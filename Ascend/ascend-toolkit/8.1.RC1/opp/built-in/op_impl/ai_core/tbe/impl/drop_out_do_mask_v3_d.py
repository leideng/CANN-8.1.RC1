#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
drop_out_do_mask_v3_d
"""
import operator
from functools import reduce as  functools_reduce

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util import util_select_op_base


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    MATMUL_BATCH_SIZE = 0
    BATCH_MATMUL_BATCH_SIZE1 = 1
    BATCH_MATMUL_BATCH_SIZE2 = 2
    BATCH_MATMUL_BATCH_SIZE3 = 3
    BATCH_MATMUL_BATCH_SIZE4 = 4
    SHAPE_SIZE_LIMIT = 1 << 30
    SIZE_SIXTEEN = 16


def _division_sixteen(shape):
    """
    judge whether the last two dimensions are divided by 16
    Parameters
    ----------
    shape : input shape
    Returns : true or false
    """
    if len(shape) < 2:

        return False

    if shape[-1] == 0 or shape[-2] == 0:

        return False

    return shape[-1] % Constant.SIZE_SIXTEEN == 0 and shape[-2] % Constant.SIZE_SIXTEEN == 0


def op_select_format(input_tensor, input_mask, output, input_keep_prob, kernel_name="dropout_do_mask_v3_d"):
    """
    1.when the lengths of input_tensor's shape dim -1 and dim -2 % 16 == 0
    support NZ, ND format.\n

        example:\n
        original:\n
        input_tensor's Tensor(shape=(16, 16), "ND")\n
        input_mask's Tensor(shape=(16, 16), "ND")\n
        support conversion to NZ operation:\n
        input_tensor's Tensor(shape=(1, 1, 16, 16), "FRACTAL_NZ")\n
        input_mask's Tensor(shape=(16, 16), "ND")\n
    """
    shape_input_tensor = input_tensor.get("ori_shape")
    if _division_sixteen(shape_input_tensor):
        # Nz+ND
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x",
                                               datatype="float16,float16,float,float16,float16,float,float",
                                               format="FRACTAL_NZ,ND,ND,FRACTAL_NZ,ND,FRACTAL_NZ,ND")
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="mask",
                                               datatype="uint8,uint8,uint8,bool,bool,bool,bool",
                                               format="ND,ND,ND,FRACTAL_NZ,ND,FRACTAL_NZ,ND")
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype="float16,float16,float,float16,float16,float,float",
                                                format="FRACTAL_NZ,ND,ND,FRACTAL_NZ,ND,FRACTAL_NZ,ND")
    else:
        # ND+ND
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x",
                                               datatype="float16,float,float16,float",
                                               format="ND,ND,ND,ND")
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="mask",
                                               datatype="uint8,uint8,bool,bool",
                                               format="ND,ND,ND,ND")
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype="float16,float,float16,float",
                                                format="ND,ND,ND,ND")

    param_list = [input0, input1, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def reshape_input_mask(input_tensor, input_mask, kernel_name):
    """
    Reshape mask shape ND to matmul shape FRACTAL_NZ,
    e.g. [batch1, batch2, K//16, M//16, 16, 16] -> [batch, K//16, M//16, 16, 16].

    Params
    ----------------
    input_tensor: matmul, tvm.tensor, fp16/fp32
    input_mask: dropout_gen_mask, tvm.tensor, uint8
    kernel_name: str

    Returns
    ----------------
    input_mask: reshaped mask
    """
    matmul_flag = "matmul" in input_tensor.op.tag \
        and input_tensor.op.attrs["format"] == "FRACTAL_NZ"
    matmul_shape = shape_util.shape_to_list(input_tensor.shape)
    mask_shape = shape_util.shape_to_list(input_mask.shape)
    batch_shape = mask_shape[:-4]

    if matmul_flag:
        lambda_expression = None
        if len(batch_shape) == Constant.MATMUL_BATCH_SIZE:
            lambda_expression = lambda *indices: input_mask(*indices)
        elif len(batch_shape) == Constant.BATCH_MATMUL_BATCH_SIZE1:
            lambda_expression = lambda *indices: input_mask(*indices)
        elif len(batch_shape) == Constant.BATCH_MATMUL_BATCH_SIZE2:
            lambda_expression = lambda *indices: input_mask(
                indices[0] // batch_shape[-1],
                indices[0] % batch_shape[-1],
                indices[-4],
                indices[-3],
                indices[-2],
                indices[-1]
            )
        elif len(batch_shape) == Constant.BATCH_MATMUL_BATCH_SIZE3:
            lambda_expression = lambda *indices: input_mask(
                indices[0] // batch_shape[-1] // batch_shape[-2],
                indices[0] // batch_shape[-1] % batch_shape[-2],
                indices[0] % batch_shape[-1],
                indices[-4],
                indices[-3],
                indices[-2],
                indices[-1]
            )
        elif len(batch_shape) == Constant.BATCH_MATMUL_BATCH_SIZE4:
            lambda_expression = lambda *indices: input_mask(
                indices[0] // batch_shape[-1] // batch_shape[-2] // batch_shape[-3],
                indices[0] // batch_shape[-1] // batch_shape[-2] % batch_shape[-3],
                indices[0] // batch_shape[-1] % batch_shape[-2],
                indices[0] % batch_shape[-1],
                indices[-4],
                indices[-3],
                indices[-2],
                indices[-1]
            )
        else:
            error_detail = ("Only support to adjust batch shape [2, 3, 4], " +
                "but the recent batch shape is [%d]." % (len(batch_shape)))
            error_manager_vector.raise_err_input_shape_invalid(
                kernel_name, "input_mask", error_detail
            )

        if lambda_expression:
            input_mask = tvm.compute(
                matmul_shape,
                lambda_expression,
                name="dropout_reshape",
                tag="dropout_broadcast"
            )

    return input_mask, batch_shape


@register_operator_compute("drop_out_do_mask_v3_d", op_mode="static", support_fusion=True)
def drop_out_do_mask_v3_d_compute(input_tensor: tvm.Tensor,
                                  input_mask: tvm.Tensor,
                                  output,
                                  input_keep_prob: float,
                                  kernel_name="drop_out_do_mask_v3_d"):
    """
    dropoutdomaskv3d compute
    """
    input_mask_dtype = input_mask.dtype.lower()
    if input_mask_dtype == "uint8":
        input_mask, batch_shape = reshape_input_mask(input_tensor, input_mask, kernel_name)
        output = _compute_inner(input_tensor, input_mask, input_keep_prob)
        if batch_shape:
            output.op.attrs["batch_shape"] = batch_shape
    elif input_mask_dtype == "int8":
        output = _compute_inner(input_tensor, input_mask, input_keep_prob)
    else:
        raise RuntimeError("Unsupport data type of input_mask: " + input_mask.dtype)

    return output


def _get_placeholder(dict_list, name_list):
    list_placeholder = []
    for var, name in zip(dict_list, name_list):
        shape = var.get("shape")
        shape = shape_util.scalar2tensor_one(shape)
        shape_refine = (functools_reduce(operator.mul, shape),)
        dtype = var.get("dtype").lower()
        if name == "input_tensor":
            input_shape = list(shape_refine)
        if input_shape != list(shape_refine):
            raise RuntimeError(
                "the shape of input_tensor and input_mask must be equal !")
        list_placeholder.append(
            tvm.placeholder(shape=shape_refine, name=name, dtype=dtype))
    return list_placeholder


def _compute_inner(input_tensor: tvm.Tensor,
                   input_mask: tvm.Tensor,
                   input_keep_prob: float):
    input_dtype = input_tensor.dtype
    input_mask = tbe.cast_to(input_mask, input_dtype)
    rec_keep_prob = 1 / input_keep_prob
    mul_input_mask = tbe.vmul(input_tensor, input_mask)
    return tbe.vmuls(mul_input_mask, tvm.const(rec_keep_prob, input_dtype))


@para_check.check_input_type(dict, dict, dict, float, str)
def drop_out_do_mask_v3_d(input_tensor, input_mask, output, input_keep_prob,
                          kernel_name="drop_out_do_mask_v3_d"):
    """
    algorithm: tf_drop_out_do_mask_v3_d
    scale_x = x*(1 / keep_prob)
    res = select(mask == 1, scale_x, 0)

    Parameters
    ----------
    input_tensor : dict,shape and dtype of input_tensor,only support float16 and float32
    input_mask : dict,shape and dtype of input_mask
        shape of mask,1D, dtype == uint8/int8
        length=(size(shape_tensor)+tbe_platform.ELEMENTS_VECTOR_OP_FP16
        -1)/tbe_platform.ELEMENTS_VECTOR_OP_FP16*tbe_platform.ELEMENTS_VECTOR_OP_FP16
        eg. shape_tensor=[2,5,8] shape_mask=[16] shape_res=[2,5,8]
        shape_tensor=[15,17,19] shape_mask=[608] shape_res=[15,17,19]
    input_keep_prob : dict,shape and dtype of input_keep_prob
        shape of keep_prob, only 1 parament and equals to (1)
        prob scale (0.0,1.0] NOTICE: type same as dytpe
    output : dict,shape and dtype of output
    kernel_name : str
        cce kernel name, default value is "drop_out_do_mask_v3_d"

    Returns
    -------
    None
    """
    para_check.check_kernel_name(kernel_name)
    para_check.check_dtype_rule(
        input_tensor.get('dtype').lower(), ("float16", "float32"))
    para_check.check_dtype_rule(
        input_mask.get('dtype').lower(), ("uint8", "int8"))
    para_check.check_shape_rule(input_tensor.get('shape'),
                          max_shape_num=Constant.SHAPE_SIZE_LIMIT)
    para_check.check_shape_rule(input_mask.get('shape'),
                          max_shape_num=Constant.SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(input_tensor.get('shape'), Constant.SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(input_mask.get('shape'), Constant.SHAPE_SIZE_LIMIT)
    input_name_list = ['input_tensor', 'input_mask']
    list_placeholder = _get_placeholder([input_tensor, input_mask],
                                         input_name_list)
    input_tensor = list_placeholder[0]
    input_mask = list_placeholder[1]

    output = drop_out_do_mask_v3_d_compute(input_tensor, input_mask,
                                           output, input_keep_prob)

    build_list = [input_tensor, input_mask, output]
    config = {"name": kernel_name, "tensor_list": build_list}

    with tvm.target.cce():
        sch = auto_schedule(output)

    config = {"name": kernel_name,
              "tensor_list": build_list}
    build(sch, config)
    tbe_platform.fusion_manager.set_current_op_pattern("DropOutDoMaskV3D")
