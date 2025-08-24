#!/usr/bin/python
# -*- coding: utf-8 -*-
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
softmax_grad_ext
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from te.utils.error_manager import error_manager_vector
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl import constant_util as constant
from impl.util import util_frac_z as fz
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=locally-disabled,too-many-locals,unused-variable
# 'pylint: disable=locally-disabled,invalid-name,unidiomatic-typecheck
# 'pylint: disable=locally-disabled,too-many-branches
# 'pylint: disable=unnecessary-comprehension
def _division_sixteen(shape):

    if len(shape) < 2:
        if shape[-1] == 0:
            error_manager_vector.raise_err_specific_reson("softmax_grad_ext", "value of shape is illegal")
            return False

    if shape[-1] == 0 or shape[-2] == 0:
        error_manager_vector.raise_err_specific_reson("softmax_grad_ext", "value of shape is illegal")
        return False

    if shape[-1] % constant.SIZE_SIXTEEN == 0 and shape[-2] % constant.SIZE_SIXTEEN == 0:
        return True

    return False


# 'pylint: disable = unused-argument
def op_select_format(grad, x1, x2, y, axes, keep_dims,
                     kernel_name="softmax_grad_ext"):
    """select format dynamically"""
    origin_shape0 = shape_util.scalar2tensor_one(grad.get("ori_shape"))
    origin_shape1 = shape_util.scalar2tensor_one(x1.get("ori_shape"))
    origin_shape2 = shape_util.scalar2tensor_one(x2.get("ori_shape"))

    condition_0 = len(origin_shape2) == 1 and origin_shape2[0] == 1
    condition_1 = _division_sixteen(origin_shape0)
    condition_2 = _division_sixteen(origin_shape1)

    if condition_0 and condition_1 and condition_2:
        # NZ + NZ + Scalar
        input0 = gen_param(classify="input0", name="grad",
                           datatype="bfloat16,float16,float,bfloat16,float16,float",
                           format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,ND,ND,ND",
                           unknownshape_format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,ND,ND,ND")
        input1 = gen_param(classify="input1", name="x1",
                           datatype="bfloat16,float16,float,bfloat16,float16,float",
                           format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,ND,ND,ND",
                           unknownshape_format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,ND,ND,ND")
        input2 = gen_param(classify="input2", name="x2",
                           datatype="bfloat16,float16,float,bfloat16,float16,float",
                           format="ND,ND,ND,ND,ND,ND",
                           unknownshape_format="ND,ND,ND,ND,ND,ND")
        output0 = gen_param(classify="output0", name="y",
                            datatype="bfloat16,float16,float,bfloat16,float16,float",
                            format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,ND,ND,ND",
                            unknownshape_format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,ND,ND,ND")
    else:
        # ND+ND+ND
        input0 = gen_param(classify="input0", name="grad",
                           datatype="bfloat16,float16,float",
                           format="ND,ND,ND",
                           unknownshape_format="ND,ND,ND")
        input1 = gen_param(classify="input1", name="x1",
                           datatype="bfloat16,float16,float",
                           format="ND,ND,ND",
                           unknownshape_format="ND,ND,ND")
        input2 = gen_param(classify="input2", name="x2",
                           datatype="bfloat16,float16,float",
                           format="ND,ND,ND",
                           unknownshape_format="ND,ND,ND")
        output0 = gen_param(classify="output0", name="y",
                            datatype="bfloat16,float16,float",
                            format="ND,ND,ND",
                            unknownshape_format="ND,ND,ND")

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def shape_broadcast(data_1, data_2):
    """broadcast the two input

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = shape_util.shape_to_list(data_1.shape)
    shape_y = shape_util.shape_to_list(data_2.shape)
    if shape_x != shape_y:
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(data_1.shape, data_2.shape,
                                                                  param_name_input1="data_1",
                                                                  param_name_input2="data_2")
        data_1 = _broadcast_nz(data_1, shape_max)
        data_2 = _broadcast_nz(data_2, shape_max)

    return data_1, data_2


def _broadcast_nz(tensor, shape):
    broadcast_axes = []
    src_shape = shape_util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and \
            broadcast_axes[1] - broadcast_axes[0] != 1 and \
            broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = tbe.broadcast(tensor, temp_shape)
    tensor = tbe.broadcast(tensor, shape)
    return tensor


@register_operator_compute("SoftmaxGradExt", op_mode="dynamic", support_fusion=False)
# 'pylint: disable = unused-argument
def softmax_grad_ext_compute(data_grad, data_x1, data_x2,
                             y, axes, keep_dims,
                             kernel_name="softmax_grad_ext"):
    """apply one adam calculation function

    Parameters
    ----------
    data_grad: TVM tensor
         the input tensor of mul and sub
    data_x1: TVM tensor
         the input tensor of mul and mul_1
    data_x2: TVM tensor
         the input tensor of mul_1
    y: dict
         the output tensor of mul_grad
    axes: int, list, tuple
        the axes for reduce.
    keep_dims: bool
        if true, retains reduced dimensions with length 1.
    kernel_name : str
        kernel name, default value is "softmax_grad_ext"

    Returns
    -------
    output tensor
    """
    grad_dtype = data_grad.dtype
    shape = shape_util.shape_to_list(data_grad.shape)
    list_axis = list(axes)

    attributes = data_grad.op.attrs
    disable_fuse_axes = attributes["disable_fuse_axes"]
    ori_shape = shape_util.shape_to_list(attributes["ori_shape"])
    ori_format = attributes["ori_format"]
    input_format = attributes["format"]
    has_improve_precision = False
    is_use_value = False

    if grad_dtype == "bfloat16":
        data_grad = tbe.cast_to(data_grad, "float32")
        data_x1 = tbe.cast_to(data_x1, "float32")
        data_x2 = tbe.cast_to(data_x2, "float32")

    if len(list_axis) == 2:
        if input_format == "FRACTAL_NZ":
            is_use_value = True
            idc_list = shape_util.shape_to_list(disable_fuse_axes)
            idx_c1 = idc_list[0]
            idx_c0 = idc_list[1]
            c = -1
            if (idx_c0 - idx_c1) == 2:
                c = ori_shape[-1]
            else:
                c = ori_shape[-2]
            c = tbe.var('c') if c == -1 else c
            pad_c = tvm.floormod(c - 1, shape[idx_c0]) + 1

    if is_use_value:
        data_x1 = tbe.set_value(data_x1, lambda *i: tvm.all(i[list_axis[0]] > shape[list_axis[0]] - 2, \
                                                            i[list_axis[1]] > pad_c - 1), 0)
    # mul
    data_grad, data_x1 = shape_broadcast(data_grad, data_x1)
    mul_result = tbe.vmul(data_grad, data_x1)

    # sum
    dtype = mul_result.dtype
    if dtype == "float16":
        mul_result = tbe.cast_to(mul_result, "float32")
    sum_result = tbe.reduce_sum(mul_result, axis=axes, keepdims=keep_dims)
    if dtype == "float16":
        sum_result = tbe.cast_to(sum_result, "float16")

    # sub
    data_grad, sum_result = shape_broadcast(data_grad, sum_result)
    sub_result = tbe.vsub(data_grad, sum_result)

    # mul_1
    data_x1, data_x2 = shape_broadcast(data_x1, data_x2)
    mul_1_result = tbe.vmul(data_x1, data_x2)

    # mul_grad
    sub_result, mul_1_result = shape_broadcast(sub_result, mul_1_result)
    res = tbe.vmul(mul_1_result, sub_result)
    if grad_dtype == "bfloat16":
        res = tbe.round(res, "bfloat16")
    return res


@register_operator("SoftmaxGradExt")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            (para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_LIST_INT),
                            para_check.REQUIRED_ATTR_BOOL, para_check.KERNEL_NAME)
def softmax_grad_ext(grad, x1, x2, y, axes, keep_dims, kernel_name="softmax_grad_ext"):
    """function: softmax_grad_ext

    Parameters
    ----------
    grad: dict
         the input tensor of mul and sub
    x1: dict
         the input tensor of mul and mul_1
    x2: dict
         the input tensor of mul_1
    y: dict
         the output tensor of mul_grad
    axes: int, list, tuple
        the axes for reduce.
    keep_dims: bool
        if true, retains reduced dimensions with length 1.
    kernel_name : str
        kernel name, default value is "softmax_grad_ext"

    Returns
    -------
    None
    """
    shape_grad = shape_util.scalar2tensor_one(grad.get("shape"))
    shape_x1 = shape_util.scalar2tensor_one(x1.get("shape"))
    shape_x2 = shape_util.scalar2tensor_one(x2.get("shape"))
    input_format = grad.get("format")
    dtype_grad = grad.get("dtype").lower()
    ori_shape = grad.get("ori_shape")
    ori_format = grad.get("ori_format")
    if not isinstance(axes, int):
        list_axis = list(axes)
    else:
        list_axis = [axes]
    if input_format == "FRACTAL_NZ":
        list_axis = fz.to_frac_z_axis(ori_shape, list_axis)
    extra_params = {}
    if input_format == "FRACTAL_NZ" and len(list_axis) == 2:
        extra_params.update({"disable_fuse_axes": [list_axis[0], list_axis[1]]})
    extra_params.update({"input_shape_type":[0, 0, 1]})
    tensors = []
    schedules = []
    ins = classify([grad, x1, x2, list_axis], "norm", extra_params)

    for idx, (grad, x1, x2, reduce_axis) in enumerate(ins):
        with tbe.compute():
            disable_fuse_axes = []
            if "disable_fuse_axes" in extra_params:
                disable_fuse_axes = extra_params.get("disable_fuse_axes")[idx]
            shape_grad_new, x1_shape_new, x2_shape_new = shape_util.variable_shape([grad, x1, x2], op_mode="norm")
            grad_softmax = tvm.placeholder(shape_grad_new, dtype=dtype_grad, name="grad_softmax",
                                           attrs={"ori_shape": ori_shape, "ori_format": ori_format,
                                                  "format": input_format,
                                                  "disable_fuse_axes": disable_fuse_axes})
            x1_softmax = tvm.placeholder(x1_shape_new, dtype=dtype_grad, name="x1_softmax",
                                         attrs={"ori_shape": ori_shape, "ori_format": ori_format,
                                         "format": input_format, "disable_fuse_axes": disable_fuse_axes})
            x2_softmax = tvm.placeholder(x2_shape_new, dtype=dtype_grad, name="x2_softmax")
            output = softmax_grad_ext_compute(grad_softmax, x1_softmax, x2_softmax, y,
                                              reduce_axis, keep_dims, kernel_name)
            tensors.append([grad_softmax, x1_softmax, x2_softmax, output])

        with tvm.target.cce():
            sch = tbe.auto_schedule(output)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
  
