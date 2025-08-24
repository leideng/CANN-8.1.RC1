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
ascend_anti_quant
"""
import functools
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl import ascend_quant_util as util
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
from te.lang.cce import cast_to
from te.lang.cce import vadds
from te.lang.cce import vmuls


# 'pylint: disable=too-many-arguments,invalid-name,unused-argument
# 'pylint: disable=unnecessary-lambda
# 'pylint: disable=too-many-locals
# 'pylint: disable=huawei-too-many-arguments
def op_select_format(x, y, scale, offset, dtype=1, sqrt_mode=False, kernel_name="ascend_anti_quant"):
    """
    select format dynamically
    """
    x_format_list = "NC1HWC0,FRACTAL_NZ,ND"
    y_format_list = "NC1HWC0,FRACTAL_NZ,ND"

    x_dtype_list = "int8,int8,int8"
    y_dtype_list = "float16,float16,float16"


    input0 = gen_param(classify="input0", name="x",
                       datatype=x_dtype_list,
                       format=x_format_list,
                       unknownshape_format=x_format_list)

    output0 = gen_param(classify="output0", name="y",
                        datatype=y_dtype_list,
                        format=y_format_list,
                        unknownshape_format=y_format_list)
    
    param_list = [input0, output0]

    param_dynamic_in_json =  get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _check_params(x):
    """
    check the parameters including shape, dtype, kernel_name, attr.
    """
    shape = x.get("shape")
    x_format = x.get("format")
    x_dtype = x.get("dtype").lower()

    if x_format == "NC1HWC0":
        para_check.check_shape(shape, min_rank=5, max_rank=5, param_name="x")
    elif x_format == "FRACTAL_NZ":
        para_check.check_shape(shape, min_rank=4, param_name="x")
    para_check.check_dtype(x_dtype, ("int8",), param_name="x")


def _reform_compute_generate(tensor, in_shape, out_shape, scale_val, c1_index):
    """
    generate lambda func

    Parameters
    ----------
    tensor : input tensor
    in_shape : the shape of input tensor
    out_shape :the shape of output tensor
    scale_val : the value of scale

    Returns
    -------
    res lambda_func
    """
    in_shape = list(in_shape)
    out_shape = list(out_shape)
    n_dim = len(in_shape)
    c0_index = n_dim - 1

    def lambda_func(*indice):
        """
        get lambda for c0 and c1 reform
        """
        new_indice = [0] * n_dim
        for i in range(n_dim):
            if i == c1_index:
                new_indice[i] = (indice[c1_index] * out_shape[c0_index] + indice[c0_index]) // in_shape[c0_index]
            elif i == c0_index:
                new_indice[i] = (indice[c1_index] * out_shape[c0_index] + indice[c0_index]) % in_shape[c0_index]
            else:
                new_indice[i] = indice[i]
        return tensor(*new_indice) * scale_val

    return lambda_func


def _reform_by_vmuls(input_tensor, input_shape, output_shape, scale_val, c1_index):
    """
    5 dim input tensor C0 change

    Parameters
    ----------
    input_tensor : input tensor
    input_shape : the shape of input tensor
    output_shape :the shape of output tensor
    scale_val : the value of scale

    Returns
    -------
    res tensor
    """
    vmuls_vector = tvm.compute(output_shape,
                               _reform_compute_generate(input_tensor, input_shape, output_shape, scale_val, c1_index),
                               name="reform_by_vmuls", tag="anti_quant_reform_by_vmuls")

    return vmuls_vector


@register_operator_compute("ascend_anti_quant", op_mode="static", support_fusion=True)
def ascend_anti_quant_compute(x, y, scale, offset, dtype=1, sqrt_mode=False, kernel_name="ascend_anti_quant"):
    """
    int8 -> float16/float32

    Parameters:
    ----------
    x : the tensor of input
    y : the dict of output
    scale : the data of scale
    offset : the data of offset
    sqrt_mode : the sqrt mode when true the result to do sqrt
    kernel_name : cce kernel name, default value is "ascend_anti_quant"

    Returns:
    -------
    None
    """
    in_shape = shape_util.shape_to_list(x.shape)
    nz_format_flag = util.is_nz_format(x)

    c1_index = 1
    if nz_format_flag:
        c1_index = len(in_shape) - 4

    out_shape = util.get_antiquant_output_shape(in_shape, nz_format_flag, c1_transform=2)

    input_ub = tvm.compute(in_shape, lambda *i: x(*i), name="input_ub", tag="anti_quant_input_ub")
    # cast int8 to fp16
    cast_f16_ub = tvm.compute(in_shape, lambda *indice: shape_util.cast(input_ub(*indice), "float16"),
                              name="cast_f16_ub", tag="anti_quant_cast_f16_ub")

    # add offset
    offset_value = tvm.const(offset, "float16")
    offset_ub = tvm.compute(in_shape, lambda *indice: cast_f16_ub(*indice) + offset_value,
                            name="offset_ub", tag="anti_quant_offset_ub")

    scale_value = tvm.const(scale, "float16")
    if sqrt_mode:
        scale_sqrt_ub = tvm.compute(in_shape, lambda *indice: offset_ub(*indice) * scale_value,
                                    name="scale_sqrt_ub", tag="anti_quant_scale_sqrt_ub")
        scale_ub = _reform_by_vmuls(scale_sqrt_ub, in_shape, out_shape, scale_value, c1_index)
    else:
        # mul scale and convert 32 to 16 of C0
        scale_ub = _reform_by_vmuls(offset_ub, in_shape, out_shape, scale_value, c1_index)

    ori_shape = y.get("ori_shape")
    ori_format = y.get("ori_format")
    ori_c_index = ori_format.find("C")
    if ori_c_index < 0:
        ori_c_index = len(ori_shape) - 1

    ori_c = ori_shape[ori_c_index]
    # remove pad
    if 0 < ori_c % 32 <= 16:
        align_shape = out_shape[:]
        align_shape[c1_index] = (ori_c + util.Constant.FP16_BLOCK_VALUE - 1) // util.Constant.FP16_BLOCK_VALUE

        res = tvm.compute(align_shape, lambda *indice: scale_ub(*indice), name="res", tag="anti_quant",
                          attrs={"scale": scale,
                                 "sqrt_mode": sqrt_mode,
                                 "offset": offset})
    else:
        res = tvm.compute(out_shape, lambda *indice: scale_ub(*indice), name="res", tag="anti_quant",
                          attrs={"scale": scale,
                                 "sqrt_mode": sqrt_mode,
                                 "offset": offset})

    return res


def get_op_support_info(x, y, scale, offset, sqrt_mode=False, kernel_name="ascend_anti_quant"):
    """
    get split info
    """
    return util.get_quant_support_info(x)


# 'pylint: disable=too-many-arguments,invalid-name,unused-argument
# 'pylint: disable=huawei-too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def ascend_anti_quant(x, y, scale, offset, dtype=1, sqrt_mode=False, kernel_name="ascend_anti_quant"):
    """
    int8 -> float16

    Parameters:
    ----------
    x : the dict of input, format is NC1HWC0
    y : the dict of output, format is NC1HWC0
    scale : the data of scale
    offset : the data of offset
    sqrt_mode : the sqrt mode when true the result to do sqrt
    kernel_name : cce kernel name, default value is "ascend_anti_quant"

    Returns:
    -------
    None
    """
    _check_params(x)
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    x_format = x.get("format")

    if x_format == "NC1HWC0":
        # change to N,C1,H*W,C0
        input_shape = (input_shape[0], input_shape[1], input_shape[2] * input_shape[3], input_shape[4])
    elif x_format == "FRACTAL_NZ":
        batch = 1
        if len(input_shape) > 4:
            batch = functools.reduce(lambda x, y: x * y, input_shape[:-4])
        input_shape = (batch, input_shape[-4], input_shape[-3] * input_shape[-2], input_shape[-1])
    input_x = tvm.placeholder(input_shape, name="input_x", dtype=input_dtype)

    if x_format in ("NC1HWC0", "FRACTAL_NZ"):
        res = ascend_anti_quant_compute(input_x, y, scale, offset, dtype, sqrt_mode, kernel_name)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        config = {"name": kernel_name,
                  "print_ir": False,
                  "tensor_list": (input_x, res)}
        tbe.build(sch, config)
    else:
        cast_f16_ub = cast_to(input_x, "float16")
        offset_ub = vadds(cast_f16_ub, offset)
        if sqrt_mode:
            scale_sqrt_ub = vmuls(offset_ub, scale)
            res = vmuls(scale_sqrt_ub, scale)
        else:
            res = vmuls(offset_ub, scale)
        with tvm.target.cce():
            auto_sch = auto_schedule(res)
        config = {"name": kernel_name,
                  "print_ir": False,
                  "tensor_list": (input_x, res)}
        build(auto_sch, config)
