# Copyright 2021 Huawei Technologies Co., Ltd
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
mod
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("Mod", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def mod_compute(input_x, input_y, output_z, kernel_name="mod"):
    """
    Returns element-wise remainder of division.
    the result here is consistent with a truncating divide.
    'truncate_mod(x, y) = x - truncate_div(x, y) * y'.

    Parameters
    ----------
    input_x: TVM tensor
        input tensor contains shape and dtype attributes.
        source data type support "bfloat16", "float16", "float32", "int8", "uint8", "int32".
    input_y: TVM tensor
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_x'.
    output_z: dict
        data of output.
        Must have the same type as 'input_x'.
    kernel_name: str
        kernel name, default value is "mod"

    Returns:
    res: TVM tensor
        output tensor. Has the same type as "input_x".
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)
    dtype = input_x.dtype.lower()

    has_improve_precision = False
    if dtype != "float32" and \
            tbe_platform.api_check_support("te.lang.cce.vdiv", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")
        has_improve_precision = True

    if list(shape_x) != list(shape_y):
        shape_x, shape_y, shape_broadcast = shape_util.broadcast_shapes(input_x.shape, input_y.shape,
                                                                        param_name_input1="input_x",
                                                                        param_name_input2="input_y")
        input_x = tbe.broadcast(input_x, shape_broadcast, "float32")
        input_y = tbe.broadcast(input_y, shape_broadcast, "float32")
    else:
        shape_broadcast = shape_x

    data_div = tbe.vdiv(input_x, input_y)
    data_zero = tbe.broadcast(tvm.const(0, "float32"), shape_broadcast, "float32")
    data_div_min = tbe.vmin(data_div, data_zero)
    data_div_max = tbe.vmax(data_div, data_zero)
    if tbe_platform.api_check_support("tbe.dsl.ceil", "f322f32"):
        data_div_max_floor = tbe.floor(data_div_max, "float32")
        data_div_min_ceil = tbe.ceil(data_div_min, "float32")
    else:
        data_div_max_floor = tbe.floor(data_div_max)
        data_div_min_ceil = tbe.ceil(data_div_min)

    if dtype != "int32" and \
            tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
        data_div_max_floor = tbe.cast_to(data_div_max_floor, "float32")
        data_div_min_ceil = tbe.cast_to(data_div_min_ceil, "float32")

    data_div_res = tbe.vadd(data_div_max_floor, data_div_min_ceil)
    data_div_res = tbe.cast_to(data_div_res, input_y.dtype.lower())
    data_mul = tbe.vmul(data_div_res, input_y)
    res = tbe.vsub(input_x, data_mul)

    if has_improve_precision:
        res = tbe.cast_to(res, dtype)

    return res


@register_operator("Mod")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def mod(input_x, input_y, output_z, kernel_name="mod"):
    """
    Returns element-wise remainder of division.

    Parameters
    ----------
    input_x: dict
        input tensor contains shape and dtype attributes.
        source data type support "bfloat16", "float16", "float32", "int8", "uint8", "int32".
    input_y: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_x'.
    output_z: dict
        data of output.
        Must have the same type as 'input_x'.
    kernel_name: str
        kernel name, default value is "mod"

    Returns:
    None
    """
    shape_util.compare_tensor_dict_key(input_x, input_y, "dtype")

    check_list = ("float16", "float32", "int8", "uint8", "int32", "bfloat16")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedule, tensors = [], []

    for (input_x_dict, input_y_dict) in ins:
        with tbe.compute():
            shape_x, shape_y = shape_util.variable_shape([input_x_dict, input_y_dict])
            reshape_x, reshape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)
            data_x = tvm.placeholder(reshape_x, dtype=input_dtype, name="data_x")
            data_y = tvm.placeholder(reshape_y, dtype=input_dtype, name="data_y")
            res = mod_compute(data_x, data_y, output_z, kernel_name="mod")

            tensors.append([data_x, data_y, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedule.append(sch)

    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensors
    }

    tbe.build(schedule, config)
