# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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
trunc
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute

CONST_ZERO = 0.0
SHAPE_SIZE_LIMIT = 2147483648


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@register_operator_compute("Trunc", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def trunc_compute(x, y, kernel_name="trunc"):
    """
    returns the interger part of a number by removing any fractional digits

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : TVM tensor
        the dict of y
    kernel_name : str
        kernel name, default value is "trunc"

    Returns
    -------
    result res
    """
    dtype_x = x.dtype.lower()
    shape_x = x.shape
    # the int32 type is processed separately
    if dtype_x == "int32":
        zero = tvm.const(0, dtype_x)
        input_data_zero = tbe.broadcast(zero, x.shape, dtype_x)
        res = tbe.vadd(x, input_data_zero)
        return res
    
    if not tbe_platform.api_check_support("tbe.dsl.trunc", dtype_x):
        x = tbe.cast_to(x, "float16")

    check_type_tuple = ("int8", "uint8")
    if dtype_x not in check_type_tuple and tbe_platform.api_check_support("tbe.dsl.trunc", "f322f32"):
        x_f32 = tbe.cast_to(x, "float32")
        res = tbe.trunc(x_f32, "float32")
        res = tbe.cast_to(res, dtype_x)
    else:
        tensor_zero = tbe.broadcast(tvm.const(CONST_ZERO, x.dtype), shape_x)
        data_res1 = tbe.vmax(x, tensor_zero)
        data_res2 = tbe.vmin(x, tensor_zero)
        data_res1 = tbe.floor(data_res1)
        data_res2 = tbe.ceil(data_res2)
        res = tbe.vadd(data_res1, data_res2)
        if res.dtype != dtype_x:
            res = tbe.cast_to(res, dtype_x, f1628IntegerFlag=True)

    return res


# 'pylint: disable=locally-disabled,redefined-builtin
@register_operator("Trunc")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def trunc(x, y, kernel_name="trunc"):
    """
    returns the interger part of a number by removing any fractional digits

    Parameters
    ----------
    x : the dict of x
         include shape and dtype.

    y : the dict of y
         include shape and dtype.

    kernel_name : str
        kernel name, default value is "trunc"

    Returns
    -------
    None
    """
    x_dtype = x.get("dtype").lower()

    #Check whether the dtype of the input parameter is supported.
    check_tuple = ("bfloat16", "float16", "float32", "int8", "int32", "uint8")
    para_check.check_dtype_rule(x_dtype, check_tuple, param_name="x")
    para_check.check_kernel_name(kernel_name)

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_x = tvm.placeholder(x_shape[0], dtype=x_dtype, name="data_x")
            res = trunc_compute(data_x, y, kernel_name)
            tensors.append([data_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
