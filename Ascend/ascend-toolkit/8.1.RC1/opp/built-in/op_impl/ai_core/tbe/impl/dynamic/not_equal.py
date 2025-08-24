#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
not_equal
"""
from __future__ import absolute_import

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util_soc_common import is_v200


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # define a scalar, value = 2**(-126), minimun num of float32 2**(-126)
    SCALAR_MIN_FP32 = 2**(-126)
    # define a scalar, `value = 2**(50)`
    SCALAR_MUL_FP32 = 2**(50)
    # define a scalar, `value = 2**(26)`
    SCALAR_MUL2_FP32 = 2**(26)
    # define a scalar, value = 2**(-24), minimun num of float16 2**(-24)
    SCALAR_MIN_FP16 = 2**(-24)
    # define a scalar, `value = 2**(12)`
    SCALAR_MUL_FP16 = 2**(12)


def not_equal_compute_with_cmp(input_x, input_y, shape_broadcast):
    """
    b64 compute for not_equal: input data type is int64 or uint64
    """
    input_x = tbe.broadcast(input_x, shape_broadcast)
    input_y = tbe.broadcast(input_y, shape_broadcast)

    res = tbe.vcmp(input_x, input_y, "ne", "bool")

    return res


def not_equal_compute_with_cmpsel(input_x, input_y, shape_broadcast, dtype_x):
    """
    vcmpsel compute for not_equal: input data type is float32
    """
    input_x = tbe.broadcast(input_x, shape_broadcast)
    input_y = tbe.broadcast(input_y, shape_broadcast)

    ones = tbe.broadcast(tvm.const(1, dtype=dtype_x), shape_broadcast, output_dtype=dtype_x)
    zeros = tbe.broadcast(tvm.const(0, dtype=dtype_x), shape_broadcast, output_dtype=dtype_x)
    res = tbe.vcmpsel(input_x, input_y, "eq", zeros, ones)
    res = tbe.cast_to(res, "int8")

    return res


def not_equal_compute_b32(input_x, input_y, shape_broadcast):
    """
    b32 compute for not_equal: input data type is int32
    """
    input_x = tbe.broadcast(input_x, shape_broadcast)
    input_y = tbe.broadcast(input_y, shape_broadcast)

    eq_mask = tbe.vcmp(input_x, input_y, "eq", "bit")
    res = tbe.vsel(eq_mask, 0, 1)
    res = tbe.cast_to(res, "int8")

    return res


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("NotEqual", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def not_equal_compute(input_x, input_y, output_z, kernel_name="not_equal"):
    """
    compute for not_equal

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict info of output_z
    kernel_name: str
        cce kernel name, default value is "not_equal"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype_x = input_x.dtype
    shape_x, shape_y, shape_broadcast = shape_util.broadcast_shapes(input_x.shape,
                                                                    input_y.shape,
                                                                    param_name_input1="input_x",
                                                                    param_name_input2="input_y")

    if dtype_x in ("int64", "uint64"):
        return not_equal_compute_with_cmp(input_x, input_y, shape_broadcast)

    if is_v200() and dtype_x in ("float32", "float16"):
        return not_equal_compute_with_cmpsel(input_x, input_y, shape_broadcast, dtype_x)
    elif dtype_x == "int32" and tbe_platform.api_check_support("tbe.dsl.vcmp", "int32"):
        return not_equal_compute_b32(input_x, input_y, shape_broadcast)
    elif dtype_x in ("float32", "int32"):
        tensor_min = tbe.broadcast(tvm.const(Constant.SCALAR_MIN_FP32, dtype="float32"), shape_broadcast)
        tensor_mul = tbe.broadcast(tvm.const(Constant.SCALAR_MUL_FP32, dtype="float32"), shape_broadcast)
        tensor_mul1 = tbe.broadcast(tvm.const(Constant.SCALAR_MUL2_FP32, dtype="float32"), shape_broadcast)
    else:
        tensor_min = tbe.broadcast(tvm.const(Constant.SCALAR_MIN_FP16, dtype="float16"), shape_broadcast)
        tensor_mul = tbe.broadcast(tvm.const(Constant.SCALAR_MUL_FP16, dtype="float16"), shape_broadcast)

    if dtype_x in ("int8", "uint8"):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")

    input_x = tbe.broadcast(input_x, shape_broadcast)
    input_y = tbe.broadcast(input_y, shape_broadcast)
    res_vsub = tbe.vsub(input_x, input_y)

    if dtype_x in ("float32", "int32"):
        res_vsub = tbe.cast_to(res_vsub, "float32")
        res_vabs = tbe.vabs(res_vsub)
    else:
        res_vabs = tbe.vabs(res_vsub)

    res_min = tbe.vmin(res_vabs, tensor_min)
    res_vmul = tbe.vmul(res_min, tensor_mul)
    res_vmul1 = tbe.vmul(res_vmul, tensor_mul)

    if dtype_x in ("float32", "int32"):
        res_vmul2 = tbe.vmul(res_vmul1, tensor_mul1)
        res = tbe.cast_to(res_vmul2, "int8", True)
    else:
        res = tbe.cast_to(res_vmul1, "int8", True)

    return res


@register_operator("NotEqual")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def not_equal(input_x, input_y, output_z, kernel_name="not_equal"):
    """
    Returns the truth value of (x != y) element-wise

    Parameters
    ----------
    input_x: dict
        dict of input_x, include keys(shape and dtype)
    input_y: dict
        dict of input_y, include keys(shape and dtype)
    output_z: dict
        dict of  output
    kernel_name: str
        cce kernel name, default value is "not_equal"

    Returns
    -------
    None
    """
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()
    x_dtype = "int8" if x_dtype == "bool" else x_dtype
    y_dtype = "int8" if y_dtype == "bool" else y_dtype

    check_list = ("bfloat16", "float16", "float32", "int32", "int64", "uint64", "uint8", "int8")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(y_dtype, check_list, param_name="input_y")
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("not_equal", "input_x", "input_y", str(x_dtype),
                                                              str(y_dtype))

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_x_, input_y_) in ins:
        with tbe.compute():
            x_shape, y_shape = shape_util.variable_shape([input_x_, input_y_])
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, y_dtype, "tensor_y")
            res = not_equal_compute(tensor_x, tensor_y, output_z, kernel_name)
            tensors.append([tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
