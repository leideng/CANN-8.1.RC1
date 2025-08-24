#!/usr/bin/env python
# coding: utf-8
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
tanh
"""
import functools
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm

# shape size limit for aicore is 2**31
SHAPE_SIZE_LIMIT = 2147483648


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name
@register_operator_compute("tanh", op_mode="static", support_fusion=True)
def tanh_compute(input_x, output_y, kernel_name="tanh", impl_mode=None):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is tanh

    Returns
    -------
    res : tvm.tensor
        the result of tanh
    """
    if impl_mode == OpImplMode.HIGH_PERFORMANCE:
        return fast_tanh_compute(input_x, output_y, kernel_name)

    input_dtype = input_x.dtype
    # positive min float32 value
    MIN_FP_DATA = 2 ** (-126)
    CONST_DTYPE = input_dtype
    # positive min float16 value
    if input_dtype == "float16":
        MIN_FP_DATA = 2 ** (-14)

    has_improve_precision = False

    if input_dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True
        CONST_DTYPE = "float32"

    input_abs = tbe.vabs(input_x)
    power_val = tbe.vmuls(input_abs, tvm.const(-2, CONST_DTYPE))
    exp_val = tbe.vexp(power_val)

    up_val_tmp = tbe.vmul(exp_val, input_x)
    up_val = tbe.vsub(input_x, up_val_tmp)

    input_x_tmp = tbe.vadds(input_abs, MIN_FP_DATA)
    down_val_tmp = tbe.vadds(exp_val, tvm.const(1, CONST_DTYPE))
    down_val = tbe.vmul(down_val_tmp, input_x_tmp)

    res = tbe.vdiv(up_val, down_val)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


def fast_tanh_compute(input_x, output_y, kernel_name="tanh"):
    """
    algorithm: tanh
    calculating data's tanh
        x2 = x * x;
        a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        y = a / b;

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is tanh

    Returns
    -------
    res : tvm.tensor
        the result of tanh
    """
    input_dtype = input_x.dtype
    const_b1 = tvm.const(135135.0, "float32")
    const_b2 = tvm.const(17325.0, "float32")
    const_b3 = tvm.const(378.0, "float32")
    const_b4 = tvm.const(62370.0, "float32")
    const_b5 = tvm.const(3150.0, "float32")
    const_b6 = tvm.const(28.0, "float32")
    one_const = tvm.const(1.0, "float32")
    negative_one_const = tvm.const(-1.0, "float32")

    has_improve_precision = False
    if input_dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True

    input_mul = tbe.vmul(input_x, input_x)
    input_add = tbe.vadds(input_mul, const_b3)
    input_mul_1 = tbe.vmul(input_mul, input_add)
    input_add_1 = tbe.vadds(input_mul_1, const_b2)
    input_mul_2 = tbe.vmul(input_mul, input_add_1)
    input_add_2 = tbe.vadds(input_mul_2, const_b1)
    input_mul_3 = tbe.vmul(input_x, input_add_2)

    input_mul2 = tbe.vmuls(input_mul, const_b6)
    input_add2 = tbe.vadds(input_mul2, const_b5)
    input_mul2_1 = tbe.vmul(input_mul, input_add2)
    input_add2_1 = tbe.vadds(input_mul2_1, const_b4)
    input_mul2_2 = tbe.vmul(input_mul, input_add2_1)
    input_add_2 = tbe.vadds(input_mul2_2, const_b1)

    input_div = tbe.vdiv(input_mul_3, input_add_2)
    result1 = tbe.vmins(input_div, one_const)
    res = tbe.vmaxs(result1, negative_one_const)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def tanh(input_x, output_y, kernel_name="tanh", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is tanh
    impl_mode:str
        impl_mode, default value is None
    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    para_check.check_shape(input_shape, param_name="input_x")

    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    # fuse single axis
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x * y, input_shape)

    data = tvm.placeholder(fuseshape, name="data", dtype=input_dtype)
    res = tanh_compute(data, output_y, kernel_name, impl_mode=impl_mode)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}

    build(sch, config)
