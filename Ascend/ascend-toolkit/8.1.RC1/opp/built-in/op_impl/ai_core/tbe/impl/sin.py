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
sin
"""
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from tbe import tvm
from te.utils import para_check
from te.utils import shape_util

# define a string name of "float16"
FLOAT_16 = "float16"
# define a string name of "float32"
FLOAT_32 = "float32"

PI = 3.14159265358979

# the first factor to use Taylor series in circle
FIRST_ORDER = 5
# the last factor to use Taylor series in circle
LAST_ORDER = 13
# the first factor of Taylor series
FIRST_FACTOR = -1.0 / 6.0


# 'pylint: disable=invalid-name,too-many-locals
def _sin(x):
    """algorithm: sin
    calculating data's sin x = x-x^3/3!+x ^5/5!-x^7/7!+x^9/9!-x^11/11! (-pai/2 < x < pai/2)

    Parameters
    ----------
    x : TVM tensor
        the placeholders of input data

    Returns
    -------
    res : the res of sin
    """
    input_x_power = tbe.vmul(x, x)
    iter_value = tbe.vmul(tbe.vmuls(input_x_power, FIRST_FACTOR), x)
    res = tbe.vadd(x, iter_value)

    i = FIRST_ORDER
    while i < LAST_ORDER:
        iter_value = tbe.vmuls(tbe.vmul(input_x_power, iter_value),
                                       -1.0 / (i*(i - 1)))
        res = tbe.vadd(res, iter_value)
        # add 2 to get the next order
        i = i + 2

    return res


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("sin")
def sin_compute(x, y, kernel_name="sin"):
    """algorithm: sin
    calculating data's sin x = x - x^3/3! + x ^5/5! + ... + (-1)^k*x^2(k+1)/(2(k+1))!

    Parameters
    ----------
    x : TVM tensor
        the placeholders of input data
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is "sin"

    Returns
    -------
    res : the res of sin
    """
    dtype = x.dtype
    shape = shape_util.shape_to_list(x.shape)

    has_improve_precision = False
    cast_dtype = FLOAT_16
    if tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
        has_improve_precision = True
        cast_dtype = FLOAT_32

    # cast to type float32 when type is float16
    if dtype == FLOAT_16 and has_improve_precision:
        x = tbe.cast_to(x, FLOAT_32)

    pai_multiple = tbe.vmuls(x, 1 / PI)
    round_float = tbe.cast_to(tbe.round(pai_multiple), cast_dtype)
    # to adjust x to [-pai/2,pai/2]
    x = tbe.vsub(x, tbe.vmuls(round_float, PI))

    res = _sin(x)

    # if round is odd, the final result need to mutiply -1.Need to multipy 1/2 to get the ceil value
    ceil_value = tbe.ceil(tbe.vmuls(round_float, 1 / 2))
    # if odd, ceil*2-round is 1,if even, the value is 0
    sub_value = tbe.vsub(tbe.vmuls(ceil_value, tvm.const(2, dtype)), round_float)
    tensor_one = tbe.broadcast(tvm.const(1, cast_dtype), shape)
    odd_tensor = tbe.vsub(tensor_one, sub_value)
    even_tensor = tbe.vsub(odd_tensor, tensor_one)
    odd_even_tensor = tbe.vadd(odd_tensor, even_tensor)
    res = tbe.vmul(res, odd_even_tensor)

    # cast the dtype to float16
    if dtype == FLOAT_16 and has_improve_precision:
        res = tbe.cast_to(res, FLOAT_16)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sin(x, y, kernel_name="sin"):
    """algorithm: sin
    calculating data's sin x = x - x^3/3! + x^5/5! + ... + (-1)^k*x^2(k+1)/(2(k+1))!

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support float16, float32
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "sin"

    Returns
    -------
    None
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype").lower()

    para_check.check_shape(shape_input, param_name="x")
    check_list = (FLOAT_16, FLOAT_32)
    para_check.check_dtype(dtype_input, check_list, param_name="x")
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape_input)
    data_input = tvm.placeholder(fuseshape, name="data_input", dtype=dtype_input)
    res = sin_compute(data_input, y, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_input, res)}
    tbe.cce_build_code(sch, config)
