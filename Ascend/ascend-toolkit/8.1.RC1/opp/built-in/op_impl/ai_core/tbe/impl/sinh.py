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
sinh
"""
import functools

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    SCALAR_NEGATIVE_ONE = -1
    SCALAR_ZERO_POINT_FIVE = 0.5
    SCALAR_TWO = 2


# 'pylint: disable=unused-argument
@register_operator_compute("sinh", op_mode="static", support_fusion=True)
def sinh_compute(input_data, output_data, kernel_name="sinh"):
    """algorithm: sinh
    calculating data's sinh = (exp(x) - exp(-x)) / 2

    Parameters
    ----------
    input_data: TVM tensor
        data of input.
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "sinh"

    Returns
    -------
    res: TVM tensor
        the res of sinh
    """

    dtype = input_data.dtype
    shape = input_data.shape

    # in order to get the precise calcuate result
    has_improve_precision = False
    if dtype.lower() == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_data = tbe.cast_to(input_data, "float32")
        dtype = "float32"
        has_improve_precision = True

    data_mul = tbe.vmuls(input_data, tvm.const(Constant.SCALAR_NEGATIVE_ONE, dtype))
    data_exp = tbe.vexp(data_mul)
    data_exp_x = tbe.vmuls(data_exp, tvm.const(Constant.SCALAR_ZERO_POINT_FIVE, dtype))

    tensor_two = tbe.broadcast(tvm.const(Constant.SCALAR_TWO, dtype), shape)
    data_ln2 = tbe.vlog(tensor_two)
    data_neg_ln2 = tbe.vmuls(data_ln2, tvm.const(Constant.SCALAR_NEGATIVE_ONE, dtype))
    data_x = tbe.vadd(input_data, data_neg_ln2)
    data_exp_data = tbe.vexp(data_x)

    res = tbe.vsub(data_exp_data, data_exp_x)

    # cast the dtype to float16
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sinh(input_data, output_data, kernel_name="sinh"):
    """algorithm: sinh
    calculating data's sinh = (exp(x) - exp(-x)) / 2

    Parameters
    ----------
    input_data: dict
        shape and dtype of input, only support float16, float32
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "sinh"

    Returns
    -------
    None
    """
    shape_input = input_data.get("shape")
    dtype_input = input_data.get("dtype")

    para_check.check_shape(shape_input, param_name="input_data")
    check_list = ("float16", "float32")
    input_dtype = dtype_input.lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_data")

    reshape_input = (functools.reduce(lambda x, y: x * y, shape_input[:]),)
    data_input = tvm.placeholder(reshape_input,
                                 name="data_input", dtype=dtype_input)
    res = sinh_compute(data_input, output_data, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    build(sch, config)
