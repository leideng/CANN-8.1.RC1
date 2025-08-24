#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
celu
"""
import te.lang.cce as tbe
from tbe.dsl.api import build
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util


# 'pylint:disable=too-many-arguments, too-many-locals
@register_operator_compute("celu", op_mode="static", support_fusion=True)
def celu_compute(x, y, a1, a2, a3, kernel_name="celu"):
    """
    Implement the operator by referring to  the
            TBE Operator Development Guide.
    celu:
    if x >= 0
        y = alpha3 * 3
    else
        y = alpha1 * (exp(x/alpha2)-1)
    x:dict of x, include shape and dtype
    y:dict of y, include shape and dtype
    a1: scalar, alpha1
    a2: scalar, alpha2
    a3: scalar, alpha3

    """

    dtype = x.dtype

    vexp_support = False
    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        vexp_support = True

    if dtype.lower() == "float16" and vexp_support:
        x = tbe.cast_to(x, "float32")
        compute_dtype = "float32"
    else:
        compute_dtype = dtype

    rec_a2 = tvm.const(-1/a2, compute_dtype)
    negative_x = tbe.vmuls(x, tvm.const(-1, compute_dtype))
    vmax_x = tbe.vmaxs(negative_x, tvm.const(0, compute_dtype))
    div_a2x = tbe.vmuls(vmax_x, rec_a2)
    exp_a2x = tbe.vexp(div_a2x)
    neg_part = tbe.vadds(exp_a2x, tvm.const(-1, compute_dtype))
    pos_part = tbe.vmaxs(x, tvm.const(0, compute_dtype))
    mul_a1 = tbe.vmuls(neg_part, tvm.const(a1, compute_dtype))
    mul_a3 = tbe.vmuls(pos_part, tvm.const(a3, compute_dtype))
    res = tbe.vadd(mul_a1, mul_a3)
    if dtype.lower() == "float16" and vexp_support:
        res = tbe.cast_to(res, dtype)

    return res


# 'pylint: disable=invalid-name,too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def celu(x, y, alpha1=1.0, alpha2=1.0, alpha3=1.0, kernel_name="celu"):
    """
    Implement the operator by referring to  the
            TBE Operator Development Guide.
    celu:
    if x >= 0
        y = alpha3 * 3
    else
        y = alpha1 * (exp(x/alpha2)-1)
    x:dict of x, include shape and dtype
    y:dict of y, include shape and dtype
    a1: scalar, alpha1
    a2: scalar, alpha2
    a3: scalar, alpha3

    """
    util.check_kernel_name(kernel_name)
    shape_input = x.get("shape")
    dtype_input = x.get("dtype")
    input_dtype = dtype_input.lower()

    para_check.check_shape(shape_input, param_name="x")

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    if alpha2 == 0:
        error_manager_vector.raise_err_input_value_invalid("celu", "alpha2", "non-zero", "zero")

    data_input = tvm.placeholder(shape_input, name="data_input", dtype=input_dtype)

    res = celu_compute(data_input, y, alpha1, alpha2, alpha3, kernel_name)

    with tvm.target.cce():
        auto_sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    build(auto_sch, config)
