#!usr/bin/env python
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
sigmoid
"""
import functools
import math
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
import te.lang.cce as tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import check_op_impl_mode
from impl.util.platform_adapter import OpImplMode


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    CONST_FP32_MAX = 3.4e+38
    CONST_FP16_MAX = 65504


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
def sigmoid_high_performance_compute(x, y, kernel_name="sigmoid"):
    """calculating data

    Calculation principle
    ---------------------
    `L(x) = 0.229270815*x - 0.0102459298*x^3 + 0.000207697530*x^5 + 0.5`
    `L(x) = a*x + b*x^3 + c*x^5 + d = x(a + x^2(b + cx^2)) + d`
    `sigmoid = max(0, min(1,L(x)))`

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : dict
        dict of y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid"

    Returns
    -------
    output tensor
    """

    mul_support = tbe_platform.api_check_support("tbe.dsl.vmuls", "float32")
    dtype = x.dtype
    if dtype == "float32" and not mul_support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'x', ("float16",), dtype)

    const_num_one = tvm.const(1, dtype=dtype)
    const_num_zero = tvm.const(0, dtype=dtype)
    const_num_a = tvm.const(0.229270815, dtype=dtype)
    const_num_b = tvm.const(-0.0102459298, dtype=dtype)
    const_num_c = tvm.const(0.000207697530, dtype=dtype)
    const_num_d = tvm.const(0.5, dtype=dtype)
    # `x^2`
    tmp_x2 = tbe.vmul(x, x)
    # `cx^2`
    tmp_cx2 = tbe.vmuls(tmp_x2, const_num_c)
    # `b + cx^2`
    tmp_bsum = tbe.vadds(tmp_cx2, const_num_b)
    # `x^2(b + cx^2)`
    tmop_cx4 = tbe.vmul(tmp_x2, tmp_bsum)
    # `a + x^2(b + cx^2)`
    tmp_asum = tbe.vadds(tmop_cx4, const_num_a)
    # `x(a + x^2(b + cx^2))`
    tmp_cx5 = tbe.vmul(x, tmp_asum)
    # `x(a + x^2(b + cx^2)) + d`
    tmp_d = tbe.vadds(tmp_cx5, const_num_d)

    tmp_min = tbe.vmins(tmp_d, const_num_one)
    tmp_max = tbe.vmaxs(tmp_min, const_num_zero)

    return tmp_max


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator_compute("sigmoid", op_mode="static", support_fusion=True)
def sigmoid_compute(x, y, kernel_name="sigmoid", impl_mode="high_precision"):
    """calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : dict
        dict of y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid"
    impl_mode : str
        impl_mode, default value is "high_precision"

    Returns
    -------
    output tensor
    """
    soc_version = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)

    if soc_version in ("Ascend310P", ) and impl_mode == "high_performance":
        return sigmoid_high_performance_compute(x, y, kernel_name)

    data_input = x
    dtype = x.dtype
    exp_support = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
    mul_support = tbe_platform.api_check_support("tbe.dsl.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'x', ("float16", ), dtype)

    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend910", "Ascend910B", "Ascend910_93"):
        if dtype == "float32" and exp_support:
            ln_res = -math.log(Constant.CONST_FP32_MAX)
            ln_res = int(ln_res * 10) / 10
        else:
            ln_res = -math.log(Constant.CONST_FP16_MAX)
            ln_res = int(ln_res * 1000) / 1000
        data_input = tbe.vmaxs(data_input, ln_res)

    const_num_neg_one = tvm.const(-1, dtype=dtype)
    const_num_one = tvm.const(1, dtype=dtype)
    tmp_negative = tbe.vmuls(data_input, const_num_neg_one)
    if dtype == "float32" and not exp_support:
        tmp_negative = tbe.cast_to(tmp_negative, "float16")
    tmp_exp = tbe.vexp(tmp_negative)
    if dtype == "float32" and not exp_support:
        tmp_exp = tbe.cast_to(tmp_exp, "float32")
    tmp_sum = tbe.vadds(tmp_exp, const_num_one)
    if dtype == "float32":
        inp_shape = tmp_sum.shape
        tensor_one = tbe.broadcast(tvm.const(1, dtype), inp_shape)
        tmp_rec = tbe.vdiv(tensor_one, tmp_sum)
    else:
        tmp_rec = tbe.vrec(tmp_sum)
    return tmp_rec


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sigmoid(x, y, kernel_name="sigmoid", impl_mode="high_precision"):
    """calculating data

    Parameters
    ----------
    x : dict
        dict of x, include keys(shape and dtype)
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "sigmoid"
    impl_mode : str
        impl_mode, default value is "high_precision"

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    shape = x.get("shape")
    dtype = x.get("dtype")
    para_check.check_shape(shape, param_name="x")
    input_dtype = dtype.lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="x")

    fused_shape = [functools.reduce(lambda a, b: a * b, shape[:])]
    data_input = tvm.placeholder(fused_shape, name="data_input", dtype=input_dtype)

    res = sigmoid_compute(data_input, y, kernel_name, impl_mode)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}
    tbe.cce_build_code(sch, config)
