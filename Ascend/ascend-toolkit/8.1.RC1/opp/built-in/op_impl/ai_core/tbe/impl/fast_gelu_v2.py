#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
fast_gelu_v2
"""
import functools

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member,too-many-locals
@register_operator_compute("fast_gelu_v2", op_mode="static", support_fusion=True)
def fast_gelu_v2_compute(input_x, output_y, kernel_name="fast_gelu_v2",
                         impl_mode=None):
    """
    mathematical formula of fast_gelu_v2(x):
    sgn(x) = (x+0.000000000001)/|(x+0.000000000001)|
    fast_gelu_v2(x) = x*(sgn(x)*[(a/2)*(clip(|x|,max=-b) + b)^2 + 0.5] + 0.5)

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input input_x
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is fast_gelu_v2
    impl_mode: str
        impl_mode, default value is None

    Returns
    -------
     A TVM tensor same as input placeholders.
    """
    dtype = input_x.dtype
    has_improve_precision = False

    if dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        if impl_mode == OpImplMode.HIGH_PRECISION or (impl_mode is None):
            has_improve_precision = True
            input_x = tbe.cast_to(input_x, "float32")
    const_b = tvm.const(-1.769, "float32")
    const_b_ = tvm.const(1.769, "float32")
    const_a_half = tvm.const(-0.1444, "float32")
    const_c = tvm.const(0.7071, "float32")
    const_offset = tvm.const(0.000000000001, "float32")
    const_d = tvm.const(0.5, "float32")

    muls_0 = tbe.vmuls(input_x, const_c)
    abs_muls_0 = tbe.vabs(muls_0)
    max_abs_muls_0 = tbe.vmins(abs_muls_0, const_b_)
    vadds = tbe.vadds(max_abs_muls_0, const_b)
    temp = tbe.vmul(vadds, vadds)
    temp_0 = tbe.vmuls(temp, const_a_half)
    temp_0 = tbe.vadds(temp_0, const_d)
    x_adds = tbe.vadds(input_x, const_offset)
    abs_x = tbe.vabs(x_adds)
    if impl_mode == OpImplMode.HIGH_PERFORMANCE:
        vrec_abs = tbe.vrec(abs_x, priority_flag=0)
        sgn = tbe.vmul(x_adds, vrec_abs)
    else:
        sgn = tbe.vdiv(x_adds, abs_x)
    temp_1 = tbe.vmul(temp_0, sgn)
    temp_1 = tbe.vadds(temp_1, const_d)
    result = tbe.vmul(input_x, temp_1)

    if has_improve_precision:
        result = tbe.cast_to(result, "float16")

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def fast_gelu_v2(input_x, output_y, kernel_name="fast_gelu_v2", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    mathematical formula of fast_gelu_v2(x):
    sgn(x) = (x+0.000000000001)/|(x+0.000000000001)|
    fast_gelu_v2(x) = x*(sgn(x)*[(a/2)*(clip(|x|,max=-b) + b)^2 + 0.5] + 0.5)

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is fast_gelu_v2
    impl_mode:str
        impl_mode, default value is None

    Returns
    -------
    None.
    """
    shape = input_x.get("shape")
    para_check.check_shape(shape, param_name="input_x")

    check_list = ("float16", "float32")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x * y, shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=input_dtype, attrs={"shape_desc": ["A"]})
    result = fast_gelu_v2_compute(data, output_y, kernel_name, impl_mode=impl_mode)

    with tvm.target.cce():
        sch = auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, result]}

    build(sch, config)
