#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
celu_v2
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_attr_common import CeluV2AttrInfo


# 'pylint:disable=too-many-arguments,too-many-locals,unused-argument
def celu_v2_compute(x, y, alpha=1.0, kernel_name="celu_v2"):
    """
    Implement the operator by referring to  the
            TBE Operator Development Guide.
    scale * (max(0, x) + min(0, alpha * (exp(x/input_scale) - 1)
    x:dict of x, include shape and data_type
    y:dict of y, include shape and data_type
    alpha: attr, alpha of the min
    input_scale: attr, input scale of the input

    """
    data_type = x.dtype
    high_perf = False
    if tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        high_perf = True

    type_fp16 = "float16"
    type_fp32 = "float32"
    type_bf16 = "bfloat16"

    if (data_type.lower() == type_fp16 and high_perf) or (data_type.lower() == type_bf16):
        compute_type = type_fp32
        x = tbe.cast_to(x, type_fp32)
    else:
        compute_type = data_type

    # apply min part of formula
    alpha_scalar = get_attr_by_cls(alpha, CeluV2AttrInfo.ATTR_ALPHA, compute_type)
    x_mul_rec_input_scale = tbe.vmuls(x, 1.0/alpha_scalar)
    exp_x_mul_rec_input_scale = tbe.vexp(x_mul_rec_input_scale)
    exp_x_mul_rec_input_scale_minus1 = tbe.vadds(exp_x_mul_rec_input_scale, tvm.const(-1.0, compute_type))
    alpha_times_exp_minus1 = tbe.vmuls(exp_x_mul_rec_input_scale_minus1, alpha_scalar)
    vmin_x = tbe.vmins(alpha_times_exp_minus1, tvm.const(0.0, compute_type))

    # apply max part of formula
    vmax_x = tbe.vmaxs(x, tvm.const(0.0, compute_type))

    # add min max
    result = tbe.vadd(vmax_x, vmin_x)

    if data_type.lower() == type_fp16 and high_perf:
        result = tbe.cast_to(result, data_type)
    elif data_type.lower() == type_bf16:
        result = tbe.round(result, data_type)

    return result


# 'pylint: disable=redefined-builtin
@register_operator("CeluV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def celu_v2(x, y, alpha=1.0, kernel_name="celu_v2"):
    """
    algorithm: celu_v2

    scale * (max(0, x) + min(0, alpha * (exp(x/input_scale) - 1)

    Parameters
    ----------
    x: dict
        shape and dtype of input
    y: dict
        shape and dtype of output, should be same shape and type as input
    alpha: float
        alpha of the min
    kernel_name : str
        cce kernel name, default value is celu_v2

    Returns
    -------
    None
    """
    x_type_ = x.get("dtype")
    dtype_input_ = x_type_.lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_input_, check_list, param_name="x")
    if alpha == 0:
        raise ZeroDivisionError("alpha is zero, zero division error occur!")
    ins = classify([x], OpPatternMode.ELEWISE)
    schedules = []
    tensors = []
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], name="data_input",
                                         dtype=dtype_input_)
            res = celu_v2_compute(data_input, y, alpha=alpha, kernel_name=kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
