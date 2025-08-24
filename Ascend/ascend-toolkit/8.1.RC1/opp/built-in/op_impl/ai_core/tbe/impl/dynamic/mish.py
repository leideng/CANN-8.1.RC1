#!/usr/bin/python
# -*- coding: utf-8 -*-
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
mish
"""
import functools

from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpTbeImplMode


# 'pylint: disable=unused-argument
@register_operator_compute("Mish", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def mish_compute(x, y, kernel_name="mish", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: mish
    calculating data's mish,y= x*(1 - 2/(1+(1+exp(x))^2))

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input data
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is mish

    Returns
    -------
    res : tvm.tensor
        the result of mish
    """
    x_shape = x.shape
    x_dtype = x.dtype.lower()
    if impl_mode == OpImplMode.SUPER_PERFORMANCE:
        """
        `x*(1-2/(1+(1+(1+x/64)^64)^2))`
        """
        res_1 = tbe.vadds(tbe.vmuls(x, 0.015625), 1)
        res_pow_2 = tbe.vmul(res_1, res_1)
        res_pow_4 = tbe.vmul(res_pow_2, res_pow_2)
        res_pow_8 = tbe.vmul(res_pow_4, res_pow_4)
        res_pow_16 = tbe.vmul(res_pow_8, res_pow_8)
        res_pow_32 = tbe.vmul(res_pow_16, res_pow_16)
        res_pow_64 = tbe.vmul(res_pow_32, res_pow_32)
        res_2 = tbe.vadds(res_pow_64, 1)
        res_3 = tbe.vmul(res_2, res_2)
        res_4 = tbe.vadds(res_3, 1)
        tmp = tbe.broadcast(tvm.const(-2, x_dtype), x_shape)
        res_rec = tbe.vdiv(tmp, res_4)
        res_6 = tbe.vadds(res_rec, 1)
        res = tbe.vmul(res_6, x)
        return res

    dtype = x.dtype.lower()
    is_cast = False
    if dtype in ("float16",) and tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        x = tbe.cast_to(x, "float32")
        dtype = "float32"
        is_cast = True

    # x > 0 : (2e^(-x) + 1) / (2e^(-2x) + 2e^(-x) + 1)
    mask = tbe.vcmp(x, 0, 'gt', mode='bit')
    neg_x_one = tbe.vmuls(x, -1)
    neg_x_two = tbe.vmuls(x, -2)
    exp_val_neg_one = tbe.vexp(neg_x_one)
    exp_val_neg_two = tbe.vexp(neg_x_two)
    two_exp_neg_one = tbe.vmuls(exp_val_neg_one, 2)
    two_exp_neg_two = tbe.vmuls(exp_val_neg_two, 2)
    tmp1 = tbe.vadds(two_exp_neg_one, 1)
    tmp2 = tbe.vadd(two_exp_neg_one, two_exp_neg_two)
    tmp2 = tbe.vadds(tmp2, 1)
    tanh_val_1 = tbe.vdiv(tmp1, tmp2)
    tanh_val_1 = tbe.vsel(mask, tanh_val_1, 0)

    # x <= 0: (2e^x + e^2x) / (2 + 2e^x + e^2x)
    mask = tbe.vcmp(x, 0, 'le', mode='bit')
    two_x = tbe.vmuls(x, 2)
    exp_val = tbe.vexp(x)
    two_exp_val = tbe.vmuls(exp_val, 2)
    exp_val_two = tbe.vexp(two_x)
    tmp1 = tbe.vadd(two_exp_val, exp_val_two)
    tmp2 = tbe.vadds(tmp1, 2)
    tanh_val_2 = tbe.vdiv(tmp1, tmp2)
    tanh_val_2 = tbe.vsel(mask, tanh_val_2, 0)

    tanh_val = tbe.vadd(tanh_val_1, tanh_val_2)
    res = tbe.vmul(x, tanh_val)

    if is_cast:
        res = tbe.cast_to(res, "float16")
    return res


@register_operator("Mish")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def mish(x, y, kernel_name="mish", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: mish
    calculating data's mish,y= x*(1 - 2/(1+(1+exp(x))^2))

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support float16, float32, bfloat16
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is mish

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode,
                       [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION, OpImplMode.SUPER_PERFORMANCE],
                       kernel_name)

    input_dtype = x.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], name="data_input", dtype=input_dtype)
            res = mish_compute(data_input, y, kernel_name, impl_mode)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
