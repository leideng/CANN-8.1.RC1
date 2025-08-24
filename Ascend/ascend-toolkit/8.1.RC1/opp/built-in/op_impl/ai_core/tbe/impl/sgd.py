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
sgd
"""
import tbe.dsl as tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_apply_op_schedule import ApplyOpConfig
from impl.util.util_apply_op_schedule import common_apply_op_process


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,too-many-locals
@register_operator_compute("sgd", op_mode="static", support_fusion=True)
def sgd_compute(parameters,
                gradient,
                learning_rate,
                accum,
                momentum,
                stat,
                update,
                dampening,
                weight_decay,
                nesterov,
                kernel_name="sgd"):
    """Update '*parameters' according to the SGD algorithm.

    accum = accum * momentum + grad
    if use_nesterov is True:
        parameters -= grad * lr + accum * momentum * lr
    else:
        parameters -= accum * lr

    Parameters:
    ----------
    parameters : mutable tensor parameters.

    gradient : tensor grad.

    learning_rate : scalar lr.

    accum: mutable tensor accum.

    momentum : scalar momentum.

    stat : mutable tensor stat.

    update: out dict.

    dampening: (float, optional): dampening for momentum (default: 0)

    weight_decay: weight decay (L2 penalty) (default: 0)

    nesterov: bool. If true, use nesterov computing grad,
    default value is False.

    kernel_name : cce kernel name, default value is "sgd" (optional).

    Returns:
    -------
    outs
    """
    dtype = parameters.dtype
    support = tbe_platform.api_check_support("tbe.dsl.vmuls", "float32")
    if not support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'parameters', [], dtype)
    if dtype in ("float16", "bfloat16"):
        parameters = tbe.cast_to(parameters, "float32")
        accum = tbe.cast_to(accum, "float32")
        learning_rate = tbe.cast_to(learning_rate, "float32")
        gradient = tbe.cast_to(gradient, "float32")
        momentum = tbe.cast_to(momentum, "float32")
        stat = tbe.cast_to(stat, "float32")

    # if weight_decay != 0.0
    if weight_decay != 0.0:
        parameters = tbe.vmuls(parameters, tvm.const(1.0, 'float32'))
        grad_delta = tbe.vmuls(parameters, weight_decay)
        gradient = tbe.vadd(gradient, grad_delta)

    stat_mid = tbe.vmuls(stat, tvm.const(-1, "float32"))
    stat_act = tbe.vadds(stat_mid, tvm.const(1, "float32"))

    dampening_t = tbe.vmuls(stat_act, dampening)

    # update accum
    accum_delta = tvm.compute(accum.shape, lambda *indice: accum(*indice) * momentum[0], tag='elewise_single_VS_mul')

    gradient_damp = tbe.vmul(gradient, dampening_t)
    accum_t = tbe.vadd(accum_delta, gradient)
    if dampening != 0.0:
        accum_t = tbe.vsub(accum_t, gradient_damp)

    # update parameters
    if nesterov:
        parameters_delta = tvm.compute(gradient.shape,
                                       lambda *indice: gradient(*indice) * learning_rate[0],
                                       tag='elewise_single_VS_mul')
        parameters_delta_2 = tvm.compute(accum_t.shape,
                                         lambda *indice: accum_t(*indice) * momentum[0],
                                         tag='elewise_single_VS_mul')
        parameters_delta_2 = tvm.compute(parameters_delta_2.shape,
                                         lambda *indice: parameters_delta_2(*indice) * learning_rate[0],
                                         tag='elewise_single_VS_mul')
        parameters_delta = tbe.vadd(parameters_delta, parameters_delta_2)
        parameters_t = tbe.vsub(parameters, parameters_delta)
    else:
        parameters_delta = tvm.compute(accum_t.shape,
                                       lambda *indice: accum_t(*indice) * learning_rate[0],
                                       tag='elewise_single_VS_mul')
        parameters_t = tbe.vsub(parameters, parameters_delta)

    # update stat
    stat_t = tbe.vmuls(stat_act, tvm.const(0.0, 'float32'))

    if dtype == "float16":
        parameters_t = tbe.cast_to(parameters_t, "float16")
        accum_t = tbe.cast_to(accum_t, "float16")
        stat_t = tbe.cast_to(stat_t, "float16")
    if dtype == "bfloat16":
        parameters_t = tbe.round(parameters_t, "bfloat16")
        accum_t = tbe.round(accum_t, "bfloat16")
        stat_t = tbe.round(stat_t, "bfloat16")

    if not tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322s32f"):
        momentum = tbe.cast_to(momentum, "float16")
    momentum_floor = tbe.floor(momentum)
    momentum_ceil = tbe.ceil(momentum)

    accum_out = tvm.compute(
        accum_t.shape,
        lambda *indice:
            tvm.select(
                tvm.any(momentum_floor[0] != 0, momentum_ceil[0] != 0),
                accum_t(*indice)
            ),
        name="accum_out"
    )

    parameters_out = tvm.compute(parameters_t.shape, lambda *indice: parameters_t(*indice), name="parameters_out")

    stat_out = tvm.compute(
        stat_t.shape,
        lambda *indice:
            tvm.select(
                tvm.any(momentum_floor[0] != 0, momentum_ceil[0] != 0),
                stat_t(*indice)
            ),
        name="stat_out"
    )

    update = tvm.compute(parameters_t.shape, lambda *indice: parameters_t(*indice), name="update")

    outputs = tvm.compute(
        parameters_t.shape,
        lambda *indice: accum_out(*indice) + parameters_out(*indice) + stat_out(*indice) + update(*indice)
    )

    outs =  (accum_out, parameters_out, stat_out, update, outputs)
    return outs


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sgd(parameters,
        gradient,
        learning_rate,
        accum,
        momentum,
        stat,
        update,
        dampening,
        weight_decay,
        nesterov,
        kernel_name="sgd"):
    """Update '*parameters' according to the SGD algorithm.

    accum = accum * momentum + grad
    if use_nesterov is True:
        parameters -= grad * lr + accum * momentum * lr
    else:
        parameters -= accum * lr

    Parameters:
    ----------
    parameters : mutable tensor parameters.

    gradient : tensor grad.

    learning_rate : scalar lr.

    accum: mutable tensor accum.

    momentum : scalar momentum.

    stat : mutable tensor stat.

    update: out dict.

    dampening: (float, optional): dampening for momentum (default: 0)

    weight_decay: weight decay (L2 penalty) (default: 0)

    nesterov: bool. If true, use nesterov computing grad,
    default value is False.

    kernel_name : cce kernel name, default value is "sgd" (optional).

    Returns:
    -------
    None
    """
    if nesterov and dampening != 0:
        error_manager_vector.raise_err_check_params_rules(kernel_name,
                                                          "nesterov requires zero dampening when nesterov is true",
                                                          'dampening', dampening)
    if weight_decay < 0:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "weight_decay can not be less than 0",
                                                          'weight_decay', weight_decay)

    input_dict = (parameters, gradient, learning_rate, accum, momentum, stat)
    args = ApplyOpConfig.TensorArgs(
        input_dict,
        sgd_compute,
        update,
        17 if nesterov else 9,
    )

    name = ApplyOpConfig.TensorName(all=('parameters', 'gradient', 'learning_rate', 'accum', 'momentum', 'stat'),
                                    scalar=('learning_rate', 'momentum'),
                                    reuse=('accum', 'parameters', 'stat'))
    options = ApplyOpConfig.TensorOptions(attrs=[dampening, weight_decay, nesterov],
                                          build=None, dtype=("float16", "float32", "bfloat16"), optype="SGD")
    common_apply_op_process(ApplyOpConfig(args, name, options), kernel_name)
