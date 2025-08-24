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
poisson_nll_loss
"""

import operator
import math
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util as tsu


@register_operator_compute("poisson_nll_loss", op_mode="static", support_fusion=True)
# 'pylint: disable=unused-argument,too-many-locals,invalid-name,too-many-arguments
def poisson_nll_loss_compute(input_x, target, loss, log_input=True, full=False, eps=1e-8, reduction="mean", number=0.0):
    """
    possion_nll_loss

    Parameters
    ----------
    input_x: shape dtype
    target: shape dtype
    log_input: scalar parameter, default `value = True``
    full: scalar parameter, dafault value = False
    eps: scalar parameter, dafault value = 1e-8
    reduction: scalar parameter, dafault value = "mean"
    kernel name, default value is "poisson_nll_loss"
    """
    shape_input_x = tsu.shape_to_list(input_x.shape)
    shape_target = tsu.shape_to_list(target.shape)
    shape_input_x, shape_target, shape_max = tsu.broadcast_shapes(shape_input_x, shape_target)
    para_check.check_shape(shape_max)
    input_x = tbe.broadcast(input_x, shape_max)
    target = tbe.broadcast(target, shape_max)

    dtype = input_x.dtype
    shape = input_x.shape

    if dtype.lower() == "float16":
        input_x = tbe.cast_to(input_x, "float32")
        target = tbe.cast_to(target, "float32")

    scalar_param_eps = tvm.const(eps, dtype=input_x.dtype)
    scalar_param_doublepi = tvm.const(2 * math.pi, dtype=input_x.dtype)
    scalar_param_half = tvm.const(0.5, dtype=input_x.dtype)

    tensor_scalar_number = tbe.broadcast(tvm.const(number, dtype=input_x.dtype), [
        1,
    ])
    tensor_scalar_one = tbe.broadcast(tvm.const(1.0, dtype=input_x.dtype), shape)
    tensor_scalar_zero = tbe.broadcast(tvm.const(0, dtype=input_x.dtype), shape)

    if log_input:
        tensor_output = tbe.vsub(tbe.vexp(input_x), tbe.vmul(input_x, target))
    else:
        tensor_logfalse_log = tbe.vlog(tbe.vadds(input_x, scalar_param_eps))
        tensor_output = tbe.vsub(input_x, tbe.vmul(target, tensor_logfalse_log))

    if full:
        tensor_full_double_pitarget = tbe.vmuls(target, scalar_param_doublepi)
        tensoe_full_left = tbe.vmul(target, tbe.vlog(target))
        tensoe_full_right = tbe.vmuls(tbe.vlog(tensor_full_double_pitarget), scalar_param_half)
        tensor_output_full = tbe.vadd(tbe.vsub(tensoe_full_left, target), tensoe_full_right)
        tensor_output_full = tbe.vcmpsel(target, tensor_scalar_one, 'le', tensor_scalar_zero, tensor_output_full)
        tensor_output = tbe.vadd(tensor_output, tensor_output_full)

    if reduction == "none":
        output = tensor_output
    elif reduction == "mean":
        tensor_output = tbe.sum(tensor_output, axis=0, keepdims=True)
        output = tbe.vdiv(tensor_output, tensor_scalar_number)
    else:
        output = tbe.sum(tensor_output, axis=0, keepdims=True)

    if dtype.lower() == "float16":
        output = tbe.cast_to(output, "float16")

    return output


# 'pylint: disable = unused-argument,redefined-builtin,too-many-arguments,invalid-name,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def poisson_nll_loss(input_x,
                     target,
                     loss,
                     log_input=True,
                     full=False,
                     eps=1e-8,
                     reduction="mean",
                     kernel_name="poisson_nll_loss"):
    """
    possion_nll_loss

    Parameters
    ----------
    input_x: shape dtype
    target: shape dtype
    log_input: scalar parameter, default value True
    full: scalar parameter, dafault value False
    eps: scalar parameter, dafault value 1e-8
    reduction: scalar parameter, dafault value "mean"
    kernel name, default value is "poisson_nll_loss"
    """
    shape_input = input_x.get("shape")
    shape_target = target.get("shape")
    dtype_input = input_x.get("dtype")
    dtype_target = target.get("dtype")

    para_check.check_kernel_name(kernel_name)

    para_check.check_shape_rule(shape_input)
    para_check.check_shape_rule(shape_target)
    if not operator.eq(shape_input, shape_target):
        raise RuntimeError("All input shape must be equal.")
    shape_input, _ = tsu.refine_shape_axes(shape_input, [])
    shape_target, _ = tsu.refine_shape_axes(shape_target, [])
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input.lower(), check_list)
    para_check.check_dtype(dtype_target.lower(), check_list)

    if dtype_input.lower() != dtype_target.lower():
        raise RuntimeError("All input dtype must be same.")

    if reduction != "none" and reduction != "mean" and reduction != "sum":
        raise RuntimeError("Invalid reduction which should be none, mean or sum.")

    if eps == 0:
        raise RuntimeError("Invalid eps which should not be zero.")

    dtype = dtype_input.lower()
    data_input = tvm.placeholder(shape_input, name="data_input", dtype=dtype)
    data_target = tvm.placeholder(shape_target, name="data_target", dtype=dtype)

    output = poisson_nll_loss_compute(data_input, data_target, loss, log_input, full, eps, reduction,
                                      tvm.all(shape_input)[0])

    # auto schedule
    with tvm.target.cce():
        schedule = auto_schedule(output)

    config = {"name": kernel_name, "print_ir": False, "tensor_list": [data_input, data_target, output]}

    build(schedule, config)
