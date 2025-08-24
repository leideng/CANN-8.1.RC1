#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

weights adaptive range quantization
"""
from __future__ import absolute_import
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648


# 'pylint: disable=invalid-name, too-many-locals
def wts_arq_compute(w, w_min, w_max, num_bits, offset_flag):
    """
    wts_arq compute
    do fake quantize for weights

    Parameters
    ----------
    w: TVM tensor
        the placeholder of w
    w_min: TVM tensor
        the placeholder of w_min
    w_max: TVM tensor
        the placeholder of w_max
    num_bits: int
        the bits num used for quantize
    offset_flag: bool
        whether use offset for quantize

    Returns
    -------
    y: TVM tensor
        the fake quantized weights
    """
    w_type = w.dtype
    shape_w = shape_util.shape_to_list(w.shape)

    const_0 = tvm.const(0.0, w_type)
    w_min = tbe.vmins(w_min, const_0)
    w_max = tbe.vmaxs(w_max, const_0)

    # const defination
    eps = 1.1920929e-07
    const_eps = tbe.broadcast(tvm.const(eps, w_type), w_min.shape, w_type)
    const_1 = tbe.broadcast(tvm.const(1.0, w_type), w_min.shape, w_type)

    if offset_flag:
        # multiply w_max/w_min with const_step_reciprocal firstly,
        # incase (w_max - w_min) overflow float16
        const_step_reciprocal = tvm.const(1.0 / (2 ** num_bits - 1), w_type)
        scale_upper_bound = tbe.vmuls(w_max, const_step_reciprocal)
        scale_low_bound = tbe.vmuls(w_min, const_step_reciprocal)
        scale = tbe.vsub(scale_upper_bound, scale_low_bound)
        scale = tbe.vcmpsel(scale, const_eps, operation='lt', slhs=const_1, srhs=scale)

        offset = tbe.round(tbe.vdiv(w_min, scale))
        offset = tbe.vmuls(offset, tvm.const(-1, w_type))
        const_minus_bias = tvm.const(-1 * (2 ** (num_bits - 1)), w_type)
        offset = tbe.vadds(offset, const_minus_bias)
    else:
        const_step_low = tvm.const(2 ** (num_bits - 1), w_type)
        const_step_high = tvm.const(2 ** (num_bits - 1) - 1, w_type)

        step_low = tbe.broadcast(const_step_low, w_max.shape, w_type)
        step_upper = tbe.broadcast(const_step_high, w_max.shape, w_type)

        scale_1 = tbe.vdiv(tbe.vabs(w_min), step_low)
        scale_2 = tbe.vdiv(w_max, step_upper)
        scale = tbe.vmax(scale_1, scale_2)
        scale = tbe.vcmpsel(scale, const_eps, operation='lt', slhs=const_1, srhs=scale)

    scale_bc = tbe.broadcast(scale, shape_w, w_type)
    y = tbe.vdiv(w, scale_bc)
    y = tbe.round(y)
    if offset_flag:
        offset_bc = tbe.broadcast(offset, shape_w, w_type)
        y = tbe.vadd(y, offset_bc)
    const_int8_low = tvm.const(-1 * 2 ** (num_bits - 1), y.dtype)
    const_int8_high = tvm.const(2 ** (num_bits - 1) - 1, y.dtype)
    y = tbe.vmaxs(y, const_int8_low)
    y = tbe.vmins(y, const_int8_high)
    if offset_flag:
        y = tbe.vsub(y, offset_bc)
    y = tbe.cast_to(y, scale_bc.dtype)
    y = tbe.vmul(y, scale_bc)

    return y


# 'pylint: disable=invalid-name, too-many-locals, too-many-arguments
@para_check.check_input_type(dict, dict, dict, dict, int, bool, str)
def wts_arq(w,
            w_min,
            w_max,
            y,
            num_bits=8,
            offset_flag=False,
            kernel_name="wts_arq"):
    """
    algorithm: weights adaptive range quantization
    get scale and offset, do fake quantize for weights

    Parameters
    ----------
    w: dict
        dict with keys(shape and dtype) of w
    w_min: dict
        dict with keys(shape and dtype) of w_min
    w_max: dict
        dict with keys(shape and dtype) of w_max
    y: dict
        dict with keys(shape and dtype) of y
    num_bits: int
        the bits num used for quantize
    offset_flag: bool
        whether use offset for quantize
    kernel_name : str
        cce kernel name, default value is "wts_arq"

    Returns
    -------
    None
    """
    shape_w = w.get("shape")
    para_check.check_shape_rule(shape_w)
    para_check.check_shape_size(shape_w, SHAPE_SIZE_LIMIT)

    check_tuple = ("float16", "float32")
    w_type = w.get("dtype").lower()
    para_check.check_dtype_rule(w_type, check_tuple)

    w_min_type = w_min.get("dtype").lower()
    w_max_type = w_max.get("dtype").lower()
    if w_min_type != w_max_type:
        raise ValueError("w_min type:'{}' is different from w_max type:'{}'".format(w_min_type, w_max_type))
    if w_min_type != w_type:
        raise ValueError("w_min type:'{}' is different from w type:'{}'".format(w_min_type, w_type))

    shape_w_min = w_min.get("shape")
    shape_w_max = w_max.get("shape")
    if shape_w_min != shape_w_max:
        raise ValueError("w_min shape:'{}' is different from w_max shape:'{}'".format(shape_w_min, shape_w_max))

    if len(shape_w_min) != len(shape_w):
        raise ValueError(
            "w_min shape dim size:'{}' is different from w shape dim size:'{}'".format(len(shape_w_min), len(shape_w)))

    # 'pylint: disable=consider-using-enumerate
    for i in range(len(shape_w_min)):
        if shape_w_min[i] != 1 and shape_w_min[i] != shape_w[i]:
            raise ValueError(
                "w_min shape dim[{0}]:'{1}' should be equal to w shape dim[{0}]:'{2}'".format(i, shape_w_min[i],
                                                                                              shape_w[i]))

    if num_bits != 8:
        raise ValueError(
            "num_bits can only be 8 in current implementation, but is {}".format(num_bits))

    para_check.check_kernel_name(kernel_name)

    weights = tvm.placeholder(shape_w, name="w", dtype=w_type)
    weights_min = tvm.placeholder(w_min.get("shape"), name="w_min", dtype=w_min.get("dtype"))
    weights_max = tvm.placeholder(w_max.get("shape"), name="w_max", dtype=w_max.get("dtype"))
    y = wts_arq_compute(weights, weights_min, weights_max, num_bits,
                        offset_flag)

    with tvm.target.cce():
        schedule = auto_schedule(y)

    config = {
        "print_ir": False,
        "name": kernel_name,
        "bool_storage_as_1bit": True,
        "tensor_list": (weights, weights_min, weights_max, y)
    }
    build(schedule, config)
