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
fake_quant_with_min_max_args
"""
import functools

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-locals,unused-variable
@register_operator_compute("fake_quant_with_min_max_args", op_mode="static", support_fusion=True)
def fake_quant_with_min_max_args_compute(x, y, min=-6, max=6, num_bits=8,
                                         narrow_range=False,
                                         kernel_name="fake_quant_with_min_"
                                                     "max_args"):
    """
    Computes Fake-quantize the 'x' tensor,
    type float32 to 'y' tensor of same type
    calculating data's :
    y = (floor(clamped_shifted * inv_nudged_scale + 0.5f)))*scale+nudged_min
    scale=(max-min)/(quant_max-quant_min)

    Parameters
    ----------
    x: TVM tenor
        the placeholder of input data,type is float32
    y: dict
        the dict of output data
    min: scalar float int
        Defaults to -6
    max: scalar float int
        Defaults to 6
        [min; max] define the clamping range for the x data
    num_bits: float int
        Defaults to 8.num_bits is the bitwidth of the quantization,
        between 2 and 16
    narrow_range: bool
        True or False.if None,narrow_range=False
        if True x values are quantized into the quantization range
        [1; 2^num_bits - 1]
        if False x values are quantized into the quantization range
        [0; 2^num_bits - 1]
    kernel_name: str
        cce kernel name, default value is "fake_quant_with_min_max_args"

    Returns
    -------
    res: TVM tensor
        the result of fake_quant_with_min_max_args_compute
    """
    shape_x = shape_util.shape_to_list(x.shape)
    output_dtype = x.dtype

    nudged_min, nudged_max, scale = _nudge_min_max(min, max, num_bits,
                                                   narrow_range)

    if tbe_platform.api_check_support("tbe.dsl.vmaxs", x.dtype):
        nudged_min_neg = nudged_min * (-1.0)
        inv_nudged_scale = 1.00 / scale

        # Transform the input between nudged_max and nudged_min
        clamped_vmin = tbe.vmins(x, nudged_max)
        clamped = tbe.vmaxs(clamped_vmin, nudged_min)

        # Calculate the quantized and dequantized results
        clamped_shifted = tbe.vadds(clamped, nudged_min_neg)
        vmul_shifted = tbe.vmuls(clamped_shifted, inv_nudged_scale)
        vadds_shifted = tbe.vadds(vmul_shifted, tvm.const(0.5, dtype="float32"))

        vadds_shifted = tbe.cast_to(vadds_shifted, "float16")
        floor_vadds_shifted = tbe.floor(vadds_shifted)
        floor_cast = tbe.cast_to(floor_vadds_shifted, output_dtype)
        res_scale = tbe.vmuls(floor_cast, scale)
        res = tbe.vadds(res_scale, nudged_min)

    else:
        zero_tensor = tbe.vmuls(x, 0)
        nudged_max_tensor = tbe.vadds(zero_tensor, nudged_max)
        nudged_min_tensor = tbe.vadds(zero_tensor, nudged_min)
        inv_nudged_scale = 1.00 / scale
        inv_nudged_scale_const = tvm.const(inv_nudged_scale, dtype=output_dtype)

        # Transform the input between nudged_max and nudged_min
        clamped_vmin = tbe.vmin(x, nudged_max_tensor)
        clamped = tbe.vmax(clamped_vmin, nudged_min_tensor)

        # Calculate the quantized and dequantized results
        clamped_shifted = tbe.vsub(clamped, nudged_min_tensor)
        vmul_shifted = tbe.vmuls(clamped_shifted, inv_nudged_scale_const)
        vadds_shifted = tbe.vadds(vmul_shifted, tvm.const(0.5, dtype="float32"))
        floor_vadds_shifted = tbe.floor(vadds_shifted)
        floor_cast = tbe.cast_to(floor_vadds_shifted, output_dtype)
        res_scale = tbe.vmuls(floor_cast, scale)
        res = tbe.vadd(res_scale, nudged_min_tensor)

    return res


def _nudge_min_max(min, max, num_bits, narrow_range):
    """
    Calculate the maximum and minimum values of the quantization

    Parameters
    ----------
    min: scalar
        input min
    max: TVM tenor
        input max
    num_bits: scalar
        Defaults to 8.num_bits is the bitwidth of the quantization,
        between 2 and 16
    narrow_range: bool

    Returns
    -------
    res: nudged_min, nudged_max, scale
    """
    quant_max = (2**num_bits) - 1

    if narrow_range is False:
        quant_min = 0.00
    else:
        quant_min = 1.00

    scale = (max - min) / (float(quant_max) - quant_min)

    zeor_point_from_min = quant_min - min / scale

    # Calculate the maximum and minimum values of the quantization
    if zeor_point_from_min < quant_min:
        nudged_zero_point = quant_min
    elif zeor_point_from_min > quant_max:
        nudged_zero_point = quant_max
    else:
        nudged_zero_point = (zeor_point_from_min + 0.5) // 1

    nudged_min = (quant_min - nudged_zero_point) * scale
    nudged_max = (quant_max - nudged_zero_point) * scale

    return nudged_min, nudged_max, scale


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT),
                            (para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT),
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def fake_quant_with_min_max_args(x, y, min=-6, max=6, num_bits=8,
                                 narrow_range=False, kernel_name="fake_quant_"
                                                                 "with_min_max_args"):
    """
    Computes Fake-quantize the 'x' tensor,
    type float32 to 'y' tensor of same type
    calculating data's :
    y = (floor(clamped_shifted * inv_nudged_scale + 0.5f)))*scale+nudged_min
    scale=(max-min)/(quant_max-quant_min)

    Parameters
    ----------
    x: dict
        shape and dtype of input,only support float32
    y: dict
        the dict of output data
    min: scalar float int
        Defaults to -6
    max: scalar float int
        Defaults to 6
        [min; max] define the clamping range for the x data
    num_bits: float int
        Defaults to 8.num_bits is the bitwidth of the quantization,
        between 2 and 16
    narrow_range: bool or None
        True or False.if None,narrow_range=False
        if True x values are quantized into the quantization range
        [1; 2^num_bits - 1]
        if False x values are quantized into the quantization range
        [0; 2^num_bits - 1]
    kernel_name: str
        cce kernel name, default value is "fake_quant_with_min_max_args"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    para_check.check_shape(shape_x, param_name="x")
    input_dtype = x.get("dtype").lower()
    para_check.check_dtype(input_dtype, ["float32"], param_name="x")
    if min >= max:
        excepted_value = "min must be less than max"
        real_value = "min is (%d) and max is (%d)" % (min, max)
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "min", excepted_value, real_value)
    if num_bits < 2 or num_bits > 16:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, "num_bits", "2", "16", num_bits)
    shape_x = (functools.reduce(lambda x, y: x * y, shape_x[:]),)
    x = tvm.placeholder(shape_x, name="x", dtype=input_dtype)
    res = fake_quant_with_min_max_args_compute(x, y, float(min), float(max),
                                               num_bits, narrow_range,
                                               kernel_name)

    with tvm.target.cce():
        auto_sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [x, res]}
    build(auto_sch, config)
