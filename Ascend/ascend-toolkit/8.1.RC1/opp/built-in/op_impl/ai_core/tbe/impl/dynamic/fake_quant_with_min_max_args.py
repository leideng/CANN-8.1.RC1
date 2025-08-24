#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=locally-disabled,huawei-too-many-arguments,too-many-arguments,unused-argument,invalid-name
# 'pylint: redefined-builtin,too-many-locals,unused-variable
@register_operator_compute("FakeQuantWithMinMaxArgs", op_mode="dynamic", support_fusion=False)
def fake_quant_with_min_max_args_compute(x, y, min_value=-6.0, max_value=6.0, num_bits=8,
                                         narrow_range=False,
                                         kernel_name="fake_quant_with_min_max_args"):
    """
    Computes Fake-quantize the 'x' tensor,
    type float32 to 'y' tensor of same type
    calculating data's :
    `y = (floor(clamped_shifted * inv_nudged_scale_const + 0.5f))) * scale + nudged_min`
    `scale = (max_value - min_value) / (quant_max - quant_min)`
    `quant_max = (2**num_bits) - 1`
    `quant_min = 1.0 if narrow_range else 0.0`
    `inv_nudged_scale_const = 1 / scale`
    `clamped_shifted = max(min(x, nudged_max), nudged_min) - nudged_min`
    `zero_point_from_min = quant_min - min_value / scale`
    `nudged_zero_point = quant_min if zero_point_from_min < quant_min else`
                        `quant_max if zero_point_from_min > quant_max else round(zero_point_from_min)`
    `nudged_min = (quant_min - nudged_zero_point) * scale`
    `nudged_max = (quant_max - nudged_zero_point) * scale`


    Parameters
    ----------
    x: TVM tenor
        the placeholder of input data,type is float32
    y: dict
        the dict of output data
    min_value: scalar float
        Defaults to -6
    max_value: scalar float
        Defaults to 6
        [min_value; max_value] define the clamping range for the x data
    num_bits: int
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

    nudged_min, nudged_max, scale = _nudge_min_max(min_value, max_value, num_bits,
                                                   narrow_range)

    zero_tensor = tbe.vmuls(x, 0)
    nudged_max_tensor = tbe.vadds(zero_tensor, nudged_max)
    nudged_min_tensor = tbe.vadds(zero_tensor, nudged_min)
    inv_nudged_scale = 1.00 / scale
    inv_nudged_scale_const = tvm.const(inv_nudged_scale, dtype=output_dtype)

    # Transform the input between nudged_max and nudged_min
    if tbe_platform.api_check_support("te.lang.cce.vmaxs", x.dtype):
        clamped_vmin = tbe.vmins(x, nudged_max)
        clamped = tbe.vmaxs(clamped_vmin, nudged_min)
    else:
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


def _nudge_min_max(min_value, max_value, num_bits, narrow_range):
    """
    Calculate the maximum and minimum values of the quantization

    Parameters
    ----------
    min_value: scalar
        input min
    max_value: TVM tenor
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
    quant_min = 0.00 if narrow_range is False else 1.00

    scale = (max_value - min_value) / (float(quant_max) - quant_min)

    zeor_point_from_min = quant_min - min_value / scale

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


@register_operator("FakeQuantWithMinMaxArgs")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def fake_quant_with_min_max_args(x, y, min=-6.0, max=6.0, num_bits=8,
                                 narrow_range=False, kernel_name="fake_quant_with_min_max_args"):
    """
    Computes Fake-quantize the 'x' tensor,
    type float32 to 'y' tensor of same type
    calculating data's :
    `y = (floor(clamped_shifted * inv_nudged_scale_const + 0.5f))) * scale + nudged_min`
    `scale = (max_value - min_value) / (quant_max - quant_min)`
    `quant_max = (2**num_bits) - 1`
    `quant_min = 1.0 if narrow_range else 0.0`
    `inv_nudged_scale_const = 1 / scale`
    `clamped_shifted = max(min(x, nudged_max), nudged_min) - nudged_min`
    `zero_point_from_min = quant_min - min_value / scale`
    `nudged_zero_point = quant_min if zero_point_from_min < quant_min else`
                        `quant_max if zero_point_from_min > quant_max else round(zero_point_from_min)`
    `nudged_min = (quant_min - nudged_zero_point) * scale`
    `nudged_max = (quant_max - nudged_zero_point) * scale`

    Parameters
    ----------
    x: dict
        shape and dtype of input,only support float32
    y: dict
        the dict of output data
    min_value: scalar float
        Defaults to -6
    max_value: scalar float
        Defaults to 6
        [min_value; max_value] define the clamping range for the x data
    num_bits: int
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
    min_value = min
    max_value = max
    shape_x = x.get("shape")
    para_check.check_shape(shape_x, param_name="x")
    input_dtype = x.get("dtype").lower()
    para_check.check_dtype(input_dtype, ["float32"], param_name="x")

    if min_value >= max_value:
        excepted_value = "Min must be less than max."
        real_value = "Min is (%2f) and max is (%2f)." % (min_value, max_value)
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "min", excepted_value, real_value)
    if num_bits < 2 or num_bits > 16:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, "num_bits", "2", "16", num_bits)

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x])
            data_x = tvm.placeholder(shape_x[0], name="data_x", dtype=input_dtype)
            res = fake_quant_with_min_max_args_compute(data_x, y, min_value, max_value,
                                                       num_bits, narrow_range,
                                                       kernel_name)
            tensors.append([data_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
