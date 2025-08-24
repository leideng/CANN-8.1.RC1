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
fake_quant_with_min_max_args_gradient
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
@register_operator_compute("FakeQuantWithMinMaxArgsGradient", op_mode="dynamic", support_fusion=False)
def fake_quant_with_min_max_args_gradient_compute(gradients, x, y,
                                                  min_value=-6.0, max_value=6.0, num_bits=8,
                                                  narrow_range=False,
                                                  kernel_name="fake_quant_with_min_max_args_gradient"):
    """
    Compute gradients for a FakeQuantWithMinMaxArgs operation.
    calculating data's :
    `y = gradients * (if x>=nudged_min and <=nudged_max 1 else 0)`
    `quant_max = (2**num_bits) - 1`
    `quant_min = 1.0 if narrow_range else 0.0`
    `zero_point_from_min = quant_min - min_value / scale`
    `scale = (max_value - min_value) / (quant_max - quant_min)`
    `nudged_zero_point = quant_min if zero_point_from_min < quant_min else`
                        `quant_max if zero_point_from_min > quant_max else round(zero_point_from_min)`
    `nudged_min = (quant_min - nudged_zero_point) * scale`
    `nudged_max = (quant_max - nudged_zero_point) * scale`

    Parameters
    ----------
    gradients: TVM rensor
        the placeholder of input data,type is float32,
        Backpropagated gradients above the FakeQuantWithMinMaxArgs operation
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
        the result of fake_quant_with_min_max_args_gradient_compute
    """
    shape_x = shape_util.shape_to_list(x.shape)
    output_dtype = x.dtype
    nudged_min, nudged_max = _nudge_min_max_gradient(min_value, max_value, num_bits,
                                                     narrow_range)

    # where((x<=nudged_max)&(x>=nudged_min),1,0),Convert the input to 0 and 1 tensor
    between_nudged_min_max = _cmpare_value(x, nudged_min, nudged_max)

    res = tbe.vmul(gradients, between_nudged_min_max)

    return res


def _nudge_min_max_gradient(min_value, max_value, num_bits, narrow_range):
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
   res: nudged_min, nudged_max
   """
    quant_max = (2 ** num_bits) - 1

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

    return nudged_min, nudged_max


def _cmpare_value(x, nudged_min, nudged_max):
    """
    where `((x<=nudged_max) & (x>=nudged_min), 1, 0)`

    Parameters
    ----------
    x: TVM rensor
        Input data
    nudged_min: TVM tenor
        Minimum value of comparison
    nudged_max: TVM rensor
        Maximum value of comparison

    Returns
    -------
    res: TVM tensor
        the result of f_cmpare_value
    """
    min_value = tvm.const(2 ** (-126), dtype="float32")
    # `(2**(-126))*(2**(62))*(2**(62))*(2**(2)) = 1`
    # so `min_value*max_value*max_value*max_value_one = 1`
    max_value = tvm.const(2 ** (62), dtype="float32")
    max_value_one = tvm.const(2 ** (2), dtype="float32")

    data_zero = tbe.vmuls(x, 0)
    max_value_tensor = tbe.vadds(data_zero, max_value)
    min_value_tensor = tbe.vadds(data_zero, min_value)
    max_value_one_tensor = tbe.vadds(data_zero, max_value_one)
    nudged_max_tensor = tbe.vadds(data_zero, nudged_max)
    nudged_min_tensor = tbe.vadds(data_zero, nudged_min)

    sub_tmp = tbe.vsub(x, nudged_min_tensor)
    sub_min = tbe.vadds(sub_tmp, min_value)
    if tbe_platform.api_check_support("te.lang.cce.vmaxs", x.dtype):
        vmax_tmp = tbe.vmaxs(sub_min, tvm.const(0, sub_min.dtype))
    else:
        vmax_tmp = tbe.vmax(sub_min, data_zero)

    sub_tmp_max = tbe.vsub(nudged_max_tensor, x)
    sub_max = tbe.vadds(sub_tmp_max, min_value)
    if tbe_platform.api_check_support("te.lang.cce.vmaxs", x.dtype):
        vmin_tmp = tbe.vmaxs(sub_max, tvm.const(0, sub_min.dtype))
    else:
        vmin_tmp = tbe.vmax(sub_max, data_zero)

    one_tmp = tbe.vmul(vmax_tmp, vmin_tmp)
    if tbe_platform.api_check_support("te.lang.cce.vmaxs", x.dtype):
        one_min = tbe.vmins(one_tmp, min_value)
    else:
        one_min = tbe.vmin(one_tmp, min_value_tensor)

    vmul_max_value = tbe.vmul(one_min, max_value_tensor)
    vmul_max_value_one = tbe.vmul(vmul_max_value, max_value_tensor)
    between_nudged_min_max = tbe.vmul(vmul_max_value_one, max_value_one_tensor)

    return between_nudged_min_max


@register_operator("FakeQuantWithMinMaxArgsGradient")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def fake_quant_with_min_max_args_gradient(gradients, x, y, min=-6.0,
                                          max=6.0, num_bits=8, narrow_range=False,
                                          kernel_name="fake_quant_with_min_max_args"):
    """
    Compute gradients for a FakeQuantWithMinMaxArgs operation.
    calculating data's :
    `y = gradients * (if x>=nudged_min and <=nudged_max 1 else 0)`
    `quant_max = (2**num_bits) - 1`
    `quant_min = 1.0 if narrow_range else 0.0`
    `zero_point_from_min = quant_min - min_value / scale`
    `scale = (max_value - min_value) / (quant_max - quant_min)`
    `nudged_zero_point = quant_min if zero_point_from_min < quant_min else`
                        `quant_max if zero_point_from_min > quant_max else round(zero_point_from_min)`
    `nudged_min = (quant_min - nudged_zero_point) * scale`
    `nudged_max = (quant_max - nudged_zero_point) * scale`

    Parameters
    ----------
    gradients:dict
              shape and dtype of input gradients,only support float32
    x: dict
        shape and dtype of input x,only support float32
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
        True or False
        if True x values are quantized into the quantization range
        [1; 2^num_bits - 1]
        if False x values are quantized into the quantization range
        [0; 2^num_bits - 1]
    kernel_name: str
        cce kernel name, default value is
        "fake_quant_with_min_max_args_gradient"

    Returns
    -------
    None
    """
    min_value = min
    max_value = max
    shape_gradients = gradients.get("shape")
    shape_x = x.get("shape")
    if len(shape_gradients) != len(shape_x):
        error_detail = "The size of the shape of gradients should be same as the size of the shape of x."
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "gradients", "x", error_detail)
    else:
        for grad_, x_ in zip(shape_gradients, shape_x):
            if grad_ != x_ and grad_ != -1 and x_ != -1:
                error_detail = "The shape of gradients and the shape of x should be the same."
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "gradients", "x", error_detail)
    shape_util.compare_tensor_dict_key(gradients, x, "dtype")

    para_check.check_shape(shape_x, param_name="x")
    input_dtype = x.get("dtype").lower()
    para_check.check_dtype(input_dtype, ["float32"], param_name="x")
    if min_value >= max_value:
        excepted_value = "Min must be less than max."
        real_value = "Min is (%2f) and max is (%2f)." % (min_value, max_value)
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "min", excepted_value, real_value)
    if num_bits < 2 or num_bits > 16:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, "num_bits", "2", "16", num_bits)

    ins = classify([gradients, x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_gradient, _x) in ins:
        with tbe.compute():
            shape_gradients, shape_x = shape_util.variable_shape([_gradient, _x])
            data_gradients = tvm.placeholder(shape_gradients, dtype=input_dtype, name="data_gradients")
            data_x = tvm.placeholder(shape_x, dtype=input_dtype, name="data_x")
            res = fake_quant_with_min_max_args_gradient_compute(data_gradients,
                                                                data_x,
                                                                y,
                                                                min_value,
                                                                max_value,
                                                                num_bits,
                                                                narrow_range,
                                                                kernel_name)
            tensors.append([data_gradients, data_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
