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
fake_quant_with_min_max_args_gradient
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
@register_operator_compute("fake_quant_with_min_max_args_gradient", op_mode="static", support_fusion=True)
def fake_quant_with_min_max_args_gradient_compute(gradients, x, y,
                                                  min=-6, max=6, num_bits=8,
                                                  narrow_range=False,
                                                  kernel_name="fake_quant_with_min"
                                                              "_max_args_gradient"):
    """
    Compute gradients for a FakeQuantWithMinMaxArgs operation.
    calculating data's :
    y = gradients*(if x>=nudged_min and <=nudged_max 1 else 0)

    Parameters
    ----------
    gradients: TVM rensor
        the placeholder of input data,type is float32,
        Backpropagated gradients above the FakeQuantWithMinMaxArgs operation
    x: TVM tenor
        the placeholder of input data,type is float32
    y: dict
        the dict of output data
    min: scalar int or float
        Defaults to -6
    max: scalar int or float
        Defaults to 6
        [min; max] define the clamping range for the x data
    num_bits: int  or float
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
    nudged_min, nudged_max = _nudge_min_max_gradient(min, max, num_bits,
                                                     narrow_range)

    # where((x<=nudged_max)&(x>=nudged_min),1,0),Convert the input to 0 and 1 tensor
    between_nudged_min_max = _cmpare_value(x, nudged_min, nudged_max)

    res = tbe.vmul(gradients, between_nudged_min_max)

    return res


def _nudge_min_max_gradient(min, max, num_bits, narrow_range):
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
   res: nudged_min, nudged_max
   """
    quant_max = (2 ** num_bits) - 1

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

    return nudged_min, nudged_max


def _cmpare_value(x, nudged_min, nudged_max):
    """
    where((x<=nudged_max)&(x>=nudged_min),1,0)

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

    if tbe_platform.api_check_support("tbe.dsl.vmaxs", x.dtype):
        nudged_min_neg = nudged_min * (-1.0)
        nudged_max_neg = nudged_max * (-1.0)

        sub_tmp = tbe.vadds(x, nudged_min_neg)
        sub_min = tbe.vadds(sub_tmp, min_value)
        vmax_tmp = tbe.vmaxs(sub_min, tvm.const(0, sub_min.dtype))

        sub_tmp_max1 = tbe.vadds(x, nudged_max_neg)
        sub_tmp_max2 = tbe.vmuls(sub_tmp_max1, tvm.const(-1.0, sub_tmp_max1.dtype))
        sub_max = tbe.vadds(sub_tmp_max2, min_value)
        vmin_tmp = tbe.vmaxs(sub_max, tvm.const(0, sub_min.dtype))

        one_tmp = tbe.vmul(vmax_tmp, vmin_tmp)
        one_min = tbe.vmins(one_tmp, min_value)

        vmul_max_value = tbe.vmuls(one_min, max_value)
        vmul_max_value_one = tbe.vmuls(vmul_max_value, max_value)
        between_nudged_min_max = tbe.vmuls(vmul_max_value_one, max_value_one)
    else:
        data_zero = tbe.vmuls(x, 0)
        max_value_tensor = tbe.vadds(data_zero, max_value)
        min_value_tensor = tbe.vadds(data_zero, min_value)
        max_value_one_tensor = tbe.vadds(data_zero, max_value_one)
        nudged_max_tensor = tbe.vadds(data_zero, nudged_max)
        nudged_min_tensor = tbe.vadds(data_zero, nudged_min)

        sub_tmp = tbe.vsub(x, nudged_min_tensor)
        sub_min = tbe.vadds(sub_tmp, min_value)
        vmax_tmp = tbe.vmax(sub_min, data_zero)

        sub_tmp_max = tbe.vsub(nudged_max_tensor, x)
        sub_max = tbe.vadds(sub_tmp_max, min_value)
        vmin_tmp = tbe.vmax(sub_max, data_zero)

        one_tmp = tbe.vmul(vmax_tmp, vmin_tmp)
        one_min = tbe.vmin(one_tmp, min_value_tensor)

        vmul_max_value = tbe.vmul(one_min, max_value_tensor)
        vmul_max_value_one = tbe.vmul(vmul_max_value, max_value_tensor)
        between_nudged_min_max = tbe.vmul(vmul_max_value_one, max_value_one_tensor)

    return between_nudged_min_max


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT),
                            (para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT),
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def fake_quant_with_min_max_args_gradient(gradients, x, y, min=-6,
                                          max=6, num_bits=8, narrow_range=False,
                                          kernel_name="fake_quant_"
                                                      "with_min_max_args"):
    """
    Compute gradients for a FakeQuantWithMinMaxArgs operation.
    calculating data's :
    y = gradients*(if x>=nudged_min and <=nudged_max 1 else 0)

    Parameters
    ----------
    gradients:dict
              shape and dtype of input gradients,only support float32
    x: dict
        shape and dtype of input x,only support float32
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
    shape_gradients = gradients.get("shape")
    shape_x = x.get("shape")
    if shape_gradients != shape_x:
        error_detail = "shape of gradients and x should be same"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "gradients", "x", error_detail)
    shape_util.compare_tensor_dict_key(gradients, x, "dtype")

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
    gradients = tvm.placeholder(shape_x, name="gradients", dtype=input_dtype)
    x = tvm.placeholder(shape_x, name="x", dtype=input_dtype)
    res = fake_quant_with_min_max_args_gradient_compute(gradients, x,
                                                        y, float(min),
                                                        float(max),
                                                        num_bits, narrow_range,
                                                        kernel_name)
    with tvm.target.cce():
        auto_sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [gradients, x, res]}
    build(auto_sch, config)
