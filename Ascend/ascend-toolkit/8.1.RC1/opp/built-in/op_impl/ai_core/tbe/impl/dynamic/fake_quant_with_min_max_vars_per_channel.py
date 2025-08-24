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
fake_quant_with_min_max_vars_per_channel:
Fake-quantize the 'inputs' tensor of type float and one of the shapes: [d],[b, d] [b, h, w, d]
via per-channel floats min and max of shape [d] to 'outputs' tensor of same shape as inputs.
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


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # define a scalar for add
    HALF_ONE = 0.5
    ONE_VALUE = 1
    # define zero and one for broadcast
    ZERO_VALUE = 0


def _less_compare_float32(data_x, data_y):
    """
    if x is less than y, then return 1, else return 0.

    Parameters:
    ----------
    data_x : TVM tensor
        tensor x
    data_y : TVM tensor
        tensor y

    Returns
    -------
    the compare result
    """
    # minimun num of float32 2**(-126)
    min_value = tvm.const(2 ** (-126), dtype="float32")

    if tbe_platform.api_check_support("te.lang.cce.vmaxs", data_x.dtype):
        res_sub = tbe.vsub(data_y, data_x)
        res_min = tbe.vmins(res_sub, min_value)
        res_max = tbe.vmaxs(res_min, tvm.const(0, dtype="float32"))
    else:
        data_zero = tbe.vmuls(data_x, 0)
        data_min = tbe.vadds(data_zero, min_value)
        res_sub = tbe.vsub(data_y, data_x)
        res_min = tbe.vmin(res_sub, data_min)
        res_max = tbe.vmax(res_min, data_zero)

    # max num of float32 is 2**126
    # but cce can only support 2**62, so use 62/62/2 to adaptor 126
    res_mul_fierst = tbe.vmuls(res_max, tvm.const(2 ** 62, dtype="float32"))
    res_mul_second = tbe.vmuls(res_mul_fierst, tvm.const(2 ** 62, dtype="float32"))
    res = tbe.vmuls(res_mul_second, tvm.const(2 ** 2, dtype="float32"))

    return res


# 'pylint: disable=locally-disabled,too-many-locals
def _nudged_min_max_compute(min_broadcast, max_broadcast, num_bits,
                            narrow_range):
    """
    compute nudged_min and nudged_max by input parameters

    Parameters:
    ----------
    min : TVM tensor
        tensor min has broadcast to x shape
    max : TVM tensor
        tensor max has broadcast to x shape
    num_bits: int
        An optional int,defaults to 8,shoud be range [2,16]
    narrow_range: bool
        is narrow_range or not,defaults to False

    Returns
    -------
    res: tensor list
        [nudged_min, nudged_max, scale]
    """
    dtype = min_broadcast.dtype
    quant_min = 0.00 if narrow_range is False else 1.00
    quant_max = 2 ** num_bits - 1
    tensor_zero = tbe.broadcast(tvm.const(Constant.ZERO_VALUE, dtype), min_broadcast.shape)
    quant_min_float = tbe.vadds(tensor_zero, tvm.const(quant_min, dtype))
    quant_max_float = tbe.vadds(tensor_zero, tvm.const(quant_max, dtype))
    max_sub_min = tbe.vsub(max_broadcast, min_broadcast)
    quant_max_sub_quant_min = tbe.vsub(quant_max_float, quant_min_float)
    scale = tbe.vdiv(max_sub_min, quant_max_sub_quant_min)
    min_div_scale = tbe.vdiv(min_broadcast, scale)
    zero_point_from_min = tbe.vsub(quant_min_float, min_div_scale)

    bool_less_quant_min_float = _less_compare_float32(zero_point_from_min, quant_min_float)
    bool_more_quant_max_float = _less_compare_float32(quant_max_float, zero_point_from_min)
    less_quant_min_float = tbe.vmul(quant_min_float, bool_less_quant_min_float)
    more_quant_max_float = tbe.vmul(quant_max_float, bool_more_quant_max_float)
    tensor_one = tbe.broadcast(tvm.const(Constant.ONE_VALUE, dtype), tensor_zero.shape)
    bool_not_less_quant_min_float = tbe.vsub(tensor_one, bool_less_quant_min_float)
    bool_not_more_quant_max_float = tbe.vsub(tensor_one, bool_more_quant_max_float)
    bool_between_min_max = tbe.vmul(bool_not_less_quant_min_float, bool_not_more_quant_max_float)
    between_min_max_float = tbe.vmul(zero_point_from_min, bool_between_min_max)
    between_min_max_add_half_one = tbe.vadds(between_min_max_float, tvm.const(Constant.HALF_ONE, dtype))
    between_min_max_round = tbe.floor(between_min_max_add_half_one)
    between_min_max_round = tbe.cast_to(between_min_max_round, dtype)
    nudged_zero_point_tmp = tbe.vadd(less_quant_min_float, more_quant_max_float)
    nudged_zero_point = tbe.vadd(nudged_zero_point_tmp, between_min_max_round)

    nudged_min_tmp = tbe.vsub(quant_min_float, nudged_zero_point)
    nudged_max_tmp = tbe.vsub(quant_max_float, nudged_zero_point)
    nudged_min = tbe.vmul(nudged_min_tmp, scale)
    nudged_max = tbe.vmul(nudged_max_tmp, scale)
    res = [nudged_min, nudged_max, scale]

    return res


def _bool_both_zero_compute(juduged_min, juduged_max):
    """
    if input min and max are both zero then output_date will be all zero
    so need a juduge compute tensor

    Parameters:
    ----------
    min : TVM tensor
        tensor min
    max : TVM tensor
        tensor max

    Returns
    -------
    res : TVM tensor
        a tensor for juduge compute
    """
    dtype = juduged_min.dtype
    tensor_zero = tbe.vmuls(juduged_min, tvm.const(Constant.ZERO_VALUE, dtype))
    min_abs = tbe.vabs(juduged_min)
    max_abs = tbe.vabs(juduged_max)
    min_max_replace = tbe.vadd(min_abs, max_abs)
    bool_min_max_product_less_zero = _less_compare_float32(min_max_replace, tensor_zero)
    bool_min_max_product_more_zero = _less_compare_float32(tensor_zero, min_max_replace)
    bool_both_zero = tbe.vadd(bool_min_max_product_less_zero, bool_min_max_product_more_zero)
    res = bool_both_zero

    return res


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,too-many-arguments
# 'pylint: invalid-name,redefined-builtin
@register_operator_compute("FakeQuantWithMinMaxVarsPerChannel", op_mode="dynamic", support_fusion=False)
def fake_quant_with_min_max_vars_per_channel_compute(x, min_value, max_value, y, num_bits=8, narrow_range=False,
                                                     kernel_name="fake_quant_with_min_max_vars_per_channel"):
    """
    Fake-quantize the 'inputs' tensor of type float and one of the shapes:
                  [d],[b, d] [b, h, w, d]
    via per-channel floats min and max of shape [d] to 'outputs' tensor
                  of same shape as inputs.

    Parameters
    ----------
    x: TVM tensor
        input tensor has shape and dtype attributes
        shape, x_shape equals y_shape,
        dtype, x_dtype equals y_dtype, only support fp32
    min_value: TVM tensor
        input tensor has shape and dtype attributes
        shape of min_value, min_value shape equals to max_value shape
        The shapes of min_value, max_value and shape_inputs last one dimension shoud be same
        the min_value data type,only support fp32
    max_value: TVM tensor
        input tensor has shape and dtype attributes
        shape of max_value, min_value shape equals to max_value shape
        The shapes of min_value, max_value and shape_inputs last one dimension shoud be same
        the max_value data type,only support fp32
    num_bits: int
        An optional int,defaults to 8,shoud be range [2,16]
    narrow_range: bool
        is narrow_range or not,defaults to False
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is
        "fake_quant_with_min_max_vars_per_channel"

    Returns
    ------
    res: TVM tensor
        output tensor has shape and dtype attributes
        shape, y_shape equals x_shape
        dtype, y_dtype equals x_dtype, only support fp32
    """
    x_dtype = x.dtype
    min_broadcast = tbe.broadcast(min_value, x.shape, x_dtype)
    max_broadcast = tbe.broadcast(max_value, x.shape, x_dtype)
    # get nudged_min and nudged_max by _nudged_min_max_compute function
    nudged_min_nudged_max = _nudged_min_max_compute(min_broadcast, max_broadcast, num_bits, narrow_range)
    clamped_tmp = tbe.vmin(x, nudged_min_nudged_max[1])
    clamped = tbe.vmax(clamped_tmp, nudged_min_nudged_max[0])
    clamped_shifted = tbe.vsub(clamped, nudged_min_nudged_max[0])
    clamped_shifted_div_scale = tbe.vdiv(clamped_shifted, nudged_min_nudged_max[2])
    result_tmp = tbe.vadds(clamped_shifted_div_scale, tvm.const(0.5, x_dtype))
    floor_result_tmp = tbe.floor(result_tmp)
    floor_result_tmp = tbe.cast_to(floor_result_tmp, x_dtype)
    scale_product = tbe.vmul(floor_result_tmp, nudged_min_nudged_max[2])
    tmp_res = tbe.vadd(scale_product, nudged_min_nudged_max[0])
    # get bool_both_zero_value by _bool_both_zero_compute function
    bool_both_zero_value = _bool_both_zero_compute(min_broadcast, max_broadcast)
    res = tbe.vmul(tmp_res, bool_both_zero_value)

    return res


# 'pylint: disable=locally-disabled,redefined-builtin,invalid-name
@register_operator("FakeQuantWithMinMaxVarsPerChannel")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def fake_quant_with_min_max_vars_per_channel(x, min_value, max_value, y, num_bits=8, narrow_range=False,
                                             kernel_name="fake_quant_with_min_max_vars_per_channel"):
    """
    Generate fake_quant_with_min_max_vars_per_channel cce operator use
    fake_quant_with_min_max_vars_per_channel_compute

    Parameters
    ----------
    x: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data, assume x_shape equals y_shape,
        the data type, src_dtype equals dst_dtype, support fp32
    min_value: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of min_value, min_value shape equals to max_value shape and only 1 rank
        The shapes of min_value, max_value and shape_inputs last one dimension shoud be same
        the min_value data type,only support fp32
    max_value: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of max_value, min_value shape equals to max_value shape and only 1 rank
        The shapes of min_value, max_value and shape_inputs last one dimension shoud be same
        the max_value data type, only support fp32
    num_bits: int
        An optional int,defaults to 8,shoud be range [2,16]
    narrow_range: bool
        is narrow_range or not,defaults to False
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is
        "fake_quant_with_min_max_vars_per_channel"

    Returns
    ------
    None
    """
    # get dtype and shape attributes
    shape_inputs = x.get("shape")
    dtype_inputs = x.get("dtype")
    shape_min = min_value.get("shape")
    dtype_min = min_value.get("dtype")
    shape_max = max_value.get("shape")
    dtype_max = max_value.get("dtype")
    # check_kernel_name & shape
    dtype_inputs = dtype_inputs.lower()
    dtype_min = dtype_min.lower()
    dtype_max = dtype_max.lower()
    para_check.check_shape(shape_inputs, param_name="x")
    para_check.check_shape(shape_min, min_rank=1, max_rank=1, param_name="min")
    para_check.check_shape(shape_max, min_rank=1, max_rank=1, param_name="max")
    # check input tensor data_type
    para_check.check_dtype(dtype_inputs, "float32", param_name="x")
    para_check.check_dtype(dtype_min, "float32", param_name="min")
    para_check.check_dtype(dtype_max, "float32", param_name="max")
    # check shape_min & shape_max
    if len(shape_min) != len(shape_max):
        error_detail = "The size of the shape of min_value should be same as the size of the shape of max_value."
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "min_value", "max_value", error_detail)
    else:
        for m, n in zip(shape_min, shape_max):
            if m != n and m != -1 and n != -1:
                error_detail = "The shape of min_value and the shape of max_value should be the same."
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name,
                                                                       "min_value",
                                                                       "max_value",
                                                                       error_detail)
    if shape_min[0] != shape_inputs[-1] and shape_min[0] != -1 and shape_inputs[-1] != -1:
        error_detail = "The shapes of min,max and shape_inputs last one dimension shoud be same."
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "min_value", "x", error_detail)
    # check num_bits range
    if num_bits > 16 or num_bits < 2:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, "num_bits", "2", "16", num_bits)

    ins = classify([x, min_value, max_value], OpPatternMode.ELEWISE_WITH_BROADCAST,
                   extra_params={"disable_optimization": False})
    schedules, tensors = [], []
    for (_x, _min, _max) in ins:
        with tbe.compute():
            shape_x, shape_min, shape_max = shape_util.variable_shape([_x, _min, _max])
            data_x = tvm.placeholder(shape_x, dtype=dtype_inputs, name="data_x")
            data_min = tvm.placeholder(shape_min, dtype=dtype_min, name="data_min")
            data_max = tvm.placeholder(shape_max, dtype=dtype_max, name="data_max")
            res = fake_quant_with_min_max_vars_per_channel_compute(data_x,
                                                                   data_min,
                                                                   data_max,
                                                                   y,
                                                                   num_bits,
                                                                   narrow_range,
                                                                   kernel_name)
            tensors.append([data_x, data_min, data_max, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
