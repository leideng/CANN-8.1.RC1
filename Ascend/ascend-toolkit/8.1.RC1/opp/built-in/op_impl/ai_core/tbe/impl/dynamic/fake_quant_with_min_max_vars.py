# Copyright 2021 Huawei Technologies Co., Ltd
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
fake_quant_with_min_max_vars
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # define minimum value for float32
    MIN_FLOAT = 2 ** (-126)
    # define reciprocal of half of minimum value for float32
    HALF_MIN_FLOAT_R = 2 ** 62


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-locals
# 'pylint: disable=locally-disabled,unused-variable,invalid-name
def _less_compare_float32(data_x, data_y):
    """
    if x is less than y, then return 1, else return 0.

    Parameters:
    ----------
    data_x : tensor
        tensor x
    data_y : tensor
        tensor y

    Returns
    -------
    the compare result
    """

    # minimum num of float32 2**(-126)
    if tbe_platform.api_check_support("tbe.dsl.vmaxs", data_x.dtype):
        res_sub = tbe.vsub(data_y, data_x)
        res_min = tbe.vmins(res_sub, tvm.const(Constant.MIN_FLOAT, dtype="float32"))
        res_max = tbe.vmaxs(res_min, tvm.const(0, dtype="float32"))
    else:
        data_zero = tbe.vmuls(data_x, 0)
        data_min = tbe.vadds(data_zero, tvm.const(Constant.MIN_FLOAT, dtype="float32"))
        res_sub = tbe.vsub(data_y, data_x)
        res_min = tbe.vmin(res_sub, data_min)
        res_max = tbe.vmax(res_min, data_zero)

    # max num of float32 is 2**126
    # but cce can only support 2**62, so use (2**62)*(2**62)*(2**2) instead
    res_muled = tbe.vmuls(res_max, tvm.const(Constant.HALF_MIN_FLOAT_R, dtype="float32"))
    res_muled = tbe.vmuls(res_muled, tvm.const(Constant.HALF_MIN_FLOAT_R, dtype="float32"))
    res = tbe.vmuls(res_muled, tvm.const(2 ** 2, dtype="float32"))

    return res


def _bool_both_zero_compute(juduged_min, juduged_max):
    """
    if input min and max are both zero then output_data will be all zero
    so need a judge compute tensor

    Parameters:
    ----------
    judged_min : tensor
        tensor min
    judged_max : tensor
        tensor max

    Returns
    -------
    res : tensor
        a tensor for judge compute
    """
    tensor_zero = tbe.vmuls(juduged_min, tvm.const(0, juduged_min.dtype))
    min_abs = tbe.vabs(juduged_min)
    max_abs = tbe.vabs(juduged_max)
    min_max_replace = tbe.vadd(min_abs, max_abs)
    bool_min_max_product_less_zero = _less_compare_float32(min_max_replace, tensor_zero)
    bool_min_max_product_more_zero = _less_compare_float32(tensor_zero, min_max_replace)
    bool_both_zero = tbe.vadd(bool_min_max_product_less_zero, bool_min_max_product_more_zero)
    res = bool_both_zero

    return res


def _nudged_min_max_compute(zero_point_from_min, quant_min, quant_max, scale, min):
    """
    Compute nudged_min, nudged_max operation.

    Parameters
    ----------
    zero_point_from_min: TVM tensor
            the placeholder of zerp_point_from_min
    quant_min: TVM tensor
            the placeholder of quant_min
    quant_max: TVM tensor
            the placeholder of quant_max
    scale: TVM tensor
            the placeholder of scale
    min: TVM tensor
            the placeholder of min

    Returns
    ------
    res: list
        the calculation results
    """
    shape = shape_util.shape_to_list(quant_min.shape)
    tensor_one = tbe.broadcast(tvm.const(1, "float32"), shape)

    bool_less_quant_min_float = _less_compare_float32(zero_point_from_min, quant_min)
    bool_more_quant_max_float = _less_compare_float32(quant_max, zero_point_from_min)
    less_quant_min_float = tbe.vmul(quant_min, bool_less_quant_min_float)
    more_quant_max_float = tbe.vmul(quant_max, bool_more_quant_max_float)
    bool_not_less_quant_min_float = tbe.vsub(tensor_one, bool_less_quant_min_float)
    bool_not_more_quant_max_float = tbe.vsub(tensor_one, bool_more_quant_max_float)
    bool_between_min_max = tbe.vmul(bool_not_less_quant_min_float, bool_not_more_quant_max_float)
    between_min_max_float = tbe.vmul(zero_point_from_min, bool_between_min_max)
    between_min_max_add_half_one = tbe.vadds(between_min_max_float, tvm.const(0.5, "float32"))

    if not tbe_platform.api_check_support("tbe.dsl.floor", "f322s32"):
        between_min_max_add_half_one = tbe.cast_to(between_min_max_add_half_one, "float16")

    between_min_max_round = tbe.floor(between_min_max_add_half_one)
    between_min_max_round_fp32 = tbe.cast_to(between_min_max_round, "float32")
    nudged_zero_point_tmp = tbe.vadd(less_quant_min_float, more_quant_max_float)
    nudged_zero_point = tbe.vadd(nudged_zero_point_tmp, between_min_max_round_fp32)
    nudged_min_tmp = tbe.vsub(quant_min, nudged_zero_point)
    nudged_max_tmp = tbe.vsub(quant_max, nudged_zero_point)
    nudged_min = tbe.vmul(nudged_min_tmp, scale)
    nudged_max = tbe.vmul(nudged_max_tmp, scale)

    return nudged_min, nudged_max


@register_operator_compute("FakeQuantWithMinMaxVars", op_mode="dynamic", support_fusion=False)
def fake_quant_with_min_max_vars_compute(x, min, max, y, num_bits, narrow_range):
    """
    Compute FakeQuantWithMinMaxVars operation.

    Parameters
    ----------
    x: TVM tensor
           the placeholder of x
    min: TVM tensor
           the placeholder of min
    max: TVM tensor
           the placeholder of max
    y: dict
            shape and dtype of fake quant output
    num_bits: int
            define the range of quant max
    narrow_range: bool
            define the range of quant min
    kernel_name : string
            cce kernel name, default value is "fake_quant_with_min_max_vars"

    Returns
    ------
    res: tensor
        the calculation results
    """

    shape = shape_util.shape_to_list(x.shape)
    quant_max = 2 ** num_bits - 1

    if not narrow_range:
        quant_min = 0
    else:
        quant_min = 1

    quant_max = tbe.broadcast(quant_max, shape)
    quant_min = tbe.broadcast(quant_min, shape)

    quant_max_fp32 = tbe.cast_to(quant_max, "float32")
    quant_min_fp32 = tbe.cast_to(quant_min, "float32")

    max = tbe.broadcast(max, shape)
    min = tbe.broadcast(min, shape)

    scale = tbe.vdiv(tbe.vsub(max, min),
                     tbe.vsub(quant_max_fp32, quant_min_fp32))
    zero_point_from_min = tbe.vsub(quant_min_fp32,
                                   tbe.vdiv(min, scale))
    nudged_min, nudged_max = _nudged_min_max_compute(zero_point_from_min, quant_min_fp32,
                                                     quant_max_fp32, scale, min)

    clamped_tmp = tbe.vmin(x, nudged_max)
    clamped = tbe.vmax(clamped_tmp, nudged_min)
    clamped_shifted = tbe.vsub(clamped, nudged_min)
    clamped_scaled = tbe.vadds(tbe.vdiv(clamped_shifted, scale),
                               tvm.const(0.5, "float32"))

    if not tbe_platform.api_check_support("tbe.dsl.floor", "f322s32"):
        clamped_scaled = tbe.cast_to(clamped_scaled, "float16")

    result_tmp = tbe.floor(clamped_scaled)
    result_tmp_fp32 = tbe.cast_to(result_tmp, "float32")
    if not tbe_platform.api_check_support("tbe.dsl.vmla", "float32"):
        tmp_add = tbe.vmul(result_tmp_fp32, scale)
        result = tbe.vadd(tmp_add, nudged_min)
    else:
        result = tbe.vmla(result_tmp_fp32, scale, nudged_min)

    bool_both_zero_value = _bool_both_zero_compute(min, max)
    res = tbe.vmul(result, bool_both_zero_value)

    return res


@register_operator("FakeQuantWithMinMaxVars")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def fake_quant_with_min_max_vars(x, min, max, y, num_bits=8, narrow_range=False,
                                 kernel_name="fake_quant_with_min_max_vars"):
    """
    algorithm: calculate the fake quant value of input tensor
    calculating data's fake quant

    Parameters
    ----------
    x: dict
           shape and dtype of input data
    min: dict
         shape and dtype of min
    max: dict
         shape and dtype of max
    y: dict
            shape and dtype of fake quant output
    num_bits: int
                  define the range of quant max
    narrow_range: bool
                  define the range of quant min
    kernel_name : string
                  cce kernel name, default value is
                  "fake_quant_with_min_max_vars"

    Returns
    -------
    None
    """

    input_shape = x.get("shape")
    min_shape = min.get("shape")
    max_shape = max.get("shape")

    input_dtype = x.get("dtype").lower()
    min_dtype = min.get("dtype").lower()
    max_dtype = max.get("dtype").lower()

    min_shape = shape_util.scalar2tensor_one(min_shape)
    max_shape = shape_util.scalar2tensor_one(max_shape)
    para_check.check_shape(input_shape, param_name="x")
    para_check.check_shape(min_shape, min_rank=1, max_rank=1,
                           max_dim=1, param_name="min")
    para_check.check_shape(max_shape, min_rank=1, max_rank=1,
                           max_dim=1, param_name="max")

    if num_bits > 16 or num_bits < 2:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, "num_bits",
                                                                "2", "16", num_bits)

    check_tuple = ("float32",)

    para_check.check_dtype(input_dtype, check_tuple, param_name="x")
    para_check.check_dtype(min_dtype, check_tuple, param_name="min")
    para_check.check_dtype(max_dtype, check_tuple, param_name="max")

    ins = classify([x, min, max], OpPatternMode.ELEWISE_WITH_BROADCAST,
                   extra_params={"disable_optimization": False})
    schedules, tensors = [], []

    for (_x, _min, _max) in ins:
        with tbe.compute():
            shape_x, shape_min, shape_max = shape_util.variable_shape([_x, _min, _max])

            data = tvm.placeholder(shape_x, dtype=input_dtype, name="data")
            data_min = tvm.placeholder(shape_min, dtype=min_dtype, name="data_min")
            data_max = tvm.placeholder(shape_max, dtype=max_dtype, name="data_max")

            res = fake_quant_with_min_max_vars_compute(data, data_min, data_max,
                                                       y, num_bits, narrow_range)

            tensors.append([data, data_min, data_max, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
