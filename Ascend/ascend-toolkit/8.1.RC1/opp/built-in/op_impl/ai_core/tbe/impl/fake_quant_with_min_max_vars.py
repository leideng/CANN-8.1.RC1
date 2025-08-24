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
fake_quant_with_min_max_vars
"""
import functools

import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # define a scalar for add
    HALF_ONE = 0.5
    # define zero for broadcast
    ZERO_VALUE = 0
    # define one for broadcast
    ONE_VALUE = 1


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-locals,unused-variable
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
    shape_inputs = shape_util.shape_to_list(data_x.shape)
    # minimum num of float32 2**(-126)
    min_value = tvm.const(2 ** (-126), dtype="float32")

    if tbe_platform.api_check_support("tbe.dsl.vmaxs", data_x.dtype):
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
    res_muled = tbe.vmuls(res_max, tvm.const(2 ** 62, dtype="float32"))
    res_mul = tbe.vmuls(res_muled, tvm.const(2 ** 62, dtype="float32"))
    res = tbe.vmuls(res_mul, tvm.const(2 ** 2, dtype="float32"))

    return res


def _bool_both_zero_compute(juduged_min, juduged_max):
    """
    if input min and max are both zero then output_date will be all zero
    so need a juduge compute tensor

    Parameters:
    ----------
    juduged_min : tensor
        tensor min
    juduged_max : tensor
        tensor max

    Returns
    -------
    res : tensor
        a tensor for juduge compute
    """
    dtype = juduged_min.dtype
    tensor_zero = tbe.vmuls(juduged_min, tvm.const(Constant.ZERO_VALUE, dtype))
    min_abs = tbe.vabs(juduged_min)
    max_abs = tbe.vabs(juduged_max)
    min_max_replace = tbe.vadd(min_abs, max_abs)
    bool_min_max_product_less_zero = _less_compare_float32(min_max_replace,
                                                           tensor_zero)
    bool_min_max_product_more_zero = _less_compare_float32(tensor_zero,
                                                           min_max_replace)
    bool_both_zero = tbe.vadd(bool_min_max_product_less_zero, bool_min_max_product_more_zero)
    res = bool_both_zero

    return res


def _nudged_min_max_compute(zero_point_from_min, quant_min, quant_max, scale,
                            min):
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
    tensor_zero = tbe.vmuls(min, tvm.const(Constant.ZERO_VALUE, "float32"))
    bool_less_quant_min_float = _less_compare_float32(zero_point_from_min, quant_min)
    bool_more_quant_max_float = _less_compare_float32(quant_max, zero_point_from_min)
    less_quant_min_float = tbe.vmul(quant_min, bool_less_quant_min_float)
    more_quant_max_float = tbe.vmul(quant_max, bool_more_quant_max_float)
    tensor_one = tbe.vadds(tensor_zero, tvm.const(Constant.ONE_VALUE, "float32"))
    bool_not_less_quant_min_float = tbe.vsub(tensor_one, bool_less_quant_min_float)
    bool_not_more_quant_max_float = tbe.vsub(tensor_one, bool_more_quant_max_float)
    bool_between_min_max = tbe.vmul(bool_not_less_quant_min_float, bool_not_more_quant_max_float)
    between_min_max_float = tbe.vmul(zero_point_from_min, bool_between_min_max)
    between_min_max_add_half_one = tbe.vadds(between_min_max_float, tvm.const(Constant.HALF_ONE, "float32"))
    between_min_max_round = tbe.floor(between_min_max_add_half_one)
    nudged_zero_point_tmp = tbe.vadd(less_quant_min_float, more_quant_max_float)
    nudged_zero_point = tbe.vadd(nudged_zero_point_tmp, between_min_max_round)
    nudged_min_tmp = tbe.vsub(quant_min, nudged_zero_point)
    nudged_max_tmp = tbe.vsub(quant_max, nudged_zero_point)
    nudged_min = tbe.vmul(nudged_min_tmp, scale)
    nudged_max = tbe.vmul(nudged_max_tmp, scale)

    return nudged_min, nudged_max


@register_operator_compute("fake_quant_with_min_max_vars", op_mode="static", support_fusion=True)
def fake_quant_with_min_max_vars_compute(x, min, max, y, num_bits, narrow_range,
                                         kernel_name="fake_quant_with_min_max_vars"):
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
            cce kernel name, default value is "bitwise_or"

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
    max = tbe.broadcast(max, shape)
    min = tbe.broadcast(min, shape)

    scale = tbe.vdiv(tbe.vsub(max, min), tbe.vsub(quant_max, quant_min))
    zero_point_from_min = tbe.vsub(quant_min, tbe.vdiv(min, scale))
    nudged_min, nudged_max = _nudged_min_max_compute(zero_point_from_min, quant_min, quant_max, scale, min)

    clamped_tmp = tbe.vmin(x, nudged_max)
    clamped = tbe.vmax(clamped_tmp, nudged_min)
    clamped_shifted = tbe.vsub(clamped, nudged_min)
    result_tmp = tbe.floor(tbe.vadds(tbe.vdiv(clamped_shifted, scale), tvm.const(0.5, "float32")))
    result = tbe.vadd(tbe.vmul(result_tmp, scale), nudged_min)

    bool_both_zero_value = _bool_both_zero_compute(min, max)
    res = tbe.vmul(result, bool_both_zero_value)

    return res


# 'pylint: disable=locally-disabled,redefined-builtin,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def fake_quant_with_min_max_vars(x, min, max, y, num_bits, narrow_range,
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
    input_dtype = x.get("dtype")
    min_shape = min.get("shape")
    min_dtype = min.get("dtype")
    max_shape = max.get("shape")
    max_dtype = max.get("dtype")

    min_shape = shape_util.scalar2tensor_one(min_shape)
    max_shape = shape_util.scalar2tensor_one(max_shape)
    para_check.check_shape(input_shape, param_name="x")
    para_check.check_shape(min_shape, min_rank=1, max_rank=1, param_name="min")
    para_check.check_shape(max_shape, min_rank=1, max_rank=1, param_name="max")

    if num_bits > 16 or num_bits < 2:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, "num_bits", "2", "16", num_bits)

    check_tuple = ("float32",)
    x_type = input_dtype.lower()
    min_dtype = min_dtype.lower()
    max_dtype = max_dtype.lower()
    para_check.check_dtype(x_type, check_tuple, param_name="x")
    para_check.check_dtype(min_dtype, check_tuple, param_name="min")
    para_check.check_dtype(max_dtype, check_tuple, param_name="max")
    input_shape = (functools.reduce(lambda x, y: x * y, input_shape[:]),)
    shape_min, shape_max, shape_broadcast = shape_util.broadcast_shapes(min_shape, input_shape,
                                                                        param_name_input1="min",
                                                                        param_name_input2="x")
    data = tvm.placeholder(input_shape, dtype=x_type, name="data_input")
    data_min = tvm.placeholder(shape_min, dtype=min_dtype, name="data_min")
    data_max = tvm.placeholder(shape_min, dtype=max_dtype, name="data_max")

    res = fake_quant_with_min_max_vars_compute(data, data_min, data_max, y, num_bits, narrow_range, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": (data, data_min, data_max, res)}

    tbe.cce_build_code(schedule, config)
