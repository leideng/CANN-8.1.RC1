# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
layer_norm_update
"""
"""
layer_norm_update
"""
from tbe import tvm
from tbe.common.utils.errormgr import error_manager_vector
from impl.util import util_select_op_base
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import is_unknown
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import OpPatternMode
from impl.util.norm_pattern_adapter import NormPattern
from impl import constant_util as constant

class LayerNormUpdateAttrInfo:
    ATTR_EPSILON = OpAttr(0, "epsilon", "Float", 0.00001)

#'pylint: disable=huawei-too-many-arguments
def layer_norm_update_compute(input_x, input_gamma, input_beta, input_sum, input_square_sum, output_y,
                              reduce_axis, epsilon, kernel_name="layer_norm_elem",
                              impl_mode="high_performance"):
    """
    DSL description of the layernorm update operator's mathematical calculation process

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of x input data
    input_gamma: TVM tensor
        the placeholder of gamma input data
    input_beta: TVM tensor
        the placeholder of beta input data
    input_sum: TVM tensor
        the placeholder of sum input data
    input_square_sum: TVM tensor
        the placeholder of square sum input data
    output_y: dict
        shape and dtype of output_y
    reduce_axis: list
    the reduce axis
    epsilon: float,
    Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "layer_norm_elem"

    Returns
    -------
    res_tuple: tuple
        (result, mean, rstd)
    """

    shape_x = shape_util.shape_to_list(input_x.shape)
    input_x = tbe.cast_to(input_x, "float32")
    input_gamma = tbe.cast_to(input_gamma, "float32")
    input_beta = tbe.cast_to(input_beta, "float32")
    coeff = reduce_axis ** (-1)
    coeff_const = tvm.const(coeff, dtype="float32")
    epsilon_const = get_attr_by_cls(epsilon, LayerNormUpdateAttrInfo.ATTR_EPSILON, "float32")
    mean = tbe.vmuls(input_sum, coeff_const)
    square_mean = tbe.vmuls(input_square_sum, coeff_const)
    mean_square = tbe.vmul(mean, mean)
    variance = tbe.vsub(square_mean, mean_square)
    normalize_add = tbe.vadds(variance, epsilon_const)
    normalize_log = tbe.vlog(normalize_add)
    normalize_log_mul = tbe.vmuls(normalize_log, tvm.const(-0.5, dtype="float32"))
    rstd = tbe.vexp(normalize_log_mul)
    normalize_rstd_broadcast = tbe.broadcast(rstd, input_x.shape)
    mean_variance_broadcast = tbe.broadcast(mean, input_x.shape)
    variance_sub = tbe.vsub(input_x, mean_variance_broadcast)
    normalize_mul = tbe.vmul(variance_sub, normalize_rstd_broadcast)
    gamma_broadcast = tbe.broadcast(input_gamma, input_x.shape)
    beta_broadcast = tbe.broadcast(input_beta, input_x.shape)
    scale_mul = tbe.vmul(normalize_mul, gamma_broadcast)
    res = tbe.vadd(scale_mul, beta_broadcast)
    res = tbe.cast_to(res, "float16")
    return res


#'pylint: disable=huawei-too-many-arguments
@register_operator("LayerNormUpdate")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def layer_norm_update(input_x,
                      input_gamma,
                      input_beta,
                      input_sum,
                      input_square_sum,
                      output_y,
                      epsilon=1e-5,
                      kernel_name="layer_norm_update",
                      impl_mode="high_performance"):
    """
    layernorm update operator interface implementation
    calculating: x, gamma, beta, input_sum, square_sum
    mean = sum / reduce_axis
    variance = (square_sum / reduce_axis) - mean * mean
    result = gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta

    Parameters
    ----------
    input_x : dict
        shape and dtype of input x, only support float16
    input_gamma: dict
        shape and dtype of input gamma, only support float16
    input_beta: dict
        shape and dtype of input beta, only support float16
    input_sum: dict
        shape and dtype of input mean sum, only support float
    input_square_sum: dict
        shape and dtype of input square mean sum, only support float
    output_y: dict
        shape and dtype of output, only support float16
    epsilon: float,
    Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "layer_norm_elem"

    Returns
    -------
    None
    """
    input_format = input_x.get("format").upper()
    check_list = ("float16")
    dtype = input_x.get("dtype").lower()
    dtype_gamma = input_gamma.get("dtype").lower()
    dtype_beta = input_gamma.get("dtype").lower()
    para_check.check_dtype(dtype, check_list, param_name="input_x")
    para_check.check_dtype(dtype_gamma, check_list, param_name="input_gamma")
    para_check.check_dtype(dtype_beta, check_list, param_name="input_beta")
    check_list = ("float32")
    dtype_sum = input_sum.get("dtype").lower()
    dtype_square_sum = input_square_sum.get("dtype").lower()
    para_check.check_dtype(dtype_sum, check_list, param_name="input_sum")
    para_check.check_dtype(dtype_square_sum, check_list, param_name="input_square_sum")
    shape_x = list(input_x.get("shape"))
    orign_shape_x = list(input_x.get("ori_shape"))
    shape_gamma = list(input_gamma.get("shape"))
    shape_beta = list(input_beta.get("shape"))
    shape_sum = list(input_sum.get("shape"))
    shape_square_sum = list(input_square_sum.get("shape"))
    if shape_gamma != shape_beta:
        error_detail = "gamma and beta's shape must be same."
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_gamma", "input_beta",
                                                                error_detail)

    if shape_sum != shape_square_sum:
        error_detail = "sum and square_sum's shape must be same."
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_sum", "input_square_sum",
                                                                error_detail)
    reduce_axis = shape_x[-1]
    schedules, tensors = [], []
    ins = classify([input_x, input_gamma, input_beta, input_sum, input_square_sum], OpPatternMode.ELEWISE_WITH_BROADCAST)
    for (dy_shape_x, dy_shape_gamma, dy_shape_beta, dy_shape_sum, dy_shape_square_sum) in ins:
        with tbe.compute():
            x_var, gamma_var, beta_var, sum_var, square_sum_var = shape_util.variable_shape([dy_shape_x, dy_shape_gamma, 
                                                                                        dy_shape_beta, dy_shape_sum, dy_shape_square_sum])
            data_x = tvm.placeholder(x_var, name="x", dtype=dtype)
            data_gamma = tvm.placeholder(gamma_var, name="gamma", dtype=dtype_gamma)
            data_beta = tvm.placeholder(beta_var, name="beta", dtype=dtype_beta)
            data_sum = tvm.placeholder(sum_var, name="sum", dtype=dtype_sum)
            data_square_sum = tvm.placeholder(square_sum_var, name="square_sum", dtype=dtype_square_sum)
            res = layer_norm_update_compute(data_x, data_gamma, data_beta, data_sum, data_square_sum, output_y, reduce_axis, epsilon, kernel_name, impl_mode)
            tensors.append([data_x, data_gamma, data_beta, data_sum, data_square_sum, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)

        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
