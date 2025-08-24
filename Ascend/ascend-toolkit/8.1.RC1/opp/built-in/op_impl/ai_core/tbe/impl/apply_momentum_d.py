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
apply_momentum
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_apply_op_schedule
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable=too-few-public-methods, not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    NUM_ZERO = 0.0


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=unused-argument,invalid-name,too-many-locals
def get_op_support_info(var,
                        accum,
                        lr,
                        grad,
                        momentum,
                        var_out,
                        accum_out,
                        use_nesterov=False,
                        kernel_name="apply_momentum_d"):
    """
    get_op_support_info
    """
    format_var = var.get("format").upper()
    if format_var in ("NC1HWC0",):
        # cut N
        axis_split_matrix = [[
            SplitInput([0, [0], [-1], [-1]], [1, [0], [-1], [-1]], [2, [0], [-1], [-1]], [3, [0], [-1], [-1]],
                       [4, [0], [-1], [-1]]),
            SplitOutput([0, [0]], [1, [0]])
        ]]
    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


@register_operator_compute("apply_momentum_d", op_mode="static", support_fusion=True)
def apply_momentum_compute_d(var,
                             accum,
                             lr,
                             grad,
                             momentum,
                             var_out,
                             accum_out,
                             use_nesterov,
                             kernel_name='apply_momentum_d'):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    accum = accum * momentum + grad
    if use_nesterov is True:
        var -= grad * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : mutable tensor var.

    accum: mutable tensor accum.

    lr : scalar lr.

    grad : tensor grad.

    momentum : scalar momentum.

    var_out : the dict of output var.

    accum_out : the dict of output accum.

    use_nesterov: bool. If true, use nesterov computing grad,
        default value is False.

    kernel_name : cce kernel name, default value is "apply_momentum_d".

    Returns:
    -------
    None
    """

    # cast to float32 for higher accuracy
    dtype = var.dtype
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        var = tbe.cast_to(var, "float32")
        accum = tbe.cast_to(accum, "float32")
        lr = tbe.cast_to(lr, "float32")
        grad = tbe.cast_to(grad, "float32")
        momentum = tbe.cast_to(momentum, "float32")

    # update accum
    accum_delta = tvm.compute(accum.shape,
                              lambda *indice: accum(*indice) * momentum[0],
                              tag='elewise_single_VS_mul')
    accum_t = tbe.vadd(accum_delta, grad)

    # update var
    if use_nesterov:
        var_delta = tvm.compute(grad.shape,
                                lambda *indice: grad(*indice) * lr[0],
                                tag='elewise_single_VS_mul')
        var_delta_2 = tvm.compute(
            accum_t.shape,
            lambda *indice: accum_t(*indice) * momentum[0],
            tag='elewise_single_VS_mul')
        var_delta_2 = tvm.compute(var_delta_2.shape,
                                  lambda *indice: var_delta_2(*indice) * lr[0],
                                  tag='elewise_single_VS_mul')
        var_delta = tbe.vadd(var_delta, var_delta_2)
        var_t = tbe.vsub(var, var_delta)
    else:
        var_delta = tvm.compute(accum_t.shape,
                                lambda *indice: accum_t(*indice) * lr[0],
                                tag='elewise_single_VS_mul')
        var_t = tbe.vsub(var, var_delta)

    if dtype == "float16":
        var_t = tbe.cast_to(var_t, "float16")
        accum_t = tbe.cast_to(accum_t, "float16")

    var_out_data = tbe.vadds(
        var_t, tvm.const(Constant.NUM_ZERO, var_t.dtype))
    accum_out_data = tbe.vadds(
        accum_t, tvm.const(Constant.NUM_ZERO, accum_t.dtype))

    def _compute(*index):
        return [accum_t(*index), var_t(*index), var_out_data(*index), \
               accum_out_data(*index)]

    return tvm.compute(var.shape, _compute, name="outputs")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def apply_momentum_d(var,
                     accum,
                     lr,
                     grad,
                     momentum,
                     var_out,
                     accum_out,
                     use_nesterov=False,
                     kernel_name="apply_momentum_d"):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    accum = accum * momentum + grad
    if use_nesterov is True:
        var -= gard * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : the dict of mutable tensor var, only support float16, float32.

    accum : the dict of mutable tensor accum.
        Must have the same data type as `var`.

    lr : the dict of scalar lr. Must have the same data type as `var`.

    grad : the dict of tensor grad. Must have the same data type as `var`.

    momentum : the dict of scalar momentum.
        Must have the same data type as `var`.

    var_out : the dict of output var.

    accum_out : the dict of output accum.

    use_nesterov: bool. If true, use nesterov computing grad,
        default value is False.

    kernel_name : cce kernel name, default value is "apply_momentum_d".

    Returns
    -------
    None
    """

    input_dict = (var, accum, lr, grad, momentum)

    args = util_apply_op_schedule.ApplyOpConfig.TensorArgs(
        input_dict,
        apply_momentum_compute_d,
        [var_out, accum_out],
        8 if use_nesterov else 6,
    )
    name = util_apply_op_schedule.ApplyOpConfig.TensorName(all=('var', 'accum', 'lr', 'grad', 'momentum'),
                                                           scalar=('lr', 'momentum'),
                                                           reuse=('accum', 'var'))
    options = util_apply_op_schedule.ApplyOpConfig.TensorOptions(attrs=use_nesterov)

    util_apply_op_schedule.common_apply_op_process(util_apply_op_schedule.ApplyOpConfig(args, name, options),
                                                   kernel_name)
