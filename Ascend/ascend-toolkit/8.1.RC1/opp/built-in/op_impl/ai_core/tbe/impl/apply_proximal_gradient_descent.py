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
apply_proximal_gradient_descent
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_apply_op_schedule
from impl.util import util_build
from impl.util import util_compute


# 'pylint: disable=too-few-public-methods, not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    CONST_ZERO = 0
    CONST_ONE = 1
    CONST_ONE_NEG = -1


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("apply_proximal_gradient_descent", op_mode="static", support_fusion=True)
def apply_proximal_gradient_descent_compute(
        var,
        alpha,
        l1,
        l2,
        delta,
        out,
        kernel_name="apply_proximal_gradient_descent"):
    """
    the operator's compute
    prox_v = var - alpha * delta
    if l1 > 0 :
        var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}
    else:
        var = prox_v / (var + l2 * delta)

    Parameters:
    ----------
    var: the dict of var, only support float16, float32
    alpha: the dict of alpha, only support float16, float32
    l1: the dict of l1, only support float16, float32
    l2: the dict of l2, only support float16, float32
    delta: the dict of delta, only support float16, float32
    out: the dict of output, only support float16, float32

    Returns
    the value of out_var
    output_data
    """
    dtype = var.dtype

    if dtype == "float16":
        var = tbe.cast_to(var, "float32")
        alpha = tbe.cast_to(alpha, "float32")
        l1 = tbe.cast_to(l1, "float32")
        l2 = tbe.cast_to(l2, "float32")
        delta = tbe.cast_to(delta, "float32")

    alpha_broad = tbe.broadcast(alpha, var.shape)
    l1_broad = tbe.broadcast(l1, var.shape)
    l2_broad = tbe.broadcast(l2, var.shape)

    var_out = _compute_process(var, alpha_broad, l1_broad, l2_broad, delta)

    if dtype == "float16":
        var_out = tbe.cast_to(var_out, "float16")
    else:
        var_out = tbe.cast_to(var_out, "float32")

    # this compute is for muti output
    def _compute(*index):
        return var_out(*index), var_out(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


def _compute_process(var, alpha_broad, l1_broad, l2_broad, delta):
    """
    the operator's compute
    prox_v = var - alpha * delta
    if l1 > 0 :
        var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}
    else:
        var = prox_v / (var + l2 * delta)

    Parameters:
    ----------
    var: the value of var
    alpha_broad: the value of alpha_broad
    l1_broad: the value of l1_broad
    l2_broad: the value of l2_broad
    delta: the value of delta

    Returns
    the value of out_var
    output_data
    """
    # var - alpha * delta
    alpha_delta = tbe.vmul(alpha_broad, delta)
    alpha_delta = tbe.vmuls(alpha_delta, tvm.const(Constant.CONST_ONE_NEG, "float32"))
    prox_v = tbe.vadd(var, alpha_delta)

    const_zero_tensor = tbe.broadcast(tvm.const(Constant.CONST_ZERO, var.dtype.lower()), delta.shape)
    situation = tbe.vcmp(l1_broad, const_zero_tensor, 'gt')

    var_res = _compute_positive(prox_v, alpha_broad, l1_broad, l2_broad)

    # prox_var / 1 + l2 * alpha
    l2_lr = tbe.vmul(l2_broad, alpha_broad)
    l2_lr_1 = tbe.vadds(l2_lr, tvm.const(Constant.CONST_ONE, "float32"))
    var_t_neg = tbe.vdiv(prox_v, l2_lr_1)

    var_out = tbe.vsel(situation, var_res, var_t_neg)

    return var_out


def _compute_positive(prox_v, alpha_broad, l1_broad, l2_broad):
    """
    the operator's compute
    var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

    Parameters:
    ----------
    prox_v: the value of prox_v
    alpha_broad: the value of alpha_broad
    l1_broad: the value of l1_broad
    l2_broad: the value of l2_broad

    Returns
    the value of var_res
    """
    prox_v_abs = tbe.vabs(prox_v)
    prox_v_sign = util_compute.sign(prox_v)
    # 1+alpha*l2
    alpha_l2 = tbe.vmul(alpha_broad, l2_broad)
    alpha_l2_1 = tbe.vadds(alpha_l2, tvm.const(Constant.CONST_ONE, "float32"))
    # this step is max{|prox_v|-alpha*l1,0}
    alpha_l1 = tbe.vmul(alpha_broad, l1_broad)
    alpha_l1_neg = tbe.vmuls(alpha_l1, tvm.const(Constant.CONST_ONE_NEG, "float32"))
    prox_v_l1 = tbe.vadd(prox_v_abs, alpha_l1_neg)
    max_value = tbe.vmax(
        prox_v_l1,
        tbe.broadcast(tvm.const(Constant.CONST_ZERO, "float32"), prox_v.shape))
    # this step is sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}
    res = tbe.vdiv(prox_v_sign, alpha_l2_1)
    var_res = tbe.vmul(res, max_value)

    return var_res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def apply_proximal_gradient_descent(
        var,
        alpha,
        l1,
        l2,
        delta,
        out,
        kernel_name="apply_proximal_gradient_descent"):
    """
    Update '*var' as FOBOS algorithm with fixed learning rate..

    prox_v = var - alpha * delta
    var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

    Parameters:
    ----------
    var: the dict of var, only support float16, float32
    alpha: the dict of alpha, only support float16, float32
    l1: the dict of l1, only support float16, float32
    l2: the dict of l2, only support float16, float32
    delta: the dict of delta, only support float16, float32
    out: the dict of output, only support float16, float32

    kernel_name : cce kernel name, default value is
        "apply_proximal_gradient_descent"

    Returns
    -------
    None
    """

    check_list = ('float16', 'float32')
    dtype = var.get('dtype')
    para_check.check_dtype(dtype, check_list, param_name="var")
    dtype = dtype.lower()

    input_dict = (var, alpha, l1, l2, delta)

    args = util_apply_op_schedule.ApplyOpConfig.TensorArgs(input_dict,
                                                           apply_proximal_gradient_descent_compute,
                                                           out, 5 if dtype == 'float32' else 10)
    name = util_apply_op_schedule.ApplyOpConfig.TensorName(all=('var', 'alpha', 'l1', 'l2', 'delta'),
                                                           scalar=('alpha', 'l1', 'l2'),
                                                           reuse=('var', ))
    options = util_apply_op_schedule.ApplyOpConfig.TensorOptions(build=util_build.set_bool_storage_config())

    util_apply_op_schedule.common_apply_op_process(util_apply_op_schedule.ApplyOpConfig(args, name, options),
                                                   kernel_name)
