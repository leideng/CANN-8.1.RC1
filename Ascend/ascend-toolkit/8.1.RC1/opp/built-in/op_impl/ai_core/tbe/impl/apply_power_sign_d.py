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
apply_power_sign
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_compute
from impl.util import util_apply_op_schedule


# 'pylint: disable=too-few-public-methods, not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    CONST_ONE = 1.0
    CONST_ONE_NA = -1.0


# 'pylint: disable=locally-disabled,invalid-name
# 'pylint: disable=too-many-locals
def _compute_m_t(m, beta, grad):
    beta_tmp = tvm.compute(m.shape,
                           lambda *indice: m(*indice) * beta[0],
                           tag='elewise_single_VS_mul')
    beta_na = tvm.compute(beta.shape,
                          lambda *indice: beta(*indice) * tvm.const(Constant.CONST_ONE_NA, beta.dtype),
                          tag='elewise_single_VS_mul')
    beta_na = tvm.compute(beta_na.shape,
                          lambda *indice: beta_na(*indice) + tvm.const(Constant.CONST_ONE, beta_na.dtype),
                          tag='elewise_single_VS_add')
    beta_sub_tmp = tvm.compute(grad.shape,
                               lambda *indice: grad(*indice) * beta_na[0],
                               tag='elewise_single_VS_mul')

    m_t = tbe.vadd(beta_tmp, beta_sub_tmp)
    return m_t


def _compute_update(logbase, sign_decay, sign_gm, grad):
    vmul_tmp = tvm.compute(sign_gm.shape,
                           lambda *indice: sign_gm(*indice) * sign_decay[0],
                           tag='elewise_single_VS_mul')
    vmul_tmp = tvm.compute(vmul_tmp.shape,
                           lambda *indice: vmul_tmp(*indice) * logbase[0],
                           tag='elewise_single_VS_mul')
    exp_tmp = tbe.vexp(vmul_tmp)
    update = tbe.vmul(exp_tmp, grad)
    return update


def _compute_var(var, lr, update):
    lt_tmp = tvm.compute(update.shape,
                         lambda *indice: update(*indice) * lr[0],
                         tag='elewise_single_VS_mul')
    var_t = tbe.vsub(var, lt_tmp)
    return var_t


def _compute_process(input_list):
    var, m, lr, logbase, sign_decay, beta, grad = input_list[0], input_list[1], \
                                                  input_list[2], input_list[3], \
                                                  input_list[4], input_list[5], input_list[6]

    m_t = _compute_m_t(m, beta, grad)

    sign_gm = tbe.vmul(util_compute.sign(m_t), util_compute.sign(grad))

    update = _compute_update(logbase, sign_decay, sign_gm, grad)

    var_t = _compute_var(var, lr, update)

    return var_t, m_t


# 'pylint: disable=locally-disabled, too-many-arguments, unused-argument
@register_operator_compute("apply_power_sign_d", op_mode="static", support_fusion=True)
def apply_power_sign_d_compute(var,
                               m,
                               lr,
                               logbase,
                               sign_decay,
                               beta,
                               grad,
                               var_out,
                               m_out,
                               kernel_name="apply_power_sign_d"):
    """
    Calculate the algorithm

    Parameters:
    ----------
    var : mutable tensor var
    m : mutable tensor m
    lr : scalar lr
    logbase : scalar logbase
    sign_decay : scalar sign_decay
    beta : scalar beta
    grad : mutable tensor grad
    var_out : var output
    m_out : m output

    Algorithm :
    ----------
    m_t <- beta * m_{t-1} + (1 - beta) * grad
    update <- exp(logbase * sign_decay * sign(grad) * sign(m_t)) * grad
    variable <- variable - lr_t * update

    Returns
    ----------
    out_var, out_m, out_var
    """

    dtype = var.dtype
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        var = tbe.cast_to(var, "float32")
        m = tbe.cast_to(m, "float32")
        lr = tbe.cast_to(lr, "float32")
        logbase = tbe.cast_to(logbase, "float32")
        sign_decay = tbe.cast_to(sign_decay, "float32")
        beta = tbe.cast_to(beta, "float32")
        grad = tbe.cast_to(grad, "float32")

    input_list = [var, m, lr, logbase, sign_decay, beta, grad]
    var_t, m_t = _compute_process(input_list)
    res1 = tbe.vadds(var_t, tvm.const(0.0, dtype=dtype))
    res2 = tbe.vadds(m_t, tvm.const(0.0, dtype=dtype))

    if dtype == "float16":
        res1 = tbe.cast_to(res1, "float16")
        res2 = tbe.cast_to(res2, "float16")
        var_t = tbe.cast_to(var_t, "float16")
        m_t = tbe.cast_to(m_t, "float16")

    #this compute is for muti output
    def _compute(*index):
        return [var_t(*index), m_t(*index), res1(*index), res2(*index)]

    return tvm.compute(var.shape, _compute, name="outputs")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def apply_power_sign_d(var,
                       m,
                       lr,
                       logbase,
                       sign_decay,
                       beta,
                       grad,
                       var_out,
                       m_out,
                       kernel_name="apply_power_sign_d"):
    """
    Update '*var' according to the AddSign update

    Parameters:
    ----------
    var: dict of Variable, only support float16, float32
    m : dict of input_grad, only support float16, float32
    lr : dict of lr, only support float16, float32
    logbase : dict of logbase, only support float16, float32
    sign_decay : dict of sign_decay, only support float16, float32
    grad : dict of grad, only support float16, float32
    beta : dict of beta, only support float16, float32
    var_out : dict of output, only support float16, float32
    m_out : dict of output, only support float16, float32
    kernel_name : cce kernel name, default value is apply_power_sign

    Algorithm :
    ----------
    m_t <- beta * m_{t-1} + (1 - beta) * grad
    update <- exp(logbase * sign_decay * sign(grad) * sign(m_t)) * grad
    variable <- variable - lr_t * update


    Returns
    ----------
    None
    """
    input_dict = (var, m, lr, logbase, sign_decay, beta, grad)

    check_list = ('float16', 'float32')
    dtype = var.get('dtype')
    para_check.check_dtype(dtype, check_list, param_name="var")
    dtype = dtype.lower()

    args = util_apply_op_schedule.ApplyOpConfig.TensorArgs(input_dict, apply_power_sign_d_compute, [var_out, m_out],
                                                           6 if dtype == 'float32' else 10)
    name = util_apply_op_schedule.ApplyOpConfig.TensorName(all=('var', 'm', 'lr', 'logbase',
                                                                'sign_decay', 'beta', 'grad'),
                                                           scalar=('lr', 'logbase', 'sign_decay', 'beta'),
                                                           reuse=('var', 'm'))

    util_apply_op_schedule.common_apply_op_process(util_apply_op_schedule.ApplyOpConfig(args, name), kernel_name)
