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
apply_centered_rms_prop_d.py
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_apply_op_schedule


# 'pylint: disable=too-few-public-methods, not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    NUM_ONE = 1.0
    NUM_ZERO = 0.0
    NUM_ONE_NA = -1.0


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-locals
@register_operator_compute("apply_centered_rms_prop_d", op_mode="static", support_fusion=True)
def apply_centered_rms_prop_d_compute(var,
                                      mg,
                                      ms,
                                      mom,
                                      lr,
                                      rho,
                                      momentum,
                                      epsilon,
                                      grad,
                                      var_out,
                                      mg_out,
                                      ms_out,
                                      mom_out,
                                      kernel_name="apply_centered_rms_prop_d"):
    """
    Update '*var' according to the centered RMSProp algorithm.

    mean_square = decay * mean_square + (1-decay) * gradient ** 2
    mean_grad = decay * mean_grad + (1-decay) * gradient
    Delta = learning_rate*gradient/sqrt(mean_square+epsilon-mean_grad**2)
    mg_{t} <- rho * mg_{t-1} + (1-rho) * grad
    ms_{t} <- rho * ms_{t-1} + (1-rho) * grad * grad
    mom_{t} <- momentum*mom_{t-1}+lr*grad/sqrt(ms_{t}-mg{t}*mg{t}+epsilon)
    var_{t} <- var_{t-1} - mom_{t}

    Parameters:
    ----------
    var: dict of tensor var, include shape and dtype,
        dtype support float16 and float32.

    mg: dict of tensor mg(mean_grad), include shape and dtype,
        dtype support float16 and float32.

    ms: dict of tensor ms(mean_square), include shape and dtype,
        dtype support float16 and float32.

    mom: dict of tensor mom, include shape and dtype,
        dtype support float16 and float32.

    lr: dict of scalar lr(learning rate). Must have the same dtype as var.

    rho: dict of scalar rho(decay rate). Must have the same dtype as var.

    momentum: dict of scalar momentum. Must have the same dtype as var.

    epsilon: dict of scalar epsilon. Must have the same dtype as var.

    grad: dict of tensor grad. Must have the same dtype as var.

    out: dict of output out.

    kernel_name : cce kernel name, default value is "apply_centered_rms_prop_d".

    Returns
    -------
    None
    """

    inp_dtype = var.dtype
    if inp_dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        var = tbe.cast_to(var, "float32")
        mg = tbe.cast_to(mg, "float32")
        ms = tbe.cast_to(ms, "float32")
        mom = tbe.cast_to(mom, "float32")
        lr = tbe.cast_to(lr, "float32")
        rho = tbe.cast_to(rho, "float32")
        momentum = tbe.cast_to(momentum, "float32")
        epsilon = tbe.cast_to(epsilon, "float32")
        grad = tbe.cast_to(grad, "float32")

    tensor_one_rho = tvm.compute(rho.shape,
                                 lambda *indices: rho(*indices)
                                     * tvm.const(Constant.NUM_ONE_NA, rho.dtype),
                                 tag='elewise_single_VS_mul')
    tensor_one_rho = tvm.compute(
        tensor_one_rho.shape,
        lambda *indices: tensor_one_rho(*indices)
                         + tvm.const(Constant.NUM_ONE, tensor_one_rho.dtype),
        tag='elewise_single_VS_add')

    mg_rho = tvm.compute(mg.shape,
                         lambda *indices: mg(*indices) * rho[0],
                         tag='elewise_single_VS_mul')
    rhs = tvm.compute(grad.shape,
                      lambda *indices: grad(*indices) * tensor_one_rho[0],
                      tag='elewise_single_VS_mul')
    out_mg = tbe.vadd(mg_rho, rhs)

    ms_rho = tvm.compute(ms.shape,
                         lambda *indices: ms(*indices) * rho[0],
                         tag='elewise_single_VS_mul')
    rhs = tbe.vmul(grad, grad)
    rhs = tvm.compute(rhs.shape,
                      lambda *indices: rhs(*indices) * tensor_one_rho[0],
                      tag='elewise_single_VS_mul')
    out_ms = tbe.vadd(ms_rho, rhs)

    lhs_mom = tvm.compute(mom.shape,
                          lambda *indices: mom(*indices) * momentum[0],
                          tag='elewise_single_VS_mul')
    lr_grad = tvm.compute(grad.shape,
                          lambda *indices: grad(*indices) * lr[0],
                          tag='elewise_single_VS_mul')
    rhs = tbe.vmul(out_mg, out_mg)
    rhs = tbe.vsub(out_ms, rhs)
    rhs_eps = tvm.compute(rhs.shape,
                          lambda *indices: rhs(*indices) + epsilon[0],
                          tag='elewise_single_VS_add')
    rhs_eps = tbe.vsqrt(rhs_eps)
    rhs_eps = tbe.vdiv(lr_grad, rhs_eps)
    out_mom = tbe.vadd(lhs_mom, rhs_eps)

    out_var = tbe.vsub(var, out_mom)

    if inp_dtype == "float16":
        out_var = tbe.cast_to(out_var, "float16")
        out_mg = tbe.cast_to(out_mg, "float16")
        out_ms = tbe.cast_to(out_ms, "float16")
        out_mom = tbe.cast_to(out_mom, "float16")

    mg_output_data = tbe.vadds(out_mg, Constant.NUM_ZERO)
    ms_output_data = tbe.vadds(out_ms, Constant.NUM_ZERO)
    mom_output_data = tbe.vadds(out_mom, Constant.NUM_ZERO)

    # this compute is for multi output
    def _compute(*index):
        return [out_mg(*index), out_ms(*index), out_mom(*index), out_var(
            *index), out_var(*index), mg_output_data(*index), ms_output_data(
            *index), mom_output_data(*index)]

    return tvm.compute(var.shape, _compute, name="outputs")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def apply_centered_rms_prop_d(var,
                              mg,
                              ms,
                              mom,
                              lr,
                              rho,
                              momentum,
                              epsilon,
                              grad,
                              var_out,
                              mg_out,
                              ms_out,
                              mom_out,
                              kernel_name="apply_centered_rms_prop_d"):
    """
    Update '*var' according to the centered RMSProp algorithm.

    mean_square = decay * mean_square + (1-decay) * gradient ** 2
    mean_grad = decay * mean_grad + (1-decay) * gradient
    Delta = learning_rate*gradient/sqrt(mean_square+epsilon-mean_grad**2)
    mg_{t} <- rho * mg_{t-1} + (1-rho) * grad
    ms_{t} <- rho * ms_{t-1} + (1-rho) * grad * grad
    mom_{t} <- momentum*mom_{t-1}+lr*grad/sqrt(ms_{t}-mg{t}*mg{t}+epsilon)
    var_{t} <- var_{t-1} - mom_{t}

    Parameters:
    ----------
    var: dict of tensor var, include shape and dtype,
        dtype support float16 and float32.

    mg: dict of tensor mg(mean_grad), include shape and dtype,
        dtype support float16 and float32.

    ms: dict of tensor ms(mean_square), include shape and dtype,
        dtype support float16 and float32.

    mom: dict of tensor mom, include shape and dtype,
        dtype support float16 and float32.

    lr: dict of scalar lr(learning rate). Must have the same dtype as var.

    rho: dict of scalar rho(decay rate). Must have the same dtype as var.

    momentum: dict of scalar momentum. Must have the same dtype as var.

    epsilon: dict of scalar epsilon. Must have the same dtype as var.

    grad: dict of tensor grad. Must have the same dtype as var.

    var_out: the dict of var output, only support float16, float32

    mg_out: the dict of mg output, only support float16, float32

    ms_out: the dict of ms output, only support float16, float32

    mom_out: the dict of mom output, only support float16, float32

    kernel_name : cce kernel name, default value is "apply_centered_rms_prop_d".

    Returns
    -------
    None
    """

    input_dict = (var, mg, ms, mom, lr, rho, momentum, epsilon, grad)
    out = [var_out, mg_out, ms_out, mom_out]
    check_list = ('float16', 'float32')
    dtype = var.get('dtype')
    para_check.check_dtype(dtype, check_list, param_name="var")
    dtype = dtype.lower()

    args = util_apply_op_schedule.ApplyOpConfig.TensorArgs(input_dict,
                                                           apply_centered_rms_prop_d_compute, out,
                                                           6 if dtype == "float32" else 12)
    name = util_apply_op_schedule.ApplyOpConfig.TensorName(all=('var', 'mg', 'ms', 'mom', 'lr', 'rho',
                                                                'momentum', 'epsilon', 'grad'),
                                                           scalar=('lr', 'rho', 'momentum', 'epsilon'),
                                                           reuse=('mg', 'ms', 'mom', 'var'))

    util_apply_op_schedule.common_apply_op_process(util_apply_op_schedule.ApplyOpConfig(args, name), kernel_name)
