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
this file achieved the apply_adadelta_d which is a optimizer operator
to update weight
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_apply_op_schedule


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Const:
    """
    The class for constant.
    """
    NUM_ONE = 1.0
    NUM_ZERO = 0.0


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=unused-argument,invalid-name,too-many-locals
@register_operator_compute("apply_adadelta_d", op_mode="static", support_fusion=True)
def apply_adadelta_d_compute(var,
                             accum,
                             accum_update,
                             lr,
                             rho,
                             epsilon,
                             grad,
                             var_out,
                             accum_out,
                             accum_update_out,
                             kernel_name='apply_adadelta_d'):
    """
    Update '*var' according to the adadelta scheme.

    accum = rho * accum + (1 - rho) * grad ** 2
    update = (update_accum + epsilon).sqrt() * (accum + epsilon).rsqrt()*grad
    update_accum = rho * update_accum + (1 - rho) * update.square();
    var -= update * lr;

    Parameters:
    ----------
    var : mutable tensor var.

    accum: mutable tensor accum.

    accum_update : mutable tensor accum_update.

    lr : scalar lr.

    rho : scalar rho.

    epsilon : scalar epsilon.

    grad : tensor grad.

    var_out : the dict of var output.

    accum_out : the dict of accum output.

    accum_update_out : the dict of accum_update output.

    kernel_name : cce kernel name, default value is "apply_adadelta_d".

    Returns:
    -------
    None
    """

    dtype = var.dtype
    has_improve_precision = False
    cast_type = "float16"
    if dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vsqrt", "float32"):
        cast_type = "float32"
        has_improve_precision = True

    if dtype == "float16" and has_improve_precision:
        var = tbe.cast_to(var, "float32")
        accum = tbe.cast_to(accum, "float32")
        accum_update = tbe.cast_to(accum_update, "float32")
        lr = tbe.cast_to(lr, "float32")
        rho = tbe.cast_to(rho, "float32")
        epsilon = tbe.cast_to(epsilon, "float32")
        grad = tbe.cast_to(grad, "float32")

    tensor_one = tbe.broadcast(tvm.const(Const.NUM_ONE, cast_type), var.shape)
    tensor_rho = tbe.broadcast(rho, var.shape)
    tensor_one = tbe.cast_to(tensor_one, tensor_rho.dtype)
    tensor_rho_gs = tbe.vsub(tensor_one, tensor_rho)
    tensor_epsilon = tbe.broadcast(epsilon, var.shape)

    # step1: update accum
    rhs = tbe.vmul(accum, tensor_rho)
    lhs = tbe.vmul(grad, grad)
    lhs = tbe.vmul(lhs, tensor_rho_gs)
    accum_res = tbe.vadd(lhs, rhs)

    accum_update_orig = tbe.vadds(accum_update, Const.NUM_ZERO)
    # step2
    rhs = tbe.vadd(accum_update_orig, tensor_epsilon)
    rhs = tbe.vsqrt(rhs)
    lhs = tbe.vadd(accum_res, tensor_epsilon)
    lhs = tbe.vsqrt(lhs)
    lhs = tbe.vdiv(grad, lhs)
    update = tbe.vmul(lhs, rhs)

    # step3: update var
    var_res = tbe.broadcast(lr, var.shape)
    var_res = tbe.vmul(update, var_res)
    var_res = tbe.vsub(var, var_res)

    # step4: update accum_update
    rhs = tbe.vmul(accum_update_orig, tensor_rho)
    lhs = tbe.vmul(update, update)
    lhs = tbe.vmul(lhs, tensor_rho_gs)
    accum_update_res = tbe.vadd(lhs, rhs)

    # out
    output_data = tbe.vadds(var_res, Const.NUM_ZERO)
    accum_output_data = tbe.vadds(accum_res, Const.NUM_ZERO)
    accum_update_output_data = tbe.vadds(accum_update_res, Const.NUM_ZERO)

    if dtype == "float16" and has_improve_precision:
        var_res = tbe.cast_to(var_res, "float16")
        accum_res = tbe.cast_to(accum_res, "float16")
        accum_update_res = tbe.cast_to(accum_update_res, "float16")
        output_data = tbe.cast_to(output_data, "float16")
        accum_output_data = tbe.cast_to(accum_output_data, "float16")
        accum_update_output_data = \
            tbe.cast_to(accum_update_output_data, "float16")

    # this compute is for muti output
    def _compute(*index):
        return [var_res(*index), accum_res(*index), accum_update_res(*index), \
               output_data(*index), accum_output_data(*index), \
               accum_update_output_data(*index)]

    return tvm.compute(var.shape, _compute, name="outputs")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def apply_adadelta_d(var,
                     accum,
                     accum_update,
                     lr,
                     rho,
                     epsilon,
                     grad,
                     var_out,
                     accum_out,
                     accum_update_out,
                     kernel_name="apply_adadelta_d"):
    """
    Update '*var' according to the adadelta scheme.

    accum = rho * accum + (1 - rho) * grad ** 2
    update = (update_accum + epsilon).sqrt() * (accum + epsilon).rsqrt() * grad
    update_accum = rho * update_accum + (1 - rho) * update.square();
    var -= update * lr;

    Parameters:
    ----------
    var: the dict of input, only support float16, float32

    accum: the dict of accum, only support float16, float32

    accum_update: the dict of accum_update, only support float16, float32

    lr: the dict of lr, only support float16, float32

    rho: the dict of rho, only support float16, float32

    epsilon: the dict of epsilon, only support float16, float32

    grad: the dict of grad, only support float16, float32

    var_out: the dict of var output data

    accum_out: the dict of accum output data

    accum_update_out: the dict of accum_update output data

    kernel_name : cce kernel name, default value is "apply_adadelta_d"

    Returns
    -------
    None
    """
    input_dict = (var, accum, accum_update, lr, rho, epsilon, grad)

    args = util_apply_op_schedule.ApplyOpConfig.TensorArgs(input_dict, apply_adadelta_d_compute,
                                    [var_out, accum_out, accum_update_out], 16)
    name = util_apply_op_schedule.ApplyOpConfig.TensorName(all=('var', 'accum', 'accum_update', 'lr',
                                                                'rho', 'epsilon', 'grad'),
                                                           scalar=('lr', 'rho', 'epsilon'),
                                                           reuse=('var', 'accum', 'accum_update'))

    util_apply_op_schedule.common_apply_op_process(util_apply_op_schedule.ApplyOpConfig(args, name), kernel_name)
