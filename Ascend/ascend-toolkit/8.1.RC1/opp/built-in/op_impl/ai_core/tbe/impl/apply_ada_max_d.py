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
apply_ada_max_d
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_apply_op_schedule


# 'pylint: disable=locally-disabled, too-many-arguments
# 'pylint: disable=unused-argument, invalid-name, too-many-locals
@register_operator_compute("apply_ada_max_d", op_mode="static", support_fusion=True)
def apply_ada_max_d_compute(var,
                            m,
                            v,
                            beta1_power,
                            lr,
                            beta1,
                            beta2,
                            epsilon,
                            grad,
                            var_out,
                            m_out,
                            v_out,
                            kernel_name='apply_ada_max_d'):
    """
    Update '*var' according to the AdaMax algorithm.

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    v_t <- max(beta2 * v_{t-1}, abs(g))
    variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)

    Parameters:
    ----------
    var : mutable tensor var.

    m : mutable tensor m.

    v : mutable tensor v.

    beta1_power : scalar beta1_power.

    lr : scalar lr.

    beta1 : scalar beta1.

    beta2 : scalar beta2.

    epsilon : scalar epsilon.

    grad : tensor grad.

    var_out : the dict of var output.

    m_out : the dict of m output.

    v_out : the dict of v output.

    kernel_name : cce kernel name, default value is "apply_ada_max_d" (optional).

    Returns:
    -------
    None
    """

    # cast to float32 for improved accuracy
    inp_dtype = var.dtype
    if inp_dtype == 'float16' and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        var = tbe.cast_to(var, 'float32')
        m = tbe.cast_to(m, 'float32')
        v = tbe.cast_to(v, 'float32')
        lr = tbe.cast_to(lr, 'float32')
        beta1_power = tbe.cast_to(beta1_power, 'float32')
        beta1 = tbe.cast_to(beta1, 'float32')
        beta2 = tbe.cast_to(beta2, 'float32')
        epsilon = tbe.cast_to(epsilon, 'float32')
        grad = tbe.cast_to(grad, 'float32')
    else:
        m = tbe.vmuls(m, tvm.const(1, dtype=inp_dtype))

    rhs = tvm.compute(beta1.shape,
                      lambda *indices: beta1(*indices) * -1,
                      tag='elewise_single_VS_mul')
    rhs = tvm.compute(rhs.shape,
                      lambda *indices: rhs(*indices)
                                       + tvm.const(1.0, dtype=rhs.dtype),
                      tag='elewise_single_VS_add')
    lhs = tbe.vsub(grad, m)
    rhs = tvm.compute(lhs.shape,
                      lambda *indices: lhs(*indices) * rhs[0],
                      tag='elewise_single_VS_mul')
    m = tbe.vadd(m, rhs)

    lhs = tvm.compute(v.shape,
                      lambda *indices: v(*indices) * beta2[0],
                      tag='elewise_single_VS_mul')
    rhs = tbe.vabs(grad)
    v = tbe.vmax(lhs, rhs)

    rhs = tvm.compute(v.shape,
                      lambda *indices: v(*indices) + epsilon[0],
                      tag='elewise_single_VS_add')
    lhs = tvm.compute(beta1_power.shape,
                      lambda *indices: beta1_power(*indices) * -1,
                      tag='elewise_single_VS_mul')
    lhs = tvm.compute(lhs.shape,
                      lambda *indices: lhs(*indices)
                                       + tvm.const(1.0, dtype=lhs.dtype),
                      tag='elewise_single_VS_add')
    rhs = tvm.compute(rhs.shape,
                      lambda *indices: rhs(*indices) * lhs[0],
                      tag='elewise_single_VS_mul')
    lhs = tvm.compute(m.shape,
                      lambda *indices: m(*indices) * lr[0],
                      tag='elewise_single_VS_mul')
    rhs = tbe.vdiv(lhs, rhs)
    var = tbe.vsub(var, rhs)

    res1 = tbe.vadds(var, tvm.const(0.0, dtype="float32"))
    res2 = tbe.vadds(m, tvm.const(0.0, dtype="float32"))
    res3 = tbe.vadds(v, tvm.const(0.0, dtype="float32"))

    if inp_dtype == 'float16':
        res1 = tbe.cast_to(res1, inp_dtype)
        res2 = tbe.cast_to(res2, inp_dtype)
        res3 = tbe.cast_to(res3, inp_dtype)

    def _compute(*index):
        return [m(*index), v(*index), var(*index), res1(*index), res2(*index), res3(*index)]

    return tvm.compute(var.shape, _compute, name="outputs")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def apply_ada_max_d(var,
                    m,
                    v,
                    beta1_power,
                    lr,
                    beta1,
                    beta2,
                    epsilon,
                    grad,
                    var_out,
                    m_out,
                    v_out,
                    kernel_name='apply_ada_max_d'):
    """
    Update '*var' according to the AdaMax algorithm.

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    v_t <- max(beta2 * v_{t-1}, abs(g))
    variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)

    Parameters:
    ----------
    var : the dict of mutable tensor var. Must be one of the following data types:
          `float32`, `float16`.

    m: the dict of mutable tensor m. Must have the same data type as `var`.

    v : the dict of mutable tensor v. Must have the same data type as `var`.

    beta1_power : the dict of scalar beta1_power.
        Must have the same data type as `var`.

    lr : the dict of scalar lr. Must have the same data type as `var`.

    beta1 : the dict of scalar beta1. Must have the same data type as `var`.

    beta2 : the dict of scalar beta2. Must have the same data type as `var`.

    epsilon : the dict of scalar epsilon. Must have the same data type as `var`.

    grad : the dict of tensor grad. Must have the same data type as `var`.

    var_out : the dict of var output.

    m_out : the dict of m output.

    v_out : the dict of v output.

    kernel_name : cce kernel name, default value is "apply_ada_max" (optional).

    Returns:
    -------
    None
    """

    input_dict = (var, m, v, beta1_power, lr, beta1, beta2, epsilon, grad)

    args = util_apply_op_schedule.ApplyOpConfig.TensorArgs(input_dict, apply_ada_max_d_compute,
                                    [var_out, m_out, v_out], 14)
    name = util_apply_op_schedule.ApplyOpConfig.TensorName(all=('var', 'm', 'v', 'beta1_power', 'lr',
                                                                'beta1', 'beta2', 'epsilon', 'grad'),
                                                           scalar=('lr', 'beta1_power', 'beta1',
                                                                   'beta2', 'epsilon'),
                                                           reuse=('m', 'v', 'var'))

    util_apply_op_schedule.common_apply_op_process(util_apply_op_schedule.ApplyOpConfig(args, name), kernel_name)
