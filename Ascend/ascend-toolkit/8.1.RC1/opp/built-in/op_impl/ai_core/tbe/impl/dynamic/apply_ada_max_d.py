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
apply_ada_max

Op_description :
Update '*var' according to the AdaMax algorithm.

# apply_ada_max_d(var,
#   m,
#   v,
#   beta1_power,
#   lr,
#   beta1,
#   beta2,
#   epsilon,
#   grad,
#   m_out,
#   v_out,
#   kernel_name='apply_ada_max_d')

Supportive_dtype_format :
['int32', 'int8', 'uint8', 'float32', 'float16']
['ND', 'NCHW', 'NHWC', 'NC1HWC0']

Constraint :
[1] All : the input tensors must have the same shape and type.
[2] All : shape size limit is 2147483648.
"""

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode


# 'pylint: disable=locally-disabled, too-many-arguments
# 'pylint: disable=unused-argument, invalid-name, too-many-locals
@register_operator_compute("ApplyAdaMaxD", op_mode="dynamic", support_fusion=True)
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

    inp_dtype = var.dtype
    if inp_dtype == 'float16' and tbe_platform.api_check_support("te.lang.cce.vadd", "float32"):
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

    neg_one = tvm.const(-1, dtype=var.dtype)
    rhs = tbe.vmuls(beta1, neg_one)

    one = tvm.const(1, dtype=var.dtype)
    rhs = tbe.vadds(rhs, one)
    lhs = tbe.vsub(grad, m)

    rhs = tbe.vmuls(lhs, rhs[0])

    m = tbe.vadd(m, rhs)

    lhs = tbe.vmuls(v, beta2[0])

    rhs = tbe.vabs(grad)
    v = tbe.vmax(lhs, rhs)

    rhs = tbe.vadds(v, epsilon[0])

    lhs = tbe.vmuls(beta1_power, neg_one)

    lhs = tbe.vadds(lhs, one)

    rhs = tbe.vmuls(rhs, lhs[0])

    lhs = tbe.vmuls(m, lr[0])

    rhs = tbe.vdiv(lhs, rhs)
    var = tbe.vsub(var, rhs)

    res1 = tbe.vadds(var, tvm.const(0.0, dtype="float32"))
    res2 = tbe.vadds(m, tvm.const(0.0, dtype="float32"))
    res3 = tbe.vadds(v, tvm.const(0.0, dtype="float32"))

    if inp_dtype == 'float16':
        res1 = tbe.cast_to(res1, inp_dtype)
        res2 = tbe.cast_to(res2, inp_dtype)
        res3 = tbe.cast_to(res3, inp_dtype)

    return [res1, res2, res3]


@register_operator("ApplyAdaMaxD")
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
    compute_type = var.get("dtype").lower()

    check_list = ("float32", "float16")

    dtype_var = var.get("dtype").lower()
    para_check.check_dtype(dtype_var, check_list, param_name="dtype_var")

    dtype_m = m.get("dtype").lower()
    para_check.check_dtype(dtype_m, check_list, param_name="dtype_m")

    dtype_v = v.get("dtype").lower()
    para_check.check_dtype(dtype_v, check_list, param_name="dtype_v")

    dtype_grad = grad.get("dtype").lower()
    para_check.check_dtype(dtype_grad, check_list, param_name="dtype_grad")

    shape_scalar = [1]
    data_beta1_power = tvm.placeholder(shape_scalar, name="data_beta1_power", dtype=compute_type)
    data_lr = tvm.placeholder(shape_scalar, name="data_lr", dtype=compute_type)
    data_beta1 = tvm.placeholder(shape_scalar, name="data_beta1", dtype=compute_type)
    data_beta2 = tvm.placeholder(shape_scalar, name="data_beta2", dtype=compute_type)
    data_epsilon = tvm.placeholder(shape_scalar, name="data_epsilon", dtype=compute_type)

    ins = classify([var, m, v, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (x1, x2, x3, x9) in ins:
        with tbe.compute():
            var_shape, m_shape, v_shape, grad_shape = shape_util.variable_shape([x1, x2, x3, x9])
            var_data = tvm.placeholder(var_shape, dtype=dtype_var, name="var_data")
            m_data = tvm.placeholder(m_shape, dtype=dtype_m, name="m_data")
            v_data = tvm.placeholder(v_shape, dtype=dtype_v, name="v_data")

            grad_data = tvm.placeholder(grad_shape, dtype=dtype_grad, name="grad_data")

            res = apply_ada_max_d_compute(var_data,
                                          m_data,
                                          v_data,
                                          data_beta1_power,
                                          data_lr,
                                          data_beta1,
                                          data_beta2,
                                          data_epsilon,
                                          grad_data,
                                          var_out,
                                          m_out,
                                          v_out,
                                          kernel_name='apply_ada_max_d')
            tensors.append(
                [var_data, m_data, v_data, data_beta1_power, data_lr, data_beta1, data_beta2, data_epsilon, grad_data] +
                list(res))
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
