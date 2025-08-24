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
this file achieved the apply_proximal_gradient_descent which is
a optimizer operator to update weight, this file contains compute and schedule.

apply_proximal_gradient_descent

Op_description :
Update '*var' as FOBOS algorithm with fixed learning rate.

# apply_proximal_gradient_descent(var,
#   alpha,
#   l1,
#   l2,
#   delta,
#   out,
#   kernel_name='apply_proximal_gradient_descent')

Supportive_dtype_format :
['int32', 'int8', 'uint8', 'float32', 'float16']
['ND', 'NCHW', 'NHWC', 'NC1HWC0']

Constraint :
[1] All : the input tensors must have the same shape and type.
[2] All : shape size limit is 2147483648.
"""
from impl.util import util_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument,invalid-name,too-many-arguments
@register_operator_compute("ApplyProximalGradientDescent", op_mode="dynamic", support_fusion=True)
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

    return var_out


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
    alpha_delta = tbe.vmul(alpha_broad, delta)
    alpha_delta = tbe.vmuls(alpha_delta, tvm.const(-1, "float32"))
    prox_v = tbe.vadd(var, alpha_delta)
    const_zero_tensor = tbe.broadcast(tvm.const(0, var.dtype.lower()), delta.shape)
    var_res = _compute_positive(prox_v, alpha_broad, l1_broad, l2_broad)
    l2_lr = tbe.vmul(l2_broad, alpha_broad)
    l2_lr_1 = tbe.vadds(l2_lr, tvm.const(1, "float32"))
    var_t_neg = tbe.vdiv(prox_v, l2_lr_1)

    var_out = tbe.vcmpsel(l1_broad, const_zero_tensor, 'gt', var_res, var_t_neg)

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
    alpha_l2 = tbe.vmul(alpha_broad, l2_broad)
    alpha_l2_1 = tbe.vadds(alpha_l2, tvm.const(1, "float32"))
    alpha_l1 = tbe.vmul(alpha_broad, l1_broad)
    alpha_l1_neg = tbe.vmuls(alpha_l1, tvm.const(-1, "float32"))
    prox_v_l1 = tbe.vadd(prox_v_abs, alpha_l1_neg)
    max_value = tbe.vmax(
        prox_v_l1,
        tbe.broadcast(tvm.const(0, "float32"), prox_v.shape))
    res = tbe.vdiv(prox_v_sign, alpha_l2_1)
    var_res = tbe.vmul(res, max_value)

    return var_res


# 'pylint: disable=line-too-long, too-many-locals
@register_operator("ApplyProximalGradientDescent")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
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

    var_dtype = var.get('dtype').lower()
    check_list = ('float16', 'float32')
    para_check.check_dtype(var_dtype, check_list, param_name="var")

    shape_scalar = [1]
    compute_dtype = var_dtype
    data_alpha = tvm.placeholder(shape_scalar, name="data_alpha", dtype=compute_dtype)
    data_l1 = tvm.placeholder(shape_scalar, name="data_l1", dtype=compute_dtype)
    data_l2 = tvm.placeholder(shape_scalar, name="data_l2", dtype=compute_dtype)

    ins = classify([var, delta], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_var, _delta) in ins:
        with tbe.compute():
            shape_var, shape_delta = shape_util.variable_shape([_var, _delta])
            data_var = tvm.placeholder(shape_var, name="data_var", dtype=compute_dtype)
            data_delta = tvm.placeholder(shape_delta, name="data_delta", dtype=compute_dtype)
            res = apply_proximal_gradient_descent_compute(data_var, data_alpha, data_l1, data_l2, data_delta, out)
            tensors.append([data_var, data_alpha, data_l1, data_l2, data_delta, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
