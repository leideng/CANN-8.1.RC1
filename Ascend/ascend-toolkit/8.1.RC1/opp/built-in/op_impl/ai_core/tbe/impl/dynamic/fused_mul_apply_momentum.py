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
dynamic fused_mul_apply_momentum
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.fused_mul_apply_momentum import get_op_support_info as fused_mul_apply_momentum_get_op_support_info


def get_op_support_info(var,
                        accum,
                        lr,
                        x1,
                        momentum,
                        x2,
                        out_var,
                        out_accum,
                        use_nesterov,
                        kernel_name='fused_mul_apply_momentum'):
    """
    get fusedMulApplyMomentum slice info
    """
    return fused_mul_apply_momentum_get_op_support_info(var, accum, lr, x1, momentum, x2, out_var,
                                                        out_accum, use_nesterov, kernel_name)


# 'pylint: disable=unused-argument,invalid-name, too-many-locals,too-many-arguments
@register_operator_compute("FusedMulApplyMomentum", op_mode="dynamic", support_fusion=True)
def fused_mul_apply_momentum_compute(var,
                                     accum,
                                     lr,
                                     x1,
                                     momentum,
                                     x2,
                                     out_var,
                                     out_accum,
                                     use_nesterov,
                                     kernel_name='fused_mul_apply_momentum'):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    accum = accum * momentum + x1 * x2
    if use_nesterov is True:
        var -= x1 * x2 * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : mutable tensor var.
    accum: mutable tensor accum.
    lr : scalar lr.
    x1 : tensor x1.
    momentum : scalar momentum.
    x2 : scalar x2.
    out_var : the var output.
    out_accum : the accum output
    use_nesterov: bool. If true, use nesterov computing grad,
                  default value is False.
    kernel_name : cce kernel name, default value is
                 "cce_fused_mul_apply_momentum" (optional).
    Returns:
    -------
    out_var, out_accum
    """

    # cast to float32 for higher accuracy
    dtype = var.dtype

    if dtype == "float16":
        var = tbe.cast_to(var, "float32")
        accum = tbe.cast_to(accum, "float32")
        lr = tbe.cast_to(lr, "float32")
        x1 = tbe.cast_to(x1, "float32")
        x2 = tbe.cast_to(x2, "float32")
        momentum = tbe.cast_to(momentum, "float32")

    # calc grad
    x2_brc = tbe.broadcast(x2, x1.shape)
    grad = tbe.vmul(x1, x2_brc)
    # update accum
    momentum_brc = tbe.broadcast(momentum, accum.shape)
    accum_delta = tbe.vmul(accum, momentum_brc)
    accum_t = tbe.vadd(accum_delta, grad)

    # update var
    lr_brc = tbe.broadcast(lr, accum.shape)
    if use_nesterov:
        var_delta = tbe.vmul(grad, lr_brc)
        var_delta_2 = tbe.vmul(accum_t, momentum_brc)
        var_delta_2 = tbe.vmul(var_delta_2, lr_brc)
        var_delta = tbe.vadd(var_delta, var_delta_2)
        var_t = tbe.vsub(var, var_delta)
    else:
        var_delta = tbe.vmul(accum_t, lr_brc)
        var_t = tbe.vsub(var, var_delta)

    if dtype == "float16":
        var_t = tbe.cast_to(var_t, "float16")
        accum_t = tbe.cast_to(accum_t, "float16")

    res = [var_t, accum_t]

    return res


# 'pylint: disable=unbalanced-tuple-unpacking, too-many-arguments
@register_operator("FusedMulApplyMomentum")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def fused_mul_apply_momentum(var,
                             accum,
                             lr,
                             x1,
                             momentum,
                             x2,
                             out_var,
                             out_accum,
                             use_nesterov=False,
                             kernel_name="fused_mul_apply_momentum"):
    """
    Update '*var' according to the ApplyMomentum algorithm.
    accum = accum * momentum + x1 * x2
    if use_nesterov is True:
        var -= gard * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : the dict of mutable tensor var, only support float16, float32.
    accum: the dict of mutable tensor accum. Must have the same dtype as `var`.
    lr : the dict of scalar lr. Must have the same dtype as `var`.
    x1 : the dict of tensor grad. Must have the same dtype as `var`.
    momentum : the dict of scalar momentum. Must have the same dtype as `var`.
    x2 : the dict of scalar grad. Must have the same dtype as `var`.
    out_var : the dict of var output.
    out_accum : the dict of accum output
    use_nesterov: bool. If true, use nesterov computing  grad,
                 default value is False.
    kernel_name : cce kernel name, default value is "fused_mul_apply_momentum".

    Returns
    -------
    None
    """

    var_dtype = var.get("dtype").lower()
    accum_dtype = accum.get("dtype").lower()
    lr_dtype = lr.get("dtype").lower()
    x1_dtype = x1.get("dtype").lower()
    momentum_dtype = momentum.get("dtype").lower()
    x2_dtype = x2.get("dtype").lower()

    check_list = ("float32", "float16")

    para_check.check_dtype(var_dtype, check_list, param_name="input_var")
    para_check.check_dtype(accum_dtype, check_list, param_name="input_accum")
    para_check.check_dtype(lr_dtype, check_list, param_name="input_lr")
    para_check.check_dtype(x1_dtype, check_list, param_name="input_x1")
    para_check.check_dtype(momentum_dtype, check_list, param_name="input_momentum")
    para_check.check_dtype(x2_dtype, check_list, param_name="input_x2")

    shape_scalar = [1]
    data_lr = tvm.placeholder(shape_scalar, dtype=var_dtype, name="data_lr")
    data_momentum = tvm.placeholder(shape_scalar, dtype=var_dtype, name="data_momentum")
    data_x2 = tvm.placeholder(shape_scalar, dtype=var_dtype, name="data_x2")

    ins = classify([var, accum, x1], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_var, _accum, _x1) in ins:
        with tbe.compute():
            shape_var, shape_accum, shape_x1 = shape_util.variable_shape([_var, _accum, _x1])
            data_var = tvm.placeholder(shape_var, dtype=var_dtype, name="data_var")
            data_accum = tvm.placeholder(shape_accum, dtype=var_dtype, name="data_accum")
            data_x1 = tvm.placeholder(shape_x1, dtype=var_dtype, name="data_x1")
            res = fused_mul_apply_momentum_compute(data_var, data_accum,
                                                   data_lr, data_x1,
                                                   data_momentum, data_x2,
                                                   out_var, out_accum,
                                                   use_nesterov, kernel_name)

            tensor_list = [data_var, data_accum, data_lr, data_x1,
                           data_momentum, data_x2] + list(res)
            tensors.append(tensor_list)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
