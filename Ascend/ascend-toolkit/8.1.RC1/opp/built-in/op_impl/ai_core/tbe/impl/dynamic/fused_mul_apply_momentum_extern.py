# Copyright 2022 Huawei Technologies Co., Ltd
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
dynamic fused_mul_apply_momentum_extern
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument, too-many-arguments
@register_operator_compute("FusedMulApplyMomentumExtern", op_mode="dynamic", support_fusion=True)
def fused_mul_apply_momentum_extern_compute(var,
                                            accum,
                                            lr,
                                            x1,
                                            momentum,
                                            x2,
                                            var_copy,
                                            out_fp32,
                                            out_fp16,
                                            out_accum,
                                            use_nesterov,
                                            kernel_name="fused_mul_apply_momentum_extern"):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    `accum = accum * momentum + x1 * x2`
    `if use_nesterov is True:`
        `var -= x1 * x2 * lr + accum * momentum * lr`
    `else:`
        `var -= accum * lr`

    Parameters:
    ----------
    var : mutable tensor var. Dtype is float32
    accum: mutable tensor accum.
    lr : scalar lr.
    x1 : tensor x1.
    momentum : scalar momentum.
    x2 : scalar x2.
    var_copy : mutable tensor var. Dtype is float16.
    out_fp32 : the dict of output. Dtype is float32.
    out_fp16 : the dict of output. Dtype is float16.
    out_accum : the dict of output. Dtype is same as input accum.
    use_nesterov: bool. If true, use nesterov computing grad,
                  default value is False.
    kernel_name : cce kernel name, default value is
                 "cce_fused_mul_apply_momentum_extern" (optional).

    Returns:
    -------
    out_fp32, out_fp16, out_accum
    """

    # cast to float32 for higher accuracy
    dtype = accum.dtype
    if dtype == "float16":
        accum = tbe.cast_to(accum, "float32")
        lr = tbe.cast_to(lr, "float32")
        x1 = tbe.cast_to(x1, "float32")
        x2 = tbe.cast_to(x2, "float32")
        momentum = tbe.cast_to(momentum, "float32")

    # calc grad
    grad = tvm.compute(x1.shape, lambda *indice: x1(*indice) * x2[0], tag='elewise_single_VS_mul')
    # update accum
    accum_delta = tvm.compute(accum.shape, lambda *indice: accum(*indice) * momentum[0], tag='elewise_single_VS_mul')
    accum_t = tbe.vadd(accum_delta, grad)

    # update var
    if use_nesterov:
        var_delta = tvm.compute(grad.shape, lambda *indice: grad(*indice) * lr[0], tag='elewise_single_VS_mul')
        var_delta_2 = tvm.compute(accum_t.shape,
                                  lambda *indice: accum_t(*indice) * momentum[0],
                                  tag='elewise_single_VS_mul')
        var_delta_2 = tvm.compute(var_delta_2.shape,
                                  lambda *indice: var_delta_2(*indice) * lr[0],
                                  tag='elewise_single_VS_mul')
        var_delta = tbe.vadd(var_delta, var_delta_2)
        var_t_fp32 = tbe.vsub(var, var_delta)
        var_delta_fp16 = tbe.cast_to(var_delta, "float16")
        var_t_fp16 = tbe.vsub(var_copy, var_delta_fp16)
    else:
        var_delta = tvm.compute(accum_t.shape, lambda *indice: accum_t(*indice) * lr[0], tag='elewise_single_VS_mul')
        var_t_fp32 = tbe.vsub(var, var_delta)
        var_delta_fp16 = tbe.cast_to(var_delta, "float16")
        var_t_fp16 = tbe.vsub(var_copy, var_delta_fp16)

    if dtype == "float16":
        accum_t = tbe.cast_to(accum_t, "float16")

    res = [var_t_fp32, var_t_fp16, accum_t]

    return res


# 'pylint: disable=too-many-arguments, too-many-locals
@register_operator("FusedMulApplyMomentumExtern")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def fused_mul_apply_momentum_extern(var,
                                    accum,
                                    lr,
                                    x1,
                                    momentum,
                                    x2,
                                    var_copy,
                                    out_fp32,
                                    out_fp16,
                                    out_accum,
                                    use_nesterov=False,
                                    kernel_name="fused_mul_apply_momentum"):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    `accum = accum * momentum + x1 * x2`
    `if use_nesterov is True:`
        `var -= gard * lr + accum * momentum * lr`
    `else:`
        `var -= accum * lr`

    Parameters:
    ----------
    var : the dict of mutable tensor var, Dtype is float32.
    accum: the dict of mutable tensor accum.
    lr : the dict of scalar lr.
    x1 : the dict of tensor grad.
    momentum : the dict of scalar momentum.
    x2 : the dict of tensor grad.
    var_copy : the dict of mutable tensor var, Dtype is float16.
    out_fp32 : the dict of output. Dtype is float32.
    out_fp16 : the dict of output. Dtype is float16.
    out_accum : the dict of output. Dtype is same as input accum.
    use_nesterov: bool. If true, use nesterov computing grad,
                 default value is False.
    kernel_name : cce kernel name, default value is "fused_mul_apply_momentum".

    Returns
    -------
    None
    """

    var_dtype = var.get("dtype").lower()
    var_copy_dtype = var_copy.get("dtype").lower()
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
    para_check.check_dtype(var_copy_dtype, check_list, param_name="input_var_copy")

    shape_scalar = [1]
    data_lr = tvm.placeholder(shape_scalar, dtype=lr_dtype, name="data_lr")
    data_momentum = tvm.placeholder(shape_scalar, dtype=momentum_dtype, name="data_momentum")
    data_x2 = tvm.placeholder(shape_scalar, dtype=x2_dtype, name="data_x2")

    ins = classify([var, accum, x1, var_copy], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_var, _accum, _x1, _var_copy) in ins:
        with tbe.compute():
            shape_var, shape_accum, shape_x1, shape_var_copy = shape_util.variable_shape([_var, _accum, _x1, _var_copy])
            data_var = tvm.placeholder(shape_var, dtype=var_dtype, name="data_var")
            data_accum = tvm.placeholder(shape_accum, dtype=accum_dtype, name="data_accum")
            data_x1 = tvm.placeholder(shape_x1, dtype=x1_dtype, name="data_x1")
            data_var_copy = tvm.placeholder(shape_var_copy, dtype=var_copy_dtype, name="data_var_copy")
            res = fused_mul_apply_momentum_extern_compute(data_var, data_accum,
                                                          data_lr, data_x1,
                                                          data_momentum, data_x2,
                                                          data_var_copy,
                                                          out_fp32, out_fp16, out_accum,
                                                          use_nesterov, kernel_name)
            tensor_list = [data_var, data_accum, data_lr, data_x1,
                           data_momentum, data_x2, data_var_copy] + list(res)
            tensors.append(tensor_list)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
