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
apply_adagradv2_d
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


class ApplyAdagradV2DAttrInfo:
    """
    define attr info
    """
    ATTR_EPSILON = OpAttr(0, "epsilon", "Float")


# 'pylint: disable=unused-argument,invalid-name,too-many-arguments,too-many-locals
@register_operator_compute("ApplyAdagradV2D", op_mode="dynamic", support_fusion=False)
def apply_adagradv2_d_compute(var,
                              accum,
                              lr,
                              grad,
                              out_var,
                              out_accum,
                              epsilon,
                              update_slots,
                              kernel_name="apply_adagradv2_d"):
    """
    ApplyAdagradv2 algorithm:

    if update_slots
        accum += grad ** 2
    var -= lr * grad / (accum.sqrt() + epsilon)

    Parameters:
    ----------
    var: placeholder, input tensor with dtype float32

    accum: placeholder, has same shape and dtype as var

    lr: placeholder, has same dtype as var

    grad: placeholder, has same shape and dtype as var

    out_var: output var, has same shape and dtype as var

    out_accum: output accum, has same shape and dtype as var

    epsilon: scalar, has same dtype as var

    update_slots: An optional 'bool'. Defaults to 'True',
        If True, the accum tensor will be updated;
        otherwise the option is False, the accum tensor will not be update.

    kernel_name : cce kernel name, default value is "apply_adagradv2_d".

    Returns:
    -------
    out_var, out_accum
    """
    input_dtype = var.dtype
    vmul_support = tbe_platform.api_check_support("te.lang.cce.vmul", "float32")
    if input_dtype == "float32" and not vmul_support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'var', [], input_dtype)

    if update_slots is True:
        grad_square = tbe.vmul(grad, grad)
        out_accum = tbe.vadd(accum, grad_square)
    else:
        out_accum = tbe.vadds(accum, tvm.const(0.0, input_dtype))

    lr_brc = tbe.broadcast(lr, grad.shape)
    lr_grad = tbe.vmul(grad, lr_brc)
    sqrt_accum = tbe.vsqrt(out_accum)
    scalar = get_attr_by_cls(epsilon, ApplyAdagradV2DAttrInfo.ATTR_EPSILON, input_dtype)
    sqrt_accum_epsilon = tbe.vadds(sqrt_accum, scalar)
    update = tbe.vdiv(lr_grad, sqrt_accum_epsilon)
    out_var = tbe.vsub(var, update)

    return out_var, out_accum


# 'pylint: disable=unbalanced-tuple-unpacking,invalid-name,too-many-arguments
@register_operator("ApplyAdagradV2D")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def apply_adagradv2_d(var,
                      accum,
                      lr,
                      grad,
                      out_var,
                      out_accum,
                      epsilon,
                      update_slots=True,
                      kernel_name="apply_adagradv2_d"):
    """
    Update '*var' according to the Adagrad algorithm.

    if update_slots
        accum += grad ** 2
    var -= lr * grad / (accum.sqrt() + epsilon)

    Parameters:
    ----------
    var: the dict of var, only support float32

    accum: the dict of accum, only support float32

    lr: the dict of lr, only support float32

    grad: the dict of grad, only support float32

    out_var: the dict of output, only support float32

    out_accum: the dict of output, only support float32

    epsilon: scalar, only support float32

    update_slots: An optional 'bool'. Defaults to 'True',
        If True, the accum tensor will be updated;
        otherwise the option is False, the accum tensor will not be update.

    kernel_name : cce kernel name, default value is "apply_adagradv2_d".

    Returns
    -------
    None
    """
    check_list = ("float32")

    dtype_var = var.get("dtype").lower()
    para_check.check_dtype(dtype_var, check_list, param_name="dtype_var")

    dtype_accum = accum.get("dtype").lower()
    para_check.check_dtype(dtype_accum, check_list, param_name="dtype_accum")

    dtype_lr = lr.get("dtype").lower()
    para_check.check_dtype(dtype_lr, check_list, param_name="dtype_lr")
    lr_data = tvm.placeholder([1], dtype=dtype_lr, name="lr_data")

    dtype_grad = grad.get("dtype").lower()
    para_check.check_dtype(dtype_grad, check_list, param_name="dtype_grad")

    ins = classify([var, accum, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (x1, x2, x3) in ins:
        with tbe.compute():
            var_shape, accum_shape, grad_shape = shape_util.variable_shape([x1, x2, x3])
            var_data = tvm.placeholder(var_shape, dtype=dtype_var, name="var_data")
            accum_data = tvm.placeholder(accum_shape, dtype=dtype_accum, name="accum_data")
            grad_data = tvm.placeholder(grad_shape, dtype=dtype_grad, name="grad_data")
            out_var, out_accum = apply_adagradv2_d_compute(var_data,
                                                           accum_data,
                                                           lr_data,
                                                           grad_data,
                                                           out_var,
                                                           out_accum,
                                                           epsilon,
                                                           update_slots,
                                                           kernel_name="apply_adagradv2_d")
            res = [out_var, out_accum]
            tensors.append((var_data, accum_data, lr_data, grad_data, out_var, out_accum))
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
