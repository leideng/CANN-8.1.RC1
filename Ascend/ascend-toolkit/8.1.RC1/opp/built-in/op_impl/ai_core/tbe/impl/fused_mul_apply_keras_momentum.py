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
fused_mul_apply_keras_momentum
"""
import functools
import operator

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments,invalid-name
def get_op_support_info(var,
                        accum,
                        lr,
                        x1,
                        momentum,
                        x2,
                        out_var,
                        out_accum,
                        use_nesterov,
                        kernel_name='fused_mul_apply_keras_momentum'):
    """
    get fusedMulApplyMomentum slice info
    """
    format_var = var.get("format")
    shape_var = var.get("shape")
    support_format = ["FRACTAL_Z", "C1HWNCoC0", "NC1HWC0", "ND", "NCHW", "NHWC"]
    if format_var in support_format:
        axis_reduce_list = None
        axis_split_list = []
        for idx, _ in enumerate(shape_var):
            split_info = [SplitInput([0, [idx], [-1], [-1]], [1, [idx], [-1], [-1]], [3, [idx], [-1], [-1]]),
                          SplitOutput([0, [idx]], [1, [idx]])]
            axis_split_list.append(split_info)
    else:
        axis_split_list = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_list, axis_reduce_list, 0, 0)

    return op_cal_info_in_json


# 'pylint: disable=unused-argument,invalid-name,too-many-locals,too-many-arguments
@register_operator_compute("fused_mul_apply_keras_momentum", op_mode="static", support_fusion=True)
def fused_mul_apply_keras_momentum_compute(var,
                                           accum,
                                           lr,
                                           x1,
                                           momentum,
                                           x2,
                                           out_var,
                                           out_accum,
                                           use_nesterov,
                                           kernel_name='fused_mul_apply_keras_momentum'):
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
                 "cce_fused_mul_apply_keras_momentum" (optional).

    Returns:
    -------
    out_var, out_accum
    """

    # cast to float32 for higher accuracy
    inp_dtype = var.dtype
    # check the instruction supports or not
    vmul_support = tbe_platform.api_check_support("te.lang.cce.vmul", "float32")
    if inp_dtype == "float32" and not vmul_support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'var', [], inp_dtype)

    # update var and accum according to the momentum scheme
    # `accum = accum * momentum - grad * lr`
    x2_brc = tbe.broadcast(x2, x1.shape)
    grad = tbe.vmul(x1, x2_brc)
    accum_momen = tvm.compute(accum.shape, lambda *indices: accum(*indices) * momentum[0], tag='elewise_single_VS_mul')
    grad_lr = tvm.compute(grad.shape, lambda *indices: grad(*indices) * lr[0], tag='elewise_single_VS_mul')
    out_accum = tbe.vsub(accum_momen, grad_lr)

    # `var = var + accum * momentum - grad * lr`
    if use_nesterov is True:
        accum_momen2 = tvm.compute(accum.shape,
                                   lambda *indices: out_accum(*indices) * momentum[0],
                                   tag='elewise_single_VS_mul')
        add_var_am = tbe.vadd(var, accum_momen2)
        out_var = tbe.vsub(add_var_am, grad_lr)
    # `var = var + accum`
    else:
        out_var = tbe.vadd(var, out_accum)


    return out_var, out_accum


def _get_placeholder(dict_list, name_list):
    list_placeholder = []
    var_shape = []
    for var, name in zip(dict_list, name_list):
        shape = var.get('shape')
        dtype = var.get('dtype').lower()
        if name == 'var':
            var_shape = list(shape)
        if name != 'lr' and name != 'momentum' and name != 'x2' and var_shape != list(shape):
            error_manager_vector.raise_err_inputs_shape_not_equal('fused_mul_apply_keras_momentum',
                                                                  'var',
                                                                  name,
                                                                  var_shape,
                                                                  shape,
                                                                  var_shape)
        if (name in ('lr', 'momentum', 'x2')) and shape[0] != 1:
            error_manager_vector.raise_err_check_params_rules('fused_mul_apply_keras_momentum',
                                                              'the shapes of lr, momentum and x2 must be scalar',
                                                              name,
                                                              shape)

        para_check.check_dtype(dtype, ('float32', 'float16'), param_name="var")
        para_check.check_shape(shape, param_name="var")
        shape_refine = (functools.reduce(operator.mul, shape),)
        list_placeholder.append(tvm.placeholder(shape=shape_refine, name=name, dtype=dtype))
    return list_placeholder


# 'pylint: disable=unbalanced-tuple-unpacking,too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def fused_mul_apply_keras_momentum(var,
                                   accum,
                                   lr,
                                   x1,
                                   momentum,
                                   x2,
                                   out_var,
                                   out_accum,
                                   use_locking=False,
                                   use_nesterov=False,
                                   kernel_name="fused_mul_apply_keras_momentum"):
    """
    Update '*var' according to the ApplyKerasMomentum algorithm.

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

    use_nesterov: bool. If true, use nesterov computing grad,
                 default value is False.

    kernel_name : cce kernel name, default value is "fused_mul_apply_keras_momentum".

    Returns
    -------
    None
    """

    input_name_list = ['var', 'accum', 'lr', 'x1', 'momentum', 'x2']
    var, accum, lr, x1, momentum, x2 = _get_placeholder([var, accum, lr, x1, momentum, x2], input_name_list)
    out_var, out_accum = fused_mul_apply_keras_momentum_compute(var, accum, lr, x1, momentum, x2, out_var, out_accum,
                                                                use_nesterov)
    outs = [out_var, out_accum]
    build_list = [var, accum, lr, x1, momentum, x2, out_var, out_accum]

    with tvm.target.cce():
        sch = auto_schedule(outs)
    config = {"name": kernel_name, "tensor_list": build_list}
    build(sch, config)
