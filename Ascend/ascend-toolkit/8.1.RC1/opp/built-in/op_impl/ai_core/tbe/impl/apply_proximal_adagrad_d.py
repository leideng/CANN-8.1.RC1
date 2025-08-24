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
apply_proximal_adagrad
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util import util_apply_op_schedule
from impl.util import util_compute
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# 'pylint: disable=too-few-public-methods, not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    CONST_ZERO = 0
    CONST_ONE = 1


def len_c(input_format, shape):
    """
    Judge the c axis
    :param input_format: input format
    :param shape: input shape
    :return: c axis and nd_fromat
    """
    nd_format = False
    shape_c = 1
    if input_format == "NHWC" and len(shape) == 4:
        shape_c = shape[3]
    elif input_format == "NCHW" and len(shape) == 4:
        shape_c = shape[1]
    else:
        nd_format = True
    return shape_c, nd_format


# 'pylint: disable=unused-variable,too-many-arguments,too-many-locals,unused-argument
# 'pylint: disable=too-many-boolean-expressions,too-many-statements
def op_select_format(var, accum, lr, l1, l2, grad, var_out,
                     accum_out, use_locking=False,
                     kernel_name="apply_proximal_adagrad_d"):
    """
    Select format according to the following rules.
    input2 input3 input4 only support ND,and output0 output1 is the same as input0
    1.When the input0 or input1 or input5 shape is -2, all inputs and outputs only ND format is supported
    2.Supports input0 input1 input5 ND, 5HD, FZ when the N and C axis can be divisible by 16
    3.Supports input0 input1 input5 ND, 5HD when the N and C axis can be divisible by 16
    4.In other cases, only ND is supported
    """
    var_format = var.get("format")
    accum_format = accum.get("format")
    grad_format = grad.get("format")
    var_shape = list(var.get("ori_shape"))
    accum_shape = list(accum.get("ori_shape"))
    grad_shape = list(grad.get("ori_shape"))
    var_n = var_shape[0]
    accum_n = accum_shape[0]
    grad_n = grad_shape[0]
    var_c, var_nd_format = len_c(var_format, var_shape)
    accum_c, accum_nd_format = len_c(accum_format, accum_shape)
    grad_c, grad_nd_format = len_c(grad_format, grad_shape)
    if var_n == -2 or accum_n == -2 or grad_n == -2 or var_nd_format or accum_nd_format or grad_nd_format:
        format_list = "ND,ND"
        dtype_list = "float16,float32"
        unknowshape_format = format_list
        format_list_nd = format_list
        dtype_list_nd = dtype_list
        unknowshape_format_nd = format_list
    else:
        support_fz_var = (var_c % 16 == 0) and (var_n % 16 == 0)
        support_5hd_var = var_c % 16 == 0
        support_fz_accum = (accum_c % 16 == 0) and (accum_n % 16 == 0)
        support_5hd_accum = accum_c % 16 == 0
        support_fz_grad = (grad_c % 16 == 0) and (grad_n % 16 == 0)
        support_5hd_grad = grad_c % 16 == 0
        if support_fz_var and support_fz_accum and support_fz_grad:
            format_list = "ND,ND,FRACTAL_Z,FRACTAL_Z,NC1HWC0,NC1HWC0"
            dtype_list = "float16,float32,float16,float32,float16,float32"
            unknowshape_format = format_list
            format_list_nd = "ND,ND,ND,ND,ND,ND"
            dtype_list_nd = dtype_list
            unknowshape_format_nd = format_list_nd
        elif support_5hd_var and support_5hd_accum and support_5hd_grad:
            format_list = "ND,ND,NC1HWC0,NC1HWC0"
            dtype_list = "float16,float32,float16,float32"
            unknowshape_format = format_list
            format_list_nd = "ND,ND,ND,ND"
            dtype_list_nd = dtype_list
            unknowshape_format_nd = format_list_nd
        else:
            format_list = "ND,ND"
            dtype_list = "float16,float32"
            unknowshape_format = format_list
            format_list_nd = format_list
            dtype_list_nd = dtype_list
            unknowshape_format_nd = format_list
    input0 = gen_param(classify="input0", name="var",
                       datatype=dtype_list,
                       format=format_list,
                       unknownshape_format=unknowshape_format)
    input1 = gen_param(classify="input1", name="accum",
                       datatype=dtype_list,
                       format=format_list,
                       unknownshape_format=unknowshape_format)
    input2 = gen_param(classify="input2", name="lr",
                       datatype=dtype_list_nd,
                       format=format_list_nd,
                       unknownshape_format=unknowshape_format_nd)
    input3 = gen_param(classify="input3", name="l1",
                       datatype=dtype_list_nd,
                       format=format_list_nd,
                       unknownshape_format=unknowshape_format_nd)
    input4 = gen_param(classify="input4", name="l2",
                       datatype=dtype_list_nd,
                       format=format_list_nd,
                       unknownshape_format=unknowshape_format_nd)
    input5 = gen_param(classify="input5", name="grad",
                       datatype=dtype_list,
                       format=format_list,
                       unknownshape_format=unknowshape_format)
    output0 = gen_param(classify="output0", name="var",
                       datatype=dtype_list,
                       format=format_list,
                       unknownshape_format=unknowshape_format)
    output1 = gen_param(classify="output1", name="accum",
                       datatype=dtype_list,
                       format=format_list,
                       unknownshape_format=unknowshape_format)
    param_list = [input0, input1, input2, input3, input4, input5, output0, output1]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _check_shape_is_same(var, accum, grad):
    """
    Check whether var.shape accum.shape and grad.shape is same or not.

    Parameters
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    accum: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.

    Returns:
    None
    """
    shape_var = var.get("shape")
    shape_accum = accum.get("shape")
    shape_grad = grad.get("shape")
    if shape_var != shape_accum or shape_var != shape_grad:
        error_detail = "shape of var and accum and grad should be same"
        error_manager_vector.raise_err_input_shape_invalid("apply_proximal_adagrad_d", "var or accum or grad",
                                                           error_detail)


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=too-many-locals,unused-argument,invalid-name
@register_operator_compute("apply_proximal_adagrad_d", op_mode="static", support_fusion=True)
def apply_proximal_adagrad_d_compute(var, accum, lr, l1, l2, grad, var_out,
                                     accum_out, use_locking=False,
                                     kernel_name="apply_proximal_adagrad"):
    """
    the operator's compute
    accum += grad * grad
    learning_rate = lr_broad * rsqrt(accum)
    prox_v = var - grad * learning_rate
    if l1 > 0 :
        var = sign(prox_v)/(1+learning_rate*l2)*max{|prox_v|-learning_rate*l1,0}
    else:
        var = prox_v / (1+l2*learning_rate)

    Parameters
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    accum: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    accum_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'accum'.
    use_locking: bool
        default value is "False"
    kernel_name: str
        kernel name, default value is "apply_proximal_adagrad_d"

    Returns:
        the value of out_var, accum_out, out_data
    """
    dtype = var.dtype
    has_improve_precision = False
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vsqrt", "float32"):
        var = tbe.cast_to(var, "float32")
        accum = tbe.cast_to(accum, "float32")
        lr = tbe.cast_to(lr, "float32")
        l1 = tbe.cast_to(l1, "float32")
        l2 = tbe.cast_to(l2, "float32")
        grad = tbe.cast_to(grad, "float32")
        has_improve_precision = True

    lr_broad = tbe.broadcast(lr, var.shape)
    l1_broad = tbe.broadcast(l1, var.shape)
    l2_broad = tbe.broadcast(l2, var.shape)

    grad_2 = tbe.vmul(grad, grad)
    accum_out = tbe.vadd(accum, grad_2)
    accum_sqrt = tbe.vsqrt(accum_out)
    learning_rate = tbe.vdiv(lr_broad, accum_sqrt)
    learning_rate_grad = tbe.vmul(grad, learning_rate)
    prox_v = tbe.vsub(var, learning_rate_grad)
    l2_lr = tbe.vmul(l2_broad, learning_rate)
    l2_lr_1 = tbe.vadds(l2_lr, tvm.const(Constant.CONST_ONE, "float32"))
    prox_v_abs = tbe.vabs(prox_v)
    prox_v_sign = util_compute.sign(prox_v)
    learning_rate_l1 = tbe.vmul(learning_rate, l1_broad)
    prox_v_l1 = tbe.vsub(prox_v_abs, learning_rate_l1)
    max_value = tbe.vmax(prox_v_l1, tbe.broadcast(
        tvm.const(Constant.CONST_ZERO, "float32"), prox_v.shape))
    var_res = tbe.vmul(prox_v_sign, max_value)
    var_new = tbe.vdiv(var_res, l2_lr_1)
    output_data = tbe.vadds(var_new, tvm.const(Constant.CONST_ZERO, "float32"))
    output_accum_data = tbe.vadds(accum_out, tvm.const(Constant.CONST_ZERO, "float32"))

    if has_improve_precision:
        var_new = tbe.cast_to(var_new, "float16")
        accum_out = tbe.cast_to(accum_out, "float16")
        output_data = tbe.cast_to(output_data, "float16")
        output_accum_data = tbe.cast_to(output_accum_data, "float16")

    # this compute is for muti output
    def _compute(*index):
        return [var_new(*index), accum_out(*index), output_data(*index), output_accum_data(*index)]

    return tvm.compute(var.shape, _compute, name="outputs")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def apply_proximal_adagrad_d(var, accum, lr, l1, l2, grad, var_out,
                             accum_out, use_locking=False,
                             kernel_name="apply_proximal_adagrad_d"):
    """
    Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

    Parameters
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    accum: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    accum_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'accum'.
    use_locking: bool
        default value is "False"
    kernel_name: str
        kernel name, default value is "apply_proximal_adagrad_d"

    Returns:
    None
    """
    _check_shape_is_same(var, accum, grad)

    input_dict = (var, accum, lr, l1, l2, grad)
    args = util_apply_op_schedule.ApplyOpConfig.TensorArgs(input_dict, apply_proximal_adagrad_d_compute,
                                                           [var_out, accum_out], 15)
    name = util_apply_op_schedule.ApplyOpConfig.TensorName(all=('var', 'accum', 'lr', 'l1', 'l2', 'grad'),
                                                           scalar=('lr', 'l1', 'l2'),
                                                           reuse=('var', 'accum'))
    util_apply_op_schedule.common_apply_op_process(util_apply_op_schedule.ApplyOpConfig(args, name), kernel_name)
