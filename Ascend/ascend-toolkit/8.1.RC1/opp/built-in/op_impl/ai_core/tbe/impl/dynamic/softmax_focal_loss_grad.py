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
softmax_focal_loss_grad
"""
from ast import Param
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=too-many-locals,invalid-name,unused-argument,too-many-arguments
@register_operator_compute("SoftmaxFocalLossGrad", op_mode="dynamic", support_fusion=False)
def softmax_focal_loss_grad_compute(pred,
                                    target,
                                    dout,
                                    axis,
                                    gamma,
                                    alpha,
                                    weight,
                                    reduction,
                                    kernel_name):
    """
    computing logic

    Parameters
    ----------
    pred : TVM tensor
       the placeholder of pred
    target : TVM tensor
       the placeholder of target
    dout : TVM tensor
       the placeholder of dout
    gamma : float
        default value is 2.0
    alpha : float
        default value is 0.25
    weight : TVM tensor
        the placeholder of sample weight, or None
    reduction: str
       type of result, default value is "mean"
    kernel_name : str
       kernel name, default value is "softmax_focal_loss_grad"

    As the softmax calculation has been done in mmcv preprocessing, here pred is softmax probs.
    softmax calculation is like such codes below. 
    
    # pred_exp = tbe.vexp(pred)
    # exp_sum = tbe.reduce_sum(pred_exp, axis=axis, keepdims=True)
    # sum_broadcast = tbe.broadcast(exp_sum, pred_shape)
    # probs = tbe.vdiv(pred_exp, sum_broadcast)

    Returns
    -------
    output tensor
    """

    ori_dtype = pred.dtype
    support = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
    if support:
        cast_dtype = "float32"
    else:
        cast_dtype = "float16"

    pred = tbe.cast_to(pred, cast_dtype)
    target = tbe.cast_to(target, cast_dtype)
    dout = tbe.cast_to(dout, cast_dtype)

    pred_shape = shape_util.shape_to_list(pred.shape)
    probs = pred
   
    # define constant
    const_n1 = tvm.const(-1.0, cast_dtype)
    const_p1 = tvm.const(1.0, cast_dtype)

    probs_opp = tbe.vmuls(probs, const_n1)
    probs_nadd = tbe.vadds(probs_opp, const_p1)

    # calculate "log(p), log(1-p)"
    probs_log = tbe.vlog(probs)
    probs_nlog = tbe.vlog(probs_nadd)

    # calculate "(1-p)^r = exp(r * ln(1-p))" 
    pow_x = tbe.vmuls(probs_nlog, tvm.const(gamma, cast_dtype))
    pow_y = tbe.vexp(pow_x)
    
    wf = tbe.vmul(pow_y, target)
    wf = tbe.vmuls(wf, tvm.const(alpha, cast_dtype))

    wf_reduce = tbe.reduce_sum(wf, axis=axis, keepdims=True)
    wf_broadcast = tbe.broadcast(wf_reduce, pred.shape)

    pow_u = tbe.vmuls(probs_nlog, tvm.const(gamma - 1.0, cast_dtype))
    pow_v = tbe.vexp(pow_u)
    
    wb = tbe.vmul(pow_v, target)
    wb = tbe.vmuls(wb, tvm.const(alpha, cast_dtype))

    wb_reduce = tbe.reduce_sum(wb, axis=axis, keepdims=True)
    wb_broadcast = tbe.broadcast(wb_reduce, pred.shape)
    
    ce = tbe.vmul(probs_log, target)
    if weight is not None:
        weight = tbe.cast_to(weight, cast_dtype)
        ce = tbe.vmul(ce, weight)
    ce = tbe.vmuls(ce, const_n1)
    ce_reduce = tbe.reduce_sum(ce, axis=axis, keepdims=True)
    ce_broadcast = tbe.broadcast(ce_reduce, pred.shape)

    weight_u = tbe.vmul(weight, target)
    weight_reduce = tbe.reduce_sum(weight_u, axis=axis, keepdims=True)
    weight_broadcast = tbe.broadcast(weight_reduce, pred.shape)
    
    d_ce_u = tbe.vmul(probs, weight_broadcast)
    d_ce_v = tbe.vmul(target, weight)

    d_ce = tbe.vsub(d_ce_u, d_ce_v)
    d_wf = tbe.vsub(wf_broadcast, wb_broadcast)
    d_wf = tbe.vadd(d_wf, wb)
    d_wf = tbe.vmul(d_wf, probs)
    d_wf = tbe.vmuls(d_wf, tvm.const(-gamma, cast_dtype))

    grad_font = tbe.vmul(d_ce, wf_broadcast)
    grad_back = tbe.vmul(d_wf, ce_broadcast)

    res = tbe.vadd(grad_font, grad_back)
 
    res = tbe.vmul(res, dout)

    # if choose "mean", res should be divided by batch_size
    if reduction == "mean":
        element = 1.0
        for i in pred_shape:
            element *= i
        if isinstance(element, float):
            coef = element ** (-1)
            coef = tvm.const(coef, dtype=cast_dtype)
        else:
            coef = tbe.var("coef", dtype=cast_dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_coef_dtype", cast_dtype)
        res = tbe.vmuls(res, coef)

    # calculate finish and return
    if ori_dtype != res.dtype:
        res = tbe.cast_to(res, ori_dtype)

    return res


# 'pylint: disable=too-many-arguments,too-many-locals
@register_operator("SoftmaxFocalLossGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def softmax_focal_loss_grad(pred,
                            target,
                            dout,
                            weight,
                            grad,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            kernel_name="softmax_focal_loss_grad"):
    """
    calculating the grad of softmax focal loss

    Parameters
    ----------
    pred : dict
        the shape and dtype of the predicted tensor
    target : dict
        the shape and dtype of the ground truth label
    dout : dict
        the shape and dtype of the gradient input
    grad : dict
        the shape and dtype of the gradient output
    gamma : float
        param gamma, default = 2.0
    alpha : float
        param alpha, default = 0.25
    weight: dict
        the shape and dtype of the sample weight
    reduction : str
        param reduction, default value is "mean"
    kernel_name : str
        kernel name, default value is "softmax_focal_loss_grad"

    Returns
    -------
    None
    """
    # input range resticiton for templates
    pred["range"] = ((1, None), (1, None))
    target["range"] = ((1, None), (1, None))
    dout["range"] = ((1, None), (1, None))
    weight["range"] = ((1, None), (1, None))
    grad["range"] = ((1, None), (1, None))
    # check input: predict label dout
    float_list = ("float16", "float32")

    pred_dtype = pred.get("dtype").lower()
    para_check.check_dtype(pred_dtype, float_list, param_name="pred")

    dout_dtype = dout.get("dtype").lower()
    para_check.check_dtype(dout_dtype, float_list, param_name='dout')

    grad_dtype = grad.get("dtype").lower()
    para_check.check_dtype(grad_dtype, float_list, param_name="grad")

    int_list = ("int32")
    
    target_dtype = target.get("dtype").lower()
    para_check.check_dtype(target_dtype, int_list, param_name="target")

    # check reduction
    check_list_reduction = ("none", "mean", "sum")
    reduction_type = reduction.lower()
    para_check.check_dtype(reduction_type, check_list_reduction, param_name="reduction")

    extra_params = {}
    list_axis = [-1, ]

    if weight is None:
        
        schedules, tensors = [], []
        ins = classify([pred, target, dout, list_axis], OpPatternMode.NORM, extra_params)
   
        for (_pred, _target, _dout, reduce_axis) in ins:
            with tbe.compute():
                pred_shape, target_shape, dout_shape = shape_util.variable_shape(
                    [_pred, _target, _dout], op_mode='norm')
                tensor_pred = tvm.placeholder(pred_shape, name="pred", dtype=pred_dtype)
                tensor_target = tvm.placeholder(target_shape, name="target", dtype=target_dtype)
                tensor_dout = tvm.placeholder(dout_shape, name='dout', dtype=dout_dtype)

                res = softmax_focal_loss_grad_compute(tensor_pred, tensor_target, tensor_dout, reduce_axis,
                                                      gamma, alpha, None, reduction_type, kernel_name)

                tensors.append([tensor_pred, tensor_target, tensor_dout, res])
            
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    
    else:

        schedules, tensors = [], []
        ins = classify([pred, target, dout, weight, list_axis], OpPatternMode.NORM, extra_params)
        weight_dtype = weight.get('dtype')

        for (_pred, _target, _dout, _weight, reduce_axis) in ins:
            with tbe.compute():
                pred_shape, target_shape, dout_shape, weight_shape = \
                    shape_util.variable_shape([_pred, _target, _dout, _weight], op_mode='norm')
                tensor_pred = tvm.placeholder(pred_shape, name="pred", dtype=pred_dtype)
                tensor_target = tvm.placeholder(target_shape, name="target", dtype=target_dtype)
                tensor_dout = tvm.placeholder(dout_shape, name='dout', dtype=dout_dtype)
                tensor_weight = tvm.placeholder(weight_shape, name='weight', dtype=weight_dtype)

                res = softmax_focal_loss_grad_compute(tensor_pred, tensor_target, tensor_dout, reduce_axis,
                                                      gamma, alpha, tensor_weight, reduction_type, kernel_name)

                tensors.append([tensor_pred, tensor_target, tensor_dout, tensor_weight, res])
            
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    # build
    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)