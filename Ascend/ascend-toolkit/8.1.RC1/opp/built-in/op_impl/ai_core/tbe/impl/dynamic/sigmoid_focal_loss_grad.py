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
sigmoid_focal_loss_grad
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


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    CONST_FP_MIN = 1.17549435e-38


# 'pylint: disable=too-many-locals,invalid-name,unused-argument,too-many-arguments
@register_operator_compute("SigmoidFocalLossGrad", op_mode="dynamic", support_fusion=False)
def sigmoid_focal_loss_grad_compute(pred,
                                    target,
                                    dout,
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
       kernel name, default value is "sigmoid_focal_loss_grad"

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

    # calculate the sigmoid probs
    const_n1 = tvm.const(-1.0, cast_dtype)
    const_p1 = tvm.const(1.0, cast_dtype)
    
    pred_n = tbe.vmuls(pred, const_n1)
    pred_nexp = tbe.vexp(pred_n)
    prob_deno = tbe.vadds(pred_nexp, const_p1)
    inp_shape = prob_deno.shape
    tensor_one = tbe.broadcast(tvm.const(1, cast_dtype), inp_shape)
    tensor_one.op.attrs["broadcast_flag"] = "brdcast_for_vdiv"
    probs = tbe.vdiv(tensor_one, prob_deno)

    # calculate "-p, 1-p"
    probs_opp = tbe.vmuls(probs, const_n1)
    probs_nadd = tbe.vadds(probs_opp, const_p1)
    
    # calculate "log(p), log(1 - p)" 
    probs_clamp = tbe.vmaxs(probs, Constant.CONST_FP_MIN)
    probs_nadd_clamp = tbe.vmaxs(probs_nadd, Constant.CONST_FP_MIN)
    probs_log = tbe.vlog(probs_clamp)
    probs_nlog = tbe.vlog(probs_nadd_clamp)
  
    # calculate "(1-p)^(r+1) = exp((r+1) * ln(1-p))"
    pow_x = tbe.vmuls(probs_nlog, tvm.const(gamma + 1.0, cast_dtype))
    pow_y = tbe.vexp(pow_x)

    dpos_front = tbe.vmuls(pow_y, tvm.const(-alpha, cast_dtype))
    # calculate "(1-p)^r = exp(r * ln(1-p))"
    pow_u = tbe.vmuls(probs_nlog, tvm.const(gamma, cast_dtype))
    pow_v = tbe.vexp(pow_u)

    dpos_back = tbe.vmul(pow_v, probs_log)
    dpos_back = tbe.vmul(dpos_back, probs)
    dpos_back = tbe.vmuls(dpos_back, tvm.const(gamma * alpha, cast_dtype))
    dpos = tbe.vadd(dpos_front, dpos_back)

    # calculate "p^r = exp(r * lnp)"
    pow_m = tbe.vmuls(probs_log, tvm.const(gamma, cast_dtype))
    pow_n = tbe.vexp(pow_m)

    dneg_front = tbe.vmul(pow_n, probs_nadd)
    dneg_front = tbe.vmul(dneg_front, probs_nlog)
    dneg_front = tbe.vmuls(dneg_front, tvm.const(gamma * (alpha-1.0), cast_dtype))

    pow_p = tbe.vmuls(probs_log, tvm.const(gamma + 1.0, cast_dtype))
    pow_q = tbe.vexp(pow_p)

    dneg_back = tbe.vmuls(pow_q, tvm.const(1.0-alpha, cast_dtype))
    dneg = tbe.vadd(dneg_front, dneg_back)
    
    target_opp = tbe.vmuls(target, const_n1)
    target_nadd = tbe.vadds(target_opp, const_p1)

    grad_front = tbe.vmul(dpos, target_nadd)
    grad_back = tbe.vmul(dneg, target) 

    res = tbe.vadd(grad_front, grad_back)
    if weight is not None:
        weight = tbe.cast_to(weight, cast_dtype)
        res = tbe.vmul(res, weight)
    res = tbe.vmul(res, dout)

    # if choose "mean", res should divide over n
    if reduction == "mean":
        reduce_elts = 1.0
        for i in pred_shape:
            reduce_elts *= i
        if isinstance(reduce_elts, float):
            coef = reduce_elts ** (-1)
            coef = tvm.const(coef, dtype=cast_dtype)
        else:
            coef = tbe.var("coef", dtype=cast_dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_coef", cast_dtype)
        res = tbe.vmuls(res, coef)

    # calculate finish and return
    if ori_dtype != res.dtype:
        res = tbe.cast_to(res, ori_dtype)

    return res


# 'pylint: disable=too-many-arguments,too-many-locals
@register_operator("SigmoidFocalLossGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def sigmoid_focal_loss_grad(pred,
                            target,
                            dout,
                            weight,
                            grad,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            kernel_name="sigmoid_focal_loss_grad"):
    """
    calculating the grad of sigmoid focal loss

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
        kernel name, default value is "sigmoid_focal_loss_grad"

    Returns
    -------
    None
    """
    # check input: predict label dout
    float_list = ("float16", "float32")

    pred_dtype = pred.get("dtype").lower()
    para_check.check_dtype(pred_dtype, float_list, param_name="pred")

    dout_dtype = dout.get("dtype").lower()
    para_check.check_dtype(dout_dtype, float_list, param_name='dout')

    grad_dtype = grad.get("dtype").lower()
    para_check.check_dtype(grad_dtype, float_list, param_name="grad")

    int_list = ("int32", "int64")
    
    target_dtype = target.get("dtype").lower()
    para_check.check_dtype(target_dtype, int_list, param_name="target")

    # check reduction
    check_list_reduction = ("none", "mean", "sum")
    reduction_type = reduction.lower()
    para_check.check_dtype(reduction_type, check_list_reduction, param_name="reduction")

    if weight is None:
        
        schedules, tensors = [], []
        ins = classify([pred, target, dout], OpPatternMode.ELEWISE)
   
        for (_pred, _target, _dout) in ins:
            with tbe.compute():
                pred_shape, _, _ = shape_util.variable_shape([_pred, _target, _dout])
                tensor_pred = tvm.placeholder(pred_shape, name="pred", dtype=pred_dtype)
                tensor_target = tvm.placeholder(pred_shape, name="target", dtype=target_dtype)
                tensor_dout = tvm.placeholder(pred_shape, name='dout', dtype=dout_dtype)

                res = sigmoid_focal_loss_grad_compute(tensor_pred, tensor_target, tensor_dout, 
                                                      gamma, alpha, None, reduction_type, kernel_name)

                tensors.append([tensor_pred, tensor_target, tensor_dout, res])
            
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    
    else:
        
        schedules, tensors = [], []
        ins = classify([pred, target, dout, weight], OpPatternMode.ELEWISE)
        weight_dtype = weight.get('dtype')

        for (_pred, _target, _dout, _weight) in ins:
            with tbe.compute():
                pred_shape, _, _, _ = shape_util.variable_shape([_pred, _target, _dout, _weight])
                tensor_pred = tvm.placeholder(pred_shape, name="pred", dtype=pred_dtype)
                tensor_target = tvm.placeholder(pred_shape, name="target", dtype=target_dtype)
                tensor_dout = tvm.placeholder(pred_shape, name='dout', dtype=dout_dtype)
                tensor_weight = tvm.placeholder(pred_shape, name='weight', dtype=weight_dtype)

                res = sigmoid_focal_loss_grad_compute(tensor_pred, tensor_target, tensor_dout, 
                                                      gamma, alpha, tensor_weight, reduction_type, kernel_name)

                tensors.append([tensor_pred, tensor_target, tensor_dout, tensor_weight, res])
            
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    # build
    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)