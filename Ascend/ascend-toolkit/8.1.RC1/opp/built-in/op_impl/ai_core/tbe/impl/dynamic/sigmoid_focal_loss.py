#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
sigmoid focal loss
"""
import math
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    CONST_FP_MIN = 1.17549435e-38


# 'pylint: disable=unused-argument
@register_operator_compute("SigmoidFocalLoss", op_mode="dynamic", support_fusion=True)
def sigmoid_focal_loss_compute(pred, target, weight, y, gamma, alpha, 
                                reduction, kernel_name, axis, impl_mode):
    """
    algorithm: sigmoid_focal_loss
    calculate the sigmoid focal loss
        prob = 1 / (1 + e^-pred)
        bce_loss = - target * log(prob) - (1 - target) * log(1 - prob)
        pt = (1 - prob) * target + prob * (1 - target)
        fl_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt ** gamma
        loss = bce_loss * fl_weight
        if weight is not None:
            loss = loss * weight
        if reduction = "mean":
            res = mean(loss)
        elif reduction = "sum":
            res = sum(loss)
        elif reduction is None:
            res = loss

    Parameters
    ----------
    pred : TVM Tensor, float16 and float32
        tensor of pred output from previous layer
    target : TVM Tensor, int32
        tensor of target indicate the ground truth
    weight : TVM Tensor, float16 and float32
        tensor of weight that weighted the loss of each class
    y : TVM Tensor, float16 and float32
        tensor of output y
    gamma : float16 or float32
        value of attr gamma, exponential coefficient of focal loss
    alpha : float16 or float32
        value of attr alpha, weighted coefficient of focal loss
    reduction : str
        value of attr reduction, specifies the reduction method of loss, the default value is "mean"
    kernel_name : str
        value of kernel name, default value is "sigmoid_focal_loss"
    impl_mode: str
        value of impl_mode

    Returns
    -------
    res
    """
    # get pred shape and dtype
    ori_dtype = pred.dtype
    pred_dtype = ori_dtype
    if pred_dtype == "float16" and impl_mode == "high_precision":
        pred_dtype = "float32"
        pred = tbe.cast_to(pred, pred_dtype)
    target_pos = tbe.cast_to(target, pred_dtype)
    pred_shape = shape_util.shape_to_list(pred.shape)

    # calculate Sigmoid
    const_one = tvm.const(1.0, pred_dtype)
    const_one_neg = tvm.const(-1.0, pred_dtype)
    pred_neg = tbe.vmuls(pred, const_one_neg)
    pred_neg_exp = tbe.vexp(pred_neg)
    pred_neg_exp_plus_one = tbe.vadds(pred_neg_exp, const_one)
    inp_shape = pred_neg_exp_plus_one.shape
    tensor_one = tbe.broadcast(tvm.const(1, pred_dtype), inp_shape)
    tensor_one.op.attrs["broadcast_flag"] = "brdcast_for_vdiv"
    prob_pos = tbe.vdiv(tensor_one, pred_neg_exp_plus_one)
    prob_neg = tbe.vadds(tbe.vmuls(prob_pos, const_one_neg), const_one)

    prob_pos_clamp = tbe.vmaxs(prob_pos, Constant.CONST_FP_MIN)
    prob_neg_clamp = tbe.vmaxs(prob_neg, Constant.CONST_FP_MIN)

    # calculate binary cross entropy
    target_neg = tbe.vadds(tbe.vmuls(target_pos, const_one_neg), const_one)
    log_prob_pos = tbe.vlog(prob_pos_clamp, impl_mode)
    log_prob_neg = tbe.vlog(prob_neg_clamp, impl_mode)
    first_half_bce_loss = tbe.vmul(target_pos, log_prob_pos)
    second_half_bce_loss = tbe.vmul(target_neg, log_prob_neg)
    bce_loss = tbe.vadd(first_half_bce_loss, second_half_bce_loss)
    bce_loss = tbe.vmuls(bce_loss, const_one_neg)
    
    # calculate focal loss weight
    pt_first_half = tbe.vmul(prob_neg, target_pos)
    pt_second_half = tbe.vmul(prob_pos, target_neg)
    pt = tbe.vadd(pt_first_half, pt_second_half)
    const_alpha = tvm.const(alpha, pred_dtype)
    const_alpha_neg = tvm.const(1-alpha, pred_dtype)
    const_gamma = tvm.const(gamma, pred_dtype)
    alpha_weighted_target_pos = tbe.vmuls(target_pos, const_alpha)
    alpha_weighted_target_neg = tbe.vmuls(target_neg, const_alpha_neg)
    alpha_weighted_target = tbe.vadd(alpha_weighted_target_pos, alpha_weighted_target_neg)
    pt_clamp = tbe.vmaxs(pt, Constant.CONST_FP_MIN)
    pt_log = tbe.vlog(pt_clamp, impl_mode)
    pt_log_mul_gamma = tbe.vmuls(pt_log, const_gamma)
    pt_pow_gamma = tbe.vexp(pt_log_mul_gamma)
    fl_weight = tbe.vmul(alpha_weighted_target, pt_pow_gamma)
    
    # calculate focal loss
    loss = tbe.vmul(bce_loss, fl_weight)

    # calculate weighted focal loss
    if weight is not None:
        if weight.dtype != pred_dtype:
            weight = tbe.cast_to(weight, pred_dtype)
        loss = tbe.vmul(loss, weight)

    # reduce
    if reduction == "none":
        res = loss
    elif reduction == "sum":
        res = tbe.reduce_sum(loss, axis=axis["value"], keepdims=False)
    else:
        reduce_elts = 1.0
        for i in pred_shape:
            reduce_elts *= i
        if isinstance(reduce_elts, float):
            cof = reduce_elts if math.isclose(reduce_elts, 0.0) else reduce_elts ** (-1)
            cof = tvm.const(cof, dtype=pred_dtype)
        else:
            cof = tbe.var("cof", dtype=pred_dtype)
            if pred_dtype == "float16":
                tbe.var("cof_empty", dtype=pred_dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", pred_dtype)
        res = tbe.vmuls(loss, cof)
        res = tbe.reduce_sum(res, axis=axis["value"], keepdims=False)

    if ori_dtype != pred_dtype:
        res = tbe.cast_to(res, ori_dtype)

    return res


@register_operator("SigmoidFocalLoss")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def sigmoid_focal_loss(pred, target, weight, y, gamma=2.0, alpha=0.25,
                       reduction="mean", kernel_name="sigmoid_focal_loss", 
                       impl_mode="high_performance"):
    """
    algorithm: sigmoid_focal_loss
    calculate the sigmoid focal loss

    Parameters
    ----------
    pred : dict
        dict that include dtype and shape of input pred(logits), support float16 and float32
    target : dict
        dict that include dtype and shape of input target(labels), support int32
    weight : dict
        dict that include dtype and shape of optional input weight, support float16 and float32
    y : dict
        dict that include dtype and shape of output y, dtype and shape should be same with pred
    gamma : float16 or float32
        value of attr gamma, exponential coefficient of focal loss, support float16 and float 32
    alpha : float16 or float32
        value of attr alpha, weighted coefficient of focal loss, support float16 and float 32
    reduction : str
        value of attr reduction, specifies the reduction method of loss, the default value is "mean"
    kernel_name : str
        value of kernel name, default value is "sigmoid_focal_loss"
    impl_mode: str
        value of impl_mode, default value is "high_performance" which means the api that support impl_mode will
        run on high_performance. Another mode is "high_precision"

    Returns
    -------
    None
    """
    pred_dtype = pred.get("dtype").lower()
    pred_shape = pred.get("shape")
    pred["rel_pos_to_reduce"] = "before"
    check_list = ("float16", "float32")
    para_check.check_dtype(pred_dtype, check_list, param_name="pred")

    target_dtype = target.get("dtype").lower()
    target_shape = target.get("shape")
    target["rel_pos_to_reduce"] = "before"
    target_check_list = ("int32")
    para_check.check_dtype(target_dtype, target_check_list, param_name="target")

    if not is_unknown_rank_input(pred) and len(pred_shape) != 2:
        error_manager_vector.raise_err_input_shape_invalid(kernel_name,
                                                           "pred", "it should be (batch_size, num_classes).")
    if not is_unknown_rank_input(target) and len(target_shape) != 2:
        error_manager_vector.raise_err_input_shape_invalid(kernel_name,
                                                           "target", "it should be (batch_size, num_classes).")
    if not is_unknown_rank_input(target) and not is_unknown_rank_input(pred):
        if pred_shape[0] != target_shape[0] or pred_shape[1] != target_shape[1]:
            error_manager_vector.raise_err_inputs_shape_not_equal(kernel_name, "target", 
                                                                target_shape, "pred", 
                                                                pred_shape, pred_shape)

    if reduction not in ("mean", "sum", "none"):
        rule_desc = "Reduction should be mean, sum or none."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "reduction", reduction)
        
    tbe_context.get_context().add_compile_info("reduction", reduction)

    extra_params = None
    if reduction == "none":
        mode = OpPatternMode.ELEWISE
    else:
        mode = OpPatternMode.REDUCE
        if is_unknown_rank_input(pred):
            input_axis = {"shape": [2, ], "value": [], "rel_pos_to_reduce": "axis"}
        else:
            axis = []
            for i, _ in enumerate(pred_shape):
                axis.append(i)
            input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
        extra_params = {"keepdims": False, "reduce_axes_type": "all"}

    schedules, tensors = [], []
    if weight is not None:
        weight_dtype = weight.get("dtype").lower()
        para_check.check_dtype(weight_dtype, check_list, param_name="weight")
        weight_shape = weight.get("shape")
        weight["rel_pos_to_reduce"] = "before"
        if not is_unknown_rank_input(weight) and len(weight_shape) != 2:
            error_manager_vector.raise_err_input_shape_invalid(kernel_name,
                                                           "weight", "it should be (batch_size, num_classes).")
        if not is_unknown_rank_input(target) and not is_unknown_rank_input(pred) and not is_unknown_rank_input(weight):
            if pred_shape[0] != weight_shape[0] or pred_shape[1] != weight_shape[1]:
                error_manager_vector.raise_err_inputs_shape_not_equal(kernel_name, "weight", 
                                                                    weight_shape, "pred", pred_shape, 
                                                                    pred_shape)

        if extra_params is not None:
            ins = classify([pred, target, weight, input_axis], mode, extra_params)
            for (p, t, w, axis) in ins:
                with tbe.compute():
                    shape_pred, shape_target, shape_weight = shape_util.variable_shape([p, t, w, axis], 
                                                                                        op_mode="reduce")[0:3]
                    pred_data = tvm.placeholder(shape_pred, dtype=pred_dtype, name="pred_data")
                    target_data = tvm.placeholder(shape_target, dtype=target_dtype, name="target_data")
                    weight_data = tvm.placeholder(shape_weight, dtype=weight_dtype, name="weight_data")
                    res = sigmoid_focal_loss_compute(pred_data, target_data, weight_data, y, 
                                                    gamma, alpha, reduction, kernel_name, axis, impl_mode)
                    tensors.append((pred_data, target_data, weight_data, res))
                with tvm.target.cce():
                    sch = tbe.auto_schedule(res)
                schedules.append(sch)
        else:
            ins = classify([pred, target, weight], mode)
            for (p, t, w) in ins:
                with tbe.compute():
                    shape_pred, shape_target, shape_weight = shape_util.variable_shape([p, t, w])
                    pred_data = tvm.placeholder(shape_pred, dtype=pred_dtype, name="pred_data")
                    target_data = tvm.placeholder(shape_target, dtype=target_dtype, name="target_data")
                    weight_data = tvm.placeholder(shape_weight, dtype=weight_dtype, name="weight_data")
                    res = sigmoid_focal_loss_compute(pred_data, target_data, weight_data, y, 
                                                    gamma, alpha, reduction, kernel_name, [], impl_mode)
                    tensors.append((pred_data, target_data, weight_data, res))
                with tvm.target.cce():
                    sch = tbe.auto_schedule(res)
                schedules.append(sch)
    else:
        if extra_params is not None:
            ins = classify([pred, target, input_axis], mode, extra_params)
            for (p, t, axis) in ins:
                with tbe.compute():
                    shape_pred, shape_target = shape_util.variable_shape([p, t, axis], op_mode="reduce")[0:2]
                    pred_data = tvm.placeholder(shape_pred, dtype=pred_dtype, name="pred_data")
                    target_data = tvm.placeholder(shape_target, dtype=target_dtype, name="target_data")
                    res = sigmoid_focal_loss_compute(pred_data, target_data, weight, y, 
                                                    gamma, alpha, reduction, kernel_name, axis, impl_mode)
                    tensors.append((pred_data, target_data, res))
                with tvm.target.cce():
                    sch = tbe.auto_schedule(res)
                schedules.append(sch)
        else:
            ins = classify([pred, target], mode)
            for (p, t) in ins:
                with tbe.compute():
                    shape_pred, shape_target = shape_util.variable_shape([p, t])
                    pred_data = tvm.placeholder(shape_pred, dtype=pred_dtype, name="pred_data")
                    target_data = tvm.placeholder(shape_target, dtype=target_dtype, name="target_data")
                    res = sigmoid_focal_loss_compute(pred_data, target_data, weight, y, 
                                                    gamma, alpha, reduction, kernel_name, [], impl_mode)
                    tensors.append((pred_data, target_data, res))
                with tvm.target.cce():
                    sch = tbe.auto_schedule(res)
                schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
