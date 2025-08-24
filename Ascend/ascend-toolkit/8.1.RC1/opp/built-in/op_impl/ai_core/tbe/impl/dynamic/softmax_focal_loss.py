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
softmax focal loss
"""
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_common import check_op_impl_mode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument
@register_operator_compute("SoftmaxFocalLoss", op_mode="dynamic", support_fusion=True)
def softmax_focal_loss_compute(pred, target, weight, y, gamma, alpha, 
                                reduction, kernel_name, axis=-1, impl_mode="high_performance"):
    """
    algorithm: softmax_focal_loss
    calculate the softmax focal loss
        ce_loss = - sum(target * log(pred), axis=-1)
        fl_weight = sum((1 - pred)**gamma * y * alpha), axis=-1)
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
        tensor of output loss
    gamma : float16 or float32
        value of attr gamma, exponential coefficient of focal loss
    alpha : float16 or float32
        value of attr alpha, weighted coefficient of focal loss
    reduction : str
        value of attr reduction, specifies the reduction method of loss, the default value is "mean"
    kernel_name : str
        value of kernel name, default value is "softmax_focal_loss"
    impl_mode: str
        value of impl_mode, default value is "high_performance" which means the api that support impl_mode will
        run on high_performance. Another mode is "high_precision"

    Returns
    -------
    res
    """
    # get pred shape and dtype, cast f16 to f32 for higher precision
    ori_dtype = pred.dtype
    pred_dtype = ori_dtype
    if pred_dtype == "float16" and impl_mode == "high_precision":
        pred_dtype = "float32"
        pred = tbe.cast_to(pred, pred_dtype)
    target = tbe.cast_to(target, pred_dtype)
    if weight is not None:
        weight = tbe.cast_to(weight, pred_dtype)

    # calculate cross entropy loss
    log_prob = tbe.vlog(pred, impl_mode)
    log_prob = tbe.vmuls(log_prob, tvm.const(-1.0, pred_dtype))
    ce_loss = tbe.vmul(log_prob, target)

    # calculate weighted CE loss
    if weight is not None:
        ce_loss = tbe.vmul(ce_loss, weight)

    ce_loss_sum = tbe.reduce_sum(ce_loss, axis=axis, keepdims=True)
    ce_loss_sum = tbe.broadcast(ce_loss_sum, pred.shape)

    # calculate focal weight
    prob_neg = tbe.vadds(tbe.vmuls(pred, tvm.const(-1.0, pred_dtype)), tvm.const(1.0, pred_dtype))
    log_prob_neg = tbe.vlog(prob_neg, impl_mode)
    log_prob_neg_muls_gamma = tbe.vmuls(log_prob_neg, tvm.const(gamma, pred_dtype))
    exp_log_prob_neg_muls_gamma = tbe.vexp(log_prob_neg_muls_gamma)
    exp_log_prob_neg_muls_gamma_mul_target = tbe.vmul(exp_log_prob_neg_muls_gamma, target)
    focal_weight = tbe.vmuls(exp_log_prob_neg_muls_gamma_mul_target, tvm.const(alpha, pred_dtype))
    focal_weight_sum = tbe.reduce_sum(focal_weight, axis=axis, keepdims=True)
    focal_weight_sum = tbe.broadcast(focal_weight_sum, pred.shape)
    
    # calculate focal loss
    res = tbe.vmul(ce_loss_sum, focal_weight_sum)

    if ori_dtype != pred_dtype:
        res = tbe.cast_to(res, ori_dtype)
    
    return res


def _check_parameter(op_input, check_list, param_name, pred_shape, kernel_name):
    """
    check inputs' shape and dtype, modify unknown rank shape

    Parameters
    ----------
    op_input : dict
        dict that include dtype and shape of op_input
    check_list : list
        list that include support dtype of this op_input
    param_name : str
        name of this op_input
    pred_shape : list
        shape of pred, used to be compared with inputs
    kernel_name : str
        value of kernel name

    Returns
    -------
    op_input : dict
        op_input that has been changed
    input_dtype : str
        dtype of op_input
    """
    if op_input is None:
        return None, None
    input_dtype = op_input.get("dtype").lower()
    input_shape = op_input.get("shape")
    para_check.check_dtype(input_dtype, check_list, param_name=param_name)
    if is_unknown_rank_input(op_input):
        input_shape = [-1, -1]
        input_range = [(1, None), (1, None)]
        op_input["shape"] = input_shape
        op_input["range"] = input_range
    if len(input_shape) != 2:
        error_manager_vector.raise_err_input_shape_invalid(kernel_name,
                                                    "pred", "pred's shape should be (batch_size, num_classes).")
    if input_shape[0] != pred_shape[0] or input_shape[1] != pred_shape[1]:
        error_manager_vector.raise_err_inputs_shape_not_equal(kernel_name, param_name, input_shape, 
                                                                "pred", pred_shape, pred_shape)
    return op_input, input_dtype


@register_operator("SoftmaxFocalLoss")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME,
                            para_check.OPTION_ATTR_STR)
def softmax_focal_loss(pred, target, weight, y, gamma=2.0, alpha=0.25,
                       reduction="mean", kernel_name="softmax_focal_loss", 
                       impl_mode="high_performance"):
    """
    algorithm: softmax_focal_loss
    calculate the softmax focal loss

    Parameters
    ----------
    pred : dict
        dict that include dtype and shape of input pred(logits), support float16 and float32
    target : dict
        dict that include dtype and shape of input target(onehot labels), support int32
    weight : dict
        dict that include dtype and shape of optional input weight, support float16 and float32
    y : dict
        dict that include dtype and shape of output loss, dtype and shape should be same with pred
    gamma : float16 or float32
        value of attr gamma, exponential coefficient of focal loss, support float16 and float 32
    alpha : float16 or float32
        value of attr alpha, weighted coefficient of focal loss, support float16 and float 32
    reduction : str
        value of attr reduction, specifies the reduction method of loss, the default value is "mean"
    kernel_name : str
        value of kernel name, default value is "softmax_focal_loss"
    impl_mode: str
        value of impl_mode, default value is "high_performance" which means the api that support impl_mode will
        run on high_performance. Another mode is "high_precision"

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    pred_shape = pred.get("shape")
    if is_unknown_rank_input(pred):
        pred_shape = [-1, -1]
    pred, pred_dtype = _check_parameter(pred, ("float16", "float32"), "pred", pred_shape, kernel_name)
    target, target_dtype = _check_parameter(target, ("int32"), "target", pred_shape, kernel_name)
    weight, weight_dtype = _check_parameter(weight, ("float16", "float32"), "weight", pred_shape, kernel_name)

    if reduction not in ("none"):
        rule_desc = "Reduction only support none currently."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "reduction", reduction)

    input_axis = [-1, ]
    extra_params = {}
    
    schedules, tensors = [], []
    if weight is not None:
        ins = classify([pred, target, weight, input_axis], OpPatternMode.NORM, extra_params)
        for (p, t, w, axis) in ins:
            with tbe.compute():
                shape_pred, shape_target, shape_weight = shape_util.variable_shape([p, t, w], op_mode="norm")
                data_pred = tvm.placeholder(shape_pred, name="pred", dtype=pred_dtype)
                data_target = tvm.placeholder(shape_target, name="target", dtype=target_dtype)
                data_weight = tvm.placeholder(shape_weight, name="weight", dtype=weight_dtype)
                res = softmax_focal_loss_compute(data_pred, data_target, data_weight, y, gamma, alpha, 
                                                reduction, kernel_name, axis, impl_mode)
                tensors.append([data_pred, data_target, data_weight, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    else:
        ins = classify([pred, target, input_axis], OpPatternMode.NORM, extra_params)
        for (p, t, axis) in ins:
            with tbe.compute():
                shape_pred, shape_target = shape_util.variable_shape([p, t], op_mode="norm")
                data_pred = tvm.placeholder(shape_pred, name="pred", dtype=pred_dtype)
                data_target = tvm.placeholder(shape_target, name="target", dtype=target_dtype)
                res = softmax_focal_loss_compute(data_pred, data_target, None, y, gamma, alpha, 
                                                reduction, kernel_name, axis, impl_mode)
                tensors.append([data_pred, data_target, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
