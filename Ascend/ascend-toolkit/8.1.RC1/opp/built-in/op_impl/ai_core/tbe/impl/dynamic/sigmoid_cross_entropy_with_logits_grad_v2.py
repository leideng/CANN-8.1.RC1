#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
sigmoid_cross_entropy_with_logits_grad_v2
"""

# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
import math
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tbe_context
from impl.util.util_compute import only_static_support


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    CONST_FP32_MAX = 3.4e+38


def get_cof_by_shape(predict_shape, precision_dtype):
    """
    get cof by shape
    """
    reduce_elts = 1.0
    for i in predict_shape:
        if isinstance(i, tvm.tir.IntImm):
            reduce_elts *= i.value
        else:
            reduce_elts *= i

    if isinstance(reduce_elts, float):
        cof = reduce_elts if math.isclose(reduce_elts, 0.0) else reduce_elts ** (-1)
        cof = tvm.const(cof, dtype=precision_dtype)

    else:
        cof = tbe.var("cof", dtype=precision_dtype)
        tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", precision_dtype)

    return cof


def get_weight_shape(weight, predict_shape):
    """
    get weight shape
    """
    if weight is not None:
        weight_shape = shape_util.shape_to_list(weight.shape)
    else:
        weight_shape = predict_shape
    return weight_shape


# 'pylint: disable=too-many-arguments
@register_operator_compute("SigmoidCrossEntropyWithLogitsGradV2",
                           op_mode="dynamic",
                           support_fusion=only_static_support)
def sigmoid_cross_entropy_with_logits_grad_v2_compute(predict, target, dout, weight, pos_weight, reduction="mean"):
    """
    sigmoid_cross_entropy_with_logits_grad_v2 compute function

    Parameters
    ----------
    predict : tvm.tensor
        tensor of predict
    target : tvm.tensor
        tensor of target
    dout : tvm.tensor
        tensor of dout
    weight : tvm.tensor
        tensor of weight
    pos_weight : tvm.tensor
        tensor of pos_weight
    reduction : str
        specifies the reduction mode :'none' | 'mean' | 'sum'
    Returns
    -------
    res : tvm.tensor
        tensor of result
    """
    predict_shape = shape_util.shape_to_list(predict.shape)

    weight_shape = get_weight_shape(weight, predict_shape)
    pos_weight_shape = get_weight_shape(pos_weight, predict_shape)

    predict_shape, target_shape, dout_shape, weight_shape, pos_weight_shape, max_shape = \
            shape_util.unify_broadcast_shapes([predict.shape, target.shape, dout.shape, weight_shape, pos_weight_shape])

    predict = tbe.broadcast(predict, max_shape)
    target = tbe.broadcast(target, max_shape)
    dout = tbe.broadcast(dout, max_shape)

    predict_origin_dtype = predict.dtype.lower()
    precision_dtype = "float32"

    if predict_origin_dtype in ("float16", "bfloat16"):
        predict = tbe.cast_to(predict, precision_dtype)
        target = tbe.cast_to(target, precision_dtype)
        dout = tbe.cast_to(dout, precision_dtype)

    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") == "Ascend910":
        ln_res = -math.log(Constant.CONST_FP32_MAX)
        ln_res = int(ln_res * 10) / 10
        predict = tbe.vmaxs(predict, ln_res)

    # calculate sigmoid(predict)
    const_num_neg_one = tvm.const(-1, dtype="float32")
    const_num_one = tvm.const(1, dtype="float32")
    tmp_negative = tbe.vmuls(predict, const_num_neg_one)
    tmp_exp = tbe.vexp(tmp_negative)
    tmp_sum = tbe.vadds(tmp_exp, const_num_one)
    tensor_one = tbe.broadcast(tvm.const(1, "float32"), predict_shape)
    sigmoid_res = tbe.vdiv(tensor_one, tmp_sum)

    # calculate the result of gradient = ((log_weight + 1 - target) * sigmoid(predict) - log_weight) * dout
    if pos_weight is not None:
        if pos_weight.dtype.lower() in ("float16", "bfloat16"):
            pos_weight = tbe.cast_to(pos_weight, precision_dtype)

        pos_weight = tbe.broadcast(pos_weight, max_shape)

        log_weight = tbe.vmul(pos_weight, target)
        weight_tmp = tbe.vadds(log_weight, tvm.const(1, precision_dtype))
        weight_sub = tbe.vsub(weight_tmp, target)
        grad_tmp = tbe.vmul(weight_sub, sigmoid_res)
        grad_cur = tbe.vsub(grad_tmp, log_weight)
        grad_output = tbe.vmul(grad_cur, dout)
    else:
        grad_cur = tbe.vsub(sigmoid_res, target)
        grad_output = tbe.vmul(grad_cur, dout)

    # calculate the result of gradient = gradient * weight
    if weight is not None:
        if weight.dtype.lower() in ("float16", "bfloat16"):
            weight = tbe.cast_to(weight, precision_dtype)

        weight = tbe.broadcast(weight, max_shape)
        grad_output = tbe.vmul(grad_output, weight)

    # calculate the result of gradient = gradient / num
    if reduction == "mean":
        cof = get_cof_by_shape(max_shape, precision_dtype)
        grad_output = tbe.vmuls(grad_output, cof)

    if predict_origin_dtype == "float16":
        grad_output = tbe.cast_to(grad_output, predict_origin_dtype)
    elif predict_origin_dtype == "bfloat16":
        grad_output = tbe.round(grad_output, predict_origin_dtype)

    return grad_output


# 'pylint: disable=too-many-statements,too-many-arguments
@register_operator("SigmoidCrossEntropyWithLogitsGradV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def sigmoid_cross_entropy_with_logits_grad_v2(predict,
                                              target,
                                              dout,
                                              weight,
                                              pos_weight,
                                              gradient,
                                              reduction="mean",
                                              kernel_name="sigmoid_cross_entropy_with_logits_grad_v2"):
    """
    Update var by subtracting value from it.

    Parameters:
    ----------
    predict : dict
        shape and dtype of input, required
        dtype support float16, float32
    target : dict
        dict,shape and dtype of target, should be same shape and type as predict, required
        dtype support float16, float32.
    dout : dict
        dict,shape and dtype of dout, should be same shape and type as predict, required
        dtype support float16, float32.
    weight : dict
        dict,shape and dtype of weight, should be same shape and type as predict, optional
        dtype support float16, float32.
    pos_weight : dict
        dict,shape and dtype of pos_weight, should be same shape and type as predict, optional
        dtype support float16, float32.
    gradient : dict
        dict of out,shape and dtype of target, should be same shape and type as predict, required
        dtype support float16, float32.
    reduction : str
        str, specifies the reduction mode: 'none' | 'mean' | 'sum', default to 'mean'
    kernel_name : str
        cce kernel name, default to 'sigmoid_cross_entropy_with_logits_grad_v2'

    Returns
    -------
    None
    """
    predict_dtype = predict.get("dtype").lower()

    dtype_list = ["float16", "float32", "bfloat16"]
    para_check.check_dtype(predict_dtype, dtype_list)

    shape_util.compare_tensor_dict_key(predict, target, "dtype")
    shape_util.compare_tensor_dict_key(predict, dout, "dtype")

    reduction_list = ["none", "mean", "sum"]
    if reduction not in reduction_list:
        raise RuntimeError("reduction should be one of ['none','mean','sum']")

    weight_data = None
    pos_weight_data = None

    if (weight is not None) and (pos_weight is not None):

        shape_util.compare_tensor_dict_key(weight, target, "dtype")
        shape_util.compare_tensor_dict_key(pos_weight, dout, "dtype")

        schedules, tensors = [], []
        ins = classify([predict, target, dout, weight, pos_weight], OpPatternMode.ELEWISE_WITH_BROADCAST)

        for (data1, data2, data3, data4, data5) in ins:
            with tbe.compute():
                predict_shape, target_shape, dout_shape, weight_shape, pos_weight_shape = \
                    shape_util.variable_shape([data1, data2, data3, data4, data5])

                predict_data = tvm.placeholder(predict_shape, predict_dtype, name="predict_data")
                target_data = tvm.placeholder(target_shape, predict_dtype, name="target_data")
                dout_data = tvm.placeholder(dout_shape, predict_dtype, name="dout_data")
                weight_data = tvm.placeholder(weight_shape, predict_dtype, name="weight_data")
                pos_weight_data = tvm.placeholder(pos_weight_shape, predict_dtype, name="pos_weight_data")

                res = sigmoid_cross_entropy_with_logits_grad_v2_compute(predict_data, target_data, dout_data,
                                                                        weight_data, pos_weight_data, reduction)

                tensors.append([predict_data, target_data, dout_data, weight_data, pos_weight_data, res])

            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    elif (weight is not None) and (pos_weight is None):

        shape_util.compare_tensor_dict_key(weight, target, "dtype")

        schedules, tensors = [], []
        ins = classify([predict, target, dout, weight], OpPatternMode.ELEWISE_WITH_BROADCAST)

        for (data1, data2, data3, data4) in ins:
            with tbe.compute():
                predict_shape, target_shape, dout_shape, weight_shape = \
                    shape_util.variable_shape([data1, data2, data3, data4])

                predict_data = tvm.placeholder(predict_shape, predict_dtype, name="predict_data")
                target_data = tvm.placeholder(target_shape, predict_dtype, name="target_data")
                dout_data = tvm.placeholder(dout_shape, predict_dtype, name="dout_data")
                weight_data = tvm.placeholder(weight_shape, predict_dtype, name="weight_data")

                res = sigmoid_cross_entropy_with_logits_grad_v2_compute(predict_data, target_data, dout_data,
                                                                        weight_data, pos_weight_data, reduction)

                tensors.append([predict_data, target_data, dout_data, weight_data, res])

            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    elif (weight is None) and (pos_weight is not None):

        shape_util.compare_tensor_dict_key(pos_weight, dout, "dtype")

        schedules, tensors = [], []
        ins = classify([predict, target, dout, pos_weight], OpPatternMode.ELEWISE_WITH_BROADCAST)

        for (data1, data2, data3, data4) in ins:
            with tbe.compute():
                predict_shape, target_shape, dout_shape, pos_weight_shape = \
                    shape_util.variable_shape([data1, data2, data3, data4])

                predict_data = tvm.placeholder(predict_shape, predict_dtype, name="predict_data")
                target_data = tvm.placeholder(target_shape, predict_dtype, name="target_data")
                dout_data = tvm.placeholder(dout_shape, predict_dtype, name="dout_data")
                pos_weight_data = tvm.placeholder(pos_weight_shape, predict_dtype, name="pos_weight_data")

                res = sigmoid_cross_entropy_with_logits_grad_v2_compute(predict_data, target_data, dout_data,
                                                                        weight_data, pos_weight_data, reduction)

                tensors.append([predict_data, target_data, dout_data, pos_weight_data, res])

            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    elif (weight is None) and (pos_weight is None):

        schedules, tensors = [], []
        ins = classify([predict, target, dout], OpPatternMode.ELEWISE_WITH_BROADCAST)

        for (data1, data2, data3) in ins:
            with tbe.compute():
                predict_shape, target_shape, dout_shape = \
                    shape_util.variable_shape([data1, data2, data3])

                predict_data = tvm.placeholder(predict_shape, predict_dtype, name="predict_data")
                target_data = tvm.placeholder(target_shape, predict_dtype, name="target_data")
                dout_data = tvm.placeholder(dout_shape, predict_dtype, name="dout_data")

                res = sigmoid_cross_entropy_with_logits_grad_v2_compute(predict_data, target_data, dout_data,
                                                                        weight_data, pos_weight_data, reduction)

                tensors.append([predict_data, target_data, dout_data, res])

            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    # build
    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
