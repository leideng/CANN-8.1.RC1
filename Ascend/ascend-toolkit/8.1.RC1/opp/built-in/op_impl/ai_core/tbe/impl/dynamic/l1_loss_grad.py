#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
l1_loss_grad
"""
import math
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=invalid-name,too-many-arguments,unused-argument,too-many-locals
@register_operator_compute("L1LossGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def l1_loss_grad_compute(grads, predict, label, y, reduction="mean", kernel_name="l1_loss_grad"):
    """
    l1_loss_grad_compute
    """
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

    data_type = predict.dtype.lower()
    ori_type = data_type
    need_cast = data_type == "float32" and cce_product == "Ascend310"
    if need_cast:
        data_type = "float16"
        grads = tbe.cast_to(grads, data_type, False)
        label = tbe.cast_to(label, data_type, False)
        predict = tbe.cast_to(predict, data_type, False)
    elif ori_type == "float16" and cce_product != "Ascend310":
        data_type = "float32"
        grads = tbe.cast_to(grads, data_type, False)
        label = tbe.cast_to(label, data_type, False)
        predict = tbe.cast_to(predict, data_type, False)


    zero_tensor = tbe.broadcast(tvm.const(0, data_type), predict.shape)
    sign = tbe.vcmpsel(predict, label, "gt", 1.0, -1.0)
    sign = tbe.cast_to(sign, data_type)
    sign = tbe.vcmpsel(predict, label, "ne", sign, zero_tensor)
    sign = tbe.cast_to(sign, data_type)
    # if choose "mean", grads should divide over n
    if reduction == "mean":
        shape = shape_util.shape_to_list(grads.shape)
        reduce_elts = 1.0
        for i in shape:
            reduce_elts *= i
        if isinstance(reduce_elts, float):
            if math.isclose(reduce_elts, 0.0):
                if need_cast:
                    inf_data = tvm.const(65504, data_type)
                else:
                    inf_data = tvm.const(2**62, data_type)
                grads = tbe.vadds(grads, inf_data)
                res = tbe.vmul(sign, grads)
                if need_cast:
                    res = tbe.cast_to(res, "float32", False)
                return res
            cof = reduce_elts ** (-1)
            cof = tvm.const(cof, dtype=data_type)
        else:
            cof = tbe.var("cof", dtype=data_type)
            if data_type == "float16":
                tbe.var("cof_empty", dtype=data_type)
            tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", data_type)
        grads = tbe.vmuls(grads, cof)

    # chain multiplication to get the gradient of L1 with respect to weights(grads)
    res = tbe.vmul(sign, grads)
    if need_cast:
        res = tbe.cast_to(res, "float32", False)
    elif ori_type == "float16" and data_type == "float32":
        res = tbe.cast_to(res, "float16", False)


    return res


# 'pylint: disable=locally-disabled,invalid-name,unused-argument,too-many-locals
@register_operator("L1LossGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def l1_loss_grad(grads, predict, label, y, reduction="mean", kernel_name="l1_loss_grad"):
    """
    Parameters
    ----------
    grads : dict
        shape and dtype of grad_out as input
    predict : dict
        shape and dtype of predict as input, should be same shape and type as grads
    label : dict
        shape and dtype of label as input, should be same shape and type as grads
    y : dict
        shape and dtype of output, should be same shape and type as grads
    reduction: string
        reduction name, default value is "mean"
    kernel_name : str
        kernel name, default value is "l1_loss_grad"

    Returns
    -------
    None
    """

    dtype_list = ["float16", "float32", "bfloat16"]
    reduction_list = ["none", "mean", "sum"]
    grads_dtype = grads.get("dtype").lower()
    label_dtype = label.get("dtype").lower()
    predict_dtype = predict.get("dtype").lower()
    para_check.check_dtype(grads_dtype, dtype_list, param_name="grads")
    para_check.check_dtype(label_dtype, dtype_list, param_name="label")
    para_check.check_dtype(predict_dtype, dtype_list, param_name="predict")

    if reduction not in reduction_list:
        raise RuntimeError("reduction should be one of ['none','mean','sum']")

    ins = classify([grads, predict, label], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_grads, _predict, _label) in ins:
        with tbe.compute():
            shape_grads, shape_predict, shape_label = shape_util.variable_shape([_grads, _predict, _label])
            tensor_grads = tvm.placeholder(shape_grads, name="tensor_grads", dtype=grads_dtype)
            tensor_label = tvm.placeholder(shape_label, name="tensor_label", dtype=label_dtype)
            tensor_predict = tvm.placeholder(shape_predict, name="tensor_predict", dtype=predict_dtype)

            res = l1_loss_grad_compute(tensor_grads, tensor_predict, tensor_label, y,
                                       reduction=reduction, kernel_name=kernel_name)
            tensors.append([tensor_grads, tensor_predict, tensor_label, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
