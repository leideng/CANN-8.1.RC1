#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
sigmoid_cross_entropy_with_logits
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("sigmoid_cross_entropy_with_logits", op_mode="static", support_fusion=True)
def sigmoid_cross_entropy_with_logits_compute(predict,
                                              target,
                                              loss,
                                              kernel_name):
    """calculating data

    Parameters
    ----------
    predict : TVM tensor
        the placeholder of predict
    target : TVM tensor
        the placeholder of target
    loss : dict
        dict of loss, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits"

    Returns
    -------
    output tensor
    """
    predict_dtype = predict.dtype
    target_dtype = target.dtype
    if predict_dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vsub", "float32"):
        predict = tbe.cast_to(predict, "float32")
    if target_dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        target = tbe.cast_to(target, "float32")

    dtype_predict = predict.dtype
    shape_predict = shape_util.shape_to_list(predict.shape)

    const_zero = tvm.const(0, dtype=dtype_predict)
    max_predict_zero = tbe.vmaxs(predict, const_zero)

    abs_predict = tbe.vabs(predict)
    const_zero_broadcast = tbe.broadcast(const_zero, shape_predict)
    reverse_abs_predict = tbe.vsub(const_zero_broadcast, abs_predict)
    vexp_predict = tbe.vexp(reverse_abs_predict)
    const_one = tvm.const(1, dtype=dtype_predict)
    vadds_res = tbe.vadds(vexp_predict, const_one)
    vlog_res = tbe.vlog(vadds_res, priority_flag=1)
    vmul_res = tbe.vmul(predict, target)
    res = tbe.vsub(vlog_res, vmul_res)
    loss = tbe.vadd(res, max_predict_zero)

    if predict_dtype == "float16":
        loss = tbe.cast_to(loss, "float16")

    return loss


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sigmoid_cross_entropy_with_logits(
        predict, target, loss,
        kernel_name="sigmoid_cross_entropy_with_logits"):
    """calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of predict
    target : dict
        shape and dtype of target
    loss : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits"

    Returns
    -------
    None
    """
    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype")
    input_dtype_predict = dtype_predict.lower()
    para_check.check_shape(shape_predict, param_name="predict")

    shape_target = target.get("shape")
    dtype_target = target.get("dtype")
    input_dtype_target = dtype_target.lower()
    para_check.check_shape(shape_target, param_name="target")

    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype_predict, check_list, param_name="predict")
    para_check.check_dtype(input_dtype_target, check_list, param_name="target")
    shape_predict, shape_target = \
        shape_util.refine_shapes_for_broadcast(shape_predict, shape_target)
    data_predict = tvm.placeholder(shape_predict,
                                   name="data_predict",
                                   dtype=input_dtype_predict)
    data_target = tvm.placeholder(shape_target,
                                  name="data_target",
                                  dtype=input_dtype_target)
    loss = sigmoid_cross_entropy_with_logits_compute(data_predict,
                                                     data_target,
                                                     loss,
                                                     kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(loss)

    config = {"name": kernel_name,
              "tensor_list": [data_predict, data_target, loss]}

    tbe.cce_build_code(sch, config)
