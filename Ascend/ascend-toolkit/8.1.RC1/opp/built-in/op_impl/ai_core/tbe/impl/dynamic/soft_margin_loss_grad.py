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
soft_margin_loss_grad
"""

import functools
import math
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context


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


# 'pylint: disable=unused-argument,too-many-locals
@register_operator_compute("SoftMarginLossGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def soft_margin_loss_gard_compute(input_predict, input_label, input_dout, output_gradient,
                                  reduction, kernel_name="soft_margin_loss_grad"):
    """calculating data

    Parameters
    ----------
    :param input_predict: TVM tensor
        the placeholder of input_predict
    :param input_label: TVM tensor
        the placeholder of input_label
    :param input_dout: TVM tensor
        the placeholder of input_dout
    :param output_gradient: dict
        shape and dtype of output, should be same shape and type as predict
    :param reduction: str
        the method of reduction
    :param kernel_name: str
        kernel name, default value is "soft_margin_loss_gard"

    Returns
    -------
    output tensor
    """
    predict_shape = shape_util.shape_to_list(input_predict.shape)
    _, _, shape_max = shape_util.broadcast_shapes(input_predict.shape, input_label.shape)

    input_predict = tbe.broadcast(input_predict, shape_max)
    input_label = tbe.broadcast(input_label, shape_max)
    input_dout = tbe.broadcast(input_dout, shape_max)

    dtype = input_predict.dtype
    precision_dtype = "float32"

    if (dtype == "float16" or dtype == "bfloat16"):
        input_predict = tbe.cast_to(input_predict, precision_dtype)
        input_label = tbe.cast_to(input_label, precision_dtype)
        input_dout = tbe.cast_to(input_dout, precision_dtype)

    neg_label = tbe.vmuls(input_label, tvm.const(-1, precision_dtype))
    predict_mul_label = tbe.vmul(input_predict, neg_label)
    res_exp = tbe.vexp(predict_mul_label)

    res_div = tbe.vdiv(res_exp, tbe.vadds(res_exp, tvm.const(1, dtype)))
    res_neg = tbe.vmul(neg_label, res_div)
    res = tbe.vmul(res_neg, input_dout)

    if reduction == "mean":
        cof = get_cof_by_shape(predict_shape, precision_dtype)
        res = tbe.vmuls(res, cof)

    if (dtype == "float16" or dtype == "bfloat16"):
        res = tbe.cast_to(res, dtype)
    return res


# 'pylint: disable=too-many-arguments,too-many-locals
@register_operator("SoftMarginLossGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def soft_margin_loss_grad(input_predict, input_label, input_dout, output_gradient,
                          reduction="mean",
                          kernel_name="soft_margin_loss_grad"):
    """calculating data

    Parameters
    ----------
    :param input_predict: dict
        shape and dtype of predict
    :param input_label: dict
        shape and dtype of label
    :param input_dout: dict
        shape and dtype of dout
    :param output_gradient: dict
        shape and dtype of output, should be same shape and type as predict
    :param reduction: str
        the method of reduction
    :param kernel_name: str
        kernel name, default value is "soft_margin_loss_gard"

    Returns
    -------
    None
    """

    predict_dtype = input_predict.get("dtype")
    label_dtype = input_label.get("dtype")
    dout_dtype = input_dout.get("dtype")

    dtype_list = ["float16", "float32", "bfloat16"]
    para_check.check_dtype(predict_dtype, dtype_list)

    shape_util.compare_tensor_dict_key(input_predict, input_label, "dtype")
    shape_util.compare_tensor_dict_key(input_predict, input_dout, "dtype")

    reduction_list = ("mean", "none", "sum")
    if reduction not in reduction_list:
        raise RuntimeError("reduction of soft_margin_loss_gard only supports [mean, none, sum], but actual is %s. "
                           % reduction)

    ins = classify([input_predict, input_label, input_dout], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_predict, _label, _dout) in ins:
        with tbe.compute():
            predict_shape, label_shape, dout_shape = shape_util.variable_shape([_predict, _label, _dout])
            predict_data = tvm.placeholder(predict_shape, name="predict_data", dtype=predict_dtype)
            label_data = tvm.placeholder(label_shape, name="label_data", dtype=label_dtype)
            dout_data = tvm.placeholder(dout_shape, name="dout_data", dtype=dout_dtype)
            res = soft_margin_loss_gard_compute(predict_data, label_data, dout_data, output_gradient, reduction)

            tensors.append([predict_data, label_data, dout_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
