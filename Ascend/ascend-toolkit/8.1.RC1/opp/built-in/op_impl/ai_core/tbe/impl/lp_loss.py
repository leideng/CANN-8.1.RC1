# Copyright 2020 Huawei Technologies Co., Ltd
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
lp_loss
"""

import functools
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=invalid-name,unused-argument,too-many-arguments,too-many-locals
def l1_loss_compute(predict, label, reduction):
    """
    :param predict: TVM tensor
        the placeholder of predict
    :param label: TVM tensor
        the placeholder of label
    :param reduction: str
        reduce mode, can be 'mean','sum' or 'none'
    :return: output tensor
    """
    predict_shape = shape_util.shape_to_list(predict.shape)

    # float16 cast to float32
    precision_dtype = "float32"
    predict_dtype = predict.dtype.lower()
    if predict_dtype == "float16":
        predict = tbe.cast_to(predict, precision_dtype)
        label = tbe.cast_to(label, precision_dtype)

    # calculate the result of loss = |predict-label|
    loss = tbe.vabs(tbe.vsub(predict, label))

    # calculate the result of sum(loss)
    if reduction == "sum":
        dims = list(range(len(predict_shape)))
        loss = tbe.sum(loss, dims)

    # calculate the result of mean(loss)
    if reduction == "mean":
        dims = list(range(len(predict_shape)))
        sum_loss = tbe.sum(loss, dims)
        num = functools.reduce(lambda x, y: x * y, predict_shape)
        norm = 1.0 / num
        loss = tbe.vmuls(sum_loss, norm)

    loss = tbe.cast_to(loss, predict_dtype)
    return loss


@register_operator_compute("lp_loss", op_mode="static", support_fusion=True)
def lp_loss_compute(predict, label, p, reduction, kernel_name="lp_loss"):
    """
    :param predict: TVM tensor
        the placeholder of predict
    :param label: TVM tensor
        the placeholder of label
    :param p: int
        decides which loss to compute, now the p only can be 1 to compute l1_loss
    :param reduction: str
        reduce mode,can be 'mean','sum' or 'none'
    :param kernel_name: ernel name, default value is "lp_loss"
    :return: output tensor
    """
    res = l1_loss_compute(predict, label, reduction)
    return res


@para_check.check_op_params(dict, dict, dict, int, str, str)
def lp_loss(predict, label, y, p, reduction="mean", kernel_name="lp_loss"):
    """
    :param predict: dict
        shape and dtype of input
    :param label: dict
        shape and dtype of label, should be same shape and type as predict
    :param y: dict
        shape and dtype of y, should be same shape and type as predict
    :param p: int
        decides which loss to compute, now the p only can be 1 to compute l1_loss
    :param reduction: str
        reduce mode,can be 'mean','sum' or 'none'
    :param kernel_name: kernel name, default value is "lp_loss"
    :return:
        None
    """
    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype").lower()
    label_shape = label.get("shape")
    label_dtype = label.get("dtype").lower()

    dtype_list = ["float16", "float32"]
    reduction_list = ["none", "mean", "sum"]

    para_check.check_dtype(predict_dtype, dtype_list)
    para_check.check_dtype(label_dtype, dtype_list)
    para_check.check_shape(predict_shape)
    para_check.check_shape(label_shape)

    shape_util.compare_tensor_dict_key(predict, label, "shape")
    shape_util.compare_tensor_dict_key(predict, label, "dtype")

    if p != 1:
        raise RuntimeError("lp_loss only supports l1_loss")

    if reduction not in reduction_list:
        raise RuntimeError("reduction should be one of ['none','mean','sum']")

    predict_data = tvm.placeholder(predict_shape, dtype=predict_dtype, name="predict_data")
    label_data = tvm.placeholder(label_shape, dtype=label_dtype, name="label_data")

    res = lp_loss_compute(predict_data, label_data, p, reduction, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [predict_data, label_data, res]}
    tbe.cce_build_code(schedule, config)
