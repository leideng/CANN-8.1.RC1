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
soft_margin_loss_grad
"""

import math
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    SHAPE_SIZE_LIMIT = 2147483648


def get_cof_by_shape(predict_shape, precision_dtype):
    """
    get cof by shape
    """
    reduce_elts = 1.0
    for i in predict_shape:
        reduce_elts *= i

    cof = reduce_elts if math.isclose(reduce_elts, 0.0) else reduce_elts ** (-1)
    cof = tvm.const(cof, dtype=precision_dtype)

    return cof


# 'pylint: disable=unused-argument,too-many-locals
@register_operator_compute("soft_margin_loss_grad", op_mode="static", support_fusion=True)
def soft_margin_loss_gard_compute(input_predict, input_label, input_dout,
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
    :param reduction: str
        the method of reduction
    :param kernel_name: str
        kernel name, default value is "soft_margin_loss_gard"

    Returns
    -------
    output tensor
    """
    predict_shape = shape_util.shape_to_list(input_predict.shape)
    label_shape = shape_util.shape_to_list(input_label.shape)
    _, _, shape_max = shape_util.broadcast_shapes(predict_shape, label_shape)
    para_check.check_shape_size(shape_max, Constant.SHAPE_SIZE_LIMIT)

    input_predict = tbe.broadcast(input_predict, shape_max)
    input_label = tbe.broadcast(input_label, shape_max)
    input_dout = tbe.broadcast(input_dout, shape_max)

    dtype = input_predict.dtype
    precision_dtype = "float32"

    if dtype == "float16":
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

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")
    return res


# 'pylint: disable=too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def soft_margin_loss_grad(input_predict, input_label, input_dout, output_gdient,
                          reduction="mean",
                          kernel_name="soft_margin_loss_grad"):
    """calculating data

    Parameters
    ----------
    :param input_predict: dict
        shape and dtype of predpict
    :param input_label: dict
        shape and dtype of label
    :param input_dout: dict
        shape and dtype of dout
    :param output_gdient: dict
        shape and dtype of output, should be same shape and type as predpict
    :param reduction: str
        the method of reduction
    :param kernel_name: str
        kernel name, default value is "soft_margin_loss_gard"

    Returns
    -------
    None
    """

    predict_shape = shape_util.scalar2tensor_one(input_predict.get("shape"))
    predict_dtype = input_predict.get("dtype")
    label_shape = shape_util.scalar2tensor_one(input_label.get("shape"))
    label_dtype = input_label.get("dtype")
    dout_shape = shape_util.scalar2tensor_one(input_dout.get("shape"))
    dout_dtype = input_dout.get("dtype")

    # reshape
    predict_shape = list(predict_shape)
    label_shape = list(label_shape)
    dout_shape = list(dout_shape)
    if len(predict_shape) > len(label_shape):
        times = len(predict_shape) - len(label_shape)
        cnt = 0
        while cnt < times:
            label_shape.insert(0, 1)
            cnt += 1

    if len(predict_shape) > len(dout_shape):
        times = len(predict_shape) - len(dout_shape)
        cnt = 0
        while cnt < times:
            dout_shape.insert(0, 1)
            cnt += 1

    # initialize data
    predict_data = tvm.placeholder(predict_shape, name="predict_data",
                                   dtype=predict_dtype)
    label_data = tvm.placeholder(label_shape, name="label_data",
                                 dtype=label_dtype)
    dout_data = tvm.placeholder(dout_shape, name="dout_data",
                                dtype=dout_dtype)
    res = soft_margin_loss_gard_compute(predict_data, label_data, dout_data,
                                        reduction)

    # auto schedule
    with tvm.target.cce():
        schedule = auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [predict_data, label_data, dout_data, res]}
    build(schedule, config)
