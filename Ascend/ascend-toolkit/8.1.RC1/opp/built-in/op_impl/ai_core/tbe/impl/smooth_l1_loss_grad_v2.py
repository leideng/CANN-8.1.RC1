#!/usr/bin/python
# -*- coding: utf-8 -*-
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
smooth_l1_loss_grad_v2
"""
import functools
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-many-locals,invalid-name,unused-argument,too-many-arguments
@tbe_platform.fusion_manager.register("smooth_l1_loss_grad_v2")
def smooth_l1_loss_grad_v2_compute(input_predict,
                                   input_label,
                                   input_dout,
                                   sigma,
                                   reduction):
    """calculating data

    Parameters
    ----------
    input_predict : TVM tensor
       the placeholder of input_predict
    input_label : TVM tensor
       the placeholder of input_label
    input_dout : TVM tensor
        the placeholder of input_dout
    sigma : float
        default value is 1.0
    reduction: str
       type of result, default value is "mean"
    kernel_name : str
       kernel name, default value is "smooth_l1_loss_grad_v2"

    Returns
    -------
    output tensor
    """

    ori_dtype = input_predict.dtype

    # vcmpsel of Ascend310 only support float16
    product = tbe_platform.api_check_support("tbe.dsl.vcmpsel", "float32")
    if product:
        all_dtype = "float32"
    else:
        all_dtype = "float16"

    if ori_dtype != all_dtype:
        input_predict = tbe.cast_to(input_predict, all_dtype)
        input_label = tbe.cast_to(input_label, all_dtype)
        input_dout = tbe.cast_to(input_dout, all_dtype)

    # broadcast inputs
    predict_shape = shape_util.shape_to_list(input_predict.shape)
    label_shape = shape_util.shape_to_list(input_label.shape)
    dout_shape = shape_util.shape_to_list(input_dout.shape)

    predict_shape, label_shape, dout_shape, all_shape = shape_util.unify_broadcast_shapes(
        [predict_shape, label_shape, dout_shape])

    input_predict = tbe.broadcast(input_predict, all_shape)
    input_label = tbe.broadcast(input_label, all_shape)
    input_dout = tbe.broadcast(input_dout, all_shape)

    # calculate input_predict-input_label
    x = tbe.vsub(input_predict, input_label)

    # calculate |input_predict-input_label|
    x_abs = tbe.vabs(x)

    # create sigma_tensor and negative_sigma_tensor
    sigma_const = tvm.const(sigma, dtype=all_dtype)
    negative_sigma_const = tvm.const(-sigma, dtype=all_dtype)
    sigma_tensor = tbe.broadcast(sigma_const, all_shape)
    negative_sigma_tensor = tbe.broadcast(negative_sigma_const, all_shape)

    # calculate smooth
    temp = tbe.vdiv(x, sigma_tensor)
    smooth1 = tbe.vcmpsel(x, negative_sigma_tensor, 'le', -1.0, 0.0)
    smooth2 = tbe.vcmpsel(x, sigma_tensor, 'ge', 1.0, 0.0)
    smooth3 = tbe.vcmpsel(x_abs, sigma, 'lt', temp, 0.0)
    smooth1_2 = tbe.vadd(smooth1, smooth2)
    smooth = tbe.vadd(smooth1_2, smooth3)

    # calculate the res value and return
    res = tbe.vmul(smooth, input_dout)

    if reduction == "mean":
        reduce_elts = functools.reduce(lambda x, y:x * y, all_shape)
        cof = reduce_elts ** (-1)
        cof = tvm.const(cof, dtype=all_dtype)
        res = tbe.vmuls(res, cof)

    if ori_dtype != all_dtype:
        res = tbe.cast_to(res, ori_dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def smooth_l1_loss_grad_v2(predict,
                           label,
                           dout,
                           gradient,
                           sigma=1.0,
                           reduction='mean',
                           kernel_name="smooth_l1_loss_grad_v2"):
    """calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of input
    label : dict
        shape and dtype of input
    dout : dict
        shape and dtype of input
    gradient : dict
        shape and dtype of output, should be same shape and type as predict
    sigma : float
        sigma
    reduction : str
        default value is "mean"
    kernel_name : str
        kernel name, default value is "smooth_l1_loss_grad_v2"

    Returns
    -------
    None
    """
    # check input: predict label dout
    check_list = ("float16", "float32")

    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype").lower()
    para_check.check_dtype(dtype_predict, check_list, param_name="predict")

    shape_label = label.get("shape")
    dtype_label = label.get("dtype").lower()
    para_check.check_dtype(dtype_label, check_list, param_name="label")

    shape_dout = dout.get("shape")
    dtype_dout = dout.get("dtype").lower()
    para_check.check_dtype(dtype_dout, check_list, param_name="dout")

    para_check.check_shape(shape_predict, param_name="predict")
    para_check.check_shape(shape_label, param_name="label")
    para_check.check_shape(shape_dout, param_name="dout")

    # check reduction
    check_list_reduction = ("none", "mean", "sum")
    reduction_type = reduction.lower()
    para_check.check_dtype(reduction_type, check_list_reduction, param_name="reduction")

    if para_check.is_scalar(shape_dout):
        shape_dout = tuple([1] * (len(shape_label) - len(shape_dout))) + tuple(shape_dout)

    input_predict = tvm.placeholder(
        shape_predict, name="predict", dtype=dtype_predict)
    input_label = tvm.placeholder(
        shape_label, name="label", dtype=dtype_label)
    input_dout = tvm.placeholder(
        shape_dout, name="dout", dtype=dtype_dout)

    res = smooth_l1_loss_grad_v2_compute(input_predict,
                                         input_label,
                                         input_dout,
                                         sigma,
                                         reduction_type)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [input_predict, input_label, input_dout, res]
    }

    tbe.build(sch, config)
