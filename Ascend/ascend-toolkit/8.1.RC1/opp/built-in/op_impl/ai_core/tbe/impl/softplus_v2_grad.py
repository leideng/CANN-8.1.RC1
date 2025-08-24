#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
softplus_v2_grad
"""

import math
import te.lang.cce as tbe
from tbe import tvm
from te.utils import para_check
import te.platform as tbe_platform
from te.utils.shape_util import broadcast_shapes
from te.utils.shape_util import shape_to_list


# 'pylint: disable=too-many-locals,too-many-arguments,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("softplus_v2_grad")
def softplus_v2_grad_compute(input_gradients, input_features, beta, threshold):
    """
    calculating data

    Parameters
    ----------
    input_gradients : TVM tensor
       the placeholder of input_gradients
    input_features : TVM tensor
       the placeholder of input_features
    beta : float
       scale factor, default value is 1.0
    threshold: float
       boundary value, default value is 20.0
    kernel_name : str
       kernel name, default value is "softplus_v2_grad"

    Returns
    -------
    output tensor
    """
    input_dtype = input_gradients.dtype
    shape_grad = shape_to_list(input_gradients.shape)
    shape_feature = shape_to_list(input_features.shape)
    if list(shape_grad) != list(shape_feature):
        _, _, shape_max = broadcast_shapes(shape_grad, shape_feature,
                                           param_name_input1="input_gradients",
                                           param_name_input2="input_features")
        input_gradients = tbe.broadcast(input_gradients, shape_max, input_dtype)
        input_features = tbe.broadcast(input_features, shape_max, input_dtype)
    if input_dtype != "float32":
        input_gradients = tbe.cast_to(input_gradients, "float32")
        input_features = tbe.cast_to(input_features, "float32")

    one_const_tensor = tbe.broadcast(tvm.const(1.0, dtype="float32"), input_features.shape)
    beta_const_tensor = tbe.broadcast(tvm.const(beta, dtype="float32"), input_features.shape)

    left_cmp = tbe.vmul(input_features, beta_const_tensor)
    method_two_get_res = tbe.vcmpsel(left_cmp, tvm.const(threshold, dtype="float32"), 'gt', 1.0, 0.0)
    is_method_one = tbe.vsub(one_const_tensor, method_two_get_res)
    # Prevent exp overflow caused by large number calculation
    input_features_select = tbe.vmul(left_cmp, is_method_one)
    # calculate exp(beta*x)/ (1 + exp(beta*x))
    exp_res = tbe.vexp(input_features_select)
    exp_add_one_res = tbe.vadd(exp_res, one_const_tensor)
    method_one_res = tbe.vdiv(exp_res, exp_add_one_res)

    method_one_get_res = tbe.vmul(method_one_res, is_method_one)
    grad_out = tbe.vadd(method_one_get_res, method_two_get_res)
    res_tmp = tbe.vmul(grad_out, input_gradients)

    if input_dtype == "float16":
        res = tbe.cast_to(res_tmp, "float16")
    else:
        res = res_tmp

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def softplus_v2_grad(input_gradients, input_features, output_backprops,
                     beta=1.0, threshold=20.0, kernel_name="softplus_v2_grad"):
    """
    Computes softplus gradients for a softplus operation.
    The gradients: "dy * exp(x)/ (1 + exp(x))" if x/beta <= threshold else dy.

    Parameters
    ----------
    input_gradients: dict
        The backpropagated gradients to the corresponding softplus operation.
    input_features: dict
        The input_features passed as input to the corresponding softplus operation.
        source data type support "float16", "float32".
    output_backprops: dict
        data of output.
    beta: float16/float32, option, default:1.0
    threshold: float16/float32, option, default:20.0

    kernel_name: str
        kernel name, default value is "softplus_grad_v2".
    Returns
    -------
    None
    """
    shape_grad = input_gradients.get("shape")
    shape_feature = input_features.get("shape")
    dtype_grad = input_gradients.get("dtype")
    dtype_feature = input_features.get("dtype")
    # check dtype and shape
    if dtype_grad.lower() != dtype_feature.lower():
        raise RuntimeError(
            "type of grads and type of feature must be same, \
             while the types are different")
    input_dtype = dtype_grad.lower()

    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_gradients")

    para_check.check_shape(shape_grad, param_name="input_gradients")
    para_check.check_shape(shape_feature, param_name="input_features")
    if math.isclose(beta, 0.0):
        raise ZeroDivisionError("the value of beta must be non-zero")
    # broadcast grad and feature
    if len(list(shape_grad)) != len(list(shape_feature)):
        raise RuntimeError(
            "shape of grads and shape of feature \
             must have the same length")
    shape_grad, shape_feature, _ = broadcast_shapes(shape_grad, shape_feature,
                                                    param_name_input1="input_gradients",
                                                    param_name_input2="input_features")

    data_grads = tvm.placeholder(shape_grad, dtype=input_dtype, name="data_grads")
    data_features = tvm.placeholder(shape_feature, dtype=input_dtype, name="data_features")

    res = softplus_v2_grad_compute(data_grads, data_features, beta, threshold)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_grads, data_features, res]}

    tbe.cce_build_code(schedule, config)
