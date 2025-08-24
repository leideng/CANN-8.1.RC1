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
softplus_v2_grad
"""

import math
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_attr_common import SoftplusV2GradAttrInfo


class Constant:
    """
    The class for constant
    """
    ONE_CONST = 1.0
    EXP_RATE = -50.0


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
# 'pylint: disable=unused-variable
@register_operator_compute("SoftplusV2Grad",
                           op_mode="dynamic",
                           support_fusion=True,
                           support_bfp16=True)
def softplus_v2_grad_compute(input_gradients,
                             input_features,
                             output_backprops,
                             beta,
                             threshold,
                             kernel_name="softplus_v2_grad"):
    """
    calculating data
    The compute: "dy / (1 + exp(-/beta*x))" if x/beta <= threshold else dy.
    Parameters
    ----------
    input_gradients : TVM tensor
       the placeholder of input_gradients
    input_features : TVM tensor
       the placeholder of input_features
    output_backprops: dict
        data of output.
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
    gradients_shape = input_gradients.shape
    feature_shape = input_features.shape
    shape_grad = shape_util.shape_to_list(gradients_shape)
    shape_feature = shape_util.shape_to_list(feature_shape)

    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        target_type = "float32"
    else:
        target_type = "float16"

    if input_dtype != target_type:
        input_gradients = tbe.cast_to(input_gradients, target_type)
        input_features = tbe.cast_to(input_features, target_type)

    if list(shape_grad) != list(shape_feature):
        _, _, shape_max = shape_util.broadcast_shapes(
            gradients_shape,
            feature_shape,
            param_name_input1="input_gradients",
            param_name_input2="input_features")
        input_gradients = tbe.broadcast(input_gradients, shape_max)
        input_features = tbe.broadcast(input_features, shape_max)

    beta = get_attr_by_cls(beta, SoftplusV2GradAttrInfo.ATTR_BETA, target_type)
    threshold = get_attr_by_cls(threshold, SoftplusV2GradAttrInfo.ATTR_THRESHOLD, target_type)
    
    neg_one_scalar = tvm.const(-1.0, dtype=target_type)
    beta_const_tensor = tbe.broadcast(beta, input_features.shape)
    neg_beta_const_tensor = tbe.vmuls(beta_const_tensor, neg_one_scalar)

    left_cmp = tbe.vmul(input_features, neg_beta_const_tensor)
    threshold_const_tensor = tbe.broadcast(threshold, left_cmp.shape)
    neg_threshold_const_tensor = tbe.vmuls(threshold_const_tensor, neg_one_scalar)

    exp_x = tbe.vcmpsel(left_cmp, neg_threshold_const_tensor, 'lt', Constant.EXP_RATE, left_cmp)

    # when beta*x<=thresh  output = dy / (1 + exp(-beta*x)) else output = dy/(1+exp(-50.0)) = dy
    exp_res = tbe.vexp(exp_x)
    exp_add_one_res = tbe.vadds(exp_res, tvm.const(Constant.ONE_CONST, dtype=target_type))
    res = tbe.vdiv(input_gradients, exp_add_one_res)

    if target_type != input_dtype:
        res = tbe.cast_to(res, input_dtype)

    return res


@register_operator("SoftplusV2Grad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def softplus_v2_grad(input_gradients,
                     input_features,
                     output_backprops,
                     beta=1.0,
                     threshold=20.0,
                     kernel_name="softplus_v2_grad"):
    """
    Computes softplus gradients for a softplus operation.
    The gradients: "dy / (1 + exp(-/beta*x))" if x/beta <= threshold else dy.

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
    dtype_dy = input_gradients.get("dtype").lower()
    dtype_x = input_features.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_dy, check_list, param_name="input_g")
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    if dtype_dy != dtype_x:
        error_detail = "Dtype of tensor input_gradients and input_features must be same!"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "input_gradients", \
                                                               "input_features", error_detail)

    ins = classify([input_gradients, input_features],
                   OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_g, input_x) in ins:
        with tbe.compute():
            g_shape, x_shape = shape_util.variable_shape([input_g, input_x])
            tensor_g = tvm.placeholder(g_shape, dtype_dy, "tensor_g")
            tensor_x = tvm.placeholder(x_shape, dtype_x, "tensor_x")
            res = softplus_v2_grad_compute(tensor_g, tensor_x,
                                           output_backprops, beta, threshold,
                                           kernel_name)
            tensors.append([tensor_g, tensor_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
