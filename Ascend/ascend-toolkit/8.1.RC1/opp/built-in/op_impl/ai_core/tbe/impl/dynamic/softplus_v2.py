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
softplus_v2
"""

import math

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_attr_common import SoftplusV2AttrInfo
from impl.dynamic.log1p import log1p_compute


# 'pylint: disable=too-many-locals,unused-argument,invalid-name
@register_operator_compute("SoftplusV2",
                           op_mode="dynamic",
                           support_fusion=True,
                           support_bfp16=True)
def softplus_v2_compute(input_features,
                        y,
                        beta,
                        threshold,
                        kernel_name="softplus_v2"):
    """
    calculating data
    The compute: "if beta*x<thresh log(1+exp(beta*x))/beta else x".
    Parameters
    ----------
    input_features : TVM tensor
       the placeholder of input_features
    y: dict
        data of output.
    beta : float
       scale factor, default value is 1.0
    threshold: float
       boundary value, default value is 20.0
    kernel_name : str
       kernel name, default value is "softplus_v2"

    Returns
    -------
    output tensor
    """
    input_dtype = input_features.dtype
    input_shape = input_features.shape

    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        target_type = "float32"
    else:
        target_type = "float16"

    if input_dtype != target_type:
        input_features = tbe.cast_to(input_features, target_type)

    beta = get_attr_by_cls(beta, SoftplusV2AttrInfo.ATTR_BETA, target_type)
    threshold = get_attr_by_cls(threshold, SoftplusV2AttrInfo.ATTR_THRESHOLD, target_type)

    beta_const_tensor = tbe.broadcast(beta, input_shape)
    one_const_tensor = tbe.broadcast(tvm.const(1.0, dtype=target_type), input_shape)
    one_div_beta_const_tensor = tbe.vdiv(one_const_tensor, beta_const_tensor)
    left_cmp = tbe.vmul(input_features, beta_const_tensor)
    
    is_method_two = tbe.vcmpsel(left_cmp, threshold, 'gt', 1.0, 0.0)
    is_method_one = tbe.vsub(one_const_tensor, is_method_two)
    # Prevent exp overflow caused by large number calculation
    input_features_select = tbe.vmul(left_cmp, is_method_one)
    # calculate log(1+exp(beta*x))/beta
    exp_pos = tbe.vexp(input_features_select)
    res_pos = log1p_compute(exp_pos, y)
    method_one_res = tbe.vmul(res_pos, one_div_beta_const_tensor)

    res = tbe.vcmpsel(is_method_one, 1.0, "eq", method_one_res, input_features)
    if input_dtype != target_type:
        res = tbe.cast_to(res, input_dtype)

    return res


@register_operator("SoftplusV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def softplus_v2(x, y, beta=1.0, threshold=20.0, kernel_name="softplus_v2"):
    """
    Computes softplus operation with attribute beta and threshold.
    The output: log(1+exp(beta*x))/beta if x/beta <= threshold else x.

    Parameters
    ----------
    x: dict
        The input_features passed as input to the corresponding softplus operation.
        source data type support "float16", "float32".
    y: dict
        data of output.
    beta: float16/float32, option, default:1.0
    threshold: float16/float32, option, default:20.0

    kernel_name: str
        kernel name, default value is "softplus_v2".
    Returns
    -------
    None
    """
    shape_feature = x.get("shape")
    dtype_feature = x.get("dtype")
    # check dtype and shape
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_feature, check_list, param_name="x")
    para_check.check_shape(shape_feature, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (input_x, ) in ins:
        # op compute
        with tbe.compute():
            x_shape = shape_util.variable_shape([input_x])
            input_data = tvm.placeholder(x_shape[0],
                                         name="input_data",
                                         dtype=dtype_feature)

            res = softplus_v2_compute(input_data, y, beta, threshold,
                                      kernel_name)
            tensors.append([input_data, res])
        # target auto schedule
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        # append schedule 2 schedules
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
