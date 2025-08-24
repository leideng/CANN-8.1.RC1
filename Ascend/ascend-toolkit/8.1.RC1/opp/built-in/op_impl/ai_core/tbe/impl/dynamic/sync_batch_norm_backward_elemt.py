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
sync_batch_norm_backward_element
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute


# 'pylint:disable=too-many-arguments,too-many-locals,unused-argument
@register_operator_compute("SyncBatchNormBackwardElemt", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def sync_batch_norm_backward_elemt_compute(grad_output,
                                           save_input,
                                           mean,
                                           invstd,
                                           weight,
                                           mean_dy,
                                           mean_dy_xmu,
                                           grad_input,
                                           kernel_name="sync_batch_norm_backward_elemt"):
    """
    sync_batch_norm_backward_elemt_compute

    Parameters:
    ----------
    grad_output: TVM tensor
        The result of forward
    save_input: TVM tensor
        The input of forward
    mean: TVM tensor
        Mean of saved forward input
    invstd: TVM tensor
        Reciprocal of the variance of the saved forward input
    weight: TVM tensor
        parameters
    mean_dy: TVM tensor
        A part of sum_dy
    mean_dy_xmu: TVM tensor
        A part of sum_dy_xmu
    kernel_name: str
        cce kernel name, default value is sync_batch_norm_backward_elemt
    Returns
    -------
    grad_input: TVM tensor
        Gradient of save_input
    """
    output_dy_dtype = grad_output.dtype
    mean_dtype = mean.dtype
    if output_dy_dtype == "float16" and mean_dtype == "float32":
        grad_output_in = tbe.cast_to(grad_output, "float32")
        save_input_in = tbe.cast_to(save_input, "float32")
    else:
        grad_output_in = grad_output
        save_input_in = save_input

    output_dy = tbe.vsub(grad_output_in, mean_dy)
    input_mean = tbe.vsub(save_input_in, mean)
    invstd_sq = tbe.vmul(invstd, invstd)
    invstd_dy_xmu = tbe.vmul(invstd_sq, mean_dy_xmu)
    input_invstd = tbe.vmul(input_mean, invstd_dy_xmu)
    output_input = tbe.vsub(output_dy, input_invstd)
    invstd_w = tbe.vmul(invstd, weight)
    grad_input_result = tbe.vmul(output_input, invstd_w)

    if output_dy_dtype == "float16" and mean_dtype == "float32":
        grad_input = tbe.cast_to(grad_input_result, "float16")
    else:
        grad_input = grad_input_result    
    return grad_input


# 'pylint:disable=too-many-arguments,too-many-locals
def sync_batch_norm_backward_elemt(grad_output,
                                   save_input,
                                   mean,
                                   invstd,
                                   weight,
                                   mean_dy,
                                   mean_dy_xmu,
                                   grad_input,
                                   kernel_name="sync_batch_norm_backward_elemt"):
    """
    sync_batch_norm_backward_elemt

    Parameters:
    ----------
    grad_output: TVM tensor
        The result of forward
    save_input: TVM tensor
        The input of forward
    mean: TVM tensor
        Mean of saved forward input
    invstd: TVM tensor
        Reciprocal of the variance of the saved forward input
    weight: TVM tensor
        parameters
    mean_dy: TVM tensor
        A part of sum_dy
    mean_dy_xmu: TVM tensor
        A part of sum_dy_xmu
    kernel_name: str
        cce kernel name, default value is sync_batch_norm_backward_elemt
    Returns
    -------
    None
    """
    dtype_lower_grad_output = grad_output["dtype"].lower()
    dtype_lower_save_input = save_input["dtype"].lower()
    dtype_lower_mean = mean["dtype"].lower()
    dtype_lower_invstd = invstd["dtype"].lower()
    dtype_lower_weight = weight["dtype"].lower()
    dtype_lower_mean_dy = mean_dy["dtype"].lower()
    dtype_lower_mean_dy_xmu = mean_dy_xmu["dtype"].lower()

    check_list = ("float16", "bfloat16", "float32")
    para_check.check_dtype(dtype_lower_grad_output, check_list, param_name="grad_output")
    grad_output["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_lower_save_input, check_list, param_name="save_input")
    save_input["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_lower_mean, check_list, param_name="mean")
    mean["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_lower_invstd, check_list, param_name="invstd")
    invstd["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_lower_weight, check_list, param_name="weight")
    weight["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_lower_mean_dy, check_list, param_name="mean_dy")
    mean_dy["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_lower_mean_dy_xmu, check_list, param_name="mean_dy_xmu")
    mean_dy_xmu["rel_pos_to_reduce"] = "before"

    tensors = []
    schedules = []
    ins = classify([grad_output, save_input, mean, invstd, weight, mean_dy, mean_dy_xmu], OpPatternMode.ELEWISE)

    for (_grad_output, _save_input, _mean, _invstd, _weight, _mean_dy, _mean_dy_xmu) in ins:
        with tbe.compute():
            shape_grad_output, shape_save_input, shape_mean, shape_invstd, \
            shape_weight, shape_mean_dy, shape_mean_dy_xmu = \
                shape_util.variable_shape([_grad_output, _save_input, _mean,
                                           _invstd, _weight, _mean_dy, _mean_dy_xmu])
            data_input_grad_output = tvm.placeholder(shape_grad_output, name="data_input_grad_output",
                                                     dtype=dtype_lower_grad_output)
            data_input_save_input = tvm.placeholder(shape_save_input, name="data_input_save_input",
                                                    dtype=dtype_lower_save_input)
            data_input_mean = tvm.placeholder(shape_mean, name="data_input_mean",
                                              dtype=dtype_lower_mean)
            data_input_invstd = tvm.placeholder(shape_invstd, name="data_input_invstd",
                                                dtype=dtype_lower_invstd)
            data_input_weight = tvm.placeholder(shape_weight, name="data_input_weight",
                                                dtype=dtype_lower_weight)
            data_input_mean_dy = tvm.placeholder(shape_mean_dy, name="data_input_mean_dy",
                                                 dtype=dtype_lower_mean_dy)
            data_input_mean_dy_xmu = tvm.placeholder(shape_mean_dy_xmu, name="data_input_mean_dy_xmu",
                                                     dtype=dtype_lower_mean_dy_xmu)
            res = sync_batch_norm_backward_elemt_compute(data_input_grad_output, data_input_save_input,
                                                         data_input_mean, data_input_invstd,
                                                         data_input_weight, data_input_mean_dy,
                                                         data_input_mean_dy_xmu, grad_input)
            tensors.append([data_input_grad_output, data_input_save_input, data_input_mean, data_input_invstd,
                            data_input_weight, data_input_mean_dy, data_input_mean_dy_xmu, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
