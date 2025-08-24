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
dynamic sync_batch_norm_gather_stats_with_counts
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import SyncBatchNormGatherStatsWithCountsAttrInfo
from impl.util.util_attr_common import get_attr_by_cls


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals,too-many-statements
# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("SyncBatchNormGatherStatsWithCounts", op_mode="dynamic", 
                           support_fusion=True, support_bfp16=True)
def sync_batch_norm_gather_stats_with_counts_compute(mean_all,
                                                     invert_std_all,
                                                     count_all,
                                                     mean_broadcast,
                                                     count_sum,
                                                     running_var,
                                                     invert_std,
                                                     running_var_update,
                                                     axes,
                                                     momentum,
                                                     epsilon,
                                                     kernel_name="sync_batch_norm_gather_stats_with_counts",
                                                     impl_mode=OpImplMode.HIGH_PERFORMANCE,
                                                     is_5hdc=False):
    """sync_batch_norm_gather_stats_with_counts compute

    Parameters:
    ----------
    mean_all : TVM Tensor
        the mean of each device
    invert_std_all : TVM Tensor
        reciprocal of the variances of each device
    count_all : TVM Tensor
        number of data for each device
    mean_broadcast : TVM Tensor
        the overall average and broadcast
    count_sum : TVM Tensor
        general statistics
    running_var : TVM Tensor
        runtime variance
    invert_std : TVM Tensor
        reciprocal of total variance
    running_var_update : TVM Tensor
        updated runtime variance
    momentum : float
        the update step length
    epsilon : float
        anti-zero parameter
    kernel_name : str
        cce kernel name, default value is sync_batch_norm_gather_stats_with_counts

    Returns
    -------
    res: TVM tensor
        output tensor.
    """

    dtype = mean_all.dtype
    sum_support_fp32 = tbe_platform.api_check_support("te.lang.cce.sum", "float32")
    vdiv_support_fp32 = tbe_platform.api_check_support("te.lang.cce.vdiv", "float32")
    if dtype == "float32":
        calc_dtype = "float32"
    elif dtype == "float16":
        cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        if not sum_support_fp32 and not vdiv_support_fp32:
            calc_dtype = "float16"
        elif cce_product == "Ascend310" and impl_mode == OpImplMode.HIGH_PERFORMANCE:
            calc_dtype = "float16"
        else:
            calc_dtype = "float32"
    else:
        # int8 and uint8
        calc_dtype = "float16"

    if dtype != calc_dtype:
        data_mean_all_tmp = tbe.cast_to(mean_all, calc_dtype)
        data_invert_std_all_tmp = tbe.cast_to(invert_std_all, calc_dtype)
        data_count_all_tmp = tbe.cast_to(count_all, calc_dtype)
        data_mean_broadcast_tmp = tbe.cast_to(mean_broadcast, calc_dtype)
        data_count_sum_tmp = tbe.cast_to(count_sum, calc_dtype)
        data_running_var_tmp = tbe.cast_to(running_var, calc_dtype)
    else:
        data_mean_all_tmp = mean_all
        data_invert_std_all_tmp = invert_std_all
        data_count_all_tmp = count_all
        data_mean_broadcast_tmp = mean_broadcast
        data_count_sum_tmp = count_sum
        data_running_var_tmp = running_var

    var_all = tbe.vrec(data_invert_std_all_tmp, "high_precision")
    var_all_square = tbe.vmul(var_all, var_all)
    epsilon_scalar = get_attr_by_cls(epsilon,
                                     SyncBatchNormGatherStatsWithCountsAttrInfo.ATTR_EPSILON,
                                     calc_dtype)
    var_all_square_epsilon = tbe.vadds(var_all_square, -epsilon_scalar)
    mean_sub = tbe.vsub(data_mean_all_tmp, data_mean_broadcast_tmp)
    mean_var = tbe.vmul(mean_sub, mean_sub)
    mean_var_sum = tbe.vadd(var_all_square_epsilon, mean_var)
    mean_var_count = tbe.vmul(mean_var_sum, data_count_all_tmp)
    var_sum = tbe.reduce_sum(mean_var_count, axis=axes, keepdims=True)
    var_sum_count = tbe.vdiv(var_sum, data_count_sum_tmp)
    var_sum_count_epsilon = tbe.vadds(var_sum_count, epsilon_scalar)
    var_sqrt = tbe.vsqrt(var_sum_count_epsilon)
    invert_std = tbe.vrec(var_sqrt, "high_precision")
    count_sum_one = tbe.vadds(data_count_sum_tmp, tvm.const(-1, dtype=calc_dtype))
    unbiased_var = tbe.vdiv(var_sum, count_sum_one)
    momentum_scalar = get_attr_by_cls(momentum,
                                      SyncBatchNormGatherStatsWithCountsAttrInfo.ATTR_MOMENTUM,
                                      calc_dtype)
    running_var_tmp = tbe.vmuls(data_running_var_tmp, 1 - momentum_scalar)
    running_var_update_tmp = tbe.vmuls(unbiased_var, momentum_scalar)
    running_var_update = tbe.vadd(running_var_tmp, running_var_update_tmp)

    if dtype != calc_dtype:
        invert_std = tbe.cast_to(invert_std, dtype)
        running_var_update = tbe.cast_to(running_var_update, dtype)

    return [invert_std, running_var_update]


@register_operator("SyncBatchNormGatherStatsWithCounts")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals,too-many-statements
# 'pylint: disable=unused-argument,invalid-name
def sync_batch_norm_gather_stats_with_counts(mean_all,
                                             invert_std_all,
                                             count_all,
                                             mean_broadcast,
                                             count_sum,
                                             running_var,
                                             invert_std,
                                             running_var_update,
                                             momentum=0.01,
                                             epsilon=0.00001,
                                             kernel_name="sync_batch_norm_gather_stats_with_counts",
                                             impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    Reduce a tensor on a certa in axes based on mean.

    Parameters:
    ----------
    mean_all : dict
        the mean of each device
    invert_std_all : dict
        reciprocal of the variances of each device
    count_all : dict
        number of data for each device
    mean_broadcast : dict
        the overall average and broadcast
    count_sum : dict
        general statistics
    running_var : dict
        runtime variance
    invert_std : dict
        reciprocal of total variance
    running_var_update : dict
        updated runtime variance
    momentum : float
        the update step length
    epsilon : float
        anti-zero parameter
    kernel_name : str
        cce kernel name, default value is sync_batch_norm_gather_stats_with_counts

    Returns
    -------
    None
    """
    dtype_mean_all = mean_all["dtype"]
    dtype_mean_all_lower = dtype_mean_all.lower()
    dtype_invert_std_all = invert_std_all["dtype"]
    dtype_invert_std_all_lower = dtype_invert_std_all.lower()
    dtype_count_all = count_all["dtype"]
    dtype_count_all_lower = dtype_count_all.lower()
    dtype_mean_broadcast = mean_broadcast["dtype"]
    dtype_mean_broadcast_lower = dtype_mean_broadcast.lower()
    dtype_count_sum = count_sum["dtype"]
    dtype_count_sum_lower = dtype_count_sum.lower()
    dtype_running_var = running_var["dtype"]
    dtype_running_var_lower = dtype_running_var.lower()

    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_mean_all_lower, check_list)
    mean_all["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_invert_std_all_lower, check_list)
    invert_std_all["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_count_all_lower, check_list)
    count_all["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_mean_broadcast_lower, check_list)
    mean_broadcast["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_count_sum_lower, check_list)
    count_sum["rel_pos_to_reduce"] = "after"
    para_check.check_dtype(dtype_running_var_lower, check_list)
    running_var["rel_pos_to_reduce"] = "after"

    shape = mean_all["shape"]
    shape_len = len(shape)
    axes = [0, ]
    axes = shape_util.axis_check(shape_len, axes)
    input_axis = {"shape": [len(axes), ], "value": axes, "rel_pos_to_reduce": "axis"}
    keepdims = True

    schedules = []
    tensors = []
    ins = classify([mean_all, invert_std_all, count_all, mean_broadcast, count_sum, running_var, input_axis],
                   OpPatternMode.REDUCE, {"keepdims": keepdims is True})
    for (_mean_all, _invert_std_all, _count_all, _mean_broadcast, _count_sum, _running_var, _axes) in ins:
        with tbe.compute():
            # not support 5HD
            is_5hdc = False
            [shape_mean_all, shape_invert_std_all, shape_count_all, shape_mean_broadcast, shape_count_sum,
             shape_running_var] = shape_util.variable_shape([_mean_all, _invert_std_all, _count_all, _mean_broadcast,
                                                             _count_sum, _running_var, _axes], op_mode="reduce")[:6]
            data_mean_all = tvm.placeholder(shape_mean_all, name="data_mean_all", dtype=dtype_mean_all_lower)
            data_invert_std_all = tvm.placeholder(shape_invert_std_all,
                                                  name="data_invert_std_all",
                                                  dtype=dtype_invert_std_all_lower)
            data_count_all = tvm.placeholder(shape_count_all, name="data_count_all", dtype=dtype_count_all_lower)
            data_mean_broadcast = tvm.placeholder(shape_mean_broadcast,
                                                  name="data_mean_broadcast",
                                                  dtype=dtype_mean_broadcast_lower)
            data_count_sum = tvm.placeholder(shape_count_sum, name="data_count_sum", dtype=dtype_count_sum_lower)
            data_running_var = tvm.placeholder(shape_running_var,
                                               name="data_running_var",
                                               dtype=dtype_running_var_lower)
            res = sync_batch_norm_gather_stats_with_counts_compute(data_mean_all, data_invert_std_all, data_count_all,
                                                                   data_mean_broadcast, data_count_sum,
                                                                   data_running_var, invert_std, running_var_update,
                                                                   _axes.get("value"), momentum, epsilon,
                                                                   impl_mode=impl_mode, is_5hdc=is_5hdc)
            tensors.append([data_mean_all, data_invert_std_all, data_count_all, data_mean_broadcast, data_count_sum,
                            data_running_var] + res)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
