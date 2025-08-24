#!/usr/bin/python
# -*- coding: utf-8 -*-
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
unsorted_segment_min_d
"""
import functools

import te.platform as tbe_platform
from tbe import tvm
from te.lang import cce as tbe
from te.utils import para_check
from te.utils.error_manager import error_manager_vector

DYNAMIC_UNRANK = [-2]
# block length in number
BLOCK_LENGTH = 32
# max ub size
UB_SIZE_MAX = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)


# 'pylint: disable=unused-argument,invalid-name
def check_supported(x, segment_ids, y, num_segments, kernel_name="unsorted_segment_min_d"):
    """
    fusion pass test if num_segments is int32
    num_segments should > 0 and x's first dim != segment_ids's first dim
    (num_segments + x's first dim) * BLOCK_LENGTH + (
        (BLOCK_LENGTH / 2 - x's first dim % (BLOCK_LENGTH / 4)) + x's first dim) * (BLOCK_LENGTH / 8)
    should <= UB_SIZE_MAX / 2
    """
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        shape = x.get("ori_shape")
        dtype = x.get("dtype").lower()
        segment_ids_shape = segment_ids.get("ori_shape")
        segment_ids_dtype = segment_ids.get("dtype").lower()
        check_list = ("float16", "float32", "int32", "int16")
        para_check.check_dtype(dtype, check_list, param_name="x")
        check_list_ids = ("int32")
        para_check.check_dtype(segment_ids_dtype, check_list_ids, param_name="segment_ids")
        if list(shape) == DYNAMIC_UNRANK or list(segment_ids_shape) == DYNAMIC_UNRANK:
            return True, ""
        para_check.check_shape(shape, param_name="x")
        para_check.check_shape(segment_ids_shape, param_name="segment_ids")
        if num_segments <= 0:
            error_manager_vector.raise_err_check_params_rules(kernel_name, 'num_segments must greater than 0',
                                                              "num_segments", num_segments)
            return False, "num_segments should > 0"
        first_shape = int(shape[0])
        ids_length = int(segment_ids_shape[0])
        if first_shape != ids_length:
            error_manager_vector.raise_err_specific_reson(
                kernel_name,
                "only supported x.shape[0] equals to segment_ids.shape[0],"
                " while x.shape[0] is %d, segment_ids.shape[0] is %d"
                % (first_shape, ids_length))
            return False, "x's first dim != segment_ids's first dim"
        total_ub_size = (num_segments + first_shape) * BLOCK_LENGTH + (
            (BLOCK_LENGTH // 2 - first_shape % (BLOCK_LENGTH // 4)) + first_shape) * (BLOCK_LENGTH // 8)
        # not supported in aicore now
        if total_ub_size > UB_SIZE_MAX // 2:
            reason = "total_ub_size is bigger than (UB_SIZE_MAX // 2), not supported in aicore now, "\
                     "total_ub_size:%s, UB_SIZE_MAX:%s" % (total_ub_size, UB_SIZE_MAX)
            return False, reason
        return True, ""
    else:
        return False, "unsorted_segment_min_d is not support"


# 'pylint: disable=unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("unsorted_segment_min_d")
def unsorted_segment_min_d_compute(x, segment_ids, y, num_segments, kernel_name="unsorted_segment_min_d"):
    """
    compute for unsorted_segment_min_d_compute
    """
    res = tbe.unsorted_segment_min(x, segment_ids, num_segments)
    return res


# 'pylint: disable =too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def unsorted_segment_min_d(x, segment_ids, y, num_segments, kernel_name="unsorted_segment_min_d"):
    """
    Operation and Schedule for unsorted_segment_min_d.

    Parameters
    ----------
    x: dict
        shape and dtype of input.
        dtype only support float16, float32, int32

    segment_ids : dict
        should be the size of the first dimension
        need not cover all values in the full range of valid values
        dtype only support int32

    y: dict
        shape and dtype of output.

    num_segments : the dimension of the first axis of
                   the output tensor(>= max(segment_ids) + 1)

    kernel_name : cce kernel name,
                  default value is "unsorted_segment_min_d"

    Returns
    -------
        None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    segment_ids_shape = segment_ids.get("shape")
    segment_ids_dtype = segment_ids.get("dtype")

    para_check.check_shape(shape, param_name="x")
    para_check.check_shape(segment_ids_shape, param_name="segment_ids")

    check_list = ("float16", "float32", "int32", "int16")
    para_check.check_dtype(dtype, check_list, param_name="x")
    check_list_ids = ("int32", )
    para_check.check_dtype(segment_ids_dtype, check_list_ids, param_name="segment_ids")
    min_support = tbe_platform.api_check_support("te.lang.cce.unsorted_segment_min", "float32")
    if not min_support:
        new_check_list = list(check_list)
        new_check_list.remove('float16')
        para_check.check_dtype(dtype, new_check_list, param_name="x")
    if num_segments <= 0:
        error_manager_vector.raise_err_check_params_rules(kernel_name, 'num_segments must greater than 0',
                                                          "num_segments", num_segments)

    first_shape = int(shape[0])
    ids_length = int(segment_ids_shape[0])
    if first_shape != ids_length:
        error_manager_vector.raise_err_specific_reson(
            kernel_name,
            "only supported x.shape[0] equals to segment_ids.shape[0], while x.shape[0] is %d,"
            " segment_ids.shape[0] is %d"
            % (first_shape, ids_length))

    total_ub_size = (num_segments + first_shape) * BLOCK_LENGTH + (
        (BLOCK_LENGTH // 2 - first_shape % (BLOCK_LENGTH // 4)) + first_shape) * (BLOCK_LENGTH // 8)
    if total_ub_size > UB_SIZE_MAX // 2:
        error_manager_vector.raise_err_specific_reson(
            kernel_name, "the memory usage is greater than UB_SIZE_MAX when num_segments=%d and shape[0]=%d" %
            (num_segments, shape[0]))

    dtype = dtype.lower()
    if len(shape) != 1:
        shape = (first_shape, functools.reduce(lambda x, y: x * y, shape[1:]))
    data_inputs = tvm.placeholder(shape, name="data_inputs", dtype=dtype)
    data_segments_id = tvm.placeholder(segment_ids_shape, name="data_segments_id", dtype=segment_ids_dtype)
    with tvm.target.cce():
        res = unsorted_segment_min_d_compute(data_inputs, data_segments_id, y, num_segments, kernel_name)
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_inputs, data_segments_id, res]}
    tbe.cce_build_code(sch, config)
