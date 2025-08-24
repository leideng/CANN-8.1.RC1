#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

unsorted_segment_max_d
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
def check_supported(x, segment_ids, y, num_segments, kernel_name="unsorted_segment_max_d"):
    """
    fusion pass test if num_segments is int32
    num_segments should > 0 and x's first dim != segment_ids's first dim
    (num_segments + x's first dim) * BLOCK_LENGTH + (
        (BLOCK_LENGTH / 2 - x's first dim % (BLOCK_LENGTH / 4)) + x's first dim) * (BLOCK_LENGTH / 8)
    should <= UB_SIZE_MAX / 2
    """
    return False, "unsorted_segment_max_d not support"


# 'pylint: disable=unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("unsorted_segment_max_d")
def unsorted_segment_max_d_compute(x, segment_ids, y, num_segments, kernel_name="unsorted_segment_max_d"):
    """
    compute for unsorted_segment_max_d_compute
    """
    res = tbe.unsorted_segment_max(x, segment_ids, num_segments)
    return res


# 'pylint: disable =too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def unsorted_segment_max_d(x, segment_ids, y, num_segments, kernel_name="unsorted_segment_max_d"):
    """
    Operation and Schedule for unsorted_segment_max_d.

    Parameters
    ----------
    x: dict
        shape and dtype of input.
        dtype only support float16, float32, int32
        on Ascend310P, dtype also support int16

    segment_ids : dict
        should be the size of the first dimension
        need not cover all values in the full range of valid values
        dtype only support int32

    y: dict
        shape and dtype of output.

    num_segments : the dimension of the first axis of
                   the output tensor(>= max(segment_ids) + 1)

    kernel_name : cce kernel name,
                  default value is "unsorted_segment_max_d"

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

    segment_max_support = tbe_platform.api_check_support("te.lang.cce.unsorted_segment_max", "float32")
    if not segment_max_support:
        para_check.check_dtype(dtype, ("float16", "int32", "int16"), param_name="x")
    if num_segments <= 0:
        error_manager_vector.raise_err_check_params_rules(kernel_name, 'num_segments must greater than 0',
                                                          "num_segments", num_segments)

    first_shape = int(shape[0])
    ids_length = int(segment_ids_shape[0])
    if first_shape != ids_length:
        error_manager_vector.raise_err_specific_reson(
            kernel_name,
            "only supported x.shape[0] equals to segment_ids.shape[0],"
            " while x.shape[0] is %d, segment_ids.shape[0] is %d"
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
        res = unsorted_segment_max_d_compute(data_inputs, data_segments_id, y, num_segments, kernel_name)
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_inputs, data_segments_id, res]}
    tbe.cce_build_code(sch, config)
