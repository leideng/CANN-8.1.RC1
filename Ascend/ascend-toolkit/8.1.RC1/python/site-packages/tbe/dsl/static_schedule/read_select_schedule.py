#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
read_select_schedule
"""
from functools import reduce as functools_reduce

from tbe import tvm
from tbe.common.platform import scope_ubuf
from tbe.common.platform import scope_cbuf_fusion
from tbe.common.platform.platform_info import get_soc_spec
from tbe.dsl.instrinsic.cce_intrin import get_bit_len


def _tilling_axis(valid_shape, input_dtype):
    ub_size_bytes = get_soc_spec("UB_SIZE") - 32
    dtype_bytes_size = get_bit_len(input_dtype) // 8

    total_ele = int(ub_size_bytes // dtype_bytes_size)
    split_axis = 0
    split_factor = 1

    for i, _ in enumerate(valid_shape):
        ele_cnt = int(functools_reduce(lambda x, y: x*y, valid_shape[i:]))
        if ele_cnt <= total_ele:
            split_axis = i - 1
            split_factor = total_ele // ele_cnt
            break
        if i == len(valid_shape) - 1:
            split_axis = i
            split_factor = total_ele
            break

    if split_axis < 0:
        split_axis = 0
        split_factor = valid_shape[0]

    return split_axis, split_factor


def _get_tensor_map(res, tensor_map):
    """
    get the compute tensors

    Parameters
    ----------
    res: the placeholder of result
    tensor_map: the compute tensors

    Returns
    -------
    None
    """
    if res is None:
        return
    stack = [res]
    visited_list = []
    while len(stack) > 0:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
                tensor_map[in_tensor.name] = in_tensor


def read_select_schedule(res, input_tensors):  # pylint: disable=locally-disabled,unused-argument
    """
    the schedule processes of read_select

    Parameters
    ----------
    res: the placeholder of result
    input_tensors: the placeholder of input

    Returns
    -------
    the result of schedule
    """
    tensor_map = {}
    _get_tensor_map(res, tensor_map)

    tensor_input = tensor_map.get("input_tensor")

    src_in_flag = "DDR"
    if "src_in_flag" in tensor_input.op.attrs:
        src_in_flag = tensor_input.op.attrs['src_in_flag']

    valid_shape = tensor_map.get("output_ub_5d").shape
    input_dtype = tensor_map.get("output_ub_5d").dtype

    sch = tvm.create_schedule(res.op)

    split_axis, split_factor = _tilling_axis(valid_shape, input_dtype)
    axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis], factor=split_factor)
    sch[tensor_map.get("output_ub_5d")].compute_at(sch[res], axis_outer)

    if src_in_flag == "L1":
        sch[tensor_input].set_scope(scope_cbuf_fusion)
    sch[tensor_map.get("output_ub_5d")].set_scope(scope_ubuf)
    sch[tensor_map.get("output_ub_5d")].emit_insn(
        tensor_map.get("output_ub_5d").op.axis[split_axis], 'dma_copy')
    sch[res].emit_insn(axis_inner, 'dma_copy')

    return sch
