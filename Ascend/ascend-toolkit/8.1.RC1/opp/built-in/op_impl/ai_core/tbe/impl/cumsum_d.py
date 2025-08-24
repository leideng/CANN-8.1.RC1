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
cumsum_d
"""

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl import cum_computer
from impl.util import util_select_op_base

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform

from impl.constant_util import STRIDE_ONE
from impl.constant_util import REPEAT_STRIDE_EIGHT


# 'pylint: disable=too-few-public-methods,redefined-outer-name,too-many-statements,too-many-locals,too-many-lines
def check_supported(x, y, axis=0, exclusive=False, reverse=False, kernel_name="cumsum_d"):
    """
    check whether cumsum_d is supported
    only Ascend310 and Ascend910 set needCheckSupport.flag=false
    """
    return False, ""


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    Constant
    """
    # the computer type
    SUM_TYPE = "sum"


# 'pylint: disable = unused-argument,too-many-arguments,invalid-name,consider-using-in
def get_op_support_info(x, y, axis=0, exclusive=False, reverse=False, kernel_name="cumsum_d"):
    """
    get_op_support_info
    """
    format_x = x.get("format")
    shape = x.get("shape")
    if axis < 0:
        axis = len(shape) + axis
    if format_x == "ND" or format_x == "NHWC":
        axis_split_list = []
        for i in range(0, axis):
            split_0 = [util_select_op_base.SplitInput([0, [i], [-1], [-1]]),
                       util_select_op_base.SplitOutput([0, [i]])]
            axis_split_list.append(split_0)
        axis_reduce_list = None
    else:
        axis_split_list = None
        axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_list, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=locally-disabled, unused-argument,invalid-name
# 'pylint: disable=locally-disabled, too-many-arguments, not-callable
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def cumsum_d(x, y, axis=0, exclusive=False, reverse=False,
             kernel_name="cumsum_d"):
    """
    Compute the cumulative sum of the input tensor along `axis`.

    Parameters
    ----------
    x: dict, shape and dtype, dtype must be in ('float16','float32','int32',
    'int8','uint8')
    y: the dict of output
    axis: a number of int32(default:0), cumulative axis, must be in the range
    [-rank(x), rank(x))
    exclusive: if `True`, perform exclusive cumsum
    reverse: a `bool` (default: False)
    kernel_name: kernel name

    Returns
    -------
    tik_instance: tik_instance

    """
    shape = x.get("shape")
    if axis < 0:
        axis = len(shape) + axis
    check_param(x, axis, kernel_name)
    
    tik_instance = special_case(x, axis, kernel_name)
    if tik_instance is not None:
        return tik_instance

    cumsum_template = cum_computer.get_computer_by_ctype(x, axis, kernel_name, Constant.SUM_TYPE)
    cumsum_template.set_ext_params(exclusive, reverse)

    return cumsum_template.get_tik_instance()


def check_param(input_x, axis, kernel_name):
    """
    check the parameters is valid, if one is invalid,then raise error

    Parameters
    ----------
    input_x: dict,shape and datatype
    axis: cumulative axis
    kernel_name: kernel_name
    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    para_check.check_shape(input_shape, param_name="input_x")
    para_check.check_dtype(input_dtype,
                           ("float16", "float32", "int32", "int8", "uint8"), param_name="input_x")

    if axis < len(input_shape) * (-1) or axis >= len(input_shape):
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, "axis", \
                                                                len(input_shape)*(-1), len(input_shape), axis)


def special_case(x, axis, kernel_name):
    dtype = x.get("dtype")
    shape = x.get("shape")
    block_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    cumsum_axis_align = block_num * 8 * 32
    max_shape_one = 64
    if not (dtype == "float32" and block_num == 32 and axis == 0 and len(shape) == 2
            and (shape[0] % cumsum_axis_align == 0) and (shape[1] < max_shape_one)):
        return None
    if shape[1] < 8:
        return special_case2(x, axis, kernel_name)
    
    tik_instance = tik.Tik()
    block_item_num = 8
    repeat_time = 32
    atomic_buf_item = block_item_num * repeat_time
    atomic_buf_size = shape[1] * atomic_buf_item
    items_per_block = shape[0] / block_num
    tail_idx = shape[1] - block_item_num
    x_item_size = shape[1] // block_item_num * block_item_num
    has_tail = shape[1] != x_item_size
    if x_item_size == 0:
        x_item_size = block_item_num
        has_tail = False
    x_item_burst_len = x_item_size // block_item_num
    x_item_size_tail = block_item_num
    x_item_mask = x_item_size % 64 if x_item_size != 64 else 64

    input_x_gm = tik_instance.Tensor(dtype, shape, name="input_x_gm", scope=tik.scope_gm)
    output_out_gm = tik_instance.Tensor(dtype, shape, name="output_out_gm", scope=tik.scope_gm)

    input_x_ub = tik_instance.Tensor(dtype, (x_item_size,), name="input_x_ub", scope=tik.scope_ubuf)
    out_ub = tik_instance.Tensor(dtype, (x_item_size + x_item_size_tail,), name="out_ub", scope=tik.scope_ubuf)
    input_x_ub_tail = tik_instance.Tensor(dtype, (x_item_size_tail,), name="input_x_ub_tail", scope=tik.scope_ubuf)
    out_ub_tail = tik_instance.Tensor(dtype, (x_item_size_tail,), name="out_ub_tail", scope=tik.scope_ubuf)

    out_ub_x = tik_instance.Tensor(dtype, (atomic_buf_size,), name="out_ub_x", scope=tik.scope_ubuf)
    sync_workspace = tik_instance.Tensor("int64", (block_num * 32 // 8,), name="barrier_workspace",
                                         scope=tik.scope_gm, is_workspace=True, is_atomic_add=True)
    
    with tik_instance.for_range(0, block_num, block_num=block_num) as block_i:
        start = items_per_block * block_i
        tik_instance.data_move(out_ub, input_x_gm[start, 0], 0, 1, x_item_burst_len, 0, 0)
        tik_instance.data_move(output_out_gm[start, 0], out_ub, 0, 1, x_item_burst_len, 0, 0)
        if has_tail:
            tik_instance.data_move(out_ub_tail, input_x_gm[start, tail_idx], 0, 1, 1, 0, 0)
            tik_instance.data_move(output_out_gm[start, tail_idx], out_ub_tail, 0, 1, 1, 0, 0)
        with tik_instance.for_range(1, items_per_block) as i:
            tik_instance.data_move(input_x_ub, input_x_gm[start+i, 0], 0, 1, x_item_burst_len, 0, 0)
            tik_instance.vadd(x_item_mask, out_ub, input_x_ub, out_ub, 1, STRIDE_ONE, STRIDE_ONE,
                              STRIDE_ONE, REPEAT_STRIDE_EIGHT,
                              REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT)
            tik_instance.data_move(output_out_gm[start+i, 0], out_ub, 0, 1, x_item_burst_len, 0, 0)
            if has_tail:
                tik_instance.data_move(input_x_ub_tail, input_x_gm[start+i, tail_idx], 0, 1, 1, 0, 0)
                tik_instance.vadd(block_item_num, out_ub_tail, input_x_ub_tail, out_ub_tail, 1, STRIDE_ONE, STRIDE_ONE,
                                  STRIDE_ONE, REPEAT_STRIDE_EIGHT,
                                  REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT)
                tik_instance.data_move(output_out_gm[start+i, tail_idx], out_ub_tail, 0, 1, 1, 0, 0)
    
        # up to shape[1]
        if has_tail:
            with tik_instance.for_range(tail_idx, shape[1]) as t_idx:
                out_ub[t_idx].set_as(out_ub_tail[t_idx-tail_idx])
        
        # dump to repeat
        with tik_instance.for_range(0, atomic_buf_size) as t_idx:
            out_ub_x[t_idx].set_as(out_ub[t_idx % shape[1]])

        # atomic add
        tik_instance.block_barrier(sync_workspace)
        tik_instance.set_atomic_add(1)
        with tik_instance.for_range(0, (shape[0] - start - items_per_block) / atomic_buf_item) as t_idx:
            tik_instance.data_move(output_out_gm[start + items_per_block + t_idx * atomic_buf_item, 0],
                                   out_ub_x, 0, 1, atomic_buf_size // block_item_num, 0, 0)
        tik_instance.set_atomic_add(0)
    
    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(input_x_gm,), outputs=(output_out_gm, ), enable_l2=False)
    return tik_instance
    

def special_case2(x, axis, kernel_name):
    dtype = x.get("dtype")
    shape = x.get("shape")
    block_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)    
    tik_instance = tik.Tik()
    block_item_num = 8
    repeat_time = 32
    atomic_buf_item = block_item_num * repeat_time
    atomic_buf_size = shape[1] * atomic_buf_item
    items_per_block = shape[0] // block_num

    x_item_size = block_item_num
    x_item_burst_len = x_item_size // block_item_num
    x_item_mask = shape[1]

    input_x_gm = tik_instance.Tensor(dtype, shape, name="input_x_gm", scope=tik.scope_gm)
    output_out_gm = tik_instance.Tensor(dtype, shape, name="output_out_gm", scope=tik.scope_gm)

    input_x_ub = tik_instance.Tensor(dtype, (x_item_size,), name="input_x_ub", scope=tik.scope_ubuf)
    out_ub = tik_instance.Tensor(dtype, (x_item_size,), name="out_ub", scope=tik.scope_ubuf)

    out_ub_x = tik_instance.Tensor(dtype, (atomic_buf_size,), name="out_ub_x", scope=tik.scope_ubuf)
    sync_workspace = tik_instance.Tensor("int64", (block_num * 32 // 8,), name="barrier_workspace",
                                         scope=tik.scope_gm, is_workspace=True, is_atomic_add=True)
    
    with tik_instance.for_range(0, block_num, block_num=block_num) as block_i:
        start = items_per_block * block_i
        tik_instance.data_move(out_ub, input_x_gm[start, 0], 0, 1, x_item_burst_len, 0, 0)
        tik_instance.data_move(output_out_gm[start, 0], out_ub, 0, 1, x_item_burst_len, 0, 0)
        with tik_instance.for_range(1, items_per_block-8) as i:
            tik_instance.data_move(input_x_ub, input_x_gm[start+i, 0], 0, 1, x_item_burst_len, 0, 0)
            tik_instance.vadd(x_item_mask, out_ub, input_x_ub, out_ub, 1, STRIDE_ONE, STRIDE_ONE,
                              STRIDE_ONE, REPEAT_STRIDE_EIGHT,
                              REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT)
            tik_instance.data_move(output_out_gm[start+i, 0], out_ub, 0, 1, x_item_burst_len, 0, 0)
        # last 8
        idx = tik_instance.Scalar("int32")
        with tik_instance.for_range(items_per_block-8, items_per_block) as i:
            tik_instance.data_move(input_x_ub, input_x_gm[start+i, 0], 0, 1, x_item_burst_len, 0, 0)
            tik_instance.vadd(x_item_mask, out_ub, input_x_ub, out_ub, 1, STRIDE_ONE, STRIDE_ONE,
                              STRIDE_ONE, REPEAT_STRIDE_EIGHT,
                              REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT)
            with tik_instance.for_range(0, shape[1]) as sum_i:
                out_ub_x[idx].set_as(out_ub[sum_i])
                idx.set_as(idx + 1)
        tik_instance.data_move(output_out_gm[start + items_per_block - 8, 0], out_ub_x, 0, 1, shape[1], 0, 0)
        
        # dump to repeat
        with tik_instance.for_range(0, atomic_buf_size) as t_idx:
            out_ub_x[t_idx].set_as(out_ub[t_idx % shape[1]])

        # atomic add
        tik_instance.block_barrier(sync_workspace)
        tik_instance.set_atomic_add(1)
        with tik_instance.for_range(0, (shape[0] - start - items_per_block) / atomic_buf_item) as t_idx:
            tik_instance.data_move(output_out_gm[start + items_per_block + t_idx * atomic_buf_item, 0],
                                   out_ub_x, 0, 1, atomic_buf_size // block_item_num, 0, 0)
        tik_instance.set_atomic_add(0)
    
    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(input_x_gm,), outputs=(output_out_gm, ), enable_l2=False)
    return tik_instance
