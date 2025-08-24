#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#tbe_platform.get_bit_len
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
strided_slice_assign_d
"""
import copy
import functools
import math

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import build_config
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


class Constant:
    """
    constant of limitation for blockdim
    """
    MAX_BLOCKDIM = 65535


def _check_num_is_power_of_two(num):
    """
    check the num is power of two
    """
    return num & (num - 1) == 0


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-statements
def _check_parameter(input_shape, begin, end, strides, ellipsis_mask, new_axis_mask, shrink_axis_mask):
    """
    check the input parameters
    """
    if not _check_num_is_power_of_two(ellipsis_mask):
        error_manager_vector.raise_err_specific_reson("strided_slice_assign_d",
                                                      "Multiple ellipses in slice spec not allowed!")
    if len(end) != len(begin) or len(begin) != len(strides):
        error_manager_vector.raise_err_specific_reson("strided_slice_assign_d",
                                                      "end shape, begin shape and strides shape length must be equal!")

    for i, _ in enumerate(begin):
        if strides[i] <= 0:
            error_manager_vector.raise_err_specific_reson("strided_slice_assign_d",
                                                          "strides[%s] must be greater than 0 " % strides)
        if begin[i] < 0:
            begin[i] = begin[i] + input_shape[i]
            if begin[i] < 0:
                begin[i] = 0
        if end[i] < 0:
            end[i] = end[i] + input_shape[i]
            if end[i] < 0:
                error_manager_vector.raise_err_specific_reson("strided_slice_assign_d",
                                                              "end shape is invaild value in dim")
        if begin[i] >= input_shape[i]:
            error_manager_vector.raise_err_specific_reson(
                "strided_slice_assign_d",
                "begin[%s] shouldn't greater than or equal to input[%s] shape in dim" % (begin, input_shape))
        if begin[i] >= end[i]:
            error_manager_vector.raise_err_specific_reson("strided_slice_assign_d", "begin must be less than end")
        if end[i] > input_shape[i]:
            end[i] = input_shape[i]


def _make_up_slice_params(input_shape, begin, end, strides):
    """
    if input params begin, end and strides len less than input_shape dims.
    we should make up begin, end and strides according to input_shape
    """
    for index, element in enumerate(input_shape):
        if index >= len(begin):
            end.append(element)
            begin.append(0)
            strides.append(1)


def _update_params_by_new_axis_mask(params, new_axis_mask):
    """
    update the params according to new_axis_mask.
    """
    temp_params = copy.deepcopy(params)
    params.clear()
    for i, element in enumerate(temp_params):
        if (new_axis_mask & 2**i) != 2**i:
            params.append(element)


def _update_begin_end_by_begin_end_mask(input_shape, begin, end, begin_mask, end_mask):
    """
    init the begin and end parameters according to begin_mask and end_mask .
    """
    for i, (_, _, input_shape_value) in enumerate(zip(begin, end, input_shape)):
        if (begin_mask & 2**i) == 2**i:
            begin[i] = 0
        if (end_mask & 2**i) == 2**i:
            end[i] = input_shape_value


def _update_begin_end_strides_by_ellipsis_mask(input_shape, begin, end, strides, ellipsis_mask):
    """
    init the begin,end and strides parameters according to ellipsis_mask.
    """
    for i, (_, _, _, input_shape_value) in enumerate(zip(begin, end, strides, input_shape)):
        if (ellipsis_mask & 2**i) == 2**i:
            begin[i] = 0
            end[i] = input_shape_value
            strides[i] = 1


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-statements
def _update_slice_params(input_shape, begin, end, strides, new_axis_mask, shrink_axis_mask):
    """
    1. update the begin,end and strides parameters according to new_axis_mask.
    2. update the end parameters according to shrink_axis_mask.
    """
    slice_shape = []
    temp_input_shape = list(input_shape)
    for i, _ in enumerate(input_shape):
        if (new_axis_mask & 2**i) == 2**i:
            temp_input_shape.insert(i, 1)
            begin.append(0)
            strides.append(1)

    for i, element in enumerate(temp_input_shape):
        if i >= len(input_shape):
            end.append(element)

    for i, element in enumerate(begin):
        if (shrink_axis_mask & 2**i) == 2**i:
            end[i] = element + 1

    for i, (begin_value, end_value, stride_value,
            temp_input_value) in enumerate(zip(begin, end, strides, temp_input_shape)):
        if end_value > temp_input_value:
            end_value = temp_input_value
            end[i] = temp_input_value
        if begin_value >= temp_input_value and (new_axis_mask & 2**i) != 2**i:
            error_manager_vector.raise_err_specific_reson(
                "strided_slice_assign_d", "begin shouldn't greater than or equal to input shape in dim")
        slice_shape.append(int(math.ceil((end_value - begin_value) / (stride_value * 1.0))))

    _update_params_by_new_axis_mask(slice_shape, new_axis_mask)
    _update_params_by_new_axis_mask(begin, new_axis_mask)
    _update_params_by_new_axis_mask(end, new_axis_mask)
    _update_params_by_new_axis_mask(strides, new_axis_mask)

    # if 'i'th dimension value in slice_shape is 1 equals to
    # strides[i]=1 && end[i] = begin[i] + 1
    for i, element in enumerate(slice_shape):
        if element == 1:
            strides[i] = 1
            end[i] = begin[i] + 1

    if strides[-1] != 1:
        error_manager_vector.raise_err_specific_reson("strided_slice_assign_d",
                                                      "Only support strides with 1 at last value.")

    return slice_shape


def _tilling_axis(input_value_shape, input_ref_shape, dtype):
    """
    split axis and return split_factor
    """
    ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // 4

    # Convert byts to Bytes
    dtype_bytes_size = tbe_platform.get_bit_len(dtype) // 8

    # 32 means one block size(32 Bytes), divide by 32 to get the numbers of data
    # that can be stored in one block.
    one_block_bytes_size = tbe_platform.VECTOR_INST_BLOCK_WIDTH // tbe_platform.VECTOR_INST_BLOCK_NUM
    flag = one_block_bytes_size // dtype_bytes_size
    element_new = math.ceil(input_value_shape[-1] / flag) * flag
    shape_new = []
    for i in input_value_shape:
        shape_new.append(i)
    shape_new[-1] = int(element_new)

    # gm->ub maximum copy data at a time
    total_ele = ub_size_bytes // dtype_bytes_size

    split_axis = 0
    split_factor = 1
    not_storage_align = False
    for i, _ in enumerate(shape_new):
        ele_cnt = functools.reduce(lambda x, y: x * y, shape_new[i:])
        if ele_cnt <= total_ele:
            split_axis = i - 1
            split_factor = total_ele // ele_cnt
            break

    if shape_new[-1] > total_ele:
        split_axis = len(shape_new) - 1
        split_factor = total_ele

    if split_axis < 0:
        if len(input_value_shape) == 1:
            not_storage_align = True
        split_axis = 0
        split_factor = shape_new[0]

    split_factor_1 = math.ceil(input_ref_shape[split_axis] // split_factor / Constant.MAX_BLOCKDIM)
    split_factor_list = [split_factor, split_factor_1]
    
    return split_axis, split_factor_list, not_storage_align


# 'pylint: disable=too-many-statements,unnecessary-lambda
@register_operator_compute("strided_slice_assign_d", op_mode="static", support_fusion=True)
def strided_slice_assign_compute(data_ref, data_value, slice_params):
    """
    strided_slice_assign compute function

    Parameters
    ----------
    data_ref : tvm.tensor
        tensor of ref
    data_value : tvm.tensor
        tensor of input value
    slice_params : list
        include begin, end and strides params

    Returns
    -------
    Computational process for TVM compute.
    """
    ref_shape = data_ref.shape
    ref_shape = [i.value for i in ref_shape]

    input_value_shape = data_value.shape
    input_value_shape = [i.value for i in input_value_shape]

    begin = slice_params[0]
    end = slice_params[1]
    strides = slice_params[2]

    def _map_input_value_index(*index):
        input_value_index = None
        for i, _ in enumerate(begin):
            if input_value_index is None:
                input_value_index = ((index[i] - begin[i]) // strides[i], )
            else:
                input_value_index = input_value_index + ((index[i] - begin[i]) // strides[i], )
        return input_value_index

    input_value_ub = tvm.compute(input_value_shape, lambda *index: data_value(*index), name="input_value_ub")

    def _copy_from_input_value(index, shape):
        for idx, _ in enumerate(shape):
            i = len(shape) - idx - 1
            if idx == 0:
                select_result = tvm.select(begin[i] <= index[i], input_value_ub(*_map_input_value_index(*index)))
                select_result = tvm.select((index[i] - begin[i]) % strides[i] == 0, select_result)
                select_result = tvm.select(end[i] > index[i], select_result)
            else:
                select_result = tvm.select(begin[i] <= index[i], select_result)
                select_result = tvm.select((index[i] - begin[i]) % strides[i] == 0, select_result)
                select_result = tvm.select(end[i] > index[i], select_result)
        return select_result

    out = tvm.compute(ref_shape, lambda *i: _copy_from_input_value(i, ref_shape), name="res")

    return input_value_ub, out


# 'pylint: disable=too-many-locals
def _strided_slice_assign_schedule(schedule_list, out, input_value_shape, input_ref_shape, data_dtype):
    """
    strided_slice_assign schedule function

    Parameters
    ----------
    schedule_list : tuple
        include tvm.tensor of ref  and tvm.tensor of input_value in ub
    out : tvm.tensor
        tvm.tensor of out
    input_value_shape : list
        input_value shape
    input_shape : list
        ref shape
    data_dtype : str
        input data dtype

    Returns
    -------
    sch : tvm.schedule
        the compute schedule
    """
    dtype_bytes_size = tbe_platform.get_bit_len(data_dtype) // 8
    # 32 means one block size(32 Bytes), divide by 32 to get the numbers of data
    # that can be stored in one block.
    one_block_bytes_size = tbe_platform.VECTOR_INST_BLOCK_WIDTH // tbe_platform.VECTOR_INST_BLOCK_NUM
    element = one_block_bytes_size // dtype_bytes_size
    input_value_ub = schedule_list[1]

    sch = tvm.create_schedule(out.op)

    sch[input_value_ub].set_scope(tbe_platform.scope_ubuf)

    split_axis, split_factor_list, not_storage_align = \
        _tilling_axis(input_value_shape, input_ref_shape, dtype=data_dtype)

    split_factor, split_factor_1 = split_factor_list[0], split_factor_list[1]

    if not_storage_align:
        split_factor = input_ref_shape[0]
    
    axis_outer, axis_inner = sch[out].split(out.op.axis[split_axis], factor=split_factor)
    if split_factor_1 > 1:
        axis_outer, _ = sch[out].split(axis_outer, factor=split_factor_1)

    # multi core
    device_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    if split_axis == 0:
        init_core_num = (input_value_shape[0] + split_factor - 1) // split_factor
        if init_core_num > device_core_num:
            forward_axis_outer, forward_axis_inner = sch[out].split(axis_outer, nparts=device_core_num)
            sch[out].bind(forward_axis_outer, tvm.thread_axis('blockIdx.x'))
            compute_at_axis = forward_axis_inner
        else:
            sch[out].bind(axis_outer, tvm.thread_axis('blockIdx.x'))
            compute_at_axis = axis_outer
    else:
        all_move_count = input_value_shape[0]
        if all_move_count > device_core_num:
            fused_axis_outer, _ = sch[out].split(sch[out].op.axis[0], nparts=device_core_num)
            sch[out].bind(fused_axis_outer, tvm.thread_axis('blockIdx.x'))
        else:
            sch[out].bind(sch[out].op.axis[0], tvm.thread_axis('blockIdx.x'))
        compute_at_axis = axis_outer

    # compute_at
    sch[input_value_ub].compute_at(sch[out], compute_at_axis)

    # storage_align
    if not_storage_align:
        pass
    elif len(input_value_shape) == 1:
        sch[input_value_ub].storage_align(compute_at_axis, element, 0)
    else:
        sch[input_value_ub].storage_align(input_value_ub.op.axis[-2], element, 0)

    # emit insn
    sch[input_value_ub].emit_insn(input_value_ub.op.axis[split_axis], tbe_platform.DMA_COPY)
    sch[out].emit_insn(axis_inner, tbe_platform.DMA_COPY, {"no_overlap": "default"})
    sch[input_value_ub].double_buffer()

    return sch


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def strided_slice_assign_d(ref_dict,
                           input_value_dict,
                           output_ref_dict,
                           begin,
                           end,
                           strides=None,
                           begin_mask=0,
                           end_mask=0,
                           ellipsis_mask=0,
                           new_axis_mask=0,
                           shrink_axis_mask=0,
                           kernel_name="strided_slice_assign_d"):
    """
    Assign `value` to the sliced l-value reference of `ref`.

    The values of `value` are assigned to the positions in the variable
    `ref` that are selected by the slice parameters. The slice parameters
   `begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.

    NOTE this op currently does not support broadcasting and so `value`'s
    shape must be exactly the shape produced by the slice of `ref`.

    Parameters
    ----------
    ref_dict: dict.
        shape, dtype of ref.
    input_value_dict: dict
        shape, dtype of value
    output_ref_dict:dict
        shape, dtype of output
    begin: list.
        represents the index of the first value to select.
    end: list.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin value and
        instead use the largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position is actually an
        ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification should
        shrink the dimensionality.
    kernel_name : str
        cce kernel name, default value is "strided_slice_assign"

    Returns
    -------
    None
    """
    input_shape = ref_dict.get("shape")
    input_dtype = ref_dict.get("dtype").lower()
    if input_dtype == "bfloat16":
        input_dtype = "float16"
    check_list = ("float16", "float32", "int32", "int16")

    para_check.check_shape(input_shape, param_name="ref_dict")
    para_check.check_dtype(input_dtype, check_list, param_name="ref_dict")

    begin = list(begin)
    end = list(end)
    strides = [1] * len(begin) if strides is None else list(strides)

    _check_parameter(input_shape, begin, end, strides, ellipsis_mask, new_axis_mask, shrink_axis_mask)

    _make_up_slice_params(input_shape, begin, end, strides)

    # a bitmask in new_axis_mask where bit `i` being 1, then the same bit 'i'
    # being 1 in shrink_axis_mask is not take effect
    shrink_axis_mask = (new_axis_mask ^ shrink_axis_mask) & shrink_axis_mask
    begin_mask = (shrink_axis_mask ^ begin_mask) & begin_mask
    _update_begin_end_by_begin_end_mask(input_shape, begin, end, begin_mask, end_mask)
    _update_begin_end_strides_by_ellipsis_mask(input_shape, begin, end, strides, ellipsis_mask)
    slice_shape = _update_slice_params(input_shape, begin, end, strides, new_axis_mask, shrink_axis_mask)
    if input_dtype in ("int32", "float32"):
        if len(slice_shape) > 1 and slice_shape[-1] < 8:
            error_manager_vector.raise_err_specific_reson(
                "strided_slice_assign_d",
                "Under int32 and fp32, except dim num is 1, \
                last dimension of input shape must bigger and equal than 8.")
    if input_dtype == "float16":
        if len(slice_shape) > 1 and slice_shape[-1] < 16:
            error_manager_vector.raise_err_specific_reson(
                "strided_slice_assign_d",
                "Under fp16, except dim num is 1, \
                last dimension of input shape must bigger and equal than 16.")

    input_tensor = tvm.placeholder(input_shape, dtype=input_dtype, name='input_tensor')
    input_value_tensor = tvm.placeholder(slice_shape, dtype=input_dtype, name='input_value_tensor')

    slice_params = [begin, end, strides]
    input_value_ub, out = strided_slice_assign_compute(input_tensor, input_value_tensor, slice_params)
    schedule_list = (input_tensor, input_value_ub)
    sch = _strided_slice_assign_schedule(schedule_list, out, slice_shape, input_shape, input_dtype)

    tensor_list = (out, input_value_tensor, input_tensor)
    
    with build_config(dummy_placeholder=False):
        tvm.build(sch, tensor_list, "cce", name=kernel_name)
