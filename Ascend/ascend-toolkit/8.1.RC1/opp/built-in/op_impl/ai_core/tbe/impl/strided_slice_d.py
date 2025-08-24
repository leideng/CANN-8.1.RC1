#!/usr/bin/env python
# coding: utf-8
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
strided_slice_d
"""
# 'pylint: disable=too-many-lines
import copy
import math
import functools
import itertools

from impl import copy_only
from impl import common_util
from impl import strided_slice_strides_larger_than_one
from impl import strided_slice_for_last_dim_mte
from impl import strided_slice_last_dim_one
from impl import strided_slice_last_dim_with_vreducev2
from impl import strided_slice_for_last_dim
from impl import strided_slice_fast_last_dim
from impl.strided_slice_for_axis1 import SliceWithAxis1
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_tvm as tvm
from impl.util.platform_adapter import rl_bank
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import buildcfg
from impl.util.platform_adapter import check_support_block_size_16
from impl.util import util_common
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_binary import get_bit_len

SHRINK_AXIS = -1
NEW_AXIS = -2
BURST_LEN = 65535


def _fill_list_with_ones(length):
    """
    fill a list array with ones
    """
    result_list = [1] * length

    return result_list


# 'pylint: disable = unused-argument,too-many-arguments,too-many-locals,too-many-boolean-expressions
def get_op_support_info(input_x, output_x, begin, end, strides=None,
                        begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0,
                        kernel_name="strided_slice_d"):
    """
    get_op_support_info
    """
    input_ori_shape = input_x.get("ori_shape")
    input_format = input_x.get("format").upper()
    input_ori_format = input_x.get("ori_format").upper()
    begin = list(begin)
    end = list(end)
    if strides is None:
        strides = _fill_list_with_ones(len(input_ori_shape))
    else:
        strides = list(strides)

    axis_reduce_list = []
    axis_split_matrix = []
    to_check_shapes = [input_x, output_x]
    if (not is_unknown_rank_input(to_check_shapes) and
            0 not in strides and new_axis_mask == 0 and shrink_axis_mask == 0):
        _, input_shape, begin, end, strides = _infer_shape(input_ori_shape, begin, end,
                                                           strides, begin_mask, end_mask,
                                                           ellipsis_mask, new_axis_mask,
                                                           shrink_axis_mask)
        # _infer_shape make sure than begin, end and strides has same length
        output_shape = list(map(lambda x, y, z: (x - y) // z, end, begin, strides))
        for i, _ in enumerate(input_shape):
            if input_shape[i] == output_shape[i] and input_shape[i] != -1:
                axis = i
                if input_ori_format != input_format:
                    axis = util_common.update_axis_for_other_format(input_shape, i, input_format,
                                                                    input_ori_format, False)
                split_0 = [SplitInput([0, [axis], [-1], [-1]]), SplitOutput([0, [axis]])]
                axis_split_matrix.append(split_0)

    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def _build_dense_spec(sparse: dict, dense: dict):
    """
    Build expanded begin, end, strides, begin_mask, end_mask
    """

    # to remove any ellipsis
    if len(dense["begin"]) < dense["dims"]:
        pad = [0] * (dense["dims"] - len(dense["begin"]))
        dense["begin"] += pad
        dense["end"] += pad
        dense["strides"] += pad
    else:
        dense["begin"] = dense["begin"][0:dense["dims"]]
        dense["end"] = dense["end"][0:dense["dims"]]
        dense["strides"] = dense["strides"][0:dense["dims"]]

    # What indices to get the final shape from.
    dense["begin_mask"] = 0
    dense["end_mask"] = 0
    dense["shrink_axis_mask"] = 0

    full_index = 0
    for index, _ in enumerate(range(0, sparse["dims"])):
        bit_value = 1 << index
        if sparse["ellipsis_mask"] & bit_value != 0:
            # Expand the ellipsis into the appropriate indices
            # NOTE: this only works because we guaranteed one ellipsis
            next_index = min(dense["dims"] - (sparse["dims"] - index) + 1 + sparse["num_add_axis_after_ellipsis"],
                             dense["dims"])
            for i in range(full_index, next_index):
                full_index = i
                dense["begin"][full_index] = 0
                dense["end"][full_index] = 0
                dense["strides"][full_index] = 1
                dense["begin_mask"] |= (1 << full_index)
                dense["end_mask"] |= (1 << full_index)
                dense["final_shape_gather_indices"].append(full_index)
            if next_index > full_index:
                full_index = next_index
        elif bit_value & sparse["new_axis_mask"] != 0:
            dense["final_shape_gather_indices"].append(NEW_AXIS)
        else:
            # Gather slicing spec into appropriate index
            dense["begin"][full_index] = sparse["begin"][index]
            dense["end"][full_index] = sparse["end"][index]
            dense["strides"][full_index] = sparse["strides"][index]
            if sparse["begin_mask"] & bit_value != 0:
                dense["begin_mask"] |= (1 << full_index)
            if sparse["end_mask"] & bit_value != 0:
                dense["end_mask"] |= (1 << full_index)

            # If shrink, record where to get the dimensionality from (i.e.
            # new_axis creates a fake 1 size dimension. Also remember shrink
            # axis (now in dense form) so we can ignore dense->end below.
            if sparse["shrink_axis_mask"] & bit_value != 0:
                dense["final_shape_gather_indices"].append(SHRINK_AXIS)
                dense["shrink_axis_mask"] |= (1 << full_index)
            else:
                dense["final_shape_gather_indices"].append(full_index)

            full_index += 1


# 'pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-branches,too-many-statements
def _infer_shape(shape, begin, end, stride, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):
    """
    inference output shape, begin value, end value and strides.

    Returns
    -------
    output shape, input shape, begin value, end value, stride value. Except the output shape, the other return value
    has the same length.
    """
    sparse_spec = {
        "dims": len(begin),
        "num_add_axis_after_ellipsis": 0,
        "begin": list(begin),
        "end": list(end),
        "strides": list(stride),
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask
    }

    # Step 1: Account for ellipsis and new axis
    ellipsis_seen = False
    for index, _ in enumerate(sparse_spec.get("begin")):
        bit_value = 1 << index
        if ellipsis_seen and (new_axis_mask & bit_value != 0):
            sparse_spec["num_add_axis_after_ellipsis"] = sparse_spec.get("num_add_axis_after_ellipsis") + 1

        if ellipsis_mask & bit_value != 0:
            ellipsis_seen = True

    # If no ellipsis insert one at the end
    if not ellipsis_seen:
        sparse_spec["ellipsis_mask"] |= (1 << sparse_spec.get("dims"))
        sparse_spec["dims"] += 1

    # Step 2: Make a sparse spec into a full index spec
    #
    # The sparse spec does not correspond to the number of dimensions
    # Make a dense spec that corresponds to the number of dimensions
    #
    # For example suppose foo[...,3:] on foo.shape=(2,2,3) then
    # we need to produce the missing begin_mask for the first two
    # dimensions i.e. from begin_mask_spec=0, end_mask_spec=2
    # we achieve begin_mask=6, end_mask=7
    dense_spec = {
        "dims": len(shape),
        "begin_mask": 0,
        "end_mask": 0,
        "begin_valid": True,
        "end_valid": True,
        "begin": list(begin),
        "end": list(end),
        "strides": list(stride),
        "final_shape_gather_indices": [],
        "shrink_axis_mask": 0
    }
    _build_dense_spec(sparse_spec, dense_spec)
    begin = list(dense_spec.get("begin"))
    end = list(dense_spec.get("end"))
    stride = list(dense_spec.get("strides"))
    # Step 3: Make implicit ranges (non-zero begin_masks and end_masks) explicit
    #         and bounds check!
    is_identity = True
    slice_dim0 = True
    is_simple_slice = True
    processing_shape = []
    processing_begin = []
    processing_end = []
    processing_stride = []
    for i, dim_i in enumerate(shape):
        bit_value = 1 << i
        shrink_i = (dense_spec.get("shrink_axis_mask") & bit_value)
        if dim_i == -1:
            processing_shape.append(1 if shrink_i != 0 else -1)
            processing_begin.append(begin[i])
            processing_end.append(begin[i] + 1 if shrink_i != 0 else -1)
            processing_stride.append(stride[i])
            continue

        masks = (dense_spec.get("begin_mask") & bit_value, dense_spec.get("end_mask") & bit_value)
        valid_range = (0 if stride[i] > 0 else -1, dim_i if stride[i] > 0 else dim_i - 1)

        # 'pylint: disable=invalid-name,cell-var-from-loop
        def canonical(x, c):
            if masks[c] != 0:
                return valid_range[c] if stride[i] > 0 else valid_range[(c + 1) & 1]
            x_fwd = (dim_i + x) if x < 0 else x
            return valid_range[0] if x_fwd < valid_range[0] else min(x_fwd, valid_range[1])

        is_simple_slice &= (stride[i] == 1)
        begin_and_end_masked = (
            (dense_spec.get("begin_mask") & bit_value != 0) and (dense_spec.get("end_mask") & bit_value != 0))
        if dense_spec.get("begin_valid") and dense_spec.get("end_valid"):
            if shrink_i != 0:
                # If we are shrinking, the end index is now possibly incorrect. In
                # particular foo[-1] produces sparse_begin = -1, sparse_end = 0.
                # and canonical puts these to n-1 and 0, which implies a degenerate
                # interval. Fortunately, it is now safe to re-create end as begin+1.
                x_fwd = (dim_i + begin[i]) if begin[i] < 0 else begin[i]
                begin[i] = x_fwd
                end[i] = begin[i] + 1
            else:
                begin[i] = canonical(begin[i], 0)
                end[i] = canonical(end[i], 1)

            processing_begin.append(begin[i])
            processing_end.append(end[i])
            processing_stride.append(stride[i])

            # Update optimization values
            take_all_in_dimension = (stride[i] == 1 and begin[i] == 0 and end[i] == dim_i)
            is_identity &= take_all_in_dimension
            slice_dim0 &= ((i == 0 and stride[i] == 1) or take_all_in_dimension)
        else:
            is_identity &= (stride[i] == 1 and begin_and_end_masked)
            slice_dim0 &= ((i == 0 and stride[i] == 1) or begin_and_end_masked)

        # Compute the processing shape (the intermediate Eigen will produce)
        interval_length = 0
        known_interval = False
        if dense_spec.get("begin_valid") and dense_spec.get("end_valid"):
            interval_length = end[i] - begin[i]
            known_interval = True
        elif shrink_i != 0:
            # The dimension is still known as 1 for the processing_shape, but will be
            # discarded for the final shape.
            interval_length = 1
            known_interval = True
        elif begin_and_end_masked:
            # Even if we don't have values for begin or end, we do know that this
            # dimension covers the whole interval. If we have shape information for
            # this dimension, that tells us the interval length.
            if dim_i >= 0:
                interval_length = dim_i if stride[i] > 0 else -dim_i
                known_interval = True

        if known_interval:
            size_i = 0
            # Hold zero if the interval is degenerate, otherwise account for remainder
            if interval_length == 0 or ((interval_length < 0) != (stride[i] < 0)):
                size_i = 0
            else:
                size_i = interval_length // stride[i] + (1 if interval_length % stride[i] != 0 else 0)
            processing_shape.append(size_i)
        else:
            processing_shape.append(-1)

    # Step 4: Compute the final shape
    # new_axis will increase dimension by 1 (with a one-size dimension)
    # slices like foo[3,...] will reduce dimension by 1.
    # This cannot be done earlier, because it depends on Step 3.
    final_shape = []
    final_input_shape = []
    final_input_begin = []
    final_input_end = []
    final_input_stride = []
    shrink_gather_index = 0
    for _, gather_index in enumerate(dense_spec.get("final_shape_gather_indices")):
        if gather_index >= 0:
            final_shape.append(processing_shape[gather_index])
            final_input_shape.append(shape[gather_index])
            final_input_begin.append(processing_begin[gather_index])
            final_input_end.append(processing_end[gather_index])
            final_input_stride.append(processing_stride[gather_index])
            shrink_gather_index = gather_index + 1
        elif gather_index == NEW_AXIS:
            final_shape.append(1)
            # input is scalar
            if len(shape) == 0:
                final_input_shape.append(1)
                final_input_begin.append(0)
                final_input_end.append(1)
                final_input_stride.append(1)
        else:
            final_input_shape.append(shape[shrink_gather_index])
            final_input_begin.append(processing_begin[shrink_gather_index])
            final_input_end.append(processing_begin[shrink_gather_index] + 1)
            final_input_stride.append(1)
            shrink_gather_index += 1

    return tuple(final_shape), final_input_shape, final_input_begin, final_input_end, final_input_stride

# 'pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-branches,too-many-statements
# 'pylint: disable=unused-variable
def _init_parameter(input_list, begin_shape, end_shape, stride_shape,
                    begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                    shrink_axis_mask):
    """
    initialize params
    """
    output_shape, final_input_shape, final_input_begin, final_input_end, final_input_stride = \
        _infer_shape(shape=input_list,
                     begin=begin_shape,
                     end=end_shape,
                     stride=stride_shape,
                     begin_mask=begin_mask,
                     end_mask=end_mask,
                     ellipsis_mask=ellipsis_mask,
                     new_axis_mask=new_axis_mask,
                     shrink_axis_mask=shrink_axis_mask)
    return final_input_shape, final_input_begin, final_input_end, final_input_stride


# 'pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-branches,unused-argument
def strided_slice_d_compute(input_data,
                            output_x,
                            begin,
                            end,
                            stride_shape=None,
                            begin_mask=0,
                            end_mask=0,
                            ellipsis_mask=0,
                            new_axis_mask=0,
                            shrink_axis_mask=0,
                            kernel_name="strided_slice_d"):
    """
    extracts a slice of size (end-begin)/stride from the given input_data.

    Parameters:
    ----------
    input_data: TVM tensor.
        Tensor to be segmented.
    output_x : dict
        shape and dtype of out
    begin: list.
        represents the index of the first value to select.
    end: list.
        represents the index of the last value to select.
    stride_shape: list or tuple.
        step length to select.
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position is actually an ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification should shrink
        the dimensionality.

    Returns
    -------
    Computational process for TVM compute.
    """
    begin_shape = copy.deepcopy(begin)
    end_shape = copy.deepcopy(end)
    stride_shape = copy.deepcopy(stride_shape)

    output_shape = [
        int(math.ceil((end - begin) / (stride * 1.0)))
        for begin, end, stride in zip(begin_shape, end_shape, stride_shape)
    ]

    # AICore don't support scalar
    if not output_shape:
        output_shape = [1]

    def _map_index_norm(*index):
        """
        calculate normal index by strided and begin parameters.
        """
        for i, _ in enumerate(zip(begin_shape, stride_shape)):
            if i == 0:
                index_org = (index[i] * stride_shape[i] + begin_shape[i],)
            else:
                index_org = index_org + (index[i] * stride_shape[i] +
                                         begin_shape[i],)
        return index_org

    output = tvm.compute(output_shape,
                         lambda *i: input_data(*_map_index_norm(*i)),
                         name='output', tag='strided_slice_d|1')

    return [output, output_shape]


# 'pylint: disable=locally-disabled,too-many-return-statements
def _check_parameter(input_shape, begin, end, strides, ellipsis_mask, new_axis_mask, shrink_axis_mask):
    """
    check if the input parameters shape
    """
    ellipsis_dim = 0
    if len(end) != len(begin):
        return False, f"Length of begin[{len(begin)}] and end[{len(end)}] should be equal."

    if strides is not None and new_axis_mask == 0 and shrink_axis_mask == 0:
        if len(end) != len(strides) or len(begin) != len(strides):
            return False, f"Length of begin[{len(begin)}], end[{len(end)}] and strides[{len(strides)}] should be equal."
        for i, _ in enumerate(begin):
            if strides[i] == 0:
                return False, f"Strides[{str(strides)}] should not contain zero."

    if ellipsis_mask != 0:
        for i, _ in enumerate(input_shape):
            if (ellipsis_mask & 2 ** i) == 2 ** i:
                ellipsis_dim += 1
        if ellipsis_dim > 1:
            return False, f"The ellipsis_mask[{ellipsis_mask}] is invalid."

    if strides[-1] <= 0:
        return False, f"The last stride[{strides[-1]}] should be larger than 0."
    return True, ""


def _is_special_shape(out_shape):
    """
    whether the shape needs special tilling method

    """
    if not (len(out_shape) > 2 and out_shape[-1] == 1):
        return False

    if out_shape[-1] == 1:
        count = 0
        for item in out_shape:
            if item != 1:
                count += 1
        if count > 1:
            return True

    return False


def _get_last_not_one(shape):
    """
    get the first axis which is not one from the back

    """
    flag = -1
    axis = 0
    for i, item in enumerate(reversed(shape)):
        if item > 1:
            flag = i
            break
    if flag != -1:
        axis = len(shape) - flag - 1

    return axis


def _get_factor(ele_zero, ele_cnt, total_ele, no_remainder):
    """
    get split factor for _tilling_one_axis function

    """
    split_factor = 1
    if no_remainder:
        for i in reversed(list(range(1, ele_zero))):
            if ele_zero % i == 0 and i*ele_cnt <= total_ele:
                split_factor = i
                break
    else:
        for i in reversed(list(range(1, ele_zero))):
            if i*ele_cnt <= total_ele:
                split_factor = i
                break

    return split_factor


def _tilling_axis(shape, dtype, no_remainder):
    """
    calculate the split parameters according to different shapes
    """
    # size of ub
    ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 1024
    dtype_bytes_size = get_bit_len(dtype) // 8
    # 32 means one block size(32 Bytes), divide by 32 to get
    # the numbers of data that can be stored in one block.
    flag = tbe_platform.get_block_size() // dtype_bytes_size
    element_new = ((shape[-1] + flag - 1) // flag)*flag
    shape_new = []
    for i in shape:
        shape_new.append(i)
    shape_new[-1] = int(element_new)

    total_ele = ub_size_bytes // dtype_bytes_size
    split_axis = 0
    split_factor = 1
    for i, _ in enumerate(shape_new):
        ele_cnt = functools.reduce(lambda x, y: x*y, shape_new[i:])
        if ele_cnt <= total_ele:
            split_axis = i - 1
            split_factor = total_ele // ele_cnt
            break
        if i == len(shape) - 1:
            split_axis = i
            split_factor = total_ele
            break

    if split_axis < 0:
        split_axis = 0
        split_factor = shape_new[0]

    if _is_special_shape(shape):
        last_not_one_axis = _get_last_not_one(shape)
        if split_axis < last_not_one_axis:
            split_axis = last_not_one_axis
            split_factor = shape[last_not_one_axis]

    if no_remainder:
        device_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        if len(shape) >= 2 and split_axis == 0 \
                and device_core_num <= shape[0] < (2 * device_core_num) \
                and shape[0] < BURST_LEN:
            split_factor = 1

    return split_axis, split_factor


def _get_align_axis(out_shape):
    """
    get the axis_info when applying the align
    """
    flag = -1
    if out_shape[-1] != 1:
        axis = len(out_shape) - 2
    else:
        for i, item in enumerate(reversed(out_shape)):
            if item > 1:
                # the first dim greater than 1 in reverse order
                flag = i
                break
        if flag in (-1, 0):
            axis = 0
        else:
            axis = len(out_shape) - flag - 1

    return axis


def _get_target_core_num(first_axis_size):
    cloud_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    target_core_num = cloud_core_num
    for i in reversed(list(range(1, cloud_core_num + 1))):
        if first_axis_size % i == 0:
            target_core_num = i
            break
    
    if target_core_num == 1 and cloud_core_num > 1 and first_axis_size <= 128:
        target_core_num = first_axis_size

    return target_core_num


def _get_ub_block_num(dtype):
    """
    get the ub_size for dtype, get the block_size for dtype
    """
    ub_size_bytes = \
        tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 1024
    # Convert byts to Bytes
    dtype_bytes_size = get_bit_len(dtype) // 8
    ub_number = ub_size_bytes // dtype_bytes_size
    block_number = tbe_platform.get_block_size() // dtype_bytes_size

    return ub_number, block_number


def _get_multicore(input_shape, dtype, split_axis, split_factor):
    """
     judge the input args multicore situation
    """
    length = len(input_shape) - 1
    aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    ub_number, block_number = _get_ub_block_num(dtype)
    result = False
    if split_axis == length:
        last_number = input_shape[length] % split_factor
        if last_number == 0:
            result = split_factor % block_number == 0
        else:
            result = split_factor % block_number == 0 and last_number % block_number == 0
    elif input_shape[length] % block_number == 0 or (input_shape[length] > block_number and aicore_num > 2):
        result = True
    else:
        result = False

    return result


def _check_tik_branch(sch_input_shape, output_shape, begin, strides, input_dtype):
    """
    check last dim
    """
    for i in strides:
        if i != 1:
            return False
    result = True
    sch_input_shape = list(sch_input_shape)

    last_dim = sch_input_shape[len(sch_input_shape) - 1]
    if (len(sch_input_shape) - len(output_shape)) == 1:
        length = len(output_shape)
    elif len(output_shape) == len(sch_input_shape):
        length = len(output_shape) - 1
    else:
        return False

    for i in range(0, length):
        if sch_input_shape[i] != output_shape[i]:
            result = False
            break
    if last_dim == begin[len(begin) - 1]:
        result = False

    _, block_number = _get_ub_block_num(input_dtype)
    if output_shape[-1] % block_number == 0 and sch_input_shape[-1] // output_shape[-1] > tbe_platform.get_block_size():
        result = False

    return result


def _check_last_dim_with_vreducev2(input_shape, output_shape, begin, end, strides, dtype):
    """
    check last dim with vreducev2
    """
    check_vreducev2_supported = tbe_platform.api_check_support("tik.vreducev2")
    if not check_vreducev2_supported:
        return False
    for i in strides:
        if i != 1:
            return False
    if len(output_shape) != len(input_shape):
        return False
    if dtype not in ["float16", "float32"]:
        return False
    for i in range(0, len(output_shape) - 1):
        if input_shape[i] != output_shape[i]:
            return False
    dtype_size = common_util.get_data_size(dtype)
    total_ub_length = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // dtype_size
    if len(input_shape) == 1:
        total_dim = input_shape[-1]
    else:
        total_dim = functools.reduce(lambda x, y: x * y, input_shape[0:])
    if input_shape[-1] >= tbe_platform.get_block_size() and input_shape[-1] < 7500 and len(output_shape) > 1 and \
            output_shape[-1] >= tbe_platform.get_block_size() and total_dim > total_ub_length * 0.8:
        return False
    if 0 <= begin[-1] < end[-1] <= input_shape[-1]:
        return True
    return False


def _check_strides_larger_than_one(strides):
    """
    check strides larger than one
    """
    flag = True
    for value in strides:
        if value < 1:
            flag = False
    for value in strides:
        if value > 1 and flag:
            return True
    return False


def _ceil_div(value, block):
    """
    integrate the input value by block

    """
    return (value + block - 1) // block


# 'pylint: disable=huawei-too-many-arguments
def _update_params_for_other_format(input_dtype, input_shape, begin, end, strides, input_format, input_ori_format):
    """
    update begin, end,  strides base on  ori_format
    """
    if check_support_block_size_16():
        align_c0 = 8
    else:
        align_c0 = 16
    if input_dtype == "int8" or input_dtype == "uint8":
        align_c0 *= 2
    begin = list(begin)
    end = list(end)
    strides = list(strides)
    if input_format in ["NDC1HWC0"]:
        # when NDC1HWC0 will update the C1 and C0 for begin, end and strides
        # ex: begin [N, D, C, H, W] -> [N, D, C // 16, H, W, 0]
        #     end  [N, D, C, H, W] -> [N, D, (C + 15) // 16, H, W, 16]
        #     strides  [N, D, C, H, W] -> [N, D, 1, H, W, 1]
        #     strides[c] only support 1
        begin_ndchw = [begin[input_ori_format.index("N")], begin[input_ori_format.index("D")],
                       begin[input_ori_format.index("C")],
                       begin[input_ori_format.index("H")], begin[input_ori_format.index("W")]]
        end_ndchw = [end[input_ori_format.index("N")], end[input_ori_format.index("D")],
                     end[input_ori_format.index("C")],
                     end[input_ori_format.index("H")], end[input_ori_format.index("W")]]
        strides_ndchw = [strides[input_ori_format.index("N")], strides[input_ori_format.index("D")],
                         strides[input_ori_format.index("C")],
                         strides[input_ori_format.index("H")], strides[input_ori_format.index("W")]]
        input_shape_ndchw = [input_shape[input_ori_format.index("N")], input_shape[input_ori_format.index("D")],
                             input_shape[input_ori_format.index("C")],
                             input_shape[input_ori_format.index("H")], input_shape[input_ori_format.index("W")]]
        # strides[c] ！= 1，raise exception
        # begin[c] is not c0 align, raise exception
        # end[c] is not c0 align and end[c] != input_shape[c], raise exception
        if strides_ndchw[2] != 1 or begin_ndchw[2] % align_c0 != 0 or (
                end_ndchw[2] % align_c0 != 0 and end_ndchw[2] != input_shape_ndchw[2]):
            error_manager_vector.raise_err_specific_reson("strided_slice_d", "Parameter Invalid!")

        begin_c1 = begin_ndchw[2] // align_c0
        begin_c0 = 0
        end_c1 = _ceil_div(end_ndchw[2], align_c0)
        end_c0 = align_c0
        strides_c1 = 1
        strides_c0 = 1

        begin_new = [begin_ndchw[0], begin_ndchw[1],
                     begin_c1, begin_ndchw[3], begin_ndchw[4], begin_c0]
        end_new = [end_ndchw[0], end_ndchw[1],
                   end_c1, end_ndchw[3], end_ndchw[4], end_c0]
        strides_new = [strides_ndchw[0], strides_ndchw[1],
                       strides_c1, strides_ndchw[3], strides_ndchw[4], strides_c0]
        return begin_new, end_new, strides_new

    if input_format in ["NC1HWC0"]:
        # when NC1HWC0 will update the C1 and C0 for begin, end and strides
        # ex: begin [N, C, H, W] -> [N, C // 16, H, W, 0]
        #     end  [N, C, H, W] -> [N, (C + 15) // 16, H, W, 16]
        #     strides  [N, C, H, W] -> [N, 1, H, W, 1]
        #     strides[c] only support 1
        begin_nchw = [begin[input_ori_format.index("N")], begin[input_ori_format.index("C")],
                      begin[input_ori_format.index("H")], begin[input_ori_format.index("W")]]
        end_nchw = [end[input_ori_format.index("N")], end[input_ori_format.index("C")],
                    end[input_ori_format.index("H")], end[input_ori_format.index("W")]]
        strides_nchw = [strides[input_ori_format.index("N")], strides[input_ori_format.index("C")],
                        strides[input_ori_format.index("H")], strides[input_ori_format.index("W")]]
        input_shape_nchw = [input_shape[input_ori_format.index("N")], input_shape[input_ori_format.index("C")],
                            input_shape[input_ori_format.index("H")], input_shape[input_ori_format.index("W")]]
        # strides[c] ！= 1，raise exception
        # begin[c] is not c0 align, raise exception
        # end[c] is not c0 align and end[c] != input_shape[c], raise exception
        if strides_nchw[1] != 1 or begin_nchw[1] % align_c0 != 0 or (
                end_nchw[1] % align_c0 != 0 and end_nchw[1] != input_shape_nchw[1]):
            error_manager_vector.raise_err_specific_reson("strided_slice_d", "Parameter Invalid!")

        begin_c1 = begin_nchw[1] // align_c0
        begin_c0 = 0
        end_c1 = _ceil_div(end_nchw[1], align_c0)
        end_c0 = align_c0
        strides_c1 = 1
        strides_c0 = 1

        begin_new = [begin_nchw[0], begin_c1, begin_nchw[2], begin_nchw[3], begin_c0]
        end_new = [end_nchw[0], end_c1, end_nchw[2], end_nchw[3], end_c0]
        strides_new = [strides_nchw[0], strides_c1, strides_nchw[2], strides_nchw[3], strides_c0]
        return begin_new, end_new, strides_new

    return begin, end, strides


def get_fused_str(format_char_list):
    """get_fused_str for format
    """
    format_iter = itertools.permutations(format_char_list, len(format_char_list))
    format_char_list = list(format_iter)
    format_str_list = []
    for i, char_list in enumerate(format_char_list):
        format_str_list.append(''.join(list(char_list)))

    return format_str_list


def op_select_format(input_x,
                     output_x,
                     begin,
                     end,
                     strides=None,
                     begin_mask=0,
                     end_mask=0,
                     ellipsis_mask=0,
                     new_axis_mask=0,
                     shrink_axis_mask=0,
                     kernel_name="strided_slice_d"):
    """
        select format dynamically
        op_select_format support desc:
        1. when:
        input x's ori_shape is 5
        strides[c] = 1,begins[c] is c0 align
        end[c] is 16 align or end[c] = ori_shape[c]
        The Op StridedSliceD can support 6HD
        > for example:
        > input_x : Tensor (shape=(16, 16, 16, 16, 32), "NDHWC")
        > begin : [0, 0, 0, 0, 0]
        > end : [0, 0, 0, 0, 16]
        > strides : [1, 1, 1, 1, 1]
        > begin_mask : 0
        > end_mask : 0
        > ellipsis_mask : 0
        > new_axis_mask : 0
        > shrink_axis_mask : 0
        > the Op StridedSliceD can process with NDC1HWC0:
        > input_x : Tensor of (shape=(16, 2, 16, 16, 16), "NC1HWC0")
        > output_x: Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")

        2. when:
        input x's ori_shape 's length is 4
        strides[c] = 1, begins[c] is c0 align
        end[c] is c0 align or end[c] = ori_shape[c]
        The Op stridedSliceD can support 5HD
        > for example:
        > input_x : Tensor (shape=(16, 16, 16, 32), "NHWC")
        > begin : [0, 0, 0, 0]
        > end : [0, 0, 0, 16]
        > strides : [1, 1, 1, 1]
        > begin_mask : 0
        > end_mask : 0
        > ellipsis_mask : 0
        > new_axis_mask : 0
        > shrink_axis_mask : 0
        > the Op StridedSliceD can process with NDC1HWC0:
        > input_x : Tensor of (shape=(16, 2, 16, 16, 16), "NC1HWC0")
        > output_x: Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")

        3. In other scenes, The Op StridedSliceD only support ND
        > for example:
        > input_x : Tensor (shape=(16, 16, 16, 16, 32), "NDHWC")
        > begin : [0, 0, 0, 0, 1]
        > end : [0, 0, 0, 0, 15]
        > strides : [1, 1, 1, 1, 1]
    """
    input_ori_shape = input_x.get("ori_shape")
    input_ori_format = input_x.get("ori_format")
    input_dtype = input_x.get("dtype").lower()
    input_shape = input_x.get("shape")
    output_shape = output_x.get("shape")
    output_ori_shape = output_x.get("ori_shape")
    hd_format_c0 = 16
    if input_dtype == "int8" or input_dtype == "uint8" or input_dtype == "bool":
        hd_format_c0 = 32
    begin = list(begin)
    end = list(end)
    if strides is None:
        strides = _fill_list_with_ones(len(input_shape))
    else:
        strides = list(strides)
    bfp16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")
    base_data_type = ["float", "float16", "int8", "int32", "int64", "uint8", "bool", "complex32", "complex64"]
    other_data_type = ["float", "float16", "int32", "int64", "complex32", "complex64"]
    vadd_support_fp32 = tbe_platform.api_check_support("te.lang.cce.vadd", "float32")
    if not vadd_support_fp32:
        base_data_type.remove("float")
        other_data_type.remove("float")
    elif bfp16_support:
        base_data_type.append("bfloat16")
        other_data_type.append("bfloat16")

    dtype_base_out = base_data_type.copy()
    format_base_out = ["ND"] * len(base_data_type)

    if -2 not in input_shape and -2 not in input_ori_shape and -2 not in output_shape and -2 not in output_ori_shape:
        valid_param, _ = _check_parameter(input_ori_shape, begin, end, strides, ellipsis_mask, new_axis_mask,
                                          shrink_axis_mask)
        if valid_param:
            # update input_shape, begin_shape, end_shape
            output_shape, input_shape, begin, end, strides = _infer_shape(input_ori_shape, begin, end,
                                                                          strides, begin_mask, end_mask,
                                                                          ellipsis_mask, new_axis_mask,
                                                                          shrink_axis_mask)

            # charge whether support 6HD
            # conditions:
            # 1.C dim in begin is c0 align
            # 2.C dim in end is c0 align or C dim in end is equal to C dim in shape
            # 3.C dim in strides is 1
            hd_support_format_5d = get_fused_str(["N", "D", "C", "H", "W"])
            hd_support_format_4d = get_fused_str(["N", "C", "H", "W"])
            if list(input_shape) == list(input_ori_shape) and len(input_ori_format) == len(input_ori_shape) and \
                (input_ori_format in hd_support_format_5d or input_ori_format in hd_support_format_4d) and \
                len(input_shape) == len(output_shape):
                dict_zip_begin = dict(zip(list(input_ori_format), begin))
                dict_zip_end = dict(zip(list(input_ori_format), end))
                dict_zip_strides = dict(zip(list(input_ori_format), strides))
                dict_zip_shape = dict(zip(list(input_ori_format), input_ori_shape))
                begin_c_align_flag = dict_zip_begin["C"] % hd_format_c0 == 0
                end_c_align_flag = dict_zip_end["C"] % hd_format_c0 == 0 or dict_zip_end["C"] == dict_zip_shape["C"]
                strides_c_align_flag = dict_zip_strides["C"] == 1
                if begin_c_align_flag and end_c_align_flag and strides_c_align_flag:
                    dtype_base_out = dtype_base_out + other_data_type
                    if input_ori_format in hd_support_format_5d:
                        format_base_out = format_base_out + ["NDC1HWC0"] * len(other_data_type)
                    if input_ori_format in hd_support_format_4d:
                        format_base_out = format_base_out + ["NC1HWC0"] * len(other_data_type)

    dtype_str = ','.join(dtype_base_out)
    format_str = ','.join(format_base_out)

    input0 = gen_param(
        classify="input0", name="x", datatype=dtype_str, format=format_str)
    output0 = gen_param(
        classify="output0", name="y", datatype=dtype_str, format=format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def make_perf_params(output_shape, input_shape, input_begin, input_end, input_strides):
    """
    fused the no sliced dims to make better performance

    Parameters
    ----------
    output_shape : list or tuple
        origin output shape
    input_shape : list or tuple
        origin input shape
    input_begin : list or tuple
        origin begin values
    input_end : list or tuple
        origin end values
    input_strides : list or tuple
        origin stride values
    Returns
    -------
    list or tuple
        the new params of fused the no sliced dims: output_shape, input_shape, input_begin, input_end, input_strides.
    """
    for _, value in enumerate(input_strides):
        if value < 0:
            return output_shape, input_shape, input_begin, input_end, input_strides

    last_same = False
    perf_size = 0
    perf_output_shape = []
    perf_input_shape = []
    perf_input_begin = []
    perf_input_end = []
    perf_input_strides = []
    for i, _ in enumerate(input_shape):
        if input_shape[i] != output_shape[i]:
            last_same = False
            perf_output_shape.append(output_shape[i])
            perf_input_shape.append(input_shape[i])
            perf_input_begin.append(input_begin[i])
            perf_input_end.append(input_end[i])
            perf_input_strides.append(input_strides[i])
            perf_size += 1
            continue

        if not last_same:
            last_same = True
            perf_output_shape.append(output_shape[i])
            perf_input_shape.append(input_shape[i])
            perf_input_begin.append(input_begin[i])
            perf_input_end.append(input_end[i])
            perf_input_strides.append(input_strides[i])
            perf_size += 1
            continue

        index = perf_size - 1
        perf_output_shape[index] *= output_shape[i]
        perf_input_shape[index] *= input_shape[i]
        perf_input_begin[index] = 0
        perf_input_end[index] = perf_input_shape[index]
        perf_input_strides[index] = 1

    if len(perf_input_shape) > 1 and perf_input_shape[-1] == perf_output_shape[-1] and perf_input_strides[-2] == 1:
        index = -2
        perf_output_shape[index] *= perf_output_shape[-1]
        perf_input_shape[index] *= perf_input_shape[-1]
        perf_input_begin[index] *= perf_input_shape[-1]
        perf_input_end[index] *= perf_input_shape[-1]
        perf_input_strides[index] = 1

        perf_output_shape.pop(-1)
        perf_input_shape.pop(-1)
        perf_input_begin.pop(-1)
        perf_input_end.pop(-1)
        perf_input_strides.pop(-1)

    aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    if (perf_output_shape[0] == perf_input_shape[0] and
            perf_output_shape[0] % aicore_num == 0 and
            perf_output_shape[0] != aicore_num):
        first_dim = perf_output_shape[0]
        perf_output_shape.insert(0, aicore_num)
        perf_input_shape.insert(0, aicore_num)
        perf_input_begin.insert(0, 0)
        perf_input_end.insert(0, aicore_num)
        perf_input_strides.insert(0, 1)
        loop = first_dim // aicore_num
        perf_output_shape[1] = loop
        perf_input_shape[1] = loop
        perf_input_begin[1] = 0
        perf_input_end[1] = loop
        perf_input_strides[1] = 1

    output_shape = perf_output_shape
    input_shape = perf_input_shape
    input_begin = perf_input_begin
    input_end = perf_input_end
    input_strides = perf_input_strides

    return output_shape, input_shape, input_begin, input_end, input_strides


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def strided_slice_d(input_x,
                    output_x,
                    begin,
                    end,
                    strides=None,
                    begin_mask=0,
                    end_mask=0,
                    ellipsis_mask=0,
                    new_axis_mask=0,
                    shrink_axis_mask=0,
                    kernel_name="strided_slice_d"):
    """
    Extracts a strided slice of a tensor (generalized python array indexing).
    Roughly speaking, this op extracts a slice of size (end-begin)/stride
    from the given input_ tensor.
    Starting at the location specified by begin the slice continues
     by adding stride to the index
    until all dimensions are not less than end. Note that a stride
    can be negative, which causes a reverse slice.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of out
    begin: list.
        represents the index of the first value to select.
    end: list.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin
        value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position
        is actually an ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification
        should shrink the dimensionality.
    kernel_name : str
        cce kernel name, default value is "strided_slice_d"

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_ori_shape = input_x.get("ori_shape")
    input_format = input_x.get("format")
    input_ori_format = input_x.get("ori_format")
    input_dtype = input_x.get("dtype").lower()
    input_dtype, input_x["dtype"], output_x["dtype"] = (input_dtype, input_dtype,
                                                        input_dtype) if input_dtype != "bool" else ("int8", "int8",
                                                                                                    "int8")
    input_dtype, input_x["dtype"], output_x["dtype"] = (input_dtype, input_dtype,
                                                        input_dtype) if input_dtype != "bfloat16" else ("float16",
                                                                                                        "float16",
                                                                                                        "float16")
    input_dtype, input_x["dtype"], output_x["dtype"] = (input_dtype, input_dtype,
                                                        input_dtype) if input_dtype != "complex32" else ("int32",
                                                                                                         "int32",
                                                                                                         "int32")
    input_dtype, input_x["dtype"], output_x["dtype"] = (input_dtype, input_dtype,
                                                        input_dtype) if input_dtype != "complex64" else ("int64",
                                                                                                         "int64",
                                                                                                         "int64")
    para_check.check_shape(input_shape, param_name="input_x")

    begin = list(begin)
    end = list(end)
    if strides is None:
        strides = _fill_list_with_ones(len(input_shape))
    else:
        strides = list(strides)

    valid_param, msg = _check_parameter(input_ori_shape, begin, end, strides, ellipsis_mask, new_axis_mask,
                                        shrink_axis_mask)
    if not valid_param:
        error_manager_vector.raise_err_specific_reson("strided_slice_d", msg)

    ssa1 = SliceWithAxis1(input_x, output_x, begin, end, strides, begin_mask, end_mask,
                          ellipsis_mask, new_axis_mask, shrink_axis_mask, kernel_name)
    if ssa1.check_params():
        ssa1.compute()
        return

    # update input_shape, begin_shape, end_shape
    input_shape_new, begin, end, strides = _init_parameter(input_ori_shape, begin, end,
                                                           strides, begin_mask, end_mask,
                                                           ellipsis_mask, new_axis_mask,
                                                           shrink_axis_mask)

    # update begin, end, strides according to format
    if input_format in ["NDC1HWC0", "NC1HWC0"]:
        begin, end, strides = _update_params_for_other_format(input_dtype, input_ori_shape, begin, end, strides,
                                                              input_format, input_ori_format)
    else:
        input_shape = input_shape_new.copy()

    input_tensor = tvm.placeholder(input_shape,
                                   dtype=input_dtype,
                                   name='input_tensor')

    [output, out_shape] = strided_slice_d_compute(input_tensor, output_x, begin, end, strides,
                                                  begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
                                                  kernel_name=kernel_name)

    # 'pylint: disable=locally-disabled,unnecessary-lambda
    out_tensor = tvm.compute(out_shape,
                             lambda *i: output(*i),
                             name='out_tensor',
                             tag='strided_slice_d|3')

    input_size = functools.reduce(lambda x, y: x * y, input_shape[0:])
    out_size = functools.reduce(lambda x, y: x * y, out_shape[0:])

    output_dtype = output_x.get("dtype").lower()
    output_shape = output_x.get("shape")
    input_shape = input_x.get("shape")
    if input_size == out_size and -1 not in strides:
        if output_dtype == "bool":
            input_x["dtype"] = "int8"
            output_x["dtype"] = "int8"
        if not output_shape:
            output_x["shape"] = (1,)
        if not input_shape:
            input_x["shape"] = (1,)
        copy_only.copy_only(input_x, output_x, kernel_name)
        return

    output_shape_one = list(output_shape)
    if ellipsis_mask == 0 and shrink_axis_mask != 0:
        for i, _ in enumerate(list(input_shape)):
            if (shrink_axis_mask & 2 ** i) == 2 ** i:
                output_shape_one.insert(i, 1)
    output_shape = tuple(output_shape_one)

    # for RL tune getting res
    tbe_platform.fusion_manager.set_op_res(out_tensor)

    ret, sch = rl_bank.query_rl_bank([out_tensor])
    if ret and sch:
        with buildcfg.build_config():
            tvm.build(sch, [input_tensor, out_tensor], "cce", name=kernel_name)
        return

    if _check_strides_larger_than_one(strides) and not check_support_block_size_16():
        begin_shape = copy.deepcopy(begin)
        end_shape = copy.deepcopy(end)
        strides_shape = copy.deepcopy(strides)
        res = strided_slice_strides_larger_than_one.strided_slice_strides_larger_than_one(input_shape,
                                                                                          input_dtype,
                                                                                          begin_shape,
                                                                                          end_shape,
                                                                                          strides_shape,
                                                                                          kernel_name)
        if res:
            return

    if _check_last_dim_with_vreducev2(input_shape, output_shape, begin, end, strides, input_dtype) and \
       not check_support_block_size_16():
        begin_shape = copy.deepcopy(begin)
        end_shape = copy.deepcopy(end)
        strides_shape = copy.deepcopy(strides)
        res = strided_slice_last_dim_with_vreducev2.strided_slice_last_dim_with_vreducev2(input_shape,
                                                                                          input_dtype,
                                                                                          begin_shape,
                                                                                          end_shape,
                                                                                          strides_shape,
                                                                                          kernel_name)
        if res:
            return

    output_shape = list(map(lambda x, y, z: (x - y) // z, end, begin, strides))
    output_shape, input_shape, begin, end, strides = make_perf_params(output_shape, input_shape, begin, end, strides)
    if _check_tik_branch(input_shape, output_shape, begin, strides, input_dtype):
        begin_shape = copy.deepcopy(begin)
        end_shape = copy.deepcopy(end)
        stride_shape = list(strides)
        stride_shape = copy.deepcopy(stride_shape)
        input_list = list(input_shape)
        head_size = 1
        for i in range(0, (len(input_shape) - 1)):
            head_size = head_size * input_shape[i]
        if input_list[-1] > 80 and output_shape[-1] == 80:
            res1 = strided_slice_fast_last_dim.strided_slice_last_dim_only(input_shape, input_dtype,
                                                                           output_shape, begin_shape,
                                                                           kernel_name)
            if res1:
                return
        if tbe_platform.get_block_size() <= input_list[-1] < 7500 and len(output_shape) > 1 \
           and output_shape[-1] >= tbe_platform.get_block_size():
            res = strided_slice_for_last_dim_mte.strided_slice_last_dim_mte(input_shape, input_dtype,
                                                                            output_shape, begin_shape,
                                                                            kernel_name)
            if res:
                return
        res = strided_slice_for_last_dim.strided_slice_last_dim(input_shape, input_dtype,
                                                                output_shape, begin_shape,
                                                                end_shape, stride_shape,
                                                                kernel_name)
        if res:
            return
        res1 = strided_slice_last_dim_one.strided_slice_last_dim_one(input_shape, input_dtype,
                                                                     output_shape, begin_shape,
                                                                     kernel_name)
        if res1:
            return

    input_tensor = tvm.placeholder(input_shape, dtype=input_dtype, name='input_tensor')

    [output, out_shape] = strided_slice_d_compute(input_tensor, output_x, begin, end, strides,
                                                  begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
                                                  kernel_name=kernel_name)

    # 'pylint: disable=locally-disabled,unnecessary-lambda
    out_tensor = tvm.compute(out_shape, lambda *i: output(*i), name='out_tensor', tag='strided_slice_d|3')

    # for RL tune getting res
    tbe_platform.fusion_manager.set_op_res(out_tensor)

    ret, sch = rl_bank.query_rl_bank([out_tensor])
    if ret and sch:
        with buildcfg.build_config():
            tvm.build(sch, [input_tensor, out_tensor], "cce", name=kernel_name)
        return

    sch = tvm.create_schedule(out_tensor.op)
    sch[output].set_scope(tbe_platform.scope_ubuf)

    dtype_size = common_util.get_data_size(input_dtype)
    element_align = tbe_platform.BLOCK_REDUCE_INT8 // dtype_size
    core_state = False
    if out_shape[-1] < element_align:
        split_axis, split_factor = _tilling_axis(out_shape, input_dtype, False)
    else:
        split_axis, split_factor = _tilling_axis(out_shape, input_dtype, True)
    core_state = _get_multicore(out_shape, input_dtype, split_axis, split_factor)

    axis_outer, axis_inner = sch[out_tensor].split(out_tensor.op.axis[split_axis], factor=split_factor)

    if split_axis == 0:
        core_num = _get_target_core_num(out_shape[split_axis] // split_factor)
        axis_outer_outer, axis_outer_inter = sch[out_tensor].split(
            axis_outer, nparts=core_num)
    else:
        core_num = _get_target_core_num(out_shape[0])
        axis_outer_outer, axis_outer_inter = sch[out_tensor].split(out_tensor.op.axis[0], nparts=core_num)
        for i in range(1, split_axis):
            axis_outer_inter = sch[out_tensor].fuse(axis_outer_inter, out_tensor.op.axis[i])
        axis_outer_inter = sch[out_tensor].fuse(axis_outer_inter, axis_outer)

    sch[output].compute_at(sch[out_tensor], axis_outer_inter)

    sch[output].emit_insn(output.op.axis[0], tbe_platform.DMA_COPY)  # gm-ub
    if len(out_shape) >= 2:
        # Convert bytes to Bytes
        dtype_bytes_size = get_bit_len(input_dtype) // 8
        # 32 means one block size(32 Bytes), divide by 32 to
        # get the numbers of data that
        # can be stored in one block.
        element = tbe_platform.get_block_size() // dtype_bytes_size
        align_axis = _get_align_axis(out_shape)
        sch[output].storage_align(output.op.axis[align_axis], element, 0)

    if core_state:
        thread_block = tvm.thread_axis("blockIdx.x")
        sch[out_tensor].bind(axis_outer_outer, thread_block)

    sch[out_tensor].emit_insn(axis_inner, tbe_platform.DMA_COPY)  # ub-gm

    with buildcfg.build_config():
        tvm.build(sch, [input_tensor, out_tensor], "cce", name=kernel_name)
