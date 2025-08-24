#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

strided slice
"""
from __future__ import with_statement
from types import MethodType
import math
from functools import reduce
from impl import common_util
from impl import constant_util as constant
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.dynamic.strided_slice import StridedSlice
from impl.dynamic.strided_slice import ceil_32bytes_align_count
from impl.strided_slice_d import _fill_list_with_ones
import te.platform as tbe_platform


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    SHRINK_AXIS = -1
    NEW_AXIS = -2
    MAX_SHAPE_DIM = 8
    MAX_STRIDES = 65535
    SHAPE_LEN = 2
    TILING_FACTOR_256 = 256
    TILING_UB_SIZE = 382
    TILING_FACTOR_2 = 2
    TILING_FACTOR_16 = 16


# 'pylint: disable=too-many-arguments,unused-argument
def check_supported(input_x,
                    begin,
                    end,
                    strides=None,
                    output_x=None,
                    begin_mask=0,
                    end_mask=0,
                    ellipsis_mask=0,
                    new_axis_mask=0,
                    shrink_axis_mask=0,
                    kernel_name="strided_slice"):
    """
    check_supported
    """
    strides_value = strides.get("const_value")
    if not strides_value:
        return False, "strides is not const."

    begin_value = begin.get("const_value")
    end_value = end.get("const_value")
    if begin_value or end_value:
        return False, "begin/end is const."

    for i in strides_value:
        if i != 1:
            return False, "strides has not 1 value."

    return True


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
# 'pylint: disable=too-many-locals
# 'pylint: disable=too-many-statements
def strided_slice(input_x,
                  begin,
                  end,
                  strides=None,
                  output_x=None,
                  begin_mask=0,
                  end_mask=0,
                  ellipsis_mask=0,
                  new_axis_mask=0,
                  shrink_axis_mask=0,
                  kernel_name="strided_slice"):
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
    begin: dict.
        shape and dtype of begin, represents the index of the first value to select.
    end: dict.
        shape and dtype of end, represents the index of the last value to select.
    strides: dict.
        shape and dtype of strides, step length to select.
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
        cce kernel name, default value is "strided_slice"

    Returns
    -------
    tik_instance
    """
    strided_slice_instance = StridedSlice(input_x, None, 0, 0, 0, 0, 0, kernel_name)
    strided_slice_instance.begin_gm = strided_slice_instance.tik_instance.Tensor(begin.get("dtype"),
                                                                                 begin.get("shape"),
                                                                                 scope=tik.scope_gm,
                                                                                 name="begin_gm")
    strided_slice_instance.end_gm = strided_slice_instance.tik_instance.Tensor(end.get("dtype"),
                                                                               end.get("shape"),
                                                                               scope=tik.scope_gm,
                                                                               name="end_gm")
    strided_slice_instance.strides_gm = strided_slice_instance.tik_instance.Tensor(strides.get("dtype"),
                                                                                   strides.get("shape"),
                                                                                   scope=tik.scope_gm,
                                                                                   name="strides_gm")

    strided_slice_instance.input_gm = strided_slice_instance.tik_instance.Tensor(input_x.get("dtype"),
                                                                                 input_x.get("shape"),
                                                                                 scope=tik.scope_gm,
                                                                                 name="input_gm")
    strided_slice_instance.output_gm = strided_slice_instance.tik_instance.Tensor(input_x.get("dtype"),
                                                                                  output_x.get("shape"),
                                                                                  scope=tik.scope_gm,
                                                                                  name="output_gm")
    strided_slice_instance.tiling_param = strided_slice_instance.TilingParam(input_x.get("shape"),
                                                                             strided_slice_instance.tik_instance)

    # 'pylint: disable=too-many-locals
    # 'pylint: disable=too-many-statements
    def init_tiling(tiling_inst: StridedSlice.TilingParam):
        begin_dtype = begin.get("dtype")
        dtype = input_x.get("dtype")
        begin_len = begin.get("shape")[0]
        input_shape = input_x.get("shape")
        need_ub_size = ceil_32bytes_align_count(begin_len, begin_dtype)
        dtype_size = common_util.get_data_size(dtype)
        core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        block_element = constant.BLOCK_SIZE // dtype_size
        reserve_ub_size = Constant.TILING_UB_SIZE
        ub_size = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - reserve_ub_size) // \
                   dtype_size // block_element * block_element

        def _data_move(tik_instance: tik.Tik, dest: tik.Tensor, src: tik.Tensor, count):
            dtype_size = common_util.get_data_size(src.dtype)
            burst = math.ceil(count * dtype_size / constant.BLOCK_SIZE)
            tik_instance.data_move(dest, src, 0, 1, burst, 0, 0)

        def _gen_shape(name, name_index):
            name += str(name_index)
            return tiling_inst.tik_instance.Scalar(begin_dtype, name=name)

        # 'pylint: disable=too-many-branches
        def _build_dense_spec(sparse: dict, dense: dict):
            """
            Build expanded begin, end, strides, begin_mask, end_mask
            """
            for idx in range(len(dense["strides"])):
                dense["begin"][idx].set_as(sparse["begin"][idx])
                dense["end"][idx].set_as(sparse["end"][idx])
            # to remove any ellipsis
            if len(sparse["strides"]) < dense["dims"]:
                for idx in range(len(sparse["strides"]), dense["dims"], 1):
                    dense["begin"][idx].set_as(0)
                    dense["end"][idx].set_as(0)
                    dense["strides"].append(0)

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
                    next_index = min(dense["dims"] - (sparse["dims"] - index) + 1 + \
                                     sparse["num_add_axis_after_ellipsis"],
                                    dense["dims"])
                    for i in range(full_index, next_index):
                        full_index = i
                        dense["begin"][full_index].set_as(0)
                        dense["end"][full_index].set_as(0)
                        dense["strides"][full_index] = 1
                        dense["begin_mask"] |= (1 << full_index)
                        dense["end_mask"] |= (1 << full_index)
                        dense["final_shape_gather_indices"].append(full_index)
                    if next_index > full_index:
                        full_index = next_index
                elif bit_value & sparse["new_axis_mask"] != 0:
                    dense["final_shape_gather_indices"].append(Constant.NEW_AXIS)
                else:
                    # Gather slicing spec into appropriate index
                    dense["begin"][full_index].set_as(sparse["begin"][index])
                    dense["end"][full_index].set_as(sparse["end"][index])
                    dense["strides"][full_index] = sparse["strides"][index]
                    if sparse["begin_mask"] & bit_value != 0:
                        dense["begin_mask"] |= (1 << full_index)
                    if sparse["end_mask"] & bit_value != 0:
                        dense["end_mask"] |= (1 << full_index)

                    # If shrink, record where to get the dimensionality from (i.e.
                    # new_axis creates a fake 1 size dimension. Also remember shrink
                    # axis (now in dense form) so we can ignore dense->end below.
                    if sparse["shrink_axis_mask"] & bit_value != 0:
                        dense["final_shape_gather_indices"].append(Constant.SHRINK_AXIS)
                        dense["shrink_axis_mask"] |= (1 << full_index)
                    else:
                        dense["final_shape_gather_indices"].append(full_index)

                    full_index += 1

        # 'pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-branches,too-many-statements
        def _infer_shape(shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask,
                         new_axis_mask, shrink_axis_mask):
            """
            inference output shape, begin value, end value and strides.

            Returns
            -------
            output shape, input shape, begin value, end value, stride value.
            Except the output shape, the other return value has the same length.
            """
            sparse_spec = {
                "dims": len(strides),
                "num_add_axis_after_ellipsis": 0,
                "begin": list(begin),
                "end": list(end),
                "strides": list(strides),
                "begin_mask": begin_mask,
                "end_mask": end_mask,
                "ellipsis_mask": ellipsis_mask,
                "new_axis_mask": new_axis_mask,
                "shrink_axis_mask": shrink_axis_mask
            }

            # Step 1: Account for ellipsis and new axis
            ellipsis_seen = False
            for index, _ in enumerate(sparse_spec.get("strides")):
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
            len_array = max(len(shape), len(strides))
            begin_array = tiling_inst.tik_instance.ScalarArray(dtype=begin_dtype, length=len_array, name="begin_array")
            end_array = tiling_inst.tik_instance.ScalarArray(dtype=begin_dtype, length=len_array, name="end_array")
            dense_spec = {
                "dims": len(shape),
                "begin_mask": 0,
                "end_mask": 0,
                "begin_valid": True,
                "end_valid": True,
                "begin": begin_array,
                "end": end_array,
                "strides": list(strides),
                "final_shape_gather_indices": [],
                "shrink_axis_mask": 0
            }
            _build_dense_spec(sparse_spec, dense_spec)
            dense_begin = dense_spec.get("begin")
            dense_end = dense_spec.get("end")
            dense_stride = list(dense_spec.get("strides"))
            # Step 3: Make implicit ranges (non-zero begin_masks and end_masks) explicit
            #         and bounds check!
            is_simple_slice = True
            processing_shape = []
            processing_begin = tiling_inst.tik_instance.ScalarArray(dtype=begin_dtype,
                                                                    length=len_array, name="processing_begin")
            processing_end = tiling_inst.tik_instance.ScalarArray(dtype=begin_dtype,
                                                                  length=len_array, name="processing_end")
            processing_stride = []
            x_fwd = tiling_inst.tik_instance.Scalar(dtype=begin_dtype, name="x_fwd")
            is_identity = tiling_inst.tik_instance.Scalar(dtype=begin_dtype, name="is_identity", init_value=1)
            slice_dim0 = tiling_inst.tik_instance.Scalar(dtype=begin_dtype, name="slice_dim0", init_value=1)
            take_all_in_dimension = tiling_inst.tik_instance.Scalar(dtype=begin_dtype, name="take_all_in_dimension")

            for i, dim_i in enumerate(shape):
                bit_value = 1 << i
                shrink_i = (dense_spec.get("shrink_axis_mask") & bit_value)
                if dim_i == -1:
                    processing_shape.append(1 if shrink_i != 0 else -1)
                    processing_begin[i].set_as(dense_begin[i])
                    if shrink_i != 0:
                        processing_end[i].set_as(dense_begin[i] + 1)
                    else:
                        processing_end[i].set_as(-1)
                    processing_stride.append(dense_stride[i])
                    continue

                masks = (dense_spec.get("begin_mask") & bit_value, dense_spec.get("end_mask") & bit_value)
                valid_range = (0 if dense_stride[i] > 0 else -1, dim_i if dense_stride[i] > 0 else dim_i - 1)

                # 'pylint: disable=invalid-name,cell-var-from-loop
                def canonical(x, c):
                    res = tiling_inst.tik_instance.Scalar(dtype=begin_dtype, name="res")
                    if masks[c] != 0:
                        return valid_range[c] if dense_stride[i] > 0 else valid_range[(c + 1) & 1]
                    with tiling_inst.tik_instance.if_scope(x < 0):
                        x_fwd.set_as(dim_i + x)
                    with tiling_inst.tik_instance.else_scope():
                        x_fwd.set_as(x)
                    with tiling_inst.tik_instance.if_scope(x_fwd < valid_range[0]):
                        res.set_as(valid_range[0])
                    with tiling_inst.tik_instance.elif_scope(x_fwd < valid_range[1]):
                        res.set_as(x_fwd)
                    with tiling_inst.tik_instance.else_scope():
                        res.set_as(valid_range[1])
                    return res

                is_simple_slice &= (dense_stride[i] == 1)
                begin_and_end_masked = (
                    (dense_spec.get("begin_mask") & bit_value != 0) and (dense_spec.get("end_mask") & bit_value != 0))
                if dense_spec.get("begin_valid") and dense_spec.get("end_valid"):
                    if shrink_i != 0:
                        with tiling_inst.tik_instance.if_scope(dense_begin[i] < 0):
                            x_fwd.set_as(dim_i + dense_begin[i])
                        with tiling_inst.tik_instance.else_scope():
                            x_fwd.set_as(dense_begin[i])
                        dense_begin[i].set_as(x_fwd)
                        dense_end[i].set_as(dense_begin[i] + 1)
                    else:
                        dense_begin[i].set_as(canonical(dense_begin[i], 0))
                        dense_end[i].set_as(canonical(dense_end[i], 1))

                    processing_begin[i].set_as(dense_begin[i])
                    processing_end[i].set_as(dense_end[i])
                    processing_stride.append(dense_stride[i])

                    # Update optimization values
                    with tiling_inst.tik_instance.if_scope(tik.all(dense_stride[i] == 1, dense_begin[i] == 0,
                                                                   dense_end[i] == dim_i)):
                        take_all_in_dimension.set_as(1)
                    with tiling_inst.tik_instance.else_scope():
                        take_all_in_dimension.set_as(0)
                    is_identity.set_as(is_identity & take_all_in_dimension)
                    slice_dim0.set_as(slice_dim0 & ((i == 0 and dense_stride[i] == 1) or take_all_in_dimension))
                else:
                    is_identity.set_as(is_identity & (dense_stride[i] == 1 and begin_and_end_masked))
                    slice_dim0.set_as(slice_dim0 & ((i == 0 and dense_stride[i] == 1) or begin_and_end_masked))

                # Compute the processing shape (the intermediate Eigen will produce)
                known_interval = False
                interval_length = 0
                if shrink_i != 0:
                    # The dimension is still known as 1 for the processing_shape, but will be
                    # discarded for the final shape.
                    interval_length = 1
                    known_interval = True
                elif begin_and_end_masked:
                    # Even if we don't have values for begin or end, we do know that this
                    # dimension covers the whole interval. If we have shape information for
                    # this dimension, that tells us the interval length.
                    if dim_i >= 0:
                        if dense_stride[i] > 0:
                            interval_length = dim_i
                        else:
                            interval_length = -dim_i
                        known_interval = True

                if known_interval:
                    if interval_length == 0 or (interval_length < 0) != (dense_stride[i] < 0):
                        size_i = 0
                    elif interval_length % dense_stride[i] != 0:
                        size_i = interval_length // dense_stride[i] + 1
                    else:
                        size_i = interval_length // dense_stride[i]
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
                elif gather_index == Constant.NEW_AXIS:
                    final_shape.append(1)
                    final_input_shape.append(1)
                    final_input_begin.append(0)
                    final_input_end.append(1)
                    final_input_stride.append(1)
                else:
                    final_shape.append(1)
                    final_input_shape.append(shape[shrink_gather_index])
                    final_input_begin.append(processing_begin[shrink_gather_index])
                    final_input_end.append(processing_begin[shrink_gather_index] + 1)
                    final_input_stride.append(1)
                    shrink_gather_index += 1

            res = (tuple(final_shape), final_input_shape, final_input_begin, final_input_end, final_input_stride)
            return res

        def _set_params(scope_gm, scope_ub, value_list):
            with tiling_inst.tik_instance.new_stmt_scope():
                _data_move(tiling_inst.tik_instance, scope_ub, scope_gm, need_ub_size)
                # set begin values
                for tmp_idx, tmp_value in enumerate(value_list):
                    tmp_value.set_as(scope_ub[tmp_idx])

        def _make_perf_params(ori_input_shape, out_shape, ori_begin, ori_end, ori_strides):
            last_same = False
            perf_size = 0
            perf_params_input = []
            perf_params_output = []
            perf_params_begin = tiling_inst.tik_instance.ScalarArray(dtype=begin_dtype, length=len(ori_strides),
                                                                     name="perf_params_begin")
            perf_params_end = tiling_inst.tik_instance.ScalarArray(dtype=begin_dtype, length=len(ori_strides),
                                                                   name="perf_params_end")
            perf_params_strides = []
            for idx, val in enumerate(ori_input_shape):
                if val != out_shape[idx]:
                    perf_params_input.append(val)
                    perf_params_begin[perf_size].set_as(ori_begin[idx])
                    perf_params_end[perf_size].set_as(ori_end[idx])
                    perf_params_strides.append(ori_strides[idx])
                    perf_params_output.append(out_shape[idx])
                    perf_size = perf_size + 1
                    last_same = False
                    continue

                if not last_same:
                    last_same = True
                    perf_params_input.append(val)
                    perf_params_begin[perf_size].set_as(ori_begin[idx])
                    perf_params_end[perf_size].set_as(ori_end[idx])
                    perf_params_strides.append(ori_strides[idx])
                    perf_params_output.append(out_shape[idx])
                    perf_size = perf_size + 1
                    continue

                perf_index = perf_size - 1
                perf_params_input[perf_index] = perf_params_input[perf_index] * val
                perf_params_begin[perf_index].set_as(perf_params_begin[perf_index] * ori_begin[idx])
                perf_params_end[perf_index].set_as(perf_params_end[perf_index] * ori_end[idx])
                perf_params_strides[perf_index] = perf_params_strides[perf_index] * ori_strides[idx]
                perf_params_output[perf_index] = perf_params_output[perf_index] * out_shape[idx]

            last_second_index = len(perf_params_input) - 2
            if len(perf_params_input) > 1 and perf_params_input[-1] == perf_params_output[-1] and \
                perf_params_strides[last_second_index] == 1:
                perf_params_input[last_second_index] = perf_params_input[last_second_index] * perf_params_input[-1]
                perf_params_begin[last_second_index].set_as(perf_params_begin[last_second_index] *
                                                            perf_params_input[len(perf_params_input) - 1])
                perf_params_end[last_second_index].set_as(perf_params_end[last_second_index] *
                                                          perf_params_input[len(perf_params_input) - 1])
                perf_params_strides[last_second_index] = perf_params_strides[last_second_index] * \
                                                         perf_params_strides[-1]
                perf_params_output[last_second_index] = perf_params_output[last_second_index] * perf_params_output[-1]

                perf_params_input.pop()
                perf_params_strides.pop()
                perf_params_output.pop()

            res = (perf_params_input, perf_params_output, perf_params_begin, perf_params_end, perf_params_strides)
            return res

        def _cal_vnchw_ub_size():
            need_ub_size = (ub_size // dtype_size - block_element) // 2 // block_element * block_element
            return need_ub_size

        def _is_shape_equal_except_last(input_shape, out_shape, length):
            for i in range(length + 1):
                if input_shape[i] != out_shape[i]:
                    return False
            return True

        # 'pylint: disable=too-many-boolean-expressions
        def _get_tiling_mode(input_shape, out_shape):
            tiling_mode = 0
            byte_block = constant.MAX_BLOCK_NUMBER
            stride_limit = Constant.MAX_STRIDES * byte_block
            shape_len = len(out_shape)
            float16_type_size = constant.DATA_SIZE_TWO

            if out_shape[-1] * dtype_size < byte_block:
                tiling_mode = 1
            else:
                tiling_mode = 2

            reduce_shape = 1
            if shape_len > Constant.SHAPE_LEN:
                reduce_shape = reduce(lambda x, y: x * y, out_shape[0:-2])
            if out_shape[-1] * dtype_size < byte_block and shape_len >= Constant.SHAPE_LEN and \
                dtype == constant.DATA_TYPE_FP16 and reduce_shape % core_num == 0 and \
                    out_shape[-2] >= constant.SIZE_SIXTEEN and \
                        input_shape[-1] * Constant.TILING_FACTOR_256 <= _cal_vnchw_ub_size():
                tiling_mode = 3

            shape_equal_except_last = _is_shape_equal_except_last(input_shape, out_shape,
                                                                 shape_len - Constant.SHAPE_LEN)
            if shape_len >= Constant.SHAPE_LEN and out_shape[-1] * dtype_size % byte_block == 0 and \
                input_shape[-1] * dtype_size % byte_block == 0 and shape_equal_except_last and \
                ub_size >= 2 * out_shape[-1] * dtype_size and \
                (input_shape[-1] - out_shape[-1]) * dtype_size <= stride_limit:
                tiling_mode = 4

            if shape_len == Constant.SHAPE_LEN and shape_equal_except_last \
                and out_shape[-1] * dtype_size > byte_block \
                and input_shape[-1] * byte_block * Constant.TILING_FACTOR_2 <= ub_size \
                and input_shape[-1] * dtype_size > byte_block:
                tiling_mode = 6

            if shape_len == Constant.SHAPE_LEN and shape_equal_except_last \
                and out_shape[-1] == 1 and byte_block * (input_shape[-1] + 1) < ub_size:
                tiling_mode = 8

            multi_times = dtype_size // float16_type_size
            input_inner_dims = input_shape[-1] * multi_times
            output_inner_dims = out_shape[-1] * multi_times
            output_32bytes_align_rows = byte_block // float16_type_size
            if output_inner_dims > 0 and output_32bytes_align_rows % output_inner_dims == 0:
                output_32bytes_align_rows = output_32bytes_align_rows // output_inner_dims
            elif output_inner_dims % output_32bytes_align_rows == 0:
                output_32bytes_align_rows = 1
            need_ub_size = input_inner_dims * Constant.TILING_FACTOR_16 * Constant.TILING_FACTOR_2
            if shape_len == Constant.SHAPE_LEN and shape_equal_except_last \
                and need_ub_size * byte_block // dtype_size * output_32bytes_align_rows < ub_size \
                and dtype_size % float16_type_size == 0:
                tiling_mode = 5

            if shape_len == 1 and strides_list[0] == 1:
                tiling_mode = 7

            return tiling_mode

        def _set_tiling_data(tiling_data):
            perf_input_shape, perf_output_shape, perf_begin, perf_end, perf_strides, mode, core_num =\
                tiling_data[0], tiling_data[1], tiling_data[2], tiling_data[3], tiling_data[4], tiling_data[5], \
                tiling_data[6]
            input_length = len(perf_input_shape)
            tiling_inst.tiling_mode.set_as(mode)
            tiling_inst.shape_length.set_as(input_length)
            tiling_inst.core_num.set_as(core_num)
            dst_items = (tiling_inst.input_shape, tiling_inst.output_shape, tiling_inst.begin,
                     tiling_inst.end, tiling_inst.stride)
            from_items = (perf_input_shape, perf_output_shape, perf_begin, perf_end, perf_strides)
            for item_idx, item in enumerate(dst_items):
                for dim_idx in range(input_length):
                    item[dim_idx].set_as(from_items[item_idx][dim_idx])

            tiling_inst.out_dim.set_as(1)
            tiling_inst.out_dim_with_vnchwconv.set_as(1)
            with tiling_inst.tik_instance.for_range(0, input_length) as index:
                dim = tiling_inst.output_shape[index]
                with tiling_inst.tik_instance.if_scope(index < input_length - 1):
                    tiling_inst.out_dim.set_as(tiling_inst.out_dim * dim)
                with tiling_inst.tik_instance.if_scope(index < input_length - 2):
                    tiling_inst.out_dim_with_vnchwconv.set_as(tiling_inst.out_dim_with_vnchwconv * dim)

            with tiling_inst.tik_instance.for_range(0, input_length) as index:
                dim_idx = input_length - 1 - index
                tiling_inst.input_steps[index].set_as(tiling_inst.input_shape[dim_idx])
                tiling_inst.output_steps[index].set_as(tiling_inst.output_shape[dim_idx])
                with tiling_inst.tik_instance.if_scope(index > 0):
                    tiling_inst.input_steps[index].set_as(tiling_inst.input_steps[index] *
                                                          tiling_inst.input_steps[index - 1])
                    tiling_inst.output_steps[index].set_as(tiling_inst.output_steps[index] *
                                                           tiling_inst.output_steps[index - 1])

        begin_ub = tiling_inst.tik_instance.Tensor(begin_dtype, (need_ub_size,), name="begin_ub", scope=tik.scope_ubuf)
        end_ub = tiling_inst.tik_instance.Tensor(begin_dtype, (need_ub_size,), name="end_ub", scope=tik.scope_ubuf)
        begin_list = tuple(map(lambda x: _gen_shape("begin_", x), list(range(begin_len))))
        end_list = tuple(map(lambda x: _gen_shape("end_", x), list(range(begin_len))))
        # set begin values
        _set_params(strided_slice_instance.begin_gm, begin_ub, begin_list)
        # set end values
        _set_params(strided_slice_instance.end_gm, end_ub, end_list)
        # set strides
        strides_list = tuple(strides.get("const_value"))
        if strides_list is None:
            strides_list = _fill_list_with_ones(begin_len)

        final_output_shape, final_input_shape, final_begin, final_end, \
        final_stride = _infer_shape(input_shape, begin_list, end_list, strides_list,
                                    begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                    shrink_axis_mask)

        perf_input_shape, perf_output_shape, perf_begin, perf_end, \
        perf_stride = _make_perf_params(final_input_shape, final_output_shape, final_begin, final_end, final_stride)

        mode = _get_tiling_mode(perf_input_shape, perf_output_shape)

        tiling_data = [perf_input_shape, perf_output_shape, perf_begin, perf_end, perf_stride, mode, core_num]
        _set_tiling_data(tiling_data)

    strided_slice_instance.tiling_param.init = MethodType(init_tiling, strided_slice_instance.tiling_param)
    strided_slice_instance.strided_slice()
    inst = strided_slice_instance.tik_instance
    opt_config = strided_slice_instance.get_opt_config()
    inst.BuildCCE(kernel_name=strided_slice_instance.kernel_name,
                  inputs=(strided_slice_instance.input_gm, strided_slice_instance.begin_gm,
                          strided_slice_instance.end_gm, strided_slice_instance.strides_gm),
                  outputs=(strided_slice_instance.output_gm,),
                  config=opt_config,
                  enable_l2=False)
    return inst
