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

strided slice
"""

from __future__ import absolute_import
import math
from ..util.platform_adapter import para_check
from ..util.platform_adapter import tik

from .. import common_util
from .. import constant_util as constant
from ..util.platform_adapter import register_operator
from ..util.platform_adapter import tbe_context

MAX_SIZE = 2 ** 31 - 1
MAX_NBURST = 4095


def ceil_32bytes_align_count(count, dtype):
    """
    ceil_32bytes_align_count
    """
    type_size = common_util.get_data_size(dtype)
    block_count = math.ceil(count * type_size / constant.BLOCK_SIZE)
    return block_count * constant.BLOCK_SIZE // type_size


def _data_move(tik_instance: tik.Tik, dest: tik.Tensor, src: tik.Tensor, count):
    """
    _data_move
    """
    dtype_size = common_util.get_data_size(src.dtype)
    burst = math.ceil(count * dtype_size / constant.BLOCK_SIZE)
    tik_instance.data_move(dest, src, 0, 1, burst, 0, 0)


# pylint: disable=too-many-locals, too-many-statements, too-many-instance-attributes
# pylint: disable=too-few-public-methods
class StridedSlice:
    """
    StridedSlice
    """

    # pylint: disable=too-many-locals, too-many-statements, too-many-instance-attributes
    class TilingParam:
        """
        TilingParam
        """

        def __init__(self, input_x_shape, inst: tik.Tik):
            """
            tiling param
            :param input_x_shape: input shape
            :param inst: tik instance
            """
            self.tik_instance = inst
            dtype = "int64"
            self.dtype = dtype
            # mode_type, shape_length, input_shape, output_shape, begin, end, stride
            tiling_gm_size = 2 + len(input_x_shape) * 5
            self.tiling_gm = inst.Tensor(dtype, (tiling_gm_size,), name="tiling_gm", scope=tik.scope_gm)

            self.input_shape = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="input_dims")
            self.begin = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="begin_dims")
            self.end = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="end_dims")
            self.stride = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="stride_dims")
            self.output_shape = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="output_shape_dims")
            self.input_steps = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="input_steps")
            self.output_steps = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="output_steps")

            self.shape_length = inst.Scalar(dtype, name="shape_length", init_value=len(input_x_shape))
            self.tiling_mode = inst.Scalar(dtype, name="tiling_mode")
            self.out_dim = inst.Scalar(dtype, name="out_dim")
            self.out_dim_with_vnchwconv = inst.Scalar(dtype, name="out_dim_with_vnchwconv")

        # pylint: disable=invalid-name
        def init(self):
            """
            init process data
            """
            with self.tik_instance.new_stmt_scope():
                need_ub_size = ceil_32bytes_align_count(self.tiling_gm.shape[0], self.dtype)
                tiling_ub = self.tik_instance.Tensor(self.dtype, (need_ub_size,), name="tiling_ub",
                                                     scope=tik.scope_ubuf)
                _data_move(self.tik_instance, tiling_ub, self.tiling_gm, need_ub_size)

                self.tiling_mode.set_as(tiling_ub[0])
                self.shape_length.set_as(tiling_ub[1])
                index = self.tik_instance.Scalar(init_value=2)
                items = (self.input_shape, self.output_shape, self.begin, self.end, self.stride)
                for item in items:
                    with self.tik_instance.for_range(0, self.shape_length) as dim_idx:
                        item[dim_idx].set_as(tiling_ub[index])
                        index.set_as(index + 1)

            self.out_dim.set_as(1)
            self.out_dim_with_vnchwconv.set_as(1)
            with self.tik_instance.for_range(0, self.shape_length) as index:
                dim = self.output_shape[index]
                with self.tik_instance.if_scope(index < self.shape_length - 1):
                    self.out_dim.set_as(self.out_dim * dim)
                with self.tik_instance.if_scope(index < self.shape_length - 2):
                    self.out_dim_with_vnchwconv.set_as(self.out_dim_with_vnchwconv * dim)

            with self.tik_instance.for_range(0, self.shape_length) as index:
                dim_idx = self.shape_length - 1 - index
                self.input_steps[index].set_as(self.input_shape[dim_idx])
                self.output_steps[index].set_as(self.output_shape[dim_idx])
                with self.tik_instance.if_scope(index > 0):
                    self.input_steps[index].set_as(self.input_steps[index] * self.input_steps[index - 1])
                    self.output_steps[index].set_as(self.output_steps[index] * self.output_steps[index - 1])

    # pylint: disable=locally-disabled,too-many-arguments,
    # pylint: disable=unused-argument,too-many-locals
    def __init__(self, input_x, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
                 kernel_name="strided_slice"):
        self.strides = strides
        self.begin_mask = begin_mask
        self.end_mask = end_mask
        self.ellipsis_mask = ellipsis_mask
        self.new_axis_mask = new_axis_mask
        self.shrink_axis_mask = shrink_axis_mask
        self.kernel_name = kernel_name

        inst = tik.Tik()
        self.tik_instance = inst
        self.tik_profiling = tik.Dprofile()
        max_dim_supported = 8
        self.tiling_param = self.TilingParam([1] * max_dim_supported, inst)
        self.dtype = input_x.get("dtype").lower()
        self.dtype_size = common_util.get_data_size(self.dtype)
        self.input_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="input_gm", scope=tik.scope_gm)
        self.begin_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="begin_gm", scope=tik.scope_gm)
        self.end_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="end_gm", scope=tik.scope_gm)
        self.strides_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="strides_gm", scope=tik.scope_gm)
        self.output_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="output_gm", scope=tik.scope_gm)
        self.aicore_num = self.tik_profiling.get_aicore_num()
        self.block_element = constant.BLOCK_SIZE // self.dtype_size
        self.reserve_ub_size = 0
        self.ub_size = (self.tik_profiling.get_unified_buffer_size() // self.dtype_size // self.block_element *
                        self.block_element) - self.reserve_ub_size
        self.ub_size_with_vnchwconv = \
            (self.tik_profiling.get_unified_buffer_size() // self.dtype_size - self.block_element) \
            // 2 // self.block_element * self.block_element
        self.max_gap = 65535 * self.block_element
        self.max_last_dim = (self.max_gap + self.ub_size) // self.block_element
        self.shape_length = None

    def _ceil_div(self, int1: tik.Scalar, int2):
        """
        get ceil for (int1 / int2)
        """
        result = self.tik_instance.Scalar("int64")
        with self.tik_instance.if_scope(int1 == 0):
            result.set_as(1)
        with self.tik_instance.else_scope():
            result.set_as(int1 // int2)
        with self.tik_instance.if_scope(int1 % int2 != 0):
            result.set_as(result + 1)

        return result

    def _ceil_32bytes_count(self, count: tik.Scalar):
        """
        _ceil_32bytes_count
        """
        ceil_num = self._ceil_div(count, self.block_element)
        return ceil_num * self.block_element

    def _get_input_gm_addr(self, cur_index: tik.Scalar):
        """
        _get_input_gm_addr
        """
        inst = self.tik_instance
        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        dim_count = self.tiling_param.shape_length
        addr = inst.Scalar(self.tiling_param.dtype, name="input_addr")
        addr.set_as(self.tiling_param.begin[dim_count - 1])
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")

        with inst.for_range(2, dim_count + 1) as dim_idx:
            dim = self.tiling_param.output_shape[dim_count - dim_idx]
            step = self.tiling_param.input_steps[dim_idx - 2]
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * (tmp + self.tiling_param.begin[dim_count - dim_idx]))
            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _get_output_gm_addr(self, cur_index: tik.Scalar):
        """
        _get_output_gm_addr
        """
        inst = self.tik_instance
        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        dim_count = self.tiling_param.shape_length
        addr = inst.Scalar(self.tiling_param.dtype, name="output_addr")
        addr.set_as(0)
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")

        with inst.for_range(2, dim_count + 1) as dim_idx:
            dim = self.tiling_param.output_shape[dim_count - dim_idx]
            step = self.tiling_param.output_steps[dim_idx - 2]
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * tmp)
            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _data_move(self, dest: tik.Tensor, src: tik.Tensor, count: tik.Scalar):
        """
        _data_move
        """
        dtype_size = common_util.get_data_size(src.dtype)
        burst = self._ceil_div(count * dtype_size, constant.BLOCK_SIZE)
        self.tik_instance.data_move(dest, src, 0, 1, burst, 0, 0)

    def strided_slice(self):
        """
        strided_slice
        """
        inst = self.tik_instance
        core_num = self.aicore_num
        if self.dtype == "float16":
            with inst.for_range(0, core_num, block_num=core_num, name="core_idx") as i:
                self.tiling_param.init()
                self.shape_length = self.tiling_param.shape_length
                with inst.if_scope(self.tiling_param.tiling_mode == 1):
                    self._do_small_last_dim(i)
                with inst.else_scope():
                    with inst.if_scope(self.tiling_param.tiling_mode == 2):
                        self._do_large_last_dim(i)
                    with inst.else_scope():
                        with inst.if_scope(self.tiling_param.tiling_mode == 3):
                            self._do_small_last_dim_with_vnchwconv(i)
                        with inst.else_scope():
                            self._do_large_last_dim_multi_rows(i)
        else:
            with inst.for_range(0, core_num, block_num=core_num, name="core_idx") as i:
                self.tiling_param.init()
                self.shape_length = self.tiling_param.shape_length
                with inst.if_scope(tik.any(self.tiling_param.tiling_mode == 1, self.tiling_param.tiling_mode == 3)):
                    self._do_small_last_dim(i)
                with inst.else_scope():
                    with inst.if_scope(self.tiling_param.tiling_mode == 2):
                        self._do_large_last_dim(i)
                    with inst.else_scope():
                        self._do_large_last_dim_multi_rows(i)

    def _do_large_last_dim_multi_rows(self, core_idx):
        """
        _do_large_last_dim_multi_rows
        """
        inst = self.tik_instance
        core_num = self.aicore_num
        # out_dim = 67, num_rows_per_aicore: 3, 3, 3, 2, 2, 2 ...
        num_rows_per_aicore = inst.Scalar("int64", name="num_rows_per_aicore")
        num_tail_rows = inst.Scalar("int64", name="num_tail_rows")
        num_rows_per_aicore.set_as(self.tiling_param.out_dim // core_num)
        num_tail_rows.set_as(self.tiling_param.out_dim - core_num * num_rows_per_aicore)

        row_idx = inst.Scalar("int64", name="row_idx")
        with inst.if_scope(core_idx < num_tail_rows):
            row_idx.set_as(core_idx + core_idx * num_rows_per_aicore)
        with inst.else_scope():
            row_idx.set_as(num_tail_rows + core_idx * num_rows_per_aicore)

        with inst.if_scope(core_idx < num_tail_rows):
            num_rows_per_aicore.set_as(num_rows_per_aicore + 1)

        with inst.if_scope(row_idx < self.tiling_param.out_dim):
            self._do_large_last_dim_multi_rows_per_aicore(row_idx, num_rows_per_aicore)

    def _do_large_last_dim_multi_rows_per_aicore(self, row_idx, num_rows):
        """
        _do_large_last_dim_multi_rows_per_aicore
        """
        inst = self.tik_instance
        output_shape = self.tiling_param.output_shape
        input_shape = self.tiling_param.input_shape
        dim_count = self.tiling_param.shape_length
        max_rows_per_data_move = inst.Scalar("int64", name="loops")
        max_rows_per_data_move.set_as(self.ub_size // output_shape[dim_count - 1])
        loops = inst.Scalar("int64", name="loops")
        loops.set_as(num_rows // max_rows_per_data_move)

        row_loop_idx = inst.Scalar("int64", name="row_loop_idx")

        with inst.new_stmt_scope():
            ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="ub")
            with inst.for_range(0, loops, name="loops_for_range") as loop_idx:
                row_loop_idx.set_as(row_idx + loop_idx * max_rows_per_data_move)
                input_gm_addr = self._get_input_gm_addr(row_loop_idx)
                output_gm_addr = self._get_output_gm_addr(row_loop_idx)
                inst.data_move(
                    ub,
                    self.input_gm[input_gm_addr],
                    0,
                    max_rows_per_data_move,
                    output_shape[dim_count - 1] // self.block_element,
                    (input_shape[dim_count - 1] - output_shape[dim_count - 1]) // self.block_element,
                    0)  # input gm -> ub

                inst.data_move(
                    self.output_gm[output_gm_addr],
                    ub,
                    0,
                    1,
                    max_rows_per_data_move * output_shape[dim_count - 1] // self.block_element,
                    0,
                    0)  # ub -> output gm

            num_remain_rows = num_rows - loops * max_rows_per_data_move
            with inst.if_scope(num_remain_rows > 0):
                row_loop_idx.set_as(row_idx + loops * max_rows_per_data_move)
                input_gm_addr = self._get_input_gm_addr(row_loop_idx)
                output_gm_addr = self._get_output_gm_addr(row_loop_idx)
                inst.data_move(
                    ub,
                    self.input_gm[input_gm_addr],
                    0,
                    num_remain_rows,
                    output_shape[dim_count - 1] // self.block_element,
                    (input_shape[dim_count - 1] - output_shape[dim_count - 1]) // self.block_element,
                    0)  # input gm -> ub
                inst.data_move(
                    self.output_gm[output_gm_addr],
                    ub,
                    0,
                    1,
                    num_remain_rows * output_shape[dim_count - 1] // self.block_element,
                    0,
                    0)  # ub -> output gm

    def _do_large_last_dim(self, core_idx):
        self._do_large_last_dim_normal(core_idx)

    def _do_large_last_dim_normal(self, core_idx):
        """
        _do_large_last_dim_normal
        """
        inst = self.tik_instance
        core_num = self.aicore_num
        output_shape = self.tiling_param.output_shape
        inner_loops = self._ceil_div(output_shape[self.shape_length - 1], self.ub_size)
        out_loops = self._ceil_div(self.tiling_param.out_dim, core_num)
        with inst.for_range(0, out_loops, name="out_loop") as loop_idx:
            idx = core_idx * out_loops + loop_idx
            with inst.if_scope(idx < self.tiling_param.out_dim):
                input_gm_addr = self._get_input_gm_addr(idx)
                output_gm_addr = self._get_output_gm_addr(idx)
                with inst.for_range(0, inner_loops, name="inner_loop") as inner_loop_idx:
                    with inst.if_scope(output_shape[self.shape_length - 1] % self.block_element == 0):
                        self._do_large_last_dim_align(input_gm_addr, output_gm_addr, inner_loop_idx)
                    with inst.else_scope():
                        self._do_large_last_dim_not_align(input_gm_addr, output_gm_addr, inner_loop_idx)

    # pylint: disable=invalid-name
    def _do_small_last_dim(self, core_idx):
        """
        _do_small_last_dim
        """
        inst = self.tik_instance
        core_num = self.aicore_num
        output_shape = self.tiling_param.output_shape
        inner_dim = output_shape[self.shape_length - 1]
        out_dim = self.tiling_param.out_dim
        out_loops = self._ceil_div(out_dim, core_num)
        tmp_ub_size = self.block_element
        ub_size = self.ub_size - self.block_element
        ub_data_count = inst.Scalar("int32", name="out_ub_data_count")
        ub_data_count.set_as(0)
        input_gm = self.input_gm
        output_gm = self.output_gm
        need_update_out_addr = inst.Scalar("int32", name="need_update_out_addr")
        need_update_out_addr.set_as(1)
        output_gm_addr = inst.Scalar(self.tiling_param.dtype,
                                     name="output_addr")
        with inst.new_stmt_scope():
            tmp_ub = inst.Tensor(self.dtype, (tmp_ub_size,), scope=tik.scope_ubuf, name="tmp_ub")
            ub = inst.Tensor(self.dtype, (ub_size,), scope=tik.scope_ubuf, name="out_ub")
            with inst.for_range(0, out_loops, name="out_loop") as loop_idx:
                idx = core_idx * out_loops + loop_idx
                with inst.if_scope(idx < self.tiling_param.out_dim):
                    input_gm_addr = self._get_input_gm_addr(idx)
                    with inst.if_scope(need_update_out_addr == 1):
                        need_update_out_addr.set_as(0)
                        output_gm_addr.set_as(self._get_output_gm_addr(idx))

                    with inst.if_scope(ub_data_count + inner_dim > ub_size):
                        self._data_move(output_gm[output_gm_addr], ub, ub_data_count)
                        ub_data_count.set_as(0)
                        output_gm_addr.set_as(self._get_output_gm_addr(idx))
                    self._data_move(tmp_ub, input_gm[input_gm_addr], self.block_element)

                    with inst.for_range(0, inner_dim) as index:
                        ub[ub_data_count + index] = tmp_ub[index]
                    ub_data_count.set_as(ub_data_count + inner_dim)

                    with inst.if_scope(loop_idx == out_loops - 1):
                        self._add_tail(ub, tmp_ub, idx, ub_data_count)
            with inst.if_scope(ub_data_count != 0):
                self._data_move(output_gm[output_gm_addr], ub, ub_data_count)

    def _do_large_last_dim_align(self, input_gm_addr, output_gm_addr, inner_loop_idx):
        """
        _do_large_last_dim_align
        """
        inst = self.tik_instance
        total = self.tiling_param.output_shape[self.shape_length - 1]
        input_gm = self.input_gm
        output_gm = self.output_gm
        count = inst.Scalar("int32", name="remain")
        with inst.new_stmt_scope():
            ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="out_ub")
            count.set_as(total - self.ub_size * inner_loop_idx)
            with inst.if_scope(count > self.ub_size):
                count.set_as(self.ub_size)

            self._data_move(ub, input_gm[input_gm_addr + inner_loop_idx * self.ub_size], count)
            self._data_move(output_gm[output_gm_addr + inner_loop_idx * self.ub_size], ub, count)

    # pylint: disable=too-many-locals,invalid-name
    def _do_large_last_dim_not_align(self, input_gm_addr, output_gm_addr, inner_loop_idx):
        """
        _do_large_last_dim_not_align
        """
        inst = self.tik_instance
        total = self.tiling_param.output_shape[self.shape_length - 1]
        input_gm = self.input_gm
        output_gm = self.output_gm
        count = inst.Scalar("int32", name="remain")
        with inst.new_stmt_scope():
            ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="out_ub")
            count.set_as(total - self.ub_size * inner_loop_idx)
            with inst.if_scope(count >= self.ub_size):
                self._data_move(ub, input_gm[input_gm_addr + inner_loop_idx * self.ub_size], self.ub_size)
                self._data_move(output_gm[output_gm_addr + inner_loop_idx * self.ub_size], ub, self.ub_size)
            with inst.else_scope():
                with inst.if_scope(inner_loop_idx > 0):
                    align_count = self._ceil_32bytes_count(count)
                    redundant_count = align_count - count
                    new_in_start_index = (input_gm_addr + inner_loop_idx * self.ub_size - redundant_count)
                    new_out_start_index = (output_gm_addr + inner_loop_idx * self.ub_size - redundant_count)
                    self._data_move(ub, input_gm[new_in_start_index:], align_count)
                    self._data_move(output_gm[new_out_start_index:], ub, align_count)
                with inst.else_scope():
                    in_start_index = (input_gm_addr + inner_loop_idx * self.ub_size)
                    out_start_index = (output_gm_addr + inner_loop_idx * self.ub_size)
                    self._data_move(ub, input_gm[in_start_index:], self.block_element)
                    self._data_move(output_gm[out_start_index:], ub, self.block_element)

                    in_start_index += self.block_element
                    out_start_index += self.block_element
                    align_count = self._ceil_32bytes_count(count - self.block_element)
                    redundant_count = align_count - count + self.block_element
                    new_in_start_index = in_start_index - redundant_count
                    new_out_start_index = out_start_index - redundant_count
                    self._data_move(ub, input_gm[new_in_start_index:], align_count)
                    self._data_move(output_gm[new_out_start_index:], ub, align_count)

    def _add_tail(self, ub, tmp_ub, idx, ub_data_count):
        """
        _add_tail
        """
        inst = self.tik_instance
        inner_dim = self.tiling_param.output_shape[self.shape_length - 1]
        out_dim = self.tiling_param.out_dim
        align_count = self._ceil_32bytes_count(ub_data_count)
        overlap_count = align_count - ub_data_count
        ext_rows = self._ceil_div(overlap_count, inner_dim)
        input_gm = self.input_gm
        with inst.for_range(1, ext_rows + 1, name="ext_row") as row_idx:
            with inst.if_scope(idx + row_idx < out_dim):
                input_addr = self._get_input_gm_addr(idx + row_idx)
                self._data_move(tmp_ub, input_gm[input_addr], self.block_element)
                with inst.for_range(0, inner_dim) as index:
                    with inst.if_scope(ub_data_count < align_count):
                        ub[ub_data_count] = tmp_ub[index]
                        ub_data_count.set_as(ub_data_count + 1)

    def _do_small_last_dim_with_vnchwconv(self, core_idx):
        """
        _do_small_last_dim_with_vnchwconv
        """
        inst = self.tik_instance
        core_num = self.aicore_num
        input_shape = self.tiling_param.input_shape
        output_shape = self.tiling_param.output_shape
        out_loops = self._ceil_div(self.tiling_param.out_dim_with_vnchwconv, core_num)
        compute_rows_each_inner_loops = self.ub_size_with_vnchwconv // (
            16 * input_shape[self.shape_length - 1]) // 16 * 16

        inner_loops = self._ceil_div(output_shape[self.shape_length - 2], compute_rows_each_inner_loops) - 1
        compute_rows_tail = output_shape[self.shape_length - 2] - inner_loops * compute_rows_each_inner_loops

        with inst.for_range(0, out_loops, name="out_loops") as out_loops_idx:
            idx = core_idx * out_loops + out_loops_idx
            with inst.if_scope(idx < self.tiling_param.out_dim_with_vnchwconv):
                input_gm_base_addr = self._get_input_base_gm_addr_with_vnchwconv(idx)
                output_gm_base_addr = self._get_output_base_gm_addr_with_vnchwconv(idx)
                with inst.for_range(0, inner_loops, name="inner_loops") as inner_loops_idx:
                    input_gm_addr = input_gm_base_addr + inner_loops_idx * \
                        compute_rows_each_inner_loops * input_shape[self.shape_length - 1]
                    output_gm_addr = output_gm_base_addr + inner_loops_idx * \
                        compute_rows_each_inner_loops * output_shape[self.shape_length - 1]

                    self._do_each_matrix_align(input_gm_addr, output_gm_addr, compute_rows_each_inner_loops)

                input_gm_addr = input_gm_base_addr + inner_loops * compute_rows_each_inner_loops * input_shape[
                    self.shape_length - 1]
                output_gm_addr = output_gm_base_addr + inner_loops * compute_rows_each_inner_loops * output_shape[
                    self.shape_length - 1]
                self._do_each_matrix_tail(input_gm_addr, output_gm_addr, compute_rows_tail, out_loops_idx, out_loops)

    def _get_input_base_gm_addr_with_vnchwconv(self, cur_index: tik.Scalar):
        """
        _get_input_base_gm_addr_vnchw
        """
        inst = self.tik_instance
        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        dim_count = self.tiling_param.shape_length
        addr = inst.Scalar(self.tiling_param.dtype, name="input_addr")
        addr.set_as(self.tiling_param.begin[dim_count - 2] * self.tiling_param.input_shape[dim_count - 1])
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")

        with inst.for_range(3, dim_count + 1) as dim_idx:
            dim = self.tiling_param.output_shape[dim_count - dim_idx]
            step = self.tiling_param.input_steps[dim_idx - 2]
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * (tmp + self.tiling_param.begin[dim_count - dim_idx]))
            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _get_output_base_gm_addr_with_vnchwconv(self, cur_index: tik.Scalar):
        """
        _get_output_base_gm_addr_vnchw
        """
        inst = self.tik_instance
        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        dim_count = self.tiling_param.shape_length
        addr = inst.Scalar(self.tiling_param.dtype, name="output_addr")
        addr.set_as(0)
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")

        with inst.for_range(3, dim_count + 1) as dim_idx:
            dim = self.tiling_param.output_shape[dim_count - dim_idx]
            step = self.tiling_param.output_steps[dim_idx - 2]
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * tmp)
            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _do_each_matrix_align(self, input_gm_addr, output_gm_addr, rows):
        """
        _do_each_matrix_align
        """
        inst = self.tik_instance
        output_matrix_count = self.tiling_param.output_shape[self.shape_length - 1] * rows
        with inst.new_stmt_scope():
            ub1 = inst.Tensor(self.dtype, (self.ub_size_with_vnchwconv,), scope=tik.scope_ubuf, name="ub1")
            ub2 = inst.Tensor(self.dtype, (self.ub_size_with_vnchwconv,), scope=tik.scope_ubuf, name="ub2")
            self._do_each_matrix_except_move_output_gm(input_gm_addr, output_gm_addr, rows, [ub1, ub2])
            self._data_move(self.output_gm[output_gm_addr], ub2, output_matrix_count)

    def _do_each_matrix_except_move_output_gm(self, input_gm_addr, output_gm_addr, rows, ub_list):
        """
        _do_each_matrix_except_move_output_gm
        """
        inst = self.tik_instance
        input_shape = self.tiling_param.input_shape
        output_shape = self.tiling_param.output_shape
        begin = self.tiling_param.begin
        ub1, ub2 = ub_list
        input_matrix_count = input_shape[self.shape_length - 1] * rows
        self._data_move(ub1, self.input_gm[input_gm_addr], input_matrix_count)

        # first vnchwconv_loop: ub1(32, 31) -> ub2(32 * 31, 16)
        vnchwconv_loop = self._ceil_div(input_matrix_count, 16)
        with inst.for_range(0, vnchwconv_loop) as i:
            src_addr = [ub1[16 * i + 16 * j] for j in range(16)]
            dst_addr = [ub2[16 * 16 * i + 16 * j] for j in range(16)]
            inst.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

        nburst_loop = rows // MAX_NBURST
        with inst.for_range(0, nburst_loop) as i:
            inst.data_move(
                ub1[i * MAX_NBURST * output_shape[self.shape_length - 1] * 16],
                ub2[(i * MAX_NBURST * input_shape[self.shape_length - 1] + begin[self.shape_length - 1]) * 16],
                0,
                MAX_NBURST,
                output_shape[self.shape_length - 1],
                input_shape[self.shape_length - 1] - output_shape[self.shape_length - 1],
                0)

        with inst.if_scope(rows % MAX_NBURST != 0):
            inst.data_move(
                ub1[nburst_loop * MAX_NBURST * output_shape[self.shape_length - 1] * 16],
                ub2[(nburst_loop * MAX_NBURST * input_shape[self.shape_length - 1] + begin[
                    self.shape_length - 1]) * 16],
                0,
                rows % MAX_NBURST,
                output_shape[self.shape_length - 1],
                input_shape[self.shape_length - 1] - output_shape[self.shape_length - 1],
                0)

        output_matrix_count = output_shape[self.shape_length - 1] * rows
        vnchwconv_loop = self._ceil_div(output_matrix_count, 16)
        with inst.for_range(0, vnchwconv_loop) as i:
            src_addr = [ub1[16 * 16 * i + 16 * j] for j in range(16)]
            dst_addr = [ub2[16 * i + 16 * j] for j in range(16)]
            inst.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

    def _do_each_matrix_tail(self, input_gm_addr, output_gm_addr, rows, out_loops_idx, out_loops):
        """
        _do_each_matrix_tail
        """
        inst = self.tik_instance
        output_matrix_count = self.tiling_param.output_shape[self.shape_length - 1] * rows
        with inst.new_stmt_scope():
            ub1 = inst.Tensor(self.dtype, (self.ub_size_with_vnchwconv,), scope=tik.scope_ubuf, name="ub1")
            ub2 = inst.Tensor(self.dtype, (self.ub_size_with_vnchwconv,), scope=tik.scope_ubuf, name="ub2")
            ub_block = inst.Tensor(self.dtype, (self.block_element,), scope=tik.scope_ubuf, name="ub_block")

            self._do_each_matrix_except_move_output_gm(input_gm_addr, output_gm_addr, rows, [ub1, ub2])

            with inst.if_scope(tik.all(out_loops_idx == out_loops - 1, output_matrix_count % self.block_element != 0)):
                floor_align_count = output_matrix_count // self.block_element * self.block_element
                self._data_move(self.output_gm[output_gm_addr], ub2, floor_align_count)
                with inst.for_range(0, self.block_element, name="block_element_loop") as element_id:
                    ub_block[element_id] = ub2[output_matrix_count - self.block_element + element_id]
                self._data_move(self.output_gm[output_gm_addr + output_matrix_count -
                                               self.block_element], ub_block, self.block_element)

            with inst.else_scope():
                self._data_move(self.output_gm[output_gm_addr], ub2, output_matrix_count)


# pylint: disable=locally-disabled,too-many-arguments,
# pylint: disable=unused-argument,too-many-locals
@register_operator("StridedSlice")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def strided_slice(input_x, begin, end, strides=None, output_x=None, begin_mask=0, end_mask=0,
                  ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, kernel_name="strided_slice"):
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
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "uint8", "bool", "int8")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    strided_slice_instance = StridedSlice(input_x, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                          shrink_axis_mask, kernel_name)
    strided_slice_instance.strided_slice()
    inst = strided_slice_instance.tik_instance
    opt_config = {"out_of_bound_sync_check": True}
    inst.BuildCCE(kernel_name=strided_slice_instance.kernel_name,
                  inputs=(strided_slice_instance.input_gm,
                          strided_slice_instance.begin_gm,
                          strided_slice_instance.end_gm,
                          strided_slice_instance.strides_gm),
                  outputs=(strided_slice_instance.output_gm,),
                  flowtable=[strided_slice_instance.tiling_param.tiling_gm],
                  config=opt_config,
                  enable_l2=False)

    tbe_context.get_context().add_compile_info("vars", {"block_dim": strided_slice_instance.aicore_num,
                                                        "begin_mask": strided_slice_instance.begin_mask,
                                                        "end_mask": strided_slice_instance.end_mask,
                                                        "ellipsis_mask": strided_slice_instance.ellipsis_mask,
                                                        "new_axis_mask": strided_slice_instance.new_axis_mask,
                                                        "shrink_axis_mask": strided_slice_instance.shrink_axis_mask,
                                                        "ub_size": tik.Dprofile().get_unified_buffer_size()})
    return inst
