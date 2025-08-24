#!/usr/bin/env python
# coding: utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
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
strided_slice_strides_larger_than_one
"""
import functools
import math
import te.platform as tbe_platform
from impl import common_util
from impl.util.util_tik_comm_func import floor_align
from impl.util.util_tik_comm_func import ceil_align
from impl.util.util_tik_comm_func import ceil_div
from impl.util.platform_adapter import tik

# the max value of repeat_times in vnchwconv
VNCHW_REPEAT_TIMES = 255
# max value of nburst in data_move
N_BURST = 4095
# max value of src_stride/dst_stride in data_move
MAX_STRIDE = 65535


# 'pylint: disable=too-few-public-methods,too-many-arguments,disable=too-many-branches,disable=too-many-statements
class StridedSliceStridesLargerThanOne:
    """
    StridedSliceStridesLargerThanOne
    """

    def __init__(self, input_shape, dtype, begin, end, strides, kernel_name):
        """
        init parameters.

        Returns
        -------
        None

        """
        self.tik_instance = tik.Tik()
        self.kernel_name = kernel_name
        self.dtype = dtype
        self.scalar_type = "int64"
        self.dtype_size = common_util.get_data_size(dtype)
        self.float16_type_size = common_util.get_data_size("float16")
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.input_shape = input_shape
        self.begin = begin
        self.end = end
        self.strides = strides
        self.output_shape = list(map(lambda x, y, z: math.ceil((x - y) / z), end, begin, strides))
        if self.dtype_size % self.float16_type_size == 1:
            self.multi_times = 1
            self.tensor_type = self.dtype
            self.tensor_type_size = self.dtype_size
            self.vnchwconv_column = 32
        else:
            self.multi_times = self.dtype_size // self.float16_type_size
            self.tensor_type = "float16"
            self.tensor_type_size = self.float16_type_size
            self.vnchwconv_column = 16
        self.output_count = functools.reduce(lambda x, y: x * y, self.output_shape[0:])
        self.total_ub_length = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // self.tensor_type_size
        self.element_each_block = common_util.constant.BLOCK_SIZE // self.tensor_type_size
        self.output_element_each_block = common_util.constant.BLOCK_SIZE // self.dtype_size
        self.output_count = ceil_align(self.output_count, self.output_element_each_block) * self.multi_times
        self.input_inner_need_count = ceil_align(end[-1] - begin[-1], strides[-1]) * self.multi_times
        self.input_inner_dim = ceil_align(self.input_inner_need_count, self.element_each_block)
        self.output_inner_dim = self.output_shape[-1] * self.multi_times
        self.shape_len = len(input_shape)
        if self.shape_len == 1:
            self.out_dims = 1
        else:
            self.out_dims = functools.reduce(lambda x, y: x * y, self.output_shape[0:-1])
        self.begin_value = begin[-1] * self.multi_times
        self.end_value = end[-1] * self.multi_times
        self.strides_value = strides[-1] * self.multi_times
        self.input_steps = []
        for i in range(self.shape_len):
            dim_idx = self.shape_len - 1 - i
            if i == 0:
                self.input_steps.append(self.input_shape[dim_idx] * self.multi_times)
            else:
                self.input_steps.append(self.input_shape[dim_idx])
            if i > 0:
                self.input_steps[i] = self.input_steps[i] * self.input_steps[i - 1]
        self.ub_size = floor_align(self.total_ub_length // 2, self.element_each_block)
        self.output_32bytes_align_rows = self.element_each_block
        if self.output_32bytes_align_rows % self.output_inner_dim == 0:
            self.output_32bytes_align_rows = self.output_32bytes_align_rows // self.output_inner_dim
        if self.output_inner_dim % self.element_each_block == 0:
            self.output_32bytes_align_rows = 1
        self.max_rows_in_ub = floor_align(self.ub_size // (self.input_inner_dim * self.vnchwconv_column),
                                          self.output_32bytes_align_rows)
        self.output_inner_dim_align = 0
        self.output_inner_num_in_ub = 0
        self.inner_loop_num = 0
        self.inner_loop_count = 0
        self.inner_loops = 0
        self.last_inner_loop_num = 0
        self.last_inner_loop_num_align = 0
        self.roll_back_num = 0
        self.last_inner_loop_count = 0
        self.rows_each_core = 0
        self.aicore_num_used = 0
        self.loop_times = 0
        self.last_loop_rows = 0
        self.tail_rows = 0
        self.tail_loop_times = 0
        self.tail_last_loop_rows = 0
        self.roll_back_count = 0
        self.gm_ub_count = 0
        self.repeat_times = 0
        self.rows_each_repeat = 0
        self.tail_rows_repeat_times = 0
        self.tail_rows_repeat_tail_count = 0
        self.out_ub_size = 0

        # whether a line of output data is less than 32B; value: 1, greater than 32B; value: 2, less than 32B.
        self.large_last_dim = 0
        # ub can store multiple lines of input data
        self.normal_last_dim = 0
        if self.max_rows_in_ub < 1:
            if self.output_inner_dim < self.element_each_block:
                self.large_last_dim = 2
            else:
                self.large_last_dim = 1
        else:
            self.normal_last_dim = 1
            last_dim_strides_flag = True
            for stride in self.strides[0:self.shape_len - 1]:
                if stride > 1:
                    last_dim_strides_flag = False
            if self.shape_len > 1 and last_dim_strides_flag:
                for i in range(1, self.shape_len - 1):
                    if self.input_shape[i] != self.output_shape[i]:
                        last_dim_strides_flag = False
            one_row_need_ub = self.input_shape[-1] * self.multi_times * self.vnchwconv_column
            max_rows = floor_align((self.ub_size - 32 * self.vnchwconv_column) // one_row_need_ub,
                                   self.output_32bytes_align_rows)
            if max_rows >= 2 and last_dim_strides_flag and \
                (self.input_shape[-1] * self.dtype_size < 256 or \
                    self.input_shape[-1] / ((self.output_shape[-1] - 1) * self.strides[-1] + 1) <= 2):
                self.normal_last_dim = 2

        if self.normal_last_dim == 1:
            self._get_normal_parameters()
        elif self.large_last_dim == 1:
            self._get_large_parameters()
        elif self.large_last_dim == 2:
            self._get_small_parameters()
        elif self.normal_last_dim == 2:
            self._get_last_strides_parameters()

        self.input_gm = self.tik_instance.Tensor(self.dtype, self.input_shape, scope=tik.scope_gm, name="input_gm")
        self.output_gm = self.tik_instance.Tensor(self.dtype, self.output_shape, scope=tik.scope_gm, name="output_gm")
        if self.large_last_dim != 2 and self.dtype_size % self.float16_type_size == 0:
            self.input_gm = self.input_gm.reinterpret_cast_to("float16")
            self.output_gm = self.output_gm.reinterpret_cast_to("float16")
        self.support_dmp = tbe_platform.api_check_support("tik.data_move_pad")
        self.block_bytes = 32

    def _gm2ub(self, inst, data_ub, data_gm, data_len):
        dtype_size = common_util.get_data_size(data_ub.dtype)
        ele_per_block = self.block_bytes // dtype_size
        if self.support_dmp is True:
            data_ub_b8 = data_ub.reinterpret_cast_to("int8")
            data_gm_b8 = data_gm.reinterpret_cast_to("int8")
            inst.data_move_pad(data_ub_b8, data_gm_b8, 1, data_len * dtype_size, 0, 0)
        else:
            inst.data_move(data_ub, data_gm, 0, 1, ceil_div(data_len, ele_per_block), 0, 0)

    def _ub2gm(self, inst, data_gm, data_ub, data_len):
        dtype_size = common_util.get_data_size(data_ub.dtype)
        ele_per_block = self.block_bytes // dtype_size
        if self.support_dmp is True:
            data_ub_b8 = data_ub.reinterpret_cast_to("int8")
            data_gm_b8 = data_gm.reinterpret_cast_to("int8")
            inst.data_move_pad(data_gm_b8, data_ub_b8, 1, data_len * dtype_size, 0, 0)
        else:
            inst.data_move(data_gm, data_ub, 0, 1, ceil_div(data_len, ele_per_block), 0, 0)

    def _get_last_strides_parameters(self):
        """
        init parameters, on the condition only the last dim stride is lager then one.
        And a column ub can fit multiple rows of input data.

        Returns
        -------
        None

        """
        self.input_inner_dim = self.input_shape[-1] * self.multi_times
        one_row_need_ub = self.input_inner_dim * self.vnchwconv_column
        self.max_rows_in_ub = floor_align((self.ub_size - 32 * self.vnchwconv_column) // one_row_need_ub,
                                          self.output_32bytes_align_rows)
        self.rows_each_core = ceil_align(ceil_div(self.out_dims, self.aicore_num), self.output_32bytes_align_rows)
        self.repeat_times = ceil_div(self.rows_each_core, self.max_rows_in_ub)
        self.rows_each_repeat = ceil_align(ceil_div(self.rows_each_core, self.repeat_times),
                                           self.output_32bytes_align_rows)
        self.rows_each_core = self.rows_each_repeat * self.repeat_times
        self.aicore_num_used = ceil_div(self.out_dims, self.rows_each_core)
        if self.aicore_num_used == 1:
            self.rows_each_core = self.out_dims
            self.rows_each_repeat = min(self.rows_each_repeat, self.out_dims)
        self.loop_times = ceil_div(self.repeat_times, 16)
        self.last_loop_rows = self.repeat_times % 16
        if self.last_loop_rows == 0:
            self.last_loop_rows = 16
        self.tail_rows = self.out_dims % self.rows_each_core
        self.tail_rows_repeat_times = ceil_div(self.tail_rows, self.rows_each_repeat)
        self.tail_rows_repeat_tail_count = self.tail_rows % self.rows_each_repeat
        if self.tail_rows_repeat_tail_count == 0:
            self.tail_rows_repeat_tail_count = self.rows_each_repeat
        if self.tail_rows_repeat_times == 1:
            self.tail_loop_times = 1
        else:
            self.tail_loop_times = ceil_div(self.tail_rows_repeat_times - 1, 16)
        self.tail_last_loop_rows = (self.tail_rows_repeat_times - 1) % 16
        if self.tail_last_loop_rows == 0 and self.tail_rows_repeat_times > 1:
            self.tail_last_loop_rows = 16

    def _get_small_parameters(self):
        """
        init large parameters. When a column ub can not fit a row of input data,
        and a line of output data is less than 32B.

        Returns
        -------
        None
        """
        if self.dtype_size % self.float16_type_size == 0:
            self.total_ub_length = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // self.dtype_size
            self.element_each_block = common_util.constant.BLOCK_SIZE // self.dtype_size
            self.input_inner_dim = ceil_align(ceil_align(self.end[-1] - self.begin[-1], self.strides[-1]),
                                              self.element_each_block)
            self.output_inner_dim = self.output_shape[-1]
            self.output_32bytes_align_rows = self.element_each_block
            if self.output_32bytes_align_rows % self.output_inner_dim == 0:
                self.output_32bytes_align_rows = self.output_32bytes_align_rows // self.output_inner_dim
            if self.output_inner_dim % self.element_each_block == 0:
                self.output_32bytes_align_rows = 1
            self.begin_value = self.begin[-1]
            self.input_steps = []
            for i in range(self.shape_len):
                dim_idx = self.shape_len - 1 - i
                self.input_steps.append(self.input_shape[dim_idx])
                if i > 0:
                    self.input_steps[i] = self.input_steps[i] * self.input_steps[i - 1]

        self.out_ub_size = self.element_each_block * self.output_32bytes_align_rows
        self.ub_size = floor_align(self.total_ub_length - self.out_ub_size, self.element_each_block)
        if self.input_inner_dim > self.ub_size:
            self.inner_loops = ceil_div(self.input_inner_dim, self.ub_size)
            self.inner_loop_num = ceil_div(self.output_inner_dim, self.inner_loops)
        else:
            self.inner_loops = 1
            self.inner_loop_num = self.output_inner_dim
        self.inner_loop_count = self.inner_loop_num * self.strides[-1]
        self.ub_size = ceil_align(self.inner_loop_count, self.element_each_block)
        self.out_ub_size = floor_align(self.total_ub_length - self.ub_size, self.element_each_block)
        self.rows_each_core = ceil_div(self.out_dims, self.aicore_num)

    def _get_large_parameters(self):
        """
        init large parameters. When a column ub can not fit a row of input data,
        and a line of output data is greater than or equal to 32B.

        Returns
        -------
        None
        """
        self.output_inner_dim_align = floor_align(self.output_inner_dim, self.element_each_block)
        self.output_inner_num_in_ub = floor_align(self.ub_size // (self.strides[-1] * self.vnchwconv_column),
                                                  self.element_each_block)
        if self.output_inner_dim_align < self.output_inner_num_in_ub:
            self.inner_loop_num = self.output_inner_dim_align
        else:
            self.inner_loop_num = self.output_inner_num_in_ub
        self.inner_loop_count = self.inner_loop_num * self.strides[-1]
        self.inner_loops = ceil_div(self.output_inner_dim, self.inner_loop_num)
        self.last_inner_loop_num = self.output_inner_dim % self.inner_loop_num
        if self.last_inner_loop_num == 0:
            self.last_inner_loop_num = self.inner_loop_num
        self.last_inner_loop_num_align = ceil_align(self.last_inner_loop_num, self.element_each_block)
        self.roll_back_num = self.last_inner_loop_num_align - self.last_inner_loop_num
        self.last_inner_loop_count = self.last_inner_loop_num_align * self.strides[-1]
        self.rows_each_core = ceil_div(self.out_dims, self.aicore_num)
        self.aicore_num_used = ceil_div(self.out_dims, self.rows_each_core)
        if self.aicore_num_used == 1:
            self.rows_each_core = self.out_dims
        self.loop_times = ceil_div(self.rows_each_core, 16)
        self.last_loop_rows = self.rows_each_core % 16
        if self.last_loop_rows == 0:
            self.last_loop_rows = 16
        self.tail_rows = self.out_dims % self.rows_each_core
        self.tail_loop_times = ceil_div(self.tail_rows, 16)
        self.tail_last_loop_rows = self.tail_rows % 16
        if self.tail_last_loop_rows == 0:
            self.tail_last_loop_rows = 16

    def _get_normal_parameters(self):
        """
        init normal parameters. When a column ub can fit multiple rows of input data.

        Returns
        -------
        None

        """
        self.roll_back_count = self.input_inner_dim - self.input_inner_need_count
        self.gm_ub_count = (self.output_shape[-1] - 1) * self.strides[-1] * self.multi_times + 1 * self.multi_times
        self.rows_each_core = ceil_align(ceil_div(self.out_dims, self.aicore_num), self.output_32bytes_align_rows)
        self.repeat_times = ceil_div(self.rows_each_core, self.max_rows_in_ub)
        self.rows_each_repeat = ceil_align(ceil_div(self.rows_each_core, self.repeat_times),
                                           self.output_32bytes_align_rows)
        self.rows_each_core = self.rows_each_repeat * self.repeat_times
        self.aicore_num_used = ceil_div(self.out_dims, self.rows_each_core)
        if self.aicore_num_used == 1:
            self.rows_each_core = self.out_dims
            if self.rows_each_repeat > self.out_dims:
                self.rows_each_repeat = self.out_dims
            if self.rows_each_repeat == 1:
                self.input_inner_dim = self.end_value - self.begin_value
                self.roll_back_count = 0
        self.loop_times = ceil_div(self.repeat_times, 16)
        self.last_loop_rows = self.repeat_times % 16
        if self.last_loop_rows == 0:
            self.last_loop_rows = 16
        self.tail_rows = self.out_dims % self.rows_each_core
        self.tail_rows_repeat_times = ceil_div(self.tail_rows, self.rows_each_repeat)
        self.tail_rows_repeat_tail_count = self.tail_rows % self.rows_each_repeat
        if self.tail_rows_repeat_tail_count == 0:
            self.tail_rows_repeat_tail_count = self.rows_each_repeat
        if self.tail_rows_repeat_times == 1:
            self.tail_loop_times = 1
        else:
            self.tail_loop_times = ceil_div(self.tail_rows_repeat_times - 1, 16)
        self.tail_last_loop_rows = (self.tail_rows_repeat_times - 1) % 16
        if self.tail_last_loop_rows == 0 and self.tail_rows_repeat_times > 1:
            self.tail_last_loop_rows = 16

    def _do_slice_with_vnchwconv_last_strides(self):
        """
        slice data when a column ub can fit multiple rows of input data,
        on the condition only the last dim stride is lager then one.

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.aicore_num_used, block_num=self.aicore_num_used) as blk_idx:
            if self.tail_rows != 0:
                with self.tik_instance.if_scope(blk_idx < self.aicore_num_used - 1):
                    with self.tik_instance.for_range(0, self.loop_times - 1) as loop_idx:
                        row_id = blk_idx * self.rows_each_core + loop_idx * 16 * self.rows_each_repeat
                        self._do_with_vnchwconv_last_strides_per_loop(self.rows_each_repeat, row_id, 16)
                    row_id = blk_idx * self.rows_each_core + (self.loop_times - 1) * 16 * self.rows_each_repeat
                    self._do_with_vnchwconv_last_strides_per_loop(self.rows_each_repeat, row_id, self.last_loop_rows)
                with self.tik_instance.else_scope():
                    if self.tail_rows_repeat_times > 1:
                        with self.tik_instance.for_range(0, self.tail_loop_times - 1) as loop_idx:
                            row_id = blk_idx * self.rows_each_core + loop_idx * 16 * self.rows_each_repeat
                            self._do_with_vnchwconv_last_strides_per_loop(self.rows_each_repeat, row_id, 16)
                        row_id = blk_idx * self.rows_each_core + (self.tail_loop_times - 1) * 16 * \
                                 self.rows_each_repeat
                        self._do_with_vnchwconv_last_strides_per_loop(self.rows_each_repeat, row_id,
                                                                      self.tail_last_loop_rows)
                    row_id = blk_idx * self.rows_each_core + (self.tail_loop_times - 1) * 16 * \
                             self.rows_each_repeat + self.tail_last_loop_rows * self.rows_each_repeat
                    self._do_with_vnchwconv_last_strides_per_loop(self.tail_rows_repeat_tail_count, row_id, 1)
            else:
                with self.tik_instance.for_range(0, self.loop_times - 1) as loop_idx:
                    row_id = blk_idx * self.rows_each_core + loop_idx * 16 * self.rows_each_repeat
                    self._do_with_vnchwconv_last_strides_per_loop(self.rows_each_repeat, row_id, 16)
                row_id = blk_idx * self.rows_each_core + (self.loop_times - 1) * 16 * self.rows_each_repeat
                self._do_with_vnchwconv_last_strides_per_loop(self.rows_each_repeat, row_id, self.last_loop_rows)

    # 'pylint: disable=too-many-locals
    def _do_with_vnchwconv_last_strides_per_loop(self, rows_each_repeat, row_id, loop_rows):
        """
        slice multiple rows of input data at one loop
        Parameters
        ----------
        rows_each_repeat: number of input rows stored in one column of ub
        row_id: id of output row
        loop_rows: number of columns stored in ub

        Returns
        -------
        None
        """
        inst = self.tik_instance
        output_addr = row_id * self.output_inner_dim
        with inst.new_stmt_scope():
            input_ub = inst.Tensor(self.tensor_type, (self.ub_size, ), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(self.tensor_type, (self.ub_size, ), scope=tik.scope_ubuf, name="vnchwconv_ub")
            vnchw_conv_repeat_times = ceil_div(rows_each_repeat * self.input_inner_dim, self.vnchwconv_column)
            loop_count = ceil_align(rows_each_repeat * self.input_inner_dim, self.element_each_block)
            with inst.for_range(0, loop_rows) as loop_rows_idx:
                output_idx = row_id + loop_rows_idx * rows_each_repeat
                src_addr_in = self._get_input_gm_addr(output_idx) - self.begin_value
                dst_addr_in = loop_rows_idx * loop_count
                self._gm2ub(inst, input_ub[dst_addr_in], self.input_gm[src_addr_in],
                            rows_each_repeat * self.input_inner_dim)
            self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times)
            with inst.for_range(0, rows_each_repeat) as rows_idx:
                src_addr = rows_idx * self.input_inner_dim * self.element_each_block + \
                           self.begin_value * self.element_each_block
                dst_addr = self.output_inner_dim * rows_idx * self.element_each_block
                inst.data_move(input_ub[dst_addr:], vnchw_conv_ub[src_addr:], 0, self.output_shape[-1],
                               self.multi_times, self.strides_value - self.multi_times, 0)
            vnchw_conv_repeat_times = ceil_div(rows_each_repeat * self.output_inner_dim, self.vnchwconv_column)
            self._do_with_vnchwconv2output(vnchw_conv_ub, input_ub, vnchw_conv_repeat_times)
            with inst.if_scope(output_addr + loop_rows * rows_each_repeat * self.output_inner_dim < self.out_dims *
                               self.output_inner_dim):
                count = loop_rows * rows_each_repeat * self.output_inner_dim
                self._ub2gm(inst, self.output_gm[output_addr], vnchw_conv_ub, count)
            with inst.else_scope():
                self._ub2gm(inst, self.output_gm[output_addr], vnchw_conv_ub,
                            self.out_dims * self.output_inner_dim - output_addr)

    def _do_slice_with_vnchwconv_small(self):
        """
        slice data when a column ub can not fit a row of input data,
        and a line of output data is less than 32B.

        Returns
        -------
        None
        """
        inst = self.tik_instance
        with inst.for_range(0, self.aicore_num, block_num=self.aicore_num) as blk_idx:
            with inst.new_stmt_scope():
                input_ub = inst.Tensor(self.dtype, (self.ub_size, ), scope=tik.scope_ubuf, name="input_ub")
                out_ub = inst.Tensor(self.dtype, (self.out_ub_size, ), scope=tik.scope_ubuf, name="out_ub")
                ub_data_count = inst.Scalar("int32", name="ub_data_count")
                ub_data_count.set_as(0)
                dst_addr = inst.Scalar("int32", name="dst_addr")
                dst_addr.set_as(blk_idx * self.rows_each_core * self.output_inner_dim)
                count = inst.Scalar("int32", name="count")
                count.set_as(0)
                with inst.for_range(0, self.rows_each_core) as loop_idx:
                    row_id = blk_idx * self.rows_each_core + loop_idx
                    with inst.if_scope(row_id < self.out_dims):
                        input_gm_addr = self._get_input_gm_addr(row_id)
                        with inst.for_range(0, self.inner_loops) as inner_loop_idx:
                            src_addr = input_gm_addr + inner_loop_idx * self.inner_loop_count
                            count.set_as(self.output_inner_dim - self.inner_loop_num * inner_loop_idx)
                            with inst.if_scope(count > self.inner_loop_num):
                                count.set_as(self.inner_loop_num)
                            with inst.if_scope(ub_data_count + count > self.out_ub_size):
                                self._ub2gm(inst, self.output_gm[dst_addr], out_ub, ub_data_count)
                                dst_addr.set_as(blk_idx * self.rows_each_core * self.output_inner_dim + \
                                                loop_idx * self.output_inner_dim + \
                                                inner_loop_idx * self.inner_loop_num)
                                ub_data_count.set_as(0)
                            self._gm2ub(inst, input_ub, self.input_gm[src_addr], (count - 1) * self.strides[-1] + 1)
                            with inst.for_range(0, count) as count_idx:
                                out_ub[ub_data_count + count_idx] = input_ub[count_idx * self.strides[-1]]
                            ub_data_count.set_as(ub_data_count + count)
                        with inst.if_scope(loop_idx == self.rows_each_core - 1):
                            self._add_tail(input_ub, out_ub, row_id, ub_data_count)
                with inst.if_scope(ub_data_count > 0):
                    self._ub2gm(inst, self.output_gm[dst_addr], out_ub, ub_data_count)

    # 'pylint: disable=too-many-locals
    def _add_tail(self, input_ub, out_ub, row_id, ub_data_count):
        """
        additional data rows are processed so that the output data is 32B aligned.
        Parameters
        ----------
        input_ub: ub for storing input data
        out_ub: ub for storing output data
        row_id: id of output row
        ub_data_count: data count in output ub

        Returns
        -------
        None
        """
        inst = self.tik_instance
        tmp_count = inst.Scalar("int32", name="tmp_count")
        tmp_count.set_as(ub_data_count)
        align_count = ceil_align(tmp_count, self.element_each_block)
        overlap_count = align_count - tmp_count
        ext_rows = ceil_div(overlap_count, self.output_inner_dim)
        count_tail = inst.Scalar("int32", name="count_tail")
        count_tail.set_as(0)
        with inst.for_range(1, ext_rows + 1) as idx:
            with inst.if_scope(row_id + idx < self.out_dims):
                input_gm_addr = self._get_input_gm_addr(row_id + idx)
                with inst.for_range(0, self.inner_loops) as inner_loop_idx:
                    src_addr = input_gm_addr + inner_loop_idx * self.inner_loop_count
                    count_tail.set_as(self.output_inner_dim - self.inner_loop_num * inner_loop_idx)
                    with inst.if_scope(count_tail > self.inner_loop_num):
                        count_tail.set_as(self.inner_loop_num)
                    self._gm2ub(inst, input_ub, self.input_gm[src_addr], (count_tail - 1) * self.strides[-1] + 1)
                    with inst.for_range(0, count_tail) as count_idx:
                        with inst.if_scope(ub_data_count < align_count):
                            out_ub[ub_data_count] = input_ub[count_idx * self.strides[-1]]
                            ub_data_count.set_as(ub_data_count + 1)

    def _do_slice_with_vnchwconv_large(self):
        """
        slice data when a column ub can not fit a row of input data.

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.aicore_num_used, block_num=self.aicore_num_used) as blk_idx:
            if self.tail_rows == 0:
                with self.tik_instance.for_range(0, self.loop_times - 1) as loop_idx:
                    row_id = blk_idx * self.rows_each_core + loop_idx * 16
                    self._do_with_vnchwconv_per_loop_large(row_id, 16)
                row_id = blk_idx * self.rows_each_core + (self.loop_times - 1) * 16
                self._do_with_vnchwconv_per_loop_large(row_id, self.last_loop_rows)
            else:
                with self.tik_instance.if_scope(blk_idx < self.aicore_num_used - 1):
                    with self.tik_instance.for_range(0, self.loop_times - 1) as loop_idx:
                        row_id = blk_idx * self.rows_each_core + loop_idx * 16
                        self._do_with_vnchwconv_per_loop_large(row_id, 16)
                    row_id = blk_idx * self.rows_each_core + (self.loop_times - 1) * 16
                    self._do_with_vnchwconv_per_loop_large(row_id, self.last_loop_rows)
                with self.tik_instance.for_range(0, self.tail_loop_times - 1) as tail_loop_idx:
                    row_id = (self.aicore_num_used - 1) * self.rows_each_core + tail_loop_idx * 16
                    self._do_with_vnchwconv_per_loop_large(row_id, 16)
                row_id = (self.aicore_num_used - 1) * self.rows_each_core + (self.tail_loop_times - 1) * 16
                self._do_with_vnchwconv_per_loop_large(row_id, self.tail_last_loop_rows)

    # 'pylint: disable=too-many-locals
    # 'pylint: disable=too-many-arguments
    def _do_with_vnchwconv_per_loop_large(self, row_id, loop_rows):
        """
        slice a row of input data multiple times
        Parameters
        ----------
        row_id: id of output row
        loop_rows: number of output rows processed at one loop

        Returns
        -------
        None
        """
        inst = self.tik_instance
        with inst.for_range(0, self.inner_loops - 1) as inner_loops_idx:
            self._do_with_vnchwconv_per_inner_loop_large(row_id, loop_rows, inner_loops_idx, self.inner_loop_count, 0)
        self._do_with_vnchwconv_per_inner_loop_large(row_id, loop_rows, self.inner_loops - 1,
                                                     self.last_inner_loop_count, self.roll_back_num)

    # 'pylint: disable=too-many-locals
    def _do_with_vnchwconv_per_inner_loop_large(self, row_id, loop_rows, inner_loops_idx, loop_count, roll_back_num):
        """
        slice data with the instruction of vnchwconv
        Parameters
        ----------
        row_id: id of output row
        loop_rows: number of output rows processed at one loop
        inner_loops_idx: id of input data loops
        loop_count: number of data processed at one loop
        roll_back_num: roll back num, in order to output data align of 32B

        Returns
        -------
        None
        """
        inst = self.tik_instance
        output_addr = self.output_inner_dim * row_id
        loop_num = loop_count // self.strides[-1]
        with inst.new_stmt_scope():
            input_ub = inst.Tensor(self.tensor_type, (self.ub_size, ), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(self.tensor_type, (self.ub_size, ), scope=tik.scope_ubuf, name="vnchwconv_ub")
            vnchw_conv_repeat_times = loop_count // self.vnchwconv_column
            with inst.for_range(0, loop_rows) as loop_rows_idx:
                output_idx = row_id + loop_rows_idx
                src_addr_in = self._get_input_gm_addr(output_idx) + inner_loops_idx * self.inner_loop_count - \
                              roll_back_num * self.strides[-1]
                dst_addr_in = loop_rows_idx * loop_count
                self._gm2ub(inst, input_ub[dst_addr_in], self.input_gm[src_addr_in], loop_count)
            self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times)
            nburst_length = loop_count // self.strides_value
            nburst_loop = ceil_div(nburst_length, N_BURST)
            nburst_tail = nburst_length % N_BURST
            if nburst_tail == 0:
                nburst_tail = N_BURST
            with inst.for_range(0, nburst_loop - 1) as loop_idx:
                inst.data_move(
                    input_ub[N_BURST * self.multi_times * self.element_each_block * loop_idx:],
                    vnchw_conv_ub[N_BURST * loop_idx * self.strides[-1] * self.multi_times * self.element_each_block:],
                    0, N_BURST, self.multi_times, (self.strides[-1] - 1) * self.multi_times, 0)
            inst.data_move(
                input_ub[N_BURST * self.multi_times * self.element_each_block * (nburst_loop - 1):],
                vnchw_conv_ub[N_BURST * (nburst_loop - 1) * self.strides[-1] * self.multi_times *
                              self.element_each_block:], 0, nburst_tail, self.multi_times,
                (self.strides[-1] - 1) * self.multi_times, 0)
            vnchw_conv_repeat_times = ceil_div(loop_num, self.vnchwconv_column)
            self._do_with_vnchwconv2output(vnchw_conv_ub, input_ub, vnchw_conv_repeat_times)
            tmp_stride = (self.inner_loops - 1) * self.inner_loop_num // self.element_each_block
            if self.roll_back_num == 0 and tmp_stride <= MAX_STRIDE:
                dst_addr_out = output_addr + inner_loops_idx * self.inner_loop_num
                dst_stride = inst.Scalar("int32", name="dst_stride", init_value=0)
                with inst.if_scope(tik.all(self.inner_loops > 1, inner_loops_idx == self.inner_loops - 1)):
                    dst_stride.set_as((self.inner_loops - 1) * self.inner_loop_num // self.element_each_block)
                with inst.elif_scope(self.inner_loops > 1):
                    dst_stride.set_as(((self.inner_loops - 2) * self.inner_loop_num + self.last_inner_loop_num) //
                                      self.element_each_block)
                inst.data_move(self.output_gm[dst_addr_out], vnchw_conv_ub, 0, loop_rows, vnchw_conv_repeat_times, 0,
                               dst_stride)
            else:
                with inst.for_range(0, loop_rows) as loop_rows_idx:
                    dst_addr_out = (output_addr + loop_rows_idx * self.output_inner_dim +
                                    inner_loops_idx * self.inner_loop_num - roll_back_num)
                    self._ub2gm(inst, self.output_gm[dst_addr_out], vnchw_conv_ub[loop_num * loop_rows_idx], loop_num)

    def _do_slice_with_vnchwconv_normal(self):
        """
        slice data when a column ub can fit multiple rows of input data.

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.aicore_num_used, block_num=self.aicore_num_used) as blk_idx:
            if self.tail_rows != 0:
                with self.tik_instance.if_scope(blk_idx < self.aicore_num_used - 1):
                    with self.tik_instance.for_range(0, self.loop_times - 1) as loop_idx:
                        row_id = blk_idx * self.rows_each_core + loop_idx * 16 * self.rows_each_repeat
                        self._do_with_vnchwconv_per_loop(self.rows_each_repeat, row_id, 16)
                    row_id = blk_idx * self.rows_each_core + (self.loop_times - 1) * 16 * self.rows_each_repeat
                    self._do_with_vnchwconv_per_loop(self.rows_each_repeat, row_id, self.last_loop_rows)
                with self.tik_instance.else_scope():
                    if self.tail_rows_repeat_times > 1:
                        with self.tik_instance.for_range(0, self.tail_loop_times - 1) as loop_idx:
                            row_id = blk_idx * self.rows_each_core + loop_idx * 16 * self.rows_each_repeat
                            self._do_with_vnchwconv_per_loop(self.rows_each_repeat, row_id, 16)
                        row_id = blk_idx * self.rows_each_core + (self.tail_loop_times - 1) * 16 * \
                                 self.rows_each_repeat
                        self._do_with_vnchwconv_per_loop(self.rows_each_repeat, row_id, self.tail_last_loop_rows)
                    row_id = blk_idx * self.rows_each_core + (self.tail_loop_times - 1) * 16 * \
                             self.rows_each_repeat + self.tail_last_loop_rows * self.rows_each_repeat
                    self._do_with_vnchwconv_per_loop(self.tail_rows_repeat_tail_count, row_id, 1)
            else:
                with self.tik_instance.for_range(0, self.loop_times - 1) as loop_idx:
                    row_id = blk_idx * self.rows_each_core + loop_idx * 16 * self.rows_each_repeat
                    self._do_with_vnchwconv_per_loop(self.rows_each_repeat, row_id, 16)
                row_id = blk_idx * self.rows_each_core + (self.loop_times - 1) * 16 * self.rows_each_repeat
                self._do_with_vnchwconv_per_loop(self.rows_each_repeat, row_id, self.last_loop_rows)

    def _do_with_vnchwconv_per_loop(self, rows_each_repeat, row_id, loop_rows):
        """
        slice multiple rows of input data at one loop
        Parameters
        ----------
        rows_each_repeat: number of input rows stored in one column of ub
        row_id: id of output row
        loop_rows: number of columns stored in ub

        Returns
        -------
        None
        """
        inst = self.tik_instance
        output_addr = row_id * self.output_inner_dim
        with inst.new_stmt_scope():
            input_ub = inst.Tensor(self.tensor_type, (self.ub_size, ), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(self.tensor_type, (self.ub_size, ), scope=tik.scope_ubuf, name="vnchwconv_ub")
            vnchw_conv_repeat_times = ceil_div(rows_each_repeat * self.input_inner_dim, self.vnchwconv_column)
            with inst.for_range(0, loop_rows) as loop_rows_idx:
                with inst.for_range(0, rows_each_repeat) as rows_idx:
                    output_idx = row_id + loop_rows_idx * rows_each_repeat + rows_idx
                    src_addr_in = self._get_input_gm_addr(output_idx)
                    dst_addr_in = (loop_rows_idx * rows_each_repeat + rows_idx) * self.input_inner_dim
                    self._gm2ub(inst, input_ub[dst_addr_in], self.input_gm[src_addr_in], self.gm_ub_count)
            loop_count = ceil_align(rows_each_repeat * self.input_inner_dim, self.element_each_block)
            self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times)
            if self.roll_back_count > 0:
                inst.data_move(input_ub, vnchw_conv_ub, 0, rows_each_repeat, self.input_inner_need_count,
                               self.roll_back_count, 0)
                self._do_with_vnchwconv_ub2gm(input_ub, vnchw_conv_ub, output_addr, loop_rows, rows_each_repeat)
            else:
                self._do_with_vnchwconv_ub2gm(vnchw_conv_ub, input_ub, output_addr, loop_rows, rows_each_repeat)

    # 'pylint: disable=too-many-arguments
    def _do_with_vnchwconv_ub2gm(self, vnchw_conv_ub, input_ub, output_addr, loop_rows, rows_each_repeat):
        """
        rearrange data in ub with vnchwconv instruction, and move data form ub to gm
        Parameters
        ----------
        vnchw_conv_ub: data before rearrangement
        input_ub: data after rearrangement
        output_addr: the output addr when move data from ub to gm
        rows_each_repeat: number of input rows stored in one column of ub
        loop_rows: number of columns calculated by ub

        Returns
        -------
        None
        """
        inst = self.tik_instance
        nburst_length = rows_each_repeat * self.output_shape[-1]
        nburst_loop = ceil_div(nburst_length, N_BURST)
        nburst_tail = nburst_length % N_BURST
        if nburst_tail == 0:
            nburst_tail = N_BURST
        with inst.for_range(0, nburst_loop - 1) as loop_idx:
            inst.data_move(input_ub[N_BURST * self.multi_times * self.element_each_block * loop_idx:],
                           vnchw_conv_ub[N_BURST * loop_idx * self.strides[-1] * self.multi_times * \
                                         self.element_each_block:],
                           0, N_BURST, self.multi_times, (self.strides[-1] - 1) * self.multi_times, 0)
        inst.data_move(input_ub[N_BURST * self.multi_times * self.element_each_block * (nburst_loop - 1):],
                       vnchw_conv_ub[N_BURST * (nburst_loop - 1) * self.strides[-1] * self.multi_times \
                                     * self.element_each_block:],
                       0, nburst_tail, self.multi_times, (self.strides[-1] - 1) * self.multi_times, 0)
        vnchw_conv_repeat_times = ceil_div(rows_each_repeat * self.output_inner_dim, self.vnchwconv_column)
        self._do_with_vnchwconv2output(vnchw_conv_ub, input_ub, vnchw_conv_repeat_times)
        with inst.if_scope(output_addr + loop_rows * rows_each_repeat * self.output_inner_dim <= self.output_count):
            count = loop_rows * rows_each_repeat * self.output_inner_dim
            self._ub2gm(inst, self.output_gm[output_addr], vnchw_conv_ub, count)
        with inst.else_scope():
            count = self.output_count - output_addr
            self._ub2gm(inst, self.output_gm[output_addr], vnchw_conv_ub, count)

    def _get_input_gm_addr(self, cur_index):
        """
        get input data address according to the number of output data lines
        Parameters
        ----------
        cur_index: the row id of output data lines

        Returns
        -------
        addr: input data address
        """
        dim_count = self.shape_len
        tmp_cur_index = cur_index
        addr = self.begin_value
        for dim_idx in range(2, dim_count + 1):
            dim = self.output_shape[dim_count - dim_idx]
            tmp_out_idx = tmp_cur_index % dim
            tmp_begin_idx = self.begin[dim_count - dim_idx] + tmp_out_idx * self.strides[dim_count - dim_idx]
            addr = addr + tmp_begin_idx * self.input_steps[dim_idx - 2]
            tmp_cur_index = tmp_cur_index // dim
        return addr

    def _do_with_vnchwconv2output(self, vnchw_conv_ub, input_ub, vnchw_conv_repeat_times):
        """
        rearrange data in ub with vnchwconv instruction
        Parameters
        ----------
        vnchw_conv_ub: data after rearrangement
        input_ub: data before rearrangement
        vnchw_conv_repeat_times: the repeat times of vnchwconv instruction

        Returns
        -------
        None
        """
        inst = self.tik_instance
        if self.dtype_size % self.float16_type_size == 0:
            dst_list = [vnchw_conv_ub[i * vnchw_conv_repeat_times * self.element_each_block] for i in range(16)]
            src_list = [input_ub[i * self.element_each_block] for i in range(16)]
            repeat_loop_times = ceil_div(vnchw_conv_repeat_times, VNCHW_REPEAT_TIMES)
            repeat_last_times = vnchw_conv_repeat_times % VNCHW_REPEAT_TIMES
            if repeat_last_times == 0:
                repeat_last_times = VNCHW_REPEAT_TIMES
            if repeat_last_times == 1:
                inst.vnchwconv(False, False, dst_list, src_list, repeat_last_times, 0, 0)
            else:
                inst.vnchwconv(False, False, dst_list, src_list, repeat_last_times, 1, 16)
            with inst.for_range(0, repeat_loop_times - 1) as repeat_loop_idx:
                src_offset = repeat_last_times * 16 * self.element_each_block + \
                            VNCHW_REPEAT_TIMES * 16 * self.element_each_block * repeat_loop_idx
                dst_offset = repeat_last_times * self.element_each_block + \
                            VNCHW_REPEAT_TIMES * self.element_each_block * repeat_loop_idx
                dst_list = [vnchw_conv_ub[dst_offset + i * vnchw_conv_repeat_times * self.element_each_block] \
                            for i in range(16)]
                src_list = [input_ub[src_offset + i * self.element_each_block] for i in range(16)]
                inst.vnchwconv(False, False, dst_list, src_list, VNCHW_REPEAT_TIMES, 1, 16)
        else:
            dst_list = [vnchw_conv_ub[i * vnchw_conv_repeat_times * self.element_each_block] for i in range(16)]
            src_list = [input_ub[i * self.element_each_block] for i in range(16)]
            if vnchw_conv_repeat_times == 1:
                inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
            else:
                inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 1, self.vnchwconv_column)
            dst_list = [vnchw_conv_ub[i * vnchw_conv_repeat_times * self.element_each_block] for i in range(16)]
            src_list = [input_ub[32 * 16 + i * self.element_each_block] for i in range(16)]
            if vnchw_conv_repeat_times == 1:
                inst.vnchwconv(True, False, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
            else:
                inst.vnchwconv(True, False, dst_list, src_list, vnchw_conv_repeat_times, 1, self.vnchwconv_column)

    def _do_with_input2vnchwconv(self, vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times):
        """
        rearrange data in ub with vnchwconv instruction
        Parameters
        ----------
        vnchw_conv_ub: data after rearrangement
        input_ub: data before rearrangement
        loop_count: number of data processed in one column of ub
        vnchw_conv_repeat_times: the repeat times of vnchwconv instruction

        Returns
        -------
        None
        """
        inst = self.tik_instance
        if self.dtype_size % self.float16_type_size == 0:
            dst_list = [vnchw_conv_ub[i * self.element_each_block] for i in range(16)]
            src_list = [input_ub[i * loop_count] for i in range(16)]
            repeat_loop_times = ceil_div(vnchw_conv_repeat_times, VNCHW_REPEAT_TIMES)
            repeat_last_times = vnchw_conv_repeat_times % VNCHW_REPEAT_TIMES
            if repeat_last_times == 0:
                repeat_last_times = VNCHW_REPEAT_TIMES
            if repeat_last_times == 1:
                inst.vnchwconv(False, False, dst_list, src_list, repeat_last_times, 0, 0)
            else:
                inst.vnchwconv(False, False, dst_list, src_list, repeat_last_times, 16, 1)
            with inst.for_range(0, repeat_loop_times - 1) as repeat_loop_idx:
                dst_offset = repeat_last_times * 16 * self.element_each_block + \
                             VNCHW_REPEAT_TIMES * repeat_loop_idx * 16 * self.element_each_block
                src_offset = repeat_last_times * self.element_each_block + \
                             VNCHW_REPEAT_TIMES * self.element_each_block * repeat_loop_idx
                dst_list = [vnchw_conv_ub[dst_offset + i * self.element_each_block] for i in range(16)]
                src_list = [input_ub[src_offset + i * loop_count] for i in range(16)]
                inst.vnchwconv(False, False, dst_list, src_list, VNCHW_REPEAT_TIMES, 16, 1)
        else:
            dst_list = [vnchw_conv_ub[i * self.element_each_block] for i in range(16)]
            src_list = [input_ub[i * loop_count] for i in range(16)]
            if vnchw_conv_repeat_times == 1:
                inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
            else:
                inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, self.vnchwconv_column, 1)
            dst_list = [vnchw_conv_ub[32 * 16 + i * self.element_each_block] for i in range(16)]
            src_list = [input_ub[i * loop_count] for i in range(16)]
            if vnchw_conv_repeat_times == 1:
                inst.vnchwconv(False, True, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
            else:
                inst.vnchwconv(False, True, dst_list, src_list, vnchw_conv_repeat_times, self.vnchwconv_column, 1)

    def compute(self):
        """
        Compute entry of strided_slice

        Returns
        -------
        tik_instance: tik_instance
        """
        if self.normal_last_dim == 1:
            self._do_slice_with_vnchwconv_normal()
        elif self.large_last_dim == 1:
            self._do_slice_with_vnchwconv_large()
        elif self.large_last_dim == 2:
            self._do_slice_with_vnchwconv_small()
        elif self.normal_last_dim == 2:
            self._do_slice_with_vnchwconv_last_strides()

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}

        self.tik_instance.BuildCCE(self.kernel_name,
                                   inputs=[self.input_gm],
                                   outputs=[self.output_gm],
                                   config=opt_config)
        return self.tik_instance


# 'pylint: disable=too-many-arguments
def strided_slice_strides_larger_than_one(input_shape, dtype, begin, end, strides, kernel_name):
    """
    strided_slice for strides larger than one

    Returns
    -------
    tik_instance: tik_instance
    """
    strides_not_equal_one_function = StridedSliceStridesLargerThanOne(input_shape, dtype, begin, end, strides,
                                                                      kernel_name)
    return strides_not_equal_one_function.compute()
