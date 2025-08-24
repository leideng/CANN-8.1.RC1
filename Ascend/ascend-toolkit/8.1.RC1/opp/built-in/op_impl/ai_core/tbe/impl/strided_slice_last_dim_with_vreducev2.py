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
strided_slice_last_dim_with_vreducev2
"""
import functools

from impl import common_util
from impl.util.util_tik_comm_func import floor_align
from impl.util.util_tik_comm_func import ceil_align
from impl.util.util_tik_comm_func import ceil_div
from impl.util.util_tik_comm_func import gm2ub
from impl.util.util_tik_comm_func import ub2gm
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform

# the number of bits per byte
BYTE_SIZE = 8
DATA_SIZE = 512
# the max value of nburst in data_move_pad
N_BURST = 4095
# the max value of src_gap/dst_gap in data_move_pad
MAX_GAP = 65535


class StridedSliceLastDimWithVreducev2:
    """
    StridedSliceLastDimWithVreducev2
    """

    def __init__(self, input_shape, dtype, begin, end, stride, kernel_name):
        """
        init parameters.

        Returns
        -------
        None

        """
        self.tik_instance = tik.Tik()
        self.profile = tik.Dprofile()
        self.kernel_name = kernel_name
        self.dtype = dtype
        self.dtype_size = common_util.get_data_size(dtype)
        self.total_ub_length = self.profile.get_unified_buffer_size() // self.dtype_size
        self.aicore_num = self.profile.get_aicore_num()
        self.element_each_block = common_util.constant.BLOCK_SIZE // self.dtype_size
        # the number of bits of dtype
        self.mask_bit = self.dtype_size * BYTE_SIZE
        self.repeat_rows_mask_num = 0
        self.repeat_tail_mask_num = 0
        self.input_inner_dims = input_shape[-1]
        self.output_inner_dims = (end[-1] - begin[-1]) // stride[-1]
        self.output_inner_dims_32b_align = ceil_align(self.output_inner_dims, self.element_each_block)
        if len(input_shape) == 1:
            self.out_dims = 1
        else:
            self.out_dims = functools.reduce(lambda x, y: x * y, input_shape[0:-1])
        self.input_gm = self.tik_instance.Tensor(self.dtype, (self.out_dims, self.input_inner_dims),
                                                 scope=tik.scope_gm,
                                                 name="input_gm")
        self.output_gm = self.tik_instance.Tensor(self.dtype, (self.out_dims, self.output_inner_dims),
                                                  scope=tik.scope_gm,
                                                  name="output_gm")
        self.begin_value = begin[-1]
        self.end_value = end[-1]
        self.src_gap = (self.input_inner_dims - self.end_value + self.begin_value) * self.dtype_size
        self.mask_num = ceil_div(self.input_inner_dims, self.mask_bit)
        self.thread_num = 2
        self.ub_size = self.total_ub_length // self.thread_num
        self.max_rows_in_ub = (self.ub_size - self.element_each_block * 2) // \
                              (self.input_inner_dims + self.mask_num * 2)
        self.input_ub_size = ceil_align(self.max_rows_in_ub * self.input_inner_dims, self.element_each_block)
        self.pattern_ub_size = ceil_align(self.max_rows_in_ub * self.mask_num, self.element_each_block)
        self.output_32b_align_rows = self.element_each_block
        if self.element_each_block % self.output_inner_dims == 0:
            self.output_32b_align_rows = self.element_each_block // self.output_inner_dims
        elif self.output_inner_dims % self.element_each_block == 0:
            self.output_32b_align_rows = 1

        if self.dtype_size == 2:
            self.mask_dtype = common_util.constant.DATA_TYPE_UINT16
        elif self.dtype_size == 4:
            self.mask_dtype = common_util.constant.DATA_TYPE_UINT32

        self.rows_each_core = ceil_align(ceil_div(self.out_dims, self.aicore_num), self.output_32b_align_rows)
        self.aicore_num_used = ceil_div(self.out_dims, self.rows_each_core)
        self.tail_rows = self.out_dims % self.rows_each_core
        if self.aicore_num_used == 1:
            self.rows_each_core = self.out_dims
            self.tail_rows = 0

        # whether a line of output data exceeds the size of ub; default value:0, not exceed.
        self.big_last_dim = 0
        # whether to use the vreducev2 instruction; default value:0, used vreducev2.
        self.vreducev2_flag = 1
        # whether to use the data_move instruction or data_move_pad instruction to move data from ub to gm;
        # default value:0, used data_move, when the data in ub is 32B aligned.
        self.vreducev2_and_datamove_flag = 1

        if self.input_inner_dims > floor_align(self.ub_size, self.element_each_block):
            self.vreducev2_flag = 0
        if self.vreducev2_flag == 0:
            self.max_rows_in_ub = self.total_ub_length // self.output_inner_dims_32b_align
            if self.max_rows_in_ub == 0 or self.src_gap > MAX_GAP or \
                self.output_inner_dims * self.dtype_size > MAX_GAP:
                self.big_last_dim = 1
                self.max_rows_in_ub = 1
                self.input_ub_size = floor_align(self.total_ub_length, self.element_each_block)
            else:
                self.input_ub_size = ceil_align(self.max_rows_in_ub * self.output_inner_dims_32b_align,
                                                self.element_each_block)
            self.rows_each_core = ceil_div(self.out_dims, self.aicore_num)
            self.aicore_num_used = ceil_div(self.out_dims, self.rows_each_core)
            self.tail_rows = self.out_dims % self.rows_each_core
            if self.aicore_num_used == 1:
                self.rows_each_core = self.out_dims
                self.tail_rows = 0

        if self.vreducev2_flag == 1:
            pattern_tensor_dict_each_core = self.get_pattern_tensor(self.rows_each_core)
            self.rows_each_repeat_tensor_each_core = pattern_tensor_dict_each_core.get("rows_each_repeat_tensor")
            self.repeat_rows_mask_num_each_core = pattern_tensor_dict_each_core.get("repeat_rows_mask_num")
            self.thread_num_each_core = pattern_tensor_dict_each_core.get("thread_num")
            self.rows_each_repeat_each_core = pattern_tensor_dict_each_core.get("rows_each_repeat")
            self.repeat_times_each_core = pattern_tensor_dict_each_core.get("repeat_times")
            self.roll_back_rows_each_core = pattern_tensor_dict_each_core.get("roll_back_rows")
            self.rows_repeat_core_gm = self.tik_instance.Tensor(self.mask_dtype,
                                                                (len(self.rows_each_repeat_tensor_each_core),),
                                                                name="rows_repeat_core_gm",
                                                                scope=tik.scope_gm,
                                                                init_value=self.rows_each_repeat_tensor_each_core)
            self.rows_repeat_tail_gm = self.tik_instance.Tensor(self.mask_dtype,
                                                                (len(self.rows_each_repeat_tensor_each_core),),
                                                                name="rows_repeat_tail_gm",
                                                                scope=tik.scope_gm,
                                                                init_value=self.rows_each_repeat_tensor_each_core)
            self.rows_each_repeat_tensor_tail_row = self.rows_each_repeat_tensor_each_core

            if self.tail_rows != 0:
                pattern_tensor_dict_tail_rows = self.get_pattern_tensor(self.tail_rows)
                self.rows_each_repeat_tensor_tail_row = pattern_tensor_dict_tail_rows.get("rows_each_repeat_tensor")
                self.repeat_rows_mask_num_tail_row = pattern_tensor_dict_tail_rows.get("repeat_rows_mask_num")
                self.thread_num_tail_row = pattern_tensor_dict_tail_rows.get("thread_num")
                self.rows_each_repeat_tail_row = pattern_tensor_dict_tail_rows.get("rows_each_repeat")
                self.repeat_times_tail_row = pattern_tensor_dict_tail_rows.get("repeat_times")
                self.roll_back_rows_tail_row = pattern_tensor_dict_tail_rows.get("roll_back_rows")
                self.rows_repeat_tail_gm = self.tik_instance.Tensor(self.mask_dtype,
                                                                    (len(self.rows_each_repeat_tensor_tail_row),),
                                                                    name="rows_repeat_tail_gm",
                                                                    scope=tik.scope_gm,
                                                                    init_value=self.rows_each_repeat_tensor_tail_row)

    def _gm2ub(self, inst, input_ub, input_gm, data_len):
        if tbe_platform.api_check_support("tik.data_move_pad", self.dtype) is True:
            inst.data_move_pad(input_ub, input_gm, 1, data_len * self.dtype_size, 0, 0)
        else:
            gm2ub(inst, input_ub, input_gm, data_len)

    def get_mask_tensor_and_num(self, repeat_rows, mask_str_list_one_line):
        """
        get the src1_pattern and mask of vreducev2 instruction.

        Returns
        -------
        mask_tensor: src1_pattern of vreducev2 instruction
        mask_num: mask number of vreducev2 instruction

        """
        mask_str_repeat_rows = mask_str_list_one_line * repeat_rows
        mask_num = len(mask_str_repeat_rows)
        mask_tensor_num = ceil_div(mask_num, self.mask_bit)
        mask_tensor = []
        for i in range(0, mask_tensor_num - 1):
            begin_bit = i * self.mask_bit
            end_bit = (i + 1) * self.mask_bit
            mask_bit = ''.join(reversed(mask_str_repeat_rows[begin_bit:end_bit]))
            mask_tensor.append(int(mask_bit, 2))
        mask_bit = ''.join(reversed(mask_str_repeat_rows[((mask_tensor_num - 1) * self.mask_bit):]))
        mask_tensor.append(int(mask_bit, 2))
        return mask_tensor, mask_num

    def get_pattern_tensor(self, to_do_rows):
        """
        get the src1_pattern and mask of vreducev2 instruction.

        Returns
        -------
        pattern_tensor_dict, the parameters are as follows:
        rows_each_repeat_tensor: src1_pattern of vreducev2 instruction
        repeat_rows_mask_num: mask number of vreducev2 instruction
        thread_num: doublebuffer
        rows_each_repeat: number of rows processed at a repeat
        repeat_times: the time of repeats per core
        roll_back_rows: the number of rows that need to be rolled back in the last repeat

        """
        repeat_params_dict = self.get_repeat_params(to_do_rows)
        mask_str_list_left = ['0' for i in range(self.begin_value)]
        mask_str_list_middle = ['1' for i in range(self.end_value - self.begin_value)]
        mask_str_list_right = ['0' for i in range(self.input_inner_dims - self.end_value)]
        mask_str_list_one_line = ''.join(mask_str_list_left) + ''.join(mask_str_list_middle) + ''.join(
            mask_str_list_right)
        rows_each_repeat_tensor, repeat_rows_mask_num = self.get_mask_tensor_and_num(
            repeat_params_dict["rows_each_repeat"], mask_str_list_one_line)
        pattern_tensor_dict = {
            "rows_each_repeat_tensor": rows_each_repeat_tensor,
            "repeat_rows_mask_num": repeat_rows_mask_num,
            "thread_num": repeat_params_dict.get("thread_num"),
            "rows_each_repeat": repeat_params_dict.get("rows_each_repeat"),
            "repeat_times": repeat_params_dict.get("repeat_times"),
            "roll_back_rows": repeat_params_dict.get("roll_back_rows")
        }
        return pattern_tensor_dict

    def get_repeat_params(self, to_do_rows):
        """
        get the repeat params in each core,
        when uses vreducev2 instruction to process.

        Returns
        -------
        repeat_params_dict, the parameters are as follows:
        thread_num: doublebuffer
        repeat_times: the number of repeats per core
        rows_each_repeat: number of rows processed at a repeat
        roll_back_rows: the number of rows that need to be rolled back in the last repeat

        """
        max_rows = self.max_rows_in_ub
        max_rows = min(max_rows, to_do_rows)
        thread_num = min(to_do_rows, self.thread_num)
        repeat_times = ceil_align(ceil_div(to_do_rows, max_rows), thread_num)
        if to_do_rows < repeat_times:
            repeat_times = to_do_rows
        rows_each_repeat = ceil_align(to_do_rows // repeat_times, self.output_32b_align_rows)
        if rows_each_repeat > max_rows:
            rows_each_repeat = floor_align(to_do_rows // repeat_times, self.output_32b_align_rows)
            if rows_each_repeat == 0:
                self.vreducev2_and_datamove_flag = 0
                rows_each_repeat = max_rows
        repeat_times = ceil_div(to_do_rows, rows_each_repeat)
        repeat_tail_count = to_do_rows % rows_each_repeat
        roll_back_rows = 0
        if repeat_tail_count != 0 and repeat_times != 1:
            roll_back_rows = rows_each_repeat - repeat_tail_count
            repeat_tail_count = rows_each_repeat
        elif repeat_tail_count == 0:
            repeat_tail_count = rows_each_repeat

        repeat_params_dict = {
            "thread_num": thread_num,
            "rows_each_repeat": rows_each_repeat,
            "repeat_times": repeat_times,
            "roll_back_rows": roll_back_rows
        }

        return repeat_params_dict

    def vreducev2_process(self, blk_idx, repeat_rows_mask_num, thread_num, rows_each_repeat, repeat_times,
                          roll_back_rows, gm_flag):
        """
        the last dim of input_shape is smaller than 1/2 ub,
        uses vreducev2 instruction to process.

        Returns
        -------
        None
        """
        input_addr = self.rows_each_core * self.input_inner_dims * blk_idx
        output_addr = self.rows_each_core * self.output_inner_dims * blk_idx
        curr_rows = self.tik_instance.Scalar(dtype="int64", name="curr_rows", init_value=rows_each_repeat)

        gm_flag = self.tik_instance.Scalar(dtype="int32", name="gm_flag", init_value=gm_flag)
        pattern_ub_repeat = self.tik_instance.Tensor(self.mask_dtype, (self.pattern_ub_size,),
                                                     scope=tik.scope_ubuf,
                                                     name="pattern_ub_repeat")
        with self.tik_instance.if_scope(gm_flag == 0):
            self._gm2ub(self.tik_instance, pattern_ub_repeat, self.rows_repeat_core_gm,
                        len(self.rows_each_repeat_tensor_each_core))
        with self.tik_instance.else_scope():
            self._gm2ub(self.tik_instance, pattern_ub_repeat, self.rows_repeat_tail_gm,
                        len(self.rows_each_repeat_tensor_tail_row))

        with self.tik_instance.new_stmt_scope():
            if roll_back_rows == 0:
                thread_num = min(repeat_times, thread_num)
                with self.tik_instance.for_range(0, repeat_times, thread_num=thread_num) as repeat_idx:
                    input_ub = self.tik_instance.Tensor(self.dtype, (self.input_ub_size,),
                                                        scope=tik.scope_ubuf,
                                                        name="input_ub")
                    with self.tik_instance.if_scope((self.rows_each_core * blk_idx + repeat_idx * rows_each_repeat +
                                                     curr_rows) > self.out_dims):
                        self._gm2ub(self.tik_instance, input_ub,
                                    self.input_gm[input_addr + repeat_idx * rows_each_repeat * self.input_inner_dims],
                                    (self.out_dims - (self.rows_each_core * blk_idx + repeat_idx * rows_each_repeat)) *
                                    self.input_inner_dims)
                    with self.tik_instance.else_scope():
                        self._gm2ub(self.tik_instance, input_ub,
                                    self.input_gm[input_addr + repeat_idx * rows_each_repeat * self.input_inner_dims],
                                    curr_rows * self.input_inner_dims)
                    self.tik_instance.vreducev2(repeat_rows_mask_num, input_ub, input_ub, pattern_ub_repeat, 1, 1, 0, 0,
                                                None, "counter")
                    if self.vreducev2_and_datamove_flag == 1:
                        with self.tik_instance.if_scope(
                            (output_addr + rows_each_repeat * repeat_idx * self.output_inner_dims +
                             self.output_inner_dims * curr_rows) > self.out_dims * self.output_inner_dims):
                            ub2gm(
                                self.tik_instance,
                                self.output_gm[output_addr + rows_each_repeat * repeat_idx * self.output_inner_dims],
                                input_ub, self.output_inner_dims * self.out_dims -
                                (output_addr + rows_each_repeat * repeat_idx * self.output_inner_dims))
                        with self.tik_instance.else_scope():
                            ub2gm(self.tik_instance,
                                  self.output_gm[output_addr + rows_each_repeat * repeat_idx * self.output_inner_dims],
                                  input_ub, self.output_inner_dims * curr_rows)
                    else:
                        self.tik_instance.data_move_pad(
                            self.output_gm[output_addr + rows_each_repeat * repeat_idx * self.output_inner_dims],
                            input_ub, 1, self.output_inner_dims * curr_rows * self.dtype_size, 0, 0)
            else:
                thread_num = min(repeat_times - 1, thread_num)
                if thread_num == 0:
                    thread_num = 1
                with self.tik_instance.for_range(0, repeat_times - 1, thread_num=thread_num) as repeat_idx:
                    input_ub = self.tik_instance.Tensor(self.dtype, (self.input_ub_size,),
                                                        scope=tik.scope_ubuf,
                                                        name="input_ub")
                    self._gm2ub(self.tik_instance, input_ub,
                                self.input_gm[input_addr + repeat_idx * rows_each_repeat * self.input_inner_dims],
                                curr_rows * self.input_inner_dims)
                    self.tik_instance.vreducev2(repeat_rows_mask_num, input_ub, input_ub, pattern_ub_repeat, 1, 1, 0, 0,
                                                None, "counter")
                    if self.vreducev2_and_datamove_flag == 1:
                        ub2gm(self.tik_instance,
                              self.output_gm[output_addr + rows_each_repeat * repeat_idx * self.output_inner_dims],
                              input_ub, self.output_inner_dims * curr_rows)
                    else:
                        self.tik_instance.data_move_pad(
                            self.output_gm[output_addr + rows_each_repeat * repeat_idx * self.output_inner_dims],
                            input_ub, 1, self.output_inner_dims * curr_rows * self.dtype_size, 0, 0)
                input_ub = self.tik_instance.Tensor(self.dtype, (self.input_ub_size,),
                                                    scope=tik.scope_ubuf,
                                                    name="input_ub")
                self._gm2ub(
                    self.tik_instance, input_ub,
                    self.input_gm[input_addr + (repeat_times - 1) * rows_each_repeat * self.input_inner_dims -
                                  roll_back_rows * self.input_inner_dims], curr_rows * self.input_inner_dims)
                self.tik_instance.vreducev2(repeat_rows_mask_num, input_ub, input_ub, pattern_ub_repeat, 1, 1, 0, 0,
                                            None, "counter")
                if self.vreducev2_and_datamove_flag == 1:
                    ub2gm(
                        self.tik_instance,
                        self.output_gm[output_addr + rows_each_repeat * (repeat_times - 1) * self.output_inner_dims -
                                       roll_back_rows * self.output_inner_dims], input_ub,
                        self.output_inner_dims * curr_rows)
                else:
                    self.tik_instance.data_move_pad(
                        self.output_gm[output_addr + rows_each_repeat * (repeat_times - 1) * self.output_inner_dims -
                                       roll_back_rows * self.output_inner_dims], input_ub, 1,
                        self.output_inner_dims * curr_rows * self.dtype_size, 0, 0)

    def do_with_vreducev2(self):
        """
        Multi core processing with vreducev2 instruction.

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.aicore_num_used, block_num=self.aicore_num_used) as blk_idx:
            if self.tail_rows != 0:
                with self.tik_instance.if_scope(blk_idx < self.aicore_num_used - 1):
                    self.vreducev2_process(blk_idx, self.repeat_rows_mask_num_each_core, self.thread_num_each_core,
                                           self.rows_each_repeat_each_core, self.repeat_times_each_core,
                                           self.roll_back_rows_each_core, 0)
                with self.tik_instance.if_scope(blk_idx == self.aicore_num_used - 1):
                    self.vreducev2_process(blk_idx, self.repeat_rows_mask_num_tail_row, self.thread_num_tail_row,
                                           self.rows_each_repeat_tail_row, self.repeat_times_tail_row,
                                           self.roll_back_rows_tail_row, 1)
            else:
                self.vreducev2_process(blk_idx, self.repeat_rows_mask_num_each_core, self.thread_num_each_core,
                                       self.rows_each_repeat_each_core, self.repeat_times_each_core,
                                       self.roll_back_rows_each_core, 0)

    def data_move_pad_process(self, blk_idx, to_do_rows):
        """
        the last dim of output_shape is smaller than ub, and the last dim of input_shape is larger than 1/2 ub,
        uses data_move_pad instruction to move in and out.

        Returns
        -------
        None
        """
        max_rows = min(self.max_rows_in_ub, N_BURST)
        repeat_times = ceil_div(to_do_rows, max_rows)
        rows_each_repeat = ceil_div(to_do_rows, repeat_times)
        if rows_each_repeat > max_rows:
            rows_each_repeat = to_do_rows // repeat_times
            repeat_times = ceil_div(to_do_rows, rows_each_repeat)
            rows_each_repeat = ceil_div(to_do_rows, repeat_times)
        repeat_times = ceil_div(to_do_rows, rows_each_repeat)
        repeat_tail_count = to_do_rows % rows_each_repeat
        if repeat_tail_count == 0:
            repeat_tail_count = rows_each_repeat
        input_addr = self.rows_each_core * self.input_inner_dims * blk_idx + self.begin_value
        output_addr = self.rows_each_core * self.output_inner_dims * blk_idx
        curr_rows = self.tik_instance.Scalar(dtype="int64", name="curr_rows", init_value=rows_each_repeat)

        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, repeat_times) as repeat_idx:
                with self.tik_instance.if_scope(repeat_idx == repeat_times - 1):
                    curr_rows.set_as(repeat_tail_count)
                input_ub = self.tik_instance.Tensor(self.dtype, (self.input_ub_size,),
                                                    scope=tik.scope_ubuf,
                                                    name="input_ub")
                self.tik_instance.data_move_pad(
                    input_ub, self.input_gm[input_addr + rows_each_repeat * repeat_idx * self.input_inner_dims],
                    curr_rows, self.output_inner_dims * self.dtype_size, 0, self.src_gap)
                self.tik_instance.data_move_pad(
                    self.output_gm[output_addr + rows_each_repeat * repeat_idx * self.output_inner_dims], input_ub,
                    curr_rows, self.output_inner_dims * self.dtype_size, 0, 0)

    def do_with_data_move_pad(self):
        """
        Multi core processing with data_move_pad instruction.

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.aicore_num_used, block_num=self.aicore_num_used) as blk_idx:
            if self.tail_rows != 0:
                with self.tik_instance.if_scope(blk_idx == self.aicore_num_used - 1):
                    self.data_move_pad_process(blk_idx, self.tail_rows)
                with self.tik_instance.else_scope():
                    self.data_move_pad_process(blk_idx, self.rows_each_core)
            else:
                self.data_move_pad_process(blk_idx, self.rows_each_core)

    def data_move_process(self, blk_idx, to_do_rows):
        """
        the last dim of output_shape is larger than ub, and uses data_move instruction to move in and out.

        Returns
        -------
        None
        """
        repeat_times = ceil_div(to_do_rows, self.max_rows_in_ub)
        rows_each_repeat = ceil_div(to_do_rows, repeat_times)
        input_addr = self.rows_each_core * self.input_inner_dims * blk_idx + self.begin_value
        output_addr = self.rows_each_core * self.output_inner_dims * blk_idx
        num_each_loop = floor_align(self.total_ub_length, self.element_each_block)
        one_line_loop_times = ceil_div(self.output_inner_dims, num_each_loop)
        loop_tail_count = (num_each_loop if self.output_inner_dims % num_each_loop == 0 else self.output_inner_dims %
                           num_each_loop)
        loop_tail_count_32b_align = ceil_align(loop_tail_count, self.element_each_block)
        curr_roll_back_num = self.tik_instance.Scalar(dtype="int64", name="curr_roll_back_num",
                                                      init_value=loop_tail_count_32b_align - loop_tail_count)
        with self.tik_instance.if_scope(one_line_loop_times == 1):
            curr_roll_back_num.set_as(0)

        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, repeat_times) as repeat_idx:
                repeat_input_addr = input_addr + rows_each_repeat * repeat_idx * self.input_inner_dims
                repeat_output_addr = output_addr + rows_each_repeat * repeat_idx * self.output_inner_dims

                with self.tik_instance.for_range(0, one_line_loop_times) as loop_time_idx:
                    input_ub = self.tik_instance.Tensor(self.dtype, (self.input_ub_size,),
                                                        scope=tik.scope_ubuf,
                                                        name="input_ub")
                    with self.tik_instance.if_scope(loop_time_idx == one_line_loop_times - 1):
                        if tbe_platform.api_check_support("tik.data_move_pad", self.dtype) is False:
                            gm2ub(self.tik_instance, input_ub,
                                  self.input_gm[repeat_input_addr + loop_time_idx * num_each_loop -
                                                curr_roll_back_num],
                                  loop_tail_count_32b_align)
                            ub2gm(self.tik_instance,
                                  self.output_gm[repeat_output_addr + loop_time_idx * num_each_loop -
                                                 curr_roll_back_num],
                                  input_ub, loop_tail_count_32b_align)
                        else:
                            self.tik_instance.data_move_pad(input_ub,
                                                            self.input_gm[repeat_input_addr +
                                                                          loop_time_idx * num_each_loop],
                                                            1, loop_tail_count * self.dtype_size, 0, 0)
                            self.tik_instance.data_move_pad(self.output_gm[repeat_output_addr +
                                                                           loop_time_idx * num_each_loop],
                                                            input_ub,
                                                            1, loop_tail_count * self.dtype_size, 0, 0)
                    with self.tik_instance.else_scope():
                        gm2ub(self.tik_instance, input_ub,
                              self.input_gm[repeat_input_addr + loop_time_idx * num_each_loop], num_each_loop)
                        ub2gm(self.tik_instance, self.output_gm[repeat_output_addr + loop_time_idx * num_each_loop],
                              input_ub, num_each_loop)

    def do_with_data_move(self):
        """
        Multi core processing with data_move instruction.

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.aicore_num_used, block_num=self.aicore_num_used) as blk_idx:
            if self.tail_rows != 0:
                with self.tik_instance.if_scope(blk_idx == self.aicore_num_used - 1):
                    self.data_move_process(blk_idx, self.tail_rows)
                with self.tik_instance.else_scope():
                    self.data_move_process(blk_idx, self.rows_each_core)
            else:
                self.data_move_process(blk_idx, self.rows_each_core)

    def compute_start(self):
        """
        Compute entry of strideslice

        Returns
        -------
        tik_instance: tik_instance
        """
        if self.vreducev2_flag == 1:
            self.do_with_vreducev2()
        elif self.big_last_dim == 0:
            self.do_with_data_move_pad()
        else:
            self.do_with_data_move()

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}

        if self.vreducev2_flag == 1:
            tbe_context.get_context().add_compile_info("global_variable_link", True)

        self.tik_instance.BuildCCE(self.kernel_name,
                                   inputs=[self.input_gm],
                                   outputs=[self.output_gm],
                                   config=opt_config)
        return self.tik_instance


def strided_slice_last_dim_with_vreducev2(input_shape, dtype, begin, end, stride, kernel_name):
    """
    strided slice for only last dim to slice

    Returns
    -------
    tik_instance: tik_instance
    """
    last_dim_with_vreducev2_function = StridedSliceLastDimWithVreducev2(input_shape, dtype, begin, end, stride,
                                                                        kernel_name)
    return last_dim_with_vreducev2_function.compute_start()
