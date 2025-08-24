#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
rotary_mul
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform.platform_info import get_soc_spec


# 'pylint: disable=huawei-too-many-arguments
class Constant:
    MAX_INT32 = 2 ** 31 - 1
    TILING_SCALAR_DTYPE = "int64"
    TILING_PARAMS_NUM = 22
    TILING_MODE_NO_BROADCAST = 0
    TILING_MODE_BROADCAST_SEP = 1
    TILING_MODE_BROADCAST = 2
    TILING_MODE_BROADCAST_ALL = 3
    TILING_MODE_BROADCAST_REPEAT = 4
    TILING_MODE_SINGLE_BROADCAST = 5
    MASK_32 = 64
    MASK_16 = 128
    DATA_EACH_BLOCK_32 = 8
    DATA_EACH_BLOCK_16 = 16
    MOVE_STRIDE_MAX = 65535


class RotaryMul():
    '''RotaryMul'''
    def __init__(self, x, r1, r2):
        self.tik_instance = tik.Tik()
        # get inputs' dtype.
        self.dtype = x.get("dtype").lower()

        # get soc's aicore num and ub size.
        self.aicore_num = get_soc_spec("CORE_NUM")
        self.ub_size = get_soc_spec("UB_SIZE")

        # init inputs and output gm.
        self.init_inputs_and_output_gm()

        # init tiling gm and scalar.
        self.init_tiling_gm_and_scalar()

        if self.dtype == "float16":
            self.mask = Constant.MASK_16
            self.data_each_block = Constant.DATA_EACH_BLOCK_16
        else:
            self.mask = Constant.MASK_32
            self.data_each_block = Constant.DATA_EACH_BLOCK_32
        
        if self.dtype == "bfloat16":
            self.ub_dtype = "float32"
        else:
            self.ub_dtype = self.dtype
    
    def init_tiling_gm_and_scalar(self):
        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_SCALAR_DTYPE, [Constant.TILING_PARAMS_NUM],
                                                  tik.scope_gm, "tiling_gm")
        self.tiling_mode = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "tiling_mode")
        self.last_dim = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "last_dim")
        self.total_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "total_num")
        self.broadcast_first_dim = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "broadcast_first_dim")
        self.broadcast_second_dim = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "broadcast_second_dim")
        self.tiling_core_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "tiling_core_num")
        self.data_align = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "data_align")
        self.task_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "task_num")
        self.tail_align = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "tail_align")
        self.task_ub_size = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "task_ub_size")
        self.r1_move_rep_times = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "r1_move_rep_times")
        self.repeat_cycle = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "repeat_cycle")
        self.repeat_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "repeat_tail")
        self.left_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "left_num")
        self.task_ub_size_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "task_ub_size_tail")
        self.r1_move_rep_times_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "r1_move_rep_times_tail")
        self.repeat_cycle_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "repeat_cycle_tail")
        self.repeat_tail_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "repeat_tail_tail")
        self.left_num_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "left_num_tail")
        self.move_stride = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_stride")
        self.x_half_burst = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "x_half_burst")
        self.x_burst = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "x_burst")

    def init_inputs_and_output_gm(self):
        self.x_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], tik.scope_gm, "x_gm")
        self.r1_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], tik.scope_gm, "r1_gm")
        self.r2_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], tik.scope_gm, "r2_gm")
        self.output_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], tik.scope_gm, "output_gm")

    def get_tiling_data(self):
        tiling_ub = self.tik_instance.Tensor(
            Constant.TILING_SCALAR_DTYPE,
            (Constant.TILING_PARAMS_NUM,),
            tik.scope_ubuf,
            "tiling_ub"
        )

        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 6, 0, 0)
        self.tiling_mode.set_as(tiling_ub[0])
        self.last_dim.set_as(tiling_ub[1])
        self.total_num.set_as(tiling_ub[2])
        self.broadcast_first_dim.set_as(tiling_ub[3])
        self.broadcast_second_dim.set_as(tiling_ub[4])
        self.tiling_core_num.set_as(tiling_ub[5])
        self.data_align.set_as(tiling_ub[6])
        self.task_num.set_as(tiling_ub[7])
        self.tail_align.set_as(tiling_ub[8])
        self.task_ub_size.set_as(tiling_ub[9])
        self.r1_move_rep_times.set_as(tiling_ub[10])
        self.repeat_cycle.set_as(tiling_ub[11])
        self.repeat_tail.set_as(tiling_ub[12])
        self.left_num.set_as(tiling_ub[13])
        self.task_ub_size_tail.set_as(tiling_ub[14])
        self.r1_move_rep_times_tail.set_as(tiling_ub[15])
        self.repeat_cycle_tail.set_as(tiling_ub[16])
        self.repeat_tail_tail.set_as(tiling_ub[17])
        self.left_num_tail.set_as(tiling_ub[18])
        self.move_stride.set_as(tiling_ub[19])
        self.x_half_burst.set_as(tiling_ub[20])
        self.x_burst.set_as(tiling_ub[21])

    # 'pylint: disable=huawei-too-many-arguments
    def tik_func_calcu(self, mode, x_ub, r_ub, repeat_cycle, repeat_tail, left_num):
        if mode == "vmul":
            tik_fun = self.tik_instance.vmul
        if mode == "vadd":
            tik_fun = self.tik_instance.vadd

        with self.tik_instance.for_range(0, repeat_cycle) as i:
            offset = self.mask * 255 * i
            tik_fun(self.mask, x_ub[offset], r_ub[offset], x_ub[offset],
                    255, 1, 1, 1, 8, 8, 8)
        with self.tik_instance.if_scope(repeat_tail > 0):
            offset = self.mask * 255 * repeat_cycle
            tik_fun(self.mask, x_ub[offset], r_ub[offset], x_ub[offset],
                    repeat_tail, 1, 1, 1, 8, 8, 8)
        with self.tik_instance.if_scope(left_num > 0):
            offset = self.mask * 255 * repeat_cycle + self.mask * repeat_tail
            tik_fun(left_num, x_ub[offset], r_ub[offset], x_ub[offset],
                    1, 1, 1, 1, 8, 8, 8)
    
    # 'pylint: disable=huawei-too-many-arguments
    def do_cal_with_x(self, task_id, x_ub, x_concat_ub, r1_ub, r2_ub, align,
                      repeat, tail, move_repeat, left_num, ub_size, gm_offset):
        if self.dtype == "bfloat16":
            x_ub_bf16 = self.tik_instance.Tensor("bfloat16", [ub_size], tik.scope_ubuf, "x_ub_fp16")
        with self.tik_instance.for_range(0, self.broadcast_first_dim) as i:
            x_offset = task_id * self.task_ub_size + i * self.total_num * self.last_dim + gm_offset
            if self.dtype == "bfloat16":
                self.tik_instance.data_move(x_ub_bf16, self.x_gm[x_offset], 0, 1, move_repeat, 0, 0)
                self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", x_ub_bf16, x_ub, ub_size)
            else:
                self.tik_instance.data_move(x_ub, self.x_gm[x_offset], 0, 1, move_repeat, 0, 0)
            self.tik_instance.data_move(x_concat_ub[self.last_dim // 2], x_ub, 0, align,
                                        self.x_half_burst, self.x_half_burst, self.x_half_burst)
            self.tik_instance.data_move(x_concat_ub, x_ub[self.last_dim // 2], 0, align,
                                        self.x_half_burst, self.x_half_burst, self.x_half_burst)
            
            self.tik_func_calcu("vmul", x_ub, r1_ub, repeat, tail, left_num)
            self.tik_func_calcu("vmul", x_concat_ub, r2_ub, repeat, tail, left_num)
            self.tik_func_calcu("vadd", x_ub, x_concat_ub, repeat, tail, left_num)

            if self.dtype == "bfloat16":
                self.vec_conv_bfp16_and_fp32("fp32_to_bfp16", x_ub, x_ub_bf16, ub_size)
                self.tik_instance.data_move(self.output_gm[x_offset], x_ub_bf16, 0, 1, move_repeat, 0, 0)
            else:
                self.tik_instance.data_move(self.output_gm[x_offset], x_ub, 0, 1, move_repeat, 0, 0) 
    
    def vec_conv_bfp16_and_fp32(self, mode, src_ub, dst_ub, num):
        mask = Constant.MASK_32
        repeat = num // mask
        repeat_tail = num % mask

        if mode == "bfp16_to_fp32":
            with self.tik_instance.if_scope(repeat > 0):
                self.tik_instance.vec_conv(mask, "none", dst_ub, src_ub, repeat, 8, 4)
            with self.tik_instance.if_scope(repeat_tail > 0):
                offset = mask * repeat
                self.tik_instance.vec_conv(mask, "none", dst_ub[offset], src_ub[offset], repeat_tail, 
                  repeat_tail // Constant.DATA_EACH_BLOCK_32, repeat_tail // Constant.DATA_EACH_BLOCK_16)
        else:
            with self.tik_instance.if_scope(repeat > 0):
                self.tik_instance.vec_conv(mask, "round", dst_ub, src_ub, repeat, 4, 8)
            with self.tik_instance.if_scope(repeat_tail > 0):
                offset = mask * repeat
                self.tik_instance.vec_conv(mask, "round", dst_ub[offset], src_ub[offset], repeat_tail,
                  repeat_tail // Constant.DATA_EACH_BLOCK_16, repeat_tail // Constant.DATA_EACH_BLOCK_32)
    
    def update_r1_and_r2(self, task_id, r1_ub, r2_ub, move_rep_time, gm_offset):
        r_offset = task_id * self.task_ub_size + gm_offset
        self.tik_instance.data_move(r1_ub, self.r1_gm[r_offset], 0, 1, move_rep_time, 0, 0)
        self.tik_instance.data_move(r2_ub, self.r2_gm[r_offset], 0, 1, move_rep_time, 0, 0)

    # 'pylint: disable=huawei-too-many-arguments
    def update_r1_and_r2_bfloat16(self, task_id, r1_ub, r2_ub, move_rep_time, ub_size, gm_offset):
        with self.tik_instance.new_stmt_scope():
            r1_ub_bfloat16 = self.tik_instance.Tensor("bfloat16", [ub_size], tik.scope_ubuf, "r1_ub_float16")
            r2_ub_bfloat16 = self.tik_instance.Tensor("bfloat16", [ub_size], tik.scope_ubuf, "r2_ub_float16")
            r_offset = task_id * self.task_ub_size + gm_offset
            self.tik_instance.data_move(r1_ub_bfloat16, self.r1_gm[r_offset], 0, 1, move_rep_time, 0, 0)
            self.tik_instance.data_move(r2_ub_bfloat16, self.r2_gm[r_offset], 0, 1, move_rep_time, 0, 0)

            self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", r1_ub_bfloat16, r1_ub, ub_size)
            self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", r2_ub_bfloat16, r2_ub, ub_size)
    
    def update_r1_and_r2_single_broadcast(self, task_id, r1_ub, r2_ub, align):
        with self.tik_instance.new_stmt_scope():
            r1_tmp_ub = self.tik_instance.Tensor(self.dtype, [self.last_dim], tik.scope_ubuf, "r1_tmp_ub")
            r2_tmp_ub = self.tik_instance.Tensor(self.dtype, [self.last_dim], tik.scope_ubuf, "r2_tmp_ub")
            self.tik_instance.data_move(r1_tmp_ub, self.r1_gm, 0, 1, self.move_stride, 0, 0)
            self.tik_instance.data_move(r2_tmp_ub, self.r2_gm, 0, 1, self.move_stride, 0, 0)

            if self.dtype == "bfloat16":
                r1_tmp_ub_fp32 = self.tik_instance.Tensor(self.ub_dtype, [self.last_dim],
                                                          tik.scope_ubuf, "r1_tmp_ub_fp32")
                r2_tmp_ub_fp32 = self.tik_instance.Tensor(self.ub_dtype, [self.last_dim],
                                                          tik.scope_ubuf, "r2_tmp_ub_fp32")
                self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", r1_tmp_ub, r1_tmp_ub_fp32, self.last_dim)
                self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", r2_tmp_ub, r2_tmp_ub_fp32, self.last_dim)

                with self.tik_instance.for_range(0, align) as i:
                    r_ub_offset = i * self.last_dim
                    self.tik_instance.data_move(r1_ub[r_ub_offset], r1_tmp_ub_fp32, 0, 1, self.move_stride * 2, 0, 0)
                    self.tik_instance.data_move(r2_ub[r_ub_offset], r2_tmp_ub_fp32, 0, 1, self.move_stride * 2, 0, 0)
            else:
                with self.tik_instance.for_range(0, align) as i:
                    r_ub_offset = i * self.last_dim
                    self.tik_instance.data_move(r1_ub[r_ub_offset], r1_tmp_ub, 0, 1, self.move_stride, 0, 0)
                    self.tik_instance.data_move(r2_ub[r_ub_offset], r2_tmp_ub, 0, 1, self.move_stride, 0, 0)
    
    def r2_ub_vmuls_scalar(self, r2_ub, length, align):
        rep_stride = self.last_dim // self.data_each_block
        repeat = length // self.mask
        with self.tik_instance.for_range(0, repeat) as i:
            offset = self.mask * i
            self.tik_instance.vmuls(self.mask, r2_ub[offset], r2_ub[offset], -1.0,
                                    align, 1, 1, rep_stride, rep_stride)
        repeat_tail = length % self.mask
        with self.tik_instance.if_scope(repeat_tail > 0):
            offset = self.mask * repeat
            self.tik_instance.vmuls(repeat_tail, r2_ub[offset], r2_ub[offset], -1.0,
                                    align, 1, 1, rep_stride, rep_stride)

    # 'pylint: disable=huawei-too-many-arguments
    def do_cal_with_x_sep_bf16(self, task_id, x_ub, x_concat_ub, r1_ub, r2_ub,
                               align, repeat, tail, left_num, ub_size):
        x_ub_bf16 = self.tik_instance.Tensor("bfloat16", [ub_size], tik.scope_ubuf, "x_ub_fp16")
        with self.tik_instance.if_scope(self.move_stride < Constant.MOVE_STRIDE_MAX):
            with self.tik_instance.for_range(0, self.broadcast_first_dim) as i:
                with self.tik_instance.for_range(0, self.broadcast_second_dim) as j:
                    x_offset = task_id * self.task_ub_size * self.broadcast_second_dim + \
                                i * self.total_num * self.last_dim * self.broadcast_second_dim + \
                                j * self.last_dim

                    self.tik_instance.data_move(x_ub_bf16, self.x_gm[x_offset], 0, align, 
                                                self.x_burst, self.move_stride, 0)
                    self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", x_ub_bf16, x_ub, ub_size)

                    self.tik_instance.data_move(x_concat_ub[self.last_dim // 2], x_ub, 0, align,
                                                self.x_half_burst, self.x_half_burst, self.x_half_burst)
                    self.tik_instance.data_move(x_concat_ub, x_ub[self.last_dim // 2], 0, align,
                                                self.x_half_burst, self.x_half_burst, self.x_half_burst)

                    self.tik_func_calcu("vmul", x_ub, r1_ub, repeat, tail, left_num)
                    self.tik_func_calcu("vmul", x_concat_ub, r2_ub, repeat, tail, left_num)
                    self.tik_func_calcu("vadd", x_ub, x_concat_ub, repeat, tail, left_num)

                    self.vec_conv_bfp16_and_fp32("fp32_to_bfp16", x_ub, x_ub_bf16, ub_size)
                    self.tik_instance.data_move(self.output_gm[x_offset], x_ub_bf16, 0, align,
                                                self.x_burst, 0, self.move_stride)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.broadcast_first_dim) as i:
                with self.tik_instance.for_range(0, self.broadcast_second_dim) as j:
                    x_offset = task_id * self.task_ub_size * self.broadcast_second_dim + \
                                i * self.total_num * self.last_dim * self.broadcast_second_dim + \
                                j * self.last_dim

                    with self.tik_instance.for_range(0, align) as align_offset:
                        x_ub_offset = x_offset + align_offset * self.broadcast_second_dim * self.last_dim
                        ub_offset = align_offset * self.last_dim
                        self.tik_instance.data_move(x_ub_bf16[ub_offset], self.x_gm[x_ub_offset], 0, 1, 
                                                    self.x_burst, 0, 0)
                    self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", x_ub_bf16, x_ub, ub_size)

                    self.tik_instance.data_move(x_concat_ub[self.last_dim // 2], x_ub, 0, align,
                                                self.x_half_burst, self.x_half_burst, self.x_half_burst)
                    self.tik_instance.data_move(x_concat_ub, x_ub[self.last_dim // 2], 0, align,
                                                self.x_half_burst, self.x_half_burst, self.x_half_burst)

                    self.tik_func_calcu("vmul", x_ub, r1_ub, repeat, tail, left_num)
                    self.tik_func_calcu("vmul", x_concat_ub, r2_ub, repeat, tail, left_num)
                    self.tik_func_calcu("vadd", x_ub, x_concat_ub, repeat, tail, left_num)

                    self.vec_conv_bfp16_and_fp32("fp32_to_bfp16", x_ub, x_ub_bf16, ub_size)
                    with self.tik_instance.for_range(0, align) as align_offset:
                        x_ub_offset = x_offset + align_offset * self.broadcast_second_dim * self.last_dim
                        ub_offset = align_offset * self.last_dim
                        self.tik_instance.data_move(self.output_gm[x_ub_offset], x_ub_bf16[ub_offset], 0, 1,
                                                    self.x_burst, 0, 0)

    # 'pylint: disable=huawei-too-many-arguments
    def do_cal_with_x_sep(self, task_id, x_ub, x_concat_ub, r1_ub, r2_ub,
                          align, repeat, tail, left_num, ub_size):
        with self.tik_instance.if_scope(self.move_stride < Constant.MOVE_STRIDE_MAX):
            with self.tik_instance.for_range(0, self.broadcast_first_dim) as i:
                with self.tik_instance.for_range(0, self.broadcast_second_dim) as j:
                    x_offset = task_id * self.task_ub_size * self.broadcast_second_dim + \
                                i * self.total_num * self.last_dim * self.broadcast_second_dim + \
                                j * self.last_dim

                    self.tik_instance.data_move(x_ub, self.x_gm[x_offset], 0, align,
                                                self.x_burst, self.move_stride, 0)

                    self.tik_instance.data_move(x_concat_ub[self.last_dim // 2], x_ub, 0, align,
                                                self.x_half_burst, self.x_half_burst, self.x_half_burst)
                    self.tik_instance.data_move(x_concat_ub, x_ub[self.last_dim // 2], 0, align,
                                                self.x_half_burst, self.x_half_burst, self.x_half_burst)

                    self.tik_func_calcu("vmul", x_ub, r1_ub, repeat, tail, left_num)
                    self.tik_func_calcu("vmul", x_concat_ub, r2_ub, repeat, tail, left_num)
                    self.tik_func_calcu("vadd", x_ub, x_concat_ub, repeat, tail, left_num)

                    self.tik_instance.data_move(self.output_gm[x_offset], x_ub, 0, align,
                                                self.x_burst, 0, self.move_stride)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.broadcast_first_dim) as i:
                with self.tik_instance.for_range(0, self.broadcast_second_dim) as j:
                    x_offset = task_id * self.task_ub_size * self.broadcast_second_dim + \
                                i * self.total_num * self.last_dim * self.broadcast_second_dim + \
                                j * self.last_dim

                    with self.tik_instance.for_range(0, align) as align_offset:
                        x_ub_offset = x_offset + align_offset * self.broadcast_second_dim * self.last_dim
                        ub_offset = align_offset * self.last_dim
                        self.tik_instance.data_move(x_ub[ub_offset], self.x_gm[x_ub_offset], 0, 1,
                                                    self.x_burst, 0, 0)

                    self.tik_instance.data_move(x_concat_ub[self.last_dim // 2], x_ub, 0, align,
                                                self.x_half_burst, self.x_half_burst, self.x_half_burst)
                    self.tik_instance.data_move(x_concat_ub, x_ub[self.last_dim // 2], 0, align,
                                                self.x_half_burst, self.x_half_burst, self.x_half_burst)

                    self.tik_func_calcu("vmul", x_ub, r1_ub, repeat, tail, left_num)
                    self.tik_func_calcu("vmul", x_concat_ub, r2_ub, repeat, tail, left_num)
                    self.tik_func_calcu("vadd", x_ub, x_concat_ub, repeat, tail, left_num)

                    with self.tik_instance.for_range(0, align) as align_offset:
                        x_ub_offset = x_offset + align_offset * self.broadcast_second_dim * self.last_dim
                        ub_offset = align_offset * self.last_dim
                        self.tik_instance.data_move(self.output_gm[x_ub_offset], x_ub[ub_offset], 0, 1,
                                                    self.x_burst, 0, 0)

    def rotary_process_single_broadcast(self, task_id):
        r1_ub = self.tik_instance.Tensor(self.ub_dtype,
                                         [self.task_ub_size],
                                         tik.scope_ubuf,
                                         "r1_ub")
        r2_ub = self.tik_instance.Tensor(self.ub_dtype,
                                         [self.task_ub_size],
                                         tik.scope_ubuf,
                                         "r2_ub")
        x_ub = self.tik_instance.Tensor(self.ub_dtype,
                                        [self.task_ub_size],
                                        tik.scope_ubuf,
                                        "x_ub")
        x_concat_ub = self.tik_instance.Tensor(self.ub_dtype,
                                               [self.task_ub_size],
                                               tik.scope_ubuf,
                                               "x_concat_ub")

        self.update_r1_and_r2_single_broadcast(task_id, r1_ub, r2_ub, self.data_align)

        self.r2_ub_vmuls_scalar(r2_ub, self.last_dim // 2, self.data_align)

        if self.dtype == "bfloat16":
            x_ub_bf16 = self.tik_instance.Tensor("bfloat16", [self.task_ub_size], tik.scope_ubuf, "x_ub_fp16")
            self.tik_instance.data_move(x_ub_bf16, self.x_gm[task_id * self.task_ub_size],
                                        0, 1, self.r1_move_rep_times, 0, 0)
            self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", x_ub_bf16, x_ub, self.task_ub_size)
        else:
            self.tik_instance.data_move(x_ub, self.x_gm[task_id * self.task_ub_size], 0, 1,
                                        self.r1_move_rep_times, 0, 0)
        
        self.tik_instance.data_move(x_concat_ub[self.last_dim // 2], x_ub, 0, self.data_align,
                                    self.x_half_burst, self.x_half_burst, self.x_half_burst)
        self.tik_instance.data_move(x_concat_ub, x_ub[self.last_dim // 2], 0, self.data_align,
                                    self.x_half_burst, self.x_half_burst, self.x_half_burst)
        
        self.tik_func_calcu("vmul", x_ub, r1_ub, self.repeat_cycle, self.repeat_tail, self.left_num)
        self.tik_func_calcu("vmul", x_concat_ub, r2_ub, self.repeat_cycle, self.repeat_tail, self.left_num)
        self.tik_func_calcu("vadd", x_ub, x_concat_ub, self.repeat_cycle, self.repeat_tail, self.left_num)

        if self.dtype == "bfloat16":
            self.vec_conv_bfp16_and_fp32("fp32_to_bfp16", x_ub, x_ub_bf16, self.task_ub_size)
            self.tik_instance.data_move(self.output_gm[task_id * self.task_ub_size], x_ub_bf16,
                                        0, 1, self.r1_move_rep_times, 0, 0)
        else:
            self.tik_instance.data_move(self.output_gm[task_id * self.task_ub_size], x_ub, 0, 1,
                                        self.r1_move_rep_times, 0, 0) 

    def rotary_process_single_broadcast_tail(self, task_id):
        r1_ub = self.tik_instance.Tensor(self.ub_dtype,
                                         [self.task_ub_size_tail],
                                         tik.scope_ubuf,
                                         "r1_ub")
        r2_ub = self.tik_instance.Tensor(self.ub_dtype,
                                         [self.task_ub_size_tail],
                                         tik.scope_ubuf,
                                         "r2_ub")
        x_ub = self.tik_instance.Tensor(self.ub_dtype,
                                        [self.task_ub_size_tail],
                                        tik.scope_ubuf,
                                        "x_ub")
        x_concat_ub = self.tik_instance.Tensor(self.ub_dtype,
                                               [self.task_ub_size_tail],
                                               tik.scope_ubuf,
                                               "x_concat_ub")

        self.update_r1_and_r2_single_broadcast(task_id, r1_ub, r2_ub, self.tail_align)

        self.r2_ub_vmuls_scalar(r2_ub, self.last_dim // 2, self.tail_align)

        if self.dtype == "bfloat16":
            x_ub_bf16 = self.tik_instance.Tensor("bfloat16", [self.task_ub_size_tail], tik.scope_ubuf, "x_ub_fp16")
            self.tik_instance.data_move(x_ub_bf16, self.x_gm[task_id * self.task_ub_size], 0, 1,
                                        self.r1_move_rep_times_tail, 0, 0)
            self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", x_ub_bf16, x_ub, self.task_ub_size_tail)
        else:
            self.tik_instance.data_move(x_ub, self.x_gm[task_id * self.task_ub_size], 0, 1,
                                        self.r1_move_rep_times_tail, 0, 0)
        
        self.tik_instance.data_move(x_concat_ub[self.last_dim // 2], x_ub, 0, self.tail_align,
                                    self.x_half_burst, self.x_half_burst, self.x_half_burst)
        self.tik_instance.data_move(x_concat_ub, x_ub[self.last_dim // 2], 0, self.tail_align,
                                    self.x_half_burst, self.x_half_burst, self.x_half_burst)
        
        self.tik_func_calcu("vmul", x_ub, r1_ub, self.repeat_cycle_tail, self.repeat_tail_tail, self.left_num_tail)
        self.tik_func_calcu("vmul", x_concat_ub, r2_ub, self.repeat_cycle_tail,
                            self.repeat_tail_tail, self.left_num_tail)
        self.tik_func_calcu("vadd", x_ub, x_concat_ub, self.repeat_cycle_tail,
                            self.repeat_tail_tail, self.left_num_tail)

        if self.dtype == "bfloat16":
            self.vec_conv_bfp16_and_fp32("fp32_to_bfp16", x_ub, x_ub_bf16, self.task_ub_size_tail)
            self.tik_instance.data_move(self.output_gm[task_id * self.task_ub_size], x_ub_bf16,
                                        0, 1, self.r1_move_rep_times_tail, 0, 0)
        else:
            self.tik_instance.data_move(self.output_gm[task_id * self.task_ub_size], x_ub, 0, 1,
                                        self.r1_move_rep_times_tail, 0, 0) 

    def rotary_process_broadcast(self, task_id):
        r1_ub = self.tik_instance.Tensor(self.ub_dtype,
                                         [self.task_ub_size],
                                         tik.scope_ubuf,
                                         "r1_ub")
        r2_ub = self.tik_instance.Tensor(self.ub_dtype,
                                         [self.task_ub_size],
                                         tik.scope_ubuf,
                                         "r2_ub")
        x_ub = self.tik_instance.Tensor(self.ub_dtype,
                                        [self.task_ub_size],
                                        tik.scope_ubuf,
                                        "x_ub")
        x_concat_ub = self.tik_instance.Tensor(self.ub_dtype,
                                               [self.task_ub_size],
                                               tik.scope_ubuf,
                                               "x_concat_ub")

        with self.tik_instance.for_range(0, self.broadcast_second_dim) as i:
            r_gm_offset = i * self.total_num * self.last_dim
            if self.dtype == "bfloat16":
                self.update_r1_and_r2_bfloat16(task_id, r1_ub, r2_ub, self.r1_move_rep_times,
                                               self.task_ub_size, r_gm_offset)
            else:
                self.update_r1_and_r2(task_id, r1_ub, r2_ub, self.r1_move_rep_times, r_gm_offset)

            self.r2_ub_vmuls_scalar(r2_ub, self.last_dim // 2, self.data_align)
            
            x_gm_offset = i * self.total_num * self.last_dim * self.broadcast_first_dim
            self.do_cal_with_x(task_id, x_ub, x_concat_ub, r1_ub, r2_ub, self.data_align, self.repeat_cycle,
                               self.repeat_tail, self.r1_move_rep_times, self.left_num, self.task_ub_size, x_gm_offset)

    def rotary_process_broadcast_tail(self, task_id):
        r1_ub = self.tik_instance.Tensor(self.ub_dtype,
                                         [self.task_ub_size_tail],
                                         tik.scope_ubuf,
                                         "r1_ub")
        r2_ub = self.tik_instance.Tensor(self.ub_dtype,
                                         [self.task_ub_size_tail],
                                         tik.scope_ubuf,
                                         "r2_ub")
        x_ub = self.tik_instance.Tensor(self.ub_dtype,
                                        [self.task_ub_size_tail],
                                        tik.scope_ubuf,
                                        "x_ub")
        x_concat_ub = self.tik_instance.Tensor(self.ub_dtype,
                                               [self.task_ub_size_tail],
                                               tik.scope_ubuf,
                                               "x_concat_ub")

        with self.tik_instance.for_range(0, self.broadcast_second_dim) as i:
            r_gm_offset = i * self.total_num * self.last_dim
            if self.dtype == "bfloat16":
                self.update_r1_and_r2_bfloat16(task_id, r1_ub, r2_ub, self.r1_move_rep_times_tail,
                                               self.task_ub_size_tail, r_gm_offset)
            else:
                self.update_r1_and_r2(task_id, r1_ub, r2_ub, self.r1_move_rep_times_tail, r_gm_offset)

            self.r2_ub_vmuls_scalar(r2_ub, self.last_dim // 2, self.tail_align)

            x_gm_offset = i * self.total_num * self.last_dim * self.broadcast_first_dim
            self.do_cal_with_x(task_id, x_ub, x_concat_ub, r1_ub, r2_ub, self.tail_align, self.repeat_cycle_tail,
                                self.repeat_tail_tail, self.r1_move_rep_times_tail,
                                self.left_num_tail, self.task_ub_size_tail, x_gm_offset)

    def rotary_process_broadcast_seperate(self, task_id):
        r1_ub = self.tik_instance.Tensor(self.ub_dtype,
                                         [self.task_ub_size],
                                         tik.scope_ubuf,
                                         "r1_ub")
        r2_ub = self.tik_instance.Tensor(self.ub_dtype,
                                         [self.task_ub_size],
                                         tik.scope_ubuf,
                                         "r2_ub")
        x_ub = self.tik_instance.Tensor(self.ub_dtype,
                                        [self.task_ub_size],
                                        tik.scope_ubuf,
                                        "x_ub")
        x_concat_ub = self.tik_instance.Tensor(self.ub_dtype,
                                               [self.task_ub_size],
                                               tik.scope_ubuf,
                                               "x_concat_ub")

        if self.dtype == "bfloat16":
            self.update_r1_and_r2_bfloat16(task_id, r1_ub, r2_ub, self.r1_move_rep_times, self.task_ub_size, 0)
        else:
            self.update_r1_and_r2(task_id, r1_ub, r2_ub, self.r1_move_rep_times, 0)
        
        self.r2_ub_vmuls_scalar(r2_ub, self.last_dim // 2, self.data_align)

        if self.dtype == "bfloat16":
            self.do_cal_with_x_sep_bf16(task_id, x_ub, x_concat_ub, r1_ub, r2_ub,
                                        self.data_align, self.repeat_cycle, self.repeat_tail,
                                        self.left_num, self.task_ub_size)
        else:
            self.do_cal_with_x_sep(task_id, x_ub, x_concat_ub, r1_ub, r2_ub,
                                   self.data_align, self.repeat_cycle, self.repeat_tail,
                                   self.left_num, self.task_ub_size)
    
    def rotary_process_broadcast_seperate_tail(self, task_id):
        r1_ub = self.tik_instance.Tensor(self.ub_dtype,
                                         [self.task_ub_size_tail],
                                         tik.scope_ubuf,
                                         "r1_ub")
        r2_ub = self.tik_instance.Tensor(self.ub_dtype,
                                         [self.task_ub_size_tail],
                                         tik.scope_ubuf,
                                         "r2_ub")
        x_ub = self.tik_instance.Tensor(self.ub_dtype,
                                        [self.task_ub_size_tail],
                                        tik.scope_ubuf,
                                        "x_ub")
        x_concat_ub = self.tik_instance.Tensor(self.ub_dtype,
                                               [self.task_ub_size_tail],
                                               tik.scope_ubuf,
                                               "x_concat_ub")

        if self.dtype == "bfloat16":
            self.update_r1_and_r2_bfloat16(task_id, r1_ub, r2_ub, self.r1_move_rep_times_tail,
                                           self.task_ub_size_tail, 0)
        else:
            self.update_r1_and_r2(task_id, r1_ub, r2_ub, self.r1_move_rep_times_tail, 0)

        self.r2_ub_vmuls_scalar(r2_ub, self.last_dim // 2, self.tail_align)

        if self.dtype == "bfloat16":
            self.do_cal_with_x_sep_bf16(task_id, x_ub, x_concat_ub, r1_ub, r2_ub,
                                        self.tail_align, self.repeat_cycle_tail, self.repeat_tail_tail,
                                        self.left_num_tail, self.task_ub_size_tail)
        else:
            self.do_cal_with_x_sep(task_id, x_ub, x_concat_ub, r1_ub, r2_ub,
                                   self.tail_align, self.repeat_cycle_tail, self.repeat_tail_tail,
                                   self.left_num_tail, self.task_ub_size_tail)
    
    def rotary_process_broadcast_repeat(self, task_id):
        r1_ub = self.tik_instance.Tensor(self.ub_dtype, [self.last_dim], tik.scope_ubuf, "r1_ub")
        r2_ub = self.tik_instance.Tensor(self.ub_dtype, [self.last_dim], tik.scope_ubuf, "r2_ub")
        x_ub = self.tik_instance.Tensor(self.ub_dtype, [self.task_ub_size], tik.scope_ubuf, "x_ub")
        x_concat_ub = self.tik_instance.Tensor(self.ub_dtype, [self.task_ub_size], tik.scope_ubuf, "x_concat_ub")

        if self.dtype == "bfloat16":
            x_ub_bf16 = self.tik_instance.Tensor("bfloat16", [self.task_ub_size], tik.scope_ubuf, "x_ub_fp16")
        with self.tik_instance.for_range(0, self.broadcast_first_dim) as i:
            r_offset = i * self.last_dim
            if self.dtype == "bfloat16":
                r1_ub_bfloat16 = self.tik_instance.Tensor("bfloat16", [self.last_dim], tik.scope_ubuf, "r1_ub_bfloat16")
                r2_ub_bfloat16 = self.tik_instance.Tensor("bfloat16", [self.last_dim], tik.scope_ubuf, "r2_ub_bfloat16")
                self.tik_instance.data_move(r1_ub_bfloat16, self.r1_gm[r_offset], 0, 1, self.r1_move_rep_times, 0, 0)
                self.tik_instance.data_move(r2_ub_bfloat16, self.r2_gm[r_offset], 0, 1, self.r1_move_rep_times, 0, 0)
                self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", r1_ub_bfloat16, r1_ub, self.last_dim)
                self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", r2_ub_bfloat16, r2_ub, self.last_dim)
            else:
                self.tik_instance.data_move(r1_ub, self.r1_gm[r_offset], 0, 1, self.r1_move_rep_times, 0, 0)
                self.tik_instance.data_move(r2_ub, self.r2_gm[r_offset], 0, 1, self.r1_move_rep_times, 0, 0)
            
            self.r2_ub_vmuls_scalar(r2_ub, self.last_dim // 2, 1)

            x_offset = i * self.total_num * self.last_dim + task_id * self.task_ub_size
            if self.dtype == "bfloat16":
                self.tik_instance.data_move(x_ub_bf16, self.x_gm[x_offset], 0, 1, self.x_burst, 0, 0)
                self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", x_ub_bf16, x_ub, self.task_ub_size)
            else:
                self.tik_instance.data_move(x_ub, self.x_gm[x_offset], 0, 1, self.x_burst, 0, 0)
                
            self.tik_instance.data_move(x_concat_ub[self.last_dim // 2], x_ub, 0, self.data_align,
                                        self.x_half_burst, self.x_half_burst, self.x_half_burst)
            self.tik_instance.data_move(x_concat_ub, x_ub[self.last_dim // 2], 0, self.data_align,
                                        self.x_half_burst, self.x_half_burst, self.x_half_burst)

            with self.tik_instance.for_range(0, self.data_align) as j:
                offset = j * self.last_dim
                repeat_time = self.last_dim // self.mask
                self.tik_instance.vmul(self.mask, x_ub[offset], r1_ub, x_ub[offset], repeat_time, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vmul(self.mask, x_concat_ub[offset], r2_ub, x_concat_ub[offset], repeat_time,
                                    1, 1, 1, 8, 8, 8)
            repeat_time = self.task_ub_size // self.mask
            self.tik_instance.vadd(self.mask, x_ub, x_concat_ub, x_ub, repeat_time, 1, 1, 1, 8, 8, 8)            

            if self.dtype == "bfloat16":
                self.vec_conv_bfp16_and_fp32("fp32_to_bfp16", x_ub, x_ub_bf16, self.task_ub_size)
                self.tik_instance.data_move(self.output_gm[x_offset], x_ub_bf16, 0, 1, self.x_burst, 0, 0)
            else:
                self.tik_instance.data_move(self.output_gm[x_offset], x_ub, 0, 1, self.x_burst, 0, 0)
  
    def rotary_process_broadcast_repeat_tail(self, task_id):
        r1_ub = self.tik_instance.Tensor(self.ub_dtype, [self.last_dim], tik.scope_ubuf, "r1_ub")
        r2_ub = self.tik_instance.Tensor(self.ub_dtype, [self.last_dim], tik.scope_ubuf, "r2_ub")
        x_ub = self.tik_instance.Tensor(self.ub_dtype, [self.task_ub_size_tail], tik.scope_ubuf, "x_ub")
        x_concat_ub = self.tik_instance.Tensor(self.ub_dtype, [self.task_ub_size_tail], tik.scope_ubuf, "x_concat_ub")

        if self.dtype == "bfloat16":
            x_ub_bf16 = self.tik_instance.Tensor("bfloat16", [self.task_ub_size_tail], tik.scope_ubuf, "x_ub_fp16")
        with self.tik_instance.for_range(0, self.broadcast_first_dim) as i:
            r_offset = i * self.last_dim
            if self.dtype == "bfloat16":
                r1_ub_bfloat16 = self.tik_instance.Tensor("bfloat16", [self.last_dim], tik.scope_ubuf, "r1_ub_bfloat16")
                r2_ub_bfloat16 = self.tik_instance.Tensor("bfloat16", [self.last_dim], tik.scope_ubuf, "r2_ub_bfloat16")
                self.tik_instance.data_move(r1_ub_bfloat16, self.r1_gm[r_offset], 0, 1, self.r1_move_rep_times, 0, 0)
                self.tik_instance.data_move(r2_ub_bfloat16, self.r2_gm[r_offset], 0, 1, self.r1_move_rep_times, 0, 0)
                self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", r1_ub_bfloat16, r1_ub, self.last_dim)
                self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", r2_ub_bfloat16, r2_ub, self.last_dim)
            else:
                self.tik_instance.data_move(r1_ub, self.r1_gm[r_offset], 0, 1, self.r1_move_rep_times, 0, 0)
                self.tik_instance.data_move(r2_ub, self.r2_gm[r_offset], 0, 1, self.r1_move_rep_times, 0, 0)
            
            self.r2_ub_vmuls_scalar(r2_ub, self.last_dim // 2, 1)

            x_offset = i * self.total_num * self.last_dim + task_id * self.task_ub_size
            if self.dtype == "bfloat16":
                self.tik_instance.data_move(x_ub_bf16, self.x_gm[x_offset], 0, 1, self.left_num, 0, 0)
                self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", x_ub_bf16, x_ub, self.task_ub_size_tail)
            else:
                self.tik_instance.data_move(x_ub, self.x_gm[x_offset], 0, 1, self.left_num, 0, 0)
                
            self.tik_instance.data_move(x_concat_ub[self.last_dim // 2], x_ub, 0, self.tail_align,
                                        self.x_half_burst, self.x_half_burst, self.x_half_burst)
            self.tik_instance.data_move(x_concat_ub, x_ub[self.last_dim // 2], 0, self.tail_align,
                                        self.x_half_burst, self.x_half_burst, self.x_half_burst)

            with self.tik_instance.for_range(0, self.tail_align) as j:
                offset = j * self.last_dim
                repeat_time = self.last_dim // self.mask
                self.tik_instance.vmul(self.mask, x_ub[offset], r1_ub, x_ub[offset], repeat_time, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vmul(self.mask, x_concat_ub[offset], r2_ub, x_concat_ub[offset], repeat_time,
                                       1, 1, 1, 8, 8, 8)
            repeat_time = self.task_ub_size_tail // self.mask
            self.tik_instance.vadd(self.mask, x_ub, x_concat_ub, x_ub, repeat_time, 1, 1, 1, 8, 8, 8)            

            if self.dtype == "bfloat16":
                self.vec_conv_bfp16_and_fp32("fp32_to_bfp16", x_ub, x_ub_bf16, self.task_ub_size_tail)
                self.tik_instance.data_move(self.output_gm[x_offset], x_ub_bf16, 0, 1, self.left_num, 0, 0)
            else:
                self.tik_instance.data_move(self.output_gm[x_offset], x_ub, 0, 1, self.left_num, 0, 0)

    def rotary_process_no_broadcast(self, task_id):
        r1_ub = self.tik_instance.Tensor(self.ub_dtype, [self.task_ub_size], tik.scope_ubuf, "r1_ub")
        r2_ub = self.tik_instance.Tensor(self.ub_dtype, [self.task_ub_size], tik.scope_ubuf, "r2_ub")
        x_ub = self.tik_instance.Tensor(self.ub_dtype, [self.task_ub_size], tik.scope_ubuf, "x_ub")
        x_concat_ub = self.tik_instance.Tensor(self.ub_dtype, [self.task_ub_size], tik.scope_ubuf, "x_concat_ub")

        if self.dtype == "bfloat16":
            self.update_r1_and_r2_bfloat16(task_id, r1_ub, r2_ub, self.r1_move_rep_times, self.task_ub_size, 0)
        else:
            self.update_r1_and_r2(task_id, r1_ub, r2_ub, self.r1_move_rep_times, 0)
        
        self.r2_ub_vmuls_scalar(r2_ub, self.last_dim // 2, self.data_align)

        if self.dtype == "bfloat16":
            x_ub_bf16 = self.tik_instance.Tensor("bfloat16", [self.task_ub_size], tik.scope_ubuf, "x_ub_bf16")
            self.tik_instance.data_move(x_ub_bf16, self.x_gm[task_id * self.task_ub_size], 0, 1,
                                        self.r1_move_rep_times, 0, 0)
            self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", x_ub_bf16, x_ub, self.task_ub_size)
        else:
            self.tik_instance.data_move(x_ub, self.x_gm[task_id * self.task_ub_size], 0, 1,
                                        self.r1_move_rep_times, 0, 0)
        
        self.tik_instance.data_move(x_concat_ub[self.last_dim // 2], x_ub, 0, self.data_align,
                                    self.x_half_burst, self.x_half_burst, self.x_half_burst)
        self.tik_instance.data_move(x_concat_ub, x_ub[self.last_dim // 2], 0, self.data_align,
                                    self.x_half_burst, self.x_half_burst, self.x_half_burst)
            
        self.tik_func_calcu("vmul", x_ub, r1_ub, self.repeat_cycle, self.repeat_tail, self.left_num)
        self.tik_func_calcu("vmul", x_concat_ub, r2_ub, self.repeat_cycle, self.repeat_tail, self.left_num)
        self.tik_func_calcu("vadd", x_ub, x_concat_ub, self.repeat_cycle, self.repeat_tail, self.left_num)

        if self.dtype == "bfloat16":
            self.vec_conv_bfp16_and_fp32("fp32_to_bfp16", x_ub, x_ub_bf16, self.task_ub_size)
            self.tik_instance.data_move(self.output_gm[task_id * self.task_ub_size], x_ub_bf16, 0, 1,
                                        self.r1_move_rep_times, 0, 0)
        else:
            self.tik_instance.data_move(self.output_gm[task_id * self.task_ub_size], x_ub, 0, 1,
                                        self.r1_move_rep_times, 0, 0)
    
    def rotary_process_no_broadcast_tail(self, task_id):
        r1_ub = self.tik_instance.Tensor(self.ub_dtype, [self.task_ub_size_tail], tik.scope_ubuf, "r1_ub")
        r2_ub = self.tik_instance.Tensor(self.ub_dtype, [self.task_ub_size_tail], tik.scope_ubuf, "r2_ub")
        x_ub = self.tik_instance.Tensor(self.ub_dtype, [self.task_ub_size_tail], tik.scope_ubuf, "x_ub")
        x_concat_ub = self.tik_instance.Tensor(self.ub_dtype, [self.task_ub_size_tail], tik.scope_ubuf, "x_concat_ub")

        if self.dtype == "bfloat16":
            self.update_r1_and_r2_bfloat16(task_id, r1_ub, r2_ub, self.r1_move_rep_times_tail,
                                           self.task_ub_size_tail, 0)
        else:
            self.update_r1_and_r2(task_id, r1_ub, r2_ub, self.r1_move_rep_times_tail, 0)
        
        self.r2_ub_vmuls_scalar(r2_ub, self.last_dim // 2, self.tail_align)

        if self.dtype == "bfloat16":
            x_ub_bf16 = self.tik_instance.Tensor("bfloat16", [self.task_ub_size_tail], tik.scope_ubuf, "x_ub_bf16")
            self.tik_instance.data_move(x_ub_bf16, self.x_gm[task_id * self.task_ub_size], 0, 1,
                                        self.r1_move_rep_times_tail, 0, 0)
            self.vec_conv_bfp16_and_fp32("bfp16_to_fp32", x_ub_bf16, x_ub, self.task_ub_size_tail)
        else:
            self.tik_instance.data_move(x_ub, self.x_gm[task_id * self.task_ub_size], 0, 1,
                                        self.r1_move_rep_times_tail, 0, 0)
        
        self.tik_instance.data_move(x_concat_ub[self.last_dim // 2], x_ub, 0, self.tail_align,
                                    self.x_half_burst, self.x_half_burst, self.x_half_burst)
        self.tik_instance.data_move(x_concat_ub, x_ub[self.last_dim // 2], 0, self.tail_align,
                                    self.x_half_burst, self.x_half_burst, self.x_half_burst)
            
        self.tik_func_calcu("vmul", x_ub, r1_ub, self.repeat_cycle_tail, self.repeat_tail_tail, self.left_num_tail)
        self.tik_func_calcu("vmul", x_concat_ub, r2_ub, self.repeat_cycle_tail,
                            self.repeat_tail_tail, self.left_num_tail)
        self.tik_func_calcu("vadd", x_ub, x_concat_ub, self.repeat_cycle_tail,
                            self.repeat_tail_tail, self.left_num_tail)

        if self.dtype == "bfloat16":
            self.vec_conv_bfp16_and_fp32("fp32_to_bfp16", x_ub, x_ub_bf16, self.task_ub_size_tail)
            self.tik_instance.data_move(self.output_gm[task_id * self.task_ub_size], x_ub_bf16, 0, 1,
                                        self.r1_move_rep_times_tail, 0, 0)
        else:
            self.tik_instance.data_move(self.output_gm[task_id * self.task_ub_size], x_ub, 0, 1,
                                        self.r1_move_rep_times_tail, 0, 0)

    def run_core(self, task_id):
        with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_BROADCAST_SEP):
            with self.tik_instance.if_scope(tik.any(task_id != self.task_num - 1, self.tail_align == 0)):
                self.rotary_process_broadcast_seperate(task_id)
            with self.tik_instance.else_scope():
                self.rotary_process_broadcast_seperate_tail(task_id)
        
        with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_BROADCAST):
            with self.tik_instance.if_scope(tik.any(task_id != self.task_num - 1, self.tail_align == 0)):
                self.rotary_process_broadcast(task_id)
            with self.tik_instance.else_scope():
                self.rotary_process_broadcast_tail(task_id)
        
        with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_NO_BROADCAST):
            with self.tik_instance.if_scope(tik.any(task_id != self.task_num - 1, self.tail_align == 0)):
                self.rotary_process_no_broadcast(task_id)
            with self.tik_instance.else_scope():
                self.rotary_process_no_broadcast_tail(task_id)
        
        with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_BROADCAST_REPEAT):
            with self.tik_instance.if_scope(tik.any(task_id != self.task_num - 1, self.tail_align == 0)):
                self.rotary_process_broadcast_repeat(task_id)
            with self.tik_instance.else_scope():
                self.rotary_process_broadcast_repeat_tail(task_id)
        
        with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_SINGLE_BROADCAST):
            with self.tik_instance.if_scope(tik.any(task_id != self.task_num - 1, self.tail_align == 0)):
                self.rotary_process_single_broadcast(task_id)
            with self.tik_instance.else_scope():
                self.rotary_process_single_broadcast_tail(task_id)

    def run_tik(self, kernel_name):
        self.get_tiling_data()

        batch_core_num = self.task_num // self.tiling_core_num
        batch_core_tail = self.task_num % self.tiling_core_num

        with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as i:
            with self.tik_instance.for_range(0, batch_core_num) as j:
                self.run_core(i + j * self.tiling_core_num)
            with self.tik_instance.if_scope(i < batch_core_tail):
                self.run_core(batch_core_num * self.tiling_core_num + i)

        tbe_context.get_context().add_compile_info(
            "vars", {
                "full_core_num": self.aicore_num,
                "ub_size": self.ub_size
            }
        )

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=[self.x_gm, self.r1_gm, self.r2_gm],
                                   outputs=[self.output_gm],
                                   flowtable = [self.tiling_gm])

        return self.tik_instance


# 'pylint: disable=huawei-too-many-arguments
@register_operator("RotaryMul")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def rotary_mul(x, r1, r2, result, kernel_name="rotary_mul"):
    rotary_mul_obj = RotaryMul(x, r1, r2)
    res = rotary_mul_obj.run_tik(kernel_name)

    return res