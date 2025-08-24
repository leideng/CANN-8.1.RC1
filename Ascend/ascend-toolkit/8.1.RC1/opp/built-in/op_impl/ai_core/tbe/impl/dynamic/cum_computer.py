#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
cum_computer
"""
from impl import common_util
from impl import constant_util
from impl.util import util_soc_common
from impl.util.util_tik_comm_func import ceil_div
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import check_support_block_size_16


class Constant:
    """
    The class for constant
    """
    TILING_ARG_NUM = 8
    MAX_INT64 = 2**64 - 1
    MAX_MOVE_BURST = 65535
    MAX_REPEAT_TIMES = 255
    BF16_DATA_SIZE = 2
    MASK = 64

    # define for transpose
    MAX_REPEAT_TIME = 255
    VECTOR_BLOCK_SIZE_BYTE2 = 256
    VECTOR_BLOCK_SIZE_BYTE4 = 128
    VECTOR_BLOCK_SIZE_BYTE2_FOR_NANO = 64
    TRANS_BLOCK_NUM_PER_VECTOR_F16 = 8
    TRANS_BLOCK_NUM_PER_VECTOR_F32 = 4
    FP16_ALIGN_NUM = 16
    NUMBER_SIXTEEN = 16

    # transpose mode
    LINE_TO_COL = 0
    COL_TO_LINE = 1

    # calc mode
    CALC_FULL_MODE = 0
    CALC_MORE_THAN_ONE_REPEAT = 1
    CALC_LESS_THAN_ONE_REPEAT = 2

    # addr mode
    ADD_BY_VECTOR = 0
    ROUND = "round"


# 'pylint: disable=attribute-defined-outside-init
class CumsumComputer:

    def __init__(self, x, axis, y, exclusive, reverse, kernel_name):
        self.x_dtype = x.get("dtype")
        self.x_size_orig = self._get_data_size(self.x_dtype)
        self.new_x_dtype = x.get("dtype")
        if self.new_x_dtype == "bfloat16":
            self.new_x_dtype = "float32"

        self.new_x_dtype_transpose = self.new_x_dtype
        self.exclusive = exclusive
        self.reverse = reverse
        self.kernel_name = kernel_name

        self.block_size = tbe_platform.get_block_size()

        self.tik_instance = tik.Tik(block_size=self.block_size)
        self.tik_profiling = tik.Dprofile()

        # ub size
        self.ub_size_bytes = self.tik_profiling.get_unified_buffer_size()
        self.x_dtype_size = self._get_data_size(self.new_x_dtype)
        self.data_one_block = self.block_size // self.x_dtype_size
        self.x_dtype_size_transpose = self._get_data_size(self.new_x_dtype_transpose)

        self.one_max_nums = (self.ub_size_bytes - 128) // 2 // self.block_size
        if self.x_dtype == "bfloat16":
            self.one_max_nums = self.one_max_nums // 2

        if not tbe_platform.api_check_support("tik.vcopy") and self.x_dtype_size == 4:
            self.ub_size_bytes = self.ub_size_bytes - 18 * 1024 if self.ub_size_bytes - 18 * 1024 > 0 \
                else self.ub_size_bytes

        #ub size for transpose
        if self.x_dtype != "bfloat16":
            self.loop_times_max = (self.ub_size_bytes - 2048) // 2
        else:
            # one for bf16, two for transpose src, two for transpose des
            self.loop_times_max = (self.ub_size_bytes - 2048) // 5
        self.src_transpose_line = 256 // self.x_dtype_size_transpose if not check_support_block_size_16() \
                else 128 // self.x_dtype_size_transpose
        self.loop_times_max = self.loop_times_max // self.x_dtype_size_transpose // 16 // self.src_transpose_line
        self.full_loop_times_max_num = self.src_transpose_line * 16 * self.loop_times_max
        self.one_loop_time_max_num = self.src_transpose_line * 16
        self.one_line_max_num = self.loop_times_max * 16
        self.trans_mod_num = 16 if not check_support_block_size_16() else 8

        self.is_support_transpose_mode = 0
        if (tbe_platform.api_check_support("tik.vcopy") and self.x_dtype in ("float32", "float16")) or (self.x_dtype
                                                                                                        in ("float16")):
            self.is_support_transpose_mode = 1

        if tbe_platform.api_check_support("tik.set_atomic_add", self.x_dtype) and \
                self.x_dtype != "bfloat16":
            self.support_atomic = 1
        else:
            self.support_atomic = 0

        self._init_gm()
        self._init_tiling_scalars()
        self._cum_compute_tiling()
        self._init_ub()

    @staticmethod
    def _get_data_size(dtype):
        return Constant.BF16_DATA_SIZE if dtype == "bfloat16" else common_util.get_data_size(dtype)

    def cum_computer(self):
        self._compute()

        exclusive_value = 1 if self.exclusive else 0
        reverse_value = 1 if self.reverse else 0
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size_bytes,
                "support_atomic": self.support_atomic,
                "support_transpose": self.is_support_transpose_mode,
                "exclusive": exclusive_value,
                "reverse": reverse_value,
                "one_line_max_num": self.one_line_max_num,
                "block_size": self.block_size
            })

        axis = self.tik_instance.Tensor("int64", (1,), name="axis", scope=tik.scope_gm)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm, axis],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm])
        return self.tik_instance

    def _init_tiling_scalars(self):
        # number of running cores
        self.act_core_num = self.tik_instance.Scalar("int64", name="act_core_num", init_value=0)
        # cycle of core
        self.core_cycle = self.tik_instance.Scalar("int64", name="core_cycle", init_value=0)
        # tail of core
        self.core_tail = self.tik_instance.Scalar("int64", name="core_tail", init_value=0)
        # per cum data number
        self.per_cum_num = self.tik_instance.Scalar("int64", name="per_cum_num", init_value=0)
        # per core cum times
        self.cum_times = self.tik_instance.Scalar("int64", name="cum_times", init_value=0)
        # all core number
        self.core_num = self.tik_instance.Scalar("int64", name="core_num", init_value=0)
        # tiling mode
        self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_mode", init_value=0)
        # last 8 need cycle
        self.remain_cycle = self.tik_instance.Scalar("int64", name="remain_cycle", init_value=0)
        self.each_line_max_cycle = self.tik_instance.Scalar("int64", name="each_line_max_cycle", init_value=0)
        self.each_line_max_number = self.tik_instance.Scalar("int64", name="each_line_max_number", init_value=0)

    def _init_ub(self):
        self.input_ub_orig = None
        self.ub_temp_orig = None
        self.input_ub = None
        self.out_ub = None
        self.trans_ub = None
        self.add_ub = None
        self.ub_temp = None
        self.add_data1 = None
        self.add_data2 = None

    def _init_gm(self):
        self.input_gm = self.tik_instance.Tensor(self.x_dtype, (Constant.MAX_INT64,),
                                                 name="input_gm",
                                                 scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(
            self.x_dtype, (Constant.MAX_INT64,), name="output_gm", scope=tik.scope_gm,
            is_atomic_add=True) if self.support_atomic == 1 else self.tik_instance.Tensor(
                self.x_dtype, (Constant.MAX_INT64,), name="output_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)

    def _cum_compute_tiling(self):
        """
        The function of tiling
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
        self.act_core_num.set_as(self.tiling_ub[0])
        self.core_cycle.set_as(self.tiling_ub[1])
        self.core_tail.set_as(self.tiling_ub[2])
        self.per_cum_num.set_as(self.tiling_ub[3])
        self.cum_times.set_as(self.tiling_ub[4])
        self.core_num.set_as(self.tiling_ub[5])
        self.tiling_mode.set_as(self.tiling_ub[6])
        self.remain_cycle.set_as(self.tiling_ub[7])

    def _dup_0_ub(self, dst_ub, data_len):
        """
        dup 0 ub
        """
        scalar_0 = self.tik_instance.Scalar(self.new_x_dtype, name="scalar_0", init_value=0)
        repeat_data_num = 8 * self.data_one_block
        dup_0_repeat = data_len // repeat_data_num
        dup_0_repeat_tail = data_len % repeat_data_num
        loop_repeat_cnt = dup_0_repeat // constant_util.MAX_REPEAT_TIMES
        with self.tik_instance.if_scope(dup_0_repeat >= constant_util.MAX_REPEAT_TIMES):
            with self.tik_instance.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
                offset = repeat_lp_cnt * constant_util.MAX_REPEAT_TIMES * repeat_data_num
                self.tik_instance.vector_dup(repeat_data_num, dst_ub[offset:], scalar_0, constant_util.MAX_REPEAT_TIMES,
                                             1, 8)
            remaining_repeat = dup_0_repeat - loop_repeat_cnt * constant_util.MAX_REPEAT_TIMES
        with self.tik_instance.if_scope(remaining_repeat > 0):
            offset = loop_repeat_cnt * constant_util.MAX_REPEAT_TIMES * repeat_data_num
            self.tik_instance.vector_dup(repeat_data_num, dst_ub[offset:], scalar_0, remaining_repeat, 1, 8)
        with self.tik_instance.if_scope(dup_0_repeat_tail > 0):
            offset = (loop_repeat_cnt * constant_util.MAX_REPEAT_TIMES + remaining_repeat) * repeat_data_num
            self.tik_instance.vector_dup(dup_0_repeat_tail, dst_ub[offset:], scalar_0, 1, 1, 8)

    def _gm_2_ub(self, dst_ub, gm_start_idx, burst):
        """
        move data from input gm to ub
        """
        loop_repeat_cnt = burst // Constant.MAX_MOVE_BURST
        gm_2_ub_repeat_tail = burst % Constant.MAX_MOVE_BURST
        with self.tik_instance.if_scope(burst >= Constant.MAX_MOVE_BURST):
            with self.tik_instance.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
                offset = gm_start_idx + repeat_lp_cnt * Constant.MAX_MOVE_BURST
                self.tik_instance.data_move(dst_ub, self.input_gm[offset], 0, 1, Constant.MAX_MOVE_BURST, 0, 0)
        with self.tik_instance.if_scope(gm_2_ub_repeat_tail > 0):
            offset = gm_start_idx + loop_repeat_cnt * Constant.MAX_MOVE_BURST
            if self.x_dtype == "bfloat16":
                shape_size = (self.one_max_nums * self.data_one_block // 32 + 1) * 32
                bf16_ub_tmp = self.tik_instance.Tensor("bfloat16", (shape_size,), tik.scope_ubuf, "bf16_ub_tmp")
                self.tik_instance.data_move_pad(bf16_ub_tmp, self.input_gm[offset], 1,
                                                (burst - loop_repeat_cnt * Constant.MAX_MOVE_BURST) * 8 * 2, 0, 0)
                vconv_loop = (burst - loop_repeat_cnt * Constant.MAX_MOVE_BURST) * 8 // \
                    (Constant.MASK * Constant.MAX_REPEAT_TIMES)
                repeat_tail = (burst - loop_repeat_cnt * Constant.MAX_MOVE_BURST) * 8 % \
                    (Constant.MASK * Constant.MAX_REPEAT_TIMES) // Constant.MASK
                mask = (burst - loop_repeat_cnt * Constant.MAX_MOVE_BURST) * 8 % \
                    (Constant.MASK * Constant.MAX_REPEAT_TIMES) % Constant.MASK
                with self.tik_instance.for_range(0, vconv_loop) as vconv_loop_idx:
                    loop_offset = vconv_loop_idx * Constant.MASK * Constant.MAX_REPEAT_TIMES
                    self.tik_instance.vconv(Constant.MASK, "", dst_ub[loop_offset], bf16_ub_tmp[loop_offset],
                                            Constant.MAX_REPEAT_TIMES, 1, 1, 8, 4)
                with self.tik_instance.if_scope(repeat_tail > 0):
                    loop_offset = vconv_loop * Constant.MASK * Constant.MAX_REPEAT_TIMES
                    self.tik_instance.vconv(Constant.MASK, "", dst_ub[loop_offset], bf16_ub_tmp[loop_offset],
                                            repeat_tail, 1, 1, 8, 4)
                with self.tik_instance.if_scope(mask > 0):
                    loop_offset = vconv_loop * Constant.MASK * Constant.MAX_REPEAT_TIMES + repeat_tail * Constant.MASK
                    self.tik_instance.vconv(mask, "", dst_ub[loop_offset], bf16_ub_tmp[loop_offset], 1, 1, 1, 8, 4)
            else:
                self.tik_instance.data_move(dst_ub, self.input_gm[offset], 0, 1,
                                            burst - loop_repeat_cnt * Constant.MAX_MOVE_BURST, 0, 0)

    def _gm_2_ub_by_ele(self, dst_ub, gm_start_idx, number):
        """
        move data from ub to output gm by number, only support last one block
        """
        if tbe_platform.api_check_support("tik.data_move_pad", self.x_dtype):
            offset = gm_start_idx
            if self.x_dtype == "bfloat16":
                shape_size = (self.one_max_nums * self.data_one_block // 32 + 1) * 32
                bf16_ub_tmp = self.tik_instance.Tensor("bfloat16", (shape_size,), tik.scope_ubuf, "bf16_ub_tmp")
                self.tik_instance.data_move_pad(bf16_ub_tmp, self.input_gm[offset], 1, number * self.x_size_orig, 0, 0)
                vconv_loop = number // (Constant.MASK * Constant.MAX_REPEAT_TIMES)
                with self.tik_instance.for_range(0, vconv_loop) as vconv_loop_idx:
                    loop_offset = vconv_loop_idx * Constant.MASK * Constant.MAX_REPEAT_TIMES
                    self.tik_instance.vconv(Constant.MASK, "", dst_ub[loop_offset], bf16_ub_tmp[loop_offset],
                                            Constant.MAX_REPEAT_TIMES, 1, 1, 8, 4)
                repeat_tail = number % (Constant.MASK * Constant.MAX_REPEAT_TIMES) // Constant.MASK
                with self.tik_instance.if_scope(repeat_tail > 0):
                    loop_offset = vconv_loop * Constant.MASK * Constant.MAX_REPEAT_TIMES
                    self.tik_instance.vconv(Constant.MASK, "", dst_ub[loop_offset], bf16_ub_tmp[loop_offset],
                                            repeat_tail, 1, 1, 8, 4)
                mask = number % (Constant.MASK * Constant.MAX_REPEAT_TIMES) % Constant.MASK
                with self.tik_instance.if_scope(mask > 0):
                    loop_offset = vconv_loop * Constant.MASK * Constant.MAX_REPEAT_TIMES + repeat_tail * Constant.MASK
                    self.tik_instance.vconv(mask, "", dst_ub[loop_offset], bf16_ub_tmp[loop_offset], 1, 1, 1, 8, 4)
            else:
                self.tik_instance.data_move_pad(dst_ub, self.input_gm[offset], 1, number * self.x_size_orig, 0, 0)
        else:
            block_num = ceil_div(number, self.data_one_block)
            self._gm_2_ub(dst_ub, gm_start_idx, block_num)

    def _ub_2_gm(self, src_ub, gm_start_idx, burst):
        """
        move data from ub to output gm
        """
        loop_repeat_cnt = burst // Constant.MAX_MOVE_BURST
        ub_2_gm_repeat_tail = burst % Constant.MAX_MOVE_BURST
        with self.tik_instance.if_scope(burst >= Constant.MAX_MOVE_BURST):
            with self.tik_instance.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
                offset = gm_start_idx + repeat_lp_cnt * Constant.MAX_MOVE_BURST
                self.tik_instance.data_move(self.output_gm[offset], src_ub, 0, 1, Constant.MAX_MOVE_BURST, 0, 0)
        with self.tik_instance.if_scope(ub_2_gm_repeat_tail > 0):
            offset = gm_start_idx + loop_repeat_cnt * Constant.MAX_MOVE_BURST
            if self.x_dtype == "bfloat16":
                shape_size = (self.one_max_nums * self.data_one_block // 32 + 1) * 32
                bf16_ub_tmp = self.tik_instance.Tensor("bfloat16", (shape_size,), tik.scope_ubuf, "bf16_ub_tmp")
                vconv_loop = (burst - loop_repeat_cnt * Constant.MAX_MOVE_BURST) * 8 // \
                    (Constant.MASK * Constant.MAX_REPEAT_TIMES)
                repeat_tail = (burst - loop_repeat_cnt * Constant.MAX_MOVE_BURST) * 8 % \
                    (Constant.MASK * Constant.MAX_REPEAT_TIMES) // Constant.MASK
                mask = (burst - loop_repeat_cnt * Constant.MAX_MOVE_BURST) * 8 % \
                    (Constant.MASK * Constant.MAX_REPEAT_TIMES) % Constant.MASK
                with self.tik_instance.for_range(0, vconv_loop) as vconv_loop_idx:
                    loop_offset = vconv_loop_idx * Constant.MASK * Constant.MAX_REPEAT_TIMES
                    self.tik_instance.vconv(Constant.MASK, "round", bf16_ub_tmp[loop_offset], src_ub[loop_offset],
                                            Constant.MAX_REPEAT_TIMES, 1, 1, 4, 8)
                with self.tik_instance.if_scope(repeat_tail > 0):
                    loop_offset = vconv_loop * Constant.MASK * Constant.MAX_REPEAT_TIMES
                    self.tik_instance.vconv(Constant.MASK, "round", bf16_ub_tmp[loop_offset], src_ub[loop_offset],
                                            repeat_tail, 1, 1, 4, 8)
                with self.tik_instance.if_scope(mask > 0):
                    loop_offset = vconv_loop * Constant.MASK * Constant.MAX_REPEAT_TIMES + repeat_tail * Constant.MASK
                    self.tik_instance.vconv(mask, "round", bf16_ub_tmp[loop_offset], src_ub[loop_offset], 1, 1, 1, 4, 8)
                self.tik_instance.data_move_pad(self.output_gm[offset], bf16_ub_tmp, 1,
                                                (burst - loop_repeat_cnt * Constant.MAX_MOVE_BURST) * 8 * 2, 0, 0)
            else:
                self.tik_instance.data_move(self.output_gm[offset], src_ub, 0, 1,
                                            burst - loop_repeat_cnt * Constant.MAX_MOVE_BURST, 0, 0)

    def _ub_2_gm_by_ele(self, src_ub, gm_start_idx, num):
        """
        move data from ub to output gm, only support last one block
        """
        if tbe_platform.api_check_support("tik.data_move_pad", self.x_dtype):
            offset = gm_start_idx
            if self.x_dtype == "bfloat16":
                shape_size = (self.one_max_nums * self.data_one_block // 32 + 1) * 32
                bf16_ub_tmp = self.tik_instance.Tensor("bfloat16", (shape_size,), tik.scope_ubuf, "bf16_ub_tmp")

                vconv_loop = num // (Constant.MASK * Constant.MAX_REPEAT_TIMES)
                with self.tik_instance.for_range(0, vconv_loop) as vconv_loop_idx:
                    loop_offset = vconv_loop_idx * Constant.MASK * Constant.MAX_REPEAT_TIMES
                    self.tik_instance.vconv(Constant.MASK, Constant.ROUND, bf16_ub_tmp[loop_offset],
                                            src_ub[loop_offset], Constant.MAX_REPEAT_TIMES, 1, 1, 4, 8)

                repeat_tail = num % (Constant.MASK * Constant.MAX_REPEAT_TIMES) // Constant.MASK
                with self.tik_instance.if_scope(repeat_tail > 0):
                    loop_offset = vconv_loop * Constant.MASK * Constant.MAX_REPEAT_TIMES
                    self.tik_instance.vconv(Constant.MASK, Constant.ROUND, bf16_ub_tmp[loop_offset],
                                            src_ub[loop_offset], repeat_tail, 1, 1, 4, 8)

                mask = num % (Constant.MASK * Constant.MAX_REPEAT_TIMES) % Constant.MASK
                with self.tik_instance.if_scope(mask > 0):
                    loop_offset = vconv_loop * Constant.MASK * Constant.MAX_REPEAT_TIMES + repeat_tail * Constant.MASK
                    self.tik_instance.vconv(mask, Constant.ROUND, bf16_ub_tmp[loop_offset], src_ub[loop_offset], 1, 1,
                                            1, 4, 8)
                    self.tik_instance.data_move_pad(self.output_gm[offset], bf16_ub_tmp, 1, num * self.x_size_orig, 0,
                                                    0)
            else:
                self.tik_instance.data_move_pad(self.output_gm[offset], src_ub, 1, num * self.x_size_orig, 0, 0)
        else:
            block_num = ceil_div(num, self.data_one_block)
            self._ub_2_gm(src_ub, gm_start_idx, block_num)

    def _batch_add(self, data_len, dst, src1, src2):
        """
        batch add
        """
        repeat_data_num = 8 * self.data_one_block
        add_repeat = self.tik_instance.Scalar("int64", "add_repeat")
        add_repeat.set_as(data_len // repeat_data_num)
        add_repeat_tail = data_len % repeat_data_num
        loop_repeat_cnt = add_repeat // constant_util.MAX_REPEAT_TIMES

        with self.tik_instance.if_scope(add_repeat >= constant_util.MAX_REPEAT_TIMES):
            with self.tik_instance.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
                offset = repeat_lp_cnt * constant_util.MAX_REPEAT_TIMES * repeat_data_num
                self.tik_instance.vadd(repeat_data_num, dst[offset], src1[offset], src2[offset],
                                       constant_util.MAX_REPEAT_TIMES, 1, 1, 1, 8, 8, 8)
            remaining_repeat = add_repeat - loop_repeat_cnt * constant_util.MAX_REPEAT_TIMES
        with self.tik_instance.if_scope(remaining_repeat > 0):
            offset = loop_repeat_cnt * constant_util.MAX_REPEAT_TIMES * repeat_data_num
            self.tik_instance.vadd(repeat_data_num, dst[offset], src1[offset], src2[offset], remaining_repeat, 1, 1, 1,
                                   8, 8, 8)
        with self.tik_instance.if_scope(add_repeat_tail > 0):
            offset = (loop_repeat_cnt * constant_util.MAX_REPEAT_TIMES + remaining_repeat) * repeat_data_num
            self.tik_instance.vadd(add_repeat_tail, dst[offset], src1[offset], src2[offset], 1, 1, 1, 1, 8, 8, 8)

    def _transpose_for_byte2(self, src_ub, dst_ub, trans_mode, loop_times):
        loop_times_max = self.loop_times_max

        if trans_mode == Constant.LINE_TO_COL:
            with self.tik_instance.for_range(0, Constant.TRANS_BLOCK_NUM_PER_VECTOR_F16) as trans_block_index:
                with self.tik_instance.for_range(0, loop_times) as loop_idx:
                    src_list = [
                        src_ub[16 * (i * loop_times + trans_block_index * loop_times_max * 16 + loop_idx)]
                        for i in range(Constant.FP16_ALIGN_NUM)
                    ]
                    dst_list = [
                        dst_ub[128 * i + trans_block_index * 16 +
                               (loop_idx * Constant.TRANS_BLOCK_NUM_PER_VECTOR_F16) * Constant.VECTOR_BLOCK_SIZE_BYTE2]
                        for i in range(Constant.FP16_ALIGN_NUM)
                    ]
                    self.tik_instance.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

        elif trans_mode == Constant.COL_TO_LINE:
            with self.tik_instance.for_range(0, Constant.TRANS_BLOCK_NUM_PER_VECTOR_F16) as trans_block_index:
                with self.tik_instance.for_range(0, loop_times) as loop_idx:
                    src_list = [
                        src_ub[128 * i + trans_block_index * 16 +
                               (loop_idx * Constant.TRANS_BLOCK_NUM_PER_VECTOR_F16) * Constant.VECTOR_BLOCK_SIZE_BYTE2]
                        for i in range(Constant.FP16_ALIGN_NUM)
                    ]
                    dst_list = [
                        dst_ub[16 * (i * loop_times + trans_block_index * loop_times_max * 16 + loop_idx)]
                        for i in range(Constant.FP16_ALIGN_NUM)
                    ]
                    self.tik_instance.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

    def _transpose_for_byte4(self, src_ub, dst_ub, trans_mode, loop_times):
        loop_times_max = self.loop_times_max if not check_support_block_size_16() else loop_times
        dst_list = [i for i in range(16)]
        src_list = [i for i in range(16)]

        if trans_mode == Constant.LINE_TO_COL:
            for j in range(Constant.TRANS_BLOCK_NUM_PER_VECTOR_F32):
                with self.tik_instance.for_range(0, loop_times) as loop_idx:
                    for i in range(16):
                        a = i // 2 * 64
                        if i % 2 == 1:
                            a += 8
                        src_list[i] = src_ub[16 * (i * loop_times + j * loop_times_max * 16 + loop_idx)]
                        dst_list[i] = dst_ub[a + j * 16 + 128 * loop_idx * Constant.TRANS_BLOCK_NUM_PER_VECTOR_F32 * 2]
                    self.tik_instance.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)
            for j in range(Constant.TRANS_BLOCK_NUM_PER_VECTOR_F32):
                with self.tik_instance.for_range(0, loop_times) as loop_idx:
                    for i in range(16):
                        a = i // 2 * 64
                        if i % 2 == 1:
                            a += 8
                        src_list[i] = src_ub[16 * (i * loop_times + j * loop_times_max * 16 + loop_idx) + 8]
                        dst_list[i] = dst_ub[a + j * 16 +
                                             (loop_idx * 2 + 1) * 128 * Constant.TRANS_BLOCK_NUM_PER_VECTOR_F32]
                    self.tik_instance.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

        elif trans_mode == Constant.COL_TO_LINE:
            for j in range(Constant.TRANS_BLOCK_NUM_PER_VECTOR_F32):
                with self.tik_instance.for_range(0, loop_times) as loop_idx:
                    for i in range(16):
                        if i < 8:
                            a = i * 64
                        else:
                            a = (i - 8) * 64
                            a += 8
                        src_list[i] = src_ub[a + j * 16 + 128 * Constant.TRANS_BLOCK_NUM_PER_VECTOR_F32 * loop_idx * 2]
                    for i in range(8):
                        # loop_times_max * 16 * 16 is one loopnum offset
                        dst_list[2 * i] = dst_ub[16 * (loop_times * i + j * loop_times_max * 16 + loop_idx)]
                        dst_list[2 * i +
                                 1] = dst_ub[16 *
                                             (loop_times * i + 8 * loop_times + j * loop_times_max * 16 + loop_idx)]
                    self.tik_instance.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

            for j in range(Constant.TRANS_BLOCK_NUM_PER_VECTOR_F32):
                with self.tik_instance.for_range(0, loop_times) as loop_idx:
                    for i in range(16):
                        if i < 8:
                            a = i * 64
                        else:
                            a = (i - 8) * 64
                            a += 8
                        # (loop_idx * 2 + 1) * 128 * 4 is second part for f32
                        src_list[i] = src_ub[a + j * 16 +
                                             (loop_idx * 2 + 1) * 128 * Constant.TRANS_BLOCK_NUM_PER_VECTOR_F32]
                    for i in range(8):
                        dst_list[2 * i] = dst_ub[16 * (loop_times * i + j * loop_times_max * 16 + loop_idx) + 8]
                        dst_list[2 * i +
                                 1] = dst_ub[16 *
                                             (loop_times * i + 8 * loop_times + j * loop_times_max * 16 + loop_idx) + 8]
                    self.tik_instance.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

    def _add_scalar_data(self, data_num, src_ub, scalar_value):
        rept = data_num // self.src_transpose_line
        max_rept = rept // Constant.MAX_REPEAT_TIMES
        mod_rept = rept % Constant.MAX_REPEAT_TIMES

        with self.tik_instance.if_scope(max_rept != 0):
            with self.tik_instance.for_range(0, max_rept) as idx:
                offset = idx * self.src_transpose_line * Constant.MAX_REPEAT_TIMES
                self.tik_instance.vadds(self.src_transpose_line, src_ub[offset], src_ub[offset], scalar_value,
                                        Constant.MAX_REPEAT_TIMES, 1, 1, 8, 8)

        with self.tik_instance.if_scope(mod_rept != 0):
            offset = self.src_transpose_line * max_rept * Constant.MAX_REPEAT_TIMES
            self.tik_instance.vadds(self.src_transpose_line, src_ub[offset], src_ub[offset], scalar_value, mod_rept, 1,
                                    1, 8, 8)

        mod_num = data_num % self.src_transpose_line
        with self.tik_instance.if_scope(mod_num != 0):
            offset = self.src_transpose_line * rept
            self.tik_instance.vadds(mod_num, src_ub[offset], src_ub[offset], scalar_value, 1, 1, 1, 8, 8)

    def _calc_add_base_attr(self, data_num, add_calc_mode):
        if add_calc_mode == Constant.ADD_BY_VECTOR:
            loop_num = data_num // self.one_loop_time_max_num * 16
            if self.reverse:
                with self.tik_instance.for_range(0, loop_num - 1) as line_idx:
                    dest_offset = (loop_num - line_idx - 2) * self.src_transpose_line
                    src_offset = (loop_num - line_idx - 1) * self.src_transpose_line
                    self.tik_instance.vadd(self.src_transpose_line, self.trans_ub[dest_offset],
                                           self.trans_ub[dest_offset], self.trans_ub[src_offset], 1, 1, 1, 1, 8, 8, 8)
            else:
                self.tik_instance.vadd(self.src_transpose_line, self.trans_ub[0], self.ub_temp, self.trans_ub[0], 1, 1,
                                       1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, loop_num - 1) as line_idx:
                    dest_offset = (line_idx + 1) * self.src_transpose_line
                    src_offset = line_idx * self.src_transpose_line
                    self.tik_instance.vadd(self.src_transpose_line, self.trans_ub[dest_offset],
                                           self.trans_ub[dest_offset], self.trans_ub[src_offset], 1, 1, 1, 1, 8, 8, 8)
                burst_len = ceil_div(self.src_transpose_line * self.x_dtype_size_transpose, self.block_size)
                self.tik_instance.data_move(self.ub_temp[0], self.trans_ub[(loop_num - 1) * self.src_transpose_line], 0,
                                            1, burst_len, 0, 0)

    def _do_data_move_in(self, ub_offset, gm_offset, data_num):
        burst_len = ceil_div(data_num * self.x_size_orig, self.block_size)
        if self.x_dtype == "bfloat16":
            self.tik_instance.data_move(self.input_ub_orig[ub_offset], self.input_gm[gm_offset], 0, 1, burst_len, 0, 0)
            common_util.conv_s4_to_s8(self.tik_instance, self.input_ub[ub_offset:], self.input_ub_orig[ub_offset:],
                                      data_num)
        else:
            if tbe_platform.api_check_support("tik.data_move_pad", self.x_dtype):
                self.tik_instance.data_move_pad(self.input_ub[ub_offset], self.input_gm[gm_offset], 1,
                                                data_num * self.x_size_orig, 0, 0)
            else:
                block_num = data_num // self.data_one_block
                with self.tik_instance.if_scope(block_num != 0):
                    self.tik_instance.data_move(self.input_ub[ub_offset], self.input_gm[gm_offset], 0, 1, block_num, 0,
                                                0)
                mod = data_num % self.data_one_block
                with self.tik_instance.if_scope(mod != 0):
                    self._gm_2_ub_by_ele(self.input_ub[ub_offset + block_num * self.data_one_block],
                                         gm_offset + block_num * self.data_one_block, mod)

    def _add_lines_value(self, start, add_e_numbers, init_value):
        with self.tik_instance.if_scope(add_e_numbers != 0):
            self.ub_temp[0].set_as(init_value)
            with self.tik_instance.for_range(0, self.src_transpose_line) as idx:
                self._add_line_value(start + idx * add_e_numbers, add_e_numbers, init_value)
                self.ub_temp[idx + 1].set_as(self.add_ub[0])
                init_value.set_as(self.add_ub[0])

    def _do_nburst_data_move_in(self, gm_offset, n_burst):
        if self.x_dtype == "bfloat16":
            dest_stride = 7
            burst_len = ceil_div(16 * self.x_size_orig, self.block_size)
            self.tik_instance.data_move(self.input_ub_orig[0], self.input_gm[gm_offset], 0, n_burst, burst_len, 0,
                                        dest_stride)
            common_util.conv_s4_to_s8(self.tik_instance, self.input_ub[0], self.input_ub_orig[0], n_burst * 16 * 8)
        else:
            dest_stride = 15 if self.x_dtype_size_transpose == 2 else 14
            if check_support_block_size_16():
                dest_stride = 7
            burst_len = ceil_div(self.trans_mod_num * self.x_size_orig, self.block_size)
            self.tik_instance.data_move(self.input_ub[0], self.input_gm[gm_offset], 0, n_burst, burst_len, 0,
                                        dest_stride)

    def _add_line_value(self, inner_start, add_e_numbers, init_value):
        input_ub_num = 16 * 15
        loop = add_e_numbers // input_ub_num
        with self.tik_instance.for_range(0, loop) as idx:
            gm_offset = inner_start + idx * input_ub_num
            self._do_nburst_data_move_in(gm_offset, input_ub_num // self.trans_mod_num)
            self._do_reduce_sum(self.input_ub, self.add_ub, input_ub_num, init_value)
            init_value.set_as(self.add_ub[0])

        mod = add_e_numbers % input_ub_num
        with self.tik_instance.if_scope(mod != 0):
            gm_offset = inner_start + loop * input_ub_num
            self._do_nburst_data_move_in(gm_offset, mod // self.trans_mod_num)
            self._do_reduce_sum(self.input_ub, self.add_ub, mod, init_value)

    def _do_reduce_sum(self, input_ub, out_ub, data_num, first_value):
        work_tensor_ub = self.tik_instance.Tensor(self.new_x_dtype_transpose, (self.src_transpose_line * 4,),
                                                  tik.scope_ubuf, "work_tensor_ub")
        out_ub[0].set_as(first_value)

        rpt = self.tik_instance.Scalar("int32", name="rpt", init_value=0)
        rpt.set_as(data_num // self.trans_mod_num)

        dst_list = [0] * 16
        src_list = [0] * 16
        if not check_support_block_size_16() and self.x_dtype_size_transpose == 2:
            dst_list = [self.trans_ub[16 * i] for i in range(16)]
            src_list = [input_ub[16 * i] for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list, rpt, 16, 16)
            if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend910B", "Ascend910_93"):
                self.tik_instance.vadd(1, self.trans_ub[0], self.trans_ub[0], out_ub, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vec_reduce_add(1, out_ub, self.trans_ub[0], work_tensor_ub, rpt * 16, 1)
            else:
                with self.tik_instance.for_range(0, data_num) as idx:
                    self.tik_instance.vadd(1, out_ub, self.trans_ub[idx * 16], out_ub, 1, 1, 1, 1, 8, 8, 8)

        elif not check_support_block_size_16():
            dst_list = [0] * 16
            src_list = [0] * 16
            for i in range(16):
                if i < 8:
                    a = i * 64
                else:
                    a = (i - 8) * 64
                    a += 8
                src_list[i] = input_ub[a]
            for i in range(8):
                dst_list[2 * i] = self.trans_ub[16 * i]
                dst_list[2 * i + 1] = self.trans_ub[16 * i + 8]

            self.tik_instance.vnchwconv(False, False, dst_list, src_list, rpt, 16, 16)
            self.tik_instance.vadd(1, self.trans_ub[0], self.trans_ub[0], out_ub, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vec_reduce_add(1, out_ub, self.trans_ub[0], work_tensor_ub, rpt * 16, 1)

        else:
            src_list = [input_ub[8 * i] for i in range(16)]
            for i in range(8):
                dst_list[2 * i] = self.trans_ub[8 * i]
                dst_list[2 * i + 1] = self.trans_ub[8 * (i + 8)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list, rpt, 16, 16)
            self.tik_instance.vadd(1, out_ub, self.trans_ub[0], out_ub, rpt * 8, 1, 1, 1, 0, 1, 0)

    def _do_transpose_for_last_mod(self, data_num):
        dst_list = [0] * 16
        src_list = [0] * 16
        if not check_support_block_size_16() and self.x_dtype_size_transpose == 2:
            dst_list = [self.trans_ub[16 * i] for i in range(16)]
            src_list = [self.input_ub[16 * i] for i in range(16)]
        else:
            if not check_support_block_size_16():
                src_list = [self.input_ub[8 * i] for i in range(8)] + \
                    [self.input_ub[8 * i + 8] for i in range(8)]
            else:
                src_list = [self.input_ub[8 * i] for i in range(16)]
            for i in range(8):
                dst_list[2 * i] = self.trans_ub[8 * i]
                dst_list[2 * i + 1] = self.trans_ub[8 * (i + 8)]
        self.tik_instance.vnchwconv(False, False, dst_list, src_list, ceil_div(data_num, 16) + 1, 16, 16)

        # new each transpose block shape is f16[16, 16] and f32[16, 8]
        data_offset = 16 if not check_support_block_size_16() and self.x_dtype_size_transpose == 2 \
            else 8
        with self.tik_instance.for_range(0, data_num - 1) as line_idx:
            self.tik_instance.vadd(1, self.trans_ub[(line_idx + 1) * data_offset],
                                   self.trans_ub[(line_idx + 1) * data_offset], self.trans_ub[line_idx * data_offset],
                                   1, 1, 1, 1, 8, 8, 8)

        if not check_support_block_size_16() and self.x_dtype_size_transpose == 2:
            dst_list = [self.input_ub[16 * i] for i in range(16)]
            src_list = [self.trans_ub[16 * i] for i in range(16)]
        elif not check_support_block_size_16():
            for i in range(16):
                src_list[i] = self.trans_ub[8 * i]
                dst_list[i] = self.input_ub[8 * i]
        else:
            src_list = [self.trans_ub[8 * i] for i in range(16)]
            for i in range(8):
                dst_list[2 * i] = self.input_ub[8 * i]
                dst_list[2 * i + 1] = self.input_ub[8 * (i + 8)]

        self.tik_instance.vnchwconv(False, False, dst_list, src_list, ceil_div(data_num, 16) + 1, 16, 16)

    def _calc_data_last_mod(self, start, data_num, scalar_value):
        if self.reverse:
            self._do_data_move_in(0, start, data_num)
        else:
            # move data to input_ub ,and reshape [ceil_div(data_num, 16), 256]
            n_burst = data_num // self.trans_mod_num
            with self.tik_instance.if_scope(n_burst != 0):
                self._do_nburst_data_move_in(start, n_burst)

            mod = data_num % self.trans_mod_num
            with self.tik_instance.if_scope(mod != 0):
                if not check_support_block_size_16() and self.x_dtype_size_transpose == 2:
                    self._do_data_move_in(Constant.VECTOR_BLOCK_SIZE_BYTE2 * n_burst,
                                          start + n_burst * self.trans_mod_num, mod)
                elif not check_support_block_size_16():
                    self._do_data_move_in(Constant.VECTOR_BLOCK_SIZE_BYTE4 * n_burst,
                                          start + n_burst * self.trans_mod_num, mod)
                else:
                    self._do_data_move_in(Constant.VECTOR_BLOCK_SIZE_BYTE2_FOR_NANO * n_burst,
                                          start + n_burst * self.trans_mod_num, mod)

        self._add_scalar_data(1, self.input_ub, scalar_value)
        self._do_transpose_for_last_mod(data_num)

    def _calc_data(self, start, data_num, calc_mode, scalar_value):
        trans_func = self._transpose_for_byte4 if check_support_block_size_16() or \
            self.x_dtype_size_transpose == 4 else self._transpose_for_byte2

        if calc_mode == Constant.CALC_FULL_MODE:
            self._do_data_move_in(0, start, data_num)
            trans_func(self.input_ub, self.trans_ub, Constant.LINE_TO_COL, self.loop_times_max)
            # add data
            self._calc_add_base_attr(data_num, Constant.ADD_BY_VECTOR)
            trans_func(self.trans_ub, self.input_ub, Constant.COL_TO_LINE, self.loop_times_max)

        elif calc_mode == Constant.CALC_MORE_THAN_ONE_REPEAT:
            loop_num = data_num // self.one_loop_time_max_num
            each_line_num = loop_num * 16
            with self.tik_instance.for_range(0, self.src_transpose_line) as line_idx:
                self._do_data_move_in(line_idx * self.one_line_max_num, start + line_idx * each_line_num, each_line_num)
            trans_func(self.input_ub, self.trans_ub, Constant.LINE_TO_COL, self.loop_times_max)
            # add data
            self._calc_add_base_attr(data_num, Constant.ADD_BY_VECTOR)
            trans_func(self.trans_ub, self.input_ub, Constant.COL_TO_LINE, self.loop_times_max)

        else:
            self._calc_data_last_mod(start, data_num, scalar_value)

    def _do_data_move_out(self, out_offset, ub_offset, data_num):
        if self.x_dtype == "bfloat16":
            common_util.conv_s8_to_s4(self.tik_instance, self.input_ub_orig[ub_offset], self.input_ub[ub_offset],
                                      data_num)
            ub_addr = self.input_ub_orig
        else:
            ub_addr = self.input_ub
        burst_len = ceil_div(data_num * self.x_size_orig, self.block_size)
        each_block_num = self.block_size // self.x_size_orig
        mod_block = each_block_num - data_num % each_block_num
        with self.tik_instance.if_scope(mod_block != 0):
            with self.tik_instance.for_range(0, mod_block) as idx:
                ub_addr[data_num + idx].set_as(0)
        self.tik_instance.data_move(self.output_gm[out_offset], ub_addr[ub_offset], 0, 1, burst_len, 0, 0)

    def _do_cumsum_for_last_dim(self, start, cumsum_data_num):
        if self.x_dtype == "bfloat16":
            self.input_ub_orig = self.tik_instance.Tensor(self.x_dtype,
                                                          (self.src_transpose_line * self.loop_times_max * 16 + 32,),
                                                          name="input_ub_orig",
                                                          scope=tik.scope_ubuf)
        self.input_ub = self.tik_instance.Tensor(self.new_x_dtype_transpose,
                                                 (self.src_transpose_line * self.loop_times_max * 16 + 32,),
                                                 name="input_ub",
                                                 scope=tik.scope_ubuf)
        self._dup_0_ub(self.input_ub, self.src_transpose_line * self.loop_times_max * 16 + 32)
        self.trans_ub = self.tik_instance.Tensor(self.new_x_dtype_transpose,
                                                 (self.loop_times_max * 16 * self.src_transpose_line + 32,),
                                                 name="out_ub",
                                                 scope=tik.scope_ubuf)
        self.add_ub = self.tik_instance.Tensor(self.new_x_dtype_transpose, (self.src_transpose_line,),
                                               name="add_ub",
                                               scope=tik.scope_ubuf)
        self.ub_temp = self.tik_instance.Tensor(self.new_x_dtype_transpose, (self.src_transpose_line * 2,),
                                                name="ub_temp",
                                                scope=tik.scope_ubuf)
        self._dup_0_ub(self.ub_temp, self.src_transpose_line * 2)
        base_value = self.tik_instance.Scalar(self.new_x_dtype_transpose, name="base_value", init_value=0)
        full_loop_nums = cumsum_data_num // self.full_loop_times_max_num
        not_full_loop_nums = cumsum_data_num % self.full_loop_times_max_num // self.one_loop_time_max_num
        with self.tik_instance.for_range(0, full_loop_nums) as full_loop_index:
            data_offset = start + full_loop_index * self.full_loop_times_max_num
            self._add_lines_value(data_offset, self.loop_times_max * 16, base_value)
            self._calc_data(data_offset, self.full_loop_times_max_num, Constant.CALC_FULL_MODE, base_value)
            base_value.set_as(self.input_ub[self.full_loop_times_max_num - 1])
            # do datamove by attr; based support atomic add
            self._do_data_move_out(data_offset, 0, self.full_loop_times_max_num)

        not_full_loop_nums = cumsum_data_num % self.full_loop_times_max_num // self.one_loop_time_max_num
        # not_full_loop_nums and calc base attr
        with self.tik_instance.if_scope(not_full_loop_nums != 0):
            data_offset = start + full_loop_nums * self.full_loop_times_max_num
            self._add_lines_value(data_offset, not_full_loop_nums * 16, base_value)
            self._calc_data(data_offset, not_full_loop_nums * self.one_loop_time_max_num,
                            Constant.CALC_MORE_THAN_ONE_REPEAT, base_value)

            # do datamove by attr; based support atomic add
            if self.x_dtype == "bfloat16":
                with self.tik_instance.for_range(0, self.src_transpose_line) as line_idx:
                    self._do_data_move_out(data_offset + line_idx * not_full_loop_nums * 16,
                                           line_idx * self.one_line_max_num, not_full_loop_nums * 16)
            else:
                burst_len = ceil_div(not_full_loop_nums * 16 * self.x_size_orig, self.block_size)
                src_sride = ceil_div((self.loop_times_max - not_full_loop_nums) * 16 * self.x_size_orig,
                                     self.block_size)
                self.tik_instance.data_move(self.output_gm[data_offset], self.input_ub[0], 0, self.src_transpose_line,
                                            burst_len, src_sride, 0)

            base_value.set_as(self.input_ub[(self.src_transpose_line - 1) * self.one_line_max_num +
                                            not_full_loop_nums * 16 - 1])
        latest_data_num = cumsum_data_num % self.full_loop_times_max_num % self.one_loop_time_max_num
        with self.tik_instance.if_scope(latest_data_num != 0):
            data_offset = start + full_loop_nums * self.full_loop_times_max_num + \
                    not_full_loop_nums * self.one_loop_time_max_num
            self._calc_data(data_offset, latest_data_num, Constant.CALC_LESS_THAN_ONE_REPEAT, base_value)
            # do datamove by attr; based support atomic add
            if self.exclusive:
                pass
            self._do_mod_data_move_out(data_offset, latest_data_num)

    def _do_mod_data_move_out(self, gm_offset, latest_data_num):
        n_burst = latest_data_num // self.trans_mod_num
        burst_len = ceil_div(self.trans_mod_num * self.x_size_orig, self.block_size)
        trans_block_number = 256 if self.x_dtype_size_transpose == 2 else 128
        if check_support_block_size_16():
            trans_block_number = 64
        if self.x_dtype == "bfloat16":
            with self.tik_instance.for_range(0, n_burst) as idx:
                self._do_data_move_out(gm_offset + idx * 16, idx * trans_block_number, 16)
        else:
            src_stride = 15 if self.x_dtype_size_transpose == 2 else 14
            if check_support_block_size_16():
                src_stride = 7
            # data shape is [x, 16] so this place need div 16
            n_burst = latest_data_num // self.trans_mod_num
            with self.tik_instance.if_scope(n_burst != 0):
                self.tik_instance.data_move(self.output_gm[gm_offset], self.input_ub[0], 0,
                                            latest_data_num // self.trans_mod_num, burst_len, src_stride, 0)

        mod_num = latest_data_num % self.trans_mod_num
        with self.tik_instance.if_scope(mod_num != 0):
            revert_num = self.trans_mod_num - mod_num
            out_offset = gm_offset + n_burst * self.trans_mod_num - revert_num
            with self.tik_instance.if_scope(n_burst != 0):
                with self.tik_instance.for_range(0, revert_num) as idx:
                    self.ub_temp[idx].set_as(self.input_ub[(n_burst - 1) * trans_block_number + mod_num + idx])
                with self.tik_instance.for_range(0, mod_num) as idx:
                    # latest dim size need greatequel than 16
                    self.ub_temp[revert_num + idx].set_as(self.input_ub[n_burst * trans_block_number + idx])
                # if need support bf16, this place should be deal
                self.tik_instance.data_move(self.output_gm[out_offset], self.ub_temp[0], 0, 1, burst_len, 0, 0)
            with self.tik_instance.else_scope():
                self.ub_temp_orig = self.tik_instance.Tensor(self.x_dtype, (32,),
                                                             name="ub_temp_orig",
                                                             scope=tik.scope_ubuf)
                self.tik_instance.data_move(self.ub_temp_orig, self.output_gm[out_offset], 0, 1, burst_len, 0, 0)
                # if need support bf16, this place should be deal
                with self.tik_instance.for_range(0, mod_num) as idx:
                    self.ub_temp_orig[revert_num + idx].set_as(self.input_ub[idx])
                self.tik_instance.data_move(self.output_gm[out_offset], self.ub_temp_orig[0], 0, 1, burst_len, 0, 0)

    def _calc_add_for_last_dim_small_reverse(self, first_line_cycle_num):
        with self.tik_instance.for_range(0, first_line_cycle_num) as cycle_idx:
            base_cycle_addr = cycle_idx * self.src_transpose_line * self.cum_times
            with self.tik_instance.for_range(0, self.cum_times - 1) as cum_idx:
                dest_offset = base_cycle_addr + (self.cum_times - 2 - cum_idx) * self.src_transpose_line
                src_offset = base_cycle_addr + (self.cum_times - 1 - cum_idx) * self.src_transpose_line
                self.tik_instance.vadd(self.src_transpose_line, self.trans_ub[dest_offset], self.trans_ub[dest_offset],
                                       self.trans_ub[src_offset], 1, 1, 1, 1, 8, 8, 8)
        if self.exclusive:
            burst_len = (self.cum_times - 1) * self.src_transpose_line * \
                self.x_dtype_size_transpose // self.block_size
            with self.tik_instance.for_range(0, first_line_cycle_num) as cycle_idx:
                with self.tik_instance.if_scope(burst_len != 0):
                    self.tik_instance.data_move(
                        self.trans_ub[self.src_transpose_line * cycle_idx * self.cum_times],
                        self.trans_ub[self.src_transpose_line * (1 + cycle_idx * self.cum_times)], 0, 1, burst_len, 0,
                        0)
                self._dup_0_ub(self.trans_ub[self.src_transpose_line * (self.cum_times * (cycle_idx + 1) - 1)],
                               self.src_transpose_line)

    def _calc_add_for_last_dim_small_copy_by_line(self, first_line_cycle_num):
        burst_len = self.src_transpose_line * self.x_dtype_size_transpose // self.block_size
        with self.tik_instance.for_range(0, first_line_cycle_num) as cycle_idx:
            with self.tik_instance.for_range(0, self.cum_times - 1) as cum_idx:
                self.tik_instance.data_move(
                    self.trans_ub[self.src_transpose_line *
                                  (self.cum_times - 1 - cum_idx + cycle_idx * self.cum_times)],
                    self.trans_ub[self.src_transpose_line *
                                  (self.cum_times - 2 - cum_idx + cycle_idx * self.cum_times)], 0, 1, burst_len, 0, 0)
            self._dup_0_ub(self.trans_ub[self.src_transpose_line * cycle_idx * self.cum_times], self.src_transpose_line)

    def _calc_add_for_last_dim_small_copy_by_cumsum(self, first_line_cycle_num):
        burst_len = (self.cum_times - 1) * self.src_transpose_line * \
                    self.x_dtype_size_transpose // self.block_size
        with self.tik_instance.for_range(0, first_line_cycle_num) as cycle_idx:
            with self.tik_instance.if_scope(burst_len != 0):
                self.tik_instance.data_move(self.trans_ub[self.src_transpose_line * (1 + cycle_idx * self.cum_times)],
                                            self.trans_ub[self.src_transpose_line * cycle_idx * self.cum_times], 0, 1,
                                            burst_len, 0, 0)
            self._dup_0_ub(self.trans_ub[self.src_transpose_line * cycle_idx * self.cum_times], self.src_transpose_line)

    def _calc_add_for_last_dim_small_not_reverse(self, first_line_cycle_num):
        with self.tik_instance.for_range(0, first_line_cycle_num) as cycle_idx:
            base_cycle_addr = cycle_idx * self.src_transpose_line * self.cum_times
            with self.tik_instance.for_range(0, self.cum_times - 1) as cum_idx:
                dest_offset = base_cycle_addr + (cum_idx + 1) * self.src_transpose_line
                src_offset = base_cycle_addr + cum_idx * self.src_transpose_line
                self.tik_instance.vadd(self.src_transpose_line, self.trans_ub[dest_offset], self.trans_ub[dest_offset],
                                       self.trans_ub[src_offset], 1, 1, 1, 1, 8, 8, 8)
        if self.exclusive:
            if tbe_platform.api_check_support("tik.vcopy"):
                self._calc_add_for_last_dim_small_copy_by_cumsum(first_line_cycle_num)
            else:
                self._calc_add_for_last_dim_small_copy_by_line(first_line_cycle_num)

    def _calc_add_for_last_dim_small(self, first_line_cycle_num):
        if self.reverse:
            self._calc_add_for_last_dim_small_reverse(first_line_cycle_num)
        else:
            self._calc_add_for_last_dim_small_not_reverse(first_line_cycle_num)

    def _do_data_move_out_for_last_dim_small(self, out_offset, ub_offset, data_num):
        if self.x_dtype == "bfloat16":
            common_util.conv_s8_to_s4(self.tik_instance, self.input_ub_orig[ub_offset], self.input_ub[ub_offset],
                                      data_num, 'round')
            ub_addr = self.input_ub_orig
        else:
            ub_addr = self.input_ub
        each_block_num = self.block_size // self.x_size_orig
        with self.tik_instance.if_scope(data_num >= each_block_num):
            burst_len = data_num * self.x_size_orig // self.block_size
            self.tik_instance.data_move(self.output_gm[out_offset], ub_addr[ub_offset], 0, 1, burst_len, 0, 0)
            mod = data_num % each_block_num
            with self.tik_instance.if_scope(mod != 0):
                gm_out_rollback_offset = out_offset + data_num - each_block_num
                ub_out_rollback_offset = ub_offset + data_num - each_block_num
                with self.tik_instance.for_range(0, each_block_num) as idx:
                    self.ub_temp[idx].set_as(ub_addr[ub_out_rollback_offset + idx])
                self.tik_instance.data_move(self.output_gm[gm_out_rollback_offset], self.ub_temp, 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            mod = each_block_num - data_num
            gm_out_rollback_offset = out_offset - mod
            self.tik_instance.data_move(self.ub_temp, self.output_gm[gm_out_rollback_offset], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, data_num) as idx:
                self.ub_temp[mod + idx].set_as(ub_addr[ub_offset + idx])
            self.tik_instance.data_move(self.output_gm[gm_out_rollback_offset], self.ub_temp, 0, 1, 1, 0, 0)

    def _do_cumsum_for_last_dim_small(self, start, num_cycle, block_id):
        if self.x_dtype == "bfloat16":
            self.input_ub_orig = self.tik_instance.Tensor(self.x_dtype,
                                                          (self.src_transpose_line * self.loop_times_max * 16 + 32,),
                                                          name="input_ub_orig",
                                                          scope=tik.scope_ubuf)
        self.input_ub = self.tik_instance.Tensor(self.new_x_dtype_transpose,
                                                 (self.src_transpose_line * self.loop_times_max * 16 + 32,),
                                                 name="input_ub",
                                                 scope=tik.scope_ubuf)
        self._dup_0_ub(self.input_ub, self.src_transpose_line * self.loop_times_max * 16 + 32)
        self.trans_ub = self.tik_instance.Tensor(self.new_x_dtype_transpose,
                                                 (self.loop_times_max * 16 * self.src_transpose_line + 32,),
                                                 name="out_ub",
                                                 scope=tik.scope_ubuf)
        self.ub_temp = self.tik_instance.Tensor(self.x_dtype, (self.src_transpose_line,),
                                                name="ub_temp",
                                                scope=tik.scope_ubuf)
        # move data to ub [src_transpose_line, loop_times_max, 16]
        loop_times = ceil_div(num_cycle, self.each_line_max_cycle)
        with self.tik_instance.for_range(0, loop_times - 1) as line_idx:
            self._do_data_move_in(line_idx * self.one_line_max_num, start + line_idx * self.each_line_max_number,
                                  self.each_line_max_number)
        mod_times = num_cycle % self.each_line_max_cycle
        with self.tik_instance.if_scope(mod_times != 0):
            self._do_data_move_in((loop_times - 1) * self.one_line_max_num,
                                  start + (loop_times - 1) * self.each_line_max_number, mod_times * self.cum_times)
        with self.tik_instance.else_scope():
            self._do_data_move_in((loop_times - 1) * self.one_line_max_num,
                                  start + (loop_times - 1) * self.each_line_max_number, self.each_line_max_number)
        # do transpose
        trans_func = self._transpose_for_byte4 if check_support_block_size_16() or \
            self.x_dtype_size_transpose == 4 else self._transpose_for_byte2
        trans_func(self.input_ub, self.trans_ub, Constant.LINE_TO_COL, self.loop_times_max)
        # add data
        with self.tik_instance.if_scope(loop_times > 1):
            self._calc_add_for_last_dim_small(self.each_line_max_cycle)
        with self.tik_instance.elif_scope(mod_times != 0):
            self._calc_add_for_last_dim_small(mod_times)
        with self.tik_instance.else_scope():
            self._calc_add_for_last_dim_small(self.each_line_max_cycle)
        # do transpose
        trans_func(self.trans_ub, self.input_ub, Constant.COL_TO_LINE, self.loop_times_max)
        # move data to gm
        with self.tik_instance.for_range(0, loop_times - 1) as line_idx:
            out_offset = start + line_idx * self.each_line_max_number
            ub_offset = line_idx * self.one_line_max_num
            self._do_data_move_out_for_last_dim_small(out_offset, ub_offset, self.each_line_max_number)

        with self.tik_instance.if_scope(mod_times != 0):
            out_offset = start + (loop_times - 1) * self.each_line_max_number
            ub_offset = (loop_times - 1) * self.one_line_max_num
            self._do_data_move_out_for_last_dim_small(out_offset, ub_offset, mod_times * self.cum_times)
        with self.tik_instance.else_scope():
            out_offset = start + (loop_times - 1) * self.each_line_max_number
            ub_offset = (loop_times - 1) * self.one_line_max_num
            self._do_data_move_out_for_last_dim_small(out_offset, ub_offset, self.each_line_max_number)

    def _handle_last_dim_small(self, block_id):
        '''
        do cumsum for axis is -1, this function used transpose for improve calc speed
        '''
        if self.is_support_transpose_mode == 1:
            self.each_line_max_cycle.set_as(16 * self.loop_times_max // self.cum_times)
            self.each_line_max_number.set_as(self.each_line_max_cycle * self.cum_times)
            each_time_max_cycle = self.tik_instance.Scalar("int64",
                                                           name="each_time_max_cycle",
                                                           init_value=self.each_line_max_cycle *
                                                           self.src_transpose_line)
            core_cycle = self.tik_instance.Scalar("int64", name="core_cycle", init_value=0)
            core_start = self.tik_instance.Scalar("int64", name="core_start", init_value=0)
            with self.tik_instance.if_scope(block_id < self.core_tail):
                core_cycle.set_as(self.core_cycle + 1)
                core_start.set_as(core_cycle * self.cum_times * self.per_cum_num * block_id)
            with self.tik_instance.else_scope():
                core_cycle.set_as(self.core_cycle)
                core_start.set_as(core_cycle * self.cum_times * self.per_cum_num * block_id + \
                    self.core_tail * self.cum_times * self.per_cum_num)

            looptimes = core_cycle // each_time_max_cycle
            with self.tik_instance.for_range(0, looptimes) as line_idx:
                start = core_start + line_idx * self.each_line_max_number * self.src_transpose_line
                self._do_cumsum_for_last_dim_small(start, each_time_max_cycle, block_id)

            modtimes = core_cycle % each_time_max_cycle
            with self.tik_instance.if_scope(modtimes != 0):
                start = core_start + looptimes * self.each_line_max_number * self.src_transpose_line
                self._do_cumsum_for_last_dim_small(start, modtimes, block_id)

    def _handle_last_dim_large(self, block_id):
        '''
        do cumsum for axis is -1, this function used transpose for improve calc speed
        '''
        if self.is_support_transpose_mode == 1:
            if not check_support_block_size_16():
                with self.tik_instance.for_range(0, self.core_cycle) as cycle:
                    start = (cycle * self.act_core_num + block_id) * self.cum_times * self.per_cum_num
                    self._do_cumsum_for_last_dim(start, self.cum_times)

                with self.tik_instance.if_scope(block_id < self.core_tail):
                    start_tail = (block_id + self.act_core_num * self.core_cycle) * self.cum_times * self.per_cum_num
                    self._do_cumsum_for_last_dim(start_tail, self.cum_times)
            else:
                self._handle_last_dim_large_nano(block_id)

    def _do_cumsum_for_last_dim_nano_full(self, args):
        (start, full_loop_nums, line_num, one_calc_num, cumsum_data_num) = args
        with self.tik_instance.for_range(0, full_loop_nums) as full_loop_index:
            with self.tik_instance.for_range(0, line_num) as line_idx:
                self.tik_instance.data_move(
                    self.input_ub[line_idx * one_calc_num],
                    self.input_gm[start + full_loop_index * one_calc_num + line_idx * cumsum_data_num], 0, 1,
                    ceil_div(one_calc_num * self.x_dtype_size_transpose, self.block_size), 0, 0)
            # transpose
            trans_func = self._transpose_for_byte4
            trans_func(self.input_ub, self.trans_ub, Constant.LINE_TO_COL, self.loop_times_max)
            # calc
            self.tik_instance.vadd(line_num, self.trans_ub[0], self.ub_temp, self.trans_ub[0], 1, 1, 1, 1, 8, 8, 8)
            with self.tik_instance.for_range(0, one_calc_num - 1) as line_idx:
                dest_offset = (line_idx + 1) * self.src_transpose_line
                src_offset = line_idx * self.src_transpose_line
                self.tik_instance.vadd(line_num, self.trans_ub[dest_offset], self.trans_ub[dest_offset],
                                       self.trans_ub[src_offset], 1, 1, 1, 1, 8, 8, 8)
            burst_len = ceil_div(line_num * self.x_dtype_size_transpose, self.block_size)
            self.tik_instance.data_move(self.ub_temp, self.trans_ub[(one_calc_num - 1) * self.src_transpose_line], 0, 1,
                                        burst_len, 0, 0)
            # transpose
            trans_func(self.trans_ub, self.input_ub, Constant.COL_TO_LINE, self.loop_times_max)
            with self.tik_instance.for_range(0, line_num) as line_idx:
                self.tik_instance.data_move(
                    self.output_gm[start + full_loop_index * one_calc_num + line_idx * cumsum_data_num],
                    self.input_ub[line_idx * one_calc_num], 0, 1,
                    ceil_div(one_calc_num * self.x_dtype_size_transpose, self.block_size), 0, 0)

    def _do_cumsum_for_last_dim_nano_mod(self, args):
        (start, full_loop_nums, not_full_loop_nums, line_num, one_calc_num, cumsum_data_num) = args
        with self.tik_instance.if_scope(not_full_loop_nums != 0):
            # do data move in
            mod_calc_num = ceil_div(not_full_loop_nums, Constant.NUMBER_SIXTEEN) * Constant.NUMBER_SIXTEEN
            with self.tik_instance.for_range(0, line_num) as line_idx:
                self.tik_instance.data_move(
                    self.input_ub[line_idx * mod_calc_num],
                    self.input_gm[start + full_loop_nums * one_calc_num + line_idx * cumsum_data_num], 0, 1,
                    ceil_div(not_full_loop_nums * self.x_dtype_size_transpose, self.block_size), 0, 0)
            # transpose
            trans_func = self._transpose_for_byte4
            trans_func(self.input_ub, self.trans_ub, Constant.LINE_TO_COL, ceil_div(not_full_loop_nums, 16))
            # calc
            self.tik_instance.vadd(line_num, self.trans_ub[0], self.ub_temp, self.trans_ub[0], 1, 1, 1, 1, 8, 8, 8)
            with self.tik_instance.for_range(0, not_full_loop_nums - 1) as line_idx:
                dest_offset = (line_idx + 1) * self.src_transpose_line
                src_offset = line_idx * self.src_transpose_line
                self.tik_instance.vadd(line_num, self.trans_ub[dest_offset], self.trans_ub[dest_offset],
                                       self.trans_ub[src_offset], 1, 1, 1, 1, 8, 8, 8)
            # transpose
            trans_func(self.trans_ub, self.input_ub, Constant.COL_TO_LINE, ceil_div(not_full_loop_nums, 16))
            # do data move out
            with self.tik_instance.for_range(0, line_num) as line_idx:
                self._do_data_move_out_for_last_dim_small(
                    start + full_loop_nums * one_calc_num + line_idx * cumsum_data_num, line_idx * mod_calc_num,
                    not_full_loop_nums)

    def _do_cumsum_for_last_dim_nano(self, start, line_num, cumsum_data_num):
        self.input_ub = self.tik_instance.Tensor(
            self.new_x_dtype_transpose, (self.src_transpose_line * self.loop_times_max * Constant.NUMBER_SIXTEEN + 32,),
            name="input_ub",
            scope=tik.scope_ubuf)
        self._dup_0_ub(self.input_ub, self.src_transpose_line * self.loop_times_max * Constant.NUMBER_SIXTEEN + 32)
        self.trans_ub = self.tik_instance.Tensor(
            self.new_x_dtype_transpose,
            (self.loop_times_max * Constant.NUMBER_SIXTEEN * self.src_transpose_line + self.src_transpose_line,),
            name="out_ub",
            scope=tik.scope_ubuf)
        self.ub_temp = self.tik_instance.Tensor(self.new_x_dtype_transpose, (self.src_transpose_line * 2,),
                                                name="ub_temp",
                                                scope=tik.scope_ubuf)
        self._dup_0_ub(self.ub_temp, line_num * 2)
        one_calc_num = self.loop_times_max * Constant.NUMBER_SIXTEEN
        full_loop_nums = cumsum_data_num // one_calc_num
        not_full_loop_nums = cumsum_data_num % one_calc_num

        args = (start, full_loop_nums, line_num, one_calc_num, cumsum_data_num)
        self._do_cumsum_for_last_dim_nano_full(args)
        args = (start, full_loop_nums, not_full_loop_nums, line_num, one_calc_num, cumsum_data_num)
        self._do_cumsum_for_last_dim_nano_mod(args)

    def _handle_last_dim_large_nano(self, block_id):
        total_cycle = self.tik_instance.Scalar("int64", name="core_cycle", init_value=0)
        total_cycle.set_as(self.core_cycle)
        base_start = self.tik_instance.Scalar("int64", name="base_start", init_value=0)
        with self.tik_instance.if_scope(block_id < self.core_tail):
            total_cycle.set_as(total_cycle + 1)
            base_start.set_as(total_cycle * block_id * self.cum_times * self.per_cum_num)
        with self.tik_instance.else_scope():
            base_start.set_as((self.core_cycle * block_id + self.core_tail) * self.cum_times * self.per_cum_num)
        loop_rpt = total_cycle // self.src_transpose_line
        loop_mod = total_cycle % self.src_transpose_line
        with self.tik_instance.for_range(0, loop_rpt) as loop_idx:
            start = base_start + loop_idx * self.src_transpose_line * self.cum_times * self.per_cum_num
            self._do_cumsum_for_last_dim_nano(start, self.src_transpose_line, self.cum_times)
        with self.tik_instance.if_scope(loop_mod != 0):
            start = base_start + loop_rpt * self.src_transpose_line * self.cum_times * self.per_cum_num
            self._do_cumsum_for_last_dim_nano(start, loop_mod, self.cum_times)

    def _compute(self):
        with self.tik_instance.for_range(0, self.act_core_num, block_num=self.act_core_num) as block_id:
            if self.support_atomic == 1:
                with self.tik_instance.if_scope(self.tiling_mode == 201):
                    self._handle_small_special(block_id)
                with self.tik_instance.elif_scope(self.tiling_mode == 202):
                    self._handle_default(block_id)
                with self.tik_instance.elif_scope(self.tiling_mode == 203):
                    self._handle_loop_speed(block_id)
                with self.tik_instance.elif_scope(self.tiling_mode == 204):
                    self._handle_last_dim_large(block_id)
                with self.tik_instance.elif_scope(self.tiling_mode == 205):
                    self._handle_last_dim_small(block_id)
            else:
                with self.tik_instance.if_scope(self.tiling_mode == 201):
                    self._handle_small_special(block_id)
                with self.tik_instance.elif_scope(self.tiling_mode == 202):
                    self._handle_default(block_id)
                with self.tik_instance.elif_scope(self.tiling_mode == 204):
                    self._handle_last_dim_large(block_id)
                with self.tik_instance.elif_scope(self.tiling_mode == 205):
                    self._handle_last_dim_small(block_id)

    def _handle_default(self, block_id):
        with self.tik_instance.for_range(0, self.core_cycle) as cycle:
            start = (cycle * self.act_core_num + block_id) * self.cum_times * self.per_cum_num
            with self.tik_instance.if_scope(self.per_cum_num < self.data_one_block):
                self.__handle_small_piece(start)
            with self.tik_instance.else_scope():
                self.__handle_one_core(start)
        with self.tik_instance.if_scope(block_id < self.core_tail):
            start_tail = (block_id + self.act_core_num * self.core_cycle) * self.cum_times * self.per_cum_num
            with self.tik_instance.if_scope(self.per_cum_num < self.data_one_block):
                self.__handle_small_piece(start_tail)
            with self.tik_instance.else_scope():
                self.__handle_one_core(start_tail)

    def __handle_one_core(self, start_idx):
        out_ub = self.tik_instance.Tensor(self.new_x_dtype, (self.one_max_nums * self.data_one_block,),
                                          name="out_ub",
                                          scope=tik.scope_ubuf)
        out_ub_tail = self.tik_instance.Tensor(self.new_x_dtype, (self.data_one_block,),
                                               name="out_ub_tail",
                                               scope=tik.scope_ubuf)
        input_ub = self.tik_instance.Tensor(self.new_x_dtype, (self.one_max_nums * self.data_one_block,),
                                            name="input_ub",
                                            scope=tik.scope_ubuf)
        input_ub_tail = self.tik_instance.Tensor(self.new_x_dtype, (self.data_one_block,),
                                                 name="input_ub_tail",
                                                 scope=tik.scope_ubuf)
        if self.reverse:
            start_idx = start_idx + (self.cum_times - 1) * self.per_cum_num
        max_ub_cycle = self.per_cum_num // (self.one_max_nums * self.data_one_block)
        remaining_ub_nums = self.per_cum_num % (self.one_max_nums * self.data_one_block)
        last_ub_nums = remaining_ub_nums % self.data_one_block
        with self.tik_instance.for_range(0, max_ub_cycle) as burst_cycle:
            max_ub_start_idx = start_idx + burst_cycle * (self.one_max_nums * self.data_one_block)
            if self.exclusive:
                self._dup_0_ub(out_ub, (self.one_max_nums * self.data_one_block))
            else:
                self._gm_2_ub(out_ub, max_ub_start_idx, self.one_max_nums)
            self._ub_2_gm(out_ub, max_ub_start_idx, self.one_max_nums)
            with self.tik_instance.for_range(1, self.cum_times) as cum_cycle:
                if self.exclusive:
                    cum_cycle = cum_cycle - 1
                axis_start_idx = max_ub_start_idx + self.per_cum_num * cum_cycle
                if self.reverse:
                    axis_start_idx = max_ub_start_idx - self.per_cum_num * cum_cycle
                self._gm_2_ub(input_ub, axis_start_idx, self.one_max_nums)
                self._batch_add(self.one_max_nums * self.data_one_block, out_ub, input_ub, out_ub)
                if self.exclusive and self.reverse:
                    self._ub_2_gm(out_ub, axis_start_idx - self.per_cum_num, self.one_max_nums)
                elif self.exclusive and not self.reverse:
                    self._ub_2_gm(out_ub, axis_start_idx + self.per_cum_num, self.one_max_nums)
                else:
                    self._ub_2_gm(out_ub, axis_start_idx, self.one_max_nums)
        with self.tik_instance.if_scope(remaining_ub_nums >= self.data_one_block):
            remaining_ub_start_idx = start_idx + max_ub_cycle * (self.one_max_nums * self.data_one_block)
            burst_len = remaining_ub_nums // self.data_one_block
            if self.exclusive:
                self._dup_0_ub(out_ub, burst_len * self.data_one_block)
            else:
                self._gm_2_ub(out_ub, remaining_ub_start_idx, burst_len)
            self._ub_2_gm(out_ub, remaining_ub_start_idx, burst_len)
            with self.tik_instance.for_range(1, self.cum_times) as cum_cycle:
                if self.exclusive:
                    cum_cycle = cum_cycle - 1
                remaining_axis_start_idx = remaining_ub_start_idx + self.per_cum_num * cum_cycle
                if self.reverse:
                    remaining_axis_start_idx = remaining_ub_start_idx - self.per_cum_num * cum_cycle
                self._gm_2_ub(input_ub, remaining_axis_start_idx, burst_len)
                self._batch_add(burst_len * self.data_one_block, out_ub, input_ub, out_ub)
                if self.exclusive and self.reverse:
                    self._ub_2_gm(out_ub, remaining_axis_start_idx - self.per_cum_num, burst_len)
                elif self.exclusive and not self.reverse:
                    self._ub_2_gm(out_ub, remaining_axis_start_idx + self.per_cum_num, burst_len)
                else:
                    self._ub_2_gm(out_ub, remaining_axis_start_idx, burst_len)
        with self.tik_instance.if_scope(last_ub_nums > 0):
            last_ub_start_idx = start_idx + self.per_cum_num - self.data_one_block
            if self.exclusive:
                self._dup_0_ub(out_ub_tail, self.data_one_block)
            else:
                self._gm_2_ub(out_ub_tail, last_ub_start_idx, 1)
            self._ub_2_gm(out_ub_tail, last_ub_start_idx, 1)
            with self.tik_instance.for_range(1, self.cum_times) as cum_cycle:
                if self.exclusive:
                    cum_cycle = cum_cycle - 1
                last_axis_start_idx = last_ub_start_idx + self.per_cum_num * cum_cycle
                if self.reverse:
                    last_axis_start_idx = last_ub_start_idx - self.per_cum_num * cum_cycle
                self._gm_2_ub(input_ub_tail, last_axis_start_idx, 1)
                self.tik_instance.vadd(self.data_one_block, out_ub_tail, input_ub_tail, out_ub_tail, 1, 1, 1, 1, 8, 8,
                                       8)
                if self.exclusive and self.reverse:
                    self._ub_2_gm(out_ub_tail, last_axis_start_idx - self.per_cum_num, 1)
                elif self.exclusive and not self.reverse:
                    self._ub_2_gm(out_ub_tail, last_axis_start_idx + self.per_cum_num, 1)
                else:
                    self._ub_2_gm(out_ub_tail, last_axis_start_idx, 1)

    def __handle_small_piece(self, start_idx):
        if self.reverse:
            start_idx = start_idx + (self.cum_times - 1) * self.per_cum_num
        out_ub = self.tik_instance.Tensor(self.new_x_dtype, (self.data_one_block,), name="out_ub", scope=tik.scope_ubuf)
        input_ub = self.tik_instance.Tensor(self.new_x_dtype, (self.data_one_block,),
                                            name="input_ub",
                                            scope=tik.scope_ubuf)
        last_out_ub = self.tik_instance.Tensor(self.new_x_dtype, (self.data_one_block,),
                                               name="last_out_ub",
                                               scope=tik.scope_ubuf)
        # cumulative
        if not self.reverse and not self.exclusive:
            self._gm_2_ub(out_ub, start_idx, 1)
            self._ub_2_gm(out_ub, start_idx, 1)
            # middle
            with self.tik_instance.for_range(1, self.cum_times - (self.remain_cycle - 1)) as cum_cycle:
                axis_start_idx = start_idx + self.per_cum_num * cum_cycle
                self._gm_2_ub(input_ub, axis_start_idx, 1)
                self.tik_instance.vadd(self.data_one_block, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                self._ub_2_gm(out_ub, axis_start_idx, 1)
            # last 8
            with self.tik_instance.for_range(0, self.data_one_block -
                                             (self.remain_cycle - 1) * self.per_cum_num) as last_ub_idx:
                last_out_ub[last_ub_idx].set_as(out_ub[self.per_cum_num - (self.data_one_block -
                                                                           (self.remain_cycle - 1) * self.per_cum_num) +
                                                       last_ub_idx])
            with self.tik_instance.for_range(0, self.remain_cycle - 1) as last_cum_cycle:
                last_axis_stat_idx = start_idx + self.per_cum_num * last_cum_cycle + self.per_cum_num * (
                    self.cum_times - (self.remain_cycle - 1))
                self._gm_2_ub_by_ele(input_ub, last_axis_stat_idx, self.per_cum_num)
                self.tik_instance.vadd(self.per_cum_num, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, self.per_cum_num) as last_cum_ub_idx:
                    last_out_ub[self.data_one_block - (self.remain_cycle - 1) * self.per_cum_num + \
                                self.per_cum_num * last_cum_cycle + last_cum_ub_idx].set_as(
                        out_ub[last_cum_ub_idx])
            self._ub_2_gm(last_out_ub,
                          self.per_cum_num * (self.cum_times - 1) - self.data_one_block + self.per_cum_num + start_idx,
                          1)
        elif self.reverse and not self.exclusive:
            # last 8
            self._gm_2_ub_by_ele(out_ub, start_idx, self.per_cum_num)
            with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as tem_idx:
                out_ub[self.per_cum_num + tem_idx].set_as(0)
            with self.tik_instance.for_range(0, self.per_cum_num) as last_ub_idx:
                last_out_ub[self.data_one_block - self.per_cum_num + last_ub_idx].set_as(out_ub[last_ub_idx])
            with self.tik_instance.for_range(1, self.remain_cycle - 1) as last_cum_cycle:
                temp_start_idx = start_idx - self.per_cum_num * last_cum_cycle
                self._gm_2_ub_by_ele(input_ub, temp_start_idx, self.per_cum_num)
                with self.tik_instance.for_range(0, (self.data_one_block - self.per_cum_num) - \
                                                    self.per_cum_num * last_cum_cycle) as tem_idx:
                    input_ub[self.per_cum_num * (last_cum_cycle + 1) + tem_idx].set_as(0)
                self.tik_instance.vadd(self.data_one_block, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, self.per_cum_num) as last_ub_idx:
                    last_out_ub[self.data_one_block - self.per_cum_num * (last_cum_cycle + 1) + last_ub_idx].set_as(
                        out_ub[last_ub_idx])
            self._ub_2_gm(last_out_ub, start_idx - self.data_one_block + self.per_cum_num, 1)
            # middle
            with self.tik_instance.for_range(0, self.cum_times - (self.remain_cycle - 1)) as most_idx:
                axis_start_idx = start_idx - (self.remain_cycle - 1) * self.per_cum_num - self.per_cum_num * most_idx
                self._gm_2_ub(input_ub, axis_start_idx, 1)
                self.tik_instance.vadd(self.data_one_block, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                self._ub_2_gm_by_ele(out_ub, axis_start_idx, self.per_cum_num)
        elif not self.reverse and self.exclusive:
            self._dup_0_ub(out_ub, self.data_one_block)
            self._ub_2_gm(out_ub, start_idx, 1)
            # middle
            with self.tik_instance.for_range(1, self.cum_times - (self.remain_cycle - 1)) as cum_cycle:
                axis_start_idx = start_idx + self.per_cum_num * (cum_cycle - 1)
                self._gm_2_ub(input_ub, axis_start_idx, 1)
                self.tik_instance.vadd(self.data_one_block, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                self._ub_2_gm(out_ub, axis_start_idx + self.per_cum_num, 1)
            # last 8
            with self.tik_instance.for_range(0, self.data_one_block -
                                             (self.remain_cycle - 1) * self.per_cum_num) as last_ub_idx:
                last_out_ub[last_ub_idx].set_as(out_ub[self.per_cum_num - (self.data_one_block -
                                                                           (self.remain_cycle - 1) * self.per_cum_num) +
                                                       last_ub_idx])
            with self.tik_instance.for_range(0, self.remain_cycle - 1) as last_cum_cycle:
                last_axis_stat_idx = start_idx + self.per_cum_num * last_cum_cycle + self.per_cum_num * (
                    self.cum_times - (self.remain_cycle - 1) - 1)
                self._gm_2_ub_by_ele(input_ub, last_axis_stat_idx, self.per_cum_num)
                self.tik_instance.vadd(self.per_cum_num, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, self.per_cum_num) as last_cum_ub_idx:
                    last_out_ub[self.data_one_block - (
                            self.remain_cycle - 1) * self.per_cum_num + \
                                self.per_cum_num * last_cum_cycle + last_cum_ub_idx].set_as(out_ub[last_cum_ub_idx])
            self._ub_2_gm(last_out_ub,
                          self.per_cum_num * (self.cum_times - 1) - self.data_one_block + self.per_cum_num + start_idx,
                          1)
        else:  # self.reverse and self.exclusive
            # last 8
            self._dup_0_ub(out_ub, self.data_one_block)
            with self.tik_instance.for_range(0, self.per_cum_num) as last_ub_idx:
                last_out_ub[self.data_one_block - self.per_cum_num + last_ub_idx].set_as(out_ub[last_ub_idx])
            with self.tik_instance.for_range(1, self.remain_cycle - 1) as last_cum_cycle:
                temp_start_idx = start_idx - self.per_cum_num * (last_cum_cycle - 1)
                self._gm_2_ub_by_ele(input_ub, temp_start_idx, self.per_cum_num)
                self.tik_instance.vadd(self.per_cum_num, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, self.per_cum_num) as last_ub_idx:
                    last_out_ub[self.data_one_block - self.per_cum_num * (last_cum_cycle + 1) + last_ub_idx].set_as(
                        out_ub[last_ub_idx])
            self._ub_2_gm(last_out_ub, start_idx - self.data_one_block + self.per_cum_num, 1)
            next_temp_start_idx = start_idx - (self.remain_cycle - 2) * self.per_cum_num
            self._gm_2_ub_by_ele(input_ub, next_temp_start_idx, self.per_cum_num)
            self.tik_instance.vadd(self.per_cum_num, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
            with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as out_ub_idx:
                out_ub[self.per_cum_num + out_ub_idx].set_as(
                    last_out_ub[self.data_one_block - (self.per_cum_num * (self.remain_cycle - 1)) + out_ub_idx])
            self._ub_2_gm(out_ub, next_temp_start_idx - self.per_cum_num, 1)
            # middle
            with self.tik_instance.for_range(0, self.cum_times - self.remain_cycle) as most_idx:
                axis_start_idx = start_idx - self.remain_cycle * self.per_cum_num - self.per_cum_num * (most_idx - 1)
                self._gm_2_ub(input_ub, axis_start_idx, 1)
                self.tik_instance.vadd(self.data_one_block, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                self._ub_2_gm(out_ub, axis_start_idx - self.per_cum_num, 1)

    def _handle_small_special_do_move_out(self, core_num, start_idx_org, last_out_ub):
        data_num = core_num * self.cum_times * self.per_cum_num
        with self.tik_instance.if_scope(data_num >= self.data_one_block):
            burst_len = data_num // self.data_one_block
            self._ub_2_gm(last_out_ub, start_idx_org, burst_len)
            mod = data_num % self.data_one_block
            with self.tik_instance.if_scope(mod != 0):
                gm_out_rollback_offset = start_idx_org + data_num - self.data_one_block
                ub_out_rollback_offset = data_num - self.data_one_block
                with self.tik_instance.for_range(0, self.data_one_block) as idx:
                    self.out_ub[idx].set_as(last_out_ub[ub_out_rollback_offset + idx])
                self._ub_2_gm(self.out_ub, gm_out_rollback_offset, 1)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(start_idx_org >= self.data_one_block):
                mod = self.data_one_block - data_num
            with self.tik_instance.else_scope():
                mod = 0
            gm_out_rollback_offset = start_idx_org - mod
            self.tik_instance.data_move(self.out_ub, self.output_gm[gm_out_rollback_offset], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, data_num) as idx:
                self.out_ub[mod + idx].set_as(last_out_ub[idx])
            self._ub_2_gm(self.out_ub, gm_out_rollback_offset, 1)

    def _handle_small_special_per_cum_num(self, last_out_offset, start_idx, last_out_ub, out_ub_fp32, input_ub_fp32):
        self.tik_instance.vconv(self.data_one_block, "", out_ub_fp32, self.out_ub, 1, 1, 1, 8, 4)
        with self.tik_instance.for_range(0, self.per_cum_num) as first_idx:
            last_out_ub[last_out_offset + first_idx].set_as(self.out_ub[first_idx])
        
        self._gm_2_ub_by_ele(self.input_ub, start_idx, self.cum_times)
        self.tik_instance.vconv(self.cum_times, "", input_ub_fp32, self.input_ub, 1, 1, 1, 8, 4)
        with self.tik_instance.for_range(1, self.cum_times) as cycle_idx:
            input_data = self.tik_instance.Scalar("float32", name="input_data", 
                                                  init_value=input_ub_fp32[cycle_idx])
            self.tik_instance.vadds(self.per_cum_num, out_ub_fp32, out_ub_fp32, input_data, 1, 1, 1, 8, 8)
            self.tik_instance.vconv(self.per_cum_num, "", self.out_ub, out_ub_fp32, 1, 1, 1, 4, 8)
            with self.tik_instance.for_range(0, self.per_cum_num) as cumulative_cycle_idx:
                last_out_ub[last_out_offset + cycle_idx * self.per_cum_num + cumulative_cycle_idx].set_as(
                    self.out_ub[cumulative_cycle_idx])

    def _handle_small_special_do_calc_by_attr(self, last_out_offset, start_idx, last_out_ub):
        if not self.exclusive and not self.reverse:
            with self.tik_instance.for_range(0, self.per_cum_num) as first_idx:
                last_out_ub[last_out_offset + first_idx].set_as(self.out_ub[first_idx])
            with self.tik_instance.for_range(1, self.cum_times) as cycle_idx:
                self._gm_2_ub_by_ele(self.input_ub, start_idx + cycle_idx * self.per_cum_num, self.per_cum_num)
                self.tik_instance.vadd(self.per_cum_num, self.out_ub, self.input_ub, self.out_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, self.per_cum_num) as cumulative_cycle_idx:
                    last_out_ub[last_out_offset + cycle_idx * self.per_cum_num + cumulative_cycle_idx].set_as(
                        self.out_ub[cumulative_cycle_idx])
        elif self.exclusive and not self.reverse:
            with self.tik_instance.for_range(0, self.per_cum_num) as first_idx:
                last_out_ub[last_out_offset + first_idx].set_as(self.out_ub[first_idx])
            with self.tik_instance.for_range(1, self.cum_times) as cycle_idx:
                self._gm_2_ub_by_ele(self.input_ub, start_idx + (cycle_idx - 1) * self.per_cum_num, self.per_cum_num)
                self.tik_instance.vadd(self.per_cum_num, self.out_ub, self.input_ub, self.out_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, self.per_cum_num) as cumulative_cycle_idx:
                    last_out_ub[last_out_offset + cycle_idx * self.per_cum_num + cumulative_cycle_idx].set_as(
                        self.out_ub[cumulative_cycle_idx])
        elif not self.exclusive and self.reverse:
            with self.tik_instance.for_range(0, self.per_cum_num) as first_idx:
                last_out_ub[last_out_offset + (self.cum_times - 1) * self.per_cum_num + first_idx].set_as(
                    self.out_ub[first_idx])
            with self.tik_instance.for_range(1, self.cum_times) as cycle_idx:
                self._gm_2_ub_by_ele(self.input_ub, start_idx - cycle_idx * self.per_cum_num, self.per_cum_num)
                self.tik_instance.vadd(self.per_cum_num, self.out_ub, self.input_ub, self.out_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, self.per_cum_num) as cumulative_cycle_idx:
                    last_out_ub[last_out_offset + (self.cum_times - 1) * self.per_cum_num -
                                cycle_idx * self.per_cum_num + cumulative_cycle_idx].set_as(
                                    self.out_ub[cumulative_cycle_idx])
        else:  # self.reverse and self.exclusive
            with self.tik_instance.for_range(0, self.per_cum_num) as first_idx:
                last_out_ub[last_out_offset + (self.cum_times - 1) * self.per_cum_num + first_idx].set_as(
                    self.out_ub[first_idx])
            with self.tik_instance.for_range(1, self.cum_times) as cycle_idx:
                self._gm_2_ub_by_ele(self.input_ub, start_idx - (cycle_idx - 1) * self.per_cum_num, self.per_cum_num)
                self.tik_instance.vadd(self.per_cum_num, self.out_ub, self.input_ub, self.out_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, self.per_cum_num) as cumulative_cycle_idx:
                    last_out_ub[last_out_offset + (self.cum_times - 1) * self.per_cum_num -
                                cycle_idx * self.per_cum_num + cumulative_cycle_idx].set_as(
                                    self.out_ub[cumulative_cycle_idx])
                    
    def _handle_small_special_do_calc_deal_attr(self, core_num, start_idx_org, last_out_ub):
        with self.tik_instance.for_range(0, core_num) as cycle:
            start_idx = start_idx_org + cycle * self.cum_times * self.per_cum_num
            last_out_offset = cycle * self.cum_times * self.per_cum_num
            if self.reverse:
                start_idx = start_idx + (self.cum_times - 1) * self.per_cum_num
            if self.exclusive:
                self._dup_0_ub(self.out_ub, self.data_one_block)
            else:
                self._gm_2_ub_by_ele(self.out_ub, start_idx, self.per_cum_num)
            self._handle_small_special_do_calc_by_attr(last_out_offset, start_idx, last_out_ub)

    def _handle_small_special_do_calc_deal_per_cum_num(self, core_num, start_idx_org, last_out_ub):
        out_ub_fp32 = self.tik_instance.Tensor("float32", (self.data_one_block,),
                                                    name="out_ub_fp32", scope=tik.scope_ubuf)
        input_ub_fp32 = self.tik_instance.Tensor("float32", (self.data_one_block,),
                                                    name="input_ub_fp32", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, core_num) as cycle:
            start_idx = start_idx_org + cycle * self.cum_times * self.per_cum_num
            last_out_offset = cycle * self.cum_times * self.per_cum_num
            self._gm_2_ub_by_ele(self.out_ub, start_idx, self.per_cum_num)
            self._handle_small_special_per_cum_num(last_out_offset, start_idx, last_out_ub, out_ub_fp32, input_ub_fp32)

    def _handle_small_special_do_calc(self, core_num, start_idx_org, last_out_ub):
        if self.x_dtype == 'float16' and util_soc_common.is_v220() and (not self.exclusive) and (not self.reverse):
            with self.tik_instance.if_scope(tik.all(self.per_cum_num == 1, self.cum_times < self.data_one_block)):
                self._handle_small_special_do_calc_deal_per_cum_num(core_num, start_idx_org, last_out_ub)
            with self.tik_instance.else_scope():
                self._handle_small_special_do_calc_deal_attr(core_num, start_idx_org, last_out_ub)
        else:
            self._handle_small_special_do_calc_deal_attr(core_num, start_idx_org, last_out_ub)
        self._handle_small_special_do_move_out(core_num, start_idx_org, last_out_ub)

    def _handle_small_special(self, block_id):
        start = block_id * self.cum_times * self.per_cum_num
        self.out_ub = self.tik_instance.Tensor(self.new_x_dtype, (self.data_one_block,),
                                               name="out_ub",
                                               scope=tik.scope_ubuf)
        self.input_ub = self.tik_instance.Tensor(self.new_x_dtype, (self.data_one_block,),
                                                 name="input_ub",
                                                 scope=tik.scope_ubuf)
        max_last_out_ub_ele = (self.one_max_nums * self.data_one_block // 32 + 1) * 32
        last_out_ub = self.tik_instance.Tensor(self.new_x_dtype, (max_last_out_ub_ele,),
                                               name="last_out_ub",
                                               scope=tik.scope_ubuf)
        max_per_cumsum = max_last_out_ub_ele // (self.cum_times * self.per_cum_num)
        loop_core_num = self.core_num // max_per_cumsum
        mod_core_num = self.core_num % max_per_cumsum

        with self.tik_instance.for_range(0, loop_core_num) as core_num_loop_idx:
            start_idx = start + core_num_loop_idx * max_per_cumsum * self.cum_times * self.per_cum_num
            self._handle_small_special_do_calc(max_per_cumsum, start_idx, last_out_ub)

        with self.tik_instance.if_scope(mod_core_num != 0):
            start_idx = start + loop_core_num * max_per_cumsum * self.cum_times * self.per_cum_num
            self._handle_small_special_do_calc(mod_core_num, start_idx, last_out_ub)

    def _handle_loop_speed(self, block_id):
        self.tik_instance.set_atomic_add(self.x_dtype)
        with self.tik_instance.if_scope(block_id < self.core_tail):
            start_idx = (self.core_cycle + 1) * block_id * self.per_cum_num
            if self.reverse:
                start_idx = (self.cum_times - (self.core_cycle + 1) * (block_id + 1)) * self.per_cum_num
            with self.tik_instance.if_scope(self.per_cum_num < self.data_one_block):
                self.__handle_small_piece_speed(start_idx, self.core_cycle + 1)
            with self.tik_instance.else_scope():
                self.__handle_one_core_speed(start_idx, self.core_cycle + 1)
        with self.tik_instance.else_scope():
            start_idx = (self.core_cycle + 1) * self.core_tail * self.per_cum_num + (
                block_id - self.core_tail) * self.core_cycle * self.per_cum_num
            if self.reverse:
                start_idx = ((self.cum_times - ((block_id - self.core_tail) + 1) * self.core_cycle) -
                             (self.core_cycle + 1) * self.core_tail) * self.per_cum_num
            with self.tik_instance.if_scope(self.per_cum_num < self.data_one_block):
                self.__handle_small_piece_speed(start_idx, self.core_cycle)
            with self.tik_instance.else_scope():
                self.__handle_one_core_speed(start_idx, self.core_cycle)
        self.tik_instance.set_atomic_add(0)

    def __handle_one_core_speed(self, start_idx, cal_times):
        out_ub = self.tik_instance.Tensor(self.new_x_dtype, (self.one_max_nums * self.data_one_block,),
                                          name="out_ub",
                                          scope=tik.scope_ubuf)
        input_ub = self.tik_instance.Tensor(self.new_x_dtype, (self.one_max_nums * self.data_one_block,),
                                            name="input_ub",
                                            scope=tik.scope_ubuf)
        max_ub_cycle = self.per_cum_num // (self.one_max_nums * self.data_one_block)
        remaining_ub_nums = self.per_cum_num % (self.one_max_nums * self.data_one_block)
        last_ub_nums = remaining_ub_nums % self.data_one_block
        if self.reverse:
            start_idx = start_idx + (cal_times - 1) * self.per_cum_num
        with self.tik_instance.for_range(0, max_ub_cycle) as burst_cycle:
            max_ub_start_idx = start_idx + burst_cycle * (self.one_max_nums * self.data_one_block)
            if self.exclusive and not self.reverse:
                with self.tik_instance.if_scope(start_idx == 0):
                    self._dup_0_ub(out_ub, (self.one_max_nums * self.data_one_block))
                with self.tik_instance.else_scope():
                    self._gm_2_ub(out_ub, max_ub_start_idx - self.per_cum_num, self.one_max_nums)
            elif self.exclusive and self.reverse:
                with self.tik_instance.if_scope(start_idx == (self.cum_times - 1) * self.per_cum_num):
                    self._dup_0_ub(out_ub, (self.one_max_nums * self.data_one_block))
                with self.tik_instance.else_scope():
                    self._gm_2_ub(out_ub, max_ub_start_idx + self.per_cum_num, self.one_max_nums)
            else:
                self._gm_2_ub(out_ub, max_ub_start_idx, self.one_max_nums)
            self._ub_2_gm(out_ub, max_ub_start_idx, self.one_max_nums)
            with self.tik_instance.for_range(1, cal_times) as cum_cycle:
                if self.exclusive:
                    cum_cycle = cum_cycle - 1
                axis_start_idx = max_ub_start_idx + self.per_cum_num * cum_cycle
                if self.reverse:
                    axis_start_idx = max_ub_start_idx - self.per_cum_num * cum_cycle
                self._gm_2_ub(input_ub, axis_start_idx, self.one_max_nums)
                self._batch_add(self.one_max_nums * self.data_one_block, out_ub, input_ub, out_ub)
                if self.exclusive and self.reverse:
                    self._ub_2_gm(out_ub, axis_start_idx - self.per_cum_num, self.one_max_nums)
                elif self.exclusive and not self.reverse:
                    self._ub_2_gm(out_ub, axis_start_idx + self.per_cum_num, self.one_max_nums)
                else:
                    self._ub_2_gm(out_ub, axis_start_idx, self.one_max_nums)
            if self.reverse:
                with self.tik_instance.for_range(self.cum_times - (start_idx // self.per_cum_num - (cal_times - 1)),
                                                 self.cum_times) as t_idx:
                    atomic_idx = (self.cum_times - 1 - t_idx) * self.per_cum_num + burst_cycle * (self.one_max_nums *
                                                                                                  self.data_one_block)
                    self._ub_2_gm(out_ub, atomic_idx, self.one_max_nums)
            else:
                with self.tik_instance.for_range((start_idx // self.per_cum_num) + cal_times, self.cum_times) as t_idx:
                    atomic_idx = t_idx * self.per_cum_num + burst_cycle * (self.one_max_nums * self.data_one_block)
                    self._ub_2_gm(out_ub, atomic_idx, self.one_max_nums)
        with self.tik_instance.if_scope(remaining_ub_nums >= self.data_one_block):
            remaining_ub_start_idx = start_idx + max_ub_cycle * (self.one_max_nums * self.data_one_block)
            burst_len = remaining_ub_nums // self.data_one_block
            if self.exclusive and not self.reverse:
                with self.tik_instance.if_scope(start_idx == 0):
                    self._dup_0_ub(out_ub, burst_len * self.data_one_block)
                with self.tik_instance.else_scope():
                    self._gm_2_ub(out_ub, remaining_ub_start_idx - self.per_cum_num, burst_len)
            elif self.exclusive and self.reverse:
                with self.tik_instance.if_scope(start_idx == (self.cum_times - 1) * self.per_cum_num):
                    self._dup_0_ub(out_ub, burst_len * self.data_one_block)
                with self.tik_instance.else_scope():
                    self._gm_2_ub(out_ub, remaining_ub_start_idx + self.per_cum_num, burst_len)
            else:
                self._gm_2_ub(out_ub, remaining_ub_start_idx, burst_len)
            self._ub_2_gm(out_ub, remaining_ub_start_idx, burst_len)
            with self.tik_instance.for_range(1, cal_times) as cum_cycle:
                if self.exclusive:
                    cum_cycle = cum_cycle - 1
                remaining_axis_start_idx = remaining_ub_start_idx + self.per_cum_num * cum_cycle
                if self.reverse:
                    remaining_axis_start_idx = remaining_ub_start_idx - self.per_cum_num * cum_cycle
                self._gm_2_ub(input_ub, remaining_axis_start_idx, burst_len)
                self._batch_add(burst_len * self.data_one_block, out_ub, input_ub, out_ub)
                if self.exclusive and self.reverse:
                    self._ub_2_gm(out_ub, remaining_axis_start_idx - self.per_cum_num, burst_len)
                elif self.exclusive and not self.reverse:
                    self._ub_2_gm(out_ub, remaining_axis_start_idx + self.per_cum_num, burst_len)
                else:
                    self._ub_2_gm(out_ub, remaining_axis_start_idx, burst_len)
            if self.reverse:
                with self.tik_instance.for_range(self.cum_times - 1 - (start_idx // self.per_cum_num) + cal_times,
                                                 self.cum_times) as t_idx:
                    atomic_idx = (self.cum_times - 1 - t_idx) * self.per_cum_num + max_ub_cycle * (self.one_max_nums *
                                                                                                   self.data_one_block)
                    self._ub_2_gm(out_ub, atomic_idx, burst_len)
            else:
                with self.tik_instance.for_range((start_idx // self.per_cum_num) + cal_times, self.cum_times) as t_idx:
                    atomic_idx = t_idx * self.per_cum_num + max_ub_cycle * (self.one_max_nums * self.data_one_block)
                    self._ub_2_gm(out_ub, atomic_idx, burst_len)
        with self.tik_instance.if_scope(last_ub_nums > 0):
            last_ub_start_idx = start_idx + self.per_cum_num - self.data_one_block
            if self.exclusive and not self.reverse:
                with self.tik_instance.if_scope(start_idx == 0):
                    self._dup_0_ub(out_ub, self.data_one_block)
                with self.tik_instance.else_scope():
                    self._gm_2_ub(out_ub, last_ub_start_idx - self.per_cum_num, 1)
            elif self.exclusive and self.reverse:
                with self.tik_instance.if_scope(start_idx == (self.cum_times - 1) * self.per_cum_num):
                    self._dup_0_ub(out_ub, self.data_one_block)
                with self.tik_instance.else_scope():
                    self._gm_2_ub(out_ub, last_ub_start_idx + self.per_cum_num, 1)
            else:
                self._gm_2_ub(out_ub, last_ub_start_idx, 1)
            with self.tik_instance.for_range(0, self.data_one_block - last_ub_nums) as last_ub_idx:
                out_ub[last_ub_idx].set_as(0)
            self._ub_2_gm(out_ub, last_ub_start_idx, 1)
            with self.tik_instance.for_range(1, cal_times) as cum_cycle:
                if self.exclusive:
                    cum_cycle = cum_cycle - 1
                last_axis_start_idx = last_ub_start_idx + self.per_cum_num * cum_cycle
                if self.reverse:
                    last_axis_start_idx = last_ub_start_idx - self.per_cum_num * cum_cycle
                self._gm_2_ub(input_ub, last_axis_start_idx, 1)
                with self.tik_instance.for_range(0, self.data_one_block - last_ub_nums) as last_ub_idx:
                    input_ub[last_ub_idx].set_as(0)
                self.tik_instance.vadd(self.data_one_block, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                if self.exclusive and self.reverse:
                    self._ub_2_gm(out_ub, last_axis_start_idx - self.per_cum_num, 1)
                elif self.exclusive and not self.reverse:
                    self._ub_2_gm(out_ub, last_axis_start_idx + self.per_cum_num, 1)
                else:
                    self._ub_2_gm(out_ub, last_axis_start_idx, 1)
            if self.reverse:
                with self.tik_instance.for_range(self.cum_times - 1 - (start_idx // self.per_cum_num) + cal_times,
                                                 self.cum_times) as t_idx:
                    atomic_idx = (self.cum_times - 1 - t_idx) * self.per_cum_num + \
                                 self.per_cum_num - self.data_one_block
                    self._ub_2_gm(out_ub, atomic_idx, 1)
            else:
                with self.tik_instance.for_range((start_idx // self.per_cum_num) + cal_times, self.cum_times) as t_idx:
                    atomic_idx = t_idx * self.per_cum_num + self.per_cum_num - self.data_one_block
                    self._ub_2_gm(out_ub, atomic_idx, 1)

    def __handle_small_piece_speed(self, start_idx, cal_times):
        out_ub = self.tik_instance.Tensor(self.new_x_dtype, (self.data_one_block,), name="out_ub", scope=tik.scope_ubuf)
        input_ub = self.tik_instance.Tensor(self.new_x_dtype, (self.data_one_block,),
                                            name="input_ub",
                                            scope=tik.scope_ubuf)
        last_out_ub = self.tik_instance.Tensor(self.new_x_dtype, (self.data_one_block,),
                                               name="last_out_ub",
                                               scope=tik.scope_ubuf)
        if self.reverse:
            start_idx = start_idx + (cal_times - 1) * self.per_cum_num
        # cumulative
        if not self.reverse and not self.exclusive:
            self._gm_2_ub(out_ub, start_idx, 1)
            with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as out_ub_idx:
                out_ub[self.per_cum_num + out_ub_idx].set_as(0)
            self._ub_2_gm(out_ub, start_idx, 1)
            # middle
            with self.tik_instance.for_range(1, cal_times - (self.remain_cycle - 1)) as cum_cycle:
                axis_start_idx = start_idx + self.per_cum_num * cum_cycle
                self._gm_2_ub(input_ub, axis_start_idx, 1)
                with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as input_ub_idx:
                    input_ub[self.per_cum_num + input_ub_idx].set_as(0)
                self.tik_instance.vadd(self.data_one_block, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                self._ub_2_gm(out_ub, axis_start_idx, 1)
            # last 8
            with self.tik_instance.for_range(0, self.data_one_block) as last_ub_idx:
                last_out_ub[last_ub_idx].set_as(0)
            with self.tik_instance.for_range(0, self.remain_cycle - 1) as last_cum_cycle:
                last_axis_stat_idx = start_idx + self.per_cum_num * last_cum_cycle + self.per_cum_num * (
                    cal_times - (self.remain_cycle - 1))
                self._gm_2_ub_by_ele(input_ub, last_axis_stat_idx, self.per_cum_num)
                with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as input_ub_idx:
                    input_ub[self.per_cum_num + input_ub_idx].set_as(0)
                self.tik_instance.vadd(self.per_cum_num, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, self.per_cum_num) as last_cum_ub_idx:
                    last_out_ub[self.data_one_block - (self.remain_cycle - 1) * self.per_cum_num + \
                                self.per_cum_num * last_cum_cycle + last_cum_ub_idx].set_as(
                        out_ub[last_cum_ub_idx])
            self._ub_2_gm(last_out_ub,
                          self.per_cum_num * (cal_times - 1) - self.data_one_block + self.per_cum_num + start_idx, 1)
            with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as zero_idx:
                last_out_ub[zero_idx].set_as(0)
            with self.tik_instance.for_range((start_idx // self.per_cum_num) + cal_times, self.cum_times) as t_idx:
                atomic_idx = t_idx * self.per_cum_num + self.per_cum_num - self.data_one_block
                self._ub_2_gm(last_out_ub, atomic_idx, 1)
        elif self.reverse and not self.exclusive:
            # last 8
            with self.tik_instance.for_range(0, self.data_one_block) as last_ub_idx:
                last_out_ub[last_ub_idx].set_as(0)
            self._gm_2_ub_by_ele(out_ub, start_idx, self.per_cum_num)
            with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as tem_idx:
                out_ub[self.per_cum_num + tem_idx].set_as(0)
            with self.tik_instance.for_range(0, self.per_cum_num) as last_ub_idx:
                last_out_ub[self.data_one_block - self.per_cum_num + last_ub_idx].set_as(out_ub[last_ub_idx])
            with self.tik_instance.for_range(1, self.remain_cycle - 1) as last_cum_cycle:
                temp_start_idx = start_idx - self.per_cum_num * last_cum_cycle
                self._gm_2_ub_by_ele(input_ub, temp_start_idx, self.per_cum_num)
                with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as tem_idx:
                    input_ub[self.per_cum_num + tem_idx].set_as(0)
                self.tik_instance.vadd(self.data_one_block, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, self.per_cum_num) as last_ub_idx:
                    last_out_ub[self.data_one_block - self.per_cum_num * (last_cum_cycle + 1) + last_ub_idx].set_as(
                        out_ub[last_ub_idx])
            self._ub_2_gm(last_out_ub, start_idx - self.data_one_block + self.per_cum_num, 1)
            # middle
            with self.tik_instance.for_range(0, cal_times - (self.remain_cycle - 1)) as most_idx:
                axis_start_idx = start_idx - (self.remain_cycle - 1) * self.per_cum_num - self.per_cum_num * most_idx
                self._gm_2_ub(input_ub, axis_start_idx, 1)
                with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as tem_idx:
                    input_ub[self.per_cum_num + tem_idx].set_as(0)
                self.tik_instance.vadd(self.data_one_block, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                self._ub_2_gm(out_ub, axis_start_idx, 1)
            with self.tik_instance.for_range(self.cum_times - 1 - (start_idx // self.per_cum_num) + cal_times,
                                             self.cum_times) as t_idx:
                atomic_idx = (self.cum_times - 1 - t_idx) * self.per_cum_num
                self._ub_2_gm(out_ub, atomic_idx, 1)
        elif not self.reverse and self.exclusive:
            with self.tik_instance.if_scope(start_idx == 0):
                self._dup_0_ub(out_ub, self.data_one_block)
            with self.tik_instance.else_scope():
                self._gm_2_ub(out_ub, start_idx - self.per_cum_num, 1)
                with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as tem_idx:
                    out_ub[self.per_cum_num + tem_idx].set_as(0)
            self._ub_2_gm(out_ub, start_idx, 1)
            # middle
            with self.tik_instance.for_range(1, cal_times - (self.remain_cycle - 1)) as cum_cycle:
                axis_start_idx = start_idx + self.per_cum_num * (cum_cycle - 1)
                self._gm_2_ub(input_ub, axis_start_idx, 1)
                with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as tem_idx:
                    input_ub[self.per_cum_num + tem_idx].set_as(0)
                self.tik_instance.vadd(self.data_one_block, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                self._ub_2_gm(out_ub, axis_start_idx + self.per_cum_num, 1)
            # last 8
            with self.tik_instance.for_range(0, self.data_one_block) as last_ub_idx:
                last_out_ub[last_ub_idx].set_as(0)
            with self.tik_instance.for_range(0, self.remain_cycle - 1) as last_cum_cycle:
                last_axis_stat_idx = start_idx + self.per_cum_num * last_cum_cycle + self.per_cum_num * (
                    cal_times - (self.remain_cycle - 1) - 1)
                self._gm_2_ub_by_ele(input_ub, last_axis_stat_idx, self.per_cum_num)
                with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as tem_idx:
                    input_ub[self.per_cum_num + tem_idx].set_as(0)
                self.tik_instance.vadd(self.per_cum_num, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, self.per_cum_num) as last_cum_ub_idx:
                    last_out_ub[self.data_one_block - (
                            self.remain_cycle - 1) * self.per_cum_num + \
                                self.per_cum_num * last_cum_cycle + last_cum_ub_idx].set_as(out_ub[last_cum_ub_idx])
            self._ub_2_gm(last_out_ub,
                          self.per_cum_num * (cal_times - 1) - self.data_one_block + self.per_cum_num + start_idx, 1)
            with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as zero_idx:
                last_out_ub[zero_idx].set_as(0)
            with self.tik_instance.for_range((start_idx // self.per_cum_num) + cal_times, self.cum_times) as t_idx:
                atomic_idx = t_idx * self.per_cum_num + self.per_cum_num - self.data_one_block
                self._ub_2_gm(last_out_ub, atomic_idx, 1)
        else:  # self.reverse and self.exclusive
            # last 8
            with self.tik_instance.for_range(0, self.data_one_block) as last_ub_idx:
                last_out_ub[last_ub_idx].set_as(0)
            with self.tik_instance.if_scope(start_idx == (self.cum_times - 1) * self.per_cum_num):
                self._dup_0_ub(out_ub, self.data_one_block)
            with self.tik_instance.else_scope():
                self._gm_2_ub_by_ele(out_ub, start_idx + self.per_cum_num, self.per_cum_num)
                with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as zero_idx:
                    out_ub[self.per_cum_num + zero_idx].set_as(0)
            with self.tik_instance.for_range(0, self.per_cum_num) as last_ub_idx:
                last_out_ub[self.data_one_block - self.per_cum_num + last_ub_idx].set_as(out_ub[last_ub_idx])
            with self.tik_instance.for_range(1, self.remain_cycle - 1) as last_cum_cycle:
                temp_start_idx = start_idx - self.per_cum_num * (last_cum_cycle - 1)
                self._gm_2_ub_by_ele(input_ub, temp_start_idx, self.per_cum_num)
                with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as tem_idx:
                    input_ub[self.per_cum_num + tem_idx].set_as(0)
                self.tik_instance.vadd(self.per_cum_num, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance.for_range(0, self.per_cum_num) as last_ub_idx:
                    last_out_ub[self.data_one_block - self.per_cum_num * (last_cum_cycle + 1) + last_ub_idx].set_as(
                        out_ub[last_ub_idx])
            self._ub_2_gm(last_out_ub, start_idx - self.data_one_block + self.per_cum_num, 1)
            # middle
            with self.tik_instance.for_range(self.remain_cycle - 1, cal_times) as remain_idx:
                axis_start_idx = start_idx - self.per_cum_num * (remain_idx - 1)
                self._gm_2_ub_by_ele(input_ub, axis_start_idx, self.per_cum_num)
                with self.tik_instance.for_range(0, self.data_one_block - self.per_cum_num) as tem_idx:
                    input_ub[self.per_cum_num + tem_idx].set_as(0)
                self.tik_instance.vadd(self.data_one_block, out_ub, input_ub, out_ub, 1, 1, 1, 1, 8, 8, 8)
                self._ub_2_gm(out_ub, axis_start_idx - self.per_cum_num, 1)
            with self.tik_instance.for_range(self.cum_times - 1 - (start_idx // self.per_cum_num) + cal_times,
                                             self.cum_times) as t_idx:
                atomic_idx = (self.cum_times - 1 - t_idx) * self.per_cum_num
                self._ub_2_gm(out_ub, atomic_idx, 1)
