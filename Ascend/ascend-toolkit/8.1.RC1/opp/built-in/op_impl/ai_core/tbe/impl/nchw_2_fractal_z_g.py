#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
nchw_2_fractal_z_g
"""
import math
from te import tik
import te.platform as tbe_platform
from te.utils import para_check


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    This class for Constant.
    """
    EPB = 16
    CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    OFFSET_1 = 256 * 256
    OFFSET_2 = 0
    SRC_C_UNIT = 16


# 'pylint: disable=superfluous-parens,useless-object-inheritance,invalid-name
# 'pylint: disable=too-many-instance-attributes,too-few-public-methods,unused-argument,no-self-use
# 'pylint: disable=too-many-arguments,too-many-locals,useless-return,protected-access
class Nchw2Fractalzg(object):
    """
    Nchw2Fractalzg
    """
    class TilingParam(object):
        """
        TilingParam
        """
        class PerCoreParam(object):
            """
            PerCoreParam
            """
            def __init__(self):
                #le 16
                self.loop_src_n_base = []
                self.loop_src_n_repeat = []
                self.loop_src_c_base = []
                self.loop_src_c_repeat = []
                self.loop_src_c_tail = []

                #gt 16
                self.loop_gc_base = []
                self.loop_gc_repeat = []
                self.loop_gc_tail = []
                self.loop_cout_orig_base = []
                self.loop_cout_orig_repeat = []
                self.loop_cout_orig_tail = []

            def composite_c_le_16(self, pf_src_n_base, pf_src_n_repeat, pf_src_c_base, pf_src_c_repeat):
                """
                composite_c_le_16
                """
                len_n = len(pf_src_n_base)
                len_c = len(pf_src_c_base)
                for n in range(len_n):
                    for c in range(len_c):
                        self.loop_src_n_base.append(pf_src_n_base[n])
                        self.loop_src_n_repeat.append(pf_src_n_repeat[n])
                        self.loop_src_c_base.append(pf_src_c_base[c])
                        self.loop_src_c_repeat.append(pf_src_c_repeat[c])
                self._pad_zero()

            def composite_c_gt_16(self, pf_gc_base, pf_gc_repeat, gc_tail,
                                  pf_cout_orig_base, pf_cout_orig_repeat, cout_orig_tail):
                """
                composite_c_gt_16
                """
                len_gc = len(pf_gc_base)
                len_cout = len(pf_cout_orig_base)
                for i in range(len_gc):
                    for j in range(len_cout):
                        self.loop_gc_base.append(pf_gc_base[i])
                        self.loop_gc_repeat.append(pf_gc_repeat[i])
                        self.loop_cout_orig_base.append(pf_cout_orig_base[j])
                        self.loop_cout_orig_repeat.append(pf_cout_orig_repeat[j])
                        if i == len_gc - 1:
                            self.loop_gc_tail.append(gc_tail)
                        else:
                            self.loop_gc_tail.append(0)
                        if j == len_cout - 1:
                            self.loop_cout_orig_tail.append(cout_orig_tail)
                        else:
                            self.loop_cout_orig_tail.append(0)
                self._pad_zero()

            def _pad_zero_impl(self, v, n):
                while(len(v) < n):
                    v.append(0)

            def _pad_zero(self):
                self._pad_zero_impl(self.loop_src_n_base, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_src_n_repeat, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_src_c_base, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_src_c_repeat, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_src_c_tail, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_gc_base, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_gc_repeat, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_gc_tail, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_cout_orig_base, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_cout_orig_repeat, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_cout_orig_tail, Constant.CORE_NUM)

        def __init__(self, shape_in, shape_out, groups):
            self.src_n = shape_in[0]
            self.cout_orig = shape_in[0] // groups
            self.cin_orig = shape_in[1]
            self.src_c = shape_in[1]
            self.cin_orig_ms = Constant.EPB if self.cin_orig > Constant.EPB else  self.cin_orig # mjaor section
            self.kh = shape_in[2]
            self.kw = shape_in[3]
            self.dst_n = shape_out[1] * shape_out[2]
            self.groups = groups
            self.loop_on_chw = 1
            self.loop_on_groups = self.groups
            self.n = shape_out[1] * shape_out[2]
            self.n16 = self.n // Constant.EPB
            self.loop_on_cout_orig = self.cout_orig // Constant.EPB
            self.cout_orig_tail = self.cout_orig - self.loop_on_cout_orig * Constant.EPB
            self.src_c_unit = Constant.SRC_C_UNIT
            self.loop_on_src_c = 0
            self.c0 = shape_out[3]
            self.khw = self.kh * self.kw
            self.vol_chw = self.src_c * self.khw

            #c_le_16
            self.pf_src_n_base = []
            self.pf_src_n_repeat = []
            self.pf_src_c_base = []
            self.pf_src_c_repeat = []

            #c_gt_16
            self.gc_tail = self.groups * self.src_c % Constant.SRC_C_UNIT
            self.pf_cout_orig_base = []
            self.pf_cout_orig_repeat = []
            self.pf_cout_orig_tail = []
            self.pf_gc_base = []
            self.pf_gc_repeat = []
            self.pf_gc_tail = 0

            self._calc_src_c_unit()


            self.pcp = self.PerCoreParam()
            if (self.cin_orig <= 16):
                self._dispatch_loop_c_le_16()
                self.pcp.composite_c_le_16(self.pf_src_n_base, self.pf_src_n_repeat,
                                           self.pf_src_c_base, self.pf_src_c_repeat)
            else:
                self._dispatch_loop_c_gt_16()
                self.pcp.composite_c_gt_16(self.pf_gc_base, self.pf_gc_repeat, self.gc_tail,
                                           self.pf_cout_orig_base, self.pf_cout_orig_repeat, self.cout_orig_tail)

        def _calc_src_c_unit(self):
            if self.cin_orig <= Constant.EPB:
                self.src_c_unit = self.src_c
            else:
                self.src_c_unit = Constant.SRC_C_UNIT
                self.loop_on_src_c = self.src_c // self.src_c_unit
                self.src_c_tail = self.src_c - self.loop_on_src_c * self.src_c_unit

        def _sum_all(self, loop):
            res = 0
            for i in range(len(loop)):
                res = res + loop[i]
            return res

        def _remove_zero_data(self, tup):
            tup = [x for x in tup if x != 0]

        def _each_factor_process_num(self, total, factor, unit, base, repeat):
            if total < unit:
                base.append(0)
                repeat.append(0)
            elif total == unit:
                base.append(0)
                repeat.append(1)
            else:
                each_factor = []
                share = total // unit
                item = math.ceil(share / factor)
                for i in range(factor):
                    each_factor.append(item)
                for i in reversed(range(len(each_factor))):
                    if self._sum_all(each_factor) > share:
                        each_factor[i] = each_factor[i] - 1
                self._remove_zero_data(each_factor)
                r = 0
                for i in range(len(each_factor)):
                    base.append(r)
                    r = r + each_factor[i]
                    repeat.append(each_factor[i])

        def _dispatch_loop_c_le_16(self):
            # step 1: src_n factor
            factor_n = self.groups if self.groups < Constant.CORE_NUM else Constant.CORE_NUM
            unit = self.cout_orig
            self._each_factor_process_num(self.src_n, factor_n, unit, self.pf_src_n_base, self.pf_src_n_repeat)

            # step 2: src_c factor
            factor_c = Constant.CORE_NUM // factor_n
            if factor_c == 0:
                factor_c = 1
            unit = self.src_c_unit
            self._each_factor_process_num(self.src_c, factor_c, unit, self.pf_src_c_base, self.pf_src_c_repeat)

        def _dispatch_loop_c_gt_16(self):
            # step 1: src_n factor
            total = self.groups * self.src_c
            unit = Constant.SRC_C_UNIT
            factor_gc = (total // unit) if (total // unit) < Constant.CORE_NUM else Constant.CORE_NUM
            self._each_factor_process_num(total, factor_gc, unit, self.pf_gc_base, self.pf_gc_repeat)

            # step 2: src_c factor
            total = self.loop_on_cout_orig
            unit = 1
            factor_cout = Constant.CORE_NUM // factor_gc
            if factor_cout == 0:
                factor_cout = 1
            self._each_factor_process_num(total, factor_cout, unit, self.pf_cout_orig_base, self.pf_cout_orig_repeat)

    def __init__(self, tik_inst, data_in, data_out):
        UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.tik_inst = tik_inst
        self.ub_input = tik_inst.Tensor("float16", (UB_SIZE // 2, ), tik.scope_ubuf, "ub_input")
        self.data_in = data_in
        self.data_out = data_out

    def _tiling(self, shape_in, shape_out, groups):
        tp = self.TilingParam(shape_in, shape_out, groups)
        return tp

    def _get_param_by_block_idx_le_16(self, block_idx, tp, pc_src_n_base, pc_src_n_repeat):
        for i in range(Constant.CORE_NUM):
            with self.tik_inst.if_scope(block_idx == i):
                pc_src_n_base.set_as(tp.pcp.loop_src_n_base[i])
                pc_src_n_repeat.set_as(tp.pcp.loop_src_n_repeat[i])

    def _clear_tail_memory(self, tp, ub_input, left_zero, left_part, right_part, is_left):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(is_left == 1):
            with self.tik_inst.if_scope(left_part * tp.khw % Constant.EPB != 0):
                self.tik_inst.vector_dup(16, ub_input[Constant.OFFSET_2 + left_zero * tp.khw *
                                                      Constant.EPB + left_part * tp.khw * Constant.EPB],
                                         0, Constant.EPB - (left_part * tp.khw % Constant.EPB), 1, 1)
        with tik_inst.else_scope():
            with self.tik_inst.if_scope(right_part * tp.khw % Constant.EPB != 0):
                self.tik_inst.vector_dup(16, ub_input[Constant.OFFSET_2 + left_zero *
                                                      tp.khw * Constant.EPB + right_part * tp.khw * Constant.EPB],
                                         0, Constant.EPB - (right_part * tp.khw % Constant.EPB), 1, 1)

    def _zero_offset_1(self, ub_input, repeat=248):
        # mask,dst,scalar, repeat_times, dst_blk_stride, dst_rep_stride
        self.tik_inst.vector_dup(128, ub_input[Constant.OFFSET_1], 0, repeat, 1, 8)

    def _zero_offset_2(self, ub_input, repeat=248):
        self.tik_inst.vector_dup(128, ub_input[Constant.OFFSET_2], 0, repeat, 1, 8)

    def _calc_src_addr_le_16(self, tp, lsn, lct, left_part, is_left, src_addr):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(is_left == 1):
            src_addr.set_as(lsn * tp.cout_orig * tp.vol_chw + lct * Constant.EPB * tp.vol_chw)
        with tik_inst.else_scope():
            src_addr.set_as(lsn * tp.cout_orig * tp.vol_chw + lct * Constant.EPB * tp.vol_chw + left_part * tp.khw)

    def _calc_dst_addr_le_16(self, tp, lct, top_distance, nc0_counter, dst_addr):
        dst_addr.set_as(nc0_counter * tp.khw * tp.dst_n * Constant.EPB + lct *
                        Constant.EPB * Constant.EPB + top_distance * Constant.EPB)

    def _calc_burst_len(self, tp, left_part, right_part, is_left, burst_len):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(is_left == 1):
            burst_len.set_as((left_part * tp.khw + Constant.EPB - 1) / Constant.EPB)
        with tik_inst.else_scope():
            burst_len.set_as((right_part * tp.khw + Constant.EPB - 1) / Constant.EPB)

    def _copy_in_le_16(self, tp, ub_input, ub_offset, lsn, lct, cout_orig_tail,
                       left_zero, left_part, right_part, is_left):
        tik_inst = self.tik_inst
        src_addr = tik_inst.Scalar("int64", init_value=0)
        burst_len = tik_inst.Scalar("int64", init_value=0)
        self._calc_src_addr_le_16(tp, lsn, lct, left_part, is_left, src_addr)
        self._calc_burst_len(tp, left_part, right_part, is_left, burst_len)
        loop_num = tik_inst.Scalar("int64", init_value=cout_orig_tail)
        with tik_inst.if_scope(cout_orig_tail == 0):
            loop_num.set_as(Constant.EPB)
        with tik_inst.for_range(0, loop_num) as i:
            tik_inst.data_move(self.ub_input[Constant.OFFSET_1 + ub_offset * Constant.EPB],
                               self.data_in[src_addr + i * tp.vol_chw], 0, 1, burst_len, 0, 0)
            ub_offset.set_as(ub_offset + burst_len)

    def _reorder_le_16(self, tp, ub_input, left_zero, left_part, right_part, is_left):
        tik_inst = self.tik_inst
        # step 1 : first vnchwconv, make line be vertical
        src_stride = tik_inst.Scalar("int64", init_value=0)
        dst_stride = tik_inst.Scalar("int64", init_value=0)
        with tik_inst.if_scope(is_left == 1):
            line_blocks = (left_part * tp.khw + Constant.EPB - 1) / Constant.EPB
            ele_num_per_line = line_blocks * Constant.EPB
            src_addr_list = [ub_input[Constant.OFFSET_1 + ele_num_per_line * i] for i in range(Constant.EPB)]
            dst_addr_list = [ub_input[Constant.OFFSET_2 + left_zero * tp.khw *
                                      Constant.EPB + Constant.EPB * i] for i in range(Constant.EPB)]
            repeat_cnt_first = line_blocks
            with tik_inst.if_scope(repeat_cnt_first != 1):
                src_stride.set_as(1)
                dst_stride.set_as(16)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt_first, dst_stride, src_stride)
        with tik_inst.else_scope():
            line_blocks = (right_part * tp.khw + Constant.EPB - 1) / Constant.EPB
            ele_num_per_line = line_blocks * Constant.EPB
            src_addr_list = [ub_input[Constant.OFFSET_1 + ele_num_per_line * i] for i in range(Constant.EPB)]
            # step 1 : first vnchwconv
            src_addr_list = [ub_input[Constant.OFFSET_1 + ele_num_per_line * i] for i in range(Constant.EPB)]
            dst_addr_list = [ub_input[Constant.OFFSET_2 + left_zero * tp.khw *
                                      Constant.EPB + Constant.EPB * i] for i in range(Constant.EPB)]
            repeat_cnt_first = line_blocks
            with tik_inst.if_scope(repeat_cnt_first != 1):
                src_stride.set_as(1)
                dst_stride.set_as(16)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt_first, dst_stride, src_stride)

        # step 2 : clear dirty data, because unused data should be zero, and step3 will access these data
        self._clear_tail_memory(tp, ub_input, left_zero, left_part, right_part, is_left)

        # step 3 : second vnchwconv, move block data together
        src_addr_list = [ub_input[Constant.OFFSET_2 + tp.khw * Constant.EPB * i] for i in range(Constant.EPB)]
        dst_addr_list = [ub_input[Constant.OFFSET_1 + Constant.EPB * i] for i in range(Constant.EPB)]
        repeat_cnt_second = tp.khw
        with tik_inst.if_scope(repeat_cnt_second == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(1)
            dst_stride.set_as(16)
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt_second, dst_stride, src_stride)

    def _copy_out_le_16(self, tp, ub_input, lsn, lct, cout_orig_tail, top_distance, nc0_counter):
        tik_inst = self.tik_inst
        dst_addr = tik_inst.Scalar("int64", init_value=0)
        cout_orig_tail_s = tik_inst.Scalar("int64", init_value=cout_orig_tail)
        self._calc_dst_addr_le_16(tp, lct, top_distance, nc0_counter, dst_addr)
        with tik_inst.if_scope(cout_orig_tail == 0):
            tik_inst.data_move(self.data_out[dst_addr], self.ub_input[Constant.OFFSET_1], 0,
                               tp.khw, Constant.EPB, 0, tp.dst_n - Constant.EPB)
        with tik_inst.else_scope():
            tik_inst.data_move(self.data_out[dst_addr], self.ub_input[Constant.OFFSET_1],
                               0, tp.khw, cout_orig_tail_s, Constant.EPB - cout_orig_tail_s,
                               tp.dst_n - cout_orig_tail_s)

    def _update_param_lr_part_le_16(self, tp, left_zero, left_part, right_part):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(left_zero + tp.cin_orig > Constant.EPB):
            left_part.set_as(Constant.EPB - left_zero)
            right_part.set_as(tp.cin_orig - left_part)
        with tik_inst.else_scope():
            left_part.set_as(tp.cin_orig)
            right_part.set_as(0)

    def _update_param_right_le_16(self, tp, is_left, left_zero, nc0_counter):
        is_left.set_as(0)
        left_zero.set_as(0)
        nc0_counter.set_as(nc0_counter + 1)

    def _update_param_le_16(self, tp, left_zero, top_distance, nc0_counter, left_part, right_part, is_left):
        tik_inst = self.tik_inst
        top_distance.set_as((top_distance + tp.cout_orig) % tp.dst_n)
        with tik_inst.if_scope(is_left == 0):
            left_zero.set_as(right_part)
        with tik_inst.else_scope():
            with tik_inst.if_scope(left_zero + tp.cin_orig < Constant.EPB):
                left_zero.set_as(left_zero + tp.cin_orig)
            with tik_inst.else_scope():
                nc0_counter.set_as(nc0_counter + 1)
                left_zero.set_as(0)

    def _copy_left_part_le_16(self, tp, ub_input, ub_offset, lsn, lct, cout_orig_tail, left_zero, top_distance,
                              nc0_counter, left_part, right_part, is_left):
        ub_offset.set_as(0)
        is_left.set_as(1)
        self._zero_offset_1(ub_input)
        self._zero_offset_2(ub_input)
        self._copy_in_le_16(tp, ub_input, ub_offset, lsn, lct, cout_orig_tail,
                            left_zero, left_part, right_part, is_left)
        self._reorder_le_16(tp, ub_input, left_zero, left_part, right_part, is_left)
        self._copy_out_le_16(tp, ub_input, lsn, lct, cout_orig_tail, top_distance, nc0_counter)

    def _copy_right_part_le_16(self, tp, ub_input, ub_offset, lsn, lct, cout_orig_tail, left_zero, top_distance,
                               nc0_counter, left_part, right_part, is_left):
        ub_offset.set_as(0)
        self._zero_offset_1(ub_input)
        self._zero_offset_2(ub_input)
        self._copy_in_le_16(tp, ub_input, ub_offset, lsn, lct, cout_orig_tail,
                            left_zero, left_part, right_part, is_left)
        self._reorder_le_16(tp, ub_input, left_zero, left_part, right_part, is_left)
        self._copy_out_le_16(tp, ub_input, lsn, lct, cout_orig_tail, top_distance, nc0_counter)

    def _calc_base_pos_le_16(self, tp, pc_src_n_base, left_zero, top_distance, nc0_counter):
        left_zero.set_as(pc_src_n_base * tp.cin_orig % Constant.EPB)
        top_distance.set_as(pc_src_n_base * tp.cout_orig % tp.dst_n)
        nc0_counter.set_as(pc_src_n_base * tp.cin_orig / Constant.EPB)

    def compute_c_le_16(self, tp, data_in, data_out):
        """
        nchw_2_fractal_z_g entrance function
        """
        tik_inst = self.tik_inst
        ub_offset = tik_inst.Scalar("int64", init_value=0)
        pc_src_n_base = tik_inst.Scalar("int64", init_value=0)
        pc_src_n_repeat = tik_inst.Scalar("int64", init_value=0)
        left_zero = tik_inst.Scalar("int64", init_value=0)
        top_distance = tik_inst.Scalar("int64", init_value=0)
        nc0_counter = tik_inst.Scalar("int64", init_value=0)
        left_part = tik_inst.Scalar("int64", init_value=tp.cin_orig)
        right_part = tik_inst.Scalar("int64", init_value=0)
        is_left = tik_inst.Scalar("int64", init_value=1)

        with tik_inst.for_range(0, Constant.CORE_NUM, block_num=Constant.CORE_NUM) as block_idx:
            self._get_param_by_block_idx_le_16(block_idx, tp, pc_src_n_base, pc_src_n_repeat)
            self._calc_base_pos_le_16(tp, pc_src_n_base, left_zero, top_distance, nc0_counter)

            with tik_inst.for_range(pc_src_n_base, pc_src_n_base + pc_src_n_repeat) as lsn:
                self._update_param_lr_part_le_16(tp, left_zero, left_part, right_part)
                with tik_inst.for_range(0, tp.loop_on_cout_orig) as lct:
                    self._copy_left_part_le_16(tp, self.ub_input, ub_offset, lsn, lct, 0, left_zero, top_distance,
                                               nc0_counter, left_part, right_part, is_left)

                with tik_inst.if_scope(tp.cout_orig_tail != 0):
                    self._copy_left_part_le_16(tp, self.ub_input, ub_offset, lsn,
                                               tp.loop_on_cout_orig, tp.cout_orig_tail, left_zero,
                                               top_distance, nc0_counter, left_part, right_part, is_left)

                with tik_inst.if_scope(right_part != 0):
                    self._update_param_right_le_16(tp, is_left, left_zero, nc0_counter)
                    with tik_inst.for_range(0, tp.loop_on_cout_orig) as lct:
                        self._copy_right_part_le_16(tp, self.ub_input, ub_offset, lsn, lct, 0, left_zero, top_distance,
                                                    nc0_counter, left_part, right_part, is_left)

                    with tik_inst.if_scope(tp.cout_orig_tail != 0):
                        self._copy_right_part_le_16(tp, self.ub_input, ub_offset, lsn, tp.loop_on_cout_orig,
                                                    tp.cout_orig_tail, left_zero, top_distance,
                                                    nc0_counter, left_part, right_part, is_left)

                self._update_param_le_16(tp, left_zero, top_distance, nc0_counter, left_part, right_part, is_left)

    def _get_param_by_block_idx_gt_16(self, block_idx, tp, pc_gc_base, pc_gc_repeat, pc_gc_tail,
                                      pc_cout_orig_base, pc_cout_orig_repeat, pc_cout_orig_tail):
        for i in range(Constant.CORE_NUM):
            with self.tik_inst.if_scope(block_idx == i):
                pc_gc_base.set_as(tp.pcp.loop_gc_base[i])
                pc_gc_repeat.set_as(tp.pcp.loop_gc_repeat[i])
                pc_gc_tail.set_as(tp.pcp.loop_gc_tail[i])
                pc_cout_orig_base.set_as(tp.pcp.loop_cout_orig_base[i])
                pc_cout_orig_repeat.set_as(tp.pcp.loop_cout_orig_repeat[i])
                pc_cout_orig_tail.set_as(tp.pcp.loop_cout_orig_tail[i])

    def _calc_param_gt_16(self, tp, lgc, left_zero, top_distance, nc0_counter, left_part, right_part):
        tik_inst = self.tik_inst
        top_distance.set_as(lgc * Constant.EPB // tp.cin_orig * tp.cout_orig % tp.dst_n)
        nc0_counter.set_as(lgc)
        left_zero.set_as(0)
        left_part.set_as((tp.src_c - (lgc * Constant.EPB) % tp.src_c))
        with tik_inst.if_scope(left_part >= Constant.EPB):
            left_part.set_as(Constant.EPB)
            right_part.set_as(0)
        with tik_inst.else_scope():
            right_part.set_as(Constant.EPB - left_part)

    def _calc_param_cg_tail_gt_16(self, tp, lgc, lct, left_zero, top_distance,
                                  nc0_counter, left_part, right_part, is_left):
        left_zero.set_as(0)
        top_distance.set_as(lgc * Constant.EPB // tp.cin_orig * tp.cout_orig % tp.dst_n)
        nc0_counter.set_as(lgc)
        left_part.set_as((tp.src_c - (lgc * Constant.EPB) % tp.src_c))
        right_part.set_as(0)
        is_left.set_as(1)

    def _calc_src_addr_gt_16(self, tp, lgc, lct, left_part, is_left, src_addr):
        tik_inst = self.tik_inst
        hwc_counter = tik_inst.Scalar("int64", init_value=0)
        with tik_inst.if_scope(is_left == 1):
            hwc_counter.set_as(lgc * Constant.EPB // tp.src_c)
            src_addr.set_as(hwc_counter * tp.cout_orig * tp.vol_chw +\
                            lct * Constant.EPB * tp.vol_chw +\
                            lgc * Constant.EPB % tp.src_c * tp.khw)
        with tik_inst.else_scope():
            hwc_counter.set_as((lgc * Constant.EPB + left_part) // tp.src_c)
            src_addr.set_as(hwc_counter * tp.cout_orig * tp.vol_chw + lct * Constant.EPB * tp.vol_chw)

    def _calc_dst_addr_gt_16(self, tp, lgc, lct, top_distance, nc0_counter, dst_addr):
        dst_addr.set_as(nc0_counter * tp.khw * tp.dst_n * Constant.EPB + lct *
                        Constant.EPB * Constant.EPB + top_distance * Constant.EPB)

    def _copy_in_gt_16(self, tp, ub_input, ub_offset, lgc, lct, cout_orig_tail,
                       left_zero, left_part, right_part, is_left):
        tik_inst = self.tik_inst
        src_addr = tik_inst.Scalar("int64", init_value=0)
        burst_len = tik_inst.Scalar("int64", init_value=0)
        self._calc_src_addr_gt_16(tp, lgc, lct, left_part, is_left, src_addr)
        self._calc_burst_len(tp, left_part, right_part, is_left, burst_len)
        loop_num = tik_inst.Scalar("int64", init_value=cout_orig_tail)
        with tik_inst.if_scope(cout_orig_tail == 0):
            loop_num.set_as(Constant.EPB)
        with tik_inst.for_range(0, loop_num) as i:
            tik_inst.data_move(self.ub_input[Constant.OFFSET_1 + ub_offset * Constant.EPB],
                               self.data_in[src_addr + i * tp.vol_chw], 0, 1, burst_len, 0, 0)
            ub_offset.set_as(ub_offset + burst_len)

    def _copy_out_gt_16(self, tp, ub_input, lgc, lct, cout_orig_tail, top_distance, nc0_counter):
        tik_inst = self.tik_inst
        dst_addr = tik_inst.Scalar("int64", init_value=0)
        cout_orig_tail_s = tik_inst.Scalar("int64", init_value=cout_orig_tail)
        self._calc_dst_addr_gt_16(tp, lgc, lct, top_distance, nc0_counter, dst_addr)
        with tik_inst.if_scope(cout_orig_tail == 0):
            tik_inst.data_move(self.data_out[dst_addr], self.ub_input[Constant.OFFSET_1],
                               0, tp.khw, Constant.EPB, 0, tp.dst_n - Constant.EPB)
        with tik_inst.else_scope():
            tik_inst.data_move(self.data_out[dst_addr], self.ub_input[Constant.OFFSET_1],
                               0, tp.khw, cout_orig_tail_s, Constant.EPB - cout_orig_tail_s,
                               tp.dst_n - cout_orig_tail_s)

    def _copy_left_part_gt_16(self, tp, ub_input, ub_offset, lgc, lct, cout_orig_tail, left_zero, top_distance,
                              nc0_counter, left_part, right_part, is_left):
        ub_offset.set_as(0)
        is_left.set_as(1)
        self._zero_offset_1(ub_input)
        self._zero_offset_2(ub_input)
        self._copy_in_gt_16(tp, ub_input, ub_offset, lgc, lct, cout_orig_tail,
                            left_zero, left_part, right_part, is_left)
        self._reorder_le_16(tp, ub_input, left_zero, left_part, right_part, is_left)
        self._copy_out_gt_16(tp, ub_input, lgc, lct, cout_orig_tail, top_distance, nc0_counter)

    def _copy_right_part_gt_16(self, tp, ub_input, ub_offset, lgc, lct, cout_orig_tail, left_zero, top_distance,
                               nc0_counter, left_part, right_part, is_left):
        ub_offset.set_as(0)
        is_left.set_as(0)
        self._zero_offset_1(ub_input)
        self._zero_offset_2(ub_input)
        self._copy_in_gt_16(tp, ub_input, ub_offset, lgc, lct, cout_orig_tail,
                            left_zero, left_part, right_part, is_left)
        self._reorder_le_16(tp, ub_input, left_zero, left_part, right_part, is_left)
        self._copy_out_gt_16(tp, ub_input, lgc, lct, cout_orig_tail, top_distance, nc0_counter)

    def _update_param_right_gt_16(self, tp, is_left, left_part, left_zero, top_distance):
        is_left.set_as(0)
        left_zero.set_as(left_part)
        top_distance.set_as((top_distance + tp.cout_orig) % tp.dst_n)

    def compute_c_gt_16(self, tp, data_in, data_out):
        """
        nchw_2_fractal_z_g entrance function
        """
        tik_inst = self.tik_inst
        ub_offset = tik_inst.Scalar("int64", init_value=0)

        pc_gc_base = tik_inst.Scalar("int64", init_value=0)
        pc_gc_repeat = tik_inst.Scalar("int64", init_value=0)
        pc_gc_tail = tik_inst.Scalar("int64", init_value=0)
        pc_cout_orig_base = tik_inst.Scalar("int64", init_value=0)
        pc_cout_orig_repeat = tik_inst.Scalar("int64", init_value=0)
        pc_cout_orig_tail = tik_inst.Scalar("int64", init_value=0)

        left_zero = tik_inst.Scalar("int64", init_value=0)
        top_distance = tik_inst.Scalar("int64", init_value=0)
        nc0_counter = tik_inst.Scalar("int64", init_value=0)
        left_part = tik_inst.Scalar("int64", init_value=tp.cin_orig)
        right_part = tik_inst.Scalar("int64", init_value=0)
        is_left = tik_inst.Scalar("int64", init_value=1)

        with tik_inst.for_range(0, Constant.CORE_NUM, block_num=Constant.CORE_NUM) as block_idx:
            self._get_param_by_block_idx_gt_16(block_idx, tp, pc_gc_base, pc_gc_repeat, pc_gc_tail,
                                               pc_cout_orig_base, pc_cout_orig_repeat, pc_cout_orig_tail)

            with tik_inst.for_range(pc_gc_base, pc_gc_base + pc_gc_repeat) as lgc:
                self._calc_param_gt_16(tp, lgc, left_zero, top_distance, nc0_counter, left_part, right_part)

                with tik_inst.for_range(pc_cout_orig_base, pc_cout_orig_base + pc_cout_orig_repeat) as lct:
                    self._copy_left_part_gt_16(tp, self.ub_input, ub_offset, lgc, lct, 0, left_zero,
                                               top_distance, nc0_counter, left_part, right_part, is_left)

                with tik_inst.if_scope(pc_cout_orig_tail != 0):
                    self._copy_left_part_gt_16(tp, self.ub_input, ub_offset,
                                               lgc, pc_cout_orig_base + pc_cout_orig_repeat,
                                               pc_cout_orig_tail, left_zero, top_distance,
                                               nc0_counter, left_part, right_part, is_left)

                with tik_inst.if_scope(right_part != 0):
                    self._update_param_right_gt_16(tp, is_left, left_part, left_zero, top_distance)
                    with tik_inst.for_range(pc_cout_orig_base, pc_cout_orig_base + pc_cout_orig_repeat) as lct:
                        self._copy_right_part_gt_16(tp, self.ub_input, ub_offset, lgc, lct, 0, left_zero,
                                                    top_distance, nc0_counter, left_part, right_part, is_left)

                    with tik_inst.if_scope(pc_cout_orig_tail != 0):
                        self._copy_right_part_gt_16(tp, self.ub_input, ub_offset, lgc,
                                                    pc_cout_orig_base + pc_cout_orig_repeat,
                                                    pc_cout_orig_tail, left_zero, top_distance,
                                                    nc0_counter, left_part, right_part, is_left)

            with tik_inst.if_scope(pc_gc_tail != 0):
                self._calc_param_cg_tail_gt_16(tp, pc_gc_base + pc_gc_repeat, lct, left_zero,
                                               top_distance, nc0_counter, left_part, right_part, is_left)

                with tik_inst.for_range(pc_cout_orig_base, pc_cout_orig_base + pc_cout_orig_repeat) as lct:
                    self._copy_left_part_gt_16(tp, self.ub_input, ub_offset, pc_gc_base + pc_gc_repeat,
                                               lct, 0, left_zero, top_distance, nc0_counter,
                                               left_part, right_part, is_left)

                with tik_inst.if_scope(pc_cout_orig_tail != 0):
                    self._copy_left_part_gt_16(tp, self.ub_input, ub_offset,
                                               pc_gc_base + pc_gc_repeat, pc_cout_orig_base + pc_cout_orig_repeat,
                                               pc_cout_orig_tail, left_zero, top_distance,
                                               nc0_counter, left_part, right_part, is_left)


# 'pylint: disable=unused-argument
@para_check.check_input_type(dict, dict, str, str, int, str)
def nchw_2_fractal_z_g(src, dst, src_format, dst_format, groups, kernel_name="nchw_2_fractal_z_g"):
    """
    algorithm: nchw_2_fractal_z_g

    Parameters
    ----------
    src: dict
        dict with keys(shape, dtype) of src
    dst: dict
        dict with keys(shape, dtype) of dst
    src_format: str
        data format of src
    dst_format: str
        data format of dst
    kernel_name: str
        kernel name, default value is "nchw_2_fractal_z_g"

    Returns
    -------
    tik_instance: tik_instance
    """
    shape_in = src.get("shape")
    shape_out = dst.get("shape")
    in_dtype = src.get("dtype").lower()
    out_dtype = dst.get("dtype").lower()
    tik_inst = tik.Tik()
    data_in = tik_inst.Tensor(in_dtype, shape_in, tik.scope_gm, name="data_in")
    data_out = tik_inst.Tensor(out_dtype, shape_out, tik.scope_gm, name="data_out", is_atomic_add=True)
    instance = Nchw2Fractalzg(tik_inst, data_in, data_out)
    tp = instance._tiling(shape_in, shape_out, groups)
    if (shape_in[1] <= Constant.EPB):
        instance.compute_c_le_16(tp, data_in, data_out)
    else:
        instance.compute_c_gt_16(tp, data_in, data_out)
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[data_in], outputs=[data_out])
