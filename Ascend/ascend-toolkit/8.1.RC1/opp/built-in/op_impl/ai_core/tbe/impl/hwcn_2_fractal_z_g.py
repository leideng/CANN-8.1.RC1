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
hwcn_2_fractal_z_g
"""
import math
from te import tik
import te.platform as tbe_platform
from te.utils import para_check


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    EPB = 16
    UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    OFFSET_1 = 256 * 256
    OFFSET_2 = 0
    MAX_CORE_NUM = 32


# 'pylint: disable=superfluous-parens,useless-object-inheritance
# 'pylint: disable=invalid-name,too-many-instance-attributes,too-many-arguments
# 'pylint: disable=unused-variable,too-few-public-methods,too-many-statements
# 'pylint: disable=no-self-use,consider-using-enumerate,unused-argument,too-many-locals
# 'pylint: disable=singleton-comparison,useless-return,protected-access
class Hwcn2Fractalzg(object):
    """
    Hwcn2Fractalzg
    """
# 'pylint: disable=too-few-public-methods
    class TilingParam(object):
        """
        TilingParam
        """
        class PerCoreParam(object):
            """
            PerCoreParam
            """
            def __init__(self):
                self.loop_kernel_base = []
                self.loop_kernel_repeat = []

                self.loop_src_c_base = []
                self.loop_src_c_repeat = []

                self.loop_src_n_base = []
                self.loop_src_n_repeat = []
                self.loop_src_n_tail = []

                self.loop_groups_base = []
                self.loop_groups_repeat = []

            # 'pylint: disable=too-many-arguments
            def composite_c_le_16(self, pf_kernel_base, pf_kernel_repeat, pf_src_c_base, pf_src_c_repeat,
                                  pf_src_n_base, pf_src_n_repeat, pf_src_n_tail):
                """
                composite_c_le_16
                """
                len_k = len(pf_kernel_base)
                len_c = len(pf_src_c_base)
                len_n = len(pf_src_n_base)
                for k in range(len_k):
                    for c in range(len_c):
                        for n in range(len_n):
                            self.loop_kernel_base.append(pf_kernel_base[k])
                            self.loop_kernel_repeat.append(pf_kernel_repeat[k])
                            self.loop_src_c_base.append(pf_src_c_base[c])
                            self.loop_src_c_repeat.append(pf_src_c_repeat[c])
                            self.loop_src_n_base.append(pf_src_n_base[n])
                            self.loop_src_n_repeat.append(pf_src_n_repeat[n])
                            self.loop_src_n_tail.append(pf_src_n_tail[n])
                self._pad_zero()

            def composite_c_gt_16(self, pf_kernel_base, pf_kernel_repeat, pf_groups_base, pf_groups_repeat):
                """
                composite_c_gt_16
                """
                len_k = len(pf_kernel_base)
                len_g = len(pf_groups_base)
                for k in range(len_k):
                    for g in range(len_g):
                        self.loop_kernel_base.append(pf_kernel_base[k])
                        self.loop_kernel_repeat.append(pf_kernel_repeat[k])
                        self.loop_groups_base.append(pf_groups_base[g])
                        self.loop_groups_repeat.append(pf_groups_repeat[g])
                self._pad_zero()

            @staticmethod
            def _pad_zero_impl(v, n):
                while(len(v) < n):
                    v.append(0)

            def _pad_zero(self):
                self._pad_zero_impl(self.loop_kernel_base, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_kernel_repeat, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_src_c_base, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_src_c_repeat, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_src_n_base, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_src_n_repeat, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_src_n_tail, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_groups_base, Constant.CORE_NUM)
                self._pad_zero_impl(self.loop_groups_repeat, Constant.CORE_NUM)

        def __init__(self, shape_in, shape_out, groups):
            self.cout_orig = shape_in[3] // groups
            self.cout_orig_aligned = ((self.cout_orig + Constant.EPB - 1) // Constant.EPB) * Constant.EPB
            self.cout_orig_burst_len = self.cout_orig_aligned // Constant.EPB
            self.cin_orig = shape_in[2]
            self.max_block_num = math.ceil((self.cin_orig - 1) / Constant.EPB) + 1
            self.kh = shape_in[0]
            self.kw = shape_in[1]
            self.groups = groups
            self.src_c = shape_in[2]
            self.src_n = shape_in[3]
            self.dst_n = shape_out[1] * shape_out[2]
            self.vol_cn = self.src_c * self.src_n
            self.c0 = shape_out[3]
            self.khw = self.kh * self.kw
            self.vol_gc1 = shape_out[0] // self.khw
            self.g = 1
            self.c1 = 1
            self.pf_kernel_base = [] # pf means per factor
            self.pf_kernel_repeat = []
            self.pf_src_c_base = []
            self.pf_src_c_repeat = []
            self.pf_src_n_base = []
            self.pf_src_n_repeat = []
            self.pf_src_n_tail = []
            self.pf_groups_base = []
            self.pf_groups_repeat = []
            self.loop_left_zero = []
            self.loop_top_distance = []
            self.loop_nc0_counter = []
            self.loop_left_zero_tail = []
            self.loop_top_distance_tail = []
            self.loop_nc0_counter_tail = []
            self._update_g()

            self.vol_hwnc0 = self.khw * self.dst_n * Constant.EPB
            self.src_n_unit = 1
            self.burst_len_unit = 1
            self.src_n_tail = 1
            self.burst_len_tail = 1

            self.pcp = self.PerCoreParam()
            if (self.cin_orig <= 16):
                self._dispatch_loop_c_le_16()
                self.pcp.composite_c_le_16(self.pf_kernel_base, self.pf_kernel_repeat,
                                           self.pf_src_c_base, self.pf_src_c_repeat,
                                           self.pf_src_n_base, self.pf_src_n_repeat, self.pf_src_n_tail)
                self._calc_init_position()
                self._calc_init_position_tail()
            else:
                self._dispatch_loop_c_gt_16()
                self.pcp.composite_c_gt_16(self.pf_kernel_base, self.pf_kernel_repeat,
                                           self.pf_groups_base, self.pf_groups_repeat)

        @staticmethod
        def _ceil(m, n):
            return (m + n - 1) // n

        @staticmethod
        def _lcm(m, n):
            return (m * n) // math.gcd(m, n)

        def _update_g(self):
            e = min(self._lcm(self._lcm(self.cin_orig, Constant.EPB) // self.cin_orig,
                              self._lcm(self.cout_orig, Constant.EPB) // self.cout_orig),
                    self.groups)
            self.g = self._ceil(self.groups, e)
            self.c1 = self.vol_gc1 // self.g

        @staticmethod
        def _sum_all(loop):
            res = 0
            for i, _ in enumerate(loop):
                res = res + loop[i]
            return res

        @staticmethod
        def _remove_zero_data(tup):
            tup = [x for x in tup if x != 0]

        def _calc_src_n_unit(self):
            # purpose is not split one block into two
            # `when src_c<=16, src_n_unit=lcm(cout_orig, 16) * K, 1<=cout_orig<=16, K is scale`
            # lcm(cout_orig, 16) 1    2    3    4    5    6    7    8    9    10    11    12    13    14    15    16
            # ---------------------------------------------------------------------------------------------------------
            #                    16   16   48   16   80   48   112  16   144  80    176   48    208   112   240   16
            #                 K  20   20   5    20   3    5    2    20   2    3     2     5     1     2     1     20
            #                    320  320  240  320  240  240  224  320  288  240   352   240   208   224   240   320
            src_n_unit_dict = {1:320, 2:320, 3:240, 4:320, 5:240, 6:240, 7:224, 8:320,
                   9:288, 10:240, 11:352, 12:240, 13:208, 14:224, 15:240, 16:320, 17:112}
            if self.src_c <= Constant.EPB:
                self.src_n_unit = src_n_unit_dict[self.src_c]
                if (self.src_n != 17) and (self.src_n_unit > self.src_n):
                    self.src_n_unit = self.src_n
                else:
                    pass
            else:
                pass

            self.burst_len_unit = math.ceil(self.src_n_unit / Constant.EPB)

        def _calc_src_n_tail(self):
            if self.src_c <= Constant.EPB:
                self.src_n_tail = self.src_n % self.src_n_unit
            self.burst_len_tail = math.ceil(self.src_n_tail / Constant.EPB)

        # 'pylint: disable=too-many-arguments
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
                for i, _ in enumerate(each_factor):
                    base.append(r)
                    r = r + each_factor[i]
                    repeat.append(each_factor[i])

        def _bind_src_n_tail(self, factor_n):
            for _ in range(factor_n - 1):
                self.pf_src_n_tail.append(0)
            self.pf_src_n_tail.append(self.src_n_tail)

        def _dispatch_loop_c_le_16(self):
            factor_k = self.khw
            factor_c = 1
            factor_n = 1
            if factor_k >= Constant.CORE_NUM:
                factor_k = Constant.CORE_NUM

            # step 1: kernel factor
            unit = 1
            self._each_factor_process_num(self.khw, factor_k, unit, self.pf_kernel_base, self.pf_kernel_repeat)

            # step 2: src_c factor
            if self.src_c > Constant.EPB:
                factor_c = self.src_c // Constant.EPB if Constant.CORE_NUM // factor_k > self.src_c // \
                                                         Constant.EPB else Constant.CORE_NUM // factor_k
            unit = Constant.EPB
            self._each_factor_process_num(self.src_c, factor_c, unit, self.pf_src_c_base, self.pf_src_c_repeat)

            # step 3: src_n factor
            factor_n = Constant.CORE_NUM // (factor_k * factor_c)
            self._calc_src_n_unit()
            self._calc_src_n_tail()
            if factor_n == 0:
                factor_n = 1
            self._each_factor_process_num(self.src_n, factor_n, self.src_n_unit,
                                          self.pf_src_n_base, self.pf_src_n_repeat)

            # step 4 : bind src_n_tail to the last one of pf_src_n
            self._bind_src_n_tail(factor_n)

        def _dispatch_loop_c_gt_16(self):
            factor_k = self.khw
            if factor_k >= Constant.CORE_NUM:
                factor_k = Constant.CORE_NUM

            # step 1: kernel factor
            unit = 1
            self._each_factor_process_num(self.khw, factor_k, unit, self.pf_kernel_base, self.pf_kernel_repeat)

            # step 2: src_n factor
            factor_g = Constant.CORE_NUM // (factor_k)
            if factor_g == 0:
                factor_g = 1
            unit = 1
            self._each_factor_process_num(self.groups, factor_g, unit, self.pf_groups_base, self.pf_groups_repeat)

        def _calc_init_position_common(self, block_idx, loop_left_zero, loop_top_distance, loop_nc0_counter):
            #eg, left_zero_cycle(3) eq 16
            #  3 3 3 3 3 1
            #  2 3 3 3 3 2
            #  1 3 3 3 3 3
            left_zero_cycle_dict = {1:16, 2:8, 3:16, 4:4, 5:16, 6:8, 7:16, 8:2, 9:16,
                                    10:8, 11:16, 12:3, 13:16, 14:8, 15:16, 16:1}
            left_zero = 0
            consumed = 0
            cycle = left_zero_cycle_dict[self.src_c]
            nc0_counter = block_idx * self.cin_orig // Constant.EPB
            top_distance = block_idx % cycle * self.cout_orig
            block_idx = block_idx % cycle
            while block_idx > 0:
                if consumed == 0:
                    if left_zero + self.cin_orig < Constant.EPB:
                        left_zero = left_zero + self.cin_orig
                        block_idx = block_idx - 1
                    elif left_zero + self.cin_orig == Constant.EPB:
                        left_zero = 0
                        block_idx = block_idx - 1
                    elif left_zero + self.cin_orig > Constant.EPB:
                        consumed = Constant.EPB - left_zero
                        left_zero = 0
                else:
                    left_zero = self.cin_orig - consumed
                    consumed = 0
                    block_idx = block_idx - 1
            loop_left_zero.append(left_zero)
            loop_nc0_counter.append(nc0_counter)
            loop_top_distance.append(top_distance)

        def _calc_init_position(self):
            for i in range(Constant.CORE_NUM):
                block_idx = self.pcp.loop_src_n_base[i] * self.src_n_unit // self.cout_orig
                self._calc_init_position_common(block_idx, self.loop_left_zero,
                                                self.loop_top_distance,
                                                self.loop_nc0_counter)

        def _calc_init_position_tail(self):
            for i in range(Constant.CORE_NUM):
                block_idx = (self.pcp.loop_src_n_base[i] + self.pcp.loop_src_n_repeat[i]) * self.src_n_unit //\
                             self.cout_orig
                self._calc_init_position_common(block_idx, self.loop_left_zero_tail,
                                                self.loop_top_distance_tail,
                                                self.loop_nc0_counter_tail)

    def __init__(self, tik_inst, data_in, data_out, data_workspace):
        self.tik_inst = tik_inst
        self.ub_input = tik_inst.Tensor("float16", (Constant.UB_SIZE // 2, ), tik.scope_ubuf, "ub_input")
        self.data_in = data_in
        self.data_out = data_out
        self.data_workspace = data_workspace

    def tiling(self, shape_in, shape_out, groups):
        """
        tiling
        """
        tp = self.TilingParam(shape_in, shape_out, groups)
        return tp

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _get_param_by_block_idx_le_16(self, block_idx, tp, pc_kernel_base, pc_kernel_repeat,
                                      pc_src_c_base, pc_src_c_repeat, pc_src_n_base, pc_src_n_repeat,
                                      left_zero, top_distance, nc0_counter,
                                      src_n_tail, left_zero_tail, top_distance_tail, nc0_counter_tail):
        for i in range(Constant.CORE_NUM):
            with self.tik_inst.if_scope(block_idx == i):
                pc_kernel_base.set_as(tp.pcp.loop_kernel_base[i])
                pc_kernel_repeat.set_as(tp.pcp.loop_kernel_repeat[i])
                pc_src_c_base.set_as(tp.pcp.loop_src_c_base[i])
                pc_src_c_repeat.set_as(tp.pcp.loop_src_c_repeat[i])
                pc_src_n_base.set_as(tp.pcp.loop_src_n_base[i])
                pc_src_n_repeat.set_as(tp.pcp.loop_src_n_repeat[i])
                left_zero.set_as(tp.loop_left_zero[i])
                top_distance.set_as(tp.loop_top_distance[i])
                nc0_counter.set_as(tp.loop_nc0_counter[i])
                src_n_tail.set_as(tp.pcp.loop_src_n_tail[i])
                left_zero_tail.set_as(tp.loop_left_zero_tail[i])
                top_distance_tail.set_as(tp.loop_top_distance_tail[i])
                nc0_counter_tail.set_as(tp.loop_nc0_counter_tail[i])

    # 'pylint: disable=too-many-arguments
    def _get_param_by_block_idx_gt_16(self, block_idx, tp, pc_kernel_base, pc_kernel_repeat,
                                      pc_groups_base, pc_groups_repeat):
        for i in range(Constant.CORE_NUM):
            with self.tik_inst.if_scope(block_idx == i):
                pc_kernel_base.set_as(tp.pcp.loop_kernel_base[i])
                pc_kernel_repeat.set_as(tp.pcp.loop_kernel_repeat[i])
                pc_groups_base.set_as(tp.pcp.loop_groups_base[i])
                pc_groups_repeat.set_as(tp.pcp.loop_groups_repeat[i])

    @staticmethod
    def _calc_src_addr_ws(tp, lk, lsn, src_addr):
        src_addr.set_as(lk * tp.vol_cn + lsn * tp.src_n_unit)

    @staticmethod
    def _calc_dst_addr_ws(tp, lk, lsn, dst_addr):
        dst_addr.set_as(lk * tp.src_n * Constant.EPB + lsn * tp.src_n_unit * Constant.EPB)

    # 'pylint: disable=too-many-arguments
    def  _copy_in_ws(self, tp, ub_input, ub_offset, src_addr, is_tail):
        tik_inst = self.tik_inst
        ub_offset.set_as(0)
        burst_len_tail = tik_inst.Scalar("int64", init_value=tp.burst_len_tail)
        with tik_inst.if_scope(is_tail == 0):
            with tik_inst.for_range(0, tp.src_c) as i:
                tik_inst.data_move(ub_input[ub_offset * Constant.EPB], self.data_in[src_addr + i * tp.src_n],
                                   0, 1, tp.burst_len_unit, 0, 0)
                ub_offset.set_as(ub_offset + tp.burst_len_unit)
        with tik_inst.else_scope():
            with tik_inst.for_range(0, tp.src_c) as i:
                tik_inst.data_move(ub_input[ub_offset * Constant.EPB], self.data_in[src_addr + i * tp.src_n],
                                   0, 1, burst_len_tail, 0, 0)
                ub_offset.set_as(ub_offset + burst_len_tail)

    def _zero_offset_1(self, ub_input, repeat=248):
        # mask,dst,scalar, repeat_times, dst_blk_stride, dst_rep_stride
        self.tik_inst.vector_dup(128, ub_input[Constant.OFFSET_1], 0, repeat, 1, 8)

    def _zero_offset_2(self, ub_input, repeat=248):
        self.tik_inst.vector_dup(128, ub_input[Constant.OFFSET_2], 0, repeat, 1, 8)

    def _reorder_ws(self, tp, ub_input, is_tail):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(is_tail == 0):
            src_addr_list = [ub_input[tp.src_n_unit * i] for i in range(Constant.EPB)]
            dst_addr_list = [ub_input[Constant.OFFSET_1 + Constant.EPB * i] for i in range(Constant.EPB)]
            repeat_cnt = tp.burst_len_unit
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride = 0
                dst_stride = 0
            with tik_inst.else_scope():
                src_stride = 1
                dst_stride = 16
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        with tik_inst.else_scope():
            if tp.src_n == 17:
                return
            src_addr_list = [ub_input[tp.src_n_tail * i] for i in range(Constant.EPB)]
            dst_addr_list = [ub_input[Constant.OFFSET_1 + Constant.EPB * i] for i in range(Constant.EPB)]
            repeat_cnt = tik_inst.Scalar("int64", init_value=tp.burst_len_tail)
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride = 0
                dst_stride = 0
            with tik_inst.else_scope():
                src_stride = 1
                dst_stride = 16
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

    def _copy_out_ws(self, tp, ub_input, dst_addr, is_tail):
        tik_inst = self.tik_inst
        src_n_tail = tik_inst.Scalar("int64", init_value=tp.src_n_tail)
        with tik_inst.if_scope(is_tail == 0):
            tik_inst.data_move(self.data_workspace[dst_addr], ub_input[Constant.OFFSET_1], 0, 1, tp.src_n_unit, 0, 0)
        with tik_inst.else_scope():
            tik_inst.data_move(self.data_workspace[dst_addr], ub_input[Constant.OFFSET_1], 0, 1, src_n_tail, 0, 0)

    # 'pylint: disable=too-many-arguments
    def _prepare_by_workspace(self, tp, ub_input, ub_offset, lk, lsn, is_tail):
        tik_inst = self.tik_inst
        src_addr = tik_inst.Scalar("int64", init_value=0)
        dst_addr = tik_inst.Scalar("int64", init_value=0)
        self._calc_src_addr_ws(tp, lk, lsn, src_addr)
        self._calc_dst_addr_ws(tp, lk, lsn, dst_addr)
        self._copy_in_ws(tp, ub_input, ub_offset, src_addr, is_tail)
        self._reorder_ws(tp, ub_input, is_tail)
        self._copy_out_ws(tp, ub_input, dst_addr, is_tail)

    @staticmethod
    def _calc_src_addr_new(tp, lk, lb, src_addr):
        src_addr.set_as(lk * tp.src_n * Constant.EPB + lb * tp.cout_orig * Constant.EPB)

    # 'pylint: disable=too-many-arguments
    @staticmethod
    def _calc_dst_addr_new(tp, lk, top_distance, left_zero, nc0_counter, dst_addr):
        dst_addr.set_as(lk * tp.dst_n * Constant.EPB +\
                        nc0_counter * tp.vol_hwnc0 +\
                        top_distance * Constant.EPB +\
                        left_zero)

    # 'pylint: disable=too-many-arguments
    def _update_param(self, tp, top_distance, left_zero, nc0_counter, consumed):
        tik_inst = self.tik_inst

        with tik_inst.if_scope(consumed != 0):
            top_distance.set_as((top_distance + tp.cout_orig) % tp.dst_n)
            left_zero.set_as(tp.cin_orig - consumed)
            consumed.set_as(0)
        with tik_inst.else_scope():
            with tik_inst.if_scope(left_zero + tp.cin_orig < Constant.EPB):
                top_distance.set_as((top_distance + tp.cout_orig) % tp.dst_n)
                left_zero.set_as(left_zero + tp.cin_orig)
            with tik_inst.else_scope():
                nc0_counter.set_as(nc0_counter + 1)
                with tik_inst.if_scope(left_zero + tp.cin_orig == Constant.EPB):
                    left_zero.set_as(0)
                    top_distance.set_as((top_distance + tp.cout_orig) % tp.dst_n)
                with tik_inst.else_scope(): # (left_zero + tp.cin_orig > Constant.EPB):
                    consumed.set_as(Constant.EPB - left_zero)
                    left_zero.set_as(0)

    def _zero_back(self, tp, ub_input, left_zero):
        tik_inst = self.tik_inst
        zero = self.tik_inst.Scalar("float16", init_value=0)
        with tik_inst.for_range(0, tp.cout_orig) as i:
            with tik_inst.for_range(Constant.EPB - left_zero, tp.cin_orig) as j:
                ub_input[i * Constant.EPB + j].set_as(zero)

    def _move_back_to_front(self, tp, ub_input, consumed):
        tik_inst = self.tik_inst
        zero = self.tik_inst.Scalar("float16", init_value=0)
        scalar_value = self.tik_inst.Scalar("float16", init_value=0)

        #move back to front
        with tik_inst.for_range(0, tp.cout_orig) as i:
            with tik_inst.for_range(0, tp.cin_orig - consumed) as j:
                scalar_value.set_as(ub_input[i * Constant.EPB + j + consumed])
                ub_input[i * Constant.EPB + j].set_as(scalar_value)

        #set tial with consumed len to zero
        with tik_inst.for_range(0, tp.cout_orig) as i:
            with tik_inst.for_range(tp.cin_orig - consumed, tp.cin_orig) as k:
                ub_input[i * Constant.EPB + k].set_as(zero)

    def _move_out_wrapper(self, tp, ub_input, dst_addr, left_zero):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(left_zero + tp.cin_orig < Constant.EPB):
            tik_inst.data_move(self.data_out[dst_addr], ub_input, 0, 1, tp.cout_orig, 0, 0)
        with tik_inst.else_scope(): # overlap scenario
            burst_len = tik_inst.Scalar("int64", init_value=tp.cout_orig - 1)
            with tik_inst.if_scope(burst_len != 0):
                tik_inst.data_move(self.data_out[dst_addr], ub_input, 0, 1, burst_len, 0, 0)
            tik_inst.vector_dup(128, self.ub_input[Constant.OFFSET_1], 0, 1, 1, 8)
            dst_addr.set_as(dst_addr + (tp.cout_orig - 1) * Constant.EPB - (Constant.EPB - tp.cin_orig))
            with tik_inst.for_range(0, tp.cin_orig) as k:
                self.ub_input[Constant.OFFSET_1 + Constant.EPB - tp.cin_orig + k] = \
                    self.ub_input[(tp.cout_orig - 1) * Constant.EPB + k]
            tik_inst.data_move(self.data_out[dst_addr], self.ub_input[Constant.OFFSET_1], 0, 1, 1, 0, 0)

    # 'pylint: disable=too-many-arguments
    def _copy_ws_out_aligned(self, tp, ub_input, lk, lb, left_zero, top_distance, nc0_counter, consumed):
        tik_inst = self.tik_inst
        src_addr = tik_inst.Scalar("int64", init_value=0)
        dst_addr = tik_inst.Scalar("int64", init_value=0)
        self._calc_src_addr_new(tp, lk, lb, src_addr)
        self._calc_dst_addr_new(tp, lk, top_distance, left_zero, nc0_counter, dst_addr)
        tik_inst.data_move(ub_input, self.data_workspace(src_addr), 0, 1, tp.cout_orig, 0, 0)
        self._move_back_to_front(tp, ub_input, consumed)
        self._move_out_wrapper(tp, ub_input, dst_addr, left_zero)
        self._update_param(tp, top_distance, left_zero, nc0_counter, consumed)

    # 'pylint: disable=too-many-arguments
    def _copy_ws_out(self, tp, ub_input, lk, lb, left_zero, top_distance, nc0_counter, consumed):
        tik_inst = self.tik_inst
        src_addr = tik_inst.Scalar("int64", init_value=0)
        dst_addr = tik_inst.Scalar("int64", init_value=0)

        self._calc_src_addr_new(tp, lk, lb, src_addr)
        self._calc_dst_addr_new(tp, lk, top_distance, left_zero, nc0_counter, dst_addr)

        tik_inst.data_move(ub_input, self.data_workspace(src_addr), 0, 1, tp.cout_orig, 0, 0)

        with tik_inst.if_scope(left_zero + tp.cin_orig > Constant.EPB):
            self._zero_back(tp, ub_input, left_zero)
            self._move_out_wrapper(tp, ub_input, dst_addr, left_zero)
        with tik_inst.else_scope():
            self._move_out_wrapper(tp, ub_input, dst_addr, left_zero)

        self._update_param(tp, top_distance, left_zero, nc0_counter, consumed)

        with tik_inst.if_scope(consumed != 0):
            self._copy_ws_out_aligned(tp, ub_input, lk, lb, left_zero, top_distance, nc0_counter, consumed)

    # 'pylint: disable=too-many-arguments
    @staticmethod
    def _calc_block_param(tp, pc_src_n_base, pc_src_n_repeat, block_base, block_repeat,
                          src_n_tail, block_base_tail, block_repeat_tail):
        block_base.set_as((pc_src_n_base * tp.src_n_unit) // tp.cout_orig)
        block_repeat.set_as((pc_src_n_repeat * tp.src_n_unit) // tp.cout_orig)
        block_base_tail.set_as(((pc_src_n_base + pc_src_n_repeat) * tp.src_n_unit) // tp.cout_orig)
        block_repeat_tail.set_as(src_n_tail // tp.cout_orig)

    # 'pylint: disable=too-many-arguments
    @staticmethod
    def _backup(left_zero, top_distance, nc0_counter,
                left_zero_tail, top_distance_tail, nc0_counter_tail,
                left_zero_bk, top_distance_bk, nc0_counter_bk,
                left_zero_tail_bk, top_distance_tail_bk, nc0_counter_tail_bk):
        left_zero_bk.set_as(left_zero)
        top_distance_bk.set_as(top_distance)
        nc0_counter_bk.set_as(nc0_counter)
        left_zero_tail_bk.set_as(left_zero_tail)
        top_distance_tail_bk.set_as(top_distance_tail)
        nc0_counter_tail_bk.set_as(nc0_counter_tail)

    # 'pylint: disable=too-many-arguments
    @staticmethod
    def _restore(left_zero, top_distance, nc0_counter,
                 left_zero_tail, top_distance_tail, nc0_counter_tail,
                 left_zero_bk, top_distance_bk, nc0_counter_bk,
                 left_zero_tail_bk, top_distance_tail_bk, nc0_counter_tail_bk):
        left_zero.set_as(left_zero_bk)
        top_distance.set_as(top_distance_bk)
        nc0_counter.set_as(nc0_counter_bk)
        left_zero_tail.set_as(left_zero_tail_bk)
        top_distance_tail.set_as(top_distance_tail_bk)
        nc0_counter_tail.set_as(nc0_counter_tail_bk)

    # 'pylint: disable=too-many-statements,too-many-locals
    def compute_c_le_16(self, tp):
        """
        hwcn_2_fractal_z_g entrance function
        """
        tik_inst = self.tik_inst
        ub_offset = tik_inst.Scalar("int64", init_value=0)

        pc_kernel_base = tik_inst.Scalar("int64", init_value=0)
        pc_kernel_repeat = tik_inst.Scalar("int64", init_value=0)
        pc_src_c_base = tik_inst.Scalar("int64", init_value=0)
        pc_src_c_repeat = tik_inst.Scalar("int64", init_value=0)
        pc_src_n_base = tik_inst.Scalar("int64", init_value=0)
        pc_src_n_repeat = tik_inst.Scalar("int64", init_value=0)

        left_zero = tik_inst.Scalar("int64", init_value=0)
        top_distance = tik_inst.Scalar("int64", init_value=0)
        nc0_counter = tik_inst.Scalar("int64", init_value=0)

        left_zero_tail = tik_inst.Scalar("int64", init_value=0)
        top_distance_tail = tik_inst.Scalar("int64", init_value=0)
        nc0_counter_tail = tik_inst.Scalar("int64", init_value=0)

        left_zero_bk = tik_inst.Scalar("int64", init_value=0)
        top_distance_bk = tik_inst.Scalar("int64", init_value=0)
        nc0_counter_bk = tik_inst.Scalar("int64", init_value=0)

        left_zero_tail_bk = tik_inst.Scalar("int64", init_value=0)
        top_distance_tail_bk = tik_inst.Scalar("int64", init_value=0)
        nc0_counter_tail_bk = tik_inst.Scalar("int64", init_value=0)

        src_n_tail = tik_inst.Scalar("int64", init_value=0)
        is_tail = tik_inst.Scalar("int64", init_value=0)
        consumed = tik_inst.Scalar("int64", init_value=0)

        block_base = tik_inst.Scalar("int64", init_value=0)
        block_repeat = tik_inst.Scalar("int64", init_value=0)
        block_base_tail = tik_inst.Scalar("int64", init_value=0)
        block_repeat_tail = tik_inst.Scalar("int64", init_value=0)

        with tik_inst.for_range(0, Constant.CORE_NUM, block_num=Constant.CORE_NUM) as block_idx:
            self._get_param_by_block_idx_le_16(block_idx, tp, pc_kernel_base, pc_kernel_repeat, pc_src_c_base,
                                               pc_src_c_repeat, pc_src_n_base, pc_src_n_repeat,
                                               left_zero, top_distance, nc0_counter,
                                               src_n_tail, left_zero_tail, top_distance_tail, nc0_counter_tail)

            self._calc_block_param(tp, pc_src_n_base, pc_src_n_repeat, block_base, block_repeat,
                                   src_n_tail, block_base_tail, block_repeat_tail)

            self._backup(left_zero, top_distance, nc0_counter,
                         left_zero_tail, top_distance_tail, nc0_counter_tail,
                         left_zero_bk, top_distance_bk, nc0_counter_bk,
                         left_zero_tail_bk, top_distance_tail_bk, nc0_counter_tail_bk)

            self._zero_offset_1(self.ub_input)
            self._zero_offset_2(self.ub_input)

            with tik_inst.for_range(pc_kernel_base, pc_kernel_base + pc_kernel_repeat) as lk:
                self._restore(left_zero, top_distance, nc0_counter,
                              left_zero_tail, top_distance_tail, nc0_counter_tail,
                              left_zero_bk, top_distance_bk, nc0_counter_bk,
                              left_zero_tail_bk, top_distance_tail_bk, nc0_counter_tail_bk)

                with tik_inst.for_range(pc_src_n_base, pc_src_n_base + pc_src_n_repeat) as lsn:
                    is_tail.set_as(0)
                    self._prepare_by_workspace(tp, self.ub_input, ub_offset, lk, lsn, is_tail)

                if tp.src_n != 17:
                    with tik_inst.if_scope(src_n_tail != 0):
                        self._zero_offset_1(self.ub_input)
                        self._zero_offset_2(self.ub_input)
                        is_tail.set_as(1)
                        self._prepare_by_workspace(tp, self.ub_input, ub_offset,
                                                   lk, pc_src_n_base + pc_src_n_repeat, is_tail)

            with tik_inst.for_range(pc_kernel_base, pc_kernel_base + pc_kernel_repeat) as lk:
                self._restore(left_zero, top_distance, nc0_counter,
                              left_zero_tail, top_distance_tail, nc0_counter_tail,
                              left_zero_bk, top_distance_bk, nc0_counter_bk,
                              left_zero_tail_bk, top_distance_tail_bk, nc0_counter_tail_bk)
                with tik_inst.for_range(block_base, block_base + block_repeat) as lb:
                    self._copy_ws_out(tp, self.ub_input, lk, lb, left_zero, top_distance, nc0_counter, consumed)
                with tik_inst.if_scope(src_n_tail != 0):
                    with tik_inst.for_range(block_base_tail, block_base_tail + block_repeat_tail) as lb:
                        self._copy_ws_out(tp, self.ub_input, lk, lb,
                                          left_zero_tail, top_distance_tail, nc0_counter_tail, consumed)
        return

    @staticmethod
    def _calc_src_addr_gt_16(tp, lk, lg, consumed_line, src_addr):
        src_addr.set_as(lk * tp.vol_cn + lg * tp.cout_orig + consumed_line * tp.src_n)

    @staticmethod
    def _calc_dst_addr_gt_16(tp, lk, lg, block_seq, dst_addr):
        dst_addr.set_as(lk * tp.dst_n * Constant.EPB + block_seq * tp.vol_hwnc0)
        dst_addr.set_as(dst_addr + (lg * tp.cout_orig) % tp.dst_n * Constant.EPB)

    # 'pylint: disable=too-many-arguments
    def _copy_in_gt_16(self, tp, ub_input, ub_offset, lk, lg, pad_line, cur_line, consumed_line):
        tik_inst = self.tik_inst
        src_addr = tik_inst.Scalar("int64", init_value=0)
        self._calc_src_addr_gt_16(tp, lk, lg, consumed_line, src_addr)
        with tik_inst.for_range(0, cur_line) as i:
            tik_inst.data_move(ub_input[Constant.OFFSET_1 + pad_line * Constant.EPB + ub_offset * Constant.EPB],
                               self.data_in[src_addr + i * tp.src_n],
                               0, 1, tp.cout_orig_burst_len, 0, 0)
            ub_offset.set_as(ub_offset + tp.cout_orig_burst_len)

    def _reorder_gt_16(self, tp, ub_input):
        tik_inst = self.tik_inst
        src_stride = tik_inst.Scalar("int64", init_value=0)
        dst_stride = tik_inst.Scalar("int64", init_value=0)
        src_addr_list = [ub_input[Constant.OFFSET_1 + tp.cout_orig_aligned * i] for i in range(Constant.EPB)]
        dst_addr_list = [ub_input[Constant.OFFSET_2 + Constant.EPB * i] for i in range(Constant.EPB)]
        repeat_cnt = tik_inst.Scalar("int64", init_value=tp.cout_orig_burst_len)

        with tik_inst.if_scope(repeat_cnt > 1):
            src_stride.set_as(1)
            dst_stride.set_as(16)

        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

    # 'pylint: disable=too-many-arguments
    def _copy_out_gt_16(self, tp, ub_input, lk, lg, block_seq):
        tik_inst = self.tik_inst
        dst_addr = tik_inst.Scalar("int64", init_value=0)
        self._calc_dst_addr_gt_16(tp, lk, lg, block_seq, dst_addr)
        tik_inst.data_move(self.data_out[dst_addr], ub_input[Constant.OFFSET_2], 0, 1, tp.cout_orig, 0, 0)

    # 'pylint: disable=too-many-arguments
    @staticmethod
    def _init_line_param(tp, lg, pad_line, last_line, cur_line, consumed_line, block_seq):
        pad_line.set_as((lg * tp.cin_orig) % Constant.EPB)
        last_line.set_as(0)
        cur_line.set_as(Constant.EPB - pad_line)
        consumed_line.set_as(0)
        block_seq.set_as(lg * tp.cin_orig // Constant.EPB)

    # 'pylint: disable=too-many-arguments
    def _update_line_param(self, tp, pad_line, last_line, cur_line, consumed_line, block_seq):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(pad_line != 0):
            last_line.set_as(Constant.EPB - pad_line)
            pad_line.set_as(0)
        with tik_inst.else_scope():
            with tik_inst.if_scope(consumed_line + Constant.EPB <= tp.cin_orig):
                last_line.set_as(Constant.EPB)
            with tik_inst.else_scope():
                last_line.set_as(tp.cin_orig - consumed_line)
        consumed_line.set_as(consumed_line + last_line)
        with tik_inst.if_scope(tp.cin_orig - consumed_line < Constant.EPB):
            cur_line.set_as(tp.cin_orig - consumed_line)
        with tik_inst.else_scope():
            cur_line.set_as(Constant.EPB)
        block_seq.set_as(block_seq + 1)

    # 'pylint: disable=too-many-locals
    def compute_c_gt_16(self, tp):
        """
        hwcn_2_fractal_z_g entrance function
        """
        tik_inst = self.tik_inst
        ub_offset = tik_inst.Scalar("int64", init_value=0)

        pc_kernel_base = tik_inst.Scalar("int64", init_value=0)
        pc_kernel_repeat = tik_inst.Scalar("int64", init_value=0)
        pc_groups_base = tik_inst.Scalar("int64", init_value=0)
        pc_groups_repeat = tik_inst.Scalar("int64", init_value=0)

        pad_line = tik_inst.Scalar("int64", init_value=0)
        last_line = tik_inst.Scalar("int64", init_value=0)
        cur_line = tik_inst.Scalar("int64", init_value=0)
        consumed_line = tik_inst.Scalar("int64", init_value=0)
        block_seq = tik_inst.Scalar("int64", init_value=0)

        with tik_inst.for_range(0, Constant.CORE_NUM, block_num=Constant.CORE_NUM) as block_idx:
            self._get_param_by_block_idx_gt_16(block_idx, tp, pc_kernel_base, pc_kernel_repeat,
                                               pc_groups_base, pc_groups_repeat)

            with tik_inst.for_range(pc_kernel_base, pc_kernel_base + pc_kernel_repeat) as lk:
                with tik_inst.for_range(pc_groups_base, pc_groups_base + pc_groups_repeat) as lg:
                    self._init_line_param(tp, lg, pad_line, last_line, cur_line, consumed_line, block_seq)
                    with tik_inst.for_range(0, tp.max_block_num):
                        with tik_inst.if_scope(consumed_line != tp.cin_orig):
                            self._zero_offset_1(self.ub_input, tp.cout_orig_burst_len * 2)
                            ub_offset.set_as(0)
                            self._copy_in_gt_16(tp, self.ub_input, ub_offset, lk, lg, pad_line, cur_line, consumed_line)
                            self._reorder_gt_16(tp, self.ub_input)
                            self._copy_out_gt_16(tp, self.ub_input, lk, lg, block_seq)
                            self._update_line_param(tp, pad_line, last_line, cur_line, consumed_line, block_seq)


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
@para_check.check_input_type(dict, dict, str, str, int, str)
def hwcn_2_fractal_z_g(src, dst, src_format, dst_format, groups, kernel_name="hwcn_2_fractal_z_g"):
    """
    algorithm: hwcn_2_fractal_z_g

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
        kernel name, default value is "hwcn_2_fractal_z_g"

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
    data_workspace = tik_inst.Tensor(in_dtype, shape_out, tik.scope_gm, name="data_workspace", is_workspace=True)
    instance = Hwcn2Fractalzg(tik_inst, data_in, data_out, data_workspace)
    tp = instance.tiling(shape_in, shape_out, groups)
    if tp.cin_orig <= Constant.EPB:
        instance.compute_c_le_16(tp)
    else:
        instance.compute_c_gt_16(tp)
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[data_in], outputs=[data_out])
