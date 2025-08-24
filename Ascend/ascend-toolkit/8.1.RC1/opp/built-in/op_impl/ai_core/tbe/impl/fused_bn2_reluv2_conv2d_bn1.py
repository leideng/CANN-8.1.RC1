#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
fused_bn2_reluv2_conv2d_bn1
"""

import tbe
from tbe import tik
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils.errormgr import error_manager_cube as err_man
from impl.fused_bn2_reluv2_conv2d_bn1_base import Bn2ReluV2Conv2dBn1NoRedundant


class Bn2ReluV2Conv2dBn1BatchByBatchForStrideTwoHandSync(Bn2ReluV2Conv2dBn1NoRedundant):
    """
    class of Bn2ReluV2Conv2dBn1BatchByBatchForStrideTwoHandSync
    """

    def __init__(self, fmap_ori_shape, filters_ori_shape, padding, stride, dilation, groups, factor, epsilon, tiling,
                 kernel_name="Bn2ReluV2Conv2dBn1BatchByBatchForStrideTwoHandSync"):
        super(Bn2ReluV2Conv2dBn1BatchByBatchForStrideTwoHandSync, self).__init__(fmap_ori_shape, filters_ori_shape,
                                                                                 padding, stride, dilation, groups,
                                                                                 factor, epsilon, tiling, kernel_name)

    def compute(self):
        """
        main process
        """
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            self._init_all_ub_tensor()
            self._init_l1_l0a_l0b_l0c_tensor()
            self._bn2_mean_varience_process_handsync()
            self._init_sum_and_square_sum()
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE2", 1)
            self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_MTE2", 0)
            self.tik_instance.set_flag("PIPE_V", "PIPE_M", 0)
            self.tik_instance.set_flag("PIPE_V", "PIPE_M", 1)
            self.tik_instance.set_flag("PIPE_M", "PIPE_MTE2", 3)
            # 绑多核
            with self.tik_instance.for_range(0, self.b_block_factor, block_num=self.b_block_factor,
                                             name="b_block") as b_block:
                with self.tik_instance.for_range(0, self.b_redundant_factor,
                                                 name="pre_core_precess_batch") as pre_core_process_batch:
                    axises = {
                        "b_block": b_block,
                        "pre_core_process_batch": pre_core_process_batch,
                        "n_redundant": 0,
                    }
                    self._first_bn_update_process_handsync(axises)
                    self.tik_instance.set_flag("PIPE_MTE3", "PIPE_MTE1", 0)
                    self.next_batch.set_as(
                        b_block * self.b_redundant_factor * self.b_inner_factor +
                        pre_core_process_batch * self.b_inner_factor)
                    self.preload_next_batch.set_as(self.next_batch)
                    self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE2", 0)
                    with self.tik_instance.if_scope(self.already_load_l1_h_length < self.height_in):
                        self._preload_input_x_handsync()
                    self.kernel_process_handsync(axises)
                    self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE1", 0)
                    self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE1", 1)
                    self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE2", 0)
                    self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE2", 1)
                    self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 0)
                    self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 1)
                    self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 2)
                    self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 3)
                    self.last_bn_reduce_process_handsync()
                    with self.tik_instance.if_scope(self.l0c_pingpong == 1):
                        self.tik_instance.set_flag("PIPE_V", "PIPE_M", 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.set_flag("PIPE_V", "PIPE_M", 1)
                self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE2", 3)
                self.tik_instance.wait_flag("PIPE_V", "PIPE_M", 0)
                self.tik_instance.wait_flag("PIPE_V", "PIPE_M", 1)
                self.move_sum_and_square_sum_toddr_handsync()
        self.tik_instance.BuildCCE(self.kernel_name, self.input_tensors, self.output_tensors)

    def move_sum_and_square_sum_toddr_handsync(self):
        """
        move sum and square sum to ddr
        """
        self.tik_instance.vadd(16, self.last_sum_output_ub,
                               self.output_sum_inub,
                               self.output_sum_inub[16],
                               self.channel_out1,
                               1, 1, 1, 2, 8, 8)
        self.tik_instance.vadd(16, self.last_square_sum_output_ub,
                               self.output_square_sum_inub,
                               self.output_square_sum_inub[16],
                               self.channel_out1,
                               1, 1, 1, 2, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        for i in range(2, 4):
            self.tik_instance.vadd(16, self.last_sum_output_ub,
                                   self.output_sum_inub[i * 16],
                                   self.last_sum_output_ub,
                                   self.channel_out1,
                                   1, 1, 1, 2, 8, 2)
            if i == 3:
                self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
            self.tik_instance.vadd(16, self.last_square_sum_output_ub,
                                   self.output_square_sum_inub[i * 16],
                                   self.last_square_sum_output_ub,
                                   self.channel_out1,
                                   1, 1, 1, 2, 8, 2)
            self.tik_instance.pipe_barrier("PIPE_V")
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 1)
        self.tik_instance.set_atomic_add(1)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.data_move(
            self.output_sum[0, 0, 0, 0, 0],
            self.last_sum_output_ub,
            0,
            1, self.channel_out1 * 2, 0, 0)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 1)
        self.tik_instance.data_move(
            self.output_square_sum[0, 0, 0, 0, 0],
            self.last_square_sum_output_ub,
            0,
            1, self.channel_out1 * 2, 0, 0)
        self.tik_instance.set_atomic_add(0)

    def last_bn_reduce_process_handsync(self):
        """
        the last bn reduce process for every batch
        """
        howo_lo_index = self.front_m * self.m_l0a_factor * self.hw_out0
        howo_lo_length = self.m_l0a_factor * self.hw_out0
        with self.tik_instance.if_scope((howo_lo_index + howo_lo_length) <= (self.height_out * self.width_out)):
            with self.tik_instance.for_range(0, self.n_l0a_factor, name="n_l0") as n_l0:
                square_size = self.b_inner_factor * self.m_l0a_factor * self.hw_out0 * self.channel_out0
                with self.tik_instance.if_scope(self.l0c_pingpong == 1):
                    self.tik_instance.pipe_barrier("PIPE_V")
                    with self.tik_instance.if_scope(n_l0 == 0):
                        self.tik_instance.wait_flag("PIPE_M", "PIPE_V", 0)
                    self.tik_instance.data_move(
                        self.fmap_ub,
                        self.fmap_l0c_ping[0, n_l0, 0, 0, 0],
                        0,
                        self.b_inner_factor,
                        self.m_l0a_factor,
                        (self.n_l0a_factor - 1) * self.m_l0a_factor,
                        0)
                with self.tik_instance.else_scope():
                    self.tik_instance.pipe_barrier("PIPE_V")
                    with self.tik_instance.if_scope(n_l0 == 0):
                        self.tik_instance.wait_flag("PIPE_M", "PIPE_V", 1)
                    self.tik_instance.data_move(
                        self.fmap_ub,
                        self.fmap_l0c_pong[0, n_l0, 0, 0, 0],
                        0,
                        self.b_inner_factor,
                        self.m_l0a_factor,
                        (self.n_l0a_factor - 1) * self.m_l0a_factor,
                        0)
                self.tik_instance.pipe_barrier("PIPE_V")
                with self.tik_instance.if_scope(n_l0 > 0):
                    self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 0)
                # mv ub to gm
                fp32_loop_times, fp32_redundant, fp32_repeat_times = self.calu_loop_time(square_size)
                self.res_fmap_convertf16(self.fmap_f16_ub_ping, self.fmap_ub, {"fp32_loop_times": fp32_loop_times,
                                                                               "fp32_redundant": fp32_redundant,
                                                                               "fp32_repeat_times": fp32_repeat_times})
                dst_b_offset, dst_m_offset, dst_n_offset = self.calc_dst_b_n_m_offset(m_l0=0, n_l0=n_l0)
                self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
                self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
                self.tik_instance.data_move(
                    self.output_convolution[dst_b_offset, dst_n_offset, dst_m_offset, 0],
                    self.fmap_f16_ub_ping,
                    0,
                    self.b_inner_factor,
                    self.m_l0a_factor * self.hw_out0,
                    0,
                    self.channel_out1 * self.height_out *
                    self.width_out - self.m_l0a_factor * self.hw_out0)
                # square sum
                with self.tik_instance.if_scope(n_l0 < (self.n_l0a_factor - 1)):
                    self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 0)
                self.calu_square_sum(self.fmap_ub, fp32_loop_times, fp32_redundant, fp32_repeat_times)
                self.tik_instance.pipe_barrier("PIPE_V")
                self.dichotomy_sum_and_square_sum_handsync(self.fmap_ub, self.square_mul_ub,
                                                           square_size, dst_n_offset)
        with self.tik_instance.else_scope():
            self.last_bn_reduce_handsync_tail_process()

    def last_bn_reduce_handsync_tail_process(self):
        """
        tail process for bn reduce
        """
        m_tail_size = self.height_out * self.width_out - \
                      (self.m_redundant_factor - 1) * self.m_l0a_factor * self.hw_out0
        tail_square_size = self.b_inner_factor * m_tail_size * self.channel_out0
        with self.tik_instance.for_range(0, self.n_l0a_factor, name="n_l0") as n_l0:
            with self.tik_instance.if_scope(self.l0c_pingpong == 1):
                self.tik_instance.pipe_barrier("PIPE_V")
                with self.tik_instance.if_scope(n_l0 == 0):
                    self.tik_instance.wait_flag("PIPE_M", "PIPE_V", 0)
                self.tik_instance.data_move(
                    self.fmap_ub,
                    self.fmap_l0c_ping[0, n_l0, 0, 0, 0],
                    0,
                    self.b_inner_factor,
                    self.m_l0a_factor,
                    (self.n_l0a_factor - 1) * self.m_l0a_factor,
                    0)
            with self.tik_instance.else_scope():
                self.tik_instance.pipe_barrier("PIPE_V")
                with self.tik_instance.if_scope(n_l0 == 0):
                    self.tik_instance.wait_flag("PIPE_M", "PIPE_V", 1)
                self.tik_instance.data_move(
                    self.fmap_ub,
                    self.fmap_l0c_pong[0, n_l0, 0, 0, 0],
                    0,
                    self.b_inner_factor,
                    self.m_l0a_factor,
                    (self.n_l0a_factor - 1) * self.m_l0a_factor,
                    0)

            # mv ub to gm
            res_conv = self.fmap_ub
            self.tik_instance.pipe_barrier("PIPE_V")
            with self.tik_instance.if_scope(n_l0 > 0):
                self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 0)
            fp32_loop_times, fp32_redundant, fp32_repeat_times = self.calu_loop_time(tail_square_size)
            self.res_fmap_convertf16(self.fmap_f16_ub_ping, res_conv, {"fp32_loop_times": fp32_loop_times,
                                                                       "fp32_redundant": fp32_redundant,
                                                                       "fp32_repeat_times": fp32_repeat_times})
            self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 1)
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 1)
            dst_b_offset, dst_m_offset, dst_n_offset = self.calc_dst_b_n_m_offset(m_l0=0, n_l0=n_l0)
            self.tik_instance.data_move(
                self.output_convolution[dst_b_offset, dst_n_offset, dst_m_offset, 0],
                self.fmap_f16_ub_ping,
                0,
                self.b_inner_factor,
                m_tail_size,
                0,
                self.channel_out1 * self.height_out *
                self.width_out - m_tail_size)
            # square sum
            with self.tik_instance.if_scope(n_l0 < (self.n_l0a_factor - 1)):
                self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 0)
            self.calu_square_sum(res_conv, fp32_loop_times, fp32_redundant, fp32_repeat_times)
            self.tik_instance.pipe_barrier("PIPE_V")
            if self.b_inner_factor == 2:
                for i in range(self.b_inner_factor):
                    self.dichotomy_sum_and_square_sum_handsync(res_conv, self.square_mul_ub,
                                                               tail_square_size // self.b_inner_factor,
                                                               dst_n_offset)
                    if i + 1 < self.b_inner_factor:
                        self.tik_instance.data_move(res_conv,
                                                    res_conv[tail_square_size // self.b_inner_factor],
                                                    0, 1, tail_square_size // self.b_inner_factor // 8,
                                                    0, 0)
                        self.tik_instance.data_move(self.square_mul_ub,
                                                    self.square_mul_ub[tail_square_size // self.b_inner_factor],
                                                    0, 1, tail_square_size // self.b_inner_factor // 8,
                                                    0, 0)
            else:
                self.dichotomy_sum_and_square_sum_handsync(res_conv, self.square_mul_ub,
                                                           tail_square_size, dst_n_offset)

    def kernel_process_handsync(self, axises):
        """
        kernel process for conv calculation
        """
        self.tik_instance.pipe_barrier("PIPE_MTE3")
        self.tik_instance.set_flag("PIPE_M", "PIPE_MTE1", 0)
        self.tik_instance.set_flag("PIPE_M", "PIPE_MTE1", 1)
        self.tik_instance.set_flag("PIPE_M", "PIPE_MTE2", 0)
        self.tik_instance.set_flag("PIPE_M", "PIPE_MTE2", 1)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 0)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 1)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 2)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 3)
        with self.tik_instance.for_range(0, self.n_redundant_factor, name='n_redundant') as n_redundant:
            axises['n_redundant'] = n_redundant
            with self.tik_instance.for_range(0, self.n_l1a_factor, name="n_l1a") as n_l1a:
                with self.tik_instance.for_range(0, self.m_redundant_factor, name="m_redundant") as m_redundant:
                    with self.tik_instance.for_range(0, self.k_redundant_factor, name="k_redundant") as k_redundant:
                        with self.tik_instance.if_scope(self.l0b_pingpong == 0):
                            self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE2", 0)
                        with self.tik_instance.else_scope():
                            self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE2", 1)
                        axises["m_redundant"] = m_redundant
                        axises["n_l1a"] = n_l1a
                        axises["k_redundant"] = k_redundant
                        self.filters_l0b_process_handsync(axises)
                        with self.tik_instance.if_scope(self.l0b_pingpong == 0):
                            self.tik_instance.set_flag("PIPE_MTE2", "PIPE_M", 0)
                        with self.tik_instance.else_scope():
                            self.tik_instance.set_flag("PIPE_MTE2", "PIPE_M", 1)
                        with self.tik_instance.if_scope(self.l0a_pingpong == 0):
                            self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE1", 0)
                        with self.tik_instance.else_scope():
                            self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE1", 1)
                        with self.tik_instance.for_range(0, self.b_inner_factor, name="b_inner") as b_inner:
                            axises["b_inner"] = b_inner
                            self.fmap_l0a_process_handsync(axises)
                            self.mad_process_handsync(axises)
                            self.l0a_pingpong.set_as(1 - self.l0a_pingpong)
                        self.l0b_pingpong.set_as(1 - self.l0b_pingpong)
                        self._bn2_and_bn1_merge(axises)

    def bn_reduce_process_handsync(self, axises):
        """
        bn reduce process in kernel_process
        """
        n_l0_and_kl1_comm = Bn2ReluV2Conv2dBn1NoRedundant.get_gcd(self.n_l0a_factor, self.k_redundant_factor)
        n_l0_single_length = self.n_l0a_factor // n_l0_and_kl1_comm
        k_redundant = axises["k_redundant"]
        howo_lo_index = self.front_m * self.m_l0a_factor * self.hw_out0
        howo_lo_length = self.m_l0a_factor * self.hw_out0

        with self.tik_instance.if_scope(k_redundant % (self.k_redundant_factor // n_l0_and_kl1_comm) == 0):
            n_l0_gap = k_redundant // (self.k_redundant_factor // n_l0_and_kl1_comm)
            with self.tik_instance.for_range(n_l0_gap * n_l0_single_length, (n_l0_gap + 1) * n_l0_single_length,
                                             name="n_l0") as n_l0:
                with self.tik_instance.if_scope((howo_lo_index + howo_lo_length) <= (self.height_out * self.width_out)):
                    self._full_conv_res_bn_reduce_process(k_redundant, n_l0)

                with self.tik_instance.else_scope():
                    self._tail_bn_reduce_process_handsync(k_redundant, n_l0)
        with self.tik_instance.if_scope(k_redundant == (self.k_redundant_factor - 1)):
            with self.tik_instance.if_scope(self.l0c_pingpong == 1):
                self.tik_instance.set_flag("PIPE_V", "PIPE_M", 0)
            with self.tik_instance.else_scope():
                self.tik_instance.set_flag("PIPE_V", "PIPE_M", 1)

    def dichotomy_sum_and_square_sum_handsync(self, sum_ub, square_sum_ub, total_blocks, dst_n_offset):
        """
        dichotomy sum and square sum to avoid issue queue full
        """
        # 如果不能被64整除就直接做累加操作
        if total_blocks % 64 != 0:
            total_repeat = total_blocks // 64
            redundant = total_blocks % 64
            self.tik_instance.vadd(64,
                                   self.output_sum_inub[0, dst_n_offset, 0, 0, 0],
                                   sum_ub,
                                   self.output_sum_inub[0, dst_n_offset, 0, 0, 0],
                                   total_repeat, 1, 1, 1, 0, 8, 0)
            self.tik_instance.vadd(64,
                                   self.output_square_sum_inub[0, dst_n_offset, 0, 0, 0],
                                   square_sum_ub,
                                   self.output_square_sum_inub[0, dst_n_offset, 0, 0, 0],
                                   total_repeat, 1, 1, 1, 0, 8, 0)
            self.tik_instance.pipe_barrier("PIPE_V")
            if redundant > 0:
                self.tik_instance.vadd(redundant,
                                       self.output_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       sum_ub[total_blocks // 64 * 64],
                                       self.output_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       1, 1, 1, 1,
                                       0, 8, 0)
                self.tik_instance.vadd(redundant,
                                       self.output_square_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       square_sum_ub[total_blocks // 64 * 64],
                                       self.output_square_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       1, 1, 1, 1, 0, 8, 0)
                self.tik_instance.pipe_barrier("PIPE_V")
        else:
            threshold = 7
            total_repeat = total_blocks // 64
            dich_repeat = total_repeat // 2
            redundant = total_repeat % 2
            self.tik_instance.vadd(64,
                                   self.ub_fm1_workbuf2,
                                   sum_ub,
                                   sum_ub[64],
                                   dich_repeat,
                                   1, 1, 1, 8, 16, 16)
            self.tik_instance.vadd(64,
                                   self.ub_fm1_workbuf2[total_blocks // 2],
                                   square_sum_ub,
                                   square_sum_ub[64],
                                   dich_repeat,
                                   1, 1, 1, 8, 16, 16)
            self.tik_instance.pipe_barrier("PIPE_V")
            if redundant > 0:
                self.tik_instance.vadd(64,
                                       self.ub_fm1_workbuf2,
                                       self.ub_fm1_workbuf2,
                                       sum_ub[total_repeat // 2 * 2 * 64],
                                       1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vadd(64,
                                       self.ub_fm1_workbuf2[total_blocks // 2],
                                       self.ub_fm1_workbuf2[total_blocks // 2],
                                       square_sum_ub[total_repeat // 2 * 2 * 64],
                                       1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.pipe_barrier("PIPE_V")
            total_repeat = dich_repeat
            dich_repeat = total_repeat // 2
            vadd_0 = square_sum_ub
            vadd_1 = self.ub_fm1_workbuf2
            index = 0
            while (total_repeat % 2 == 0 and dich_repeat >= threshold):
                self.tik_instance.vadd(64,
                                       vadd_0,
                                       vadd_1[64],
                                       vadd_1,
                                       dich_repeat,
                                       1, 1, 1, 8, 16, 16)
                self.tik_instance.vadd(64,
                                       vadd_0[total_blocks // 2],
                                       vadd_1[total_blocks // 2],
                                       vadd_1[total_blocks // 2 + 64],
                                       dich_repeat,
                                       1, 1, 1, 8, 16, 16)
                self.tik_instance.pipe_barrier("PIPE_V")
                total_repeat = dich_repeat
                dich_repeat = total_repeat // 2
                vadd_0, vadd_1 = vadd_1, vadd_0
                index += 1
            if index % 2 == 0:
                self.tik_instance.vadd(64,
                                       self.output_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       self.ub_fm1_workbuf2,
                                       self.output_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       total_repeat,
                                       1, 1, 1, 0, 8, 0)
                self.tik_instance.vadd(64,
                                       self.output_square_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       self.ub_fm1_workbuf2[total_blocks // 2],
                                       self.output_square_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       total_repeat,
                                       1, 1, 1, 0, 8, 0)
                self.tik_instance.pipe_barrier("PIPE_V")
            else:
                self.tik_instance.vadd(64,
                                       self.output_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       square_sum_ub,
                                       self.output_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       total_repeat,
                                       1, 1, 1, 0, 8, 0)
                self.tik_instance.vadd(64,
                                       self.output_square_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       square_sum_ub[total_blocks // 2],
                                       self.output_square_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       total_repeat,
                                       1, 1, 1, 0, 8, 0)
                self.tik_instance.pipe_barrier("PIPE_V")

    def bn_update_process_handsync(self, axises):
        """
        bn update process in kernel process
        """
        k_redundant = axises["k_redundant"]
        # kh * kw * cin1 与cin1提公共轴，下面的默认逻辑是每次都只算1个cin1，不做特殊处理
        cin1_and_k_common_length = Bn2ReluV2Conv2dBn1NoRedundant.get_gcd(self.k_redundant_factor, self.channel_in1)

        cin1_and_k_single_jump = self.k_redundant_factor // cin1_and_k_common_length
        cin1_single_length = self.channel_in1 // cin1_and_k_common_length
        with self.tik_instance.if_scope(k_redundant % cin1_and_k_single_jump == 0):
            self.this_time_calu_length.set_as(self.preload_this_time_calu_length)
            with self.tik_instance.if_scope(self.this_time_calu_length > 0):
                cin1_length_gap = k_redundant // cin1_and_k_single_jump
                with self.tik_instance.for_range(cin1_length_gap * cin1_single_length,
                                                 (cin1_length_gap + 1) * cin1_single_length,
                                                 name="cur_cin1") as cur_cin1:
                    self._bn_update_in_conv_single_chanel_process(axises, cur_cin1)

    def mad_process_handsync(self, axises):
        """
        mmad process
        """
        k_redundant = axises["k_redundant"]
        b_inner = axises["b_inner"]
        dst_offset = b_inner
        src_offset = b_inner
        self._mmad_handsync_before(k_redundant)
        with self.tik_instance.if_scope(self.l0c_pingpong == 0):
            self._mad_l0c_ping(dst_offset, k_redundant, src_offset)
        with self.tik_instance.else_scope():
            self._mad_l0c_pong(dst_offset, k_redundant, src_offset)
        self._mmad_handsync_after(axises, k_redundant)

    def fmap_l0a_process_handsync(self, axises):
        """
        fmap l1 to l0a process
        """
        k_redundant = axises["k_redundant"]
        m_redundant = axises["m_redundant"]
        batch_index = 0
        howo_l0_index = m_redundant * self.m_l0a_factor * self.hw_out0
        ho_l0_index = howo_l0_index // self.width_out
        hin_index = ho_l0_index * self.stride_h
        wo_l0_index = howo_l0_index % self.width_out
        cin1_l0_index = (k_redundant * self.k_l0a_factor) // (self.kernel_h * self.kernel_w)
        khkw_l0_index = (k_redundant * self.k_l0a_factor) % (self.kernel_h * self.kernel_w)
        kh_l0_index = khkw_l0_index // self.kernel_w
        kw_l0_index = khkw_l0_index % self.kernel_w
        b_inner = axises["b_inner"]
        pad = [self.pad_l, self.pad_r, self.pad_t, self.pad_b]
        self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_MTE1", 0)
        with self.tik_instance.if_scope(hin_index > 0):
            self.hin_l1_start_pos.set_as(hin_index - self.pad_t)
        with self.tik_instance.else_scope():
            self.hin_l1_start_pos.set_as(0)
        with self.tik_instance.if_scope(self.l0a_pingpong == 0):
            self._load_l0a_ping([b_inner, batch_index, cin1_l0_index, kh_l0_index, kw_l0_index, pad, wo_l0_index])
        with self.tik_instance.else_scope():
            self._load_l0a_pong([b_inner, batch_index, cin1_l0_index, kh_l0_index, kw_l0_index, pad, wo_l0_index])

    def filters_l0b_process_handsync(self, axises):
        """
        load weight from ddr or l1 to l0b
        """
        k_redundant = axises["k_redundant"]
        n_l1 = axises["n_l1a"]
        with self.tik_instance.if_scope(self.l0b_pingpong == 0):
            if self.filters_load_from_l1_flag:
                self.tik_instance.pipe_barrier("PIPE_MTE1")
                with self.tik_instance.for_range(0, self.k_l0a_factor, name="k_l0") as k_l0:
                    dst_offset = k_l0

                    src_k_offset = (k_redundant * self.k_l0a_factor + k_l0)
                    src_n_offset = n_l1 * self.n_l0a_factor
                    self.tik_instance.load2dv1(self.filters_l0b_ping[dst_offset, 0, 0, 0],
                                                self.filters_l1[src_k_offset, src_n_offset, 0, 0], 0, self.n_l0a_factor,
                                                1, 0)
            else:
                self.tik_instance.pipe_barrier("PIPE_MTE2")
                with self.tik_instance.for_range(0, self.k_l0a_factor, name="k_l0") as k_l0:
                    dst_offset = k_l0

                    src_k_offset = (k_redundant * self.k_l0a_factor + k_l0)
                    src_n_offset = n_l1 * self.n_l0a_factor
                    self.tik_instance.load2dv1(self.filters_l0b_ping[dst_offset, 0, 0, 0],
                                                self.filters_input[src_k_offset, src_n_offset, 0, 0], 0,
                                                self.n_l0a_factor, 1, 0)
        with self.tik_instance.else_scope():
            if self.filters_load_from_l1_flag:
                self.tik_instance.pipe_barrier("PIPE_MTE1")
                with self.tik_instance.for_range(0, self.k_l0a_factor, name="k_l0") as k_l0:
                    dst_offset = k_l0

                    src_k_offset = (k_redundant * self.k_l0a_factor + k_l0)
                    src_n_offset = n_l1 * self.n_l0a_factor
                    self.tik_instance.load2dv1(self.filters_l0b_pong[dst_offset, 0, 0, 0],
                                                self.filters_l1[src_k_offset, src_n_offset, 0, 0], 0, self.n_l0a_factor,
                                                1, 0)
            else:
                self.tik_instance.pipe_barrier("PIPE_MTE2")
                with self.tik_instance.for_range(0, self.k_l0a_factor, name="k_l0") as k_l0:
                    dst_offset = k_l0

                    src_k_offset = (k_redundant * self.k_l0a_factor + k_l0)
                    src_n_offset = n_l1 * self.n_l0a_factor
                    self.tik_instance.load2dv1(self.filters_l0b_pong[dst_offset, 0, 0, 0],
                                                self.filters_input[src_k_offset, src_n_offset, 0, 0], 0,
                                                self.n_l0a_factor, 1, 0)

    def calc_mask_in_bn_update(self, batch_index, cin1):
        """
        calculate mask
        """
        # 输出mask外提可以减少scala 操作
        with self.tik_instance.for_range(0, self.b_inner_factor, "inner_batch") as b_inner:
            mask_loop_times = (self.already_load_l1_h_length * self.width_in *
                               self.channel_in0) // \
                              (128 * 255)
            mask_loop_redundant = (self.already_load_l1_h_length * self.width_in *
                                   self.channel_in0) % \
                                  (128 * 255)
            mask_repeat = (mask_loop_redundant + 127) // 128
            # channel_in0在，所以一定能被16除尽
            batch_start_pos = b_inner * self.already_load_l1_h_length * self.width_in * self.channel_in0
            with self.tik_instance.if_scope(mask_loop_times > 0):
                with self.tik_instance.for_range(0, mask_loop_times, name="mask_loop") as mask_loop:
                    self.tik_instance.pipe_barrier("PIPE_V")
                    self.tik_instance.vcmpv_lt(
                        self.input_x_mask_ub[mask_loop * 255 * 8],
                        self.zeros_const_ub,
                        self.input_x_fp16_ub[batch_start_pos + mask_loop * 255 * 128],
                        255,
                        1, 1, 0, 8)
            with self.tik_instance.if_scope(mask_repeat > 0):
                self.tik_instance.pipe_barrier("PIPE_V")
                self.tik_instance.vcmpv_lt(
                    self.input_x_mask_ub[mask_loop_times * 255 * 8],
                    self.zeros_const_ub,
                    self.input_x_fp16_ub[batch_start_pos + mask_loop_times * 255 * 128],
                    mask_repeat,
                    1, 1, 0, 8)

            mask_real_length = self.already_load_l1_h_length * self.width_in
            mask_burst_length = self.already_load_l1_h_length * self.width_in // 16

            mask_hin_offset = 0
            self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
            with self.tik_instance.if_scope(mask_real_length % 16 != 0):
                self.mask_tail_process_handsync([batch_index + b_inner, cin1, self.input_x_mask_ub, mask_hin_offset,
                                                 mask_real_length, self.tem_ub, self.save_ub])
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(
                    self.output_reluv2_mask[batch_index + b_inner, cin1, mask_hin_offset, 0],
                    self.input_x_mask_ub,
                    0,
                    1, mask_burst_length,
                    0, 0)
            with self.tik_instance.if_scope(cin1 < (self.channel_in1 - 1)):
                self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 0)

    def mask_tail_process_handsync(self, input_list):
        """
        tail mask process to align 32Byte for last part of mask
        """
        # 尾块部分往外搬时会有内存踩踏，先搬不踩的，最后32bit要特殊处理
        batch_l1_offset, cin1_l1_offset, input_x_mask_ub, mask_hin_offset, \
        mask_real_length, tem_ub, save_ub = input_list

        self.tik_instance.data_move(
            self.output_reluv2_mask[batch_l1_offset, cin1_l1_offset, mask_hin_offset, 0],
            input_x_mask_ub,
            0,
            1, mask_real_length // 16,
            0, 0)
        # 最后剩余部分的数据先重排，再与前面ub中的数据凑成32bit往外搬
        dst_list = [tem_ub[16 * i] for i in range(16)]
        src_list = [input_x_mask_ub[(mask_real_length // 16 - 1) * 16]] * 16
        self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
        self.tik_instance.pipe_barrier("PIPE_V")
        self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 16, 1)
        self.tik_instance.pipe_barrier("PIPE_V")
        start_pos = mask_real_length % 16
        src_list = [tem_ub[(start_pos + i) * 16] for i in range(16)]
        dst_list = [save_ub[i * 16] for i in range(16)]

        self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

        total_offset = batch_l1_offset * (self.channel_in1 * self.height_in * self.width_in) + \
                       cin1_l1_offset * (self.height_in * self.width_in) + mask_hin_offset * self.width_in + \
                       mask_real_length - 16
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.data_move(
            self.output_reluv2_mask[total_offset],
            save_ub,
            0,
            1, 1,
            0, 0)

    def _init_l1_l0a_l0b_l0c_tensor(self):
        """
        init l1 l0a l0b l0c tensor
        """
        self.fmap_l1 = self.tik_instance.Tensor("float16",
                                                [self.b_inner_factor, self.channel_in1,
                                                 self.height_in,
                                                 self.width_in, self.channel_in0],
                                                scope=tik.scope_cbuf,
                                                name="fmap_l1")
        if self.filters_load_from_l1_flag:
            self.filters_l1 = self.tik_instance.Tensor(
                "float16",
                [self.k_l1a_factor * self.k_l0a_factor,
                 self.n_l1a_factor * self.n_l0a_factor,
                 self.channel_out0,
                 self.channel_in0],
                name="filters_l1",
                scope=tik.scope_cbuf)
            self.filters_l1_size = self.k_l1a_factor * self.k_l0a_factor * self.n_l1a_factor * \
                                   self.n_l0a_factor * self.channel_out0 * self.channel_in0 * 2
        else:
            self.filters_l1 = None
        self.fmap_l0a_ping = self.tik_instance.Tensor(dtype="float16",
                                                      shape=[self.b_inner_factor,
                                                             self.m_l0a_factor,
                                                             self.k_l0a_factor,
                                                             self.hw_out0,
                                                             self.channel_in0],
                                                      scope=tik.scope_ca,
                                                      name="fmap_l0a_ping")
        self.filters_l0b_ping = self.tik_instance.Tensor(dtype="float16",
                                                         shape=[
                                                             self.k_l0a_factor,
                                                             self.n_l0a_factor,
                                                             self.channel_out0,
                                                             self.channel_in0],
                                                         scope=tik.scope_cb,
                                                         name="filters_l0b_ping")
        self.fmap_l0c_ping = self.tik_instance.Tensor(dtype="float32",
                                                      shape=[
                                                          self.b_inner_factor,
                                                          self.n_l0a_factor,
                                                          self.m_l0a_factor,
                                                          self.hw_out0,
                                                          self.channel_out0],
                                                      scope=tik.scope_cc,
                                                      name="fmap_l0c_ping")
        self.fmap_l0a_pong = self.tik_instance.Tensor(dtype="float16",
                                                      shape=[self.b_inner_factor,
                                                             self.m_l0a_factor,
                                                             self.k_l0a_factor,
                                                             self.hw_out0,
                                                             self.channel_in0],
                                                      scope=tik.scope_ca,
                                                      name="fmap_l0a_pong")
        self.filters_l0b_pong = self.tik_instance.Tensor(dtype="float16",
                                                         shape=[
                                                             self.k_l0a_factor,
                                                             self.n_l0a_factor,
                                                             self.channel_out0,
                                                             self.channel_in0],
                                                         scope=tik.scope_cb,
                                                         name="filters_l0b_pong")
        self.fmap_l0c_pong = self.tik_instance.Tensor(dtype="float32",
                                                      shape=[
                                                          self.b_inner_factor,
                                                          self.n_l0a_factor,
                                                          self.m_l0a_factor,
                                                          self.hw_out0,
                                                          self.channel_out0],
                                                      scope=tik.scope_cc,
                                                      name="fmap_l0c_pong")

    def _init_all_ub_tensor(self):
        """
        init all ub tensor addr to avoid bank conflict
        """
        self._allocate_ub_in_area_one()
        self._allocate_ub_in_area_two()
        self._allocate_ub_in_area_three()
        self._allocate_ub_in_area_four()

    def _allocate_ub_in_area_four(self):
        """
        allocate ub in 192kB ~ 256kB
        """
        # area4
        ub_fm1_workbuf2_start_addr = 192 * 1024
        square_size = self.b_inner_factor * self.m_l0a_factor * self.hw_out0 * self.channel_out0
        self.ub_fm1_workbuf2 = self.tik_instance.Tensor(
            "float32",
            [square_size],
            name="ub_fm1_workbuf2",
            scope=tik.scope_ubuf,
            start_addr=ub_fm1_workbuf2_start_addr)
        fmap_f16_ub_pong_start_addr = self.get_start_addr(ub_fm1_workbuf2_start_addr, square_size * 4)
        self.fmap_f16_ub_pong = self.tik_instance.Tensor(
            "float16",
            [square_size],
            name="fmap_f16_ub_pong",
            scope=tik.scope_ubuf,
            start_addr=fmap_f16_ub_pong_start_addr)
        input_x_fp16_ub_pong_start_addr = self.get_start_addr(fmap_f16_ub_pong_start_addr, square_size * 2)
        self.input_x_fp16_ub_pong = self.tik_instance.Tensor(
            "float16",
            [(self.b_inner_factor * self.get_max_h_length() * self.width_in * self.channel_in0 + 127) // 128 * 128],
            name="input_x_fp16_ub_pong",
            scope=tik.scope_ubuf,
            start_addr=input_x_fp16_ub_pong_start_addr)
        tem_ub_start_addr = self.get_start_addr(input_x_fp16_ub_pong_start_addr,
                                                 (self.b_inner_factor * self.get_max_h_length() * self.width_in * \
                                                  self.channel_in0 + 127) // 128 * 128 * 2)
        self.tem_ub = self.tik_instance.Tensor("uint16", [512], name="temp_ub",
                                               scope=tik.scope_ubuf,
                                               start_addr=tem_ub_start_addr)
        tem_ub_pong_start_addr = self.get_start_addr(tem_ub_start_addr, 512 * 2)
        self.tem_ub_pong = self.tik_instance.Tensor("uint16", [512], name="temp_ub_pong",
                                                    scope=tik.scope_ubuf,
                                                    start_addr=tem_ub_pong_start_addr)

    def _allocate_ub_in_area_three(self):
        """
        allocate ub in 128kB ~ 196kB
        """
        # area 3
        scale_ub_64_start_addr = 128 * 1024
        square_size = self.b_inner_factor * self.m_l0a_factor * self.hw_out0 * self.channel_out0
        self.scale_ub_64 = self.tik_instance.Tensor("float32",
                                                    [4 * self.channel_in0 * self.channel_in1],
                                                    name="scale_ub_64",
                                                    scope=tik.scope_ubuf,
                                                    start_addr=scale_ub_64_start_addr)
        offset_ub_64_start_addr = self.get_start_addr(scale_ub_64_start_addr,
                                                       4 * self.channel_in0 * self.channel_in1 * 4)
        self.offset_ub_64 = self.tik_instance.Tensor("float32",
                                                     [4 * self.channel_in0 * self.channel_in1],
                                                     name="offset_ub_64",
                                                     scope=tik.scope_ubuf,
                                                     start_addr=offset_ub_64_start_addr)
        input_x_mask_ub_pong_start_addr = self.get_start_addr(offset_ub_64_start_addr,
                                                               4 * self.channel_in0 * self.channel_in1 * 4)
        self.input_x_mask_ub_pong = self.tik_instance.Tensor(
            "uint16",
            [(self.get_max_h_length() * self.width_in + 15) // 16 * 16],
            name="input_x_mask_ub_pong",
            scope=tik.scope_ubuf,
            start_addr=input_x_mask_ub_pong_start_addr)
        fmap_f16_ub_ping_start_addr = self.get_start_addr(input_x_mask_ub_pong_start_addr, (
            self.get_max_h_length() * self.width_in + 15) // 16 * 16 * 2)
        self.fmap_f16_ub_ping = self.tik_instance.Tensor(
            "float16",
            [square_size],
            name="fmap_f16_ub_ping",
            scope=tik.scope_ubuf,
            start_addr=fmap_f16_ub_ping_start_addr)
        input_x_fp16_ub_start_addr = self.get_start_addr(fmap_f16_ub_ping_start_addr, square_size * 2)
        self.input_x_fp16_ub = self.tik_instance.Tensor(
            "float16",
            [(self.b_inner_factor * (
                self.get_max_h_length()) * self.width_in * self.channel_in0 + 127) // 128 * 128],
            name="input_x_fp16_ub",
            scope=tik.scope_ubuf,
            start_addr=input_x_fp16_ub_start_addr)
        square_mul_ub_start_addr = self.get_start_addr(input_x_fp16_ub_start_addr,
                                                        (self.b_inner_factor * (self.get_max_h_length()) * \
                                                         self.width_in * self.channel_in0 + 127) // 128 * 128 * 2)
        self.square_mul_ub = self.tik_instance.Tensor(
            "float32",
            [square_size],
            name="square_mul_ub",
            scope=tik.scope_ubuf,
            start_addr=square_mul_ub_start_addr)

    def _allocate_ub_in_area_two(self):
        """
        allocate ub tensor in 64kb ~ 128kb
        """
        # area 2
        last_sum_output_ub_start_addr = 64 * 1024
        self.last_sum_output_ub = self.tik_instance.Tensor(
            "float32", self.channel_wise_shape_out,
            name="last_sum_output_ub",
            scope=tik.scope_ubuf,
            start_addr=last_sum_output_ub_start_addr)
        last_square_sum_output_ub_start_addr = self.get_start_addr(last_sum_output_ub_start_addr,
                                                                    self.channel_out1 * self.channel_out0 * 4)
        self.last_square_sum_output_ub = self.tik_instance.Tensor(
            "float32", self.channel_wise_shape_out,
            name="last_square_sum_output_ub",
            scope=tik.scope_ubuf,
            start_addr=last_square_sum_output_ub_start_addr)
        zeros_const_ub_start_addr = self.get_start_addr(last_square_sum_output_ub_start_addr,
                                                         self.channel_out1 * self.channel_out0 * 4)
        self.zeros_const_ub = self.tik_instance.Tensor(
            "float16",
            [128],
            name="zeros_const_ub_128",
            scope=tik.scope_ubuf,
            start_addr=zeros_const_ub_start_addr)
        # 这个可以和上面的zeros_const_ub_128，共用地址，不需要初始化
        self.zeros_const_ub_16 = self.tik_instance.Tensor(
            "float16",
            [16],
            name="zeros_const_ub_16",
            scope=tik.scope_ubuf,
            start_addr=zeros_const_ub_start_addr)
        input_x_fp32_ub_start_addr = self.get_start_addr(zeros_const_ub_start_addr, 128 * 2)
        self.input_x_fp32_ub = self.tik_instance.Tensor(
            "float32",
            [self.b_inner_factor,
             self.get_max_h_length(),
             self.width_in,
             self.channel_in0],
            name="input_x_fp32_ub",
            scope=tik.scope_ubuf,
            start_addr=input_x_fp32_ub_start_addr)
        save_ub_start_addr = self.get_start_addr(input_x_fp32_ub_start_addr,
                                                  self.b_inner_factor * self.get_max_h_length() * \
                                                  self.width_in * self.channel_in0 * 4)
        self.save_ub = self.tik_instance.Tensor("uint16", [16 * 16], tik.scope_ubuf,
                                                name="save_last_inub",
                                                start_addr=save_ub_start_addr)
        save_ub_pong_start_addr = self.get_start_addr(save_ub_start_addr, 16 * 16 * 2)
        self.save_ub_pong = self.tik_instance.Tensor("uint16", [16 * 16], tik.scope_ubuf,
                                                     name="save_last_inub_pong",
                                                     start_addr=save_ub_pong_start_addr)

    def _allocate_ub_in_area_one(self):
        """
        allocate ub in 0kB~64kB
        """
        # area 1
        elem_num = self.channel_in1 * self.channel_in0
        self.y_scale_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="y_scale_ub",
            scope=tik.scope_ubuf,
            start_addr=0)
        y_offset_ub_start_addr = self.get_start_addr(0, elem_num * 4)
        self.y_offset_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="y_offset_ub",
            scope=tik.scope_ubuf,
            start_addr=y_offset_ub_start_addr)
        sum_input_ub_start_addr = self.get_start_addr(y_offset_ub_start_addr, elem_num * 4)
        self.sum_input_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="sum_ub",
            scope=tik.scope_ubuf,
            start_addr=sum_input_ub_start_addr)
        square_sum_input_ub_start_addr = self.get_start_addr(sum_input_ub_start_addr, elem_num * 4)
        self.square_sum_input_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="square_sum_input_ub",
            scope=tik.scope_ubuf,
            start_addr=square_sum_input_ub_start_addr)
        y_scale_sqrt_ub_start_addr = self._allocate_ub_in_area_one_sub(elem_num, square_sum_input_ub_start_addr)
        self._fetch_allocate_ub_in_area_one(elem_num, y_scale_sqrt_ub_start_addr)
        self._override_used_ub()

    def _fetch_allocate_ub_in_area_one(self, elem_num, y_scale_sqrt_ub_start_addr):
        """
        fetch allocate ub in area one
        """
        offset_ub_start_addr = self.get_start_addr(y_scale_sqrt_ub_start_addr, elem_num * 4)
        self.offset_ub = self.tik_instance.Tensor("float32", [elem_num], name="offset_ub",
                                                  scope=tik.scope_ubuf, start_addr=offset_ub_start_addr)
        y_offset_mul_ub_start_addr = self.get_start_addr(offset_ub_start_addr, elem_num * 4)
        self.y_offset_mul_ub = self.tik_instance.Tensor("float32", [elem_num], name="y_offset_mul_ub",
                                                        scope=tik.scope_ubuf, start_addr=y_offset_mul_ub_start_addr)
        pre_moving_mean_ub_start_addr = self.get_start_addr(y_offset_mul_ub_start_addr, elem_num * 4)
        self.pre_moving_mean_ub = self.tik_instance.Tensor("float32", [elem_num], name="pre_moving_mean_ub",
                                                           scope=tik.scope_ubuf,
                                                           start_addr=pre_moving_mean_ub_start_addr)
        mean_mul_ub_start_addr = self.get_start_addr(pre_moving_mean_ub_start_addr, elem_num * 4)
        self.mean_mul_ub = self.tik_instance.Tensor("float32", [elem_num], name="mean_mul_ub",
                                                    scope=tik.scope_ubuf, start_addr=mean_mul_ub_start_addr)
        pre_moving_variance_ub_start_addr = self.get_start_addr(mean_mul_ub_start_addr, elem_num * 4)
        self.pre_moving_variance_ub = self.tik_instance.Tensor("float32", [elem_num], name="pre_moving_variance_ub",
                                                               scope=tik.scope_ubuf,
                                                               start_addr=pre_moving_variance_ub_start_addr)
        mean_mul_rev_ub_start_addr = self.get_start_addr(pre_moving_variance_ub_start_addr, elem_num * 4)
        self.mean_mul_rev_ub = self.tik_instance.Tensor("float32", [elem_num], name="mean_mul_rev_ub",
                                                        scope=tik.scope_ubuf, start_addr=mean_mul_rev_ub_start_addr)
        moving_mean_ub_start_addr = self.get_start_addr(mean_mul_rev_ub_start_addr, elem_num * 4)
        self.moving_mean_ub = self.tik_instance.Tensor("float32", [elem_num], name="moving_mean_ub",
                                                       scope=tik.scope_ubuf, start_addr=moving_mean_ub_start_addr)
        variance_batch_ub_start_addr = self.get_start_addr(moving_mean_ub_start_addr, elem_num * 4)
        self.variance_batch_ub = self.tik_instance.Tensor("float32", [elem_num], name="variance_batch_ub",
                                                          scope=tik.scope_ubuf, start_addr=variance_batch_ub_start_addr)
        variance_mul_ub_start_addr = self.get_start_addr(variance_batch_ub_start_addr, elem_num * 4)
        self.variance_mul_ub = self.tik_instance.Tensor("float32", [elem_num], name="variance_mul_ub",
                                                        scope=tik.scope_ubuf, start_addr=variance_mul_ub_start_addr)
        variance_mul_rev_ub_start_addr = self.get_start_addr(variance_mul_ub_start_addr, elem_num * 4)
        self.variance_mul_rev_ub = self.tik_instance.Tensor("float32", [elem_num], name="variance_mul_rev_ub",
                                                            scope=tik.scope_ubuf,
                                                            start_addr=variance_mul_rev_ub_start_addr)
        moving_variance_ub_start_addr = self.get_start_addr(variance_mul_rev_ub_start_addr, elem_num * 4)
        self.moving_variance_ub = self.tik_instance.Tensor("float32", [elem_num], name="moving_variance_ub",
                                                           scope=tik.scope_ubuf,
                                                           start_addr=moving_variance_ub_start_addr)

    def _override_used_ub(self):
        """
        override the used ub
        """
        # 上面的参数只在bn2_mean_varience_process当中使用,在后面不会用到，下面的地址将覆盖上面的
        self.input_x_mask_ub = self.tik_instance.Tensor(
            "uint16",
            [((self.get_max_h_length()) * self.width_in + 15) // 16 * 16],
            name="input_x_mask_ub",
            scope=tik.scope_ubuf,
            start_addr=0)
        input_x_ub_start_addr = self.get_start_addr(0,
                                                     ((self.get_max_h_length()) * self.width_in + 15) // 16 * 16 * 2)
        self.input_x_ub = self.tik_instance.Tensor(
            "float16",
            [self.b_inner_factor * (self.get_max_h_length()) * self.width_in * self.channel_in0],
            name="input_x_ub",
            scope=tik.scope_ubuf,
            start_addr=input_x_ub_start_addr)
        square_size = self.b_inner_factor * self.m_l0a_factor * self.hw_out0 * self.channel_out0
        fmap_ub_start_addr = self.get_start_addr(input_x_ub_start_addr,
                                                  self.b_inner_factor * (self.get_max_h_length()) * \
                                                  self.width_in * self.channel_in0 * 2)
        self.fmap_ub = self.tik_instance.Tensor(
            "float32",
            [square_size],
            name="fmap_ub",
            scope=tik.scope_ubuf,
            start_addr=fmap_ub_start_addr)
        output_sum_inub_start_addr = self.get_start_addr(fmap_ub_start_addr, square_size * 4)
        self.output_sum_inub = self.tik_instance.Tensor(dtype="float32",
                                                        shape=[1, self.channel_out1, 1, 1, self.channel_out0 * 4],
                                                        scope=tik.scope_ubuf,
                                                        name="output_sum_gm_inub",
                                                        start_addr=output_sum_inub_start_addr)
        output_square_sum_inub_start_addr = self.get_start_addr(output_sum_inub_start_addr,
                                                                 self.channel_out1 * self.channel_out0 * 4 * 4)
        self.output_square_sum_inub = self.tik_instance.Tensor(dtype="float32",
                                                               shape=[1, self.channel_out1, 1, 1,
                                                                      self.channel_out0 * 4],
                                                               scope=tik.scope_ubuf,
                                                               name="output_square_sum_gm_inub",
                                                               start_addr=output_square_sum_inub_start_addr)

    def _init_sum_and_square_sum(self):
        """
        init sum and square_sum with zero
        """
        mask_loop = self.channel_out1 * self.channel_out0 * 4 // (64 * 255)
        mask_repeat = (self.channel_out1 * self.channel_out0 * 4 % (64 * 255)) // 64
        mask_redundant = (self.channel_out1 * self.channel_out0 * 4 % (64 * 255)) % 64
        self.tik_instance.pipe_barrier("PIPE_V")
        if mask_loop > 0:
            with self.tik_instance.for_range(0, mask_loop, name='mask_loop') as loop_index:
                self.tik_instance.vec_dup(64,
                                          self.output_sum_inub[loop_index * 255 * 64],
                                          0,
                                          255,
                                          8)
                self.tik_instance.vec_dup(64,
                                          self.output_square_sum_inub[loop_index * 255 * 64],
                                          0,
                                          255,
                                          8)
        if mask_repeat > 0:
            self.tik_instance.vec_dup(64,
                                      self.output_sum_inub[mask_loop * 255 * 64],
                                      0,
                                      mask_repeat,
                                      8)
            self.tik_instance.vec_dup(64,
                                      self.output_square_sum_inub[mask_loop * 255 * 64],
                                      0,
                                      mask_repeat,
                                      8)
        if mask_redundant > 0:
            self.tik_instance.vec_dup(mask_redundant,
                                      self.output_sum_inub[mask_loop * 255 * 64 + mask_repeat * 64],
                                      0,
                                      1,
                                      1)
            self.tik_instance.vec_dup(mask_redundant,
                                      self.output_square_sum_inub[mask_loop * 255 * 64 + mask_repeat * 64],
                                      0,
                                      1,
                                      1)
        self.tik_instance.pipe_barrier("PIPE_V")

    def _bn2_and_bn1_merge(self, axises):
        """
        bn_update and bn_reduce in conv2d process
        """
        k_redundant = axises["k_redundant"]
        m_redundant = axises["m_redundant"]
        n_l1a = axises["n_l1a"]
        n_redundant = axises['n_redundant']
        with self.tik_instance.if_scope(tik.all(n_l1a == 0, n_redundant == 0)):
            with self.tik_instance.if_scope(k_redundant == 0):
                self.next_hin_l1_start_pos.set_as(self.preload_next_hin_l1_start_pos)
            with self.tik_instance.if_scope(tik.any(m_redundant != (self.m_redundant_factor - 1),
                                                    self.already_load_l1_h_length < self.height_in)):
                self.bn_update_process_handsync(axises)
        with self.tik_instance.if_scope((k_redundant + m_redundant) != (self.k_redundant_factor +
                                                                        self.m_redundant_factor - 2)):
            self.tik_instance.set_flag("PIPE_MTE3", "PIPE_MTE1", 0)
        with self.tik_instance.if_scope(tik.any(n_redundant != 0,
                                                n_l1a != 0,
                                                m_redundant != 0)):
            self.bn_reduce_process_handsync(axises)
        with self.tik_instance.if_scope(k_redundant + 1 == self.k_redundant_factor):
            self.l0c_pingpong.set_as(1 - self.l0c_pingpong)
            self.front_m.set_as(m_redundant)
            self.front_n_l1.set_as(n_l1a)
            self.front_batch.set_as(
                axises["b_block"] * self.b_redundant_factor * self.b_inner_factor +
                axises["pre_core_process_batch"] * self.b_inner_factor)
            self.front_n_redundant.set_as(n_redundant)

    def _full_conv_res_bn_reduce_process(self, k_redundant, n_l0):
        """
        bn reduce process for full conv res
        """
        square_size = self.b_inner_factor * self.m_l0a_factor * self.hw_out0 * self.channel_out0
        with self.tik_instance.if_scope(self.l0c_pingpong == 1):
            with self.tik_instance.if_scope(k_redundant == 0):
                self.tik_instance.wait_flag("PIPE_M", "PIPE_V", 0)
            self.tik_instance.data_move(
                self.fmap_ub,
                self.fmap_l0c_ping[0, n_l0, 0, 0, 0],
                0,
                self.b_inner_factor,
                self.m_l0a_factor,
                (self.n_l0a_factor - 1) * self.m_l0a_factor,
                0)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(k_redundant == 0):
                self.tik_instance.wait_flag("PIPE_M", "PIPE_V", 1)
            self.tik_instance.data_move(
                self.fmap_ub,
                self.fmap_l0c_pong[0, n_l0, 0, 0, 0],
                0,
                self.b_inner_factor,
                self.m_l0a_factor,
                (self.n_l0a_factor - 1) * self.m_l0a_factor,
                0)
        fp32_loop_times, fp32_redundant, fp32_repeat_times = self.calu_loop_time(square_size)
        dst_b_offset, dst_m_offset, dst_n_offset = self.calc_dst_b_n_m_offset(m_l0=0, n_l0=n_l0)
        with self.tik_instance.if_scope(self.bn_reduce_ub_output_pingpong == 0):
            # mv ub to gm
            self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 2)
            self.tik_instance.pipe_barrier("PIPE_V")
            self.res_fmap_convertf16(self.fmap_f16_ub_ping, self.fmap_ub, {"fp32_loop_times": fp32_loop_times,
                                                                           "fp32_redundant": fp32_redundant,
                                                                           "fp32_repeat_times": fp32_repeat_times})
            self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 2)
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 2)
            self.tik_instance.data_move(
                self.output_convolution[dst_b_offset, dst_n_offset, dst_m_offset, 0],
                self.fmap_f16_ub_ping,
                0,
                self.b_inner_factor,
                self.m_l0a_factor * self.hw_out0,
                0,
                self.channel_out1 * self.height_out *
                self.width_out - self.m_l0a_factor * self.hw_out0)
            self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 2)
        with self.tik_instance.else_scope():
            # mv ub to gm
            self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 3)
            self.tik_instance.pipe_barrier("PIPE_V")
            self.res_fmap_convertf16(self.fmap_f16_ub_pong, self.fmap_ub, {"fp32_loop_times": fp32_loop_times,
                                                                           "fp32_redundant": fp32_redundant,
                                                                           "fp32_repeat_times": fp32_repeat_times})
            self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 3)
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 3)
            self.tik_instance.data_move(
                self.output_convolution[dst_b_offset, dst_n_offset, dst_m_offset, 0],
                self.fmap_f16_ub_pong,
                0,
                self.b_inner_factor,
                self.m_l0a_factor * self.hw_out0,
                0,
                self.channel_out1 * self.height_out *
                self.width_out - self.m_l0a_factor * self.hw_out0)
            self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 3)
        self.bn_reduce_ub_output_pingpong.set_as(1 - self.bn_reduce_ub_output_pingpong)
        # square sum
        self.calu_square_sum(self.fmap_ub, fp32_loop_times, fp32_redundant, fp32_repeat_times)
        self.tik_instance.pipe_barrier("PIPE_V")
        self.dichotomy_sum_and_square_sum_handsync(self.fmap_ub, self.square_mul_ub,
                                                   square_size, dst_n_offset)

    def _tail_bn_reduce_process_handsync(self, k_redundant, n_l0):
        """
        process the tail of bn_reduce
        """
        with self.tik_instance.if_scope(self.l0c_pingpong == 1):
            with self.tik_instance.if_scope(k_redundant == 0):
                self.tik_instance.wait_flag("PIPE_M", "PIPE_V", 0)
            self.tik_instance.data_move(self.fmap_ub, self.fmap_l0c_ping[0, n_l0, 0, 0, 0], 0, self.b_inner_factor,
                                        self.m_l0a_factor, (self.n_l0a_factor - 1) * self.m_l0a_factor, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(k_redundant == 0):
                self.tik_instance.wait_flag("PIPE_M", "PIPE_V", 1)
            self.tik_instance.data_move(self.fmap_ub, self.fmap_l0c_pong[0, n_l0, 0, 0, 0], 0, self.b_inner_factor,
                                        self.m_l0a_factor, (self.n_l0a_factor - 1) * self.m_l0a_factor, 0)
        m_tail_size = self.height_out * self.width_out - \
                      (self.m_redundant_factor - 1) * self.m_l0a_factor * self.hw_out0
        tail_square_size = self.b_inner_factor * m_tail_size * self.channel_out0
        dst_n_offset, fp32_loop_times, fp32_redundant, fp32_repeat_times, res_conv = self._move_conv_res_to_ddr(
            m_tail_size, n_l0, tail_square_size)
        self.bn_reduce_ub_output_pingpong.set_as(1 - self.bn_reduce_ub_output_pingpong)
        # square sum
        self.calu_square_sum(res_conv, fp32_loop_times, fp32_redundant, fp32_repeat_times)
        self.tik_instance.pipe_barrier("PIPE_V")
        if self.b_inner_factor == 2:
            for i in range(self.b_inner_factor):
                self.dichotomy_sum_and_square_sum_handsync(res_conv, self.square_mul_ub,
                                                           tail_square_size // self.b_inner_factor, dst_n_offset)
                if i + 1 < self.b_inner_factor:
                    self.tik_instance.data_move(res_conv, res_conv[tail_square_size // self.b_inner_factor],
                                                0, 1, tail_square_size // self.b_inner_factor // 8, 0, 0)
                    self.tik_instance.data_move(self.square_mul_ub,
                                                self.square_mul_ub[tail_square_size // self.b_inner_factor],
                                                0, 1, tail_square_size // self.b_inner_factor // 8, 0, 0)
        else:
            self.dichotomy_sum_and_square_sum_handsync(res_conv, self.square_mul_ub, tail_square_size, dst_n_offset)

    def _move_conv_res_to_ddr(self, m_tail_size, n_l0, tail_square_size):
        """
        move conv2d res to ddr
        """
        # inner batch 场景需要剔除脏数据
        res_conv = self.fmap_ub
        fp32_loop_times, fp32_redundant, fp32_repeat_times = self.calu_loop_time(tail_square_size)
        dst_b_offset, dst_m_offset, dst_n_offset = self.calc_dst_b_n_m_offset(m_l0=0, n_l0=n_l0)
        with self.tik_instance.if_scope(self.bn_reduce_ub_output_pingpong == 0):
            # mv ub to gm
            self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 2)
            self.tik_instance.pipe_barrier("PIPE_V")
            self.res_fmap_convertf16(self.fmap_f16_ub_ping, res_conv, {"fp32_loop_times": fp32_loop_times,
                                                                       "fp32_redundant": fp32_redundant,
                                                                       "fp32_repeat_times": fp32_repeat_times})
            self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 2)
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 2)
            self.tik_instance.data_move(
                self.output_convolution[dst_b_offset, dst_n_offset, dst_m_offset, 0], self.fmap_f16_ub_ping, 0,
                self.b_inner_factor, m_tail_size, 0, self.channel_out1 * self.height_out * self.width_out - m_tail_size)
            self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 2)
        with self.tik_instance.else_scope():
            # mv ub to gm
            self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 3)
            self.tik_instance.pipe_barrier("PIPE_V")
            self.res_fmap_convertf16(self.fmap_f16_ub_pong, res_conv, {"fp32_loop_times": fp32_loop_times,
                                                                       "fp32_redundant": fp32_redundant,
                                                                       "fp32_repeat_times": fp32_repeat_times})
            self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 3)
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 3)
            self.tik_instance.data_move(self.output_convolution[dst_b_offset, dst_n_offset, dst_m_offset, 0],
                                        self.fmap_f16_ub_pong, 0, self.b_inner_factor, m_tail_size, 0,
                                        self.channel_out1 * self.height_out *
                                        self.width_out - m_tail_size)
            self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 3)
        return dst_n_offset, fp32_loop_times, fp32_redundant, fp32_repeat_times, res_conv

    def _bn_update_in_conv_single_chanel_process(self, axises, cur_cin1):
        """
        single chanel process for bn update in conv
        """
        batch_index = self.next_batch
        self.fm_ddr2ub_burst_length.set_as(self.preload_fm_ddr2ub_burst_length)
        self.fm_ddr2ub_src_stride.set_as(self.preload_fm_ddr2ub_src_stride)
        fm_ddr2ub_nburst = self.b_inner_factor
        fp32_loop_times = self.tik_instance.Scalar(name="fp32_loop_times",
                                                   init_value=self.this_time_calu_length * self.width_in * \
                                                              self.channel_in0 // (64 * 255))
        fp32_loop_redundant = self.tik_instance.Scalar(name="fp32_loop_redundant",
                                                       init_value=self.this_time_calu_length *
                                                       self.width_in * self.channel_in0 % (64 * 255))
        fp32_repeat = self.tik_instance.Scalar(name="fp32_repeat", init_value=fp32_loop_redundant // 64)
        fp32_redundant = self.tik_instance.Scalar(name="fp32_redundant",
                                                  init_value=fp32_loop_redundant % 64)
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 0)
        self.tik_instance.pipe_barrier("PIPE_V")
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(0, fp32_loop_times, name="fp32_loop") as fp32_loop:
                self.tik_instance.vconv(
                    64,
                    "none",
                    self.input_x_fp32_ub[fp32_loop * 255 * 64],
                    self.input_x_ub[fp32_loop * 255 * 64],
                    255,
                    1, 1, 8, 4)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vconv(
                64,
                "none",
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                self.input_x_ub[fp32_loop_times * 255 * 64],
                fp32_repeat,
                1, 1, 8, 4)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vconv(
                fp32_redundant,
                "none",
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                self.input_x_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                1,
                1, 1, 8, 4)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE2", 1)
        self.tik_instance.pipe_barrier("PIPE_V")
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(
                    0, fp32_loop_times, name="loop_time") as loop_time:
                self.tik_instance.vmaddrelu(64, self.input_x_fp32_ub[loop_time * 255 * 64],
                                            self.scale_ub_64[cur_cin1 * 64],
                                            self.offset_ub_64[cur_cin1 * 64], 255, 1, 1, 1, 8, 0, 0)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vmaddrelu(64, self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                                        self.scale_ub_64[cur_cin1 * 64],
                                        self.offset_ub_64[cur_cin1 * 64], fp32_repeat, 1, 1, 1, 8, 0, 0)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vmaddrelu(fp32_redundant,
                                        self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                                        self.scale_ub_64[cur_cin1 * 64],
                                        self.offset_ub_64[cur_cin1 * 64], 1, 1, 1, 1, 8, 0, 0)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
        with self.tik_instance.if_scope(self.bn_update_lastub_output_pingpong == 0):
            self._bn_update_process_ping(batch_index, cur_cin1, fm_ddr2ub_nburst, {"fp32_loop_times": fp32_loop_times,
                                                                                   "fp32_redundant": fp32_redundant,
                                                                                   "fp32_repeat": fp32_repeat})
        with self.tik_instance.else_scope():
            self._bn_update_process_pong(batch_index, cur_cin1, fm_ddr2ub_nburst, {"fp32_loop_times": fp32_loop_times,
                                                                                   "fp32_redundant": fp32_redundant,
                                                                                   "fp32_repeat": fp32_repeat})
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE2", 1)
        self.bn_update_lastub_output_pingpong.set_as(1 - self.bn_update_lastub_output_pingpong)
        self._preload_next_input_x_in_conv(axises, cur_cin1, fm_ddr2ub_nburst)

    def _preload_next_input_x_in_conv(self, axises, cur_cin1, fm_ddr2ub_nburst):
        """
        preload input x in conv process
        """
        with self.tik_instance.if_scope(cur_cin1 + 1 == self.channel_in1):
            self.already_load_l1_h_length.set_as(self.preload_already_load_l1_h_length)
            m_redundant = axises['m_redundant']
            with self.tik_instance.if_scope(self.already_load_l1_h_length >= self.height_in):
                # 下一个batch的
                pass
            with self.tik_instance.else_scope():
                next_ho_index = self.tik_instance.Scalar(name="next_ho_index", init_value=0)
                with self.tik_instance.if_scope((m_redundant + 2) <= self.m_redundant_factor):
                    next_ho_index.set_as((m_redundant + 2) * self.m_l0a_factor * 16 // self.width_out)
                next_hin_index = next_ho_index * self.stride_h
                self.preload_next_hin_l1_start_pos.set_as(next_hin_index - self.pad_t)
                self.preload_this_time_calu_length.set_as(self.preload_next_hin_l1_start_pos +
                                                          self.get_conv_h_length() -
                                                          self.preload_already_load_l1_h_length)

                with self.tik_instance.if_scope((self.preload_already_load_l1_h_length +
                                                 self.preload_this_time_calu_length) > self.height_in):
                    self.preload_this_time_calu_length.set_as(
                        self.height_in - self.preload_already_load_l1_h_length)
                self.preload_already_load_l1_h_length.set_as(self.preload_already_load_l1_h_length +
                                                             self.preload_this_time_calu_length)
                self.preload_fm_ddr2ub_burst_length.set_as(
                    self.preload_this_time_calu_length * self.width_in)
                self.preload_fm_ddr2ub_src_stride.set_as(
                    self.channel_in1 * self.height_in * self.width_in - \
                    self.preload_this_time_calu_length * self.width_in)

                self.tik_instance.data_move(
                    self.input_x_ub,
                    self.fmap_input[self.preload_next_batch, 0,
                                    self.already_load_l1_h_length, 0, 0],
                    0,
                    self.b_inner_factor,
                    self.preload_fm_ddr2ub_burst_length,
                    self.preload_fm_ddr2ub_src_stride, 0)
                self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.input_x_ub,
                self.fmap_input[self.preload_next_batch, cur_cin1 + 1,
                                self.already_load_l1_h_length, 0, 0],
                0,
                fm_ddr2ub_nburst, self.preload_fm_ddr2ub_burst_length,
                self.preload_fm_ddr2ub_src_stride, 0)
            self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)

    def _bn_update_process_ping(self, batch_index, cur_cin1, fm_ddr2ub_nburst, loop_time):
        """
        ping of bn update process
        """
        fp32_loop_times = loop_time.get("fp32_loop_times", 0)
        fp32_redundant = loop_time.get("fp32_redundant", 0)
        fp32_repeat = loop_time.get("fp32_repeat", 0)
        self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 0)
        self.tik_instance.pipe_barrier("PIPE_V")
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(
                    0, fp32_loop_times, name="fp32_loop") as fp32_loop:
                self.tik_instance.vconv(
                    64,
                    "none",
                    self.input_x_fp16_ub[fp32_loop * 255 * 64],
                    self.input_x_fp32_ub[fp32_loop * 255 * 64],
                    255,
                    1, 1, 4, 8)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vconv(
                64,
                "none",
                self.input_x_fp16_ub[fp32_loop_times * 255 * 64],
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                fp32_repeat,
                1, 1, 4, 8)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vconv(
                fp32_redundant,
                "none",
                self.input_x_fp16_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                1,
                1, 1, 4, 8)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
        # 将数据搬到L1
        self.tik_instance.data_move(
            self.fmap_l1[0, cur_cin1, self.already_load_l1_h_length, 0, 0],
            self.input_x_fp16_ub,
            0,
            fm_ddr2ub_nburst, self.fm_ddr2ub_burst_length,
            0, self.fm_ddr2ub_src_stride)
        self.tik_instance.pipe_barrier("PIPE_MTE3")
        self.tik_instance.data_move(
            self.output_relu[batch_index, cur_cin1, self.already_load_l1_h_length, 0, 0],
            self.input_x_fp16_ub,
            0,
            fm_ddr2ub_nburst, self.fm_ddr2ub_burst_length, 0, self.fm_ddr2ub_src_stride)
        self._calc_reluv2_mask(batch_index, cur_cin1)

    def _calc_reluv2_mask(self, batch_index, cur_cin1):
        """
        calu mask
        """
        # 输出mask外提可以减少scala 操作
        with self.tik_instance.for_range(0, self.b_inner_factor, name="b_inner") as b_inner:
            mask_loop_times = self.tik_instance.Scalar(name="mask_loop_times",
                                                       init_value=(self.this_time_calu_length *
                                                                   self.width_in * self.channel_in0) // \
                                                                  (128 * 255))
            mask_loop_redundant = self.tik_instance.Scalar(name="mask_loop_redundant",
                                                           init_value=self.this_time_calu_length * \
                                                                      self.width_in * self.channel_in0 % \
                                                                      (128 * 255))
            mask_repeat = self.tik_instance.Scalar(name="mask_repeat",
                                                   init_value=(mask_loop_redundant + 127) // 128)

            batch_start_pos = self.tik_instance.Scalar(name="batch_start_pos",
                                                       init_value=b_inner * self.this_time_calu_length *
                                                       self.width_in * self.channel_in0)
            self.tik_instance.pipe_barrier("PIPE_V")
            with self.tik_instance.if_scope(mask_loop_times > 0):
                with self.tik_instance.for_range(0, mask_loop_times, name="mask_loop") as mask_loop:
                    self.tik_instance.vcmpv_lt(
                        self.input_x_mask_ub[mask_loop * 255 * 8],
                        self.zeros_const_ub,
                        self.input_x_fp16_ub[batch_start_pos + mask_loop * 255 * 128],
                        255,
                        1, 1, 0, 8)
            with self.tik_instance.if_scope(mask_repeat > 0):
                self.tik_instance.vcmpv_lt(
                    self.input_x_mask_ub[mask_loop_times * 255 * 8],
                    self.zeros_const_ub,
                    self.input_x_fp16_ub[batch_start_pos + mask_loop_times * 255 * 128],
                    mask_repeat,
                    1, 1, 0, 8)
            self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
            mask_hin_offset = self.already_load_l1_h_length
            mask_real_length = self.tik_instance.Scalar(name="mask_real_length",
                                                        init_value=self.this_time_calu_length * self.width_in)
            mask_burst_length = self.tik_instance.Scalar(name="mask_burst_length",
                                                         init_value=mask_real_length // 16)
            with self.tik_instance.if_scope(mask_real_length % 16 != 0):
                self.mask_tail_process_handsync([batch_index + b_inner, cur_cin1, self.input_x_mask_ub,
                                                 mask_hin_offset, mask_real_length,
                                                 self.tem_ub, self.save_ub])
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(
                    self.output_reluv2_mask[batch_index + b_inner, cur_cin1, mask_hin_offset, 0],
                    self.input_x_mask_ub,
                    0,
                    1, mask_burst_length,
                    0, 0)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 0)

    def _bn_update_process_pong(self, batch_index, cur_cin1, fm_ddr2ub_nburst, loop_time):
        """
        the pong of bn_update process
        """
        fp32_loop_times = loop_time.get("fp32_loop_times", 0)
        fp32_redundant = loop_time.get("fp32_redundant", 0)
        fp32_repeat = loop_time.get("fp32_repeat", 0)
        self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 1)
        self.tik_instance.pipe_barrier("PIPE_V")
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(
                    0, fp32_loop_times, name="fp32_loop") as fp32_loop:
                self.tik_instance.vconv(
                    64,
                    "none",
                    self.input_x_fp16_ub_pong[fp32_loop * 255 * 64],
                    self.input_x_fp32_ub[fp32_loop * 255 * 64],
                    255,
                    1, 1, 4, 8)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vconv(
                64,
                "none",
                self.input_x_fp16_ub_pong[fp32_loop_times * 255 * 64],
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                fp32_repeat,
                1, 1, 4, 8)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vconv(
                fp32_redundant,
                "none",
                self.input_x_fp16_ub_pong[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                1,
                1, 1, 4, 8)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 1)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 1)
        # 将数据搬到L1
        self.tik_instance.data_move(
            self.fmap_l1[0, cur_cin1, self.already_load_l1_h_length, 0, 0],
            self.input_x_fp16_ub_pong,
            0,
            fm_ddr2ub_nburst, self.fm_ddr2ub_burst_length,
            0, self.fm_ddr2ub_src_stride)
        self.tik_instance.pipe_barrier("PIPE_MTE3")
        self.tik_instance.data_move(
            self.output_relu[batch_index, cur_cin1, self.already_load_l1_h_length, 0, 0],
            self.input_x_fp16_ub_pong,
            0,
            fm_ddr2ub_nburst, self.fm_ddr2ub_burst_length, 0, self.fm_ddr2ub_src_stride)
        self._calu_reluv2_mask_pong(batch_index, cur_cin1)

    def _calu_reluv2_mask_pong(self, batch_index, cur_cin1):
        """
        calculate mask
        """
        # 输出mask外提可以减少scala 操作
        with self.tik_instance.for_range(0, self.b_inner_factor, name="b_inner") as b_inner:
            self.tik_instance.pipe_barrier("PIPE_V")
            mask_loop_times = self.tik_instance.Scalar(name="mask_loop_times",
                                                       init_value=self.this_time_calu_length * \
                                                                  self.width_in * self.channel_in0 // \
                                                                  (128 * 255))
            mask_loop_redundant = self.tik_instance.Scalar(name="mask_loop_redundant",
                                                           init_value=self.this_time_calu_length * \
                                                                      self.width_in * \
                                                                      self.channel_in0 % (128 * 255))
            mask_repeat = self.tik_instance.Scalar(name="mask_repeat",
                                                   init_value=(mask_loop_redundant + 127) // 128)

            batch_start_pos = self.tik_instance.Scalar(name="batch_start_pos",
                                                       init_value=b_inner * self.this_time_calu_length *
                                                       self.width_in * self.channel_in0)
            with self.tik_instance.if_scope(mask_loop_times > 0):
                with self.tik_instance.for_range(0, mask_loop_times, name="mask_loop") as mask_loop:
                    self.tik_instance.vcmpv_lt(
                        self.input_x_mask_ub_pong[mask_loop * 255 * 8],
                        self.zeros_const_ub,
                        self.input_x_fp16_ub_pong[batch_start_pos + mask_loop * 255 * 128],
                        255,
                        1, 1, 0, 8)
            with self.tik_instance.if_scope(mask_repeat > 0):
                self.tik_instance.vcmpv_lt(
                    self.input_x_mask_ub_pong[mask_loop_times * 255 * 8],
                    self.zeros_const_ub,
                    self.input_x_fp16_ub_pong[batch_start_pos + mask_loop_times * 255 * 128],
                    mask_repeat,
                    1, 1, 0, 8)
            self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 1)
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 1)
            mask_hin_offset = self.already_load_l1_h_length
            mask_real_length = self.tik_instance.Scalar(name="mask_real_length",
                                                        init_value=self.this_time_calu_length * \
                                                                   self.width_in)
            mask_burst_length = self.tik_instance.Scalar(name="mask_burst_length",
                                                         init_value=mask_real_length // 16)
            with self.tik_instance.if_scope(mask_real_length % 16 != 0):
                self.mask_tail_process_handsync([batch_index + b_inner, cur_cin1,
                                                 self.input_x_mask_ub_pong,
                                                 mask_hin_offset, mask_real_length,
                                                 self.tem_ub_pong, self.save_ub_pong])
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(
                    self.output_reluv2_mask[batch_index + b_inner, cur_cin1, mask_hin_offset, 0],
                    self.input_x_mask_ub_pong,
                    0,
                    1, mask_burst_length,
                    0, 0)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 1)

    def _mmad_handsync_after(self, axises, k_redundant):
        """
        instr sync after mmad
        """
        with self.tik_instance.if_scope(tik.all((axises["m_redundant"] == self.m_redundant_factor - 1),
                                                k_redundant == self.k_redundant_factor - 1)):
            self.tik_instance.set_flag("PIPE_M", "PIPE_MTE2", 3)
        with self.tik_instance.if_scope(self.l0a_pingpong == 0):
            self.tik_instance.set_flag("PIPE_M", "PIPE_MTE1", 0)
        with self.tik_instance.else_scope():
            self.tik_instance.set_flag("PIPE_M", "PIPE_MTE1", 1)
        with self.tik_instance.if_scope(k_redundant == self.k_redundant_factor - 1):
            with self.tik_instance.if_scope(self.l0c_pingpong == 0):
                self.tik_instance.set_flag("PIPE_M", "PIPE_V", 0)
            with self.tik_instance.else_scope():
                self.tik_instance.set_flag("PIPE_M", "PIPE_V", 1)
        with self.tik_instance.if_scope(self.l0b_pingpong == 0):
            self.tik_instance.set_flag("PIPE_M", "PIPE_MTE2", 0)
        with self.tik_instance.else_scope():
            self.tik_instance.set_flag("PIPE_M", "PIPE_MTE2", 1)

    def _mmad_handsync_before(self, k_redundant):
        """
        instr sync before mmad
        """
        with self.tik_instance.if_scope(self.l0a_pingpong == 0):
            self.tik_instance.set_flag("PIPE_MTE1", "PIPE_M", 0)
        with self.tik_instance.else_scope():
            self.tik_instance.set_flag("PIPE_MTE1", "PIPE_M", 1)
        with self.tik_instance.if_scope(self.l0b_pingpong == 0):
            self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_M", 0)
        with self.tik_instance.else_scope():
            self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_M", 1)
        with self.tik_instance.if_scope(k_redundant == 0):
            with self.tik_instance.if_scope(self.l0c_pingpong == 0):
                self.tik_instance.wait_flag("PIPE_V", "PIPE_M", 0)
            with self.tik_instance.else_scope():
                self.tik_instance.wait_flag("PIPE_V", "PIPE_M", 1)
        with self.tik_instance.if_scope(self.l0a_pingpong == 0):
            self.tik_instance.wait_flag("PIPE_MTE1", "PIPE_M", 0)
        with self.tik_instance.else_scope():
            self.tik_instance.wait_flag("PIPE_MTE1", "PIPE_M", 1)

    def _load_l0a_pong(self, input_list):
        """
        load fmap from l1 to l0a pong
        """
        b_inner, batch_index, cin1_l0_index, kh_l0_index, kw_l0_index, pad, wo_l0_index = input_list
        # 性能优化，按搬移次数少的来搬
        with self.tik_instance.if_scope(self.m_l0a_factor < self.k_l0a_factor):
            with self.tik_instance.for_range(0, self.m_l0a_factor, name="m_l0") as m_l0:
                dst_b_offset = b_inner
                dst_m_offset = m_l0
                src_b_offset = b_inner + batch_index
                current_w_offset = m_l0 * self.hw_out0 % self.width_out
                self.tik_instance.pipe_barrier("PIPE_MTE1")
                self.tik_instance.load3dv1(
                    self.fmap_l0a_pong[dst_b_offset, dst_m_offset, 0, 0, 0],
                    self.fmap_l1[src_b_offset, 0, 0, 0, 0], pad, self.height_in, self.width_in, cin1_l0_index,
                    kw_l0_index, kh_l0_index, (wo_l0_index + current_w_offset) * self.stride_w - self.pad_l,
                    self.hin_l1_start_pos, self.stride_w, self.stride_h, self.kernel_w, self.kernel_h,
                    self.dilation_w, self.dilation_w, 1, 0, self.k_l0a_factor, 0, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.k_l0a_factor, name="k_l0") as k_l0:
                dst_b_offset = b_inner
                dst_k_offset = k_l0
                src_b_offset = batch_index + b_inner
                current_c1_offset = k_l0 // (self.kernel_h * self.kernel_w)
                current_kh_offset = k_l0 % (self.kernel_h * self.kernel_w) // self.kernel_w
                current_kw_offset = k_l0 % (self.kernel_h * self.kernel_w) % self.kernel_w
                self.tik_instance.pipe_barrier("PIPE_MTE1")
                self.tik_instance.load3dv1(
                    self.fmap_l0a_pong[dst_b_offset, 0, dst_k_offset, 0, 0],
                    self.fmap_l1[src_b_offset, 0, 0, 0, 0], pad, self.height_in, self.width_in,
                    cin1_l0_index + current_c1_offset, kw_l0_index + current_kw_offset, kh_l0_index + current_kh_offset,
                    wo_l0_index * self.stride_w - self.pad_l, self.hin_l1_start_pos, self.stride_w, self.stride_h,
                    self.kernel_w, self.kernel_h, self.dilation_w, self.dilation_h, self.k_l0a_factor, 1,
                    self.m_l0a_factor, 0, 0)

    def _load_l0a_ping(self, input_list):
        """
        load fmap from l1 to l0a ping
        """
        # 性能优化，按搬移次数少的来搬
        b_inner, batch_index, cin1_l0_index, kh_l0_index, kw_l0_index, pad, wo_l0_index = input_list
        with self.tik_instance.if_scope(self.m_l0a_factor < self.k_l0a_factor):
            with self.tik_instance.for_range(0, self.m_l0a_factor, name="m_l0") as m_l0:
                dst_b_offset = b_inner
                dst_m_offset = m_l0
                src_b_offset = b_inner + batch_index
                current_w_offset = m_l0 * self.hw_out0 % self.width_out
                self.tik_instance.pipe_barrier("PIPE_MTE1")
                self.tik_instance.load3dv1(
                    self.fmap_l0a_ping[dst_b_offset, dst_m_offset, 0, 0, 0],
                    self.fmap_l1[src_b_offset, 0, 0, 0, 0],
                    pad,
                    self.height_in,
                    self.width_in,
                    cin1_l0_index,
                    kw_l0_index,
                    kh_l0_index,
                    (wo_l0_index + current_w_offset) * self.stride_w - self.pad_l,
                    self.hin_l1_start_pos,
                    self.stride_w,
                    self.stride_h,
                    self.kernel_w,
                    self.kernel_h,
                    self.dilation_w,
                    self.dilation_w,
                    1,
                    0,
                    self.k_l0a_factor,
                    0,
                    0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.k_l0a_factor, name="k_l0") as k_l0:
                dst_b_offset = b_inner
                dst_k_offset = k_l0
                src_b_offset = batch_index + b_inner
                current_c1_offset = k_l0 // (self.kernel_h * self.kernel_w)
                current_kh_offset = k_l0 % (self.kernel_h * self.kernel_w) // self.kernel_w
                current_kw_offset = k_l0 % (self.kernel_h * self.kernel_w) % self.kernel_w
                self.tik_instance.pipe_barrier("PIPE_MTE1")
                self.tik_instance.load3dv1(
                    self.fmap_l0a_ping[dst_b_offset, 0, dst_k_offset, 0, 0],
                    self.fmap_l1[src_b_offset, 0, 0, 0, 0],
                    pad,
                    self.height_in,
                    self.width_in,
                    cin1_l0_index + current_c1_offset,
                    kw_l0_index + current_kw_offset,
                    kh_l0_index + current_kh_offset,
                    wo_l0_index * self.stride_w - self.pad_l,
                    self.hin_l1_start_pos,
                    self.stride_w,
                    self.stride_h,
                    self.kernel_w,
                    self.kernel_h,
                    self.dilation_w,
                    self.dilation_h,
                    self.k_l0a_factor,
                    1,
                    self.m_l0a_factor,
                    0,
                    0)

    def _preload_input_x_handsync(self):
        """
        preload input for bn update in kernel process
        """
        next_conv_need_length = self.get_conv_h_length()
        next_ho_index = self.m_l0a_factor * 16 // self.width_out
        next_hin_index = next_ho_index * self.stride_h
        h1_start_pos = next_hin_index - self.pad_t
        self.preload_next_hin_l1_start_pos.set_as(h1_start_pos)
        self.preload_this_time_calu_length.set_as(
            h1_start_pos + next_conv_need_length - self.already_load_l1_h_length)
        with self.tik_instance.if_scope(
                (self.preload_this_time_calu_length + self.already_load_l1_h_length) > self.height_in):
            self.preload_this_time_calu_length.set_as(self.height_in - self.already_load_l1_h_length)
        self.preload_fm_ddr2ub_burst_length.set_as(self.preload_this_time_calu_length * self.width_in)
        self.preload_fm_ddr2ub_src_stride.set_as(
            self.channel_in1 * self.height_in * self.width_in - self.preload_this_time_calu_length * self.width_in)
        self.preload_already_load_l1_h_length.set_as(
            self.already_load_l1_h_length + self.preload_this_time_calu_length)
        self.tik_instance.data_move(
            self.input_x_ub,
            self.fmap_input[self.preload_next_batch, 0,
                            self.already_load_l1_h_length, 0, 0],
            0,
            self.b_inner_factor,
            self.preload_fm_ddr2ub_burst_length,
            self.preload_fm_ddr2ub_src_stride, 0)
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)

    def _first_bn_update_process_handsync(self, axises):
        """
        first bn update split cin1
        """
        b_block = axises.get("b_block")
        pre_core_process_batch = axises.get("pre_core_process_batch")
        batch_index = b_block * (self.b_redundant_factor * self.b_inner_factor) + \
                      pre_core_process_batch * self.b_inner_factor
        self.already_load_l1_h_length.set_as(self.get_conv_h_length() - self.pad_t)

        with self.tik_instance.if_scope(self.pad_t + self.already_load_l1_h_length > self.height_in):
            self.already_load_l1_h_length.set_as(self.height_in)
        fm_ddr2ub_nburst = self.b_inner_factor
        fm_ddr2ub_burst_length = self.already_load_l1_h_length * self.width_in
        fm_ddr2ub_src_stride = self.channel_in1 * self.height_in * self.width_in - \
                               self.already_load_l1_h_length * self.width_in
        self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE2", 3)
        # inner batch 场景下直接跳读
        self.tik_instance.data_move(
            self.input_x_ub,
            self.fmap_input[batch_index, 0, 0, 0, 0],
            0,
            fm_ddr2ub_nburst,
            fm_ddr2ub_burst_length,
            fm_ddr2ub_src_stride,
            0)  # float32 多一倍 block
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 1)
        self.tik_instance.pipe_barrier("PIPE_MTE2")
        with self.tik_instance.for_range(0, self.channel_in1, name="cin1") as cin1:
            self._first_bn_update_single_chanel_process([batch_index, cin1, fm_ddr2ub_burst_length, fm_ddr2ub_nburst,
                                                         fm_ddr2ub_src_stride])

    def _first_bn_update_single_chanel_process(self, input_list):
        """
        single chanel process in first bn update
        """
        batch_index, cin1, fm_ddr2ub_burst_length, fm_ddr2ub_nburst, fm_ddr2ub_src_stride = input_list
        with self.tik_instance.if_scope(cin1 > 0):
            self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 0)
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 1)
        fp32_loop_times = self.b_inner_factor * self.already_load_l1_h_length * self.width_in * \
                          self.channel_in0 // (64 * 255)
        fp32_loop_redundant = self.b_inner_factor * self.already_load_l1_h_length * self.width_in * \
                              self.channel_in0 % (64 * 255)
        fp32_repeat = fp32_loop_redundant // 64
        fp32_redundant = fp32_loop_redundant % 64
        self.tik_instance.pipe_barrier("PIPE_V")
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(0, fp32_loop_times, name="fp32_loop") as fp32_loop:
                self.tik_instance.vconv(
                    64,
                    "none",
                    self.input_x_fp32_ub[fp32_loop * 255 * 64],
                    self.input_x_ub[fp32_loop * 255 * 64],
                    255,
                    1, 1, 8, 4)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vconv(
                64,
                "none",
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                self.input_x_ub[fp32_loop_times * 255 * 64],
                fp32_repeat,
                1, 1, 8, 4)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vconv(
                fp32_redundant,
                "none",
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                self.input_x_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                1,
                1, 1, 8, 4)
        # preload fm ddr 2 ub
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE2", 0)
        with self.tik_instance.if_scope((cin1 + 1) < self.channel_in1):
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE2", 0)
            self.tik_instance.data_move(
                self.input_x_ub,
                self.fmap_input[batch_index, cin1 + 1, 0, 0, 0],
                0,
                fm_ddr2ub_nburst,
                fm_ddr2ub_burst_length,
                fm_ddr2ub_src_stride,
                0)
            self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 1)
        self.tik_instance.pipe_barrier("PIPE_V")
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(
                    0, fp32_loop_times, name="loop_time") as loop_time:
                self.tik_instance.vmaddrelu(64, self.input_x_fp32_ub[loop_time * 255 * 64],
                                            self.scale_ub_64[cin1 * 64],
                                            self.offset_ub_64[cin1 * 64], 255, 1, 1, 1, 8, 0, 0)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vmaddrelu(64, self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                                        self.scale_ub_64[cin1 * 64],
                                        self.offset_ub_64[cin1 * 64], fp32_repeat, 1, 1, 1, 8, 0, 0)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vmaddrelu(fp32_redundant,
                                        self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                                        self.scale_ub_64[cin1 * 64],
                                        self.offset_ub_64[cin1 * 64], 1, 1, 1, 1, 8, 0, 0)
        self.tik_instance.pipe_barrier("PIPE_V")
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(
                    0, fp32_loop_times, name="fp32_loop") as fp32_loop:
                self.tik_instance.vconv(
                    64,
                    "none",
                    self.input_x_fp16_ub[fp32_loop * 255 * 64],
                    self.input_x_fp32_ub[fp32_loop * 255 * 64],
                    255,
                    1, 1, 4, 8)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vconv(
                64,
                "none",
                self.input_x_fp16_ub[fp32_loop_times * 255 * 64],
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                fp32_repeat,
                1, 1, 4, 8)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vconv(
                fp32_redundant,
                "none",
                self.input_x_fp16_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                1,
                1, 1, 4, 8)
        self._fetch_first_bn_update_process([batch_index, cin1, fm_ddr2ub_burst_length, fm_ddr2ub_nburst,
                                             fm_ddr2ub_src_stride])

    def _fetch_first_bn_update_process(self, input_list):
        """
        fetch first bn update
        """
        batch_index, cin1, fm_ddr2ub_burst_length, fm_ddr2ub_nburst, fm_ddr2ub_src_stride = input_list
        # move relu result
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.pipe_barrier("PIPE_MTE3")
        self.tik_instance.data_move(
            self.output_relu[batch_index, cin1, 0, 0, 0],
            self.input_x_fp16_ub,
            0,
            fm_ddr2ub_nburst,
            fm_ddr2ub_burst_length, 0, fm_ddr2ub_src_stride)
        # 将数据搬到L1
        self.tik_instance.data_move(
            self.fmap_l1[0, cin1, 0, 0, 0],
            self.input_x_fp16_ub,
            0,
            fm_ddr2ub_nburst,
            fm_ddr2ub_burst_length,
            0, fm_ddr2ub_src_stride)
        self.calc_mask_in_bn_update(batch_index, cin1)

    def _bn2_mean_varience_process_handsync(self):
        """
        prepare process, calucate the offset and scale
        """
        start_pos = 0
        elem_num = self.channel_in1 * self.channel_in0
        burst_length = elem_num // 8
        num = self.batch * self.height_in * self.width_in
        num_rec = 1.0 / num

        redundant, repeat = self._first_part_bn_mean_varience_process(burst_length, elem_num, num_rec, start_pos)
        self._second_part_bn_mean_varience_process(burst_length, redundant, repeat, start_pos)
        self._third_part_bn_mean_varience_process([burst_length, num, redundant, repeat, start_pos])

    def _third_part_bn_mean_varience_process(self, input_list):
        """
        third part of calculate mean and varience process
        """
        burst_length, num, redundant, repeat, start_pos = input_list
        # 计算均值的滑动平均 输出到ddr
        factor_reverse = self._third_part_bn_mean_varience_process_sub(burst_length, redundant, repeat, start_pos)
        # 这一块计算逻辑不知道为什么要加上去，理论上不需要 乘以 batch_var_scalar，
        # 为了和库上的bn_training_update 保持一致，有可能会有精度抖动
        if num == 1:
            batch_var_scalar = 0.0
        else:
            batch_var_scalar = num / (num - 1)
        if repeat >= 1:
            self.tik_instance.vmuls(
                64,
                self.variance_batch_ub,
                self.varience_ub,
                batch_var_scalar,
                repeat,
                1, 1, 8, 8)  # float32 一个repeat 64个元素
        if redundant >= 1:
            self.tik_instance.vmuls(
                redundant,
                self.variance_batch_ub[repeat * 64],
                self.varience_ub[repeat * 64],
                batch_var_scalar,
                1,
                1, 1, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vmuls(
                64,
                self.variance_mul_ub,
                self.variance_batch_ub,
                self.factor,
                repeat,
                1, 1, 8, 8)  # float32 一个repeat 64个元素
        if redundant >= 1:
            self.tik_instance.vmuls(
                redundant,
                self.variance_mul_ub[repeat * 64],
                self.variance_batch_ub[repeat * 64],
                self.factor,
                1,
                1, 1, 8, 8)
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 1)
        self._fetch_third_part_bn_mean_varience_process([burst_length, factor_reverse, redundant, repeat, start_pos])

    def _fetch_third_part_bn_mean_varience_process(self, input_list):
        """
        fetch third part bn mean and varience process
        """
        burst_length, factor_reverse, redundant, repeat, start_pos = input_list
        if repeat >= 1:
            self.tik_instance.vmuls(
                64, self.variance_mul_rev_ub, self.pre_moving_variance_ub, factor_reverse, repeat, 1, 1, 8, 8)
        if redundant >= 1:
            self.tik_instance.vmuls(
                redundant * 8, self.variance_mul_rev_ub[repeat * 64], self.pre_moving_variance_ub[repeat * 64],
                factor_reverse, 1, 1, 1, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vadd(
                64, self.moving_variance_ub, self.variance_mul_ub, self.variance_mul_rev_ub, repeat,
                1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vadd(
                redundant * 8, self.moving_variance_ub[repeat * 64], self.variance_mul_ub[repeat * 64],
                self.variance_mul_rev_ub[repeat * 64], 1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 3)
        dup_scalar = self.tik_instance.Scalar(dtype="float16", name="dup_zero_scalar", init_value=0)
        self.tik_instance.vector_dup(
            128, self.zeros_const_ub, dup_scalar, 1, 1, 1)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.data_move(
            self.output_mean[start_pos], self.mean_ub, 0, 1, burst_length, 0, 0)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 1)
        self.tik_instance.data_move(
            self.output_varience[start_pos], self.varience_ub, 0, 1, burst_length, 0, 0)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 2)
        self.tik_instance.data_move(
            self.output_moving_mean[start_pos], self.moving_mean_ub, 0, 1, burst_length, 0, 0)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 3)
        self.tik_instance.data_move(
            self.output_moving_varience[start_pos], self.moving_variance_ub, 0, 1, burst_length, 0, 0)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_MTE2", 0)
        for i in range(4):
            self.tik_instance.data_move(self.offset_ub_64[16 * i], self.y_offset_ub,
                                        0, self.channel_in1, 2, 0, 6)
            self.tik_instance.data_move(self.scale_ub_64[16 * i], self.y_scale_ub,
                                        0, self.channel_in1, 2, 0, 6)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE2", 1)

    def _second_part_bn_mean_varience_process(self, burst_length, redundant, repeat, start_pos):
        """
        second part of calculate mean and varience process
        """
        ## 计算整体的缩放和平移参数
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 1)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vadds(64, self.y_scale_add_ub, self.varience_ub, self.epsilon, repeat, 1, 1, 8, 8)
        if redundant >= 1:
            self.tik_instance.vadds(redundant, self.y_scale_add_ub[repeat * 64], self.varience_ub[repeat * 64],
                                    self.epsilon, 1, 1, 1, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vsqrt(64, self.y_scale_sqrt_ub, self.y_scale_add_ub, repeat, 1, 1, 8, 8)
        if redundant >= 1:
            self.tik_instance.vsqrt(redundant, self.y_scale_sqrt_ub[repeat * 64], self.y_scale_add_ub[repeat * 64],
                                    1, 1, 1, 8, 8)
        self.tik_instance.data_move(self.offset_ub, self.offset_input[start_pos], 0, 1, burst_length, 0, 0)
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 1)
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 0)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vdiv(64, self.y_scale_ub, self.scale_ub, self.y_scale_sqrt_ub, repeat, 1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vdiv(redundant, self.y_scale_ub[repeat * 64], self.scale_ub[repeat * 64],
                                   self.y_scale_sqrt_ub[repeat * 64], 1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vmul(64, self.y_offset_mul_ub, self.mean_ub, self.y_scale_ub, repeat, 1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vmul(redundant, self.y_offset_mul_ub[repeat * 64], self.mean_ub[repeat * 64],
                                   self.y_scale_ub[repeat * 64], 1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.data_move(self.pre_moving_mean_ub, self.pre_moving_mean_input[start_pos], 0, 1, burst_length,
                                    0, 0)  # float32 多一倍 block
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 1)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vsub(64, self.y_offset_ub, self.offset_ub, self.y_offset_mul_ub, repeat, 1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vsub(redundant, self.y_offset_ub[repeat * 64], self.offset_ub[repeat * 64],
                                   self.y_offset_mul_ub[repeat * 64], 1, 1, 1, 1, 8, 8, 8)

    def _first_part_bn_mean_varience_process(self, burst_length, elem_num, num_rec, start_pos):
        """
        first part of calculate mean and varience
        """
        self.tik_instance.data_move(
            self.sum_input_ub,
            self.sum_input[start_pos],
            0, 1, burst_length, 0, 0)  # float32 多一倍 block
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)
        self.tik_instance.data_move(
            self.square_sum_input_ub,
            self.square_sum_input[start_pos],
            0,
            1, burst_length,
            0, 0)
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 1)
        repeat = elem_num // 64
        redundant = elem_num % 64
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 0)
        if repeat >= 1:
            self.tik_instance.vmuls(
                64,
                self.mean_ub,
                self.sum_input_ub,  # float32 一个repeat 64个元素
                num_rec,
                repeat,
                1, 1, 8, 8)
        if redundant >= 1:
            self.tik_instance.vmuls(
                redundant,
                self.mean_ub[repeat * 64],
                self.sum_input_ub[repeat * 64],
                num_rec,
                1, 1, 1, 8, 8)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.data_move(
            self.scale_ub,
            self.scale_input[start_pos],
            0,
            1, burst_length,
            0, 0)  # float32 多一倍 block
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 1)
        if repeat >= 1:
            self.tik_instance.vmuls(
                64,
                self.square_sum_div_ub,
                self.square_sum_input_ub,  # float32 一个repeat 64个元素
                num_rec,
                repeat,
                1, 1, 8, 8)
        if redundant >= 1:
            self.tik_instance.vmuls(
                redundant,
                self.square_sum_div_ub[repeat * 64],
                self.square_sum_input_ub[repeat * 64],
                num_rec,
                1, 1, 1, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vmul(
                64,
                self.square_mean_ub,
                self.mean_ub,
                self.mean_ub,  # float32 一个repeat 64个元素
                repeat,
                1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vmul(
                redundant,
                self.square_mean_ub[repeat * 64],
                self.mean_ub[repeat * 64],
                self.mean_ub[repeat * 64],
                1,
                1, 1, 1, 8, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vsub(
                64,
                self.varience_ub,
                self.square_sum_div_ub,
                self.square_mean_ub,  # float32 一个repeat 64个元素
                repeat,
                1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vsub(
                redundant,
                self.varience_ub[repeat * 64],
                self.square_sum_div_ub[repeat * 64],
                self.square_mean_ub[repeat * 64],
                1,
                1, 1, 1, 8, 8, 8)
        return redundant, repeat


class Bn2ReluV2Conv2dBn1HandSync(Bn2ReluV2Conv2dBn1NoRedundant):
    """
    class of Bn2ReluV2Conv2dBn1HandSync
    """

    def __init__(self, fmap_ori_shape, filters_ori_shape, padding, stride, dilation, groups, factor, epsilon, tiling,
                 kernel_name="Bn2ReluV2Conv2dBn1HandSync"):
        super(Bn2ReluV2Conv2dBn1HandSync, self).__init__(fmap_ori_shape, filters_ori_shape, padding, stride, dilation,
                                                         groups, factor, epsilon, tiling, kernel_name)

    def mask_tail_process_handsync(self, input_list):
        """
        mask tail process for align 32 Byte for last part of mask
        """
        # 尾块部分往外搬时会有内存踩踏，先搬不踩的，最后32bit要特殊处理
        batch_l1_offset, cin1_l1_offset, input_x_mask_ub, mask_hin_offset, \
        mask_real_length, input_x_mask_ub_length, tem_ub, save_ub = input_list

        self.tik_instance.data_move(
            self.output_reluv2_mask[batch_l1_offset, cin1_l1_offset, mask_hin_offset, 0],
            input_x_mask_ub,
            0,
            1, mask_real_length // 16,
            0, 0)
        self.tik_instance.pipe_barrier("PIPE_V")
        # 最后剩余部分的数据先重排，再与前面ub中的数据凑成32bit往外搬
        dst_list = [tem_ub[16 * i] for i in range(16)]
        src_list = [input_x_mask_ub[input_x_mask_ub_length - 32]] * 16
        self.tik_instance.vnchwconv(True, False, dst_list, src_list, 2, 16, 1)
        self.tik_instance.pipe_barrier("PIPE_V")
        start_pos = mask_real_length % 16
        src_list = [tem_ub[(start_pos + i) * 16] for i in range(16)]
        dst_list = [save_ub[i * 16] for i in range(16)]
        self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.pipe_barrier("PIPE_MTE3")
        total_offset = batch_l1_offset * (self.channel_in1 * self.height_in * self.width_in) + cin1_l1_offset * \
                       self.height_in * self.width_in + mask_hin_offset * self.width_in + mask_real_length - 16
        self.tik_instance.data_move(
            self.output_reluv2_mask[total_offset],
            save_ub,
            0,
            1, 1,
            0, 0)

    def fmap_l0a_process_handsync(self, axises):
        """
        load fmap from l1 to l0a
        """
        k_redundant = axises["k_redundant"]
        m_redundant = axises["m_redundant"]
        batch_index = 0
        howo_l0_index = m_redundant * self.m_l0a_factor * self.hw_out0
        wo_l0_index = howo_l0_index % self.width_out
        cin1_l0_index = (k_redundant * self.k_l0a_factor) // (self.kernel_h * self.kernel_w)
        khkw_l0_index = (k_redundant * self.k_l0a_factor) % (self.kernel_h * self.kernel_w)
        kh_l0_index = khkw_l0_index // self.kernel_w
        kw_l0_index = khkw_l0_index % self.kernel_w
        b_inner = axises["b_inner"]
        pad = [self.pad_l, self.pad_r, self.pad_t, self.pad_b]
        with self.tik_instance.if_scope(k_redundant == 0):
            self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_MTE1", 0)
        with self.tik_instance.if_scope(self.l0a_pingpong == 0):
            self._load_l0a_ping([b_inner, batch_index, cin1_l0_index, kh_l0_index, kw_l0_index, pad, wo_l0_index])
        with self.tik_instance.else_scope():
            self._load_l0a_pong([b_inner, batch_index, cin1_l0_index, kh_l0_index, kw_l0_index, pad, wo_l0_index])

    def filters_l0b_process_handsync(self, axises):
        """
        load filters from ddr or l1 to l0b
        """
        k_redundant = axises["k_redundant"]
        n_l1 = axises["n_l1a"]
        with self.tik_instance.if_scope(self.l0b_pingpong == 0):
            self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE1", 0)
            if self.filters_load_from_l1_flag:
                with self.tik_instance.for_range(0, self.k_l0a_factor, name="k_l0") as k_l0:
                    dst_offset = k_l0

                    src_k_offset = (k_redundant * self.k_l0a_factor + k_l0)
                    src_n_offset = n_l1 * self.n_l0a_factor
                    self.tik_instance.load2dv1(
                        self.filters_l0b_ping[dst_offset, 0, 0, 0],
                        self.filters_l1[src_k_offset, src_n_offset, 0, 0],
                        0,
                        self.n_l0a_factor,
                        1, 0)
            else:
                with self.tik_instance.for_range(0, self.k_l0a_factor, name="k_l0") as k_l0:
                    dst_offset = k_l0

                    src_k_offset = (k_redundant * self.k_l0a_factor + k_l0)
                    src_n_offset = n_l1 * self.n_l0a_factor
                    self.tik_instance.load2dv1(
                        self.filters_l0b_ping[dst_offset, 0, 0, 0],
                        self.filters_input[src_k_offset, src_n_offset, 0, 0],
                        0,
                        self.n_l0a_factor,
                        1, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE1", 1)
            if self.filters_load_from_l1_flag:
                with self.tik_instance.for_range(0, self.k_l0a_factor, name="k_l0") as k_l0:
                    dst_offset = k_l0

                    src_k_offset = (k_redundant * self.k_l0a_factor + k_l0)
                    src_n_offset = n_l1 * self.n_l0a_factor
                    self.tik_instance.load2dv1(
                        self.filters_l0b_pong[dst_offset, 0, 0, 0],
                        self.filters_l1[src_k_offset, src_n_offset, 0, 0],
                        0,
                        self.n_l0a_factor,
                        1, 0)
            else:
                with self.tik_instance.for_range(0, self.k_l0a_factor, name="k_l0") as k_l0:
                    dst_offset = k_l0

                    src_k_offset = (k_redundant * self.k_l0a_factor + k_l0)
                    src_n_offset = n_l1 * self.n_l0a_factor
                    self.tik_instance.load2dv1(
                        self.filters_l0b_pong[dst_offset, 0, 0, 0],
                        self.filters_input[src_k_offset, src_n_offset, 0, 0],
                        0,
                        self.n_l0a_factor,
                        1, 0)

    def filters_l1_process_handsync(self):
        """
        load filters from ddr to l1
        """
        if self.filters_load_from_l1_flag:
            with self.tik_instance.if_scope(self.n_redundant < self.n_redundant_factor):
                src_k_offset = 0
                src_n_offset = self.n_redundant * self.n_l1a_factor * self.n_l0a_factor

                self.tik_instance.data_move(
                    self.filters_l1,
                    self.filters_input[src_k_offset, src_n_offset, 0, 0],
                    0,
                    self.k_l1a_factor * self.k_l0a_factor,
                    self.n_l1a_factor * self.n_l0a_factor * self.channel_out0,
                    self.channel_out1 * self.channel_out0 -
                    self.n_l1a_factor * self.n_l0a_factor * self.channel_out0,
                    0)
            self.tik_instance.set_flag("PIPE_MTE2", "PIPE_MTE1", 0)

    def compute(self):
        """
        main process
        """
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            self._init_all_ub_tensor()
            self._init_l0a_l0b_l0c_tensor()
            self._init_output_sum_and_square_sum()
            self._bn2_mean_varience_process_handsync()
            # 绑多核
            with self.tik_instance.for_range(0, self.b_block_factor, block_num=self.b_block_factor,
                                             name="b_block") as b_block:
                axises = {
                    "b_block": b_block,
                    "pre_core_process_batch": 0,
                    "n_redundant": 0,
                }
                self._first_bn_update_process_handsync(axises)
                self.tik_instance.set_flag("PIPE_MTE3", "PIPE_MTE1", 0)
                self.next_batch.set_as(b_block * self.b_redundant_factor * self.b_inner_factor)
                self.preload_next_batch.set_as(self.next_batch)
                self.preload_already_load_l1_h_length.set_as(self.already_load_l1_h_length)
                self._preload_input_x_handsync()
                if (self.filters_ddr_size == self.filters_l1_size) and self.filters_load_from_l1_flag:
                    self.filters_l1_process_handsync()
                self.kernel_process_with_instr_sync(axises)
            self.last_bn_reduce_process_handsync()
            self.move_sum_and_square_sum_toddr_handsync()
        self.tik_instance.BuildCCE(self.kernel_name, self.input_tensors, self.output_tensors)

    def kernel_process_with_instr_sync(self, axises):
        """
        instr sync and kernle process
        """
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_MTE1", 0)
        self.tik_instance.pipe_barrier("PIPE_V")
        self.tik_instance.pipe_barrier("PIPE_MTE3")
        self.tik_instance.pipe_barrier("PIPE_MTE2")
        self.tik_instance.set_flag("PIPE_M", "PIPE_MTE1", 0)
        self.tik_instance.set_flag("PIPE_M", "PIPE_MTE1", 1)
        self.tik_instance.set_flag("PIPE_V", "PIPE_M", 0)
        self.tik_instance.set_flag("PIPE_V", "PIPE_M", 1)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 0)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 1)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 2)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 3)
        with self.tik_instance.for_range(0, self.b_redundant_factor,
                                         name="pre_core_precess_batch") as pre_core_process_batch:
            axises["pre_core_process_batch"] = pre_core_process_batch
            self.kernel_process_handsync(axises)
        self.tik_instance.pipe_barrier("PIPE_MTE3")
        self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE1", 0)
        self.tik_instance.wait_flag("PIPE_M", "PIPE_MTE1", 1)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_M", 0)
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 0)
        self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 0)
        self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 1)
        self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 2)
        self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 3)

    def spilt_allocate_ub_area_one(self, elem_num, y_offset_mul_ub_start_addr):
        """
        split allocate ub area one
        """
        pre_moving_mean_ub_start_addr = self.get_start_addr(y_offset_mul_ub_start_addr, elem_num * 4)
        self.pre_moving_mean_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="pre_moving_mean_ub",
            scope=tik.scope_ubuf,
            start_addr=pre_moving_mean_ub_start_addr)
        mean_mul_ub_start_addr = self.get_start_addr(pre_moving_mean_ub_start_addr, elem_num * 4)
        self.mean_mul_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="mean_mul_ub",
            scope=tik.scope_ubuf,
            start_addr=mean_mul_ub_start_addr)
        pre_moving_variance_ub_start_addr = self.get_start_addr(mean_mul_ub_start_addr, elem_num * 4)
        self.pre_moving_variance_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="pre_moving_variance_ub",
            scope=tik.scope_ubuf,
            start_addr=pre_moving_variance_ub_start_addr)
        mean_mul_rev_ub_start_addr = self.get_start_addr(pre_moving_variance_ub_start_addr, elem_num * 4)
        self.mean_mul_rev_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="mean_mul_rev_ub",
            scope=tik.scope_ubuf,
            start_addr=mean_mul_rev_ub_start_addr)
        moving_mean_ub_start_addr = self.get_start_addr(mean_mul_rev_ub_start_addr, elem_num * 4)
        self.moving_mean_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="moving_mean_ub",
            scope=tik.scope_ubuf,
            start_addr=moving_mean_ub_start_addr)
        variance_batch_ub_start_addr = self.get_start_addr(moving_mean_ub_start_addr, elem_num * 4)
        self.variance_batch_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="variance_batch_ub",
            scope=tik.scope_ubuf,
            start_addr=variance_batch_ub_start_addr)
        variance_mul_ub_start_addr = self.get_start_addr(variance_batch_ub_start_addr, elem_num * 4)
        self.variance_mul_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="variance_mul_ub",
            scope=tik.scope_ubuf,
            start_addr=variance_mul_ub_start_addr)
        variance_mul_rev_ub_start_addr = self.get_start_addr(variance_mul_ub_start_addr, elem_num * 4)
        self.variance_mul_rev_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="variance_mul_rev_ub",
            scope=tik.scope_ubuf,
            start_addr=variance_mul_rev_ub_start_addr)
        moving_variance_ub_start_addr = self.get_start_addr(variance_mul_rev_ub_start_addr, elem_num * 4)
        self.moving_variance_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="moving_variance_ub",
            scope=tik.scope_ubuf,
            start_addr=moving_variance_ub_start_addr)

    def move_sum_and_square_sum_toddr_handsync(self):
        """
        move sum and square sum to ddr
        """
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.set_atomic_add(1)
        self.tik_instance.data_move(
            self.output_sum[0, 0, 0, 0, 0],
            self.output_sum_inub,
            0,
            1, 8, 0, 0)
        self.tik_instance.data_move(
            self.output_square_sum[0, 0, 0, 0, 0],
            self.output_square_sum_inub,
            0,
            1, 8, 0, 0)
        self.tik_instance.pipe_barrier("PIPE_ALL")
        self.tik_instance.set_atomic_add(0)

    def last_bn_reduce_process_handsync(self):
        """
        last bn reduce process
        """
        with self.tik_instance.for_range(0, self.n_l0a_factor, name="n_l0") as n_l0:
            square_size = self.b_inner_factor * self.m_l0a_factor * self.hw_out0 * self.channel_out0
            with self.tik_instance.if_scope(self.l0c_pingpong == 1):
                self.tik_instance.pipe_barrier("PIPE_V")
                with self.tik_instance.if_scope(n_l0 == 0):
                    self.tik_instance.wait_flag("PIPE_M", "PIPE_V", 0)
                self.tik_instance.data_move(
                    self.fmap_ub,
                    self.fmap_l0c_ping[0, n_l0, 0, 0, 0],
                    0,
                    self.b_inner_factor,
                    self.m_l0a_factor,
                    (self.n_l0a_factor - 1) * self.m_l0a_factor,
                    0)
            with self.tik_instance.else_scope():
                self.tik_instance.pipe_barrier("PIPE_V")
                with self.tik_instance.if_scope(n_l0 == 0):
                    self.tik_instance.wait_flag("PIPE_M", "PIPE_V", 1)
                self.tik_instance.data_move(
                    self.fmap_ub,
                    self.fmap_l0c_pong[0, n_l0, 0, 0, 0],
                    0,
                    self.b_inner_factor,
                    self.m_l0a_factor,
                    (self.n_l0a_factor - 1) * self.m_l0a_factor,
                    0)

            # mv ub to gm
            with self.tik_instance.if_scope(n_l0 > 0):
                self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 0)
            fp32_loop_times, fp32_redundant, fp32_repeat_times = self.calu_loop_time(square_size)
            self.res_fmap_convertf16_handsync(self.fmap_f16_ub_ping, self.fmap_ub,
                                              {"fp32_loop_times": fp32_loop_times,
                                               "fp32_redundant": fp32_redundant,
                                               "fp32_repeat_times": fp32_repeat_times})
            dst_b_offset, dst_m_offset, dst_n_offset = self.calc_dst_b_n_m_offset(m_l0=0, n_l0=n_l0)
            self.tik_instance.data_move(
                self.output_convolution[dst_b_offset, dst_n_offset, dst_m_offset, 0],
                self.fmap_f16_ub_ping,
                0,
                self.b_inner_factor,
                self.m_l0a_factor * self.hw_out0,
                0,
                self.channel_out1 * self.height_out *
                self.width_out - self.m_l0a_factor * self.hw_out0)
            # square sum
            with self.tik_instance.if_scope(n_l0 < (self.n_l0a_factor - 1)):
                self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 0)
            self.calu_square_sum_handsync(self.fmap_ub, fp32_loop_times, fp32_redundant, fp32_repeat_times)
            self.dichotomy_sum_and_square_sum_handsync(self.fmap_ub, self.square_mul_ub, square_size)
            self.tik_instance.vadd(16,
                                   self.output_sum_inub[0, dst_n_offset, 0, 0, 0],
                                   self.output_sum_inub[0, dst_n_offset, 0, 0, 0],
                                   self.fmap_ub,
                                   1, 1, 1, 1, 2, 2, 2)
            self.tik_instance.vadd(16,
                                   self.output_square_sum_inub[0, dst_n_offset, 0, 0, 0],
                                   self.output_square_sum_inub[0, dst_n_offset, 0, 0, 0],
                                   self.square_mul_ub,
                                   1, 1, 1, 1, 2, 2, 2)

    def kernel_process_handsync(self, axises):
        """
        kernel process for conv
        """
        with self.tik_instance.for_range(0, self.n_l1a_factor, name="n_l1a") as n_l1a:
            if self.filters_load_from_l1_flag and (self.filters_ddr_size != self.filters_l1_size):
                with self.tik_instance.if_scope(self.n_redundant < self.n_redundant_factor):
                    self.front_n_redundant.set_as(self.n_redundant)
                    self.filters_l1_process_handsync()
                    self.n_redundant.set_as(self.n_redundant + 1)
                with self.tik_instance.else_scope():
                    # 开始加载下一轮batch的filter
                    self.n_redundant.set_as(0)
                    self.filters_l1_process_handsync()
            with self.tik_instance.for_range(0, self.m_redundant_factor, name="m_redundant") as m_redundant:
                with self.tik_instance.for_range(0, self.k_redundant_factor, name="k_redundant") as k_redundant:
                    axises["m_redundant"] = m_redundant
                    axises["n_l1a"] = n_l1a
                    axises["k_redundant"] = k_redundant
                    self.filters_l0b_process_handsync(axises)
                    with self.tik_instance.for_range(0, self.b_inner_factor, name="b_inner") as b_inner:
                        axises["b_inner"] = b_inner
                        self.fmap_l0a_process_handsync(axises)
                        self.mad_process_handsync(axises)
                        self.l0a_pingpong.set_as(1 - self.l0a_pingpong)
                    self.l0b_pingpong.set_as(1 - self.l0b_pingpong)
                    with self.tik_instance.if_scope(n_l1a == 0):
                        with self.tik_instance.if_scope(k_redundant == 0):
                            with self.tik_instance.if_scope(m_redundant + 1 < self.m_redundant_factor):
                                self.next_hin_l1_start_pos.set_as(self.preload_next_hin_l1_start_pos)
                            with self.tik_instance.else_scope():
                                # 下一个batch的bn_update的开始
                                self.next_hin_l1_start_pos.set_as(self.preload_next_hin_l1_start_pos)
                                self.already_load_l1_h_length.set_as(self.preload_already_load_l1_h_length)
                                self.next_batch.set_as(self.preload_next_batch)
                        with self.tik_instance.if_scope(
                                tik.any(axises["pre_core_process_batch"] != (self.b_redundant_factor - 1),
                                        m_redundant != (self.m_redundant_factor - 1))):
                            self.bn_update_process_handsync(axises)
                    with self.tik_instance.if_scope(tik.any(axises["pre_core_process_batch"] != 0,
                                                            n_l1a != 0,
                                                            m_redundant != 0)):
                        self.bn_reduce_process_handsync(axises)
                    with self.tik_instance.if_scope(k_redundant + 1 == self.k_redundant_factor):
                        self.l0c_pingpong.set_as(1 - self.l0c_pingpong)
                        self.front_m.set_as(m_redundant)
                        self.front_n_l1.set_as(n_l1a)
                        self.hin_l1_start_pos.set_as(self.next_hin_l1_start_pos)
                        self.front_batch.set_as(
                            axises["b_block"] * self.b_redundant_factor + axises["pre_core_process_batch"])

    def res_fmap_convertf16_handsync(self, fmap_f16_ub, fmap_ub, loop_time):
        """
        convert dtype form fp32 to fp16
        """
        self.res_fmap_convertf16(fmap_f16_ub, fmap_ub, loop_time)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)

    def bn_reduce_process_handsync(self, axises):
        """
        bn reduce process in kernel process
        """
        n_l0_and_kl1_comm = Bn2ReluV2Conv2dBn1NoRedundant.get_gcd(self.n_l0a_factor, self.k_redundant_factor)
        n_l0_single_length = self.n_l0a_factor // n_l0_and_kl1_comm
        k_redundant = axises["k_redundant"]
        with self.tik_instance.if_scope(k_redundant % (self.k_redundant_factor // n_l0_and_kl1_comm) == 0):
            n_l0_gap = k_redundant // (self.k_redundant_factor // n_l0_and_kl1_comm)
            with self.tik_instance.for_range(n_l0_gap * n_l0_single_length, (n_l0_gap + 1) * n_l0_single_length,
                                             name="n_l0") as n_l0:
                square_size = self.b_inner_factor * self.m_l0a_factor * self.hw_out0 * self.channel_out0

                with self.tik_instance.if_scope(self.l0c_pingpong == 1):
                    self.tik_instance.pipe_barrier("PIPE_V")
                    with self.tik_instance.if_scope(k_redundant == 0):
                        self.tik_instance.wait_flag("PIPE_M", "PIPE_V", 0)
                    self.tik_instance.data_move(
                        self.fmap_ub,
                        self.fmap_l0c_ping[0, n_l0, 0, 0, 0],
                        0,
                        self.b_inner_factor,
                        self.m_l0a_factor,
                        (self.n_l0a_factor - 1) * self.m_l0a_factor,
                        0)
                with self.tik_instance.else_scope():
                    self.tik_instance.pipe_barrier("PIPE_V")
                    with self.tik_instance.if_scope(k_redundant == 0):
                        self.tik_instance.wait_flag("PIPE_M", "PIPE_V", 1)
                    self.tik_instance.data_move(
                        self.fmap_ub,
                        self.fmap_l0c_pong[0, n_l0, 0, 0, 0],
                        0,
                        self.b_inner_factor,
                        self.m_l0a_factor,
                        (self.n_l0a_factor - 1) * self.m_l0a_factor,
                        0)
                fp32_loop_times, fp32_redundant, fp32_repeat_times = self.calu_loop_time(square_size)
                dst_b_offset, dst_m_offset, dst_n_offset = self.calc_dst_b_n_m_offset(m_l0=0, n_l0=n_l0)
                with self.tik_instance.if_scope(self.bn_reduce_ub_output_pingpong == 0):
                    self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 0)
                    # mv ub to gm
                    self.res_fmap_convertf16_handsync(self.fmap_f16_ub_ping, self.fmap_ub,
                                                      {"fp32_loop_times": fp32_loop_times,
                                                       "fp32_redundant": fp32_redundant,
                                                       "fp32_repeat_times": fp32_repeat_times})
                    self.tik_instance.pipe_barrier("PIPE_MTE3")
                    self.tik_instance.data_move(
                        self.output_convolution[dst_b_offset, dst_n_offset, dst_m_offset, 0],
                        self.fmap_f16_ub_ping,
                        0,
                        self.b_inner_factor,
                        self.m_l0a_factor * self.hw_out0,
                        0,
                        self.channel_out1 * self.height_out *
                        self.width_out - self.m_l0a_factor * self.hw_out0)
                    self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 1)
                    # mv ub to gm
                    self.res_fmap_convertf16_handsync(self.fmap_f16_ub_pong, self.fmap_ub,
                                                      {"fp32_loop_times": fp32_loop_times,
                                                       "fp32_redundant": fp32_redundant,
                                                       "fp32_repeat_times": fp32_repeat_times})
                    self.tik_instance.pipe_barrier("PIPE_MTE3")
                    self.tik_instance.data_move(
                        self.output_convolution[dst_b_offset, dst_n_offset, dst_m_offset, 0],
                        self.fmap_f16_ub_pong,
                        0,
                        self.b_inner_factor,
                        self.m_l0a_factor * self.hw_out0,
                        0,
                        self.channel_out1 * self.height_out *
                        self.width_out - self.m_l0a_factor * self.hw_out0)
                    self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 1)
                self.bn_reduce_ub_output_pingpong.set_as(1 - self.bn_reduce_ub_output_pingpong)
                # square sum
                self.calu_square_sum_handsync(self.fmap_ub, fp32_loop_times, fp32_redundant, fp32_repeat_times)
                self.dichotomy_sum_and_square_sum_handsync(self.fmap_ub, self.square_mul_ub, square_size)
                with self.tik_instance.if_scope(k_redundant == ((self.n_l0a_factor - 1) *
                                                                (self.k_redundant_factor // n_l0_and_kl1_comm))):
                    with self.tik_instance.if_scope(self.l0c_pingpong == 1):
                        self.tik_instance.set_flag("PIPE_V", "PIPE_M", 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.set_flag("PIPE_V", "PIPE_M", 1)
                self.tik_instance.pipe_barrier("PIPE_V")
                self.tik_instance.vadd(16,
                                       self.output_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       self.output_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       self.fmap_ub,
                                       1, 1, 1, 1, 2, 2, 2)
                self.tik_instance.vadd(16,
                                       self.output_square_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       self.output_square_sum_inub[0, dst_n_offset, 0, 0, 0],
                                       self.square_mul_ub,
                                       1, 1, 1, 1, 2, 2, 2)

    def effecient_sum_and_square_sum_handsync(self, sum_ub, square_sum_ub, total_blocks):
        """
        part of dichotomy
        """
        threshold = 6
        total_repeat = total_blocks // 64
        dichotomy_repeat = total_repeat // 2
        redundant = total_blocks % 64
        # separate src & des op to boost performance
        if dichotomy_repeat > threshold:
            self.tik_instance.vadd(64, self.ub_fm1_workbuf2, sum_ub, sum_ub[64], dichotomy_repeat, 1, 1, 1, 8, 16, 16)
            self.tik_instance.vadd(
                64, self.ub_fm1_workbuf2[total_blocks // 2], square_sum_ub, square_sum_ub[64], dichotomy_repeat,
                1, 1, 1, 8, 16, 16)
            self.tik_instance.pipe_barrier("PIPE_V")
            if redundant > 0:
                self.tik_instance.vadd(
                    16, self.ub_fm1_workbuf2, sum_ub[total_blocks // 64 * 64], sum_ub, redundant // 16,
                    1, 1, 1, 0, 2, 0)
                self.tik_instance.vadd(
                    16, self.ub_fm1_workbuf2[total_blocks // 2], square_sum_ub[total_blocks // 64 * 64], square_sum_ub,
                    redundant // 16, 1, 1, 1, 0, 2, 0)
                redundant = 0
                self.tik_instance.pipe_barrier("PIPE_V")
            dichotomy_repeat, total_repeat, ub_add_0 = self._dich_loop_add(dichotomy_repeat, threshold, total_blocks)

            self.tik_instance.vadd(64, sum_ub, ub_add_0[64], ub_add_0, dichotomy_repeat, 1, 1, 1, 8, 16, 16)
            self.tik_instance.vadd(
                64, square_sum_ub, ub_add_0[total_blocks // 2 + 64], ub_add_0[total_blocks // 2],
                dichotomy_repeat, 1, 1, 1, 8, 16, 16)
            self.tik_instance.pipe_barrier("PIPE_V")
            if total_repeat % 2 != 0:
                self.tik_instance.vadd(
                    64, sum_ub, ub_add_0[dichotomy_repeat * 64 * 2], sum_ub, total_repeat % 2, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vadd(
                    64, square_sum_ub, ub_add_0[total_blocks // 2 + dichotomy_repeat * 64 * 2], square_sum_ub,
                    total_repeat % 2, 1, 1, 1, 8, 8, 8)
                self.tik_instance.pipe_barrier("PIPE_V")
            total_repeat = dichotomy_repeat
            dichotomy_repeat = total_repeat // 2

        return total_repeat, dichotomy_repeat, redundant

    def dichotomy_sum_and_square_sum_handsync(self, sum_ub, square_sum_ub, total_blocks):
        """
        dichotomy sum and square sum
        """
        total_repeat, dichotomy_repeat, redundant = self.effecient_sum_and_square_sum_handsync(sum_ub, square_sum_ub,
                                                                                               total_blocks)
        while dichotomy_repeat > 0:
            self.tik_instance.vadd(64, sum_ub, sum_ub[dichotomy_repeat * 64], sum_ub,
                                   dichotomy_repeat, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vadd(64, square_sum_ub, square_sum_ub[dichotomy_repeat * 64], square_sum_ub,
                                   dichotomy_repeat, 1, 1, 1, 8, 8, 8)
            self.tik_instance.pipe_barrier("PIPE_V")
            if total_repeat % 2 != 0:
                self.tik_instance.vadd(64, sum_ub, sum_ub[dichotomy_repeat * 64 * 2], sum_ub, total_repeat % 2,
                                       1, 1, 1, 8, 8, 8)
                self.tik_instance.vadd(64, square_sum_ub, square_sum_ub[dichotomy_repeat * 64 * 2], square_sum_ub,
                                       total_repeat % 2, 1, 1, 1, 8, 8, 8)
                self.tik_instance.pipe_barrier("PIPE_V")
            total_repeat = dichotomy_repeat
            dichotomy_repeat = total_repeat // 2

        if total_repeat > 0:
            self.tik_instance.vadd(16, sum_ub, sum_ub[16], sum_ub, (total_repeat * 64 // 16 - 1), 1, 1, 1, 0, 2, 0)
            self.tik_instance.vadd(16, square_sum_ub, square_sum_ub[16], square_sum_ub, (total_repeat * 64 // 16 - 1),
                                   1, 1, 1, 0, 2, 0)
            self.tik_instance.pipe_barrier("PIPE_V")
        if redundant > 0:
            self.tik_instance.vadd(16, sum_ub, sum_ub[total_blocks // 64 * 64], sum_ub, redundant // 16,
                                   1, 1, 1, 0, 2, 0)
            self.tik_instance.vadd(16, square_sum_ub, square_sum_ub[total_blocks // 64 * 64], square_sum_ub,
                                   redundant // 16, 1, 1, 1, 0, 2, 0)
            self.tik_instance.pipe_barrier("PIPE_V")

    def calu_square_sum_handsync(self, fmap_ub, fp32_loop_times, fp32_redundant, fp32_repeat_times):
        """
        calcuate the square sum
        """
        self.calu_square_sum(fmap_ub, fp32_loop_times, fp32_redundant, fp32_repeat_times) 
        self.tik_instance.pipe_barrier("PIPE_V")

    def bn_update_process_handsync(self, axises):
        """
        bn update process in kernel process
        """
        k_redundant = axises["k_redundant"]
        # kh * kw * cin1 与cin1提公共轴，下面的默认逻辑是每次都只算1个cin1，不做特殊处理
        cin1_and_k_common_length = Bn2ReluV2Conv2dBn1NoRedundant.get_gcd(self.k_redundant_factor, self.channel_in1)

        cin1_and_k_single_jump = self.k_redundant_factor // cin1_and_k_common_length
        cin1_single_length = self.channel_in1 // cin1_and_k_common_length
        with self.tik_instance.if_scope(k_redundant % cin1_and_k_single_jump == 0):
            self.this_time_calu_length.set_as(self.preload_this_time_calu_length)
            with self.tik_instance.if_scope(self.this_time_calu_length > 0):
                self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 0)
                cin1_length_gap = k_redundant // cin1_and_k_single_jump
                with self.tik_instance.for_range(cin1_length_gap * cin1_single_length,
                                                 (cin1_length_gap + 1) * cin1_single_length,
                                                 name="cur_cin1") as cur_cin1:
                    self.bn_update_single_chanel_process(axises, cur_cin1)

    def bn_update_single_chanel_process(self, axises, cur_cin1):
        """
        process single chanel in bn update
        """
        batch_index = self.next_batch
        self.fm_ddr2ub_burst_length.set_as(self.preload_fm_ddr2ub_burst_length)
        self.fm_ddr2ub_src_stride.set_as(self.preload_fm_ddr2ub_src_stride)
        fm_ddr2ub_nburst = self.b_inner_factor
        fp32_loop_times = self.tik_instance.Scalar(name="fp32_loop_times",
                                                   init_value=self.this_time_calu_length * self.width_in * \
                                                              self.channel_in0 // (64 * 255))
        fp32_loop_redundant = self.tik_instance.Scalar(name="fp32_loop_redundant",
                                                       init_value=self.this_time_calu_length *
                                                       self.width_in * self.channel_in0 % (64 * 255))
        fp32_repeat = self.tik_instance.Scalar(name="fp32_repeat", init_value=fp32_loop_redundant // 64)
        fp32_redundant = self.tik_instance.Scalar(name="fp32_redundant", init_value=fp32_loop_redundant % 64)
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(0, fp32_loop_times, name="fp32_loop") as fp32_loop:
                self.tik_instance.vconv(64, "none", self.input_x_fp32_ub[fp32_loop * 255 * 64],
                                        self.input_x_ub[fp32_loop * 255 * 64], 255, 1, 1, 8, 4)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vconv(64, "none", self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                                    self.input_x_ub[fp32_loop_times * 255 * 64], fp32_repeat, 1, 1, 8, 4)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vconv(fp32_redundant, "none",
                                    self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                                    self.input_x_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64], 1, 1, 1, 8, 4)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE2", 0)
        self.tik_instance.vec_dup(64, self.scale_ub_64, 0, 1, 0)
        self.tik_instance.vec_dup(64, self.offset_ub_64, 0, 1, 0)
        self.tik_instance.vec_add(16, self.scale_ub_64, self.scale_ub_64, self.y_scale_ub[cur_cin1 * 16], 4, 2, 2, 0)
        self.tik_instance.vec_add(16, self.offset_ub_64, self.offset_ub_64,
                                  self.y_offset_ub[cur_cin1 * 16], 4, 2, 2, 0)
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(0, fp32_loop_times, name="loop_time") as loop_time:
                self.tik_instance.vmaddrelu(64, self.input_x_fp32_ub[loop_time * 255 * 64],
                                            self.scale_ub_64, self.offset_ub_64, 255, 1, 1, 1, 8, 0, 0)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vmaddrelu(64, self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                                        self.scale_ub_64, self.offset_ub_64, fp32_repeat, 1, 1, 1, 8, 0, 0)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vmaddrelu(fp32_redundant,
                                        self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                                        self.scale_ub_64, self.offset_ub_64, 1, 1, 1, 1, 8, 0, 0)
        # 这里开double buffer，防止vector流水中断
        with self.tik_instance.if_scope(self.bn_update_lastub_output_pingpong == 0):
            self._bn_update_ping([batch_index, cur_cin1, fm_ddr2ub_nburst, fp32_loop_times, fp32_redundant,
                                  fp32_repeat])
        with self.tik_instance.else_scope():
            self._bn_update_pong([batch_index, cur_cin1, fm_ddr2ub_nburst, fp32_loop_times, fp32_redundant,
                                  fp32_repeat])
        self.bn_update_lastub_output_pingpong.set_as(1 - self.bn_update_lastub_output_pingpong)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE2", 0)
        self._preload_next_input_x_in_conv(axises, cur_cin1, fm_ddr2ub_nburst)

    def calc_mask_in_bn_update_pong(self, batch_index, cur_cin1):
        """
        calculate the mask in bn_update pong
        """
        # 输出mask外提可以减少scala 操作
        with self.tik_instance.for_range(0, self.b_inner_factor, name="b_inner") as b_inner:
            input_x_mask_ub_length = self.tik_instance.Scalar(name="input_x_mask_ub_length",
                                                              init_value=(self.this_time_calu_length * \
                                                                          self.width_in + 15) // \
                                                                         16 * 16)
            mask_loop_times = self.tik_instance.Scalar(name="mask_loop_times",
                                                       init_value=self.this_time_calu_length * \
                                                                  self.width_in * self.channel_in0 // \
                                                                  (128 * 255))
            mask_loop_redundant = self.tik_instance.Scalar(name="mask_loop_redundant",
                                                           init_value=self.this_time_calu_length * \
                                                                      self.width_in * \
                                                                      self.channel_in0 % (128 * 255))
            mask_repeat = self.tik_instance.Scalar(name="mask_repeat",
                                                   init_value=mask_loop_redundant // 128)
            last_mask_repeat = self.tik_instance.Scalar(name="last_mask_repeat",
                                                        init_value=(mask_loop_redundant % 128 + \
                                                                    15) // 16)
            batch_start_pos = self.tik_instance.Scalar(name="batch_start_pos",
                                                       init_value=b_inner * self.this_time_calu_length *
                                                       self.width_in * self.channel_in0)
            with self.tik_instance.if_scope(mask_loop_times > 0):
                with self.tik_instance.for_range(0, mask_loop_times,
                                                 name="mask_loop") as mask_loop:
                    self.tik_instance.pipe_barrier("PIPE_V")
                    self.tik_instance.vcmpv_lt(
                        self.input_x_mask_ub[mask_loop * 255 * 8],
                        self.zeros_const_ub,
                        self.input_x_fp16_ub_pong[batch_start_pos + mask_loop * 255 * 128],
                        255,
                        1, 1, 0, 8)
            with self.tik_instance.if_scope(mask_repeat > 0):
                self.tik_instance.pipe_barrier("PIPE_V")
                self.tik_instance.vcmpv_lt(
                    self.input_x_mask_ub[mask_loop_times * 255 * 8],
                    self.zeros_const_ub,
                    self.input_x_fp16_ub_pong[batch_start_pos + mask_loop_times * 255 * 128],
                    mask_repeat,
                    1, 1, 0, 8)
            with self.tik_instance.if_scope(last_mask_repeat > 0):
                self.tik_instance.pipe_barrier("PIPE_V")
                self.tik_instance.vcmpv_lt(
                    self.input_x_mask_ub[mask_loop_times * 255 * 8 + mask_repeat * 8],
                    self.zeros_const_ub_16,
                    self.input_x_fp16_ub_pong[
                        batch_start_pos + mask_loop_times * 255 * 128 + mask_repeat * 128],
                    last_mask_repeat,
                    1, 1, 0, 1)
            self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 3)
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 3)
            mask_hin_offset = self.already_load_l1_h_length
            mask_real_length = self.tik_instance.Scalar(name="mask_real_length",
                                                        init_value=self.this_time_calu_length * \
                                                                   self.width_in)
            mask_burst_length = self.tik_instance.Scalar(name="mask_burst_length",
                                                         init_value=mask_real_length // 16)
            with self.tik_instance.if_scope(mask_real_length % 16 != 0):
                self.mask_tail_process_handsync([batch_index + b_inner, cur_cin1,
                                                 self.input_x_mask_ub,
                                                 mask_hin_offset, mask_real_length,
                                                 input_x_mask_ub_length, self.tem_ub_pong,
                                                 self.save_ub_pong])
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(
                    self.output_reluv2_mask[
                        batch_index + b_inner, cur_cin1, mask_hin_offset, 0],
                    self.input_x_mask_ub,
                    0,
                    1, mask_burst_length,
                    0, 0)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 3)

    def calc_mask_in_bn_update_ping(self, batch_index, cur_cin1):
        """
        calculate the mask in bn_update ping
        """
        # 输出mask外提可以减少scala 操作
        with self.tik_instance.for_range(0, self.b_inner_factor, name="b_inner") as b_inner:
            input_x_mask_ub_length = self.tik_instance.Scalar(name="input_x_mask_ub_length",
                                                              init_value=(self.this_time_calu_length * \
                                                                          self.width_in + 15) // \
                                                                         16 * 16)
            mask_loop_times = self.tik_instance.Scalar(name="mask_loop_times",
                                                       init_value=(self.this_time_calu_length *
                                                                   self.width_in *
                                                                   self.channel_in0) // (128 * 255))
            mask_loop_redundant = self.tik_instance.Scalar(name="mask_loop_redundant",
                                                           init_value=self.this_time_calu_length * \
                                                                      self.width_in * \
                                                                      self.channel_in0 % (128 * 255))
            mask_repeat = self.tik_instance.Scalar(name="mask_repeat",
                                                   init_value=mask_loop_redundant // 128)
            last_mask_repeat = self.tik_instance.Scalar(name="last_mask_repeat",
                                                        init_value=(mask_loop_redundant % 128 + \
                                                                    15) // 16)
            batch_start_pos = self.tik_instance.Scalar(name="batch_start_pos",
                                                       init_value=b_inner *
                                                       self.this_time_calu_length * self.width_in * self.channel_in0)
            with self.tik_instance.if_scope(mask_loop_times > 0):
                with self.tik_instance.for_range(0, mask_loop_times,
                                                 name="mask_loop") as mask_loop:
                    self.tik_instance.pipe_barrier("PIPE_V")
                    self.tik_instance.vcmpv_lt(
                        self.input_x_mask_ub_pong[mask_loop * 255 * 8],
                        self.zeros_const_ub,
                        self.input_x_fp16_ub[batch_start_pos + mask_loop * 255 * 128],
                        255,
                        1, 1, 0, 8)
            with self.tik_instance.if_scope(mask_repeat > 0):
                self.tik_instance.pipe_barrier("PIPE_V")
                self.tik_instance.vcmpv_lt(
                    self.input_x_mask_ub_pong[mask_loop_times * 255 * 8],
                    self.zeros_const_ub,
                    self.input_x_fp16_ub[batch_start_pos + mask_loop_times * 255 * 128],
                    mask_repeat,
                    1, 1, 0, 8)
            with self.tik_instance.if_scope(last_mask_repeat > 0):
                self.tik_instance.pipe_barrier("PIPE_V")
                self.tik_instance.vcmpv_lt(
                    self.input_x_mask_ub_pong[mask_loop_times * 255 * 8 + mask_repeat * 8],
                    self.zeros_const_ub_16,
                    self.input_x_fp16_ub[
                        batch_start_pos + mask_loop_times * 255 * 128 + mask_repeat * 128],
                    last_mask_repeat,
                    1, 1, 0, 1)
            self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 2)
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 2)
            mask_hin_offset = self.already_load_l1_h_length
            mask_real_length = self.tik_instance.Scalar(name="mask_real_length",
                                                        init_value=self.this_time_calu_length * \
                                                                   self.width_in)
            mask_burst_length = self.tik_instance.Scalar(name="mask_burst_length",
                                                         init_value=mask_real_length // 16)
            with self.tik_instance.if_scope(mask_real_length % 16 != 0):
                self.mask_tail_process_handsync([batch_index + b_inner, cur_cin1,
                                                 self.input_x_mask_ub_pong, mask_hin_offset,
                                                 mask_real_length, input_x_mask_ub_length,
                                                 self.tem_ub, self.save_ub])
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(
                    self.output_reluv2_mask[
                        batch_index + b_inner, cur_cin1, mask_hin_offset, 0],
                    self.input_x_mask_ub_pong,
                    0,
                    1, mask_burst_length,
                    0, 0)
        self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 2)

    def mad_process_handsync(self, axises):
        """
        mmad process
        """
        k_redundant = axises["k_redundant"]
        b_inner = axises["b_inner"]
        dst_offset = b_inner
        src_offset = b_inner
        with self.tik_instance.if_scope(self.l0a_pingpong == 0):
            self.tik_instance.wait_flag("PIPE_MTE1", "PIPE_M", 0)
        with self.tik_instance.else_scope():
            self.tik_instance.wait_flag("PIPE_MTE1", "PIPE_M", 1)
        with self.tik_instance.if_scope(k_redundant == 0):
            with self.tik_instance.if_scope(self.l0c_pingpong == 0):
                self.tik_instance.wait_flag("PIPE_V", "PIPE_M", 0)
            with self.tik_instance.else_scope():
                self.tik_instance.wait_flag("PIPE_V", "PIPE_M", 1)
        with self.tik_instance.if_scope(self.l0c_pingpong == 0):
            self._mad_l0c_ping(dst_offset, k_redundant, src_offset)
        with self.tik_instance.else_scope():
            self._mad_l0c_pong(dst_offset, k_redundant, src_offset)
        with self.tik_instance.if_scope(self.l0b_pingpong == 0):
            self.tik_instance.set_flag("PIPE_M", "PIPE_MTE1", 0)
        with self.tik_instance.else_scope():
            self.tik_instance.set_flag("PIPE_M", "PIPE_MTE1", 1)
        with self.tik_instance.if_scope(k_redundant == (self.k_redundant_factor - 1)):
            with self.tik_instance.if_scope(self.l0c_pingpong == 0):
                self.tik_instance.set_flag("PIPE_M", "PIPE_V", 0)
            with self.tik_instance.else_scope():
                self.tik_instance.set_flag("PIPE_M", "PIPE_V", 1)

    def _preload_next_input_x_in_conv(self, axises, cur_cin1, fm_ddr2ub_nburst):
        """
        preload next input x in conv
        """
        with self.tik_instance.if_scope(cur_cin1 + 1 == self.channel_in1):
            with self.tik_instance.if_scope(self.already_load_l1_h_length + \
                                            self.this_time_calu_length <= self.height_in + self.pad_t):
                self.already_load_l1_h_length.set_as(
                    self.this_time_calu_length + self.already_load_l1_h_length)
            with self.tik_instance.else_scope():
                self.already_load_l1_h_length.set_as(self.height_in)
            m_redundant = axises['m_redundant']
            with self.tik_instance.if_scope((m_redundant + 2) == self.m_redundant_factor):
                # 下一个batch的
                self.preload_next_hin_l1_start_pos.set_as(-self.pad_t)
                self.preload_next_batch.set_as(self.preload_next_batch + 1)
                self.preload_already_load_l1_h_length.set_as(0)
            with self.tik_instance.else_scope():
                self.preload_already_load_l1_h_length.set_as(self.already_load_l1_h_length)
                next_ho_index = self.tik_instance.Scalar(name="next_ho_index", init_value=0)
                with self.tik_instance.if_scope((m_redundant + 1) == self.m_redundant_factor):
                    next_ho_index.set_as(self.m_l0a_factor * 16 // self.width_out)
                with self.tik_instance.else_scope():
                    next_ho_index.set_as(
                        (m_redundant + 2) * self.m_l0a_factor * 16 // self.width_out)
                next_hin_index = next_ho_index * self.stride_h
                self.preload_next_hin_l1_start_pos.set_as(next_hin_index - self.pad_t)
            self.preload_this_time_calu_length.set_as(self.preload_next_hin_l1_start_pos +
                                                      self.get_conv_h_length() -
                                                      self.preload_already_load_l1_h_length)

            with self.tik_instance.if_scope((self.preload_already_load_l1_h_length + \
                                             self.preload_this_time_calu_length) > self.height_in):
                self.preload_this_time_calu_length.set_as(
                    self.height_in - self.preload_already_load_l1_h_length)
            self.preload_fm_ddr2ub_burst_length.set_as(
                self.preload_this_time_calu_length * self.width_in)
            self.preload_fm_ddr2ub_src_stride.set_as(self.b_inner_factor * self.height_in * \
                                                     self.width_in - self.preload_this_time_calu_length * \
                                                     self.width_in)
            self.tik_instance.pipe_barrier("PIPE_MTE2")
            self.tik_instance.data_move(
                self.input_x_ub,
                self.fmap_input[self.preload_next_batch, 0,
                                self.preload_already_load_l1_h_length, 0, 0],
                0,
                self.b_inner_factor,
                self.preload_fm_ddr2ub_burst_length,
                self.preload_fm_ddr2ub_src_stride, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.pipe_barrier("PIPE_MTE2")
            self.tik_instance.data_move(
                self.input_x_ub,
                self.fmap_input[self.preload_next_batch, cur_cin1 + 1,
                                self.preload_already_load_l1_h_length, 0, 0],
                0,
                fm_ddr2ub_nburst, self.preload_fm_ddr2ub_burst_length,
                self.preload_fm_ddr2ub_src_stride, 0)
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)
        with self.tik_instance.if_scope(cur_cin1 + 1 == self.channel_in1):
            self.tik_instance.set_flag("PIPE_MTE3", "PIPE_MTE1", 0)

    def _bn_update_pong(self, input_list):
        """
        bn update pong
        """
        batch_index, cur_cin1, fm_ddr2ub_nburst, fp32_loop_times, fp32_redundant, fp32_repeat = input_list
        self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 3)
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(
                    0, fp32_loop_times, name="fp32_loop") as fp32_loop:
                self.tik_instance.vconv(
                    64,
                    "none",
                    self.input_x_fp16_ub_pong[fp32_loop * 255 * 64],
                    self.input_x_fp32_ub[fp32_loop * 255 * 64],
                    255,
                    1, 1, 4, 8)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vconv(
                64,
                "none",
                self.input_x_fp16_ub_pong[fp32_loop_times * 255 * 64],
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                fp32_repeat,
                1, 1, 4, 8)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vconv(
                fp32_redundant,
                "none",
                self.input_x_fp16_ub_pong[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                1,
                1, 1, 4, 8)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 3)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 3)
        # 将数据搬到L1
        self.tik_instance.data_move(
            self.fmap_l1[0, cur_cin1, self.already_load_l1_h_length, 0, 0],
            self.input_x_fp16_ub_pong,
            0,
            fm_ddr2ub_nburst, self.fm_ddr2ub_burst_length,
            0, self.fm_ddr2ub_src_stride)
        self.tik_instance.pipe_barrier("PIPE_MTE3")
        self.tik_instance.data_move(
            self.output_relu[batch_index, cur_cin1, self.already_load_l1_h_length, 0, 0],
            self.input_x_fp16_ub_pong,
            0,
            fm_ddr2ub_nburst, self.fm_ddr2ub_burst_length, 0, self.fm_ddr2ub_src_stride)
        self.calc_mask_in_bn_update_pong(batch_index, cur_cin1)

    def _bn_update_ping(self, input_list):
        """
        bn_update ping
        """
        batch_index, cur_cin1, fm_ddr2ub_nburst, fp32_loop_times, fp32_redundant, fp32_repeat = input_list
        self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 2)
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(
                    0, fp32_loop_times, name="fp32_loop") as fp32_loop:
                self.tik_instance.vconv(
                    64,
                    "none",
                    self.input_x_fp16_ub[fp32_loop * 255 * 64],
                    self.input_x_fp32_ub[fp32_loop * 255 * 64],
                    255,
                    1, 1, 4, 8)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vconv(
                64,
                "none",
                self.input_x_fp16_ub[fp32_loop_times * 255 * 64],
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                fp32_repeat,
                1, 1, 4, 8)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vconv(
                fp32_redundant,
                "none",
                self.input_x_fp16_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                1,
                1, 1, 4, 8)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 2)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 2)
        # 将数据搬到L1
        self.tik_instance.data_move(
            self.fmap_l1[0, cur_cin1, self.already_load_l1_h_length, 0, 0],
            self.input_x_fp16_ub,
            0,
            fm_ddr2ub_nburst, self.fm_ddr2ub_burst_length,
            0, self.fm_ddr2ub_src_stride)
        self.tik_instance.pipe_barrier("PIPE_MTE3")
        self.tik_instance.data_move(
            self.output_relu[batch_index, cur_cin1, self.already_load_l1_h_length, 0, 0],
            self.input_x_fp16_ub,
            0,
            fm_ddr2ub_nburst, self.fm_ddr2ub_burst_length, 0, self.fm_ddr2ub_src_stride)
        self.calc_mask_in_bn_update_ping(batch_index, cur_cin1)

    def _load_l0a_pong(self, input_list):
        """
        load fmap from l1 to l0a pong
        """
        b_inner, batch_index, cin1_l0_index, kh_l0_index, kw_l0_index, pad, wo_l0_index = input_list
        # 性能优化，按搬移次数少的来搬
        with self.tik_instance.if_scope(self.m_l0a_factor < self.k_l0a_factor):
            with self.tik_instance.for_range(0, self.m_l0a_factor, name="m_l0") as m_l0:
                dst_b_offset = b_inner
                dst_m_offset = m_l0
                src_b_offset = b_inner + batch_index
                current_w_offset = m_l0 * self.hw_out0 % self.width_out
                self.tik_instance.pipe_barrier("PIPE_MTE1")
                self.tik_instance.load3dv1(
                    self.fmap_l0a_pong[dst_b_offset, dst_m_offset, 0, 0, 0],
                    self.fmap_l1[src_b_offset, 0, 0, 0, 0],
                    pad,
                    self.height_in,
                    self.width_in,
                    cin1_l0_index,
                    kw_l0_index,
                    kh_l0_index,
                    (wo_l0_index + current_w_offset) * self.stride_w - self.pad_l,
                    self.hin_l1_start_pos,
                    self.stride_w,
                    self.stride_h,
                    self.kernel_w,
                    self.kernel_h,
                    self.dilation_w,
                    self.dilation_w,
                    1,
                    0,
                    self.k_l0a_factor,
                    0,
                    0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.k_l0a_factor, name="k_l0") as k_l0:
                dst_b_offset = b_inner
                dst_k_offset = k_l0
                src_b_offset = batch_index + b_inner
                current_c1_offset = k_l0 // (self.kernel_h * self.kernel_w)
                current_kh_offset = k_l0 % (self.kernel_h * self.kernel_w) // self.kernel_w
                current_kw_offset = k_l0 % (self.kernel_h * self.kernel_w) % self.kernel_w
                self.tik_instance.pipe_barrier("PIPE_MTE1")
                self.tik_instance.load3dv1(
                    self.fmap_l0a_pong[dst_b_offset, 0, dst_k_offset, 0, 0],
                    self.fmap_l1[src_b_offset, 0, 0, 0, 0],
                    pad,
                    self.height_in,
                    self.width_in,
                    cin1_l0_index + current_c1_offset,
                    kw_l0_index + current_kw_offset,
                    kh_l0_index + current_kh_offset,
                    wo_l0_index * self.stride_w - self.pad_l,
                    self.hin_l1_start_pos,
                    self.stride_w,
                    self.stride_h,
                    self.kernel_w,
                    self.kernel_h,
                    self.dilation_w,
                    self.dilation_h,
                    self.k_l0a_factor,
                    1,
                    self.m_l0a_factor,
                    0,
                    0)
        self.tik_instance.set_flag("PIPE_MTE1", "PIPE_M", 1)

    def _load_l0a_ping(self, input_list):
        """
        load fmap from l1 to l0a ping
        """
        b_inner, batch_index, cin1_l0_index, kh_l0_index, kw_l0_index, pad, wo_l0_index = input_list
        # 性能优化，按搬移次数少的来搬
        with self.tik_instance.if_scope(self.m_l0a_factor < self.k_l0a_factor):
            with self.tik_instance.for_range(0, self.m_l0a_factor, name="m_l0") as m_l0:
                dst_b_offset = b_inner
                dst_m_offset = m_l0
                src_b_offset = b_inner + batch_index
                current_w_offset = m_l0 * self.hw_out0 % self.width_out
                self.tik_instance.pipe_barrier("PIPE_MTE1")
                self.tik_instance.load3dv1(
                    self.fmap_l0a_ping[dst_b_offset, dst_m_offset, 0, 0, 0],
                    self.fmap_l1[src_b_offset, 0, 0, 0, 0],
                    pad,
                    self.height_in,
                    self.width_in,
                    cin1_l0_index,
                    kw_l0_index,
                    kh_l0_index,
                    (wo_l0_index + current_w_offset) * self.stride_w - self.pad_l,
                    self.hin_l1_start_pos,
                    self.stride_w,
                    self.stride_h,
                    self.kernel_w,
                    self.kernel_h,
                    self.dilation_w,
                    self.dilation_w,
                    1,
                    0,
                    self.k_l0a_factor,
                    0,
                    0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.k_l0a_factor, name="k_l0") as k_l0:
                dst_b_offset = b_inner
                dst_k_offset = k_l0
                src_b_offset = batch_index + b_inner
                current_c1_offset = k_l0 // (self.kernel_h * self.kernel_w)
                current_kh_offset = k_l0 % (self.kernel_h * self.kernel_w) // self.kernel_w
                current_kw_offset = k_l0 % (self.kernel_h * self.kernel_w) % self.kernel_w
                self.tik_instance.pipe_barrier("PIPE_MTE1")
                self.tik_instance.load3dv1(
                    self.fmap_l0a_ping[dst_b_offset, 0, dst_k_offset, 0, 0],
                    self.fmap_l1[src_b_offset, 0, 0, 0, 0],
                    pad,
                    self.height_in,
                    self.width_in,
                    cin1_l0_index + current_c1_offset,
                    kw_l0_index + current_kw_offset,
                    kh_l0_index + current_kh_offset,
                    wo_l0_index * self.stride_w - self.pad_l,
                    self.hin_l1_start_pos,
                    self.stride_w,
                    self.stride_h,
                    self.kernel_w,
                    self.kernel_h,
                    self.dilation_w,
                    self.dilation_h,
                    self.k_l0a_factor,
                    1,
                    self.m_l0a_factor,
                    0,
                    0)
        self.tik_instance.set_flag("PIPE_MTE1", "PIPE_M", 0)

    def _preload_input_x_handsync(self):
        """
        preload input for bn update in kernel process
        """
        next_conv_need_length = self.get_conv_h_length()
        next_ho_index = self.m_l0a_factor * 16 // self.width_out
        next_hin_index = next_ho_index * self.stride_h
        h1_start_pos = next_hin_index - self.pad_t
        self.preload_next_hin_l1_start_pos.set_as(h1_start_pos)
        self.preload_this_time_calu_length.set_as(
            h1_start_pos + next_conv_need_length - self.already_load_l1_h_length)
        with self.tik_instance.if_scope((self.preload_this_time_calu_length +
                                         self.already_load_l1_h_length) > self.height_in):
            self.preload_this_time_calu_length.set_as(
                self.height_in - self.already_load_l1_h_length)
        self.preload_fm_ddr2ub_burst_length.set_as(
            self.preload_this_time_calu_length * self.width_in)
        self.preload_fm_ddr2ub_src_stride.set_as(
            self.b_inner_factor * self.height_in * self.width_in - self.preload_this_time_calu_length * self.width_in)

        self.tik_instance.data_move(
            self.input_x_ub,
            self.fmap_input[self.preload_next_batch, 0,
                            self.preload_already_load_l1_h_length, 0, 0],
            0,
            self.b_inner_factor,
            self.preload_fm_ddr2ub_burst_length,
            self.preload_fm_ddr2ub_src_stride, 0)
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)

    def _first_bn_update_process_handsync(self, axises):
        """
        first bn update
        """
        b_block = axises.get("b_block")
        pre_core_process_batch = axises.get("pre_core_process_batch")
        batch_index = b_block * self.b_redundant_factor + pre_core_process_batch
        self.already_load_l1_h_length.set_as(self.get_max_h_length())

        with self.tik_instance.if_scope(
                self.pad_t + self.already_load_l1_h_length > self.height_in):
            self.already_load_l1_h_length.set_as(self.height_in)
        fm_ddr2ub_nburst = self.b_inner_factor
        fm_ddr2ub_burst_length = self.already_load_l1_h_length * self.width_in
        fm_ddr2ub_src_stride = self.b_inner_factor * self.height_in * self.width_in - \
                               self.already_load_l1_h_length * self.width_in

        # inner batch 场景下直接跳读
        self.tik_instance.data_move(
            self.input_x_ub,
            self.fmap_input[batch_index, 0, 0, 0, 0],
            0,
            fm_ddr2ub_nburst,
            fm_ddr2ub_burst_length,
            fm_ddr2ub_src_stride,
            0)  # float32 多一倍 block
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)
        self.tik_instance.pipe_barrier("PIPE_MTE2")
        with self.tik_instance.for_range(0, self.channel_in1, name="cin1") as cin1:
            self._first_bn_update_single_chanel_process([batch_index, cin1, fm_ddr2ub_burst_length, fm_ddr2ub_nburst,
                                                         fm_ddr2ub_src_stride])

    def _first_bn_update_single_chanel_process(self, input_list):
        """
        first bn update process on single chanel
        """
        batch_index, cin1, fm_ddr2ub_burst_length, fm_ddr2ub_nburst, fm_ddr2ub_src_stride = input_list
        fp32_loop_times = self.b_inner_factor * self.already_load_l1_h_length * self.width_in * \
                          self.channel_in0 // (64 * 255)
        fp32_loop_redundant = self.b_inner_factor * self.already_load_l1_h_length * self.width_in * \
                              self.channel_in0 % (64 * 255)
        fp32_repeat = fp32_loop_redundant // 64
        fp32_redundant = fp32_loop_redundant % 64
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 0)
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(0, fp32_loop_times, name="fp32_loop") as fp32_loop:
                self.tik_instance.vconv(
                    64,
                    "none",
                    self.input_x_fp32_ub[fp32_loop * 255 * 64],
                    self.input_x_ub[fp32_loop * 255 * 64],
                    255,
                    1, 1, 8, 4)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vconv(
                64,
                "none",
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                self.input_x_ub[fp32_loop_times * 255 * 64],
                fp32_repeat,
                1, 1, 8, 4)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vconv(
                fp32_redundant,
                "none",
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                self.input_x_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                1,
                1, 1, 8, 4)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE2", 0)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE2", 0)
        # preload fm ddr 2 ub
        with self.tik_instance.if_scope((cin1 + 1) < self.channel_in1):
            self.tik_instance.pipe_barrier("PIPE_MTE2")
            self.tik_instance.data_move(
                self.input_x_ub,
                self.fmap_input[batch_index, cin1 + 1, 0, 0, 0],
                0,
                fm_ddr2ub_nburst,
                fm_ddr2ub_burst_length,
                fm_ddr2ub_src_stride,
                0)
            self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)
        self.tik_instance.vec_dup(64, self.scale_ub_64, 0, 1, 0)
        self.tik_instance.vec_dup(64, self.offset_ub_64, 0, 1, 0)
        self.tik_instance.vec_add(16, self.scale_ub_64, self.scale_ub_64,
                                  self.y_scale_ub[cin1 * 16], 4, 2, 2, 0)
        self.tik_instance.vec_add(16, self.offset_ub_64, self.offset_ub_64,
                                  self.y_offset_ub[cin1 * 16], 4, 2, 2, 0)
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(
                    0, fp32_loop_times, name="loop_time") as loop_time:
                self.tik_instance.vmaddrelu(64, self.input_x_fp32_ub[loop_time * 255 * 64],
                                            self.scale_ub_64,
                                            self.offset_ub_64, 255, 1, 1, 1, 8, 0, 0)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vmaddrelu(64, self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                                        self.scale_ub_64,
                                        self.offset_ub_64, fp32_repeat, 1, 1, 1, 8, 0, 0)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vmaddrelu(fp32_redundant,
                                        self.input_x_fp32_ub[
                                            fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                                        self.scale_ub_64,
                                        self.offset_ub_64, 1, 1, 1, 1, 8, 0, 0)
        with self.tik_instance.if_scope(cin1 > 0):
            self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 0)
        with self.tik_instance.if_scope(fp32_loop_times > 0):
            with self.tik_instance.for_range(
                    0, fp32_loop_times, name="fp32_loop") as fp32_loop:
                self.tik_instance.vconv(
                    64,
                    "none",
                    self.input_x_fp16_ub[fp32_loop * 255 * 64],
                    self.input_x_fp32_ub[fp32_loop * 255 * 64],
                    255,
                    1, 1, 4, 8)
        with self.tik_instance.if_scope(fp32_repeat > 0):
            self.tik_instance.vconv(
                64,
                "none",
                self.input_x_fp16_ub[fp32_loop_times * 255 * 64],
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64],
                fp32_repeat,
                1, 1, 4, 8)
        with self.tik_instance.if_scope(fp32_redundant > 0):
            self.tik_instance.vconv(
                fp32_redundant,
                "none",
                self.input_x_fp16_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                self.input_x_fp32_ub[fp32_loop_times * 255 * 64 + fp32_repeat * 64],
                1,
                1, 1, 4, 8)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
        self._fetch_first_bn_update_single_chanel_process([batch_index, cin1, fm_ddr2ub_burst_length, fm_ddr2ub_nburst,
                                                           fm_ddr2ub_src_stride])

    def _fetch_first_bn_update_single_chanel_process(self, input_list):
        """
        fetch first bn update single chanel process
        """
        batch_index, cin1, fm_ddr2ub_burst_length, fm_ddr2ub_nburst, fm_ddr2ub_src_stride = input_list
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.pipe_barrier("PIPE_MTE3")
        # move relu result
        self.tik_instance.data_move(
            self.output_relu[batch_index, cin1, 0, 0, 0],
            self.input_x_fp16_ub,
            0,
            fm_ddr2ub_nburst,
            fm_ddr2ub_burst_length, 0, fm_ddr2ub_src_stride)
        # 将数据搬到L1
        self.tik_instance.data_move(
            self.fmap_l1[0, cin1, 0, 0, 0],
            self.input_x_fp16_ub,
            0,
            fm_ddr2ub_nburst,
            fm_ddr2ub_burst_length,
            0, fm_ddr2ub_src_stride)
        with self.tik_instance.if_scope(cin1 < (self.channel_in1 - 1)):
            self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 0)
        self._calc_mask_in_first_bn_update(batch_index, cin1)

    def _calc_mask_in_first_bn_update(self, batch_index, cin1):
        """
        calculate mask in first bn update
        """
        # 输出mask外提可以减少scala 操作
        with self.tik_instance.for_range(0, self.b_inner_factor, "inner_batch") as b_inner:
            input_x_mask_ub_length = (self.already_load_l1_h_length * self.width_in + 15) // 16 * 16
            mask_loop_times = (self.already_load_l1_h_length * self.width_in *
                               self.channel_in0) // \
                              (128 * 255)
            mask_loop_redundant = (self.already_load_l1_h_length * self.width_in *
                                   self.channel_in0) % (128 * 255)
            mask_repeat = mask_loop_redundant // 128
            # channel_in0在，所以一定能被16除尽
            last_mask_repeat = (mask_loop_redundant % 128 + 15) // 16
            batch_start_pos = b_inner * self.already_load_l1_h_length * self.width_in * self.channel_in0
            with self.tik_instance.if_scope(cin1 > 0):
                self.tik_instance.wait_flag("PIPE_MTE3", "PIPE_V", 2)
            with self.tik_instance.if_scope(mask_loop_times > 0):
                self.tik_instance.pipe_barrier("PIPE_V")
                with self.tik_instance.for_range(0, mask_loop_times,
                                                 name="mask_loop") as mask_loop:
                    self.tik_instance.pipe_barrier("PIPE_V")
                    self.tik_instance.vcmpv_lt(
                        self.input_x_mask_ub[mask_loop * 255 * 8],
                        self.zeros_const_ub,
                        self.input_x_fp16_ub[batch_start_pos + mask_loop * 255 * 128],
                        255,
                        1, 1, 0, 8)
            with self.tik_instance.if_scope(mask_repeat > 0):
                self.tik_instance.pipe_barrier("PIPE_V")
                self.tik_instance.vcmpv_lt(
                    self.input_x_mask_ub[mask_loop_times * 255 * 8],
                    self.zeros_const_ub,
                    self.input_x_fp16_ub[batch_start_pos + mask_loop_times * 255 * 128],
                    mask_repeat,
                    1, 1, 0, 8)
            with self.tik_instance.if_scope(last_mask_repeat > 0):
                self.tik_instance.pipe_barrier("PIPE_V")
                self.tik_instance.vcmpv_lt(
                    self.input_x_mask_ub[mask_loop_times * 255 * 8 + mask_repeat * 8],
                    self.zeros_const_ub_16,
                    self.input_x_fp16_ub[
                        batch_start_pos + mask_loop_times * 255 * 128 + mask_repeat * 128],
                    last_mask_repeat,
                    1, 1, 0, 1)
            mask_real_length = self.already_load_l1_h_length * self.width_in
            mask_burst_length = self.already_load_l1_h_length * self.width_in // 16

            mask_hin_offset = 0
            self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
            self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
            with self.tik_instance.if_scope(mask_real_length % 16 != 0):
                self.mask_tail_process_handsync([batch_index + b_inner, cin1,
                                                 self.input_x_mask_ub,
                                                 mask_hin_offset, mask_real_length,
                                                 input_x_mask_ub_length, self.tem_ub,
                                                 self.save_ub])
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(
                    self.output_reluv2_mask[batch_index + b_inner, cin1, mask_hin_offset, 0],
                    self.input_x_mask_ub,
                    0,
                    1, mask_burst_length,
                    0, 0)
            with self.tik_instance.if_scope(cin1 < (self.channel_in1 - 1)):
                self.tik_instance.set_flag("PIPE_MTE3", "PIPE_V", 2)

    def _bn2_mean_varience_process_handsync(self):
        """
        prepare process to calucate the scale and offset
        """
        start_pos = 0
        elem_num = self.channel_in1 * self.channel_in0
        burst_length = elem_num // 8
        num = self.batch * self.height_in * self.width_in
        num_rec = 1.0 / num

        redundant, repeat = self._first_part_bn_mean_varience_process(burst_length, elem_num, num_rec, start_pos)

        self._second_part_bn_mean_varience_process(burst_length, redundant, repeat, start_pos)
        factor_reverse = self._third_part_bn_mean_varience_process_sub(burst_length, redundant, repeat, start_pos)

        self._fourth_part_bn_mean_varience_process([burst_length, factor_reverse, num, redundant, repeat, start_pos])

    def _fourth_part_bn_mean_varience_process(self, input_list):
        """
        fourth part of calculate mean and varience
        """
        # 这一块计算逻辑不知道为什么要加上去，理论上不需要 乘以 batch_var_scalar，
        # 为了和库上的bn_training_update 保持一致，有可能会有精度抖动
        burst_length, factor_reverse, num, redundant, repeat, start_pos = input_list
        if num == 1:
            batch_var_scalar = 0.0
        else:
            batch_var_scalar = num / (num - 1)
        if repeat >= 1:
            self.tik_instance.vmuls(
                64,
                self.variance_batch_ub,
                self.varience_ub,
                batch_var_scalar,
                repeat,
                1, 1, 8, 8)  # float32 一个repeat 64个元素
        if redundant >= 1:
            self.tik_instance.vmuls(
                redundant,
                self.variance_batch_ub[repeat * 64],
                self.varience_ub[repeat * 64],
                batch_var_scalar,
                1,
                1, 1, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vmuls(
                64,
                self.variance_mul_ub,
                self.variance_batch_ub,
                self.factor,
                repeat,
                1, 1, 8, 8)  # float32 一个repeat 64个元素
        if redundant >= 1:
            self.tik_instance.vmuls(
                redundant,
                self.variance_mul_ub[repeat * 64],
                self.variance_batch_ub[repeat * 64],
                self.factor,
                1,
                1, 1, 8, 8)
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 1)
        if repeat >= 1:
            self.tik_instance.vmuls(
                64,
                self.variance_mul_rev_ub,
                self.pre_moving_variance_ub,
                factor_reverse,
                repeat,
                1, 1, 8, 8)  # float32 一个repeat 64个元素
        if redundant >= 1:
            self.tik_instance.vmuls(
                redundant * 8,
                self.variance_mul_rev_ub[repeat * 64],
                self.pre_moving_variance_ub[repeat * 64],
                factor_reverse,
                1,
                1, 1, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vadd(
                64,
                self.moving_variance_ub,
                self.variance_mul_ub,
                self.variance_mul_rev_ub,  # float32 一个repeat 64个元素
                repeat,
                1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vadd(
                redundant * 8,
                self.moving_variance_ub[repeat * 64],
                self.variance_mul_ub[repeat * 64],
                self.variance_mul_rev_ub[repeat * 64],
                1,
                1, 1, 1, 8, 8, 8)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 3)
        dup_scalar = self.tik_instance.Scalar(
            dtype="float16",
            name="dup_zero_scalar",
            init_value=0)
        self.tik_instance.vector_dup(
            128,
            self.zeros_const_ub,
            dup_scalar,
            1,
            1, 1)
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.data_move(
            self.output_mean[start_pos],
            self.mean_ub,
            0,
            1, burst_length,
            0, 0)  # float32 多一倍 block
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 1)
        self.tik_instance.data_move(
            self.output_varience[start_pos],
            self.varience_ub,
            0,
            1, burst_length,
            0, 0)  # float32 多一倍 block
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 2)
        self.tik_instance.data_move(
            self.output_moving_mean[start_pos],
            self.moving_mean_ub,
            0,
            1, burst_length, 0, 0)  # float32 多一倍 block
        self.tik_instance.wait_flag("PIPE_V", "PIPE_MTE3", 3)
        self.tik_instance.data_move(
            self.output_moving_varience[start_pos],
            self.moving_variance_ub,
            0,
            1, burst_length, 0, 0)  # float32 多一倍 block

    def _second_part_bn_mean_varience_process(self, burst_length, redundant, repeat, start_pos):
        """
        second part of calculate mean and varience
        """
        # 计算整体的缩放和平移参数
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 1)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vadds(
                64,
                self.y_scale_add_ub,
                self.varience_ub,  # float32 一个repeat 64个元素
                self.epsilon,
                repeat,
                1, 1, 8, 8)
        if redundant >= 1:
            self.tik_instance.vadds(
                redundant,
                self.y_scale_add_ub[repeat * 64],
                self.varience_ub[repeat * 64],
                self.epsilon,
                1,
                1, 1, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vsqrt(
                64,
                self.y_scale_sqrt_ub,
                self.y_scale_add_ub,  # float32 一个repeat 64个元素
                repeat,
                1, 1, 8, 8)
        if redundant >= 1:
            self.tik_instance.vsqrt(
                redundant,
                self.y_scale_sqrt_ub[repeat * 64],
                self.y_scale_add_ub[repeat * 64],
                1,
                1, 1, 8, 8)
        self.tik_instance.data_move(
            self.offset_ub,
            self.offset_input[start_pos],
            0,
            1, burst_length,
            0, 0)  # float32 多一倍 block
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 1)
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 0)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vdiv(
                64,
                self.y_scale_ub,
                self.scale_ub,
                self.y_scale_sqrt_ub,  # float32 一个repeat 64个元素
                repeat, 1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vdiv(
                redundant,
                self.y_scale_ub[repeat * 64],
                self.scale_ub[repeat * 64],
                self.y_scale_sqrt_ub[repeat * 64],
                1,
                1, 1, 1, 8, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vmul(
                64,
                self.y_offset_mul_ub,
                self.mean_ub,
                self.y_scale_ub,  # float32 一个repeat 64个元素
                repeat,
                1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vmul(
                redundant,
                self.y_offset_mul_ub[repeat * 64],
                self.mean_ub[repeat * 64],
                self.y_scale_ub[repeat * 64],
                1,
                1, 1, 1, 8, 8, 8)
        self.tik_instance.data_move(
            self.pre_moving_mean_ub,
            self.pre_moving_mean_input[start_pos],
            0,
            1, burst_length,
            0, 0)  # float32 多一倍 block
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 1)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vsub(
                64,
                self.y_offset_ub,
                self.offset_ub,
                self.y_offset_mul_ub,  # float32 一个repeat 64个元素
                repeat,
                1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vsub(
                redundant,
                self.y_offset_ub[repeat * 64],
                self.offset_ub[repeat * 64],
                self.y_offset_mul_ub[repeat * 64],
                1,
                1, 1, 1, 8, 8, 8)

    def _first_part_bn_mean_varience_process(self, burst_length, elem_num, num_rec, start_pos):
        """
        first part of calculate mean and varience
        """
        self.tik_instance.data_move(
            self.sum_input_ub,
            self.sum_input[start_pos],
            0, 1, burst_length, 0, 0)  # float32 多一倍 block
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)
        self.tik_instance.data_move(
            self.square_sum_input_ub,
            self.square_sum_input[start_pos],
            0,
            1, burst_length,
            0, 0)
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 1)
        repeat = elem_num // 64
        redundant = elem_num % 64
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 0)
        if repeat >= 1:
            self.tik_instance.vmuls(
                64,
                self.mean_ub,
                self.sum_input_ub,  # float32 一个repeat 64个元素
                num_rec,
                repeat,
                1, 1, 8, 8)
        if redundant >= 1:
            self.tik_instance.vmuls(
                redundant,
                self.mean_ub[repeat * 64],
                self.sum_input_ub[repeat * 64],
                num_rec,
                1, 1, 1, 8, 8)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 0)
        self.tik_instance.data_move(
            self.scale_ub,
            self.scale_input[start_pos],
            0,
            1, burst_length,
            0, 0)  # float32 多一倍 block
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 0)
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 1)
        if repeat >= 1:
            self.tik_instance.vmuls(
                64,
                self.square_sum_div_ub,
                self.square_sum_input_ub,  # float32 一个repeat 64个元素
                num_rec,
                repeat,
                1, 1, 8, 8)
        if redundant >= 1:
            self.tik_instance.vmuls(
                redundant,
                self.square_sum_div_ub[repeat * 64],
                self.square_sum_input_ub[repeat * 64],
                num_rec,
                1, 1, 1, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vmul(
                64,
                self.square_mean_ub,
                self.mean_ub,
                self.mean_ub,  # float32 一个repeat 64个元素
                repeat,
                1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vmul(
                redundant,
                self.square_mean_ub[repeat * 64],
                self.mean_ub[repeat * 64],
                self.mean_ub[repeat * 64],
                1,
                1, 1, 1, 8, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vsub(
                64,
                self.varience_ub,
                self.square_sum_div_ub,
                self.square_mean_ub,  # float32 一个repeat 64个元素
                repeat,
                1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vsub(
                redundant,
                self.varience_ub[repeat * 64],
                self.square_sum_div_ub[repeat * 64],
                self.square_mean_ub[repeat * 64],
                1,
                1, 1, 1, 8, 8, 8)
        return redundant, repeat

    def _init_l0a_l0b_l0c_tensor(self):
        """
        init l0a, l0b, l0c tensor
        """
        self.fmap_l1 = self.tik_instance.Tensor("float16",
                                                [self.b_inner_factor, self.channel_in1,
                                                 self.height_in,
                                                 self.width_in, self.channel_in0],
                                                scope=tik.scope_cbuf,
                                                name="fmap_l1")
        if self.filters_load_from_l1_flag:
            self.filters_l1 = self.tik_instance.Tensor(
                "float16",
                [self.k_l1a_factor * self.k_l0a_factor,
                 self.n_l1a_factor * self.n_l0a_factor,
                 self.channel_out0,
                 self.channel_in0],
                name="filters_l1",
                scope=tik.scope_cbuf)
            self.filters_l1_size = self.k_l1a_factor * self.k_l0a_factor * self.n_l1a_factor * \
                                   self.n_l0a_factor * self.channel_out0 * self.channel_in0 * 2
        else:
            self.filters_l1 = None
        self.fmap_l0a_ping = self.tik_instance.Tensor(dtype="float16",
                                                      shape=[self.b_inner_factor,
                                                             self.m_l0a_factor,
                                                             self.k_l0a_factor,
                                                             self.hw_out0,
                                                             self.channel_in0],
                                                      scope=tik.scope_ca,
                                                      name="fmap_l0a_ping")
        self.filters_l0b_ping = self.tik_instance.Tensor(dtype="float16",
                                                         shape=[self.k_l0a_factor,
                                                                self.n_l0a_factor,
                                                                self.channel_out0,
                                                                self.channel_in0],
                                                         scope=tik.scope_cb,
                                                         name="filters_l0b_ping")
        self.fmap_l0c_ping = self.tik_instance.Tensor(dtype="float32",
                                                      shape=[self.b_inner_factor,
                                                             self.n_l0a_factor,
                                                             self.m_l0a_factor,
                                                             self.hw_out0,
                                                             self.channel_out0],
                                                      scope=tik.scope_cc,
                                                      name="fmap_l0c_ping")
        self.fmap_l0a_pong = self.tik_instance.Tensor(dtype="float16",
                                                      shape=[self.b_inner_factor,
                                                             self.m_l0a_factor,
                                                             self.k_l0a_factor,
                                                             self.hw_out0,
                                                             self.channel_in0],
                                                      scope=tik.scope_ca,
                                                      name="fmap_l0a_pong")
        self.filters_l0b_pong = self.tik_instance.Tensor(dtype="float16",
                                                         shape=[
                                                             self.k_l0a_factor,
                                                             self.n_l0a_factor,
                                                             self.channel_out0,
                                                             self.channel_in0
                                                         ],
                                                         scope=tik.scope_cb,
                                                         name="filters_l0b_pong")
        self.fmap_l0c_pong = self.tik_instance.Tensor(dtype="float32",
                                                      shape=[self.b_inner_factor,
                                                             self.n_l0a_factor,
                                                             self.m_l0a_factor,
                                                             self.hw_out0,
                                                             self.channel_out0],
                                                      scope=tik.scope_cc,
                                                      name="fmap_l0c_pong")

    def _init_output_sum_and_square_sum(self):
        """
        init sum and square sum with zero
        """
        mask_loop = self.channel_out1 * self.channel_out0 // (64 * 255)
        mask_repeat = self.channel_out1 * self.channel_out0 % (64 * 255) // 64
        mask_redundant = self.channel_out1 * self.channel_out0 % (64 * 255) % 64
        if mask_loop > 0:
            with self.tik_instance.for_range(0, mask_loop, name='mask_loop') as loop_index:
                self.tik_instance.vec_dup(64,
                                          self.output_sum_inub[loop_index * 255 * 64],
                                          0,
                                          255,
                                          8)
                self.tik_instance.vec_dup(64,
                                          self.output_square_sum_inub[loop_index * 255 * 64],
                                          0,
                                          255,
                                          8)
                self.tik_instance.pipe_barrier("PIPE_V")
        if mask_repeat > 0:
            self.tik_instance.vec_dup(64,
                                      self.output_sum_inub[mask_loop * 255 * 64],
                                      0,
                                      mask_repeat,
                                      8)
            self.tik_instance.vec_dup(64,
                                      self.output_square_sum_inub[mask_loop * 255 * 64],
                                      0,
                                      mask_repeat,
                                      8)
            self.tik_instance.pipe_barrier("PIPE_V")
        if mask_redundant > 0:
            self.tik_instance.vec_dup(mask_redundant,
                                      self.output_sum_inub[mask_loop * 255 * 64 + mask_repeat * 64],
                                      0,
                                      1,
                                      1)
            self.tik_instance.vec_dup(mask_redundant,
                                      self.output_square_sum_inub[mask_loop * 255 * 64 + mask_repeat * 64],
                                      0,
                                      1,
                                      1)
            self.tik_instance.pipe_barrier("PIPE_V")

    def _init_all_ub_tensor(self):
        """
        init all ub tensor to avoid bank conflict
        """
        # avoid bank conflict
        self._allocate_ub_in_area_one()
        self._allocate_ub_in_area_two()
        self._allocate_ub_in_area_three()
        self._allocate_ub_in_area_four()

    def _allocate_ub_in_area_four(self):
        """
        allocate ub in 192kB ~ 256kB
        """
        # area4
        input_x_fp16_ub_pong_start_addr = self.get_start_addr(192 * 1024, 0)
        square_size = self.b_inner_factor * self.m_l0a_factor * self.hw_out0 * self.channel_out0
        fmap_input_size = self.b_inner_factor * self.get_max_h_length() * self.width_in * self.channel_in0
        self.input_x_fp16_ub_pong = self.tik_instance.Tensor(
            "float16",
            [self.b_inner_factor * self.get_max_h_length() * self.width_in * self.channel_in0],
            name="input_x_fp16_ub_pong",
            scope=tik.scope_ubuf,
            start_addr=input_x_fp16_ub_pong_start_addr)
        fmap_ub_start_addr = self.get_start_addr(input_x_fp16_ub_pong_start_addr, fmap_input_size * 2)
        self.fmap_ub = self.tik_instance.Tensor(
            "float32",
            [square_size],
            name="fmap_ub",
            scope=tik.scope_ubuf,
            start_addr=fmap_ub_start_addr)
        output_sum_inub_start_addr = self.get_start_addr(fmap_ub_start_addr, square_size * 4)
        self.output_sum_inub = self.tik_instance.Tensor(dtype="float32",
                                                        shape=self.channel_wise_shape_out,
                                                        scope=tik.scope_ubuf,
                                                        name="output_sum_gm_inub",
                                                        start_addr=output_sum_inub_start_addr)
        output_square_sum_inub_start_addr = self.get_start_addr(output_sum_inub_start_addr,
                                                                 self.channel_out1 * self.channel_out0 * 4)
        self.output_square_sum_inub = self.tik_instance.Tensor(dtype="float32",
                                                               shape=self.channel_wise_shape_out,
                                                               scope=tik.scope_ubuf,
                                                               name="output_square_sum_gm_inub",
                                                               start_addr=output_square_sum_inub_start_addr)
        save_ub_start_addr = self.get_start_addr(output_square_sum_inub_start_addr,
                                                  self.channel_out1 * self.channel_out0 * 4)
        self.save_ub = self.tik_instance.Tensor("uint16", [16 * 16], tik.scope_ubuf,
                                                name="save_last_inub",
                                                start_addr=save_ub_start_addr)
        tem_ub_start_addr = self.get_start_addr(save_ub_start_addr, 16 * 16 * 2)
        self.tem_ub = self.tik_instance.Tensor("uint16", [512], name="temp_ub",
                                               scope=tik.scope_ubuf,
                                               start_addr=tem_ub_start_addr)
        save_ub_pong_start_addr = self.get_start_addr(tem_ub_start_addr, 512 * 2)
        self.save_ub_pong = self.tik_instance.Tensor("uint16", [16 * 16], tik.scope_ubuf,
                                                     name="save_last_inub_pong",
                                                     start_addr=save_ub_pong_start_addr)
        tem_ub_pong_start_addr = self.get_start_addr(save_ub_pong_start_addr, 16 * 16 * 2)
        self.tem_ub_pong = self.tik_instance.Tensor("uint16", [512], name="temp_ub_pong",
                                                    scope=tik.scope_ubuf,
                                                    start_addr=tem_ub_pong_start_addr)

    def _allocate_ub_in_area_three(self):
        """
        allocate ub in 128kB ~ 196kB
        """
        # area3
        input_x_ub_start_addr = self.get_start_addr(128 * 1024, 0)
        square_size = self.b_inner_factor * self.m_l0a_factor * self.hw_out0 * self.channel_out0
        fmap_input_size = self.b_inner_factor * self.get_max_h_length() * self.width_in * self.channel_in0
        self.input_x_ub = self.tik_instance.Tensor(
            "float16",
            [self.b_inner_factor * (self.get_max_h_length()) * self.width_in * self.channel_in0],
            name="input_x_ub",
            scope=tik.scope_ubuf,
            start_addr=input_x_ub_start_addr)
        fmap_f16_ub_pong_start_addr = self.get_start_addr(input_x_ub_start_addr, fmap_input_size * 2)
        self.fmap_f16_ub_pong = self.tik_instance.Tensor(
            "float16",
            [square_size],
            name="fmap_f16_ub_pong",
            scope=tik.scope_ubuf,
            start_addr=fmap_f16_ub_pong_start_addr)

    def _allocate_ub_in_area_two(self):
        """
        allocate ub in 64kB ~ 128kB
        """
        # area 2
        input_x_fp32_ub_start_addr = self.get_start_addr(64 * 1024, 0)
        square_size = self.b_inner_factor * self.m_l0a_factor * self.hw_out0 * self.channel_out0
        self.input_x_fp32_ub = self.tik_instance.Tensor(
            "float32",
            [self.b_inner_factor,
             self.get_max_h_length(),
             self.width_in,
             self.channel_in0],
            name="input_x_fp32_ub",
            scope=tik.scope_ubuf,
            start_addr=input_x_fp32_ub_start_addr)
        fmap_input_size = self.b_inner_factor * self.get_max_h_length() * self.width_in * self.channel_in0
        input_x_fp16_ub_start_addr = self.get_start_addr(input_x_fp32_ub_start_addr, fmap_input_size * 4)
        self.input_x_fp16_ub = self.tik_instance.Tensor(
            "float16",
            [(self.b_inner_factor * (self.get_max_h_length()) * self.width_in * self.channel_in0)],
            name="input_x_fp16_ub",
            scope=tik.scope_ubuf,
            start_addr=input_x_fp16_ub_start_addr)
        square_mul_ub_start_addr = self.get_start_addr(input_x_fp16_ub_start_addr, fmap_input_size * 2)
        self.square_mul_ub = self.tik_instance.Tensor(
            "float32",
            [square_size],
            name="square_mul_ub",
            scope=tik.scope_ubuf,
            start_addr=square_mul_ub_start_addr)

    def _allocate_ub_in_area_one(self):
        """
        allocate ub in 0kB ~ 64kB
        """
        # area 1
        self.zeros_const_ub = self.tik_instance.Tensor(
            "float16",
            [128],
            name="zeros_const_ub_128",
            scope=tik.scope_ubuf,
            start_addr=0)
        self.zeros_const_ub_16 = self.tik_instance.Tensor(
            "float16",
            [16],
            name="zeros_const_ub_16",
            scope=tik.scope_ubuf,
            start_addr=256)
        elem_num = self.channel_in1 * self.channel_in0
        self.y_scale_ub = self.tik_instance.Tensor(
            "float32",
            [self.channel_in0 * self.channel_in1],
            name="y_scale_ub",
            scope=tik.scope_ubuf,
            start_addr=288)
        self.y_offset_ub_start_addr = self.get_start_addr(288, self.channel_in0 * self.channel_in1 * 4)
        self.y_offset_ub = self.tik_instance.Tensor(
            "float32",
            [self.channel_in0 * self.channel_in1],
            name="y_offset_ub",
            scope=tik.scope_ubuf,
            start_addr=self.y_offset_ub_start_addr)
        scale_ub_64_start_addr = self.get_start_addr(self.y_offset_ub_start_addr, elem_num * 4)
        self.scale_ub_64 = self.tik_instance.Tensor("float32",
                                                    [64],
                                                    name="scale_ub_64",
                                                    scope=tik.scope_ubuf,
                                                    start_addr=scale_ub_64_start_addr)
        offset_ub_64_start_addr = self.get_start_addr(scale_ub_64_start_addr, 64 * 4)
        self.offset_ub_64 = self.tik_instance.Tensor("float32",
                                                     [64],
                                                     name="offset_ub_64",
                                                     scope=tik.scope_ubuf,
                                                     start_addr=offset_ub_64_start_addr)
        input_x_mask_ub_start_addr = self.get_start_addr(offset_ub_64_start_addr, 64 * 4)
        self.input_x_mask_ub = self.tik_instance.Tensor(
            "uint16",
            [((self.get_max_h_length()) * self.width_in + 15) // 16 * 16],
            name="input_x_relu_ub",
            scope=tik.scope_ubuf,
            start_addr=input_x_mask_ub_start_addr)
        input_x_mask_ub_pong_start_addr = self.get_start_addr(input_x_mask_ub_start_addr,
                                                               (self.get_max_h_length() * self.width_in +
                                                                15) // 16 * 16 * 2)
        self.input_x_mask_ub_pong = self.tik_instance.Tensor(
            "uint16",
            [(self.get_max_h_length() * self.width_in + 15) // 16 * 16],
            name="input_x_relu_ub_pong",
            scope=tik.scope_ubuf,
            start_addr=input_x_mask_ub_pong_start_addr)
        square_size = self.b_inner_factor * self.m_l0a_factor * self.hw_out0 * self.channel_out0
        fmap_f16_ub_ping_start_addr = self.get_start_addr(input_x_mask_ub_pong_start_addr,
                                                           (self.get_max_h_length() * self.width_in + \
                                                           15) // 16 * 16 * 2)
        self.fmap_f16_ub_ping = self.tik_instance.Tensor(
            "float16",
            [square_size],
            name="fmap_f16_ub_ping",
            scope=tik.scope_ubuf,
            start_addr=fmap_f16_ub_ping_start_addr)
        ub_fm1_workbuf_start_addr = self.get_start_addr(fmap_f16_ub_ping_start_addr, square_size * 4)
        self.ub_fm1_workbuf = self.tik_instance.Tensor(
            "float32",
            [square_size],
            name="ub_fm1_workbuf",
            scope=tik.scope_ubuf,
            start_addr=ub_fm1_workbuf_start_addr)
        ub_fm1_workbuf2_start_addr = self.get_start_addr(ub_fm1_workbuf_start_addr, square_size * 4)
        self.ub_fm1_workbuf2 = self.tik_instance.Tensor(
            "float32",
            [square_size],
            name="ub_fm1_workbuf2",
            scope=tik.scope_ubuf,
            start_addr=ub_fm1_workbuf2_start_addr)
        sum_input_ub_start_addr = self.get_start_addr(fmap_f16_ub_ping_start_addr, square_size * 2)
        self.sum_input_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="sum_ub",
            scope=tik.scope_ubuf,
            start_addr=sum_input_ub_start_addr)
        square_sum_input_ub_start_addr = self.get_start_addr(sum_input_ub_start_addr, elem_num * 4)
        self.square_sum_input_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="square_sum_input_ub",
            scope=tik.scope_ubuf,
            start_addr=square_sum_input_ub_start_addr)
        self._fetch_allocate_ub_in_area_one(elem_num, square_sum_input_ub_start_addr)

    def _fetch_allocate_ub_in_area_one(self, elem_num, square_sum_input_ub_start_addr):
        """
        fetch allocate ub in area one
        """
        y_scale_sqrt_ub_start_addr = self._allocate_ub_in_area_one_sub(elem_num, square_sum_input_ub_start_addr)
        offset_ub_start_addr = self.get_start_addr(y_scale_sqrt_ub_start_addr, elem_num * 4)
        self.offset_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="offset_ub",
            scope=tik.scope_ubuf,
            start_addr=offset_ub_start_addr)
        y_offset_mul_ub_start_addr = self.get_start_addr(offset_ub_start_addr, elem_num * 4)
        self.y_offset_mul_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="y_offset_mul_ub",
            scope=tik.scope_ubuf,
            start_addr=y_offset_mul_ub_start_addr)
        self.spilt_allocate_ub_area_one(elem_num, y_offset_mul_ub_start_addr)

    def _dich_loop_add(self, dichotomy_repeat, threshold, total_blocks):
        """
        dichotomy loop add
        """
        total_repeat = dichotomy_repeat
        dichotomy_repeat = total_repeat // 2
        ub_add_0 = self.ub_fm1_workbuf2
        ub_add_1 = self.ub_fm1_workbuf
        while dichotomy_repeat > threshold:
            self.tik_instance.vadd(64, ub_add_1, ub_add_0[64], ub_add_0, dichotomy_repeat, 1, 1, 1, 8, 16, 16)
            self.tik_instance.vadd(
                64, ub_add_1[total_blocks // 2], ub_add_0[total_blocks // 2 + 64], ub_add_0[total_blocks // 2],
                dichotomy_repeat, 1, 1, 1, 8, 16, 16)
            self.tik_instance.pipe_barrier("PIPE_V")
            if total_repeat % 2 != 0:
                self.tik_instance.vadd(64, ub_add_1, ub_add_0[dichotomy_repeat * 64 * 2], ub_add_1,
                                       total_repeat % 2, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vadd(
                    64, ub_add_1[total_blocks // 2], ub_add_0[total_blocks // 2 + dichotomy_repeat * 64 * 2],
                    ub_add_1[total_blocks // 2], total_repeat % 2, 1, 1, 1, 8, 8, 8)
                self.tik_instance.pipe_barrier("PIPE_V")
            total_repeat = dichotomy_repeat
            dichotomy_repeat = total_repeat // 2
            ub_add_0, ub_add_1 = ub_add_1, ub_add_0
        return dichotomy_repeat, total_repeat, ub_add_0


def fused_bn2_reluv2_conv2d_bn1(fmap_input, sum_input, square_sum_input,
                                scale_input, offset_input, moving_mean_pre_input,
                                moving_variance_pre_input, filters_input, bias_input,
                                moving_mean_cur_output, moving_variance_cur_output,
                                mean_output, variance_out,
                                relu_output, mask_output,
                                convolution_output, sum_output, square_sum_output,
                                factor, epsilon, stride, padding, dilation, groups,
                                kernel_name="fused_bn2_reluv2_conv2d_bn1"):
    """
    algorithm: fuse_bn_update_reluv2_conv2d_bn_reduce

    Parameters
    ----------
    fmap_input: dict
        dict of s, A 5D Tensor for input x of conv.
        source data type, support , "float16".
    sum_input: dict
        dict of sum, A 5HD Tensor for sum.
        The output of batch_normalization_forward_training_reduce.
    square_sum_input: dict
        dict of square_sum, A 5HD Tensor for square_sum.
        The output of batch_normalization_forward_training_reduce.
    scale_input: dict
        dict of scale, A 5HD Tensor for mean.
    offset_input: dict
        dict of offset, A 5HD Tensor for variance.
    moving_mean_pre_input: dict
        dict of mean, A 5HD Tensor for mean.
    moving_variance_pre_input: dict
        dict of variance, A 5HD Tensor for variance.
    filters_input: dict with keys(shape and dtype)
        input 4d weight tensor
    bias_input: dict with keys(shape and dtype) or None
        input bias tensor
    mean_output: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
    relu_output: the dict of relu output
    mask_output: the dict of mask_output
    convolution_output: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    sum_output: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum_output: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    factor: float
        A retio to caculate the update mean or variance.
    epsilon: float
        A small float number added to the variance of x.
        weights: dict with keys(shape and dtype)
        input 4d weight tensor
    stride: tuple/list of 4 integers
        stride on H/W, format sensitive
    padding: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilation: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    kernel_name: str
        kernel name, default value is "fused_bn2_reluv2_conv2d_bn1"

    Returns
    -------
    None
    """
    bn2_reluv2_conv2d_bn1 = None
    fmap_ori_format = fmap_input["ori_format"]
    fmap_ori_shape = fmap_input["ori_shape"]
    if fmap_ori_format == "NHWC":
        batch, height, width, channel_in = fmap_ori_shape
        fmap_ori_shape = [batch, channel_in, height, width]
        _, stride_h, stride_w, _ = stride
        _, dilation_h, dilation_w, _ = dilation
    else:
        _, _, stride_h, stride_w = stride
        _, _, dilation_h, dilation_w = dilation
    strides = [stride_h, stride_w]
    dilations = [dilation_h, dilation_w]

    filters_ori_format = filters_input["ori_format"]
    filters_ori_shape = filters_input["ori_shape"]
    if filters_ori_format == "HWCN":
        kernel_h, kernel_w, channel_in, chanel_out = filters_ori_shape
        filters_ori_shape = [chanel_out, channel_in, kernel_h, kernel_w]

    tiling = None
    # case1
    if is_same_value(fmap_ori_shape, [256, 64, 56, 56]) and is_same_value(
            filters_ori_shape, [64, 64, 3, 3]) and \
            is_same_value(padding, [1, 1, 1, 1]) and is_same_value(strides, [1, 1]):
        tiling = {'BL1': [12, 1], 'AL0': [14, 3], 'BL0': [3, 4],
                  'BLOCK_DIM': [32, 1, 1],
                  'BLOCK_INNER': 1}
        bn2_reluv2_conv2d_bn1 = Bn2ReluV2Conv2dBn1HandSync(fmap_ori_shape,
                                                           filters_ori_shape,
                                                           padding,
                                                           strides,
                                                           dilations,
                                                           groups,
                                                           factor,
                                                           epsilon,
                                                           tiling, kernel_name)
        bn2_reluv2_conv2d_bn1.compute()
        return bn2_reluv2_conv2d_bn1

    # case7
    if is_same_value(fmap_ori_shape, [256, 128, 56, 56]) and is_same_value(
            filters_ori_shape, [128, 128, 3, 3]) and \
            is_same_value(padding, [0, 1, 0, 1]) and is_same_value(strides, [2, 2]):
        tiling = {"AL1": [4, 24], "BL1": None, "AL0": [14, 3], "BL0": [3, 8],
                  "BLOCK_DIM": [32, 1, 1], "BLOCK_INNER": 1}
        bn2_reluv2_conv2d_bn1 = Bn2ReluV2Conv2dBn1BatchByBatchForStrideTwoHandSync(
            fmap_ori_shape,
            filters_ori_shape,
            padding,
            strides,
            dilations,
            groups,
            factor,
            epsilon,
            tiling, kernel_name)
        bn2_reluv2_conv2d_bn1.compute()
        return bn2_reluv2_conv2d_bn1
    err_man.raise_err_specific("fused_bn2_reluv2_conv2d_bn1", "no match shape, please check!!!")
    return bn2_reluv2_conv2d_bn1


def is_same_value(a_value, b_value):
    """
    check shape
    """
    for elem_a, elem_b in zip(a_value, b_value):
        if elem_a != elem_b:
            return False
    return True
