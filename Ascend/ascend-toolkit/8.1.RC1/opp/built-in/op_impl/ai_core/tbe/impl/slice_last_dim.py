#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
slice_last_dim
"""
# 'pylint: disable=too-many-statements,invalid-name,too-many-branches,unused-argument,too-many-locals

import math
from functools import reduce
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check

BLOCK_SIZE = 32
C0 = 16
DATA_MOVE_MAX_NBURST = 4095
UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
RESERVED_UB_SIZE = 4 * 1024
AVAILABLE_UB_SIZE = UB_SIZE - RESERVED_UB_SIZE


class SliceLastDim():
    """slice last dim main method"""
    # 'pylint: disable=too-many-arguments
    def __init__(self, x, y, start, end, stride=1, kernel_name="slice_last_dim"):
        self.kernel_name = kernel_name
        x_shape, self.x_dtype = x.get("shape"), x.get("dtype")

        self.start = start if start >= 0 else start + x_shape[-2]  # in case of start or end is negative
        self.end = end if end >= 0 else end + x_shape[-2]  # same as above
        self.stride = stride
        self.length = (end - start + stride - 1) // stride

        # [NC1H, W, C0)
        self.reformat_x_shape = [reduce(lambda x_, y_: x_ * y_, x_shape[:-2]), x_shape[-2], C0]
        self.total_burst, self.data_frag, self.wc0_data = self.get_nburst_and_data_frag(
            self.reformat_x_shape, self.stride, self.length)

        self.out_shape = self.reformat_x_shape[:-2] + [self.length, C0]  # NC1HWC0
        self.x_byte_size = tbe_platform.get_bit_len(self.x_dtype) // 8

        # get core num and nburst_per_core
        self.block_num, self.nburst_per_core, self.nburst_last_core = self.get_block_num_and_nburst_per_core(
            self.total_burst)

        # do param check
        self.param_check()

        # start TIK process
        self.tik_instance = tik.Tik()
        self.x_gm = self.tik_instance.Tensor(
            self.x_dtype, self.reformat_x_shape, name="x_gm", scope=tbe_platform.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.x_dtype, self.out_shape, name="out_gm", scope=tbe_platform.scope_gm)

    def slice_last_dim_compute(self):
        """
        slice compute main logic here
        :return:
        """
        with self.tik_instance.for_range(0, self.block_num, block_num=self.block_num) as block_i:
            with self.tik_instance.if_scope(block_i < self.block_num - 1):
                self.data_move_per_core(block_i, self.nburst_per_core)
            with self.tik_instance.else_scope():
                self.data_move_per_core(block_i, self.nburst_last_core)

        # Build cce
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.x_gm,),
            outputs=(self.out_gm,),
            enable_l2=False
        )

    def data_move_per_core(self, block_i, num_burst_per_core):
        """
        main logic of data move per core
        :param block_i: core num
        :param num_burst_per_core: number of burst per core
        :return:
        """
        data_move_times, num_burst_per_data_move, tail_num_burst = \
            self.get_num_of_burst_per_data_move(num_burst_per_core, self.data_frag, self.x_byte_size)

        # future improvement: double buffer.
        with self.tik_instance.for_range(0, data_move_times) as data_move_i:
            if self.stride == 1:
                # case a[..., 5:15]
                self.data_move_case_stride1(block_i, data_move_i, data_move_times,
                                            num_burst_per_data_move, tail_num_burst)

            elif self.start < self.stride and self.reformat_x_shape[-2] % self.stride == 0 \
                    and self.end > self.reformat_x_shape[-2] - self.stride + self.start:
                # case a[..., ::2] or case a[..., 1::2], stride > 1
                self.data_move_case_stride_gt1(block_i, data_move_i, data_move_times,
                                               num_burst_per_data_move, tail_num_burst)

            else:
                # case a[..., 3:15:stride], stride > 1
                self.data_move_case_stride_gt1_with_loop(block_i, data_move_i, data_move_times,
                                                         num_burst_per_data_move, tail_num_burst)

    # 'pylint: disable=too-many-arguments
    def data_move_case_stride1(self, block_i, data_move_i, data_move_times,
                               num_burst_per_data_move, tail_num_burst):
        """
        data move with stride 1
        :param block_i: block loop i
        :param data_move_i: data move loop i
        :param data_move_times: data move times
        :param num_burst_per_data_move: number of burst every data move
        :param tail_num_burst:
        :return:
        """
        gm_to_ub_offset = (self.start * C0 + data_move_i * num_burst_per_data_move * self.wc0_data
                           + block_i * self.nburst_per_core * self.wc0_data)
        ub_to_gm_offset = (self.nburst_per_core * self.data_frag * block_i
                           + num_burst_per_data_move * self.data_frag * data_move_i)
        # stride
        gm_to_ub_src_stride = (self.wc0_data - self.data_frag) * self.x_byte_size // BLOCK_SIZE
        ub_to_gm_dst_stride = 0
        # burst
        loop_ub_to_gm_n_burst = 1
        tail_ub_to_gm_n_burst = 1

        with self.tik_instance.if_scope(data_move_i < data_move_times - 1):
            self.data_move_every_time(gm_to_ub_offset, ub_to_gm_offset, gm_to_ub_src_stride, ub_to_gm_dst_stride,
                                      num_burst_per_data_move, loop_ub_to_gm_n_burst)
        with self.tik_instance.else_scope():
            self.data_move_every_time(gm_to_ub_offset, ub_to_gm_offset, gm_to_ub_src_stride, ub_to_gm_dst_stride,
                                      tail_num_burst, tail_ub_to_gm_n_burst)

    # 'pylint: disable=too-many-arguments
    def data_move_case_stride_gt1(self, block_i, data_move_i, data_move_times,
                                  num_burst_per_data_move, tail_num_burst):
        """
        data move with stride 1
        :param block_i: block loop i
        :param data_move_i: data move loop i
        :param data_move_times: data move times
        :param num_burst_per_data_move: number of burst every data move
        :param tail_num_burst:
        :return:
        """
        gm_to_ub_offset = (self.start * self.data_frag
                           + data_move_i * num_burst_per_data_move * self.data_frag * self.stride
                           + block_i * self.nburst_per_core * self.data_frag * self.stride)
        ub_to_gm_offset = (self.nburst_per_core * self.data_frag * block_i
                           + num_burst_per_data_move * self.data_frag * data_move_i)
        # stride
        gm_to_ub_src_stride = self.data_frag * (self.stride - 1) * self.x_byte_size // BLOCK_SIZE
        ub_to_gm_dst_stride = 0
        # burst
        loop_ub_to_gm_n_burst = 1
        tail_ub_to_gm_n_burst = 1

        with self.tik_instance.if_scope(data_move_i < data_move_times - 1):
            self.data_move_every_time(gm_to_ub_offset, ub_to_gm_offset, gm_to_ub_src_stride, ub_to_gm_dst_stride,
                                      num_burst_per_data_move, loop_ub_to_gm_n_burst)
        with self.tik_instance.else_scope():
            self.data_move_every_time(gm_to_ub_offset, ub_to_gm_offset, gm_to_ub_src_stride, ub_to_gm_dst_stride,
                                      tail_num_burst, tail_ub_to_gm_n_burst)

    # 'pylint: disable=too-many-arguments
    def data_move_case_stride_gt1_with_loop(self, block_i, data_move_i, data_move_times,
                                            num_burst_per_data_move, tail_num_burst):
        """
        data move with stride 1
        :param block_i: block loop i
        :param data_move_i: data move loop i
        :param data_move_times: data move times
        :param num_burst_per_data_move: number of burst every data move
        :param tail_num_burst:
        :return:
        """
        with self.tik_instance.for_range(0, self.length) as w_i:
            gm_to_ub_offset = (self.start + w_i * self.stride) * self.data_frag \
                              + data_move_i * num_burst_per_data_move * self.wc0_data \
                              + block_i * self.nburst_per_core * self.wc0_data
            ub_to_gm_offset = (w_i * self.data_frag
                               + num_burst_per_data_move * self.data_frag * self.length * data_move_i
                               + self.nburst_per_core * self.data_frag * self.length * block_i)

            # stride
            gm_to_ub_src_stride = (self.wc0_data - self.data_frag) * self.x_byte_size // BLOCK_SIZE
            ub_to_gm_dst_stride = (self.length * C0 - self.data_frag) * self.x_byte_size // BLOCK_SIZE
            # burst
            loop_ub_to_gm_n_burst = num_burst_per_data_move
            tail_ub_to_gm_n_burst = tail_num_burst

            with self.tik_instance.if_scope(data_move_i < data_move_times - 1):
                self.data_move_every_time(gm_to_ub_offset, ub_to_gm_offset, gm_to_ub_src_stride, ub_to_gm_dst_stride,
                                          num_burst_per_data_move, loop_ub_to_gm_n_burst)
            with self.tik_instance.else_scope():
                self.data_move_every_time(gm_to_ub_offset, ub_to_gm_offset, gm_to_ub_src_stride, ub_to_gm_dst_stride,
                                          tail_num_burst, tail_ub_to_gm_n_burst)

    # 'pylint: disable=too-many-arguments
    def data_move_every_time(self, gm_to_ub_offset, ub_to_gm_offset, gm_to_ub_stride, ub_to_gm_stride,
                             gm_to_ub_nburst, ub_to_gm_nburst):
        """
        everty data move step of gm to ub and ub to gm
        :param gm_to_ub_offset: gm to ub offset
        :param ub_to_gm_offset: ub to gm offset
        :param gm_to_ub_stride: gm to ub stride
        :param ub_to_gm_stride: ub to gm stride
        :param gm_to_ub_nburst: gm to ub nburst
        :param ub_to_gm_nburst: ub to gm nburst
        :return:
        """
        x_ub = self.tik_instance.Tensor(self.x_dtype, (gm_to_ub_nburst * self.data_frag,), name="x_ub",
                                        scope=tbe_platform.scope_ubuf)

        # data move from gm to ub using stride
        gm_to_ub_bur_len = self.data_frag * self.x_byte_size // BLOCK_SIZE

        # check some data move param
        if gm_to_ub_bur_len * BLOCK_SIZE > AVAILABLE_UB_SIZE:
            raise RuntimeError(
                "Not supported yet, data fragment size bigger than AVAILABLE UB size!")

        if not 0 <= gm_to_ub_stride <= 65535:
            raise RuntimeError(
                "gm to ub stride should between [0, 65535]!")

        if not 0 <= ub_to_gm_stride <= 65535:
            raise RuntimeError(
                "ub to gm stride should between [0, 65535]!")

        self.tik_instance.data_move(x_ub, self.x_gm[gm_to_ub_offset], 0, gm_to_ub_nburst,
                                    gm_to_ub_bur_len, src_stride=gm_to_ub_stride, dst_stride=0)

        # data move from ub to gm
        ub_to_gm_bur_len = gm_to_ub_nburst // ub_to_gm_nburst * self.data_frag * self.x_byte_size // BLOCK_SIZE
        self.tik_instance.data_move(self.out_gm[ub_to_gm_offset], x_ub, 0, ub_to_gm_nburst,
                                    ub_to_gm_bur_len, src_stride=0, dst_stride=ub_to_gm_stride)

    @staticmethod
    def get_block_num_and_nburst_per_core(total_burst):
        """
        get block_num and pre loops per core
        :return: block_num and pre loops per core
        """
        # if data_move unit less than 32B, can't open multi_core mode, because the data may overwrite
        block_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        if total_burst <= block_num:
            block_num = total_burst
            pre_num_burst = 1
        else:
            pre_num_burst = math.ceil(total_burst / block_num)
            block_num = math.ceil(total_burst / pre_num_burst)
        tail_num_burst = total_burst - (block_num - 1) * pre_num_burst

        return block_num, pre_num_burst, tail_num_burst

    @staticmethod
    def get_num_of_burst_per_data_move(num_burst, data_frag, x_byte_size):
        """
        get data move times, cause the data_move interface's nburst max size is 4095, and considering the UB size,
        we may not be able to load all the data from gm to ub within one data move
        :param num_burst: total pre l
        :param data_frag: dim of data_frag
        :param x_byte_size: data size of an element.
        :return: pre_loops_per_data_move, loop times
        """
        nburst_per_data_move = min(AVAILABLE_UB_SIZE // (data_frag * x_byte_size), DATA_MOVE_MAX_NBURST)
        data_move_times = (num_burst + nburst_per_data_move - 1) // nburst_per_data_move
        tail_nburst = num_burst % nburst_per_data_move

        return data_move_times, nburst_per_data_move, tail_nburst

    def get_nburst_and_data_frag(self, x_shape, stride, length):
        """
        get total nburst, num of data fragment every data move and num of w*c0 data.
        :param x_shape: input shape,[NC1H, W, C0]
        :param stride: slice stride
        :param length: result length, e.x. the 4th dim: [N, C1, H, length, C0]
        :return:
        """
        if stride == 1:
            nburst, data_frag, wc0_data = x_shape[0], length * x_shape[-1], x_shape[-1] * x_shape[-2]
        elif self.start < self.stride and self.reformat_x_shape[-2] % self.stride == 0 \
                and self.end > self.reformat_x_shape[-2] - self.stride + self.start:
            nburst, data_frag, wc0_data = x_shape[0] * self.length, x_shape[2], x_shape[-1] * x_shape[-2]
        else:
            nburst, data_frag, wc0_data = x_shape[0], x_shape[2], x_shape[-1] * x_shape[-2]

        return nburst, data_frag, wc0_data

    def param_check(self):
        """
        do some parameters check
        :return:
        """
        # stride should greater than or equal to 1
        if self.stride < 1:
            raise RuntimeError(
                "stride should be greater than or equal to 1!")

        if self.x_dtype not in ("float16", "float32", "int16", "int32"):
            raise RuntimeError(
                "{} data type not supported yet!".format(self.x_dtype))


# 'pylint: disable=too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def slice_last_dim(x, y, start, end, stride=1, kernel_name="slice_last_dim"):
    """
    interface of slice_last_dim.
    :param x: dict, input tensor
    :param y: dict, output tensor
    :param start: int, attr of start index
    :param end: int, attr of end to sliced
    :param stride: int, attr of stride to slice
    :param kernel_name:
    :return:
    """
    slice_last_dim_instance = SliceLastDim(x, y, start, end, stride, kernel_name)
    slice_last_dim_instance.slice_last_dim_compute()
