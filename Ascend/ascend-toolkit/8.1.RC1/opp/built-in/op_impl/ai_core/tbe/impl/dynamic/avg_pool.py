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
avg_pool
"""
import math
import re
import copy
import warnings
from typing import Union
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_register
from impl.util.util_conv2d_dynamic import Conv2dParaProcess
from impl.util.util_conv2d_dynamic import modify_input_range
from impl.util.util_conv2d_dynamic import check_l1_size
from impl.util.util_conv2d_dynamic import create_fuzz_range
from impl.util.util_conv2d_dynamic import correct_input_range
from impl.util.util_conv2d_dynamic import check_graph_mode
from impl.util.util_conv2d_dynamic import check_input_range
from impl.util.util_conv2d_dynamic import check_range_l1_size
from impl.util.util_conv2d_dynamic import check_range_value
from impl.util.util_cube_dynamic import BIT_RATIO_DICT
from tbe.dsl.compute.conv_compute import conv
from tbe.dsl.compute.conv_compute import ConvParam
from tbe.common.utils import log
from tbe.common.platform import get_bit_len


AVG_KERNEL_SIZE_H_MUL_W = 255  # kernel_h * kernel_w
AVG_KERNEL_SIZE = 20  # maximum ksize
MAX_CUBE_STRIDE = 63  # maximum cube stride
NONETYPE = type(None)
MAX_INT32 = 2 ** 31 - 1
TILING_ARG_NUM = 24
RESERVED_UB_SIZE = 8 * 1024
EIGHT_BIT = 8
BLOCK_BYTES = 32
C_ZERO = 16
REPEAT_LIMIT = 255
MASK = 128
MIN_FP16 = 0
INPUT_DIM = 4
SHAPE_LEN = 5
H_DIM = 2
W_DIM = 3
ORI_SHAPE_LEN = 4
DYNAMIC_VALUE = -1


class AvgPool:
    def __init__(self, dtype, ksize, strides, padding, kernel_name):
        self.dtype = dtype
        self.ksize_h = ksize[0]
        self.ksize_w = ksize[1]
        self.strides_h = strides[0]
        self.strides_w = strides[1]
        self.padding = 0 if padding == "SAME" else 1
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_size = get_bit_len(self.dtype) // EIGHT_BIT
        self.ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB_SIZE) // self.dtype_size
        self.one_fourth_ub_ele = self.ub_ele // 4
        """
        init_gm_tensor
        """
        self.tiling_gm = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)
        self.input_gm = self.tik_instance.Tensor(self.dtype, (MAX_INT32,), name="input_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype, (MAX_INT32,), name="output_gm", scope=tik.scope_gm)
        """
        init_tiling_args
        """
        self.tiling_mode = None
        self.act_core_num = None
        self.one_core_ele = None
        self.last_core_ele = None
        self.input_h = None
        self.input_w = None
        self.output_h = None
        self.output_w = None
        self.pad_h = None
        self.pad_w = None
        self.pad_t = None
        self.pad_b = None  
        self.pad_l = None
        self.pad_r = None   
        self.c_factor = None
        self.h_factor = None
        self.w_factor = None
        self.one_core_loop_num = None
        self.one_core_loop_left = None
        self.last_core_loop_num = None
        self.last_core_loop_left = None
        """
        init_ub_tensor
        """
        self.ub_a = None
        self.ub_b = None
        self.ub_c = None
        self.ub_d = None
        """
        init_ub_scalar_single
        """
        self.size = None
        self.offset = None
        self.core_ele = None
        self.loop_num = None
        self.loop_left = None
        self.before_h = None
        self.after_h = None
        self.before_w = None
        self.after_w = None
        self.len_h = None
        self.len_w = None
        self.offset_ub = None
        self.nburst = None
        self.src_stride = None
        self.dst_stride = None
        self.burst_len_in = None
        self.burst_len_out = None
        self.size_1 = None
        self.offset_2 = None
        self.size_2 = None
        self.offset_3 = None
        self.repeat_3 = None
        self.size_3 = None
        self.size_w = None
        self.repeat_w = None
        self.rep_blk_h = None
        self.factor_total = None
        self.fp_dtype = None
        self.int_dtype = None
        self.real_h = None
        self.real_w = None
        self.pos_h = None
        self.pos_w = None
        """
        init_ub_scalar_double
        """
        self.size_loop = None
        self.size_left = None
        self.repeat_loop = None
        self.repeat_left = None
        self.size_offset = None
        self.repeat_offset = None
        self.offset_in = None
        self.offset_out = None
        self.repeat_offset_src0 = None
        self.repeat_offset_src1 = None
        self.size_offset_src0 = None
        self.size_offset_src1 = None
        self.offset_dst = None
        self.offset_src0 = None
        self.offset_src1 = None
        self.offset_h = None
        self.offset_w = None

    def tiling_args(self):
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.act_core_num = self.tik_instance.Scalar("int32", name="act_core_num")
        self.act_core_num.set_as(self.tiling_ub[1])
        self.one_core_ele = self.tik_instance.Scalar("int32", name="one_core_ele")
        self.one_core_ele.set_as(self.tiling_ub[2])
        self.last_core_ele = self.tik_instance.Scalar("int32", name="last_core_ele")
        self.last_core_ele.set_as(self.tiling_ub[3])
        self.input_h = self.tik_instance.Scalar("int32", name="input_h")
        self.input_h.set_as(self.tiling_ub[4])
        self.input_w = self.tik_instance.Scalar("int32", name="input_w")
        self.input_w.set_as(self.tiling_ub[5])
        self.output_h = self.tik_instance.Scalar("int32", name="output_h")
        self.output_h.set_as(self.tiling_ub[6])
        self.output_w = self.tik_instance.Scalar("int32", name="output_w")
        self.output_w.set_as(self.tiling_ub[7])
        self.pad_h = self.tik_instance.Scalar("int32", name="pad_h")
        self.pad_h.set_as(self.tiling_ub[8])
        self.pad_w = self.tik_instance.Scalar("int32", name="pad_w")
        self.pad_w.set_as(self.tiling_ub[9])
        self.pad_t = self.tik_instance.Scalar("int32", name="pad_t")
        self.pad_t.set_as(self.tiling_ub[10])
        self.pad_b = self.tik_instance.Scalar("int32", name="pad_b")
        self.pad_b.set_as(self.tiling_ub[11])
        self.pad_l = self.tik_instance.Scalar("int32", name="pad_l")
        self.pad_l.set_as(self.tiling_ub[12])
        self.pad_r = self.tik_instance.Scalar("int32", name="pad_r")
        self.pad_r.set_as(self.tiling_ub[13])
        self.c_factor = self.tik_instance.Scalar("int32", name="c_factor")
        self.c_factor.set_as(self.tiling_ub[14])
        self.h_factor = self.tik_instance.Scalar("int32", name="h_factor")
        self.h_factor.set_as(self.tiling_ub[15])
        self.w_factor = self.tik_instance.Scalar("int32", name="w_factor")
        self.w_factor.set_as(self.tiling_ub[16])
        self.one_core_loop_num = self.tik_instance.Scalar("int32", name="one_core_loop_num")
        self.one_core_loop_num.set_as(self.tiling_ub[17])
        self.one_core_loop_left = self.tik_instance.Scalar("int32", name="one_core_loop_left")
        self.one_core_loop_left.set_as(self.tiling_ub[18])
        self.last_core_loop_num = self.tik_instance.Scalar("int32", name="last_core_loop_num")
        self.last_core_loop_num.set_as(self.tiling_ub[19])
        self.last_core_loop_left = self.tik_instance.Scalar("int32", name="last_core_loop_left")
        self.last_core_loop_left.set_as(self.tiling_ub[20])

    def init_ub_tensor(self):
        self.ub_a = self.tik_instance.Tensor(self.dtype, (self.one_fourth_ub_ele,), name="ub_a", scope=tik.scope_ubuf)
        self.ub_b = self.tik_instance.Tensor(self.dtype, (self.one_fourth_ub_ele,), name="ub_b", scope=tik.scope_ubuf)
        self.ub_c = self.tik_instance.Tensor(self.dtype, (self.one_fourth_ub_ele,), name="ub_c", scope=tik.scope_ubuf)
        self.ub_d = self.tik_instance.Tensor(self.dtype, (self.one_fourth_ub_ele,), name="ub_d", scope=tik.scope_ubuf)

    def init_ub_scalar_single(self):
        self.size = self.tik_instance.Scalar("int32", name="size")
        self.offset = self.tik_instance.Scalar("int32", name="offset")
        self.core_ele = self.tik_instance.Scalar("int32", name="core_ele")
        self.loop_num = self.tik_instance.Scalar("int32", name="loop_num")
        self.loop_left = self.tik_instance.Scalar("int32", name="loop_left")
        self.before_h = self.tik_instance.Scalar("int32", name="before_h")
        self.after_h = self.tik_instance.Scalar("int32", name="after_h")
        self.before_w = self.tik_instance.Scalar("int32", name="before_w")
        self.after_w = self.tik_instance.Scalar("int32", name="after_w")
        self.len_h = self.tik_instance.Scalar("int32", name="len_h")
        self.len_w = self.tik_instance.Scalar("int32", name="len_w")
        self.offset_ub = self.tik_instance.Scalar("int32", name="offset_ub")
        self.nburst = self.tik_instance.Scalar("int32", name="nburst")
        self.src_stride = self.tik_instance.Scalar("int32", name="src_stride")
        self.dst_stride = self.tik_instance.Scalar("int32", name="dst_stride")
        self.burst_len_in = self.tik_instance.Scalar("int32", name="burst_len_in")
        self.burst_len_out = self.tik_instance.Scalar("int32", name="burst_len_out")
        self.size_1 = self.tik_instance.Scalar("int32", name="size_1")
        self.offset_2 = self.tik_instance.Scalar("int32", name="offset_2")
        self.size_2 = self.tik_instance.Scalar("int32", name="size_2")
        self.offset_3 = self.tik_instance.Scalar("int32", name="offset_3")
        self.repeat_3 = self.tik_instance.Scalar("int32", name="repeat_3")
        self.size_3 = self.tik_instance.Scalar("int32", name="size_3")
        self.size_w = self.tik_instance.Scalar("int32", name="size_w")
        self.repeat_w = self.tik_instance.Scalar("int32", name="repeat_w")
        self.rep_blk_h = self.tik_instance.Scalar("int32", name="rep_blk_h")
        self.factor_total = self.tik_instance.Scalar(self.dtype, name="factor_total")
        self.fp_dtype = "float32"
        self.int_dtype = "int32"
        if not tbe_platform.api_check_support("tik.vconv", "s322f32") or \
                not tbe_platform.api_check_support("tik.vmuls", "float32"):
            self.fp_dtype = "float16"
            if not tbe_platform.api_check_support("tik.vconv", "s322f16"):
                self.int_dtype = "int16"

        self.real_h = self.tik_instance.Scalar("int32", name="real_h")
        self.real_w = self.tik_instance.Scalar("int32", name="real_w")
        self.pos_h = self.tik_instance.Scalar("int32", name="pos_h")
        self.pos_w = self.tik_instance.Scalar("int32", name="pos_w")

    def init_ub_scalar_double(self):
        self.size_loop = self.tik_instance.Scalar("int32", name="size_loop")
        self.size_left = self.tik_instance.Scalar("int32", name="size_left")
        self.repeat_loop = self.tik_instance.Scalar("int32", name="repeat_loop")
        self.repeat_left = self.tik_instance.Scalar("int32", name="repeat_left")
        self.size_offset = self.tik_instance.Scalar("int32", name="size_offset")
        self.repeat_offset = self.tik_instance.Scalar("int32", name="repeat_offset")
        self.offset_in = self.tik_instance.Scalar("int32", name="offset_in")
        self.offset_out = self.tik_instance.Scalar("int32", name="offset_out")
        self.repeat_offset_src0 = self.tik_instance.Scalar("int32", name="repeat_offset_src0")
        self.repeat_offset_src1 = self.tik_instance.Scalar("int32", name="repeat_offset_src1")
        self.size_offset_src0 = self.tik_instance.Scalar("int32", name="size_offset_src0")
        self.size_offset_src1 = self.tik_instance.Scalar("int32", name="size_offset_src1")
        self.offset_dst = self.tik_instance.Scalar("int32", name="offset_dst")
        self.offset_src0 = self.tik_instance.Scalar("int32", name="offset_src0")
        self.offset_src1 = self.tik_instance.Scalar("int32", name="offset_src1")
        self.offset_h = self.tik_instance.Scalar("int32", name="offset_h")
        self.offset_w = self.tik_instance.Scalar("int32", name="offset_w")

    def select_tiling_mode(self, core_idx, core_ele, loop_num, loop_left):
        with self.tik_instance.if_scope(self.tiling_mode == 1):
            self.tiling_c_dim_core_nc(core_idx, core_ele)
        with self.tik_instance.if_scope(self.tiling_mode == 2):
            self.tiling_h_dim_core_nc(core_idx, core_ele, loop_num, loop_left)
        with self.tik_instance.if_scope(self.tiling_mode == 3):
            self.tiling_w_dim_core_nc(core_idx, core_ele, loop_num, loop_left)

    def vector_dup_continuous(self, src, size):
        with self.tik_instance.if_scope(size > 0):
            self.size_loop.set_as(size // MASK)
            self.size_left.set_as(size % MASK)
            self.repeat_loop.set_as(self.size_loop // REPEAT_LIMIT)
            self.repeat_left.set_as(self.size_loop % REPEAT_LIMIT)
            with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * MASK)
                self.tik_instance.vector_dup(MASK, src[self.repeat_offset], MIN_FP16, REPEAT_LIMIT, 1, 8)
            with self.tik_instance.if_scope(self.repeat_left > 0):
                self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * MASK)
                self.tik_instance.vector_dup(MASK, src[self.repeat_offset], MIN_FP16, self.repeat_left, 1, 8)
            with self.tik_instance.if_scope(self.size_left > 0):
                self.size_offset.set_as(self.size_loop * MASK)
                self.tik_instance.vector_dup(self.size_left, src[self.size_offset], MIN_FP16, 1, 1, 8)

    def vector_dup_discrete(self, src, repeat, size, dst_blk=1, dst_rep=8):
        with self.tik_instance.if_scope(tik.all(size > 0, repeat > 0)):
            self.size_loop.set_as(size // MASK)
            self.size_left.set_as(size % MASK)
            self.repeat_loop.set_as(repeat // REPEAT_LIMIT)
            self.repeat_left.set_as(repeat % REPEAT_LIMIT)

            def _inner(src, mask_len):
                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.tik_instance.vector_dup(mask_len, src[self.repeat_offset], MIN_FP16, REPEAT_LIMIT, dst_blk,
                                                 dst_rep)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.tik_instance.vector_dup(mask_len, src[self.repeat_offset], MIN_FP16, self.repeat_left, dst_blk,
                                                 dst_rep)

            with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                self.size_offset.set_as(size_loop_idx * MASK)
                _inner(src[self.size_offset:], MASK)
            with self.tik_instance.if_scope(self.size_left > 0):
                self.size_offset.set_as(self.size_loop * MASK)
                _inner(src[self.size_offset:], self.size_left)

    def reduce_max_rw(self, dst, src0, src1, size, src0_blk=1, src1_blk=1):
        if self.strides_w <= 31:
            with self.tik_instance.if_scope(size > 0):
                self.size_loop.set_as(size // MASK)
                self.size_left.set_as(size % MASK)
                self.repeat_loop.set_as(self.size_loop // REPEAT_LIMIT)
                self.repeat_left.set_as(self.size_loop % REPEAT_LIMIT)
                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * MASK)
                    self.repeat_offset_src0.set_as(repeat_loop_idx * REPEAT_LIMIT * src0_blk * MASK)
                    self.repeat_offset_src1.set_as(repeat_loop_idx * REPEAT_LIMIT * src1_blk * MASK)
                    self.tik_instance.vmax(MASK, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], REPEAT_LIMIT, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * MASK)
                    self.repeat_offset_src0.set_as(self.repeat_loop * REPEAT_LIMIT * src0_blk * MASK)
                    self.repeat_offset_src1.set_as(self.repeat_loop * REPEAT_LIMIT * src1_blk * MASK)
                    self.tik_instance.vmax(MASK, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], self.repeat_left, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
                with self.tik_instance.if_scope(self.size_left > 0):
                    self.size_offset.set_as(self.size_loop * MASK)
                    self.size_offset_src0.set_as(self.size_loop * src0_blk * MASK)
                    self.size_offset_src1.set_as(self.size_loop * src1_blk * MASK)
                    self.tik_instance.vmax(self.size_left, dst[self.size_offset], src0[self.size_offset_src0],
                                           src1[self.size_offset_src1], 1, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
        else:
            with self.tik_instance.if_scope(size > 0):
                self.size_loop.set_as(size // C_ZERO)
                with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                    self.size_offset.set_as(size_loop_idx * C_ZERO)
                    self.size_offset_src0.set_as(size_loop_idx * src0_blk * C_ZERO)
                    self.size_offset_src1.set_as(size_loop_idx * src1_blk * C_ZERO)
                    self.tik_instance.vmax(C_ZERO, dst[self.size_offset], src0[self.size_offset_src0],
                                           src1[self.size_offset_src1], 1, 1, 1, 1, 1, 1, 1)

    def reduce_avg_rw(self, dst, src0, src1, size, src0_blk=1, src1_blk=1):
        if self.strides_w <= 31:
            with self.tik_instance.if_scope(size > 0):
                self.size_loop.set_as(size // MASK)
                self.size_left.set_as(size % MASK)
                self.repeat_loop.set_as(self.size_loop // REPEAT_LIMIT)
                self.repeat_left.set_as(self.size_loop % REPEAT_LIMIT)
                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * MASK)
                    self.repeat_offset_src0.set_as(repeat_loop_idx * REPEAT_LIMIT * src0_blk * MASK)
                    self.repeat_offset_src1.set_as(repeat_loop_idx * REPEAT_LIMIT * src1_blk * MASK)
                    self.tik_instance.vadd(MASK, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], REPEAT_LIMIT, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * MASK)
                    self.repeat_offset_src0.set_as(self.repeat_loop * REPEAT_LIMIT * src0_blk * MASK)
                    self.repeat_offset_src1.set_as(self.repeat_loop * REPEAT_LIMIT * src1_blk * MASK)
                    self.tik_instance.vadd(MASK, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], self.repeat_left, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
                with self.tik_instance.if_scope(self.size_left > 0):
                    self.size_offset.set_as(self.size_loop * MASK)
                    self.size_offset_src0.set_as(self.size_loop * src0_blk * MASK)
                    self.size_offset_src1.set_as(self.size_loop * src1_blk * MASK)
                    self.tik_instance.vadd(self.size_left, dst[self.size_offset], src0[self.size_offset_src0],
                                           src1[self.size_offset_src1], 1, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
        else:
            with self.tik_instance.if_scope(size > 0):
                self.size_loop.set_as(size // C_ZERO)
                with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                    self.size_offset.set_as(size_loop_idx * C_ZERO)
                    self.size_offset_src0.set_as(size_loop_idx * src0_blk * C_ZERO)
                    self.size_offset_src1.set_as(size_loop_idx * src1_blk * C_ZERO)
                    self.tik_instance.vadd(C_ZERO, dst[self.size_offset], src0[self.size_offset_src0],
                                           src1[self.size_offset_src1], 1, 1, 1, 1, 1, 1, 1)

    def reduce_max_rh(self, dst, src0, src1, repeat, size, src0_blk=1, src1_blk=1, dst_rep=8, src0_rep=8, src1_rep=8):
        with self.tik_instance.if_scope(tik.all(size > 0, repeat > 0)):
            self.size_loop.set_as(size // MASK)
            self.size_left.set_as(size % MASK)
            self.repeat_loop.set_as(repeat // REPEAT_LIMIT)
            self.repeat_left.set_as(repeat % REPEAT_LIMIT)

            def _inner(dst, src0, src1, mask_len):
                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.repeat_offset_src0.set_as(repeat_loop_idx * REPEAT_LIMIT * src0_rep * C_ZERO)
                    self.repeat_offset_src1.set_as(repeat_loop_idx * REPEAT_LIMIT * src1_rep * C_ZERO)
                    self.tik_instance.vmax(mask_len, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], REPEAT_LIMIT, 1, src0_blk, src1_blk, dst_rep,
                                           src0_rep, src1_rep)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.repeat_offset_src0.set_as(self.repeat_loop * REPEAT_LIMIT * src0_rep * C_ZERO)
                    self.repeat_offset_src1.set_as(self.repeat_loop * REPEAT_LIMIT * src1_rep * C_ZERO)
                    self.tik_instance.vmax(mask_len, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], self.repeat_left, 1, src0_blk, src1_blk,
                                           dst_rep, src0_rep, src1_rep)

            with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                self.size_offset.set_as(size_loop_idx * MASK)
                self.size_offset_src0.set_as(size_loop_idx * src0_blk * MASK)
                self.size_offset_src1.set_as(size_loop_idx * src1_blk * MASK)
                _inner(dst[self.size_offset:], src0[self.size_offset_src0:], src1[self.size_offset_src1:], MASK)
            with self.tik_instance.if_scope(self.size_left > 0):
                self.size_offset.set_as(self.size_loop * MASK)
                self.size_offset_src0.set_as(self.size_loop * src0_blk * MASK)
                self.size_offset_src1.set_as(self.size_loop * src1_blk * MASK)
                _inner(dst[self.size_offset:], src0[self.size_offset_src0:], src1[self.size_offset_src1:],
                       self.size_left)

    def reduce_avg_rh(self, dst, src0, src1, repeat, size, src0_blk=1, src1_blk=1, dst_rep=8, src0_rep=8, src1_rep=8):
        with self.tik_instance.if_scope(tik.all(size > 0, repeat > 0)):
            self.size_loop.set_as(size // MASK)
            self.size_left.set_as(size % MASK)
            self.repeat_loop.set_as(repeat // REPEAT_LIMIT)
            self.repeat_left.set_as(repeat % REPEAT_LIMIT)

            def _inner(dst, src0, src1, mask_len):
                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.repeat_offset_src0.set_as(repeat_loop_idx * REPEAT_LIMIT * src0_rep * C_ZERO)
                    self.repeat_offset_src1.set_as(repeat_loop_idx * REPEAT_LIMIT * src1_rep * C_ZERO)
                    self.tik_instance.vadd(mask_len, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1],
                                           REPEAT_LIMIT, 1, src0_blk, src1_blk, dst_rep,
                                           src0_rep, src1_rep)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.repeat_offset_src0.set_as(self.repeat_loop * REPEAT_LIMIT * src0_rep * C_ZERO)
                    self.repeat_offset_src1.set_as(self.repeat_loop * REPEAT_LIMIT * src1_rep * C_ZERO)
                    self.tik_instance.vadd(mask_len, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], self.repeat_left, 1, src0_blk, src1_blk,
                                           dst_rep, src0_rep, src1_rep)

            with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                self.size_offset.set_as(size_loop_idx * MASK)
                self.size_offset_src0.set_as(size_loop_idx * src0_blk * MASK)
                self.size_offset_src1.set_as(size_loop_idx * src1_blk * MASK)
                _inner(dst[self.size_offset:], src0[self.size_offset_src0:], src1[self.size_offset_src1:], MASK)
            with self.tik_instance.if_scope(self.size_left > 0):
                self.size_offset.set_as(self.size_loop * MASK)
                self.size_offset_src0.set_as(self.size_loop * src0_blk * MASK)
                self.size_offset_src1.set_as(self.size_loop * src1_blk * MASK)
                _inner(dst[self.size_offset:], src0[self.size_offset_src0:], src1[self.size_offset_src1:],
                       self.size_left)

    def mean_func(self, repeat_o, ub_x, h_idx, w_idx):
        def _int_conv_float(win_int):
            win_fp = self.tik_instance.Scalar(self.fp_dtype, name="win_fp")
            if not tbe_platform.api_check_support("tik.vconv", "s322f32") or \
                    not tbe_platform.api_check_support("tik.vmuls", "float32"):
                int_temp = self.tik_instance.Tensor(self.int_dtype, (1,), name="int_temp", scope=tik.scope_ubuf)
                fp_temp = self.tik_instance.Tensor(self.fp_dtype, (1,), name="int_temp", scope=tik.scope_ubuf)
                int_temp[0].set_as(win_int)
                self.tik_instance.vec_conv(1, '', fp_temp, int_temp, 1, 8, 8, deqscale=1.0)
                self.tik_instance.vec_rec(1, fp_temp, fp_temp, 1, 8, 8)
                win_fp.set_as(fp_temp[0])
            else:
                self.tik_instance.scalar_conv('none', win_fp, win_int)
                win_fp.set_as(1.0 / win_fp)

            return win_fp

        self.size_loop.set_as(repeat_o * self.size_w // C_ZERO)
        if self.padding:
            self.factor_total.set_as(1.0 / (self.ksize_h * self.ksize_w))
            with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                self.size_offset.set_as(size_loop_idx * C_ZERO)
                self.tik_instance.vmuls(C_ZERO, ub_x[self.size_offset], ub_x[self.size_offset],
                                        self.factor_total, 1, 1, 1, 1, 1)
        else:
            with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                self.pos_h.set_as(h_idx + size_loop_idx // self.output_w)
                self.pos_w.set_as((w_idx + size_loop_idx) % self.output_w)
                self.real_h.set_as(self.ksize_h)
                self.real_w.set_as(self.ksize_w)

                with self.tik_instance.if_scope(self.pos_h * self.strides_h < self.pad_t):
                    self.real_h.set_as(self.ksize_h - (self.pad_t - self.pos_h * self.strides_h))
                with self.tik_instance.if_scope(self.pos_h * self.strides_h + self.ksize_h >
                                                (self.pad_h - self.pad_b)):
                    self.real_h.set_as(
                        self.ksize_h - (self.pos_h * self.strides_h + self.ksize_h - (self.pad_h - self.pad_b)))
                with self.tik_instance.if_scope(self.pos_w * self.strides_w < self.pad_l):
                    self.real_w.set_as(self.ksize_w - (self.pad_l - self.pos_w * self.strides_w))
                with self.tik_instance.if_scope(self.pos_w * self.strides_w + self.ksize_w >
                                                self.pad_w - self.pad_r):
                    self.real_w.set_as(
                        self.ksize_w - (self.pos_w * self.strides_w + self.ksize_w - (self.pad_w - self.pad_r)))
                window_int = self.tik_instance.Scalar(self.int_dtype, name="window_int")
                window_int.set_as(self.real_h * self.real_w)
                self.factor_total.set_as(_int_conv_float(window_int))
                self.size_offset.set_as(size_loop_idx * C_ZERO)
                self.tik_instance.vmuls(C_ZERO, ub_x[self.size_offset], ub_x[self.size_offset],
                                        self.factor_total, 1, 1, 1, 1, 1)

    def reduce_max(self, ub_x, ub_y, repeat_p, repeat_o, pos_idx):
        if self.ksize_w == 1:
            if self.strides_w == self.ksize_w:
                self.reduce_max_rw(ub_y, ub_x, ub_x, repeat_p * self.size_w, self.strides_w, self.strides_w)
            elif self.strides_w <= REPEAT_LIMIT:
                with self.tik_instance.if_scope(
                        tik.all(repeat_p >= self.repeat_w, self.output_w <= REPEAT_LIMIT,
                                self.pad_w <= REPEAT_LIMIT)):
                    self.reduce_max_rh(ub_y, ub_x, ub_x, repeat_p, self.size_w, self.strides_w, self.strides_w,
                                       self.output_w, self.pad_w, self.pad_w)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, repeat_p) as h_idx:
                        self.offset_dst.set_as(h_idx * self.size_w)
                        self.offset_src0.set_as(h_idx * self.pad_w * C_ZERO)
                        self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src0:],
                                           self.size_w, self.strides_w, self.strides_w)
            else:
                with self.tik_instance.for_range(0, repeat_p) as h_idx:
                    self.offset_dst.set_as(h_idx * self.size_w)
                    self.offset_src0.set_as(h_idx * self.pad_w * C_ZERO)
                    self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src0:],
                                       self.size_w, self.strides_w, self.strides_w)
        else:
            if self.strides_w == self.ksize_w:
                self.reduce_avg_rw(ub_y, ub_x, ub_x[C_ZERO:], repeat_p * self.size_w, self.strides_w, self.strides_w)
                with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                    self.offset_w.set_as((idx + 2) * C_ZERO)
                    self.reduce_avg_rw(ub_y, ub_x[self.offset_w:], ub_y, repeat_p * self.size_w, self.strides_w, 1)
            elif self.strides_w <= REPEAT_LIMIT:
                with self.tik_instance.if_scope(
                        tik.all(repeat_p >= self.repeat_w, self.output_w <= REPEAT_LIMIT,
                                self.pad_w <= REPEAT_LIMIT)):
                    self.reduce_avg_rh(ub_y, ub_x, ub_x[C_ZERO:],
                                       repeat_p, self.size_w, self.strides_w, self.strides_w,
                                       self.output_w, self.pad_w, self.pad_w)
                    with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                        self.offset_w.set_as((idx + 2) * C_ZERO)
                        self.reduce_avg_rh(ub_y, ub_x[self.offset_w:], ub_y, repeat_p, self.size_w, self.strides_w, 1,
                                           self.output_w, self.pad_w, self.output_w)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, repeat_p) as h_idx:
                        self.offset_dst.set_as(h_idx * self.size_w)
                        self.offset_src0.set_as(h_idx * self.pad_w * C_ZERO)
                        self.offset_src1.set_as(self.offset_src0 + C_ZERO)
                        self.reduce_avg_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src1:],
                                           self.size_w, self.strides_w, self.strides_w)
                        with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                            self.offset_w.set_as(self.offset_src0 + (idx + 2) * C_ZERO)
                            self.reduce_avg_rw(ub_y[self.offset_dst:], ub_x[self.offset_w:], ub_y[self.offset_dst:],
                                               self.size_w, self.strides_w, 1)
            else:
                with self.tik_instance.for_range(0, repeat_p) as h_idx:
                    self.offset_dst.set_as(h_idx * self.size_w)
                    self.offset_src0.set_as(h_idx * self.pad_w * C_ZERO)
                    self.offset_src1.set_as(self.offset_src0 + C_ZERO)
                    self.reduce_avg_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src1:],
                                       self.size_w, self.strides_w, self.strides_w)
                    with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                        self.offset_w.set_as(self.offset_src0 + (idx + 2) * C_ZERO)
                        self.reduce_avg_rw(ub_y[self.offset_dst:], ub_x[self.offset_w:], ub_y[self.offset_dst:],
                                           self.size_w, self.strides_w, 1)

        if self.ksize_h == 1:
            if self.strides_h == 1:
                self.reduce_max_rw(ub_x, ub_y, ub_y, repeat_o * self.size_w, 1, 1)
            else:
                with self.tik_instance.if_scope(
                        tik.all(repeat_o >= self.repeat_w, self.output_w <= REPEAT_LIMIT,
                                self.rep_blk_h <= REPEAT_LIMIT)):
                    self.reduce_max_rh(ub_x, ub_y, ub_y, repeat_o, self.size_w, 1, 1,
                                       self.output_w, self.rep_blk_h, self.rep_blk_h)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, repeat_o) as h_idx:
                        self.offset_dst.set_as(h_idx * self.size_w)
                        self.offset_src0.set_as(h_idx * self.strides_h * self.size_w)
                        self.reduce_max_rw(ub_x[self.offset_dst:], ub_y[self.offset_src0:], ub_y[self.offset_src0:],
                                           self.size_w, 1, 1)
        else:
            if self.strides_h == 1:
                self.reduce_avg_rw(ub_x, ub_y, ub_y[self.size_w:], repeat_o * self.size_w, 1, 1)
                with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                    self.offset_h.set_as((idx + 2) * self.size_w)
                    self.reduce_avg_rw(ub_x, ub_y[self.offset_h:], ub_x, repeat_o * self.size_w, 1, 1)
            else:
                with self.tik_instance.if_scope(
                        tik.all(repeat_o >= self.repeat_w, self.output_w <= REPEAT_LIMIT,
                                self.rep_blk_h <= REPEAT_LIMIT)):
                    self.reduce_avg_rh(ub_x, ub_y, ub_y[self.size_w:], repeat_o, self.size_w, 1, 1,
                                       self.output_w, self.rep_blk_h, self.rep_blk_h)
                    with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                        self.offset_h.set_as((idx + 2) * self.size_w)
                        self.reduce_avg_rh(ub_x, ub_y[self.offset_h:], ub_x, repeat_o, self.size_w, 1, 1,
                                           self.output_w, self.rep_blk_h, self.output_w)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, repeat_o) as h_idx:
                        self.offset_dst.set_as(h_idx * self.size_w)
                        self.offset_src0.set_as(h_idx * self.strides_h * self.size_w)
                        self.offset_src1.set_as(self.offset_src0 + self.size_w)
                        self.reduce_avg_rw(ub_x[self.offset_dst:], ub_y[self.offset_src0:], ub_y[self.offset_src1:],
                                           self.size_w, 1, 1)
                        with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                            self.offset_h.set_as(self.offset_src0 + (idx + 2) * self.size_w)
                            self.reduce_avg_rw(ub_x[self.offset_dst:], ub_y[self.offset_h:], ub_x[self.offset_dst:],
                                               self.size_w, 1, 1)
        self.mean_func(repeat_o, ub_x, pos_idx[0], pos_idx[1])

    def tiling_c_dim_core_nc(self, core_idx, core_ele):
        '''Tiling c1 dim
        '''
        self.size_1.set_as(self.pad_t * self.pad_w * C_ZERO + self.pad_l * C_ZERO)
        self.offset_2.set_as((self.pad_h - self.pad_b) * self.pad_w * C_ZERO - self.pad_r * C_ZERO)
        self.size_2.set_as(self.pad_b * self.pad_w * C_ZERO + self.pad_r * C_ZERO)
        self.offset_3.set_as(self.size_1 + self.input_w * C_ZERO)
        self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
        self.size_w.set_as(self.output_w * C_ZERO)
        self.burst_len_out.set_as(self.output_h * self.output_w)
        self.size.set_as(self.pad_h * self.pad_w * C_ZERO)
        self.repeat_w.set_as(self.size_w // MASK)
        self.rep_blk_h.set_as(self.output_w * self.strides_h)

        with self.tik_instance.if_scope(self.pad_h <= self.input_h):
            self.nburst.set_as(self.pad_h)
        with self.tik_instance.else_scope():
            self.nburst.set_as(self.input_h)
        with self.tik_instance.if_scope(self.pad_w <= self.input_w):
            self.repeat_3.set_as(0)
            self.burst_len_in.set_as(self.pad_w)
            self.src_stride.set_as(self.input_w - self.pad_w)
            self.dst_stride.set_as(0)
        with self.tik_instance.else_scope():
            self.repeat_3.set_as(self.pad_h - 1)
            self.burst_len_in.set_as(self.input_w)
            self.src_stride.set_as(0)
            self.dst_stride.set_as(self.pad_r + self.pad_l)

        with self.tik_instance.new_stmt_scope():
            def _inner(ele_idx, ub_x, ub_y):
                self.offset_in.set_as((core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w * C_ZERO)
                self.offset_out.set_as(
                    (core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w * C_ZERO)

                with self.tik_instance.if_scope(tik.all(self.pad_w > REPEAT_LIMIT, self.size_3 > 0)):
                    self.vector_dup_continuous(ub_x, self.size)
                with self.tik_instance.else_scope():
                    self.vector_dup_continuous(ub_x, self.size_1)
                    self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                    self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.pad_w)
                self.tik_instance.data_move(ub_x[self.size_1], self.input_gm[self.offset_in], 0, self.nburst,
                                            self.burst_len_in, self.src_stride, self.dst_stride)
                pos_idx = [0, 0]
                self.reduce_max(ub_x, ub_y, self.pad_h, self.output_h, pos_idx)

                self.tik_instance.data_move(self.output_gm[self.offset_out], ub_x, 0, 1, self.burst_len_out, 0, 0)

            self.init_ub_tensor()
            self.init_ub_scalar_double()
            with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                _inner(ele_idx * 2, self.ub_a, self.ub_b)
                _inner(ele_idx * 2 + 1, self.ub_c, self.ub_d)
            with self.tik_instance.if_scope(core_ele % 2 == 1):
                _inner(core_ele - 1, self.ub_a, self.ub_b)

    def tiling_h_dim_core_nc(self, core_idx, core_ele, loop_num, loop_left):
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.tiling_h_dim_core_nc_process(core_idx, core_ele, loop_idx, self.h_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_h_dim_core_nc_process(core_idx, core_ele, loop_num, loop_left)

    def tiling_h_dim_core_nc_process(self, core_idx, core_ele, loop_idx, ele):
        # common params
        self.before_h.set_as(loop_idx * self.h_factor * self.strides_h)
        self.after_h.set_as((loop_idx * self.h_factor + ele - 1) * self.strides_h + self.ksize_h)
        self.len_h.set_as(self.after_h - self.before_h)
        self.size_w.set_as(self.output_w * C_ZERO)
        self.burst_len_out.set_as(ele * self.output_w)
        self.size.set_as(self.len_h * self.pad_w * C_ZERO)
        self.repeat_w.set_as(self.size_w // MASK)
        self.rep_blk_h.set_as(self.output_w * self.strides_h)

        with self.tik_instance.if_scope(self.before_h < self.pad_t):
            self.size_1.set_as((self.pad_t - self.before_h) * self.pad_w * C_ZERO + self.pad_l * C_ZERO)
            self.offset_3.set_as(self.size_1 + self.input_w * C_ZERO)
            self.offset.set_as(0)
            with self.tik_instance.if_scope(self.after_h <= self.pad_h - self.pad_b):
                self.offset_2.set_as(self.len_h * self.pad_w * C_ZERO - self.pad_r * C_ZERO)
                self.size_2.set_as(self.pad_r * C_ZERO)
                self.repeat_3.set_as(self.after_h - self.pad_t - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
                self.nburst.set_as(self.after_h - self.pad_t)
            with self.tik_instance.else_scope():
                self.offset_2.set_as((self.pad_h - self.pad_b - self.before_h) * self.pad_w * C_ZERO -
                                     self.pad_r * C_ZERO)
                self.size_2.set_as((self.after_h - (self.pad_h - self.pad_b)) * self.pad_w * C_ZERO +
                                   self.pad_r * C_ZERO)
                self.repeat_3.set_as(self.input_h - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
                self.nburst.set_as(self.input_h)
        with self.tik_instance.else_scope():
            self.size_1.set_as(self.pad_l * C_ZERO)
            self.offset_3.set_as(self.size_1 + self.input_w * C_ZERO)
            self.offset.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO)
            with self.tik_instance.if_scope(self.after_h <= self.pad_h - self.pad_b):
                self.offset_2.set_as(self.len_h * self.pad_w * C_ZERO - self.pad_r * C_ZERO)
                self.size_2.set_as(self.pad_r * C_ZERO)
                self.repeat_3.set_as(self.len_h - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
                self.nburst.set_as(self.len_h)
            with self.tik_instance.else_scope():
                self.offset_2.set_as((self.pad_h - self.pad_b - self.before_h) * self.pad_w * C_ZERO -
                                     self.pad_r * C_ZERO)
                self.size_2.set_as((self.after_h - (self.pad_h - self.pad_b)) * self.pad_w * C_ZERO +
                                   self.pad_r * C_ZERO)
                self.repeat_3.set_as(self.pad_h - self.pad_b - self.before_h - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
                self.nburst.set_as(self.pad_h - self.pad_b - self.before_h)

        with self.tik_instance.if_scope(self.pad_w <= self.input_w):
            self.burst_len_in.set_as(self.pad_w)
            self.src_stride.set_as(self.input_w - self.pad_w)
            self.dst_stride.set_as(0)
        with self.tik_instance.else_scope():
            self.burst_len_in.set_as(self.input_w)
            self.src_stride.set_as(0)
            self.dst_stride.set_as(self.pad_r + self.pad_l)

        with self.tik_instance.new_stmt_scope():
            def _inner(ele_idx, ub_x, ub_y):
                self.offset_in.set_as((core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w * C_ZERO +
                                      self.offset)
                self.offset_out.set_as((core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w *
                                       C_ZERO + loop_idx * self.h_factor * self.output_w * C_ZERO)

                with self.tik_instance.if_scope(tik.all(self.pad_w > REPEAT_LIMIT, self.size_3 > 0)):
                    self.vector_dup_continuous(ub_x, self.size)
                with self.tik_instance.else_scope():
                    self.vector_dup_continuous(ub_x, self.size_1)
                    self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                    self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.pad_w)
                self.tik_instance.data_move(ub_x[self.size_1], self.input_gm[self.offset_in], 0, self.nburst,
                                            self.burst_len_in, self.src_stride, self.dst_stride)

                pos_idx = [loop_idx * self.h_factor, 0]
                self.reduce_max(ub_x, ub_y, self.len_h, ele, pos_idx)

                self.tik_instance.data_move(self.output_gm[self.offset_out], ub_x, 0, 1, self.burst_len_out, 0, 0)

            self.init_ub_tensor()
            self.init_ub_scalar_double()
            with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                _inner(ele_idx * 2, self.ub_a, self.ub_b)
                _inner(ele_idx * 2 + 1, self.ub_c, self.ub_d)
            with self.tik_instance.if_scope(core_ele % 2 == 1):
                _inner(core_ele - 1, self.ub_a, self.ub_b)

    def tiling_w_dim_core_nc(self, core_idx, core_ele, loop_num, loop_left):
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.tiling_w_dim_core_nc_process(core_idx, core_ele, loop_idx, self.w_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_w_dim_core_nc_process(core_idx, core_ele, loop_num, loop_left)

    def tiling_w_dim_core_nc_process(self, core_idx, core_ele, loop_idx, ele):
        # common params
        self.before_w.set_as(loop_idx * self.w_factor * self.strides_w)
        self.after_w.set_as((loop_idx * self.w_factor + ele - 1) * self.strides_w + self.ksize_w)
        self.len_w.set_as(self.after_w - self.before_w)
        self.size_w.set_as(ele * C_ZERO)
        self.burst_len_out.set_as(ele)
        self.size.set_as(self.ksize_h * self.len_w * C_ZERO)

        with self.tik_instance.for_range(0, self.output_h) as h_idx:
            self.before_h.set_as(h_idx * self.h_factor * self.strides_h)
            self.after_h.set_as(self.before_h + self.ksize_h)
            with self.tik_instance.if_scope(self.before_h < self.pad_t):
                self.size_1.set_as((self.pad_t - self.before_h) * self.len_w * C_ZERO)
                self.offset_2.set_as(0)
                self.size_2.set_as(0)
                self.repeat_3.set_as(self.after_h - self.pad_t)
                self.nburst.set_as(self.after_h - self.pad_t)

                with self.tik_instance.if_scope(self.before_w < self.pad_l):
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(self.size_1)
                        self.size_3.set_as((self.pad_l - self.before_w) * C_ZERO)
                        self.offset.set_as(0)
                        self.offset_ub.set_as(self.size_1 + (self.pad_l - self.before_w) * C_ZERO)
                        self.burst_len_in.set_as(self.after_w - self.pad_l)
                        self.src_stride.set_as(self.input_w - (self.after_w - self.pad_l))
                        self.dst_stride.set_as(self.pad_l - self.before_w)
                with self.tik_instance.else_scope():
                    self.offset.set_as((self.before_w - self.pad_l) * C_ZERO)
                    self.offset_ub.set_as(self.size_1)
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as(0)
                        self.burst_len_in.set_as(self.len_w)
                        self.src_stride.set_as(self.input_w - self.len_w)
                        self.dst_stride.set_as(0)
                    with self.tik_instance.else_scope():
                        self.offset_3.set_as(self.size_1 + (self.pad_w - self.pad_r - self.before_w) * C_ZERO)
                        self.size_3.set_as((self.after_w - (self.pad_w - self.pad_r)) * C_ZERO)
                        self.burst_len_in.set_as(self.pad_w - self.pad_r - self.before_w)
                        self.src_stride.set_as(self.before_w - self.pad_l)
                        self.dst_stride.set_as(self.after_w - (self.pad_w - self.pad_r))
            with self.tik_instance.else_scope():
                self.size_1.set_as(0)
                with self.tik_instance.if_scope(self.after_h <= self.pad_h - self.pad_b):
                    self.offset_2.set_as(0)
                    self.size_2.set_as(0)
                    self.repeat_3.set_as(self.ksize_h)
                    self.nburst.set_as(self.ksize_h)
                with self.tik_instance.else_scope():
                    self.offset_2.set_as((self.pad_h - self.pad_b - self.before_h) * self.len_w * C_ZERO)
                    self.size_2.set_as((self.after_h - (self.pad_h - self.pad_b)) * self.len_w * C_ZERO)
                    self.repeat_3.set_as(self.pad_h - self.pad_b - self.before_h)
                    self.nburst.set_as(self.pad_h - self.pad_b - self.before_h)

                with self.tik_instance.if_scope(self.before_w < self.pad_l):
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as((self.pad_l - self.before_w) * C_ZERO)
                        self.offset.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO)
                        self.offset_ub.set_as(self.size_3)
                        self.burst_len_in.set_as(self.after_w - self.pad_l)
                        self.src_stride.set_as(self.input_w - (self.after_w - self.pad_l))
                        self.dst_stride.set_as(self.pad_l - self.before_w)
                with self.tik_instance.else_scope():
                    self.offset.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO +
                                       (self.before_w - self.pad_l) * C_ZERO)
                    self.offset_ub.set_as(0)
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as(0)
                        self.burst_len_in.set_as(self.len_w)
                        self.src_stride.set_as(self.input_w - self.len_w)
                        self.dst_stride.set_as(0)
                    with self.tik_instance.else_scope():
                        self.offset_3.set_as((self.pad_w - self.pad_r - self.before_w) * C_ZERO)
                        self.size_3.set_as((self.after_w - (self.pad_w - self.pad_r)) * C_ZERO)
                        self.burst_len_in.set_as(self.pad_w - self.pad_r - self.before_w)
                        self.src_stride.set_as(self.before_w - self.pad_l)
                        self.dst_stride.set_as(self.after_w - (self.pad_w - self.pad_r))

            with self.tik_instance.new_stmt_scope():
                def _inner(ele_idx, ub_x, ub_y):
                    self.offset_in.set_as((core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w *
                                          C_ZERO + self.offset)
                    self.offset_out.set_as((core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w *
                                           C_ZERO + h_idx * self.output_w * C_ZERO + loop_idx * self.w_factor * C_ZERO)
                    with self.tik_instance.if_scope(tik.all(self.len_w > REPEAT_LIMIT, self.size_3 > 0)):
                        self.vector_dup_continuous(ub_x, self.size)
                    with self.tik_instance.else_scope():
                        self.vector_dup_continuous(ub_x, self.size_1)
                        self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                        self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.len_w)
                    self.tik_instance.data_move(ub_x[self.offset_ub], self.input_gm[self.offset_in], 0, self.nburst,
                                                self.burst_len_in, self.src_stride, self.dst_stride)
                    if self.ksize_w == 1:
                        with self.tik_instance.for_range(0, self.ksize_h) as k_idx:
                            self.offset_dst.set_as(k_idx * self.size_w)
                            self.offset_src0.set_as(k_idx * self.len_w * C_ZERO)
                            self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:],
                                               ub_x[self.offset_src0:],
                                               self.size_w, self.strides_w, self.strides_w)
                    else:
                        with self.tik_instance.for_range(0, self.ksize_h) as k_idx:
                            self.offset_dst.set_as(k_idx * self.size_w)
                            self.offset_src0.set_as(k_idx * self.len_w * C_ZERO)
                            self.offset_src1.set_as(self.offset_src0 + C_ZERO)
                            self.reduce_avg_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:],
                                               ub_x[self.offset_src1:],
                                               self.size_w, self.strides_w, self.strides_w)
                            with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                                self.offset_w.set_as(self.offset_src0 + (idx + 2) * C_ZERO)
                                self.reduce_avg_rw(ub_y[self.offset_dst:], ub_x[self.offset_w:],
                                                   ub_y[self.offset_dst:],
                                                   self.size_w, self.strides_w, 1)
                    if self.ksize_h == 1:
                        self.reduce_max_rw(ub_x, ub_y, ub_y, self.size_w, 1, 1)
                    else:
                        self.reduce_avg_rw(ub_x, ub_y, ub_y[self.size_w:], self.size_w, 1, 1)
                        with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                            self.offset_h.set_as((idx + 2) * self.size_w)
                            self.reduce_avg_rw(ub_x, ub_y[self.offset_h:], ub_x, self.size_w, 1, 1)

                    self.mean_func(1, ub_x, h_idx, loop_idx * self.w_factor)
                    self.tik_instance.data_move(self.output_gm[self.offset_out], ub_x, 0, 1, self.burst_len_out, 0, 0)

                self.init_ub_tensor()
                self.init_ub_scalar_double()
                with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                    _inner(ele_idx * 2, self.ub_a, self.ub_b)
                    _inner(ele_idx * 2 + 1, self.ub_c, self.ub_d)
                with self.tik_instance.if_scope(core_ele % 2 == 1):
                    _inner(core_ele - 1, self.ub_a, self.ub_b)

    def avg_pool_compute_tiling(self):
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_idx:
            self.tiling_ub = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,),
                                                      name="tiling_ub",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 3, 0, 0)
            self.tiling_args()
            self.init_ub_scalar_single()
            with self.tik_instance.if_scope(core_idx < self.act_core_num - 1):
                self.core_ele.set_as(self.one_core_ele)
                self.loop_num.set_as(self.one_core_loop_num)
                self.loop_left.set_as(self.one_core_loop_left)
            with self.tik_instance.if_scope(core_idx == self.act_core_num - 1):
                self.core_ele.set_as(self.last_core_ele)
                self.loop_num.set_as(self.last_core_loop_num)
                self.loop_left.set_as(self.last_core_loop_left)
            self.select_tiling_mode(core_idx, self.core_ele, self.loop_num, self.loop_left)

    def avg_pool_operator(self):
        self.avg_pool_compute_tiling()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm])
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_ele": self.ub_ele,
                "core_num": self.core_num,
                "ksize_h": self.ksize_h,
                "ksize_w": self.ksize_w,
                "strides_h": self.strides_h,
                "strides_w": self.strides_w,
                "padding": self.padding
            }
        )
        return self.tik_instance


def check_avg_pool_range(inputs: dict, ksize: Union[tuple, list], strides: Union[tuple, list], padding: str) -> list:
    """
    graph mode fuzz, check input range
    """
    op_type = "avg_pool"
    input_range = inputs.get("ori_range")
    x_format = inputs.get("ori_format")

    if x_format == "NCHW":
        idx_h = 2
        idx_w = 3
    elif x_format == "NHWC":
        idx_h = 1
        idx_w = 2
    else:
        error_manager_cube.raise_err_specific_user(op_type, "input fmap format only support NCHW or NHWC.")

    check_range_value(op_type, input_range, idx_h, idx_w)

    kh = ksize[idx_h]
    kw = ksize[idx_w]
    if padding == "SAME":
        pads = [-1, -1, -1, -1]
    else:
        pads = [0, 0, 0, 0]

    low_check = check_input_range(input_range, idx_h, idx_w, kh, kw, pads)
    up_check = check_range_l1_size(inputs, kh, kw, strides, pads)

    if not up_check and not low_check:
        return []

    type_info = []
    if low_check:
        type_info.append(low_check)
    elif up_check:
        type_info.append(up_check)

    json_str = [{"result": "UNSUPPORTED", "reason": {"param_index": [0], "type": type_info}}]
    return json_str


def gen_avg_pool_range(inputs, ksize, strides, padding):
    """
    fuzz input range
    """
    op_type = "avg_pool"
    x_shape = inputs.get("ori_shape")
    x_format = inputs.get("ori_format")
    x_range = inputs.get("ori_range")
    data_type = inputs.get("dtype")

    idx_n = 0
    if x_format == "NCHW":
        idx_c = 1
        idx_h = 2
        idx_w = 3
    elif x_format == "NHWC":
        idx_h = 1
        idx_w = 2
        idx_c = 3
    else:
        error_manager_cube.raise_err_specific_user(op_type, "input fmap format only support NCHW or NHWC")

    # x_range instance when empty
    if not x_range:
        x_range = []
        for idx, _ in enumerate(x_shape):
            if x_shape[idx] == DYNAMIC_VALUE:
                x_range.append([1, -1])
            else:
                x_range.append([x_shape[idx], x_shape[idx]])

    kh = ksize[idx_h]
    kw = ksize[idx_w]
    if padding == "SAME":
        pads = [-1, -1, -1, -1]
    else:
        pads = [0, 0, 0, 0]
    grade_n = [0, 1, 3, 7, 15, 31, ((1 << 31) - 1)]
    grade_h = [0, 3, 15, 63, 127, 191, 255, 511, 767, 1023, 4096]
    grade_w = [0, 3, 15, 63, 127, 191, 255, 511, 767, 1023, 4096]
    grade_map = {idx_n : grade_n, idx_h : grade_h, idx_w : grade_w}
    input_range = [[], [], [], []]

    for idx, grade_item in grade_map.items():
        # allow input_shape -1 with range
        if x_shape[idx] == DYNAMIC_VALUE:
            input_range[idx] = x_range[idx]
        else:
            input_range[idx] = create_fuzz_range(op_type, x_shape[idx], grade_item)

    input_range[idx_c] = [x_shape[idx_c], x_shape[idx_c]]
    log.debug("avgpool fuzz input range is :%s", input_range)
    # output_h or output_w > 0
    correct_input_range(op_type, input_range, x_shape, idx_h, idx_w, kh, kw, pads)
    log.debug("avgpool fuzz input range is corrected for output_w > 0, :%s", input_range)
    # check fmap exceed l1buffer
    if x_shape[idx_w] != DYNAMIC_VALUE:
        check_l1_size(op_type, inputs, kh, kw, strides, pads)
    attr_params = [strides, kh, kw, pads]
    new_in_range = modify_input_range(inputs, input_range, data_type, idx_h, idx_w, attr_params)
    log.debug("avgpool fuzz input range is modified for no exceed l1buffer, :%s", new_in_range)
    return new_in_range


@tbe_register.register_param_generalization("AvgPool")
def avg_pool_generalization(x: dict, filter: dict, bias: dict, y: dict, ksize: Union[tuple, list],
                            strides: Union[tuple, list], padding: str = "VALID", data_format: str = 'NHWC',
                            offset_x: int = 0, kernel_name: str = "avg_pool", generalize_config: dict = None) -> list:
    """
    avg_pool generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to avg_pool

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    if generalize_config is None:
        generalize_config = {"mode": "keep_rank"}
    support_mode = ["keep_rank"]
    if generalize_config.get("mode") not in support_mode:
        error_manager_cube.raise_err_specific_user("avg_pool", "invalid generalize mode {}, only support {}".format(
            str(generalize_config.get("mode")), str(support_mode)))

    # unknow_rank inputs ori_shape is [-2], others' shape length is 4
    unknow_rank = len(x["ori_shape"]) == 1 and x["ori_shape"][0] == -2
    if unknow_rank:
        error_manager_cube.raise_err_specific_user("avg_pool", "not support unknow_rank under mode {}".format(
            generalize_config["mode"]))
    log.debug("avgpool generalization inputs: %s", x)

    if not check_graph_mode(x):
        x_range = gen_avg_pool_range(x, ksize, strides, padding)
        x["ori_range"] = x_range
        have_range = {"x": x, "y": y}
        result = []
        for name, tensor in have_range.items():
            # only change shape NHW dim to -1, range is already set at infershape
            valid = isinstance(tensor.get("ori_shape"), (list, tuple)) and len(tensor["ori_shape"]) == ORI_SHAPE_LEN
            if not valid:
                error_manager_cube.raise_err_specific_user("avg_pool", "invalid {} ori_shape {}, only support {}d"
                    .format(name, str(tensor.get("ori_shape")), str(ORI_SHAPE_LEN)))
            if tensor.get("ori_format") == "NCHW":
                tensor["ori_shape"] = [-1, tensor["ori_shape"][1], -1, -1]
            else:
                tensor["ori_shape"] = [-1, -1, -1, tensor["ori_shape"][3]]
        result.append([x, filter, bias, y, ksize, strides, padding, data_format, offset_x, kernel_name])
        log.debug("avgpool generalization result: %s", result)
    else:
        json_str = check_avg_pool_range(x, ksize, strides, padding)
        if not json_str:
            result = [[x, filter, bias, y, ksize, strides, padding, data_format, offset_x, kernel_name]]
            log.debug("avgpool generalization result: %s", result)
        else:
            result = json_str
            log.debug("avgpool generalization invalid range, check result: %s", result)
    return result


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=invalid-name,redefined-builtin,too-many-locals,unused-argument,unused-variable,unnecessary-lambda
def check_supported(x, filter, bias, y, ksize, strides,
                    padding="VALID", data_format="NHWC", offset_x=0,
                    kernel_name="avg_pool"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16 or int8

    filter : dict, optional input, shape and dtype of input_data, only support float16 or int8

    bias : dict, optional input, shape and dtype of input_data, only support int32

    y : dict, shape and dtype of output_data, only support float16 or int32

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support padding and not padding

    data_format : str, default = "NHWC"

    offset_x : int, quantization parameter

    kernel_name : cce kernel name, default value is "avg_pool_cce"

    Returns
    -------
    True or False
    """
    ori_shape = y.get("ori_shape")
    if data_format == "NHWC":
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        outputh = ori_shape[1]
        outputw = ori_shape[2]
    elif data_format == "NCHW":
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        outputh = ori_shape[2]
        outputw = ori_shape[3]
    else:
        reason = "data_format[%s] is not in [NHWC, NCHW]." % data_format
        return False, reason
    is_support_kernel = (ksize_h * ksize_w <= AVG_KERNEL_SIZE_H_MUL_W) or \
                        (ksize_h <= AVG_KERNEL_SIZE and ksize_w <= AVG_KERNEL_SIZE)
    if not is_support_kernel and outputh != 1 and outputw == 1:
        reason = "ksize is too large, output_h is not equal to 1, and output_w is equal to 1,\
                  this shape is not supported by schedule, ksize:%s ori_shape:%s"\
                  % (str(ksize), str(ori_shape))
        return False, reason
    if not is_support_kernel and not (outputh == 1 and outputw == 1):
        reason = "ksize is too large, output_h and output_w are not equal to 1 at the same time,\
                  this shape is not supported by schedule, ksize:%s ori_shape:%s"\
                  % (str(ksize), str(ori_shape))
        return False, reason
    return True


def get_op_specific_info(x, filter, bias, y, ksize, strides,
                         padding="VALID", data_format="NHWC", offset_x=0,
                         kernel_name="avg_pool"):
    """
    get the avgpool prebuild pattern

    """
    if bias is None and filter is not None:
        return '{"prebuildPattern": "Convolution"}'
    return '{"prebuildPattern": "undefined"}'


def get_op_support_info(x, filter, bias, y, ksize, strides,
                        padding="VALID", data_format="NHWC", offset_x=0,
                        kernel_name="avg_pool"):
    """
    get the avgpool split
    """
    format_x = x.get("format")
    input_shape = x.get("shape")

    if data_format in ("NHWC",):
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        window = [ksize[1], ksize[2]]
    else:
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        window = [ksize[2], ksize[3]]

    if format_x == "NC1HWC0":
        if (ksize_h == window[0] and ksize_w == window[1]) or padding == "SAME":
            axis_split_matrix = [[util_select_op_base.SplitInput([0, [0], [-1], [-1]]),
                                 util_select_op_base.SplitOutput([0, [0]])]]
        elif padding == "VALID":
            axis_split_matrix = [
                [util_select_op_base.SplitInput([0, [0], [-1], [-1]]), util_select_op_base.SplitOutput([0, [0]])],
                [util_select_op_base.SplitInput([0, [2], [0], [0]]), util_select_op_base.SplitOutput([0, [2]])]]
        else:
            axis_split_matrix = None
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_matrix, axis_reduce_list, 2, 0)

    return op_cal_info_in_json


def set_default_para():
    """
    set default parameter value
    """
    default_para = {}
    default_para["res_dtype"] = "float16"
    default_para["optim_dict"] = {"c0_optim_flg": False}
    default_para["fusion_para"] = {"input_memory_type": 0, "output_memory_type": 0,
                                   "valid_shape": (), "slice_offset": (),
                                   "l1_fusion_type": -1}
    default_para["ori_shape"] = [0, 0, 0, 0]
    return default_para


def _collect_org_tensors(ori_paras):
    """
    get valid tensors
    """
    ori_tensors = {}
    for key, value in ori_paras.items():
        valid_tensor = isinstance(value, dict) \
                       and isinstance(value.get("ori_shape"), (list, tuple)) \
                       and len(value.get("ori_shape")) > 0
        if valid_tensor:
            ori_tensors[key] = value
    return ori_tensors


# pylint: disable=locally-disabled,too-many-arguments,too-many-statements
# pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
def _check_window_rule(ksize, strides, padding, data_format, offset_x):
    """
    check ksize and strides of window in pooling
    :param ksize: list or tuple, the length must be 4
    :param strides: list or tuple, the length must be 4
    :param data_format: input format
    :return: None
    """
    if len(ksize) != 4:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_012
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'ksize'
        error_info['min_value'] = '4'
        error_info['max_value'] = '4'
        error_info['real_value'] = len(ksize)
        raise RuntimeError(error_info,
                           "In op[%s], the num of dimensions of input[%s] "
                           "should be in the range of [%s, %s], "
                           "but actually is [%s]." %
                           (error_info.get('op_name'), error_info.get('param_name'),
                            error_info.get('min_value'), error_info.get('max_value'),
                            error_info.get('real_value')))

    if len(strides) != 4:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_012
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'strides'
        error_info['min_value'] = '4'
        error_info['max_value'] = '4'
        error_info['real_value'] = len(strides)
        raise RuntimeError(error_info,
                           "In op[%s], the num of dimensions of input[%s] "
                           "should be in the range of [%s, %s], "
                           "but actually is [%s]." %
                           (error_info.get('op_name'), error_info.get('param_name'),
                            error_info.get('min_value'), error_info.get('max_value'),
                            error_info.get('real_value')))

    ksize_c = ksize[3] if data_format in ("NHWC",) else ksize[1]
    strides_c = strides[3] if data_format in ("NHWC",) else strides[1]
    window_size = ksize[1] * ksize[2] if data_format in ("NHWC",) else ksize[2] * ksize[3]

    if ksize[0] != 1 or (ksize_c != 1):
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_000
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = ",".join(("ksize[1]", "ksize[3]"))
        error_info['expected_value'] = '1'
        error_info['real_value'] = ",".join((str(ksize[1]), str(ksize[3])))
        raise RuntimeError("In op[%s], the parameter[%s] should be [%s], "
                           "but actually is [%s]." %
                           (error_info.get('op_name'), error_info.get('param_name'),
                            error_info.get('expected_value'),
                            error_info.get('real_value')))

    if strides[0] != 1 or strides_c != 1:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_000
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = ",".join(("strides[1]", "strodes[3]"))
        error_info['expected_value'] = '1'
        error_info['real_value'] = ",".join((str(strides[1]), str(strides[3])))
        raise RuntimeError(error_info, "In op[%s], the parameter[%s] should be [%s], "
                                       "but actually is [%s]." % (error_info.get('op_name'),
                                                                  error_info.get('param_name'),
                                                                  error_info.get('expected_value'),
                                                                  error_info.get('real_value')))

    if padding not in ("SAME", "VALID"):
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_015
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'padding'
        error_info['expected_value_list'] = ",".join(("SAME", "VALID"))
        error_info['real_value'] = padding
        raise RuntimeError(error_info, "In op[%s], parameter[%s] should be one of [%s], "
                                       "but actually is [%s]." % (error_info.get('op_name'),
                                                                  error_info.get('param_name'),
                                                                  error_info.get('expected_value_list'),
                                                                  error_info.get('real_value')))

    if data_format not in ("NCHW", "NHWC", "NC1HWC0"):
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_015
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'x'
        error_info['excepted_format_list'] = ",".join(("NC1HWC0",
                                                       "NCHW", "NHWC"))
        error_info['format'] = data_format
        raise RuntimeError(error_info, "In op[%s], the format[%s] of input "
                                       "should be one of [%s], "
                                       "but actuall"
                                       "y is [%s]."
                           % (error_info.get('op_name'),
                              error_info.get('param_name'),
                              error_info.get('excepted_format_list'),
                              error_info.get('format')))

    if offset_x != 0:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_000
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'offset_x'
        error_info['expected_value'] = '0'
        error_info['real_value'] = str(offset_x)
        raise RuntimeError(error_info, "In op[%s], the parameter[%s] should be [%s], "
                                       "but actually is [%s]."
                           % (error_info.get('op_name'),
                              error_info.get('param_name'),
                              error_info.get('expected_value'),
                              error_info.get('real_value')))


def _avg_pool_check_rule(input_shape, input_dtype, output_dtype,
                          ksize, strides, padding, data_format, offset_x, kernel_name):
    """
    :param input_shape: shape of input_data
    :param input_dtype: dtype of input_data
    :param output_dtype: dtype of output_data
    :param ksize: the window of avgpooling
    :param strides: the stride of avgpooling window
    :param padding: padding_mode of avgpool
    :param data_format: NHWC default
    :param offset_x: default 0
    :param kernel_name: cce kernel name
    :return: None
    """
    if len(input_shape) != INPUT_DIM:
        error_manager_cube.raise_err_specific_user("avg_pool",
                                                   "ori_input_shape dim should be 4.")
    para_check.check_shape(input_shape)
    para_check.check_dtype(input_dtype, ["float16", "int8"])
    para_check.check_dtype(output_dtype, ["float16", "int8", "int32"])

    _check_window_rule(ksize, strides, padding, data_format, offset_x)


def _check_filter_window(fmap, filter, window, stride):
    """
    check filter window size
    """
    fmap_shape = fmap.get("ori_shape")
    filter_shape = filter.get("ori_shape")
    filter_format = filter.get("ori_format")
    if stride[0] > MAX_CUBE_STRIDE or stride[1] > MAX_CUBE_STRIDE:
        raise RuntimeError("In op[%s], the [%s] should less than [%s] when filter is None"
                           % ('avgpool', 'stride', str(MAX_CUBE_STRIDE)))
    if filter_format not in ("NCHW", "NHWC"):
        raise RuntimeError("In op[%s], the ori_format of filter "
                                       "should be [%s] or [%s]"
                           % ('avgpool', 'NCHW', 'NHWC'))
    h_index = filter_format.index("H")
    w_index = filter_format.index("W")
    c_index = filter_format.index("C")
    n_index = filter_format.index("N")
    if filter_shape[h_index] != window[0] or filter_shape[w_index] != window[1]:
        raise RuntimeError("In op[%s], the h_shape of filter "
                                       "should be equal with [%s],"
                           % ('avgpool', 'ksize'))
    if filter_shape[c_index] != 1:
        raise RuntimeError("In op[%s], the c_shape of filter "
                                       "should be [%s],"
                           % ('avgpool', '1'))
    if filter_shape[n_index] != fmap_shape[fmap.get("ori_format").index("C")]:
        raise RuntimeError("In op[%s], the N shape of filter "
                                       "should be equal with C shape of fmap,"
                           % ('avgpool'))


def _avgpool_compute(x, filters, bias, offset_w, y, strides, padding, dilations,
                    groups=1, data_format='NHWC', offset_x=0, kernel_name="avg_pool",
                    dsl_flag=True):

    """
    avg_pool compute

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    x: dict with keys(shape and dtype)
        input 4d feature map tensor
    filters: dict with keys(shape and dtype)
        input 4d weight tensor
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    offset_w: keys(shape and dtype) or None
        input offset_w tensor
    y: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    padding:  "SAME", "VALID"
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset for fmap

    Returns
    -------
    tvm compute
    """

    default_para = set_default_para()
    if not y.get("ori_shape"):
        y["ori_shape"] = default_para["ori_shape"]
    if padding == "SAME":
        pads = [-1, -1, -1, -1]
    else:
        pads = [0, 0, 0, 0]
    ori_paras = {
        "inputs": x, "weights": filters, "bias": bias, "offset_w": offset_w,
        "outputs": y, "strides": strides, "pads": pads, "dilations": dilations,
        "groups": groups, "data_format": data_format, "offset_x": offset_x,
        "kernel_name": kernel_name, "optim_dict": default_para.get("optim_dict"),
    }

    avgpool_para = Conv2dParaProcess(ori_paras)
    paras = avgpool_para.config_paras()

    pad_t, pad_b, pad_l, pad_r = avgpool_para.pads
    op_res = conv(paras.get("input_tensor"), paras.get("weight_tensor"),
                  {"bias_tensor": paras.get("bias_tensor"),
                   "offset_w_tensor": offset_w,
                   "pad_h": [pad_t, pad_b], "pad_w": [pad_l, pad_r],
                   "stride_h": avgpool_para.strides[H_DIM], "stride_w": avgpool_para.strides[W_DIM],
                   "dilate_h": avgpool_para.dilations[H_DIM], "dilate_w": avgpool_para.dilations[W_DIM],
                   "filter_h": paras.get("w_shape")[H_DIM],
                   "filter_w": paras.get("w_shape")[W_DIM],
                   "offset_x": offset_x,
                   "res_dtype": default_para.get("res_dtype"),
                   "fusion_para": default_para.get("fusion_para"),
                   "kernel_name": kernel_name,
                   "group": avgpool_para.groups,
                   "enlarge": paras.get("group_para").get("enlarge"),
                   "c1_opt": paras.get("group_para").get("c1_opt"),
                   "cout1_opt": paras.get("group_para").get("cout1_opt"),
                   "group_opt": paras.get("group_para").get("group_opt"),
                   "a_shape": paras.get("in_shape_nc1hwc0"),
                   "weight_fracz_shape": paras.get("w_shape_frac_z"),
                   "weight_ori_shape_nchw": paras.get("w_shape"),
                   "correct_range_flag": paras.get("correct_range_flag", False),
                   "new_in_range": paras.get("new_in_range"),
                   "ori_tensors": _collect_org_tensors(ori_paras)},
                  optim_dict=default_para.get("optim_dict"),
                  dsl_flag=dsl_flag)

    if avgpool_para.is_tensor is True:
        return op_res
    return {"op_placeholder": [paras.get("input_tensor"), paras.get("weight_tensor")], "op_res": [op_res]}


def assist_matrix_compute(res):
    """
    construnt assist_matrix
    """
    out_h = ConvParam.h_out
    out_w = ConvParam.w_out
    pad_t, _, pad_l, _ = ConvParam.padding
    input_h, input_w = ConvParam.h_in, ConvParam.w_in
    filter_h, filter_w = ConvParam.filter_h, ConvParam.filter_w
    stride_h, stride_w = ConvParam.stride_h, ConvParam.stride_w
    c_ub = res.get("op_res")[0]
    res_dtype = c_ub.dtype
    conv_shape = c_ub.shape
    if ConvParam.v200_width_out_1_flag:
        out_w = 1
    if res["padding_mode"] == "VALID":
        if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend310",):
            c_ub_avg = tvm.compute(conv_shape,
                                   lambda n, c1, m, c0:
                                   tvm.div(c_ub(n, c1, m, c0), filter_h * filter_w).astype(res_dtype),
                                   name="c_ub_avg",
                                   tag="elewise_binary_div")
        else:
            mean_matrix_shape = c_ub.shape[3]
            mean_matrix = tvm.compute(mean_matrix_shape, lambda * index:
                                    tvm.const(filter_h * filter_w).astype(res_dtype),
                                    name="mean_matrix")
            c_ub_avg = tvm.compute(conv_shape,
                                   lambda n, c1, m, c0:
                                   tvm.div(c_ub(n, c1, m, c0), mean_matrix(c0)).astype(res_dtype),
                                   name="c_ub_avg",
                                   tag="elewise_binary_div")
            ConvParam.tensor_map["mean_matrix"] = mean_matrix
    else:
        mean_matrix_shape = c_ub.shape[2:4]
        mean_matrix = tvm.compute(mean_matrix_shape, lambda m, c0:
                                  tvm.max(
                                      (tvm.min((m // out_w) * stride_h - pad_t + filter_h, input_h) -
                                      tvm.max((m // out_w) * stride_h - pad_t, 0)) * \
                                      (tvm.min((m % out_w) * stride_w - pad_l + filter_w, input_w) -
                                      tvm.max((m % out_w) * stride_w - pad_l, 0)), 1
                                  ).astype("int"),
                                  name="mean_matrix")
        mean_matrix_fp16 = tvm.compute(mean_matrix_shape, lambda *index:
                                       mean_matrix(*index).astype(res_dtype),
                                       name="mean_matrix_fp16",
                                       tag="elewise_single_cast")
        if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend310",):
            mean_matrix_rec = tvm.compute(mean_matrix_shape, lambda *index:
                                          1 / mean_matrix_fp16(*index),
                                          name="mean_matrix_rec",
                                          tag="elewise_single_rec")
            c_ub_avg = tvm.compute(conv_shape, lambda n, c1, m, c0:
                                   c_ub(n, c1, m, c0) * mean_matrix_rec(m, c0),
                                   name="c_ub_avg",
                                   tag="elewise_binary_mul")
        else:
            c_ub_avg = tvm.compute(conv_shape, lambda n, c1, m, c0:
                                   tvm.div(c_ub(n, c1, m, c0), mean_matrix_fp16(m, c0)).astype(res_dtype),
                                   name="c_ub_avg",
                                   tag="elewise_binary_div")
        ConvParam.tensor_map["mean_matrix"] = mean_matrix
    return c_ub_avg


@register_operator("AvgPool")
@para_check.check_input_type(dict, (dict, NONETYPE), (dict, NONETYPE), dict,
                             (tuple, list), (tuple, list),
                             str, str, int, str)
def avg_pool(x, filter, bias, y, ksize, strides,
             padding="VALID", data_format="NHWC", offset_x=0,
             kernel_name="avg_pool"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16 or int8

    filter : dict, optional input, shape and dtype of input_data, only support float16 or int8

    bias : dict, optional input, shape and dtype of input_data, only support int32

    y : dict, shape and dtype of output_data, only support float16 or int32

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support padding and not padding

    data_format : str, default = "NHWC"

    offset_x : int, quantization parameter

    kernel_name : cce kernel name, default value is "avg_pool_cce"

    Returns
    -------
    None
    """

    # get shape&dtype
    # input_shape only support format NCHW
    input_shape = x.get("ori_shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()
    input_format = x.get("ori_format")
    output_dtype = y.get("dtype")
    output_dtype = output_dtype.lower()
    output_format = y.get("ori_format")
    if not output_format:
        y["ori_format"] = input_format
    elif output_format not in ("NCHW", "NHWC"):
        error_manager_cube.raise_err_one_para("E62006", "avg_pool",
                                              "output_format should be 'NCHW or 'NHWC'")

    _avg_pool_check_rule(input_shape, input_dtype, output_dtype, ksize, strides, padding,
                         data_format, offset_x, kernel_name)
    if input_format == "NCHW":
        input_c, input_h, input_w = input_shape[1:4]
        stride = [-1, -1, strides[2], strides[3]]
        window = [ksize[2], ksize[3]]
        strides_h = strides[2]
        strides_w = strides[3]

    elif input_format == "NHWC":
        input_h, input_w, input_c = input_shape[1:4]
        stride = [-1, strides[1], strides[2], -1]
        window = [ksize[1], ksize[2]]
        strides_h = strides[1]
        strides_w = strides[2]
    else:
        raise RuntimeError("Unsupported input format!")
    tbe_context.get_context().add_compile_info("strides_h", strides_h)
    tbe_context.get_context().add_compile_info("strides_w", strides_w)
    tbe_context.get_context().add_compile_info("k_size_h", window[0])
    tbe_context.get_context().add_compile_info("k_size_w", window[1])

    if bias is None and filter is not None:
        log.info("[%s]: Enter avgpool cube dynamic branch", kernel_name)
        tbe_context.get_context().add_compile_info("filter", 1)
        dilations = (1, 1, 1, 1)
        _check_filter_window(x, filter, window, stride)
        offset_w = None
        with tbe.compute():
            res = _avgpool_compute(
                x, filter, bias, offset_w, y, stride, padding, dilations,
                input_c, data_format, offset_x, kernel_name, dsl_flag=True)
        res["padding_mode"] = padding
        c_ub_avg = assist_matrix_compute(res)
        with tvm.target.cce():
            sch = tbe.auto_schedule(c_ub_avg)
        tensor_list = res.get("op_placeholder") + [c_ub_avg]
        config = {
            "print_ir": False,
            "name": kernel_name,
            "tensor_list": tensor_list,
            "build_args": {"constant_realize_extent_in_infer_bound": False, "dummy_placeholder": True}
        }
        tbe.build(sch, config)
    else:
        log.info("[%s]: Enter avgpool vector dynamic branch", kernel_name)
        if filter is None:
            tbe_context.get_context().add_compile_info("filter", 0)
            obj = AvgPool(input_dtype, window, [strides_h, strides_w], padding, kernel_name)
            obj.avg_pool_operator()
        if bias is not None:
            error_manager_cube.raise_err_input_params_not_expected("dynamic_avg_pool", "bias", "None",
                                                                   "dict")
