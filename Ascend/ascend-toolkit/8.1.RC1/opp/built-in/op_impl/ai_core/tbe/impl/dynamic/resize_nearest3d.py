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
resize_nearest3d.py
"""

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tik


MAX_INT32 = 2**31 - 1
TILING_NUM = 64


class ResizeNearest3D(object):
    # 'pylint: disable=too-many-arguments,unused-argument
    def __init__(self, image, output, size, scale_factor, align_corners, half_pixel_centers, nearest_mode, kernel_name):
        self.tik_instance = tik.Tik()
        self.dtype = image.get("dtype")
        self.size = size
        self.scale = scale_factor
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers
        self.nearest_mode = nearest_mode
        self.kernel_name = kernel_name

        if self.dtype != "float32":
            raise RuntimeError("ResizeNearest3D only support float32")

        self.c0 = 16
        self.block_byte_size = 32
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_byte_size = self.get_dtype_size(self.dtype)
        self.data_each_block = self.block_byte_size // self.dtype_byte_size
        self.c_burst = self.c0 // self.data_each_block
        self.max_mask = 64

        self.left_ub = self.ub_size - (5 * 1024 + 3 * 64) * 4
        self.left_number = self.left_ub // self.dtype_byte_size
        self.left_w = self.left_number // self.c0
        self.gm_shape = [MAX_INT32] * 6

        self.tiling_gm = self.tik_instance.Tensor("int64", (TILING_NUM,), scope=tik.scope_gm, name="tiling_gm")
        self.image_gm = self.tik_instance.Tensor(self.dtype, self.gm_shape, scope=tik.scope_gm, name="image_gm")
        self.output_gm = self.tik_instance.Tensor(self.dtype, self.gm_shape, scope=tik.scope_gm, name="output_gm")
        if self.size is not None:
            self.size_dtype = size.get("dtype")
            self.size_gm = self.tik_instance.Tensor(self.size_dtype, [MAX_INT32], scope=tik.scope_gm, name="size_gm")
        if self.scale is not None:
            self.scale_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="scale_gm")

        self.scale_d, self.scale_h, self.scale_w = None, None, None

        self.tiling_mode = None
        self.batch_n, self.batch_c1 = None, None
        self.input_d, self.input_h, self.input_w = None, None, None
        self.output_d, self.output_h, self.output_w = None, None, None
        self.avg_input, self.loop_input, self.tail_input = None, None, None
        self.nd = None
        self.avg_nd = None
        self.last_nd = None
        self.block_num = None
        self.loop_h = None
        self.tail_h = None
        self.loop_w = None
        self.tail_w = None
        self.move_c1 = None
        self.loop_c1 = None
        self.tail_c1 = None

    def get_tiling_params(self):
        """
        get runtime params from tiling
        :return: None
        """
        self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_mode")
        self.batch_n = self.tik_instance.Scalar("int64", name="batch_n")
        self.batch_c1 = self.tik_instance.Scalar("int64", name="batch_c1")
        self.input_d = self.tik_instance.Scalar("int64", name="input_d")
        self.input_h = self.tik_instance.Scalar("int64", name="input_h")
        self.input_w = self.tik_instance.Scalar("int64", name="input_w")
        self.output_d = self.tik_instance.Scalar("int64", name="output_d")
        self.output_h = self.tik_instance.Scalar("int64", name="output_h")
        self.output_w = self.tik_instance.Scalar("int64", name="output_w")
        self.avg_input = self.tik_instance.Scalar("int64", name="avg_input")
        self.loop_input = self.tik_instance.Scalar("int64", name="loop_input")
        self.tail_input = self.tik_instance.Scalar("int64", name="tail_input")
        self.nd = self.tik_instance.Scalar("int64", name="nd")
        self.avg_nd = self.tik_instance.Scalar("int64", name="avg_nd")
        self.last_nd = self.tik_instance.Scalar("int64", name="last_nd")
        self.block_num = self.tik_instance.Scalar("int64", name="block_num")
        self.loop_h = self.tik_instance.Scalar("int64", name="loop_h")
        self.tail_h = self.tik_instance.Scalar("int64", name="tail_h")
        self.loop_w = self.tik_instance.Scalar("int64", name="loop_w")
        self.tail_w = self.tik_instance.Scalar("int64", name="tail_w")
        self.move_c1 = self.tik_instance.Scalar("int64", name="move_c1")
        self.loop_c1 = self.tik_instance.Scalar("int64", name="loop_c1")
        self.tail_c1 = self.tik_instance.Scalar("int64", name="tail_c1")

        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int64", shape=(TILING_NUM,), scope=tik.scope_ubuf, name="tiling_ub")
            self.data_move(tiling_ub, self.tiling_gm, num=TILING_NUM)

            self.tiling_mode.set_as(tiling_ub[0])
            self.batch_n.set_as(tiling_ub[1])
            self.batch_c1.set_as(tiling_ub[2])
            self.input_d.set_as(tiling_ub[3])
            self.input_h.set_as(tiling_ub[4])
            self.input_w.set_as(tiling_ub[5])
            self.output_d.set_as(tiling_ub[6])
            self.output_h.set_as(tiling_ub[7])
            self.output_w.set_as(tiling_ub[8])
            self.avg_input.set_as(tiling_ub[9])
            self.loop_input.set_as(tiling_ub[10])
            self.tail_input.set_as(tiling_ub[11])
            self.nd.set_as(tiling_ub[12])
            self.avg_nd.set_as(tiling_ub[13])
            self.last_nd.set_as(tiling_ub[14])
            self.block_num.set_as(tiling_ub[15])
            self.loop_h.set_as(tiling_ub[16])
            self.tail_h.set_as(tiling_ub[17])
            self.loop_w.set_as(tiling_ub[18])
            self.tail_w.set_as(tiling_ub[19])
            self.move_c1.set_as(tiling_ub[20])
            self.loop_c1.set_as(tiling_ub[21])
            self.tail_c1.set_as(tiling_ub[22])

        self.get_scale()

    def get_scale(self):
        """
        get scale
        :return:
        """
        self.scale_d = self.tik_instance.Scalar("float32", init_value=0)
        self.scale_h = self.tik_instance.Scalar("float32", init_value=0)
        self.scale_w = self.tik_instance.Scalar("float32", init_value=0)

        input_d_fp = self.tik_instance.Scalar("float32")
        input_h_fp = self.tik_instance.Scalar("float32")
        input_w_fp = self.tik_instance.Scalar("float32")
        output_d_fp = self.tik_instance.Scalar("float32")
        output_h_fp = self.tik_instance.Scalar("float32")
        output_w_fp = self.tik_instance.Scalar("float32")

        input_d_fp.set_as(self.input_d)
        input_h_fp.set_as(self.input_h)
        input_w_fp.set_as(self.input_w)
        output_d_fp.set_as(self.output_d)
        output_h_fp.set_as(self.output_h)
        output_w_fp.set_as(self.output_w)

        if not self.align_corners:
            self.scale_d.set_as(input_d_fp / output_d_fp)
            self.scale_h.set_as(input_h_fp / output_h_fp)
            self.scale_w.set_as(input_w_fp / output_w_fp)
        else:
            with self.tik_instance.if_scope(output_d_fp > 1):
                self.scale_d.set_as((input_d_fp - 1) / (output_d_fp - 1))

            with self.tik_instance.if_scope(output_h_fp > 1):
                self.scale_h.set_as((input_h_fp - 1) / (output_h_fp - 1))

            with self.tik_instance.if_scope(output_w_fp > 1):
                self.scale_w.set_as((input_w_fp - 1) / (output_w_fp - 1))

    def tiling_compute(self, block_idx, nd_num):
        """
        tiling compute func
        :param block_idx:
        :param nd_num:
        :return:
        """
        with self.tik_instance.if_scope(tik.any(self.tiling_mode == 0, self.tiling_mode == 1, self.tiling_mode == 2,
                                                self.tiling_mode == 3)):
            src_ub = self.tik_instance.Tensor(self.dtype, [1024, self.c0], scope=tik.scope_ubuf,
                                              name="src_ub")
            dst_ub = self.tik_instance.Tensor(self.dtype, [1024, self.c0], scope=tik.scope_ubuf,
                                              name="dst_ub")
            self.compute_per_core(block_idx, nd_num, src_ub, dst_ub)
        with self.tik_instance.if_scope(self.tiling_mode == 4):
            src_ub = self.tik_instance.Tensor(self.dtype, [self.input_w, self.c0], scope=tik.scope_ubuf,
                                              name="src_ub")
            dst_ub = self.tik_instance.Tensor(self.dtype, [self.output_w, self.c0], scope=tik.scope_ubuf,
                                              name="dst_ub")
            self.compute_per_core(block_idx, nd_num, src_ub, dst_ub)
        with self.tik_instance.if_scope(self.tiling_mode == 5):
            src_ub = self.tik_instance.Tensor(self.dtype, [self.avg_input, self.c0], scope=tik.scope_ubuf,
                                              name="src_ub")
            dst_ub = self.tik_instance.Tensor(self.dtype, [self.output_w, self.c0], scope=tik.scope_ubuf,
                                              name="dst_ub")
            self.compute_per_core(block_idx, nd_num, src_ub, dst_ub)
        with self.tik_instance.if_scope(self.tiling_mode == 6):
            src_ub = self.tik_instance.Tensor(self.dtype, [self.input_w, self.c0], scope=tik.scope_ubuf,
                                              name="src_ub")
            dst_ub = self.tik_instance.Tensor(self.dtype, [1024, self.c0], scope=tik.scope_ubuf,
                                              name="dst_ub")
            self.compute_per_core(block_idx, nd_num, src_ub, dst_ub)
        with self.tik_instance.if_scope(self.tiling_mode == 7):
            src_ub = self.tik_instance.Tensor(self.dtype, [self.avg_input, self.c0], scope=tik.scope_ubuf,
                                              name="src_ub")
            dst_ub = self.tik_instance.Tensor(self.dtype, [1024, self.c0], scope=tik.scope_ubuf,
                                              name="dst_ub")
            self.compute_per_core(block_idx, nd_num, src_ub, dst_ub)
        with self.tik_instance.if_scope(self.tiling_mode == 8):
            self.memcpy_nearest(block_idx, nd_num)

    def compute(self):
        """
        op compute func
        :return:
        """
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_idx:
            self.get_tiling_params()

            nd_num = self.tik_instance.Scalar("int32")
            with self.tik_instance.if_scope(block_idx < self.block_num):
                with self.tik_instance.if_scope(block_idx < self.block_num - 1):
                    nd_num.set_as(self.avg_nd)
                with self.tik_instance.if_scope(block_idx == self.block_num - 1):
                    nd_num.set_as(self.last_nd)

                self.tiling_compute(block_idx, nd_num)

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num, "left_w": self.left_w})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        if self.size is not None:
            self.tik_instance.BuildCCE(inputs=[self.image_gm, self.size_gm], outputs=[self.output_gm],
                                       flowtable=[self.tiling_gm], kernel_name=self.kernel_name, config=opt_config)
        else:
            self.tik_instance.BuildCCE(inputs=[self.image_gm, self.scale_gm], outputs=[self.output_gm],
                                       flowtable=[self.tiling_gm], kernel_name=self.kernel_name, config=opt_config)

        return self.tik_instance

    def compute_per_core(self, block_idx, nd_num, src_ub, dst_ub):
        """
        compute per core
        :param block_idx:
        :param nd_num:
        :param src_ub:
        :param dst_ub:
        :return:
        """
        offset_idx = self.tik_instance.Scalar("int32")
        nd_idx = self.tik_instance.Scalar("int32")
        dst_n = self.tik_instance.Scalar("int32")
        dst_d = self.tik_instance.Scalar("int32")
        offset_idx.set_as(block_idx * self.avg_nd)

        idx_1024 = self.tik_instance.Tensor("float32", [1024], scope=tik.scope_ubuf, name="idx_1024")
        h_idx_fp = self.tik_instance.Tensor("float32", [1024], scope=tik.scope_ubuf, name="h_idx_fp")
        w_idx_fp = self.tik_instance.Tensor("float32", [1024], scope=tik.scope_ubuf, name="w_idx_fp")
        h_idx = self.tik_instance.Tensor("int32", [1024], scope=tik.scope_ubuf, name="h_idx")
        w_idx = self.tik_instance.Tensor("int32", [1024], scope=tik.scope_ubuf, name="w_idx")
        input_h_ub = self.tik_instance.Tensor("int32", [64], scope=tik.scope_ubuf, name="input_h_ub")
        input_w_ub = self.tik_instance.Tensor("int32", [64], scope=tik.scope_ubuf, name="input_w_ub")
        zero_ub = self.tik_instance.Tensor("int32", [64], scope=tik.scope_ubuf, name="zero_ub")
        self.dup_value(input_h_ub, num=64, dup_value=self.input_h - 1)
        self.dup_value(input_w_ub, num=64, dup_value=self.input_w - 1)
        self.dup_value(zero_ub, num=64, dup_value=0)
        self.gen_1024(idx_1024)

        with self.tik_instance.if_scope(tik.all(self.loop_h == 1, self.loop_w == 1)):
            self.loop_shsw(nd_num, nd_idx, offset_idx, dst_n, dst_d, input_h_ub, input_w_ub, idx_1024, h_idx_fp, h_idx,
                           w_idx_fp, w_idx, src_ub, dst_ub, zero_ub)

        with self.tik_instance.elif_scope(tik.all(self.loop_h > 1, self.loop_w == 1)):
            self.loop_bhsw(nd_num, nd_idx, offset_idx, dst_n, dst_d, input_h_ub, input_w_ub, idx_1024, h_idx_fp, h_idx,
                           w_idx_fp, w_idx, src_ub, dst_ub, zero_ub)

        with self.tik_instance.elif_scope(tik.all(self.loop_h == 1, self.loop_w > 1)):
            self.loop_shbw(nd_num, nd_idx, offset_idx, dst_n, dst_d, input_h_ub, input_w_ub, idx_1024, h_idx_fp, h_idx,
                           w_idx_fp, w_idx, src_ub, dst_ub, zero_ub)

        with self.tik_instance.else_scope():
            self.loop_bhbw(nd_num, nd_idx, offset_idx, dst_n, dst_d, input_h_ub, input_w_ub, idx_1024, h_idx_fp, h_idx,
                           w_idx_fp, w_idx, src_ub, dst_ub, zero_ub)

    def loop_shsw(self, nd_num, nd_idx, offset_idx, dst_n, dst_d, input_h_ub, input_w_ub, idx_1024, h_idx_fp, h_idx,
                  w_idx_fp, w_idx, src_ub, dst_ub, zero_ub):
        """
        loop h <= 1024 and w <= 1024
        """
        self.calc_idx(idx_1024, h_idx_fp, h_idx, input_h_ub, self.scale_h, zero_ub)
        self.calc_idx(idx_1024, w_idx_fp, w_idx, input_w_ub, self.scale_w, zero_ub)

        src_d = self.tik_instance.Scalar("int32")
        src_h = self.tik_instance.Scalar("int32")
        src_w = self.tik_instance.Scalar("int32")
        num_h = self.tik_instance.Scalar("int32", init_value=self.output_h)
        num_w = self.tik_instance.Scalar("int32", init_value=self.output_w)

        self.interpolate(nd_num, nd_idx, offset_idx, dst_n, dst_d, src_d, num_h, src_h, num_w, src_w,
                         h_idx, w_idx, src_ub, dst_ub, 0, 0)

    def loop_bhsw(self, nd_num, nd_idx, offset_idx, dst_n, dst_d, input_h_ub, input_w_ub, idx_1024, h_idx_fp, h_idx,
                  w_idx_fp, w_idx, src_ub, dst_ub, zero_ub):
        """
        loop h > 1024 and w <= 1024
        """
        self.calc_idx(idx_1024, w_idx_fp, w_idx, input_w_ub, self.scale_w, zero_ub)
        src_d = self.tik_instance.Scalar("int32")
        src_h = self.tik_instance.Scalar("int32")
        src_w = self.tik_instance.Scalar("int32")
        num_h = self.tik_instance.Scalar("int32", init_value=1024)
        num_w = self.tik_instance.Scalar("int32", init_value=self.output_w)

        with self.tik_instance.for_range(0, self.loop_h) as h_loop_idx:
            self.calc_idx(idx_1024, h_idx_fp, h_idx, input_h_ub, self.scale_h, zero_ub, h_loop_idx)
            with self.tik_instance.if_scope(h_loop_idx == self.loop_h - 1):
                num_h.set_as(self.tail_h)

            self.interpolate(nd_num, nd_idx, offset_idx, dst_n, dst_d, src_d, num_h, src_h, num_w, src_w,
                             h_idx, w_idx, src_ub, dst_ub, h_loop_idx, 0)

    def loop_shbw(self, nd_num, nd_idx, offset_idx, dst_n, dst_d, input_h_ub, input_w_ub, idx_1024, h_idx_fp, h_idx,
                  w_idx_fp, w_idx, src_ub, dst_ub, zero_ub):
        """
        loop h <= 1024 and w > 1024
        """
        self.calc_idx(idx_1024, h_idx_fp, h_idx, input_h_ub, self.scale_h, zero_ub)
        src_d = self.tik_instance.Scalar("int32")
        src_h = self.tik_instance.Scalar("int32")
        src_w = self.tik_instance.Scalar("int32")
        num_h = self.tik_instance.Scalar("int32", init_value=self.output_h)
        num_w = self.tik_instance.Scalar("int32", init_value=1024)

        with self.tik_instance.for_range(0, self.loop_w) as w_loop_idx:
            self.calc_idx(idx_1024, w_idx_fp, w_idx, input_w_ub, self.scale_w, zero_ub, w_loop_idx)
            with self.tik_instance.if_scope(w_loop_idx == self.loop_w - 1):
                num_w.set_as(self.tail_w)

            self.interpolate(nd_num, nd_idx, offset_idx, dst_n, dst_d, src_d, num_h, src_h, num_w, src_w,
                             h_idx, w_idx, src_ub, dst_ub, 0, w_loop_idx)

    def loop_bhbw(self, nd_num, nd_idx, offset_idx, dst_n, dst_d, input_h_ub, input_w_ub, idx_1024, h_idx_fp, h_idx,
                  w_idx_fp, w_idx, src_ub, dst_ub, zero_ub):
        """
        loop h > 1024 and w > 1024
        """
        src_d = self.tik_instance.Scalar("int32")
        src_h = self.tik_instance.Scalar("int32")
        src_w = self.tik_instance.Scalar("int32")
        num_h = self.tik_instance.Scalar("int32", init_value=1024)
        num_w = self.tik_instance.Scalar("int32", init_value=1024)

        with self.tik_instance.for_range(0, self.loop_h) as h_loop_idx:
            self.calc_idx(idx_1024, h_idx_fp, h_idx, input_h_ub, self.scale_h, zero_ub, h_loop_idx)
            with self.tik_instance.if_scope(h_loop_idx == self.loop_h - 1):
                num_h.set_as(self.tail_h)

            with self.tik_instance.for_range(0, self.loop_w) as w_loop_idx:
                self.calc_idx(idx_1024, w_idx_fp, w_idx, input_w_ub, self.scale_w, zero_ub, w_loop_idx)
                with self.tik_instance.if_scope(w_loop_idx == self.loop_w - 1):
                    num_w.set_as(self.tail_w)

                self.interpolate(nd_num, nd_idx, offset_idx, dst_n, dst_d, src_d, num_h, src_h, num_w, src_w,
                                 h_idx, w_idx, src_ub, dst_ub, h_loop_idx, w_loop_idx)

    def interpolate(self, nd_num, nd_idx, offset_idx, dst_n, dst_d, src_d, num_h, src_h, num_w, src_w,
                    h_idx, w_idx, src_ub, dst_ub, h_loop_idx, w_loop_idx):
        """
        interpolate
        """

        with self.tik_instance.for_range(0, nd_num) as idx:
            nd_idx.set_as(offset_idx + idx)
            dst_n.set_as(nd_idx // self.output_d)
            dst_d.set_as(nd_idx % self.output_d)
            self.compute_src_index(self.scale_d, dst_d, src_d, self.input_d)

            with self.tik_instance.for_range(0, num_h) as dst_h:
                src_h.set_as(h_idx[dst_h])
                with self.tik_instance.if_scope(tik.any(self.tiling_mode == 0, self.tiling_mode == 1,
                                                        self.tiling_mode == 2)):
                    self.compute_012(w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                                     h_loop_idx, w_loop_idx)

                with self.tik_instance.if_scope(self.tiling_mode == 3):
                    self.compute_3(w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                                   h_loop_idx, w_loop_idx)

                with self.tik_instance.if_scope(tik.any(self.tiling_mode == 4, self.tiling_mode == 6)):
                    self.compute_46(w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                                    h_loop_idx, w_loop_idx)
                with self.tik_instance.if_scope(tik.any(self.tiling_mode == 5, self.tiling_mode == 7)):
                    self.compute_57(w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                                    h_loop_idx, w_loop_idx)

    def compute_012(self, w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                    h_loop_idx, w_loop_idx):
        """
        compute func when tiling_mode in [0, 1, 2]
        """
        with self.tik_instance.if_scope(self.tiling_mode == 0):
            self.compute_0(w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                           h_loop_idx, w_loop_idx)
        with self.tik_instance.if_scope(self.tiling_mode == 1):
            self.compute_1(w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                           h_loop_idx, w_loop_idx)
        with self.tik_instance.if_scope(self.tiling_mode == 2):
            self.compute_2(w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                           h_loop_idx, w_loop_idx)

    def compute_0(self, w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                  h_loop_idx, w_loop_idx):
        """
        compute func when tiling_mode is 0
        """
        image_offset = self.cal_offset(dst_n, src_d, 0, src_h, 0, True)
        self.data_move_batch(src_ub, self.image_gm, [0, image_offset], self.batch_c1, self.input_h, self.input_w)
        with self.tik_instance.for_range(0, num_w) as dst_w:
            src_w.set_as(w_idx[dst_w])
            self.data_move_ub(dst_ub[dst_w * self.c0], src_ub[src_w * self.c0], self.batch_c1)

        output_offset = self.cal_offset(dst_n, dst_d, 0, dst_h + h_loop_idx * 1024, w_loop_idx * 1024)
        self.data_move_batch(self.output_gm, dst_ub, [output_offset, 0], self.batch_c1, self.output_h, self.output_w,
                             "out")

    def compute_1(self, w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                  h_loop_idx, w_loop_idx):
        """
        compute func when tiling_mode is 1
        """
        with self.tik_instance.for_range(0, self.loop_c1) as loop:
            with self.tik_instance.if_scope(loop != self.loop_c1 - 1):
                image_offset = self.cal_offset(dst_n, src_d, loop * self.move_c1, src_h, 0, True)
                self.data_move_batch(src_ub, self.image_gm, [0, image_offset], self.move_c1, self.input_h,
                                     self.input_w)

                with self.tik_instance.for_range(0, num_w) as dst_w:
                    src_w.set_as(w_idx[dst_w])
                    dst_offset = loop * self.move_c1 * self.output_w * self.c0 + dst_w * self.c0
                    self.data_move_ub(dst_ub[dst_offset], src_ub[src_w * self.c0], self.move_c1)

            with self.tik_instance.else_scope():
                image_offset = self.cal_offset(dst_n, src_d, loop * self.move_c1, src_h, 0, True)
                self.data_move_batch(src_ub, self.image_gm, [0, image_offset], self.tail_c1, self.input_h,
                                     self.input_w)

                with self.tik_instance.for_range(0, num_w) as dst_w:
                    src_w.set_as(w_idx[dst_w])
                    dst_offset = loop * self.move_c1 * self.output_w * self.c0 + dst_w * self.c0
                    self.data_move_ub(dst_ub[dst_offset], src_ub[src_w * self.c0], self.tail_c1)

        output_offset = self.cal_offset(dst_n, dst_d, 0, dst_h + h_loop_idx * 1024, w_loop_idx * 1024)
        self.data_move_batch(self.output_gm, dst_ub, [output_offset, 0], self.batch_c1, self.output_h, self.output_w,
                             "out")

    def compute_2(self, w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                  h_loop_idx, w_loop_idx):
        """
        compute func when tiling_mode is 2
        """
        with self.tik_instance.for_range(0, self.batch_c1) as c1:
            image_offset = self.cal_offset(dst_n, src_d, c1, src_h, 0, True)
            self.data_move(src_ub, self.image_gm[image_offset], num=self.avg_input * self.c0)

            loop = self.tik_instance.Scalar("int32", init_value=0)
            with self.tik_instance.for_range(0, num_w) as dst_w:
                src_w.set_as(w_idx[dst_w])

                self.data_move_src(src_ub, src_w, loop, dst_n, src_d, c1, src_h)
                dst_offset = c1 * self.output_w * self.c0 + dst_w * self.c0
                self.data_move(dst_ub[dst_offset], src_ub[src_w % self.avg_input, 0], num=16)

        output_offset = self.cal_offset(dst_n, dst_d, 0, dst_h + h_loop_idx * 1024, w_loop_idx * 1024)
        self.data_move_batch(self.output_gm, dst_ub, [output_offset, 0], self.batch_c1, self.output_h, self.output_w,
                             "out")

    def compute_3(self, w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                  h_loop_idx, w_loop_idx):
        """
        compute func when tiling_mode is 3
        """
        image_offset = self.cal_offset(dst_n, src_d, 0, src_h, 0, True)
        self.data_move_batch(src_ub, self.image_gm, [0, image_offset], self.batch_c1, self.input_h, self.input_w)

        with self.tik_instance.for_range(0, self.loop_c1) as loop:
            with self.tik_instance.if_scope(loop != self.loop_c1 - 1):
                with self.tik_instance.for_range(0, num_w) as dst_w:
                    src_w.set_as(w_idx[dst_w])
                    src_offset = loop * self.move_c1 * self.input_w * self.c0 + src_w * self.c0
                    self.data_move_ub(dst_ub[dst_w * self.c0], src_ub[src_offset], self.move_c1)

                output_offset = self.cal_offset(dst_n, dst_d, loop * self.move_c1, dst_h + h_loop_idx * 1024,
                                                w_loop_idx * 1024)
                self.data_move_batch(self.output_gm, dst_ub, [output_offset, 0], self.move_c1, self.output_h,
                                     self.output_w, "out")
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, num_w) as dst_w:
                    src_w.set_as(w_idx[dst_w])
                    src_offset = loop * self.move_c1 * self.input_w * self.c0 + src_w * self.c0
                    self.data_move_ub(dst_ub[dst_w * self.c0], src_ub[src_offset], self.tail_c1)

                output_offset = self.cal_offset(dst_n, dst_d, loop * self.move_c1, dst_h + h_loop_idx * 1024,
                                                w_loop_idx * 1024)
                self.data_move_batch(self.output_gm, dst_ub, [output_offset, 0], self.tail_c1, self.output_h,
                                     self.output_w, "out")

    def compute_46(self, w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                   h_loop_idx, w_loop_idx):
        """
        compute func when tiling_mode is 4 or 6
        """
        with self.tik_instance.for_range(0, self.batch_c1) as c1:
            image_offset = self.cal_offset(dst_n, src_d, c1, src_h, 0, True)
            self.data_move(src_ub, self.image_gm[image_offset], num=self.input_w * self.c0)
            with self.tik_instance.if_scope(num_w > 1):
                with self.tik_instance.for_range(0, num_w, thread_num=2) as dst_w:
                    src_w.set_as(w_idx[dst_w])
                    self.data_move(dst_ub[dst_w, 0], src_ub[src_w, 0], num=16)

            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, num_w) as dst_w:
                    src_w.set_as(w_idx[dst_w])
                    self.data_move(dst_ub[dst_w, 0], src_ub[src_w, 0], num=16)

            output_offset = self.cal_offset(dst_n, dst_d, c1, dst_h + h_loop_idx * 1024, w_loop_idx * 1024)
            self.data_move(self.output_gm[output_offset], dst_ub, num=num_w * self.c0)

    def compute_57(self, w_idx, src_ub, dst_ub, dst_n, dst_d, dst_h, src_d, src_h, src_w, num_w,
                   h_loop_idx, w_loop_idx):
        """
        compute func when tiling_mode is  5 or 7
        """
        with self.tik_instance.for_range(0, self.batch_c1) as c1:
            image_offset = self.cal_offset(dst_n, src_d, c1, src_h, 0, True)
            self.data_move(src_ub, self.image_gm[image_offset], num=self.avg_input * self.c0)
            loop = self.tik_instance.Scalar("int32", init_value=0)
            with self.tik_instance.for_range(0, num_w) as dst_w:
                src_w.set_as(w_idx[dst_w])

                self.data_move_src(src_ub, src_w, loop, dst_n, src_d, c1, src_h)
                self.data_move(dst_ub[dst_w, 0], src_ub[src_w % self.avg_input, 0], num=16)

            output_offset = self.cal_offset(dst_n, dst_d, c1, dst_h + h_loop_idx * 1024, w_loop_idx * 1024)
            self.data_move(self.output_gm[output_offset], dst_ub, num=num_w * self.c0)

    def memcpy_nearest(self, block_idx, nd_num):
        """
        only move in and out
        :param block_idx:
        :param nd_num:
        :return:
        """
        offset_idx = self.tik_instance.Scalar("int32")
        dst_nd = self.tik_instance.Scalar("int32")
        loop = self.tik_instance.Scalar("int32")
        tail_n = self.tik_instance.Scalar("int32")

        offset_idx.set_as(block_idx * self.avg_nd)
        move_n = 2048

        move_num = self.batch_c1 * self.output_h * self.output_w
        loop_num = move_n * 16
        loop.set_as((move_num + move_n - 1) // move_n)

        tail_n.set_as(move_num % move_n)
        with self.tik_instance.if_scope(tail_n == 0):
            tail_n.set_as(move_n)
        tail_num = tail_n * 16

        dst_ub = self.tik_instance.Tensor(self.dtype, [move_n, self.c0], scope=tik.scope_ubuf, name="dst_ub")

        with self.tik_instance.for_range(0, nd_num) as idx:
            dst_nd.set_as(offset_idx + idx)
            with self.tik_instance.if_scope(loop > 1):
                with self.tik_instance.for_range(0, loop) as loop_idx:
                    with self.tik_instance.if_scope(loop_idx != loop - 1):
                        self.data_move(dst_ub, self.image_gm[dst_nd * move_num + loop_idx * move_n * 16], num=loop_num)
                        self.data_move(self.output_gm[dst_nd * move_num + loop_idx * move_n * 16], dst_ub, num=loop_num)
                    with self.tik_instance.else_scope():
                        self.data_move(dst_ub, self.image_gm[dst_nd * move_num + loop_idx * move_n * 16], num=tail_num)
                        self.data_move(self.output_gm[dst_nd * move_num + loop_idx * move_n * 16], dst_ub, num=tail_num)

            with self.tik_instance.else_scope():
                self.data_move(dst_ub, self.image_gm[dst_nd * move_num], num=move_num)
                self.data_move(self.output_gm[dst_nd * move_num], dst_ub, num=move_num)

    def cal_offset(self, n, d, c1, h, w, src=False):
        """
        calculate buffer offset
        """
        if src:
            one_d = self.input_d * self.batch_c1 * self.input_h * self.input_w * self.c0
            one_c1 = self.batch_c1 * self.input_h * self.input_w * self.c0
            one_h = self.input_h * self.input_w * self.c0
            one_w = self.input_w * self.c0
            return n * one_d + d * one_c1 + c1 * one_h + h * one_w + w * self.c0
        else:
            one_d = self.output_d * self.batch_c1 * self.output_h * self.output_w * self.c0
            one_c1 = self.batch_c1 * self.output_h * self.output_w * self.c0
            one_h = self.output_h * self.output_w * self.c0
            one_w = self.output_w * self.c0
            return n * one_d + d * one_c1 + c1 * one_h + h * one_w + w * self.c0

    def gen_1024(self, idx_ub):
        """
        generate 1024 indice
        """
        tmp_scalar = self.tik_instance.Scalar("float32")
        with self.tik_instance.for_range(0, 1024) as n:
            tmp_scalar.set_as(n)
            idx_ub[n].set_as(tmp_scalar)

    def calc_by_mode(self, idx_ub_fp, idx_ub):
        """
        calculate index by nearest mode
        """
        if self.nearest_mode == "round_prefer_floor":
            fp_scalar = self.tik_instance.Scalar("float32")
            int_scalar = self.tik_instance.Scalar("int32")
            fp_scalar2 = self.tik_instance.Scalar("float32", init_value=0.5)
            with self.tik_instance.for_range(0, 1024) as idx:
                fp_scalar.set_as(idx_ub_fp[idx])
                int_scalar.set_as(fp_scalar)
                with self.tik_instance.if_scope(fp_scalar == int_scalar + fp_scalar2):
                    idx_ub[idx].set_as(int_scalar)
                with self.tik_instance.else_scope():
                    self.tik_instance.scalar_conv("round", int_scalar, fp_scalar)
                    idx_ub[idx].set_as(int_scalar)
        elif self.nearest_mode == "round_prefer_ceil":
            self.data_conv(idx_ub, idx_ub_fp, [0, 0], mode="round", num=1024)
        else:
            self.data_conv(idx_ub, idx_ub_fp, [0, 0], mode=self.nearest_mode, num=1024)

    def calc_idx(self, ub_1024, idx_ub_fp, idx_ub, src_dim, scale, zero_ub, loop=0):
        """
        calculate index
        """
        with self.tik_instance.if_scope(loop == 0):
            if not self.half_pixel_centers:
                self.data_muls(idx_ub_fp, ub_1024, scale, [0, 0], 1024)
            else:
                self.data_adds(idx_ub_fp, ub_1024, 0.5, [0, 0], 1024)
                self.data_muls(idx_ub_fp, idx_ub_fp, scale, [0, 0], 1024)
                self.data_adds(idx_ub_fp, idx_ub_fp, -0.5, [0, 0], 1024)

            self.calc_by_mode(idx_ub_fp, idx_ub)
            self.tik_instance.vec_min(64, idx_ub, idx_ub, src_dim, 16, 8, 8, 0)

        with self.tik_instance.else_scope():
            self.data_adds(idx_ub_fp, ub_1024, loop * 1024, [0, 0], num=1024)
            if not self.half_pixel_centers:
                self.data_muls(idx_ub_fp, idx_ub_fp, scale, [0, 0], 1024)
            else:
                self.data_adds(idx_ub_fp, idx_ub_fp, 0.5, [0, 0], 1024)
                self.data_muls(idx_ub_fp, idx_ub_fp, scale, [0, 0], 1024)
                self.data_adds(idx_ub_fp, idx_ub_fp, -0.5, [0, 0], 1024)

            self.calc_by_mode(idx_ub_fp, idx_ub)
            self.tik_instance.vec_min(64, idx_ub, idx_ub, src_dim, 16, 8, 8, 0)

        if self.half_pixel_centers:
            self.tik_instance.vec_max(64, idx_ub, idx_ub, zero_ub, 16, 8, 8, 0)

    def compute_src_index(self, scale, dst_idx, src_idx, src_dim):
        """
        compute src index
        """
        src_idx_fp = self.tik_instance.Scalar("float32")
        dst_idx_fp = self.tik_instance.Scalar("float32")

        dst_idx_fp.set_as(dst_idx)
        if not self.half_pixel_centers:
            src_idx_fp.set_as(scale * dst_idx_fp)
        else:
            src_idx_fp.set_as((dst_idx_fp + 0.5) * scale - 0.5)

        if self.nearest_mode == "round_prefer_floor":
            fp_scalar2 = self.tik_instance.Scalar("float32", init_value=0.5)
            int_scalar = self.tik_instance.Scalar("int32")
            int_scalar.set_as(src_idx_fp)
            with self.tik_instance.if_scope(src_idx_fp == int_scalar + fp_scalar2):
                self.tik_instance.scalar_conv("floor", src_idx, src_idx_fp)
            with self.tik_instance.else_scope():
                self.tik_instance.scalar_conv("round", src_idx, src_idx_fp)

        elif self.nearest_mode == "round_prefer_ceil":
            self.tik_instance.scalar_conv("round", src_idx, src_idx_fp)
        else:
            self.tik_instance.scalar_conv(self.nearest_mode, src_idx, src_idx_fp)
        with self.tik_instance.if_scope(src_idx > src_dim - 1):
            src_idx.set_as(src_dim - 1)

        if self.half_pixel_centers:
            with self.tik_instance.if_scope(src_idx < 0):
                src_idx.set_as(0)

    def get_dtype_size(self, dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2, "int64": 8}
        return dtype_dict.get(dtype)

    def data_move(self, dst, src, num, src_stride=0, dst_stride=0):
        """
        move data
        """
        sid = 0
        nburst = 1
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        burst_len = (num + data_each_block - 1) // data_each_block
        self.tik_instance.data_move(dst, src, sid, nburst, burst_len, src_stride=src_stride, dst_stride=dst_stride)

    def data_move_batch(self, dst, src, offsets, nburst, h, w, mode="in"):
        """
        batch data move
        """
        dst_offset, src_offset = offsets

        if mode == "in":
            src_stride = (h * w - w) * self.c_burst
            with self.tik_instance.if_scope(src_stride < 65535):
                self.tik_instance.data_move(dst[dst_offset], src[src_offset], 0, nburst, w * self.c_burst,
                                            src_stride, 0)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, nburst) as c_idx:
                    self.tik_instance.data_move(dst[dst_offset + c_idx * w * self.c0],
                                                src[src_offset + c_idx * h * w * self.c0],
                                                0, 1, w * self.c_burst, 0, 0)
        else:
            dst_stride = (h * w - w) * self.c_burst
            with self.tik_instance.if_scope(dst_stride < 65535):
                self.tik_instance.data_move(dst[dst_offset], src[src_offset], 0, nburst, w * self.c_burst,
                                            0, dst_stride)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, nburst) as c_idx:
                    self.tik_instance.data_move(dst[dst_offset + c_idx * h * w * self.c0],
                                                src[src_offset + c_idx * w * self.c0],
                                                0, 1, w * self.c_burst, 0, 0)

    def data_move_ub(self, dst, src, nburst):
        """
        data move ub 2 ub
        """
        src_stride = (self.input_w - 1) * self.c_burst
        dst_stride = (self.output_w - 1) * self.c_burst

        self.tik_instance.data_move(dst, src, 0, nburst, self.c_burst, src_stride, dst_stride)

    def data_move_src(self, src_ub, src_w, loop, dst_n, src_d, c1, src_h):
        """
        move src data from gm to ub
        """
        with self.tik_instance.if_scope(src_w >= (loop + 1) * self.avg_input):
            loop.set_as(src_w // self.avg_input)
            gm_offset = self.cal_offset(dst_n, src_d, c1, src_h, loop * self.avg_input, True)
            with self.tik_instance.if_scope(loop != self.loop_input - 1):
                self.data_move(src_ub, self.image_gm[gm_offset], num=self.avg_input * self.c0)
            with self.tik_instance.else_scope():
                self.data_move(src_ub, self.image_gm[gm_offset], num=self.tail_input * self.c0)

    def dup_value(self, dst, num, dup_value=0, offset=0):
        """
        dup value to ub
        """
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        mask = 256 // dtype_byte_size
        stride = 8

        loop = num // (mask * 255)
        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
            offset += loop * mask * 255

        repeat_time = (num % (mask * 255)) // mask
        if repeat_time > 0:
            self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
            offset += repeat_time * mask

        last_num = num % mask
        if last_num > 0:
            self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, stride)

    def single_operator_template(self, op_obj, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        vector_mask_max = 64
        dst_offset, src_offset = offsets

        tensor_size = num if num else src.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src[tmp_src_offset], scalar, 255,
                       dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            op_obj(vector_mask_max, dst[dst_offset], src[src_offset], scalar, repeat_time, dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            op_obj(last_num, dst[dst_offset], src[src_offset], scalar, 1, dst_stride, src_stride)

    def data_adds(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik adds
        """
        self.single_operator_template(self.tik_instance.vec_adds, dst, src, scalar, offsets, num, dst_stride,
                                      src_stride)

    def data_muls(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik muls
        """
        self.single_operator_template(self.tik_instance.vec_muls, dst, src, scalar, offsets, num, dst_stride,
                                      src_stride)

    def data_conv(self, dst, src, offsets, mode="ceil", num=0, dst_stride=8, src_stride=8):
        """
        tik conv
        """
        vector_mask_max = 64
        dst_offset, src_offset = offsets

        tensor_size = num if num else src.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                self.tik_instance.vec_conv(vector_mask_max, mode, dst[tmp_dst_offset], src[tmp_src_offset], 255,
                                           dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            self.tik_instance.vec_conv(vector_mask_max, mode, dst[dst_offset], src[src_offset], repeat_time,
                                       dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)
