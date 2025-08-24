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
sync_resize_bilinear_v2_grad.py
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len


# 'pylint:disable=too-many-instance-attributes,too-many-public-methods
class SyncResizeBilinearV2Grad(object):
    """
    The class of SyncResizeBilinearV2Grad op
    """

    # 'pylint:disable=too-few-public-methods,too-many-instance-attributes
    class CommomScalar(object):
        """
        define some scalar
        """

        def __init__(self, tik_instance):
            self.dst_h = tik_instance.Scalar(dtype="float32", name="dst_h")
            self.dst_w = tik_instance.Scalar(dtype="float32", name="dst_w")
            self.src_h = tik_instance.Scalar(dtype="float32", name="src_h")
            self.h_idx = tik_instance.Scalar(dtype="int32", name="h_idx")
            self.h_stride = tik_instance.Scalar(dtype="int32", name="h_stride")
            self.hl_ratio = tik_instance.Scalar(dtype="float32", name="hl_ratio")
            self.hr_ratio = tik_instance.Scalar(dtype="float32", name="hr_ratio")

            self.src_w = tik_instance.Scalar(dtype="float32", name="src_w")
            self.w_idx = tik_instance.Scalar(dtype="int32", name="w_idx")
            self.w_stride = tik_instance.Scalar(dtype="int32", name="w_stride")
            self.wl_ratio = tik_instance.Scalar(dtype="float32", name="wl_ratio")
            self.wr_ratio = tik_instance.Scalar(dtype="float32", name="wr_ratio")
            self.l_ratio = tik_instance.Scalar(dtype="float32", name="l_ratio")
            self.r_ratio = tik_instance.Scalar(dtype="float32", name="r_ratio")

            self.output_idx00 = tik_instance.Scalar(dtype="int32", name="output_idx00", init_value=0)
            self.output_idx01 = tik_instance.Scalar(dtype="int32", name="output_idx01", init_value=0)
            self.output_idx10 = tik_instance.Scalar(dtype="int32", name="output_idx10")
            self.output_idx11 = tik_instance.Scalar(dtype="int32", name="output_idx11")
            self.grads_idx = tik_instance.Scalar(dtype="int32", name="grads_idx")
            self.next_loop = tik_instance.Scalar(dtype="int32", name="next_loop", init_value=0)

    MAX_INT32 = 2**31 - 1
    TILING_NUM = 64
    MASK = 64

    TILING_MODE0 = 0
    TILING_MODE1 = 1
    TILING_MODE2 = 2
    TILING_MODE3 = 3
    TILING_MODE4 = 4
    TILING_MODE5 = 5
    TILING_MODE6 = 6

    # 'pylint:disable=too-many-arguments,invalid-name
    def __init__(self, grads, images, align_corners=False,
                 half_pixel_centers=False, kernel_name="sync_resize_bilinear_v2_grad"):

        self.tik_instance = tik.Tik()

        self.grads_dtype = grads.get("dtype")
        self.images_dtype = images.get("dtype")

        self.align_corner = align_corners
        self.half_pixel_centers = half_pixel_centers
        self.kernel_name = kernel_name

        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.block_byte_size = 32
        self.grad_byte_size = self.get_dtype_size(self.grads_dtype)
        self.data_each_block = self.block_byte_size // self.grad_byte_size
        self.max_mask = 8 * self.data_each_block
        self.tensor_size = self.ub_size // self.grad_byte_size // 2
        self.c0 = 16
        self.tensor_c = self.tensor_size // self.c0

        self.grads_gm = self.tik_instance.Tensor(self.grads_dtype, (self.MAX_INT32,),
                                                 scope=tik.scope_gm,
                                                 name="grads_gm")
        self.images_gm = self.tik_instance.Tensor(self.images_dtype, (self.MAX_INT32,),
                                                  scope=tik.scope_gm,
                                                  name="images_gm")
        self.output_gm = self.tik_instance.Tensor(self.grads_dtype, (self.MAX_INT32,),
                                                  scope=tik.scope_gm,
                                                  name="output_gm",
                                                  is_atomic_add=True)
        self.tiling_gm = self.tik_instance.Tensor("int32", (self.TILING_NUM,), scope=tik.scope_gm, name="tiling_gm")
        self.l1_status = self.l1_support()
        self.int_index_l1, self.l_ratio_l1 = None, None
        self.r_ratio_l1, self.count_num_l1 = None, None
        self._init_tiling_params()

    @staticmethod
    def l1_support():
        """
        check if support l1 buffer or not
        :return:
        """
        soc_version = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
        if soc_version in [tbe_platform.ASCEND_910, tbe_platform.ASCEND_310P, tbe_platform.ASCEND_610,
                           tbe_platform.BS9SX1A]:
            return 1
        else:
            return 0

    def l1_function(self):
        """
        use to calculate ratio and count num
        :return:
        """
        self.int_index_l1 = self.tik_instance.Tensor("int32", (4096, 8), name="int_index_l1", scope=tik.scope_cbuf)
        self.l_ratio_l1 = self.tik_instance.Tensor("int32", (4096, 8), name="wl_ratio_l1", scope=tik.scope_cbuf)
        self.r_ratio_l1 = self.tik_instance.Tensor("int32", (4096, 8), name="wr_ratio_l1", scope=tik.scope_cbuf)
        self.count_num_l1 = self.tik_instance.Tensor("int32", (640,), name="count_num_l1", scope=tik.scope_cbuf)
        with self.tik_instance.new_stmt_scope():
            const_one = self.tik_instance.Tensor("float32", (8, 8), name="const_one", scope=tik.scope_ubuf)
            const_zero = self.tik_instance.Tensor("float32", (8, 8), name="const_zero", scope=tik.scope_ubuf)
            l_ratio = self.tik_instance.Tensor("float32", (256, 8), name="l_ratio", scope=tik.scope_ubuf)
            r_ratio = self.tik_instance.Tensor("float32", (256, 8), name="r_ratio", scope=tik.scope_ubuf)
            count_num = self.tik_instance.Tensor("int32", (40, 16), name="count_num", scope=tik.scope_ubuf)
            self.dup_zero(l_ratio, 256 * 8)
            self.dup_zero(r_ratio, 256 * 8)

            self.tik_instance.vec_dup(self.MASK, const_one, 1.0, 1, 8)
            self.tik_instance.vec_dup(self.MASK, const_zero, 0.0, 1, 8)

            self.l1_ratio(const_zero, const_one, l_ratio, r_ratio)
            self.l1_count(const_zero, count_num)

    def l1_ratio(self, const_zero, const_one, l_ratio, r_ratio):
        """
        use to calculate ratio
        :param const_zero: ub tensor dup by zero
        :param const_one: ub tensor dup by one
        :param l_ratio: ub tensor
        :param r_ratio: ub tensor
        :return:
        """
        index_256 = self.tik_instance.Tensor("float32", (256, 8), name="index_256", scope=tik.scope_ubuf)
        int_index = self.tik_instance.Tensor("int32", (256, 8), name="int_index", scope=tik.scope_ubuf)
        float_index = self.tik_instance.Tensor("float32", (256, 8), name="float_index", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.w_loop) as loop_idx:
            with self.tik_instance.for_range(0, 256) as num_idx:
                self.tik_instance.vec_dup(8, index_256[num_idx * 8], loop_idx * 256 + num_idx + self.dst_start_w, 1, 8)

            if self.half_pixel_centers:
                self.tik_instance.vec_adds(self.MASK, index_256, index_256, 0.5, 32, 8, 8)
                self.tik_instance.vec_muls(self.MASK, index_256, index_256, self.scale_w, 32, 8, 8)
                self.tik_instance.vec_adds(self.MASK, index_256, index_256, -0.5, 32, 8, 8)
                self.tik_instance.vec_max(self.MASK, index_256, index_256, const_zero, 32, 8, 8, 0)
            else:
                self.tik_instance.vec_muls(self.MASK, index_256, index_256, self.scale_w, 32, 8, 8)

            self.tik_instance.vec_adds(self.MASK, index_256, index_256, -self.src_start_w, 32, 8, 8)
            self.tik_instance.vec_conv(self.MASK, "floor", int_index, index_256, 32, 8, 8)
            self.tik_instance.vec_conv(self.MASK, "", float_index, int_index, 32, 8, 8)

            self.tik_instance.vec_sub(self.MASK, r_ratio, index_256, float_index, 32, 8, 8, 8)
            self.tik_instance.vec_sub(self.MASK, l_ratio, const_one, r_ratio, 32, 8, 0, 8)

            self.data_move(self.int_index_l1, int_index, [loop_idx * 256 * 8, 0], num=256 * 8)
            self.data_move(self.l_ratio_l1, l_ratio, [loop_idx * 256 * 8, 0], num=256 * 8)
            self.data_move(self.r_ratio_l1, r_ratio, [loop_idx * 256 * 8, 0], num=256 * 8)

    def l1_count(self, const_zero, count_num):
        """
        :param const_zero: ub tensor dup by zero
        :param count_num: ub tensor
        :return:
        """
        tmp_index = self.tik_instance.Tensor("float32", (256, 16), name="tmp_index", scope=tik.scope_ubuf)
        tmp_int_index = self.tik_instance.Tensor("int32", (256, 16), name="tmp_int_index", scope=tik.scope_ubuf)
        count_scalar = self.tik_instance.Scalar("int32", name="count", init_value=0)
        src_idx = self.tik_instance.Scalar("int32", name="src_idx", init_value=0)
        loop_scalar = self.tik_instance.Scalar("int32", name="loop", init_value=1)
        base_scalar = self.tik_instance.Scalar("int32", name="base", init_value=0)
        tmp_src = self.tik_instance.Scalar("int32", name="tmp")
        self.dup_zero(tmp_index, 256 * 16)

        with self.tik_instance.for_range(0, self.grads_w) as dst_w:
            tmp_index[dst_w].set_as(dst_w + self.dst_start_w)

        if self.half_pixel_centers:
            self.tik_instance.vec_adds(self.MASK, tmp_index, tmp_index, 0.5, 64, 8, 8)
            self.tik_instance.vec_muls(self.MASK, tmp_index, tmp_index, self.scale_w, 64, 8, 8)
            self.tik_instance.vec_adds(self.MASK, tmp_index, tmp_index, -0.5, 64, 8, 8)
            self.tik_instance.vec_max(self.MASK, tmp_index, tmp_index, const_zero, 64, 8, 8, 0)
        else:
            self.tik_instance.vec_muls(self.MASK, tmp_index, tmp_index, self.scale_w, 64, 8, 8)

        self.tik_instance.vec_adds(self.MASK, tmp_index, tmp_index, -self.src_start_w, 64, 8, 8)
        self.tik_instance.vec_conv(self.MASK, "floor", tmp_int_index, tmp_index, 64, 8, 8)
        self.tik_instance.vec_dup(self.MASK, count_num, 0, 10, 8)

        with self.tik_instance.for_range(0, self.grads_w) as idx:
            with self.tik_instance.if_scope(idx < loop_scalar * 256):
                src_idx.set_as(tmp_int_index[idx])
                tmp_src.set_as(src_idx)
                src_idx.set_as(src_idx + base_scalar)
                count_scalar.set_as(count_num[src_idx])
                count_scalar.set_as(count_scalar + 1)
                count_num[src_idx].set_as(count_scalar)
            with self.tik_instance.else_scope():
                src_idx.set_as(tmp_int_index[idx])
                loop_scalar.set_as(loop_scalar + 1)
                with self.tik_instance.if_scope(tmp_src == src_idx):
                    base_scalar.set_as(base_scalar + 1)
                src_idx.set_as(src_idx + base_scalar)
                count_scalar.set_as(count_num[src_idx])
                count_scalar.set_as(count_scalar + 1)
                count_num[src_idx].set_as(count_scalar)
        self.data_move(self.count_num_l1, count_num, [0, 0], num=640)

    def get_tiling_params(self):
        """
        get runtime params from tiling
        :return: None
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int32",
                                                 shape=(self.TILING_NUM,),
                                                 scope=tik.scope_ubuf,
                                                 name="tiling_ub")
            self.data_move(tiling_ub, self.tiling_gm, [0, 0], num=self.TILING_NUM)

            self.tiling_mode.set_as(tiling_ub[0])
            self.need_core_num.set_as(tiling_ub[1])
            self.nc1_per_core.set_as(tiling_ub[2])
            self.nc1_last_core.set_as(tiling_ub[3])
            self.h_per_core.set_as(tiling_ub[4])
            self.h_last_core.set_as(tiling_ub[5])
            self.grads_h.set_as(tiling_ub[6])
            self.grads_w.set_as(tiling_ub[7])
            self.images_h.set_as(tiling_ub[8])
            self.images_w.set_as(tiling_ub[9])
            self.grad_each_core.set_as(tiling_ub[10])
            self.output_each_core.set_as(tiling_ub[11])
            self.grad_move_num.set_as(tiling_ub[12])
            self.output_move_num.set_as(tiling_ub[13])
            self.nc1.set_as(tiling_ub[14])
            self.w_loop.set_as(tiling_ub[15])
            self.w_tail.set_as(tiling_ub[16])
            self.core_num_var.set_as(tiling_ub[17])
            self.resize_h.set_as(tiling_ub[18])
            self.resize_w.set_as(tiling_ub[19])
            self.ori_h.set_as(tiling_ub[20])
            self.ori_w.set_as(tiling_ub[21])
            self.src_start_w.set_as(tiling_ub[22])
            self.dst_start_w.set_as(tiling_ub[23])
            self._get_scale(self.scale_h, self.images_h, self.grads_h)
            self._get_scale(self.scale_w, self.images_w, self.grads_w)

    def tiling_compute(self):
        """
        the main function of tiling compute
        :return:
        """
        self.get_tiling_params()

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_idx:
            with self.tik_instance.if_scope(self.tiling_mode == self.TILING_MODE0):
                with self.tik_instance.new_stmt_scope():
                    self.n2n(core_idx)
            with self.tik_instance.if_scope(self.tiling_mode == self.TILING_MODE1):
                with self.tik_instance.new_stmt_scope():
                    self.one2n_small(core_idx)
            with self.tik_instance.if_scope(self.tiling_mode == self.TILING_MODE2):
                with self.tik_instance.new_stmt_scope():
                    self.one2n_big(core_idx)
            with self.tik_instance.if_scope(self.tiling_mode == self.TILING_MODE3):
                with self.tik_instance.new_stmt_scope():
                    self.n2one(core_idx)
            with self.tik_instance.if_scope(self.tiling_mode == self.TILING_MODE4):
                with self.tik_instance.new_stmt_scope():
                    self.small_in_out(core_idx)
            if self.l1_status:
                with self.tik_instance.if_scope(self.tiling_mode == self.TILING_MODE5):
                    self.l1_function()
                    with self.tik_instance.new_stmt_scope():
                        self.normal_small(core_idx)
            with self.tik_instance.if_scope(self.tiling_mode == self.TILING_MODE6):
                with self.tik_instance.new_stmt_scope():
                    self.normal_big(core_idx)

    def compute(self):
        """
        op compute
        :return:
        """
        self.tiling_compute()
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num, "l1_support": self.l1_status,
                                                            "tensor_c": self.tensor_c})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.grads_gm, self.images_gm],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance

    @staticmethod
    def get_dtype_size(dtype):
        """
        :param dtype: data type
        :return:
        """
        dtype_byte_size = get_bit_len(dtype) // 8
        return dtype_byte_size

    def get_src_index(self, scale, dst_index_scalar, src_scalar):
        """
        calculate the source index
        :param scale: scalar of scale
        :param dst_index_scalar: scalar of dst_scalar
        :param src_scalar: scalar of src
        :return:
        """
        if self.half_pixel_centers:
            src_scalar.set_as(scale * (dst_index_scalar + 0.5) - 0.5)
            with self.tik_instance.if_scope(src_scalar < 0):
                src_scalar.set_as(0)
            with self.tik_instance.else_scope():
                pass
        else:
            src_scalar.set_as(scale * dst_index_scalar)

    @staticmethod
    def get_ratio(src_index, idx_scalar, l_ratio_scalar, r_ratio_scalar):
        """
        calculate ratio
        :param src_index: scalar of src_index
        :param idx_scalar: scalar of idx
        :param l_ratio_scalar: scalar of l_ratio
        :param r_ratio_scalar: scalar of r_ratio
        :return:
        """
        r_ratio = src_index - idx_scalar
        l_ratio = 1 - r_ratio
        l_ratio_scalar.set_as(l_ratio)
        r_ratio_scalar.set_as(r_ratio)

    def get_stride(self, idx_scalar, stride_scalar, image_size):
        """
        calculate stride
        :param idx_scalar: scalar idx
        :param stride_scalar: scalar stride
        :param image_size: scalar image_size
        :return:
        """
        with self.tik_instance.if_scope(idx_scalar < image_size - 1):
            stride_scalar.set_as(1)
        with self.tik_instance.else_scope():
            stride_scalar.set_as(0)

    def get_output_idx(self, h_idx_scalar, w_idx_scalar, images_w):
        """
        calculate index of output
        """
        output_idx = (h_idx_scalar * images_w + w_idx_scalar) * self.c0
        return output_idx

    def get_grad_idx(self, dst_h, dst_w, grads_w, big_grad=False):
        """
        calculate index of grad
        """
        if big_grad:
            grad_idx = dst_w * self.c0
        else:
            grad_idx = (dst_h * grads_w + dst_w) * self.c0
        return grad_idx

    def check_next_loop(self, w_idx_scalar, max_w, loop_init, scalar):
        """
        check if need next loop or not
        """
        with self.tik_instance.if_scope(tik.all(w_idx_scalar == (loop_init + 1) * max_w, scalar.w_stride == 1)):
            scalar.next_loop.set_as(1)
        with self.tik_instance.if_scope(w_idx_scalar > max_w * (loop_init + 1)):
            scalar.next_loop.set_as(1)
        with self.tik_instance.else_scope():
            pass

    def get_output_loop_idx(self, w_idx_scalar, max_w, scalar):
        """
        calculate output loop
        """
        with self.tik_instance.if_scope(tik.all(w_idx_scalar != 0, w_idx_scalar % max_w == 0, scalar.w_stride == 0)):
            scalar.output_idx00.set_as(max_w * self.c0)
        with self.tik_instance.else_scope():
            scalar.output_idx00.set_as(w_idx_scalar % max_w * self.c0)

    def calculate_h(self, scalar, dst_h):
        """
        calculate source h
        """
        scalar.dst_h.set_as(dst_h)
        self.get_src_index(self.scale_h, scalar.dst_h, scalar.src_h)
        self.tik_instance.scalar_conv("floor", scalar.h_idx, scalar.src_h)
        self.get_stride(scalar.h_idx, scalar.h_stride, self.images_h)
        self.get_ratio(scalar.src_h, scalar.h_idx, scalar.hl_ratio, scalar.hr_ratio)

    def calculate_w(self, scalar, dst_w):
        """
        calculate source w
        """
        scalar.dst_w.set_as(dst_w + self.dst_start_w)
        self.get_src_index(self.scale_w, scalar.dst_w, scalar.src_w)
        scalar.src_w.set_as(scalar.src_w - self.src_start_w)
        self.tik_instance.scalar_conv("floor", scalar.w_idx, scalar.src_w)
        self.get_stride(scalar.w_idx, scalar.w_stride, self.images_w)
        self.get_ratio(scalar.src_w, scalar.w_idx, scalar.wl_ratio, scalar.wr_ratio)

    # 'pylint:disable=too-many-arguments
    def calculate_grad(self,
                       grad_ub,
                       output_ub,
                       grad_idx,
                       output_idx_l,
                       output_idx_r,
                       l_ratio_scalar,
                       r_ratio_scalar,
                       repeat_time=1,
                       dst_stride=0,
                       src_stride=0):
        """
        calculate grad by axpy
        """
        self.tik_instance.vec_axpy(16, output_ub[output_idx_l], grad_ub[grad_idx], l_ratio_scalar, repeat_time,
                                   dst_stride, src_stride)
        self.tik_instance.vec_axpy(16, output_ub[output_idx_r], grad_ub[grad_idx], r_ratio_scalar, repeat_time,
                                   dst_stride, src_stride)

    def small_in_out(self, core_idx):
        """
        calculate output when input shape and grad shape are smaller than ub
        """
        # images: H0 * W0 < 2048, after resize: H1 * W1 < 2048
        grads_ub = self.tik_instance.Tensor(self.grads_dtype, (self.tensor_size,),
                                            scope=tik.scope_ubuf,
                                            name="grads_ub")
        output_ub = self.tik_instance.Tensor(self.grads_dtype, (self.tensor_size,),
                                             scope=tik.scope_ubuf,
                                             name="output_ub")

        with self.tik_instance.if_scope(core_idx < self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.nc1_per_core) as per_idx:
                self.dup_zero(output_ub, self.tensor_size)
                nc1_idx = core_idx * self.nc1_per_core + per_idx
                grad_offset = nc1_idx * self.grad_each_core
                output_offset = nc1_idx * self.output_each_core
                self.small_each_core(grads_ub, output_ub, grad_offset, output_offset)

        with self.tik_instance.if_scope(core_idx == self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.nc1_last_core) as last_idx:
                self.dup_zero(output_ub, self.tensor_size)
                nc1_idx = core_idx * self.nc1_per_core + last_idx
                grad_offset = nc1_idx * self.grad_each_core
                output_offset = nc1_idx * self.output_each_core
                self.small_each_core(grads_ub, output_ub, grad_offset, output_offset)

    def small_each_core(self, grads_ub, output_ub, grad_offset, output_offset):
        """
        core function
        """
        scalar = self.CommomScalar(self.tik_instance)

        self.data_move(grads_ub, self.grads_gm, offsets=[0, grad_offset], num=self.grad_each_core)
        with self.tik_instance.for_range(0, self.grads_h) as dst_h:
            self.calculate_h(scalar, dst_h)

            with self.tik_instance.for_range(0, self.grads_w) as dst_w:
                self.small_in_loop(grads_ub, output_ub, scalar, dst_h, dst_w)
        self.data_move(self.output_gm, output_ub, offsets=[output_offset, 0], num=self.output_each_core)

    # 'pylint:disable=too-many-arguments
    def small_in_loop(self, grads_ub, output_ub, scalar, dst_h, dst_w, base_grad_offset=0, big_grad=False):
        """
        loop function
        """
        self.calculate_w(scalar, base_grad_offset + dst_w)

        output_idx00 = self.get_output_idx(scalar.h_idx, scalar.w_idx, self.images_w)
        grads_idx = self.get_grad_idx(dst_h, dst_w, self.grads_w, big_grad)

        output_idx01 = output_idx00 + scalar.w_stride * self.c0
        output_idx10 = output_idx00 + scalar.h_stride * self.images_w * self.c0
        output_idx11 = output_idx10 + scalar.w_stride * self.c0

        scalar.l_ratio.set_as(scalar.hl_ratio * scalar.wl_ratio)
        scalar.r_ratio.set_as(scalar.hl_ratio * scalar.wr_ratio)

        self.calculate_grad(grads_ub, output_ub, grads_idx, output_idx00, output_idx01, scalar.l_ratio, scalar.r_ratio)

        scalar.l_ratio.set_as(scalar.hr_ratio * scalar.wl_ratio)
        scalar.r_ratio.set_as(scalar.hr_ratio * scalar.wr_ratio)
        self.calculate_grad(grads_ub, output_ub, grads_idx, output_idx10, output_idx11, scalar.l_ratio, scalar.r_ratio)

    def normal_small(self, core_idx):
        """
        calculate output when input shape and grad shape are small
        """
        self.tik_instance.set_atomic_add(1)
        grads_ub = self.tik_instance.Tensor(self.grads_dtype, (256, 16), scope=tik.scope_ubuf, name="grads_ub")
        output_ub = self.tik_instance.Tensor(self.grads_dtype, (640, 16), scope=tik.scope_ubuf, name="output_ub")
        output_ub2 = self.tik_instance.Tensor(self.grads_dtype, (640, 16), scope=tik.scope_ubuf, name="output_ub2")
        self.dup_zero(grads_ub, 256 * 16)

        with self.tik_instance.if_scope(core_idx < self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.h_per_core) as per_idx:
                h_idx = core_idx * self.h_per_core + per_idx
                self.small_in_each_core(grads_ub, output_ub, output_ub2, h_idx)

        with self.tik_instance.if_scope(core_idx == self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.h_last_core) as last_idx:
                h_idx = core_idx * self.h_per_core + last_idx
                self.small_in_each_core(grads_ub, output_ub, output_ub2, h_idx)
        self.tik_instance.set_atomic_add(0)

    # 'pylint:disable=too-many-locals
    def small_in_each_core(self, grads_ub, output_ub, output_ub2, h_idx):
        """
        core function
        """
        scalar = self.CommomScalar(self.tik_instance)
        self.calculate_h(scalar, h_idx)
        count_num = self.tik_instance.Tensor("int32", (640,), name="count_num", scope=tik.scope_ubuf)
        mul_out0 = self.tik_instance.Tensor("float32", (512, 16), name="mul_out0", scope=tik.scope_ubuf)
        mul_out1 = self.tik_instance.Tensor("float32", (512, 16), name="mul_out1", scope=tik.scope_ubuf)
        int_index = self.tik_instance.Tensor("int32", (256, 8), name="int_index", scope=tik.scope_ubuf)
        l_ratio = self.tik_instance.Tensor("float32", (256, 8), name="l_ratio", scope=tik.scope_ubuf)
        r_ratio = self.tik_instance.Tensor("float32", (256, 8), name="r_ratio", scope=tik.scope_ubuf)
        l0_ratio = self.tik_instance.Tensor("float32", (256, 8), name="l0_ratio", scope=tik.scope_ubuf)
        l1_ratio = self.tik_instance.Tensor("float32", (256, 8), name="l1_ratio", scope=tik.scope_ubuf)
        r0_ratio = self.tik_instance.Tensor("float32", (256, 8), name="r0_ratio", scope=tik.scope_ubuf)
        r1_ratio = self.tik_instance.Tensor("float32", (256, 8), name="r1_ratio", scope=tik.scope_ubuf)
        w_out_begin = self.tik_instance.Scalar("int32", name="w_out_begin")
        w_out_end = self.tik_instance.Scalar("int32", name="w_out_end")
        src_w_idx = self.tik_instance.Scalar("int32", name="src_w_idx")
        dst_w_idx = self.tik_instance.Scalar("int32", name="dst_w_idx")
        add_repeat = self.tik_instance.Scalar("int32", name="add_repeat")
        self.data_move(count_num, self.count_num_l1, [0, 0], num=640)

        with self.tik_instance.for_range(0, self.nc1) as nc1_idx:
            self.dup_zero(output_ub, num=640 * 16)
            self.dup_zero(output_ub2, num=640 * 16)

            src_w_idx.set_as(0)
            with self.tik_instance.for_range(0, self.w_loop) as loop_idx:
                dst_w_idx.set_as(0)
                self.data_move(int_index, self.int_index_l1, [0, loop_idx * 256 * 8], num=256 * 8)
                self.data_move(l_ratio, self.l_ratio_l1, [0, loop_idx * 256 * 8], num=256 * 8)
                self.data_move(r_ratio, self.r_ratio_l1, [0, loop_idx * 256 * 8], num=256 * 8)
                self.tik_instance.vec_muls(self.MASK, l0_ratio, l_ratio, scalar.hl_ratio, 32, 8, 8)
                self.tik_instance.vec_muls(self.MASK, r0_ratio, r_ratio, scalar.hl_ratio, 32, 8, 8)
                self.tik_instance.vec_muls(self.MASK, l1_ratio, l_ratio, scalar.hr_ratio, 32, 8, 8)
                self.tik_instance.vec_muls(self.MASK, r1_ratio, r_ratio, scalar.hr_ratio, 32, 8, 8)

                with self.tik_instance.if_scope(loop_idx != self.w_loop - 1):
                    grad_offset = (nc1_idx * self.grads_h + h_idx) * self.grad_move_num + loop_idx * 256 * 16
                    self.data_move(grads_ub, self.grads_gm, [0, grad_offset], num=256 * 16)
                    self.small_in_mul(grads_ub, l0_ratio, r0_ratio, l1_ratio, r1_ratio, mul_out0, mul_out1)
                    w_out_begin.set_as(int_index[0])
                    w_out_end.set_as(int_index[2047])
                    self.small_in_add(mul_out0, mul_out1, output_ub, output_ub2, count_num, src_w_idx, add_repeat,
                                      dst_w_idx, w_out_begin, w_out_end + 1)

                with self.tik_instance.else_scope():
                    grad_offset = (nc1_idx * self.grads_h + h_idx) * self.grad_move_num + loop_idx * 256 * 16
                    self.data_move(grads_ub, self.grads_gm, [0, grad_offset], num=self.w_tail * 16)
                    self.small_in_mul(grads_ub, l0_ratio, r0_ratio, l1_ratio, r1_ratio, mul_out0, mul_out1)
                    w_out_begin.set_as(int_index[0])
                    self.small_in_add(mul_out0, mul_out1, output_ub, output_ub2, count_num, src_w_idx, add_repeat,
                                      dst_w_idx, w_out_begin, self.images_w)

            gm_offset = (nc1_idx * self.images_h + scalar.h_idx) * self.output_move_num
            gm_offset2 = gm_offset + scalar.h_stride * self.output_move_num
            self.data_move(self.output_gm, output_ub, [gm_offset, 0], num=self.output_move_num)
            self.data_move(self.output_gm, output_ub2, [gm_offset2, 0], num=self.output_move_num)

    # 'pylint:disable=too-many-arguments
    def small_in_mul(self, grads_ub, l0_ratio, r0_ratio, l1_ratio, r1_ratio, mul_out0, mul_out1):
        """
        mul function
        """
        self.tik_instance.vmul(self.MASK, mul_out0[0], l0_ratio, grads_ub[0], 32, 4, 1, 2, 32, 8, 16)
        self.tik_instance.vmul(self.MASK, mul_out0[8], l0_ratio, grads_ub[8], 32, 4, 1, 2, 32, 8, 16)
        self.tik_instance.vmul(self.MASK, mul_out0[16], r0_ratio, grads_ub[0], 32, 4, 1, 2, 32, 8, 16)
        self.tik_instance.vmul(self.MASK, mul_out0[24], r0_ratio, grads_ub[8], 32, 4, 1, 2, 32, 8, 16)
        self.tik_instance.vmul(self.MASK, mul_out1[0], l1_ratio, grads_ub[0], 32, 4, 1, 2, 32, 8, 16)
        self.tik_instance.vmul(self.MASK, mul_out1[8], l1_ratio, grads_ub[8], 32, 4, 1, 2, 32, 8, 16)
        self.tik_instance.vmul(self.MASK, mul_out1[16], r1_ratio, grads_ub[0], 32, 4, 1, 2, 32, 8, 16)
        self.tik_instance.vmul(self.MASK, mul_out1[24], r1_ratio, grads_ub[8], 32, 4, 1, 2, 32, 8, 16)

    # 'pylint:disable=too-many-arguments
    def small_in_add(self, mul_out0, mul_out1, output_ub, output_ub2, count_num, src_w_idx, add_repeat, dst_w_idx,
                     w_out_begin, w_out_end):
        """
        add function
        """
        with self.tik_instance.for_range(w_out_begin, w_out_end) as tmp_idx:
            add_repeat.set_as(count_num[src_w_idx])

            with self.tik_instance.if_scope(tmp_idx < self.images_w - 1):
                self.tik_instance.vec_add(32, output_ub[tmp_idx * 16], mul_out0[dst_w_idx * 32],
                                          output_ub[tmp_idx * 16], add_repeat, 0, 4, 0)
                self.tik_instance.vec_add(32, output_ub2[tmp_idx * 16], mul_out1[dst_w_idx * 32],
                                          output_ub2[tmp_idx * 16], add_repeat, 0, 4, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.vec_add(16, output_ub[tmp_idx * 16], mul_out0[dst_w_idx * 32],
                                          output_ub[tmp_idx * 16], add_repeat, 0, 4, 0)
                self.tik_instance.vec_add(16, output_ub[tmp_idx * 16], mul_out0[dst_w_idx * 32 + 16],
                                          output_ub[tmp_idx * 16], add_repeat, 0, 4, 0)
                self.tik_instance.vec_add(16, output_ub2[tmp_idx * 16], mul_out1[dst_w_idx * 32],
                                          output_ub2[tmp_idx * 16], add_repeat, 0, 4, 0)
                self.tik_instance.vec_add(16, output_ub2[tmp_idx * 16], mul_out1[dst_w_idx * 32 + 16],
                                          output_ub2[tmp_idx * 16], add_repeat, 0, 4, 0)

            src_w_idx.set_as(src_w_idx + 1)
            dst_w_idx.set_as(dst_w_idx + add_repeat)

    def normal_big(self, core_idx):
        """
        calculate output when input shape and grad shape are big
        """
        self.tik_instance.set_atomic_add(1)
        grads_ub = self.tik_instance.Tensor(self.grads_dtype, (self.tensor_size,),
                                            scope=tik.scope_ubuf,
                                            name="grads_ub")
        output_ub = self.tik_instance.Tensor(self.grads_dtype, (self.tensor_size // 2,),
                                             scope=tik.scope_ubuf,
                                             name="output_ub")
        output_ub2 = self.tik_instance.Tensor(self.grads_dtype, (self.tensor_size // 2,),
                                              scope=tik.scope_ubuf,
                                              name="output_ub2")
        with self.tik_instance.if_scope(core_idx < self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.h_per_core) as per_idx:
                self.dup_zero(output_ub, num=self.tensor_size // 2)
                self.dup_zero(output_ub2, num=self.tensor_size // 2)
                h_idx = core_idx * self.h_per_core + per_idx
                self.big_each_core(grads_ub, output_ub, output_ub2, h_idx)

        with self.tik_instance.if_scope(core_idx == self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.h_last_core) as last_idx:
                self.dup_zero(output_ub, num=self.tensor_size // 2)
                self.dup_zero(output_ub2, num=self.tensor_size // 2)
                h_idx = core_idx * self.h_per_core + last_idx
                self.big_each_core(grads_ub, output_ub, output_ub2, h_idx)
        self.tik_instance.set_atomic_add(0)

    # 'pylint:disable=too-many-locals
    def big_each_core(self, grads_ub, output_ub, output_ub2, h_idx):
        """
        core function
        """
        scalar = self.CommomScalar(self.tik_instance)
        self.calculate_h(scalar, h_idx)

        max_w = self.tensor_c
        output_max_w = max_w // 2
        use_max_w = output_max_w - 1
        loop = self.grads_w // max_w
        tail_w = self.grads_w % max_w
        output_loop = self.images_w // output_max_w
        grad_move_num = max_w * self.c0
        output_move_num = self.images_w * self.c0
        max_move_num = output_max_w * self.c0
        use_move_num = use_max_w * self.c0

        with self.tik_instance.for_range(0, self.nc1) as nc1_idx:
            loop_init = self.tik_instance.Scalar(dtype="int32", name="loop_init", init_value=0)
            base_offset = (nc1_idx * self.images_h + scalar.h_idx) * output_move_num
            base_grad_offset = (nc1_idx * self.grads_h + h_idx) * self.grads_w * self.c0

            with self.tik_instance.if_scope(output_loop > 0):
                with self.tik_instance.if_scope(loop > 0):
                    with self.tik_instance.for_range(0, loop) as loop_idx:
                        grads_offset = base_grad_offset + loop_idx * grad_move_num
                        self.data_move(grads_ub, self.grads_gm, [0, grads_offset], num=grad_move_num)

                        base_w = loop_idx * max_w
                        self.big_in_loop(grads_ub, output_ub, output_ub2, scalar, h_idx, base_offset, base_w, max_w,
                                         use_max_w, use_move_num, output_move_num, max_move_num, loop_init)

                with self.tik_instance.if_scope(tail_w > 0):
                    grads_offset = base_grad_offset + loop * grad_move_num
                    self.data_move(grads_ub, self.grads_gm, [0, grads_offset], num=tail_w * self.c0)

                    base_w = loop * max_w
                    self.big_in_loop(grads_ub, output_ub, output_ub2, scalar, h_idx, base_offset, base_w, tail_w,
                                     use_max_w, use_move_num, output_move_num, max_move_num, loop_init)

            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(loop > 0):
                    with self.tik_instance.for_range(0, loop) as loop_idx:
                        grads_offset = base_grad_offset + loop_idx * grad_move_num
                        self.data_move(grads_ub, self.grads_gm, [0, grads_offset], num=grad_move_num)

                        base_w = loop_idx * max_w
                        self.big_in_loop2(grads_ub, output_ub, output_ub2, scalar, h_idx, base_w, max_w)

                with self.tik_instance.if_scope(tail_w > 0):
                    grads_offset = base_grad_offset + loop * grad_move_num
                    self.data_move(grads_ub, self.grads_gm, [0, grads_offset], num=tail_w * self.c0)

                    base_w = loop * max_w
                    self.big_in_loop2(grads_ub, output_ub, output_ub2, scalar, h_idx, base_w, tail_w)

                gm_offset = base_offset
                gm_offset2 = gm_offset + scalar.h_stride * output_move_num
                self.data_move(self.output_gm, output_ub, [gm_offset, 0], num=output_move_num)
                self.data_move(self.output_gm, output_ub2, [gm_offset2, 0], num=output_move_num)
                self.dup_zero(output_ub, num=self.tensor_size // 2)
                self.dup_zero(output_ub2, num=self.tensor_size // 2)

    # 'pylint:disable=too-many-arguments,too-many-locals
    def big_in_loop(self, grads_ub, output_ub, output_ub2, scalar, h_idx, base_offset, base_w, max_w, use_max_w,
                    use_move_num, output_move_num, max_move_num, loop_init):
        """
        loop function
        """
        with self.tik_instance.for_range(0, max_w) as dst_w:
            self.calculate_w(scalar, base_w + dst_w)
            self.check_next_loop(scalar.w_idx, use_max_w, loop_init, scalar)

            grads_idx = self.get_grad_idx(h_idx, dst_w, self.grads_w, big_grad=True)
            self.get_output_loop_idx(scalar.w_idx, use_max_w, scalar)
            output_idx01 = scalar.output_idx00 + scalar.w_stride * self.c0

            with self.tik_instance.if_scope(scalar.next_loop == 1):
                gm_offset = base_offset + loop_init * use_move_num
                gm_offset2 = gm_offset + scalar.h_stride * output_move_num
                self.data_move(self.output_gm, output_ub, [gm_offset, 0], num=max_move_num)
                self.data_move(self.output_gm, output_ub2, [gm_offset2, 0], num=max_move_num)

                loop_init.set_as(scalar.w_idx // use_max_w)
                scalar.next_loop.set_as(0)
                self.dup_zero(output_ub, num=self.tensor_size // 2)
                self.dup_zero(output_ub2, num=self.tensor_size // 2)

                scalar.l_ratio.set_as(scalar.hl_ratio * scalar.wl_ratio)
                scalar.r_ratio.set_as(scalar.hl_ratio * scalar.wr_ratio)

                self.calculate_grad(grads_ub, output_ub, grads_idx, scalar.output_idx00, output_idx01, scalar.l_ratio,
                                    scalar.r_ratio)

                scalar.l_ratio.set_as(scalar.hr_ratio * scalar.wl_ratio)
                scalar.r_ratio.set_as(scalar.hr_ratio * scalar.wr_ratio)

                self.calculate_grad(grads_ub, output_ub2, grads_idx, scalar.output_idx00, output_idx01, scalar.l_ratio,
                                    scalar.r_ratio)

            with self.tik_instance.else_scope():
                scalar.l_ratio.set_as(scalar.hl_ratio * scalar.wl_ratio)
                scalar.r_ratio.set_as(scalar.hl_ratio * scalar.wr_ratio)

                self.calculate_grad(grads_ub, output_ub, grads_idx, scalar.output_idx00, output_idx01, scalar.l_ratio,
                                    scalar.r_ratio)

                scalar.l_ratio.set_as(scalar.hr_ratio * scalar.wl_ratio)
                scalar.r_ratio.set_as(scalar.hr_ratio * scalar.wr_ratio)

                self.calculate_grad(grads_ub, output_ub2, grads_idx, scalar.output_idx00, output_idx01, scalar.l_ratio,
                                    scalar.r_ratio)

        gm_offset = base_offset + loop_init * use_move_num
        gm_offset2 = gm_offset + scalar.h_stride * output_move_num
        self.data_move(self.output_gm, output_ub, [gm_offset, 0], num=max_move_num)
        self.data_move(self.output_gm, output_ub2, [gm_offset2, 0], num=max_move_num)
        self.dup_zero(output_ub, num=self.tensor_size // 2)
        self.dup_zero(output_ub2, num=self.tensor_size // 2)

    # 'pylint:disable=too-many-arguments
    def big_in_loop2(self, grads_ub, output_ub, output_ub2, scalar, h_idx, base_w, max_w):
        """
        loop function
        """
        with self.tik_instance.for_range(0, max_w) as dst_w:
            self.calculate_w(scalar, base_w + dst_w)
            grads_idx = self.get_grad_idx(h_idx, dst_w, self.grads_w, big_grad=True)
            output_idx00 = scalar.w_idx * self.c0
            output_idx01 = output_idx00 + scalar.w_stride * self.c0

            scalar.l_ratio.set_as(scalar.hl_ratio * scalar.wl_ratio)
            scalar.r_ratio.set_as(scalar.hl_ratio * scalar.wr_ratio)

            self.calculate_grad(grads_ub, output_ub, grads_idx, output_idx00, output_idx01, scalar.l_ratio,
                                scalar.r_ratio)

            scalar.l_ratio.set_as(scalar.hr_ratio * scalar.wl_ratio)
            scalar.r_ratio.set_as(scalar.hr_ratio * scalar.wr_ratio)

            self.calculate_grad(grads_ub, output_ub2, grads_idx, output_idx00, output_idx01, scalar.l_ratio,
                                scalar.r_ratio)

    def n2n(self, core_idx):
        """
        calculate output when input shape and grad shape are same
        """
        grads_ub = self.tik_instance.Tensor(self.grads_dtype, (self.tensor_size,),
                                            scope=tik.scope_ubuf,
                                            name="grads_ub")

        with self.tik_instance.if_scope(core_idx < self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.h_per_core) as per_idx:
                h_idx = core_idx * self.h_per_core + per_idx
                self.n2n_core(grads_ub, h_idx)

        with self.tik_instance.if_scope(core_idx == self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.h_last_core) as last_idx:
                h_idx = core_idx * self.h_per_core + last_idx
                self.n2n_core(grads_ub, h_idx)

    def n2n_core(self, grads_ub, h_idx):
        """
        core function
        """
        max_w = self.tensor_size // self.c0

        with self.tik_instance.if_scope(self.grads_w < max_w):
            with self.tik_instance.for_range(0, self.nc1) as nc1_idx:
                grads_offset = (nc1_idx * self.grads_h + h_idx) * self.grads_w * self.c0
                self.data_move(grads_ub, self.grads_gm, [0, grads_offset], num=self.grads_w * self.c0)
                self.data_move(self.output_gm, grads_ub, [grads_offset, 0], num=self.grads_w * self.c0)

        with self.tik_instance.else_scope():
            tail_w = self.tik_instance.Scalar("int32")
            loop = (self.grads_w + max_w - 1) // max_w
            tail_w.set_as(self.grads_w % max_w)
            grad_move_num = max_w * self.c0

            with self.tik_instance.if_scope(tail_w == 0):
                tail_w.set_as(max_w)

            with self.tik_instance.if_scope(loop > 1):
                self.n2n_loop(grads_ub, h_idx, loop, grad_move_num, tail_w)
            with self.tik_instance.else_scope():
                self.n2n_loop2(grads_ub, h_idx, loop, grad_move_num, tail_w)

    # 'pylint:disable=too-many-arguments
    def n2n_loop(self, grads_ub, h_idx, loop, grad_move_num, tail_w):
        """
        loop function
        """
        with self.tik_instance.for_range(0, self.nc1) as nc1_idx:
            base_grad_offset = (nc1_idx * self.grads_h + h_idx) * self.grads_w * self.c0
            with self.tik_instance.for_range(0, loop, thread_num=2) as loop_idx:
                with self.tik_instance.if_scope(loop_idx != loop - 1):
                    grads_offset = base_grad_offset + loop_idx * grad_move_num
                    self.data_move(grads_ub, self.grads_gm, [0, grads_offset], num=grad_move_num)
                    self.data_move(self.output_gm, grads_ub, [grads_offset, 0], num=grad_move_num)

                with self.tik_instance.else_scope():
                    grads_offset = base_grad_offset + loop_idx * grad_move_num
                    self.data_move(grads_ub, self.grads_gm, [0, grads_offset], num=tail_w * self.c0)
                    self.data_move(self.output_gm, grads_ub, [grads_offset, 0], num=tail_w * self.c0)

    # 'pylint:disable=too-many-arguments
    def n2n_loop2(self, grads_ub, h_idx, loop, grad_move_num, tail_w):
        """
        loop function
        """
        with self.tik_instance.for_range(0, self.nc1) as nc1_idx:
            base_grad_offset = (nc1_idx * self.grads_h + h_idx) * self.grads_w * self.c0
            with self.tik_instance.for_range(0, loop, thread_num=1) as loop_idx:
                with self.tik_instance.if_scope(loop_idx != loop - 1):
                    grads_offset = base_grad_offset + loop_idx * grad_move_num
                    self.data_move(grads_ub, self.grads_gm, [0, grads_offset], num=grad_move_num)
                    self.data_move(self.output_gm, grads_ub, [grads_offset, 0], num=grad_move_num)

                with self.tik_instance.else_scope():
                    grads_offset = base_grad_offset + loop_idx * grad_move_num
                    self.data_move(grads_ub, self.grads_gm, [0, grads_offset], num=tail_w * self.c0)
                    self.data_move(self.output_gm, grads_ub, [grads_offset, 0], num=tail_w * self.c0)

    def one2n_small(self, core_idx):
        """
        calculate output when input shape is [1,1] and grad shape is small
        :param core_idx:
        :return:
        """
        with self.tik_instance.if_scope(core_idx < self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.nc1_per_core) as per_idx:
                nc1_idx = core_idx * self.nc1_per_core + per_idx
                grad_offset = nc1_idx * self.grad_each_core
                output_offset = nc1_idx * self.output_each_core
                self.one2n_small_core(grad_offset, output_offset)

        with self.tik_instance.if_scope(core_idx == self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.nc1_last_core) as last_idx:
                nc1_idx = core_idx * self.nc1_per_core + last_idx
                grad_offset = nc1_idx * self.grad_each_core
                output_offset = nc1_idx * self.output_each_core
                self.one2n_small_core(grad_offset, output_offset)

    def one2n_small_core(self, grad_offset, output_offset):
        """
        core function
        """
        unit_w = 256
        iter_num = 8
        w_grad_ub = self.tik_instance.Tensor(self.grads_dtype, (unit_w * self.c0,),
                                             scope=tik.scope_ubuf,
                                             name="w_grad_ub")
        self.dup_zero(w_grad_ub, num=unit_w * self.c0)
        grads_ub = self.tik_instance.Tensor(self.grads_dtype, (self.tensor_size,),
                                            scope=tik.scope_ubuf,
                                            name="grads_ub")
        self.data_move(grads_ub, self.grads_gm, offsets=[0, grad_offset], num=self.grad_each_core)

        self.data_merge(grads_ub, w_grad_ub, unit_w, self.grads_h * self.grads_w)
        self.data_sum(w_grad_ub, unit_w * self.c0, iter_num)

        self.data_move(self.output_gm, w_grad_ub, [output_offset, 0], num=self.c0)

    def one2n_big(self, core_idx):
        """
        calculate output when input shape is [1,1] and grad shape is big
        """
        self.tik_instance.set_atomic_add(1)
        grads_ub = self.tik_instance.Tensor(self.grads_dtype, (self.tensor_size,),
                                            scope=tik.scope_ubuf,
                                            name="grads_ub")
        with self.tik_instance.if_scope(core_idx < self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.nc1_per_core) as per_idx:
                nc1_idx = core_idx * self.nc1_per_core + per_idx
                grad_offset = nc1_idx * self.grad_each_core
                output_offset = nc1_idx * self.output_each_core
                self.one2n_big_core(grads_ub, grad_offset, output_offset)

        with self.tik_instance.if_scope(core_idx == self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.nc1_last_core) as last_idx:
                nc1_idx = core_idx * self.nc1_per_core + last_idx
                grad_offset = nc1_idx * self.grad_each_core
                output_offset = nc1_idx * self.output_each_core
                self.one2n_big_core(grads_ub, grad_offset, output_offset)
        self.tik_instance.set_atomic_add(0)

    def one2n_big_core(self, grads_ub, grad_offset, output_offset):
        """
        core function
        """
        max_w = self.tensor_size // self.c0
        unit_w = 1024
        iter_num = 10
        loop = (self.grads_h * self.grads_w) // max_w
        tail_w = (self.grads_h * self.grads_w) % max_w
        offset = self.tik_instance.Scalar(dtype="int32", name="offset")
        offset.set_as(grad_offset)

        w_grad_ub = self.tik_instance.Tensor(self.grads_dtype, (unit_w * self.c0,),
                                             scope=tik.scope_ubuf,
                                             name="w_grad_ub")
        self.dup_zero(w_grad_ub, num=unit_w * self.c0)

        move_num = max_w * self.c0
        with self.tik_instance.for_range(0, loop) as idx:
            tmp_offset = offset + idx * move_num
            self.data_move(grads_ub, self.grads_gm, [0, tmp_offset], num=move_num)
            self.data_merge(grads_ub, w_grad_ub, unit_w, max_w)

        with self.tik_instance.if_scope(tail_w > 0):
            offset.set_as(offset + loop * move_num)
            self.data_move(grads_ub, self.grads_gm, [0, offset], num=tail_w * self.c0)
            self.data_merge(grads_ub, w_grad_ub, unit_w, tail_w)

        self.data_sum(w_grad_ub, unit_w * self.c0, iter_num)

        self.data_move(self.output_gm, w_grad_ub, [output_offset, 0], num=self.c0)

    def n2one(self, core_idx):
        """
        calculate output when grad shape is [1,1]
        """
        grads_ub = self.tik_instance.Tensor(self.grads_dtype, (self.c0,), scope=tik.scope_ubuf, name="grads_ub")
        with self.tik_instance.if_scope(core_idx < self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.nc1_per_core) as per_idx:
                self.dup_zero(grads_ub, num=self.c0)
                nc1_idx = core_idx * self.nc1_per_core + per_idx
                grad_offset = nc1_idx * self.grad_each_core
                output_offset = nc1_idx * self.output_each_core
                self.n2one_core(grads_ub, grad_offset, output_offset)

        with self.tik_instance.if_scope(core_idx == self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.nc1_last_core) as last_idx:
                self.dup_zero(grads_ub, num=self.c0)
                nc1_idx = core_idx * self.nc1_per_core + last_idx
                grad_offset = nc1_idx * self.grad_each_core
                output_offset = nc1_idx * self.output_each_core
                self.n2one_core(grads_ub, grad_offset, output_offset)

    def n2one_core(self, grads_ub, grad_offset, output_offset):
        """
        core function
        """
        self.data_move(grads_ub, self.grads_gm, [0, grad_offset], num=self.c0)
        self.data_move(self.output_gm, grads_ub, [output_offset, 0], num=self.c0)

    # 'pylint:disable=too-many-arguments
    def data_move(self, dst, src, offsets, num, src_stride=0, dst_stride=0):
        """
        move data from ub or gm to ub or gm
        """
        dst_offset, src_offset = offsets
        sid = 0
        nburst = 1
        burst_len = (num + self.data_each_block - 1) // self.data_each_block
        self.tik_instance.data_move(dst[dst_offset],
                                    src[src_offset],
                                    sid,
                                    nburst,
                                    burst_len,
                                    src_stride=src_stride,
                                    dst_stride=dst_stride)

    def dup_zero(self, dst, num=0, offset=0):
        """
        dup zero to ub
        """
        dup_value = 0

        loop = num // (self.max_mask * 255)
        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * self.max_mask * 255
                self.tik_instance.vec_dup(self.max_mask, dst[tmp_offset], dup_value, 255, 8)
            offset += loop * self.max_mask * 255

        repeat_time = (num % (self.max_mask * 255)) // self.max_mask
        if repeat_time > 0:
            self.tik_instance.vec_dup(self.max_mask, dst[offset], dup_value, repeat_time, 8)
            offset += repeat_time * self.max_mask

        last_num = num % self.max_mask
        if last_num > 0:
            self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, 8)

    # 'pylint:disable=too-many-arguments,too-many-locals
    def data_add(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        add data
        """
        dst_offset = self.tik_instance.Scalar("int32", name="dst_offset")
        src0_offset = self.tik_instance.Scalar("int32", name="src0_offset")
        src1_offset = self.tik_instance.Scalar("int32", name="src1_offset")
        loop_scalar = self.tik_instance.Scalar("int32", name="loop")
        repeat_scalar = self.tik_instance.Scalar("int32", name="repeat")
        last_num_scalar = self.tik_instance.Scalar("int32", name="last_num")
        dst_offset.set_as(offsets[0])
        src0_offset.set_as(offsets[1])
        src1_offset.set_as(offsets[2])

        loop = num // (self.max_mask * 255)
        loop_scalar.set_as(loop)
        with self.tik_instance.if_scope(loop_scalar > 0):
            with self.tik_instance.for_range(0, loop_scalar) as index:
                tmp_dst_offset = dst_offset + index * self.max_mask * 255
                tmp_src0_offset = src0_offset + index * self.max_mask * 255
                tmp_src1_offset = src1_offset + index * self.max_mask * 255
                self.tik_instance.vec_add(self.max_mask, dst[tmp_dst_offset], src0[tmp_src0_offset],
                                          src1[tmp_src1_offset], 255, dst_stride, src0_stride, src1_stride)

            dst_offset.set_as(dst_offset + loop_scalar * self.max_mask * 255)
            src0_offset.set_as(src0_offset + loop_scalar * self.max_mask * 255)
            src1_offset.set_as(src1_offset + loop_scalar * self.max_mask * 255)

        repeat_time = (num % (self.max_mask * 255)) // self.max_mask
        repeat_scalar.set_as(repeat_time)
        with self.tik_instance.if_scope(repeat_scalar > 0):
            self.tik_instance.vec_add(self.max_mask, dst[dst_offset], src0[src0_offset], src1[src1_offset],
                                      repeat_scalar, dst_stride, src0_stride, src1_stride)
            dst_offset.set_as(dst_offset + repeat_scalar * self.max_mask)
            src0_offset.set_as(src0_offset + repeat_scalar * self.max_mask)
            src1_offset.set_as(src1_offset + repeat_scalar * self.max_mask)

        last_num = num % self.max_mask
        last_num_scalar.set_as(last_num)
        with self.tik_instance.if_scope(last_num_scalar > 0):
            self.tik_instance.vec_add(last_num_scalar, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1,
                                      dst_stride, src0_stride, src1_stride)

    def data_merge(self, grads_ub, w_grad_ub, unit_w, merge_w):
        """
        merge data
        """
        loop = merge_w // unit_w
        tail_num = merge_w % unit_w

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as loop_idx:
                tmp_grads_offset = loop_idx * unit_w * self.c0
                self.data_add(w_grad_ub, w_grad_ub, grads_ub, [0, 0, tmp_grads_offset], num=unit_w * self.c0)
        with self.tik_instance.if_scope(tail_num > 0):
            offset = loop * unit_w * self.c0
            self.data_add(w_grad_ub, w_grad_ub, grads_ub, [0, 0, offset], num=tail_num * self.c0)

    def data_sum(self, src, num, iter_num):
        """
        sum data
        """
        for _ in range(iter_num):
            num = num // 2
            if num // self.MASK > 0:
                mask = self.MASK
                repeat_time = num // self.MASK
            else:
                mask = num
                repeat_time = 1

            src_stride = mask // self.data_each_block
            self.tik_instance.vec_add(mask, src, src[num], src, repeat_time, 0, src_stride, 0)

    def _get_scale(self, scale_scalar, image_size, grad_size):
        """
        :param scale_scalar: scalar of scale
        :param image_size: scalar of image_size
        :param grad_size: scalar of grad_size
        :return:
        """
        image_float = self.tik_instance.Scalar("float32", name="image_float")
        grad_float = self.tik_instance.Scalar("float32", name="grad_float")
        self.tik_instance.scalar_conv("", image_float, image_size)
        self.tik_instance.scalar_conv("", grad_float, grad_size)

        if self.align_corner:
            with self.tik_instance.if_scope(grad_float < 2):
                scale_scalar.set_as(0)
            with self.tik_instance.else_scope():
                scale_scalar.set_as((image_float - 1) / (grad_float - 1))
        else:
            with self.tik_instance.if_scope(grad_float < 2):
                scale_scalar.set_as(0)
            with self.tik_instance.else_scope():
                scale_scalar.set_as(image_float / grad_float)
    
    def _init_tiling_params(self):
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.need_core_num = self.tik_instance.Scalar("int32", name="need_core_num")
        self.nc1_per_core = self.tik_instance.Scalar("int32", name="nc1_per_core")
        self.nc1_last_core = self.tik_instance.Scalar("int32", name="nc1_last_core")
        self.h_per_core = self.tik_instance.Scalar("int32", name="h_per_core")
        self.h_last_core = self.tik_instance.Scalar("int32", name="h_last_core")
        self.grads_h = self.tik_instance.Scalar("int32", name="grads_h")
        self.grads_w = self.tik_instance.Scalar("int32", name="grads_w")
        self.images_h = self.tik_instance.Scalar("int32", name="images_h")
        self.images_w = self.tik_instance.Scalar("int32", name="images_w")
        self.grad_each_core = self.tik_instance.Scalar("int32", name="grad_each_core")
        self.output_each_core = self.tik_instance.Scalar("int32", name="output_eah_core")
        self.grad_move_num = self.tik_instance.Scalar("int32", name="grad_move_num")
        self.output_move_num = self.tik_instance.Scalar("int32", name="output_move_num")
        self.nc1 = self.tik_instance.Scalar("int32", name="nc1")
        self.w_loop = self.tik_instance.Scalar("int32", name="w_loop")
        self.w_tail = self.tik_instance.Scalar("int32", name="w_tail")
        self.core_num_var = self.tik_instance.Scalar("int32", name="core_num_var")
        self.resize_h = self.tik_instance.Scalar("int32", name="resize_h")
        self.resize_w = self.tik_instance.Scalar("int32", name="resize_w")
        self.ori_h = self.tik_instance.Scalar("int32", name="ori_h")
        self.ori_w = self.tik_instance.Scalar("int32", name="ori_w")
        self.src_start_w = self.tik_instance.Scalar("float32", name="src_start_w")
        self.dst_start_w = self.tik_instance.Scalar("float32", name="dst_start_w")
        self.scale_h = self.tik_instance.Scalar("float32", name="scale_h")
        self.scale_w = self.tik_instance.Scalar("float32", name="scale_w")
        

@register_operator("SyncResizeBilinearV2Grad")
# 'pylint: disable=unused-argument,too-many-arguments
def sync_resize_bilinear_v2_grad(grads,
                                 images,
                                 y,
                                 size=None,
                                 ori_image_size=None,
                                 src_start_w=None,
                                 dst_start_w=None,
                                 align_corners=False,
                                 half_pixel_centers=False,
                                 kernel_name="resize_bilinear_v2_grad"):
    """
    algorithm:resize_bilinear_v2_grad
    Operation for resize_bilinear_v2_grad

    Parameters
    ----------
    grads : dict
        dict with keys(range and dtype) of grads
    images : dict
        dict with keys(range and dtype) of images
    y : dict
        dict with keys(range and dtype) of output
    size : list or tuple
        resize size, include H, W, default to None
    ori_image_size: list or tuple
        origin image size before split, include H, W, default to None
    src_start_w: int
        start w of src image, default to None
    dst_start_w: int
        start w of dst image, default to None
    align_corners : bool
        decide how to calculate for scale
    half_pixel_centers : bool
        decide how to calculate for location
    kernel_name : str
        kernel name, default value is "resize_bilinear_v2_grad"

    Returns
    -------
    None
    """
    obj = SyncResizeBilinearV2Grad(grads, images, align_corners, half_pixel_centers, kernel_name)
    instance = obj.compute()
    return instance
