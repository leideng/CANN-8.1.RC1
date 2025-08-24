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
resize_trilinear
"""

from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # input shape indices NDC1HWC0
    N_IDX = 0
    D_IDX = 1
    C1_IDX = 2
    H_IDX = 3
    W_IDX = 4
    # constant parameters in calculation
    VECTOR_MASK_MAX = 64
    BLOCK_NUM_FP32 = 8
    STRIDE_FP16 = 4


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,too-many-statements,too-many-instance-attributes
class ResizeTrilinear():
    """
    Function: use to store ResizeTrilinear base parameters
    """

    class CommonScalar:
        """
        define some scalar
        """
        def __init__(self, tik_instance, inner_dtype):
            self.src_dl = tik_instance.Scalar(dtype="int32", name="src_dl")
            self.src_dr = tik_instance.Scalar(dtype="int32", name="src_dr")
            self.src_hl = tik_instance.Scalar(dtype="int32", name="src_hl")
            self.src_hr = tik_instance.Scalar(dtype="int32", name="src_hr")
            self.src_wl = tik_instance.Scalar(dtype="int32", name="src_wl")
            self.src_wr = tik_instance.Scalar(dtype="int32", name="src_wr")
            self.ratio_dl = tik_instance.Scalar(dtype=inner_dtype, name="ratio_dl")
            self.ratio_dr = tik_instance.Scalar(dtype=inner_dtype, name="ratio_dr")
            self.ratio_hl = tik_instance.Scalar(dtype=inner_dtype, name="ratio_hl")
            self.ratio_hr = tik_instance.Scalar(dtype=inner_dtype, name="ratio_hr")
            self.ratio_wl = tik_instance.Scalar(dtype=inner_dtype, name="ratio_wl")
            self.ratio_wr = tik_instance.Scalar(dtype=inner_dtype, name="ratio_wr")

    def __init__(self, images, size, y, align_corners, half_pixel_centers, kernel_name):
        self.tik_instance = tik.Tik()
        self.kernel_name = kernel_name
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.images_dtype = images.get("dtype").lower()
        self.output_dtype = y.get("dtype").lower()
        self.size_dtype = size.get("dtype").lower()
        self.block_byte_size = 32
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers

        self.images_shape = images.get("shape")
        self.n = self.images_shape[Constant.N_IDX]
        self.d_in = self.images_shape[Constant.D_IDX]
        self.c1 = self.images_shape[Constant.C1_IDX]
        self.h_in = self.images_shape[Constant.H_IDX]
        self.w_in = self.images_shape[Constant.W_IDX]
        self.c0 = 16

        self.size_shape = size.get("shape")
        self.y_shape = y.get("shape")
        self.d_out = self.y_shape[Constant.D_IDX]
        self.h_out = self.y_shape[Constant.H_IDX]
        self.w_out = self.y_shape[Constant.W_IDX]

        self.inner_dtype = "float32"
        self.nd_num = self.n * self.d_out
        self.nd_per_core = self.ceil_div(self.nd_num, self.core_num)
        self.need_core_num = self.ceil_div(self.nd_num, self.nd_per_core)
        self.last_num = self.nd_num - self.nd_per_core * (self.need_core_num - 1)

        self.enlarge_factor = self.w_out // self.w_in
        self.extra_counts = self.enlarge_factor // 2
        self.repeat_stride = (self.enlarge_factor - 1) * self.c0 // Constant.BLOCK_NUM_FP32
        self.scale_d, self.scale_h, self.scale_w = self.get_scale_dhw()

        # check dtype
        para_check.check_dtype(self.size_dtype, ("int32",), param_name="size")
        para_check.check_dtype(self.images_dtype, ("float32", "float16"), param_name="images")

        # request gm
        self.images_gm = self.tik_instance.Tensor(self.images_dtype, self.images_shape,
                                                  name="images_gm", scope=tik.scope_gm)
        self.size_gm = self.tik_instance.Tensor(self.size_dtype, self.size_shape,
                                                name="size_gm", scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.output_dtype, self.y_shape,
                                               name="out_gm", scope=tik.scope_gm)
        # assist gm for dst indices 0, 1, 2, ..., 511
        max_dim_value = list(range(512))
        self.dst_idx_gm = self.tik_instance.Tensor(self.inner_dtype, [512],
                                                   name="dst_idx_gm", scope=tik.scope_gm, init_value=max_dim_value)

        # init ub tensor
        self.ub_d_src_idx_0 = None
        self.ub_h_src_idx_0 = None
        self.ub_d_src_ratio_0 = None
        self.ub_d_src_ratio_1 = None
        self.ub_h_src_ratio_0 = None
        self.ub_h_src_ratio_1 = None
        self.ub_extra_src_idx = None
        self.ub_extra_dst_idx = None
        self.ub_w_all_ratio_0 = None
        self.ub_w_all_ratio_1 = None
        self.ub_0 = None
        self.ub_1 = None
        self.ub_2 = None
        self.ub_3 = None
        self.ub_out = None

    @staticmethod
    def ceil_div(dividend, divisor):
        """
        calculate ceil division
        """
        result = (dividend + divisor - 1) // divisor
        return result

    @staticmethod
    def get_dtype_size(dtype):
        """
        :param dtype: data type
        :return:
        """
        dtype_byte_size = get_bit_len(dtype) // 8
        return dtype_byte_size

    def get_scale_dhw(self):
        scale_d = self.get_scale(self.d_in, self.d_out)
        scale_h = self.get_scale(self.h_in, self.h_out)
        scale_w = self.get_scale(self.w_in, self.w_out)
        return scale_d, scale_h, scale_w

    def get_scale(self, in_size, out_size):
        """
        :param in_size: scalar of in_float
        :param out_size: scalar of out_float
        :return:
        """
        if self.align_corners:
            return (in_size - 1) / (out_size - 1)
        else:
            return in_size / out_size

    def get_src_idx_ratio(self, scale, dim_out, dim_in, ub_dst_idx, ub_src_idx0, ub_src_ratio0, ub_src_ratio1):
        """
        calculate source indices and ratio in one dimension
        :param scale: scalar of scale
        :param dim_out: scalar of output dim
        :param dim_in: scalar of input dim
        :param ub_dst_idx: ub tensor float32 (dim_out,) of dst indices (0, 1, 2, ...)
        :return:
        :param ub_src_idx0: ub tensor int32 (dim_out,) of left src indices
        :param ub_src_ratio0: ub tensor float32 (dim_out,) of left src ratio
        :param ub_src_ratio1: ub tensor float32 (dim_out,) of right src ratio
        """
        with self.tik_instance.new_stmt_scope():
            # ub of zeros and max values
            ub_zero = self.tik_instance.Tensor(self.inner_dtype, [dim_out], scope=tik.scope_ubuf, name="ub_zero")
            ub_one = self.tik_instance.Tensor(self.inner_dtype, [dim_out], scope=tik.scope_ubuf, name="ub_one")
            ub_max = self.tik_instance.Tensor(self.inner_dtype, [dim_out], scope=tik.scope_ubuf, name="ub_max")
            ub_second_max = self.tik_instance.Tensor(self.inner_dtype, [dim_out],
                                                     scope=tik.scope_ubuf, name="ub_second_max")
            self.dup_value(ub_zero, dim_out, 0)
            self.dup_value(ub_one, dim_out, 1)
            self.dup_value(ub_max, dim_out, dim_in - 1)
            self.dup_value(ub_second_max, dim_out, dim_in - 2)

            # calculate src_idx_fp
            ub_src_idx_fp = self.tik_instance.Tensor(self.inner_dtype, [dim_out],
                                                     scope=tik.scope_ubuf, name="ub_src_idx_fp")
            tmp_ub_fp = self.tik_instance.Tensor(self.inner_dtype, [dim_out],
                                                 scope=tik.scope_ubuf, name="tmp_ub_fp")

            if self.half_pixel_centers:
                self.data_adds(tmp_ub_fp, ub_dst_idx, 0.5, [0, 0], num=dim_out)
                self.data_muls(tmp_ub_fp, tmp_ub_fp, scale, [0, 0], num=dim_out)
                self.data_adds(ub_src_idx_fp, tmp_ub_fp, -0.5, [0, 0], num=dim_out)
            else:
                self.data_muls(ub_src_idx_fp, ub_dst_idx, scale, [0, 0], num=dim_out)

            self.data_max(ub_src_idx_fp, ub_src_idx_fp, ub_zero, [0, 0, 0], num=dim_out)
            self.data_min(ub_src_idx_fp, ub_src_idx_fp, ub_max, [0, 0, 0], num=dim_out)

            # calculate src_idx0 and src_ratio1
            self.data_min(tmp_ub_fp, ub_src_idx_fp, ub_second_max, [0, 0, 0], num=dim_out)
            self.data_conv(ub_src_idx0, tmp_ub_fp, [0, 0], num=dim_out, mode="floor")
            self.data_conv(tmp_ub_fp, ub_src_idx0, [0, 0], num=dim_out)
            self.data_sub(ub_src_ratio1, ub_src_idx_fp, tmp_ub_fp, [0, 0, 0], num=dim_out)

            # calculate src_ratio0
            self.data_sub(ub_src_ratio0, ub_one, ub_src_ratio1, [0, 0, 0], num=dim_out)

    def get_all_w_ratio(self, scalar, ub_w_ratio, ub_w_all_ratio):
        """
        duplicate w ratio c0 times
        :param scalar: scalar
        :param ub_w_ratio: ub tensor float32 (dim_out, c0) of w ratio
        :return:
        :param ub_w_all_ratio: ub tensor float32 (dim_out,) of w ratio
        """
        with self.tik_instance.for_range(0, self.w_out) as w_idx:
            scalar.set_as(ub_w_ratio[w_idx])
            self.dup_value(ub_w_all_ratio, self.c0, scalar, offset=w_idx * self.c0)

    def get_extra_w_idx(self, ub_w_src_idx, enlarge_factor, ub_extra_src_idx, ub_extra_dst_idx):
        """
        get the w src and dst indices with extra counts
        :param ub_w_src_idx: ub tensor int32 (dim_out,) of w src idx
        :param enlarge_factor: scalar of ratio of w_out / w_in
        :return:
        :param ub_extra_src_idx: ub tensor int32 (dim_in,) of w src idx with extra counts
        :param ub_extra_dst_idx: ub tensor int32 (dim_in,) of w dst idx with extra counts
        """
        with self.tik_instance.new_stmt_scope():
            src_idx = self.tik_instance.Scalar(dtype="int32", name="src_idx")
            idx_counts = self.tik_instance.Scalar(dtype="int32", name="idx_counts")
            extra_idx_num = self.tik_instance.Scalar(dtype="int32", name="extra_idx_num", init_value=0)
            ub_w_cnt = self.tik_instance.Tensor("int32", [self.w_in], scope=tik.scope_ubuf, name="ub_w_cnt")
            self.dup_value(ub_w_cnt, self.w_in, 0)

            with self.tik_instance.for_range(0, self.w_out) as w_idx:
                src_idx.set_as(ub_w_src_idx[w_idx])
                idx_counts.set_as(ub_w_cnt[src_idx])
                with self.tik_instance.if_scope(idx_counts == enlarge_factor):
                    ub_extra_src_idx[extra_idx_num + 1].set_as(src_idx)
                    ub_extra_dst_idx[extra_idx_num + 1].set_as(w_idx)
                    extra_idx_num.set_as(extra_idx_num + 1)
                ub_w_cnt[src_idx].set_as(idx_counts + 1)
            idx_counts.set_as(ub_w_cnt[0])

            # store scalar idx_counts & extra_idx_num at the start
            ub_extra_src_idx[0].set_as(idx_counts)
            ub_extra_dst_idx[0].set_as(extra_idx_num)

    def compute_trilinear(self):
        """
        resize with trilinear interpolation
        w_in <= 512, w_out <= 512
        """
        with self.tik_instance.for_range(0, self.need_core_num, block_num=self.need_core_num) as core_idx:
            with self.tik_instance.if_scope(core_idx != self.need_core_num - 1):
                self.compute_trilinear_core(core_idx, self.nd_per_core)
            with self.tik_instance.else_scope():
                self.compute_trilinear_core(core_idx, self.last_num)

    def compute_trilinear_core(self, core_idx, nd_num):
        """
        core function
        """
        self.init_ub_tensor()
        scalar = self.CommonScalar(self.tik_instance, self.inner_dtype)
        n_idx = self.tik_instance.Scalar("int32")
        d_idx = self.tik_instance.Scalar("int32")

        idx_counts = self.tik_instance.Scalar(dtype="int32", name="idx_counts")
        extra_idx_num = self.tik_instance.Scalar(dtype="int32", name="extra_idx_num")
        idx_counts.set_as(self.ub_extra_src_idx[0])
        extra_idx_num.set_as(self.ub_extra_dst_idx[0])

        dst_idx = self.tik_instance.Scalar("int32")
        src_idx = self.tik_instance.Scalar("int32")
        dst_last_idx = self.tik_instance.Scalar("int32")
        src_last_idx = self.tik_instance.Scalar("int32")

        # offsets calculation for two modes
        if self.half_pixel_centers:
            offset_dst_start = self.cal_offset(0, 0, 0, 0, self.extra_counts)
            offset_dst_end = self.cal_offset(0, 0, 0, 0, self.w_out - self.extra_counts)
            offset_src_end = self.cal_offset(0, 0, 0, 0, self.w_in - 2, src=True)
        else:
            offset_src_extra = self.cal_offset(0, 0, 0, 0, 0, src=True)

        with self.tik_instance.for_range(0, nd_num) as idx:
            nd_idx = core_idx * self.nd_per_core + idx
            n_idx.set_as(nd_idx // self.d_out)
            d_idx.set_as(nd_idx % self.d_out)

            # src d
            scalar.src_dl.set_as(self.ub_d_src_idx_0[d_idx])
            scalar.src_dr.set_as(scalar.src_dl + 1)
            scalar.ratio_dl.set_as(self.ub_d_src_ratio_0[d_idx])
            scalar.ratio_dr.set_as(self.ub_d_src_ratio_1[d_idx])

            with self.tik_instance.for_range(0, self.c1) as c1_idx:
                with self.tik_instance.for_range(0, self.h_out) as h_idx:
                    # src h
                    scalar.src_hl.set_as(self.ub_h_src_idx_0[h_idx])
                    scalar.src_hr.set_as(scalar.src_hl + 1)
                    scalar.ratio_hl.set_as(self.ub_h_src_ratio_0[h_idx])
                    scalar.ratio_hr.set_as(self.ub_h_src_ratio_1[h_idx])

                    gm_bottom_left_input_offset = \
                        self.cal_offset(n_idx, scalar.src_dl, c1_idx, scalar.src_hl, 0, src=True)
                    gm_bottom_right_input_offset = \
                        self.cal_offset(n_idx, scalar.src_dr, c1_idx, scalar.src_hl, 0, src=True)
                    gm_top_left_input_offset = \
                        self.cal_offset(n_idx, scalar.src_dl, c1_idx, scalar.src_hr, 0, src=True)
                    gm_top_right_input_offset = \
                        self.cal_offset(n_idx, scalar.src_dr, c1_idx, scalar.src_hr, 0, src=True)

                    with self.tik_instance.if_scope(self.images_dtype in ("float16",)):
                        ub_cast_0 = self.tik_instance.Tensor(self.images_dtype, (self.w_in, self.c0),
                                                             scope=tik.scope_ubuf, name="ub_cast_0")
                        ub_cast_1 = self.tik_instance.Tensor(self.images_dtype, (self.w_in, self.c0),
                                                             scope=tik.scope_ubuf, name="ub_cast_1")

                        # move bottom [w,c0] to ub
                        self.data_move(ub_cast_0, self.images_gm, [0, gm_bottom_left_input_offset],
                                       num=self.w_in * self.c0)
                        self.data_move(ub_cast_1, self.images_gm, [0, gm_bottom_right_input_offset],
                                       num=self.w_in * self.c0)
                        self.data_conv(self.ub_0, ub_cast_0, [0, 0], num=self.w_in * self.c0,
                                       src_stride=Constant.STRIDE_FP16)
                        self.data_conv(self.ub_1, ub_cast_1, [0, 0], num=self.w_in * self.c0,
                                       src_stride=Constant.STRIDE_FP16)

                        # modify by ratio of d-h plane
                        self.data_muls(self.ub_0, self.ub_0, scalar.ratio_dl * scalar.ratio_hl, [0, 0],
                                       num=self.w_in * self.c0)
                        self.data_muls(self.ub_1, self.ub_1, scalar.ratio_dr * scalar.ratio_hl, [0, 0],
                                       num=self.w_in * self.c0)
                        self.data_add(self.ub_0, self.ub_0, self.ub_1, [0, 0, 0], num=self.w_in * self.c0)

                        # move top [w,c0] to ub
                        self.data_move(ub_cast_0, self.images_gm, [0, gm_top_left_input_offset],
                                       num=self.w_in * self.c0)
                        self.data_move(ub_cast_1, self.images_gm, [0, gm_top_right_input_offset],
                                       num=self.w_in * self.c0)
                        self.data_conv(self.ub_2, ub_cast_0, [0, 0], num=self.w_in * self.c0,
                                       src_stride=Constant.STRIDE_FP16)
                        self.data_conv(self.ub_3, ub_cast_1, [0, 0], num=self.w_in * self.c0,
                                       src_stride=Constant.STRIDE_FP16)

                        # modify by ratio of d-h plane
                        self.data_muls(self.ub_2, self.ub_2, scalar.ratio_dl * scalar.ratio_hr, [0, 0],
                                       num=self.w_in * self.c0)
                        self.data_muls(self.ub_3, self.ub_3, scalar.ratio_dr * scalar.ratio_hr, [0, 0],
                                       num=self.w_in * self.c0)
                        self.data_add(self.ub_1, self.ub_2, self.ub_3, [0, 0, 0], num=self.w_in * self.c0)
                        self.data_add(self.ub_0, self.ub_0, self.ub_1, [0, 0, 0], num=self.w_in * self.c0)

                    with self.tik_instance.else_scope():
                        # move bottom [w,c0] to ub
                        self.data_move(self.ub_0, self.images_gm, [0, gm_bottom_left_input_offset],
                                       num=self.w_in * self.c0)
                        self.data_move(self.ub_1, self.images_gm, [0, gm_bottom_right_input_offset],
                                       num=self.w_in * self.c0)

                        # modify by ratio of d-h plane
                        self.data_muls(self.ub_0, self.ub_0, scalar.ratio_dl * scalar.ratio_hl, [0, 0],
                                       num=self.w_in * self.c0)
                        self.data_muls(self.ub_1, self.ub_1, scalar.ratio_dr * scalar.ratio_hl, [0, 0],
                                       num=self.w_in * self.c0)
                        self.data_add(self.ub_0, self.ub_0, self.ub_1, [0, 0, 0], num=self.w_in * self.c0)

                        # move top [w,c0] to ub
                        self.data_move(self.ub_2, self.images_gm, [0, gm_top_left_input_offset],
                                       num=self.w_in * self.c0)
                        self.data_move(self.ub_3, self.images_gm, [0, gm_top_right_input_offset],
                                       num=self.w_in * self.c0)

                        # modify by ratio of d-h plane
                        self.data_muls(self.ub_2, self.ub_2, scalar.ratio_dl * scalar.ratio_hr, [0, 0],
                                       num=self.w_in * self.c0)
                        self.data_muls(self.ub_3, self.ub_3, scalar.ratio_dr * scalar.ratio_hr, [0, 0],
                                       num=self.w_in * self.c0)
                        self.data_add(self.ub_1, self.ub_2, self.ub_3, [0, 0, 0], num=self.w_in * self.c0)
                        self.data_add(self.ub_0, self.ub_0, self.ub_1, [0, 0, 0], num=self.w_in * self.c0)

                    # mode: half pixel centers
                    if self.half_pixel_centers:
                        with self.tik_instance.for_range(0, self.enlarge_factor) as repeat_time:
                            self.data_move(self.ub_2, self.ub_0, [offset_dst_start + repeat_time * self.c0, 0],
                                           num=self.c0, nburst=self.w_in - 1, dst_stride=self.repeat_stride)
                            self.data_move(self.ub_3, self.ub_0, [offset_dst_start + repeat_time * self.c0, self.c0],
                                           num=self.c0, nburst=self.w_in - 1, dst_stride=self.repeat_stride)

                        with self.tik_instance.for_range(0, self.extra_counts) as repeat_time:
                            # start
                            self.data_move(self.ub_2, self.ub_0, [repeat_time * self.c0, 0],
                                           num=self.c0, dst_stride=self.repeat_stride)
                            self.data_move(self.ub_3, self.ub_0, [repeat_time * self.c0, self.c0],
                                           num=self.c0, dst_stride=self.repeat_stride)
                            # end
                            self.data_move(self.ub_2, self.ub_0,
                                           [offset_dst_end + repeat_time * self.c0, offset_src_end],
                                           num=self.c0, dst_stride=self.repeat_stride)
                            self.data_move(self.ub_3, self.ub_0,
                                           [offset_dst_end + repeat_time * self.c0, offset_src_end + self.c0],
                                           num=self.c0, dst_stride=self.repeat_stride)

                    # mode: align corners
                    else:
                        # start
                        with self.tik_instance.for_range(0, idx_counts) as repeat_time:
                            self.data_move(self.ub_2, self.ub_0, [repeat_time * self.c0, offset_src_extra],
                                           num=self.c0, dst_stride=self.repeat_stride)
                            self.data_move(self.ub_3, self.ub_0, [repeat_time * self.c0, offset_src_extra + self.c0],
                                           num=self.c0, dst_stride=self.repeat_stride)

                        # extra counts
                        with self.tik_instance.for_range(2, extra_idx_num + 1) as idx:
                            src_idx.set_as(self.ub_extra_src_idx[idx])
                            dst_idx.set_as(self.ub_extra_dst_idx[idx])
                            src_last_idx.set_as(self.ub_extra_src_idx[idx - 1])
                            dst_last_idx.set_as(self.ub_extra_dst_idx[idx - 1])
                            offset_src_last = self.cal_offset(0, 0, 0, 0, src_last_idx + 1, src=True)
                            offset_src_extra = self.cal_offset(0, 0, 0, 0, src_idx, src=True)
                            # repeat part
                            with self.tik_instance.for_range(0, self.enlarge_factor) as repeat_time:
                                self.data_move(self.ub_2, self.ub_0,
                                               [(dst_last_idx + 1 + repeat_time) * self.c0, offset_src_last],
                                               num=self.c0, nburst=src_idx - src_last_idx,
                                               dst_stride=self.repeat_stride)
                                self.data_move(self.ub_3, self.ub_0,
                                               [(dst_last_idx + 1 + repeat_time) * self.c0, offset_src_last + self.c0],
                                               num=self.c0, nburst=src_idx - src_last_idx,
                                               dst_stride=self.repeat_stride)
                            # extra single point
                            self.data_move(self.ub_2, self.ub_0, [dst_idx * self.c0, offset_src_extra],
                                           num=self.c0, dst_stride=self.repeat_stride)
                            self.data_move(self.ub_3, self.ub_0, [dst_idx * self.c0, offset_src_extra + self.c0],
                                           num=self.c0, dst_stride=self.repeat_stride)

                    # linear interpolation
                    self.data_mul(self.ub_2, self.ub_2, self.ub_w_all_ratio_0, [0, 0, 0], num=self.w_out * self.c0)
                    self.data_mul(self.ub_3, self.ub_3, self.ub_w_all_ratio_1, [0, 0, 0], num=self.w_out * self.c0)
                    with self.tik_instance.if_scope(self.images_dtype in ("float16",)):
                        ub_cast = self.tik_instance.Tensor(self.inner_dtype, (self.w_out, self.c0),
                                                           scope=tik.scope_ubuf, name="ub_cast")
                        self.data_add(ub_cast, self.ub_2, self.ub_3, [0, 0, 0], num=self.w_out * self.c0)
                        self.data_conv(self.ub_out, ub_cast, [0, 0], num=self.w_out * self.c0,
                                       dst_stride=Constant.STRIDE_FP16)
                    with self.tik_instance.else_scope():
                        self.data_add(self.ub_out, self.ub_2, self.ub_3, [0, 0, 0], num=self.w_out * self.c0)
                    gm_output_offset = self.cal_offset(n_idx, d_idx, c1_idx, h_idx, 0)
                    self.data_move(self.out_gm, self.ub_out, [gm_output_offset, 0], num=self.w_out * self.c0)

    def init_ub_tensor(self):
        """
        init tensor
        """
        self.ub_d_src_idx_0 = self.tik_instance.Tensor("int32", [self.d_out],
                                                       scope=tik.scope_ubuf, name="ub_d_src_idx_0")
        self.ub_h_src_idx_0 = self.tik_instance.Tensor("int32", [self.h_out],
                                                       scope=tik.scope_ubuf, name="ub_h_src_idx_0")
        self.ub_d_src_ratio_0 = self.tik_instance.Tensor(self.inner_dtype, [self.d_out],
                                                         scope=tik.scope_ubuf, name="ub_d_src_ratio_0")
        self.ub_d_src_ratio_1 = self.tik_instance.Tensor(self.inner_dtype, [self.d_out],
                                                         scope=tik.scope_ubuf, name="ub_d_src_ratio_1")
        self.ub_h_src_ratio_0 = self.tik_instance.Tensor(self.inner_dtype, [self.h_out],
                                                         scope=tik.scope_ubuf, name="ub_h_src_ratio_0")
        self.ub_h_src_ratio_1 = self.tik_instance.Tensor(self.inner_dtype, [self.h_out],
                                                         scope=tik.scope_ubuf, name="ub_h_src_ratio_1")

        self.ub_extra_src_idx = self.tik_instance.Tensor("int32", [self.w_in],
                                                         scope=tik.scope_ubuf, name="ub_extra_src_idx")
        self.ub_extra_dst_idx = self.tik_instance.Tensor("int32", [self.w_in],
                                                         scope=tik.scope_ubuf, name="ub_extra_dst_idx")
        self.ub_w_all_ratio_0 = self.tik_instance.Tensor(self.inner_dtype, (self.w_out, self.c0),
                                                         scope=tik.scope_ubuf, name="ub_w_src_ratio_0")
        self.ub_w_all_ratio_1 = self.tik_instance.Tensor(self.inner_dtype, (self.w_out, self.c0),
                                                         scope=tik.scope_ubuf, name="ub_w_src_ratio_1")

        with self.tik_instance.new_stmt_scope():
            ub_d_dst_idx = self.tik_instance.Tensor(self.inner_dtype, [self.d_out],
                                                    scope=tik.scope_ubuf, name="ub_d_dst_idx")
            ub_h_dst_idx = self.tik_instance.Tensor(self.inner_dtype, [self.h_out],
                                                    scope=tik.scope_ubuf, name="ub_h_dst_idx")
            ub_w_dst_idx = self.tik_instance.Tensor(self.inner_dtype, [self.w_out],
                                                    scope=tik.scope_ubuf, name="ub_w_dst_idx")

            self.data_move(ub_d_dst_idx, self.dst_idx_gm, [0, 0], num=self.d_out)
            self.data_move(ub_h_dst_idx, self.dst_idx_gm, [0, 0], num=self.h_out)
            self.data_move(ub_w_dst_idx, self.dst_idx_gm, [0, 0], num=self.w_out)

            ub_w_src_idx_0 = self.tik_instance.Tensor("int32", [self.w_out],
                                                      scope=tik.scope_ubuf, name="ub_w_src_idx_0")
            ub_w_src_ratio_0 = self.tik_instance.Tensor(self.inner_dtype, [self.w_out],
                                                        scope=tik.scope_ubuf, name="ub_w_src_ratio_0")
            ub_w_src_ratio_1 = self.tik_instance.Tensor(self.inner_dtype, [self.w_out],
                                                        scope=tik.scope_ubuf, name="ub_w_src_ratio_1")
        
            self.get_src_idx_ratio(self.scale_d, self.d_out, self.d_in, ub_d_dst_idx,
                                   self.ub_d_src_idx_0, self.ub_d_src_ratio_0, self.ub_d_src_ratio_1)
            self.get_src_idx_ratio(self.scale_h, self.h_out, self.h_in, ub_h_dst_idx,
                                   self.ub_h_src_idx_0, self.ub_h_src_ratio_0, self.ub_h_src_ratio_1)
            self.get_src_idx_ratio(self.scale_w, self.w_out, self.w_in, ub_w_dst_idx,
                                   ub_w_src_idx_0, ub_w_src_ratio_0, ub_w_src_ratio_1)

            scalar = self.CommonScalar(self.tik_instance, self.inner_dtype)
            self.get_all_w_ratio(scalar.ratio_wl, ub_w_src_ratio_0, self.ub_w_all_ratio_0)
            self.get_all_w_ratio(scalar.ratio_wr, ub_w_src_ratio_1, self.ub_w_all_ratio_1)
            self.get_extra_w_idx(ub_w_src_idx_0, self.enlarge_factor, self.ub_extra_src_idx, self.ub_extra_dst_idx)

        self.ub_0 = self.tik_instance.Tensor(self.inner_dtype, (self.w_in, self.c0),
                                             scope=tik.scope_ubuf, name="ub_0")
        self.ub_1 = self.tik_instance.Tensor(self.inner_dtype, (self.w_in, self.c0),
                                             scope=tik.scope_ubuf, name="ub_1")
        self.ub_2 = self.tik_instance.Tensor(self.inner_dtype, (self.w_out, self.c0),
                                             scope=tik.scope_ubuf, name="ub_2")
        self.ub_3 = self.tik_instance.Tensor(self.inner_dtype, (self.w_out, self.c0),
                                             scope=tik.scope_ubuf, name="ub_3")
        self.ub_out = self.tik_instance.Tensor(self.images_dtype, (self.w_out, self.c0),
                                               scope=tik.scope_ubuf, name="ub_out")

    def compute(self):
        """
        op compute
        """
        self.compute_trilinear()

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.images_gm, self.size_gm],
                                   outputs=[self.out_gm],
                                   config=opt_config)
        return self.tik_instance

    def data_move(self, dst, src, offsets, num, nburst=1, src_stride=0, dst_stride=0):
        """
        move data from ub or gm to ub or gm
        """
        dst_offset, src_offset = offsets
        sid = 0
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        burst_len = (num + data_each_block - 1) // data_each_block
        self.tik_instance.data_move(dst[dst_offset],
                                    src[src_offset],
                                    sid,
                                    nburst,
                                    burst_len,
                                    src_stride=src_stride,
                                    dst_stride=dst_stride)

    def cal_offset(self, n, d, c1, h, w, src=False):
        """
        calculate buffer offset
        """
        if src:
            one_d = self.d_in * self.c1 * self.h_in * self.w_in * self.c0
            one_c1 = self.c1 * self.h_in * self.w_in * self.c0
            one_h = self.h_in * self.w_in * self.c0
            one_w = self.w_in * self.c0
            return n * one_d + d * one_c1 + c1 * one_h + h * one_w + w * self.c0
        else:
            one_d = self.d_out * self.c1 * self.h_out * self.w_out * self.c0
            one_c1 = self.c1 * self.h_out * self.w_out * self.c0
            one_h = self.h_out * self.w_out * self.c0
            one_w = self.w_out * self.c0
            return n * one_d + d * one_c1 + c1 * one_h + h * one_w + w * self.c0

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
        vector_mask_max = Constant.VECTOR_MASK_MAX
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

    def double_operator_template(self, op_obj, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8,
                                 src1_stride=8):
        """
        tik api template
        """
        vector_mask_max = Constant.VECTOR_MASK_MAX
        dst_offset, src0_offset, src1_offset = offsets

        tensor_size = num if num else src1.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src0_offset = src0_offset + index * vector_mask_max * 255
                tmp_src1_offset = src1_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src0[tmp_src0_offset], src1[tmp_src1_offset], 255,
                       dst_stride, src0_stride, src1_stride)

            dst_offset += loop * vector_mask_max * 255
            src0_offset += loop * vector_mask_max * 255
            src1_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            op_obj(vector_mask_max, dst[dst_offset], src0[src0_offset], src1[src1_offset], repeat_time, dst_stride,
                   src0_stride, src1_stride)
            dst_offset += repeat_time * vector_mask_max
            src0_offset += repeat_time * vector_mask_max
            src1_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            op_obj(last_num, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1, dst_stride, src0_stride,
                   src1_stride)

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

    def data_add(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik add
        """
        self.double_operator_template(self.tik_instance.vec_add, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_sub(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik sub
        """
        self.double_operator_template(self.tik_instance.vec_sub, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_mul(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik mul
        """
        self.double_operator_template(self.tik_instance.vec_mul, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_max(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik max
        """
        self.double_operator_template(self.tik_instance.vec_max, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_min(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik min
        """
        self.double_operator_template(self.tik_instance.vec_min, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_conv(self, dst, src, offsets, mode="", num=0, dst_stride=8, src_stride=8):
        """
        tik conv
        """
        vector_mask_max = Constant.VECTOR_MASK_MAX
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


def check_attrs(align_corners, half_pixel_centers, kernel_name):
    """
    check attrs
    """
    if align_corners & half_pixel_centers:
        error_manager_vector.raise_err_check_params_rules(
            kernel_name, "mode is either align_corners or half_pixel_centers",
            "align_corners & half_pixel_centers", "True")


def check_shape(images, y, kernel_name):
    """
    check shape
    """
    input_shape = images.get("shape")
    output_shape = y.get("shape")
    input_w = input_shape[Constant.W_IDX]
    output_w = output_shape[Constant.W_IDX]
    if (input_w > 512) or (output_w > 512):
        error_manager_vector.raise_err_input_shape_invalid(
            kernel_name, "input or output w", "large shape over 512 not supported")


@register_operator("ResizeTrilinear")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def resize_trilinear(images,
                     size,
                     y,
                     align_corners=False,
                     half_pixel_centers=False,
                     kernel_name="resize_trilinear"):
    """Resize `images` to `size` using trilinear interpolation.

    Parameters
    ----------
    images: dict
        the dict of input, include shape of input_tensor which layout
        only support 6HD and dtype supports 'float16', 'float32'
    size: dict
        the dict of input, the depth, height and width of output tensor
        only support ND and dtype supports 'int32'
    y: dict
        the dict of output, include shape of input_tensor which layout
        only support 6HD and dtype supports 'float16', 'float32'
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether half_pixel_centers
    kernel_name: str
        cce kernel name, default value is `resize_trilinear`

    Returns
    -------
    tik_instance
    """
    check_attrs(align_corners, half_pixel_centers, kernel_name)
    check_shape(images, y, kernel_name)
    obj = ResizeTrilinear(images, size, y, align_corners, half_pixel_centers, kernel_name)
    return obj.compute()
