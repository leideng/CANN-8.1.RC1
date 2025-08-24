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
resize_grad_d
"""

import math
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    UB_RESERVE_SIZE = 1024
    MAX_INT64_VALUE = 2 ** 64 - 1
    MAX_INT32_VALUE = 2 ** 32 - 1
    EIGHT_BIT = 8
    ONE_BLOCK_SIZE = 32
    TILING_ARG_NUM = 17
    TILING_SCALAR_DTYPE = "int32"
    NUM_EACH_BURST = 8
    TILING_MODE_1 = 1
    TILING_MODE_0 = 0
    CUBIC_COEFF = -0.75
    BLOCK_NUM_FLOAT = 8


def get_ceil_int(int1, int2):
    ceil_int = (int1 + int2 - 1) // int2
    return ceil_int


class UpSampleBicubic2dBackward:
    def __init__(self, grads, original_size, scales, coordinate_transformation_mode,
                 cubic_coeff_a, kernel_name="resize_grad_d"):
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.dtype_grads = grads.get("dtype")
        self.output_grads_gm = self.tik_instance.Tensor(
            self.dtype_grads, (Constant.MAX_INT64_VALUE,), name="output_grads_gm", scope=tik.scope_gm)
        self.input_grads_gm = self.tik_instance.Tensor(
            self.dtype_grads, (Constant.MAX_INT64_VALUE,), name="input_grads_gm", 
            scope=tik.scope_gm, is_atomic_add=True)

        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_SCALAR_DTYPE, (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)

        self.ub_size_bytes = tbe_platform.get_soc_spec("UB_SIZE")
        
        if self.dtype_grads == "float32":
            self.ub_tensor_size = int((self.ub_size_bytes - Constant.UB_RESERVE_SIZE) / 8 / 8) * 8
        elif self.dtype_grads == "float16":
            self.ub_tensor_size = int((self.ub_size_bytes - Constant.UB_RESERVE_SIZE) / 10 / 16) * 16
            
        self.dtype_bytes_size = tbe_platform.get_bit_len(self.dtype_grads) // 8
        self.dtype_bytes_size_fp32 = 4
        self.block_num = Constant.ONE_BLOCK_SIZE / self.dtype_bytes_size
        self.block_num_fp32 = Constant.BLOCK_NUM_FLOAT

        self.scales = scales
        if scales is not None:
            if len(scales) == 1:
                self.scales_h, self.scales_w = scales[0], scales[0]
            else:
                self.scales_h, self.scales_w = scales[0], scales[1]
        else:
            self.scales_h, self.scales_w = 0, 0

        self.align_corners = (coordinate_transformation_mode == "align_corners")
        self.cubic_coeff_a = Constant.CUBIC_COEFF
        self.tiling_cubic()
        if (self.dtype_grads == "float16"):
            self.fp16_ub_init()

        self.output_grad_ub = self.tik_instance.Tensor("float32", (self.ub_tensor_size + 8,), 
                                    name="output_grads_ub", scope=tik.scope_ubuf)
        self.output_grad_temp_ub = self.tik_instance.Tensor("float32", (self.ub_tensor_size + 8,), 
                                    name="output_grad_temp_ub", scope=tik.scope_ubuf)
        
        if self.dtype_grads == "float16":
            self.input_grads_gm_workspace = self.tik_instance.Tensor(
                "float32", (self.fp16_whole,), name="input_grads_gm_workspace", 
                scope=tik.scope_gm, is_workspace=True, is_atomic_add=True) 
            
            self.sync_workspace = self.tik_instance.Tensor(
                "int64", (Constant.MAX_INT32_VALUE, ), name="barrier_workspace", scope=tik.scope_gm, 
                is_workspace=True, is_atomic_add=True)
            self.atomic_add_space = self.input_grads_gm_workspace
        else:
            self.atomic_add_space = self.input_grads_gm
    
    # 'pylint: disable=too-many-arguments 
    # 'pylint: disable=too-many-locals 
    @staticmethod    
    def area_pixel_compute_source_index(scale, dst_index, align_corners):
        if align_corners:
            return scale * dst_index * 1.0
        return scale * (dst_index + 0.5) - 0.5

    def area_pixel_compute_scale(self, hw_scale, input_size, output_size, align_corners, scale=None):
        tmp1 = self.tik_instance.Scalar("float32", name="tmp1")
        tmp2 = self.tik_instance.Scalar("float32", name="tmp2")
        self.tik_instance.scalar_conv('', tmp1, input_size)
        self.tik_instance.scalar_conv('', tmp2, output_size)

        if align_corners:
            with self.tik_instance.if_scope(output_size > 1):
                hw_scale.set_as((tmp1 - 1.0) / (tmp2 - 1.0))
            with self.tik_instance.else_scope():
                hw_scale.set_as(0.0)
        else:
            with self.tik_instance.if_scope(scale > 0.0):
                hw_scale.set_as(1.0 / scale)
            with self.tik_instance.else_scope():
                hw_scale.set_as(tmp1 / tmp2)

    def fp16_ub_init(self):
        self.ub_max_size = math.floor(int(self.ub_tensor_size + 8) / 64)
        self.fp16_whole = self.tik_instance.Scalar(dtype="int32", name="input_w")
        self.fp16_whole.set_as(self.batch * self.channel * self.input_h * self.input_w)
        self.fp16_whole = 16 * 16 * 16 * 16 * 16 * 16 * 16
        
        self.output_grad_ub_fp16 = self.tik_instance.Tensor("float16", (self.ub_tensor_size + 8,), 
                                    name="output_grad_ub_fp16", scope=tik.scope_ubuf)
        self.fp16_whole_num = self.tik_instance.Scalar(dtype="int64", name="fp16_whole_num")
        self.fp16_whole_num.set_as(self.input_h * self.input_w * self.batch * self.channel)

        self.fp16_left = self.tik_instance.Scalar(dtype="int64", name="fp16_whole_num")
        self.fp16_left = self.fp16_whole_num % 64

        self.fp16_loop = self.tik_instance.Scalar(dtype="int64", name="fp16_loop")
        self.fp16_loop.set_as(self.fp16_whole_num / 64)

        self.core_fp16_loop = self.tik_instance.Scalar(dtype="int64", name="core_fp16_loop")
        self.core_fp16_loop.set_as(self.fp16_loop / (self.core_num - 1))

        self.core_fp16_loop_left = self.tik_instance.Scalar(dtype="int64", name="core_fp16_loop_left")
        self.core_fp16_loop_left.set_as(self.fp16_loop % (self.core_num - 1))

    def tiling_cubic(self):
        self.vector_max = 255 * self.block_num_fp32 * 8
        
        tiling_ub = self.tik_instance.Tensor(
            Constant.TILING_SCALAR_DTYPE, [Constant.TILING_ARG_NUM, ], name="tiling_ub", scope=tik.scope_ubuf)
        burst_val = get_ceil_int(Constant.TILING_ARG_NUM, Constant.NUM_EACH_BURST)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, burst_val, 0, 0)

        self.core_num = self.tik_instance.Scalar(dtype="int32", name="core_num")
        self.core_num.set_as(tiling_ub[16])

        self.output_h = self.tik_instance.Scalar(dtype="int32", name="output_h")
        self.output_h.set_as(tiling_ub[3])

        self.output_w = self.tik_instance.Scalar(dtype="int32", name="output_w")
        self.output_w.set_as(tiling_ub[4])

        self.batch = self.tik_instance.Scalar(dtype="int32", name="batch")
        self.batch.set_as(tiling_ub[5])

        self.channel = self.tik_instance.Scalar(dtype="int32", name="channel")
        self.channel.set_as(tiling_ub[6])

        self.input_h = self.tik_instance.Scalar(dtype="int32", name="input_h")
        self.input_h.set_as(tiling_ub[7])

        self.input_w = self.tik_instance.Scalar(dtype="int32", name="input_w")
        self.input_w.set_as(tiling_ub[8])

        self.scale_0 = self.tik_instance.Scalar(dtype="float32", name="scale_0")
        self.scale_0.set_as(tiling_ub[12])

        self.scale_1 = self.tik_instance.Scalar(dtype="float32", name="scale_1")
        self.scale_1.set_as(tiling_ub[13])

        self.check_scale(self.scale_0, self.scale_1)

        self.height_scale = self.tik_instance.Scalar("float32", name="height_scale")
        self.width_scale = self.tik_instance.Scalar("float32", name="width_scale")

        self.area_pixel_compute_scale(self.height_scale, self.input_h, self.output_h,
                                      self.align_corners, scale=self.scales_h)
        self.area_pixel_compute_scale(self.width_scale, self.input_w, self.output_w,
                                      self.align_corners, scale=self.scales_w)

        self.input_limit = self.tik_instance.Scalar(dtype="int64", name="input_limit")
        self.input_limit.set_as(self.batch * self.channel * self.input_h * self.input_w)

        self.output_limit = self.tik_instance.Scalar(dtype="int64", name="output_limit")
        self.output_limit.set_as(self.batch * self.channel * self.output_h * self.output_w)

        self.output_num = self.tik_instance.Scalar(dtype="int64", name="output_num")
        self.output_num.set_as(self.batch * self.channel)

        self.ub_loop_times = self.tik_instance.Scalar(dtype="int64", name="ub_loop_times")
        self.ub_loop_times.set_as(self.output_num / self.ub_tensor_size)
        
        self.vector_loop_times = math.floor(int(self.ub_tensor_size) / int(self.vector_max))
        self.vector_left_length = math.floor(int(self.ub_tensor_size) % int(self.vector_max))

        self.left_ubtensor_size = self.tik_instance.Scalar(dtype="int64", name="left_ubtensor_size")
        self.left_ubtensor_size.set_as(self.output_num % self.ub_tensor_size)

        self.left_vector_loop_times = self.tik_instance.Scalar(dtype="int64", name="left_vector_loop_times")
        self.left_vector_loop_times.set_as(self.left_ubtensor_size / self.vector_max)

        self.left_vector_left_length = self.tik_instance.Scalar(dtype="int64", name="left_vector_left_length")
        self.left_vector_left_length.set_as(self.left_ubtensor_size % self.vector_max)

        self.t_x = self.tik_instance.Scalar("float32", name="t_x")
        self.t_y = self.tik_instance.Scalar("float32", name="t_y")

    def conv_single(self, gm_offset, length):
        ub_loop = (length / 64) / int(self.ub_max_size)
        ub_left = (length / 64) % int(self.ub_max_size)
        with self.tik_instance.for_range(0, ub_loop) as ub_loop_index:
            self.data_move_general(self.output_grad_ub, self.input_grads_gm_workspace, 
                                [gm_offset + ub_loop_index * self.ub_max_size * 64, 0],
                                self.ub_max_size * 8, self.input_limit, self.dtype_bytes_size_fp32, "gm2ub")
            vector_loop = int(self.ub_max_size / 255)
            vecto_loop_left = int(self.ub_max_size % 255)
            with self.tik_instance.for_range(0, vector_loop) as vector_loop_index:
                self.tik_instance.vec_conv(64, 'none', self.output_grad_ub_fp16[vector_loop_index * 255 * 64],
                                self.output_grad_ub[vector_loop_index * 255 * 64], 255, 4, 8)  
            self.tik_instance.vec_conv(64, 'none', self.output_grad_ub_fp16[vector_loop * 255 * 64],
                                self.output_grad_ub[vector_loop * 255 * 64], vecto_loop_left, 4, 8)
            self.data_move_general(self.input_grads_gm, self.output_grad_ub_fp16, 
                                [0, gm_offset + ub_loop_index * self.ub_max_size * 64],
                                self.ub_max_size * 4, self.input_limit, self.dtype_bytes_size, "ub2gm") 
        with self.tik_instance.if_scope(ub_left > 0):
            self.data_move_general(self.output_grad_ub, self.input_grads_gm_workspace, 
                                [gm_offset + ub_loop * self.ub_max_size * 64, 0],
                                ub_left * 8, self.input_limit, self.dtype_bytes_size_fp32, "gm2ub")
            vector_loop = ub_left / 255
            vecto_loop_left = ub_left % 255
            with self.tik_instance.for_range(0, vector_loop) as vector_loop_index:
                self.tik_instance.vec_conv(64, 'none', self.output_grad_ub_fp16[vector_loop_index * 255 * 64],
                                self.output_grad_ub[vector_loop_index * 255 * 64], 255, 4, 8)  
            self.tik_instance.vec_conv(64, 'none', self.output_grad_ub_fp16[vector_loop * 255 * 64],
                                self.output_grad_ub[vector_loop * 255 * 64], vecto_loop_left, 4, 8)
            self.data_move_general(self.input_grads_gm, self.output_grad_ub_fp16, 
                                [0, gm_offset + ub_loop * self.ub_max_size * 64],
                                ub_left * 4, self.input_limit, self.dtype_bytes_size, "ub2gm")

    def conv_data(self, core_index):
        with self.tik_instance.if_scope(core_index < self.core_num - 2):
            gm_offset = 64 * core_index * self.core_fp16_loop
            self.conv_single(gm_offset, self.core_fp16_loop * 64)
        with self.tik_instance.elif_scope(core_index == self.core_num - 2):
            gm_offset = 64 * core_index * self.core_fp16_loop
            self.conv_single(gm_offset, (self.core_fp16_loop + self.core_fp16_loop_left) * 64)
        with self.tik_instance.elif_scope(core_index == self.core_num - 1):
            with self.tik_instance.if_scope(self.fp16_left):
                gm_offset = 64 * (core_index * self.core_fp16_loop + self.core_fp16_loop_left)
                self.data_move_general(self.output_grad_ub, self.input_grads_gm_workspace, 
                        [gm_offset, 0],
                        8, self.input_limit, self.dtype_bytes_size_fp32, "gm2ub")
                self.tik_instance.vec_conv(self.fp16_left, 'none', self.output_grad_ub_fp16,
                                            self.output_grad_ub, 1, 4, 8)
                self.data_move_general(self.input_grads_gm, self.output_grad_ub_fp16, 
                        [0, gm_offset],
                        4, self.input_limit, self.dtype_bytes_size, "ub2gm")
        
    def compute(self):
        compute_num = self.tik_instance.Scalar("int64", name="compute_num")
        compute_tail = self.tik_instance.Scalar("int64", name="compute_tail")
        compute_num.set_as((self.output_h * self.output_w) // self.core_num)
        compute_tail.set_as((self.output_h * self.output_w) % self.core_num)

        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as index:
            loop_begin = self.tik_instance.Scalar(dtype="int64", name="loop_begin")
            loop_length = self.tik_instance.Scalar(dtype="int64", name="loop_length")
            with self.tik_instance.if_scope(index < compute_tail):
                loop_begin.set_as(compute_num * index + index)
                loop_length.set_as(compute_num + 1)
            with self.tik_instance.else_scope():
                loop_begin.set_as(compute_num * index + compute_tail)
                loop_length.set_as(compute_num)

            with self.tik_instance.for_range(loop_begin, loop_begin + loop_length) as pos:
                self.single_process(pos)
            if (self.dtype_grads == "float16"):
                self.tik_instance.block_barrier(self.sync_workspace)
                self.conv_data(index)

    @staticmethod
    def cubic_convolution1(x, cubic_coeff_a):
        return ((cubic_coeff_a + 2) * x - (cubic_coeff_a + 3)) * x * x + 1

    @staticmethod
    def cubic_convolution2(x, cubic_coeff_a):
        return ((cubic_coeff_a * x - 5 * cubic_coeff_a) * x + 8 * cubic_coeff_a) * x - 4 * cubic_coeff_a

    def get_cubic_coefficients(self, t):
        coeffs_ub = self.tik_instance.Tensor("float32", (4,), name="coeffs_ub", scope=tik.scope_ubuf)
        x1 = self.tik_instance.Scalar("float32", name="x1")
        x2 = self.tik_instance.Scalar("float32", name="x2")
        x1.set_as(t)
        coeffs_ub[0].set_as(self.cubic_convolution2(x1 + 1.0, self.cubic_coeff_a))
        coeffs_ub[1].set_as(self.cubic_convolution1(x1, self.cubic_coeff_a))

        x2.set_as(1.0 - t)
        coeffs_ub[2].set_as(self.cubic_convolution1(x2, self.cubic_coeff_a))
        coeffs_ub[3].set_as(self.cubic_convolution2(x2 + 1.0, self.cubic_coeff_a))
        return coeffs_ub

    def guard_index_and_lambda(self, real_input_index, input_size, input_int, input_float):
        with self.tik_instance.if_scope(input_int > input_size - 1):
            input_int.set_as(input_size - 1)

        input_float.set_as(real_input_index - input_int)
        with self.tik_instance.if_scope(input_float < 0.0):
            input_float.set_as(0.0)
        with self.tik_instance.if_scope(input_float > 1.0):
            input_float.set_as(1.0)

    def check_scale(self, scale_0, scale_1):
        with self.tik_instance.if_scope(scale_0 > 0.0):
            self.scales_h = scale_0
        with self.tik_instance.if_scope(scale_1 > 0.0):
            self.scales_w = scale_1

    def single_process(self, pos):
        height = pos / self.output_w
        weight = pos % self.output_w

        src_scalar = self.tik_instance.Scalar(dtype="int32", init_value=height)
        dst_scalar_output_x = self.tik_instance.Scalar(dtype="float32")
        self.tik_instance.scalar_conv('none', dst_scalar_output_x, src_scalar)
        real_x = self.area_pixel_compute_source_index(self.height_scale, dst_scalar_output_x, self.align_corners)
        src_scalar = self.tik_instance.Scalar(dtype="float32", init_value=real_x)
        dst_scalar_input_x = self.tik_instance.Scalar(dtype="int32")
        self.tik_instance.scalar_conv('floor', dst_scalar_input_x, src_scalar)
        self.guard_index_and_lambda(real_x, self.input_h, dst_scalar_input_x, self.t_x)
        x_coeffs_ub = self.get_cubic_coefficients(self.t_x)

        src_scalar = self.tik_instance.Scalar(dtype="int32", init_value=weight)
        dst_scalar_output_y = self.tik_instance.Scalar(dtype="float32")
        self.tik_instance.scalar_conv('none', dst_scalar_output_y, src_scalar)
        real_y = self.area_pixel_compute_source_index(self.width_scale, dst_scalar_output_y, self.align_corners)
        src_scalar = self.tik_instance.Scalar(dtype="float32", init_value=real_y)
        dst_scalar_input_y = self.tik_instance.Scalar(dtype="int32")
        self.tik_instance.scalar_conv('floor', dst_scalar_input_y, src_scalar)
        self.guard_index_and_lambda(real_y, self.input_w, dst_scalar_input_y, self.t_y)
        y_coeffs_ub = self.get_cubic_coefficients(self.t_y)

        x_coeffs_scalar = self.tik_instance.Scalar(dtype="float32")
        y_coeffs_scalar = self.tik_instance.Scalar(dtype="float32")
        with self.tik_instance.for_range(0, 4) as i:
            with self.tik_instance.for_range(0, 4) as j: 
                x_coeffs_scalar.set_as(x_coeffs_ub[i])
                y_coeffs_scalar.set_as(y_coeffs_ub[j])
                self.calc(height, weight, dst_scalar_input_x - 1 + i, 
                        dst_scalar_input_y - 1 + j, x_coeffs_scalar, y_coeffs_scalar)
            
    def ub_fp16_process(self, gm_offset, ub_loop_time):
        self.data_move_general(self.output_grad_ub_fp16, self.output_grads_gm, 
                    [gm_offset + ub_loop_time * self.ub_tensor_size, 0],
                    int(self.ub_tensor_size / self.block_num), self.output_limit, self.dtype_bytes_size, "gm2ub")
        vector_loop = int(self.ub_tensor_size / (255 * 64))
        vector_left = self.ub_tensor_size % (255 * 64)
        small_left = int(vector_left / 64)
        small_mask = vector_left % 64
        if vector_loop:
            self.tik_instance.vec_conv(64, 'none', self.output_grad_ub, 
                                            self.output_grad_ub_fp16, 255, 8, 4)
        if vector_left:
            if small_left:
                self.tik_instance.vec_conv(64, 'none', self.output_grad_ub[vector_loop * 255 * 64], 
                            self.output_grad_ub_fp16[vector_loop * 255 * 64], small_left, 8, 4)
            if small_mask:
                self.tik_instance.vec_conv(small_mask, 'none', 
                            self.output_grad_ub[vector_loop * 255 * 64 + small_left * 64], 
                            self.output_grad_ub_fp16[vector_loop * 255 * 64 + small_left * 64], 1, 8, 4)

    def ub_process(self, gm_offset, dst_gm_offset, x_coeffs, y_coeffs):
        with self.tik_instance.for_range(0, self.ub_loop_times) as ub_loop_time:
            if (self.dtype_grads == "float16"):
                self.ub_fp16_process(gm_offset, ub_loop_time)
            else:
                self.data_move_general(self.output_grad_ub, self.output_grads_gm, 
                    [gm_offset + ub_loop_time * self.ub_tensor_size, 0],
                    int(self.ub_tensor_size / self.block_num_fp32), self.output_limit, self.dtype_bytes_size, "gm2ub")

            if self.vector_loop_times > 0:
                with self.tik_instance.for_range(0, self.vector_loop_times) as vector_loop_time:
                    self.tik_instance.vec_muls(64, self.output_grad_temp_ub[vector_loop_time * 64 * 255], 
                                                self.output_grad_ub[vector_loop_time * 64 * 255], 
                                                x_coeffs, 255, 8, 8)
                    self.tik_instance.vec_muls(64, self.output_grad_temp_ub[vector_loop_time * 64 * 255], 
                                                self.output_grad_temp_ub[vector_loop_time * 64 * 255], 
                                                y_coeffs, 255, 8, 8)
            if int(self.vector_left_length / 64) > 0:
                self.tik_instance.vec_muls(64, self.output_grad_temp_ub[int(self.vector_loop_times * 64 * 255)], 
                                                self.output_grad_ub[int(self.vector_loop_times * 64 * 255)], 
                                                x_coeffs, int(self.vector_left_length / 64), 8, 8)
                self.tik_instance.vec_muls(64, self.output_grad_temp_ub[int(self.vector_loop_times * 64 * 255)], 
                                                self.output_grad_temp_ub[int(self.vector_loop_times * 64 * 255)], 
                                                y_coeffs, int(self.vector_left_length / 64), 8, 8)
            if int(self.vector_left_length % 64) > 0:
                tmp_offset = int(self.vector_loop_times * 64 * 255) + int(self.vector_left_length / 64) * 64
                self.tik_instance.vec_muls(int(self.vector_left_length % 64), self.output_grad_temp_ub[tmp_offset],
                                                self.output_grad_ub[tmp_offset], x_coeffs, 1, 8, 8)
                self.tik_instance.vec_muls(int(self.vector_left_length % 64), self.output_grad_temp_ub[tmp_offset], 
                                                self.output_grad_temp_ub[tmp_offset], y_coeffs, 1, 8, 8)
            self.tik_instance.set_atomic_add(1)

            self.data_move_general(self.atomic_add_space, self.output_grad_temp_ub, 
                    [0, dst_gm_offset + ub_loop_time * self.ub_tensor_size],
                    int(self.ub_tensor_size / self.block_num_fp32), self.input_limit, 
                    self.dtype_bytes_size_fp32, "ub2gm")
            self.tik_instance.set_atomic_add(0)

    def align_fp16_process(self, gm_offset):
        up_length = self.tik_instance.Scalar(dtype="int64", name="up_length")
        up_length.set_as(get_ceil_int(self.left_ubtensor_size, self.block_num))
        
        self.data_move_general(self.output_grad_ub_fp16, self.output_grads_gm, 
                    [gm_offset + self.ub_loop_times * self.ub_tensor_size, 0],
                    up_length, self.output_limit, self.dtype_bytes_size, "gm2ub")

        vector_loop = self.tik_instance.Scalar(dtype="int64", name="vector_loop")
        vector_loop.set_as((up_length * self.block_num) / (255 * 64))

        vector_left = self.tik_instance.Scalar(dtype="int64", name="vector_left")
        vector_left.set_as((up_length * self.block_num) % (255 * 64))
        
        small_left = self.tik_instance.Scalar(dtype="int64", name="small_left")
        small_left.set_as(vector_left / 64)

        mask = self.tik_instance.Scalar(dtype="int64", name="mask")
        mask.set_as(self.left_ubtensor_size % 64)

        with self.tik_instance.if_scope(vector_loop > 0):
            self.tik_instance.vec_conv(64, 'none', self.output_grad_ub, self.output_grad_ub_fp16, 255, 8, 4)
        with self.tik_instance.if_scope(vector_left > 0):
            with self.tik_instance.if_scope(small_left > 0):
                self.tik_instance.vec_conv(64, 'none', self.output_grad_ub[vector_loop * 255 * 64], 
                                                self.output_grad_ub_fp16[vector_loop * 255 * 64], small_left, 8, 4)
            with self.tik_instance.if_scope(mask > 0):
                self.tik_instance.vec_conv(mask, 'none',
                                        self.output_grad_ub[vector_loop * 255 * 64 + small_left * 64], 
                                        self.output_grad_ub_fp16[vector_loop * 255 * 64 + small_left * 64], 1, 8, 4)
                
    def align_process(self, gm_offset, dst_gm_offset, x_coeffs, y_coeffs):
        if (self.dtype_grads == "float16"):
            self.align_fp16_process(gm_offset)
        else:
            self.data_move_general(self.output_grad_ub, self.output_grads_gm, 
                    [gm_offset + self.ub_loop_times * self.ub_tensor_size, 0],
                    self.left_ubtensor_size / self.block_num_fp32, self.output_limit, self.dtype_bytes_size, "gm2ub")
        
        with self.tik_instance.if_scope(self.left_vector_loop_times > 0):
            with self.tik_instance.for_range(0, self.left_vector_loop_times) as left_vector_loop_time:
                self.tik_instance.vec_muls(64, self.output_grad_temp_ub[left_vector_loop_time * 64 * 255], 
                                            self.output_grad_ub[left_vector_loop_time * 64 * 255], 
                                            x_coeffs, 255, 8, 8)
                self.tik_instance.vec_muls(64, self.output_grad_temp_ub[left_vector_loop_time * 64 * 255], 
                                            self.output_grad_temp_ub[left_vector_loop_time * 64 * 255], 
                                            y_coeffs, 255, 8, 8)
        with self.tik_instance.if_scope((self.left_vector_left_length / 64) > 0):
            self.tik_instance.vec_muls(64, self.output_grad_temp_ub[self.left_vector_loop_times * 64 * 255], 
                                            self.output_grad_ub[self.left_vector_loop_times * 64 * 255], 
                                            x_coeffs, self.left_vector_left_length / 64, 8, 8)
            self.tik_instance.vec_muls(64, self.output_grad_temp_ub[self.left_vector_loop_times * 64 * 255], 
                                            self.output_grad_temp_ub[self.left_vector_loop_times * 64 * 255], 
                                            y_coeffs, self.left_vector_left_length / 64, 8, 8)
        with self.tik_instance.if_scope((self.left_vector_left_length % 64) > 0):
            tmp_offset = self.left_vector_loop_times * 64 * 255 + (self.left_vector_left_length / 64) * 64
            self.tik_instance.vec_muls(self.left_vector_left_length % 64, self.output_grad_temp_ub[tmp_offset], 
                                            self.output_grad_ub[tmp_offset], x_coeffs, 1, 8, 8)
            self.tik_instance.vec_muls(self.left_vector_left_length % 64, self.output_grad_temp_ub[tmp_offset], 
                                            self.output_grad_temp_ub[tmp_offset], y_coeffs, 1, 8, 8)
        self.tik_instance.set_atomic_add(1)
        self.data_move_general(self.atomic_add_space, self.output_grad_temp_ub, 
                                [0, dst_gm_offset + self.ub_loop_times * self.ub_tensor_size],
                                self.left_ubtensor_size / self.block_num_fp32, self.input_limit, 
                                self.dtype_bytes_size_fp32, "ub2gm")
        self.tik_instance.set_atomic_add(0)

    def no_align_fp16_process(self, gm_offset, dst_gm_offset):
        up_length = self.tik_instance.Scalar(dtype="int64", name="up_length")
        up_length.set_as(get_ceil_int(self.left_ubtensor_size, self.block_num))

        self.data_move_general(self.output_grad_ub_fp16, self.output_grads_gm, 
                                [gm_offset + self.ub_loop_times * self.ub_tensor_size, 0], 
                                up_length, self.output_limit, self.dtype_bytes_size, "gm2ub")

        vector_loop = self.tik_instance.Scalar(dtype="int64", name="vector_loop")
        vector_loop.set_as((up_length * self.block_num) / (255 * 64))

        vector_left = self.tik_instance.Scalar(dtype="int64", name="vector_left")
        vector_left.set_as((up_length * self.block_num) % (255 * 64))
        
        small_left = self.tik_instance.Scalar(dtype="int64", name="small_left")
        small_left.set_as(vector_left / 64)

        mask = self.tik_instance.Scalar(dtype="int64", name="mask")
        mask.set_as(self.left_ubtensor_size % 64)

        with self.tik_instance.if_scope(vector_loop > 0):
            self.tik_instance.vec_conv(64, 'none', self.output_grad_ub, self.output_grad_ub_fp16, 255, 8, 4)
        with self.tik_instance.if_scope(vector_left > 0):
            with self.tik_instance.if_scope(small_left > 0):
                self.tik_instance.vec_conv(64, 'none', self.output_grad_ub[vector_loop * 255 * 64], 
                                                self.output_grad_ub_fp16[vector_loop * 255 * 64], small_left, 8, 4)
            with self.tik_instance.if_scope(mask >= 8):
                self.tik_instance.vec_conv((mask / 8) * 8, 'none', 
                                            self.output_grad_ub[vector_loop * 255 * 64 + small_left * 64], 
                                            self.output_grad_ub_fp16[vector_loop * 255 * 64 + small_left * 64], 1, 8, 4)

        left_length = self.tik_instance.Scalar(dtype="int64", name="left_length")
        left_length.set_as(self.left_ubtensor_size % self.block_num_fp32)
        with self.tik_instance.if_scope(left_length > 0):
            sort_pos = gm_offset + self.ub_loop_times * self.ub_tensor_size + self.left_ubtensor_size - 8
            sort_ub = self.tik_instance.Tensor("float16", (16,), name="sort_ub", scope=tik.scope_ubuf)
            self.data_move_general(sort_ub, self.output_grads_gm, [sort_pos, 0], 
                                1, self.output_limit, self.dtype_bytes_size, "gm2ub")

            with self.tik_instance.for_range(0, 8 - left_length) as sort_ub_index:
                sort_ub[sort_ub_index].set_as(0.0)
            self.tik_instance.vec_conv(8, 'none', 
                                        self.output_grad_ub[self.left_ubtensor_size - left_length], 
                                        sort_ub, 1, 0, 0)

    def no_align_fp32_process(self, gm_offset, align_pos):
        self.tik_instance.data_move(self.output_grad_ub, 
                                    self.output_grads_gm[gm_offset + self.ub_loop_times * self.ub_tensor_size], 
                                    0, 1, self.left_ubtensor_size / self.block_num_fp32, 0, 0)
        self.tik_instance.data_move(self.output_grad_ub[align_pos], 
                                    self.output_grads_gm[gm_offset + self.output_num - self.block_num_fp32], 
                                    0, 1, 1, 0, 0)
        self.tik_instance.vec_dup(8 - (self.left_ubtensor_size % 8), 
                                    self.output_grad_ub[align_pos], 0, 1, 0)

    def no_align_process(self, gm_offset, dst_gm_offset, x_coeffs, y_coeffs):
        align_pos = (self.left_ubtensor_size / self.block_num_fp32) * self.block_num_fp32
        if (self.dtype_grads == "float16"):
            self.no_align_fp16_process(gm_offset, dst_gm_offset)
        else:
            self.no_align_fp32_process(gm_offset, align_pos)
            
        new_block_num = self.tik_instance.Scalar(dtype="int64", name="new_block_num")
        new_block_num.set_as(get_ceil_int(self.left_ubtensor_size, self.block_num_fp32) * self.block_num_fp32)
        new_repeat_times = self.tik_instance.Scalar(dtype="int64", name="new_repeat_times")
        new_repeat_times.set_as(new_block_num / self.vector_max)
        left_new_length = self.tik_instance.Scalar(dtype="int64", name="left_new_length")
        left_new_length.set_as(new_block_num % self.vector_max)

        with self.tik_instance.if_scope(new_repeat_times > 0):
            with self.tik_instance.for_range(0, new_repeat_times) as new_repeat_time:
                self.tik_instance.vec_muls(64, self.output_grad_temp_ub[new_repeat_time * 64 * 255], 
                                            self.output_grad_ub[new_repeat_time * 64 * 255], x_coeffs, 255, 8, 8)
                self.tik_instance.vec_muls(64, self.output_grad_temp_ub[new_repeat_time * 64 * 255], 
                                            self.output_grad_temp_ub[new_repeat_time * 64 * 255], y_coeffs, 255, 8, 8)

        with self.tik_instance.if_scope((left_new_length / 64) > 0):
            self.tik_instance.vec_muls(64, self.output_grad_temp_ub[new_repeat_times * 64 * 255], 
                                        self.output_grad_ub[new_repeat_times * 64 * 255], 
                                        x_coeffs, left_new_length / 64, 8, 8)
            self.tik_instance.vec_muls(64, self.output_grad_temp_ub[new_repeat_times * 64 * 255], 
                                        self.output_grad_temp_ub[new_repeat_times * 64 * 255], 
                                        y_coeffs, left_new_length / 64, 8, 8)
        
        with self.tik_instance.if_scope((left_new_length % 64) > 0):
            tmp_offset = new_repeat_times * 64 * 255 + (left_new_length / 64) * 64
            self.tik_instance.vec_muls(left_new_length % 64, self.output_grad_temp_ub[tmp_offset], 
                                        self.output_grad_ub[tmp_offset], x_coeffs, 1, 8, 8)
            self.tik_instance.vec_muls(left_new_length % 64, self.output_grad_temp_ub[tmp_offset], 
                                        self.output_grad_temp_ub[tmp_offset], y_coeffs, 1, 8, 8)
        
        self.tik_instance.set_atomic_add(1)
        self.data_move_general(self.atomic_add_space, self.output_grad_temp_ub, 
                                [0, dst_gm_offset + self.ub_loop_times * self.ub_tensor_size],
                                self.left_ubtensor_size / self.block_num_fp32, 
                                self.input_limit, self.dtype_bytes_size_fp32, "ub2gm")
        self.data_move_general(self.atomic_add_space, self.output_grad_temp_ub, 
                                [align_pos, dst_gm_offset + self.output_num - self.block_num_fp32],
                                1, self.input_limit, self.dtype_bytes_size_fp32, "ub2gm")
        self.tik_instance.set_atomic_add(0)

    def data_move_general(self, dst, src, offsets, block_size, limit_size, dtype_size, trans):
        src_offset, dst_offset = offsets
        block_num = 32 / dtype_size
        if trans == "gm2ub":
            if tbe_platform.api_check_support("tik.data_move_pad"):
                with self.tik_instance.if_scope(src_offset + block_size * block_num >= limit_size):
                    pad_length = limit_size - src_offset
                    self.tik_instance.data_move_pad(dst[dst_offset], src[src_offset], 1, 
                                                    pad_length * dtype_size, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(dst[dst_offset], src[src_offset], 0, 1, block_size, 0, 0)
            else:
                self.tik_instance.data_move(dst[dst_offset], src[src_offset], 0, 1, block_size, 0, 0)
        else:
            if tbe_platform.api_check_support("tik.data_move_pad"):
                with self.tik_instance.if_scope(dst_offset + block_size * block_num >= limit_size):
                    pad_length = limit_size - dst_offset
                    self.tik_instance.data_move_pad(dst[dst_offset], src[src_offset], 1, 
                                                    pad_length * dtype_size, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(dst[dst_offset], src[src_offset], 0, 1, block_size, 0, 0)
            else:
                self.tik_instance.data_move(dst[dst_offset], src[src_offset], 0, 1, block_size, 0, 0)

    def block_process(self, gm_offset, dst_gm_offset, x_coeffs, y_coeffs):
        if (self.dtype_grads == "float16"):
            self.tik_instance.vec_dup(8, self.output_grad_ub, 0, 1, 0)
            self.data_move_general(self.output_grad_ub_fp16, self.output_grads_gm, 
                                [gm_offset + self.ub_loop_times * self.ub_tensor_size, 0],
                                1, self.output_limit, self.dtype_bytes_size, "gm2ub")      
            with self.tik_instance.for_range(self.left_ubtensor_size, 16) as left_ub_index:
                self.output_grad_ub_fp16[left_ub_index].set_as(0.0)
            self.tik_instance.vec_conv(self.left_ubtensor_size, 'none', self.output_grad_ub,
                                        self.output_grad_ub_fp16, 1, 8, 4)
        else:
            self.data_move_general(self.output_grad_ub, self.output_grads_gm, 
                                [gm_offset + self.ub_loop_times * self.ub_tensor_size, 0],
                                1, self.output_limit, self.dtype_bytes_size, "gm2ub")
            with self.tik_instance.for_range(self.left_ubtensor_size, 8) as left_ub_index:
                self.output_grad_ub[left_ub_index].set_as(0.0)
        self.tik_instance.vec_muls(8, self.output_grad_temp_ub, self.output_grad_ub, x_coeffs, 1, 8, 8)
        self.tik_instance.vec_muls(8, self.output_grad_temp_ub, self.output_grad_temp_ub, y_coeffs, 1, 8, 8)
        self.tik_instance.set_atomic_add(1)
        self.data_move_general(self.atomic_add_space, self.output_grad_temp_ub, 
                                [0, dst_gm_offset + self.ub_loop_times * self.ub_tensor_size],
                                1, self.input_limit, self.dtype_bytes_size_fp32, "ub2gm")
        self.tik_instance.set_atomic_add(0)

    # 'pylint: disable=huawei-too-many-arguments
    # 'pylint: disable=huawei-too-many-locals
    def calc(self, height, weight, dst_height, dst_weight, x_coeffs, y_coeffs):
        height = self.tik_instance.Scalar(dtype="int64", init_value=height)
        weight = self.tik_instance.Scalar(dtype="int64", init_value=weight)
        dst_height_scalar = self.tik_instance.Scalar(dtype="int64", init_value=dst_height)
        dst_weight_scalar = self.tik_instance.Scalar(dtype="int64", init_value=dst_weight)
        
        height_bound = self.tik_instance.Scalar(dtype="int64", init_value=self.input_h - 1)
        weight_bound = self.tik_instance.Scalar(dtype="int64", init_value=self.input_w - 1)

        temp_x = self.tik_instance.Scalar(dtype="int64")
        temp_y = self.tik_instance.Scalar(dtype="int64")

        dst_height_bound = self.tik_instance.Scalar(dtype="int64")
        dst_weight_bound = self.tik_instance.Scalar(dtype="int64")

        self.tik_instance.scalar_min(temp_x, dst_height_scalar, height_bound)
        self.tik_instance.scalar_min(temp_y, dst_weight_scalar, weight_bound)
        self.tik_instance.scalar_max(dst_height_bound, temp_x, 0)
        self.tik_instance.scalar_max(dst_weight_bound, temp_y, 0)
        
        gm_offset = (height * self.output_w + weight) * self.batch * self.channel
        dst_gm_offset = (dst_height_bound * self.input_w + dst_weight_bound) * self.batch * self.channel

        with self.tik_instance.if_scope(self.ub_loop_times > 0):
            self.ub_process(gm_offset, dst_gm_offset, x_coeffs, y_coeffs)

        with self.tik_instance.if_scope(self.left_ubtensor_size > 0):
            with self.tik_instance.if_scope(self.left_ubtensor_size % 8 == 0):
                self.align_process(gm_offset, dst_gm_offset, x_coeffs, y_coeffs)

            with self.tik_instance.elif_scope(self.left_ubtensor_size > 8):                
                self.no_align_process(gm_offset, dst_gm_offset, x_coeffs, y_coeffs)

            with self.tik_instance.elif_scope(self.left_ubtensor_size < 8):
                self.block_process(gm_offset, dst_gm_offset, x_coeffs, y_coeffs)

    def upsamplebicubic2d_backward_compute(self):
        self.compute()
        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
            })
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": False, "is_dynamic_shape": True}
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.output_grads_gm, ],
            outputs=[self.input_grads_gm, ],
            flowtable=[self.tiling_gm],
            config=opt_config)
        return self.tik_instance


class ResizeBicubicBackward:
    """
    ResizeBicubicBackward main functions
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self, grads, original_size, scales, coordinate_transformation_mode,
                 cubic_coeff_a, kernel_name="resize_grad_d"):
        """
        Init parameters
        """
        self.tik_instance = tik.Tik()
        self.dtype_grads = grads.get("dtype")
        self.align_corners = (coordinate_transformation_mode == "align_corners")
        self.kernel_name = kernel_name

        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        ub_size_bytes = tbe_platform.get_soc_spec("UB_SIZE")
        dtype_bytes_size = get_bit_len(self.dtype_grads) // Constant.EIGHT_BIT
        self.data_each_block = Constant.ONE_BLOCK_SIZE // dtype_bytes_size
        self.vector_mask_max = 8 * self.data_each_block
        self.ub_tensor_size = \
            (ub_size_bytes - Constant.UB_RESERVE_SIZE) // \
            dtype_bytes_size // self.data_each_block * self.data_each_block

        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_SCALAR_DTYPE, (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
        self.input_grads_gm = self.tik_instance.Tensor(self.dtype_grads, (Constant.MAX_INT64_VALUE,),
                                                       name="input_grads_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype_grads, (Constant.MAX_INT64_VALUE,),
                                                  name="output_gm", scope=tik.scope_gm,
                                                  is_atomic_add=True)

        self.tiling_args()

        self.pos_max_in = self.tik_instance.Scalar("int32", name="pos_max_in")
        self.pos_max_in.set_as(self.grads_shape_num - self.data_each_block)
        self.pos_max_out = self.tik_instance.Scalar("int32", name="pos_max_out")
        self.pos_max_out.set_as(self.ori_shape_num - self.data_each_block)

        with self.tik_instance.if_scope(self.grads_shape_num < self.data_each_block):
            self.pos_max_in.set_as(0)
        with self.tik_instance.if_scope(self.ori_shape_num < self.data_each_block):
            self.pos_max_in.set_as(0)

    def tiling_args(self):
        """
        Get runtime params from tiling
        """
        self.tiling_mode = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="tiling_mode")
        self.grads_shape_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="grads_shape_num")
        self.ori_shape_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="ori_shape_num")
        self.h_out = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="h_out")
        self.w_out = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="w_out")
        self.h_in = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="h_in")
        self.w_in = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="w_in")
        self.h_scale = self.tik_instance.Scalar("float32", name="h_scale")
        self.w_scale = self.tik_instance.Scalar("float32", name="w_scale")
        self.cubic_coeff_a = self.tik_instance.Scalar("float32", name="cubic_coeff_a")
        self.scale_0 = self.tik_instance.Scalar("float32", name="scale_0")
        self.scale_1 = self.tik_instance.Scalar("float32", name="scale_1")
        self.batch_core_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="batch_core_num")
        self.batch_core_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="batch_core_tail")
        self.tiling_core_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="tiling_core_num")

        tiling_ub = self.tik_instance.Tensor(Constant.TILING_SCALAR_DTYPE, [Constant.TILING_ARG_NUM, ],
                                             name="tiling_ub", scope=tik.scope_ubuf)
        burst_val = get_ceil_int(Constant.TILING_ARG_NUM, Constant.NUM_EACH_BURST)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, burst_val, 0, 0)

        self.tiling_mode.set_as(tiling_ub[0])
        self.grads_shape_num.set_as(tiling_ub[1])
        self.ori_shape_num.set_as(tiling_ub[2])
        self.h_out.set_as(tiling_ub[5])
        self.w_out.set_as(tiling_ub[6])
        self.h_in.set_as(tiling_ub[7])
        self.w_in.set_as(tiling_ub[8])
        self.h_scale.set_as(tiling_ub[9])
        self.w_scale.set_as(tiling_ub[10])
        self.cubic_coeff_a.set_as(tiling_ub[11])
        self.scale_0.set_as(tiling_ub[12])
        self.scale_1.set_as(tiling_ub[13])
        self.batch_core_num.set_as(tiling_ub[14])
        self.batch_core_tail.set_as(tiling_ub[15])
        self.tiling_core_num.set_as(tiling_ub[16])

    def resize_bicubic_backward_compute_same_size(self):
        """
        Special case, same size of input and output images, just copy
        """
        input_grads_ub = self.tik_instance.Tensor(self.dtype_grads, (self.ub_tensor_size,),
                                                  name="input_grads_ub", scope=tik.scope_ubuf)
        loop_time = self.tik_instance.Scalar("int32", name="loop_time")
        loop_time.set_as(self.grads_shape_num // self.ub_tensor_size)
        move_offset = self.tik_instance.Scalar("int32", name="move_offset")
        burst_len = get_ceil_int(self.ub_tensor_size, self.data_each_block)
        with self.tik_instance.if_scope(loop_time > 0):
            with self.tik_instance.for_range(0, loop_time) as loop:
                move_offset.set_as(loop * self.ub_tensor_size)
                self.tik_instance.data_move(input_grads_ub, self.input_grads_gm[move_offset],
                                            0, 1, burst_len, 0, 0)
                self.tik_instance.data_move(self.output_gm[move_offset], input_grads_ub,
                                            0, 1, burst_len, 0, 0)
            move_offset.set_as(loop_time * self.ub_tensor_size)

        last_num = self.tik_instance.Scalar("int32", name="last_num")
        last_num.set_as(self.grads_shape_num % self.ub_tensor_size)
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.data_move(input_grads_ub, self.input_grads_gm[move_offset],
                                        0, 1, get_ceil_int(last_num, self.data_each_block), 0, 0)
            self.tik_instance.data_move(self.output_gm[move_offset], input_grads_ub,
                                        0, 1, get_ceil_int(last_num, self.data_each_block), 0, 0)

    def resize_bicubic_backward_compute_general(self):
        with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as core_index:
            with self.tik_instance.for_range(0, self.batch_core_num) as batch:
                self.resize_bicubic_compute_core(core_index + batch * self.tiling_core_num)
            with self.tik_instance.if_scope(core_index < self.batch_core_tail):
                self.resize_bicubic_compute_core(self.batch_core_num * self.tiling_core_num + core_index)

    def resize_bicubic_compute_core(self, core_index):
        input_grads_ub = self.tik_instance.Tensor(self.dtype_grads, (self.data_each_block,),
                                                  name="input_grads_ub", scope=tik.scope_ubuf)
        output_ub = self.tik_instance.Tensor(self.dtype_grads, (self.data_each_block,),
                                             name="output_ub", scope=tik.scope_ubuf)
        value_ub = self.tik_instance.Tensor(self.dtype_grads, (1,),
                                            name="value_ub", scope=tik.scope_ubuf)
        value_temp = self.tik_instance.Tensor("float32", (self.data_each_block,),
                                              name="value_temp", scope=tik.scope_ubuf)
        assist1_ub = self.tik_instance.Tensor("float32", (1,),
                                              name="assist1_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.h_out) as output_y:
            with self.tik_instance.for_range(0, self.w_out) as output_x:
                dst_scalar_output_x = self.tik_instance.Scalar(dtype="float32")
                dst_scalar_output_y = self.tik_instance.Scalar(dtype="float32")
                dst_scalar_input_x = self.tik_instance.Scalar(dtype="int32")
                dst_scalar_input_y = self.tik_instance.Scalar(dtype="int32")

                src_scalar_int = self.tik_instance.Scalar(dtype="int32", init_value=output_x)
                self.tik_instance.scalar_conv('none', dst_scalar_output_x, src_scalar_int)
                src_scalar_int.set_as(output_y)
                self.tik_instance.scalar_conv('none', dst_scalar_output_y, src_scalar_int)

                real_w = self.area_pixel_compute_source_index(self.w_scale,
                                                              dst_scalar_output_x,
                                                              self.align_corners)
                real_h = self.area_pixel_compute_source_index(self.h_scale,
                                                              dst_scalar_output_y,
                                                              self.align_corners)

                src_scalar_fp = self.tik_instance.Scalar(dtype="float32", init_value=real_w)
                self.tik_instance.scalar_conv('floor', dst_scalar_input_x, src_scalar_fp)
                src_scalar_fp.set_as(real_h)
                self.tik_instance.scalar_conv('floor', dst_scalar_input_y, src_scalar_fp)

                # interpolation process of resize_bicubic
                x_coeffs_ub = self.get_resize_bicubic_coefficients(dst_scalar_output_x,
                                                                   dst_scalar_input_x,
                                                                   self.w_in,
                                                                   self.w_out, 1)

                y_coeffs_ub = self.get_resize_bicubic_coefficients(dst_scalar_output_y,
                                                                   dst_scalar_input_y,
                                                                   self.h_in,
                                                                   self.h_out, 0)

                x_coeffs_scalar = self.tik_instance.Scalar(dtype="float32")
                y_coeffs_scalar = self.tik_instance.Scalar(dtype="float32")

                # Move to next channel
                input_pos = self.tik_instance.Scalar(dtype="int32")
                output_pos = self.tik_instance.Scalar(dtype="int32")

                input_pos.set_as(
                    core_index * self.w_out * self.h_out + output_y * self.w_out + output_x)
                output_pos.set_as(core_index * self.w_in * self.h_in)

                with self.tik_instance.if_scope(input_pos > self.pos_max_in):
                    self.tik_instance.data_move(input_grads_ub,
                                                self.input_grads_gm[self.pos_max_in],
                                                0, 1, 1, 0, 0)
                    value_ub[0].set_as(input_grads_ub[input_pos - self.pos_max_in])
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(input_grads_ub,
                                                self.input_grads_gm[input_pos],
                                                0, 1, 1, 0, 0)
                    value_ub[0].set_as(input_grads_ub[0])

                with self.tik_instance.for_range(0, 4) as i:
                    with self.tik_instance.for_range(0, 4) as j:
                        x_coeffs_scalar.set_as(x_coeffs_ub[i])
                        y_coeffs_scalar.set_as(y_coeffs_ub[j])
                        self.tik_instance.vec_dup(self.data_each_block, value_temp, 0, 1, 1)
                        # Convert fp16 to fp32 to improve accuracy
                        if self.dtype_grads == "float16":
                            self.tik_instance.vec_conv(
                                1, 'none', assist1_ub, value_ub, 1, 1, 1)
                            self.tik_instance.vec_muls(1, value_temp,
                                                       assist1_ub, x_coeffs_scalar, 1, 1, 1)
                            self.tik_instance.vec_muls(1, value_temp,
                                                       value_temp, y_coeffs_scalar, 1, 1, 1)
                        else:
                            self.tik_instance.vec_muls(1, value_temp,
                                                       value_ub, x_coeffs_scalar, 1, 1, 1)
                            self.tik_instance.vec_muls(1, value_temp,
                                                       value_temp, y_coeffs_scalar, 1, 1, 1)

                        self.resize_increment_value_bounded(
                            output_ub,
                            output_pos,
                            self.w_in,
                            self.h_in,
                            dst_scalar_input_x - 1 + i,
                            dst_scalar_input_y - 1 + j,
                            value_temp)

    # 'pylint: disable=too-many-arguments
    # 'pylint: disable=too-many-locals
    @staticmethod
    def area_pixel_compute_source_index(scale, dst_index, align_corners):
        """
        Compute index in input image according to scale and index of output image
        :param scale : float
        :param dst_index : int index of output image
        :param align_corners : bool
        Returns
        -------
        float
        """
        if align_corners:
            return scale * dst_index * 1.0
        return scale * (dst_index + 0.5) - 0.5

    def get_resize_bicubic_coefficients(self, output_scalar, input_scalar, in_length, out_length, flag):
        """
        Interpolation process of resize_bicubic
        :param output_scalar: int (output index)
        :param input_scalar: int (integer part of real_w or real_h)
        :param in_length: int (w_in or h_in)
        :param out_length: int (w_out or h_out)
        :param flag: int
        Returns
        ----------
        :return: tensor shape [4,]
        """
        coeffs_ub = self.tik_instance.Tensor("float32", (4,),
                                             name="coeffs_ub", scope=tik.scope_ubuf)
        temp_scalar = self.tik_instance.Scalar(dtype="int32", init_value=input_scalar)
        cast_input_index = self.tik_instance.Scalar(dtype="float32")
        self.tik_instance.scalar_conv('none', cast_input_index, temp_scalar)
        input_index_scalar = self.tik_instance.Scalar(dtype="float32", init_value=cast_input_index)
        one_length = self.tik_instance.Scalar(dtype="float32")
        two_length = self.tik_instance.Scalar(dtype="float32")
        three_length = self.tik_instance.Scalar(dtype="float32")
        x1 = self.tik_instance.Scalar(dtype="float32")
        a = self.cubic_coeff_a  # -0.75
        if flag == 0:
            scale = self.scale_0  # y direction
        else:
            scale = self.scale_1  # x direction
        with self.tik_instance.if_scope(out_length > 1):
            if self.align_corners:
                one_length.set_as(out_length - 1)
                two_length.set_as((out_length - 1) * (out_length - 1))
                three_length.set_as((out_length - 1) *
                                    (out_length - 1) * (out_length - 1))
                x1.set_as(output_scalar * (in_length - 1.0) -
                          input_index_scalar * one_length)
            else:
                with self.tik_instance.if_scope(scale > 0.0):
                    one_length.set_as(scale)
                    two_length.set_as(scale * scale)
                    three_length.set_as(scale * scale * scale)
                    x1.set_as((output_scalar + 0.5) -
                              (0.5 + input_index_scalar) * scale)
                with self.tik_instance.else_scope():
                    one_length.set_as(out_length)
                    two_length.set_as(out_length * out_length)
                    three_length.set_as(out_length * out_length * out_length)
                    x1.set_as((output_scalar + 0.5) * in_length -
                              (0.5 + input_index_scalar) * out_length)
            coeffs_ub[0].set_as((((a * (x1 + one_length) - 5.0 * a * one_length) * (x1 + one_length) +
                                  8.0 * a * two_length) * (x1 + one_length) - 4.0 * a * three_length) / three_length)
            coeffs_ub[1].set_as((((a + 2.0) * x1 - (a + 3.0) * one_length)
                                 * x1 * x1 + 1.0 * three_length) / three_length)
            x2 = one_length - x1
            coeffs_ub[2].set_as((((a + 2.0) * x2 - (a + 3.0) * one_length)
                                 * x2 * x2 + 1.0 * three_length) / three_length)
            coeffs_ub[3].set_as((((a * (x2 + one_length) - 5.0 * a * one_length) * (x2 + one_length) +
                                  8.0 * a * two_length) * (x2 + one_length) - 4.0 * a * three_length) / three_length)

        with self.tik_instance.else_scope():
            coeffs_ub[0].set_as(0.0)
            coeffs_ub[1].set_as(1.0)
            coeffs_ub[2].set_as(0.0)
            coeffs_ub[3].set_as(0.0)
        return coeffs_ub

    # 'pylint: disable=too-many-arguments
    # 'pylint: disable=too-many-locals
    def resize_increment_value_bounded(self, output_ub, output_pos, width, height, x, y, value):
        """
        Compute resize increment value bounded
        """
        x_scalar = self.tik_instance.Scalar(dtype="int64", init_value=x)
        y_scalar = self.tik_instance.Scalar(dtype="int64", init_value=y)
        w_scalar = self.tik_instance.Scalar(dtype="int64", init_value=0)
        h_scalar = self.tik_instance.Scalar(dtype="int64", init_value=0)
        offset = self.tik_instance.Scalar(dtype="int64")
        temp_ub = self.tik_instance.Tensor(self.dtype_grads, (1,),
                                           name="temp_ub", scope=tik.scope_ubuf)
        assist2_ub = self.tik_instance.Tensor("float32", (1,),
                                              name="assist2_ub", scope=tik.scope_ubuf)

        with self.tik_instance.if_scope(x_scalar < w_scalar):
            x_scalar.set_as(w_scalar)
        with self.tik_instance.if_scope(y_scalar < h_scalar):
            y_scalar.set_as(h_scalar)

        w_scalar.set_as(width - 1)
        h_scalar.set_as(height - 1)
        with self.tik_instance.if_scope(x_scalar > w_scalar):
            x_scalar.set_as(w_scalar)
        with self.tik_instance.if_scope(y_scalar > h_scalar):
            y_scalar.set_as(h_scalar)

        offset.set_as(output_pos + y_scalar * width + x_scalar)

        with self.tik_instance.if_scope(offset > self.pos_max_out):
            self.tik_instance.data_move(output_ub,
                                        self.output_gm[self.pos_max_out],
                                        0, 1, 1, 0, 0)
            pos_in_output_ub = offset - self.pos_max_out

            # Convert fp16 to fp32 to improve accuracy
            if self.dtype_grads == "float16":
                temp_ub[0].set_as(output_ub[pos_in_output_ub])
                self.tik_instance.vec_conv(
                    1, 'none', assist2_ub, temp_ub, 1, 1, 1)
                self.tik_instance.vec_add(1, assist2_ub,
                                          assist2_ub, value, 1, 1, 1, 1)
                self.tik_instance.vec_conv(
                    1, 'none', temp_ub, assist2_ub, 1, 1, 1)
                output_ub[pos_in_output_ub].set_as(temp_ub[0])
                self.tik_instance.data_move(self.output_gm[self.pos_max_out],
                                            output_ub,
                                            0, 1, 1, 0, 0)
            else:
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.output_gm[self.pos_max_out + pos_in_output_ub],
                                            value, 0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(0)
        with self.tik_instance.else_scope():
            # Convert fp16 to fp32 to improve accuracy
            if self.dtype_grads == "float16":
                self.tik_instance.data_move(output_ub,
                                            self.output_gm[offset],
                                            0, 1, 1, 0, 0)
                self.tik_instance.vec_conv(
                    1, 'none', assist2_ub, output_ub, 1, 1, 1)
                self.tik_instance.vec_add(1, assist2_ub,
                                          assist2_ub, value, 1, 1, 1, 1)
                self.tik_instance.vec_conv(
                    1, 'none', output_ub, assist2_ub, 1, 1, 1)
                self.tik_instance.data_move(self.output_gm[offset],
                                            output_ub,
                                            0, 1, 1, 0, 0)
            else:
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.output_gm[offset],
                                            value, 0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(0)

    def resize_bicubic_backward_operator(self):
        """
        Tiling mode 1: h_in = h_out and w_in = w_out, copy input to output
        Tiling mode 0: do cubic interpolation sampling
        """
        with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_1):
            self.resize_bicubic_backward_compute_same_size()
        with self.tik_instance.else_scope():
            self.resize_bicubic_backward_compute_general()

        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.core_num
            })
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_grads_gm],
            outputs=[self.output_gm],
            flowtable=[self.tiling_gm],
            config=opt_config)

        return self.tik_instance


class ResizeLinearBackward:
    """
    ResizeLinearBackward main functions
    """
    # 'pylint: disable=too-many-arguments
    def __init__(self, grads, original_size, scales, coordinate_transformation_mode="half_pixel",
                 kernel_name="resize_grad_d"):
        """
        Init parameters
        """
        self.tik_instance = tik.Tik()
        self.dtype_grads = grads.get("dtype")
        self.coordinate_transformation_mode = coordinate_transformation_mode
        self.kernel_name = kernel_name

        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        dtype_bytes_size = get_bit_len(self.dtype_grads) // Constant.EIGHT_BIT
        self.data_each_block = Constant.ONE_BLOCK_SIZE // dtype_bytes_size

        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_SCALAR_DTYPE, (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
        self.input_grads_gm = self.tik_instance.Tensor(self.dtype_grads, (Constant.MAX_INT64_VALUE,),
                                                       name="input_grads_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype_grads, (Constant.MAX_INT64_VALUE,),
                                                  name="output_gm", scope=tik.scope_gm)

        self.tiling_args()

    def tiling_args(self):
        """
        Get runtime params from tiling
        """
        self.grads_shape_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="grads_shape_num")
        self.ori_shape_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="ori_shape_num")
        self.n = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="n")
        self.c = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="c")
        self.w_out = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="w_out")
        self.h_in_redundancy = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="h_in_redundancy")
        self.w_in = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="w_in")
        self.w_scale = self.tik_instance.Scalar("float32", name="w_scale")
        self.scale = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="scale")
        self.tiling_core_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="tiling_core_num")

        tiling_ub = self.tik_instance.Tensor(Constant.TILING_SCALAR_DTYPE, [Constant.TILING_ARG_NUM, ],
                                             name="tiling_ub", scope=tik.scope_ubuf)
        burst_val = get_ceil_int(Constant.TILING_ARG_NUM, Constant.NUM_EACH_BURST)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, burst_val, 0, 0)

        self.grads_shape_num.set_as(tiling_ub[1])
        self.ori_shape_num.set_as(tiling_ub[2])
        self.n.set_as(tiling_ub[3])
        self.c.set_as(tiling_ub[4])
        self.w_out.set_as(tiling_ub[6])
        self.h_in_redundancy.set_as(tiling_ub[7])
        self.w_in.set_as(tiling_ub[8])
        self.w_scale.set_as(tiling_ub[10])
        self.scale.set_as(tiling_ub[12])
        self.tiling_core_num.set_as(tiling_ub[16])

    def init_output_gm_as_zero(self):
        temp_ub = self.tik_instance.Tensor(self.dtype_grads,
                                           [self.data_each_block, ],
                                           name="temp_ub",
                                           scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.data_each_block, temp_ub, 0, 1, 8)

        loop_time = self.tik_instance.Scalar("int32")
        loop_time = get_ceil_int(self.ori_shape_num, self.data_each_block)
        with self.tik_instance.for_range(0, loop_time) as i:
            self.tik_instance.data_move(self.output_gm[i * self.data_each_block], temp_ub, 0, 1, 1, 0, 0)

    def compute_helper_backward(self, w_scale, output_block_offset, index_in_gm, input_dim_offset):
        # cal real_w
        real_w = self.tik_instance.Scalar("float32", name="real_w")
        k = self.tik_instance.Scalar("float32", init_value=output_block_offset)
        temp_w = self.tik_instance.Scalar("float32")
        with self.tik_instance.if_scope(self.coordinate_transformation_mode == "align_corners"):
            temp_w.set_as(w_scale * k)
        with self.tik_instance.else_scope():
            temp_scalar = self.tik_instance.Scalar(dtype="float32", init_value=w_scale * (k + 0.5) - 0.5)
            with self.tik_instance.if_scope(temp_scalar < 0):
                temp_w.set_as(0.)
            with self.tik_instance.else_scope():
                temp_w.set_as(temp_scalar)
        real_w.set_as(temp_w)

        # cal Integer of real_w
        coefficient_w = self.tik_instance.Scalar("int32", name="coefficient_w")
        self.tik_instance.scalar_conv('floor', coefficient_w, real_w)

        # cal Offset
        offset = self.tik_instance.Scalar(dtype="int32", name="offset", init_value=1)
        with self.tik_instance.if_scope(coefficient_w == (self.w_in - 1)):
            offset.set_as(0)

        # cal Decimal of real_w
        coefficient_lambda_1 = self.tik_instance.Scalar("float32", name="coefficient_lambda_1")
        coefficient_lambda_1.set_as(real_w - coefficient_w)

        # Cal 1.0 - Decimal of real_w
        coefficient_lambda_0 = self.tik_instance.Scalar("float32", name="coefficient_lambda_0")
        coefficient_lambda_0.set_as(1.0 - coefficient_lambda_1)

        _x = self.get_number_in_global_memory(index_in_gm)

        self.set_output_as(input_dim_offset + coefficient_w, coefficient_lambda_0 * _x)
        self.set_output_as(input_dim_offset + coefficient_w + offset, coefficient_lambda_1 * _x)

    def get_number_in_global_memory(self, index):
        """
        Get the value with given index from input tensor (in global memory)
        """
        res = self.tik_instance.Scalar(self.dtype_grads, name="res")
        index = self.tik_instance.Scalar("int32", name="index", init_value=index)
        max_offset = self.tik_instance.Scalar(dtype="int32", name="max_offset", init_value=0)
        with self.tik_instance.if_scope(self.grads_shape_num > self.data_each_block):
            max_offset.set_as(self.grads_shape_num - self.data_each_block)

        x_ub = self.tik_instance.Tensor(self.dtype_grads, [self.data_each_block], name="x_ub", scope=tik.scope_ubuf)

        with self.tik_instance.if_scope(index < max_offset):
            self.tik_instance.data_move(x_ub, self.input_grads_gm[index], 0, 1, 1, 0, 0)
            res.set_as(x_ub[0])
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(x_ub, self.input_grads_gm[max_offset], 0, 1, 1, 0, 0)
            res.set_as(x_ub[index - max_offset])

        return res

    def set_output_as(self, index, num):
        block_num = index // self.data_each_block
        block_offset = index % self.data_each_block

        temp_ub = self.tik_instance.Tensor(self.dtype_grads,
                                           [self.data_each_block, ],
                                           name="temp_ub",
                                           scope=tik.scope_ubuf)

        self.tik_instance.data_move(temp_ub, self.output_gm[block_num * self.data_each_block], 0, 1, 1, 0, 0)

        temp_scalar = self.tik_instance.Scalar(self.dtype_grads, init_value=temp_ub[block_offset])
        temp_ub[block_offset].set_as(temp_scalar + num)
        self.tik_instance.data_move(self.output_gm[block_num * self.data_each_block], temp_ub, 0, 1, 1, 0, 0)

    def resize_linear_operator(self):
        self.input_grads_gm.reshape([self.grads_shape_num, ])
        self.init_output_gm_as_zero()

        with self.tik_instance.for_range(0, self.n) as i:
            with self.tik_instance.for_range(0, self.c) as j:
                with self.tik_instance.for_range(0, self.w_out) as k:
                    self.compute_helper_backward(self.w_scale,
                                                 k,
                                                 (i * (self.c * self.w_out)) + (j * self.w_out) + k,
                                                 i * (self.c * self.w_in) + j * self.w_in)

        self.output_gm.reshape([self.n, self.c, 1, self.w_in])

        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.core_num
            })
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_grads_gm],
            outputs=[self.output_gm],
            flowtable=[self.tiling_gm],
            config=opt_config)
        return self.tik_instance


# 'pylint: disable=unused-argument
@register_operator("ResizeGradD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_FLOAT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def resize_grad_d(grads, y, original_size, roi, scales, coordinate_transformation_mode="half_pixel",
                  cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0.0,
                  mode="nearest", nearest_mode="round_prefer_floor", data_format="NCHW", kernel_name="resize_grad_d"):
    """
    Interface of resize_grad_d, compatible to PyTorch API nn.upsample
    Parameters
    ----------
    grads : dict
        shape and dtype of input
    y : dict
        shape and dtype of output
    original_size : list_int
        shape of original_image
    roi : list_float
        1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X.
        The RoIs' coordinates are normalized in the coordinate system of the input image.
        It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"
    scales : list_float
        The scale array along each dimension.
        It takes value greater than 0.
        If it's less than 1, it's sampling down, otherwise, it's upsampling.
        The number of elements of 'scales' should be the same as the rank of input 'X'.
        Only one of 'scales' and 'sizes' can be specified. If 'size' is specified,
        then set scales to empty data (zero shape) in this operator's input list.
    coordinate_transformation_mode : str
        This attribute describes how to transform the coordinate in the resized tensor to
        the coordinate in the original tensor.
    cubic_coeff_a : float
        The coefficient 'a' used in cubic interpolation.
        Two common choice are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch).
        This attribute is valid only if "mode" is "cubic".
    exclude_outside : int
        default is 0.
        If set to 1, the weight of sampling locations outside the tensor will be set to 0
        and the weight will be renormalized so that their sum is 1.0. The default value is 0.
    extrapolation_value : float (default is 0.0)
        When coordinate_transformation_mode is "tf_crop_and_resize" and x_original is outside
        the range [0, length_original - 1], this value is used as the corresponding output value.
        Default is 0.0f.
    mode : string
        default is nearest
        Three interpolation modes: nearest (default), linear and cubic.
        The "linear" mode includes linear interpolation for 1D tensor and N-linear interpolation
        for N-D tensor (for example, bilinear interpolation for 2D tensor).
        The "cubic" mode includes cubic interpolation for 1D tensor and N-cubic interpolation
        for N-D tensor (for example, bicubic interpolation for 2D tensor).
    nearest_mode : string (default is round_prefer_floor)
        Four modes: round_prefer_floor (default, as known as round half down),
        round_prefer_ceil (as known as round half up), floor, ceil.
        Only used by nearest interpolation.
        It indicates how to get "nearest" pixel in input tensor from x_original,
        so this attribute is valid only if "mode" is "nearest".
    data_format : str
        NCHW or HWNC
    kernel_name : str
        kernel name, default value is "resize_grad_d"

    Returns
    -------
    tik_instance
    """
    check_tuple = ("float32", "float16")
    input_data_type = grads.get("dtype").lower()
    para_check.check_dtype_rule(input_data_type, check_tuple, "DataType")
    para_check.check_kernel_name(kernel_name)
    if mode == "cubic":
        if data_format == "HWNC":
            resize_bicubic_backward_instance = UpSampleBicubic2dBackward(
                grads, original_size, scales,
                coordinate_transformation_mode,
                cubic_coeff_a,
                kernel_name=kernel_name)
            res = resize_bicubic_backward_instance.upsamplebicubic2d_backward_compute()
        else:
            resize_bicubic_backward_instance = ResizeBicubicBackward(
                grads, original_size, scales,
                coordinate_transformation_mode,
                cubic_coeff_a,
                kernel_name=kernel_name)
            res = resize_bicubic_backward_instance.resize_bicubic_backward_operator()
    elif mode == "linear":
        resize_linear = ResizeLinearBackward(
            grads,
            original_size,
            scales,
            coordinate_transformation_mode,
            kernel_name=kernel_name)
        res = resize_linear.resize_linear_operator()
    else:
        raise RuntimeError("Upsample Not supported.")
    return res