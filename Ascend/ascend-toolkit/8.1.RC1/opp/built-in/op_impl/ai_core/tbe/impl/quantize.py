#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
quantize
"""

import math
import functools as fctool
from impl.util.platform_adapter import tbe_platform as tp
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik


# 'pylint: disable=invalid-name, too-many-branches, too-many-arguments
# 'pylint: disable=too-many-instance-attributes, too-many-lines, too-many-statements
class Quantize():
    """
    impl of quantize
    """
    def __init__(self, input_x, scales, zero_points, axis, dtype, kernel_name):
        """
        init
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.input_x_shape = list(input_x.get("shape"))
        self.input_x_dtype = input_x.get("dtype")
        self.scales_shape = scales.get("shape")
        self.scales_dtype = scales.get("dtype")
        self.zero_points_shape = zero_points.get("shape")
        self.zero_points_dtype = zero_points.get("dtype")
        self.x_need_cast = False
        if self.input_x_dtype == "float16":
            self.x_need_cast = True
        self.max_mask_x_dtype = 256 // 4
        self.dim = axis
        self.kernel_name = kernel_name
        self.dtype_bytes_size_x = tp.get_bit_len(self.input_x_dtype) // 8
        self.data_each_block_x = 32 // self.dtype_bytes_size_x
        self.dtype_bytes_size_scales = tp.get_bit_len(self.scales_dtype) // 8
        self.data_each_block_scales = 32 // self.dtype_bytes_size_scales
        self.dtype_bytes_size_zero_points = tp.get_bit_len(self.zero_points_dtype) // 8
        self.data_each_block_zero_points = 32 // self.dtype_bytes_size_zero_points
        self.determine_upper_and_lower_bounds(dtype)
        self.dtype_bytes_size_y = tp.get_bit_len(self.output_y_dtype) // 8
        self.data_each_block_y = 32 // self.dtype_bytes_size_y
        self.dtype32 = 4
        self.dtype16 = 2
        self.dtype8 = 1
        self.per_tensor = False
        if self.scales_shape[0] == 1 and len(self.scales_shape) == 1:
            self.per_tensor = True
        if self.per_tensor:
            if self.zero_points_dtype != "int32":
                ext_size = 32 * 5 + 10 * 256
            else:
                ext_size = 32 * 4 + 10 * 256
            self.ub_size = tp.get_soc_spec(tp.UB_SIZE) - ext_size * 2
            if self.x_need_cast:
                divider = self.dtype_bytes_size_x + self.dtype32 * 3 + self.dtype16 + self.dtype8
                self.ub_tensor_size = self.ub_size // divider // 2 // self.max_mask_x_dtype * self.max_mask_x_dtype
            else:
                divider = self.dtype_bytes_size_x + self.dtype32 * 2 + self.dtype16 + self.dtype8
                self.ub_tensor_size = self.ub_size // divider // 2 // self.max_mask_x_dtype * self.max_mask_x_dtype
        else:
            ext_size = 8 * 256
            self.ub_size = tp.get_soc_spec(tp.UB_SIZE) - ext_size * 2
            if self.input_x_dtype == "float16":
                if self.zero_points_dtype != "int32":
                    divider = self.dtype_bytes_size_x + self.dtype32 * 9 + self.dtype16 * 2 + self.dtype8
                else:
                    divider = self.dtype_bytes_size_x + self.dtype32 * 9 + self.dtype16 + self.dtype8
                self.ub_tensor_size = self.ub_size // divider // 2 // self.max_mask_x_dtype * self.max_mask_x_dtype
            else:
                if self.zero_points_dtype != "int32":
                    divider = self.dtype_bytes_size_x + self.dtype32 * 8 + self.dtype16 * 2 + self.dtype8
                else:
                    divider = self.dtype_bytes_size_x + self.dtype32 * 8 + self.dtype16 + self.dtype8
                self.ub_tensor_size = self.ub_size // divider // 2 // self.max_mask_x_dtype * self.max_mask_x_dtype

        self.input_x_gm = self.tik_instance.Tensor(self.input_x_dtype,
                                                   self.input_x_shape,
                                                   name="input_x_gm",
                                                   scope=tik.scope_gm)
        self.input_zero_points_gm = self.tik_instance.Tensor(self.zero_points_dtype,
                                                             self.zero_points_shape,
                                                             name="input_zero_points_gm",
                                                             scope=tik.scope_gm)
        self.input_scales_gm = self.tik_instance.Tensor(self.scales_dtype,
                                                        self.scales_shape,
                                                        name="input_scales_gm",
                                                        scope=tik.scope_gm)
        self.output_y_gm = self.tik_instance.Tensor(self.output_y_dtype,
                                                    self.output_y_shape,
                                                    name="output_y_gm",
                                                    scope=tik.scope_gm)
        self.in_num = fctool.reduce(lambda x, y: x * y, self.input_x_shape)
        if self.in_num <= self.ub_tensor_size:
            self.ai_core_num = 1
        else:
            self.ai_core_num = tp.get_soc_spec(tp.CORE_NUM)
        if self.ai_core_num == 1:
            self.num_each_core = self.in_num // self.ai_core_num
        else:
            self.num_each_core = self.in_num // self.ai_core_num // self.max_mask_x_dtype * self.max_mask_x_dtype
        self.last_core_num = self.in_num - self.num_each_core * self.ai_core_num

    def determine_upper_and_lower_bounds(self, dtype):
        """
        get upper and lower bounds by dtype
        """
        type_dict = {"torch.qint8": "int8", "torch.quint8": "uint8", "torch.qint32": "int32"}
        self.output_y_dtype = type_dict[dtype]
        self.output_y_shape = self.input_x_shape
        if self.output_y_dtype == "int8":
            self.min_value = -128
            self.max_value = 127
        elif self.output_y_dtype == "uint8":
            self.min_value = 0
            self.max_value = 255
        else:
            self.min_value = -2147483648
            self.max_value = 2147483647

    def quantize_compute(self):
        """
        Calculate total entrance
        """
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as core_id:
            move_offset = self.num_each_core * core_id
            self.quantize_compute_each_core(move_offset, self.num_each_core)
        move_offset = self.num_each_core * self.ai_core_num
        if self.last_core_num > 0:
            self.quantize_compute_each_core(move_offset, self.last_core_num)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_x_gm, self.input_scales_gm, self.input_zero_points_gm],
                                   outputs=[self.output_y_gm])
        return self.tik_instance

    def quantize_compute_each_core(self, core_move_offset, core_move_num):
        """
        Calculation on each core
        """
        loop_time = core_move_num // self.ub_tensor_size
        move_offset = core_move_offset
        need_db = True
        if loop_time < 2:
            need_db = False
        if loop_time > 0:
            if need_db:
                with self.tik_instance.for_range(0, loop_time, thread_num=2) as loop_id:
                    move_offset = loop_id * self.ub_tensor_size + core_move_offset
                    self.quantize_compute_each_loop(move_offset, self.ub_tensor_size)
            else:
                with self.tik_instance.for_range(0, loop_time) as loop_id:
                    move_offset = loop_id * self.ub_tensor_size + core_move_offset
                    self.quantize_compute_each_loop(move_offset, self.ub_tensor_size)
            move_offset = loop_time * self.ub_tensor_size + core_move_offset
        last_num = core_move_num % self.ub_tensor_size
        if last_num > 0:
            self.quantize_compute_each_loop(move_offset, last_num)

    def quantize_compute_each_loop(self, move_offset, move_num):
        """
        Compute each loop.
        move_num <= ub_tensor_size
        """
        self.init_ub_tensor_and_scalar()
        burse_len_x = math.ceil(move_num / self.data_each_block_x)
        burse_len_y = math.ceil(move_num / self.data_each_block_y)
        self.tik_instance.data_move(self.input_x_ub, self.input_x_gm[move_offset], 0, 1, burse_len_x, 0, 0)
        if self.x_need_cast:
            self.x_to_fp32(move_num)
        if self.per_tensor:
            self.tik_instance.data_move(self.input_scales_ub, self.input_scales_gm, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.input_zero_points_ub, self.input_zero_points_gm, 0, 1, 1, 0, 0)
            if self.zero_points_dtype != "int32":
                # zp from int8/uint8 to int32
                self.tik_instance.vec_conv(self.data_each_block_zero_points, '', self.input_zero_points_ub_fp16,
                                           self.input_zero_points_ub, 1, 8, 8)
                self.tik_instance.vec_conv(self.data_each_block_zero_points, '', self.input_zero_points_ub_fp32,
                                           self.input_zero_points_ub_fp16, 1, 8, 8)
            else:
                # zp from int32 to fp32
                self.tik_instance.vec_conv(self.data_each_block_zero_points, '', self.input_zero_points_ub_fp32,
                                           self.input_zero_points_ub, 1, 8, 8)
            # scale to 1/scale
            self.tik_instance.vec_rec_high_preci(self.data_each_block_scales, self.input_scales_ub_fp32_t,
                                                 self.input_scales_ub, self.wk_tensor, 1, 8, 8)
            self.ori_scale.set_as(self.input_scales_ub[0])
            self.scale_scalar_fp32.set_as(self.input_scales_ub_fp32_t[0])
            self.zp_scalar_fp32.set_as(self.input_zero_points_ub_fp32[0])
            self.quantize_per_tensor(move_num)
            with self.tik_instance.if_scope(self.ori_scale == 0):
                self.per_tensor_x_zero(move_num)
            self.fp32x_to_int32(move_num)
            self.min_max_tensor(move_num)
        else:
            burse_len_scale = math.ceil(move_num / self.data_each_block_scales)
            burse_len_zp = math.ceil(move_num / self.data_each_block_zero_points)
            self.tik_instance.data_move(self.input_scales_ub, self.input_scales_gm[move_offset], 0, 1, burse_len_scale,
                                        0, 0)
            self.tik_instance.data_move(self.input_zero_points_ub, self.input_zero_points_gm[move_offset], 0, 1,
                                        burse_len_zp, 0, 0)
            # zp from int32 to fp32
            self.zp_to_fp32(move_num)
            # scale to 1/scale
            self.scale_t(move_num)
            self.quantize_per_channel(move_num)
            self.per_channel_x_zero(move_num)
            self.fp32x_to_int32(move_num)
            self.min_max_tensor(move_num)
        if self.output_y_dtype == "int32":
            self.tik_instance.data_move(self.output_y_gm[move_offset], self.int32_tensor, 0, 1, burse_len_y, 0, 0)
        else:
            self.y_int32_to_fp16(move_num)
            self.y_fp16_to_int(move_num)
            self.tik_instance.data_move(self.output_y_gm[move_offset], self.output_y_int, 0, 1, burse_len_y, 0, 0)

    def per_channel_x_zero(self, num):
        """
        check if x and scale are all 0, if so, take a specific value
        :param num: need dealed num
        :return:
        """
        loop = num // self.max_mask_x_dtype
        for loop_id in range(loop):
            compute_offset = loop_id * self.max_mask_x_dtype
            if self.x_need_cast:
                self.tik_instance.vec_cmpv_eq(self.is_x_zero, self.input_x_ub_fp32[compute_offset],
                                              self.zero_compare_tensor, 1, 8, 0)
            else:
                self.tik_instance.vec_cmpv_eq(self.is_x_zero, self.input_x_ub[compute_offset], self.zero_compare_tensor,
                                              1, 8, 0)
            self.tik_instance.vec_cmpv_eq(self.is_scale_zero, self.input_scales_ub[compute_offset],
                                          self.zero_compare_tensor, 1, 8, 0)
            self.tik_instance.vec_and(self.max_mask_x_dtype, self.is_zero, self.is_x_zero, self.is_scale_zero, 1, 8, 8,
                                      8)
            self.tik_instance.vec_sel(self.max_mask_x_dtype, 0, self.tmp_x_ub[compute_offset], self.is_zero,
                                      self.min_vale_tensor, self.tmp_x_ub[compute_offset], 1, 8, 8)
        compute_offset = loop * self.max_mask_x_dtype
        last_num = num % self.max_mask_x_dtype
        if last_num > 0:
            if self.x_need_cast:
                self.tik_instance.vec_cmpv_eq(self.is_x_zero, self.input_x_ub_fp32[compute_offset],
                                              self.zero_compare_tensor, 1, 8, 0)
            else:
                self.tik_instance.vec_cmpv_eq(self.is_x_zero, self.input_x_ub[compute_offset], self.zero_compare_tensor,
                                              1, 8, 0)
            self.tik_instance.vec_cmpv_eq(self.is_scale_zero, self.input_scales_ub[compute_offset],
                                          self.zero_compare_tensor, 1, 8, 0)
            self.tik_instance.vec_and(last_num, self.is_zero, self.is_x_zero, self.is_scale_zero, 1, 8, 8, 8)
            self.tik_instance.vec_sel(last_num, 0, self.tmp_x_ub[compute_offset], self.is_zero, self.min_vale_tensor,
                                      self.tmp_x_ub[compute_offset], 1, 8, 8)

    def quantize_per_channel(self, num):
        """
        quantize per channel: x * scale + zp
        :param num: need dealed num
        :return:
        """
        loop = num // (self.max_mask_x_dtype * 255)
        if self.ub_tensor_size >= self.max_mask_x_dtype * 255:
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * self.max_mask_x_dtype * 255
                    if self.x_need_cast:
                        self.tik_instance.vec_mul(self.max_mask_x_dtype, self.tmp_x_ub[compute_offset],
                                                  self.input_x_ub_fp32[compute_offset],
                                                  self.input_scales_ub_fp32_t[compute_offset], 255, 8, 8, 8)
                    else:
                        self.tik_instance.vec_mul(self.max_mask_x_dtype, self.tmp_x_ub[compute_offset],
                                                  self.input_x_ub[compute_offset],
                                                  self.input_scales_ub_fp32_t[compute_offset], 255, 8, 8, 8)
                    self.tik_instance.vec_add(self.max_mask_x_dtype, self.tmp_x_ub[compute_offset],
                                              self.tmp_x_ub[compute_offset],
                                              self.input_zero_points_ub_fp32[compute_offset], 255, 8, 8, 8)
        compute_offset = loop * self.max_mask_x_dtype * 255
        repeat_time = num % (self.max_mask_x_dtype * 255) // self.max_mask_x_dtype
        if repeat_time > 0:
            if self.x_need_cast:
                self.tik_instance.vec_mul(self.max_mask_x_dtype, self.tmp_x_ub[compute_offset],
                                          self.input_x_ub_fp32[compute_offset],
                                          self.input_scales_ub_fp32_t[compute_offset], repeat_time, 8, 8, 8)
            else:
                self.tik_instance.vec_mul(self.max_mask_x_dtype, self.tmp_x_ub[compute_offset],
                                          self.input_x_ub[compute_offset], self.input_scales_ub_fp32_t[compute_offset],
                                          repeat_time, 8, 8, 8)
            self.tik_instance.vec_add(self.max_mask_x_dtype, self.tmp_x_ub[compute_offset],
                                      self.tmp_x_ub[compute_offset], self.input_zero_points_ub_fp32[compute_offset],
                                      repeat_time, 8, 8, 8)
        compute_offset += repeat_time * self.max_mask_x_dtype
        last_num = num % self.max_mask_x_dtype
        if last_num > 0:
            if self.x_need_cast:
                self.tik_instance.vec_mul(last_num, self.tmp_x_ub[compute_offset], self.input_x_ub_fp32[compute_offset],
                                          self.input_scales_ub_fp32_t[compute_offset], 1, 8, 8, 8)
            else:
                self.tik_instance.vec_mul(last_num, self.tmp_x_ub[compute_offset], self.input_x_ub[compute_offset],
                                          self.input_scales_ub_fp32_t[compute_offset], 1, 8, 8, 8)
            self.tik_instance.vec_add(last_num, self.tmp_x_ub[compute_offset], self.tmp_x_ub[compute_offset],
                                      self.input_zero_points_ub_fp32[compute_offset], 1, 8, 8, 8)

    def scale_t(self, num):
        """
        get 1/scale
        :param num: need dealed num
        :return:
        """
        loop = num // (self.max_mask_x_dtype * 255)
        if self.ub_tensor_size >= self.max_mask_x_dtype * 255:
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * self.max_mask_x_dtype * 255
                    self.tik_instance.vec_rec_high_preci(self.max_mask_x_dtype,
                                                         self.input_scales_ub_fp32_t[compute_offset:],
                                                         self.input_scales_ub[compute_offset:], self.wk_tensor, 255, 8,
                                                         8)
        compute_offset = loop * self.max_mask_x_dtype * 255
        repeat_time = num % (self.max_mask_x_dtype * 255) // self.max_mask_x_dtype
        if repeat_time > 0:
            self.tik_instance.vec_rec_high_preci(self.max_mask_x_dtype, self.input_scales_ub_fp32_t[compute_offset:],
                                                 self.input_scales_ub[compute_offset:], self.wk_tensor, repeat_time, 8,
                                                 8)
        compute_offset += repeat_time * self.max_mask_x_dtype
        last_num = num % self.max_mask_x_dtype
        if last_num > 0:
            self.tik_instance.vec_rec_high_preci(last_num, self.input_scales_ub_fp32_t[compute_offset:],
                                                 self.input_scales_ub[compute_offset:], self.wk_tensor, 1, 8, 8)

    def per_tensor_x_zero(self, num):
        """
        check if x and scale are all 0, if so, take a specific value
        :param num: need dealed num
        :return:
        """
        loop = num // self.max_mask_x_dtype
        for loop_id in range(loop):
            compute_offset = loop_id * self.max_mask_x_dtype
            if self.x_need_cast:
                self.tik_instance.vec_cmpv_eq(self.is_x_zero, self.input_x_ub_fp32[compute_offset],
                                              self.zero_compare_tensor, 1, 8, 0)
            else:
                self.tik_instance.vec_cmpv_eq(self.is_x_zero, self.input_x_ub[compute_offset], self.zero_compare_tensor,
                                              1, 8, 0)
            self.tik_instance.vec_sel(self.max_mask_x_dtype, 0, self.tmp_x_ub[compute_offset], self.is_x_zero,
                                      self.min_vale_tensor, self.tmp_x_ub[compute_offset], 1, 8, 8)
        compute_offset = loop * self.max_mask_x_dtype
        last_num = num % self.max_mask_x_dtype
        if last_num > 0:
            if self.x_need_cast:
                self.tik_instance.vec_cmpv_eq(self.is_x_zero, self.input_x_ub_fp32[compute_offset],
                                              self.zero_compare_tensor, 1, 8, 0)
            else:
                self.tik_instance.vec_cmpv_eq(self.is_x_zero, self.input_x_ub[compute_offset], self.zero_compare_tensor,
                                              1, 8, 8)
            self.tik_instance.vec_sel(last_num, 0, self.tmp_x_ub[compute_offset], self.is_x_zero, self.min_vale_tensor,
                                      self.tmp_x_ub[compute_offset], 1, 8, 8)

    def x_to_fp32(self, num):
        """
        fp16 to fp32
        :param num: need dealed num
        :return:
        """
        loop = num // (self.max_mask_x_dtype * 255)
        if self.ub_tensor_size >= self.max_mask_x_dtype * 255:
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * self.max_mask_x_dtype * 255
                    self.tik_instance.vec_conv(self.max_mask_x_dtype, '', self.input_x_ub_fp32[compute_offset],
                                               self.input_x_ub[compute_offset], 255, 8, 4)
        compute_offset = loop * self.max_mask_x_dtype * 255
        repeat_time = num % (self.max_mask_x_dtype * 255) // self.max_mask_x_dtype
        if repeat_time > 0:
            self.tik_instance.vec_conv(self.max_mask_x_dtype, '', self.input_x_ub_fp32[compute_offset],
                                       self.input_x_ub[compute_offset], repeat_time, 8, 4)
        compute_offset += repeat_time * self.max_mask_x_dtype
        last_num = num % self.max_mask_x_dtype
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, '', self.input_x_ub_fp32[compute_offset],
                                       self.input_x_ub[compute_offset], 1, 8, 4)

    def zp_to_fp32(self, num):
        """
        int32/int8/uint8 to fp32
        :param num: need dealed num
        :return:
        """
        if self.zero_points_dtype == "int32":
            loop = num // (self.max_mask_x_dtype * 255)
            if self.ub_tensor_size >= self.max_mask_x_dtype * 255:
                if loop > 0:
                    for index in range(loop):
                        compute_offset = index * self.max_mask_x_dtype * 255
                        self.tik_instance.vec_conv(self.max_mask_x_dtype, '',
                                                   self.input_zero_points_ub_fp32[compute_offset],
                                                   self.input_zero_points_ub[compute_offset], 255, 8, 8)
            compute_offset = loop * self.max_mask_x_dtype * 255
            repeat_time = num % (self.max_mask_x_dtype * 255) // self.max_mask_x_dtype
            if repeat_time > 0:
                self.tik_instance.vec_conv(self.max_mask_x_dtype, '', self.input_zero_points_ub_fp32[compute_offset],
                                           self.input_zero_points_ub[compute_offset], repeat_time, 8, 8)
            compute_offset += repeat_time * self.max_mask_x_dtype
            last_num = num % self.max_mask_x_dtype
            if last_num > 0:
                self.tik_instance.vec_conv(last_num, '', self.input_zero_points_ub_fp32[compute_offset],
                                           self.input_zero_points_ub[compute_offset], 1, 8, 8)
        else:
            # int -> fp16 -> fp32
            self.zp_int_to_fp16(num)
            self.zp_fp16_to_fp32(num)

    def zp_int_to_fp16(self, num):
        """
        zp int8/uint8 -> fp16
        :param num: need dealed num
        :return:
        """
        mask_num = 256 // 2
        loop = num // (mask_num * 255)
        if self.ub_tensor_size >= mask_num * 255:
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * mask_num * 255
                    self.tik_instance.vec_conv(mask_num, '', self.input_zero_points_ub_fp16[compute_offset],
                                               self.input_zero_points_ub[compute_offset], 255, 8, 4)
        compute_offset = loop * mask_num * 255
        repeat_time = num % (mask_num * 255) // mask_num
        if repeat_time > 0:
            self.tik_instance.vec_conv(mask_num, '', self.input_zero_points_ub_fp16[compute_offset],
                                       self.input_zero_points_ub[compute_offset], repeat_time, 8, 4)
        compute_offset += repeat_time * mask_num
        last_num = num % mask_num
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, '', self.input_zero_points_ub_fp16[compute_offset],
                                       self.input_zero_points_ub[compute_offset], 1, 8, 4)

    def zp_fp16_to_fp32(self, num):
        """
        zp fp16 -> fp32
        :param num: need dealed num
        :return:
        """
        loop = num // (self.max_mask_x_dtype * 255)
        if self.ub_tensor_size >= self.max_mask_x_dtype * 255:
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * self.max_mask_x_dtype * 255
                    self.tik_instance.vec_conv(self.max_mask_x_dtype, '',
                                               self.input_zero_points_ub_fp32[compute_offset],
                                               self.input_zero_points_ub_fp16[compute_offset], 255, 8, 4)
        compute_offset = loop * self.max_mask_x_dtype * 255
        repeat_time = num % (self.max_mask_x_dtype * 255) // self.max_mask_x_dtype
        if repeat_time > 0:
            self.tik_instance.vec_conv(self.max_mask_x_dtype, '', self.input_zero_points_ub_fp32[compute_offset],
                                       self.input_zero_points_ub_fp16[compute_offset], repeat_time, 8, 4)
        compute_offset += repeat_time * self.max_mask_x_dtype
        last_num = num % self.max_mask_x_dtype
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, '', self.input_zero_points_ub_fp32[compute_offset],
                                       self.input_zero_points_ub_fp16[compute_offset], 1, 8, 4)

    def y_int32_to_fp16(self, num):
        """
        int32 to fp16
        :param num: need dealed num
        :return:
        """
        loop = num // (self.max_mask_x_dtype * 255)
        if self.ub_tensor_size >= self.max_mask_x_dtype * 255:
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * self.max_mask_x_dtype * 255
                    self.tik_instance.vec_conv(self.max_mask_x_dtype, '', self.output_y_fp16[compute_offset],
                                               self.int32_tensor[compute_offset], 255, 4, 8, 1.0)
        compute_offset = loop * self.max_mask_x_dtype * 255
        repeat_time = num % (self.max_mask_x_dtype * 255) // self.max_mask_x_dtype
        if repeat_time > 0:
            self.tik_instance.vec_conv(self.max_mask_x_dtype, '', self.output_y_fp16[compute_offset],
                                       self.int32_tensor[compute_offset], repeat_time, 4, 8, 1.0)
        compute_offset += repeat_time * self.max_mask_x_dtype
        last_num = num % self.max_mask_x_dtype
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, '', self.output_y_fp16[compute_offset],
                                       self.int32_tensor[compute_offset], 1, 4, 8, 1.0)

    def y_fp16_to_int(self, num):
        """
        fp16 to int8/uint8
        :param num: need dealed num
        :return:
        """
        max_mask = 256 // 2
        loop = num // (max_mask * 255)
        if self.ub_tensor_size >= max_mask * 255:
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * max_mask * 255
                    self.tik_instance.vec_conv(max_mask, '', self.output_y_int[compute_offset],
                                               self.output_y_fp16[compute_offset], 255, 4, 8)
        compute_offset = loop * max_mask * 255
        repeat_time = num % (max_mask * 255) // max_mask
        if repeat_time > 0:
            self.tik_instance.vec_conv(max_mask, '', self.output_y_int[compute_offset],
                                       self.output_y_fp16[compute_offset], repeat_time, 4, 8)
        compute_offset += repeat_time * max_mask
        last_num = num % max_mask
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, '', self.output_y_int[compute_offset],
                                       self.output_y_fp16[compute_offset], 1, 4, 8)

    def fp32x_to_int32(self, num):
        """
        fp32 to int32
        :param num: need dealed num
        :return:
        """
        loop = num // (self.max_mask_x_dtype * 255)
        if self.ub_tensor_size >= self.max_mask_x_dtype * 255:
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * self.max_mask_x_dtype * 255
                    self.tik_instance.vec_conv(self.max_mask_x_dtype, 'round', self.int32_tensor[compute_offset],
                                               self.tmp_x_ub[compute_offset], 255, 8, 8)
        compute_offset = loop * self.max_mask_x_dtype * 255
        repeat_time = num % (self.max_mask_x_dtype * 255) // self.max_mask_x_dtype
        if repeat_time > 0:
            self.tik_instance.vec_conv(self.max_mask_x_dtype, 'round', self.int32_tensor[compute_offset],
                                       self.tmp_x_ub[compute_offset], repeat_time, 8, 8)
        compute_offset += repeat_time * self.max_mask_x_dtype
        last_num = num % self.max_mask_x_dtype
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, 'round', self.int32_tensor[compute_offset],
                                       self.tmp_x_ub[compute_offset], 1, 8, 8)

    def min_max_tensor(self, num):
        """
        Limit element upper and lower bounds
        :param num: need dealed num
        :return:
        """
        loop = num // (self.max_mask_x_dtype * 255)
        if self.ub_tensor_size >= self.max_mask_x_dtype * 255:
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * self.max_mask_x_dtype * 255
                    self.tik_instance.vec_min(self.max_mask_x_dtype, self.int32_tensor[compute_offset],
                                              self.int32_tensor[compute_offset], self.compare_int32_max_tensor, 255, 8,
                                              8, 0)
                    self.tik_instance.vec_max(self.max_mask_x_dtype, self.int32_tensor[compute_offset],
                                              self.int32_tensor[compute_offset], self.compare_int32_min_tensor, 255, 8,
                                              8, 0)
        compute_offset = loop * self.max_mask_x_dtype * 255
        repeat_time = num % (self.max_mask_x_dtype * 255) // self.max_mask_x_dtype
        if repeat_time > 0:
            self.tik_instance.vec_min(self.max_mask_x_dtype, self.int32_tensor[compute_offset],
                                      self.int32_tensor[compute_offset], self.compare_int32_max_tensor, repeat_time, 8,
                                      8, 0)
            self.tik_instance.vec_max(self.max_mask_x_dtype, self.int32_tensor[compute_offset],
                                      self.int32_tensor[compute_offset], self.compare_int32_min_tensor, repeat_time, 8,
                                      8, 0)
        compute_offset += repeat_time * self.max_mask_x_dtype
        last_num = num % self.max_mask_x_dtype
        if last_num > 0:
            self.tik_instance.vec_min(last_num, self.int32_tensor[compute_offset], self.int32_tensor[compute_offset],
                                      self.compare_int32_max_tensor, 1, 8, 8, 0)
            self.tik_instance.vec_max(last_num, self.int32_tensor[compute_offset], self.int32_tensor[compute_offset],
                                      self.compare_int32_min_tensor, 1, 8, 8, 0)

    def quantize_per_tensor(self, num):
        """
        quantize per tensor
        x * scale + zp
        :param num: dealed num
        :return:
        """
        loop = num // (self.max_mask_x_dtype * 255)
        if self.ub_tensor_size >= self.max_mask_x_dtype * 255:
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * self.max_mask_x_dtype * 255
                    if self.x_need_cast:
                        self.tik_instance.vec_muls(self.max_mask_x_dtype, self.tmp_x_ub[compute_offset],
                                                   self.input_x_ub_fp32[compute_offset], self.scale_scalar_fp32, 255, 8,
                                                   8)
                    else:
                        self.tik_instance.vec_muls(self.max_mask_x_dtype, self.tmp_x_ub[compute_offset],
                                                   self.input_x_ub[compute_offset], self.scale_scalar_fp32, 255, 8, 8)
                    self.tik_instance.vec_adds(self.max_mask_x_dtype, self.tmp_x_ub[compute_offset],
                                               self.tmp_x_ub[compute_offset], self.zp_scalar_fp32, 255, 8, 8)
        compute_offset = loop * self.max_mask_x_dtype * 255
        repeat_time = num % (self.max_mask_x_dtype * 255) // self.max_mask_x_dtype
        if repeat_time > 0:
            if self.x_need_cast:
                self.tik_instance.vec_muls(self.max_mask_x_dtype, self.tmp_x_ub[compute_offset],
                                           self.input_x_ub_fp32[compute_offset], self.scale_scalar_fp32, repeat_time, 8,
                                           8)
            else:
                self.tik_instance.vec_muls(self.max_mask_x_dtype, self.tmp_x_ub[compute_offset],
                                           self.input_x_ub[compute_offset], self.scale_scalar_fp32, repeat_time, 8, 8)
            self.tik_instance.vec_adds(self.max_mask_x_dtype, self.tmp_x_ub[compute_offset],
                                       self.tmp_x_ub[compute_offset], self.zp_scalar_fp32, repeat_time, 8, 8)
        compute_offset += repeat_time * self.max_mask_x_dtype
        last_num = num % self.max_mask_x_dtype
        if last_num > 0:
            if self.x_need_cast:
                self.tik_instance.vec_muls(last_num, self.tmp_x_ub[compute_offset],
                                           self.input_x_ub_fp32[compute_offset], self.scale_scalar_fp32, 1, 8, 8)
            else:
                self.tik_instance.vec_muls(last_num, self.tmp_x_ub[compute_offset], self.input_x_ub[compute_offset],
                                           self.scale_scalar_fp32, 1, 8, 8)
            self.tik_instance.vec_adds(last_num, self.tmp_x_ub[compute_offset], self.tmp_x_ub[compute_offset],
                                       self.zp_scalar_fp32, 1, 8, 8)

    def init_ub_tensor_and_scalar(self):
        """
        init tensor and scalar
        """
        self.input_x_ub = self.tik_instance.Tensor(self.input_x_dtype, (self.ub_tensor_size, ),
                                                   name="input_x_ub",
                                                   scope=tik.scope_ubuf)
        self.tmp_x_ub = self.tik_instance.Tensor("float32", (self.ub_tensor_size, ),
                                                 name="tmp_x_ub",
                                                 scope=tik.scope_ubuf)
        if self.per_tensor:
            self.ori_scale = self.tik_instance.Scalar("float32")
            self.scale_scalar_fp32 = self.tik_instance.Scalar("float32")
            self.zp_scalar_fp32 = self.tik_instance.Scalar("float32")
            self.input_scales_ub = self.tik_instance.Tensor(self.scales_dtype, (self.data_each_block_scales, ),
                                                            name="input_scales_ub",
                                                            scope=tik.scope_ubuf)
            self.input_zero_points_ub = self.tik_instance.Tensor(self.zero_points_dtype,
                                                                 (self.data_each_block_zero_points, ),
                                                                 name="input_zero_points_ub",
                                                                 scope=tik.scope_ubuf)
            self.wk_tensor = self.tik_instance.Tensor("float32", (self.max_mask_x_dtype * 2, ),
                                                      name="wk_tensor",
                                                      scope=tik.scope_ubuf)
            self.input_scales_ub_fp32_t = self.tik_instance.Tensor("float32", (self.data_each_block_scales, ),
                                                                   name="input_scales_ub_fp32_t",
                                                                   scope=tik.scope_ubuf)
            self.input_zero_points_ub_fp32 = self.tik_instance.Tensor("float32", (self.data_each_block_zero_points, ),
                                                                      name="input_zero_points_ub_fp32",
                                                                      scope=tik.scope_ubuf)
            if self.zero_points_dtype != "int32":
                self.input_zero_points_ub_fp16 = self.tik_instance.Tensor("float16",
                                                                          (self.data_each_block_zero_points, ),
                                                                          name="input_zero_points_ub_fp16",
                                                                          scope=tik.scope_ubuf)
            if self.x_need_cast:
                self.input_x_ub_fp32 = self.tik_instance.Tensor("float32", (self.ub_tensor_size, ),
                                                                name="input_x_ub_fp32",
                                                                scope=tik.scope_ubuf)

        else:
            self.input_scales_ub = self.tik_instance.Tensor(self.scales_dtype, (self.ub_tensor_size, ),
                                                            name="input_scales_ub",
                                                            scope=tik.scope_ubuf)
            self.input_zero_points_ub = self.tik_instance.Tensor(self.zero_points_dtype, (self.ub_tensor_size, ),
                                                                 name="input_zero_points_ub",
                                                                 scope=tik.scope_ubuf)
            self.wk_tensor = self.tik_instance.Tensor("float32", (self.ub_tensor_size * 2, ),
                                                      name="wk_tensor",
                                                      scope=tik.scope_ubuf)
            self.input_scales_ub_fp32_t = self.tik_instance.Tensor("float32", (self.ub_tensor_size, ),
                                                                   name="input_scales_ub_fp32_t",
                                                                   scope=tik.scope_ubuf)
            self.input_zero_points_ub_fp32 = self.tik_instance.Tensor("float32", (self.ub_tensor_size, ),
                                                                      name="input_zero_points_ub_fp32",
                                                                      scope=tik.scope_ubuf)
            if self.zero_points_dtype != "int32":
                self.input_zero_points_ub_fp16 = self.tik_instance.Tensor("float16", (self.ub_tensor_size, ),
                                                                          name="input_zero_points_ub_fp16",
                                                                          scope=tik.scope_ubuf)
            if self.input_x_dtype == "float16":
                self.input_x_ub_fp32 = self.tik_instance.Tensor("float32", (self.ub_tensor_size, ),
                                                                name="input_x_ub_fp32",
                                                                scope=tik.scope_ubuf)
        self.zero_compare_tensor = self.tik_instance.Tensor("float32", (self.max_mask_x_dtype, ),
                                                            name="zero_compare_tensor",
                                                            scope=tik.scope_ubuf)
        self.scale_compare_tensor = self.tik_instance.Tensor("float32", (self.max_mask_x_dtype, ),
                                                             name="scale_compare_tensor",
                                                             scope=tik.scope_ubuf)
        self.min_scalar = self.tik_instance.Scalar("int32", init_value=self.min_value)
        if (not self.per_tensor) and self.output_y_dtype == 'int8':
            self.min_scalar_fp32 = self.tik_instance.Scalar("float32", init_value=0.0)
        else:
            self.min_scalar_fp32 = self.tik_instance.Scalar("float32", init_value=self.min_value)
        self.max_scalar = self.tik_instance.Scalar("int32", init_value=self.max_value)
        self.min_vale_tensor = self.tik_instance.Tensor("float32", (self.max_mask_x_dtype, ),
                                                        name="min_vale_tensor",
                                                        scope=tik.scope_ubuf)
        self.zero_scalar = self.tik_instance.Scalar(dtype="int32", init_value=0)
        self.zero_scalar_fp32 = self.tik_instance.Scalar(dtype="float32", init_value=0.0)
        self.ext = self.tik_instance.Scalar(dtype="float32", init_value=1e-20)
        self.is_x_zero = self.tik_instance.Tensor('uint16', (self.max_mask_x_dtype, ),
                                                  name="is_x_zero",
                                                  scope=tik.scope_ubuf)
        self.is_scale_zero = self.tik_instance.Tensor('uint16', (self.max_mask_x_dtype, ),
                                                      name="is_scale_zero",
                                                      scope=tik.scope_ubuf)
        self.is_zero = self.tik_instance.Tensor('uint16', (self.max_mask_x_dtype, ),
                                                name="is_zero",
                                                scope=tik.scope_ubuf)
        self.int32_tensor = self.tik_instance.Tensor("int32", (self.ub_tensor_size, ),
                                                     name="int32_tensor",
                                                     scope=tik.scope_ubuf)
        self.compare_int32_min_tensor = self.tik_instance.Tensor("int32", (self.max_mask_x_dtype, ),
                                                                 name="compare_int32_min_tensor",
                                                                 scope=tik.scope_ubuf)
        self.compare_int32_max_tensor = self.tik_instance.Tensor("int32", (self.max_mask_x_dtype, ),
                                                                 name="compare_int32_max_tensor",
                                                                 scope=tik.scope_ubuf)

        self.tik_instance.vec_dup(self.max_mask_x_dtype, self.compare_int32_min_tensor, self.min_scalar, 1, 8)
        self.tik_instance.vec_dup(self.max_mask_x_dtype, self.compare_int32_max_tensor, self.max_scalar, 1, 8)
        self.tik_instance.vec_dup(self.max_mask_x_dtype, self.zero_compare_tensor, self.zero_scalar_fp32, 1, 8)
        self.tik_instance.vec_dup(self.max_mask_x_dtype, self.min_vale_tensor, self.min_scalar_fp32, 1, 8)
        if self.output_y_dtype != "int32":
            self.output_y_fp16 = self.tik_instance.Tensor("float16", (self.ub_tensor_size, ),
                                                          name="output_y_fp16",
                                                          scope=tik.scope_ubuf)
            if self.output_y_dtype == "int8":
                self.output_y_int = self.tik_instance.Tensor("int8", (self.ub_tensor_size, ),
                                                             name="output_y_int",
                                                             scope=tik.scope_ubuf)
            elif self.output_y_dtype == "uint8":
                self.output_y_int = self.tik_instance.Tensor("uint8", (self.ub_tensor_size, ),
                                                             name="output_y_int",
                                                             scope=tik.scope_ubuf)


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def quantize(x, scales, zero_points, y, dtype, axis=1, kernel_name="quantize_per_channel"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input_x
    scales : dict
        shape and dtype of input_scales
    zero_points : dict
        shape and dtype of input_zero_points
    y : dict
        shape and dtype of output_y, should be same shape as input, dtype is same as the quantified type
    axis: int
        the processed dim
    dtype:
        quantified type
    kernel_name : str
        kernel name, default value is "quantize"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    shape_scale = scales.get("shape")
    dtype_scale = scales.get("dtype")
    shape_zp = zero_points.get("shape")
    dtype_zp = zero_points.get("dtype")
    check_x_tuple = ("float16", "float32")
    check_scale_tuple = ("float32", )
    check_zp_tuple = ("int8", "uint8", "int32")
    check_dtype = ("torch.qint8", "torch.quint8", "torch.qint32")
    if dtype_x not in check_x_tuple:
        raise RuntimeError("X only support %s while dtype is %s" % (",".join(check_x_tuple), dtype_x))
    if dtype_scale not in check_scale_tuple:
        raise RuntimeError("Scales only support %s while dtype is %s" % (",".join(check_scale_tuple), dtype_scale))
    if dtype_zp not in check_zp_tuple:
        raise RuntimeError("Zero_points only support %s while dtype is %s" % (",".join(check_zp_tuple), dtype_zp))
    if dtype not in check_dtype:
        raise RuntimeError("Dtype only support %s while dtype is %s" % (",".join(check_dtype), dtype))

    para_check.check_shape_rule(shape_x)
    para_check.check_shape_size(shape_x)
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_scale)
    para_check.check_shape_size(shape_scale)
    para_check.check_shape_rule(shape_zp)
    para_check.check_shape_size(shape_zp)

    quantize_instance = Quantize(x, scales, zero_points, axis, dtype, kernel_name)
    return quantize_instance.quantize_compute()
