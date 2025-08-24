#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
cummin
"""

import functools
import math
from te.utils import para_check
from te.utils.op_utils import check_dtype
from tbe.common.platform import get_bit_len
from te import tik
from tbe.common import platform as cce


class Cummin():
    """
    Implementation of Cummin
    """
    # 'pylint: disable=too-many-branches,too-many-statements
    def __init__(self, input_x, dim, kernel_name):
        """
        init of cummin
        """
        self.tik_instance = tik.Tik()
        self.input_x_shape = list(input_x.get("shape"))
        self.input_x_dtype = input_x.get("dtype")
        self.argmin_dtype = "int32"
        self.dim = dim
        self.kernel_name = kernel_name
        self.dtype_bytes_size = get_bit_len(self.input_x_dtype) // 8
        self.dtype_bytes_size_argmin = get_bit_len(self.argmin_dtype) // 8
        self.dtype_fp32 = 4
        self.dtype_fp16 = 2
        self.data_each_block = 32 // self.dtype_bytes_size
        self.data_each_block_argmin = 32 // self.dtype_bytes_size_argmin
        self.max_mask = 256 // 4
        if self.input_x_dtype == "float32" or self.input_x_dtype == "int32":
            self.max_mask = 256 // 4
        else:
            self.max_mask = 256 // 2
        self.pad_mask = 64
        input_x_pad_size = self.dtype_bytes_size * self.pad_mask
        if self.input_x_dtype != "float32" and self.input_x_dtype != "int32":
            self.ub_size = cce.get_soc_spec(cce.UB_SIZE) - input_x_pad_size * 2
        else:
            self.ub_size = cce.get_soc_spec(cce.UB_SIZE)
        if self.input_x_dtype == "float32":
            divider = self.dtype_bytes_size * 2 + self.dtype_bytes_size_argmin * 2 + self.dtype_fp32 * 2 + 2
            self.ub_tensor_size = self.ub_size // divider // 2 // self.max_mask * self.max_mask
        elif self.input_x_dtype == "float16":
            divider = self.dtype_bytes_size * 2 + self.dtype_bytes_size_argmin * 2 + self.dtype_fp32 * 2 + 2
            self.ub_tensor_size = self.ub_size // divider // 2 // self.max_mask * self.max_mask
        elif self.input_x_dtype == "int32":
            divider = self.dtype_bytes_size * 2 + self.dtype_bytes_size_argmin * 2 + self.dtype_fp32 * 4 + 2
            self.ub_tensor_size = self.ub_size // divider // 2 // self.max_mask * self.max_mask
        else:
            need_sub_pad_size = self.pad_mask * 2 * 2 * 2
            self.ub_size = self.ub_size - need_sub_pad_size
            d1 = self.dtype_bytes_size * 2 + self.dtype_bytes_size_argmin * 2
            d2 = self.dtype_fp16 * 2 + self.dtype_fp32 * 2 + 2
            divider = d1 + d2
            self.ub_tensor_size = self.ub_size // divider // 2 // self.max_mask * self.max_mask
        self.input_x_gm = self.tik_instance.Tensor(self.input_x_dtype,
                                                   self.input_x_shape,
                                                   name="input_x_gm",
                                                   scope=tik.scope_gm)
        self.output_y_gm = self.tik_instance.Tensor(self.input_x_dtype,
                                                    self.input_x_shape,
                                                    name="output_y_gm",
                                                    scope=tik.scope_gm)
        self.output_argmin_gm = self.tik_instance.Tensor(self.argmin_dtype,
                                                         self.input_x_shape,
                                                         name="output_argmin_gm",
                                                         scope=tik.scope_gm)
        self.dims = len(self.input_x_shape)
        self.dim = dim
        self.max_mask_x_dtype = 256 // self.dtype_bytes_size
        self.max_mask_fp16 = 256 // 2
        self.max_mask_argmin_dtype = 256 // self.dtype_bytes_size_argmin
        self.shape_dim = self.input_x_shape[dim]
        self.in_num = functools.reduce(lambda x, y: x * y, self.input_x_shape)
        shape_tmp = self.input_x_shape
        after_shape = shape_tmp[dim + 1:]
        shape_tmp.pop(dim)
        self.shape_no_dim_tmp = shape_tmp
        self.max_ai_core_num = cce.get_soc_spec(cce.CORE_NUM)
        if len(after_shape) == 0:
            self.after_num = 1
        else:
            self.after_num = functools.reduce(lambda x, y: x * y, after_shape)
        if len(self.shape_no_dim_tmp) != 0:
            self.tensor_1d_num = functools.reduce(lambda x, y: x * y, self.shape_no_dim_tmp)
        else:
            self.tensor_1d_num = 1
        if self.tensor_1d_num >= (self.max_ai_core_num - 1) * 32:
            self.ai_core_num = self.max_ai_core_num - 1
        else:
            self.ai_core_num = 1
        if self.ai_core_num == 1:
            self.num_each_core = self.tensor_1d_num
        else:
            self.num_each_core = self.tensor_1d_num // self.ai_core_num // 32 * 32
        self.last_core_num = self.tensor_1d_num - self.num_each_core * self.ai_core_num
        self.pad_num = 0
        if self.last_core_num % 32 != 0 and self.ai_core_num > 1:
            self.pad_num = 32 - self.last_core_num % 32
            self.last_core_num = self.last_core_num + self.pad_num

        self.input_x_ub = None
        self.last_x_ub = None
        self.argmin_ub = None
        self.last_argmin_ub = None
        self.argmin_ub_cast = None
        self.last_argmin_ub_cast = None
        self.input_x_ub_cast = None
        self.last_x_ub_cast = None
        self.is_le = None
        self.dim_this_time = None
        self.offset_this_dim = None
        self.last_first_offset = None
        self.zero_scalar = None

    def cummin_compute(self):
        """
        Calculate total entrance
        """
        with self.tik_instance.for_range(0, self.shape_dim) as dim_id:
            move_offset = self.tensor_1d_num * dim_id
            self.cummin_compute_each_dim(move_offset)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_x_gm],
                                   outputs=[self.output_y_gm, self.output_argmin_gm])
        return self.tik_instance

    def cummin_compute_each_dim(self, dim_move_offset):
        """
        compute on each dim.
        take the 0th dim as the unit, open muti-core calculation.
        """
        if self.ai_core_num == 1:
            self.cummin_compute_each_core(dim_move_offset, self.num_each_core)
        else:
            with self.tik_instance.for_range(0, self.max_ai_core_num, block_num=self.max_ai_core_num) as core_id:
                with self.tik_instance.if_scope(tik.all(core_id == self.ai_core_num, self.last_core_num > 0)):
                    move_offset = dim_move_offset + self.tensor_1d_num - self.last_core_num
                    self.cummin_compute_each_core(move_offset, self.last_core_num)
                with self.tik_instance.else_scope():
                    move_offset = dim_move_offset + core_id * self.num_each_core
                    self.cummin_compute_each_core(move_offset, self.num_each_core)

    def cummin_compute_each_core(self, core_move_offset, core_move_num):
        """
        compute on each core
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
                    self.cummin_compute_each_loop(move_offset,
                                                  self.ub_tensor_size)
            else:
                with self.tik_instance.for_range(0, loop_time) as loop_id:
                    move_offset = loop_id * self.ub_tensor_size + core_move_offset
                    self.cummin_compute_each_loop(move_offset,
                                                  self.ub_tensor_size)
            move_offset = loop_time * self.ub_tensor_size + core_move_offset
        last_num = core_move_num % self.ub_tensor_size
        if last_num > 0:
            self.cummin_compute_each_loop(move_offset, last_num)

    def cummin_compute_each_loop(self, move_offset, move_num):
        """
        compute each loop
        """
        self.init_ub_tensor_and_scalar()
        i = move_offset // self.tensor_1d_num
        self.offset_this_dim.set_as(move_offset)
        self.compute_each_time(i, move_num)

    def compute_each_time(self, i, num):
        """
        compute each time
        """
        burse_len = math.ceil(num / self.data_each_block)
        burse_len_argmin = math.ceil(num / self.data_each_block_argmin)
        self.tik_instance.data_move(self.input_x_ub,
                                    self.input_x_gm[self.offset_this_dim],
                                    0, 1, burse_len, 0, 0)
        self.dim_this_time.set_as(i)
        with self.tik_instance.if_scope(i == 0):
            loop = num // (self.max_mask_argmin_dtype * 255)
            if self.ub_tensor_size >= self.max_mask_argmin_dtype * 255:
                if loop > 0:
                    for index in range(loop):
                        compute_offset = index * self.max_mask_argmin_dtype * 255
                        self.tik_instance.vec_dup(self.max_mask_argmin_dtype,
                                                  self.argmin_ub[compute_offset],
                                                  self.zero_scalar,
                                                  255, 8)
            compute_offset = loop * self.max_mask_argmin_dtype * 255
            repeat_time = num % (self.max_mask_argmin_dtype * 255) // self.max_mask_argmin_dtype
            if repeat_time > 0:
                self.tik_instance.vec_dup(self.max_mask_argmin_dtype,
                                          self.argmin_ub[compute_offset],
                                          self.zero_scalar,
                                          repeat_time, 8)
            compute_offset += repeat_time * self.max_mask_argmin_dtype
            last_num = num % self.max_mask_argmin_dtype
            if last_num > 0:
                self.tik_instance.vec_dup(last_num, self.argmin_ub[compute_offset], self.zero_scalar, 1, 8)
            self.tik_instance.data_move(self.output_y_gm[self.offset_this_dim],
                                        self.input_x_ub,
                                        0, 1, burse_len, 0, 0)
            self.tik_instance.data_move(self.output_argmin_gm[self.offset_this_dim],
                                        self.argmin_ub,
                                        0, 1, burse_len_argmin, 0, 0)
        with self.tik_instance.else_scope():
            self.last_first_offset.set_as(self.offset_this_dim - self.tensor_1d_num)
            self.tik_instance.data_move(self.last_x_ub,
                                        self.output_y_gm[self.last_first_offset],
                                        0, 1, burse_len, 0, 0)
            self.tik_instance.data_move(self.last_argmin_ub,
                                        self.output_argmin_gm[self.last_first_offset],
                                        0, 1, burse_len_argmin, 0, 0)
            self.dup_argmin(num)
            self.argmin_to(num)
            self.x_to(num)
            self.compute_once(num)
            self.argmin_from(num)
            self.x_from(num)
            self.tik_instance.data_move(self.output_y_gm[self.offset_this_dim],
                                        self.input_x_ub,
                                        0, 1, burse_len, 0, 0)
            self.tik_instance.data_move(self.output_argmin_gm[self.offset_this_dim],
                                        self.argmin_ub,
                                        0, 1, burse_len_argmin, 0, 0)

    def dup_argmin(self, num):
        """
        dup argmin values with this dim
        """
        mask_int32 = 64
        loop = num // (mask_int32 * 255)
        if self.ub_tensor_size >= mask_int32 * 255:
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * mask_int32 * 255
                    self.tik_instance.vec_dup(mask_int32,
                                              self.argmin_ub[compute_offset],
                                              self.dim_this_time,
                                              255, 8)
        compute_offset = loop * mask_int32 * 255
        repeat_time = num % (mask_int32 * 255) // mask_int32
        if repeat_time > 0:
            self.tik_instance.vec_dup(mask_int32,
                                      self.argmin_ub[compute_offset],
                                      self.dim_this_time,
                                      repeat_time, 8)
        compute_offset += repeat_time * mask_int32
        last_num = num % mask_int32
        if last_num > 0:
            self.tik_instance.vec_dup(last_num,
                                      self.argmin_ub[compute_offset],
                                      self.dim_this_time,
                                      1, 8)

    def argmin_to(self, num):
        """
        Convert the type of argmin to fp32
        """
        loop = num // (self.max_mask_argmin_dtype * 255)
        if self.ub_tensor_size >= self.max_mask_argmin_dtype * 255:
            if loop > 0:
                for index in  range(loop):
                    compute_offset = index * self.max_mask_argmin_dtype * 255
                    self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                               self.last_argmin_ub_cast[compute_offset],
                                               self.last_argmin_ub[compute_offset],
                                               255, 8, 8)
                    self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                               self.argmin_ub_cast[compute_offset],
                                               self.argmin_ub[compute_offset],
                                               255, 8, 8)
        compute_offset = loop * self.max_mask_argmin_dtype * 255
        repeat_time = num % (self.max_mask_argmin_dtype * 255) // self.max_mask_argmin_dtype
        if repeat_time > 0:
            self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                       self.last_argmin_ub_cast[compute_offset],
                                       self.last_argmin_ub[compute_offset],
                                       repeat_time, 8, 8)
            self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                       self.argmin_ub_cast[compute_offset],
                                       self.argmin_ub[compute_offset],
                                       repeat_time, 8, 8)
        compute_offset += repeat_time * self.max_mask_argmin_dtype
        last_num = num % self.max_mask_argmin_dtype
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, '',
                                       self.last_argmin_ub_cast[compute_offset],
                                       self.last_argmin_ub[compute_offset],
                                       1, 8, 8)
            self.tik_instance.vec_conv(last_num, '',
                                       self.argmin_ub_cast[compute_offset],
                                       self.argmin_ub[compute_offset],
                                       1, 8, 8)

    def argmin_from(self, num):
        """
        Convert the type of argmin back
        """
        loop = num // (self.max_mask_argmin_dtype * 255)
        if self.ub_tensor_size >= self.max_mask_argmin_dtype * 255:
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * self.max_mask_argmin_dtype * 255
                    self.tik_instance.vec_conv(self.max_mask_argmin_dtype, 'round',
                                               self.argmin_ub[compute_offset],
                                               self.argmin_ub_cast[compute_offset],
                                               255, 8, 8)
        compute_offset = loop * self.max_mask_argmin_dtype * 255
        repeat_time = num % (self.max_mask_argmin_dtype * 255) // self.max_mask_argmin_dtype
        if repeat_time > 0:
            self.tik_instance.vec_conv(self.max_mask_argmin_dtype, 'round',
                                       self.argmin_ub[compute_offset],
                                       self.argmin_ub_cast[compute_offset],
                                       repeat_time, 8, 8)
        compute_offset += repeat_time * self.max_mask_argmin_dtype
        last_num = num % self.max_mask_argmin_dtype
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, 'round',
                                       self.argmin_ub[compute_offset],
                                       self.argmin_ub_cast[compute_offset],
                                       1, 8, 8)

    def x_to(self, num):
        """
        Convert the type of x, if x is not fp32 or fp16, convert to fp16
        """
        if self.input_x_dtype == "int32":
            loop = num // (self.max_mask_argmin_dtype * 255)
            if self.ub_tensor_size >= self.max_mask_argmin_dtype * 255:
                if loop > 0:
                    for index in range(loop):
                        compute_offset = index * self.max_mask_argmin_dtype * 255
                        self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                                   self.input_x_ub_cast[compute_offset],
                                                   self.input_x_ub[compute_offset],
                                                   255, 8, 8)
                        self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                                   self.last_x_ub_cast[compute_offset],
                                                   self.last_x_ub[compute_offset],
                                                   255, 8, 8)
            compute_offset = loop * self.max_mask_argmin_dtype * 255
            repeat_time = num % (self.max_mask_argmin_dtype * 255) // self.max_mask_argmin_dtype
            if repeat_time > 0:
                self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                           self.input_x_ub_cast[compute_offset],
                                           self.input_x_ub[compute_offset],
                                           repeat_time,
                                           8, 8)
                self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                           self.last_x_ub_cast[compute_offset],
                                           self.last_x_ub[compute_offset],
                                           repeat_time,
                                           8, 8)
            compute_offset += repeat_time * self.max_mask_argmin_dtype
            last_num = num % self.max_mask_argmin_dtype
            if last_num > 0:
                self.tik_instance.vec_conv(last_num, '',
                                           self.input_x_ub_cast[compute_offset],
                                           self.input_x_ub[compute_offset],
                                           1, 8, 8)
                self.tik_instance.vec_conv(last_num, '',
                                           self.last_x_ub_cast[compute_offset],
                                           self.last_x_ub[compute_offset],
                                           1, 8, 8)
        elif self.input_x_dtype != "float32" and self.input_x_dtype != "float16":
            x_stride = self.max_mask * self.dtype_bytes_size // 32
            loop = num // (self.max_mask * 255)
            if self.ub_tensor_size >= self.max_mask * 255:
                if loop > 0:
                    for index in range(loop):
                        compute_offset = index * self.max_mask * 255
                        self.tik_instance.vec_conv(self.max_mask, '',
                                                   self.input_x_ub_cast[compute_offset],
                                                   self.input_x_ub[compute_offset],
                                                   255, 8, x_stride)
                        self.tik_instance.vec_conv(self.max_mask, '',
                                                   self.last_x_ub_cast[compute_offset],
                                                   self.last_x_ub[compute_offset],
                                                   255, 8, x_stride)
            compute_offset = loop * self.max_mask * 255
            repeat_time = num % (self.max_mask * 255) // self.max_mask
            if repeat_time > 0:
                self.tik_instance.vec_conv(self.max_mask, '',
                                           self.input_x_ub_cast[compute_offset],
                                           self.input_x_ub[compute_offset],
                                           repeat_time, 8, x_stride)
                self.tik_instance.vec_conv(self.max_mask, '',
                                           self.last_x_ub_cast[compute_offset],
                                           self.last_x_ub[compute_offset],
                                           repeat_time, 8, x_stride)
            compute_offset += repeat_time * self.max_mask
            last_num = num % self.max_mask
            if last_num > 0:
                self.tik_instance.vec_conv(last_num, '',
                                           self.input_x_ub_cast[compute_offset],
                                           self.input_x_ub[compute_offset],
                                           1, 8, x_stride)
                self.tik_instance.vec_conv(last_num, '',
                                           self.last_x_ub_cast[compute_offset],
                                           self.last_x_ub[compute_offset],
                                           1, 8, x_stride)

    def x_from(self, num):
        """
        Convert the type of x back
        """
        if self.input_x_dtype == "int32":
            loop = num // (self.max_mask_argmin_dtype * 255)
            if self.ub_tensor_size >= self.max_mask_argmin_dtype * 255:
                if loop > 0:
                    for index in range(loop):
                        compute_offset = index * self.max_mask_argmin_dtype * 255
                        self.tik_instance.vec_conv(self.max_mask_argmin_dtype, 'round',
                                                   self.input_x_ub[compute_offset],
                                                   self.input_x_ub_cast[compute_offset],
                                                   255, 8, 8)
            compute_offset = loop * self.max_mask_argmin_dtype * 255
            repeat_time = num % (self.max_mask_argmin_dtype * 255) // self.max_mask_argmin_dtype
            if repeat_time > 0:
                self.tik_instance.vec_conv(self.max_mask_argmin_dtype, 'round',
                                           self.input_x_ub[compute_offset],
                                           self.input_x_ub_cast[compute_offset],
                                           repeat_time, 8, 8)
            compute_offset += repeat_time * self.max_mask_argmin_dtype
            last_num = num % self.max_mask_argmin_dtype
            if last_num > 0:
                self.tik_instance.vec_conv(last_num, 'round',
                                           self.input_x_ub[compute_offset],
                                           self.input_x_ub_cast[compute_offset],
                                           1, 8, 8)
        elif self.input_x_dtype != "float32" and self.input_x_dtype != "int32" and self.input_x_dtype != "float16":
            x_stride = self.max_mask * self.dtype_bytes_size // 32
            loop = num // (self.max_mask * 255)
            if self.ub_tensor_size >= self.max_mask * 255:
                if loop > 0:
                    for index in range(loop):
                        compute_offset = index * self.max_mask * 255
                        self.tik_instance.vec_conv(self.max_mask, '',
                                                   self.input_x_ub[compute_offset],
                                                   self.input_x_ub_cast[compute_offset],
                                                   255, x_stride, 8)
            compute_offset = loop * self.max_mask * 255
            repeat_time = num % (self.max_mask * 255) // self.max_mask
            if repeat_time > 0:
                self.tik_instance.vec_conv(self.max_mask, '',
                                           self.input_x_ub[compute_offset],
                                           self.input_x_ub_cast[compute_offset],
                                           repeat_time, x_stride, 8)
            compute_offset += repeat_time * self.max_mask
            last_num = num % self.max_mask
            if last_num > 0:
                self.tik_instance.vec_conv(last_num, '',
                                           self.input_x_ub[compute_offset],
                                           self.input_x_ub_cast[compute_offset],
                                           1, x_stride, 8)

    def compute_once(self, num):
        """
        compute once.
        x and argmin is fp32 or fp16.
        64 numbers each time.
        """
        mask_now = 64
        x_stride = self.dtype_bytes_size * mask_now // 32
        if self.input_x_dtype == "float32" or self.input_x_dtype == "float16":
            loop = num // mask_now
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * mask_now
                    self.tik_instance.vec_cmpv_le(self.is_le,
                                                  self.input_x_ub[compute_offset],
                                                  self.last_x_ub[compute_offset],
                                                  1, x_stride, x_stride)
                    self.tik_instance.vec_sel(mask_now, 0,
                                              self.input_x_ub[compute_offset],
                                              self.is_le,
                                              self.input_x_ub[compute_offset],
                                              self.last_x_ub[compute_offset],
                                              1, x_stride, x_stride, x_stride)
                    self.tik_instance.vec_sel(mask_now, 0,
                                              self.argmin_ub_cast[compute_offset],
                                              self.is_le,
                                              self.argmin_ub_cast[compute_offset],
                                              self.last_argmin_ub_cast[compute_offset],
                                              1, 8, 8, 8)
            compute_offset = loop * mask_now
            last_num = num % mask_now
            if last_num > 0:
                self.tik_instance.vec_cmpv_le(self.is_le,
                                              self.input_x_ub[compute_offset],
                                              self.last_x_ub[compute_offset],
                                              1, x_stride, x_stride)
                self.tik_instance.vec_sel(last_num, 0,
                                          self.input_x_ub[compute_offset],
                                          self.is_le,
                                          self.input_x_ub[compute_offset],
                                          self.last_x_ub[compute_offset],
                                          1, x_stride, x_stride, x_stride)
                self.tik_instance.vec_sel(last_num, 0,
                                          self.argmin_ub_cast[compute_offset],
                                          self.is_le,
                                          self.argmin_ub_cast[compute_offset],
                                          self.last_argmin_ub_cast[compute_offset],
                                          1, 8, 8, 8)
        else:
            loop = num // mask_now
            if loop > 0:
                for index in range(loop):
                    compute_offset = index * mask_now
                    self.tik_instance.vec_cmpv_le(self.is_le,
                                                  self.input_x_ub_cast[compute_offset],
                                                  self.last_x_ub_cast[compute_offset],
                                                  1, x_stride, x_stride)
                    self.tik_instance.vec_sel(mask_now, 0,
                                              self.input_x_ub_cast[compute_offset],
                                              self.is_le,
                                              self.input_x_ub_cast[compute_offset],
                                              self.last_x_ub_cast[compute_offset],
                                              1, x_stride, x_stride, x_stride)
                    self.tik_instance.vec_sel(mask_now, 0,
                                              self.argmin_ub_cast[compute_offset],
                                              self.is_le,
                                              self.argmin_ub_cast[compute_offset],
                                              self.last_argmin_ub_cast[compute_offset],
                                              1, 8, 8, 8)
            compute_offset = loop * mask_now
            last_num = num % mask_now
            if last_num > 0:
                self.tik_instance.vec_cmpv_le(self.is_le,
                                              self.input_x_ub_cast[compute_offset],
                                              self.last_x_ub_cast[compute_offset],
                                              1, x_stride, x_stride)
                self.tik_instance.vec_sel(last_num, 0,
                                          self.input_x_ub_cast[compute_offset],
                                          self.is_le,
                                          self.input_x_ub_cast[compute_offset],
                                          self.last_x_ub_cast[compute_offset],
                                          1, x_stride, x_stride, x_stride)
                self.tik_instance.vec_sel(last_num, 0,
                                          self.argmin_ub_cast[compute_offset],
                                          self.is_le,
                                          self.argmin_ub_cast[compute_offset],
                                          self.last_argmin_ub_cast[compute_offset],
                                          1, 8, 8, 8)

    def init_ub_tensor_and_scalar(self):
        """
        init tensor and scalar
        """
        if(self.input_x_dtype != "float32" and self.input_x_dtype != "int32"):
            self.input_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                       (self.ub_tensor_size + self.pad_mask,),
                                                       name="input_x_ub",
                                                       scope=tik.scope_ubuf)
            self.last_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                      (self.ub_tensor_size + self.pad_mask,),
                                                      name="last_x_ub",
                                                      scope=tik.scope_ubuf)
        else:
            self.input_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                       (self.ub_tensor_size,),
                                                       name="input_x_ub",
                                                       scope=tik.scope_ubuf)
            self.last_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                      (self.ub_tensor_size,),
                                                      name="last_x_ub",
                                                      scope=tik.scope_ubuf)
        self.argmin_ub = self.tik_instance.Tensor(self.argmin_dtype,
                                                  (self.ub_tensor_size,),
                                                  name="argmin_ub",
                                                  scope=tik.scope_ubuf)
        self.last_argmin_ub = self.tik_instance.Tensor(self.argmin_dtype,
                                                       (self.ub_tensor_size,),
                                                       name="last_argmin_ub",
                                                       scope=tik.scope_ubuf)
        self.argmin_ub_cast = self.tik_instance.Tensor("float32",
                                                       (self.ub_tensor_size,),
                                                       name="argmin_ub_cast",
                                                       scope=tik.scope_ubuf)
        self.last_argmin_ub_cast = self.tik_instance.Tensor("float32",
                                                            (self.ub_tensor_size,),
                                                            name="last_argmin_ub_cast",
                                                            scope=tik.scope_ubuf)

        if self.input_x_dtype != "float32" and self.input_x_dtype != "float16":
            if self.input_x_dtype == "int32":
                self.input_x_ub_cast = self.tik_instance.Tensor("float32",
                                                                (self.ub_tensor_size,),
                                                                name="input_x_ub_cast",
                                                                scope=tik.scope_ubuf)
                self.last_x_ub_cast = self.tik_instance.Tensor("float32",
                                                               (self.ub_tensor_size,),
                                                               name="last_x_ub_cast",
                                                               scope=tik.scope_ubuf)
            else:
                self.input_x_ub_cast = self.tik_instance.Tensor("float16",
                                                                (self.ub_tensor_size + self.pad_mask,),
                                                                name="input_x_ub_cast",
                                                                scope=tik.scope_ubuf)
                self.last_x_ub_cast = self.tik_instance.Tensor("float16",
                                                               (self.ub_tensor_size + self.pad_mask,),
                                                               name="last_x_ub_cast",
                                                               scope=tik.scope_ubuf)
        self.is_le = self.tik_instance.Tensor('uint8',
                                              (self.ub_tensor_size,),
                                              name="is_le",
                                              scope=tik.scope_ubuf)
        self.dim_this_time = self.tik_instance.Scalar(dtype="int32")
        self.offset_this_dim = self.tik_instance.Scalar(dtype="int32")
        self.last_first_offset = self.tik_instance.Scalar(dtype="int32")
        self.zero_scalar = self.tik_instance.Scalar(dtype="int32", init_value=0)


# 'pylint: disable=unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.KERNEL_NAME)
def cummin(x, y, argmin, dim, kernel_name="cummin"):
    """
    Calculate the smaller value and index on the specified dimension

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    argmin : dict
        shape and dtype of output_index, should be same shape as input
    dim : int
        the processed dimension
    kernel_name : str
        kernel name, default value is "cummin"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    check_tuple = ("float16", "float32", "int32", "int8", "uint8")
    check_dtype(dtype.lower(), check_tuple)
    dims = len(shape)
    ne_dims = dims * -1
    if(dim < ne_dims or dim > (dims - 1)):
        raise RuntimeError("Only support {} =< dim <= {} while dim is {}".format(ne_dims, dims - 1, dim))
    if dim < 0:
        dim = dim + dims

    para_check.check_shape_rule(shape)
    para_check.check_tensor_shape_size(shape)
    para_check.check_kernel_name(kernel_name)

    cummin_instance = Cummin(x, dim, kernel_name)
    return cummin_instance.cummin_compute()
