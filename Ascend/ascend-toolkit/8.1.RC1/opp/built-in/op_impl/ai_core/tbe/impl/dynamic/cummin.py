#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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

from tbe.common.platform.platform_info import get_soc_spec
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform


class Constant:
    """
    The class for constant
    """
    TILING_ARG_NUM = 7
    MAX_INT32 = 2 ** 31 - 1
    BIT_PER_BYTE = 8
    BYTES_PER_BLOCK = 32
    BYTES_PER_MASK = 256
    AICORE_DTYPES = ["int8", "uint8", "int32", "float16", "float32", "bfloat16"]
    FOUR_BYTES_CAL_DTYPES = ["bfloat16", "int32", "float32"]
    MAX_REPEAT = 255
    MASK_64_NUM = 64


def check_supported(x, y, argmin, axis, kernel_name="cummin"):
    """
    AICORE or AICPU selection
    """
    x_dtype = x.get("dtype")
    if x_dtype not in Constant.AICORE_DTYPES:
        return False, "AICORE cummin does not support this dtype, switch to AICPU."
    return True, "AICORE cummin supported."


def get_loop_args(compute_mask, compute_num, max_repeat_times):
    """
    needed args for tik vec calculating
    """
    loop = compute_num // compute_mask // max_repeat_times
    repeat_time = compute_num % (compute_mask * max_repeat_times) // compute_mask
    last_num = compute_num % (compute_mask * max_repeat_times) % compute_mask
    return loop, repeat_time, last_num


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
        self.support_data_move_pad = tbe_platform.api_check_support('tik.data_move_pad')
        self.input_x_dtype = input_x.get("dtype").lower()
        self.argmin_dtype = "int32"
        self.dim = dim
        self.kernel_name = kernel_name
        self.dtype_bytes_size = tbe_platform.get_bit_len(self.input_x_dtype) // Constant.BIT_PER_BYTE
        self.dtype_bytes_size_argmin = tbe_platform.get_bit_len(self.argmin_dtype) // Constant.BIT_PER_BYTE
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.dtype_fp32_byte_size = 4
        self.dtype_fp16_byte_size = 2
        self.pad_mask = 64

        self.check_param()
        self.init_tiling_scalars()

        self.data_each_block = Constant.BYTES_PER_BLOCK // self.dtype_bytes_size
        self.data_each_block_argmin = Constant.BYTES_PER_BLOCK // self.dtype_bytes_size_argmin
        if self.input_x_dtype in Constant.FOUR_BYTES_CAL_DTYPES:
            self.max_mask = Constant.BYTES_PER_MASK // self.dtype_fp32_byte_size
        else:
            self.max_mask = Constant.BYTES_PER_MASK // self.dtype_fp16_byte_size

        input_x_pad_size = self.dtype_bytes_size * self.pad_mask
        if self.input_x_dtype != "float32" and self.input_x_dtype != "int32":
            self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - input_x_pad_size * 2
        else:
            self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

        self.init_gm_tensor()

        self.max_mask_argmin_dtype = Constant.BYTES_PER_MASK // self.dtype_bytes_size_argmin

        self.input_x_ub = None
        self.input_x_ub_tmp = None
        self.last_x_ub = None
        self.argmin_ub = None
        self.argmin_ub_tmp = None
        self.last_argmin_ub = None
        self.argmin_ub_cast = None
        self.argmin_ub_cast_tmp = None
        self.last_argmin_ub_cast = None
        self.input_x_ub_cast = None
        self.last_x_ub_cast = None
        self.is_le = None
        self.dim_this_time = None
        self.offset_this_dim = None
        self.last_first_offset = None
        self.zero_scalar = None

    def check_param(self):
        """
        Check whether the input parameters is valid or not.
        """
        input_check_tuple = ("float16", "float32", "int32", "int8", "uint8", "bfloat16")
        para_check.check_dtype(self.input_x_dtype, input_check_tuple, param_name="x")

    def init_tiling_scalars(self):
        self.ub_tensor_size = self.tik_instance.Scalar("int32", name="ub_tensor_size", init_value=0)
        self.num_of_after_dim = self.tik_instance.Scalar("int32", name="num_of_after_dim", init_value=0)
        self.tensor_1d_num = self.tik_instance.Scalar("int32", name="tensor_1d_num", init_value=0)
        self.num_each_core = self.tik_instance.Scalar("int32", name="num_each_core", init_value=0)
        self.last_core_num = self.tik_instance.Scalar("int32", name="last_core_num", init_value=0)
        self.without_last_one_block_num = self.tik_instance.Scalar("int32",
                                                                    name="without_last_one_block_num",
                                                                    init_value=0)
        self.shape_dim = self.tik_instance.Scalar("int32", name="shape_dim", init_value=0)
        self.core_num_var = self.tik_instance.Scalar("int32", name="core_num_var", init_value=self.ai_core_num)

    def init_gm_tensor(self):
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,), name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.input_x_gm = self.tik_instance.Tensor(self.input_x_dtype,
                                                   (Constant.MAX_INT32,),
                                                   name="input_x_gm",
                                                   scope=tik.scope_gm)
        self.output_y_gm = self.tik_instance.Tensor(self.input_x_dtype,
                                                    (Constant.MAX_INT32,),
                                                    name="output_y_gm",
                                                    scope=tik.scope_gm)
        self.output_argmin_gm = self.tik_instance.Tensor(self.argmin_dtype,
                                                         (Constant.MAX_INT32,),
                                                         name="output_argmin_gm",
                                                         scope=tik.scope_gm)

    def move_tiling_args_to_ub(self):
        tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
        self.ub_tensor_size.set_as(tiling_ub[0])
        self.tensor_1d_num.set_as(tiling_ub[1])
        self.num_each_core.set_as(tiling_ub[2])
        self.last_core_num.set_as(tiling_ub[3])
        self.without_last_one_block_num.set_as(tiling_ub[4])
        self.shape_dim.set_as(tiling_ub[5])
        self.core_num_var.set_as(tiling_ub[6])

    def cummin_compute(self):
        """
        Calculate total entrance
        """
        self.move_tiling_args_to_ub()
        with self.tik_instance.for_range(0, self.shape_dim) as dim_id:
            move_offset = self.tensor_1d_num * dim_id
            with self.tik_instance.if_scope(self.without_last_one_block_num == 1):
                # single-core scenario
                self.cummin_compute_each_core(move_offset, self.num_each_core)
            with self.tik_instance.else_scope():
                # multi-core scenario
                self.cummin_compute_each_dim(move_offset)
            
        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.ai_core_num,
                "ub_size": self.ub_size
            }
        )

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_x_gm],
                                   outputs=[self.output_y_gm, self.output_argmin_gm],
                                   flowtable=[self.tiling_gm])
        return self.tik_instance

    def cummin_compute_each_dim(self, dim_move_offset):
        """
        compute on each dim with multi core.
        take the 0th dim as the unit, open multi-core calculation.
        """
        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_id:
            with self.tik_instance.if_scope(core_id == self.without_last_one_block_num):
                with self.tik_instance.if_scope(self.last_core_num > 0):
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
        with self.tik_instance.if_scope(loop_time < 2):
            need_db = False
        with self.tik_instance.if_scope(loop_time > 0):
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
        with self.tik_instance.if_scope(last_num > 0):
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
        burst_len = (num - 1) // self.data_each_block + 1
        burst_len_argmin = (num - 1) // self.data_each_block_argmin + 1
        if not self.support_data_move_pad:
            self.tik_instance.data_move(self.input_x_ub,
                                        self.input_x_gm[self.offset_this_dim],
                                        0, 1, burst_len, 0, 0)
        else:
            self.tik_instance.data_move_pad(self.input_x_ub,
                                            self.input_x_gm[self.offset_this_dim],
                                            1, num * self.dtype_bytes_size, 0, 0)
        self.dim_this_time.set_as(i)
        with self.tik_instance.if_scope(i == 0):
            loop = num // (self.max_mask_argmin_dtype * Constant.MAX_REPEAT)
            with self.tik_instance.if_scope(self.ub_tensor_size >= self.max_mask_argmin_dtype * Constant.MAX_REPEAT):
                with self.tik_instance.if_scope(loop > 0):
                    with self.tik_instance.for_range(0, loop) as index:
                        compute_offset = index * self.max_mask_argmin_dtype * Constant.MAX_REPEAT
                        self.tik_instance.vec_dup(self.max_mask_argmin_dtype,
                                                  self.argmin_ub[compute_offset],
                                                  self.zero_scalar,
                                                  Constant.MAX_REPEAT, 8)
            compute_offset = loop * self.max_mask_argmin_dtype * Constant.MAX_REPEAT
            repeat_time = num % (self.max_mask_argmin_dtype * Constant.MAX_REPEAT) // self.max_mask_argmin_dtype
            with self.tik_instance.if_scope(repeat_time > 0):
                self.tik_instance.vec_dup(self.max_mask_argmin_dtype,
                                          self.argmin_ub[compute_offset],
                                          self.zero_scalar,
                                          repeat_time, 8)
            compute_offset += repeat_time * self.max_mask_argmin_dtype
            last_num = num % self.max_mask_argmin_dtype
            with self.tik_instance.if_scope(last_num > 0):
                self.tik_instance.vec_dup(last_num, self.argmin_ub[compute_offset], self.zero_scalar, 1, 8)
        with self.tik_instance.else_scope():
            self.last_first_offset.set_as(self.offset_this_dim - self.tensor_1d_num)
            if not self.support_data_move_pad:
                self.tik_instance.data_move(self.last_x_ub,
                                            self.output_y_gm[self.last_first_offset],
                                            0, 1, burst_len, 0, 0)
                self.tik_instance.data_move(self.last_argmin_ub,
                                            self.output_argmin_gm[self.last_first_offset],
                                            0, 1, burst_len_argmin, 0, 0)
            else:
                self.tik_instance.data_move_pad(self.last_x_ub,
                                                self.output_y_gm[self.last_first_offset],
                                                1, num * self.dtype_bytes_size, 0, 0)
                self.tik_instance.data_move_pad(self.last_argmin_ub,
                                                self.output_argmin_gm[self.last_first_offset],
                                                1, num * self.dtype_bytes_size_argmin, 0, 0)
            self.dup_argmin(num)
            self.argmin_to(num)
            self.x_to(num)
            dtype_bytes_size = self.dtype_bytes_size
            if tbe_platform.api_check_support("tik.vcopy"):
                if self.input_x_dtype == "float16" or self.input_x_dtype == "float32":
                    self.tik_instance.data_move(self.input_x_ub_tmp,
                                                self.input_x_ub,
                                                0, 1, burst_len, 0, 0)
                elif self.input_x_dtype == "bfloat16":
                    burst_len_bf16 = (num - 1) // (self.data_each_block // 2) + 1
                    self.tik_instance.data_move(self.input_x_ub_tmp,
                                                self.input_x_ub_cast,
                                                0, 1, burst_len_bf16, 0, 0)
                    dtype_bytes_size = dtype_bytes_size * 2
                self.compute_once_support_nan(num, dtype_bytes_size)
            else:
                self.compute_once(num)
            self.argmin_from(num)
            self.x_from(num)
            
        if not self.support_data_move_pad:
            self.tik_instance.data_move(self.output_y_gm[self.offset_this_dim],
                                        self.input_x_ub,
                                        0, 1, burst_len, 0, 0)
            self.tik_instance.data_move(self.output_argmin_gm[self.offset_this_dim],
                                        self.argmin_ub,
                                        0, 1, burst_len_argmin, 0, 0)
        else:
            self.tik_instance.data_move_pad(self.output_y_gm[self.offset_this_dim],
                                            self.input_x_ub,
                                            1, num * self.dtype_bytes_size, 0, 0)
            self.tik_instance.data_move_pad(self.output_argmin_gm[self.offset_this_dim],
                                            self.argmin_ub,
                                            1, num * self.dtype_bytes_size_argmin, 0, 0)

    def dup_argmin(self, num):
        """
        dup argmin values with this dim
        """
        loop, repeat_time, last_num = get_loop_args(self.max_mask_argmin_dtype, num, Constant.MAX_REPEAT)
        with self.tik_instance.if_scope(self.ub_tensor_size >= self.max_mask_argmin_dtype * Constant.MAX_REPEAT):
            with self.tik_instance.if_scope(loop > 0):
                with self.tik_instance.for_range(0, loop) as index:
                    compute_offset = index * self.max_mask_argmin_dtype * Constant.MAX_REPEAT
                    self.tik_instance.vec_dup(self.max_mask_argmin_dtype,
                                              self.argmin_ub[compute_offset],
                                              self.dim_this_time,
                                              Constant.MAX_REPEAT, 8)
                    self.tik_instance.vec_dup(self.max_mask_argmin_dtype,
                                              self.argmin_ub_tmp[compute_offset],
                                              self.dim_this_time,
                                              Constant.MAX_REPEAT, 8)
        
        with self.tik_instance.if_scope(repeat_time > 0):
            compute_offset = loop * self.max_mask_argmin_dtype * Constant.MAX_REPEAT
            self.tik_instance.vec_dup(self.max_mask_argmin_dtype,
                                      self.argmin_ub[compute_offset],
                                      self.dim_this_time,
                                      repeat_time, 8)
            self.tik_instance.vec_dup(self.max_mask_argmin_dtype,
                                      self.argmin_ub_tmp[compute_offset],
                                      self.dim_this_time,
                                      repeat_time, 8)

        with self.tik_instance.if_scope(last_num > 0):
            compute_offset = num - last_num
            self.tik_instance.vec_dup(last_num,
                                      self.argmin_ub[compute_offset],
                                      self.dim_this_time,
                                      1, 8)
            self.tik_instance.vec_dup(last_num,
                                      self.argmin_ub_tmp[compute_offset],
                                      self.dim_this_time,
                                      1, 8)

    def argmin_to(self, num):
        """
        Convert the type of argmin to fp32
        """
        loop, repeat_time, last_num = get_loop_args(self.max_mask_argmin_dtype, num, Constant.MAX_REPEAT)
        with self.tik_instance.if_scope(self.ub_tensor_size >= self.max_mask_argmin_dtype * Constant.MAX_REPEAT):
            with self.tik_instance.if_scope(loop > 0):
                with self.tik_instance.for_range(0, loop) as index:
                    compute_offset = index * self.max_mask_argmin_dtype * Constant.MAX_REPEAT
                    self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                               self.last_argmin_ub_cast[compute_offset],
                                               self.last_argmin_ub[compute_offset],
                                               Constant.MAX_REPEAT, 8, 8)
                    self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                               self.argmin_ub_cast[compute_offset],
                                               self.argmin_ub[compute_offset],
                                               Constant.MAX_REPEAT, 8, 8)
                    self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                               self.argmin_ub_cast_tmp[compute_offset],
                                               self.argmin_ub_tmp[compute_offset],
                                               Constant.MAX_REPEAT, 8, 8)
        
        with self.tik_instance.if_scope(repeat_time > 0):
            compute_offset = loop * self.max_mask_argmin_dtype * Constant.MAX_REPEAT
            self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                       self.last_argmin_ub_cast[compute_offset],
                                       self.last_argmin_ub[compute_offset],
                                       repeat_time, 8, 8)
            self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                       self.argmin_ub_cast[compute_offset],
                                       self.argmin_ub[compute_offset],
                                       repeat_time, 8, 8)
            self.tik_instance.vec_conv(self.max_mask_argmin_dtype, '',
                                       self.argmin_ub_cast_tmp[compute_offset],
                                       self.argmin_ub_tmp[compute_offset],
                                       repeat_time, 8, 8)

        with self.tik_instance.if_scope(last_num > 0):
            compute_offset = num - last_num
            self.tik_instance.vec_conv(last_num, '',
                                       self.last_argmin_ub_cast[compute_offset],
                                       self.last_argmin_ub[compute_offset],
                                       1, 8, 8)
            self.tik_instance.vec_conv(last_num, '',
                                       self.argmin_ub_cast[compute_offset],
                                       self.argmin_ub[compute_offset],
                                       1, 8, 8)
            self.tik_instance.vec_conv(last_num, '',
                                       self.argmin_ub_cast_tmp[compute_offset],
                                       self.argmin_ub_tmp[compute_offset],
                                       1, 8, 8)

    def argmin_from(self, num):
        """
        Convert the type of argmin back
        """
        loop, repeat_time, last_num = get_loop_args(self.max_mask_argmin_dtype, num, Constant.MAX_REPEAT)
        with self.tik_instance.if_scope(self.ub_tensor_size >= self.max_mask_argmin_dtype * Constant.MAX_REPEAT):
            with self.tik_instance.if_scope(loop > 0):
                with self.tik_instance.for_range(0, loop) as index:
                    compute_offset = index * self.max_mask_argmin_dtype * Constant.MAX_REPEAT
                    self.tik_instance.vec_conv(self.max_mask_argmin_dtype, 'round',
                                               self.argmin_ub[compute_offset],
                                               self.argmin_ub_cast[compute_offset],
                                               Constant.MAX_REPEAT, 8, 8)
                    self.tik_instance.vec_conv(self.max_mask_argmin_dtype, 'round',
                                               self.argmin_ub_tmp[compute_offset],
                                               self.argmin_ub_cast_tmp[compute_offset],
                                               Constant.MAX_REPEAT, 8, 8)
        
        with self.tik_instance.if_scope(repeat_time > 0):
            compute_offset = loop * self.max_mask_argmin_dtype * Constant.MAX_REPEAT
            self.tik_instance.vec_conv(self.max_mask_argmin_dtype, 'round',
                                       self.argmin_ub[compute_offset],
                                       self.argmin_ub_cast[compute_offset],
                                       repeat_time, 8, 8)
            self.tik_instance.vec_conv(self.max_mask_argmin_dtype, 'round',
                                       self.argmin_ub_tmp[compute_offset],
                                       self.argmin_ub_cast_tmp[compute_offset],
                                       repeat_time, 8, 8)

        with self.tik_instance.if_scope(last_num > 0):
            compute_offset = num - last_num
            self.tik_instance.vec_conv(last_num, 'round',
                                       self.argmin_ub[compute_offset],
                                       self.argmin_ub_cast[compute_offset],
                                       1, 8, 8)
            self.tik_instance.vec_conv(last_num, 'round',
                                       self.argmin_ub_tmp[compute_offset],
                                       self.argmin_ub_cast_tmp[compute_offset],
                                       1, 8, 8)

    def x_to(self, num):
        """
        Convert the type of x.
        If x is int32 or bf16, convert to fp32.
        If x is int8 or uint8, convert to fp16.
        """
        if self.input_x_dtype != "float32" and self.input_x_dtype != "float16":
            loop, repeat_time, last_num = get_loop_args(self.max_mask, num, Constant.MAX_REPEAT)
            x_stride = self.max_mask * self.dtype_bytes_size // Constant.BYTES_PER_BLOCK
            with self.tik_instance.if_scope(self.ub_tensor_size >= self.max_mask * Constant.MAX_REPEAT):
                with self.tik_instance.if_scope(loop > 0):
                    with self.tik_instance.for_range(0, loop) as index:
                        compute_offset = index * self.max_mask * Constant.MAX_REPEAT
                        self.tik_instance.vec_conv(self.max_mask, '',
                                                   self.input_x_ub_cast[compute_offset],
                                                   self.input_x_ub[compute_offset],
                                                   Constant.MAX_REPEAT, 8, x_stride)
                        self.tik_instance.vec_conv(self.max_mask, '',
                                                   self.last_x_ub_cast[compute_offset],
                                                   self.last_x_ub[compute_offset],
                                                   Constant.MAX_REPEAT, 8, x_stride)

            with self.tik_instance.if_scope(repeat_time > 0):
                compute_offset = loop * self.max_mask * Constant.MAX_REPEAT
                self.tik_instance.vec_conv(self.max_mask, '',
                                           self.input_x_ub_cast[compute_offset],
                                           self.input_x_ub[compute_offset],
                                           repeat_time, 8, x_stride)
                self.tik_instance.vec_conv(self.max_mask, '',
                                           self.last_x_ub_cast[compute_offset],
                                           self.last_x_ub[compute_offset],
                                           repeat_time, 8, x_stride)
            
            with self.tik_instance.if_scope(last_num > 0):
                compute_offset = num - last_num
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
        if self.input_x_dtype != "float32" and self.input_x_dtype != "float16":
            loop, repeat_time, last_num = get_loop_args(self.max_mask, num, Constant.MAX_REPEAT)
            x_stride = self.max_mask * self.dtype_bytes_size // Constant.BYTES_PER_BLOCK

            def x_from_mode(mode):
                with self.tik_instance.if_scope(self.ub_tensor_size >= self.max_mask * Constant.MAX_REPEAT):
                    with self.tik_instance.if_scope(loop > 0):
                        with self.tik_instance.for_range(0, loop) as index:
                            compute_offset = index * self.max_mask * Constant.MAX_REPEAT
                            self.tik_instance.vec_conv(self.max_mask, mode,
                                                       self.input_x_ub[compute_offset],
                                                       self.input_x_ub_cast[compute_offset],
                                                       Constant.MAX_REPEAT, x_stride, 8)
                            
                with self.tik_instance.if_scope(repeat_time > 0):
                    compute_offset = loop * self.max_mask * Constant.MAX_REPEAT
                    self.tik_instance.vec_conv(self.max_mask, mode,
                                               self.input_x_ub[compute_offset],
                                               self.input_x_ub_cast[compute_offset],
                                               repeat_time, x_stride, 8)
                
                with self.tik_instance.if_scope(last_num > 0):
                    compute_offset = num - last_num
                    self.tik_instance.vec_conv(last_num, mode,
                                               self.input_x_ub[compute_offset],
                                               self.input_x_ub_cast[compute_offset],
                                               1, x_stride, 8)

            if self.input_x_dtype == "bfloat16" or self.input_x_dtype == "int32":
                x_from_mode("round")
            else:
                x_from_mode("none")                 

    def compute_once_support_nan(self, num, dtype_bytes_size):
        """
        compute once in 910B.
        x and argmin is fp32 or fp16.
        64 numbers each time.
        """
        loop, repeat_time, last_num = get_loop_args(Constant.MASK_64_NUM, num, 1)
        x_stride = Constant.MASK_64_NUM * dtype_bytes_size // Constant.BYTES_PER_BLOCK

        def do_sel_min_nan(mask_num, offset, input_x_ub, last_x_ub):
            """
            select the min value from input_x_ub[offset] and last_x_ub[offset]
            inf and nan supported
            """
            self.tik_instance.vec_cmpv_gt(self.is_le,
                                          input_x_ub[offset],
                                          last_x_ub[offset],
                                          1, x_stride, x_stride)
            self.tik_instance.vec_sel(mask_num, 0, 
                                      input_x_ub[offset],
                                      self.is_le,
                                      last_x_ub[offset],
                                      input_x_ub[offset],
                                      1, x_stride, x_stride, x_stride)
            self.tik_instance.vec_sel(mask_num, 0,
                                      self.argmin_ub_cast[offset],
                                      self.is_le,
                                      self.last_argmin_ub_cast[offset],
                                      self.argmin_ub_cast[offset],
                                      1, 8, 8, 8)
            self.tik_instance.vec_cmpv_eq(self.is_le,
                                          last_x_ub[offset],
                                          last_x_ub[offset],
                                          1, x_stride, x_stride)
            self.tik_instance.vec_sel(mask_num, 0,
                                      input_x_ub[offset],
                                      self.is_le,
                                      input_x_ub[offset],
                                      last_x_ub[offset],
                                      1, x_stride, x_stride, x_stride)
            self.tik_instance.vec_sel(mask_num, 0,
                                      self.argmin_ub_cast[offset],
                                      self.is_le,
                                      self.argmin_ub_cast[offset],
                                      self.last_argmin_ub_cast[offset],
                                      1, 8, 8, 8)
            self.tik_instance.vec_cmpv_eq(self.is_le,
                                          self.input_x_ub_tmp[offset],
                                          self.input_x_ub_tmp[offset],
                                          1, x_stride, x_stride)
            self.tik_instance.vec_sel(mask_num, 0,
                                      input_x_ub[offset],
                                      self.is_le,
                                      input_x_ub[offset],
                                      self.input_x_ub_tmp[offset],
                                      1, x_stride, x_stride, x_stride)
            self.tik_instance.vec_sel(mask_num, 0,
                                      self.argmin_ub_cast[offset],
                                      self.is_le,
                                      self.argmin_ub_cast[offset],
                                      self.argmin_ub_cast_tmp[offset],
                                      1, 8, 8, 8)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                compute_offset = index * Constant.MASK_64_NUM
                if self.input_x_dtype == "float32" or self.input_x_dtype == "float16":
                    do_sel_min_nan(Constant.MASK_64_NUM, compute_offset, self.input_x_ub, self.last_x_ub)
                elif self.input_x_dtype == "bfloat16":
                    do_sel_min_nan(Constant.MASK_64_NUM, compute_offset, self.input_x_ub_cast,
                                   self.last_x_ub_cast)
                else:
                    self.do_sel_min(self.input_x_ub_cast, self.last_x_ub_cast, compute_offset, Constant.MASK_64_NUM,
                                    x_stride)
        
        with self.tik_instance.if_scope(last_num > 0):
            compute_offset = loop * Constant.MASK_64_NUM
            if self.input_x_dtype == "float32" or self.input_x_dtype == "float16":
                do_sel_min_nan(last_num, compute_offset, self.input_x_ub, self.last_x_ub)
            elif self.input_x_dtype == "bfloat16":
                do_sel_min_nan(last_num, compute_offset, self.input_x_ub_cast, self.last_x_ub_cast)
            else:
                self.do_sel_min(self.input_x_ub_cast, self.last_x_ub_cast, compute_offset, Constant.MASK_64_NUM,
                                x_stride)      
        
    def do_sel_min(self, input_x_ub, last_x_ub, compute_offset, mask_num, x_stride):
        """
        select the min value from input_x_ub[offset] and last_x_ub[offset]
        """
        self.tik_instance.vec_cmpv_gt(self.is_le,
                                      input_x_ub[compute_offset],
                                      last_x_ub[compute_offset],
                                      1, x_stride, x_stride)
        self.tik_instance.vec_sel(mask_num, 0,
                                  input_x_ub[compute_offset],
                                  self.is_le,
                                  last_x_ub[compute_offset],
                                  input_x_ub[compute_offset],
                                  1, x_stride, x_stride, x_stride)
        self.tik_instance.vec_sel(mask_num, 0,
                                  self.argmin_ub_cast[compute_offset],
                                  self.is_le,
                                  self.last_argmin_ub_cast[compute_offset],
                                  self.argmin_ub_cast[compute_offset],
                                  1, 8, 8, 8)

    def compute_once(self, num):
        """
        compute once.
        x and argmin is fp32 or fp16.
        64 numbers each time.
        """
        loop, repeat_time, last_num = get_loop_args(Constant.MASK_64_NUM, num, 1)
        x_stride = Constant.MASK_64_NUM * self.dtype_bytes_size // Constant.BYTES_PER_BLOCK
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                compute_offset = index * Constant.MASK_64_NUM
                if self.input_x_dtype == "float32" or self.input_x_dtype == "float16":
                    self.do_sel_min(self.input_x_ub, self.last_x_ub, compute_offset, Constant.MASK_64_NUM, x_stride)
                else:
                    self.do_sel_min(self.input_x_ub_cast, self.last_x_ub_cast, compute_offset, Constant.MASK_64_NUM,
                                    x_stride)
        
        with self.tik_instance.if_scope(last_num > 0):
            compute_offset = loop * Constant.MASK_64_NUM
            if self.input_x_dtype == "float32" or self.input_x_dtype == "float16":
                self.do_sel_min(self.input_x_ub, self.last_x_ub, compute_offset, last_num, x_stride)
            else:
                self.do_sel_min(self.input_x_ub_cast, self.last_x_ub_cast, compute_offset, last_num, x_stride)

    def init_ub_tensor_and_scalar(self):
        """
        init tensor and scalar
        """
        if self.input_x_dtype == "bfloat16":
            self.input_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                       (self.ub_tensor_size + self.pad_mask,),
                                                       name="input_x_ub",
                                                       scope=tik.scope_ubuf)
            self.input_x_ub_tmp = self.tik_instance.Tensor("float32",
                                                          (self.ub_tensor_size + self.pad_mask,),
                                                          name="input_x_ub_tmp",
                                                          scope=tik.scope_ubuf)
            self.last_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                      (self.ub_tensor_size + self.pad_mask,),
                                                      name="last_x_ub",
                                                      scope=tik.scope_ubuf)
        
        elif self.input_x_dtype != "float32" and self.input_x_dtype != "int32":
            self.input_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                       (self.ub_tensor_size + self.pad_mask,),
                                                       name="input_x_ub",
                                                       scope=tik.scope_ubuf)
            self.input_x_ub_tmp = self.tik_instance.Tensor(self.input_x_dtype,
                                                          (self.ub_tensor_size + self.pad_mask,),
                                                          name="input_x_ub_tmp",
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
            self.input_x_ub_tmp = self.tik_instance.Tensor(self.input_x_dtype,
                                                          (self.ub_tensor_size,),
                                                          name="input_x_ub_tmp",
                                                          scope=tik.scope_ubuf)
            self.last_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                      (self.ub_tensor_size,),
                                                      name="last_x_ub",
                                                      scope=tik.scope_ubuf)
        self.argmin_ub = self.tik_instance.Tensor(self.argmin_dtype,
                                                  (self.ub_tensor_size,),
                                                  name="argmin_ub",
                                                  scope=tik.scope_ubuf)
        self.argmin_ub_tmp = self.tik_instance.Tensor(self.argmin_dtype,
                                                     (self.ub_tensor_size,),
                                                     name="argmin_ub_tmp",
                                                     scope=tik.scope_ubuf)
        self.last_argmin_ub = self.tik_instance.Tensor(self.argmin_dtype,
                                                       (self.ub_tensor_size,),
                                                       name="last_argmin_ub",
                                                       scope=tik.scope_ubuf)
        self.argmin_ub_cast = self.tik_instance.Tensor("float32",
                                                       (self.ub_tensor_size,),
                                                       name="argmin_ub_cast",
                                                       scope=tik.scope_ubuf)
        self.argmin_ub_cast_tmp = self.tik_instance.Tensor("float32",
                                                          (self.ub_tensor_size,),
                                                          name="argmin_ub_cast_tmp",
                                                          scope=tik.scope_ubuf)
        self.last_argmin_ub_cast = self.tik_instance.Tensor("float32",
                                                            (self.ub_tensor_size,),
                                                            name="last_argmin_ub_cast",
                                                            scope=tik.scope_ubuf)

        if self.input_x_dtype == "int32":
            self.input_x_ub_cast = self.tik_instance.Tensor("float32",
                                                            (self.ub_tensor_size,),
                                                            name="input_x_ub_cast",
                                                            scope=tik.scope_ubuf)
            self.last_x_ub_cast = self.tik_instance.Tensor("float32",
                                                            (self.ub_tensor_size,),
                                                            name="last_x_ub_cast",
                                                            scope=tik.scope_ubuf)
        elif self.input_x_dtype == "int8" or self.input_x_dtype == "uint8":
            self.input_x_ub_cast = self.tik_instance.Tensor("float16",
                                                            (self.ub_tensor_size + self.pad_mask,),
                                                            name="input_x_ub_cast",
                                                            scope=tik.scope_ubuf)
            self.last_x_ub_cast = self.tik_instance.Tensor("float16",
                                                            (self.ub_tensor_size + self.pad_mask,),
                                                            name="last_x_ub_cast",
                                                            scope=tik.scope_ubuf)
        elif self.input_x_dtype == "bfloat16":
            self.input_x_ub_cast = self.tik_instance.Tensor("float32",
                                                            (self.ub_tensor_size + self.pad_mask,),
                                                            name="input_x_ub_cast",
                                                            scope=tik.scope_ubuf)

            self.last_x_ub_cast = self.tik_instance.Tensor("float32",
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
@register_operator("Cummin")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def cummin(x, y, argmin, axis, kernel_name="cummin"):
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
    cummin_instance = Cummin(x, axis, kernel_name)
    return cummin_instance.cummin_compute()
