#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) Huawei Technologies Co., Ltd 2022-2023. All rights reserved
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

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check


# 'pylint: disable=no-init,old-style-class,too-few-public-methods
class Cst:
    """
    The class for Constant.
    """
    DIV = 1
    DIM_N = 0
    DIM_C = 1
    DIM_H = 2
    DIM_W = 3
    DIM_W1 = 2
    DIM_H1 = 3
    DIM_H0 = 4
    DIM_W0 = 5
    ALIGN_NUM = 16
    MAX_REPEAT = 255
    SCALAR_TWO = 2
    SCALAR_EIGHT = 8
    VEC_MASK_B8 = 256
    NUM_BLOCK_B8 = 32
    NUM_BLOCK_B16 = 16
    NUM_BLOCK_B32 = 8
    VEC_MASK_B16 = 128
    VEC_MASK_B32 = 64
    MAX_B16_ELE_NUM = 98304


# 'pylint: disable=old-style-class,too-many-instance-attributes
class ScaledMaskedSoftmaxNd():
    """
    Compute for ND format inputs.
    """

    # 'pylint: disable=unused-argument,too-many-arguments,too-many-locals,too-many-statements
    def __init__(self, x, mask, y, scale, fixed_triu_mask, kernel_name):
        """                                                                                                           
        Instantiate ScaledMaskedSoftmaxNd object.
        """
        self.x_dtype, self.y_dtype = x.get("dtype").lower(), y.get("dtype").lower()
        self.x_shape, self.mask_shape = x.get("shape"), mask.get("shape")
        self.scale = scale
        self.fixed_triu_mask = fixed_triu_mask
        self.kernel_name = kernel_name
        self.cal_params()

        if self.wdim < 32 or self.wdim > 8192 or self.wdim % 32 != 0:
            raise RuntimeError('Error! When the format of input is ND, the last dimension of the input tensor should ' +
                               'be within the range of [32, 8192] and be divisible by 32. Currently, ' +
                               f'it is {self.wdim}.')

        self.create_gm_tensor()

    @staticmethod
    def cal_level(length, block):
        """
        Calculate vmax and vadd times.
        """
        cnt = 0
        dividend = length // block
        while dividend % Cst.SCALAR_TWO == 0:
            dividend //= Cst.SCALAR_TWO
            cnt += 1
        return cnt, dividend - 1

    def cal_params(self):
        """
        Calculate params for following computing.
        """
        # Calculate params_list
        self.tik = tik.Tik(tik.Dprofile())
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.wdim = self.x_shape[Cst.DIM_W]
        line = 2 if self.wdim <= 512 else 1
        if self.wdim <= 1024:
            div = 1
        elif 1024 < self.wdim <= 2048:
            div = 2
        elif 2048 < self.wdim <= 4096:
            div = 4
        else:
            div = 8

        # Calculate core_attr_list
        self.line_per_iter = line * Cst.NUM_BLOCK_B16 // div
        self.total_iter = self.x_shape[Cst.DIM_N] * self.x_shape[Cst.DIM_C] * \
            self.x_shape[Cst.DIM_H] // self.line_per_iter
        self.iter_per_batch = self.x_shape[Cst.DIM_C] * self.x_shape[Cst.DIM_H] // self.line_per_iter
        self.iter_per_channel = self.x_shape[Cst.DIM_H] // self.line_per_iter
        self.iter_per_core = (self.total_iter + self.aicore_num - 1) // self.aicore_num
        self.used_core_num = (self.total_iter + self.iter_per_core - 1) // self.iter_per_core
        self.iter_last_core = self.total_iter - (self.used_core_num - 1) * self.iter_per_core
        self.element_per_iter = self.wdim * self.line_per_iter
        self.element_per_core = self.element_per_iter * self.iter_per_core
        self.shape = (self.line_per_iter, self.wdim)
        if self.line_per_iter > Cst.SCALAR_EIGHT:
            self.work_tensor_shape = (self.line_per_iter, Cst.SCALAR_TWO)
        else:
            self.work_tensor_shape = (Cst.NUM_BLOCK_B16,)
        if self.fixed_triu_mask:
            self.broad_ratio_n = self.x_shape[Cst.DIM_N]
            self.broad_ratio_c = self.x_shape[Cst.DIM_C]
        else:
            self.broad_ratio_n = self.x_shape[Cst.DIM_N] // self.mask_shape[Cst.DIM_N]
            self.broad_ratio_c = self.x_shape[Cst.DIM_C] // self.mask_shape[Cst.DIM_C]

    def create_gm_tensor(self):
        """
        Create input and output gm tensors.
        """
        self.x_gm = self.tik.Tensor(self.x_dtype, self.x_shape, name="x_gm", scope=tbe_platform.scope_gm)
        self.mask_gm = self.tik.Tensor('uint8', self.mask_shape, name="mask_gm", scope=tbe_platform.scope_gm)
        if self.fixed_triu_mask:
            shape = [1, 1, self.mask_shape[Cst.DIM_H], self.mask_shape[Cst.DIM_W]]
            self.fixed_mask_gm = self.tik.Tensor('float16', shape, name="fixed_mask_gm",
                                                 scope=tbe_platform.scope_gm, is_workspace=True)
            self.gen_triu_mask()
        self.y_gm = self.tik.Tensor(self.y_dtype, self.x_shape, name="y_gm", scope=tbe_platform.scope_gm)

    def gen_triu_mask(self):
        """
        Generate triu mask. 
        """
        tri_mask_ub = self.tik.Tensor('float16', (self.wdim,), name="tri_mask_ub", scope=tbe_platform.scope_ubuf)
        tik_vec_dup(self.tik, tri_mask_ub, 1.0, 0, self.wdim, Cst.VEC_MASK_B16)

        vec_len = self.tik.Scalar("int64", init_value=1)
        offset_gm = self.tik.Scalar("int64", init_value=0)
        move_burst = self.wdim // Cst.NUM_BLOCK_B16

        with self.tik.for_range(0, self.mask_shape[Cst.DIM_H]) as i:
            with self.tik.if_scope(vec_len <= self.wdim):
                tik_vec_dup(self.tik, tri_mask_ub, 0.0, 0, vec_len, Cst.VEC_MASK_B16)
                vec_len.set_as(vec_len + 1)
            self.tik.data_move(self.fixed_mask_gm[offset_gm], tri_mask_ub, 0, 1, move_burst, 0, 0)
            offset_gm.set_as(offset_gm + self.wdim)

    def compute(self):
        """
        Implementation of fused scaled masked softmax algorithm.
        """
        with self.tik.for_range(0, self.used_core_num, block_num=self.used_core_num) as core_index:
            loop_per_core = self.tik.Scalar("int64", name="loop_per_core")
            offset = self.tik.Scalar("int64", name="offset")
            offset_mask = self.tik.Scalar("int64", name="offset_mask")

            with self.tik.if_scope(core_index == self.used_core_num - 1):
                loop_per_core.set_as(self.iter_last_core)
            with self.tik.else_scope():
                loop_per_core.set_as(self.iter_per_core)

            with self.tik.for_range(0, loop_per_core) as it:
                self.cal_offset(offset, offset_mask, core_index, it)
                if self.x_dtype == 'float16':
                    ub_fp32 = self.tik.Tensor("float32", self.shape, scope=tbe_platform.scope_ubuf, name="ub_fp32")
                    ub_1 = self.tik.Tensor("float16", self.shape, scope=tbe_platform.scope_ubuf, name="ub_1")
                    with self.tik.new_stmt_scope():
                        ub_2 = self.tik.Tensor("float16", self.shape, scope=tbe_platform.scope_ubuf, name="ub_2")
                        ub_mask_fp16 = self.tik.Tensor("float16", self.shape,
                                                       scope=tbe_platform.scope_ubuf, name="ub_mask_fp16")
                        self.move_mask_in(ub_mask_fp16, offset_mask)
                        self.process_mask(ub_mask_fp16, ub_2)
                        self.move_x_in(ub_1, offset)
                        self.scale_x(ub_1)
                        self.smooth_by_argmax(ub_1, ub_2, ub_mask_fp16)
                        self.do_exp(ub_2, ub_fp32)
                    with self.tik.new_stmt_scope():
                        raw_input = self.tik.Tensor("float32", self.shape,
                                                    scope=tbe_platform.scope_ubuf, name="raw_input")
                        self.calc_softmax(raw_input, ub_1, ub_fp32)
                        self.move_data_out(raw_input, ub_1, offset)
                else:
                    ub_fp32 = self.tik.Tensor("float32", self.shape, scope=tbe_platform.scope_ubuf, name="ub_fp32")
                    with self.tik.new_stmt_scope():
                        self.move_x_in(ub_fp32, offset)
                        self.scale_x(ub_fp32)
                        self.smooth_by_argmax_fp32(ub_fp32, offset_mask)
                        self.do_exp(None, ub_fp32)
                    with self.tik.new_stmt_scope():
                        raw_input = self.tik.Tensor("float32", self.shape,
                                                    scope=tbe_platform.scope_ubuf, name="raw_input")
                        self.calc_softmax(raw_input, None, ub_fp32)
                        self.move_data_out(raw_input, None, offset)
        return self.build_cce()

    def cal_offset(self, offset, offset_mask, core_index, it):
        """
        Calculate offset of data move for following calculation.
        """
        curr_batch = (core_index * self.iter_per_core + it) // self.iter_per_batch
        curr_channel = (core_index * self.iter_per_core + it) % self.iter_per_batch // self.iter_per_channel
        iter_in_curr_channel = core_index * self.iter_per_core + it - curr_batch *\
            self.iter_per_batch - curr_channel * self.iter_per_channel
        offset_mask.set_as(((curr_batch // self.broad_ratio_n) * self.mask_shape[Cst.DIM_C] +
                           (curr_channel // self.broad_ratio_c)) * self.iter_per_channel * self.element_per_iter +
                           iter_in_curr_channel * self.element_per_iter)
        offset.set_as((curr_batch * self.x_shape[Cst.DIM_C] + curr_channel) * self.iter_per_channel *
                      self.element_per_iter + iter_in_curr_channel * self.element_per_iter)

    def move_mask_in(self, ub_mask_fp16, offset_mask):
        """
        Move mask from gm tensor to ub tensor.
        """
        if self.fixed_triu_mask:
            move_burst = self.element_per_iter // Cst.NUM_BLOCK_B16
            self.tik.data_move(ub_mask_fp16, self.fixed_mask_gm[offset_mask], 0, 1, move_burst, 0, 0)
        else:
            move_burst = self.element_per_iter // Cst.NUM_BLOCK_B8
            conv_times = self.element_per_iter // Cst.VEC_MASK_B16
            ub_mask = self.tik.Tensor("uint8", self.shape, scope=tbe_platform.scope_ubuf, name="ub_mask")
            self.tik.data_move(ub_mask, self.mask_gm[offset_mask], 0, 1, move_burst, 0, 0)
            vconv_ub(self.tik, Cst.VEC_MASK_B16, ub_mask_fp16, ub_mask, conv_times, 8, 4)

    def process_mask(self, ub_mask_fp16, ub_2):
        """
        Convert the upper triangular matrix to the lower triangular matrix, and fill False with -10000.
        """
        repeat_times = self.element_per_iter // Cst.VEC_MASK_B16
        self.tik.vmuls(Cst.VEC_MASK_B16, ub_2, ub_mask_fp16, self.tik.Scalar(init_value=-10000., dtype="float16"),
                       repeat_times, 1, 1, 8, 8)
        self.tik.vmuls(Cst.VEC_MASK_B16, ub_mask_fp16, ub_mask_fp16, self.tik.Scalar(init_value=-1., dtype="float16"),
                       repeat_times, 1, 1, 8, 8)
        self.tik.vadds(Cst.VEC_MASK_B16, ub_mask_fp16, ub_mask_fp16, self.tik.Scalar(init_value=1., dtype="float16"),
                       repeat_times, 1, 1, 8, 8)

    def move_x_in(self, dst, offset):
        """
        Move x from gm tensor to ub tensor.
        """
        move_burst = self.element_per_iter // Cst.NUM_BLOCK_B16
        conv_times = self.element_per_iter // Cst.VEC_MASK_B32
        if self.x_dtype == 'float16':
            self.tik.data_move(dst, self.x_gm[offset], 0, 1, move_burst, 0, 0)
        elif self.x_dtype == 'bfloat16':
            with self.tik.new_stmt_scope():
                ub_bf16 = self.tik.Tensor("bfloat16", self.shape, scope=tbe_platform.scope_ubuf, name="ub_bf16")
                self.tik.data_move(ub_bf16, self.x_gm[offset], 0, 1, move_burst, 0, 0)
                vconv_ub(self.tik, Cst.VEC_MASK_B32, dst, ub_bf16, conv_times, 8, 4)
        else:
            self.tik.data_move(dst, self.x_gm[offset], 0, 1, move_burst * Cst.SCALAR_TWO, 0, 0)

    def scale_x(self, src):
        """
        Scale x.
        """
        if self.x_dtype == 'float16':
            self.tik_func_scalar("vmuls", src, src, self.tik.Scalar(init_value=self.scale, dtype="float16"),
                                 0, 0, self.element_per_iter, Cst.VEC_MASK_B16)
        else:
            self.tik_func_scalar("vmuls", src, src, self.tik.Scalar(init_value=self.scale, dtype="float32"),
                                 0, 0, self.element_per_iter, Cst.VEC_MASK_B32)

    def smooth_by_argmax(self, ub_1, ub_2, ub_mask_fp16):
        """
        X minus maximum value of each line prevents exponent overflow.
        """
        # Mask x and fill -10000
        repeat_time = self.element_per_iter // Cst.VEC_MASK_B16
        self.tik.vmul(Cst.VEC_MASK_B16, ub_1, ub_1, ub_mask_fp16, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik.vadd(Cst.VEC_MASK_B16, ub_1, ub_1, ub_2, repeat_time, 1, 1, 1, 8, 8, 8)

        # Reduce max
        cnt, remain = self.cal_level(self.wdim, Cst.NUM_BLOCK_B16)
        half_w = self.wdim // Cst.SCALAR_TWO
        move_stride = (self.wdim - Cst.NUM_BLOCK_B16) // Cst.NUM_BLOCK_B16
        for i in range(self.line_per_iter):
            offset = self.wdim * i
            self.tik_func_binary("vmax", ub_2, ub_1, ub_1, offset, offset, offset + half_w,
                                 self.wdim // Cst.SCALAR_TWO, Cst.VEC_MASK_B16)
            self.reduce_max_per_line(ub_2, offset, cnt, remain, Cst.VEC_MASK_B16, Cst.NUM_BLOCK_B16)
        self.tik.data_move(ub_mask_fp16, ub_2, 0, self.line_per_iter, 1, move_stride, 0)
        ub_reducemax = self.tik.Tensor("float16", (Cst.NUM_BLOCK_B8,),
                                       scope=tbe_platform.scope_ubuf, name="ub_reducemax")
        self.tik.vcgmax(Cst.VEC_MASK_B16, ub_reducemax, ub_mask_fp16,
                        ceil_div(self.line_per_iter, Cst.SCALAR_EIGHT), 1, 1, 8)

        # Minus max value of each line
        self.tik.vmuls(self.line_per_iter, ub_reducemax, ub_reducemax, -1.0, 1, 1, 1, 0, 0)
        maximum = self.tik.Scalar("float16", name="maximum")
        for i in range(self.line_per_iter):
            maximum.set_as(ub_reducemax[i])
            offset = self.wdim * i
            self.tik_func_scalar("vadds", ub_2, ub_1, maximum, offset, offset, self.wdim, Cst.VEC_MASK_B16)

    def smooth_by_argmax_fp32(self, ub_fp32, offset_mask):
        """
        X(fp32) minus maximum value of each line prevents exponent overflow.
        """
        # Mask x and fill -10000
        with self.tik.new_stmt_scope():
            ub_mask_fp16 = self.tik.Tensor("float16", self.shape, scope=tbe_platform.scope_ubuf, name="ub_mask_fp16")
            ub_mask_fp32 = self.tik.Tensor("float32", self.shape, scope=tbe_platform.scope_ubuf, name="ub_mask_fp32")
            self.move_mask_in(ub_mask_fp16, offset_mask)
            conv_times = self.element_per_iter // Cst.VEC_MASK_B32
            vconv_ub(self.tik, Cst.VEC_MASK_B32, ub_mask_fp32, ub_mask_fp16, conv_times, 8, 4)
            self.tik_func_scalar("vmuls", ub_mask_fp16, ub_mask_fp16,
                                 self.tik.Scalar(init_value=-10000., dtype="float16"),
                                 0, 0, self.element_per_iter, Cst.VEC_MASK_B16)
            self.tik_func_scalar("vmuls", ub_mask_fp32, ub_mask_fp32, self.tik.Scalar(init_value=-1., dtype="float32"),
                                 0, 0, self.element_per_iter, Cst.VEC_MASK_B32)
            self.tik_func_scalar("vadds", ub_mask_fp32, ub_mask_fp32, self.tik.Scalar(init_value=1., dtype="float32"),
                                 0, 0, self.element_per_iter, Cst.VEC_MASK_B32)
            self.tik_func_binary("vmul", ub_fp32, ub_fp32, ub_mask_fp32, 0, 0, 0,
                                 self.element_per_iter, Cst.VEC_MASK_B32)
            vconv_ub(self.tik, Cst.VEC_MASK_B32, ub_mask_fp32, ub_mask_fp16, conv_times, 8, 4)
            self.tik_func_binary("vadd", ub_fp32, ub_fp32, ub_mask_fp32, 0, 0, 0,
                                 self.element_per_iter, Cst.VEC_MASK_B32)

        # Reduce max
        with self.tik.new_stmt_scope():
            ub_reducemax = self.tik.Tensor("float32", (Cst.NUM_BLOCK_B8,),
                                           scope=tbe_platform.scope_ubuf, name="ub_reducemax")
            ub_vmax = self.tik.Tensor("float32", self.shape, scope=tbe_platform.scope_ubuf, name="ub_vmax")
            half_w = self.wdim // Cst.SCALAR_TWO
            if tbe_platform.api_check_support("tik.vcgmax", "float32"):
                cnt, remain = self.cal_level(self.wdim, Cst.NUM_BLOCK_B32)
                for i in range(self.line_per_iter):
                    offset = self.wdim * i
                    self.tik_func_binary("vmax", ub_vmax, ub_fp32, ub_fp32, offset, offset, offset + half_w,
                                         self.wdim // Cst.SCALAR_TWO, Cst.VEC_MASK_B32)
                    self.reduce_max_per_line(ub_vmax, offset, cnt, remain, Cst.VEC_MASK_B32, Cst.NUM_BLOCK_B32)
                move_stride = (self.wdim - Cst.NUM_BLOCK_B32) // Cst.NUM_BLOCK_B32
                self.tik.data_move(ub_vmax[Cst.NUM_BLOCK_B32], ub_vmax[self.wdim],
                                   0, self.line_per_iter - 1, 1, move_stride, 0)
                self.tik.vcgmax(Cst.VEC_MASK_B32, ub_reducemax, ub_vmax,
                                ceil_div(self.line_per_iter, Cst.SCALAR_EIGHT), 1, 1, 8)
            else:
                with self.tik.new_stmt_scope():
                    ub_reducemax_fp16 = self.tik.Tensor("float16", (Cst.NUM_BLOCK_B8,),
                                                        scope=tbe_platform.scope_ubuf, name="ub_reducemax_fp16")
                    ub_vmax_fp16 = self.tik.Tensor("float16", self.shape,
                                                   scope=tbe_platform.scope_ubuf, name="ub_vmax_fp16")
                    cnt, remain = self.cal_level(self.wdim, Cst.NUM_BLOCK_B16)
                    for i in range(self.line_per_iter):
                        offset = self.wdim * i
                        self.tik_func_binary("vmax", ub_vmax, ub_fp32, ub_fp32, offset, offset, offset + half_w,
                                             self.wdim // Cst.SCALAR_TWO, Cst.VEC_MASK_B32)
                        self.reduce_max_per_line(ub_vmax, offset, cnt, remain, Cst.VEC_MASK_B32, Cst.NUM_BLOCK_B16)
                    move_stride = (self.wdim - Cst.NUM_BLOCK_B16) // Cst.NUM_BLOCK_B32
                    self.tik.data_move(ub_vmax[Cst.NUM_BLOCK_B16], ub_vmax[self.wdim],
                                       0, self.line_per_iter-1, 2, move_stride, 0)
                    conv_times = self.element_per_iter // Cst.VEC_MASK_B32
                    vconv_ub(self.tik, Cst.VEC_MASK_B32, ub_vmax_fp16, ub_vmax, conv_times, 4, 8)
                    self.tik.vcgmax(Cst.VEC_MASK_B16, ub_reducemax_fp16, ub_vmax_fp16,
                                    ceil_div(self.line_per_iter, Cst.NUM_BLOCK_B16), 1, 1, 8)
                    vconv_ub(self.tik, Cst.NUM_BLOCK_B8, ub_reducemax, ub_reducemax_fp16, 1, 4, 2)

            # Minus max value of each line
            self.tik.vmuls(self.line_per_iter, ub_reducemax, ub_reducemax, -1.0, 1, 1, 1, 0, 0)
            maximum = self.tik.Scalar("float32", name="maximum")
            for i in range(self.line_per_iter):
                maximum.set_as(ub_reducemax[i])
                offset = self.wdim * i
                self.tik_func_scalar("vadds", ub_fp32, ub_fp32, maximum, offset, offset, self.wdim, Cst.VEC_MASK_B32)

    # 'pylint: disable=too-many-arguments
    def reduce_max_per_line(self, src, offset, cnt, remain, mask, block):
        """
        Do reduce max for each line.
        """
        time = Cst.SCALAR_TWO
        for _ in range(1, cnt):
            time = time * Cst.SCALAR_TWO
            self.tik_func_binary("vmax", src, src, src, offset, offset,
                                 offset + self.wdim // time, self.wdim // time, mask)
        if remain > 0:
            for j in range(1, remain + 1):
                self.tik_func_binary("vmax", src, src, src, offset + block * (remain - j),
                                     offset + block * (remain - j), offset + block * (remain - j + 1),
                                     block, block)

    def do_exp(self, ub_2, ub_fp32):
        """
        Do exp on input that converted to fp32.
        """
        conv_times = self.element_per_iter // Cst.VEC_MASK_B32
        if self.x_dtype == 'float16':
            vconv_ub(self.tik, Cst.VEC_MASK_B32, ub_fp32, ub_2, conv_times, 8, 4)

        cnt = conv_times // Cst.MAX_REPEAT
        remain = conv_times % Cst.MAX_REPEAT
        for i in range(cnt):
            self.tik.vexp(Cst.VEC_MASK_B32, ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_B32 * i],
                          ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_B32 * i], Cst.MAX_REPEAT, 1, 1, 8, 8)
        self.tik.vexp(Cst.VEC_MASK_B32, ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_B32 * cnt],
                      ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_B32 * cnt], remain, 1, 1, 8, 8)

    def calc_softmax(self, raw_input, ub_1, ub_fp32):
        """
        Do Softmax on scaled masked x.
        """
        with self.tik.new_stmt_scope():
            ub_reduceadd = self.tik.Tensor("float32", (self.line_per_iter,),
                                           scope=tbe_platform.scope_ubuf, name="ub_reduceadd")
            ub_reduceadd_high_preci = self.tik.Tensor("float32", (self.line_per_iter,),
                                                      scope=tbe_platform.scope_ubuf, name="ub_reduceadd_high_preci")
            one_ub = self.tik.Tensor("float32", (self.line_per_iter,), scope=tbe_platform.scope_ubuf, name="one_ub")

            # Save the input for following calculation
            move_burst = self.element_per_iter // Cst.NUM_BLOCK_B32
            self.tik.data_move(raw_input, ub_fp32, 0, 1, move_burst, 0, 0)

            # Do reduce_add
            cnt, remain = self.cal_level(self.wdim, Cst.NUM_BLOCK_B16)
            move_time = self.line_per_iter - 1
            move_stride = (self.wdim - Cst.NUM_BLOCK_B16) // Cst.NUM_BLOCK_B32
            for i in range(self.line_per_iter):
                offset = self.wdim * i
                self.reduce_add_per_line(ub_fp32, offset, cnt, remain)
            self.tik.data_move(ub_fp32[Cst.NUM_BLOCK_B16], ub_fp32[self.wdim],
                               0, move_time, Cst.SCALAR_TWO, move_stride, 0)
            self.tik.vcadd(Cst.NUM_BLOCK_B16, ub_reduceadd, ub_fp32, self.line_per_iter, 1, 1, 2)

            # cal 1 / x_sum and muls x
            stride = self.line_per_iter // Cst.NUM_BLOCK_B32
            self.tik.vec_dup(self.line_per_iter, one_ub, 1.0, 1, stride)
            self.tik.vdiv(self.line_per_iter, ub_reduceadd_high_preci, one_ub, ub_reduceadd,
                          1, 1, 1, 1, stride, stride, stride)
            add_sum = self.tik.Scalar("float32", name="add_sum")
            for i in range(self.line_per_iter):
                add_sum.set_as(ub_reduceadd_high_preci[i])
                offset = self.wdim * i
                self.tik_func_scalar("vmuls", raw_input, raw_input, add_sum,
                                     offset, offset, self.wdim, Cst.VEC_MASK_B32)

            if self.x_dtype == "float16":
                conv_times = self.element_per_iter // Cst.VEC_MASK_B32
                vconv_ub(self.tik, Cst.VEC_MASK_B32, ub_1, raw_input, conv_times, 4, 8)

    def reduce_add_per_line(self, ub_fp32, offset, cnt, remain):
        """
        Do reduce add for each line.
        """
        time = 1
        for _ in range(cnt):
            time = time * Cst.SCALAR_TWO
            self.tik_func_binary("vadd", ub_fp32, ub_fp32, ub_fp32, offset, offset,
                                 offset + self.wdim // time, self.wdim // time, Cst.VEC_MASK_B32)
        if remain > 0:
            for j in range(1, remain + 1):
                self.tik_func_binary("vadd", ub_fp32, ub_fp32, ub_fp32,
                                     offset + Cst.NUM_BLOCK_B16 * (remain - j),
                                     offset + Cst.NUM_BLOCK_B16 * (remain - j),
                                     offset + Cst.NUM_BLOCK_B16 * (remain - j + 1),
                                     Cst.NUM_BLOCK_B16, Cst.NUM_BLOCK_B16)

    def move_data_out(self, raw_input, ub_1, offset):
        """
        Move result from ub to gm.
        """
        if self.x_dtype == "bfloat16":
            conv_times = self.element_per_iter // Cst.VEC_MASK_B32
            move_burst = self.element_per_iter // Cst.NUM_BLOCK_B16
            ub_bf16 = self.tik.Tensor("bfloat16", self.shape, scope=tbe_platform.scope_ubuf, name="ub_bf16")
            vconv_ub(self.tik, Cst.VEC_MASK_B32, ub_bf16, raw_input, conv_times, 4, 8, 'round')
            self.tik.data_move(self.y_gm[offset], ub_bf16, 0, 1, move_burst, 0, 0)
        elif self.x_dtype == "float16":
            conv_times = self.element_per_iter // Cst.VEC_MASK_B32
            move_burst = self.element_per_iter // Cst.NUM_BLOCK_B16
            self.tik.data_move(self.y_gm[offset], ub_1, 0, 1, move_burst, 0, 0)
        else:
            move_burst = self.element_per_iter // Cst.NUM_BLOCK_B32
            self.tik.data_move(self.y_gm[offset], raw_input, 0, 1, move_burst, 0, 0)

    # 'pylint: disable=too-many-arguments
    def tik_func_binary(self, func_str, dst, src0, src1, offset0, offset1, offset2, count, mask):
        """
        Do binary operation by tik function.
        """
        if func_str == "vadd":
            func = self.tik.vadd
        elif func_str == "vmax":
            func = self.tik.vmax
        elif func_str == "vmul":
            func = self.tik.vmul
        else:
            raise RuntimeError("[func_str] should be in ['vadd', 'vmax', 'vmul']")
        repeat_times = count // mask
        remain = count % mask
        if repeat_times > 0:
            max_repeat_times = repeat_times // Cst.MAX_REPEAT
            repeat_remain = repeat_times % Cst.MAX_REPEAT
            for _ in range(max_repeat_times):
                func(mask, dst[offset0], src0[offset1], src1[offset2], Cst.MAX_REPEAT, 1, 1, 1, 8, 8, 8)
                offset0 += mask * Cst.MAX_REPEAT
                offset1 += mask * Cst.MAX_REPEAT
                offset2 += mask * Cst.MAX_REPEAT
            if repeat_remain > 0:
                func(mask, dst[offset0], src0[offset1], src1[offset2], repeat_remain, 1, 1, 1, 8, 8, 8)
                offset0 += mask * repeat_remain
                offset1 += mask * repeat_remain
                offset2 += mask * repeat_remain
        if remain > 0:
            func(remain, dst[offset0], src0[offset1], src1[offset2], 1, 1, 1, 1, 0, 0, 0)

    # 'pylint: disable=too-many-arguments
    def tik_func_scalar(self, func_str, dst, src0, scalar, offset0, offset1, count, mask):
        """
        Do scalar operation by tik function.
        """
        if func_str == "vadds":
            func = self.tik.vadds
        elif func_str == "vmuls":
            func = self.tik.vmuls
        else:
            raise RuntimeError("[func_str] should be in ['vadds', 'vmuls']")
        repeat_times = count // mask
        remain = count % mask
        if repeat_times > 0:
            max_repeat_times = repeat_times // Cst.MAX_REPEAT
            repeat_remain = repeat_times % Cst.MAX_REPEAT
            for _ in range(max_repeat_times):
                func(mask, dst[offset0], src0[offset1], scalar, Cst.MAX_REPEAT, 1, 1, 8, 8)
                offset0 += mask * Cst.MAX_REPEAT
                offset1 += mask * Cst.MAX_REPEAT
            if repeat_remain > 0:
                func(mask, dst[offset0], src0[offset1], scalar, repeat_remain, 1, 1, 8, 8)
                offset0 += mask * repeat_remain
                offset1 += mask * repeat_remain
        if remain > 0:
            func(remain, dst[offset0], src0[offset1], scalar, 1, 1, 1, 0, 0)

    def build_cce(self):
        """
        Build CCE.
        """
        self.tik.BuildCCE(kernel_name=self.kernel_name,
                          inputs=[self.x_gm, self.mask_gm],
                          outputs=[self.y_gm])
        return self.tik


def ceil_div(dividend, divisor):
    """
    Calculate the minimum value that is divisible by the dividend.
    """
    return (dividend + divisor - 1) // divisor


def cal_level(dividend):
    """
    Calculate vmax and vadd times.
    """
    cnt = 0
    while dividend % Cst.SCALAR_TWO == 0:
        dividend //= Cst.SCALAR_TWO
        cnt += 1
    return cnt, dividend - 1


# 'pylint: disable=too-many-arguments
def vconv_ub(tik_instance, mask, dst, src, times, dst_stride, src_stride, round_mode=''):
    """
    Convert raw ub to another ub in different dtype.
    """
    repeat_time = times // Cst.MAX_REPEAT
    repeat_remain = times % Cst.MAX_REPEAT
    if repeat_time > 0:
        for i in range(repeat_time):
            tik_instance.vconv(mask, round_mode, dst[mask * Cst.MAX_REPEAT * i], src[mask * Cst.MAX_REPEAT * i],
                               Cst.MAX_REPEAT, 1, 1, dst_stride, src_stride)
    if repeat_remain > 0:
        tik_instance.vconv(mask, round_mode, dst[mask * Cst.MAX_REPEAT * repeat_time],
                           src[mask * Cst.MAX_REPEAT * repeat_time], repeat_remain, 1, 1, dst_stride, src_stride)


# 'pylint: disable=too-many-arguments
def move_data_out(move_params, params_list, ub_1, ub_2, ub_fp32, y_gm):
    """
    Move result from ub to gm.
    """
    tik_instance, _, _, _, line = params_list
    _, x_shape, offset, _, conv_times, _, _, if_input_nd, w_dim, ub_shape, tensor_dtype = move_params
    count = line * Cst.NUM_BLOCK_B16 // Cst.DIV
    if tensor_dtype == "bfloat16":
        if if_input_nd:
            ub_2_nd = ub_2.reshape((ub_shape[1], w_dim))
            ub_fp32_nd = ub_fp32.reshape((ub_shape[1], w_dim))
            ub_1_bf16 = ub_1.reinterpret_cast_to("bfloat16")
            ub_1_bf16_nd = ub_1_bf16.reshape((ub_shape[1], w_dim))
            with tik_instance.for_range(0, ub_shape[1]) as h_index:
                tik_instance.data_move(ub_2_nd[h_index, 0], ub_1[0, h_index, 0],
                                       0, x_shape[Cst.DIM_W1], 1, ub_shape[1] - 1, 0)

            vconv_ub(tik_instance, Cst.VEC_MASK_B32, ub_fp32_nd, ub_2_nd, conv_times, 8, 4)
            vconv_ub(tik_instance, Cst.VEC_MASK_B32, ub_1_bf16_nd, ub_fp32_nd, conv_times, 4, 8, 'round')
            tik_instance.data_move(y_gm[offset], ub_1_bf16_nd, 0, 1, x_shape[Cst.DIM_W1] * count, 0, 0)
        else:
            ub_1_bf16 = ub_1.reinterpret_cast_to("bfloat16")
            vconv_ub(tik_instance, Cst.VEC_MASK_B32, ub_fp32, ub_1, conv_times, 8, 4)
            vconv_ub(tik_instance, Cst.VEC_MASK_B32, ub_1_bf16, ub_fp32, conv_times, 4, 8, 'round')
            tik_instance.data_move(y_gm[offset], ub_1_bf16, 0, x_shape[Cst.DIM_W1],
                                   count, 0, (x_shape[Cst.DIM_H1] * Cst.NUM_BLOCK_B16 - count))
    else:
        if if_input_nd:
            ub_2_nd = ub_2.reshape((ub_shape[1], w_dim))
            with tik_instance.for_range(0, ub_shape[1]) as h_index:
                tik_instance.data_move(ub_2_nd[h_index, 0], ub_1[0, h_index, 0],
                                       0, x_shape[Cst.DIM_W1], 1, ub_shape[1] - 1, 0)
            tik_instance.data_move(y_gm[offset], ub_2_nd, 0, 1, x_shape[Cst.DIM_W1] * count, 0, 0)
        else:
            tik_instance.data_move(y_gm[offset], ub_1, 0, x_shape[Cst.DIM_W1],
                                   count, 0, (x_shape[Cst.DIM_H1] * Cst.NUM_BLOCK_B16 - count))


def calc_softmax(params_list, reduce_sum_list):
    """
    Do Softmax on input.
    """
    tik_instance, [x_shape, _], _, w_dim, line = params_list
    ub_fp32, ub_reduceadd, ub_reduceadd_high_preci, work_tensor_ub, ub_dup_fp32, ub_1 = reduce_sum_list
    counts = line * Cst.NUM_BLOCK_B16 * w_dim // Cst.DIV
    cnt, remain = cal_level(x_shape[Cst.DIM_W1])

    # Convert to fp16 for reduce_add
    conv_times = counts // Cst.VEC_MASK_B32
    vconv_ub(tik_instance, Cst.VEC_MASK_B32, ub_1, ub_fp32, conv_times, 4, 8)

    # Do reduce_add, vec_rec and braodcast
    time = 1
    for j in range(cnt):
        time *= Cst.SCALAR_TWO
        tik_instance.vadd(Cst.VEC_MASK_B32, ub_fp32, ub_fp32, ub_fp32[counts // time],
                          counts // time // Cst.VEC_MASK_B32, 1, 1, 1, 8, 8, 8)
    counts = Cst.NUM_BLOCK_B16 * Cst.NUM_BLOCK_B16 * line // Cst.DIV
    if remain > 0:
        for j in range(1, remain + 1):
            tik_instance.vadd(Cst.VEC_MASK_B32, ub_fp32[counts * (remain - j)], ub_fp32[counts * (remain - j)],
                              ub_fp32[counts * (remain - j + 1)], counts // Cst.VEC_MASK_B32, 1, 1, 1, 8, 8, 8)
    tik_instance.vcadd(Cst.NUM_BLOCK_B16, ub_reduceadd, ub_fp32, counts // Cst.NUM_BLOCK_B16, 1, 1, 2)
    tik_instance.vec_rec_high_preci(counts // Cst.NUM_BLOCK_B16, ub_reduceadd_high_preci[0], ub_reduceadd[0],
                                    work_tensor_ub[0:], 1, 4, 4)
    dup_times = line * x_shape[Cst.DIM_H0] // Cst.DIV // 8
    for j in range(dup_times):
        for k in range(8):
            tik_instance.vector_dup(Cst.NUM_BLOCK_B16, ub_dup_fp32[j * 128 + 16 * k],
                                    tik_instance.Scalar(init_value=ub_reduceadd_high_preci[j * 8 + k], dtype="float32"),
                                    1, 1, 8)

    # Covert to fp32 for doing vmul and convet to fp16 when vuml completed
    vconv_ub(tik_instance, Cst.VEC_MASK_B32, ub_fp32, ub_1, conv_times, 8, 4)
    stride = line * Cst.NUM_BLOCK_B16 * 2 // Cst.DIV
    for idx in range(counts // Cst.VEC_MASK_B32):
        tik_instance.vmul(Cst.VEC_MASK_B32, ub_fp32[idx * Cst.VEC_MASK_B32], ub_fp32[idx * Cst.VEC_MASK_B32],
                          ub_dup_fp32[idx * Cst.VEC_MASK_B32], x_shape[Cst.DIM_W1], 1, 1, 1, stride, stride, 0)
    vconv_ub(tik_instance, Cst.VEC_MASK_B32, ub_1, ub_fp32, conv_times, 4, 8)


def do_exp(params_list, exp_list):
    """
    Do exp on input converted to fp32.
    """
    tik_instance, _, _, w_dim, line = params_list
    ub_2, ub_fp32 = exp_list
    conv_time = line * Cst.NUM_BLOCK_B16 * w_dim // Cst.VEC_MASK_B32 // Cst.DIV
    vconv_ub(tik_instance, Cst.VEC_MASK_B32, ub_fp32, ub_2, conv_time, 8, 4)
    cnt = conv_time // Cst.MAX_REPEAT
    remain = conv_time % Cst.MAX_REPEAT
    for i in range(cnt):
        tik_instance.vexp(Cst.VEC_MASK_B32, ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_B32 * i],
                          ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_B32 * i], Cst.MAX_REPEAT, 1, 1, 8, 8)
    tik_instance.vexp(Cst.VEC_MASK_B32, ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_B32 * cnt],
                      ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_B32 * cnt], remain, 1, 1, 8, 8)


def smooth_by_argmax(params_list, reduce_max_list):
    """
    Input minus maximum value prevents exponent overflow.
    """
    tik_instance, [x_shape, _], _, w_dim, line = params_list
    ub_1, ub_2, ub_mask_fp16, ub_reducemax, ub_broadcast, ub_dup = reduce_max_list
    cnt, remain = cal_level(x_shape[Cst.DIM_W1])
    time = Cst.SCALAR_TWO
    counts = line * Cst.NUM_BLOCK_B16 * w_dim // Cst.DIV // Cst.VEC_MASK_B16

    # Do reduce_max and broadcast
    tik_instance.vmul(Cst.VEC_MASK_B16, ub_1, ub_1, ub_mask_fp16, counts, 1, 1, 1, 8, 8, 8)
    tik_instance.vadd(Cst.VEC_MASK_B16, ub_1, ub_1, ub_2, counts, 1, 1, 1, 8, 8, 8)
    tik_instance.vmax(Cst.VEC_MASK_B16, ub_2, ub_1, ub_1[counts * Cst.VEC_MASK_B16 // time],
                      counts // time, 1, 1, 1, 8, 8, 8)
    for j in range(1, cnt):
        time = time * Cst.SCALAR_TWO
        tik_instance.vmax(Cst.VEC_MASK_B16, ub_2, ub_2, ub_2[counts * Cst.VEC_MASK_B16 // time],
                          counts // time, 1, 1, 1, 8, 8, 8)
    vmax_counts = Cst.NUM_BLOCK_B16 * Cst.NUM_BLOCK_B16 * line // Cst.DIV
    if remain > 0:
        for j in range(1, remain + 1):
            tik_instance.vmax(Cst.VEC_MASK_B16, ub_2[vmax_counts * (remain - j)], ub_2[vmax_counts * (remain - j)],
                              ub_2[vmax_counts * (remain - j + 1)], vmax_counts // Cst.VEC_MASK_B16, 1, 1, 1, 8, 8, 8)

    tik_instance.vcgmax(Cst.VEC_MASK_B16, ub_reducemax, ub_2, Cst.SCALAR_TWO * line // Cst.DIV, 1, 1, 8)
    tik_instance.vector_dup(Cst.VEC_MASK_B16, ub_dup, tik_instance.Scalar(init_value=0, dtype="uint16"), 1, 1, 8)
    ub_reducemax_uint16 = ub_reducemax.reinterpret_cast_to("uint16")
    with tik_instance.for_range(0, line) as j:
        tik_instance.vor(Cst.NUM_BLOCK_B16, ub_broadcast[Cst.NUM_BLOCK_B16 * Cst.NUM_BLOCK_B16 * j],
                         ub_reducemax_uint16[Cst.NUM_BLOCK_B16 * j], ub_dup, Cst.NUM_BLOCK_B16, 1, 1, 0, 1, 0, 0)
    for j in range(line):
        tik_instance.vtranspose(ub_broadcast[Cst.NUM_BLOCK_B16 * Cst.NUM_BLOCK_B16 * j],
                                ub_broadcast[Cst.NUM_BLOCK_B16 * Cst.NUM_BLOCK_B16 * j])
    ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")

    # Minus max value of each line
    for idx in range(vmax_counts // 64):
        tik_instance.vsub(64, ub_2[idx * 64], ub_1[idx * 64], ub_broadcast_fp16[idx * 64], x_shape[Cst.DIM_W1],
                          1, 1, 1, vmax_counts // Cst.NUM_BLOCK_B16, vmax_counts // Cst.NUM_BLOCK_B16, 0)


# 'pylint: disable=too-many-locals,too-many-statements,too-many-arguments
def x_move_in(tik_instance, ub_1, ub_fp32, x_gm, move_params):
    """
    Move x from gm tensor to ub tensor.
    """
    _, x_shape, offset, _, conv_times, burst, src_stride, if_input_nd, w_dim, ub_shape, tensor_dtype = move_params
    if tensor_dtype == 'float16':
        if if_input_nd:
            with tik_instance.for_range(0, ub_shape[1]) as h_index:
                tik_instance.data_move(ub_1[0, h_index, 0], x_gm[offset + h_index * w_dim], 0,
                                       x_shape[Cst.DIM_W1], 1, 0, ub_shape[1] - 1)
        else:
            tik_instance.data_move(ub_1, x_gm[offset], 0, x_shape[Cst.DIM_W1], burst, src_stride, 0)
    elif tensor_dtype == 'bfloat16':
        ub_1_bf16 = ub_1.reinterpret_cast_to("bfloat16")
        if if_input_nd:
            with tik_instance.for_range(0, ub_shape[1]) as h_index:
                tik_instance.data_move(ub_1_bf16[0, h_index, 0], x_gm[offset + h_index * w_dim], 0,
                                       x_shape[Cst.DIM_W1], 1, 0, ub_shape[1] - 1)
        else:
            tik_instance.data_move(ub_1_bf16, x_gm[offset], 0, x_shape[Cst.DIM_W1], burst, src_stride, 0)
        vconv_ub(tik_instance, Cst.VEC_MASK_B32, ub_fp32, ub_1_bf16, conv_times, 8, 4)
        vconv_ub(tik_instance, Cst.VEC_MASK_B32, ub_1, ub_fp32, conv_times, 4, 8)
    else:
        tik_instance.data_move(ub_fp32, x_gm[offset], 0, x_shape[Cst.DIM_W1],
                               burst * 2, src_stride * 2, 0)
        vconv_ub(tik_instance, Cst.VEC_MASK_B32, ub_1, ub_fp32, conv_times, 4, 8)


# 'pylint: disable=too-many-locals,too-many-statements,too-many-arguments
def mask_move_in(tik_instance, ub_mask_fp16, ub_mask, ub_2, mask_gm, move_params):
    """
    Move mask from gm tensor to ub tensor.
    """
    fixed_triu_mask, x_shape, _, offset_mask, _, burst, src_stride, if_input_nd, w_dim, ub_shape, _ = move_params
    if fixed_triu_mask:
        tik_instance.data_move(ub_mask_fp16, mask_gm[offset_mask], 0,
                               x_shape[Cst.DIM_W1], burst, src_stride, 0)
    else:
        if if_input_nd:
            ub_mask_nd = ub_mask.reshape((burst, w_dim))
            ub_2_nd = ub_2.reshape((burst, w_dim))
            tik_instance.data_move(ub_mask_nd, mask_gm[offset_mask], 0, 1,
                                   x_shape[Cst.DIM_W1] * burst // Cst.SCALAR_TWO, 0, 0)
            vconv_ub(tik_instance, Cst.VEC_MASK_B16, ub_2_nd, ub_mask_nd, burst * w_dim // Cst.VEC_MASK_B16, 8, 4)
            with tik_instance.for_range(0, ub_shape[1]) as h_index:
                tik_instance.data_move(ub_mask_fp16[0, h_index, 0], ub_2_nd[h_index, 0], 0,
                                       x_shape[Cst.DIM_W1], 1, 0, ub_shape[1]-1)
        else:
            tik_instance.data_move(ub_mask, mask_gm[offset_mask], 0, x_shape[Cst.DIM_W1],
                                   burst // Cst.SCALAR_TWO, src_stride // Cst.SCALAR_TWO, 0)
            vconv_ub(tik_instance, Cst.VEC_MASK_B16, ub_mask_fp16, ub_mask, burst * w_dim // Cst.VEC_MASK_B16, 8, 4)


def create_ub(tik_instance, shape, line):
    """
    Create ub tensors.
    """
    ub_1 = tik_instance.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="ub_1")
    ub_2 = tik_instance.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="ub_2")
    ub_fp32 = tik_instance.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="ub_fp32")

    ub_shape = line * Cst.NUM_BLOCK_B16 // Cst.DIV
    ub_reducemax = tik_instance.Tensor("float16", (ub_shape,), scope=tbe_platform.scope_ubuf, name="ub_reducemax")
    ub_reduceadd = tik_instance.Tensor("float32", (ub_shape,), scope=tbe_platform.scope_ubuf, name="ub_reduceadd")
    ub_reduceadd_high_preci = tik_instance.Tensor("float32", (ub_shape,),
                                                  scope=tbe_platform.scope_ubuf, name="ub_reduceadd_high_preci")

    work_tensor_ub = tik_instance.Tensor("float32", (2 * ub_shape,),
                                         scope=tbe_platform.scope_ubuf, name="work_tensor_ub")
    ub_dup = tik_instance.Tensor("uint16", (Cst.VEC_MASK_B16,), scope=tbe_platform.scope_ubuf, name="ub_dup")
    ub_dup_fp32 = tik_instance.Tensor("float32", (32, 16), scope=tbe_platform.scope_ubuf, name="ub_dup_fp32")
    ub_broadcast = tik_instance.Tensor("uint16", (line * Cst.NUM_BLOCK_B16 * Cst.NUM_BLOCK_B16,),
                                       scope=tbe_platform.scope_ubuf, name="ub_broadcast")
    ub_mask_fp16 = tik_instance.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="ub_mask_fp16")
    ub_mask = tik_instance.Tensor("uint8", shape, scope=tbe_platform.scope_ubuf, name="ub_mask")

    loop_per_core = tik_instance.Scalar("int32", name="loop_per_core")
    offset = tik_instance.Scalar("int32", name="offset")
    offset_mask = tik_instance.Scalar("int32", name="offset_mask")

    return [ub_1, ub_2, ub_fp32, ub_reducemax, ub_reduceadd,
            ub_reduceadd_high_preci, work_tensor_ub, ub_dup, ub_dup_fp32,
            ub_broadcast, ub_mask_fp16, ub_mask, loop_per_core, offset, offset_mask]


# 'pylint: disable=too-many-locals,too-many-statements,too-many-arguments
def scaled_masked_softmax_compute(params_list, core_attr_list, gm_tensors,
                                  scale, core_index, tensor_dtype, nd_support_params):
    """
    Implementation of fused scaled masked softmax algorithm.
    """
    tik_instance, [x_shape, _], _, w_dim, line = params_list
    used_core_num, _, _, iter_per_batch, iter_per_channel, iter_per_core, \
        iter_last_core, element_per_iter, element_per_core, shape, counts, broad_ratio = core_attr_list
    x_gm, mask_gm, y_gm, fixed_triu_mask = gm_tensors
    ub_1, ub_2, ub_fp32, ub_reducemax, ub_reduceadd, ub_reduceadd_high_preci, \
        work_tensor_ub, ub_dup, ub_dup_fp32, ub_broadcast, ub_mask_fp16, ub_mask, \
        loop_per_core, offset, offset_mask = create_ub(tik_instance, shape, line)
    with tik_instance.if_scope(core_index == used_core_num - 1):
        loop_per_core.set_as(iter_last_core)
    with tik_instance.else_scope():
        loop_per_core.set_as(iter_per_core)

    with tik_instance.for_range(0, loop_per_core) as it:
        current_batch = (core_index * iter_per_core + it) // iter_per_batch
        current_channel = (core_index * iter_per_core + it) % iter_per_batch // iter_per_channel
        iter_in_curr_channel = core_index * iter_per_core + it - current_batch * iter_per_batch - \
            current_channel * iter_per_channel
        offset_mask.set_as(((current_batch * x_shape[Cst.DIM_C] + current_channel) // broad_ratio) *
                           iter_per_channel * element_per_iter + iter_in_curr_channel * counts)
        offset.set_as((current_batch * x_shape[Cst.DIM_C] + current_channel) *
                      iter_per_channel * element_per_iter + iter_in_curr_channel * counts)

        conv_times = line * Cst.NUM_BLOCK_B16 * w_dim // Cst.VEC_MASK_B32 // Cst.DIV
        burst = Cst.NUM_BLOCK_B16 * line // Cst.DIV
        src_stride = x_shape[Cst.DIM_H1] * Cst.NUM_BLOCK_B16 - line * Cst.NUM_BLOCK_B16 // Cst.DIV
        ub_shape = ub_1.shape
        repeat_times = w_dim * line * Cst.NUM_BLOCK_B16 // Cst.DIV // Cst.VEC_MASK_B16
        move_params = [fixed_triu_mask, x_shape, offset, offset_mask, conv_times,
                       burst, src_stride, nd_support_params[0], w_dim, ub_shape, tensor_dtype]

        mask_move_in(tik_instance, ub_mask_fp16, ub_mask, ub_2, mask_gm, move_params)
        tik_instance.vmuls(Cst.VEC_MASK_B16, ub_2, ub_mask_fp16,
                           tik_instance.Scalar(init_value=-10000., dtype="float16"), repeat_times, 1, 1, 8, 8)
        tik_instance.vmuls(Cst.VEC_MASK_B16, ub_mask_fp16, ub_mask_fp16,
                           tik_instance.Scalar(init_value=-1., dtype="float16"), repeat_times, 1, 1, 8, 8)
        tik_instance.vadds(Cst.VEC_MASK_B16, ub_mask_fp16, ub_mask_fp16,
                           tik_instance.Scalar(init_value=1., dtype="float16"), repeat_times, 1, 1, 8, 8)

        x_move_in(tik_instance, ub_1, ub_fp32, x_gm, move_params)
        tik_instance.vmuls(Cst.VEC_MASK_B16, ub_1, ub_1,
                           tik_instance.Scalar(init_value=scale, dtype="float16"), repeat_times, 1, 1, 8, 8)

        smooth_by_argmax(params_list, [ub_1, ub_2, ub_mask_fp16, ub_reducemax, ub_broadcast, ub_dup])
        do_exp(params_list, [ub_2, ub_fp32])
        calc_softmax(params_list, [ub_fp32, ub_reduceadd, ub_reduceadd_high_preci, work_tensor_ub, ub_dup_fp32, ub_1])
        move_data_out(move_params, params_list, ub_1, ub_2, ub_fp32, y_gm)


def gen_triu_mask(params_list, mask_gm):
    """
    Generate triu mask.
    """
    tik_instance, shape_list, h_dim, w_dim, _ = params_list
    w_dim = shape_list[0][Cst.DIM_W1] * shape_list[0][Cst.DIM_W0]
    h_dim = shape_list[0][Cst.DIM_H1] * shape_list[0][Cst.DIM_H0]

    vec_len = 1
    tri_mask_ub = tik_instance.Tensor('float16', (w_dim,), name="tri_mask_ub", scope=tbe_platform.scope_ubuf)
    tik_vec_dup(tik_instance, tri_mask_ub, 1.0, 0, w_dim, Cst.VEC_MASK_B16)
    nburst = w_dim // Cst.NUM_BLOCK_B16
    dst_stride = h_dim - 1
    offset_gm = 0

    for _ in range(h_dim):
        if vec_len < w_dim:
            tik_vec_dup(tik_instance, tri_mask_ub, 0.0, 0, vec_len, Cst.VEC_MASK_B16)
            vec_len += 1
            vec_len = vec_len if vec_len < w_dim else w_dim
        tik_instance.data_move(mask_gm[offset_gm], tri_mask_ub, 0, nburst, 1, 0, dst_stride)
        offset_gm += Cst.NUM_BLOCK_B16


# 'pylint: disable=too-many-arguments
def tik_vec_dup(tik_instance, dst, value, offset, len, mask):
    """
    Tik function vec_dup.
    """
    repeat_times = len // mask
    repeat = repeat_times // Cst.MAX_REPEAT
    repeat_remain = repeat_times % Cst.MAX_REPEAT
    remain = len % mask
    temp_offset = tik_instance.Scalar("int32", init_value=offset)
    with tik_instance.if_scope(repeat_times > 0):
        with tik_instance.if_scope(repeat > 0):
            with tik_instance.for_range(0, repeat) as i:
                tik_instance.vec_dup(mask, dst[temp_offset], value, Cst.MAX_REPEAT, 8)
                temp_offset.set_as(temp_offset + Cst.MAX_REPEAT * mask)
        with tik_instance.if_scope(repeat_remain > 0):
            tik_instance.vec_dup(mask, dst[temp_offset], value, repeat_remain, 8)
            temp_offset.set_as(temp_offset + repeat_remain * mask)
    with tik_instance.if_scope(remain > 0):
        tik_instance.vec_dup(remain, dst[temp_offset], value, 1, 0)


def create_gm_tensor(tik_instance, tensor_shape, mask_shape, x_dtype, y_dtype):
    """
    Create input and output gm tensors.
    """
    x_gm = tik_instance.Tensor(x_dtype, tensor_shape, name="x_gm", scope=tbe_platform.scope_gm)
    mask_gm = tik_instance.Tensor('bool', mask_shape, name="mask_gm", scope=tbe_platform.scope_gm)
    y_gm = tik_instance.Tensor(y_dtype, tensor_shape, name="y_gm", scope=tbe_platform.scope_gm)
    return x_gm, mask_gm, y_gm


def cal_params_list(shape_list, fixed_triu_mask, nd_support_params):
    """
    Calculate params for following computing.
    """
    # Calculate params_list
    tik_instance = tik.Tik(tik.Dprofile())
    aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    h_dim = shape_list[0][Cst.DIM_H1] * shape_list[0][Cst.DIM_H0]
    w_dim = shape_list[0][Cst.DIM_W1] * shape_list[0][Cst.DIM_W0]
    line = 2 if w_dim <= 512 else 1
    Cst.DIV = 1 if w_dim <= 1024 else 2

    # Calculate core_attr_list
    line_per_iter = line * Cst.NUM_BLOCK_B16 // Cst.DIV
    total_iter = shape_list[0][Cst.DIM_N] * shape_list[0][Cst.DIM_C] * shape_list[0][Cst.DIM_H1] * \
        shape_list[0][Cst.DIM_H0] // line_per_iter
    iter_per_batch = shape_list[0][Cst.DIM_C] * shape_list[0][Cst.DIM_H1] * shape_list[0][Cst.DIM_H0] // line_per_iter
    iter_per_channel = shape_list[0][Cst.DIM_H1] * shape_list[0][Cst.DIM_H0] // line_per_iter
    iter_per_core = (total_iter + aicore_num - 1) // aicore_num
    used_core_num = (total_iter + iter_per_core - 1) // iter_per_core
    iter_last_core = total_iter - (used_core_num - 1) * iter_per_core
    element_per_iter = w_dim * line_per_iter
    element_per_core = element_per_iter * iter_per_core
    shape = (shape_list[0][Cst.DIM_W1], line_per_iter, shape_list[0][Cst.DIM_W0])
    if nd_support_params[0]:
        counts = line * Cst.NUM_BLOCK_B16 * w_dim // Cst.DIV
    else:
        counts = line * Cst.NUM_BLOCK_B16 * Cst.NUM_BLOCK_B16 // Cst.DIV
    if fixed_triu_mask:
        broad_ratio = shape_list[0][Cst.DIM_N] * shape_list[0][Cst.DIM_C]
    else:
        broad_ratio = shape_list[0][Cst.DIM_N] * shape_list[0][Cst.DIM_C] // \
            (shape_list[1][Cst.DIM_N] * shape_list[1][Cst.DIM_C])

    return [tik_instance, shape_list, h_dim, w_dim, line], \
           [used_core_num, line_per_iter, total_iter, iter_per_batch, iter_per_channel,
            iter_per_core, iter_last_core, element_per_iter, element_per_core, shape, counts, broad_ratio]


def get_nz_shape(x):
    """
    Get x nz shape and mask nz shape.
    """
    x_shape = x.get("shape")
    if x.get("format") == "FRACTAL_NZ":
        return x_shape
    else:
        x_nz_shape = (x_shape[Cst.DIM_N],
                      x_shape[Cst.DIM_C],
                      x_shape[Cst.DIM_W] // Cst.ALIGN_NUM,
                      x_shape[Cst.DIM_H] // Cst.ALIGN_NUM,
                      Cst.ALIGN_NUM,
                      Cst.ALIGN_NUM)
        return x_nz_shape


# 'pylint: disable=unused-argument,too-many-arguments
# 'pylint: disable=too-many-locals,too-many-statements
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def scaled_masked_softmax(x, mask, y, scale=1.0, fixed_triu_mask=False, kernel_name="scaled_masked_softmax"):
    """
    Algorithm:
        mask = torch.triu(mask.shape, diagonal=1) if fixed_triu_mask else mask
        y = torch.softmax((x * scale).masked_fill(mask, -inf), dim=-1)

    Parameters
    ----------
    x : dict
        shape and dtype of input tensor, the shape must be 6D in format Fractal_NZ.
    mask : dict
        shape and dtype of mask, the shape must be broadcastble with x.
    y : dict
        shape and dtype of output, the shape must be same as x.
    scale : float
        a float scalar scaling the input tensor x
    fixed_triu_mask : boolcat a
        if true: the mask is a fixed upper triangle mask
        if false: the mask is input mask
    kernel_name : str
        kernel name, default value is "scaled_masked_softmax"

    Returns
    -------
    None
    """
    nd_support_params = (x.get("format") != "FRACTAL_NZ", x.get("shape"))
    if nd_support_params[0]:
        return ScaledMaskedSoftmaxNd(x, mask, y, scale, fixed_triu_mask, kernel_name).compute()

    x_dtype = x.get("dtype").lower()
    y_dtype = y.get("dtype").lower()
    input_nz_shape, mask_nz_shape = get_nz_shape(x), get_nz_shape(mask)
    params_list, core_attr_list = cal_params_list([input_nz_shape, mask_nz_shape], fixed_triu_mask, nd_support_params)
    tik_instance, [x_shape, mask_shape], h_dim, w_dim, _ = params_list
    if w_dim < 32 or w_dim > 2048 or w_dim % 32 != 0:
        raise RuntimeError('Error! When the format of input is NZ, the last dimension of the input tensor should ' +
                           f'be within the range of [32, 2048] and be divisible by 32. Currently, it is {w_dim}.')
    used_core_num, _, _, _, _, _, _, _, _, _, _, _ = core_attr_list
    x_gm, mask_gm, y_gm = create_gm_tensor(tik_instance, x_shape, mask_shape, x_dtype, y_dtype)

    if fixed_triu_mask:
        fixed_mask_gm = tik_instance.Tensor('float16', mask_shape, name="fixed_mask_gm",
                                            scope=tbe_platform.scope_gm, is_workspace=True)
        gm_tensors = [x_gm, fixed_mask_gm, y_gm, fixed_triu_mask]
        gen_triu_mask(params_list, fixed_mask_gm)
    else:
        mask_gm = mask_gm.reinterpret_cast_to('uint8')
        gm_tensors = [x_gm, mask_gm, y_gm, fixed_triu_mask]

    with tik_instance.for_range(0, used_core_num, block_num=used_core_num) as core_index:
        scaled_masked_softmax_compute(params_list, core_attr_list, gm_tensors,
                                      scale, core_index, x_dtype, nd_support_params)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[x_gm, mask_gm],
                          outputs=[y_gm])
    return tik_instance
