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

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.scaled_masked_softmax import gen_triu_mask
from impl.scaled_masked_softmax import get_nz_shape
from impl.scaled_masked_softmax import cal_params_list
from impl.scaled_masked_softmax import cal_level
from impl.scaled_masked_softmax import vconv_ub
from impl.scaled_masked_softmax import tik_vec_dup


# 'pylint: disable=no-init,old-style-class,too-few-public-methods
class Cst:
    """
    The class for Constant.
    """
    LEN = 2
    DIV = 1
    BLOCK = 16
    DIM_N = 0
    DIM_C = 1
    DIM_H = 2
    DIM_W = 3
    DIM_W1 = 2
    DIM_H1 = 3
    DIM_H0 = 4
    DIM_W0 = 5
    MAX_REPEAT = 255
    SCALAR_TWO = 2
    FULL_LINE = 8
    VEC_MASK_B8 = 256
    VEC_MASK_B16 = 128
    VEC_MASK_B32 = 64
    NUM_BLOCK_B8 = 32
    NUM_BLOCK_B16 = 16
    NUM_BLOCK_B32 = 8
    VEC_DUMP_SHAPE = 32
    SHAPE_SIZE_LIMIT = 1 << 30


# 'pylint: disable=old-style-class,too-many-instance-attributes
class ScaledMaskedSoftmaxGradNd():
    """
    Compute for ND format inputs.
    """

    # 'pylint: disable=unused-argument,too-many-arguments,too-many-locals,too-many-statements
    def __init__(self, y_grad, y, mask, x_grad, scale, fixed_triu_mask, kernel_name):
        """                                                                                                           
        Instantiate ScaledMaskedSoftmaxGradNd object.
        """
        self.y_grad_dtype, self.y_dtype, \
            self.x_grad_dtype = y_grad.get("dtype").lower(), y.get("dtype").lower(), x_grad.get("dtype").lower()
        self.y_grad_shape, self.y_shape, self.mask_shape = y_grad.get("shape"), y.get("shape"), mask.get("shape")
        self.scale = scale
        self.fixed_triu_mask = fixed_triu_mask
        self.kernel_name = kernel_name
        self.cal_params()
        self.create_gm_tensor()

    def cal_params(self):
        """
        Calculate params for following computing.
        """
        # Calculate params_list
        self.tik = tik.Tik(tik.Dprofile())
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.wdim = self.y_grad_shape[Cst.DIM_W]
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
        self.line_per_iter = line * Cst.BLOCK // div
        self.total_iter = self.y_grad_shape[Cst.DIM_N] * self.y_grad_shape[Cst.DIM_C] * \
            self.y_grad_shape[Cst.DIM_H] // self.line_per_iter
        self.iter_per_batch = self.y_grad_shape[Cst.DIM_C] * self.y_grad_shape[Cst.DIM_H] // self.line_per_iter
        self.iter_per_channel = self.y_grad_shape[Cst.DIM_H] // self.line_per_iter
        self.iter_per_core = (self.total_iter + self.aicore_num - 1) // self.aicore_num
        self.used_core_num = (self.total_iter + self.iter_per_core - 1) // self.iter_per_core
        self.iter_last_core = self.total_iter - (self.used_core_num - 1) * self.iter_per_core
        self.element_per_iter = self.wdim * self.line_per_iter
        self.element_per_core = self.element_per_iter * self.iter_per_core
        self.shape = (self.line_per_iter, self.wdim)
        if self.fixed_triu_mask:
            self.broad_ratio_n = self.y_grad_shape[Cst.DIM_N]
            self.broad_ratio_c = self.y_grad_shape[Cst.DIM_C]
        else:
            self.broad_ratio_n = self.y_grad_shape[Cst.DIM_N] // self.mask_shape[Cst.DIM_N]
            self.broad_ratio_c = self.y_grad_shape[Cst.DIM_C] // self.mask_shape[Cst.DIM_C]

    def create_gm_tensor(self):
        """
        Create input and output gm tensors.
        """
        self.y_grad_gm = self.tik.Tensor(self.y_grad_dtype, self.y_grad_shape,
                                         name="y_grad_gm", scope=tbe_platform.scope_gm)
        self.y_gm = self.tik.Tensor(self.y_dtype, self.y_shape, name="y_gm", scope=tbe_platform.scope_gm)
        self.mask_gm = self.tik.Tensor('uint8', self.mask_shape, name="mask_gm", scope=tbe_platform.scope_gm)
        if self.fixed_triu_mask:
            shape = [1, 1, self.mask_shape[Cst.DIM_H], self.mask_shape[Cst.DIM_W]]
            self.fixed_mask_gm = self.tik.Tensor('float16', shape, name="fixed_mask_gm",
                                                 scope=tbe_platform.scope_gm, is_workspace=True)
            self.gen_triu_mask()
        self.x_grad_gm = self.tik.Tensor(self.x_grad_dtype, self.y_grad_shape,
                                         name="x_grad_gm", scope=tbe_platform.scope_gm)

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
            grad_ub, output_ub_fp32, softmax_ub_fp32, mask_ub, \
                ub_reduceadd, loop_per_core, offset, offset_mask = self.create_ub()

            with self.tik.if_scope(core_index == self.used_core_num - 1):
                loop_per_core.set_as(self.iter_last_core)
            with self.tik.else_scope():
                loop_per_core.set_as(self.iter_per_core)

            with self.tik.for_range(0, loop_per_core) as it:
                self.cal_offset(offset, offset_mask, core_index, it)
                self.move_data_in(grad_ub, softmax_ub_fp32, self.y_gm, offset, self.y_dtype)
                self.move_data_in(grad_ub, output_ub_fp32, self.y_grad_gm, offset, self.y_grad_dtype)
                self.calc_product(output_ub_fp32, softmax_ub_fp32)
                if not self.fixed_triu_mask:
                    self.move_mask_in(mask_ub, offset_mask)
                self.calc_reduce_sum(output_ub_fp32, ub_reduceadd)
                self.move_data_in(grad_ub, output_ub_fp32, self.y_grad_gm, offset, self.y_grad_dtype)
                self.calc_softmax_grad(output_ub_fp32, ub_reduceadd, softmax_ub_fp32)
                self.scaled_masked_fill(grad_ub, mask_ub, output_ub_fp32, softmax_ub_fp32, offset_mask)
                self.move_data_out(grad_ub, output_ub_fp32, offset)
        return self.build_cce()

    def create_ub(self):
        """
        Create ub tensors.
        """
        grad_ub = self.tik.Tensor("float16", self.shape, scope=tbe_platform.scope_ubuf, name="grad_ub")
        output_ub_fp32 = self.tik.Tensor("float32", self.shape, scope=tbe_platform.scope_ubuf, name="output_ub_fp32")
        softmax_ub_fp32 = self.tik.Tensor("float32", self.shape, scope=tbe_platform.scope_ubuf, name="softmax_ub_fp32")
        mask_ub = self.tik.Tensor("uint8", self.shape, scope=tbe_platform.scope_ubuf, name="mask_ub")
        ub_reduceadd = self.tik.Tensor("float32", (self.line_per_iter,), tbe_platform.scope_ubuf, "ub_reduceadd")

        loop_per_core = self.tik.Scalar("int64", name="loop_per_core")
        offset = self.tik.Scalar("int64", name="offset")
        offset_mask = self.tik.Scalar("int64", name="offset_mask")

        return [grad_ub, output_ub_fp32, softmax_ub_fp32, mask_ub, ub_reduceadd, loop_per_core, offset, offset_mask]

    def cal_offset(self, offset, offset_mask, core_index, it):
        """
        Calculate offset of data move for following calculation.
        """
        curr_batch = (core_index * self.iter_per_core + it) // self.iter_per_batch
        curr_channel = (core_index * self.iter_per_core + it) % self.iter_per_batch // self.iter_per_channel
        iter_in_curr_channel = core_index * self.iter_per_core + it - curr_batch * self.iter_per_batch - \
            curr_channel * self.iter_per_channel
        offset_mask.set_as(((curr_batch // self.broad_ratio_n) * self.mask_shape[Cst.DIM_C] +
                           (curr_channel // self.broad_ratio_c)) * self.iter_per_channel * self.element_per_iter +
                           iter_in_curr_channel * self.element_per_iter)
        offset.set_as((curr_batch * self.y_grad_shape[Cst.DIM_C] + curr_channel) * self.iter_per_channel *
                      self.element_per_iter + iter_in_curr_channel * self.element_per_iter)

    # 'pylint: disable=too-many-arguments
    def move_data_in(self, ub_temp_fp16, dst, src, offset, src_dtype):
        """
        Move y from gm tensor to ub tensor.
        """
        move_burst = self.element_per_iter // Cst.NUM_BLOCK_B16
        conv_times = self.element_per_iter // Cst.VEC_MASK_B32
        if src_dtype == 'float16':
            self.tik.data_move(ub_temp_fp16, src[offset], 0, 1, move_burst, 0, 0)
            vconv_ub(self.tik, Cst.VEC_MASK_B32, dst, ub_temp_fp16, conv_times, 8, 4)
        elif src_dtype == 'bfloat16':
            ub_temp_bf16 = ub_temp_fp16.reinterpret_cast_to("bfloat16")
            self.tik.data_move(ub_temp_bf16, src[offset], 0, 1, move_burst, 0, 0)
            vconv_ub(self.tik, Cst.VEC_MASK_B32, dst, ub_temp_bf16, conv_times, 8, 4)
        else:
            self.tik.data_move(dst, src[offset], 0, 1, move_burst * Cst.SCALAR_TWO, 0, 0)

    def calc_product(self, output_ub_fp32, softmax_ub_fp32):
        """
        Calculate y_grad * y.
        """
        self.tik_func_binary("vmul", output_ub_fp32, output_ub_fp32, softmax_ub_fp32,
                             0, 0, 0, self.element_per_iter, Cst.VEC_MASK_B32)

    def move_mask_in(self, mask_ub, offset_mask):
        """
        Move mask from gm tensor to ub tensor.
        """
        if self.fixed_triu_mask:
            move_burst = self.element_per_iter // Cst.NUM_BLOCK_B16
            self.tik.data_move(mask_ub, self.fixed_mask_gm[offset_mask], 0, 1, move_burst, 0, 0)
        else:
            move_burst = self.element_per_iter // Cst.NUM_BLOCK_B8
            self.tik.data_move(mask_ub, self.mask_gm[offset_mask], 0, 1, move_burst, 0, 0)

    def calc_reduce_sum(self, output_ub_fp32, ub_reduceadd):
        """
        Calculate reduce sum of each line.
        """
        cnt, remain = self.cal_level(self.wdim)
        move_time = self.line_per_iter - 1
        move_stride = (self.wdim - Cst.BLOCK) // Cst.NUM_BLOCK_B32
        for i in range(self.line_per_iter):
            offset = self.wdim * i
            self.reduce_add_per_line(output_ub_fp32, offset, cnt, remain)
        self.tik.data_move(output_ub_fp32[Cst.BLOCK], output_ub_fp32[self.wdim],
                           0, move_time, Cst.SCALAR_TWO, move_stride, 0)
        self.tik.vcadd(Cst.BLOCK, ub_reduceadd, output_ub_fp32, self.line_per_iter, 1, 1, 2)

    # 'pylint: disable=no-self-use
    def cal_level(self, length):
        """
        Calculate vmax and vadd times.
        """
        cnt = 0
        dividend = length // Cst.BLOCK
        while dividend % Cst.SCALAR_TWO == 0:
            dividend //= Cst.SCALAR_TWO
            cnt += 1
        return cnt, dividend - 1

    def reduce_add_per_line(self, ub, offset, cnt, remain):
        """
        Do reduce add for each line.
        """
        time = 1
        for _ in range(cnt):
            time = time * Cst.SCALAR_TWO
            self.tik_func_binary("vadd", ub, ub, ub, offset, offset, offset + self.wdim // time,
                                 self.wdim // time, Cst.VEC_MASK_B32)
        if remain > 0:
            for j in range(1, remain + 1):
                self.tik_func_binary("vadd", ub, ub, ub, offset + Cst.BLOCK * (remain - j),
                                     offset + Cst.BLOCK * (remain - j),
                                     offset + Cst.BLOCK * (remain - j + 1), Cst.BLOCK, Cst.BLOCK)

    def calc_softmax_grad(self, output_ub_fp32, ub_reduceadd, softmax_ub_fp32):
        """
        Calculate softmax gradient of each line.
        """
        neg_one = self.tik.Scalar("float32", init_value=-1.0)
        self.tik.vec_muls(self.line_per_iter, ub_reduceadd, ub_reduceadd, neg_one, 1, 8, 8)
        add_dum = self.tik.Scalar("float32", name="add_dum")

        for i in range(self.line_per_iter):
            add_dum.set_as(ub_reduceadd[i])
            offset = self.wdim * i
            self.tik_func_scalar("vadds", output_ub_fp32, output_ub_fp32,
                                 add_dum, offset, offset, self.wdim, Cst.VEC_MASK_B32)
        self.tik_func_binary("vmul", output_ub_fp32, output_ub_fp32, softmax_ub_fp32,
                             0, 0, 0, self.element_per_iter, Cst.VEC_MASK_B32)

    # 'pylint: disable=too-many-arguments
    def scaled_masked_fill(self, grad_ub, mask_ub, output_ub_fp32, softmax_ub_fp32, offset_mask):
        """
        Calculate scaled gradient and masked fill.
        """
        # Masked fill
        if self.y_grad_dtype == "float16":
            conv_times = self.element_per_iter // Cst.VEC_MASK_B32
            vconv_ub(self.tik, Cst.VEC_MASK_B32, grad_ub, output_ub_fp32, conv_times, 4, 8)
            mask_ub_fp16 = softmax_ub_fp32.reinterpret_cast_to("float16")
        else:
            mask_ub_fp16 = grad_ub

        repeat_times = self.element_per_iter // Cst.VEC_MASK_B16
        if self.fixed_triu_mask:
            self.move_mask_in(mask_ub_fp16, offset_mask)
        else:
            vconv_ub(self.tik, Cst.VEC_MASK_B16, mask_ub_fp16, mask_ub, repeat_times, 8, 4)

        self.tik.vmuls(Cst.VEC_MASK_B16, mask_ub_fp16, mask_ub_fp16, self.tik.Scalar(init_value=-1., dtype="float16"),
                       repeat_times, 1, 1, 8, 8)
        self.tik.vadds(Cst.VEC_MASK_B16, mask_ub_fp16, mask_ub_fp16, self.tik.Scalar(init_value=1., dtype="float16"),
                       repeat_times, 1, 1, 8, 8)

        if self.y_grad_dtype == "float16":
            self.tik.vec_mul(Cst.VEC_MASK_B16, grad_ub, grad_ub, mask_ub_fp16, repeat_times, 8, 8, 8)
            self.tik_func_binary("vmul", grad_ub, grad_ub, mask_ub_fp16,
                                 0, 0, 0, self.element_per_iter, Cst.VEC_MASK_B16)
            self.tik_func_scalar("vmuls", grad_ub, grad_ub, self.tik.Scalar(init_value=self.scale, dtype="float16"),
                                 0, 0, self.element_per_iter, Cst.VEC_MASK_B16)
        else:
            conv_times = self.element_per_iter // Cst.VEC_MASK_B32
            vconv_ub(self.tik, Cst.VEC_MASK_B32, softmax_ub_fp32, mask_ub_fp16, conv_times, 8, 4)
            self.tik_func_binary("vmul", output_ub_fp32, output_ub_fp32, softmax_ub_fp32,
                                 0, 0, 0, self.element_per_iter, Cst.VEC_MASK_B32)
            self.tik_func_scalar("vmuls", output_ub_fp32, output_ub_fp32,
                                 self.tik.Scalar(init_value=self.scale, dtype="float32"),
                                 0, 0, self.element_per_iter, Cst.VEC_MASK_B32)

    def move_data_out(self, grad_ub, output_ub_fp32, offset):
        """
        Move result from ub to gm.
        """
        if self.y_grad_dtype == "float16":
            move_burst = self.element_per_iter // Cst.NUM_BLOCK_B16
            self.tik.data_move(self.x_grad_gm[offset], grad_ub, 0, 1, move_burst, 0, 0)
        elif self.y_grad_dtype == "bfloat16":
            conv_times = self.element_per_iter // Cst.VEC_MASK_B32
            move_burst = self.element_per_iter // Cst.NUM_BLOCK_B16
            grad_ub_bf16 = grad_ub.reinterpret_cast_to("bfloat16")
            vconv_ub(self.tik, Cst.VEC_MASK_B32, grad_ub_bf16, output_ub_fp32, conv_times, 4, 8, 'round')
            self.tik.data_move(self.x_grad_gm[offset], grad_ub_bf16, 0, 1, move_burst, 0, 0)
        else:
            move_burst = self.element_per_iter // Cst.NUM_BLOCK_B32
            self.tik.data_move(self.x_grad_gm[offset], output_ub_fp32, 0, 1, move_burst, 0, 0)

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
                          inputs=[self.y_grad_gm, self.y_gm, self.mask_gm],
                          outputs=[self.x_grad_gm])
        return self.tik


# 'pylint: disable=too-many-locals,too-many-statements,too-many-arguments,too-many-branches
def calc_softmax_grad_and_masked_fill(offset, params_list, ub_broadcast, grad_ub, mask_ub, output_ub_fp16,
                                      softmax_ub_fp32, y_gm, ex_params, shape, nd_support_params, grad_dtype):
    """
    Calculate Softmax grad and do scaling and mask.
    """
    tik_instance, [grad_shape, _, _], _, w_dim, line = params_list
    scale, fixed_triu_mask, mask_gm, offset_mask = ex_params
    ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")
    counts = line * Cst.BLOCK * w_dim // Cst.VEC_MASK_B16 // Cst.DIV
    stride = line * Cst.BLOCK // Cst.DIV

    for idx in range(stride * Cst.BLOCK // Cst.VEC_MASK_B16):
        tik_instance.vsub(Cst.VEC_MASK_B16, grad_ub[idx * Cst.VEC_MASK_B16], grad_ub[idx * Cst.VEC_MASK_B16],
                          ub_broadcast_fp16[idx * Cst.VEC_MASK_B16], grad_shape[Cst.DIM_W1], 1, 1, 1, stride, stride, 0)
    tik_instance.vec_mul(Cst.VEC_MASK_B16, grad_ub, grad_ub, output_ub_fp16, counts, 8, 8, 8)
    if fixed_triu_mask:
        if nd_support_params[0]:
            for i in range(grad_shape[Cst.DIM_W1]):
                tik_instance.data_move(output_ub_fp16[i * stride * Cst.BLOCK], mask_gm[offset_mask + i * Cst.BLOCK],
                                       0, stride, 1, w_dim // Cst.BLOCK - 1, 0)
        else:
            tik_instance.data_move(output_ub_fp16, mask_gm[offset_mask],
                                   0, grad_shape[Cst.DIM_W1], stride, grad_shape[Cst.DIM_H1] * Cst.BLOCK - stride, 0)
    else:
        vconv_ub(tik_instance, Cst.VEC_MASK_B16, output_ub_fp16, mask_ub, counts, 8, 4)
    tik_instance.vmuls(Cst.VEC_MASK_B16, output_ub_fp16, output_ub_fp16,
                       tik_instance.Scalar(init_value=-1., dtype="float16"), counts, 1, 1, 8, 8)
    tik_instance.vadds(Cst.VEC_MASK_B16, output_ub_fp16, output_ub_fp16,
                       tik_instance.Scalar(init_value=1., dtype="float16"), counts, 1, 1, 8, 8)
    tik_instance.vec_mul(Cst.VEC_MASK_B16, grad_ub, grad_ub, output_ub_fp16, counts, 8, 8, 8)
    tik_instance.vec_muls(Cst.VEC_MASK_B16, grad_ub, grad_ub,
                          tik_instance.Scalar(init_value=scale, dtype="float16"), counts, 8, 8)

    # Move data out
    if grad_dtype == 'bfloat16':
        counts = line * Cst.BLOCK * w_dim // Cst.VEC_MASK_B32 // Cst.DIV
        grad_ub_bf16 = grad_ub.reinterpret_cast_to('bfloat16')
        if nd_support_params[0]:
            softmax_ub_nd = output_ub_fp16.reshape((2, shape[1], w_dim))
            softmax_ub_fp32_nd = softmax_ub_fp32.reshape((1, shape[1], w_dim))
            grad_ub_nd_bf16 = grad_ub_bf16.reshape((1, shape[1], w_dim))
            for h_index in range(shape[1]):
                tik_instance.data_move(softmax_ub_nd[0, h_index, 0], grad_ub[0, h_index, 0],
                                       0, grad_shape[Cst.DIM_W1], 1, shape[1]-1, 0)
            vconv_ub(tik_instance, Cst.VEC_MASK_B32, softmax_ub_fp32_nd, softmax_ub_nd, counts, 8, 4)
            vconv_ub(tik_instance, Cst.VEC_MASK_B32, grad_ub_nd_bf16, softmax_ub_fp32_nd, counts, 4, 8, 'round')
            tik_instance.data_move(y_gm[offset], grad_ub_nd_bf16, 0, 1, grad_shape[Cst.DIM_W1] * stride, 0, 0)
        else:
            vconv_ub(tik_instance, Cst.VEC_MASK_B32, softmax_ub_fp32, grad_ub, counts, 8, 4)
            vconv_ub(tik_instance, Cst.VEC_MASK_B32, grad_ub_bf16, softmax_ub_fp32, counts, 4, 8, 'round')
            tik_instance.data_move(y_gm[offset], grad_ub, 0, grad_shape[Cst.DIM_W1],
                                   stride, 0, grad_shape[Cst.DIM_H1] * Cst.BLOCK - stride)
    else:
        if nd_support_params[0]:
            softmax_ub_nd = output_ub_fp16.reshape((2, shape[1], w_dim))
            for h_index in range(shape[1]):
                tik_instance.data_move(softmax_ub_nd[0, h_index, 0], grad_ub[0, h_index, 0],
                                       0, grad_shape[Cst.DIM_W1], 1, shape[1]-1, 0)
            tik_instance.data_move(y_gm[offset], softmax_ub_nd, 0, 1, grad_shape[Cst.DIM_W1] * stride, 0, 0)
        else:
            tik_instance.data_move(y_gm[offset], grad_ub, 0, grad_shape[Cst.DIM_W1],
                                   stride, 0, grad_shape[Cst.DIM_H1] * Cst.BLOCK - stride)


def calc_reducesum(params_list, ub_list):
    """
    Calculate reduce sum of each line.
    """
    tik_instance, [grad_shape, _, _], _, w_dim, line = params_list
    output_ub_fp32, softmax_ub_fp32, ub_reduce_add, ub_reduceadd_fp16, ub_broadcast, ub_dup = ub_list
    cnt, remain = cal_level(grad_shape[Cst.DIM_W1])
    time = 1

    # Do reduce_sum
    for _ in range(cnt):
        time = time * Cst.LEN
        counts = w_dim * Cst.BLOCK * line // Cst.DIV // time
        tik_instance.vadd(Cst.VEC_MASK_B32, output_ub_fp32, output_ub_fp32, output_ub_fp32[counts],
                          counts // Cst.VEC_MASK_B32, 1, 1, 1, 8, 8, 8)
    counts = Cst.BLOCK * Cst.BLOCK * line // Cst.DIV
    if remain > 0:
        for j in range(1, remain + 1):
            tik_instance.vadd(Cst.VEC_MASK_B32, output_ub_fp32[counts * (remain - j)],
                              output_ub_fp32[counts * (remain - j)], output_ub_fp32[counts * (remain - j + 1)],
                              counts // Cst.VEC_MASK_B32, 1, 1, 1, 8, 8, 8)
    counts = Cst.BLOCK * line // Cst.DIV
    tik_instance.vcadd(Cst.BLOCK, ub_reduce_add, output_ub_fp32, counts, 1, 1, 2)

    # Do broadcast
    vconv_ub(tik_instance, counts, ub_reduceadd_fp16, ub_reduce_add, 1, 0, 0)
    tik_instance.vector_dup(Cst.VEC_DUMP_SHAPE, ub_dup, tik_instance.Scalar(init_value=0, dtype="int16"), 1, 1, 8)
    output_ub_fp16 = output_ub_fp32.reinterpret_cast_to("float16")
    conv_times = line * Cst.BLOCK * w_dim // Cst.VEC_MASK_B32 // Cst.DIV
    vconv_ub(tik_instance, Cst.VEC_MASK_B32, output_ub_fp16, softmax_ub_fp32, conv_times, 4, 8)
    ub_reduceadd_int16 = ub_reduceadd_fp16.reinterpret_cast_to("int16")
    with tik_instance.for_range(0, line) as j:
        tik_instance.vor(Cst.BLOCK, ub_broadcast[Cst.BLOCK * Cst.BLOCK * j], ub_reduceadd_int16[Cst.BLOCK * j], ub_dup,
                         Cst.BLOCK, 1, 1, 0, 1, 0, 0)
    for j in range(line):
        tik_instance.vtranspose(ub_broadcast[Cst.BLOCK * Cst.BLOCK * j], ub_broadcast[Cst.BLOCK * Cst.BLOCK * j])

    return output_ub_fp16


def calc_product(params_list, output_ub_fp32, softmax_ub_fp32):
    """
    Calculate y_grad * y.
    """
    tik_instance, _, _, w_dim, line = params_list
    counts = line * Cst.BLOCK * w_dim // Cst.DIV
    if counts // Cst.VEC_MASK_B32 > Cst.MAX_REPEAT:
        tik_instance.vec_mul(Cst.VEC_MASK_B32, output_ub_fp32, softmax_ub_fp32, output_ub_fp32,
                             Cst.MAX_REPEAT, 8, 8, 8)
        tik_instance.vec_mul(Cst.VEC_MASK_B32, output_ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_B32],
                             softmax_ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_B32],
                             output_ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_B32], 1, 8, 8, 8)
    else:
        tik_instance.vec_mul(Cst.VEC_MASK_B32, output_ub_fp32, softmax_ub_fp32, output_ub_fp32,
                             counts // Cst.VEC_MASK_B32, 8, 8, 8)


# 'pylint: disable=too-many-locals,too-many-statements,too-many-arguments,too-many-branches
def data_move_in(offset, offset_mask, tensor_dtype, params_list, mov_lis, fixed_triu_mask, nd_support_params):
    """
    Move data from gm tensor to ub tensor.
    """
    tik_instance, [grad_shape, _, _], _, w_dim, line = params_list
    grad_ub, output_ub_fp32, grad_gm, softmax_ub_fp32, y_gm, mask_ub, mask_gm = mov_lis
    times = line * Cst.BLOCK * w_dim // Cst.VEC_MASK_B32 // Cst.DIV
    counts = Cst.BLOCK * line // Cst.DIV
    ub_shape = grad_ub.shape

    # Move mask in
    if not fixed_triu_mask:
        if nd_support_params[0]:
            with tik_instance.new_stmt_scope():
                ub_mask_nd = mask_ub.reshape((counts, w_dim))
                grad_ub_nd = grad_ub.reshape((counts, w_dim))
                tik_instance.data_move(ub_mask_nd, mask_gm[offset_mask], 0, 1,
                                       grad_shape[Cst.DIM_W1] * counts // Cst.LEN, 0, 0)
                vconv_ub(tik_instance, Cst.VEC_MASK_B16, grad_ub_nd, ub_mask_nd,
                         counts * w_dim // Cst.VEC_MASK_B16, 8, 4)
                output_ub_fp16 = output_ub_fp32.reinterpret_cast_to("float16")
                output_ub_fp16 = output_ub_fp16.reshape([ub_shape[0]*2, ub_shape[1], ub_shape[2]])
                for h_index in range(ub_shape[1]):
                    tik_instance.data_move(output_ub_fp16[0, h_index, 0], grad_ub_nd[h_index, 0],
                                           0, grad_shape[Cst.DIM_W1], 1, 0, ub_shape[1] - 1)
                vconv_ub(tik_instance, Cst.VEC_MASK_B16, mask_ub, output_ub_fp16,
                         counts * w_dim // Cst.VEC_MASK_B16, 4, 8)
        else:
            tik_instance.data_move(mask_ub, mask_gm[offset_mask], 0, grad_shape[Cst.DIM_W1], counts // Cst.LEN,
                                   (grad_shape[Cst.DIM_H1] * Cst.BLOCK - counts) // Cst.LEN, 0)

    if tensor_dtype == 'float16' or tensor_dtype == 'bfloat16':
        move_ub = grad_ub.reinterpret_cast_to('bfloat16') if tensor_dtype == 'bfloat16' else grad_ub
        # Move output of Softmax in
        if nd_support_params[0]:
            for h_index in range(ub_shape[1]):
                tik_instance.data_move(move_ub[0, h_index, 0], y_gm[offset + h_index * w_dim],
                                       0, grad_shape[Cst.DIM_W1], 1, 0, ub_shape[1] - 1)
        else:
            tik_instance.data_move(move_ub, y_gm[offset], 0, grad_shape[Cst.DIM_W1], counts,
                                   grad_shape[Cst.DIM_H1] * Cst.BLOCK - counts, 0)
        vconv_ub(tik_instance, Cst.VEC_MASK_B32, softmax_ub_fp32, move_ub, times, 8, 4)

        # Move gradient of backward in
        if nd_support_params[0]:
            for h_index in range(ub_shape[1]):
                tik_instance.data_move(move_ub[0, h_index, 0], grad_gm[offset + h_index * w_dim],
                                       0, grad_shape[Cst.DIM_W1], 1, 0,  ub_shape[1] - 1)
        else:
            tik_instance.data_move(move_ub, grad_gm[offset], 0, grad_shape[Cst.DIM_W1], counts,
                                   grad_shape[Cst.DIM_H1] * Cst.BLOCK - counts, 0)
        vconv_ub(tik_instance, Cst.VEC_MASK_B32, output_ub_fp32, move_ub, times, 8, 4)
        if tensor_dtype == 'bfloat16':
            vconv_ub(tik_instance, Cst.VEC_MASK_B32, grad_ub, output_ub_fp32, times, 4, 8)
    else:
        if nd_support_params[0]:
            for i in range(grad_shape[Cst.DIM_W1]):
                tik_instance.data_move(output_ub_fp32[i * counts * Cst.BLOCK], grad_gm[offset + i * Cst.BLOCK],
                                       0, counts, 2, (w_dim // Cst.BLOCK - 1) * 2, 0)
                tik_instance.data_move(softmax_ub_fp32[i * counts * Cst.BLOCK], y_gm[offset + i * Cst.BLOCK],
                                       0, counts, 2, (w_dim // Cst.BLOCK - 1) * 2, 0)
        else:
            tik_instance.data_move(output_ub_fp32, grad_gm[offset], 0, grad_shape[Cst.DIM_W1],
                                   counts * 2, (grad_shape[Cst.DIM_H1] * Cst.BLOCK - counts) * 2, 0)
            tik_instance.data_move(softmax_ub_fp32, y_gm[offset], 0, grad_shape[Cst.DIM_W1], counts * 2,
                                   (grad_shape[Cst.DIM_H1] * Cst.BLOCK - counts) * 2, 0)
        vconv_ub(tik_instance, Cst.VEC_MASK_B32, grad_ub, output_ub_fp32, times, 4, 8)


def create_ub(tik_instance, shape, line):
    """
    Create ub tensors.
    """
    grad_ub = tik_instance.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="grad_ub")
    softmax_ub_fp32 = tik_instance.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="softmax_ub_fp32")
    mask_ub = tik_instance.Tensor("uint8", shape, scope=tbe_platform.scope_ubuf, name="mask_ub")
    output_ub_fp32 = tik_instance.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="output_ub_fp32")

    counts = line * Cst.BLOCK // Cst.DIV
    ub_reduce_add = tik_instance.Tensor("float32", (counts,), tbe_platform.scope_ubuf, "ub_reduce_add")
    ub_reduceadd_fp16 = tik_instance.Tensor("float16", (counts,), tbe_platform.scope_ubuf, "ub_reduceadd_fp16")
    ub_dup = tik_instance.Tensor("int16", (Cst.VEC_DUMP_SHAPE,), scope=tbe_platform.scope_ubuf, name="ub_dup")
    ub_broadcast = tik_instance.Tensor("int16", (line * Cst.BLOCK, Cst.BLOCK),
                                       scope=tbe_platform.scope_ubuf, name="ub_broadcast")

    loop_per_core = tik_instance.Scalar("int32", name="loop_per_core")
    offset = tik_instance.Scalar("int32", name="offset")
    offset_mask = tik_instance.Scalar("int32", name="offset_mask")

    return [grad_ub, softmax_ub_fp32, mask_ub, output_ub_fp32, ub_reduce_add,
            ub_reduceadd_fp16, ub_dup, ub_broadcast, loop_per_core, offset, offset_mask]


# 'pylint: disable=too-many-arguments
def scaled_masked_softmax_grad_compute(params_list, core_attr_list, gm_tensors,
                                       scale, core_index, grad_dtype, nd_support_params):
    """
    Implementation of fused scaled masked softmax backward algorithm.
    """
    tik_instance, [input_shape, _, _], _, _, line = params_list
    used_core_num, _, _, iter_per_batch, iter_per_channel, iter_per_core, \
        iter_last_core, element_per_iter, _, shape, counts, broad_ratio = core_attr_list
    y_grad_gm, y_gm, mask_gm, x_grad_gm, fixed_triu_mask = gm_tensors
    grad_ub, softmax_ub_fp32, mask_ub, output_ub_fp32, ub_reduce_add, ub_reduceadd_fp16, \
        ub_dup, ub_broadcast, loop_per_core, offset, offset_mask = create_ub(tik_instance, shape, line)
    move_list = [grad_ub, output_ub_fp32, y_grad_gm, softmax_ub_fp32, y_gm, mask_ub, mask_gm]

    with tik_instance.if_scope(core_index == used_core_num - 1):
        loop_per_core.set_as(iter_last_core)
    with tik_instance.else_scope():
        loop_per_core.set_as(iter_per_core)

    with tik_instance.for_range(0, loop_per_core) as it:
        current_batch = (core_index * iter_per_core + it) // iter_per_batch
        current_channel = (core_index * iter_per_core + it) % iter_per_batch // iter_per_channel
        iter_in_curr_channel = core_index * iter_per_core + it - current_batch * iter_per_batch - \
            current_channel * iter_per_channel
        offset_mask.set_as(((current_batch * input_shape[Cst.DIM_C] + current_channel) // broad_ratio) *
                           iter_per_channel * element_per_iter + iter_in_curr_channel * counts)
        offset.set_as((current_batch * input_shape[Cst.DIM_C] + current_channel) *
                      iter_per_channel * element_per_iter + iter_in_curr_channel * counts)

        data_move_in(offset, offset_mask, grad_dtype, params_list, move_list, fixed_triu_mask, nd_support_params)
        calc_product(params_list, output_ub_fp32, softmax_ub_fp32)
        output_ub_fp16 = calc_reducesum(params_list, [output_ub_fp32, softmax_ub_fp32,
                                                      ub_reduce_add, ub_reduceadd_fp16, ub_broadcast, ub_dup])
        calc_softmax_grad_and_masked_fill(offset, params_list, ub_broadcast, grad_ub, mask_ub, output_ub_fp16,
                                          softmax_ub_fp32, x_grad_gm, [scale, fixed_triu_mask, mask_gm, offset_mask],
                                          shape, nd_support_params, grad_dtype)


# 'pylint: disable=too-many-arguments
def create_gm_tensor(tik_instance, grad_shape,  softmax_output_shape, mask_shape,
                     grad_dtype, softmax_output_dtype, output_dtype):
    """
    Create input and output gm tensors.
    """
    y_grad_gm = tik_instance.Tensor(grad_dtype, grad_shape, name="y_grad_gm", scope=tbe_platform.scope_gm)
    y_gm = tik_instance.Tensor(softmax_output_dtype, softmax_output_shape, name="y_gm", scope=tbe_platform.scope_gm)
    mask_gm = tik_instance.Tensor('bool', mask_shape, name="mask_gm", scope=tbe_platform.scope_gm)
    x_grad_gm = tik_instance.Tensor(output_dtype, grad_shape, name="x_grad_gm", scope=tbe_platform.scope_gm)
    return y_grad_gm, y_gm, mask_gm, x_grad_gm


# 'pylint: disable=unused-argument,too-many-arguments,
# 'pylint: disable=disable=too-many-locals,too-many-statements
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def scaled_masked_softmax_grad(y_grad, y, mask, x_grad, scale=1.0,
                               fixed_triu_mask=False, kernel_name="scaled_masked_softmax_grad"):
    """
    Algorithm:
        mask = torch.triu(mask.shape, diagonal=1) if fixed_triu_mask else mask
        x_grad = (y_grad - (y_grad * y).sum(-1).unsqueeze(-1)) * y
        x_grad = (x_grad * scale).masked_fill(mask, 0)

    Parameters
    ----------
    y_grad : dict
        shape and dtype of input grad tensor.
    y : dict
        shape and dtype of forward output tensor, the shape must be same as y_grad.
    mask : dict
        shape and dtype of mask, the shape must be broadcastble with y_grad.
    x_grad : dict
        shape and dtype of output grad tensor, the shape must be same as y_grad.
    scale : float
        a float scalar scaling the input_grad. 
    fixed_triu_mask : bool
        if true: the mask is a fixed upper triangle mask
        if false: the mask is input mask
    kernel_name : str
        kernel name, default value is "scaled_masked_softmax_grad"

    Returns
    -------
    None
    """
    nd_support_params = (y_grad.get("format") != "FRACTAL_NZ", y_grad.get("shape"))
    if nd_support_params[0]:
        return ScaledMaskedSoftmaxGradNd(y_grad, y, mask, x_grad, scale, fixed_triu_mask, kernel_name).compute()

    grad_dtype = y_grad.get("dtype").lower()
    softmax_output_dtype = y.get("dtype").lower()
    output_dtype = x_grad.get("dtype").lower()
    grad_shape, softmax_output_shape, mask_shape = get_nz_shape(y_grad), get_nz_shape(y), get_nz_shape(mask)
    params_list, core_attr_list = cal_params_list([grad_shape, mask_shape, softmax_output_shape],
                                                  fixed_triu_mask, nd_support_params)
    tik_instance, [grad_shape, softmax_output_shape, mask_shape], _, w_dim, _ = params_list

    if w_dim < 32 or w_dim > 2048 or w_dim % 32 != 0:
        raise RuntimeError('Error! The last dimension of the input tensor should be within the range of [32, 2048] ' +
                           f'and be divisible by 32. Currently, it is {w_dim}.')

    used_core_num, _, _, _, _, _, _, _, _, _, _, _ = core_attr_list
    Cst.DIV = 1 if w_dim <= 1024 else 2

    y_grad_gm, y_gm, mask_gm, x_grad_gm = create_gm_tensor(tik_instance, grad_shape, softmax_output_shape,
                                                           mask_shape, grad_dtype, softmax_output_dtype, output_dtype)

    if fixed_triu_mask:
        fixed_mask_gm = tik_instance.Tensor('float16', mask_shape, name="fixed_mask_gm",
                                            scope=tbe_platform.scope_gm, is_workspace=True)
        gm_tensors = [y_grad_gm, y_gm, fixed_mask_gm, x_grad_gm, fixed_triu_mask]
        gen_triu_mask(params_list, fixed_mask_gm)
    else:
        mask_tensor_gm = mask_gm.reinterpret_cast_to('uint8')
        gm_tensors = [y_grad_gm, y_gm, mask_tensor_gm, x_grad_gm, fixed_triu_mask]

    with tik_instance.for_range(0, used_core_num, block_num=used_core_num) as core_index:
        scaled_masked_softmax_grad_compute(params_list, core_attr_list, gm_tensors,
                                           scale, core_index, grad_dtype, nd_support_params)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[y_grad_gm, y_gm, mask_gm],
                          outputs=[x_grad_gm, ])
    return tik_instance
