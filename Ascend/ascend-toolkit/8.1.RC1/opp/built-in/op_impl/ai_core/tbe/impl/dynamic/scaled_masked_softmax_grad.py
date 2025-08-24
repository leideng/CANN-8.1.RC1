#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) Huawei Technologies Co., Ltd 2024-2024. All rights reserved
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
Dynamic ScaledMaskedSoftmaxGrad.
"""
from abc import ABCMeta
from impl import constant_util as constant
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator


class Constant:
    TILING_ARG_NUM = 24
    MAX_B16_UB_NUM = 6
    MAX_REPEAT = 255
    NUM_BLOCK_B8 = 32
    NUM_BLOCK_B16 = 16
    NUM_BLOCK_B32 = 8
    VEC_MASK_B16 = 128
    VEC_MASK_B32 = 64
    BYTE_B8 = 1
    BYTE_B16 = 2
    BYTE_B32 = 4
    MAX_LOOP = 1000
    PADDING_VALUE_MASK = 1
    SCALAR_TWO = 2
    SCALAR_EIGHT = 8
    SHAPE_2048 = 2048
    SEVEN_LINE_REPEAT = 224
    SHAPE_SIZE_LIMIT = 2 ** 31 + 1


class ScaledMaskedSoftmaxGrad():
    """Class for ScaledMaskedSoftmaxGrad."""

    # 'pylint: disable=too-many-arguments,unused-argument,huawei-too-many-arguments
    def __init__(self, y_grad, y, mask, x_grad, scale, fixed_triu_mask, kernel_name):
        """Init class."""
        self.tik = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.fixed_triu_mask = fixed_triu_mask
        self.kernel_name = kernel_name
        self.dtype = y_grad.get("dtype")
        self.is_nz = y_grad.get("format") == "FRACTAL_NZ"
        self.init_scalar()
        self.init_gm()
        self.get_tiling_data()

    def init_scalar(self):
        """Init scalars."""
        self.tiling_core_num = self.tik.Scalar("int32")
        self.tiling_mode = self.tik.Scalar("int32")
        self.y_grad_n = self.tik.Scalar("int32")
        self.y_grad_c = self.tik.Scalar("int32")
        self.y_grad_h1 = self.tik.Scalar("int32")
        self.y_grad_w1 = self.tik.Scalar("int32")
        self.y_grad_h0 = self.tik.Scalar("int32")
        self.y_grad_w0 = self.tik.Scalar("int32")
        self.y_n = self.tik.Scalar("int32")
        self.y_c = self.tik.Scalar("int32")
        self.y_h1 = self.tik.Scalar("int32")
        self.y_w1 = self.tik.Scalar("int32")
        self.y_h0 = self.tik.Scalar("int32")
        self.y_w0 = self.tik.Scalar("int32")
        self.mask_n = self.tik.Scalar("int32")
        self.mask_c = self.tik.Scalar("int32")
        self.mask_h1 = self.tik.Scalar("int32")
        self.mask_w1 = self.tik.Scalar("int32")
        self.mask_h0 = self.tik.Scalar("int32")
        self.mask_w0 = self.tik.Scalar("int32")
        self.scale = self.tik.Scalar("float32")

    def init_gm(self):
        """Init gm space."""
        self.y_grad_gm = self.tik.Tensor(self.dtype, [Constant.SHAPE_SIZE_LIMIT], name="x_gm", scope=tik.scope_gm)
        self.y_gm = self.tik.Tensor(self.dtype, [Constant.SHAPE_SIZE_LIMIT], name="y_gm", scope=tik.scope_gm)
        self.mask_gm = self.tik.Tensor("uint8", [Constant.SHAPE_SIZE_LIMIT], name="mask_gm", scope=tik.scope_gm)
        self.x_grad_gm = self.tik.Tensor(self.dtype, [Constant.SHAPE_SIZE_LIMIT], name="x_grad_gm", scope=tik.scope_gm)
        if self.fixed_triu_mask:
            self.fixed_mask_gm = self.tik.Tensor('float16', [Constant.SHAPE_SIZE_LIMIT],
                                                 tik.scope_gm, "fixed_mask_gm", is_workspace=True)

    def get_tiling_data(self):
        """Get tiling data."""
        core_num = tik.Dprofile().get_aicore_num()
        tbe_context.get_context().add_compile_info("vars", {"core_num": core_num})
        self.tiling_gm = self.tik.Tensor("int32", [Constant.TILING_ARG_NUM], tik.scope_gm, "tiling_gm")
        tiling_ub = self.tik.Tensor("int32", [Constant.TILING_ARG_NUM], tik.scope_ubuf, "tiling_ub")
        self.tik.data_move(tiling_ub, self.tiling_gm, 0, 1, Constant.TILING_ARG_NUM // Constant.NUM_BLOCK_B32, 0, 0)
        self.tiling_core_num.set_as(tiling_ub[0])
        self.tiling_mode.set_as(tiling_ub[1])
        self.y_grad_n.set_as(tiling_ub[2])
        self.y_grad_c.set_as(tiling_ub[3])
        self.y_grad_h1.set_as(tiling_ub[4])
        self.y_grad_w1.set_as(tiling_ub[5])
        self.y_grad_h0.set_as(tiling_ub[6])
        self.y_grad_w0.set_as(tiling_ub[7])
        self.y_n.set_as(tiling_ub[8])
        self.y_c.set_as(tiling_ub[9])
        self.y_h1.set_as(tiling_ub[10])
        self.y_w1.set_as(tiling_ub[11])
        self.y_h0.set_as(tiling_ub[12])
        self.y_w0.set_as(tiling_ub[13])
        self.mask_n.set_as(tiling_ub[14])
        self.mask_c.set_as(tiling_ub[15])
        self.mask_h1.set_as(tiling_ub[16])
        self.mask_w1.set_as(tiling_ub[17])
        self.mask_h0.set_as(tiling_ub[18])
        self.mask_w0.set_as(tiling_ub[19])
        tiling_ub_fp32 = tiling_ub.reinterpret_cast_to("float32")
        self.scale.set_as(tiling_ub_fp32[20])

    def compute(self):
        """Enter of the way to different subclass."""
        with self.tik.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as core_idx:
            with self.tik.if_scope(self.tiling_mode == 1):
                ScaledMaskedSoftmaxGradUnalignedND(self).compute(core_idx)
            with self.tik.elif_scope(self.tiling_mode == 2):
                ScaledMaskedSoftmaxGradAlignedND(self).compute(core_idx)
            with self.tik.elif_scope(self.tiling_mode == 3):
                ScaledMaskedSoftmaxGradAlignedNZ(self).compute(core_idx)
        return self.build_cce()

    def build_cce(self):
        """Build cce."""
        self.tik.BuildCCE(kernel_name=self.kernel_name,
                          inputs=[self.y_grad_gm, self.y_gm, self.mask_gm],
                          outputs=[self.x_grad_gm],
                          flowtable=[self.tiling_gm],
                          config={"enable_const_fold": True})
        return self.tik


class BaseClass(metaclass=ABCMeta):
    """Baseclass for different ScaledMaskedSoftmaxGrad implementation."""

    def __init__(self, op: "ScaledMaskedSoftmaxGrad"):
        """Init."""
        self.tik = op.tik
        self.is_nz = op.is_nz
        self.dtype = op.dtype
        self.y_grad_n = op.y_grad_n
        self.y_grad_c = op.y_grad_c
        self.y_grad_h1 = op.y_grad_h1
        self.y_grad_w1 = op.y_grad_w1
        self.y_grad_h0 = op.y_grad_h0
        self.y_grad_w0 = op.y_grad_w0
        self.y_n = op.y_n
        self.y_c = op.y_c
        self.y_h1 = op.y_h1
        self.y_w1 = op.y_w1
        self.y_h0 = op.y_h0
        self.y_w0 = op.y_w0
        self.mask_n = op.mask_n
        self.mask_c = op.mask_c
        self.mask_h1 = op.mask_h1
        self.mask_w1 = op.mask_w1
        self.mask_h0 = op.mask_h0
        self.mask_w0 = op.mask_w0
        self.scale = op.scale
        self.fixed_triu_mask = op.fixed_triu_mask
        self.tiling_core_num = op.tiling_core_num
        self.available_ub_space = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 33 * 1024

        self.y_grad_gm = op.y_grad_gm
        self.y_gm = op.y_gm
        self.mask_gm = op.mask_gm
        self.x_grad_gm = op.x_grad_gm
        if self.fixed_triu_mask:
            self.fixed_mask_gm = op.fixed_mask_gm

        self.init_scalar()

    @staticmethod
    def ceil_div(dividend, divisor):
        """
        Calculate the minimum value that is divisible by the dividend.
        """
        return (dividend + divisor - 1) // divisor

    def init_scalar(self):
        """Init scalar."""
        self.y_grad_h = self.tik.Scalar("int32")
        self.y_grad_w = self.tik.Scalar("int32")
        self.y_h = self.tik.Scalar("int32")
        self.y_w = self.tik.Scalar("int32")
        self.mask_h = self.tik.Scalar("int32")
        self.mask_w = self.tik.Scalar("int32")
        self.ele_per_line = self.tik.Scalar("int32")
        self.ele_per_iter = self.tik.Scalar("int32")
        self.ele_per_core = self.tik.Scalar("int32")
        self.total_line = self.tik.Scalar("int32")
        self.total_iter = self.tik.Scalar("int32")
        self.line_per_core = self.tik.Scalar("int32")
        self.line_per_iter = self.tik.Scalar("int32")
        self.iter_per_core = self.tik.Scalar("int32")
        self.iter_per_batch = self.tik.Scalar("int32")
        self.iter_per_channel = self.tik.Scalar("int32")
        self.used_core_num = self.tik.Scalar("int32")
        self.iter_last_core = self.tik.Scalar("int32")
        self.broad_ratio_n = self.tik.Scalar("int32")
        self.broad_ratio_c = self.tik.Scalar("int32")

    def compute(self, core_idx):
        """Main function for calculation."""
        self.get_parallel_params()
        self.cal_broadcast_params()
        if self.fixed_triu_mask:
            self.gen_triu_mask()
        with self.tik.if_scope(core_idx < self.used_core_num):
            with self.tik.if_scope(core_idx < self.used_core_num - 1):
                self.do_compute(core_idx, self.iter_per_core)
            with self.tik.else_scope():
                self.do_compute(core_idx, self.iter_last_core)

    def get_parallel_params(self):
        """Calculate parallel parameters."""

        with self.tik.if_scope(self.y_grad_w <= 512):
            self.line_per_iter.set_as(32)
        with self.tik.elif_scope(self.y_grad_w <= 1024):
            self.line_per_iter.set_as(16)
        with self.tik.elif_scope(self.y_grad_w <= 2048):
            self.line_per_iter.set_as(8)
        with self.tik.elif_scope(self.y_grad_w <= 4096):
            self.line_per_iter.set_as(4)
        with self.tik.else_scope():
            self.line_per_iter.set_as(2)
        self.total_iter.set_as(self.y_grad_n * self.y_grad_c * self.y_grad_h // self.line_per_iter)
        self.iter_per_batch.set_as(self.y_grad_c * self.y_grad_h // self.line_per_iter)
        self.iter_per_channel.set_as(self.y_grad_h // self.line_per_iter)
        self.iter_per_core.set_as(self.ceil_div(self.total_iter, self.tiling_core_num))
        self.used_core_num.set_as(self.ceil_div(self.total_iter, self.iter_per_core))
        self.iter_last_core.set_as(self.total_iter - (self.used_core_num - 1) * self.iter_per_core)
        self.ele_per_line.set_as(self.y_grad_w)
        self.ele_per_iter.set_as(self.y_grad_w * self.line_per_iter)
        self.ele_per_core.set_as(self.ele_per_iter * self.iter_per_core)

    def cal_broadcast_params(self):
        """Calculate parameters of broadcast for mask."""
        if self.fixed_triu_mask:
            self.broad_ratio_n.set_as(self.y_grad_n)
            self.broad_ratio_c.set_as(self.y_grad_c)
        else:
            self.broad_ratio_n.set_as(self.y_grad_n // self.mask_n)
            self.broad_ratio_c.set_as(self.y_grad_c // self.mask_c)

    def calc_product(self, ub_y_grad_fp32, ub_y_fp32, ele_num):
        """Calculate the result of y_gard * y."""
        self._vec_func("vmul", ub_y_grad_fp32, ub_y_grad_fp32, ub_y_fp32, 0, 0, 0, ele_num, Constant.VEC_MASK_B32)

    def cal_offset(self, core_idx, it, offset, offset_mask):
        """
        Calculate offset of data move for following calculation.
        """
        curr_batch = (core_idx * self.iter_per_core + it) // self.iter_per_batch
        curr_channel = (core_idx * self.iter_per_core + it) % self.iter_per_batch // self.iter_per_channel
        iter_in_curr_channel = core_idx * self.iter_per_core + it - curr_batch *\
            self.iter_per_batch - curr_channel * self.iter_per_channel
        offset_mask.set_as(((curr_batch // self.broad_ratio_n) * self.mask_c +
                           (curr_channel // self.broad_ratio_c)) * self.iter_per_channel * self.ele_per_iter +
                           iter_in_curr_channel * self.counts)
        offset.set_as((curr_batch * self.y_grad_c + curr_channel) * self.iter_per_channel *
                      self.ele_per_iter + iter_in_curr_channel * self.counts)

    def calc_reduce_sum(self, dst, src, line):
        """Calculate sum of each line."""
        block = Constant.VEC_MASK_B32
        src_rep_stride = self.tik.Scalar("int32", init_value=self.ele_per_line // Constant.NUM_BLOCK_B32)
        with self.tik.if_scope(self.y_grad_w > block):
            cnt, dividend, remain = self.cal_level(block)
            length = self.y_grad_w - remain
            offset = self.tik.Scalar("int32", init_value=0)
            with self.tik.for_range(0, line) as i:
                offset.set_as(self.ele_per_line * i)
                time = self.tik.Scalar("int32", init_value=Constant.SCALAR_TWO)
                with self.tik.for_range(0, cnt) as _:
                    self._vec_func("vadd", src, src, src, offset,
                                   offset + length // time, offset,
                                   length // time, block)
                    time.set_as(time * Constant.SCALAR_TWO)
                with self.tik.for_range(1, dividend) as i:
                    self._vec_func("vadd", src, src, src, offset + block * (dividend - 1 - i),
                                   offset + block * (dividend - i),
                                   offset + block * (dividend - 1 - i),
                                   block, block)
                with self.tik.if_scope(remain > 1):
                    self._vec_func("vadd", src, src, src, offset,
                                   offset + self.y_grad_w - remain, offset, remain, block)
            self._tik_vcadd(dst, src, 0, 0, line, 1, 1, src_rep_stride, block)
        with self.tik.else_scope():
            self._tik_vcadd(dst, src, 0, 0, line, 1, 1, src_rep_stride, self.y_grad_w)

    def cal_level(self, block):
        """
        Calculate vadd times.
        """
        cnt = self.tik.Scalar("int32", init_value=0)
        remain = self.tik.Scalar("int32", init_value=self.y_grad_w % block)
        dividend = self.tik.Scalar("int32", init_value=(self.y_grad_w - remain) // block)
        with self.tik.for_range(0, Constant.MAX_LOOP) as i:
            with self.tik.if_scope(dividend % Constant.SCALAR_TWO == 0):
                dividend.set_as(dividend // Constant.SCALAR_TWO)
                cnt.set_as(cnt + 1)
            with self.tik.else_scope():
                self.tik.tik_break()
        return cnt, dividend, remain

    def calc_softmax_grad(self, ub_y_grad_fp32, ub_y_fp32, ub_reduceadd, line, ele_num):
        """Calculate softmax gradient."""
        self._vec_scalar_func("vmuls", ub_reduceadd, ub_reduceadd, -1.0, 0, 0, line, Constant.VEC_MASK_B32)
        add_sum = self.tik.Scalar("float32", name="add_sum")
        with self.tik.for_range(0, line) as i:
            add_sum.set_as(ub_reduceadd[i])
            self._vec_scalar_func("vadds", ub_y_grad_fp32, ub_y_grad_fp32, add_sum, self.ele_per_line * i,
                                  self.ele_per_line * i, self.ele_per_line, Constant.VEC_MASK_B32)
        self._vec_func("vmul", ub_y_grad_fp32, ub_y_grad_fp32, ub_y_fp32, 0, 0, 0, ele_num, Constant.VEC_MASK_B32)

    def scale_grad(self, ub_y_grad_fp32, ele_num):
        """Scale the gradient."""
        self._vec_scalar_func("vmuls", ub_y_grad_fp32, ub_y_grad_fp32, self.scale, 0, 0, ele_num, Constant.VEC_MASK_B32)

    def masked_fill(self, ub_list, offset_mask):
        """Do masked_fill."""
        ub_fp16, mask_ub, ub_y_grad_fp32, ub_y_fp32 = ub_list
        if self.dtype == "float16":
            self._vconv(ub_fp16, ub_y_grad_fp32, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32, 4, 8)
            mask_ub_fp16 = ub_y_fp32.reinterpret_cast_to("float16")
        else:
            mask_ub_fp16 = ub_fp16
        if self.fixed_triu_mask:
            self.move_mask_in(mask_ub_fp16, offset_mask)
        else:
            self._vconv(mask_ub_fp16, mask_ub, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B16, 8, 4)
        self._vec_scalar_func("vmuls", mask_ub_fp16, mask_ub_fp16, -1.0, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B16)
        self._vec_scalar_func("vadds", mask_ub_fp16, mask_ub_fp16, 1, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B16)
        if self.dtype == "float16":
            self._vec_func("vmul", ub_fp16, ub_fp16, mask_ub_fp16, 0, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B16)
        else:
            self._vconv(ub_y_fp32, mask_ub_fp16, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32, 8, 4)
            self._vec_func("vmul", ub_y_grad_fp32, ub_y_grad_fp32, ub_y_fp32,
                           0, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _vec_func(self, func_str, dst, src0, src1, dst_offset, src0_offset, src1_offset, ele_num, mask):
        """
        Api for vec and scalar calculation in TIK function.
        """
        if func_str == "vadd":
            func = self.tik.vadd
        elif func_str == "vmul":
            func = self.tik.vmul
        elif func_str == "vmax":
            func = self.tik.vmax
        elif func_str == "vdiv":
            func = self.tik.vdiv
        else:
            raise RuntimeError(f"[func_str] should be in ['vadd', 'vmul', 'vdiv'], but now is [{func_str}]")
        repeat_times = ele_num // mask
        repeat = repeat_times // Constant.MAX_REPEAT
        repeat_remain = repeat_times % Constant.MAX_REPEAT
        remain = ele_num % mask
        dst_offset = self.tik.Scalar("int32", init_value=dst_offset)
        src0_offset = self.tik.Scalar("int32", init_value=src0_offset)
        src1_offset = self.tik.Scalar("int32", init_value=src1_offset)
        with self.tik.if_scope(repeat_times > 0):
            with self.tik.if_scope(repeat > 0):
                with self.tik.for_range(0, repeat) as _:
                    func(mask, dst[dst_offset], src0[src0_offset], src1[src1_offset],
                         Constant.MAX_REPEAT, 1, 1, 1, 8, 8, 8)
                    dst_offset.set_as(dst_offset + Constant.MAX_REPEAT * mask)
                    src0_offset.set_as(src0_offset + Constant.MAX_REPEAT * mask)
                    src1_offset.set_as(src1_offset + Constant.MAX_REPEAT * mask)
            with self.tik.if_scope(repeat_remain > 0):
                func(mask, dst[dst_offset], src0[src0_offset], src1[src1_offset], repeat_remain, 1, 1, 1, 8, 8, 8)
                dst_offset.set_as(dst_offset + repeat_remain * mask)
                src0_offset.set_as(src0_offset + repeat_remain * mask)
                src1_offset.set_as(src1_offset + repeat_remain * mask)
        with self.tik.if_scope(remain > 0):
            func(remain, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1, 1, 1, 1, 0, 0, 0)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _vec_scalar_func(self, func_str, dst, src, scalar, dst_offset, src_offset, ele_num, mask):
        """
        Api for vec and scalar calculation in TIK function.
        """
        if func_str == "vadds":
            func = self.tik.vadds
        elif func_str == "vmuls":
            func = self.tik.vmuls
        else:
            raise RuntimeError(f"[func_str] should be in ['vadds', 'vmuls'], but now is [{func_str}]")
        repeat_times = ele_num // mask
        repeat = repeat_times // Constant.MAX_REPEAT
        repeat_remain = repeat_times % Constant.MAX_REPEAT
        remain = ele_num % mask
        dst_offset = self.tik.Scalar("int32", init_value=dst_offset)
        src_offset = self.tik.Scalar("int32", init_value=src_offset)
        with self.tik.if_scope(repeat_times > 0):
            with self.tik.if_scope(repeat > 0):
                with self.tik.for_range(0, repeat) as _:
                    func(mask, dst[dst_offset], src[src_offset], scalar, Constant.MAX_REPEAT, 1, 1, 8, 8)
                    dst_offset.set_as(dst_offset + Constant.MAX_REPEAT * mask)
                    src_offset.set_as(src_offset + Constant.MAX_REPEAT * mask)
            with self.tik.if_scope(repeat_remain > 0):
                func(mask, dst[dst_offset], src[src_offset], scalar, repeat_remain, 1, 1, 8, 8)
                dst_offset.set_as(dst_offset + repeat_remain * mask)
                src_offset.set_as(src_offset + repeat_remain * mask)
        with self.tik.if_scope(remain > 0):
            func(remain, dst[dst_offset], src[src_offset], scalar, 1, 1, 1, 0, 0)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _vconv(self, dst, src, offset_dst, offset_src, ele_num, mask, dst_stride, src_stride, round_mode=''):
        """Api for TIK function vconv."""
        repeat_times = ele_num // mask
        remain_num = ele_num % mask
        dst_offset = self.tik.Scalar("int32", init_value=offset_dst)
        src_offset = self.tik.Scalar("int32", init_value=offset_src)
        with self.tik.if_scope(repeat_times > 0):
            max_repeat_times = repeat_times // Constant.MAX_REPEAT
            max_remain_times = repeat_times % Constant.MAX_REPEAT
            with self.tik.for_range(0, max_repeat_times) as _:
                self.tik.vconv(mask, round_mode, dst[dst_offset], src[src_offset],
                               Constant.MAX_REPEAT, 1, 1, dst_stride, src_stride)
                dst_offset.set_as(dst_offset + mask * Constant.MAX_REPEAT)
                src_offset.set_as(src_offset + mask * Constant.MAX_REPEAT)
            with self.tik.if_scope(max_remain_times > 0):
                self.tik.vconv(mask, round_mode, dst[dst_offset], src[src_offset],
                               max_remain_times, 1, 1, dst_stride, src_stride)
                dst_offset.set_as(dst_offset + mask * max_remain_times)
                src_offset.set_as(src_offset + mask * max_remain_times)
        with self.tik.if_scope(remain_num > 0):
            self.tik.vconv(remain_num, round_mode, dst[dst_offset], src[src_offset],
                           1, 1, 1, 0, 0)

    def _vec_dup(self, dst, value, offset, ele_num, mask):
        """Api for TIK function vec_dup."""
        repeat_times = ele_num // mask
        repeat = repeat_times // Constant.MAX_REPEAT
        repeat_remain = repeat_times % Constant.MAX_REPEAT
        remain = ele_num % mask
        temp_offset = self.tik.Scalar("int32", init_value=offset)
        with self.tik.if_scope(repeat_times > 0):
            with self.tik.if_scope(repeat > 0):
                with self.tik.for_range(0, repeat) as _:
                    self.tik.vec_dup(mask, dst[temp_offset], value, Constant.MAX_REPEAT, 8)
                    temp_offset.set_as(temp_offset + Constant.MAX_REPEAT * mask)
            with self.tik.if_scope(repeat_remain > 0):
                self.tik.vec_dup(mask, dst[temp_offset], value, repeat_remain, 8)
                temp_offset.set_as(temp_offset + repeat_remain * mask)
        with self.tik.if_scope(remain > 0):
            self.tik.vec_dup(remain, dst[temp_offset], value, 1, 0)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _tik_vcadd(self, dst, src, dst_offset, src_offset, repeat_times,
                   dst_rep_stride, src_blk_stride, src_rep_stride, mask):
        """
        Api for TIK function vcadd.
        """
        repeat = repeat_times // Constant.MAX_REPEAT
        repeat_remain = repeat_times % Constant.MAX_REPEAT
        dst_offset = self.tik.Scalar("int32", init_value=dst_offset)
        src_offset = self.tik.Scalar("int32", init_value=src_offset)
        with self.tik.if_scope(repeat > 0):
            with self.tik.for_range(0, repeat) as _:
                self.tik.vcadd(mask, dst[dst_offset], src[src_offset], Constant.MAX_REPEAT,
                               dst_rep_stride, src_blk_stride, src_rep_stride)
                dst_offset.set_as(dst_offset + Constant.MAX_REPEAT)
                src_offset.set_as(src_offset + Constant.MAX_REPEAT * self.ele_per_line)
        with self.tik.if_scope(repeat_remain > 0):
            self.tik.vcadd(mask, dst[dst_offset], src[src_offset], repeat_remain,
                           dst_rep_stride, src_blk_stride, src_rep_stride)


class ScaledMaskedSoftmaxGradUnalignedND(BaseClass):
    """Implementation of unaligned ScaledMaskedSoftmaxGrad in ND format."""

    def __init__(self, op: "ScaledMaskedSoftmaxGrad"):
        """Init."""
        super().__init__(op)
        self.y_grad_h.set_as(self.y_grad_h1)
        self.y_grad_w.set_as(self.y_grad_w1)
        self.y_h.set_as(self.y_h1)
        self.y_w.set_as(self.y_w1)
        self.mask_h.set_as(self.mask_h1)
        self.mask_w.set_as(self.mask_w1)
        self.ele_per_line_mask = self.tik.Scalar("int32")
        self.ele_per_channel = self.tik.Scalar("int32")
        self.padding_b8 = self.tik.Scalar("int32")
        self.move_brust_b8 = self.tik.Scalar("int32")
        self.move_brust_b16 = self.tik.Scalar("int32")
        self.move_brust_b32 = self.tik.Scalar("int32")
        self.move_ub_gap = self.tik.Scalar("int32")
        self.available_line_per_iter = self.tik.Scalar("int32")
        self.line_per_core = self.tik.Scalar("int32")
        self.line_last_iter = self.tik.Scalar("int32")
        self.max_byte_per_line = self.tik.Scalar("int32")
        self.iter_last_core = self.tik.Scalar("int32")
        self.line_last_core = self.tik.Scalar("int32")
        self.line_last_core_per_iter = self.tik.Scalar("int32")
        self.line_last_core_last_iter = self.tik.Scalar("int32")
        self.broad_mode = self.tik.Scalar("int32")
        self.line_per_batch = self.tik.Scalar("int32")

    def gen_triu_mask(self):
        """Generate triu mask tensor."""
        move_burst = self.ele_per_line // Constant.NUM_BLOCK_B16
        dup_len = self.tik.Scalar("int32", init_value=1)
        offset = self.tik.Scalar("int32", init_value=0)
        triu_mask_ub = self.tik.Tensor('float16', (self.ele_per_line,), tik.scope_ubuf, "triu_mask_ub")
        self._vec_dup(triu_mask_ub, 1.0, 0, self.ele_per_line, Constant.VEC_MASK_B16)
        with self.tik.for_range(0, self.mask_h) as i:
            with self.tik.if_scope(dup_len <= self.mask_w):
                self._vec_dup(triu_mask_ub, 0.0, 0, dup_len, Constant.VEC_MASK_B16)
                dup_len.set_as(dup_len + 1)
            self.tik.data_move(self.fixed_mask_gm[offset], triu_mask_ub, 0, 1, move_burst, 0, 0)
            offset.set_as(offset + self.ele_per_line)

    def compute(self, core_idx):
        """Main function for calculation."""
        self.get_parallel_params()
        self.cal_broadcast_params()
        if self.fixed_triu_mask:
            self.gen_triu_mask()
        with self.tik.if_scope(core_idx < self.used_core_num):
            with self.tik.if_scope(core_idx < self.used_core_num - 1):
                self.do_compute(core_idx, self.iter_per_core, self.line_per_iter, self.line_last_iter)
            with self.tik.else_scope():
                self.do_compute(core_idx, self.iter_last_core,
                                self.line_last_core_per_iter, self.line_last_core_last_iter)

    def get_parallel_params(self):
        """Calculate parallel parameters."""
        self.ele_per_line.set_as(self.ceil_div(self.y_grad_w, Constant.NUM_BLOCK_B16) * Constant.NUM_BLOCK_B16)
        if self.dtype in ("float16", "bfloat16"):
            self.move_brust_b16.set_as(self.y_grad_w * Constant.BYTE_B16)
        else:
            ele_per_line_fp32 = self.ceil_div(self.y_grad_w, Constant.NUM_BLOCK_B32) * Constant.NUM_BLOCK_B32
            self.move_ub_gap.set_as((self.ele_per_line - ele_per_line_fp32) // Constant.NUM_BLOCK_B32)
            self.move_brust_b32.set_as(self.y_grad_w * Constant.BYTE_B32)

        self.ele_per_line_mask = self.ceil_div(self.mask_w, Constant.NUM_BLOCK_B8) * Constant.NUM_BLOCK_B8
        self.ele_per_channel.set_as(self.y_grad_h * self.y_grad_w)
        self.total_line.set_as(self.y_grad_n * self.y_grad_c * self.y_grad_h)
        self.max_byte_per_line.set_as(self.ele_per_line * Constant.MAX_B16_UB_NUM * 2)
        self.available_line_per_iter = self.ceil_div(self.available_ub_space, self.max_byte_per_line)

        self.line_per_batch.set_as(self.y_grad_c * self.y_grad_h)
        self.line_per_core.set_as(self.ceil_div(self.total_line, self.tiling_core_num))
        self.iter_per_core.set_as(self.ceil_div(self.line_per_core, self.available_line_per_iter))
        self.line_per_iter.set_as(self.ceil_div(self.line_per_core, self.iter_per_core))
        self.line_last_iter.set_as(self.line_per_core - self.line_per_iter * (self.iter_per_core - 1))
        self.used_core_num.set_as(self.ceil_div(self.total_line, self.line_per_core))
        self.line_last_core.set_as(self.total_line - self.line_per_core * (self.used_core_num - 1))
        self.iter_last_core.set_as(self.ceil_div(self.line_last_core, self.available_line_per_iter))
        self.line_last_core_per_iter.set_as(self.ceil_div(self.line_last_core, self.iter_last_core))
        self.line_last_core_last_iter.set_as(self.line_last_core -
                                             self.line_last_core_per_iter * (self.iter_last_core - 1))

    def cal_broadcast_params(self):
        """Calculate parameters of broadcast."""
        super().cal_broadcast_params()
        if not self.fixed_triu_mask:
            self.move_brust_b8.set_as(self.mask_w * Constant.BYTE_B8)
            self.padding_b8.set_as(self.ele_per_line_mask - self.mask_w)
        with self.tik.if_scope(tik.all(self.mask_n == self.y_grad_n, self.mask_c == self.y_grad_c)):
            self.broad_mode.set_as(1)
        with self.tik.elif_scope(tik.all(self.mask_n == 1, self.mask_c == 1)):
            self.broad_mode.set_as(4)
        with self.tik.elif_scope(tik.all(self.mask_n == self.y_grad_n, self.mask_c < self.y_grad_c)):
            self.broad_mode.set_as(2)
        with self.tik.elif_scope(tik.all(self.mask_n < self.y_grad_n, self.mask_c == self.y_grad_c)):
            self.broad_mode.set_as(3)

    def do_compute(self, core_idx, iter_curr_core, line_per_iter, line_last_iter):
        """Implementation of fused scaled masked softmax grad algorithm."""
        line = self.tik.Scalar("int32")
        offset = self.tik.Scalar("int32")
        offset_mask = self.tik.Scalar("int32")
        ub_fp16 = self.tik.Tensor("float16", (line_per_iter, self.ele_per_line), tik.scope_ubuf, "ub_fp16")
        ub_y_fp32 = self.tik.Tensor("float32", (line_per_iter, self.ele_per_line), tik.scope_ubuf, "ub_y_fp32")
        ub_y_grad_fp32 = self.tik.Tensor("float32", (line_per_iter, self.ele_per_line),
                                         tik.scope_ubuf, "ub_y_grad_fp32")
        mask_ub = self.tik.Tensor("uint8", (line_per_iter, self.ele_per_line_mask),
                                  scope=tbe_platform.scope_ubuf, name="mask_ub")
        ub_reduceadd = self.tik.Tensor("float32", (line_per_iter,), tbe_platform.scope_ubuf, "ub_reduceadd")
        with self.tik.for_range(0, iter_curr_core) as it:
            with self.tik.if_scope(it < iter_curr_core - 1):
                line.set_as(line_per_iter)
            with self.tik.else_scope():
                line.set_as(line_last_iter)
            offset_params = self.cal_offset(core_idx, it, line_per_iter, offset, offset_mask)
            ele_num = line * self.ele_per_line
            self.move_data_in(ub_fp16, ub_y_fp32, self.y_gm, offset, line, ele_num)
            self.move_data_in(ub_fp16, ub_y_grad_fp32, self.y_grad_gm, offset, line, ele_num)
            self.calc_product(ub_y_grad_fp32, ub_y_fp32, ele_num)
            if not self.fixed_triu_mask:
                self.move_mask_in(mask_ub, offset_mask, line, offset_params)
            self.calc_reduce_sum(ub_reduceadd, ub_y_grad_fp32, line)
            self.move_data_in(ub_fp16, ub_y_grad_fp32, self.y_grad_gm, offset, line, ele_num)
            self.calc_softmax_grad(ub_y_grad_fp32, ub_y_fp32, ub_reduceadd, line, ele_num)
            self.scale_grad(ub_y_grad_fp32, ele_num)
            self.masked_fill([ub_fp16, mask_ub, ub_y_grad_fp32, ub_y_fp32], offset_mask, line, ele_num, offset_params)
            self.move_out(ub_fp16, ub_y_grad_fp32, offset, line, ele_num)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def cal_offset(self, core_idx, it, line_per_iter, offset, offset_mask):
        """Calculate the offset of inputs."""
        curr_line = core_idx * self.line_per_core + it * line_per_iter
        offset.set_as(curr_line * self.y_grad_w)
        curr_batch, curr_channel, mask_batch, mask_channel, line_in_channel = self.line_to_nc(curr_line)
        offset_mask.set_as(((mask_batch * self.mask_c + mask_channel) * self.mask_h + line_in_channel) * self.mask_w)
        return [curr_line, curr_batch, curr_channel, mask_batch, mask_channel, line_in_channel]

    def line_to_nc(self, curr_line):
        """Claculate curr batch and channel based on curr line."""
        curr_batch = curr_line // self.line_per_batch
        curr_channel = curr_line % self.line_per_batch // self.y_grad_h
        mask_batch = curr_batch // self.broad_ratio_n
        mask_channel = curr_channel // self.broad_ratio_c
        line_in_channel = curr_line - (curr_batch * self.y_grad_c + curr_channel) * self.y_grad_h
        return [curr_batch, curr_channel, mask_batch, mask_channel, line_in_channel]

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def move_data_in(self, ub_fp16, ub_fp32, src, offset, line, ele_num):
        """Move data form gm to ub."""
        if self.dtype == "float16":
            self.tik.data_move_pad(ub_fp16, src[offset], line, self.move_brust_b16, 0, 0)
            self._vconv(ub_fp32, ub_fp16, 0, 0, ele_num, Constant.VEC_MASK_B32, 8, 4)
        elif self.dtype == "bfloat16":
            with self.tik.new_stmt_scope():
                ub_bf16 = ub_fp16.reinterpret_cast_to("bfloat16")
                self.tik.data_move_pad(ub_bf16, src[offset], line, self.move_brust_b16, 0, 0)
                self._vconv(ub_fp32, ub_bf16, 0, 0, ele_num, Constant.VEC_MASK_B32, 8, 4)
        else:
            self.tik.data_move_pad(ub_fp32, src[offset], line, self.move_brust_b32, self.move_ub_gap, 0)

    def move_mask_in(self, dst, offset, line, offset_params):
        """
        Move mask from gm to ub tensor.
        Mode:
            1 for batch and channel of x and mask are the same (e.g. [4, 4, 32, 32], [4, 4, 32, 32]);
            2 for batch of x and mask are the same (e.g. [4, 4, 32, 32], [4, 1, 32, 32]);
            3 for channel of x and mask are the same (e.g. [4, 4, 32, 32], [1, 4, 32, 32]);
            4 for either batch and channel of x and mask are not same (e.g. [4, 4, 32, 32], [1, 1, 32, 32]).
        """
        with self.tik.if_scope(self.broad_mode == 1):
            if self.fixed_triu_mask:
                _, _, _, s_batch, s_channel, s_line = offset_params
                offset_temp = ((s_batch * self.mask_c + s_channel) * self.mask_h + s_line) * self.ele_per_line
                burst = line * self.ele_per_line // Constant.NUM_BLOCK_B16
                self.tik.data_move(dst, self.fixed_mask_gm[offset_temp], 0, 1, burst, 0, 0)
            else:
                self.tik.data_move_pad(dst, self.mask_gm[offset], line, self.move_brust_b8, 0, 0,
                                       right_padding=self.padding_b8, padding_value=Constant.PADDING_VALUE_MASK)
        with self.tik.elif_scope(self.broad_mode == 2):
            self.move_mask_mode_2(dst, offset, line, offset_params)
        with self.tik.elif_scope(self.broad_mode == 3):
            self.move_mask_mode_3(dst, offset, line, offset_params)
        with self.tik.else_scope():
            self.move_mask_mode_4(dst, offset, line, offset_params)

    def move_mask_mode_2(self, dst, offset, line, offset_params):
        """Move mask from gm to ub in mode 2."""
        if not self.fixed_triu_mask:
            offset_temp = self.tik.Scalar("int32", init_value=0)
            s_line, s_batch, s_channel, _, _, s_line_in_channel = offset_params
            e_line = s_line + line
            e_batch, e_channel, _, _, e_line_in_channel = self.line_to_nc(e_line)
            with self.tik.if_scope(e_batch - s_batch == 0):
                with self.tik.if_scope(e_channel - s_channel == 0):
                    self.move_mask_in_line(dst, offset_temp, offset, line)
                with self.tik.else_scope():
                    self.move_mask_in_line(dst, offset_temp, offset, self.mask_h - s_line_in_channel)
                    offset_temp.set_as((self.mask_h - s_line_in_channel) * self.ele_per_line_mask)
                    with self.tik.for_range(0, e_channel - s_channel - 1) as _:
                        self.move_mask_in_line(dst, offset_temp, s_batch * self.ele_per_channel, self.mask_h)
                        offset_temp.set_as(offset_temp + self.ele_per_line_mask * self.mask_h)
                    with self.tik.if_scope(e_line_in_channel > 0):
                        self.move_mask_in_line(dst, offset_temp, s_batch * self.ele_per_channel, e_line_in_channel)
            with self.tik.else_scope():
                self.move_mask_in_line(dst, offset_temp, offset, self.mask_h - s_line_in_channel)
                offset_temp.set_as((self.mask_h - s_line_in_channel) * self.ele_per_line_mask)
                with self.tik.for_range(0, self.y_grad_c - s_channel - 1) as _:
                    self.move_mask_in_line(dst, offset_temp, s_batch * self.ele_per_channel, self.mask_h)
                    offset_temp.set_as(offset_temp + self.ele_per_line_mask * self.mask_h)
                with self.tik.for_range(1, e_batch - s_batch) as i:
                    offset_gm = (s_batch + i) * self.ele_per_channel
                    with self.tik.for_range(0, self.y_grad_c) as _:
                        self.move_mask_in_line(dst, offset_temp, offset_gm, self.mask_h)
                        offset_temp.set_as(offset_temp + self.ele_per_line_mask * self.mask_h)
                with self.tik.if_scope(e_channel > 0):
                    with self.tik.for_range(0, e_channel) as _:
                        self.move_mask_in_line(dst, offset_temp, e_batch * self.ele_per_channel, self.mask_h)
                        offset_temp.set_as(offset_temp + self.ele_per_line_mask * self.mask_h)
                with self.tik.if_scope(e_line_in_channel > 0):
                    self.move_mask_in_line(dst, offset_temp, e_batch * self.ele_per_channel, e_line_in_channel)

    def move_mask_mode_3(self, dst, offset, line, offset_params):
        """Move mask from gm to ub in mode 3."""
        if not self.fixed_triu_mask:
            offset_temp = self.tik.Scalar("int32", init_value=0)
            s_line, s_batch, s_channel, _, _, s_line_in_channel = offset_params
            e_line = s_line + line
            e_batch, e_channel, _, _, e_line_in_channel = self.line_to_nc(e_line)
            with self.tik.if_scope(e_batch - s_batch == 0):
                self.move_mask_in_line(dst, offset_temp, offset, line)
            with self.tik.else_scope():
                line_temp = self.mask_h * self.mask_c - s_channel * self.mask_h - s_line_in_channel
                self.move_mask_in_line(dst, offset_temp, offset, line_temp)
                offset_temp.set_as(line_temp * self.ele_per_line_mask)
                with self.tik.for_range(1, e_batch - s_batch) as i:
                    self.move_mask_in_line(dst, offset_temp, 0, self.mask_h * self.mask_c)
                    offset_temp.set_as(offset_temp + self.ele_per_line_mask * self.mask_h * self.mask_c)
                line_temp = e_channel * self.mask_h + e_line_in_channel
                with self.tik.if_scope(line_temp > 0):
                    self.move_mask_in_line(dst, offset_temp, 0, line_temp)

    def move_mask_mode_4(self, dst, offset, line, offset_params):
        """Move mask from gm to ub in mode 4."""
        offset_temp = self.tik.Scalar("int32", init_value=0)
        s_line, s_batch, s_channel, mask_batch, mask_channel, s_line_in_channel = offset_params
        e_line = s_line + line
        e_batch, e_channel, _, _, e_line_in_channel = self.line_to_nc(e_line)
        s_nc = s_batch * self.y_grad_c + s_channel
        e_nc = e_batch * self.y_grad_c + e_channel
        if self.fixed_triu_mask:
            offset_mask = ((mask_batch * self.mask_c + mask_channel) *
                           self.mask_h + s_line_in_channel) * self.ele_per_line
            with self.tik.if_scope(e_nc - s_nc == 0):
                burst = line * self.ele_per_line // Constant.NUM_BLOCK_B16
                self.tik.data_move(dst, self.fixed_mask_gm[offset_mask], 0, 1, burst, 0, 0)
            with self.tik.else_scope():
                burst = (self.mask_h - s_line_in_channel) * self.ele_per_line // Constant.NUM_BLOCK_B16
                self.tik.data_move(dst, self.fixed_mask_gm[offset_mask], 0, 1, burst, 0, 0)
                offset_temp.set_as((self.mask_h - s_line_in_channel) * self.ele_per_line)
                with self.tik.for_range(1, e_nc - s_nc) as _:
                    burst = self.mask_h * self.ele_per_line // Constant.NUM_BLOCK_B16
                    self.tik.data_move(dst[offset_temp], self.fixed_mask_gm, 0, 1, burst, 0, 0)
                    offset_temp.set_as(offset_temp + self.mask_h * self.ele_per_line)
                with self.tik.if_scope(e_line_in_channel > 0):
                    burst = e_line_in_channel * self.ele_per_line // Constant.NUM_BLOCK_B16
                    self.tik.data_move(dst[offset_temp], self.fixed_mask_gm, 0, 1, burst, 0, 0)
        else:
            with self.tik.if_scope(e_nc - s_nc == 0):
                self.move_mask_in_line(dst, offset_temp, offset, line)
            with self.tik.else_scope():
                self.move_mask_in_line(dst, offset_temp, offset, self.mask_h - s_line_in_channel)
                offset_temp.set_as((self.mask_h - s_line_in_channel) * self.ele_per_line_mask)
                with self.tik.for_range(1, e_nc - s_nc) as _:
                    self.move_mask_in_line(dst, offset_temp, 0, self.mask_h)
                    offset_temp.set_as(offset_temp + self.mask_h * self.ele_per_line_mask)
                with self.tik.if_scope(e_line_in_channel > 0):
                    self.move_mask_in_line(dst, offset_temp, 0, e_line_in_channel)

    def move_mask_in_line(self, dst, dst_offset, src_offset, line):
        """Move mask from gm to ub in lines."""
        self.tik.data_move_pad(dst[dst_offset], self.mask_gm[src_offset], line, self.move_brust_b8,
                               0, 0, right_padding=self.padding_b8, padding_value=Constant.PADDING_VALUE_MASK)

    def masked_fill(self, ub_list, offset_mask, line, ele_num, offset_params):
        """Do masked_fill."""
        ub_fp16, mask_ub, ub_y_grad_fp32, ub_y_fp32 = ub_list
        if self.dtype == "float16":
            self._vconv(ub_fp16, ub_y_grad_fp32, 0, 0, ele_num, Constant.VEC_MASK_B32, 4, 8)
            mask_ub_fp16 = ub_y_fp32.reinterpret_cast_to("float16")
        else:
            mask_ub_fp16 = ub_fp16
        if self.fixed_triu_mask:
            self.move_mask_in(mask_ub_fp16, offset_mask, line, offset_params)
        else:
            self.align_mask(mask_ub_fp16, mask_ub, line, self.ele_per_line,
                            self.ele_per_line_mask, Constant.VEC_MASK_B16)
        self._vec_scalar_func("vmuls", mask_ub_fp16, mask_ub_fp16, -1.0, 0, 0, ele_num, Constant.VEC_MASK_B16)
        self._vec_scalar_func("vadds", mask_ub_fp16, mask_ub_fp16, 1, 0, 0, ele_num, Constant.VEC_MASK_B16)
        if self.dtype == "float16":
            self._vec_func("vmul", ub_fp16, ub_fp16, mask_ub_fp16, 0, 0, 0, ele_num, Constant.VEC_MASK_B16)
        else:
            self._vconv(ub_y_fp32, mask_ub_fp16, 0, 0, ele_num, Constant.VEC_MASK_B32, 8, 4)
            self._vec_func("vmul", ub_y_grad_fp32, ub_y_grad_fp32, ub_y_fp32, 0, 0, 0, ele_num, Constant.VEC_MASK_B32)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def align_mask(self, dst, src, line, ele_dst, ele_src, mask):
        """Align mask to input."""
        with self.tik.for_range(0, line) as i:
            self._vconv(dst, src, ele_dst * i, ele_src * i, ele_dst, mask, 8, 4)

    def move_out(self, ub_fp16, ub_fp32, offset, line, ele_num):
        """Move result from ub to gm."""
        if self.dtype == "float16":
            self.tik.data_move_pad(self.x_grad_gm[offset], ub_fp16, line, self.move_brust_b16, 0, 0)
        elif self.dtype == "bfloat16":
            ub_bf16 = ub_fp16.reinterpret_cast_to("bfloat16")
            self._vconv(ub_bf16, ub_fp32, 0, 0, ele_num, Constant.VEC_MASK_B32, 4, 8, 'round')
            self.tik.data_move_pad(self.x_grad_gm[offset], ub_bf16, line, self.move_brust_b16, 0, 0)
        else:
            self.tik.data_move_pad(self.x_grad_gm[offset], ub_fp32, line, self.move_brust_b32, 0, self.move_ub_gap)


class ScaledMaskedSoftmaxGradAlignedND(BaseClass):
    """Implementation of aligned ScaledMaskedSoftmaxGrad in ND format."""

    def __init__(self, op: "ScaledMaskedSoftmaxGrad"):
        """Init."""
        super().__init__(op)
        self.counts = self.tik.Scalar("int32")
        self.y_grad_h.set_as(self.y_grad_h1)
        self.y_grad_w.set_as(self.y_grad_w1)
        self.y_h.set_as(self.y_h1)
        self.y_w.set_as(self.y_w1)
        self.mask_h.set_as(self.mask_h1)
        self.mask_w.set_as(self.mask_w1)
        self.shape = (self.line_per_iter, self.y_grad_w)

    def gen_triu_mask(self):
        """
        Generate aligned triu mask if fixed_triu_mask si true.
        """
        dup_len = self.tik.Scalar("int32", init_value=1)
        offset = self.tik.Scalar("int32", init_value=0)
        triu_mask_ub = self.tik.Tensor('float16', (self.mask_w,), name="tri_mask_ub", scope=tbe_platform.scope_ubuf)
        self._vec_dup(triu_mask_ub, 1.0, 0, self.mask_w, Constant.VEC_MASK_B16)
        with self.tik.for_range(0, self.mask_h) as _:
            with self.tik.if_scope(dup_len <= self.mask_w):
                self._vec_dup(triu_mask_ub, 0.0, 0, dup_len, Constant.VEC_MASK_B16)
                dup_len.set_as(dup_len + 1)
            self.tik.data_move(self.fixed_mask_gm[offset], triu_mask_ub,
                               0, 1, self.mask_w // Constant.NUM_BLOCK_B16, 0, 0)
            offset.set_as(offset + self.mask_w)

    def do_compute(self, core_idx, iter_curr_core):
        """Implementation of fused scaled masked softmax grad algorithm."""
        offset = self.tik.Scalar("int32")
        offset_mask = self.tik.Scalar("int32")
        ub_fp16 = self.tik.Tensor("float16", self.shape, tik.scope_ubuf, "ub_fp16")
        ub_y_fp32 = self.tik.Tensor("float32", self.shape, tik.scope_ubuf, "ub_y_fp32")
        ub_y_grad_fp32 = self.tik.Tensor("float32", self.shape, tik.scope_ubuf, "ub_y_grad_fp32")
        mask_ub = self.tik.Tensor("uint8", self.shape, scope=tbe_platform.scope_ubuf, name="mask_ub")
        ub_reduceadd = self.tik.Tensor("float32", (self.line_per_iter,), tbe_platform.scope_ubuf, "ub_reduceadd")
        with self.tik.for_range(0, iter_curr_core) as it:
            self.cal_offset(core_idx, it, offset, offset_mask)
            self.move_data_in(ub_fp16, ub_y_fp32, self.y_gm, offset)
            self.move_data_in(ub_fp16, ub_y_grad_fp32, self.y_grad_gm, offset)
            self.calc_product(ub_y_grad_fp32, ub_y_fp32, self.ele_per_iter)
            self.calc_reduce_sum(ub_reduceadd, ub_y_grad_fp32)
            self.move_data_in(ub_fp16, ub_y_grad_fp32, self.y_grad_gm, offset)
            self.calc_softmax_grad(ub_y_grad_fp32, ub_y_fp32, ub_reduceadd, ub_fp16)
            if not self.fixed_triu_mask:
                self.move_mask_in(mask_ub, offset_mask)
            self.scale_grad(ub_y_grad_fp32, self.ele_per_iter)
            self.masked_fill([ub_fp16, mask_ub, ub_y_grad_fp32, ub_y_fp32], offset_mask)
            self.move_out(ub_fp16, ub_y_grad_fp32, offset)

    def get_parallel_params(self):
        """Calculate parallel parameters."""
        super().get_parallel_params()
        self.counts.set_as(self.ele_per_iter)

    def move_data_in(self, ub_fp16, ub_fp32, src, offset):
        """Move data form gm to ub."""
        if self.dtype == "float16":
            self.tik.data_move(ub_fp16, src[offset], 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B16, 0, 0)
            self._vconv(ub_fp32, ub_fp16, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32, 8, 4)
        elif self.dtype == "bfloat16":
            with self.tik.new_stmt_scope():
                ub_bf16 = ub_fp16.reinterpret_cast_to("bfloat16")
                self.tik.data_move(ub_bf16, src[offset], 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B16, 0, 0)
                self._vconv(ub_fp32, ub_bf16, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32, 8, 4)
        else:
            self.tik.data_move(ub_fp32, src[offset], 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B32, 0, 0)

    def calc_reduce_sum(self, dst, src):
        """Calculate sum of each line."""
        with self.tik.if_scope(self.ele_per_line == Constant.SHAPE_2048):
            self.tik.vcgadd(Constant.VEC_MASK_B32, src, src, Constant.SEVEN_LINE_REPEAT, 1, 1, 8)
            self.tik.vcgadd(Constant.VEC_MASK_B32, src[Constant.SEVEN_LINE_REPEAT * Constant.NUM_BLOCK_B32],
                            src[Constant.SEVEN_LINE_REPEAT * Constant.VEC_MASK_B32], 
                            Constant.SHAPE_2048 // Constant.VEC_MASK_B32, 1, 1, 8)
            self.tik.vcgadd(Constant.VEC_MASK_B32, src, src, Constant.SHAPE_2048 // Constant.VEC_MASK_B32, 1, 1, 8)
            self.tik.vcadd(Constant.SHAPE_2048 // Constant.VEC_MASK_B32, dst, src, 8, 1, 1, 4)
        with self.tik.else_scope():
            super().calc_reduce_sum(dst, src, self.line_per_iter)

    def calc_softmax_grad(self, ub_y_grad_fp32, ub_y_fp32, ub_reduceadd, ub_fp16):
        """Calculate softmax gradient."""
        with self.tik.if_scope(self.ele_per_line == Constant.SHAPE_2048):
            self.tik.vmuls(Constant.NUM_BLOCK_B32, ub_reduceadd, ub_reduceadd, -1.0, 1, 1, 1, 0, 0)
            ub_fp32_temp = ub_fp16.reinterpret_cast_to("float32")
            self.tik.vbcb(ub_fp32_temp, ub_reduceadd, 1, 1, 8)
            for i in range(8):
                self.tik.vadd(Constant.VEC_MASK_B32, ub_y_grad_fp32[Constant.SHAPE_2048 * i], 
                              ub_y_grad_fp32[Constant.SHAPE_2048 * i],
                              ub_fp32_temp[8 * i], Constant.SHAPE_2048 // Constant.VEC_MASK_B32, 1, 1, 0, 8, 8, 0)
            self._vec_func("vmul", ub_y_grad_fp32, ub_y_grad_fp32, ub_y_fp32,
                           0, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32)
        with self.tik.else_scope():
            super().calc_softmax_grad(ub_y_grad_fp32, ub_y_fp32, ub_reduceadd, self.line_per_iter, self.ele_per_iter)

    def move_mask_in(self, dst, offset):
        """Move mask from gm to ub tensor."""
        if self.fixed_triu_mask:
            move_burst = self.ele_per_iter // Constant.NUM_BLOCK_B16
            self.tik.data_move(dst, self.fixed_mask_gm[offset], 0, 1, move_burst, 0, 0)
        else:
            move_burst = self.ele_per_iter // Constant.NUM_BLOCK_B8
            self.tik.data_move(dst, self.mask_gm[offset], 0, 1, move_burst, 0, 0)

    def move_out(self, ub_fp16, ub_fp32, offset):
        """Move result from ub to gm."""
        if self.dtype == "float16":
            self.tik.data_move(self.x_grad_gm[offset], ub_fp16, 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B16, 0, 0)
        elif self.dtype == "bfloat16":
            ub_bf16 = ub_fp16.reinterpret_cast_to("bfloat16")
            self._vconv(ub_bf16, ub_fp32, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32, 4, 8, 'round')
            self.tik.data_move(self.x_grad_gm[offset], ub_bf16, 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B16, 0, 0)
        else:
            self.tik.data_move(self.x_grad_gm[offset], ub_fp32, 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B32, 0, 0)


class ScaledMaskedSoftmaxGradAlignedNZ(BaseClass):
    """Implementation of aligned ScaledMaskedSoftmaxGrad in NZ format."""

    def __init__(self, op: "ScaledMaskedSoftmaxGrad"):
        """Init."""
        super().__init__(op)
        self.counts = self.tik.Scalar("int32")
        self.ele_per_col = self.tik.Scalar("int32")
        self.y_grad_h.set_as(self.y_grad_h1 * self.y_grad_h0)
        self.y_grad_w.set_as(self.y_grad_w1 * self.y_grad_w0)
        self.y_h.set_as(self.y_h1 * self.y_h0)
        self.y_w.set_as(self.y_w1 * self.y_w0)
        self.mask_h.set_as(self.mask_h1 * self.mask_h0)
        self.mask_w.set_as(self.mask_w1 * self.mask_w0)
        self.shape = (self.line_per_iter, self.y_grad_w)

    def gen_triu_mask(self):
        """
        Generate aligned triu mask if fixed_triu_mask si true.
        """
        dup_len = self.tik.Scalar("int32", init_value=1)
        offset = self.tik.Scalar("int32", init_value=0)
        triu_mask_ub = self.tik.Tensor('float16', (self.mask_w,), name="tri_mask_ub", scope=tbe_platform.scope_ubuf)
        self._vec_dup(triu_mask_ub, 1.0, 0, self.mask_w, Constant.VEC_MASK_B16)
        with self.tik.for_range(0, self.mask_h) as _:
            with self.tik.if_scope(dup_len <= self.mask_w):
                self._vec_dup(triu_mask_ub, 0.0, 0, dup_len, Constant.VEC_MASK_B16)
                dup_len.set_as(dup_len + 1)
            self.tik.data_move(self.fixed_mask_gm[offset], triu_mask_ub,
                               0, self.mask_w // Constant.NUM_BLOCK_B16, 1, 0, self.mask_h - 1)
            offset.set_as(offset + Constant.NUM_BLOCK_B16)

    def do_compute(self, core_idx, iter_curr_core):
        """Implementation of fused scaled masked softmax grad algorithm."""
        offset = self.tik.Scalar("int32")
        offset_mask = self.tik.Scalar("int32")
        ub_fp16 = self.tik.Tensor("float16", self.shape, tik.scope_ubuf, "ub_fp16")
        ub_y_fp32 = self.tik.Tensor("float32", self.shape, tik.scope_ubuf, "ub_y_fp32")
        ub_y_grad_fp32 = self.tik.Tensor("float32", self.shape, tik.scope_ubuf, "ub_y_grad_fp32")
        mask_ub = self.tik.Tensor("uint8", self.shape, scope=tbe_platform.scope_ubuf, name="mask_ub")
        ub_dup = self.tik.Tensor("float32", (Constant.NUM_BLOCK_B8, Constant.NUM_BLOCK_B16),
                                 scope=tbe_platform.scope_ubuf, name="ub_dup")
        ub_reduceadd = self.tik.Tensor("float32", (self.line_per_iter,),
                                       scope=tbe_platform.scope_ubuf, name="ub_reduceadd")

        with self.tik.for_range(0, iter_curr_core) as it:
            self.cal_offset(core_idx, it, offset, offset_mask)
            self.move_data_in(ub_fp16, ub_y_fp32, self.y_gm, offset)
            self.move_data_in(ub_fp16, ub_y_grad_fp32, self.y_grad_gm, offset)
            self.calc_product(ub_y_grad_fp32, ub_y_fp32, self.ele_per_iter)
            self.calc_reduce_sum(ub_dup, ub_reduceadd, ub_y_grad_fp32)
            self.move_data_in(ub_fp16, ub_y_grad_fp32, self.y_grad_gm, offset)
            self.calc_softmax_grad(ub_y_grad_fp32, ub_y_fp32, ub_dup)
            if not self.fixed_triu_mask:
                self.move_mask_in(mask_ub, offset_mask)
            self.scale_grad(ub_y_grad_fp32, self.ele_per_iter)
            self.masked_fill([ub_fp16, mask_ub, ub_y_grad_fp32, ub_y_fp32], offset_mask)
            self.move_out(ub_fp16, ub_y_grad_fp32, offset)

    def get_parallel_params(self):
        """Calculate parallel parameters."""
        super().get_parallel_params()
        self.counts.set_as(self.line_per_iter * Constant.NUM_BLOCK_B16)
        self.ele_per_col.set_as(Constant.NUM_BLOCK_B16 * self.line_per_iter)

    def move_data_in(self, ub_fp16, ub_fp32, src, offset):
        """Move data form gm to ub."""
        if self.dtype == "float16":
            self.tik.data_move(ub_fp16, src[offset], 0,
                               self.y_grad_w1, self.line_per_iter, self.y_grad_h - self.line_per_iter, 0)
            self._vconv(ub_fp32, ub_fp16, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32, 8, 4)
        elif self.dtype == "bfloat16":
            with self.tik.new_stmt_scope():
                ub_bf16 = ub_fp16.reinterpret_cast_to("bfloat16")
                self.tik.data_move(ub_bf16, src[offset], 0,
                                   self.y_grad_w1, self.line_per_iter, self.y_grad_h - self.line_per_iter, 0)
                self._vconv(ub_fp32, ub_bf16, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32, 8, 4)
        else:
            self.tik.data_move(ub_fp32, src[offset], 0, self.y_grad_w1,
                               self.line_per_iter * Constant.SCALAR_TWO,
                               (self.y_grad_h - self.line_per_iter) * Constant.SCALAR_TWO, 0)

    def calc_reduce_sum(self, ub_dup, ub_reduceadd, src):
        """Calculate sum of each line."""
        cnt, remain, _ = self.cal_level(Constant.NUM_BLOCK_B16)
        dup_times = self.line_per_iter * self.y_grad_h0 // Constant.NUM_BLOCK_B16 // Constant.SCALAR_EIGHT
        time = self.tik.Scalar("int32", init_value=Constant.SCALAR_TWO)

        self._vec_func("vadd", src, src, src, 0, self.ele_per_iter // time, 0,
                       self.ele_per_iter // time, Constant.VEC_MASK_B32)
        with self.tik.for_range(1, cnt) as _:
            time.set_as(time * Constant.SCALAR_TWO)
            self._vec_func("vadd", src, src, src, 0, self.ele_per_iter // time, 0,
                           self.ele_per_iter // time, Constant.VEC_MASK_B32)
        with self.tik.if_scope(remain > 1):
            with self.tik.for_range(1, remain) as i:
                self._vec_func("vadd", src, src, src, self.ele_per_col * (remain - i - 1),
                               self.ele_per_col * (remain - i), self.ele_per_col * (remain - i - 1),
                               self.ele_per_col, Constant.VEC_MASK_B32)
        self.tik.vcadd(Constant.NUM_BLOCK_B16, ub_reduceadd, src, self.ele_per_col // Constant.NUM_BLOCK_B16, 1, 1, 2)

        self._vec_scalar_func("vmuls", ub_reduceadd, ub_reduceadd, -1.0,
                              0, 0, self.line_per_iter, Constant.VEC_MASK_B32)
        with self.tik.for_range(0, dup_times) as j:
            with self.tik.for_range(0, Constant.SCALAR_EIGHT) as k:
                self.tik.vector_dup(Constant.NUM_BLOCK_B16,
                                    ub_dup[j * Constant.VEC_MASK_B16 + Constant.NUM_BLOCK_B16 * k],
                                    self.tik.Scalar(init_value=ub_reduceadd[j * Constant.NUM_BLOCK_B32 + k],
                                                    dtype="float32"),
                                    1, 1, 8)

    def move_mask_in(self, dst, offset):
        """Move mask from gm to ub tensor."""
        if self.fixed_triu_mask:
            self.tik.data_move(dst, self.fixed_mask_gm[offset], 0,
                               self.y_grad_w1, self.line_per_iter, self.y_grad_h - self.line_per_iter, 0)
        else:
            burst = self.line_per_iter // Constant.SCALAR_TWO
            stride = (self.y_grad_h - self.line_per_iter) // Constant.SCALAR_TWO
            self.tik.data_move(dst, self.mask_gm[offset], 0, self.y_grad_w1, burst, stride, 0)

    def calc_softmax_grad(self, ub_y_grad_fp32, ub_y_fp32, ub_dup):
        """Calculate softmax gradient."""
        stride = self.line_per_iter * Constant.SCALAR_TWO
        with self.tik.for_range(0, self.ele_per_col // Constant.VEC_MASK_B32) as i:
            self.tik.vsub(Constant.VEC_MASK_B32, ub_y_grad_fp32[i * Constant.VEC_MASK_B32],
                          ub_y_grad_fp32[i * Constant.VEC_MASK_B32], ub_dup[i * Constant.VEC_MASK_B32],
                          self.y_grad_w1, 1, 1, 1, stride, stride, 0)
        self._vec_func("vmul", ub_y_grad_fp32, ub_y_grad_fp32, ub_y_fp32,
                       0, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32)

    def move_out(self, ub_fp16, ub_fp32, offset):
        """Move result from ub to gm."""
        if self.dtype == "float16":
            self.tik.data_move(self.x_grad_gm[offset], ub_fp16, 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B16, 0, 0)
        elif self.dtype == "bfloat16":
            ub_bf16 = ub_fp16.reinterpret_cast_to("bfloat16")
            self._vconv(ub_bf16, ub_fp32, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32, 4, 8, 'round')
            self.tik.data_move(self.x_grad_gm[offset], ub_bf16, 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B16, 0, 0)
        else:
            self.tik.data_move(self.x_grad_gm[offset], ub_fp32, 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B32, 0, 0)


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,huawei-too-many-arguments
@register_operator("ScaledMaskedSoftmaxGrad")
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
    if not tbe_platform.api_check_support("tik.data_move_pad"):
        raise RuntimeError("Error! The device does not support data_move_pad api.")
    return ScaledMaskedSoftmaxGrad(y_grad, y, mask, x_grad, scale, fixed_triu_mask, kernel_name).compute()
