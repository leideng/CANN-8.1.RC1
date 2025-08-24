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
Dynamic ScaledMaskedSoftmax.
"""
from abc import ABCMeta
from impl import constant_util as constant
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator


class Constant:
    TILING_ARG_NUM = 16
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


class ScaledMaskedSoftmax():
    """Class for ScaledMaskedSoftmax."""

    # 'pylint: disable=too-many-arguments,unused-argument,huawei-too-many-arguments
    def __init__(self, x, mask, y, scale, fixed_triu_mask, kernel_name):
        """Init class."""
        self.tik = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.fixed_triu_mask = fixed_triu_mask
        self.kernel_name = kernel_name
        self.dtype = x.get("dtype")
        self.is_nz = x.get("format") == "FRACTAL_NZ"
        self.init_scalar()
        self.init_gm()
        self.get_tiling_data()

    def init_scalar(self):
        """Init scalars."""
        self.tiling_core_num = self.tik.Scalar("int32")
        self.tiling_mode = self.tik.Scalar("int32")
        self.x_n = self.tik.Scalar("int32")
        self.x_c = self.tik.Scalar("int32")
        self.x_h1 = self.tik.Scalar("int32")
        self.x_w1 = self.tik.Scalar("int32")
        self.x_h0 = self.tik.Scalar("int32")
        self.x_w0 = self.tik.Scalar("int32")
        self.mask_n = self.tik.Scalar("int32")
        self.mask_c = self.tik.Scalar("int32")
        self.mask_h1 = self.tik.Scalar("int32")
        self.mask_w1 = self.tik.Scalar("int32")
        self.mask_h0 = self.tik.Scalar("int32")
        self.mask_w0 = self.tik.Scalar("int32")
        self.scale = self.tik.Scalar("float32")
        self.scale_fp16 = self.tik.Scalar("float16")

    def init_gm(self):
        """Init gm space."""
        self.x_gm = self.tik.Tensor(self.dtype, [Constant.SHAPE_SIZE_LIMIT], tik.scope_gm, "x_gm")
        self.mask_gm = self.tik.Tensor("uint8", [Constant.SHAPE_SIZE_LIMIT], tik.scope_gm, "mask_gm")
        self.y_gm = self.tik.Tensor(self.dtype, [Constant.SHAPE_SIZE_LIMIT], tik.scope_gm, "y_gm")
        if self.fixed_triu_mask:
            self.fixed_mask_gm = self.tik.Tensor('float16', [Constant.SHAPE_SIZE_LIMIT],
                                                 tik.scope_gm, "fixed_mask_gm", is_workspace=True)

    def get_tiling_data(self):
        """Get tiling data."""
        core_num = tik.Dprofile().get_aicore_num()
        tbe_context.get_context().add_compile_info("vars", {"core_num": core_num})
        self.tiling_gm = self.tik.Tensor("int32", [Constant.TILING_ARG_NUM], tik.scope_gm, "tiling_gm")
        tiling_ub = self.tik.Tensor("int32", [Constant.TILING_ARG_NUM], tik.scope_ubuf, "tiling_ub")
        self.tik.data_move(tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
        self.tiling_core_num.set_as(tiling_ub[0])
        self.tiling_mode.set_as(tiling_ub[1])
        self.x_n.set_as(tiling_ub[2])
        self.x_c.set_as(tiling_ub[3])
        self.x_h1.set_as(tiling_ub[4])
        self.x_w1.set_as(tiling_ub[5])
        self.x_h0.set_as(tiling_ub[6])
        self.x_w0.set_as(tiling_ub[7])
        self.mask_n.set_as(tiling_ub[8])
        self.mask_c.set_as(tiling_ub[9])
        self.mask_h1.set_as(tiling_ub[10])
        self.mask_w1.set_as(tiling_ub[11])
        self.mask_h0.set_as(tiling_ub[12])
        self.mask_w0.set_as(tiling_ub[13])
        tiling_ub_fp32 = tiling_ub.reinterpret_cast_to("float32")
        self.scale.set_as(tiling_ub_fp32[14])
        if self.dtype == "float16":
            self.tik.scalar_conv('', self.scale_fp16, self.scale)

    def compute(self):
        """Enter of the way to different subclass."""
        with self.tik.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as core_idx:
            with self.tik.if_scope(self.tiling_mode == 1):
                ScaledMaskedSoftmaxUnalignedND(self).compute(core_idx)
            with self.tik.elif_scope(self.tiling_mode == 2):
                ScaledMaskedSoftmaxAlignedND(self).compute(core_idx)
            with self.tik.elif_scope(self.tiling_mode == 3):
                ScaledMaskedSoftmaxAlignedNZ(self).compute(core_idx)
        return self.build_cce()

    def build_cce(self):
        """Build cce."""
        self.tik.BuildCCE(kernel_name=self.kernel_name,
                          inputs=[self.x_gm, self.mask_gm],
                          outputs=[self.y_gm],
                          flowtable=[self.tiling_gm],
                          config={"enable_const_fold": True})
        return self.tik


class BaseClass(metaclass=ABCMeta):
    """Baseclass for different ScaledMaskedSoftmax implementation."""

    def __init__(self, op: "ScaledMaskedSoftmax"):
        """Init."""
        self.tik = op.tik
        self.is_nz = op.is_nz
        self.dtype = op.dtype
        self.x_n = op.x_n
        self.x_c = op.x_c
        self.x_h1 = op.x_h1
        self.x_w1 = op.x_w1
        self.x_h0 = op.x_h0
        self.x_w0 = op.x_w0
        self.mask_n = op.mask_n
        self.mask_c = op.mask_c
        self.mask_h1 = op.mask_h1
        self.mask_w1 = op.mask_w1
        self.mask_h0 = op.mask_h0
        self.mask_w0 = op.mask_w0
        self.scale = op.scale
        self.scale_fp16 = op.scale_fp16
        self.fixed_triu_mask = op.fixed_triu_mask
        self.tiling_core_num = op.tiling_core_num
        self.available_ub_space = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 33 * 1024

        self.x_gm = op.x_gm
        self.mask_gm = op.mask_gm
        self.y_gm = op.y_gm
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
        self.x_h = self.tik.Scalar("int32")
        self.x_w = self.tik.Scalar("int32")
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
        with self.tik.if_scope(self.x_w <= 512):
            self.line_per_iter.set_as(32)
        with self.tik.elif_scope(self.x_w <= 1024):
            self.line_per_iter.set_as(16)
        with self.tik.elif_scope(self.x_w <= 2048):
            self.line_per_iter.set_as(8)
        with self.tik.elif_scope(self.x_w <= 4096):
            self.line_per_iter.set_as(4)
        with self.tik.else_scope():
            self.line_per_iter.set_as(2)
        self.total_iter.set_as(self.x_n * self.x_c * self.x_h // self.line_per_iter)
        self.iter_per_batch.set_as(self.x_c * self.x_h // self.line_per_iter)
        self.iter_per_channel.set_as(self.x_h // self.line_per_iter)
        self.iter_per_core = self.ceil_div(self.total_iter, self.tiling_core_num)
        self.used_core_num.set_as(self.ceil_div(self.total_iter, self.iter_per_core))
        self.iter_last_core.set_as(self.total_iter - (self.used_core_num - 1) * self.iter_per_core)
        self.ele_per_line.set_as(self.x_w)
        self.ele_per_iter.set_as(self.x_w * self.line_per_iter)
        self.ele_per_core.set_as(self.ele_per_iter * self.iter_per_core)

    def cal_broadcast_params(self):
        """Calculate parameters of broadcast for mask."""
        if self.fixed_triu_mask:
            self.broad_ratio_n.set_as(self.x_n)
            self.broad_ratio_c.set_as(self.x_c)
        else:
            self.broad_ratio_n.set_as(self.x_n // self.mask_n)
            self.broad_ratio_c.set_as(self.x_c // self.mask_c)

    def cal_offset(self, core_idx, it, offset, offset_mask, counts):
        """
        Calculate offset of data move for following calculation.
        """
        curr_batch = (core_idx * self.iter_per_core + it) // self.iter_per_batch
        curr_channel = (core_idx * self.iter_per_core + it) % self.iter_per_batch // self.iter_per_channel
        iter_in_curr_channel = core_idx * self.iter_per_core + it - curr_batch *\
            self.iter_per_batch - curr_channel * self.iter_per_channel
        offset_mask.set_as(((curr_batch // self.broad_ratio_n) * self.mask_c +
                           (curr_channel // self.broad_ratio_c)) * self.iter_per_channel * self.ele_per_iter +
                           iter_in_curr_channel * counts)
        offset.set_as((curr_batch * self.x_c + curr_channel) * self.iter_per_channel *
                      self.ele_per_iter + iter_in_curr_channel * counts)

    def scale_x(self, src):
        """Scale input."""
        if src.dtype == "float16":
            self._vec_scalar_func("vmuls", src, src, self.scale_fp16, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B16)
        else:
            self._vec_scalar_func("vmuls", src, src, self.scale, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32)

    def masked_fill(self, dst, ub_mask, ub_temp):
        """Do masked_fill on input."""
        if dst.dtype == "float16":
            self._vec_func("vmul", dst, dst, ub_mask, 0, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B16)
            self._vec_func("vadd", dst, dst, ub_temp, 0, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B16)
        else:
            self._vec_func("vmul", dst, dst, ub_mask, 0, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32)
            self._vconv(ub_mask, ub_temp, self.ele_per_iter, Constant.VEC_MASK_B32, 8, 4)
            self._vec_func("vadd", dst, dst, ub_mask, 0, 0, 0, self.ele_per_iter, Constant.VEC_MASK_B32)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def reduce_max_per_line(self, src, temp, offset, cnt, remain, mask, block):
        """Get the maximum value of each line of input."""
        time = self.tik.Scalar("int32", init_value=Constant.SCALAR_TWO)
        with self.tik.if_scope(cnt > 0):
            self._vec_func("vmax", temp, src, src, offset, offset + self.ele_per_line // time, offset,
                           self.ele_per_line // time, mask)
            time.set_as(time * Constant.SCALAR_TWO)
            with self.tik.for_range(1, cnt) as _:
                self._vec_func("vmax", temp, temp, temp, offset, offset + self.ele_per_line // time, offset,
                               self.ele_per_line // time, mask)
                time.set_as(time * Constant.SCALAR_TWO)
        with self.tik.if_scope(remain > 1):
            with self.tik.if_scope(cnt == 0):
                self._vec_func("vmax", temp, src, src, offset + block * (remain - 2), offset + block * (remain - 1),
                               offset + block * (remain - 2), block, mask)
                with self.tik.for_range(2, remain) as i:
                    self._vec_func("vmax", temp, temp, src, offset + block * (remain - 1 - i),
                                   offset + block * (remain - i), offset + block * (remain - 1 - i), block, mask)
            with self.tik.else_scope():
                with self.tik.for_range(1, remain) as i:
                    self._vec_func("vmax", temp, temp, temp, offset + block * (remain - 1 - i),
                                   offset + block * (remain - i), offset + block * (remain - 1 - i), block, mask)

    def cal_level(self, block):
        """
        Calculate vmax and vadd times.
        """
        cnt = self.tik.Scalar("int32", init_value=0)
        remain = self.tik.Scalar("int32", init_value=self.ele_per_line % block)
        dividend = self.tik.Scalar("int32", init_value=(self.ele_per_line - remain) // block)
        with self.tik.for_range(0, Constant.MAX_LOOP) as i:
            with self.tik.if_scope(dividend % Constant.SCALAR_TWO == 0):
                dividend.set_as(dividend // Constant.SCALAR_TWO)
                cnt.set_as(cnt + 1)
            with self.tik.else_scope():
                self.tik.tik_break()
        return cnt, dividend, remain

    def promote(self, ub_fp32, ub_fp16):
        """Conv input from low precision to high precision."""
        self._vconv(ub_fp32, ub_fp16, self.ele_per_iter, Constant.VEC_MASK_B32, 8, 4)

    def do_exp(self, dst):
        """Do exp calculation."""
        repeat_times = self.ele_per_iter // Constant.VEC_MASK_B32
        repeat = repeat_times // Constant.MAX_REPEAT
        repeat_remain = repeat_times % Constant.MAX_REPEAT
        remain = self.ele_per_iter % Constant.VEC_MASK_B32
        dst_offset = self.tik.Scalar("int32", init_value=0)
        with self.tik.if_scope(repeat_times > 0):
            with self.tik.if_scope(repeat > 0):
                with self.tik.for_range(0, repeat) as _:
                    self.tik.vexp(Constant.VEC_MASK_B32, dst[dst_offset], dst[dst_offset],
                                  Constant.MAX_REPEAT, 1, 1, 8, 8)
                    dst_offset.set_as(dst_offset + Constant.MAX_REPEAT * Constant.VEC_MASK_B32)
            with self.tik.if_scope(repeat_remain > 0):
                self.tik.vexp(Constant.VEC_MASK_B32, dst[dst_offset], dst[dst_offset], repeat_remain, 1, 1, 8, 8)
                dst_offset.set_as(dst_offset + repeat_remain * Constant.VEC_MASK_B32)
        with self.tik.if_scope(remain > 0):
            self.tik.vexp(remain, dst[dst_offset], dst[dst_offset], 1, 1, 1, 0, 0)

    def reduce_add_per_line(self, dst, src, temp, line):
        """Calculate the sum of each line."""
        block = Constant.VEC_MASK_B32
        src_rep_stride = self.tik.Scalar("int32", init_value=self.ele_per_line // Constant.NUM_BLOCK_B32)
        with self.tik.if_scope(self.ele_per_line > block):
            cnt, dividend, remain = self.cal_level(block)
            length = self.ele_per_line - remain
            offset = self.tik.Scalar("int32", init_value=0)
            with self.tik.for_range(0, line) as i:
                offset.set_as(self.ele_per_line * i)
                time = self.tik.Scalar("int32", init_value=Constant.SCALAR_TWO)
                with self.tik.if_scope(cnt > 0):
                    self.sum_line_with_cnt(temp, src, [offset, length, time, block, cnt, dividend, remain])
                with self.tik.else_scope():
                    self.sum_line_without_cnt(temp, src, [offset, block, dividend, remain])
            self._tik_vcadd(dst, temp, 0, 0, line, 1, 1, src_rep_stride, block)
        with self.tik.else_scope():
            self._tik_vcadd(dst, src, 0, 0, line, 1, 1, src_rep_stride, self.x_w)

    def sum_line_with_cnt(self, temp, src, params):
        """Sum the line when cnt grater than 0."""
        offset, length, time, block, cnt, dividend, remain = params
        self._vec_func("vadd", temp, src, src, offset,
                       offset + length // time, offset,
                       length // time, block)
        time.set_as(time * Constant.SCALAR_TWO)
        with self.tik.for_range(1, cnt) as _:
            self._vec_func("vadd", temp, temp, temp, offset,
                           offset + length // time, offset,
                           length // time, block)
            time.set_as(time * Constant.SCALAR_TWO)
        with self.tik.if_scope(dividend > 1):
            with self.tik.for_range(1, dividend) as i:
                self._vec_func("vadd", temp, temp, temp, offset + block * (dividend - 1 - i),
                               offset + block * (dividend - i),
                               offset + block * (dividend - 1 - i),
                               block, block)
        with self.tik.if_scope(remain > 1):
            self._vec_func("vadd", temp, src, temp, offset, offset + self.ele_per_line - remain,
                           offset, remain, block)

    def sum_line_without_cnt(self, temp, src, params):
        """Sum the line when cnt equal to 0."""
        offset, block, dividend, remain = params
        with self.tik.if_scope(dividend > 1):
            self._vec_func("vadd", temp, src, src, offset + block * (dividend - 2),
                           offset + block * (dividend - 1), offset + block * (dividend - 2), block, block)
            with self.tik.for_range(2, dividend) as i:
                self._vec_func("vadd", temp, temp, src, offset + block * (dividend - 1 - i),
                               offset + block * (dividend - i), offset + block * (dividend - 1 - i),
                               block, block)
            with self.tik.if_scope(remain > 1):
                self._vec_func("vadd", temp, src, temp, offset, offset + self.ele_per_line - remain,
                               offset, remain, block)
        with self.tik.else_scope():
            with self.tik.if_scope(remain > 1):
                self.tik.data_move(temp[offset], src[offset], 0, 1, Constant.NUM_BLOCK_B32, 0, 0)
                self._vec_func("vadd", temp, src, temp, offset, offset + self.ele_per_line - remain,
                               offset, remain, block)

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
        elif func_str == "vsub":
            func = self.tik.vdiv
        else:
            raise RuntimeError(f"[func_str] should be in ['vadd', 'vmul', 'vdiv', 'vsub'], but now is [{func_str}]")
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
    def _vconv(self, dst, src, ele_num, mask, dst_stride, src_stride, round_mode=''):
        """Api for TIK function vconv."""
        repeat_times = ele_num // mask
        remain_num = ele_num % mask
        offset = self.tik.Scalar("int32", init_value=0)
        with self.tik.if_scope(repeat_times > 0):
            max_repeat_times = repeat_times // Constant.MAX_REPEAT
            max_remain_times = repeat_times % Constant.MAX_REPEAT
            with self.tik.for_range(0, max_repeat_times) as _:
                self.tik.vconv(mask, round_mode, dst[offset], src[offset],
                               Constant.MAX_REPEAT, 1, 1, dst_stride, src_stride)
                offset.set_as(offset + mask * Constant.MAX_REPEAT)
            with self.tik.if_scope(max_remain_times > 0):
                self.tik.vconv(mask, round_mode, dst[offset], src[offset],
                               max_remain_times, 1, 1, dst_stride, src_stride)
                offset.set_as(offset + mask * max_remain_times)
        with self.tik.if_scope(remain_num > 0):
            self.tik.vconv(remain_num, round_mode, dst[offset], src[offset],
                           1, 1, 1, 0, 0)

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


class ScaledMaskedSoftmaxUnalignedND(BaseClass):
    """Implementation of unaligned ScaledMaskedSoftmax in ND format."""

    def __init__(self, op: "ScaledMaskedSoftmax"):
        """Init."""
        super().__init__(op)
        self.x_h.set_as(self.x_h1)
        self.x_w.set_as(self.x_w1)
        self.mask_h.set_as(self.mask_h1)
        self.mask_w.set_as(self.mask_w1)
        self.ele_per_line_b16 = self.tik.Scalar("int32")
        self.ele_per_line_mask = self.tik.Scalar("int32")
        self.padding_b8 = self.tik.Scalar("int32")
        self.move_brust_b8 = self.tik.Scalar("int32")
        self.move_brust_b16 = self.tik.Scalar("int32")
        self.move_brust_b32 = self.tik.Scalar("int32")
        self.available_line_per_iter = self.tik.Scalar("int32")
        self.max_byte_per_line = self.tik.Scalar("int32")
        self.broad_mode = self.tik.Scalar("int32")
        self.line_last_core = self.tik.Scalar("int32")
        self.line_last_core_per_iter = self.tik.Scalar("int32")
        self.line_last_core_last_iter = self.tik.Scalar("int32")
        self.line_last_iter = self.tik.Scalar("int32")
        self.line_per_batch = self.tik.Scalar("int32")
        self.ele_per_channel = self.tik.Scalar("int32")

    def gen_triu_mask(self):
        """Generate triu mask tensor."""
        move_burst = self.ele_per_line_b16 // Constant.NUM_BLOCK_B16
        dup_len = self.tik.Scalar("int32", init_value=1)
        offset = self.tik.Scalar("int32", init_value=0)
        triu_mask_ub = self.tik.Tensor('float16', (self.ele_per_line_b16,), tik.scope_ubuf, "triu_mask_ub")
        self._vec_dup(triu_mask_ub, 1.0, 0, self.ele_per_line_b16, Constant.VEC_MASK_B16)
        with self.tik.for_range(0, self.mask_h) as i:
            with self.tik.if_scope(dup_len <= self.mask_w):
                self._vec_dup(triu_mask_ub, 0.0, 0, dup_len, Constant.VEC_MASK_B16)
                dup_len.set_as(dup_len + 1)
            self.tik.data_move(self.fixed_mask_gm[offset], triu_mask_ub, 0, 1, move_burst, 0, 0)
            offset.set_as(offset + self.ele_per_line_b16)

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
        if self.dtype in ("float16", "bfloat16"):
            self.ele_per_line.set_as(self.ceil_div(self.x_w, Constant.NUM_BLOCK_B16) * Constant.NUM_BLOCK_B16)
            self.move_brust_b16.set_as(self.x_w * Constant.BYTE_B16)
        else:
            self.ele_per_line.set_as(self.ceil_div(self.x_w, Constant.NUM_BLOCK_B32) * Constant.NUM_BLOCK_B32)
            self.move_brust_b32.set_as(self.x_w * Constant.BYTE_B32)
        self.ele_per_line_b16.set_as(self.ceil_div(self.x_w, Constant.NUM_BLOCK_B16) * Constant.NUM_BLOCK_B16)
        self.ele_per_line_mask.set_as(self.ceil_div(self.mask_w, Constant.NUM_BLOCK_B8) * Constant.NUM_BLOCK_B8)
        self.ele_per_channel.set_as(self.x_h * self.x_w)
        self.total_line.set_as(self.x_n * self.x_c * self.x_h)
        self.max_byte_per_line.set_as(self.ele_per_line * Constant.MAX_B16_UB_NUM * 2)
        self.available_line_per_iter = self.ceil_div(self.available_ub_space, self.max_byte_per_line)

        self.line_per_batch.set_as(self.x_c * self.x_h)
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
        with self.tik.if_scope(tik.all(self.mask_n == self.x_n, self.mask_c == self.x_c)):
            self.broad_mode.set_as(1)
        with self.tik.elif_scope(tik.all(self.mask_n == 1, self.mask_c == 1)):
            self.broad_mode.set_as(4)
        with self.tik.elif_scope(tik.all(self.mask_n == self.x_n, self.mask_c < self.x_c)):
            self.broad_mode.set_as(2)
        with self.tik.elif_scope(tik.all(self.mask_n < self.x_n, self.mask_c == self.x_c)):
            self.broad_mode.set_as(3)

    def do_compute(self, core_idx, iter_curr_core, line_per_iter, line_last_iter):
        """Do compute in different dtype."""
        if self.dtype == "float16":
            self.compute_fp16(core_idx, iter_curr_core, line_per_iter, line_last_iter)
        else:
            self.compute_fp32_bf16(core_idx, iter_curr_core, line_per_iter, line_last_iter)

    def compute_fp16(self, core_idx, iter_curr_core, line_per_iter, line_last_iter):
        """Implementation of fused scaled masked softmax algorithm in fp16."""
        line = self.tik.Scalar("int32")
        offset = self.tik.Scalar("int32")
        offset_mask = self.tik.Scalar("int32")
        ub_fp16 = self.tik.Tensor("float16", (line_per_iter, self.ele_per_line), tik.scope_ubuf, "ub_fp16")
        ub_fp32 = self.tik.Tensor("float32", (line_per_iter, self.ele_per_line), tik.scope_ubuf, "ub_fp32")
        with self.tik.for_range(0, iter_curr_core) as it:
            with self.tik.if_scope(it < iter_curr_core - 1):
                line.set_as(line_per_iter)
            with self.tik.else_scope():
                line.set_as(line_last_iter)
            offset_params = self.cal_offset(core_idx, it, line_per_iter, offset, offset_mask)
            shape = [line, self.ele_per_line]
            ele_num = line * self.ele_per_line
            with self.tik.new_stmt_scope():
                ub_temp_fp16 = self.tik.Tensor("float16", shape, tik.scope_ubuf, "ub_temp_fp16")
                ub_mask_fp16 = self.tik.Tensor("float16", shape, tik.scope_ubuf, "ub_mask_fp16")
                self.move_mask_in(ub_mask_fp16, offset_mask, line, offset_params)
                self.process_mask(ub_mask_fp16, ub_temp_fp16, line, ele_num)
                self.move_x_in(ub_fp16, offset, line, ele_num)
                self.scale_x(ub_fp16, ele_num)
                self.masked_fill(ub_fp16, ub_mask_fp16, ub_temp_fp16, line, ele_num)
                self.smooth(ub_fp16, ub_temp_fp16, line)
            self.promote(ub_fp32, ub_fp16, ele_num)
            self.do_exp(ub_fp32, ele_num)
            ub_temp_fp32 = self.tik.Tensor("float32", shape, tik.scope_ubuf, "ub_temp_fp32")
            self.softmax(ub_fp32, ub_temp_fp32, line)
            self.move_out(ub_fp16, ub_fp32, offset, line, ele_num)

    def compute_fp32_bf16(self, core_idx, iter_curr_core, line_per_iter, line_last_iter):
        """Implementation of fused scaled masked softmax algorithm in fp32 or bf16."""
        line = self.tik.Scalar("int32")
        offset = self.tik.Scalar("int32")
        offset_mask = self.tik.Scalar("int32")
        ub_temp_fp16 = self.tik.Tensor("float16", (line_per_iter, self.ele_per_line_b16),
                                       tik.scope_ubuf, "ub_temp_fp16")
        ub_temp_fp32 = self.tik.Tensor("float32", (line_per_iter, self.ele_per_line), tik.scope_ubuf, "ub_temp_fp32")
        with self.tik.for_range(0, iter_curr_core) as it:
            with self.tik.if_scope(it < iter_curr_core - 1):
                line.set_as(line_per_iter)
            with self.tik.else_scope():
                line.set_as(line_last_iter)
            offset_params = self.cal_offset(core_idx, it, line_per_iter, offset, offset_mask)
            ele_num = line * self.ele_per_line
            self.move_mask_in(ub_temp_fp16, offset_mask, line, offset_params)
            self.process_mask(ub_temp_fp16, ub_temp_fp32, line, ele_num)
            ub_fp32 = self.tik.Tensor("float32", (line_per_iter, self.ele_per_line), tik.scope_ubuf, "ub_fp32")
            self.move_x_in(ub_fp32, offset, line, ele_num)
            self.scale_x(ub_fp32, ele_num)
            self.masked_fill(ub_fp32, ub_temp_fp32, ub_temp_fp16, line, ele_num)
            self.smooth(ub_fp32, ub_temp_fp32, line)
            self.do_exp(ub_fp32, ele_num)
            self.softmax(ub_fp32, ub_temp_fp32, line)
            self.move_out(ub_temp_fp16, ub_fp32, offset, line, ele_num)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def cal_offset(self, core_idx, it, line_per_iter, offset, offset_mask):
        """Calculate the offset of inputs."""
        curr_line = core_idx * self.line_per_core + it * line_per_iter
        offset.set_as(curr_line * self.x_w)
        curr_batch, curr_channel, mask_batch, mask_channel, line_in_channel = self.line_to_nc(curr_line)
        offset_mask.set_as(((mask_batch * self.mask_c + mask_channel) * self.mask_h + line_in_channel) * self.mask_w)
        return [curr_line, curr_batch, curr_channel, mask_batch, mask_channel, line_in_channel]

    def line_to_nc(self, curr_line):
        """Claculate curr batch and channel based on curr line."""
        curr_batch = curr_line // self.line_per_batch
        curr_channel = curr_line % self.line_per_batch // self.x_h
        mask_batch = curr_batch // self.broad_ratio_n
        mask_channel = curr_channel // self.broad_ratio_c
        line_in_channel = curr_line - (curr_batch * self.x_c + curr_channel) * self.x_h
        return [curr_batch, curr_channel, mask_batch, mask_channel, line_in_channel]

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
                offset_temp = ((s_batch * self.mask_c + s_channel) * self.mask_h + s_line) * self.ele_per_line_b16
                burst = line * self.ele_per_line_b16 // Constant.NUM_BLOCK_B16
                self.tik.data_move(dst, self.fixed_mask_gm[offset_temp], 0, 1, burst, 0, 0)
            else:
                shape = [line, self.ele_per_line_mask]
                ub_mask = self.tik.Tensor("uint8", shape, scope=tbe_platform.scope_ubuf, name="ub_mask")
                self.tik.data_move_pad(ub_mask, self.mask_gm[offset], line, self.move_brust_b8, 0, 0,
                                       right_padding=self.padding_b8, padding_value=Constant.PADDING_VALUE_MASK)
                self.align_mask(dst, ub_mask, line, self.ele_per_line_b16,
                                self.ele_per_line_mask, Constant.VEC_MASK_B16)
        with self.tik.elif_scope(self.broad_mode == 2):
            self.move_mask_mode_2(dst, offset, line, offset_params)
        with self.tik.elif_scope(self.broad_mode == 3):
            self.move_mask_mode_3(dst, offset, line, offset_params)
        with self.tik.else_scope():
            self.move_mask_mode_4(dst, offset, line, offset_params)

    def move_mask_mode_2(self, dst, offset, line, offset_params):
        """Move mask from gm to ub in mode 2."""
        shape = [line, self.ele_per_line_mask]
        ub_mask = self.tik.Tensor("uint8", shape, scope=tbe_platform.scope_ubuf, name="ub_mask")
        offset_temp = self.tik.Scalar("int32", init_value=0)
        s_line, s_batch, s_channel, _, _, s_line_in_channel = offset_params
        e_line = s_line + line
        e_batch, e_channel, _, _, e_line_in_channel = self.line_to_nc(e_line)
        with self.tik.if_scope(e_batch - s_batch == 0):
            with self.tik.if_scope(e_channel - s_channel == 0):
                self.move_mask_in_line(ub_mask, offset_temp, offset, line)
            with self.tik.else_scope():
                self.move_mask_in_line(ub_mask, offset_temp, offset, self.mask_h - s_line_in_channel)
                offset_temp.set_as((self.mask_h - s_line_in_channel) * self.ele_per_line_mask)
                with self.tik.for_range(0, e_channel - s_channel - 1) as _:
                    self.move_mask_in_line(ub_mask, offset_temp, s_batch * self.ele_per_channel, self.mask_h)
                    offset_temp.set_as(offset_temp + self.ele_per_line_mask * self.mask_h)
                with self.tik.if_scope(e_line_in_channel > 0):
                    self.move_mask_in_line(ub_mask, offset_temp, s_batch * self.ele_per_channel, e_line_in_channel)
        with self.tik.else_scope():
            self.move_mask_in_line(ub_mask, offset_temp, offset, self.mask_h - s_line_in_channel)
            offset_temp.set_as((self.mask_h - s_line_in_channel) * self.ele_per_line_mask)
            with self.tik.for_range(0, self.x_c - s_channel - 1) as _:
                self.move_mask_in_line(ub_mask, offset_temp, s_batch * self.ele_per_channel, self.mask_h)
                offset_temp.set_as(offset_temp + self.ele_per_line_mask * self.mask_h)
            with self.tik.for_range(1, e_batch - s_batch) as i:
                offset_gm = (s_batch + i) * self.ele_per_channel
                with self.tik.for_range(0, self.x_c) as _:
                    self.move_mask_in_line(ub_mask, offset_temp, offset_gm, self.mask_h)
                    offset_temp.set_as(offset_temp + self.ele_per_line_mask * self.mask_h)
            with self.tik.if_scope(e_channel > 0):
                with self.tik.for_range(0, e_channel) as _:
                    self.move_mask_in_line(ub_mask, offset_temp, e_batch * self.ele_per_channel, self.mask_h)
                    offset_temp.set_as(offset_temp + self.ele_per_line_mask * self.mask_h)
            with self.tik.if_scope(e_line_in_channel > 0):
                self.move_mask_in_line(ub_mask, offset_temp, e_batch * self.ele_per_channel, e_line_in_channel)
        self.align_mask(dst, ub_mask, line, self.ele_per_line_b16,
                        self.ele_per_line_mask, Constant.VEC_MASK_B16)

    def move_mask_mode_3(self, dst, offset, line, offset_params):
        """Move mask from gm to ub in mode 3."""
        shape = [line, self.ele_per_line_mask]
        ub_mask = self.tik.Tensor("uint8", shape, scope=tbe_platform.scope_ubuf, name="ub_mask")
        offset_temp = self.tik.Scalar("int32", init_value=0)
        s_line, s_batch, s_channel, _, _, s_line_in_channel = offset_params
        e_line = s_line + line
        e_batch, e_channel, _, _, e_line_in_channel = self.line_to_nc(e_line)
        with self.tik.if_scope(e_batch - s_batch == 0):
            self.move_mask_in_line(ub_mask, offset_temp, offset, line)
        with self.tik.else_scope():
            line_temp = self.mask_h * self.mask_c - s_channel * self.mask_h - s_line_in_channel
            self.move_mask_in_line(ub_mask, offset_temp, offset, line_temp)
            offset_temp.set_as(line_temp * self.ele_per_line_mask)
            with self.tik.for_range(1, e_batch - s_batch) as i:
                self.move_mask_in_line(ub_mask, offset_temp, 0, self.mask_h * self.mask_c)
                offset_temp.set_as(offset_temp + self.ele_per_line_mask * self.mask_h * self.mask_c)
            line_temp = e_channel * self.mask_h + e_line_in_channel
            with self.tik.if_scope(line_temp > 0):
                self.move_mask_in_line(ub_mask, offset_temp, 0, line_temp)
        self.align_mask(dst, ub_mask, line, self.ele_per_line_b16,
                        self.ele_per_line_mask, Constant.VEC_MASK_B16)

    def move_mask_mode_4(self, dst, offset, line, offset_params):
        """Move mask from gm to ub in mode 4."""
        offset_temp = self.tik.Scalar("int32", init_value=0)
        s_line, s_batch, s_channel, mask_batch, mask_channel, s_line_in_channel = offset_params
        e_line = s_line + line
        e_batch, e_channel, _, _, e_line_in_channel = self.line_to_nc(e_line)
        s_nc = s_batch * self.x_c + s_channel
        e_nc = e_batch * self.x_c + e_channel
        if self.fixed_triu_mask:
            offset_mask = ((mask_batch * self.mask_c + mask_channel) *
                           self.mask_h + s_line_in_channel) * self.ele_per_line_b16
            with self.tik.if_scope(e_nc - s_nc == 0):
                burst = line * self.ele_per_line_b16 // Constant.NUM_BLOCK_B16
                self.tik.data_move(dst, self.fixed_mask_gm[offset_mask], 0, 1, burst, 0, 0)
            with self.tik.else_scope():
                burst = (self.mask_h - s_line_in_channel) * self.ele_per_line_b16 // Constant.NUM_BLOCK_B16
                self.tik.data_move(dst, self.fixed_mask_gm[offset_mask], 0, 1, burst, 0, 0)
                offset_temp.set_as((self.mask_h - s_line_in_channel) * self.ele_per_line_b16)
                with self.tik.for_range(1, e_nc - s_nc) as _:
                    burst = self.mask_h * self.ele_per_line_b16 // Constant.NUM_BLOCK_B16
                    self.tik.data_move(dst[offset_temp], self.fixed_mask_gm, 0, 1, burst, 0, 0)
                    offset_temp.set_as(offset_temp + self.mask_h * self.ele_per_line_b16)
                with self.tik.if_scope(e_line_in_channel > 0):
                    burst = e_line_in_channel * self.ele_per_line_b16 // Constant.NUM_BLOCK_B16
                    self.tik.data_move(dst[offset_temp], self.fixed_mask_gm, 0, 1, burst, 0, 0)
        else:
            shape = [line, self.ele_per_line_mask]
            ub_mask = self.tik.Tensor("uint8", shape, scope=tbe_platform.scope_ubuf, name="ub_mask")
            with self.tik.if_scope(e_nc - s_nc == 0):
                self.move_mask_in_line(ub_mask, offset_temp, offset, line)
            with self.tik.else_scope():
                self.move_mask_in_line(ub_mask, offset_temp, offset, self.mask_h - s_line_in_channel)
                offset_temp.set_as((self.mask_h - s_line_in_channel) * self.ele_per_line_mask)
                with self.tik.for_range(1, e_nc - s_nc) as _:
                    self.move_mask_in_line(ub_mask, offset_temp, 0, self.mask_h)
                    offset_temp.set_as(offset_temp + self.mask_h * self.ele_per_line_mask)
                with self.tik.if_scope(e_line_in_channel > 0):
                    self.move_mask_in_line(ub_mask, offset_temp, 0, e_line_in_channel)
            self.align_mask(dst, ub_mask, line, self.ele_per_line_b16,
                            self.ele_per_line_mask, Constant.VEC_MASK_B16)

    def move_mask_in_line(self, dst, dst_offset, src_offset, line):
        """Move mask from gm to ub in lines."""
        self.tik.data_move_pad(dst[dst_offset], self.mask_gm[src_offset], line, self.move_brust_b8,
                               0, 0, right_padding=self.padding_b8, padding_value=Constant.PADDING_VALUE_MASK)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def align_mask(self, dst, src, line, ele_dst, ele_src, mask):
        """Align uint8 mask to x."""
        dst_offset = self.tik.Scalar("int32", init_value=0)
        src_offset = self.tik.Scalar("int32", init_value=0)
        with self.tik.for_range(0, line) as _:
            self._vconv(dst[dst_offset], src[src_offset], ele_dst, mask, 8, 4)
            dst_offset.set_as(dst_offset + ele_dst)
            src_offset.set_as(src_offset + ele_src)

    def process_mask(self, ub_mask, ub_temp, line, ele_num):
        """
        Convert the upper triangular matrix to the lower triangular matrix, and fill False with -10000.
        """
        if self.dtype == "float16":
            self._vec_scalar_func("vmuls", ub_temp, ub_mask, self.tik.Scalar("float16", init_value=-10000), 0, 0,
                                  ele_num, Constant.VEC_MASK_B16)
            self._vec_scalar_func("vmuls", ub_mask, ub_mask, self.tik.Scalar("float16", init_value=-1), 0, 0,
                                  ele_num, Constant.VEC_MASK_B16)
            self._vec_scalar_func("vadds", ub_mask, ub_mask, self.tik.Scalar("float16", init_value=1), 0, 0,
                                  ele_num, Constant.VEC_MASK_B16)
        else:
            self.align_mask(ub_temp, ub_mask, line, self.ele_per_line, self.ele_per_line_b16, Constant.VEC_MASK_B32)
            self._vec_scalar_func("vmuls", ub_mask, ub_mask, self.tik.Scalar("float16", init_value=-10000), 0, 0,
                                  line * self.ele_per_line_b16, Constant.VEC_MASK_B16)
            self._vec_scalar_func("vmuls", ub_temp, ub_temp, self.tik.Scalar("float32", init_value=-1), 0, 0,
                                  ele_num, Constant.VEC_MASK_B32)
            self._vec_scalar_func("vadds", ub_temp, ub_temp, self.tik.Scalar("float32", init_value=1), 0, 0,
                                  ele_num, Constant.VEC_MASK_B32)

    def move_x_in(self, dst, offset, line, ele_num):
        """Move x from gm to ub tensor."""
        if self.dtype == "float16":
            self.tik.data_move_pad(dst, self.x_gm[offset], line, self.move_brust_b16, 0, 0)
        elif self.dtype == "bfloat16":
            with self.tik.new_stmt_scope():
                shape = [line, self.ele_per_line]
                ub_bf16 = self.tik.Tensor("bfloat16", shape, tik.scope_ubuf, "ub_bf16")
                self.tik.data_move_pad(ub_bf16, self.x_gm[offset], line, self.move_brust_b16, 0, 0)
                self._vconv(dst, ub_bf16, ele_num, Constant.VEC_MASK_B32, 8, 4)
        else:
            self.tik.data_move_pad(dst, self.x_gm[offset], line, self.move_brust_b32, 0, 0)

    def scale_x(self, src, ele_num):
        """Scale input."""
        if self.dtype == "float16":
            self._vec_scalar_func("vmuls", src, src, self.scale_fp16, 0, 0, ele_num, Constant.VEC_MASK_B16)
        else:
            self._vec_scalar_func("vmuls", src, src, self.scale, 0, 0, ele_num, Constant.VEC_MASK_B32)

    def masked_fill(self, dst, ub_mask, ub_temp, line, ele_num):
        """Do masked_fill on input."""
        if self.dtype == "float16":
            self._vec_func("vmul", dst, dst, ub_mask, 0, 0, 0, ele_num, Constant.VEC_MASK_B16)
            self._vec_func("vadd", dst, dst, ub_temp, 0, 0, 0, ele_num, Constant.VEC_MASK_B16)
        else:
            self._vec_func("vmul", dst, dst, ub_mask, 0, 0, 0, ele_num, Constant.VEC_MASK_B32)
            self.align_mask(ub_mask, ub_temp, line, self.ele_per_line, self.ele_per_line_b16, Constant.VEC_MASK_B32)
            self._vec_func("vadd", dst, dst, ub_mask, 0, 0, 0, ele_num, Constant.VEC_MASK_B32)

    def smooth(self, ub_x, ub_temp, line):
        """Use the maximum vcalue of x to smooth the input."""
        if self.dtype == "float16":
            block = Constant.NUM_BLOCK_B16
            mask = Constant.VEC_MASK_B16
            dtype = "float16"
        else:
            block = Constant.NUM_BLOCK_B32
            mask = Constant.VEC_MASK_B32
            dtype = "float32"
        ub_reducemax = self.tik.Tensor(dtype, (self.ceil_div(line, block) * block,),
                                       scope=tbe_platform.scope_ubuf, name="ub_reducemax")
        cnt, remain, _ = self.cal_level(block)
        move_stride = self.tik.Scalar("int32", init_value=(self.ele_per_line - block) // block)
        offset = self.tik.Scalar("int32")
        with self.tik.if_scope(tik.all(cnt == 0, remain == 1)):
            self.tik.vcgmax(mask, ub_reducemax, ub_x,
                            self.ceil_div(line, Constant.SCALAR_EIGHT), 1, 1, 8)
        with self.tik.else_scope():
            with self.tik.for_range(0, line) as i:
                offset.set_as(self.ele_per_line * i)
                self.reduce_max_per_line(ub_x, ub_temp, offset, cnt, remain, mask, block)
            with self.tik.if_scope(tik.all(move_stride > 0, line > 1)):
                self.tik.data_move(ub_temp[block], ub_temp[self.ele_per_line], 0, line - 1, 1, move_stride, 0)
            self.tik.vcgmax(mask, ub_reducemax, ub_temp,
                            self.ceil_div(line, Constant.SCALAR_EIGHT), 1, 1, 8)
        self._vec_scalar_func("vmuls", ub_reducemax, ub_reducemax, -1.0, 0, 0, line, mask)

        stride = self.ele_per_line // block
        with self.tik.if_scope(tik.any(stride > Constant.MAX_REPEAT, line < Constant.SCALAR_EIGHT)):
            maximum = self.tik.Scalar(dtype, name="maximum")
            with self.tik.for_range(0, line) as i:
                maximum.set_as(ub_reducemax[i])
                offset.set_as(self.ele_per_line * i)
                self._vec_scalar_func("vadds", ub_x, ub_x, maximum, offset, offset, self.ele_per_line, mask)
        with self.tik.else_scope():
            self.smooth_with_vbcb(ub_x, ub_temp, ub_reducemax, line, stride, block, mask)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def smooth_with_vbcb(self, ub_x, ub_temp, ub_reducemax, line, stride, block, mask):
        """Input sub maximum with TIK instruction vbcb."""
        repeat_times = self.ceil_div(line, Constant.SCALAR_EIGHT)
        repeat = line // Constant.MAX_REPEAT
        repeat_remain = line % Constant.MAX_REPEAT
        offset_dst = self.tik.Scalar("int32", init_value=0)
        offset_src = self.tik.Scalar("int32", init_value=0)
        buffer = self.tik.Scalar("int32", init_value=1)
        with self.tik.if_scope(self.ele_per_line == block):
            with self.tik.if_scope(line % Constant.SCALAR_EIGHT > 0):
                if self.dtype == "float16":
                    with self.tik.if_scope(repeat_times % 2 == 0):
                        buffer.set_as(2)
                calc_num = (repeat_times - buffer) * Constant.SCALAR_EIGHT * self.ele_per_line
                remain_num = (line - (repeat_times - buffer) * Constant.SCALAR_EIGHT) * self.ele_per_line
                self.tik.vbcb(ub_temp, ub_reducemax, repeat_times - buffer, 1, 8)
                self._vec_func("vadd", ub_x, ub_x, ub_temp, 0, 0, 0, calc_num, mask)
                self.tik.vbcb(ub_temp, ub_reducemax[(repeat_times - buffer) * Constant.SCALAR_EIGHT],
                                repeat_times, 1, 8)
                self._vec_func("vadd", ub_x, ub_x, ub_temp, calc_num, calc_num, 0, remain_num, mask)
            with self.tik.else_scope():
                self.tik.vbcb(ub_temp, ub_reducemax, repeat_times, 1, 8)
                self._vec_func("vadd", ub_x, ub_x, ub_temp, 0, 0, 0, line * self.ele_per_line, mask)
        with self.tik.elif_scope(self.ele_per_line <= mask):
            self.tik.vbcb(ub_temp, ub_reducemax, repeat_times, 1, 8)
            with self.tik.for_range(0, repeat) as _:
                self.tik.vadd(self.ele_per_line, ub_x[offset_dst], ub_x[offset_dst], ub_temp[offset_src],
                              Constant.MAX_REPEAT, 1, 1, 0, stride, stride, 1)
                offset_dst.set_as(offset_dst + self.ele_per_line * Constant.MAX_REPEAT)
                offset_src.set_as(offset_src + block * Constant.MAX_REPEAT)
            with self.tik.if_scope(repeat_remain > 0):
                self.tik.vadd(self.ele_per_line, ub_x[offset_dst], ub_x[offset_dst], ub_temp[offset_src],
                              repeat_remain, 1, 1, 0, stride, stride, 1)
        with self.tik.else_scope():
            self.tik.vbcb(ub_temp, ub_reducemax, repeat_times, 1, 8)
            repeat_times = self.ele_per_line // mask
            remain_ele = self.ele_per_line % mask
            with self.tik.for_range(0, repeat_times) as _:
                with self.tik.for_range(0, repeat) as _:
                    self.tik.vadd(mask, ub_x[offset_dst], ub_x[offset_dst], ub_temp[offset_src],
                                  Constant.MAX_REPEAT, 1, 1, 0, stride, stride, 1)
                    offset_dst.set_as(offset_dst + mask * Constant.MAX_REPEAT)
                    offset_src.set_as(offset_src + block * Constant.MAX_REPEAT)
                with self.tik.if_scope(repeat_remain > 0):
                    self.tik.vadd(mask, ub_x[offset_dst], ub_x[offset_dst], ub_temp[offset_src],
                                  repeat_remain, 1, 1, 0, stride, stride, 1)
                offset_dst.set_as(offset_dst + mask)
                offset_src.set_as(0)
            self.tik.vadd(remain_ele, ub_x[offset_dst], ub_x[offset_dst], ub_temp[offset_src],
                          repeat_remain, 1, 1, 0, stride, stride, 1)

    def promote(self, ub_fp32, ub_fp16, ele_num):
        """Conv input from low precision to high precision."""
        self._vconv(ub_fp32, ub_fp16, ele_num, Constant.VEC_MASK_B32, 8, 4)

    def do_exp(self, dst, ele_num):
        """Do exp calculation."""
        repeat_times = ele_num // Constant.VEC_MASK_B32
        repeat = repeat_times // Constant.MAX_REPEAT
        repeat_remain = repeat_times % Constant.MAX_REPEAT
        remain = ele_num % Constant.VEC_MASK_B32
        dst_offset = self.tik.Scalar("int32", init_value=0)
        with self.tik.if_scope(repeat_times > 0):
            with self.tik.if_scope(repeat > 0):
                with self.tik.for_range(0, repeat) as _:
                    self.tik.vexp(Constant.VEC_MASK_B32, dst[dst_offset], dst[dst_offset],
                                  Constant.MAX_REPEAT, 1, 1, 8, 8)
                    dst_offset.set_as(dst_offset + Constant.MAX_REPEAT * Constant.VEC_MASK_B32)
            with self.tik.if_scope(repeat_remain > 0):
                self.tik.vexp(Constant.VEC_MASK_B32, dst[dst_offset], dst[dst_offset], repeat_remain, 1, 1, 8, 8)
                dst_offset.set_as(dst_offset + repeat_remain * Constant.VEC_MASK_B32)
        with self.tik.if_scope(remain > 0):
            self.tik.vexp(remain, dst[dst_offset], dst[dst_offset], 1, 1, 1, 0, 0)

    def softmax(self, ub_fp32, ub_temp, line):
        """Calculate softmax of input."""
        with self.tik.new_stmt_scope():
            block = Constant.NUM_BLOCK_B32
            mask = Constant.VEC_MASK_B32
            ub_reduceadd = self.tik.Tensor("float32", (self.ceil_div(line, block) * block,),
                                           scope=tbe_platform.scope_ubuf, name="ub_reduceadd")
            one_ub = self.tik.Tensor("float32", (line,), scope=tbe_platform.scope_ubuf, name="one_ub")

            self.reduce_add_per_line(ub_reduceadd, ub_fp32, ub_temp, line)
            self._vec_dup(one_ub, 1.0, 0, line, mask)
            self._vec_func("vdiv", ub_reduceadd, one_ub, ub_reduceadd, 0, 0, 0, line, mask)

            stride = self.ele_per_line // block
            with self.tik.if_scope(tik.any(stride > Constant.MAX_REPEAT, line < Constant.SCALAR_EIGHT)):
                add_sum = self.tik.Scalar("float32", name="add_sum")
                with self.tik.for_range(0, line) as i:
                    add_sum.set_as(ub_reduceadd[i])
                    self._vec_scalar_func("vmuls", ub_fp32, ub_fp32, add_sum, self.ele_per_line * i,
                                          self.ele_per_line * i, self.ele_per_line, mask)
            with self.tik.else_scope():
                self.softmax_with_vbcb(ub_fp32, ub_temp, ub_reduceadd, line, stride, block, mask)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def softmax_with_vbcb(self, ub_fp32, ub_temp, ub_reduceadd, line, stride, block, mask):
        """Exp div sum exp with TIK instruction vbcb."""
        repeat_times = self.ceil_div(line, Constant.SCALAR_EIGHT)
        repeat = line // Constant.MAX_REPEAT
        repeat_remain = line % Constant.MAX_REPEAT
        offset_dst = self.tik.Scalar("int32", init_value=0)
        offset_src = self.tik.Scalar("int32", init_value=0)

        with self.tik.if_scope(self.ele_per_line == block):
            with self.tik.if_scope(line % Constant.SCALAR_EIGHT > 0):
                calc_num = (repeat_times - 1) * Constant.SCALAR_EIGHT * self.ele_per_line
                remain_num = (line - (repeat_times - 1) * Constant.SCALAR_EIGHT) * self.ele_per_line
                self.tik.vbcb(ub_temp, ub_reduceadd, repeat_times - 1, 1, 8)
                self._vec_func("vmul", ub_fp32, ub_fp32, ub_temp, 0, 0, 0, calc_num, mask)
                self.tik.vbcb(ub_temp, ub_reduceadd[(repeat_times - 1) * Constant.SCALAR_EIGHT], 1, 1, 8)
                self._vec_func("vmul", ub_fp32, ub_fp32, ub_temp, calc_num, calc_num, 0, remain_num, mask)
            with self.tik.else_scope():
                self.tik.vbcb(ub_temp, ub_reduceadd, repeat_times, 1, 8)
                self._vec_func("vmul", ub_fp32, ub_fp32, ub_temp, 0, 0, 0, line * self.ele_per_line, mask)
        with self.tik.elif_scope(self.ele_per_line <= mask):
            self.tik.vbcb(ub_temp, ub_reduceadd, repeat_times, 1, 8)
            with self.tik.for_range(0, repeat) as _:
                self.tik.vmul(self.ele_per_line, ub_fp32[offset_dst], ub_fp32[offset_dst], ub_temp[offset_src],
                              Constant.MAX_REPEAT, 1, 1, 0, self.ele_per_line // block, self.ele_per_line // block, 1)
                offset_dst.set_as(offset_dst + self.ele_per_line * Constant.MAX_REPEAT)
                offset_src.set_as(offset_src + block * Constant.MAX_REPEAT)
            with self.tik.if_scope(repeat_remain > 0):
                self.tik.vmul(self.ele_per_line, ub_fp32[offset_dst], ub_fp32[offset_dst], ub_temp[offset_src],
                              repeat_remain, 1, 1, 0, self.ele_per_line // block, self.ele_per_line // block, 1)
        with self.tik.else_scope():
            self.tik.vbcb(ub_temp, ub_reduceadd, repeat_times, 1, 8)
            repeat_times = self.ele_per_line // mask
            remain_ele = self.ele_per_line % mask
            with self.tik.for_range(0, repeat_times) as _:
                with self.tik.for_range(0, repeat) as _:
                    self.tik.vmul(self.ele_per_line, ub_fp32[offset_dst], ub_fp32[offset_dst], ub_temp[offset_src],
                                  Constant.MAX_REPEAT, 1, 1, 0, stride, stride, 1)
                    offset_dst.set_as(offset_dst + self.ele_per_line * Constant.MAX_REPEAT)
                    offset_src.set_as(offset_src + block * Constant.MAX_REPEAT)
                with self.tik.if_scope(repeat_remain > 0):
                    self.tik.vmul(self.ele_per_line, ub_fp32[offset_dst], ub_fp32[offset_dst], ub_temp[offset_src],
                                  repeat_remain, 1, 1, 0, stride, stride, 1)
                offset_dst.set_as(offset_dst + mask)
                offset_src.set_as(0)
            self.tik.vmul(remain_ele, ub_fp32[offset_dst], ub_fp32[offset_dst], ub_temp[offset_src],
                          repeat_remain, 1, 1, 0, stride, stride, 1)

    def move_out(self, ub_fp16, ub_fp32, offset, line, ele_num):
        """Move result from ub to gm."""
        if self.dtype == "float16":
            self._vconv(ub_fp16, ub_fp32, ele_num, Constant.VEC_MASK_B32, 4, 8)
            self.tik.data_move_pad(self.y_gm[offset], ub_fp16, line, self.move_brust_b16, 0, 0)
        elif self.dtype == "float32":
            self.tik.data_move_pad(self.y_gm[offset], ub_fp32, line, self.move_brust_b32, 0, 0)
        else:
            ub_bf16 = ub_fp16.reinterpret_cast_to("bfloat16")
            self._vconv(ub_bf16, ub_fp32, ele_num, Constant.VEC_MASK_B32, 4, 8, 'round')
            self.tik.data_move_pad(self.y_gm[offset], ub_bf16, line, self.move_brust_b16, 0, 0)


class ScaledMaskedSoftmaxAlignedND(BaseClass):
    """Implementation of aligned ScaledMaskedSoftmax in ND format."""

    def __init__(self, op: "ScaledMaskedSoftmax"):
        """Init."""
        super().__init__(op)
        self.x_h.set_as(self.x_h1)
        self.x_w.set_as(self.x_w1)
        self.mask_h.set_as(self.mask_h1)
        self.mask_w.set_as(self.mask_w1)
        self.shape = (self.line_per_iter, self.x_w)

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
        """Do compute in different dtype."""
        if self.dtype == "float16":
            self.compute_fp16(core_idx, iter_curr_core)
        else:
            self.compute_fp32_bf16(core_idx, iter_curr_core)

    def compute_fp16(self, core_idx, iter_curr_core):
        """Implementation of fused scaled masked softmax algorithm in fp16."""
        offset = self.tik.Scalar("int32")
        offset_mask = self.tik.Scalar("int32")
        ub_fp16 = self.tik.Tensor("float16", self.shape, tik.scope_ubuf, "ub_fp16")
        ub_fp32 = self.tik.Tensor("float32", self.shape, tik.scope_ubuf, "ub_fp32")
        with self.tik.for_range(0, iter_curr_core) as it:
            self.cal_offset(core_idx, it, offset, offset_mask, self.ele_per_iter)
            with self.tik.new_stmt_scope():
                ub_temp_fp16 = self.tik.Tensor("float16", self.shape, tik.scope_ubuf, "ub_temp_fp16")
                ub_mask_fp16 = self.tik.Tensor("float16", self.shape, tik.scope_ubuf, "ub_mask_fp16")
                self.move_mask_in(ub_mask_fp16, offset_mask)
                self.process_mask(ub_mask_fp16, ub_temp_fp16)
                self.move_x_in(ub_fp16, offset)
                self.scale_x(ub_fp16)
                self.masked_fill(ub_fp16, ub_mask_fp16, ub_temp_fp16)
                self.smooth(ub_fp16, ub_temp_fp16, self.line_per_iter)
            self.promote(ub_fp32, ub_fp16)
            self.do_exp(ub_fp32)
            ub_temp_fp32 = self.tik.Tensor("float32", self.shape, tik.scope_ubuf, "ub_temp_fp32")
            self.softmax(ub_fp32, ub_temp_fp32, self.line_per_iter)
            self.move_out(ub_fp16, ub_fp32, offset)

    def compute_fp32_bf16(self, core_idx, iter_curr_core):
        """Implementation of fused scaled masked softmax algorithm in fp32 or bf16."""
        offset = self.tik.Scalar("int32")
        offset_mask = self.tik.Scalar("int32")
        ub_fp32 = self.tik.Tensor("float32", self.shape, tik.scope_ubuf, "ub_fp32")
        with self.tik.for_range(0, iter_curr_core) as it:
            self.cal_offset(core_idx, it, offset, offset_mask, self.ele_per_iter)
            self.move_x_in(ub_fp32, offset)
            ub_temp_fp16 = self.tik.Tensor("float16", self.shape, tik.scope_ubuf, "ub_temp_fp16")
            ub_temp_fp32 = self.tik.Tensor("float32", self.shape, tik.scope_ubuf, "ub_temp_fp32")
            self.move_mask_in(ub_temp_fp16, offset_mask)
            self.scale_x(ub_fp32)
            self.process_mask(ub_temp_fp16, ub_temp_fp32)
            self.masked_fill(ub_fp32, ub_temp_fp32, ub_temp_fp16)
            self.smooth(ub_fp32, ub_temp_fp32, self.line_per_iter)
            self.do_exp(ub_fp32)
            self.softmax(ub_fp32, ub_temp_fp32, self.line_per_iter)
            self.move_out(ub_temp_fp16, ub_fp32, offset)

    def move_mask_in(self, ub_mask_fp16, offset_mask):
        """Move mask from gm to ub tensor."""
        if self.fixed_triu_mask:
            move_burst = self.ele_per_iter // Constant.NUM_BLOCK_B16
            self.tik.data_move(ub_mask_fp16, self.fixed_mask_gm[offset_mask], 0, 1, move_burst, 0, 0)
        else:
            move_burst = self.ele_per_iter // Constant.NUM_BLOCK_B8
            ub_mask = self.tik.Tensor("uint8", self.shape, scope=tbe_platform.scope_ubuf, name="ub_mask")
            self.tik.data_move(ub_mask, self.mask_gm[offset_mask], 0, 1, move_burst, 0, 0)
            self._vconv(ub_mask_fp16, ub_mask, self.ele_per_iter, Constant.VEC_MASK_B16, 8, 4)

    def process_mask(self, ub_mask, ub_temp):
        """
        Convert the upper triangular matrix to the lower triangular matrix, and fill False with -10000.
        """
        if self.dtype == "float16":
            self._vec_scalar_func("vmuls", ub_temp, ub_mask, self.tik.Scalar("float16", init_value=-10000), 0, 0,
                                  self.ele_per_iter, Constant.VEC_MASK_B16)
            self._vec_scalar_func("vmuls", ub_mask, ub_mask, self.tik.Scalar("float16", init_value=-1), 0, 0,
                                  self.ele_per_iter, Constant.VEC_MASK_B16)
            self._vec_scalar_func("vadds", ub_mask, ub_mask, self.tik.Scalar("float16", init_value=1), 0, 0,
                                  self.ele_per_iter, Constant.VEC_MASK_B16)
        else:
            self._vconv(ub_temp, ub_mask, self.ele_per_iter, Constant.VEC_MASK_B32, 8, 4)
            self._vec_scalar_func("vmuls", ub_mask, ub_mask, self.tik.Scalar("float16", init_value=-10000), 0, 0,
                                  self.ele_per_iter, Constant.VEC_MASK_B16)
            self._vec_scalar_func("vmuls", ub_temp, ub_temp, self.tik.Scalar("float32", init_value=-1), 0, 0,
                                  self.ele_per_iter, Constant.VEC_MASK_B32)
            self._vec_scalar_func("vadds", ub_temp, ub_temp, self.tik.Scalar("float32", init_value=1), 0, 0,
                                  self.ele_per_iter, Constant.VEC_MASK_B32)

    def move_x_in(self, dst, offset):
        """Move x from gm to ub tensor."""
        if self.dtype == 'float16':
            self.tik.data_move(dst, self.x_gm[offset], 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B16, 0, 0)
        elif self.dtype == 'bfloat16':
            with self.tik.new_stmt_scope():
                ub_bf16 = self.tik.Tensor("bfloat16", self.shape, scope=tbe_platform.scope_ubuf, name="ub_bf16")
                self.tik.data_move(ub_bf16, self.x_gm[offset], 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B16, 0, 0)
                self._vconv(dst, ub_bf16, self.ele_per_iter, Constant.VEC_MASK_B32, 8, 4)
        else:
            self.tik.data_move(dst, self.x_gm[offset], 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B32, 0, 0)

    def smooth(self, ub_x, ub_temp, line):
        """Use the maximum vcalue of x to smooth the input."""
        if self.dtype == "float16":
            block = Constant.NUM_BLOCK_B16
            mask = Constant.VEC_MASK_B16
            dtype = "float16"
        else:
            block = Constant.NUM_BLOCK_B32
            mask = Constant.VEC_MASK_B32
            dtype = "float32"
        ub_reducemax = self.tik.Tensor(dtype, (self.ceil_div(line, block) * block,),
                                       scope=tbe_platform.scope_ubuf, name="ub_reducemax")

        with self.tik.if_scope(tik.all(self.ele_per_line == Constant.SHAPE_2048, block == Constant.NUM_BLOCK_B16)):
            self.tik.vcgmax(mask, ub_temp, ub_x, 128, 1, 1, 8)
            self.tik.vcmax(mask, ub_temp, ub_temp, 8, 1, 1, 8)
            self.tik.vreduce(block, ub_reducemax, ub_temp, 1, 1, 1, 1, 1, mask_mode="counter")
            self.tik.vmuls(block, ub_reducemax, ub_reducemax, -1.0, 1, 1, 1, 0, 0)
            self.tik.vbcb(ub_temp, ub_reducemax, 1, 1, 8)
            for i in range(8):
                self.tik.vadd(mask, ub_x[Constant.SHAPE_2048 * i], ub_x[Constant.SHAPE_2048 * i],
                              ub_temp[block * i], block, 1, 1, 0, 8, 8, 0)
        with self.tik.else_scope():
            offset = self.tik.Scalar("int32")
            cnt, remain, _ = self.cal_level(block)
            move_stride = self.tik.Scalar("int32", init_value=(self.ele_per_line - block) // block)
            with self.tik.for_range(0, line) as i:
                offset.set_as(self.ele_per_line * i)
                self.reduce_max_per_line(ub_x, ub_temp, offset, cnt, remain, mask, block)
            with self.tik.if_scope(tik.all(move_stride > 0, line > 1)):
                self.tik.data_move(ub_temp[block], ub_temp[self.ele_per_line], 0, line - 1, 1, move_stride, 0)
            self.tik.vcgmax(mask, ub_reducemax, ub_temp, self.ceil_div(line, Constant.SCALAR_EIGHT), 1, 1, 8)
            self._vec_scalar_func("vmuls", ub_reducemax, ub_reducemax, -1.0, 0, 0, line, mask)
            maximum = self.tik.Scalar(dtype, name="maximum")
            with self.tik.for_range(0, line) as i:
                offset = self.ele_per_line * i
                maximum.set_as(ub_reducemax[i])
                self._vec_scalar_func("vadds", ub_x, ub_x, maximum, offset, offset, self.ele_per_line, mask)

    def softmax(self, ub_fp32, ub_temp, line):
        """Calculate softmax of input."""
        with self.tik.new_stmt_scope():
            ub_reduceadd = self.tik.Tensor("float32", (line,),
                                           scope=tbe_platform.scope_ubuf, name="ub_reduceadd")
            one_ub = self.tik.Tensor("float32", (line,), scope=tbe_platform.scope_ubuf, name="one_ub")
            self._vec_dup(one_ub, 1.0, 0, line, Constant.VEC_MASK_B32)

            with self.tik.if_scope(self.ele_per_line == Constant.SHAPE_2048):
                self.tik.vcgadd(Constant.VEC_MASK_B32, ub_temp, ub_fp32, Constant.SEVEN_LINE_REPEAT, 1, 1, 8)
                self.tik.vcgadd(Constant.VEC_MASK_B32, ub_temp[Constant.SEVEN_LINE_REPEAT * Constant.NUM_BLOCK_B32],
                                ub_fp32[Constant.SEVEN_LINE_REPEAT * Constant.VEC_MASK_B32], 32, 1, 1, 8)
                self.tik.vcgadd(Constant.VEC_MASK_B32, ub_temp, ub_temp,
                                Constant.SHAPE_2048 // Constant.VEC_MASK_B32, 1, 1, 8)
                self.tik.vcadd(32, ub_reduceadd, ub_temp, 8, 1, 1, 4)
                self.tik.vdiv(8, ub_reduceadd, one_ub, ub_reduceadd, 1, 1, 1, 1, 0, 0, 0)
                self.tik.vbcb(ub_temp, ub_reduceadd, 1, 1, 8)
                for i in range(8):
                    self.tik.vmul(Constant.VEC_MASK_B32, ub_fp32[Constant.SHAPE_2048 * i],
                                  ub_fp32[Constant.SHAPE_2048 * i], ub_temp[8 * i],
                                  Constant.SHAPE_2048 // Constant.VEC_MASK_B32, 1, 1, 0, 8, 8, 0)
            with self.tik.else_scope():
                self.reduce_add_per_line(ub_reduceadd, ub_fp32, ub_temp, line)
                self._vec_func("vdiv", ub_reduceadd, one_ub, ub_reduceadd, 0, 0, 0, line, Constant.VEC_MASK_B32)
                add_sum = self.tik.Scalar("float32", name="add_sum")
                with self.tik.for_range(0, line) as i:
                    add_sum.set_as(ub_reduceadd[i])
                    self._vec_scalar_func("vmuls", ub_fp32, ub_fp32, add_sum, self.ele_per_line * i,
                                          self.ele_per_line * i, self.ele_per_line, Constant.VEC_MASK_B32)

    def move_out(self, ub_fp16, ub_fp32, offset):
        """Move result from ub to gm."""
        if self.dtype == "float16":
            self._vconv(ub_fp16, ub_fp32, self.ele_per_iter, Constant.VEC_MASK_B32, 4, 8)
            self.tik.data_move(self.y_gm[offset], ub_fp16, 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B16, 0, 0)
        elif self.dtype == "float32":
            self.tik.data_move(self.y_gm[offset], ub_fp32, 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B32, 0, 0)
        else:
            ub_bf16 = ub_fp16.reinterpret_cast_to("bfloat16")
            self._vconv(ub_bf16, ub_fp32, self.ele_per_iter, Constant.VEC_MASK_B32, 4, 8, 'round')
            self.tik.data_move(self.y_gm[offset], ub_bf16, 0, 1, self.ele_per_iter // Constant.NUM_BLOCK_B16, 0, 0)


class ScaledMaskedSoftmaxAlignedNZ(BaseClass):
    """Implementation of aligned ScaledMaskedSoftmax in NZ format."""

    def __init__(self, op: "ScaledMaskedSoftmax"):
        """Init."""
        super().__init__(op)
        self.counts = self.tik.Scalar("int32")
        self.ele_per_col = self.tik.Scalar("int32")
        self.x_h.set_as(self.x_h1 * self.x_h0)
        self.x_w.set_as(self.x_w1 * self.x_w0)
        self.mask_h.set_as(self.mask_h1 * self.mask_h0)
        self.mask_w.set_as(self.mask_w1 * self.mask_w0)
        self.shape = (self.line_per_iter, self.x_w)

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
            self.tik.data_move(self.fixed_mask_gm[offset], triu_mask_ub, 0,
                               self.mask_w // Constant.NUM_BLOCK_B16, 1, 0, self.mask_h - 1)
            offset.set_as(offset + Constant.NUM_BLOCK_B16)

    def do_compute(self, core_idx, iter_curr_core):
        """Implementation of fused scaled masked softmax algorithm in NZ format."""
        offset = self.tik.Scalar("int32")
        offset_mask = self.tik.Scalar("int32")
        ub_fp16 = self.tik.Tensor("float16", self.shape, tik.scope_ubuf, "ub_fp16")
        ub_fp32 = self.tik.Tensor("float32", self.shape, tik.scope_ubuf, "ub_fp32")
        with self.tik.for_range(0, iter_curr_core) as it:
            self.cal_offset(core_idx, it, offset, offset_mask, self.counts)
            with self.tik.new_stmt_scope():
                ub_temp_fp16 = self.tik.Tensor("float16", self.shape, tik.scope_ubuf, "ub_temp_fp16")
                ub_mask_fp16 = self.tik.Tensor("float16", self.shape, tik.scope_ubuf, "ub_mask_fp16")
                self.move_mask_in(ub_mask_fp16, offset_mask)
                self.process_mask(ub_mask_fp16, ub_temp_fp16)
                self.move_x_in(ub_fp16, ub_fp32, offset)
                self.scale_x(ub_fp16)
                self.masked_fill(ub_fp16, ub_mask_fp16, ub_temp_fp16)
                self.smooth(ub_fp16, ub_temp_fp16)
            self.promote(ub_fp32, ub_fp16)
            self.do_exp(ub_fp32)
            ub_temp_fp32 = self.tik.Tensor("float32", self.shape, tik.scope_ubuf, "ub_temp_fp32")
            self.softmax(ub_fp32, ub_temp_fp32)
            self.move_out(ub_fp16, ub_fp32, offset)

    def get_parallel_params(self):
        """Calculate parallel parameters."""
        super().get_parallel_params()
        self.counts.set_as(self.line_per_iter * Constant.NUM_BLOCK_B16)
        self.ele_per_col.set_as(Constant.NUM_BLOCK_B16 * self.line_per_iter)

    def move_mask_in(self, ub_mask_fp16, offset_mask):
        """Move mask from gm to ub tensor."""
        src_stride = self.x_h - self.line_per_iter
        if self.fixed_triu_mask:
            self.tik.data_move(ub_mask_fp16, self.fixed_mask_gm[offset_mask],
                               0, self.x_w1, self.line_per_iter, src_stride, 0)
        else:
            ub_mask = self.tik.Tensor("uint8", self.shape, scope=tbe_platform.scope_ubuf, name="ub_mask")
            self.tik.data_move(ub_mask, self.mask_gm[offset_mask], 0, self.x_w1,
                               self.line_per_iter // Constant.SCALAR_TWO, src_stride // Constant.SCALAR_TWO, 0)
            self._vconv(ub_mask_fp16, ub_mask, self.ele_per_iter, Constant.VEC_MASK_B16, 8, 4)

    def process_mask(self, ub_mask, ub_temp):
        """
        Convert the upper triangular matrix to the lower triangular matrix, and fill False with -10000.
        """
        self._vec_scalar_func("vmuls", ub_temp, ub_mask, self.tik.Scalar("float16", init_value=-10000), 0, 0,
                              self.ele_per_iter, Constant.VEC_MASK_B16)
        self._vec_scalar_func("vmuls", ub_mask, ub_mask, self.tik.Scalar("float16", init_value=-1), 0, 0,
                              self.ele_per_iter, Constant.VEC_MASK_B16)
        self._vec_scalar_func("vadds", ub_mask, ub_mask, self.tik.Scalar("float16", init_value=1), 0, 0,
                              self.ele_per_iter, Constant.VEC_MASK_B16)

    def move_x_in(self, ub_fp16, ub_fp32, offset):
        """Move x from gm to ub tensor."""
        src_stride = self.x_h - self.line_per_iter
        if self.dtype == "float16":
            self.tik.data_move(ub_fp16, self.x_gm[offset], 0, self.x_w1, self.line_per_iter, src_stride, 0)
        elif self.dtype == "bfloat16":
            ub_bf16 = ub_fp16.reinterpret_cast_to("bfloat16")
            self.tik.data_move(ub_bf16, self.x_gm[offset], 0, self.x_w1, self.line_per_iter, src_stride, 0)
            self._vconv(ub_fp32, ub_bf16, self.ele_per_iter, Constant.VEC_MASK_B32, 8, 4)
            self._vconv(ub_fp16, ub_fp32, self.ele_per_iter, Constant.VEC_MASK_B32, 4, 8)
        else:
            self.tik.data_move(ub_fp32, self.x_gm[offset], 0, self.x_w1,
                               self.line_per_iter * Constant.SCALAR_TWO, src_stride * Constant.SCALAR_TWO, 0)
            self._vconv(ub_fp16, ub_fp32, self.ele_per_iter, Constant.VEC_MASK_B32, 4, 8)

    def smooth(self, ub_x, ub_temp):
        """Use the maximum vcalue of x to smooth the input."""
        with self.tik.new_stmt_scope():
            ub_dup = self.tik.Tensor("uint16", (Constant.VEC_MASK_B16,), scope=tbe_platform.scope_ubuf, name="ub_dup")
            ub_broadcast = self.tik.Tensor("uint16",
                                           (Constant.SCALAR_TWO * Constant.NUM_BLOCK_B16 * Constant.NUM_BLOCK_B16,),
                                           scope=tbe_platform.scope_ubuf, name="ub_broadcast")
            ub_reducemax = self.tik.Tensor("float16", (self.line_per_iter,),
                                           scope=tbe_platform.scope_ubuf, name="ub_reducemax")

            cnt, remain, _ = self.cal_level(Constant.NUM_BLOCK_B16)
            time = self.tik.Scalar("int32", init_value=Constant.SCALAR_TWO)
            self._vec_func("vmax", ub_temp, ub_x, ub_x, 0,
                           self.ele_per_iter // time, 0, self.ele_per_iter // time, Constant.VEC_MASK_B16)
            with self.tik.for_range(1, cnt) as _:
                time.set_as(time * Constant.SCALAR_TWO)
                self._vec_func("vmax", ub_temp, ub_temp, ub_temp, 0,
                               self.ele_per_iter // time, 0, self.ele_per_iter // time, Constant.VEC_MASK_B16)
            with self.tik.if_scope(remain > 1):
                with self.tik.for_range(1, remain) as i:
                    self._vec_func("vmax", ub_temp, ub_temp, ub_temp, self.ele_per_col * (remain - i - 1),
                                   self.ele_per_col * (remain - i), self.ele_per_col * (remain - i - 1),
                                   self.ele_per_col, Constant.VEC_MASK_B16)
            self.tik.vcgmax(Constant.VEC_MASK_B16, ub_reducemax, ub_temp,
                            Constant.SCALAR_TWO * self.line_per_iter // Constant.VEC_MASK_B16, 1, 1, 8)

            self.tik.vector_dup(Constant.VEC_MASK_B16, ub_dup, self.tik.Scalar(init_value=0, dtype="uint16"), 1, 1, 8)
            ub_reducemax_uint16 = ub_reducemax.reinterpret_cast_to("uint16")
            times = self.ceil_div(self.line_per_iter, Constant.NUM_BLOCK_B16)
            with self.tik.for_range(0, times) as i:
                self.tik.vor(Constant.NUM_BLOCK_B16, ub_broadcast[Constant.NUM_BLOCK_B16 * Constant.NUM_BLOCK_B16 * i],
                             ub_reducemax_uint16[Constant.NUM_BLOCK_B16 * i],
                             ub_dup, Constant.NUM_BLOCK_B16, 1, 1, 0, 1, 0, 0)
            with self.tik.for_range(0, times) as i:
                self.tik.vtranspose(ub_broadcast[Constant.NUM_BLOCK_B16 * Constant.NUM_BLOCK_B16 * i],
                                    ub_broadcast[Constant.NUM_BLOCK_B16 * Constant.NUM_BLOCK_B16 * i])
            ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")

            times = self.ele_per_col // Constant.VEC_MASK_B32
            with self.tik.for_range(0, times) as i:
                self.tik.vsub(64, ub_x[i * Constant.VEC_MASK_B32], ub_x[i * Constant.VEC_MASK_B32],
                              ub_broadcast_fp16[i * Constant.VEC_MASK_B32], self.x_w1, 1, 1, 1,
                              self.ele_per_col // Constant.NUM_BLOCK_B16, self.ele_per_col // Constant.NUM_BLOCK_B16, 0)

    def softmax(self, ub_x, ub_temp):
        """Calculate softmax of input."""
        with self.tik.new_stmt_scope():
            ub_dup = self.tik.Tensor("float32", (Constant.NUM_BLOCK_B8, Constant.NUM_BLOCK_B16),
                                     scope=tbe_platform.scope_ubuf, name="ub_dup")
            ub_reduceadd = self.tik.Tensor("float32", (self.line_per_iter,),
                                           scope=tbe_platform.scope_ubuf, name="ub_reduceadd")
            ub_reduceadd_high_preci = self.tik.Tensor("float32", (self.line_per_iter,),
                                                      scope=tbe_platform.scope_ubuf, name="ub_reduceadd_high_preci")
            work_tensor_ub = self.tik.Tensor("float32", (Constant.SCALAR_TWO * self.line_per_iter,),
                                             scope=tbe_platform.scope_ubuf, name="work_tensor_ub")

            cnt, remain, _ = self.cal_level(Constant.NUM_BLOCK_B16)
            time = self.tik.Scalar("int32", init_value=Constant.SCALAR_TWO)
            self._vec_func("vadd", ub_temp, ub_x, ub_x, 0, self.ele_per_iter // time,
                           0, self.ele_per_iter // time, Constant.VEC_MASK_B32)
            with self.tik.for_range(1, cnt) as _:
                time.set_as(time * Constant.SCALAR_TWO)
                self._vec_func("vadd", ub_temp, ub_temp, ub_temp, 0, self.ele_per_iter // time,
                               0, self.ele_per_iter // time, Constant.VEC_MASK_B32)
            with self.tik.if_scope(remain > 1):
                with self.tik.for_range(1, remain) as i:
                    self._vec_func("vadd", ub_temp, ub_temp, ub_temp, self.ele_per_col * (remain - i - 1),
                                   self.ele_per_col * (remain - i), self.ele_per_col * (remain - i - 1),
                                   self.ele_per_col, Constant.VEC_MASK_B32)
            self.tik.vcadd(Constant.NUM_BLOCK_B16, ub_reduceadd, ub_temp,
                           self.ele_per_col // Constant.NUM_BLOCK_B16, 1, 1, 2)

            self.tik.vec_rec_high_preci(self.ele_per_col // Constant.NUM_BLOCK_B16,
                                        ub_reduceadd_high_preci[0], ub_reduceadd[0], work_tensor_ub[0:], 1, 4, 4)
            dup_times = self.line_per_iter * self.x_h0 // Constant.NUM_BLOCK_B16 // Constant.SCALAR_EIGHT
            with self.tik.for_range(0, dup_times) as j:
                with self.tik.for_range(0, Constant.SCALAR_EIGHT) as k:
                    self.tik.vector_dup(Constant.NUM_BLOCK_B16,
                                        ub_dup[j * Constant.VEC_MASK_B16 + Constant.NUM_BLOCK_B16 * k],
                                        self.tik.Scalar(
                                            init_value=ub_reduceadd_high_preci[j * Constant.NUM_BLOCK_B32 + k],
                                            dtype="float32"),
                                        1, 1, 8)

            stride = self.line_per_iter * Constant.SCALAR_TWO
            with self.tik.for_range(0, self.ele_per_col // Constant.VEC_MASK_B32) as i:
                self.tik.vmul(Constant.VEC_MASK_B32, ub_x[i * Constant.VEC_MASK_B32], ub_x[i * Constant.VEC_MASK_B32],
                              ub_dup[i * Constant.VEC_MASK_B32], self.x_w1, 1, 1, 1, stride, stride, 0)

    def move_out(self, ub_fp16, ub_fp32, offset):
        """Move result from ub to gm."""
        dst_stride = self.x_h - self.line_per_iter
        if self.dtype == "float16":
            self._vconv(ub_fp16, ub_fp32, self.ele_per_iter, Constant.VEC_MASK_B32, 4, 8)
            self.tik.data_move(self.y_gm[offset], ub_fp16, 0, self.x_w1, self.line_per_iter, 0, dst_stride)
        elif self.dtype == "bfloat16":
            ub_bf16 = ub_fp16.reinterpret_cast_to("bfloat16")
            self._vconv(ub_bf16, ub_fp32, self.ele_per_iter, Constant.VEC_MASK_B32, 4, 8, 'round')
            self.tik.data_move(self.y_gm[offset], ub_bf16, 0, self.x_w1, self.line_per_iter, 0, dst_stride)
        else:
            self.tik.data_move(self.y_gm[offset], ub_fp32, 0, self.x_w1, self.line_per_iter * Constant.SCALAR_TWO,
                               0, dst_stride * Constant.SCALAR_TWO)


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,huawei-too-many-arguments
@register_operator("ScaledMaskedSoftmax")
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
        shape and dtype of input tensor.
    mask : dict
        shape and dtype of mask, the shape must be broadcastble with x.
    y : dict
        shape and dtype of output, the shape must be same as x.
    scale : float
        a float scalar scaling the input tensor x
    fixed_triu_mask : bool
        if true: the mask is a fixed upper triangle mask
        if false: the mask is input mask
    kernel_name : str
        kernel name, default value is "scaled_masked_softmax"

    Returns
    -------
    None
    """
    if not tbe_platform.api_check_support("tik.data_move_pad"):
        raise RuntimeError("Error! The device does not support data_move_pad api.")
    return ScaledMaskedSoftmax(x, mask, y, scale, fixed_triu_mask, kernel_name).compute()
