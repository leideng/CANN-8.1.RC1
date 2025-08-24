#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
drop_out_with_muls_and_softmax_grad
"""
import math
from collections import namedtuple
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context


class Constant:
    BLOCK = 16
    # BLOCK * BLOCK
    DB = 256
    VEC_MASK = 128
    VEC_MASK_FP32 = 64
    VEC_DUMP_SHAPE = 16
    MAX_REPEAT = 255
    T_PARAMS_NUM = 32
    SPLIT = 2
    MODE = {
        "fp16_to_fp32": 1,
        "fp32_to_fp16": 2,
        "uint8_to_fp16": 3
    }
    FP32_TO_FP16 = 1
    FP16_TO_FP32 = 2
    FP16 = "float16"
    FP32 = "float32"
    UINT8 = "uint8"
    # unit is 32B
    UNIT = 32
    DTYPE_SIZE = {
        "float16": 2,
        "float32": 4,
        "uint8": 1,
        "int32": 4
    }


# 'pylint: disable=too-many-arguments
class DropoutWithMulsAndSoftmaxGrad():
    """
    arithmetic: 
        grad_mask = y_grad * mask * (1 / input_keep_prob)
        grad_mask_softmax = grad_mask * softmax_output
        reduce the grad_mask_softmax to reduce
        sub = grad_mask - reduce
        x_grad = sub * softmax * alpha
    """
    # 'pylint: disable=too-many-arguments

    def __init__(self, y_grad, mask, softmax_output, x_grad, input_keep_prob,
                 alpha, axes=-1, kernel_name="drop_out_with_muls_and_softmax_grad"):
        self.tik_inst = tik.Tik()

        Attr = namedtuple("Attr", "input_keep_prob, alpha, axes")
        self.info = namedtuple("Info", "grad, mask, softmax, output")
        # dtype
        g_dtype = y_grad.get("dtype").lower()
        m_dtype = mask.get("dtype").lower()
        s_dtype = softmax_output.get("dtype").lower()
        o_dtype = x_grad.get("dtype").lower()
        # size
        g_size = Constant.DTYPE_SIZE.get(g_dtype)
        m_size = Constant.DTYPE_SIZE.get(m_dtype)
        s_size = Constant.DTYPE_SIZE.get(s_dtype)
        o_size = Constant.DTYPE_SIZE.get(o_dtype)

        self.dtype = self.info(grad=g_dtype, mask=m_dtype,
                               softmax=s_dtype, output=o_dtype)
        self.size = self.info(grad=g_size, mask=m_size,
                              softmax=s_size, output=o_size)

        self.kernel_name = kernel_name
        core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - (16 * 1024)
        tbe_context.get_context().add_compile_info(
            "vars", {"core_num": core_num, "ub_size": ub_size})
        # apply gm
        self.t_gm = self.tik_inst.Tensor(
            "int32", (Constant.T_PARAMS_NUM,), name="t_gm", scope=tik.scope_gm)
        n, c, w1, h1, batch, batch_tail, core_num, line, ranges,\
            line_tail, cnt, remain, input_keep_prob, alpha = self.get_tiling_args(
                self.t_gm)

        self.attr = Attr(input_keep_prob=input_keep_prob,
                         alpha=alpha, axes=axes)
        names = """
                n, c, w1, h1, batch, batch_tail, core_num, line, ranges, line_tail, cnt, remain,
                dup_shape, ele_batch, ele_core, ele_core_tail, gm_shape, ele_line, line_b, line_b_b,
                line_b_b_div_mask, line_b_b_div_maskfp32, stride_sub_fp16, stride_sub_fp32,
                ele_line_tail, line_b_tail, line_b_b_tail, line_b_b_div_mask_tail, line_b_b_div_maskfp32_tail,
                stride_sub_fp16_tail, stride_sub_fp32_tail
        """
        Param = namedtuple("Param", names)

        dup_shape = Constant.VEC_DUMP_SHAPE
        ele_batch = w1 * h1 * Constant.DB
        ele_core = ele_batch * batch
        ele_core_tail = ele_batch * batch_tail
        gm_shape = n * c * ele_batch
        # line ----------------------------
        ele_line = w1 * line * Constant.DB
        line_b = line * Constant.BLOCK
        line_b_b = line * Constant.DB
        line_b_b_div_mask = line_b_b // Constant.VEC_MASK
        line_b_b_div_maskfp32 = line_b_b // Constant.VEC_MASK_FP32
        # fp16 sub stride
        stride_sub_fp16 = line_b_b * \
            Constant.DTYPE_SIZE.get("float16") // Constant.UNIT
        # fp32 sub stride
        stride_sub_fp32 = line_b_b * \
            Constant.DTYPE_SIZE.get("float32") // Constant.UNIT

        # stride
        ele_count = h1 - line
        stride_count = ele_count * Constant.DB // Constant.UNIT
        g_stride = stride_count * g_size
        m_stride = stride_count * m_size
        s_stride = stride_count * s_size
        o_stride = stride_count * o_size
        self.stride = self.info(
            grad=g_stride, mask=m_stride, softmax=s_stride, output=o_stride)
        # burst
        burst_count = line_b_b // Constant.UNIT
        g_burst = burst_count * g_size
        m_burst = burst_count * m_size
        s_burst = burst_count * s_size
        o_burst = burst_count * o_size
        self.burst = self.info(grad=g_burst, mask=m_burst,
                               softmax=s_burst, output=o_burst)
        # line_tail ----------------------------
        ele_line_tail = w1 * line_tail * Constant.DB
        line_b_tail = line_tail * Constant.BLOCK
        line_b_b_tail = line_tail * Constant.DB
        line_b_b_div_mask_tail = line_b_b_tail // Constant.VEC_MASK
        line_b_b_div_maskfp32_tail = line_b_b_tail // Constant.VEC_MASK_FP32
        # fp16 sub stride
        stride_sub_fp16_tail = line_b_b_tail * \
            Constant.DTYPE_SIZE.get("float16") // Constant.UNIT
        # fp32 sub stride
        stride_sub_fp32_tail = line_b_b_tail * \
            Constant.DTYPE_SIZE.get("float32") // Constant.UNIT

        self.param = Param(n, c, w1, h1, batch, batch_tail, core_num, line, ranges, line_tail, cnt, remain,
                           dup_shape, ele_batch, ele_core, ele_core_tail, gm_shape, ele_line, line_b, line_b_b,
                           line_b_b_div_mask, line_b_b_div_maskfp32, stride_sub_fp16, stride_sub_fp32,
                           ele_line_tail, line_b_tail, line_b_b_tail, line_b_b_div_mask_tail,
                           line_b_b_div_maskfp32_tail, stride_sub_fp16_tail, stride_sub_fp32_tail)

        # stride
        ele_count = h1 - line_tail
        stride_count = ele_count * Constant.DB // Constant.UNIT
        g_stride = stride_count * g_size
        m_stride = stride_count * m_size
        s_stride = stride_count * s_size
        o_stride = stride_count * o_size
        self.stride_tail = self.info(
            grad=g_stride, mask=m_stride, softmax=s_stride, output=o_stride)
        # burst
        burst_count = line_b_b_tail // Constant.UNIT
        g_burst = burst_count * g_size
        m_burst = burst_count * m_size
        s_burst = burst_count * s_size
        o_burst = burst_count * o_size
        self.burst_tail = self.info(
            grad=g_burst, mask=m_burst, softmax=s_burst, output=o_burst)

        self.gm = self.apply_gm(self.param.gm_shape)
        self.ub = None

    def apply_gm(self, shape):
        """apply gm

        Returns:
            namedtuple: gm
        """
        grad = self.tik_inst.Tensor(
            self.dtype.grad, (shape,), name="grad_gm", scope=tik.scope_gm)
        mask = self.tik_inst.Tensor(
            self.dtype.mask, (shape,), name="mask_gm", scope=tik.scope_gm)
        softmax = self.tik_inst.Tensor(
            self.dtype.softmax, (shape,), name="softmax_gm", scope=tik.scope_gm)
        output = self.tik_inst.Tensor(
            self.dtype.output, (shape,), name="output_gm", scope=tik.scope_gm)

        return self.info(grad=grad, mask=mask, softmax=softmax, output=output)

    def get_tiling_args(self, t_gm):
        """get data from tiling

        Returns:
            namedtuple: tiling
        """

        t_ub = self.tik_inst.Tensor(
            "int32", (Constant.T_PARAMS_NUM,), name='t_ub', scope=tik.scope_ubuf)
        self.tik_inst.data_move(t_ub, t_gm, 0, 1, 2, 0, 0)

        n = self.tik_inst.Scalar(dtype="int32", name="n", init_value=t_ub[0])
        c = self.tik_inst.Scalar(dtype="int32", name="c", init_value=t_ub[1])
        w1 = self.tik_inst.Scalar(dtype="int32", name="w1", init_value=t_ub[2])
        h1 = self.tik_inst.Scalar(dtype="int32", name="h1", init_value=t_ub[3])
        batch = self.tik_inst.Scalar(
            dtype="int32", name="batch", init_value=t_ub[4])
        batch_tail = self.tik_inst.Scalar(
            dtype="int32", name="batch_tail", init_value=t_ub[5])
        core_num = self.tik_inst.Scalar(
            dtype="int32", name="core_num", init_value=t_ub[6])
        line = self.tik_inst.Scalar(
            dtype="int32", name="line", init_value=t_ub[7])
        ranges = self.tik_inst.Scalar(
            dtype="int32", name="ranges", init_value=t_ub[8])
        line_tail = self.tik_inst.Scalar(
            dtype="int32", name="line_tail", init_value=t_ub[9])
        cnt = self.tik_inst.Scalar(
            dtype="int32", name="cnt", init_value=t_ub[10])
        remain = self.tik_inst.Scalar(
            dtype="int32", name="remain", init_value=t_ub[11])
        input_keep_prob = self.tik_inst.Scalar(
            dtype="float32", name="input_keep_prob", init_value=t_ub[12])
        alpha_fp32 = self.tik_inst.Scalar(
            dtype="float32", name="alpha_fp32", init_value=t_ub[13])
        alpha = self.tik_inst.Scalar(dtype=self.dtype.grad, name="alpha")
        if self.dtype.grad == Constant.FP16:
            self.tik_inst.scalar_conv("", alpha, alpha_fp32)
        else:
            alpha.set_as(alpha_fp32)
        return [n, c, w1, h1, batch, batch_tail, core_num, line, ranges, line_tail, cnt, remain, input_keep_prob, alpha]

    def apply_ub_fp16(self):
        """apply ub for float16

        Returns:
            namedtuple: ub
        """
        grad = self.tik_inst.Tensor(
            self.dtype.grad, (self.param.ele_line,), name="grad_ub", scope=tik.scope_ubuf)
        mask = self.tik_inst.Tensor(
            self.dtype.mask, (self.param.ele_line,), name="mask_ub", scope=tik.scope_ubuf)
        softmax = self.tik_inst.Tensor(
            self.dtype.softmax, (self.param.ele_line,), name="softmax_ub", scope=tik.scope_ubuf)
        output = self.tik_inst.Tensor(
            self.dtype.output, (self.param.ele_line,), name="output_ub", scope=tik.scope_ubuf)
        softmax_fp32 = self.tik_inst.Tensor(
            "float32", (self.param.ele_line,), name="softmax_ub_fp32", scope=tik.scope_ubuf)
        output_fp32 = self.tik_inst.Tensor(
            "float32", (self.param.ele_line,), name="output_ub_fp32", scope=tik.scope_ubuf)

        reduce_add = self.tik_inst.Tensor(
            "float32", (self.param.line_b,), name="ub_reduce_add", scope=tik.scope_ubuf)
        reduce_add_fp16 = self.tik_inst.Tensor(
            "float16", (self.param.line_b,), name="ub_reduceadd_fp16", scope=tik.scope_ubuf)
        dup = self.tik_inst.Tensor(
            "int16", (self.param.dup_shape,), name="ub_dup", scope=tik.scope_ubuf)
        broadcast = self.tik_inst.Tensor(
            "int16", (self.param.line_b_b,), name="ub_broadcast", scope=tik.scope_ubuf)

        Ub = namedtuple(
            "Ub", "grad, mask, softmax, output, softmax_fp32, output_fp32, reduce_add, reduce_add_fp16, dup, broadcast")

        return Ub(grad, mask, softmax, output, softmax_fp32, output_fp32, reduce_add, reduce_add_fp16, dup, broadcast)

    def apply_ub_fp32(self):
        """apply ub for float32

        Returns:
            namedtuple: ub
        """
        grad = self.tik_inst.Tensor(
            self.dtype.grad, (self.param.ele_line,), name="grad_ub", scope=tik.scope_ubuf)
        mask = self.tik_inst.Tensor(
            self.dtype.mask, (self.param.ele_line,), name="mask_ub", scope=tik.scope_ubuf)
        softmax = self.tik_inst.Tensor(
            self.dtype.softmax, (self.param.ele_line,), name="softmax_ub", scope=tik.scope_ubuf)
        output = self.tik_inst.Tensor(
            self.dtype.output, (self.param.ele_line,), name="output_ub", scope=tik.scope_ubuf)
        output_fp16 = self.tik_inst.Tensor(
            "float16", (self.param.ele_line,), name="output_fp16", scope=tik.scope_ubuf)
        reduce_add = self.tik_inst.Tensor(
            "float32", (self.param.line_b,), name="ub_reduce_add", scope=tik.scope_ubuf)
        broadcast = self.tik_inst.Tensor(
            "float32", (self.param.line_b_b,), name="ub_broadcast", scope=tik.scope_ubuf)

        Ub = namedtuple(
            "Ub", "grad, mask, softmax, output, output_fp16, reduce_add, broadcast")

        return Ub(grad, mask, softmax, output, output_fp16, reduce_add, broadcast)

    def drop_out_with_muls_and_softmax_grad_compute(self):

        offset = self.tik_inst.Scalar("int32", name="offset", init_value=0)
        with self.tik_inst.for_range(0, self.param.core_num, block_num=self.param.core_num) as core_index:
            if self.dtype.grad == Constant.FP16:
                self.ub = self.apply_ub_fp16()
                self.handle_batch_per_core_fp16(core_index, offset)
            else:
                self.ub = self.apply_ub_fp32()
                self.handle_batch_per_core_fp32(core_index, offset)

        opt_config = {
            "enable_const_fold": True
        }
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.gm.grad,
                                       self.gm.mask, self.gm.softmax],
                               outputs=[self.gm.output],
                               flowtable=[self.t_gm],
                               config=opt_config)
        return self.tik_inst

    def handle_batch_per_core_fp16(self, core_index, offset):
        """Process data on chip

        Args:
            core_index (int): kernel index
            offset (int): Offset when looping through tasks
        """

        with self.tik_inst.if_scope(core_index + 1 < self.param.core_num):
            with self.tik_inst.for_range(0, self.param.batch) as batch_index:
                self.handle_ele_per_batch_fp16(core_index, batch_index, offset)
                with self.tik_inst.if_scope(self.param.line_tail > 0):
                    self.handle_ele_per_batch_fp16_tail(
                        core_index, batch_index, offset)
        with self.tik_inst.else_scope():
            with self.tik_inst.for_range(0, self.param.batch_tail) as batch_index:
                self.handle_ele_per_batch_fp16(core_index, batch_index, offset)
                with self.tik_inst.if_scope(self.param.line_tail > 0):
                    self.handle_ele_per_batch_fp16_tail(
                        core_index, batch_index, offset)

    def handle_batch_per_core_fp32(self, core_index, offset):
        """Process data on chip

        Args:
            core_index (int): kernel index
            offset (int): Offset when looping through tasks
        """

        with self.tik_inst.if_scope(core_index + 1 < self.param.core_num):
            with self.tik_inst.for_range(0, self.param.batch) as batch_index:
                self.handle_ele_per_batch_fp32(core_index, batch_index, offset)
                with self.tik_inst.if_scope(self.param.line_tail > 0):
                    self.handle_ele_per_batch_fp32_tail(
                        core_index, batch_index, offset)
        with self.tik_inst.else_scope():
            with self.tik_inst.for_range(0, self.param.batch_tail) as batch_index:
                self.handle_ele_per_batch_fp32(core_index, batch_index, offset)
                with self.tik_inst.if_scope(self.param.line_tail > 0):
                    self.handle_ele_per_batch_fp32_tail(
                        core_index, batch_index, offset)

    def handle_ele_per_batch_fp16(self, core_index, batch_index, offset):
        """Process tasks on each core

        Args:
            core_index (_type_): kernel index
            batch_index (_type_): batch index
            offset (_type_): Offset when looping through tasks
        """
        with self.tik_inst.for_range(0, self.param.ranges) as i:
            offset.set_as(core_index * self.param.ele_core +
                          batch_index * self.param.ele_batch +
                          i * self.param.line_b_b)
            # gm to ub
            self.gm_to_ub(offset, self.stride, self.burst)
            # grad mul mask
            self.grad_mask_softmax_fp16()
            # reduce
            self.reduce_sum(self.ub.output_fp32, self.param.line, self.param.ele_line,
                            self.param.line_b_b, self.param.line_b_b_div_maskfp32)
            self.broadcast_fp16(self.param.line)
            # sub
            self.sub_and_mul_fp16(
                self.param.line_b_b_div_mask, self.param.stride_sub_fp16)
            # ub to gm
            self.tik_inst.data_move(
                self.gm.output[offset], self.ub.output, 0, self.param.w1, self.burst.output, 0, self.stride.output)

    def handle_ele_per_batch_fp32(self, core_index, batch_index, offset):
        with self.tik_inst.for_range(0, self.param.ranges) as i:
            offset.set_as(core_index * self.param.ele_core + batch_index *
                          self.param.ele_batch + i * self.param.line_b_b)
            # gm to ub
            self.gm_to_ub(offset, self.stride, self.burst)
            # grad mul mask mul softmax
            self.grad_mask_softmax_fp32()
            # reduce
            self.reduce_sum(self.ub.output, self.param.line, self.param.ele_line,
                            self.param.line_b_b, self.param.line_b_b_div_maskfp32)
            self.broadcast_fp32(self.param.line)
            # sub
            self.sub_and_mul_fp32(
                self.param.line_b_b_div_maskfp32, self.param.stride_sub_fp32)
            # ub to gm
            self.tik_inst.data_move(
                self.gm.output[offset], self.ub.output, 0, self.param.w1, self.burst.output, 0, self.stride.output)

    def handle_ele_per_batch_fp16_tail(self, core_index, batch_index, offset):
        """Process tasks on each core

        Args:
            core_index (_type_): kernel index
            batch_index (_type_): batch index
            offset (_type_): Offset when looping through tasks
        """
        offset.set_as(core_index * self.param.ele_core +
                      batch_index * self.param.ele_batch +
                      self.param.ranges * self.param.line_b_b)
        # gm to ub
        self.gm_to_ub(offset, self.stride_tail, self.burst_tail)
        # grad mul mask
        self.grad_mask_softmax_fp16()
        # reduce
        self.reduce_sum(self.ub.output_fp32, self.param.line_tail, self.param.ele_line_tail,
                        self.param.line_b_b_tail, self.param.line_b_b_div_maskfp32_tail)
        self.broadcast_fp16(self.param.line_tail)
        # sub
        self.sub_and_mul_fp16(
            self.param.line_b_b_div_mask_tail, self.param.stride_sub_fp16_tail)
        # ub to gm
        self.tik_inst.data_move(self.gm.output[offset], self.ub.output, 0,
                                self.param.w1, self.burst_tail.output, 0, self.stride_tail.output)

    def handle_ele_per_batch_fp32_tail(self, core_index, batch_index, offset):
        offset.set_as(core_index * self.param.ele_core + batch_index *
                      self.param.ele_batch + self.param.ranges * self.param.line_b_b)
        # gm to ub
        self.gm_to_ub(offset, self.stride_tail, self.burst_tail)
        # grad mul mask mul softmax
        self.grad_mask_softmax_fp32()
        # reduce
        self.reduce_sum(self.ub.output, self.param.line_tail, self.param.ele_line_tail,
                        self.param.line_b_b_tail, self.param.line_b_b_div_maskfp32_tail)
        self.broadcast_fp32(self.param.line_tail)
        # sub
        self.sub_and_mul_fp32(
            self.param.line_b_b_div_maskfp32_tail, self.param.stride_sub_fp32_tail)
        # ub to gm
        self.tik_inst.data_move(self.gm.output[offset], self.ub.output, 0,
                                self.param.w1, self.burst_tail.output, 0, self.stride_tail.output)

    def gm_to_ub(self, offset, stride, burst):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            self.tik_inst.data_move(
                self.ub.grad, self.gm.grad[offset], 0, self.param.w1, burst.grad, stride.grad, 0)
            self.tik_inst.data_move(
                self.ub.mask, self.gm.mask[offset], 0, self.param.w1, burst.mask, stride.mask, 0)
            self.tik_inst.data_move(
                self.ub.softmax, self.gm.softmax[offset], 0, self.param.w1, burst.softmax, stride.softmax, 0)

    def grad_mask_softmax_fp16(self):
        with self.tik_inst.if_scope(tik.all(self.attr.input_keep_prob != 0, self.attr.input_keep_prob != 1)):
            self.tik_inst.h_cast(self.ub.output, self.ub.mask, "")
            self.tik_inst.h_mul(self.ub.grad, self.ub.grad, self.ub.output)
        with self.tik_inst.if_scope(self.attr.input_keep_prob == 0):
            self.tik_inst.h_mul(self.ub.grad, self.ub.grad, self.tik_inst.Scalar(
                init_value=0, dtype=self.dtype.grad))
        with self.tik_inst.else_scope():
            self.tik_inst.h_mul(self.ub.grad, self.ub.grad, self.tik_inst.Scalar(
                init_value=1 / self.attr.input_keep_prob, dtype=self.dtype.grad))
        # grad_mask mul softmax
        self.tik_inst.h_cast(self.ub.output_fp32, self.ub.grad, "")
        self.tik_inst.h_cast(self.ub.softmax_fp32, self.ub.softmax, "")
        self.tik_inst.h_mul(self.ub.output_fp32,
                            self.ub.output_fp32, self.ub.softmax_fp32)

    def grad_mask_softmax_fp32(self):
        with self.tik_inst.if_scope(tik.all(self.attr.input_keep_prob != 0, self.attr.input_keep_prob != 1)):
            self.tik_inst.h_cast(self.ub.output_fp16, self.ub.mask, "")
            self.tik_inst.h_cast(self.ub.output, self.ub.output_fp16, "")
            self.tik_inst.h_mul(self.ub.grad, self.ub.grad, self.ub.output)
        with self.tik_inst.if_scope(self.attr.input_keep_prob == 0):
            self.tik_inst.h_mul(self.ub.grad, self.ub.grad, self.tik_inst.Scalar(
                init_value=0, dtype=self.dtype.grad))
        with self.tik_inst.else_scope():
            self.tik_inst.h_mul(self.ub.grad, self.ub.grad, self.tik_inst.Scalar(
                init_value=1 / self.attr.input_keep_prob, dtype=self.dtype.grad))

        self.tik_inst.h_mul(self.ub.output, self.ub.grad, self.ub.softmax)

    # 'pylint: disable=too-many-arguments
    def reduce_sum(self, grad_mask_softmax, line, ele_numel_line, line_b_b, line_b_b_div_maskfp32):
        """reduce grad_mask_softmax

        Args:
            grad_mask_softmax (_type_): grad * mask * softmax
            line (_type_): The number of copies of data processed at one time
            ele_numel_line (_type_): The number of all elements in each line
            line_b_b (_type_): line * block * block

        Returns:
            _type_: broadcast
        """
        self.custom_vadd(grad_mask_softmax, grad_mask_softmax, grad_mask_softmax,
                         Constant.FP32, ele_numel_line, line_b_b, line_b_b_div_maskfp32)
        self.tik_inst.vcadd(Constant.BLOCK,
                            self.ub.reduce_add,
                            grad_mask_softmax,
                            line * Constant.BLOCK,
                            1, 1,
                            Constant.BLOCK * Constant.DTYPE_SIZE.get(Constant.FP32) // Constant.UNIT)

    # 'pylint: disable=too-many-arguments
    def custom_vadd(self, dst, src0, src1, dtype, ele_numel_line, line_b_b, line_b_b_div_maskfp32):
        """grad_mask - reduce

        Args:
            dst (_type_): grad_mask_softmax
            src0 (_type_): grad_mask_softmax
            src1 (_type_): grad_mask_softmax
            dtype (_type_): float32
            ele_numel_line (_type_): The number of all elements in each line
            line_b_b (_type_): line * block * block
        """

        mask = Constant.VEC_MASK_FP32
        stride = mask * Constant.DTYPE_SIZE.get(dtype) // Constant.UNIT

        offset = self.tik_inst.Scalar("int32", init_value=ele_numel_line)
        offset1 = self.tik_inst.Scalar("int32")
        offset2 = self.tik_inst.Scalar("int32")
        with self.tik_inst.for_range(0, self.param.cnt):
            offset.set_as(offset // Constant.SPLIT)
            self.tik_inst.vadd(
                mask, dst, src0, src1[offset], offset // mask, 1, 1, 1, stride, stride, stride)
        with self.tik_inst.if_scope(self.param.remain > 0):
            with self.tik_inst.for_range(1, self.param.remain + 1) as j:
                offset1.set_as(line_b_b * (self.param.remain - j))
                offset2.set_as(line_b_b * (self.param.remain - j + 1))
                self.tik_inst.vadd(mask, dst[offset1], src0[offset1], src1[offset2], line_b_b_div_maskfp32,
                                   1, 1, 1, stride, stride, stride)

    def broadcast_fp16(self, line):

        self.tik_inst.h_cast(self.ub.reduce_add_fp16, self.ub.reduce_add, "")

        self.tik_inst.vector_dup(Constant.VEC_DUMP_SHAPE, self.ub.dup, self.tik_inst.Scalar(
            init_value=0, dtype="int16"), 1, 1, 8)
        reduce_add_int16 = self.ub.reduce_add_fp16.reinterpret_cast_to("int16")
        with self.tik_inst.for_range(0, line) as j:
            self.tik_inst.vor(Constant.BLOCK, self.ub.broadcast[Constant.BLOCK * Constant.BLOCK * j],
                              reduce_add_int16[Constant.BLOCK *
                                               j], self.ub.dup, Constant.BLOCK,
                              1, 1, 0, 1, 0, 0)
        with self.tik_inst.for_range(0, line) as j:
            self.tik_inst.vtranspose(self.ub.broadcast[Constant.BLOCK * Constant.BLOCK * j],
                                     self.ub.broadcast[Constant.BLOCK * Constant.BLOCK * j])

    def broadcast_fp32(self, line):
        ele = self.tik_inst.Scalar(Constant.FP32, name="ele")
        with self.tik_inst.for_range(0, line) as i:
            with self.tik_inst.for_range(0, Constant.BLOCK) as j:
                ele.set_as(self.ub.reduce_add[i * Constant.BLOCK + j])
                self.tik_inst.vector_dup(Constant.VEC_DUMP_SHAPE,
                                         self.ub.broadcast[i * Constant.BLOCK *
                                                           Constant.BLOCK + j * Constant.BLOCK],
                                         ele, Constant.BLOCK, 1, 0)

    def sub_and_mul_fp16(self, line_b_b_div_mask, stride):
        broadcast_fp16 = self.ub.broadcast.reinterpret_cast_to("float16")
        with self.tik_inst.for_range(0, line_b_b_div_mask) as idx:
            offset_sub = idx * Constant.VEC_MASK
            self.tik_inst.vsub(Constant.VEC_MASK, self.ub.output[offset_sub], self.ub.grad[offset_sub],
                               broadcast_fp16[offset_sub], self.param.w1, 1, 1, 1, stride, stride, 0)
        self.tik_inst.h_mul(self.ub.output, self.ub.output, self.ub.softmax)
        self.tik_inst.h_mul(self.ub.output, self.ub.output, self.attr.alpha)

    def sub_and_mul_fp32(self, line_b_b_div_maskfp32, stride):
        with self.tik_inst.for_range(0, line_b_b_div_maskfp32) as idx:
            offset_sub = idx * Constant.VEC_MASK_FP32
            self.tik_inst.vsub(Constant.VEC_MASK_FP32, self.ub.output[offset_sub], self.ub.grad[offset_sub],
                               self.ub.broadcast[offset_sub], self.param.w1, 1, 1, 1, stride, stride, 0)

        self.tik_inst.h_mul(self.ub.output, self.ub.output, self.ub.softmax)
        self.tik_inst.h_mul(self.ub.output, self.ub.output, self.attr.alpha)


# 'pylint: disable=too-many-arguments
@register_operator('DropoutWithMulsAndSoftmaxGrad')
def drop_out_with_muls_and_softmax_grad(y_grad, mask, softmax_output, x_grad, input_keep_prob, alpha, axes=-1,
                                        kernel_name="drop_out_with_muls_and_softmax_grad"):
    """
    drop_out_do_mask_v3_d + softmaxgrad + muls
    """
    op_inst = DropoutWithMulsAndSoftmaxGrad(y_grad, mask, softmax_output,
                                            x_grad,
                                            input_keep_prob, alpha, axes,
                                            kernel_name)
    tik_inst = op_inst.drop_out_with_muls_and_softmax_grad_compute()
    return tik_inst
