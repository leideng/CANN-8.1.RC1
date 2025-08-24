#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
Dynamic col2im.
"""
from impl import constant_util as constant
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from tbe.common.platform import get_bit_len


class Constant:
    """
    The class for constant.
    """
    TILING_ARG_NUM = 18
    TILING_MOVE_TIMES = 1
    TILING_MOVE_BRUST = 3
    FP16_BLK_NUM = 16
    FP32_BLK_NUM = 8
    FP16_MASK = 128
    FP32_MASK = 64
    MAX_REPEAT_TIME = 255
    MAX_MOVE_STRIDE = 65535
    BLOCK_BYTE_SIZE = 32
    FP16_REP_STRIDE = 4
    FP32_REP_STRIDE = 8


# pylint: disable=invalid-name,too-many-arguments,too-many-locals,unused-argument
class Col2im(object):
    def __init__(self, x, output_size, kernel_size, dilation, padding, stride, kernel_name):
        """__init__"""
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.core_num = tik.Dprofile().get_aicore_num()
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num, })
        self.dtype = x.get("dtype")
        self.caculate_dtype = "float32"
        self.need_cast = self.dtype != self.caculate_dtype
        self.support_atomic_add = tbe_platform.api_check_support("tik.set_atomic_add", self.caculate_dtype)
        if self.dtype == "float32":
            self.dtype_move_byte_num = constant.DATA_SIZE_FOUR
        else:
            self.dtype_move_byte_num = constant.DATA_SIZE_TWO
        self.fp32_move_byte_num = constant.DATA_SIZE_FOUR
        self.vector_length = constant.VECTOR_BYTE_SIZE // self.fp32_move_byte_num
        self.size_dtype = output_size.get("dtype")
        self.x_gm = self.tik_instance.Tensor(self.dtype, [constant.SHAPE_SIZE_LIMIT], name="x_gm", scope=tik.scope_gm)
        self.output_size_gm = self.tik_instance.Tensor(self.size_dtype, [constant.SHAPE_SIZE_LIMIT],
                                                       name="output_size_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(self.size_dtype, [Constant.TILING_ARG_NUM], name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.init_scalars()
        self.get_tiling_args()
        self.clac_params()
        if self.need_cast:
            self.y_gm_fp32 = self.tik_instance.Tensor(self.caculate_dtype, [constant.SHAPE_SIZE_LIMIT],
                                                      name="y_gm_fp32", scope=tik.scope_gm,
                                                      is_workspace=True, is_atomic_add=True)
            self.y_gm = self.tik_instance.Tensor(self.dtype, [constant.SHAPE_SIZE_LIMIT], name="y_gm",
                                                 scope=tik.scope_gm)
            sync_length = (tbe_platform.get_soc_spec(tbe_platform.CORE_NUM) *
                          constant.BLOCK_SIZE // constant.DATA_SIZE_EIGHT)
            self.sync_workspace = self.tik_instance.Tensor("int64", (sync_length, ), name="barrier_workspace",
                                                           scope=tik.scope_gm, is_workspace=True, is_atomic_add=True)
        else:
            self.y_gm = self.tik_instance.Tensor(self.dtype, [constant.SHAPE_SIZE_LIMIT], name="y_gm",
                                                 scope=tik.scope_gm, is_atomic_add=True)
            self.y_gm_fp32 = self.y_gm

    def init_scalars(self):
        self.output_batch = self.tik_instance.Scalar("int32")
        self.output_c1 = self.tik_instance.Scalar("int32")
        self.output_h = self.tik_instance.Scalar("int32")
        self.output_w = self.tik_instance.Scalar("int32")
        self.output_c0 = self.tik_instance.Scalar("int32")
        self.input_c1 = self.tik_instance.Scalar("int32")
        self.input_h = self.tik_instance.Scalar("int32")
        self.input_w = self.tik_instance.Scalar("int32")
        self.input_c0 = self.tik_instance.Scalar("int32")
        self.kernel_h = self.tik_instance.Scalar("int32")
        self.kernel_w = self.tik_instance.Scalar("int32")
        self.stride_h = self.tik_instance.Scalar("int32")
        self.stride_w = self.tik_instance.Scalar("int32")
        self.padding_h = self.tik_instance.Scalar("int32")
        self.padding_w = self.tik_instance.Scalar("int32")
        self.dilation_h = self.tik_instance.Scalar("int32")
        self.dilation_w = self.tik_instance.Scalar("int32")
        self.core_num_var = self.tik_instance.Scalar("int32")
        self.length = self.tik_instance.Scalar("int32")
        self.flag = self.tik_instance.Scalar("int32")
        self.move_in_stride_gm = self.tik_instance.Scalar("int64")
        self.move_in_stride_ub = self.tik_instance.Scalar("int64")
        self.move_in_offset_gm = self.tik_instance.Scalar("int64")
        self.move_in_offset_ub = self.tik_instance.Scalar("int64")
        self.move_out_stride = self.tik_instance.Scalar("int64")
        self.move_out_offset_gm = self.tik_instance.Scalar("int64")
        self.move_burst = self.tik_instance.Scalar("int64")
        self.fp32_move_burst = self.tik_instance.Scalar("int64")
        self.move_out_burst = self.tik_instance.Scalar("int64")

    def get_tiling_args(self):
        """
        Function: get tiling data.
        """
        tiling_ub = self.tik_instance.Tensor("int32", [Constant.TILING_ARG_NUM], name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, Constant.TILING_MOVE_TIMES,
                                    Constant.TILING_MOVE_BRUST, 0, 0)
        self.output_batch.set_as(tiling_ub[0])
        self.output_c1.set_as(tiling_ub[1])
        self.output_h.set_as(tiling_ub[2])
        self.output_w.set_as(tiling_ub[3])
        self.output_c0.set_as(tiling_ub[4])
        self.input_c1.set_as(tiling_ub[5])
        self.input_h.set_as(tiling_ub[6])
        self.input_w.set_as(tiling_ub[7])
        self.input_c0.set_as(tiling_ub[8])
        self.kernel_h.set_as(tiling_ub[9])
        self.kernel_w.set_as(tiling_ub[10])
        self.stride_h.set_as(tiling_ub[11])
        self.stride_w.set_as(tiling_ub[12])
        self.padding_h.set_as(tiling_ub[13])
        self.padding_w.set_as(tiling_ub[14])
        self.dilation_h.set_as(tiling_ub[15])
        self.dilation_w.set_as(tiling_ub[16])
        self.core_num_var.set_as(tiling_ub[17])

    def clac_params(self):
        """
        Function: Calculate parameters for following computing.
        """
        self.kernel_num = self.kernel_h * self.kernel_w
        self.ho = (self.output_h + 2 * self.padding_h - self.dilation_h * (self.kernel_h - 1) - 1) // self.stride_h + 1
        self.wo = (self.output_w + 2 * self.padding_w - self.dilation_w * (self.kernel_w - 1) - 1) // self.stride_w + 1
        self.task_num = self.output_batch * self.output_c1
        self.task_num_per_aicore = self.task_num // self.core_num_var
        self.task_tail = self.task_num % self.core_num_var
        self.move_burst.set_as(self.input_c0 * self.dtype_move_byte_num // constant.BLOCK_SIZE)
        self.fp32_move_burst.set_as(self.input_c0 * self.fp32_move_byte_num // constant.BLOCK_SIZE)
        with self.tik_instance.if_scope(tik.all(self.padding_h == 0, self.padding_w == 0)):
            kernel_w_new = self.kernel_w + (self.dilation_w - 1) * (self.kernel_w - 1)
            self.length.set_as(kernel_w_new * self.kernel_h * self.input_c0)
            self.move_in_stride_gm.set_as((self.input_w - 1) * self.input_c0 *
                                          self.dtype_move_byte_num // constant.BLOCK_SIZE)
            with self.tik_instance.if_scope(tik.any(self.dilation_h > 1, self.dilation_w > 1)):
                self.move_in_stride_ub.set_as((self.dilation_w - 1) * self.input_c0 *
                                              self.dtype_move_byte_num // constant.BLOCK_SIZE)
                self.move_in_offset_gm.set_as(self.kernel_w * self.input_w * self.input_c0)
                self.move_in_offset_ub.set_as(kernel_w_new * self.input_c0)
            self.move_out_stride.set_as((self.output_w - kernel_w_new + (self.dilation_h - 1) * self.output_w) *
                                        self.input_c0 * self.fp32_move_byte_num // constant.BLOCK_SIZE)
            self.move_out_offset_gm.set_as((self.output_w - kernel_w_new + (self.dilation_h - 1) * self.output_w) *
                                            self.input_c0)
            self.move_out_burst.set_as(kernel_w_new * self.input_c0 * self.fp32_move_byte_num // constant.BLOCK_SIZE)
            self.flag.set_as(0)
        with self.tik_instance.else_scope():
            self.length.set_as(self.input_c0)
            self.flag.set_as(1)

    def col2im_compute(self):
        """
        Function: col2im compute.
        """
        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_idx:
            input_ub = self.tik_instance.Tensor(self.dtype, (self.length,), tik.scope_ubuf, "input_ub")
            self._clear_ub(self.dtype, self.length, input_ub)

            with self.tik_instance.for_range(0, self.task_num_per_aicore) as task_idx:
                self._compute_tasks(core_idx + task_idx * self.core_num_var, input_ub)
            with self.tik_instance.if_scope(core_idx < self.task_tail):
                self._compute_tasks(core_idx + self.task_num_per_aicore * self.core_num_var, input_ub)

            if self.need_cast:
                with self.tik_instance.if_scope(self.core_num_var > 1):
                    self.tik_instance.block_barrier(self.sync_workspace)
                with self.tik_instance.for_range(0, self.task_num_per_aicore) as task_idx:
                    self._cast_each_task(core_idx + task_idx * self.core_num_var)
                with self.tik_instance.if_scope(core_idx < self.task_tail):
                    self._cast_each_task(core_idx + self.task_num_per_aicore * self.core_num_var)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.x_gm, self.output_size_gm],
                                   outputs=[self.y_gm],
                                   flowtable=[self.tiling_gm],
                                   config={"enable_const_fold": True})

    def _compute_tasks(self, offset, input_ub):
        """
        Function: compute tasks.
        """
        if self.support_atomic_add:
            with self.tik_instance.if_scope(self.flag == 0):
                self.tik_instance.set_atomic_add(self.caculate_dtype)
                self._compute_each_task_with_atomic_add(offset, input_ub)
                self.tik_instance.set_atomic_add(0)
            with self.tik_instance.else_scope():
                self._compute_each_task(offset, input_ub)
        else:
            self._compute_each_task(offset, input_ub)

    def _compute_each_task_with_atomic_add(self, nc, input_ub):
        """
        Function: col2im compute with atomic add according to each task.
        """
        n = nc // self.output_c1
        ci = nc % self.output_c1
        ub_offset = self.tik_instance.Scalar("int64")
        gm_offset = self.tik_instance.Scalar("int64")
        x_gm_idx = self.tik_instance.Scalar("int64")
        input_ub_idx = self.tik_instance.Scalar("int64")
        if self.need_cast:
            input_ub_fp32 = self.tik_instance.Tensor(self.caculate_dtype, (self.length,),
                                                     tik.scope_ubuf, "input_ub_fp32")
        else:
            input_ub_fp32 = input_ub
        with self.tik_instance.for_range(0, self.ho) as h:
            output_offset_h = h * self.stride_h
            with self.tik_instance.for_range(0, self.wo) as w:
                output_offset_w = w * self.stride_w
                ub_offset.set_as(0)
                x_gm_idx.set_as(n * self.input_c1 * self.input_h * self.input_w * self.input_c0 +
                                ci * self.input_h * self.input_w * self.input_c0 +
                                (h * self.wo + w) * self.input_c0)
                gm_offset.set_as(n * self.output_c1 * self.output_h * self.output_w * self.output_c0 +
                                 ci * self.output_h * self.output_w * self.output_c0 +
                                 output_offset_h * self.output_w * self.output_c0 +
                                 output_offset_w * self.output_c0)
                input_ub_idx.set_as(0)
                with self.tik_instance.if_scope(tik.all(self.dilation_h <= 1, self.dilation_w <= 1)):
                    self.tik_instance.data_move(input_ub, self.x_gm[x_gm_idx], constant.SID, self.kernel_num,
                                                self.move_burst, self.move_in_stride_gm, constant.STRIDE_ZERO)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.kernel_h) as line:
                        self.tik_instance.data_move(input_ub[ub_offset], self.x_gm[x_gm_idx], constant.SID,
                                                    self.kernel_w, self.move_burst, self.move_in_stride_gm,
                                                    self.move_in_stride_ub)
                        x_gm_idx.set_as(x_gm_idx + self.move_in_offset_gm)
                        ub_offset.set_as(ub_offset + self.move_in_offset_ub)

                if self.need_cast:
                    self._data_conv(input_ub_fp32, input_ub, mask=64, mode="", num=self.length,
                                    dst_stride=Constant.FP32_REP_STRIDE, src_stride=Constant.FP16_REP_STRIDE)

                with self.tik_instance.if_scope(self.move_out_stride <= Constant.MAX_MOVE_STRIDE):
                    self.tik_instance.data_move(self.y_gm_fp32[gm_offset], input_ub_fp32, constant.SID,
                                                self.kernel_h, self.move_out_burst, constant.STRIDE_ZERO,
                                                self.move_out_stride)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.kernel_h) as line:
                        self.tik_instance.data_move(self.y_gm_fp32[gm_offset], input_ub_fp32[input_ub_idx],
                                                    constant.SID, 1, self.move_out_burst, 
                                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                        gm_offset.set_as(gm_offset + self.move_out_offset_gm)
                        input_ub_idx.set_as(input_ub_idx + self.move_in_offset_ub)

    def _compute_each_task(self, nc, input_ub):
        """
        Function: col2im compute according to each task.
        """
        n = nc // self.output_c1
        ci = nc % self.output_c1
        output_ub = self.tik_instance.Tensor(self.caculate_dtype, (self.vector_length,),
                                             tik.scope_ubuf, "output_ub")
        if self.need_cast:
            input_ub_fp32 = self.tik_instance.Tensor(self.caculate_dtype, (self.length,),
                                                     tik.scope_ubuf, "input_ub_fp32")
        else:
            input_ub_fp32 = input_ub
        with self.tik_instance.for_range(0, self.kernel_num) as mask_id:
            width = mask_id % self.kernel_w
            height = mask_id // self.kernel_w
            with self.tik_instance.for_range(0, self.ho) as h:
                output_offset_h = height * self.dilation_h + h * self.stride_h - self.padding_h
                with self.tik_instance.for_range(0, self.wo) as w:
                    output_offset_w = width * self.dilation_w + w * self.stride_w - self.padding_w
                    with self.tik_instance.if_scope(tik.all(output_offset_h >= 0, output_offset_h < self.output_h,
                                                            output_offset_w >= 0, output_offset_w < self.output_w)):
                        x_gm_idx = n * self.input_c1 * self.input_h * self.input_w * self.input_c0 + \
                            ci * self.input_h * self.input_w * self.input_c0 + \
                            mask_id * self.input_w * self.input_c0 + \
                            (h * self.wo + w) * self.input_c0
                        y_gm_idx = n * self.output_c1 * self.output_h * self.output_w * self.output_c0 + \
                            ci * self.output_h * self.output_w * self.output_c0 + \
                            output_offset_h * self.output_w * self.output_c0 + \
                            output_offset_w * self.output_c0
                        self.tik_instance.data_move(input_ub, self.x_gm[x_gm_idx],
                                                    constant.SID, constant.DEFAULT_NBURST,
                                                    self.move_burst,
                                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

                        if self.need_cast:
                            self.tik_instance.vec_conv(self.output_c0, "", input_ub_fp32, input_ub,
                                                       constant.DEFAULT_REPEAT_TIME,
                                                       Constant.FP32_REP_STRIDE, Constant.FP16_REP_STRIDE)

                        self.tik_instance.data_move(output_ub, self.y_gm_fp32[y_gm_idx], constant.SID,
                                                    constant.DEFAULT_NBURST, self.fp32_move_burst,
                                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                        self.tik_instance.vadd(self.output_c0, output_ub, output_ub, input_ub_fp32,
                                               constant.DEFAULT_REPEAT_TIME, constant.BLOCK_STRIDE_ONE,
                                               constant.BLOCK_STRIDE_ONE, constant.BLOCK_STRIDE_ONE,
                                               constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT,
                                               constant.REPEAT_STRIDE_EIGHT)
                        self.tik_instance.data_move(self.y_gm_fp32[y_gm_idx], output_ub,
                                                    constant.SID, constant.DEFAULT_NBURST,
                                                    self.fp32_move_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def _cast_each_task(self, nc):
        """
        Function: cast to each task back to self.dtype.
        """
        cast_ub = self.tik_instance.Tensor(self.dtype, (self.vector_length,), tik.scope_ubuf, "cast_ub")
        cast_ub_fp32 = self.tik_instance.Tensor(self.caculate_dtype, (self.vector_length,),
                                                tik.scope_ubuf, "cast_ub_fp32")
        output_hwc0 = self.tik_instance.Scalar("int64")
        gm_offset = self.tik_instance.Scalar("int64")
        burst = self.tik_instance.Scalar("int64")
        default_burst = self.tik_instance.Scalar("int64")
        nburst = self.tik_instance.Scalar("int64")
        burst_tail = self.tik_instance.Scalar("int64")
        output_hwc0.set_as(self.output_h * self.output_w * self.output_c0)
        gm_offset.set_as(nc * output_hwc0)
        burst.set_as(output_hwc0 * self.fp32_move_byte_num // constant.BLOCK_SIZE)
        default_burst.set_as(constant.VECTOR_BYTE_SIZE // constant.BLOCK_SIZE)
        nburst.set_as(burst // default_burst)
        burst_tail.set_as(burst % default_burst)
        with self.tik_instance.for_range(0, nburst) as line:
            self.tik_instance.data_move(cast_ub_fp32, self.y_gm_fp32[gm_offset],
                                        constant.SID, constant.DEFAULT_NBURST, default_burst, 
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            if self.dtype == "float16":
                self.tik_instance.vec_conv(64, "", cast_ub, cast_ub_fp32, constant.DEFAULT_REPEAT_TIME,
                                           Constant.FP16_REP_STRIDE, Constant.FP32_REP_STRIDE)
            elif self.dtype == "bfloat16":
                self.tik_instance.vec_conv(64, "round", cast_ub, cast_ub_fp32,
                                           constant.DEFAULT_REPEAT_TIME,
                                           Constant.FP16_REP_STRIDE, Constant.FP32_REP_STRIDE)
            self.tik_instance.data_move(self.y_gm[gm_offset], cast_ub,
                                        constant.SID, constant.DEFAULT_NBURST, default_burst // 2, 
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            gm_offset.set_as(gm_offset + self.vector_length)
        with self.tik_instance.if_scope(burst_tail > 0):
            self.tik_instance.data_move(cast_ub_fp32, self.y_gm_fp32[gm_offset],
                                        constant.SID, constant.DEFAULT_NBURST, burst_tail, 
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            if self.dtype == "float16":
                self.tik_instance.vec_conv(64 * burst_tail // default_burst, "", cast_ub, cast_ub_fp32,
                                           constant.DEFAULT_REPEAT_TIME,
                                           Constant.FP16_REP_STRIDE, Constant.FP32_REP_STRIDE)
            elif self.dtype == "bfloat16":
                self.tik_instance.vec_conv(64 * burst_tail // default_burst, "round", cast_ub, cast_ub_fp32,
                                           constant.DEFAULT_REPEAT_TIME,
                                           Constant.FP16_REP_STRIDE, Constant.FP32_REP_STRIDE)
            self.tik_instance.data_move(self.y_gm[gm_offset], cast_ub,
                                        constant.SID, constant.DEFAULT_NBURST, burst_tail // 2, 
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def _clear_ub(self, dtype, length, dst):
        """
        Clear ub tensor to 0.
        """
        if dtype == "float32":
            mask = Constant.FP32_MASK
        else:
            mask = Constant.FP16_MASK
        max_numel_vec_dup_one_loop = Constant.MAX_REPEAT_TIME * mask
        repeat_times = length // max_numel_vec_dup_one_loop
        with self.tik_instance.for_range(0, repeat_times) as loop:
            self.tik_instance.vec_dup(mask, dst[loop * max_numel_vec_dup_one_loop],
                                      0, Constant.MAX_REPEAT_TIME, 8)
        remain = length % max_numel_vec_dup_one_loop
        remain_repeat_times = remain // mask
        remain_mask = remain % mask
        with self.tik_instance.if_scope(remain_repeat_times > 0):
            offset = repeat_times * max_numel_vec_dup_one_loop
            self.tik_instance.vec_dup(mask, dst[offset], 0, remain_repeat_times, 8)
            with self.tik_instance.if_scope(remain_mask > 0):
                remain_mask_offset = mask * remain_repeat_times + offset
                self.tik_instance.vec_dup(remain_mask, dst[remain_mask_offset], 0, 1, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.vec_dup(remain_mask, dst, 0, 1, 0)

    def _data_conv(self, dst, src, mask=64, mode="", num=0, dst_stride=8, src_stride=8):
        """
        tik conv
        """
        dst_offset = self.tik_instance.Scalar("int64", init_value=0)
        src_offset = self.tik_instance.Scalar("int64", init_value=0)

        tensor_size = num
        loop = tensor_size // (mask * Constant.MAX_REPEAT_TIME)

        dst_gap = dst_stride * constant.BLOCK_SIZE // (get_bit_len(dst.dtype) // 8)
        src_gap = src_stride * constant.BLOCK_SIZE // (get_bit_len(src.dtype) // 8)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * Constant.MAX_REPEAT_TIME * dst_gap
                tmp_src_offset = src_offset + index * Constant.MAX_REPEAT_TIME * src_gap
                self.tik_instance.vec_conv(mask, mode, dst[tmp_dst_offset], src[tmp_src_offset],
                                           Constant.MAX_REPEAT_TIME, dst_stride, src_stride)

            dst_offset.set_as(dst_offset + loop * dst_gap * Constant.MAX_REPEAT_TIME)
            src_offset.set_as(src_offset + loop * src_gap * Constant.MAX_REPEAT_TIME)

        repeat_time_last = (tensor_size % (mask * Constant.MAX_REPEAT_TIME)) // mask

        with self.tik_instance.if_scope(repeat_time_last > 0):
            self.tik_instance.vec_conv(mask, mode, dst[dst_offset], src[src_offset], repeat_time_last,
                                       dst_stride, src_stride)
            dst_offset.set_as(dst_offset + repeat_time_last * dst_gap)
            src_offset.set_as(src_offset + repeat_time_last * src_gap)

        last_num = tensor_size % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_conv(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@register_operator("Col2im")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.KERNEL_NAME)
def col2im(x, output_size, y, kernel_size, dilation, padding, stride, kernel_name="col2im"):
    """
    Function: do col2im operation on x, result is y, and y's height/width is value of output_size.

    Parameters:
    ----------
    x : dict 
        dict of x, include shape and dtype, dtype support float32
    output_size : dict 
        dict of output_size, include shape and dtype, dtype support int32
    y : dict 
        dict of y, include shape and dtype, dtype support float32
    kernel_size : tuple
        value of kernel_size, length 2
    dilation : tuple
        value of dilation, length 2
    padding : tuple
        value of padding, length 2
    stride : tuple
        value of stride, length 2
    kernel_name : str
        cce kernel name, default value is "Col2im"
    -------
    """
    op_obj = Col2im(x, output_size, kernel_size, dilation, padding, stride, kernel_name)
    return op_obj.col2im_compute()
