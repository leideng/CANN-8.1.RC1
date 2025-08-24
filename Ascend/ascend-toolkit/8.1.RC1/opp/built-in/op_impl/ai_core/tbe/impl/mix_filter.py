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
mix_filter
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform as cce


class Constant:
    """
    The class for constant
    """
    RESERVE_SIZE = 16 * 1024
    BLOCK = 16
    MASK = 128


class MixFilter:
    """
    Function: store MixFilter parameters  and compute MixFilter
    """
    def __init__(self, x, sin_x, cos_x, kernel, y_sin, y_cos, y_i, y_q, y_sum, kernel_name="mix_filter"):
        """
        init the MixFilter parameters
        """
        self.dtype_x = x.get("dtype")
        self.x_shape = x["shape"]
        self.sin_x_shape = sin_x["shape"]
        self.dtype_sin_x = sin_x.get("dtype")
        self.cos_x_shape = cos_x["shape"]
        self.dtype_cos_x = cos_x.get("dtype")
        self.kernel_shape = kernel["shape"]
        self.dtype_kernel = kernel.get("dtype")
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        block_bite_size = 32
        # Get the size of UB space in Bytes
        self.ub_size_bytes = cce.get_soc_spec(cce.UB_SIZE)
        self.aicore_num = cce.get_soc_spec(cce.CORE_NUM)
        self.dtype_bytes_size_x = cce.get_bit_len(self.dtype_x) // Constant.BLOCK
        self.x_each_block = block_bite_size // self.dtype_bytes_size_x

        self.vector_mask_max_x = Constant.BLOCK * self.x_each_block
        self.sync_workspace = self.tik_instance.Tensor('int64', (4 * self.aicore_num,), tik.scope_gm,
                                                       'sync_workspace', is_workspace=True, is_atomic_add=True)
        self.sync_workspace1 = self.tik_instance.Tensor('int64', (4 * self.aicore_num,), tik.scope_gm,
                                                       'sync_workspace1', is_workspace=True, is_atomic_add=True)
        self.x = None
        self.indices = None
        self.y_en = None
        self.y_cn =  None
        self.lmn0 =  None
        self.y_sin = None
        self.y_cos = None
        self.y_i = None
        self.y_q = None
        self.y_sum = None
        self.y_mid_shape, self.sin_x, self.cos_x = None, None, None
        self.kernel = None


    @staticmethod
    def get_dtype_size(dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2, "int16": 2}
        return dtype_dict.get(dtype)

    def track_compute(self):
        """
        track_compute
        """
        self.gm_for_data()
        self.cal_track()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.x, self.sin_x, self.cos_x, self.kernel],
                                   outputs=[self.y_sin, self.y_cos, self.y_i, self.y_q, self.y_sum])
        return self.tik_instance

    def gm_for_data(self):
        """
        gm_for_data
        """
        offset = 14
        self.y_mid_shape = [((self.x_shape[0] + offset) // Constant.BLOCK + 1) * Constant.BLOCK]
        self.x = self.tik_instance.Tensor(self.dtype_x, self.x_shape, name="x", scope=tik.scope_gm)
        self.sin_x = self.tik_instance.Tensor(self.dtype_sin_x, self.sin_x_shape, name="sin_x",
                                                scope=tik.scope_gm)
        self.cos_x = self.tik_instance.Tensor(self.dtype_cos_x, self.cos_x_shape, name="cos_x",
                                                scope=tik.scope_gm)
        self.kernel = self.tik_instance.Tensor(self.dtype_kernel, self.kernel_shape, name="kernel",
                                                scope=tik.scope_gm)
        self.y_sin = self.tik_instance.Tensor(self.dtype_x, self.y_mid_shape, name="y_sin", scope=tik.scope_gm,
                                              is_atomic_add=True)
        self.y_cos = self.tik_instance.Tensor(self.dtype_x, self.y_mid_shape, name="y_cos", scope=tik.scope_gm,
                                              is_atomic_add=True)
        self.y_i = self.tik_instance.Tensor(self.dtype_x, self.x_shape, name="y_i", scope=tik.scope_gm)
        self.y_q = self.tik_instance.Tensor(self.dtype_x, self.x_shape, name="y_q", scope=tik.scope_gm)
        self.y_sum = self.tik_instance.Tensor("float32", (Constant.BLOCK,), name="y_sum", scope=tik.scope_gm,
                                              is_atomic_add=True)

    def cal_track(self):
        """
        cal_en
        """
        ele_per_core = self.x_shape[0] // self.aicore_num
        core_used = self.x_shape[0] // ele_per_core
        with self.tik_instance.for_range(0, core_used, block_num=core_used) as core_index:
            self.cal_track_core(core_index)

    def ub_for_data(self):
        """
        ub_for_data
        """
        tik_inst = self.tik_instance
        shape_kernel = (16,)
        shape_x = (2048, 16)
        shape_uint = (128,)
        kernel_ub = tik_inst.Tensor("int32", shape_kernel, scope=tik.scope_ubuf, name="kernel_ub")
        conv_cos_ub = tik_inst.Tensor("float16", shape_x, scope=tik.scope_ubuf, name="conv_cos_ub")
        conv_sin_ub = tik_inst.Tensor("float16", shape_x, scope=tik.scope_ubuf, name="conv_sin_ub")
        conv_ub = tik_inst.Tensor("int16", shape_uint, scope=tik.scope_ubuf, name="conv_ub")
        negative_one_ub = tik_inst.Tensor("float16", shape_uint, scope=tik.scope_ubuf, name="negative_one_ub")
        positive_one_ub = tik_inst.Tensor("float16", shape_uint, scope=tik.scope_ubuf, name="positive_one_ub")
        zero_ub = tik_inst.Tensor("float16", shape_uint, scope=tik.scope_ubuf, name="zero_ub")
        zero_ub_int16 = tik_inst.Tensor("int16", shape_uint, scope=tik.scope_ubuf, name="zero_ub_int16")
        sum_ub = tik_inst.Tensor("float32", shape_kernel, scope=tik.scope_ubuf, name="sum_ub")
        lis = [kernel_ub, conv_cos_ub, conv_sin_ub, conv_ub, negative_one_ub, positive_one_ub, zero_ub,
               zero_ub_int16, sum_ub]
        return lis

    def cal_track_with_kernel(self, tik_instance, core_index):
        """
        cal_track_with_kernel
        """
        kernel_ub, conv_cos_ub, conv_sin_ub, conv_ub, negative_one_ub, positive_one_ub, zero_ub, \
            zero_ub_int16, sum_ub = self.ub_for_data()
        self.dup_value(negative_one_ub, 128, -1)
        self.dup_value(positive_one_ub, 128, 1)
        self.dup_value(zero_ub, 128, 0)
        self.dup_value(zero_ub_int16, 16, 0)
        with tik_instance.for_range(0, 16) as i:
            tik_instance.data_move(conv_cos_ub[i * 16], self.y_cos[core_index * 2048 + i], 0, 128, 1, 0, 15)
        with tik_instance.for_range(0, 16) as i:
            tik_instance.data_move(conv_sin_ub[i * 16], self.y_sin[core_index * 2048 + i], 0, 128, 1, 0, 15)
        tik_instance.data_move(kernel_ub, self.kernel, 0, 1, 1, 0, 0)
        kernel_ub[15].set_as(0)
        kernel_ub_int16 = kernel_ub.reinterpret_cast_to("int16")
        tik_instance.vor(16, conv_ub, kernel_ub_int16, zero_ub_int16, 8, 1, 1, 1, 1, 0, 0)
        conv_ub_fp16 = conv_ub.reinterpret_cast_to("float16")
        tik_instance.vec_mul(Constant.MASK, conv_cos_ub, conv_cos_ub, conv_ub_fp16, 255, 8, 8, 0)
        tik_instance.vec_mul(Constant.MASK, conv_cos_ub[255 * Constant.MASK],
                             conv_cos_ub[255 * Constant.MASK], conv_ub_fp16, 1, 8, 8, 0)
        tik_instance.vec_mul(Constant.MASK, conv_sin_ub, conv_sin_ub, conv_ub_fp16, 255, 8, 8, 0)
        tik_instance.vec_mul(Constant.MASK, conv_sin_ub[255 * Constant.MASK],
                             conv_sin_ub[255 * Constant.MASK], conv_ub_fp16, 1, 8, 8, 0)
        tik_instance.vcgadd(Constant.MASK, conv_sin_ub, conv_sin_ub, 255, 1, 1, 8)
        tik_instance.vcgadd(Constant.MASK, conv_sin_ub[255 * 8], conv_sin_ub[255 * Constant.MASK], 1, 1, 1, 8)
        tik_instance.vcgadd(Constant.MASK, conv_sin_ub[2048], conv_cos_ub, 255, 1, 1, 8)
        tik_instance.vcgadd(Constant.MASK, conv_sin_ub[2048 + 255 * 8],
                            conv_cos_ub[255 * Constant.MASK], 1, 1, 1, 8)
        with tik_instance.for_range(0, 16) as i:
            cmp_mask = tik_instance.vcmp_gt(Constant.MASK, conv_sin_ub[2048 + i * 128], zero_ub, 1, 1)
            tik_instance.vsel(Constant.MASK, 0, conv_cos_ub[i * 128], cmp_mask, positive_one_ub,
                              negative_one_ub, 1, 1, 1, 1, 8, 8, 8)
        with tik_instance.for_range(0, 16) as i:
            cmp_mask = tik_instance.vcmp_gt(Constant.MASK, conv_sin_ub[i * 128], zero_ub, 1, 1)
            tik_instance.vsel(Constant.MASK, 0, conv_cos_ub[2048 + i * 128], cmp_mask, positive_one_ub, negative_one_ub,
                              1, 1, 1, 1, 8, 8, 8)
        tik_instance.vec_mul(Constant.MASK, conv_cos_ub, conv_cos_ub, conv_sin_ub, 32, 8, 8, 8)
        tik_instance.vec_sub(Constant.MASK, conv_cos_ub, conv_cos_ub, conv_cos_ub[2048], 16, 8, 8, 8)
        tik_instance.vcadd(Constant.MASK, conv_cos_ub, conv_cos_ub, 16, 1, 1, 8)
        tik_instance.vcadd(16, conv_cos_ub, conv_cos_ub, 1, 1, 1, 8)
        tik_instance.vconv(16, "none", sum_ub, conv_cos_ub, 1, 1, 1, 2, 1)
        tik_instance.data_move(self.y_q[core_index * 2048], conv_sin_ub, 0, 1, 128, 0, 0)
        tik_instance.data_move(self.y_i[core_index * 2048], conv_sin_ub[2048], 0, 1, 128, 0, 0)
        tik_instance.set_atomic_add(1)
        tik_instance.data_move(self.y_sum, sum_ub, 0, 1, 2, 0, 0)
        tik_instance.set_atomic_add(0)

    def dup_value(self, dst, num, dup_value=0, offset=0):
        """
        dup value to ub
        """
        dtype_byte_size = MixFilter.get_dtype_size(dst.dtype)
        mask = 256 // dtype_byte_size
        stride = 8
        loop = num // (mask * 255)
        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
            offset += loop * mask * 255

        repeat_time = (num % (mask * 255)) // mask
        if repeat_time > 0:
            self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
            offset += repeat_time * mask
        last_num = num % mask
        if last_num > 0:
            self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, stride)

    def cal_track_core(self, core_index):
        """
        cal_track_core
        """
        tik_instance = self.tik_instance
        size_per_core = 2048
        offset = 7
        with tik_instance.new_stmt_scope():
            x_ub = tik_instance.Tensor("float16", (size_per_core,), scope=tik.scope_ubuf, name="x_ub")
            cos_ub = tik_instance.Tensor("float16", (size_per_core,), scope=tik.scope_ubuf, name="cos_ub")
            sin_ub = tik_instance.Tensor("float16", (size_per_core,), scope=tik.scope_ubuf, name="sin_ub")
            tik_instance.data_move(x_ub, self.x[core_index * size_per_core], 0, 1, 128, 0, 0)
            tik_instance.data_move(cos_ub, self.cos_x[core_index * size_per_core], 0, 1, 128, 0, 0)
            tik_instance.vec_mul(Constant.MASK, cos_ub, cos_ub, x_ub, 16, 8, 8, 8)
            tik_instance.data_move(self.y_cos[offset + core_index * size_per_core], cos_ub, 0, 1, 128, 0, 0)
            tik_instance.block_barrier(self.sync_workspace)
            tik_instance.data_move(sin_ub, self.sin_x[core_index * size_per_core], 0, 1, 128, 0, 0)
            tik_instance.vec_mul(Constant.MASK, sin_ub, sin_ub, x_ub, 16, 8, 8, 8)
            tik_instance.data_move(self.y_sin[offset + core_index * size_per_core], sin_ub, 0, 1, 128, 0, 0)
            tik_instance.block_barrier(self.sync_workspace1)
        with tik_instance.new_stmt_scope():
            self.cal_track_with_kernel(tik_instance, core_index)


def mix_filter(x, sin_x, cos_x, kernel, y_sin, y_cos, y_i, y_q, y_sum, kernel_name="mix_filter"):
    """
    the main function of mix_filter
    """
    track_instance = MixFilter(x, sin_x, cos_x, kernel, y_sin, y_cos, y_i, y_q, y_sum, kernel_name)
    tik_instance = track_instance.track_compute()
    return tik_instance
