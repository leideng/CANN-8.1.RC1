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
step_interpolate
"""

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik


class Constant:
    """
    The class for constant
    """
    BLOCK = 16


class StepInterpolate:
    """
    Function: store StepInterpolate parameters and compute StepInterpolate
    """

    def __init__(self, i, q, y_i_first, y_i_second, y_q_first, y_q_second, kernel_name="step_interpolate"):
        """
        init the StepInterpolate parameters
        """
        self.dtype_i = i.get("dtype")
        self.dtype_q = q.get("dtype")
        self.i_shape = i["shape"]
        self.q_shape = q["shape"]
        self.tik_instance = tik.Tik()
        self.kernel_name = kernel_name
        # Get the size of UB space in Bytes
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_bytes_size_i = tbe_platform.get_bit_len(self.dtype_i) // Constant.BLOCK
        self.first_shape = self.i_shape[0]
        self.alpha = 3.2

        self.i = None
        self.q = None
        self.y_i_first = None
        self.y_i_second = None
        self.y_q_first = None
        self.y_q_second = None
        self.indices = None

    @staticmethod
    def get_dtype_size(dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2, "int16": 2}
        return dtype_dict.get(dtype)

    @staticmethod
    def cal_alpha(tik_instance, lis):
        """
        cal_alpha
        """
        double_x_ub, cubic_x_ub, coe_x_ub1, coe_x_ub2, coe_x_ub3, coe_x_ub4, coe_x_ub5, res_ub, diff_ub, \
        decimal_ub = lis
        size_per_core = 640
        tik_instance.vec_mul(64, double_x_ub, diff_ub, diff_ub, 30, 8, 8, 8)
        tik_instance.vec_mul(64, cubic_x_ub, diff_ub, double_x_ub, 30, 8, 8, 8)
        tik_instance.vec_muls(64, coe_x_ub1, cubic_x_ub, 1 / 6, 30, 8, 8)
        tik_instance.vec_muls(64, coe_x_ub2, diff_ub, 1 / 6, 30, 8, 8)
        tik_instance.vec_sub(64, res_ub, coe_x_ub1, coe_x_ub2, 30, 8, 8, 8)
        tik_instance.data_move(decimal_ub, res_ub, 0, 3, 80, 0, 240)
        tik_instance.vec_muls(64, coe_x_ub3, double_x_ub, 1 / 2, 30, 8, 8)
        tik_instance.vec_muls(64, coe_x_ub4, cubic_x_ub, 1 / 2, 30, 8, 8)
        tik_instance.vec_sub(64, coe_x_ub5, coe_x_ub3, coe_x_ub4, 30, 8, 8, 8)
        tik_instance.vec_add(64, res_ub, coe_x_ub5, diff_ub, 30, 8, 8, 8)
        tik_instance.data_move(decimal_ub[size_per_core], res_ub, 0, 3, 80, 0, 240)

        tik_instance.vec_muls(64, coe_x_ub2, diff_ub, 1 / 2, 30, 8, 8)
        tik_instance.vec_sub(64, coe_x_ub5, coe_x_ub4, double_x_ub, 30, 8, 8, 8)
        tik_instance.vec_sub(64, coe_x_ub5, coe_x_ub5, coe_x_ub2, 30, 8, 8, 8)
        tik_instance.vec_adds(64, res_ub, coe_x_ub5, 1, 30, 8, 8)
        tik_instance.data_move(decimal_ub[size_per_core * 2], res_ub, 0, 3, 80, 0, 240)
        tik_instance.vec_muls(64, coe_x_ub4, diff_ub, 1 / 3, 30, 8, 8)
        tik_instance.vec_sub(64, coe_x_ub5, coe_x_ub3, coe_x_ub1, 30, 8, 8, 8)
        tik_instance.vec_sub(64, coe_x_ub5, coe_x_ub5, coe_x_ub4, 30, 8, 8, 8)
        tik_instance.data_move(decimal_ub[size_per_core * 3], coe_x_ub5, 0, 3, 80, 0, 240)

    def code_compute(self):
        """
        code_compute
        """
        self.gm_for_data()
        self.cal_code()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.i, self.q],
                                   outputs=[self.y_i_first, self.y_i_second, self.y_q_first, self.y_q_second])
        return self.tik_instance

    def gm_for_data(self):
        """
        gm_for_data
        """
        index = self.first_shape // self.alpha
        shape_indices = (index,)
        shape_y_first = (2, index)
        shape_y_second = (8,)
        self.i = self.tik_instance.Tensor(self.dtype_i, self.i_shape, name="i", scope=tbe_platform.scope_gm)
        self.q = self.tik_instance.Tensor(self.dtype_q, self.q_shape, name="q", scope=tbe_platform.scope_gm)
        lis = [i for i in range(int(index))]
        self.indices = self.tik_instance.Tensor(self.dtype_i, shape_indices, name="indices",
                                                scope=tbe_platform.scope_gm,
                                                init_value=lis)
        self.y_i_first = self.tik_instance.Tensor(self.dtype_i, shape_y_first, name="y_i_first",
                                                  scope=tbe_platform.scope_gm)
        self.y_i_second = self.tik_instance.Tensor(self.dtype_i, shape_y_second, name="y_i_second",
                                                   scope=tbe_platform.scope_gm, is_atomic_add=True)
        self.y_q_first = self.tik_instance.Tensor(self.dtype_i, shape_y_first, name="y_q_first",
                                                  scope=tbe_platform.scope_gm)
        self.y_q_second = self.tik_instance.Tensor(self.dtype_i, shape_y_second, name="y_q_second",
                                                   scope=tbe_platform.scope_gm, is_atomic_add=True)

    def cal_code(self):
        """
        cal_code
        """
        index = int(self.first_shape // self.alpha)
        ele_per_core = index // self.aicore_num
        core_used = index // ele_per_core
        with self.tik_instance.for_range(0, core_used, block_num=core_used) as core_index:
            self.cal_code_core(core_index)

    def ub_for_data(self):
        """
        ub_for_data
        """
        tik_inst = self.tik_instance
        shape_data = (640, 16)
        shape_data_conv = (16, 640)
        data_ub_fp32 = tik_inst.Tensor("float32", shape_data, scope=tbe_platform.scope_ubuf, name="data_ub_fp32")
        data_ub_fp16 = tik_inst.Tensor("float16", shape_data, scope=tbe_platform.scope_ubuf, name="data_ub_fp16")
        data_ub_fp16_conv = tik_inst.Tensor("float16", shape_data_conv, scope=tbe_platform.scope_ubuf,
                                            name="data_ub_fp16_conv")
        block_ub = tik_inst.Tensor("float32", (16,), scope=tbe_platform.scope_ubuf, name="block_ub")
        lis = [data_ub_fp32, data_ub_fp16, data_ub_fp16_conv, block_ub]
        return lis

    def cal_code_with_kernel(self, tik_instance, core_index, decimal_ub):
        """
        cal_code_with_kernel
        """
        data_ub_fp32, data_ub_fp16, data_ub_fp16_conv, block_ub = self.ub_for_data()
        offset = tik_instance.Scalar("int32", name="offset")
        offset.set_as(self.alpha * 640 * core_index - 1)
        self.dup_value(block_ub, 16, 0)
        # for i
        with tik_instance.for_range(0, 5) as i:
            tik_instance.data_move(data_ub_fp32[i * 16], self.i[offset + i * 3], 0, 128, 1, 1, 9)
        tik_instance.vec_conv(64, "none", data_ub_fp16, data_ub_fp32, 160, 4, 8)
        time_2 = 40
        time_3 = 1
        src_list = []
        dst_list = []
        for k in range(Constant.BLOCK):
            src_list.append(data_ub_fp16[time_3 * Constant.BLOCK * k])
            dst_list.append(data_ub_fp16_conv[time_2 * Constant.BLOCK * k])
        tik_instance.vnchwconv(True, False, dst_list, src_list, time_2, 1, Constant.BLOCK)
        tik_instance.data_move(data_ub_fp16, data_ub_fp16_conv, 0, 1, 160, 0, 0)
        tik_instance.data_move(data_ub_fp16[4 * 640], data_ub_fp16_conv[640], 0, 1, 160, 0, 0)
        tik_instance.data_move(data_ub_fp16[8 * 640], data_ub_fp16_conv[2 * 640], 0, 1, 160, 0, 0)
        tik_instance.vec_conv(64, "none", data_ub_fp32, data_ub_fp16, 120, 8, 4)
        tik_instance.vec_mul(64, data_ub_fp32, data_ub_fp32, decimal_ub, 120, 8, 8, 8)
        tik_instance.vec_add(64, data_ub_fp32, data_ub_fp32, data_ub_fp32[4 * 640], 40, 8, 8, 8)
        tik_instance.vec_add(64, data_ub_fp32, data_ub_fp32, data_ub_fp32[2 * 640], 20, 8, 8, 8)
        tik_instance.vec_add(64, data_ub_fp32, data_ub_fp32, data_ub_fp32[1 * 640], 10, 8, 8, 8)
        tik_instance.vec_add(64, data_ub_fp32[8 * 640], data_ub_fp32[8 * 640], data_ub_fp32[10 * 640], 20, 8, 8, 8)
        tik_instance.vec_add(64, data_ub_fp32[8 * 640], data_ub_fp32[8 * 640], data_ub_fp32[9 * 640], 10, 8, 8, 8)
        tik_instance.vcadd(64, block_ub, data_ub_fp32[8 * 640], 10, 1, 1, 8)
        tik_instance.vcadd(10, block_ub, block_ub, 1, 1, 1, 8)
        tik_instance.data_move(self.y_i_first[core_index * 640], data_ub_fp32, 0, 2, 80, 560, 2480)
        tik_instance.set_atomic_add(1)
        tik_instance.data_move(self.y_i_second, block_ub, 0, 1, 1, 0, 0)
        ub_lis = [data_ub_fp32, data_ub_fp16, data_ub_fp16_conv, block_ub]
        self.cal_q(tik_instance, ub_lis, core_index, decimal_ub, offset)
        tik_instance.set_atomic_add(0)

    def cal_q(self, tik_instance, ub_lis, core_index, decimal_ub, offset):
        """
        cal_q
        """
        data_ub_fp32, data_ub_fp16, data_ub_fp16_conv, block_ub = ub_lis
        with tik_instance.for_range(0, 5) as i:
            tik_instance.data_move(data_ub_fp32[i * 16], self.q[offset + i * 3], 0, 128, 1, 1, 9)
        tik_instance.vec_conv(64, "none", data_ub_fp16, data_ub_fp32, 160, 4, 8)
        time_2 = 40
        time_3 = 1
        src_list = []
        dst_list = []
        for k in range(Constant.BLOCK):
            src_list.append(data_ub_fp16[time_3 * Constant.BLOCK * k])
            dst_list.append(data_ub_fp16_conv[time_2 * Constant.BLOCK * k])
        tik_instance.vnchwconv(True, False, dst_list, src_list, time_2, 1, Constant.BLOCK)
        tik_instance.data_move(data_ub_fp16, data_ub_fp16_conv, 0, 1, 160, 0, 0)
        tik_instance.data_move(data_ub_fp16[4 * 640], data_ub_fp16_conv[640], 0, 1, 160, 0, 0)
        tik_instance.data_move(data_ub_fp16[8 * 640], data_ub_fp16_conv[2 * 640], 0, 1, 160, 0, 0)
        tik_instance.vec_conv(64, "none", data_ub_fp32, data_ub_fp16, 120, 8, 4)
        tik_instance.vec_mul(64, data_ub_fp32, data_ub_fp32, decimal_ub, 120, 8, 8, 8)
        tik_instance.vec_add(64, data_ub_fp32, data_ub_fp32, data_ub_fp32[4 * 640], 40, 8, 8, 8)
        tik_instance.vec_add(64, data_ub_fp32, data_ub_fp32, data_ub_fp32[2 * 640], 20, 8, 8, 8)
        tik_instance.vec_add(64, data_ub_fp32, data_ub_fp32, data_ub_fp32[1 * 640], 10, 8, 8, 8)
        tik_instance.vec_add(64, data_ub_fp32[8 * 640], data_ub_fp32[8 * 640], data_ub_fp32[10 * 640], 20, 8, 8, 8)
        tik_instance.vec_add(64, data_ub_fp32[8 * 640], data_ub_fp32[8 * 640], data_ub_fp32[9 * 640], 10, 8, 8, 8)
        tik_instance.vcadd(64, block_ub, data_ub_fp32[8 * 640], 10, 1, 1, 8)
        tik_instance.vcadd(10, block_ub, block_ub, 1, 1, 1, 8)
        tik_instance.data_move(self.y_q_first[core_index * 640], data_ub_fp32, 0, 2, 80, 560, 2480)
        tik_instance.data_move(self.y_q_second, block_ub, 0, 1, 1, 0, 0)

    def dup_value(self, dst, num, dup_value=0, offset=0):
        """
        dup value to ub
        """
        dtype_byte_size = StepInterpolate.get_dtype_size(dst.dtype)
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

    def cal_code_core(self, core_index):
        """
        cal_code_core
        """
        tik_instance = self.tik_instance
        size_per_core = 640
        times = 3
        decimal_ub = tik_instance.Tensor("float32", (12, size_per_core), scope=tbe_platform.scope_ubuf,
                                         name="decimal_ub")
        with tik_instance.new_stmt_scope():
            diff_ub = tik_instance.Tensor("float32", (times, size_per_core), scope=tbe_platform.scope_ubuf,
                                          name="diff_ub")
            with tik_instance.new_stmt_scope():
                indices_ub = tik_instance.Tensor("float32", (times, size_per_core), scope=tbe_platform.scope_ubuf,
                                                 name="indices_ub")
                int32_ub = tik_instance.Tensor("int32", (times, size_per_core),
                                               scope=tbe_platform.scope_ubuf, name="int32_ub")
                tik_instance.data_move(indices_ub, self.indices[core_index * size_per_core], 0, 1, 80, 0, 0)
                diff = tik_instance.Scalar(dtype="float32", init_value=self.alpha / times)
                tik_instance.vec_adds(64, indices_ub[size_per_core], indices_ub, diff, 10, 8, 8)
                tik_instance.vec_adds(64, indices_ub[size_per_core * 2], indices_ub[size_per_core], diff, 10, 8, 8)
                tik_instance.vec_conv(64, "floor", int32_ub, indices_ub, 30, 8, 8)
                tik_instance.vec_conv(64, "none", diff_ub, int32_ub, 30, 8, 8)
                tik_instance.vec_sub(64, diff_ub, indices_ub, diff_ub, 30, 8, 8, 8)
            double_x_ub = tik_instance.Tensor("float32", (times, size_per_core),
                                              scope=tbe_platform.scope_ubuf, name="double_x_ub")
            cubic_x_ub = tik_instance.Tensor("float32", (times, size_per_core),
                                             scope=tbe_platform.scope_ubuf, name="cubic_x_ub")
            coe_x_ub1 = tik_instance.Tensor("float32", (times, size_per_core),
                                            scope=tbe_platform.scope_ubuf, name="coe_x_ub1")  # 1/6 x3
            coe_x_ub2 = tik_instance.Tensor("float32", (times, size_per_core),
                                            scope=tbe_platform.scope_ubuf, name="coe_x_ub2")  # 1/6 x 1/2 x
            coe_x_ub3 = tik_instance.Tensor("float32", (times, size_per_core),
                                            scope=tbe_platform.scope_ubuf, name="coe_x_ub3")  # 1/2 x2
            coe_x_ub4 = tik_instance.Tensor("float32", (times, size_per_core),
                                            scope=tbe_platform.scope_ubuf, name="coe_x_ub4")  # 1/2 x3, 1/3 x
            coe_x_ub5 = tik_instance.Tensor("float32", (times, size_per_core),
                                            scope=tbe_platform.scope_ubuf, name="coe_x_ub5")
            res_ub = tik_instance.Tensor("float32", (times, size_per_core), scope=tbe_platform.scope_ubuf,
                                         name="res_ub")
            lis = [double_x_ub, cubic_x_ub, coe_x_ub1, coe_x_ub2, coe_x_ub3, coe_x_ub4, coe_x_ub5, res_ub, diff_ub,
                   decimal_ub]
            StepInterpolate.cal_alpha(tik_instance, lis)
        with tik_instance.new_stmt_scope():
            self.cal_code_with_kernel(tik_instance, core_index, decimal_ub)


def step_interpolate(i, q, y_i_first, y_i_second, y_q_first, y_q_second, kernel_name="step_interpolate"):
    """
    the main function of step_interpolate
    """
    code_instance = StepInterpolate(i, q, y_i_first, y_i_second, y_q_first, y_q_second, kernel_name)
    tik_instance = code_instance.code_compute()
    return tik_instance
