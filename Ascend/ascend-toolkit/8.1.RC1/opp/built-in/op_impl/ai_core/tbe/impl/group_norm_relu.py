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
group_norm_relu.py
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import PlatformApi
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_platform

SUPPORT_SHAPE = (
    (64, 56), (64, 120), (64, 128),
    (128, 28), (128, 56), (128, 60), (128, 64), (128, 120), (128, 128),
    (256, 14), (256, 28), (256, 30), (256, 32), (256, 56), (256, 60), (256, 64), (256, 120), (256, 128),
    (512, 7), (512, 14), (512, 15), (512, 16), (512, 28), (512, 30), (512, 32), (512, 60), (512, 64),
    (1024, 14), (1024, 30), (1024, 32),
    (2048, 7), (2048, 15), (2048, 16))


# 'pylint: disable=unused-argument,unused-variable,too-many-arguments,too-many-locals
def check_supported(input_0, input_1, input_2, output_0, num_groups, eps=0.00001, kernel_name="group_norm_relu"):
    """
    check_supported
    """
    soc_version = PlatformApi.get_soc_spec(PlatformApi.SHORT_SOC_VERSION)
    is_support_vgather = tbe_platform.api_check_support("tik.vgather", "float32")
    if not is_support_vgather:
        return False, "not support soc version"
    support_version = ("Ascend310P",)
    if soc_version not in support_version:
        return False, "not support short soc version"
    support_dtype = ("float16", "float32")
    input_params_all = (input_0, input_1, input_2, output_0)
    for input_params in input_params_all:
        if input_params.get("dtype").lower() not in support_dtype:
            return False, "not support data dtype"
    input_format = input_0.get("format").upper()
    if input_format == "NCHW":
        input_shape = tuple(input_0.get("shape"))
        output_shape = tuple(output_0.get("shape"))
        weight_shape = tuple(input_1.get("shape"))
        bias_shape = tuple(input_2.get("shape"))
    elif input_format == "NC1HWC0" and input_0.get("ori_format").upper() == "NCHW":
        input_shape = tuple(input_0.get("ori_shape"))
        output_shape = tuple(output_0.get("ori_shape"))
        weight_shape = tuple(input_1.get("ori_shape"))
        bias_shape = tuple(input_2.get("ori_shape"))
    else:
        return False, "not support data format"
    if input_shape[0] <= 0 or len(input_shape) != 4 or input_shape[1:3] not in SUPPORT_SHAPE \
            or input_shape[2] != input_shape[3]:
        return False, "not support x shape"
    if output_shape != input_shape:
        return False, "not support y shape"
    channel_num = input_shape[1]
    if (len(weight_shape) != 1 or weight_shape[0] != channel_num) and \
            (len(weight_shape) != 3 or weight_shape != (channel_num, 1, 1)):
        return False, "not support weight shape"
    if (len(bias_shape) != 1 or bias_shape[0] != channel_num) and \
            (len(bias_shape) != 3 or bias_shape != (channel_num, 1, 1)):
        return False, "not support bias shape"
    if not isinstance(num_groups, int) or num_groups != 32:
        return False, "not support num_groups"
    if eps < 0:
        return False, "not support eps"
    return True, ""


def ceil_div(dividend, divisor):
    result = (dividend + divisor - 1) // divisor
    return result


def get_align_num(data_num, align_num):
    return ceil_div(data_num, align_num) * align_num


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class GroupNormBase:
    """
    GroupNormBase
    """

    # 'pylint: disable=too-many-arguments,too-many-locals
    def __init__(self, n_num, c_num, d_num, num_groups, eps, data_type, kernel_name):
        self.n_num = n_num
        self.c_num = c_num
        self.c0_num = 16
        self.c1_num = self.c_num // self.c0_num
        self.d_num = d_num
        self.num_groups = num_groups
        self.eps = eps
        self.kernel_name = kernel_name

        self.tik_inst = tik.Tik()
        self.ub_size = PlatformApi.get_soc_spec(PlatformApi.UB_SIZE)
        self.block_size = 32
        self.input_type = data_type
        self.input_size, self.input_block_num, self.input_repeat_num = self._get_type_const(self.input_type)
        self.res_type = "float32"
        self.res_size, self.res_block_num, self.res_repeat_num = self._get_type_const(self.res_type)

        self._init_gm_tensor()

    def _init_gm_tensor(self):
        input_shape_0 = (self.n_num, self.c1_num, self.d_num, self.c0_num)
        input_shape_1 = (self.c_num,)
        output_shape_0 = (self.n_num, self.c1_num, self.d_num, self.c0_num)
        self.input_0 = self.tik_inst.Tensor(self.input_type, input_shape_0, tik.scope_gm, "input_0")
        self.input_1 = self.tik_inst.Tensor(self.input_type, input_shape_1, tik.scope_gm, "input_1")
        self.input_2 = self.tik_inst.Tensor(self.input_type, input_shape_1, tik.scope_gm, "input_2")
        self.output_0 = self.tik_inst.Tensor(self.input_type, output_shape_0, tik.scope_gm, "output_0")

    def _get_type_const(self, data_type):
        data_size_all = {"float16": 2, "float32": 4}
        data_size = data_size_all.get(data_type)
        block_data_num = self.block_size // data_size
        repeat_data_num = block_data_num * 8
        return data_size, block_data_num, repeat_data_num

    @staticmethod
    def _get_core_loop_info(batch_num, core_num_all):
        core_batch_num = ceil_div(batch_num, core_num_all)
        core_num_use = ceil_div(batch_num, core_batch_num)
        last_batch_num = batch_num - core_batch_num * (core_num_use - 1)
        return core_num_use, core_batch_num, last_batch_num

    @staticmethod
    def _get_loop_info(data_num, loop_data_num):
        loop_times = ceil_div(data_num, loop_data_num)
        last_data_num = data_num - loop_data_num * (loop_times - 1)
        return loop_times, last_data_num

    def compute(self):
        """
        start compute
        """
        core_num_all = PlatformApi.get_soc_spec(PlatformApi.CORE_NUM)
        core_num_use, core_batch_num, last_batch_num = self._get_core_loop_info(self.n_num, core_num_all)
        if last_batch_num != core_batch_num:
            with self.tik_inst.for_range(0, core_num_use, block_num=core_num_use) as core_index:
                core_start_index = core_index * core_batch_num
                with self.tik_inst.if_scope(core_index != core_num_use - 1):
                    self._compute_each_core(core_start_index, core_batch_num)
                with self.tik_inst.else_scope():
                    self._compute_each_core(core_start_index, last_batch_num)
        else:
            with self.tik_inst.for_range(0, core_num_use, block_num=core_num_use) as core_index:
                core_start_index = core_index * core_batch_num
                self._compute_each_core(core_start_index, core_batch_num)
        self.tik_inst.BuildCCE(
            inputs=[self.input_0, self.input_1, self.input_2],
            outputs=[self.output_0],
            kernel_name=self.kernel_name)

    def _compute_each_core(self, start_index, batch_num):
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            self._compute_each_batch(start_index + batch_index)

    def _count_relu(self, result_data, mask):
        self._start_tik_compute(result_data.size, mask, self._tik_maxs, (result_data, result_data, 0))

    def _start_tik_compute(self, data_num, mask, func, args=None, begin_index=0):
        repeat_num = data_num // mask
        last_num = data_num - repeat_num * mask
        max_repeat = 255
        loop_times, last_repeat = self._get_loop_info(repeat_num, max_repeat)
        for loop_index in range(loop_times):
            start_index = loop_index * max_repeat * mask + begin_index
            if loop_index != (loop_times - 1):
                func(mask, start_index, max_repeat, args)
            else:
                func(mask, start_index, last_repeat, args)
        if last_num != 0:
            start_index = repeat_num * mask + begin_index
            func(last_num, start_index, 1, args)

    def _tik_vmul(self, mask, start_index, repeat_num, args):
        dst, src0, src1 = args
        ori_size = dst.size
        dst_flatten = dst.reshape((ori_size,))
        src0_flatten = src0.reshape((ori_size,))
        src1_flatten = src1.reshape((ori_size,))
        self.tik_inst.vmul(mask, dst_flatten[start_index], src0_flatten[start_index], src1_flatten[start_index],
                           repeat_num, 1, 1, 1, 8, 8, 8)

    def _tik_vmul_broadcast(self, mask, start_index, repeat_num, args):
        dst, src0, src1 = args
        dst_flatten = dst.reshape((dst.size,))
        src0_flatten = src0.reshape((src0.size,))
        src1_flatten = src1.reshape((src1.size,))
        self.tik_inst.vmul(mask, dst_flatten[start_index], src0_flatten[start_index], src1_flatten,
                           repeat_num, 1, 1, 1, 8, 8, 0)

    def _tik_vadd_broadcast(self, mask, start_index, repeat_num, args):
        dst, src0, src1 = args
        dst_flatten = dst.reshape((dst.size,))
        src0_flatten = src0.reshape((src0.size,))
        src1_flatten = src1.reshape((src1.size,))
        self.tik_inst.vadd(mask, dst_flatten[start_index], src0_flatten[start_index], src1_flatten,
                           repeat_num, 1, 1, 1, 8, 8, 0)

    def _tik_maxs(self, mask, start_index, repeat_num, args):
        dst, src0, src1 = args
        ori_size = dst.size
        dst_flatten = dst.reshape((ori_size,))
        src0_flatten = src0.reshape((ori_size,))
        self.tik_inst.vmaxs(mask, dst_flatten[start_index], src0_flatten[start_index], src1,
                            repeat_num, 1, 1, 8, 8)

    def _tik_vconv(self, mask, start_index, repeat_num, args):
        dst, src0 = args
        dst = dst.reshape((dst.size,))
        src0 = src0.reshape((src0.size,))
        if src0.dtype == self.input_type and dst.dtype == self.res_type:
            self.tik_inst.vconv(mask, "", dst[start_index], src0[start_index], repeat_num, 1, 1, 8, 4)
        if src0.dtype == self.res_type and dst.dtype == self.input_type:
            self.tik_inst.vconv(mask, "", dst[start_index], src0[start_index], repeat_num, 1, 1, 4, 8)

    def _tik_dup(self, mask, start_index, repeat_num, args):
        dst, dup_data = args
        ori_size = dst.size
        dst_flatten = dst.reshape((ori_size,))
        self.tik_inst.vector_dup(mask, dst_flatten[start_index], dup_data, repeat_num, 1, 8)

    def _broadcast_c(self, input_data, result_data):
        block_num = self.c0_num // self.res_block_num
        self.tik_inst.data_move(result_data, input_data, 0, 1, block_num, 0, 0)
        while block_num < 8:
            self.tik_inst.data_move(result_data[block_num * self.res_block_num, ], result_data, 0, 1, block_num, 0, 0)
            block_num *= 2

    def _count_std_and_mean(self, mask, group_variance_ub, group_mean_ub, group_mean_square_ub, repeat_num):
        self.tik_inst.vmul(mask, group_variance_ub, group_mean_ub, group_mean_ub,
                           repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(mask, group_variance_ub, group_mean_square_ub, group_variance_ub,
                           repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vabs(mask, group_variance_ub, group_variance_ub,
                           repeat_num, 1, 1, 8, 8)

        self.tik_inst.vmuls(mask, group_mean_ub, group_mean_ub, -1,
                            repeat_num, 1, 1, 8, 8)
        self.tik_inst.vadds(mask, group_variance_ub, group_variance_ub, self.eps,
                            repeat_num, 1, 1, 8, 8)


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class GroupNormTilingC1(GroupNormBase):
    """
    GroupNormTilingC1
    """

    # 'pylint: disable=too-many-arguments,too-many-locals
    def __init__(self, n_num, c_num, d_num, num_groups, eps, data_type, kernel_name):
        super(GroupNormTilingC1, self).__init__(n_num, c_num, d_num, num_groups, eps, data_type, kernel_name)
        self.c_groups = self.c_num // self.num_groups
        self.c1_groups = self.c_groups // self.c0_num
        self.d_groups = self.c_groups * self.d_num
        self.c0_repeat = self.res_repeat_num // self.c0_num
        self.d_num_align = get_align_num(self.d_num, self.c0_repeat)
        self.d_groups_align = self.c_groups * self.d_num_align
        self.d_num_pad = self.d_num_align - self.d_num

        self.stand_repeat = self.d_num_align // self.c0_repeat
        self.sum_repeat = self.d_groups_align // self.res_repeat_num

    def _compute_each_batch(self, n_index):
        loop_group_num = 1
        loop_times, last_group_num = self._get_loop_info(self.num_groups, loop_group_num)
        if last_group_num != loop_group_num:
            with self.tik_inst.for_range(0, loop_times) as loop_index:
                c1_index = loop_index * loop_group_num * self.c1_groups
                ub_tensor_all = self._init_ub_tensor(loop_group_num)
                with self.tik_inst.if_scope(loop_index != loop_times - 1):
                    self._compute_each_loop(ub_tensor_all, n_index, c1_index, loop_group_num)
                with self.tik_inst.else_scope():
                    self._compute_each_loop(ub_tensor_all, n_index, c1_index, last_group_num)
        else:
            with self.tik_inst.for_range(0, loop_times) as loop_index:
                c1_index = loop_index * loop_group_num * self.c1_groups
                ub_tensor_all = self._init_ub_tensor(loop_group_num)
                self._compute_each_loop(ub_tensor_all, n_index, c1_index, loop_group_num)

    def _init_ub_tensor(self, group_num):
        data_ub_shape = (group_num, self.c1_groups, self.d_num_align, self.c0_num)
        mean_ub_shape = (group_num, self.res_block_num)
        weight_ub_shape = (group_num, self.c1_groups, self.c0_num)
        temp_ub_size = max(self.res_repeat_num,
                           self.sum_repeat,
                           self.c1_groups * self.c0_num,
                           group_num * self.res_block_num)
        temp_ub_shape = (temp_ub_size,)
        ub_tensor_all = {
            "input_data_ub":
                self.tik_inst.Tensor(self.res_type, data_ub_shape, name="input_data_ub", scope=tik.scope_ubuf),
            "group_mean_ub":
                self.tik_inst.Tensor(self.res_type, mean_ub_shape, name="group_mean_ub", scope=tik.scope_ubuf),
            "group_mean_square_ub":
                self.tik_inst.Tensor(self.res_type, mean_ub_shape, name="group_mean_square_ub", scope=tik.scope_ubuf),
            "weight_ub":
                self.tik_inst.Tensor(self.res_type, weight_ub_shape, name="weight_ub", scope=tik.scope_ubuf),
            "bias_ub":
                self.tik_inst.Tensor(self.res_type, weight_ub_shape, name="bias_ub", scope=tik.scope_ubuf),
            "temp_ub":
                self.tik_inst.Tensor(self.res_type, temp_ub_shape, name="temp_ub", scope=tik.scope_ubuf),
        }
        data_temp_shape = (self.res_size // self.input_size * group_num *
                           self.c1_groups * self.d_num_align * self.c0_num,)
        ub_tensor_all["data_temp_ub"] = self.tik_inst.Tensor(self.input_type, data_temp_shape,
                                                             tik.scope_ubuf, "data_temp_ub")
        if self.input_type != self.res_type:
            ub_tensor_all["weight_ub_fp16"] = self.tik_inst.Tensor(self.input_type, weight_ub_shape,
                                                                   tik.scope_ubuf, "weight_ub_fp16")
            ub_tensor_all["bias_ub_fp16"] = self.tik_inst.Tensor(self.input_type, weight_ub_shape,
                                                                 tik.scope_ubuf, "bias_ub_fp16")
        return ub_tensor_all

    def _compute_each_loop(self, ub_tensor_all, n_index, c1_index, group_num):
        self._data_move_in(ub_tensor_all, n_index, c1_index, group_num)
        self._vconv_data(ub_tensor_all)
        self._pad_input_data(ub_tensor_all, group_num)
        self._count_sum(ub_tensor_all, group_num)
        self._count_rec_std_ne_mean(ub_tensor_all, group_num)
        self._count_weight(ub_tensor_all, group_num)
        self._stand_data(ub_tensor_all, group_num)
        self._data_move_out(ub_tensor_all, n_index, c1_index, group_num)

    def _data_move_in(self, ub_tensor_all, n_index, c1_index, group_num):
        if self.input_type != self.res_type:
            input_data_ub = ub_tensor_all.get("data_temp_ub")
        else:
            input_data_ub = ub_tensor_all.get("input_data_ub")
        if self.d_num_pad != 0:
            nburst = group_num * self.c1_groups
            data_block_num = self.d_num * self.c0_num // self.input_block_num
            dst_stride = self.d_num_pad * self.c0_num // self.input_block_num
            self.tik_inst.data_move(input_data_ub, self.input_0[n_index, c1_index, 0, 0],
                                    0, nburst, data_block_num, 0, dst_stride)
        else:
            data_block_num = self.d_groups // self.input_block_num
            self.tik_inst.data_move(input_data_ub, self.input_0[n_index, c1_index, 0, 0], 0, 1, data_block_num, 0, 0)
        weight_block_num = group_num * self.c_groups // self.input_block_num
        if self.input_type != self.res_type:
            weight_ub = ub_tensor_all.get("weight_ub_fp16")
            bias_ub = ub_tensor_all.get("bias_ub_fp16")
        else:
            weight_ub = ub_tensor_all.get("weight_ub")
            bias_ub = ub_tensor_all.get("bias_ub")
        self.tik_inst.data_move(weight_ub, self.input_1[c1_index * self.c0_num], 0, 1, weight_block_num, 0, 0)
        self.tik_inst.data_move(bias_ub, self.input_2[c1_index * self.c0_num], 0, 1, weight_block_num, 0, 0)

    def _vconv_data(self, ub_tensor_all):
        if self.input_type != self.res_type:
            data_temp_ub = ub_tensor_all.get("data_temp_ub")
            input_data_ub = ub_tensor_all.get("input_data_ub")
            self._start_tik_compute(input_data_ub.size, self.res_repeat_num,
                                    self._tik_vconv, (input_data_ub, data_temp_ub))
            weight_ub_fp16 = ub_tensor_all.get("weight_ub_fp16")
            bias_ub_fp16 = ub_tensor_all.get("bias_ub_fp16")
            weight_ub = ub_tensor_all.get("weight_ub")
            bias_ub = ub_tensor_all.get("bias_ub")
            self._start_tik_compute(weight_ub.size, self.res_repeat_num,
                                    self._tik_vconv, (weight_ub, weight_ub_fp16))
            self._start_tik_compute(bias_ub.size, self.res_repeat_num,
                                    self._tik_vconv, (bias_ub, bias_ub_fp16))

    def _pad_input_data(self, ub_tensor_all, group_num):
        if self.d_num_pad != 0:
            input_data_ub = ub_tensor_all.get("input_data_ub")
            mask = self.d_num_pad * self.c0_num
            repeat_num = group_num * self.c1_groups
            if repeat_num == 1:
                dst_rep_stride = 0
            else:
                dst_rep_stride = self.d_num_align * self.c0_num // self.res_block_num
            if dst_rep_stride > 255:
                with self.tik_inst.for_range(0, group_num) as group_index:
                    with self.tik_inst.for_range(0, self.c1_groups) as c1_index:
                        self.tik_inst.vector_dup(mask, input_data_ub[group_index, c1_index, self.d_num, 0], 0, 1, 1, 0)
            else:
                self.tik_inst.vector_dup(mask, input_data_ub[0, 0, self.d_num, 0], 0, repeat_num, 1, dst_rep_stride)

    def _count_sum(self, ub_tensor_all, group_num):
        input_data_ub = ub_tensor_all.get("input_data_ub")
        group_mean_ub = ub_tensor_all.get("group_mean_ub")
        group_mean_square_ub = ub_tensor_all.get("group_mean_square_ub")
        data_temp_ub = ub_tensor_all.get("data_temp_ub")
        temp_ub = ub_tensor_all.get("temp_ub")

        mask = self.res_repeat_num
        if self.input_type != self.res_type:
            data_temp_ub = data_temp_ub.reinterpret_cast_to(self.res_type)
        data_temp_ub = data_temp_ub.reshape(input_data_ub.shape)
        self._start_tik_compute(input_data_ub.size, mask, self._tik_vmul, (data_temp_ub, input_data_ub, input_data_ub))

        with self.tik_inst.for_range(0, group_num) as group_index:
            self.tik_inst.vec_reduce_add(mask,
                                         group_mean_ub[group_index],
                                         input_data_ub[group_index],
                                         temp_ub, self.sum_repeat, 8)
            self.tik_inst.vec_reduce_add(mask,
                                         group_mean_square_ub[group_index],
                                         data_temp_ub[group_index],
                                         temp_ub, self.sum_repeat, 8)

    def _count_rec_std_ne_mean(self, ub_tensor_all, group_num):
        group_mean_ub = ub_tensor_all.get("group_mean_ub")
        group_mean_square_ub = ub_tensor_all.get("group_mean_square_ub")
        group_variance_ub = ub_tensor_all.get("temp_ub")
        data_num_all = group_num * self.res_block_num
        self._count_rec_std_ne_mean_loop(group_mean_ub, group_mean_square_ub, group_variance_ub, data_num_all, 1)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _count_rec_std_ne_mean_loop(self, group_mean_ub, group_mean_square_ub, group_variance_ub,
                                    mask, repeat_num):
        self.tik_inst.vmuls(mask, group_mean_ub, group_mean_ub, 1.0 / self.d_groups,
                            repeat_num, 1, 1, 8, 8)
        self.tik_inst.vmuls(mask, group_mean_square_ub, group_mean_square_ub, 1.0 / self.d_groups,
                            repeat_num, 1, 1, 8, 8)
        self._count_std_and_mean(mask, group_variance_ub, group_mean_ub, group_mean_square_ub, repeat_num)
        self.tik_inst.vsqrt(mask, group_mean_square_ub, group_variance_ub,
                            repeat_num, 1, 1, 8, 8)
        self.tik_inst.vector_dup(mask, group_variance_ub, 1, repeat_num, 1, 8)
        self.tik_inst.vdiv(mask, group_mean_square_ub, group_variance_ub, group_mean_square_ub,
                           repeat_num, 1, 1, 1, 8, 8, 8)

    def _count_weight(self, ub_tensor_all, group_num):
        rec_std_scalar = self.tik_inst.Scalar(self.res_type)
        ne_mean_scalar = self.tik_inst.Scalar(self.res_type)
        temp_ub = ub_tensor_all.get("temp_ub")
        ne_mean_ub = ub_tensor_all.get("group_mean_ub")
        rec_std_ub = ub_tensor_all.get("group_mean_square_ub")
        weight_ub = ub_tensor_all.get("weight_ub")
        bias_ub = ub_tensor_all.get("bias_ub")

        mask = self.c1_groups * self.c0_num
        with self.tik_inst.for_range(0, group_num) as group_index:
            rec_std_scalar.set_as(rec_std_ub[group_index])
            ne_mean_scalar.set_as(ne_mean_ub[group_index])
            self.tik_inst.vmuls(mask, weight_ub[group_index, 0, 0], weight_ub[group_index, 0, 0], rec_std_scalar, 1, 1,
                                1, 8, 8)
            self.tik_inst.vmuls(mask, temp_ub, weight_ub[group_index, 0, 0], ne_mean_scalar, 1, 1, 1, 8, 8)
            self.tik_inst.vadd(mask, bias_ub[group_index, 0, 0], bias_ub[group_index, 0, 0], temp_ub, 1, 1, 1, 1, 8, 8,
                               8)

    def _stand_data(self, ub_tensor_all, group_num):
        input_data_ub = ub_tensor_all.get("input_data_ub")
        weight_ub = ub_tensor_all.get("weight_ub")
        bias_ub = ub_tensor_all.get("bias_ub")
        temp_ub = ub_tensor_all.get("temp_ub")
        mask = self.res_repeat_num
        c_index_stride = self.d_num_align * self.c0_num
        g_index_stride = self.c1_groups * c_index_stride
        data_num_each_c1 = self.d_num_align * self.c0_num
        with self.tik_inst.for_range(0, group_num) as g_index:
            with self.tik_inst.for_range(0, self.c1_groups) as c_index:
                self._broadcast_c(weight_ub[g_index, c_index:, :], temp_ub)
                self._start_tik_compute(data_num_each_c1, mask,
                                        self._tik_vmul_broadcast,
                                        (input_data_ub, input_data_ub, temp_ub),
                                        c_index * c_index_stride + g_index * g_index_stride)
                self._broadcast_c(bias_ub[g_index, c_index:, :], temp_ub)
                self._start_tik_compute(data_num_each_c1, mask,
                                        self._tik_vadd_broadcast,
                                        (input_data_ub, input_data_ub, temp_ub),
                                        c_index * c_index_stride + g_index * g_index_stride)

    def _data_move_out(self, ub_tensor_all, n_index, c1_index, group_num):
        input_data_ub = ub_tensor_all.get("input_data_ub")
        self._count_relu(input_data_ub, self.res_repeat_num)
        if self.input_type != self.res_type:
            result_data_ub = ub_tensor_all.get("data_temp_ub")
            self._start_tik_compute(input_data_ub.size, self.res_repeat_num,
                                    self._tik_vconv, (result_data_ub, input_data_ub))
        else:
            result_data_ub = input_data_ub

        if self.d_num_pad != 0:
            nburst = group_num * self.c1_groups
            data_block_num = self.d_num * self.c0_num // self.input_block_num
            src_stride = self.d_num_pad * self.c0_num // self.input_block_num
            self.tik_inst.data_move(self.output_0[n_index, c1_index, 0, 0], result_data_ub,
                                    0, nburst, data_block_num, src_stride, 0)
        else:
            data_block_num = self.d_groups // self.input_block_num
            self.tik_inst.data_move(self.output_0[n_index, c1_index, 0, 0], result_data_ub, 0, 1, data_block_num, 0, 0)


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class GroupNormTilingC1D(GroupNormBase):
    """
    GroupNormTilingC1D
    """

    # 'pylint: disable=too-many-arguments,too-many-locals
    def __init__(self, n_num, c_num, d_num, num_groups, eps, data_type, kernel_name):
        super(GroupNormTilingC1D, self).__init__(n_num, c_num, d_num, num_groups, eps, data_type, kernel_name)
        self.c_groups = self.c_num // self.num_groups
        self.c1_groups = self.c_groups // self.c0_num
        self.d_groups = self.c1_groups * self.d_num
        self.c0_repeat = self.res_repeat_num // self.c0_num
        self.d_groups_align = get_align_num(self.d_groups, self.c0_repeat)
        self.d_num_pad = self.d_groups_align - self.d_groups
        self.each_loop_repeat_num = 225

    def _init_gm_tensor(self):
        c1_groups = self.c1_num // self.num_groups
        data_shape = (self.n_num, self.num_groups, c1_groups * self.d_num, self.c0_num)
        input_shape_1 = (self.c_num,)
        self.input_0 = self.tik_inst.Tensor(self.input_type, data_shape, tik.scope_gm, "input_0")
        self.input_1 = self.tik_inst.Tensor(self.input_type, input_shape_1, tik.scope_gm, "input_1")
        self.input_2 = self.tik_inst.Tensor(self.input_type, input_shape_1, tik.scope_gm, "input_2")
        self.output_0 = self.tik_inst.Tensor(self.input_type, data_shape, tik.scope_gm, "output_0")

    def _compute_each_batch(self, n_index):
        sum_shape = (self.num_groups, self.res_block_num)
        sum_ub = self.tik_inst.Tensor(self.res_type, sum_shape, tik.scope_ubuf, "sum_ub")
        sum_square_ub = self.tik_inst.Tensor(self.res_type, sum_shape, tik.scope_ubuf, "sum_square_ub")
        self._count_sum(sum_ub, sum_square_ub, n_index)
        with self.tik_inst.new_stmt_scope():
            temp_ub = self.tik_inst.Tensor(self.res_type, sum_shape,
                                           name="temp_ub", scope=tik.scope_ubuf)
            repeat_num = self.num_groups * self.res_block_num // self.res_repeat_num
            self._count_rec_std_ne_mean_loop(sum_ub, sum_square_ub, temp_ub, self.res_repeat_num, repeat_num)
        weight_shape = (self.num_groups, self.c1_groups * self.c0_num)
        weight_ub = self.tik_inst.Tensor(self.res_type, weight_shape, name="weight_ub", scope=tik.scope_ubuf)
        bias_ub = self.tik_inst.Tensor(self.res_type, weight_shape, name="bias_ub", scope=tik.scope_ubuf)
        with self.tik_inst.new_stmt_scope():
            self._count_weight(weight_ub, bias_ub, sum_ub, sum_square_ub)
        if self.c1_groups == 1:
            self._count_norm(weight_ub, bias_ub, n_index)
        else:
            self._count_norm_multiple(weight_ub, bias_ub, n_index)

    # 'pylint: disable=unused-variable
    def _count_sum(self, sum_ub, sum_square_ub, n_index):
        mask = self.res_repeat_num
        self._start_tik_compute(sum_ub.size, mask, self._tik_dup, (sum_ub, 0))
        self._start_tik_compute(sum_square_ub.size, mask, self._tik_dup, (sum_square_ub, 0))
        each_loop_d_num = self.each_loop_repeat_num * self.c0_repeat
        each_c1_loop_times, last_loop_d_num = self._get_loop_info(self.d_groups, each_loop_d_num)
        loop_times = each_c1_loop_times * self.num_groups

        with self.tik_inst.for_range(0, loop_times, thread_num=2) as loop_index:
            group_index = loop_index // each_c1_loop_times
            d_index = loop_index % each_c1_loop_times * each_loop_d_num
            c1_group_index = loop_index % each_c1_loop_times
            if each_loop_d_num != last_loop_d_num:
                with self.tik_inst.if_scope(c1_group_index != each_c1_loop_times - 1):
                    self._compute_sum_each_loop(sum_ub, sum_square_ub, n_index, group_index, d_index, each_loop_d_num)
                with self.tik_inst.else_scope():
                    self._compute_sum_each_loop(sum_ub, sum_square_ub, n_index, group_index, d_index, last_loop_d_num)
            else:
                self._compute_sum_each_loop(sum_ub, sum_square_ub, n_index, group_index, d_index, each_loop_d_num)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _compute_sum_each_loop(self, sum_ub, sum_square_ub, n_index, group_index, d_index, d_num):
        mask = self.res_repeat_num
        data_num = d_num * self.c0_num
        block_num = data_num // self.input_block_num
        input_data_ub = self.tik_inst.Tensor(self.input_type, (data_num,), tik.scope_ubuf, "input_data_ub")
        self.tik_inst.data_move(input_data_ub, self.input_0[n_index, group_index, d_index, 0], 0, 1, block_num, 0, 0)
        if self.input_type != self.res_type:
            data_ub = self.tik_inst.Tensor(self.res_type, (data_num,), tik.scope_ubuf, "data_ub")
            self._start_tik_compute(data_ub.size, self.res_repeat_num, self._tik_vconv, (data_ub, input_data_ub))
        else:
            data_ub = input_data_ub
        repeat_num = data_num // self.res_repeat_num
        repeat_num_align = get_align_num(repeat_num, self.res_block_num)
        temp_ub = self.tik_inst.Tensor(self.res_type, (repeat_num_align,), tik.scope_ubuf, "temp_ub")
        result_ub = self.tik_inst.Tensor(self.res_type, (self.res_block_num,), tik.scope_ubuf, "result_ub")

        self.tik_inst.vec_reduce_add(mask, result_ub, data_ub, temp_ub, repeat_num, 8)
        self.tik_inst.vadd(self.res_block_num, sum_ub[group_index, 0], sum_ub[group_index, 0], result_ub,
                           1, 1, 1, 1, 8, 8, 8)

        self.tik_inst.vmul(mask, data_ub, data_ub, data_ub, repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vec_reduce_add(mask, result_ub, data_ub, temp_ub, repeat_num, 8)
        self.tik_inst.vadd(self.res_block_num, sum_square_ub[group_index, 0], sum_square_ub[group_index, 0],
                           result_ub, 1, 1, 1, 1, 8, 8, 8)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _count_rec_std_ne_mean_loop(self, group_mean_ub, group_mean_square_ub, group_variance_ub,
                                    mask, repeat_num):
        data_num = self.d_groups * self.c0_num
        self.tik_inst.vmuls(mask, group_mean_ub, group_mean_ub, 1.0 / data_num,
                            repeat_num, 1, 1, 8, 8)
        self.tik_inst.vmuls(mask, group_mean_square_ub, group_mean_square_ub, 1.0 / data_num,
                            repeat_num, 1, 1, 8, 8)
        self._count_std_and_mean(mask, group_variance_ub, group_mean_ub, group_mean_square_ub, repeat_num)
        self.tik_inst.vsqrt(mask, group_mean_square_ub, group_variance_ub,
                            repeat_num, 1, 1, 8, 8)
        self.tik_inst.vector_dup(mask, group_variance_ub, 1, repeat_num, 1, 8)
        self.tik_inst.vdiv(mask, group_mean_square_ub, group_variance_ub, group_mean_square_ub,
                           repeat_num, 1, 1, 1, 8, 8, 8)

    def _count_weight(self, weight_ub, bias_ub, mean_ub, variance_ub):
        block_num = self.c_num // self.input_block_num
        if self.input_type != self.res_type:
            weight_shape = weight_ub.shape
            weight_ub_fp16 = self.tik_inst.Tensor(self.input_type, weight_shape, tik.scope_ubuf, "weight_ub_fp16")
            bias_ub_fp16 = self.tik_inst.Tensor(self.input_type, weight_shape, tik.scope_ubuf, "bias_ub_fp16")
            self.tik_inst.data_move(weight_ub_fp16, self.input_1, 0, 1, block_num, 0, 0)
            self.tik_inst.data_move(bias_ub_fp16, self.input_2, 0, 1, block_num, 0, 0)
            self._start_tik_compute(weight_ub.size, self.res_repeat_num,
                                    self._tik_vconv, (weight_ub, weight_ub_fp16))
            self._start_tik_compute(bias_ub.size, self.res_repeat_num,
                                    self._tik_vconv, (bias_ub, bias_ub_fp16))
        else:
            self.tik_inst.data_move(weight_ub, self.input_1, 0, 1, block_num, 0, 0)
            self.tik_inst.data_move(bias_ub, self.input_2, 0, 1, block_num, 0, 0)
        temp_ub = self.tik_inst.Tensor(self.res_type, (self.c1_groups * self.c0_num,), tik.scope_ubuf, "bias_ub")
        rec_std_scalar = self.tik_inst.Scalar(self.res_type)
        ne_mean_scalar = self.tik_inst.Scalar(self.res_type)

        mask = self.c1_groups * self.c0_num
        with self.tik_inst.for_range(0, self.num_groups) as group_index:
            rec_std_scalar.set_as(variance_ub[group_index, 0])
            ne_mean_scalar.set_as(mean_ub[group_index, 0])
            self.tik_inst.vmuls(mask, weight_ub[group_index, 0], weight_ub[group_index, 0], rec_std_scalar, 1, 1, 1, 8,
                                8)
            self.tik_inst.vmuls(mask, temp_ub, weight_ub[group_index, 0], ne_mean_scalar, 1, 1, 1, 8, 8)
            self.tik_inst.vadd(mask, bias_ub[group_index, 0], bias_ub[group_index, 0], temp_ub, 1, 1, 1, 1, 8, 8, 8)

    # 'pylint: disable=unused-variable
    def _count_norm(self, weight_ub, bias_ub, n_index):
        each_loop_d_num = self.each_loop_repeat_num * self.c0_repeat
        each_c1_loop_times, last_loop_d_num = self._get_loop_info(self.d_groups, each_loop_d_num)
        loop_times = each_c1_loop_times * self.num_groups

        with self.tik_inst.for_range(0, loop_times, thread_num=2) as loop_index:
            group_index = loop_index // each_c1_loop_times
            c1_group_index = loop_index % each_c1_loop_times
            d_index = loop_index % each_c1_loop_times * each_loop_d_num
            if each_loop_d_num != last_loop_d_num:
                with self.tik_inst.if_scope(c1_group_index != each_c1_loop_times - 1):
                    self._count_norm_each_loop(weight_ub, bias_ub, n_index, group_index, d_index, each_loop_d_num)
                with self.tik_inst.else_scope():
                    self._count_norm_each_loop(weight_ub, bias_ub, n_index, group_index, d_index, last_loop_d_num)
            else:
                self._count_norm_each_loop(weight_ub, bias_ub, n_index, group_index, d_index, each_loop_d_num)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _count_norm_each_loop(self, weight_ub, bias_ub, n_index, group_index, d_index, d_num):
        mask = self.res_repeat_num
        data_num = d_num * self.c0_num
        data_shape = (data_num, )
        block_num = data_num // self.input_block_num
        data_move_ub = self.tik_inst.Tensor(self.input_type, data_shape, tik.scope_ubuf, "data_move_ub")
        self.tik_inst.data_move(data_move_ub, self.input_0[n_index, group_index, d_index, 0], 0, 1, block_num, 0, 0)

        if self.input_type != self.res_type:
            data_data_ub = self.tik_inst.Tensor(self.res_type, data_shape, tik.scope_ubuf, "data_data_ub")
            self._start_tik_compute(data_data_ub.size, self.res_repeat_num, self._tik_vconv,
                                    (data_data_ub, data_move_ub))
        else:
            data_data_ub = data_move_ub

        repeat_num = data_num // self.res_repeat_num
        temp_ub = self.tik_inst.Tensor(self.res_type, (self.res_repeat_num,), tik.scope_ubuf, "temp_ub")
        self._broadcast_c(weight_ub[group_index:, :], temp_ub)
        self.tik_inst.vmul(mask,
                           data_data_ub,
                           data_data_ub,
                           temp_ub,
                           repeat_num, 1, 1, 1, 8, 8, 0)
        self._broadcast_c(bias_ub[group_index:, :], temp_ub)
        self.tik_inst.vadd(mask, data_data_ub,
                           data_data_ub,
                           temp_ub,
                           repeat_num, 1, 1, 1, 8, 8, 0)
        self._count_relu(data_data_ub, mask)
        if self.input_type != self.res_type:
            self._start_tik_compute(data_move_ub.size, self.res_repeat_num, self._tik_vconv,
                                    (data_move_ub, data_data_ub))
        else:
            data_move_ub = data_data_ub
        self.tik_inst.data_move(self.output_0[n_index, group_index, d_index, 0], data_move_ub, 0, 1, block_num, 0, 0)

    def _count_norm_multiple(self, weight_ub, bias_ub, n_index):
        each_loop_d_num = self.d_num
        each_c1_loop_times = self.c1_groups
        loop_times = each_c1_loop_times * self.num_groups
        with self.tik_inst.for_range(0, loop_times, thread_num=2) as loop_index:
            group_index = loop_index // each_c1_loop_times
            c1_group_index = loop_index % each_c1_loop_times
            d_index = loop_index % each_c1_loop_times * each_loop_d_num
            self._count_norm_multiple_each_loop(weight_ub, bias_ub, n_index, group_index, c1_group_index, d_index,
                                                each_loop_d_num)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _count_norm_multiple_each_loop(self, weight_ub, bias_ub, n_index, group_index, c1_group_index, d_index, d_num):
        data_num = d_num * self.c0_num
        block_num = data_num // self.input_block_num
        data_move_ub = self.tik_inst.Tensor(self.input_type, (data_num,), tik.scope_ubuf, "data_move_ub")
        self.tik_inst.data_move(data_move_ub, self.input_0[n_index, group_index, d_index, 0], 0, 1, block_num, 0, 0)

        if self.input_type != self.res_type:
            data_data_ub = self.tik_inst.Tensor(self.res_type, (data_num,), tik.scope_ubuf, "data_data_ub")
            self._start_tik_compute(data_data_ub.size, self.res_repeat_num, self._tik_vconv,
                                    (data_data_ub, data_move_ub))
        else:
            data_data_ub = data_move_ub

        temp_ub = self.tik_inst.Tensor(self.res_type, (self.res_repeat_num,), tik.scope_ubuf, "temp_ub")
        self._broadcast_c(weight_ub[group_index:, c1_group_index * self.c0_num:], temp_ub)
        self._start_tik_compute(data_data_ub.size, self.res_repeat_num, self._tik_vmul_broadcast,
                                (data_data_ub, data_data_ub, temp_ub))
        self._broadcast_c(bias_ub[group_index:, c1_group_index * self.c0_num:], temp_ub)
        self._start_tik_compute(data_data_ub.size, self.res_repeat_num, self._tik_vadd_broadcast,
                                (data_data_ub, data_data_ub, temp_ub))
        mask = self.res_repeat_num
        self._count_relu(data_data_ub, mask)
        if self.input_type != self.res_type:
            self._start_tik_compute(data_move_ub.size, self.res_repeat_num, self._tik_vconv,
                                    (data_move_ub, data_data_ub))
        else:
            data_move_ub = data_data_ub
        self.tik_inst.data_move(self.output_0[n_index, group_index, d_index, 0], data_move_ub, 0, 1, block_num, 0, 0)


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class GroupNormTilingC0(GroupNormBase):
    """
    GroupNormTilingC0
    """

    # 'pylint: disable=too-many-arguments,too-many-locals
    def __init__(self, n_num, c_num, d_num, num_groups, eps, data_type, kernel_name):
        super(GroupNormTilingC0, self).__init__(n_num, c_num, d_num, num_groups, eps, data_type, kernel_name)
        self.c_groups = self.c_num // self.num_groups
        self.d_groups = self.c_groups * self.d_num
        self.groups_c0 = self.c0_num // self.c_groups
        self.c0_repeat = self.res_repeat_num // self.c0_num

    def _compute_each_batch(self, n_index):
        loop_c1_num = 1
        loop_times, last_c1_num = self._get_loop_info(self.c1_num, loop_c1_num)
        tensor_const = self.tik_inst.Tensor("int32", (self.res_repeat_num,), name="tensor_const", scope=tik.scope_ubuf)
        if self.c_groups == 8:
            self.tik_inst.vector_dup(8, tensor_const, 0, 1, 1, 8)
            self.tik_inst.vector_dup(8, tensor_const[8], 4, 1, 1, 8)
        elif self.c_groups == 4:
            const_data = [4, 8, 12, 0, 20, 24, 28, 16,
                          8, 12, 0, 4, 24, 28, 16, 20,
                          12, 0, 4, 8, 28, 16, 20, 24]
            scalar_data = self.tik_inst.Scalar("int32")
            for i, data in enumerate(const_data):
                scalar_data.set_as(data)
                tensor_const[i % 8 + i // 8 * 16].set_as(scalar_data)
            self.tik_inst.vadds(32, tensor_const[8], tensor_const, 32, 1, 2, 2, 8, 8)
        if last_c1_num != loop_c1_num:
            with self.tik_inst.for_range(0, loop_times) as loop_index:
                c1_index = loop_index * loop_c1_num
                ub_tensor_all = self._init_ub_tensor(loop_c1_num)
                ub_tensor_all["tensor_const"] = tensor_const
                with self.tik_inst.if_scope(loop_index != loop_times - 1):
                    self._compute_each_loop(ub_tensor_all, n_index, c1_index, loop_c1_num)
                with self.tik_inst.else_scope():
                    self._compute_each_loop(ub_tensor_all, n_index, c1_index, last_c1_num)
        else:
            with self.tik_inst.for_range(0, loop_times) as loop_index:
                c1_index = loop_index * loop_c1_num
                ub_tensor_all = self._init_ub_tensor(loop_c1_num)
                ub_tensor_all["tensor_const"] = tensor_const
                self._compute_each_loop(ub_tensor_all, n_index, c1_index, loop_c1_num)

    def _init_ub_tensor(self, c1_num):
        data_ub_shape = (c1_num, self.d_num, self.c0_num)
        mean_ub_shape = (c1_num, self.c0_num)
        weight_ub_shape = (c1_num, self.c0_num)
        temp_ub_size = get_align_num(self.d_num * self.c0_num, self.res_repeat_num)
        temp_ub_shape = (temp_ub_size,)
        ub_tensor_all = {
            "input_data_ub":
                self.tik_inst.Tensor(self.res_type, data_ub_shape, tik.scope_ubuf, "input_data_ub", ),
            "group_mean_ub":
                self.tik_inst.Tensor(self.res_type, mean_ub_shape, name="group_mean_ub", scope=tik.scope_ubuf),
            "group_mean_square_ub":
                self.tik_inst.Tensor(self.res_type, mean_ub_shape, name="group_mean_square_ub", scope=tik.scope_ubuf),
            "group_variance_ub":
                self.tik_inst.Tensor(self.res_type, mean_ub_shape, name="group_variance_ub", scope=tik.scope_ubuf),
            "weight_ub":
                self.tik_inst.Tensor(self.res_type, weight_ub_shape, tik.scope_ubuf, "weight_ub"),
            "bias_ub":
                self.tik_inst.Tensor(self.res_type, weight_ub_shape, tik.scope_ubuf, "bias_ub"),
            "temp_ub_0":
                self.tik_inst.Tensor(self.res_type, temp_ub_shape, tik.scope_ubuf, "temp_ub_0"),
            "temp_ub_1":
                self.tik_inst.Tensor(self.res_type, temp_ub_shape, tik.scope_ubuf, "temp_ub_1"),
        }
        if self.input_type != self.res_type:
            ub_tensor_all["data_ub_fp16"] = self.tik_inst.Tensor(self.input_type, data_ub_shape,
                                                                 tik.scope_ubuf, "data_ub_fp16")
            ub_tensor_all["weight_ub_fp16"] = self.tik_inst.Tensor(self.input_type, weight_ub_shape,
                                                                   tik.scope_ubuf, "weight_ub_fp16")
            ub_tensor_all["bias_ub_fp16"] = self.tik_inst.Tensor(self.input_type, weight_ub_shape,
                                                                 tik.scope_ubuf, "bias_ub_fp16")
        return ub_tensor_all

    def _compute_each_loop(self, ub_tensor_all, n_index, c1_index, c1_num):
        self._data_move_in(ub_tensor_all, n_index, c1_index, c1_num)
        self._count_sum(ub_tensor_all, c1_num)
        self._count_rec_std_ne_mean(ub_tensor_all, c1_num)
        self._count_weight(ub_tensor_all, c1_num)
        self._stand_data(ub_tensor_all, c1_num)
        self._data_move_out(ub_tensor_all, n_index, c1_index, c1_num)

    def _data_move_in(self, ub_tensor_all, n_index, c1_index, c1_num):
        if self.input_type != self.res_type:
            data_ub_move = ub_tensor_all.get("data_ub_fp16")
            weight_ub_move = ub_tensor_all.get("weight_ub_fp16")
            bias_ub_move = ub_tensor_all.get("bias_ub_fp16")
        else:
            data_ub_move = ub_tensor_all.get("input_data_ub")
            weight_ub_move = ub_tensor_all.get("weight_ub")
            bias_ub_move = ub_tensor_all.get("bias_ub")

        data_block_num = c1_num * self.d_num * self.c0_num // self.input_block_num
        weight_block_num = c1_num * self.c0_num // self.input_block_num
        self.tik_inst.data_move(data_ub_move, self.input_0[n_index, c1_index, 0, 0], 0, 1, data_block_num, 0, 0)
        self.tik_inst.data_move(weight_ub_move, self.input_1[c1_index * self.c0_num], 0, 1, weight_block_num, 0, 0)
        self.tik_inst.data_move(bias_ub_move, self.input_2[c1_index * self.c0_num], 0, 1, weight_block_num, 0, 0)

        if self.input_type != self.res_type:
            data_ub = ub_tensor_all.get("input_data_ub")
            weight_ub = ub_tensor_all.get("weight_ub")
            bias_ub = ub_tensor_all.get("bias_ub")
            self._start_tik_compute(data_ub.size, self.res_repeat_num,
                                    self._tik_vconv, (data_ub, data_ub_move))
            self._start_tik_compute(weight_ub.size, self.res_repeat_num,
                                    self._tik_vconv, (weight_ub, weight_ub_move))
            self._start_tik_compute(bias_ub.size, self.res_repeat_num,
                                    self._tik_vconv, (bias_ub, bias_ub_move))

    def _get_sum_info(self):
        data_num = self.d_num * self.c0_num
        sum_info = []
        while data_num > self.c0_num:
            repeat_num_all = data_num // self.res_repeat_num
            if repeat_num_all > 1:
                repeat_num_loop = repeat_num_all // 2
                if_move = repeat_num_all % 2
                start_index_1 = (repeat_num_loop + if_move) * self.res_repeat_num
                sum_info.append([self.res_repeat_num, start_index_1, repeat_num_loop])
                data_num = start_index_1
            else:
                repeat_num_loop = 1
                start_index_1 = data_num // 2
                sum_info.append([start_index_1, start_index_1, repeat_num_loop])
                data_num = start_index_1
        return sum_info

    def _count_sum_c0(self, ub_tensor_all, c1_index):
        input_data_ub = ub_tensor_all.get("input_data_ub")
        temp_ub_0 = ub_tensor_all.get("temp_ub_0")
        temp_ub_1 = ub_tensor_all.get("temp_ub_1")
        group_mean_ub = ub_tensor_all.get("group_mean_ub")
        group_mean_square_ub = ub_tensor_all.get("group_mean_square_ub")
        data_num = self.d_num * self.c0_num
        block_num = data_num // self.res_block_num
        repeat_num = data_num // self.res_repeat_num
        self.tik_inst.data_move(temp_ub_0, input_data_ub[c1_index, 0, 0], 0, 1, block_num, 0, 0)
        self._start_tik_compute(data_num, self.res_repeat_num, self._tik_vmul,
                                args=(temp_ub_1, input_data_ub[c1_index, :, :], input_data_ub[c1_index, :, :]))
        sum_info_all = self._get_sum_info()
        for sum_info in sum_info_all:
            mask, start_index_1, repeat_num = sum_info
            self.tik_inst.vadd(mask, temp_ub_0, temp_ub_0, temp_ub_0[start_index_1], repeat_num, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(mask, temp_ub_1, temp_ub_1, temp_ub_1[start_index_1], repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.data_move(group_mean_ub[c1_index, 0], temp_ub_0, 0, 1, 2, 0, 0)
        self.tik_inst.data_move(group_mean_square_ub[c1_index, 0], temp_ub_1, 0, 1, 2, 0, 0)

    def _count_sum(self, ub_tensor_all, c1_num):
        with self.tik_inst.for_range(0, c1_num) as c1_index:
            self._count_sum_c0(ub_tensor_all, c1_index)
        temp_ub_0 = ub_tensor_all.get("temp_ub_0")
        temp_ub_1 = ub_tensor_all.get("temp_ub_1")
        mask = c1_num * self.c0_num
        group_mean_ub = ub_tensor_all.get("group_mean_ub")
        group_mean_square_ub = ub_tensor_all.get("group_mean_square_ub")
        tensor_const = ub_tensor_all["tensor_const"]
        if self.c_groups == 8:
            self.tik_inst.vcgadd(mask, temp_ub_0, group_mean_ub, 1, 1, 1, 8)
            self.tik_inst.vcgadd(mask, temp_ub_1, group_mean_square_ub, 1, 1, 1, 8)
            self.tik_inst.vgather(mask, group_mean_ub, temp_ub_0, tensor_const, 1, 0)
            self.tik_inst.vgather(mask, group_mean_square_ub, temp_ub_1, tensor_const, 1, 0)
        elif self.c_groups == 4:
            self.tik_inst.vgather(mask * 3, temp_ub_0[mask], group_mean_ub, tensor_const, 1, 0)
            self.tik_inst.vadd(mask, group_mean_ub, group_mean_ub, temp_ub_0[mask], 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(mask, group_mean_ub, group_mean_ub, temp_ub_0[mask * 2], 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(mask, group_mean_ub, group_mean_ub, temp_ub_0[mask * 3], 1, 1, 1, 1, 8, 8, 8)

            self.tik_inst.vgather(mask * 3, temp_ub_0[mask], group_mean_square_ub, tensor_const, 1, 0)
            self.tik_inst.vadd(mask, group_mean_square_ub, group_mean_square_ub, temp_ub_0[mask], 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(mask, group_mean_square_ub, group_mean_square_ub, temp_ub_0[mask * 2], 1, 1, 1, 1, 8, 8,
                               8)
            self.tik_inst.vadd(mask, group_mean_square_ub, group_mean_square_ub, temp_ub_0[mask * 3], 1, 1, 1, 1, 8, 8,
                               8)

    def _count_rec_std_ne_mean(self, ub_tensor_all, c1_num):
        group_mean_ub = ub_tensor_all.get("group_mean_ub")
        group_mean_square_ub = ub_tensor_all.get("group_mean_square_ub")
        group_variance_ub = ub_tensor_all.get("group_variance_ub")
        data_num_all = c1_num * self.c0_num
        repeat_times = data_num_all // self.res_repeat_num
        last_num = data_num_all - repeat_times * self.res_repeat_num
        self._count_rec_std_ne_mean_loop(group_mean_ub, group_mean_square_ub, group_variance_ub, last_num, 1)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _count_rec_std_ne_mean_loop(self, group_mean_ub, group_mean_square_ub, group_variance_ub,
                                    mask, repeat_num):
        self.tik_inst.vmuls(mask, group_mean_ub, group_mean_ub, 1.0 / self.d_groups,
                            repeat_num, 1, 1, 8, 8)
        self.tik_inst.vmuls(mask, group_mean_square_ub, group_mean_square_ub, 1.0 / self.d_groups,
                            repeat_num, 1, 1, 8, 8)
        self._count_std_and_mean(mask, group_variance_ub, group_mean_ub, group_mean_square_ub, repeat_num)
        self.tik_inst.vsqrt(mask, group_variance_ub, group_variance_ub,
                            repeat_num, 1, 1, 8, 8)
        self.tik_inst.vector_dup(mask, group_mean_square_ub, 1, repeat_num, 1, 8)
        self.tik_inst.vdiv(mask, group_variance_ub, group_mean_square_ub, group_variance_ub,
                           repeat_num, 1, 1, 1, 8, 8, 8)

    def _count_weight(self, ub_tensor_all, c1_num):
        temp_ub_0 = ub_tensor_all.get("temp_ub_0")
        group_mean_ub = ub_tensor_all.get("group_mean_ub")
        group_variance_ub = ub_tensor_all.get("group_variance_ub")
        weight_ub = ub_tensor_all.get("weight_ub")
        bias_ub = ub_tensor_all.get("bias_ub")
        mask = c1_num * self.c0_num
        self.tik_inst.vmul(mask, weight_ub, weight_ub, group_variance_ub, 1, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(mask, temp_ub_0, weight_ub, group_mean_ub, 1, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, bias_ub, bias_ub, temp_ub_0, 1, 1, 1, 1, 8, 8, 8)

    def _stand_data(self, ub_tensor_all, c1_num):
        input_data_ub = ub_tensor_all.get("input_data_ub")
        weight_ub = ub_tensor_all.get("weight_ub")
        bias_ub = ub_tensor_all.get("bias_ub")
        temp_ub = ub_tensor_all.get("temp_ub_0")
        mask = self.res_repeat_num
        data_num = self.d_num * self.c0_num
        with self.tik_inst.for_range(0, c1_num) as c1_index:
            self._broadcast_c(weight_ub[c1_index:, :], temp_ub)
            self._start_tik_compute(data_num, mask,
                                    self._tik_vmul_broadcast,
                                    (input_data_ub[c1_index, :, :],
                                     input_data_ub[c1_index, :, :],
                                     temp_ub))
            self._broadcast_c(bias_ub[c1_index:, :], temp_ub)
            self._start_tik_compute(data_num, mask,
                                    self._tik_vadd_broadcast,
                                    (input_data_ub[c1_index, :, :],
                                     input_data_ub[c1_index, :, :],
                                     temp_ub))

    def _data_move_out(self, ub_tensor_all, n_index, c1_index, c1_num):
        result_data = ub_tensor_all.get("input_data_ub")
        self._count_relu(result_data, self.res_repeat_num)
        if self.input_type != self.res_type:
            data_ub_move = ub_tensor_all.get("data_ub_fp16")
            self._start_tik_compute(data_ub_move.size, self.res_repeat_num,
                                    self._tik_vconv, (data_ub_move, result_data))
        else:
            data_ub_move = result_data
        data_block_num = c1_num * self.d_num * self.c0_num // self.input_block_num
        self.tik_inst.data_move(self.output_0[n_index, c1_index, 0, 0], data_ub_move, 0, 1, data_block_num, 0, 0)


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class GroupNormTilingD(GroupNormBase):
    """
    GroupNormTilingD
    """

    # 'pylint: disable=too-many-arguments,too-many-locals
    def __init__(self, n_num, c_num, d_num, num_groups, eps, data_type, kernel_name):
        super(GroupNormTilingD, self).__init__(n_num, c_num, d_num, num_groups, eps, data_type, kernel_name)
        self.c_groups = self.c_num // self.num_groups
        self.d_groups = self.c_groups * self.d_num
        self.groups_c0 = self.c0_num // self.c_groups
        self.c0_repeat = self.res_repeat_num // self.c0_num
        if self.d_num == 56 * 56:
            self.each_loop_repeat_num = 196
        elif self.d_num == 60 * 60:
            self.each_loop_repeat_num = 225
        elif self.d_num == 120 * 120:
            self.each_loop_repeat_num = 240
        elif self.d_num == 128 * 128:
            self.each_loop_repeat_num = 250
        elif self.d_num == 64 * 64:
            self.each_loop_repeat_num = 250

    def _compute_each_batch(self, n_index):
        sum_ub = self.tik_inst.Tensor(self.res_type, (self.c1_num, self.c0_num), tik.scope_ubuf, "sum_ub")
        sum_square_ub = self.tik_inst.Tensor(self.res_type, (self.c1_num, self.c0_num), tik.scope_ubuf, "sum_square_ub")
        self._count_sum(sum_ub, sum_square_ub, n_index)
        with self.tik_inst.new_stmt_scope():
            self._count_rec_std_ne_mean(sum_ub, sum_square_ub)
        self._count_norm(sum_square_ub, sum_ub, n_index)

    # 'pylint: disable=unused-variable
    def _count_sum(self, sum_ub, sum_square_ub, n_index):
        mask = self.res_repeat_num
        sum_repeat_num = self.c_num // mask
        self.tik_inst.vector_dup(mask, sum_ub, 0, sum_repeat_num, 1, 8)
        self.tik_inst.vector_dup(mask, sum_square_ub, 0, sum_repeat_num, 1, 8)
        each_loop_d_num = self.each_loop_repeat_num * self.c0_repeat
        each_c1_loop_times, last_c1_d_num = self._get_loop_info(self.d_num, each_loop_d_num)
        loop_times = each_c1_loop_times * self.c1_num
        if last_c1_d_num != each_loop_d_num:
            with self.tik_inst.for_range(0, loop_times, thread_num=2) as loop_index:
                c1_index = loop_index // each_c1_loop_times
                d_loop_index = loop_index % each_c1_loop_times
                d_index = d_loop_index * each_loop_d_num
                with self.tik_inst.if_scope(d_loop_index != each_c1_loop_times - 1):
                    self._compute_sum_each_loop(sum_ub, sum_square_ub, n_index, c1_index, d_index, each_loop_d_num)
                with self.tik_inst.else_scope():
                    self._compute_sum_each_loop(sum_ub, sum_square_ub, n_index, c1_index, d_index, last_c1_d_num)
        else:
            with self.tik_inst.for_range(0, loop_times, thread_num=2) as loop_index:
                c1_index = loop_index // each_c1_loop_times
                d_index = loop_index % each_c1_loop_times * each_loop_d_num
                self._compute_sum_each_loop(sum_ub, sum_square_ub, n_index, c1_index, d_index, each_loop_d_num)

    def _get_sum_info(self, data_num):
        sum_info = []
        while data_num > self.c0_num:
            repeat_num_all = data_num // self.res_repeat_num
            if repeat_num_all > 1:
                repeat_num_loop = repeat_num_all // 2
                if_move = repeat_num_all % 2
                start_index_1 = (repeat_num_loop + if_move) * self.res_repeat_num
                sum_info.append([self.res_repeat_num, start_index_1, repeat_num_loop])
                data_num = start_index_1
            else:
                repeat_num_loop = 1
                start_index_1 = data_num // 2
                sum_info.append([start_index_1, start_index_1, repeat_num_loop])
                data_num = start_index_1
        return sum_info

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _compute_sum_each_loop(self, sum_ub, sum_square_ub, n_index, c1_index, d_index, d_num):
        mask = self.res_repeat_num
        data_num = d_num * self.c0_num

        data_ub = self.tik_inst.Tensor(self.res_type, (data_num,), tik.scope_ubuf, "data_ub")
        temp_ub = self.tik_inst.Tensor(self.input_type, (self.res_size // self.input_size * data_num,),
                                       tik.scope_ubuf, "temp_ub")
        block_num = data_num // self.input_block_num

        if self.input_type != self.res_type:
            self.tik_inst.data_move(temp_ub, self.input_0[n_index, c1_index, d_index, 0], 0, 1, block_num, 0, 0)
            self._start_tik_compute(data_ub.size, self.res_repeat_num, self._tik_vconv, (data_ub, temp_ub))
            temp_ub = temp_ub.reinterpret_cast_to(self.res_type)
        else:
            self.tik_inst.data_move(data_ub, self.input_0[n_index, c1_index, d_index, 0], 0, 1, block_num, 0, 0)
        repeat_num = data_num // self.res_repeat_num

        self.tik_inst.vmul(mask, temp_ub, data_ub, data_ub, repeat_num, 1, 1, 1, 8, 8, 8)
        sum_info_all = self._get_sum_info(data_num)
        for sum_info in sum_info_all:
            mask, start_index_1, repeat_num = sum_info
            self.tik_inst.vadd(mask, data_ub, data_ub, data_ub[start_index_1], repeat_num, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(mask, temp_ub, temp_ub, temp_ub[start_index_1], repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, sum_ub[c1_index, 0], sum_ub[c1_index, 0], data_ub, 1, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, sum_square_ub[c1_index, 0], sum_square_ub[c1_index, 0], temp_ub, 1, 1, 1, 1, 8, 8, 8)

    def _count_rec_std_ne_mean(self, sum_ub, sum_square_ub):
        temp_ub_0 = self.tik_inst.Tensor(self.res_type, (self.c_num,), tik.scope_ubuf, "temp_ub_0")
        temp_ub_1 = self.tik_inst.Tensor(self.res_type, (self.c_num,), tik.scope_ubuf, "temp_ub_1")
        temp_ub_2 = self.tik_inst.Tensor(self.res_type, (self.c_num,), tik.scope_ubuf, "temp_ub_2")
        if self.c_groups == 8:
            self._count_sum_c8(sum_ub, sum_square_ub, temp_ub_0, temp_ub_1, temp_ub_2)
        elif self.c_groups == 4:
            self._count_sum_c4(sum_ub, sum_square_ub, temp_ub_2)
        elif self.c_groups == 2:
            self._count_sum_c2(sum_ub, sum_square_ub, temp_ub_2)
        mask = self.res_repeat_num
        sum_repeat = self.c_num // mask
        self._count_rec_std_ne_mean_loop(sum_ub, temp_ub_2, sum_square_ub, mask, sum_repeat)
        weight_block_num = self.c_num // self.input_block_num
        if self.input_type != self.res_type:
            weight_fp16 = self.tik_inst.Tensor(self.input_type, (self.c_num,), tik.scope_ubuf, "weight_fp16")
            bias_fp16 = self.tik_inst.Tensor(self.input_type, (self.c_num,), tik.scope_ubuf, "bias_fp16")
            self.tik_inst.data_move(weight_fp16, self.input_1, 0, 1, weight_block_num, 0, 0)
            self.tik_inst.data_move(bias_fp16, self.input_2, 0, 1, weight_block_num, 0, 0)
            self._start_tik_compute(temp_ub_0.size, self.res_repeat_num, self._tik_vconv, (temp_ub_0, weight_fp16))
            self._start_tik_compute(temp_ub_1.size, self.res_repeat_num, self._tik_vconv, (temp_ub_1, bias_fp16))
        else:
            self.tik_inst.data_move(temp_ub_0, self.input_1, 0, 1, weight_block_num, 0, 0)
            self.tik_inst.data_move(temp_ub_1, self.input_2, 0, 1, weight_block_num, 0, 0)
        self._count_weight(sum_ub, sum_square_ub, temp_ub_0, temp_ub_1, temp_ub_2, mask, sum_repeat)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _count_sum_c8(self, sum_ub, sum_square_ub, temp_ub_0, temp_ub_1, temp_ub_2):
        tensor_const = self.tik_inst.Tensor("int32", (self.c_num,), tik.scope_ubuf, "tensor_const")
        mask = self.res_repeat_num
        sum_repeat = self.c_num // mask
        self.tik_inst.vcgadd(mask, temp_ub_0, sum_ub, sum_repeat, 1, 1, 8)
        self.tik_inst.vcgadd(mask, temp_ub_1, sum_square_ub, sum_repeat, 1, 1, 8)
        self.tik_inst.vector_dup(mask, tensor_const, 0, 1, 1, 8)
        move_step = 8
        move_block = 1
        block_all = 8 * sum_repeat
        add_num = 4
        while block_all > 1:
            mask_temp = min(move_step, self.res_repeat_num)
            repeat_temp = max(1, move_step // self.res_repeat_num)
            self.tik_inst.vadds(mask_temp, tensor_const[move_step], tensor_const, add_num, repeat_temp, 1, 1, 8, 8)
            move_step *= 2
            move_block *= 2
            block_all //= 2
            add_num *= 2
        self.tik_inst.vgather(mask, sum_ub, temp_ub_0, tensor_const, sum_repeat, 8)
        self.tik_inst.vgather(mask, temp_ub_2, temp_ub_1, tensor_const, sum_repeat, 8)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _count_sum_c4(self, sum_ub, sum_square_ub, temp_ub_2):
        tensor_const = self.tik_inst.Tensor("int32", (3, self.c_num,), tik.scope_ubuf, "tensor_const")
        temp_ub = self.tik_inst.Tensor(self.res_type, (3, self.c_num,), tik.scope_ubuf, "temp_ub")
        const_data = [[4, 8, 12, 0, 20, 24, 28, 16],
                      [8, 12, 0, 4, 24, 28, 16, 20],
                      [12, 0, 4, 8, 28, 16, 20, 24]]
        scalar_data = self.tik_inst.Scalar("int32")
        for i, data_1 in enumerate(const_data):
            for j, data in enumerate(data_1):
                scalar_data.set_as(data)
                tensor_const[i, j].set_as(scalar_data)

        start_index = 8
        block_all = 8
        add_num = 32
        repeat_stride = self.c_num // self.res_block_num
        repeat_temp = 3
        while block_all > 1:
            mask_temp = start_index
            self.tik_inst.vadds(mask_temp, tensor_const[0, start_index], tensor_const, add_num,
                                repeat_temp, 1, 1, repeat_stride, repeat_stride)
            start_index *= 2
            add_num *= 2
            block_all //= 2
        mask = self.res_repeat_num
        self.tik_inst.vadds(mask, tensor_const[0, start_index], tensor_const, add_num,
                            repeat_temp, 1, 1, repeat_stride, repeat_stride)
        vgather_repeat = (3 * self.c_num) // self.res_repeat_num
        self.tik_inst.vgather(mask, temp_ub, sum_ub, tensor_const, vgather_repeat, 8)
        add_repeat = self.c_num // self.res_repeat_num
        self.tik_inst.vadd(mask, sum_ub, sum_ub, temp_ub[0, 0], add_repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, sum_ub, sum_ub, temp_ub[1, 0], add_repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, sum_ub, sum_ub, temp_ub[2, 0], add_repeat, 1, 1, 1, 8, 8, 8)

        self.tik_inst.vgather(mask, temp_ub, sum_square_ub, tensor_const, vgather_repeat, 8)
        add_repeat = self.c_num // self.res_repeat_num
        self.tik_inst.vadd(mask, sum_square_ub, sum_square_ub, temp_ub[0, 0], add_repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, sum_square_ub, sum_square_ub, temp_ub[1, 0], add_repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, temp_ub_2, sum_square_ub, temp_ub[2, 0], add_repeat, 1, 1, 1, 8, 8, 8)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _count_sum_c2(self, sum_ub, sum_square_ub, temp_ub_2):
        tensor_const = self.tik_inst.Tensor("int32", (self.c_num,), tik.scope_ubuf, "tensor_const")
        temp_ub = self.tik_inst.Tensor(self.res_type, (self.c_num,), tik.scope_ubuf, "temp_ub")
        const_data = [4, 0, 12, 8, 20, 16, 28, 24]
        scalar_data = self.tik_inst.Scalar("int32")
        for i, data in enumerate(const_data):
            scalar_data.set_as(data)
            tensor_const[i].set_as(scalar_data)
        start_index = 8
        block_all = 8
        add_num = 32
        repeat_stride = 8
        repeat_temp = 1
        while block_all > 1:
            mask_temp = start_index
            self.tik_inst.vadds(mask_temp, tensor_const[start_index], tensor_const, add_num,
                                repeat_temp, 1, 1, repeat_stride, repeat_stride)
            start_index *= 2
            add_num *= 2
            block_all //= 2
        mask = self.res_repeat_num
        vgather_repeat = self.c_num // self.res_repeat_num
        self.tik_inst.vgather(mask, temp_ub, sum_ub, tensor_const, vgather_repeat, 8)
        add_repeat = self.c_num // self.res_repeat_num
        self.tik_inst.vadd(mask, sum_ub, sum_ub, temp_ub, add_repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vgather(mask, temp_ub, sum_square_ub, tensor_const, vgather_repeat, 8)
        add_repeat = self.c_num // self.res_repeat_num
        self.tik_inst.vadd(mask, temp_ub_2, sum_square_ub, temp_ub, add_repeat, 1, 1, 1, 8, 8, 8)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _count_rec_std_ne_mean_loop(self, group_mean_ub, group_mean_square_ub, group_variance_ub,
                                    mask, repeat_num):
        self.tik_inst.vmuls(mask, group_mean_ub, group_mean_ub, 1.0 / self.d_groups,
                            repeat_num, 1, 1, 8, 8)
        self.tik_inst.vmuls(mask, group_mean_square_ub, group_mean_square_ub, 1.0 / self.d_groups,
                            repeat_num, 1, 1, 8, 8)
        self._count_std_and_mean(mask, group_variance_ub, group_mean_ub, group_mean_square_ub, repeat_num)
        self.tik_inst.vsqrt(mask, group_variance_ub, group_variance_ub,
                            repeat_num, 1, 1, 8, 8)
        self.tik_inst.vector_dup(mask, group_mean_square_ub, 1, repeat_num, 1, 8)
        self.tik_inst.vdiv(mask, group_variance_ub, group_mean_square_ub, group_variance_ub,
                           repeat_num, 1, 1, 1, 8, 8, 8)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _count_weight(self, mean_ub, variance_ub, weight_ub, bias_ub, temp_ub_0, mask, repeat_num):
        self.tik_inst.vmul(mask, variance_ub, weight_ub, variance_ub, repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(mask, temp_ub_0, variance_ub, mean_ub, repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, mean_ub, bias_ub, temp_ub_0, repeat_num, 1, 1, 1, 8, 8, 8)

    # 'pylint: disable=unused-variable
    def _count_norm(self, weight_ub, bias_ub, n_index):
        each_loop_d_num = self.each_loop_repeat_num * self.c0_repeat
        each_c1_loop_times, last_c1_d_num = self._get_loop_info(self.d_num, each_loop_d_num)
        loop_times = each_c1_loop_times * self.c1_num
        if last_c1_d_num != each_loop_d_num:
            with self.tik_inst.for_range(0, loop_times, thread_num=2) as loop_index:
                c1_index = loop_index // each_c1_loop_times
                d_loop_index = loop_index % each_c1_loop_times
                d_index = d_loop_index * each_loop_d_num
                with self.tik_inst.if_scope(d_loop_index != each_c1_loop_times - 1):
                    self._count_norm_each_loop(weight_ub, bias_ub, n_index, c1_index, d_index, each_loop_d_num)
                with self.tik_inst.else_scope():
                    self._count_norm_each_loop(weight_ub, bias_ub, n_index, c1_index, d_index, last_c1_d_num)
        else:
            with self.tik_inst.for_range(0, loop_times, thread_num=2) as loop_index:
                c1_index = loop_index // each_c1_loop_times
                d_index = loop_index % each_c1_loop_times * each_loop_d_num
                self._count_norm_each_loop(weight_ub, bias_ub, n_index, c1_index, d_index, each_loop_d_num)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _count_norm_each_loop(self, weight_ub, bias_ub, n_index, c1_index, d_index, d_num):
        mask = self.res_repeat_num
        data_num = d_num * self.c0_num
        repeat_num = data_num // self.res_repeat_num
        block_num = data_num // self.input_block_num
        data_ub = self.tik_inst.Tensor(self.res_type, (data_num,), tik.scope_ubuf, "data_ub")
        temp_ub = self.tik_inst.Tensor(self.res_type, (self.res_repeat_num,), tik.scope_ubuf, "temp_ub")

        if self.input_type != self.res_type:
            data_ub_fp16 = self.tik_inst.Tensor(self.input_type, (data_num,), tik.scope_ubuf, "data_ub_fp16")
            self.tik_inst.data_move(data_ub_fp16, self.input_0[n_index, c1_index, d_index, 0], 0, 1, block_num, 0, 0)
            self._start_tik_compute(data_ub.size, self.res_repeat_num, self._tik_vconv, (data_ub, data_ub_fp16))
        else:
            self.tik_inst.data_move(data_ub, self.input_0[n_index, c1_index, d_index, 0], 0, 1, block_num, 0, 0)

        self._broadcast_c(weight_ub[c1_index:, :], temp_ub)
        self.tik_inst.vmul(mask,
                           data_ub,
                           data_ub,
                           temp_ub,
                           repeat_num, 1, 1, 1, 8, 8, 0)
        self._broadcast_c(bias_ub[c1_index:, :], temp_ub)
        self.tik_inst.vadd(mask, data_ub,
                           data_ub,
                           temp_ub,
                           repeat_num, 1, 1, 1, 8, 8, 0)
        self._count_relu(data_ub, mask)
        if self.input_type != self.res_type:
            self._start_tik_compute(data_ub.size, self.res_repeat_num, self._tik_vconv, (data_ub_fp16, data_ub))
            self.tik_inst.data_move(self.output_0[n_index, c1_index, d_index, 0], data_ub_fp16, 0, 1, block_num, 0, 0)
        else:

            self.tik_inst.data_move(self.output_0[n_index, c1_index, d_index, 0], data_ub, 0, 1, block_num, 0, 0)


def check_input(input_0, input_1, input_2, output_0, kernel_name):
    """
    check_input
    """
    check_list = ("float16", "float32")
    para_check.check_dtype(input_0.get("dtype").lower(), check_list, param_name="x")
    para_check.check_dtype(input_1.get("dtype").lower(), check_list, param_name="gamma")
    para_check.check_dtype(input_2.get("dtype").lower(), check_list, param_name="beta")
    para_check.check_dtype(output_0.get("dtype").lower(), check_list, param_name="y")
    para_check.check_format(input_0.get("format").upper(), ("NC1HWC0",), param_name="x")
    para_check.check_format(output_0.get("format").upper(), ("NC1HWC0",), param_name="y")

    support_shape = tuple((i // 16, j) for i, j in SUPPORT_SHAPE)
    input_shape = tuple(input_0.get("shape"))
    if len(input_shape) != 5 or input_shape[1:3] not in support_shape or input_shape[2] != input_shape[3]:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "shape of x", "not support", str(input_shape))
    output_shape = tuple(output_0.get("shape"))
    if output_shape != input_shape:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "shape of y", "not support", str(input_shape))
    weight_shape = tuple(input_1.get("shape"))
    bias_shape = tuple(input_2.get("shape"))
    channel_num = input_shape[1] * input_shape[4]
    if (len(weight_shape) != 1 or weight_shape[0] != channel_num) and \
            (len(weight_shape) != 3 or weight_shape != (channel_num, 1, 1)):
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "shape of gamma", "not support", str(input_shape))
    if (len(bias_shape) != 1 or bias_shape[0] != channel_num) and \
            (len(bias_shape) != 3 or bias_shape != (channel_num, 1, 1)):
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "shape of beta", "not support", str(input_shape))


def check_attr(num_groups, eps, kernel_name):
    """
    check_attr
    """
    if num_groups != 32:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "attr num_groups", "32", str(num_groups))
    if eps < 0:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "attr eps", "not support", eps)


def judge_split_c1(input_shape, data_type, num_groups):
    n, c1, h, w, c0 = input_shape
    c1_groups = c1 // num_groups
    block_size = 32
    data_size = 4
    data_num_repeat = 64
    ub_size = PlatformApi.get_soc_spec(PlatformApi.UB_SIZE)
    repeat_num = ceil_div(c1_groups * h * w * c0, data_num_repeat)
    data_num_align = repeat_num * data_num_repeat
    ub_size_use = (data_num_align * data_size * 2 + block_size * 3 +
                   c1_groups * c0 * data_size * 3 + repeat_num * data_size)
    if data_type == "float16":
        ub_size_use += c1_groups * c0 * 2 * 2
    return ub_size_use < ub_size


# 'pylint: disable=unused-variable,too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def group_norm_relu(input_0, input_1, input_2, output_0, num_groups, eps=0.00001, kernel_name="group_norm_relu"):
    """
    algorithm: GroupNorm and Reul operator
    Parameters
    ----------
    input_0:
        A Tensor. Must be one of the following types: float16, float32.
    input_1 :
        A Tensor. Must be one of the following types: float16, float32.
    input_2:
        A Tensor. Must be one of the following types: float16, float32.
    output_0 :
        A Tensor. Must be one of the following types: float16, float32.
    num_groups :
        A require attribute, the type is int32.
    eps :
        A optional attribute, the type is float32. Defaults to 0.00001/
    kernel_name : str
        cce kernel name, default value is top_k_3
    Returns
    -------
    None
    """
    check_input(input_0, input_1, input_2, output_0, kernel_name)
    check_attr(num_groups, eps, kernel_name)
    n, c1, h, w, c0 = input_0.get("shape")
    c = c1 * c0
    d = h * w
    data_type = input_0.get("dtype")
    if c1 >= num_groups:
        if judge_split_c1(input_0.get("shape"), data_type, num_groups):
            obj = GroupNormTilingC1(n, c, d, num_groups, eps, data_type, kernel_name)
        else:
            obj = GroupNormTilingC1D(n, c, d, num_groups, eps, data_type, kernel_name)
    else:
        if h >= 56:
            obj = GroupNormTilingD(n, c, d, num_groups, eps, data_type, kernel_name)
        else:
            obj = GroupNormTilingC0(n, c, d, num_groups, eps, data_type, kernel_name)
    obj.compute()
