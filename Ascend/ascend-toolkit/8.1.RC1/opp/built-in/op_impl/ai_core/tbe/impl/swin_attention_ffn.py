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
swin_transformer_ln_qkv
"""
import math
import functools

from impl.util.attention_qkv_util import Constant
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import PlatformApi
from impl.swin_transformer_ln_qkv import ceil_div
from impl.swin_transformer_ln_qkv import get_loop_info


# 'pylint: disable=unused-argument,unused-variable,too-many-arguments,too-many-locals
def check_supported(x, weight, bias, y, shifts=(), kernel_name="swin_attention_ffn"):
    """
    check_supported
    """
    soc_version = PlatformApi.get_soc_spec(PlatformApi.SHORT_SOC_VERSION)
    support_version = ("Ascend310P",)
    if soc_version not in support_version:
        return False, "not support short soc version"
    support_dtype = ("float16", "float32")
    input_params_all = (x, weight, bias, y)
    for input_params in input_params_all:
        if input_params.get("dtype").lower() not in support_dtype:
            return False, "not support data dtype"
    if len(shifts) == 4:
        roll_num = shifts[1]
        if tuple(shifts) != (0, roll_num, roll_num, 0):
            return False, "attr shifts not support"
    elif shifts:
        return False, "attr shifts not support"
    x_shape = tuple(x.get("ori_shape"))
    weight_shape = tuple(weight.get("ori_shape"))
    bias_shape = tuple(bias.get("ori_shape"))
    output_shape = tuple(y.get("ori_shape"))
    if len(output_shape) != 3:
        return False, "output shape not support"
    batch_num = output_shape[0]
    if batch_num < tbe_platform.get_soc_spec(tbe_platform.CORE_NUM):
        return False, "input shape not support"

    support_seq_len_all = (144, 256, 64)
    support_shape_all = (
        ((9216, 128), (9216, 192), (2304, 256), (2304, 384), (576, 512), (576, 768), (144, 1536)),
        ((4096, 96), (1024, 192), (256, 384)),
        ((4096, 128), (4096, 192), (1024, 256), (1024, 384), (256, 512), (256, 768), (64, 1024), (64, 1536)),
    )
    for seq_len, support_shape in zip(support_seq_len_all, support_shape_all):
        for m_num, k_num in support_shape:
            if check_input_shape(x_shape, weight_shape, bias_shape, output_shape, batch_num, m_num, k_num, seq_len):
                return True, ""
    return False, "input shape not support"


def check_input_shape(x_shape, weight_shape, bias_shape, y_shape, batch_num, m_num, k_num, seq_len):
    x_shape_support = (x_shape == (batch_num * m_num // seq_len, seq_len, k_num))
    weight_shape_support = (weight_shape == (k_num, k_num))
    bias_shape_support = (bias_shape == (k_num,))
    y_shape_support = (y_shape == (batch_num, m_num, k_num))

    support_all = (
        x_shape_support, weight_shape_support, bias_shape_support, y_shape_support,
    )
    if all(support_all):
        return True
    else:
        return False


def get_start_index(ori_index, ori_shape):
    """
    get_start_index
    """
    start_index = 0
    ori_shape = tuple(ori_shape) + (1,)
    for dim_num, dim_index in enumerate(ori_index):
        start_index += dim_index * functools.reduce(lambda a, b: a * b, ori_shape[dim_num + 1:])
    return start_index


class TikMatmulTensor:

    def __init__(self, tik_inst, l1_m1_num, l1_k1_num, l1_n1_num, l0_k1_num, thread_num=2):
        l1_type = "float16"
        l0a_type = "float16"
        l0b_type = "float16"
        l0c_type = "float32"
        ub_type = "float16"
        self.fractal_num = 16
        self.fractal_square = self.fractal_num * self.fractal_num
        self.tik_inst = tik_inst
        self.l1_m1_num = l1_m1_num
        self.l1_k1_num = l1_k1_num
        self.l1_n1_num = l1_n1_num
        self.l0_k1_num = l0_k1_num
        self.thread_num = thread_num

        x_l1_shape = (self.l1_k1_num, self.l1_m1_num, self.fractal_square)
        x_l1_element = functools.reduce(lambda a, b: a * b, x_l1_shape)
        self.x_l1 = tik_inst.Tensor(l1_type, (x_l1_element,), tik.scope_cbuf, "x_l1")

        weight_l1_shape = (self.l1_k1_num, self.l1_n1_num, self.fractal_square)
        weight_l1_element = functools.reduce(lambda a, b: a * b, weight_l1_shape)
        self.weight_l1 = tik_inst.Tensor(l1_type, (weight_l1_element,), tik.scope_cbuf, "weight_l1")

        x_l0a_shape = (self.l1_m1_num, self.l0_k1_num, self.fractal_square)
        x_l0a_element = functools.reduce(lambda a, b: a * b, x_l0a_shape)
        self.x_l0a = [None] * thread_num
        for thread_index in range(thread_num):
            tensor_name = "x_l0a_{}".format(thread_index)
            self.x_l0a[thread_index] = tik_inst.Tensor(l0a_type, (x_l0a_element,), tik.scope_ca, tensor_name)

        weight_l0b_shape = (self.l0_k1_num, self.l1_n1_num, self.fractal_square)
        weight_l0b_element = functools.reduce(lambda a, b: a * b, weight_l0b_shape)
        self.weight_l0b = [None] * thread_num
        for thread_index in range(thread_num):
            tensor_name = "weight_l0b_0{}".format(thread_index)
            self.weight_l0b[thread_index] = tik_inst.Tensor(l0b_type, (weight_l0b_element,), tik.scope_cb, tensor_name)

        result_shape = (self.l1_n1_num, self.l1_m1_num, self.fractal_square)
        result_element = functools.reduce(lambda a, b: a * b, result_shape)
        self.result_mul_l0c = tik_inst.Tensor(l0c_type, (result_element,), tik.scope_cc, "result_mul_l0c")
        self.result_mul_ub = tik_inst.Tensor(ub_type, (result_element,), tik.scope_ubuf, "result_mul_ub")

        bias_ub_shape = (l1_n1_num, self.fractal_num)
        bias_ub_element = functools.reduce(lambda a, b: a * b, bias_ub_shape)
        self.bias_ub_ori_type = tik_inst.Tensor(ub_type, (bias_ub_element,), tik.scope_ubuf, "bias_ub_ori_type")
        self.bias_ub_lc_type = tik_inst.Tensor(l0c_type, (bias_ub_element,), tik.scope_ubuf, "bias_ub_lc_type")

    def ping_pong_matmul(self, m1_num, k1_num, n1_num, each_loop_k1_num, thread_num=None):
        """
        ping_pong_matmul
        """
        if thread_num is not None:
            self.thread_num = thread_num

        k1_loop_times, last_loop_k1_num = get_loop_info(k1_num, each_loop_k1_num)
        ping_pong_loop_times = ceil_div(k1_loop_times, self.thread_num) - 1
        ping_pong_last_times = k1_loop_times - ping_pong_loop_times * self.thread_num

        if ping_pong_loop_times > 0:
            with self.tik_inst.for_range(0, ping_pong_loop_times) as ping_pong_index:
                for thread_index in range(self.thread_num):
                    loop_index = ping_pong_index * self.thread_num + thread_index
                    k1_index = loop_index * each_loop_k1_num
                    self.tik_matmul(self.x_l0a[thread_index], self.weight_l0b[thread_index],
                                    k1_index, m1_num, k1_num, n1_num, each_loop_k1_num)

        for thread_index in range(0, ping_pong_last_times):
            loop_index = ping_pong_loop_times * self.thread_num + thread_index
            k1_index = loop_index * each_loop_k1_num
            loop_k1_num = each_loop_k1_num if loop_index < k1_loop_times - 1 else last_loop_k1_num
            self.tik_matmul(self.x_l0a[thread_index], self.weight_l0b[thread_index],
                            k1_index, m1_num, k1_num, n1_num, loop_k1_num)

    def tik_matmul(self, tensor_l0a, tensor_l0b, k1_index_start, m1_num, k1_num, n1_num, loop_k1_num):
        """
        tik_matmul
        """
        with self.tik_inst.for_range(0, loop_k1_num) as k1_index:
            # x_l1[k1_index + k1_index_start, 0, 0, 0] -> tensor_l0a[0, k1_index, 0, 0]
            src_index = (k1_index + k1_index_start) * m1_num * self.fractal_square
            dst_index = k1_index * self.fractal_square
            self.tik_inst.load2dv2(tensor_l0a[dst_index:], self.x_l1[src_index:],
                                   0, m1_num, loop_k1_num - 1, 1, 0, False)

        with self.tik_inst.for_range(0, loop_k1_num) as k1_index:
            # weight_l1[0, k1_index + k1_index_start, 0, 0] -> tensor_l0b[k_index, 0, 0, 0]
            src_index = (k1_index + k1_index_start) * self.fractal_square
            dst_index = k1_index * n1_num * self.fractal_square
            self.tik_inst.load2dv2(tensor_l0b[dst_index:], self.weight_l1[src_index:],
                                   0, n1_num, 0, k1_num, 0, True)

        self.tik_inst.mmad(self.result_mul_l0c, tensor_l0a, tensor_l0b,
                           m1_num * self.fractal_num, loop_k1_num * self.fractal_num, n1_num * self.fractal_num,
                           1)


class TikComputeInterface:

    def __init__(self, tik_inst):
        self.tik_inst = tik_inst
        self.max_repeat = 255
        self.mask_cmd = {
            "float16": 128,
            "float32": 64
        }

    def start_tik_compute(self, element_num, dtype, func, args=None, begin_index=0):
        """
        start_tik_compute
        """
        mask = self.mask_cmd.get(dtype)
        repeat_num = element_num // mask
        last_num = element_num - repeat_num * mask
        loop_times, last_repeat = get_loop_info(repeat_num, self.max_repeat)
        for loop_index in range(loop_times):
            start_index = loop_index * self.max_repeat * mask + begin_index
            if loop_index != (loop_times - 1):
                func(mask, start_index, self.max_repeat, args)
            else:
                func(mask, start_index, last_repeat, args)
        if last_num != 0:
            start_index = repeat_num * mask + begin_index
            func(last_num, start_index, 1, args)

    def tik_dup(self, mask, start_index, repeat_num, args):
        dst, dup_data = args
        dst_flatten = dst.reshape((dst.size,))
        self.tik_inst.vector_dup(mask, dst_flatten[start_index], dup_data, repeat_num, 1, 8)

    def tik_vconv(self, mask, start_index, repeat_num, args):
        dst, src = args
        dst_flatten = dst.reshape((dst.size,))
        src_flatten = src.reshape((dst.size,))
        if dst.dtype == "float32" and src.dtype == "float16":
            stride_params = [1, 1, 8, 4]
        else:
            stride_params = [1, 1, 4, 8]
        self.tik_inst.vconv(mask, "", dst_flatten[start_index:, ], src_flatten[start_index:, ],
                            repeat_num, *stride_params)


class TilingArgs:

    def __init__(self, batch_num, m_0, m_1, m_2, m_3, k_0, k_1, n_0, n_1, dtype):
        self.batch_num = batch_num
        self.m_0 = m_0
        self.m_1 = m_1
        self.m_2 = m_2
        self.m_3 = m_3
        self.k_0 = k_0
        self.k_1 = k_1
        self.n_0 = n_0
        self.n_1 = n_1
        self.data_type = dtype
        self.fp32_type = "float32"

        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = PlatformApi.get_soc_spec(PlatformApi.UB_SIZE)
        self.l1_size = PlatformApi.get_soc_spec(PlatformApi.L1_SIZE)
        self.l0a_size = PlatformApi.get_soc_spec(PlatformApi.L0A_SIZE)
        self.l0b_size = PlatformApi.get_soc_spec(PlatformApi.L0B_SIZE)
        self.l0c_size = PlatformApi.get_soc_spec(PlatformApi.L0C_SIZE)

        self.each_core_batch_num = ceil_div(self.batch_num, self.core_num)
        self.core_num, self.last_core_batch_num = get_loop_info(self.batch_num, self.each_core_batch_num)

        self.m_1_num_unit = self.m_0 * self.m_3 // math.gcd(self.m_0, self.m_3) // self.m_0

        fp32_size = 4
        fp16_size = 2

        self.each_loop_n1_num = self.n_1

        self.each_loop_m1_num = min(self.l0a_size // 2 // fp16_size // self.m_0 // self.k_0,
                                    self.l0c_size // fp32_size // self.m_0 // self.n_0 // self.each_loop_n1_num)
        self.each_loop_m1_num = self.get_align_m1_num(self.each_loop_m1_num)
        self.each_loop_k1_num = min(self.k_1, self.l1_size // fp16_size // self.m_0 // self.n_0 // (
                    self.each_loop_n1_num + self.each_loop_m1_num))

        if self.m_1 * self.m_0 * self.k_1 * self.k_0 * fp16_size <= self.l1_size // 2:
            self.each_loop_m1_num = self.m_1
            self.each_loop_k1_num = self.k_1
            self.each_loop_n1_num = min((self.l1_size // fp16_size // self.m_0 // self.n_0 // self.each_loop_k1_num
                                         - self.each_loop_m1_num),
                                        self.l0b_size // 2 // fp16_size // self.n_0 // self.k_0,
                                        self.l0c_size // fp32_size // self.m_0 // self.n_0 // self.each_loop_m1_num)

        self.each_mul_k_num = min(self.l0a_size // 2 // fp16_size // self.m_0 // self.k_0 // self.each_loop_m1_num,
                                  self.l0b_size // 2 // fp16_size // self.k_0 // self.n_0 // self.each_loop_n1_num)

        self.split_x = (self.each_loop_m1_num != self.m_1 or self.each_loop_k1_num != self.k_1)
        self.split_weight = (self.each_loop_n1_num != self.n_1 or self.each_loop_k1_num != self.k_1)

    def get_align_m1_num(self, each_loop_m1_num):
        """
        align m1 num
        """
        m_1_num_unit = self.m_0 * self.m_3 // math.gcd(self.m_0, self.m_3) // self.m_0
        each_loop_m1_num_max = m_1_num_unit
        for m_1_num in range(m_1_num_unit, each_loop_m1_num, m_1_num_unit):
            m_num = m_1_num * self.m_0
            if self.judge_div(m_num):
                each_loop_m1_num_max = m_1_num
        return each_loop_m1_num_max

    def judge_div(self, m_num):
        if m_num <= self.m_2 * self.m_3:
            if self.m_2 * self.m_3 % m_num == 0:
                return True
        elif m_num <= self.m_3 * self.m_2 * self.m_3:
            if self.m_3 * self.m_2 * self.m_3 % m_num == 0:
                return True
        elif m_num <= self.m_2 * self.m_3 * self.m_2 * self.m_3:
            if self.m_2 * self.m_3 * self.m_2 * self.m_3 % m_num == 0:
                return True
        return False


class SwinTransformMatmul(TilingArgs):

    def __init__(self, batch_num, m_0, m_1, m_2, m_3, k_0, k_1, n_0, n_1, dtype, m_split, kernel_name):
        # init tiling param
        super().__init__(batch_num, m_0, m_1, m_2, m_3, k_0, k_1, n_0, n_1, dtype)
        # init attr
        self.m_split = m_split
        self.m_num_sqrt = self.m_2 * self.m_3
        self.kernel_name = kernel_name
        self.m_num = self.m_1 * self.m_0
        self.n_num = self.n_1 * self.n_0
        self.tik_inst = tik.Tik()
        # init tik fun
        self.tik_func = TikComputeInterface(self.tik_inst)
        self.fractal_num = 16
        self.fractal_square = self.fractal_num * self.fractal_num
        # gm shape
        x_shape = (self.batch_num * self.m_2 * self.m_2, self.k_1, self.m_3 * self.m_3, self.k_0)
        x_element = functools.reduce(lambda a, b: a * b, x_shape)
        weight_shape = (self.n_1, self.k_1, self.k_0, self.n_0)
        weight_element = functools.reduce(lambda a, b: a * b, weight_shape)
        bias_shape = (self.n_1 * self.n_0,)
        bias_element = functools.reduce(lambda a, b: a * b, bias_shape)
        output_shape = (self.batch_num, self.n_1, self.m_num, self.n_0)
        output_element = functools.reduce(lambda a, b: a * b, output_shape)
        # init input gm tensor
        self.x_gm = self.tik_inst.Tensor(self.data_type, (x_element,), tik.scope_gm, "x_gm")
        self.weight_gm = self.tik_inst.Tensor(self.data_type, (weight_element,), tik.scope_gm, "weight_gm")
        self.bias_gm = self.tik_inst.Tensor(self.data_type, (bias_element,), tik.scope_gm, "bias_gm")
        self.inputs = [self.x_gm, self.weight_gm, self.bias_gm]
        # init output gm tensor
        self.y_gm = self.tik_inst.Tensor(self.data_type, (output_element,), tik.scope_gm, "y_gm")
        self.outputs = [self.y_gm]

    def start_compute(self):
        """
        start compute
        """
        if self.each_core_batch_num == self.last_core_batch_num:
            with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_index:
                self.start_compute_each_core(core_index, self.each_core_batch_num)
        else:
            with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_index:
                with self.tik_inst.if_scope(core_index < self.core_num - 1):
                    self.start_compute_each_core(core_index, self.each_core_batch_num)
                with self.tik_inst.else_scope():
                    self.start_compute_each_core(core_index, self.last_core_batch_num)
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=self.inputs,
                               outputs=self.outputs,
                               config={"enable_const_fold": True}
                               )

    def start_compute_each_core(self, core_index, batch_num):
        tik_matmul = TikMatmulTensor(self.tik_inst, self.each_loop_m1_num, self.each_loop_k1_num,
                                     self.each_loop_n1_num, self.each_mul_k_num)
        if not self.split_weight:
            weight_block_num = self.k_1 * self.n_1 * self.k_0 * self.n_0 // Constant.NUM_FP16_PER_BLOCK
            self.tik_inst.data_move(tik_matmul.weight_l1, self.weight_gm, 0, 1, weight_block_num, 0, 0)
        if self.each_loop_n1_num == self.n_1:
            self._init_bias_data(tik_matmul.bias_ub_lc_type, tik_matmul.bias_ub_ori_type, self.bias_gm, 0,
                                 self.each_loop_n1_num)
        with self.tik_inst.for_range(0, batch_num) as core_batch_index:
            batch_index = core_index * self.each_core_batch_num + core_batch_index
            self.start_compute_each_batch(batch_index, tik_matmul)

    def start_compute_each_batch(self, batch_index, tik_matmul: TikMatmulTensor):
        if not self.split_x:
            self._init_x_data(tik_matmul.x_l1, self.x_gm,
                              batch_index, 0, 0,
                              self.each_loop_m1_num, self.each_loop_k1_num)
        each_loop_m1_num = self.each_loop_m1_num
        m1_loop_times = self.m_1 // each_loop_m1_num
        for m1_loop_index in range(m1_loop_times):
            m1_index_start = m1_loop_index * each_loop_m1_num
            self.start_compute_split_n1(batch_index, m1_index_start, each_loop_m1_num, tik_matmul)

    def start_compute_split_n1(self, batch_index, m1_index_start, m1_num, tik_matmul):
        """
        split n1
        """
        each_loop_n1_num = self.each_loop_n1_num
        n1_loop_times, last_loop_n1_num = get_loop_info(self.n_1, each_loop_n1_num)
        if last_loop_n1_num == each_loop_n1_num:
            with self.tik_inst.for_range(0, n1_loop_times) as n1_loop_index:
                n1_index_start = n1_loop_index * each_loop_n1_num
                self.start_compute_split_k1(batch_index, m1_index_start, n1_index_start,
                                            m1_num, each_loop_n1_num, tik_matmul)
        else:
            with self.tik_inst.for_range(0, n1_loop_times) as n1_loop_index:
                n1_index_start = n1_loop_index * each_loop_n1_num
                with self.tik_inst.if_scope(n1_loop_index < n1_loop_times - 1):
                    self.start_compute_split_k1(batch_index, m1_index_start, n1_index_start,
                                                m1_num, each_loop_n1_num, tik_matmul)
                with self.tik_inst.else_scope():
                    self.start_compute_split_k1(batch_index, m1_index_start, n1_index_start,
                                                m1_num, last_loop_n1_num, tik_matmul)

    def start_compute_split_k1(self, batch_index, m1_index_start, n1_index_start, m1_num, n1_num, tik_matmul):
        """
        split k1
        """
        if n1_num != self.n_1:
            self._init_bias_data(tik_matmul.bias_ub_lc_type, tik_matmul.bias_ub_ori_type, self.bias_gm,
                                 n1_index_start, n1_num)
        self._init_lc_data(tik_matmul.result_mul_l0c, tik_matmul.bias_ub_lc_type,
                           m1_num, n1_num)

        each_loop_k1_num = self.each_loop_k1_num
        k1_loop_times, last_loop_k1_num = get_loop_info(self.k_1, each_loop_k1_num)
        if last_loop_k1_num == each_loop_k1_num:
            with self.tik_inst.for_range(0, k1_loop_times) as k1_loop_index:
                k1_index_start = k1_loop_index * each_loop_k1_num
                self.start_compute_loop(batch_index, m1_index_start, n1_index_start, k1_index_start,
                                        m1_num, n1_num, each_loop_k1_num, tik_matmul)
        else:
            with self.tik_inst.for_range(0, k1_loop_times) as k1_loop_index:
                k1_index_start = k1_loop_index * each_loop_k1_num
                with self.tik_inst.if_scope(k1_loop_index < k1_loop_times - 1):
                    self.start_compute_loop(batch_index, m1_index_start, n1_index_start, k1_index_start,
                                            m1_num, n1_num, each_loop_k1_num, tik_matmul)
                with self.tik_inst.else_scope():
                    self.start_compute_loop(batch_index, m1_index_start, n1_index_start, k1_index_start,
                                            m1_num, n1_num, last_loop_k1_num, tik_matmul)

        self.data_move_out(batch_index, m1_index_start, n1_index_start, m1_num, n1_num, tik_matmul)

    def start_compute_loop(self, batch_index, m1_index_start, n1_index_start, k1_index_start,
                           m1_num, n1_num, k1_num, tik_matmul):
        if self.split_x:
            self._init_x_data(tik_matmul.x_l1, self.x_gm,
                              batch_index, m1_index_start, k1_index_start,
                              m1_num, k1_num)
        if self.split_weight:
            self._init_weight_data(tik_matmul.weight_l1, self.weight_gm,
                                   n1_index_start, k1_index_start,
                                   n1_num, k1_num)
        tik_matmul.ping_pong_matmul(m1_num, k1_num, n1_num, self.each_mul_k_num)

    def get_m_3_index(self, m_index_start):
        m_3_1 = m_index_start % self.m_3
        m_3_0 = m_index_start // (self.m_3 * self.m_2) % self.m_3
        return m_3_0, m_3_1

    def get_m_2_index(self, m_index_start):
        m_2_1 = m_index_start // self.m_3 % self.m_2
        m_2_0 = m_index_start // (self.m_3 * self.m_2 * self.m_3)
        return m_2_0, m_2_1

    def data_move_out(self, batch_index, m1_index_start, n1_index_start, m1_num, n1_num, tik_matmul):
        """
        data move out
        """
        self.tik_inst.tensor_mov(tik_matmul.result_mul_ub, tik_matmul.result_mul_l0c, 'm', 1,
                                 m1_num * n1_num, 0, 0)
        if self.m_split == 0:
            gm_shape = (self.batch_num, self.n_1, self.m_1, self.m_0, self.n_0)
            gm_start_index = get_start_index((batch_index, n1_index_start, m1_index_start), gm_shape)
            ub_start_index = 0
            nburst = n1_num
            burst = m1_num * self.m_0 * self.k_0 // Constant.NUM_FP16_PER_BLOCK
            src_gap = 0
            dst_gap = (self.m_1 - m1_num) * self.m_0 * self.k_0 // Constant.NUM_FP16_PER_BLOCK
            self.tik_inst.data_move(self.y_gm[gm_start_index:], tik_matmul.result_mul_ub[ub_start_index:],
                                    0, nburst, burst, src_gap, dst_gap)
        else:
            self.data_move_out_split(batch_index, m1_index_start, n1_index_start, m1_num, n1_num, tik_matmul)

    def data_move_out_split(self, batch_index, m1_index_start, n1_index_start, m1_num, n1_num, tik_matmul):
        """
        split concat move out
        """
        m_index_start = m1_index_start * self.m_0
        m_num_move = m1_num * self.m_0
        m_split = self.m_num_sqrt - self.m_split
        if m_num_move <= self.m_num_sqrt:
            m_index_0 = m_index_start // self.m_num_sqrt
            m_index_1_start = m_index_start % self.m_num_sqrt
            m_index_1_end = m_index_start + m_num_move
            if m_index_0 <= m_split:
                m_index_0 += self.m_split
            else:
                m_index_0 = m_index_0 - m_split

            if m_index_1_end < m_split:
                self.data_move_out_split_second_0(self.y_gm, tik_matmul.result_mul_ub,
                                                  batch_index, m_index_0, m_index_1_start, 0, n1_index_start,
                                                  m_num_move, m_num_move, n1_num)
            elif m_index_1_start >= m_split:
                self.data_move_out_split_second_1(self.y_gm, tik_matmul.result_mul_ub,
                                                  batch_index, m_index_0, m_index_1_start, 0, n1_index_start,
                                                  m_num_move, m_num_move, n1_num)
            else:
                m_num_first_move = m_split - m_index_1_start
                self.data_move_out_split_second_0(self.y_gm, tik_matmul.result_mul_ub,
                                                  batch_index, m_index_0, m_index_1_start, 0, n1_index_start,
                                                  m_num_move, m_num_first_move, n1_num)
                m_num_second_move = m_index_1_end - m_split
                self.data_move_out_split_second_1(self.y_gm, tik_matmul.result_mul_ub,
                                                  batch_index, m_index_0,
                                                  m_index_1_start + m_num_first_move, m_num_first_move, n1_index_start,
                                                  m_num_move, m_num_second_move, n1_num)
        else:
            m_index_0_start = m_index_start // self.m_num_sqrt
            m_index_0_end = (m_index_start + m_num_move) // self.m_num_sqrt
            m_index_ub = 0
            for m_index_0 in range(m_index_0_start, m_index_0_end):
                if m_index_0 < m_split:
                    m_index_0 += self.m_split
                else:
                    m_index_0 = m_index_0 - m_split

                self.data_move_out_split_second(self.y_gm, tik_matmul.result_mul_ub, batch_index, m_index_0,
                                                m_index_ub, n1_index_start, m_num_move, n1_num)
                m_index_ub += self.m_num_sqrt

    def data_move_out_split_second(self, result_gm, result_ub, batch_index, m_index_0, m_index_ub, n1_index_start,
                                   m_num_ub, n1_num):
        """
        data move out split second m_sqrt
        """
        m_num_first_move = self.m_num_sqrt - self.m_split
        self.data_move_out_split_second_0(result_gm, result_ub,
                                          batch_index, m_index_0,
                                          0, m_index_ub, n1_index_start,
                                          m_num_ub, m_num_first_move, n1_num)
        m_num_second_move = self.m_split
        self.data_move_out_split_second_1(result_gm, result_ub,
                                          batch_index, m_index_0,
                                          m_num_first_move, m_index_ub + m_num_first_move, n1_index_start,
                                          m_num_ub, m_num_second_move, n1_num)

    def data_move_out_split_second_0(self, result_gm, result_ub,
                                     batch_index, m_index_0, m_index_1_start, m_index_ub, n1_index_start,
                                     m_num_ub, m_num_move, n1_num):
        gm_shape = (self.batch_num, self.n_1, self.m_num_sqrt, self.m_num_sqrt, self.n_0)
        gm_start_index = (batch_index, n1_index_start, m_index_0, self.m_split + m_index_1_start)
        ub_shape = (n1_num, m_num_ub, self.n_0)
        ub_start_index = (0, m_index_ub)
        gm_start_index_flatten = get_start_index(gm_start_index, gm_shape)
        ub_start_index_flatten = get_start_index(ub_start_index, ub_shape)
        nburst = n1_num
        burst = m_num_move * self.k_0 // Constant.NUM_FP16_PER_BLOCK
        src_gap = (m_num_ub - m_num_move) * self.k_0 // Constant.NUM_FP16_PER_BLOCK
        dst_gap = (self.m_num_sqrt * self.m_num_sqrt - m_num_move) * self.k_0 // Constant.NUM_FP16_PER_BLOCK
        self.tik_inst.data_move(result_gm[gm_start_index_flatten:], result_ub[ub_start_index_flatten:],
                                0, nburst, burst, src_gap, dst_gap)

    def data_move_out_split_second_1(self, result_gm, result_ub,
                                     batch_index, m_index_0, m_index_1_start, m_index_ub, n1_index_start,
                                     m_num_ub, m_num_move, n1_num):
        start_shift_dim = self.m_num_sqrt - self.m_split

        gm_shape = (self.batch_num, self.n_1, self.m_num_sqrt, self.m_num_sqrt, self.n_0)
        gm_start_index = (batch_index, n1_index_start, m_index_0, m_index_1_start - start_shift_dim)
        ub_shape = (n1_num, m_num_ub, self.n_0)
        ub_start_index = (0, m_index_ub)
        gm_start_index_flatten = get_start_index(gm_start_index, gm_shape)
        ub_start_index_flatten = get_start_index(ub_start_index, ub_shape)
        nburst = n1_num
        burst = m_num_move * self.k_0 // Constant.NUM_FP16_PER_BLOCK
        src_gap = (m_num_ub - m_num_move) * self.k_0 // Constant.NUM_FP16_PER_BLOCK
        dst_gap = (self.m_num_sqrt * self.m_num_sqrt - m_num_move) * self.k_0 // Constant.NUM_FP16_PER_BLOCK
        self.tik_inst.data_move(result_gm[gm_start_index_flatten:], result_ub[ub_start_index_flatten:],
                                0, nburst, burst, src_gap, dst_gap)

    def _init_x_data(self, tensor_l1, tensor_gm, batch_index, m1_index_start, k1_index_start, m1_num, k1_num):
        m_num = m1_num * self.m_0
        l1_shape = (k1_num, m_num, self.k_0)
        gm_shape = (self.batch_num, self.m_2, self.m_2, self.k_1, self.m_3, self.m_3, self.k_0)

        m_loop_times = m_num // self.m_3
        for m_loop_index_times in range(m_loop_times):
            m_index_loop = m_loop_index_times * self.m_3
            m_index_start = m1_index_start * self.m_0 + m_index_loop
            m_2_0, m_2_1 = self.get_m_2_index(m_index_start)
            m_3_0, m_3_1 = self.get_m_3_index(m_index_start)
            l1_start_index = get_start_index((0, m_index_loop), l1_shape)
            gm_start_index = get_start_index((batch_index, m_2_0, m_2_1, k1_index_start, m_3_0, m_3_1), gm_shape)
            nburst = k1_num
            burst = self.m_3 * self.k_0 // Constant.NUM_FP16_PER_BLOCK
            src_gap = (self.m_3 * self.m_3 - self.m_3) * self.k_0 // Constant.NUM_FP16_PER_BLOCK
            dst_gap = (m_num - self.m_3) * self.k_0 // Constant.NUM_FP16_PER_BLOCK
            self.tik_inst.data_move(tensor_l1[l1_start_index:], tensor_gm[gm_start_index:],
                                    0, nburst, burst, src_gap, dst_gap)

    def _init_weight_data(self, tensor_l1, tensor_gm, n1_index_start, k1_index_start, n1_num, k1_num):
        weight_block_num = k1_num * self.fractal_square // Constant.NUM_FP16_PER_BLOCK
        src_gap = (self.k_1 - k1_num) * self.fractal_square // Constant.NUM_FP16_PER_BLOCK
        weight_index = (n1_index_start * self.k_1 + k1_index_start) * self.fractal_square
        self.tik_inst.data_move(tensor_l1,
                                tensor_gm[weight_index:], 0,
                                n1_num, weight_block_num,
                                src_gap, 0)

    def _init_bias_data(self, tensor_lc_ub, tensor_ub, tensor_gm, n1_index, n1_num):
        n_num_move = n1_num * self.n_0
        bias_block_num = n_num_move // Constant.NUM_FP16_PER_BLOCK
        n_index_gm = n1_index * self.n_0
        self.tik_inst.data_move(tensor_ub, tensor_gm[n_index_gm], 0, 1, bias_block_num, 0, 0)
        self.tik_func.start_tik_compute(n1_num * self.n_0, self.fp32_type,
                                        self.tik_func.tik_vconv,
                                        args=(tensor_lc_ub, tensor_ub))

    def _init_lc_data(self, tensor_lc, tensor_ub, m1_num, n1_num):
        with self.tik_inst.for_range(0, m1_num) as m1_index:
            lc_start_index = m1_index * self.fractal_square
            self.tik_inst.broadcast_ub_to_l0c(tensor_lc[lc_start_index:],
                                              tensor_ub,
                                              n1_num, 1, 0, m1_num - 1)


# 'pylint: disable = unused-argument
# 'pylint: disable=too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def swin_attention_ffn(x, weight, bias, y, shifts=(), kernel_name="swin_attention_ffn"):
    """
    Parameters
    ----------
    x: dict
        shape and dtype of input x, only support float16
    weight: dict
        shape and dtype of input kernel_value, only support float16
    bias: dict
        shape and dtype of input gamma, only support float16
    y: dict
        shape and dtype of output, only support float16
    shifts: list_int
        roll dim of transformer Attention
    kernel_name: str
        cce kernel name, default value is "swin_attention_ffn"
    Returns
    -------
    None
    """
    m_0, k_0, n_0 = 16, 16, 16
    batch_num_x, k_1, m_3_0 = x.get("shape")[:3]
    n_1 = weight.get("shape")[0]
    dtype = x.get("dtype")
    batch_num = y.get("shape")[0]
    m_2 = int(math.sqrt(batch_num_x // batch_num))
    m_3 = int(math.sqrt(m_3_0 * m_0))
    m_1 = m_2 * m_2 * m_3 * m_3 // m_0
    if shifts:
        m_split = shifts[1]
    else:
        m_split = 0
    obj = SwinTransformMatmul(batch_num, m_0, m_1, m_2, m_3, k_0, k_1, n_0, n_1, dtype, m_split, kernel_name)
    obj.start_compute()
