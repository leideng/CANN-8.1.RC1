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
swin_transformer_ln_qkv_v2
"""
import math
import functools

from impl.util.attention_qkv_util import Constant
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import PlatformApi


# 'pylint: disable=unused-argument,unused-variable,too-many-arguments,too-many-locals
def check_supported(x, gamma, beta, weight, bias,
                    query_output, key_output, value_output,
                    head_num, head_dim, seq_length, shifts=(), epsilon=1e-7,
                    kernel_name="swin_transformer_ln_qkv"):
    """
    check_supported
    """
    x_shape = tuple(x.get("ori_shape"))
    batch_num, m_num, k_num = x_shape
    if batch_num < 4:
        return False
    support_attr_all = (
        (144, 32),
        (256, 32),
        (64, 32),
        (49, 32),
        (196, 32)
    )
    support_shape_all = (((9216, 128), (2304, 256), (576, 512), \
                          (9216, 192), (2304, 384), (576, 768), \
                          (9216, 96), (2304, 192), (576, 384), \
                          (144, 1024)), \
                         ((1024, 96), (256, 192), (64, 384), (16, 768)), \
                         ((4096, 192), (1024, 384), (256, 768), \
                          (4096, 96), (1024, 192), (256, 384), \
                          (4096, 128), (1024, 256), (256, 512), \
                          (65536, 192)), \
                         ((3136, 96), (784, 192), (196, 384), \
                          (3136, 128), (784, 256), (49, 1024), \
                          (3136, 192), (784, 384), (196, 768),
                          (196, 512), (49, 768), (49, 1536)),
                         ((196, 384), (3136, 96), (784, 192))
                        )
    gamma_shape = tuple(gamma.get("ori_shape"))
    beta_shape = tuple(beta.get("ori_shape"))
    weight_shape = tuple(weight.get("ori_shape"))
    bias_shape = tuple(bias.get("ori_shape"))
    query_output_shape = tuple(query_output.get("ori_shape"))
    key_output_shape = tuple(key_output.get("ori_shape"))
    value_output_shape = tuple(value_output.get("ori_shape"))
    for support_attr, support_shape in zip(support_attr_all, support_shape_all):
        if (seq_length, head_dim) == support_attr and (m_num, k_num) in support_shape:
            if check_input_shape(batch_num, m_num, k_num, head_num, head_dim, seq_length,
                                 x_shape, gamma_shape, beta_shape, weight_shape, bias_shape,
                                 query_output_shape, key_output_shape, value_output_shape):
                return True
    return False


# 'pylint: disable=unused-argument,unused-variable,too-many-arguments,too-many-locals
def check_input_shape(batch_num, m_num, k_num, head_num, head_dim, seq_length,
                      x_shape, gamma_shape, beta_shape, weight_shape, bias_shape,
                      query_output_shape, key_output_shape, value_output_shape):
    x_shape_support = (x_shape == (batch_num, m_num, k_num))
    gamma_shape_support = (len(gamma_shape) > 0 and functools.reduce(lambda a, b: a * b, gamma_shape) == k_num)
    beta_shape_support = (len(beta_shape) > 0 or functools.reduce(lambda a, b: a * b, beta_shape) == k_num)
    weight_shape_support = (weight_shape == (k_num, 3 * k_num))
    bias_shape_support = (bias_shape == (3 * k_num,))

    head_support = (head_dim * head_num == k_num)
    output_shape_label = (batch_num * m_num // seq_length, head_num, seq_length, head_dim)

    query_output_shape_support = (query_output_shape == output_shape_label)
    key_output_shape_support = (key_output_shape == output_shape_label)
    value_output_support = (value_output_shape == output_shape_label)
    support_all = (
        x_shape_support, gamma_shape_support, beta_shape_support, weight_shape_support, bias_shape_support,
        head_support, query_output_shape_support, key_output_shape_support, value_output_support
    )
    if all(support_all):
        return True
    else:
        return False


def ceil_div(dividend, divisor):
    return (dividend + divisor - 1) // divisor


def get_align_num(data_num, align_num):
    return ceil_div(data_num, align_num) * align_num


def get_loop_info(all_data_num, each_loop_num):
    """
    get loop info
    """
    loop_times = ceil_div(all_data_num, each_loop_num)
    last_loop_num = all_data_num - each_loop_num * (loop_times - 1)
    return loop_times, last_loop_num


def get_element_num(input_shape):
    return functools.reduce(lambda a, b: a * b, input_shape)


def get_start_index(ori_index, ori_shape):
    """
    get_start_index
    """
    start_index = 0
    ori_shape = tuple(ori_shape) + (1,)
    for dim_num, dim_index in enumerate(ori_index):
        start_index += dim_index * functools.reduce(lambda a, b: a * b, ori_shape[dim_num + 1:])
    return start_index


class TikComputeInterface:

    def __init__(self, tik_inst):
        self.tik_inst = tik_inst
        self.max_repeat = 255
        self.mask_cmd = {
            "float16": 128,
            "float32": 64
        }

    def start_tik_compute(self, element_num, dtype, func, args=None, begin_index=0):
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

    def __init__(self, batch_num, m_0, m_1, m_2, m_3, k_0, k_1, n_0, n_1, n_2, n_3, result_shape, dtype, m_split,
                 split_weight, unalign, m_unalign):
        self.batch_num = batch_num
        self.m_0 = m_0
        self.m_1 = m_1
        self.m_2 = m_2
        self.m_3 = m_3
        self.k_0 = k_0
        self.k_1 = k_1
        self.n_0 = n_0
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        self.result_shape = result_shape
        self.data_type = dtype
        self.fp32_type = "float32"
        self.split_weight = split_weight
        self.unalign = unalign
        self.m_unalign = m_unalign

        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = PlatformApi.get_soc_spec(PlatformApi.UB_SIZE)
        self.l1_size = PlatformApi.get_soc_spec(PlatformApi.L1_SIZE)
        self.l0a_size = PlatformApi.get_soc_spec(PlatformApi.L0A_SIZE)
        self.l0b_size = PlatformApi.get_soc_spec(PlatformApi.L0B_SIZE)
        self.l0c_size = PlatformApi.get_soc_spec(PlatformApi.L0C_SIZE)

        self.m_1_num_unit = self.m_0 * self.m_3 // math.gcd(self.m_0, self.m_3) // self.m_0

        self.fp32_size = 4
        self.fp16_size = 2
        self.each_ub_m1_num = 1
        self.each_loop_k1_num = self.k_1

        if self.split_weight:
            self.each_loop_n1_num = self.n_1 // 3 // self.core_num
            self.each_core_batch_num = self.batch_num
            self.last_core_batch_num = self.batch_num
        else:
            self.each_core_batch_num = ceil_div(self.batch_num, self.core_num)
            self.core_num, self.last_core_batch_num = get_loop_info(self.batch_num, self.each_core_batch_num)
            if unalign:
                self.each_loop_n1_num = self.n_1 // 3 // 2
                if self.n_1 in (48, 36):
                    self.each_loop_n1_num = self.n_1 // 3 // 2 // 2
                elif self.n_1 in (72, 144):
                    self.each_loop_n1_num = self.n_1 // 3 // 6
                elif self.n_1 in (96,):
                    self.each_loop_n1_num = self.n_1 // 3 // 4
                elif self.n_1 in (192, 288):
                    self.each_loop_n1_num = self.n_1 // 3 // 4
                    self.each_loop_k1_num = self.k_1 // 2
            else:
                self.each_loop_n1_num = self.n_1 // 3
                if self.n_1 in (48, 72, 36):
                    self.each_loop_n1_num = self.n_1 // 3 // 2
                elif self.n_1 in (96, 144, 72):
                    self.each_loop_n1_num = self.n_1 // 3 // 4
                elif self.n_1 in (192, 288, 144):
                    self.each_loop_n1_num = self.n_1 // 3 // 4
                    self.each_loop_k1_num = self.k_1 // 2

        self.each_loop_m1_num = min(self.l0a_size // 2 // self.fp16_size // self.m_0 // self.k_0,
                                    self.l0c_size // self.fp32_size // self.m_0 // self.n_0 // self.each_loop_n1_num,
                                    (self.l1_size - 3 * self.each_loop_n1_num * self.each_loop_k1_num * self.k_0 *
                                    self.n_0 * self.fp16_size) // self.m_0 // self.k_0 // self.k_1 // self.fp16_size)

        self.each_loop_m1_num = self.get_align_m1_num(self.each_loop_m1_num)

        if m_unalign:
            self.each_loop_m1_num = self.m_1

        self.each_mul_k_num = min(self.l0a_size // 2 // self.fp16_size // self.m_0 // self.k_0 // self.each_loop_m1_num,
                                  self.l0b_size // 2 // self.fp16_size // self.k_0 // self.n_0 // self.each_loop_n1_num)

    def get_max_m1_num_l1(self):
        fp16_size = 2
        fp32_size = 4
        ln_l1_space_all = self.m_0 * self.k_0 * fp16_size + self.k_1 * self.k_0 * fp32_size * 2
        mul_l1_space_all = (self.n_1 * self.n_0 * fp32_size +
                            self.each_loop_n1_num * self.each_loop_k1_num * self.k_0 * self.n_0 * fp16_size)
        l1_space_left = self.l1_size - ln_l1_space_all - mul_l1_space_all
        each_loop_m1_num_max = l1_space_left // (self.k_1 * self.m_0 * self.k_0 * fp16_size * 2)
        return each_loop_m1_num_max

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
            if self.m_3 * self.m_2 * self.m_3 % m_num == 0 and m_num % (self.m_2 * self.m_3) == 0:
                return True
        elif m_num <= self.m_2 * self.m_3 * self.m_2 * self.m_3:
            if self.m_2 * self.m_3 * self.m_2 * self.m_3 % m_num == 0 and m_num % (self.m_3 * self.m_2 * self.m_3) == 0:
                return True
        return False


class GlobalTensor:

    def __init__(self, gamma_fp32_ub, beta_fp32_ub, bias_fp32_ub, ln_one_l1, x_l1, ln_res_l1, weight_l1,
                 result_ub, x_ub, x_fp32_ub, x_sum_ub, xx_sum_ub, squared_mean_ub, var_ub, result_lc):
        self.gamma_fp32_ub = gamma_fp32_ub
        self.beta_fp32_ub = beta_fp32_ub
        self.bias_fp32_ub = bias_fp32_ub
        self.ln_one_l1 = ln_one_l1
        self.x_l1 = x_l1
        self.ln_res_l1 = ln_res_l1
        self.weight_l1 = weight_l1
        self.result_ub = result_ub
        self.x_ub = x_ub
        self.x_fp32_ub = x_fp32_ub
        self.x_sum_ub = x_sum_ub
        self.xx_sum_ub = xx_sum_ub
        self.squared_mean_ub = squared_mean_ub
        self.var_ub = var_ub
        self.result_lc = result_lc


class SwinTransformLnMatmul(TilingArgs):

    def __init__(self, batch_num, m_0, m_1, m_2, m_3, k_0, k_1, n_0, n_1, n_2, n_3, result_shape, dtype,
                 m_split, split_weight, unalign, m_unalign, epsilon, kernel_name):
        # init tiling param
        super().__init__(batch_num, m_0, m_1, m_2, m_3, k_0, k_1, n_0, n_1, n_2, n_3, result_shape, dtype,
                         m_split, split_weight, unalign, m_unalign)
        # init attr
        self.epsilon = epsilon
        self.m_split = m_split
        self.kernel_name = kernel_name
        self.mean_coeff = 1.0 / (self.k_1 * self.k_0)
        self.m_num_sqrt = self.m_2 * self.m_3
        self.m_num = self.m_1 * self.m_0
        self.m_sqrt = int(math.sqrt(self.m_num))
        self.tik_inst = tik.Tik()
        self.num_fp16_per_block = 16
        self.num_fp32_per_block = 8
        # init tik fun
        self.tik_func = TikComputeInterface(self.tik_inst)
        # gm shape
        x_shape = (self.batch_num, self.k_1, self.m_1, self.m_0, self.k_0)
        gamma_shape = (self.k_1 * self.k_0,)
        weight_shape = (self.n_1, self.k_1, self.k_0, self.n_0)
        bias_shape = (self.n_1 * self.n_0,)
        # output_shape ND：(batch_num * m_2 * m_2, head_num, m3 * m3, head_dim)
        # output_shape NZ：(batch_num * m_2 * m_2, head_num, n_2 * n_3, m3 * m3, n_0)
        if self.unalign:
            output_shape = (self.batch_num, self.m_2, self.m_2, self.n_2 * self.n_3, \
                            self.result_shape[3], self.result_shape[4], self.n_0)
        else:
            output_shape = (self.batch_num, self.m_2, self.m_2, self.n_2 * self.n_3, self.m_3, self.m_3, self.n_0)
        # init input gm tensor
        self.x_gm = self.tik_inst.Tensor(self.data_type, x_shape, tik.scope_gm, "x_gm")
        self.gamma_gm = self.tik_inst.Tensor(self.data_type, gamma_shape, tik.scope_gm, "gamma_gm")
        self.beta_gm = self.tik_inst.Tensor(self.data_type, gamma_shape, tik.scope_gm, "beta_gm")
        self.weight_gm = self.tik_inst.Tensor(self.data_type, weight_shape, tik.scope_gm, "weight_gm")
        self.bias_gm = self.tik_inst.Tensor(self.data_type, bias_shape, tik.scope_gm, "bias_gm")
        self.inputs = [self.x_gm, self.gamma_gm, self.beta_gm, self.weight_gm, self.bias_gm]
        # init output gm tensor
        if self.unalign:
            self.query_output_gm = self.tik_inst.Tensor(self.data_type, output_shape, tik.scope_gm, "query_output_gm",
                                                        is_atomic_add=True)
            self.key_output_gm = self.tik_inst.Tensor(self.data_type, output_shape, tik.scope_gm, "key_output_gm",
                                                      is_atomic_add=True)
            self.value_output_gm = self.tik_inst.Tensor(self.data_type, output_shape, tik.scope_gm, "value_output_gm",
                                                        is_atomic_add=True)
        else:
            self.query_output_gm = self.tik_inst.Tensor(self.data_type, output_shape, tik.scope_gm, "query_output_gm")
            self.key_output_gm = self.tik_inst.Tensor(self.data_type, output_shape, tik.scope_gm, "key_output_gm")
            self.value_output_gm = self.tik_inst.Tensor(self.data_type, output_shape, tik.scope_gm, "value_output_gm")
        self.outputs = [self.query_output_gm, self.key_output_gm, self.value_output_gm]
        self.weight_l1 = []

    def start_compute(self):
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_index:
            self.start_compute_each_core(core_index, self.each_core_batch_num)
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=self.inputs,
                               outputs=self.outputs,
                               config={"enable_const_fold": True}
                               )

    def start_compute_each_core(self, core_index, batch_num):
        if self.split_weight:
            ln_one_l1, gamma_fp32_ub, beta_fp32_ub = self.init_ln_tensor_split_weight()
            ln_res_l1, bias_fp32_ub = self.init_l1_tensor_split_weight()
            if self.m_unalign:
                result_lc, result_ub = self.init_result_tensor_m_unalign()
            else:
                result_lc, result_ub = self.init_result_tensor()
            weight_shape = (self.each_loop_n1_num, self.each_loop_k1_num, self.k_0, self.n_0)
            weight_l1_query = self.tensor(self.data_type, weight_shape, tik.scope_cbuf, "weight_l1_query")
            weight_l1_key = self.tensor(self.data_type, weight_shape, tik.scope_cbuf, "weight_l1_key")
            weight_l1_value = self.tensor(self.data_type, weight_shape, tik.scope_cbuf, "weight_l1_value")
            self.weight_l1 = [weight_l1_query, weight_l1_key, weight_l1_value]
            with self.tik_inst.for_range(0, batch_num) as core_batch_index:
                batch_index = core_batch_index
                self.start_compute_each_batch(core_index, batch_index, ln_one_l1, gamma_fp32_ub,
                                              beta_fp32_ub, ln_res_l1, bias_fp32_ub, result_lc, result_ub)
        else:
            ln_one_l1, gamma_fp32_l1, beta_fp32_l1 = self.init_ln_tensor()
            ln_res_l1, bias_fp32_l1 = self.init_l1_tensor()
            if self.m_unalign:
                result_lc, result_ub = self.init_result_tensor_m_unalign()
            else:
                result_lc, result_ub = self.init_result_tensor()
            with self.tik_inst.for_range(0, batch_num) as core_batch_index:
                batch_index = core_index * self.each_core_batch_num + core_batch_index
                self.start_compute_each_batch(core_index, batch_index, ln_one_l1, gamma_fp32_l1,
                                              beta_fp32_l1, ln_res_l1, bias_fp32_l1, result_lc, result_ub)

    def start_compute_each_batch(self, core_index, batch_index, ln_one_l1, gamma_fp32_l1, beta_fp32_l1, ln_res_l1,
                                 bias_fp32_l1, result_lc, result_ub):
        each_loop_m1_num = self.each_loop_m1_num
        m1_loop_times = self.m_1 // each_loop_m1_num
        for m1_loop_index in range(m1_loop_times):
            m1_index_start = m1_loop_index * each_loop_m1_num
            x_l1 = ln_res_l1
            self.data_move_in(x_l1, self.x_gm,
                              batch_index, m1_index_start, 0,
                              each_loop_m1_num, self.k_1, self.m_split)
            if self.split_weight:
                with self.tik_inst.if_scope(batch_index == 0):
                    result_nums = 3
                    n_num_result = self.n_1 // 3 // self.core_num
                    for core_result_index in range(0, result_nums):
                        n1_index = n_num_result * self.core_num * core_result_index + core_index * n_num_result
                        self._init_l1_tensor(self.weight_l1[core_result_index], self.weight_gm,
                                             n1_index, 0, self.each_loop_n1_num, self.each_loop_k1_num)
            self.start_compute_ln(x_l1, ln_one_l1, gamma_fp32_l1, beta_fp32_l1, ln_res_l1, result_lc,
                                  each_loop_m1_num)
            if self.split_weight:
                weight_l1 = self.weight_l1
                self.start_compute_split_n1(core_index, ln_res_l1, weight_l1, bias_fp32_l1,
                                            result_lc, result_ub,
                                            batch_index, m1_index_start, each_loop_m1_num)
            else:
                with self.tik_inst.new_stmt_scope():
                    weight_shape = (self.each_loop_n1_num, self.each_loop_k1_num, self.k_0, self.n_0)
                    weight_l1 = self.tensor(self.data_type, weight_shape, tik.scope_cbuf, "weight_l1")
                    self.start_compute_split_n1(core_index, ln_res_l1, weight_l1, bias_fp32_l1,
                                                result_lc, result_ub,
                                                batch_index, m1_index_start, each_loop_m1_num)

    def tensor(self, data_type, shape, scope, name):
        shape_flatten = get_element_num(shape)
        output_tensor = self.tik_inst.Tensor(data_type, (shape_flatten,), scope, name)
        return output_tensor

    def init_ln_tensor(self):
        ln_one_l1_shape = (self.m_0, self.k_0)
        ln_one_l1 = self.tensor(self.data_type, ln_one_l1_shape, tik.scope_cbuf, "ln_one_l1")
        gamma_shape = (self.k_1, self.k_0)
        gamma_fp32_l1 = self.tensor(self.fp32_type, gamma_shape, tik.scope_cbuf, "gamma_fp32_l1")
        beta_fp32_l1 = self.tensor(self.fp32_type, gamma_shape, tik.scope_cbuf, "beta_fp32_l1")
        with self.tik_inst.new_stmt_scope():
            # init fp32 gamma beta
            element_gamma = get_element_num(gamma_shape)
            gamma_ub = self.tensor(self.data_type, gamma_shape, tik.scope_ubuf, "gamma_ub")
            beta_ub = self.tensor(self.data_type, gamma_shape, tik.scope_ubuf, "beta_ub")
            gamma_fp32_ub = self.tensor(self.fp32_type, gamma_shape, tik.scope_ubuf, "gamma_fp32_ub")
            beta_fp32_ub = self.tensor(self.fp32_type, gamma_shape, tik.scope_ubuf, "beta_fp32_ub")
            gamma_fp16_block_num = element_gamma // self.num_fp16_per_block
            self.tik_inst.data_move(gamma_ub, self.gamma_gm, 0, 1, gamma_fp16_block_num, 0, 0)
            self.tik_func.start_tik_compute(element_gamma, self.fp32_type, self.tik_func.tik_vconv,
                                            args=(gamma_fp32_ub, gamma_ub))
            self.tik_inst.data_move(beta_ub, self.beta_gm, 0, 1, gamma_fp16_block_num, 0, 0)
            self.tik_func.start_tik_compute(element_gamma, self.fp32_type, self.tik_func.tik_vconv,
                                            args=(beta_fp32_ub, beta_ub))
            gamma_fp32_block_num = element_gamma // self.num_fp32_per_block
            self.tik_inst.data_move(gamma_fp32_l1, gamma_fp32_ub, 0, 1, gamma_fp32_block_num, 0, 0)
            self.tik_inst.data_move(beta_fp32_l1, beta_fp32_ub, 0, 1, gamma_fp32_block_num, 0, 0)
            # init ln_one_l1
            element_ln_one_l1 = get_element_num(ln_one_l1_shape)
            ln_one_ub = self.tensor(self.data_type, ln_one_l1_shape, tik.scope_ubuf, "ln_one_l1")
            self.tik_func.start_tik_compute(element_ln_one_l1, self.data_type, self.tik_func.tik_dup, (ln_one_ub, 1))
            self.tik_inst.data_move(ln_one_l1, ln_one_ub, 0, 1, element_ln_one_l1 // self.num_fp16_per_block, 0, 0)
        return ln_one_l1, gamma_fp32_l1, beta_fp32_l1
    
    def init_ln_tensor_split_weight(self):
        ln_one_l1_shape = (self.m_0, self.k_0)
        ln_one_l1 = self.tensor(self.data_type, ln_one_l1_shape, tik.scope_cbuf, "ln_one_l1")
        gamma_shape = (self.k_1, self.k_0)
        gamma_fp32_ub = self.tensor(self.fp32_type, gamma_shape, tik.scope_ubuf, "gamma_fp32_ub")
        beta_fp32_ub = self.tensor(self.fp32_type, gamma_shape, tik.scope_ubuf, "beta_fp32_ub")
        with self.tik_inst.new_stmt_scope():
            gamma_ub = self.tensor(self.data_type, gamma_shape, tik.scope_ubuf, "gamma_ub")
            beta_ub = self.tensor(self.data_type, gamma_shape, tik.scope_ubuf, "beta_ub")
            element_gamma = get_element_num(gamma_shape)
            gamma_fp16_block_num = element_gamma // self.num_fp16_per_block
            self.tik_inst.data_move(gamma_ub, self.gamma_gm, 0, 1, gamma_fp16_block_num, 0, 0)
            self.tik_func.start_tik_compute(element_gamma, self.fp32_type, self.tik_func.tik_vconv,
                                            args=(gamma_fp32_ub, gamma_ub))
            self.tik_inst.data_move(beta_ub, self.beta_gm, 0, 1, gamma_fp16_block_num, 0, 0)
            self.tik_func.start_tik_compute(element_gamma, self.fp32_type, self.tik_func.tik_vconv,
                                            args=(beta_fp32_ub, beta_ub))
            # init ln_one_l1
            element_ln_one_l1 = get_element_num(ln_one_l1_shape)
            ln_one_ub = self.tensor(self.data_type, ln_one_l1_shape, tik.scope_ubuf, "ln_one_l1")
            self.tik_func.start_tik_compute(element_ln_one_l1, self.data_type, self.tik_func.tik_dup, (ln_one_ub, 1))
            self.tik_inst.data_move(ln_one_l1, ln_one_ub, 0, 1, element_ln_one_l1 // self.num_fp16_per_block, 0, 0)
        return ln_one_l1, gamma_fp32_ub, beta_fp32_ub

    def init_l1_tensor(self, ):
        bias_shape = (self.n_1, self.n_0)
        bias_fp32_l1 = self.tensor(self.fp32_type, bias_shape, tik.scope_cbuf, "bias_fp32_l1")
        with self.tik_inst.new_stmt_scope():
            # init fp32 bias
            element_bias = get_element_num(bias_shape)
            bias_fp32_ub = self.tensor(self.fp32_type, bias_shape, tik.scope_ubuf, "bias_fp32_ub")
            bias_ub = self.tensor(self.data_type, bias_shape, tik.scope_ubuf, "bias_ub")
            bias_block_num_fp16 = element_bias // self.num_fp16_per_block
            self.tik_inst.data_move(bias_ub, self.bias_gm, 0, 1, bias_block_num_fp16, 0, 0)
            self.tik_func.start_tik_compute(element_bias, self.fp32_type, self.tik_func.tik_vconv,
                                            args=(bias_fp32_ub, bias_ub))
            bias_block_num_fp32 = element_bias // self.num_fp32_per_block
            self.tik_inst.data_move(bias_fp32_l1, bias_fp32_ub, 0, 1, bias_block_num_fp32, 0, 0)
        # init matmul tensor
        each_loop_m1_num = self.each_loop_m1_num
        ln_res_l1_shape = (self.k_1, each_loop_m1_num, self.m_0, self.k_0)
        ln_res_l1 = self.tensor(self.data_type, ln_res_l1_shape, tik.scope_cbuf, "ln_res_l1")
        return ln_res_l1, bias_fp32_l1

    def init_l1_tensor_split_weight(self):
        bias_shape = (self.n_1, self.n_0)
        bias_fp32_ub = self.tensor(self.fp32_type, bias_shape, tik.scope_ubuf, "bias_fp32_ub")
        with self.tik_inst.new_stmt_scope():
            # init fp32 bias
            element_bias = get_element_num(bias_shape)
            bias_ub = self.tensor(self.data_type, bias_shape, tik.scope_ubuf, "bias_ub")
            bias_block_num_fp16 = element_bias // self.num_fp16_per_block
            self.tik_inst.data_move(bias_ub, self.bias_gm, 0, 1, bias_block_num_fp16, 0, 0)
            self.tik_func.start_tik_compute(element_bias, self.fp32_type, self.tik_func.tik_vconv,
                                            args=(bias_fp32_ub, bias_ub))
        # init matmul tensor
        each_loop_m1_num = self.each_loop_m1_num
        ln_res_l1_shape = (self.k_1, each_loop_m1_num, self.m_0, self.k_0)
        ln_res_l1 = self.tensor(self.data_type, ln_res_l1_shape, tik.scope_cbuf, "ln_res_l1")
        return ln_res_l1, bias_fp32_ub

    def init_result_tensor(self):
        each_loop_m1_num = self.each_loop_m1_num
        each_loop_n1_num = self.each_loop_n1_num
        result_ub_shape = (each_loop_n1_num, each_loop_m1_num, self.m_0, self.n_0)
        result_ub = self.tensor(self.data_type, result_ub_shape, tik.scope_ubuf, "result_ub")
        result_lc = self.tik_inst.Tensor(self.fp32_type, (self.l0c_size // self.fp32_size,), tik.scope_cc, "result_lc")
        return result_lc, result_ub

    def init_result_tensor_m_unalign(self):
        m1_num_before_pad = self.m_2 * self.m_2 * self.m_3 * self.m_3
        each_loop_n1_num = self.each_loop_n1_num
        result_ub_shape = (each_loop_n1_num, m1_num_before_pad, self.n_0)
        result_ub = self.tensor(self.data_type, result_ub_shape, tik.scope_ubuf, "result_ub")
        result_lc = self.tik_inst.Tensor(self.fp32_type, (self.l0c_size // self.fp32_size,), tik.scope_cc, "result_lc")
        return result_lc, result_ub

    def data_move_in(self, x_tensor_in, x_tensor_out,
                     batch_index_out, m1_index_out, k1_index_out,
                     m1_num, k1_num, m_split):
        """
        data move in
        """
        if m_split == 0:
            start_index_in = 0
            nburst = k1_num
            burst = m1_num * self.m_0 * self.k_0 // self.num_fp16_per_block
            src_gap = (self.m_1 - m1_num) * self.m_0 * self.k_0 // self.num_fp16_per_block
            dst_gap = 0
            self.tik_inst.data_move(x_tensor_in[start_index_in:],
                                    x_tensor_out[batch_index_out, k1_index_out, m1_index_out, 0, 0],
                                    0, nburst, burst, src_gap, dst_gap)
        else:
            self.data_move_in_split(x_tensor_in, x_tensor_out,
                                    batch_index_out, m1_index_out, k1_index_out,
                                    m1_num, k1_num, m_split)

    def data_move_in_split(self, x_tensor_in, x_tensor_out, batch_index_out, m1_index_out, k1_index_out, m1_num, k1_num,
                           m_split):
        """
        split concat move in
        """
        m_index_start = m1_index_out * self.m_0
        m_num_move = m1_num * self.m_0
        start_shift_dim = self.m_num_sqrt - m_split
        if m_num_move <= self.m_num_sqrt:
            m_index_out_0 = m_index_start // self.m_num_sqrt
            if m_index_out_0 < start_shift_dim:
                m_index_out_0 += m_split
            else:
                m_index_out_0 = m_index_out_0 - start_shift_dim

            m_index_1_start = m_index_start % self.m_num_sqrt
            m_index_1_end = m_index_1_start + m_num_move
            if m_index_1_end <= start_shift_dim:
                self.data_move_in_behind_to_front(x_tensor_in, x_tensor_out,
                                                  batch_index_out, m_index_out_0, m_index_1_start, k1_index_out, 0,
                                                  m_num_move, m_num_move, k1_num, m_split)
            elif m_index_1_start >= start_shift_dim:
                self.data_move_in_front_to_behind(x_tensor_in, x_tensor_out,
                                                  batch_index_out, m_index_out_0, m_index_1_start, k1_index_out, 0,
                                                  m_num_move, m_num_move, k1_num, m_split)
            else:
                m_num_first_move = start_shift_dim - m_index_1_start
                self.data_move_in_behind_to_front(x_tensor_in, x_tensor_out,
                                                  batch_index_out, m_index_out_0, m_index_1_start, k1_index_out, 0,
                                                  m_num_move, m_num_first_move, k1_num, m_split)
                m_num_second_move = m_index_1_end - start_shift_dim
                self.data_move_in_front_to_behind(x_tensor_in, x_tensor_out,
                                                  batch_index_out, m_index_out_0, m_index_1_start + m_num_first_move,
                                                  k1_index_out, m_num_first_move,
                                                  m_num_move, m_num_second_move, k1_num, m_split)
        else:
            m_index_out_0_start = m_index_start // self.m_num_sqrt
            if self.m_unalign:
                m_index_out_0_end = (m_index_start + self.m_sqrt * self.m_sqrt) // self.m_num_sqrt
            else:
                m_index_out_0_end = (m_index_start + m_num_move) // self.m_num_sqrt
            m_index_ub = 0
            for m_index_out_0 in range(m_index_out_0_start, m_index_out_0_end):
                if m_index_out_0 < start_shift_dim:
                    m_index_out_0 += m_split
                else:
                    m_index_out_0 = m_index_out_0 - start_shift_dim

                self.data_move_in_split_second(x_tensor_in, x_tensor_out,
                                               batch_index_out, m_index_out_0, k1_index_out, m_index_ub,
                                               m_num_move, k1_num, m_split)
                m_index_ub += self.m_num_sqrt

    def data_move_in_split_second(self, x_tensor_in, x_tensor_out,
                                  batch_index_out, m_index_out_0, k1_index_out, m_index_in,
                                  m_num_ub, k1_num, m_split):
        """
        data move out split second m_sqrt
        """
        m_num_first_move = self.m_num_sqrt - m_split
        self.data_move_in_behind_to_front(x_tensor_in, x_tensor_out,
                                          batch_index_out, m_index_out_0, 0, k1_index_out,
                                          m_index_in,
                                          m_num_ub, m_num_first_move, k1_num, m_split)
        m_num_second_move = m_split
        self.data_move_in_front_to_behind(x_tensor_in, x_tensor_out,
                                          batch_index_out, m_index_out_0, m_num_first_move, k1_index_out,
                                          m_index_in + m_num_first_move,
                                          m_num_ub, m_num_second_move, k1_num, m_split)

    def data_move_in_behind_to_front(self, x_tensor_in, x_tensor_out,
                                     batch_index_out, m_index_out_0, m_index_out_1, k1_index_out, m_index_in,
                                     m_num_ub, m_num_move, k1_num, m_split):
        """
        data move in behind to front
        """
        shape_in = (k1_num, m_num_ub, self.k_0)
        start_index_in = (0, m_index_in)
        start_index_in_flatten = get_start_index(start_index_in, shape_in)
        if self.m_unalign:
            shape_out = (self.batch_num, self.k_1, self.m_1 * self.m_0, self.k_0)
            x_tensor_out = x_tensor_out.reshape(shape_out)
            nburst = k1_num
            burst = m_num_move * self.k_0 // self.num_fp16_per_block
            dst_gap = (m_num_ub - m_num_move) * self.k_0 // self.num_fp16_per_block
            src_gap = (m_num_ub - m_num_move) * self.k_0 // self.num_fp16_per_block
            self.tik_inst.data_move(x_tensor_in[start_index_in_flatten:],
                                    x_tensor_out[batch_index_out, k1_index_out, m_index_out_0 * self.m_num_sqrt + \
                                    m_split, 0], 0, nburst, burst, src_gap, dst_gap)
        else:
            shape_out = (self.batch_num, self.k_1, self.m_num_sqrt, self.m_num_sqrt, self.k_0)
            x_tensor_out = x_tensor_out.reshape(shape_out)
            nburst = k1_num
            burst = m_num_move * self.k_0 // self.num_fp16_per_block
            dst_gap = (m_num_ub - m_num_move) * self.k_0 // self.num_fp16_per_block
            src_gap = (self.m_num_sqrt * self.m_num_sqrt - m_num_move) * self.k_0 // self.num_fp16_per_block
            self.tik_inst.data_move(x_tensor_in[start_index_in_flatten:],
                                    x_tensor_out[batch_index_out, k1_index_out, m_index_out_0, \
                                    m_split + m_index_out_1, 0], 0, nburst, burst, src_gap, dst_gap)

    def data_move_in_front_to_behind(self, x_tensor_in, x_tensor_out,
                                     batch_index_out, m_index_out_0, m_index_out_1, k1_index_out, m_index_in,
                                     m_num_ub, m_num_move, k1_num, m_split):
        """
        data move in front to behind
        """
        shape_in = (k1_num, m_num_ub, self.k_0)
        start_index_in = (0, m_index_in)
        start_index_in_flatten = get_start_index(start_index_in, shape_in)
        if self.m_unalign:
            shape_out = (self.batch_num, self.k_1, m_num_ub, self.k_0)
            x_tensor_out = x_tensor_out.reshape(shape_out)

            nburst = k1_num
            burst = m_num_move * self.k_0 // self.num_fp16_per_block
            dst_gap = (m_num_ub - m_num_move) * self.k_0 // self.num_fp16_per_block
            src_gap = (m_num_ub - m_num_move) * self.k_0 // self.num_fp16_per_block
            start_shift_dim = self.m_num_sqrt - m_split
            self.tik_inst.data_move(x_tensor_in[start_index_in_flatten:],
                                    x_tensor_out[batch_index_out, k1_index_out, m_index_out_0 * self.m_num_sqrt + \
                                                 m_index_out_1 - start_shift_dim, 0], \
                                    0, nburst, burst, src_gap, dst_gap)
        else:
            shape_out = (self.batch_num, self.k_1, self.m_num_sqrt, self.m_num_sqrt, self.k_0)
            x_tensor_out = x_tensor_out.reshape(shape_out)
            nburst = k1_num
            burst = m_num_move * self.k_0 // self.num_fp16_per_block
            dst_gap = (m_num_ub - m_num_move) * self.k_0 // self.num_fp16_per_block
            src_gap = (self.m_num_sqrt * self.m_num_sqrt - m_num_move) * self.k_0 // self.num_fp16_per_block
            start_shift_dim = self.m_num_sqrt - m_split
            self.tik_inst.data_move(x_tensor_in[start_index_in_flatten:],
                                    x_tensor_out[batch_index_out, k1_index_out, m_index_out_0, \
                                                 m_index_out_1 - start_shift_dim, 0], \
                                    0, nburst, burst, src_gap, dst_gap)

    def start_compute_ln(self, x_l1, ln_one_l1, gamma_fp32_l1, beta_fp32_l1, ln_res_l1, result_lc, m1_num):
        """
        start compute layer norm
        """
        with self.tik_inst.new_stmt_scope():
            if self.split_weight:
                gamma_fp32_ub, beta_fp32_ub = gamma_fp32_l1, beta_fp32_l1
            else:
                gamma_fp32_ub, beta_fp32_ub = self.compute_ln_init_weight(gamma_fp32_l1, beta_fp32_l1)
            m1_num_ln_loop = self.each_ub_m1_num
            m1_num_loop = ceil_div(m1_num, m1_num_ln_loop)
            with self.tik_inst.for_range(0, m1_num_loop) as m1_index_loop:
                x_sum_ub, xx_sum_ub = self.compute_ln_mad_compute(result_lc, x_l1, ln_one_l1, m1_index_loop, m1_num,
                                                                  m1_num_ln_loop)
                x_ub = self.compute_ln_init_x(x_l1, m1_index_loop, m1_num, m1_num_ln_loop)
                self.compute_ln_norm(
                    xx_sum_ub, x_sum_ub, x_ub, gamma_fp32_ub,
                    beta_fp32_ub, m1_num_ln_loop)
                self.compute_ln_move_ln_result(ln_res_l1, x_ub, m1_index_loop, m1_num, m1_num_ln_loop)

    def compute_ln_init_weight(self, gamma_fp32_l1, beta_fp32_l1):
        """
        compute layer norm init weight
        """
        gamma_shape = (self.k_1, self.k_0)
        gamma_fp32_ub = self.tik_inst.Tensor(self.fp32_type, gamma_shape, tik.scope_ubuf, "gamma_fp32_ub")
        beta_fp32_ub = self.tik_inst.Tensor(self.fp32_type, gamma_shape, tik.scope_ubuf, "beta_fp32_ub")
        element_gamma = get_element_num(gamma_shape)
        gamma_fp32_block_num = element_gamma // self.num_fp32_per_block
        self.tik_inst.data_move(gamma_fp32_ub, gamma_fp32_l1, 0, 1, gamma_fp32_block_num, 0, 0)
        self.tik_inst.data_move(beta_fp32_ub, beta_fp32_l1, 0, 1, gamma_fp32_block_num, 0, 0)
        return gamma_fp32_ub, beta_fp32_ub

    def compute_ln_mad_compute(self, result_lc, x_l1, ln_one_l1, m1_index_l1, m1_num_l1, m1_num_ln):
        """
        compute layer norm sum
        """
        l1_shape = (self.k_1, m1_num_l1, self.m_0, self.k_0)
        start_index_l1 = (0, m1_index_l1)
        start_index_l1_flatten = get_start_index(start_index_l1, l1_shape)
        x_sum_shape = (1, m1_num_ln, self.m_0, self.k_0)
        x_sum_ub = self.tik_inst.Tensor(self.fp32_type, x_sum_shape, tik.scope_ubuf, "x_sum_ub")
        xx_sum_ub = self.tik_inst.Tensor(self.fp32_type, x_sum_shape, tik.scope_ubuf, "xx_sum_ub")
        with self.tik_inst.new_stmt_scope():
            l0a_shape = (m1_num_ln, self.k_1, self.m_0, self.k_0)
            l0b_shape = (self.k_1, m1_num_ln, self.m_0, self.k_0)
            mat_l0a = self.tik_inst.Tensor(self.data_type, l0a_shape, tik.scope_ca, "mat_l0a")
            mat_l0b = self.tik_inst.Tensor(self.data_type, l0b_shape, tik.scope_cb, "mat_l0b")
            # start_index, repeat, src_stride, sid, is_transpose
            with self.tik_inst.for_range(0, m1_num_ln) as m1_index:
                start_index_l1_loop = (0, m1_index)
                start_index_l1_flatten_loop = get_start_index(start_index_l1_loop, l1_shape) + start_index_l1_flatten
                self.tik_inst.load2dv2(mat_l0a[m1_index, 0, 0, 0], x_l1[start_index_l1_flatten_loop:],
                                       0, self.k_1, 0, m1_num_l1, 0, False)
            self.tik_inst.load2dv2(mat_l0b, ln_one_l1,
                                   0, self.k_1 * m1_num_ln, 0, 0, 0, False)
            # dst_fm, src_fm, src_filter, matrix_m, matrix_k, matrix_n, is_bias
            self.tik_inst.mmad(result_lc, mat_l0a, mat_l0b,
                               m1_num_ln * self.m_0,
                               self.k_1 * self.k_0,
                               m1_num_ln * self.m_0, 0)
            self.tik_inst.data_move(x_sum_ub, result_lc, 0, 1, m1_num_ln, 0, 0)
            with self.tik_inst.for_range(0, m1_num_ln) as m1_index:
                start_index_l1_loop = (0, m1_index)
                start_index_l1_flatten_loop = get_start_index(start_index_l1_loop, l1_shape) + start_index_l1_flatten
                self.tik_inst.load2dv2(mat_l0b[0, m1_index, 0, 0], x_l1[start_index_l1_flatten_loop:],
                                       0, self.k_1, 0, m1_num_l1, 0, False)
            self.tik_inst.mmad(result_lc, mat_l0a, mat_l0b,
                               m1_num_ln * self.m_0,
                               self.k_1 * self.k_0,
                               m1_num_ln * self.m_0, 0)
            self.tik_inst.data_move(xx_sum_ub, result_lc, 0, 1, m1_num_ln, 0, 0)
        return x_sum_ub, xx_sum_ub

    def compute_ln_init_x(self, x_l1, m1_index_l1, m1_num_l1, m1_num_ln):
        """
        compute layer norm init x
        """
        x_l1_shape = (self.k_1, m1_num_l1, self.m_0, self.k_0)
        start_index_l1 = (0, m1_index_l1)
        start_index_l1_flatten = get_start_index(start_index_l1, x_l1_shape)
        x_ub_shape = (self.k_1, m1_num_ln, self.m_0, self.k_0)
        x_ub = self.tik_inst.Tensor(self.data_type, x_ub_shape, tik.scope_ubuf, "x_ub")
        nburst = self.k_1
        burst = m1_num_ln * self.m_0 * self.k_0 // self.num_fp16_per_block
        src_gap = (m1_num_l1 - m1_num_ln) * self.m_0 * self.k_0 // self.num_fp16_per_block
        dst_gap = (m1_num_ln - m1_num_ln) * self.m_0 * self.k_0 // self.num_fp16_per_block
        self.tik_inst.data_move(x_ub, x_l1[start_index_l1_flatten:],
                                0, nburst, burst, src_gap, dst_gap)
        return x_ub

    def compute_ln_norm(self, xx_sum_ub, x_sum_ub, x_ub, gamma_fp32_ub,
                        beta_fp32_ub, m1_num_ln):
        # init mean
        # use diagonal element fill the row to remove invalid entry in fractal_matrix
        x_sum_shape = (1, m1_num_ln, self.m_0, self.k_0)
        squared_mean_ub = self.tik_inst.Tensor(self.fp32_type, x_sum_shape, tik.scope_ubuf, "squared_mean_ub")
        var_ub = self.tik_inst.Tensor(self.fp32_type, x_sum_shape, tik.scope_ubuf, "var_ub")
        x_fp32_ub = self.tik_inst.Tensor(self.fp32_type, x_ub.shape, tik.scope_ubuf, "var_ub")
        with self.tik_inst.for_range(0, self.m_0) as brc_idx:
            var_scalar = self.tik_inst.Scalar(self.fp32_type)
            var_scalar.set_as(xx_sum_ub[0, 0, brc_idx, brc_idx])
            # set vector mask as self.m_0 to avoid vector_dup erase next value
            self.tik_inst.vector_dup(self.m_0, xx_sum_ub[0, 0, brc_idx, 0], var_scalar, 1, 1, 0)
        element_result_lc = m1_num_ln * self.m_0 * self.k_0
        mean_repeat_num = element_result_lc // Constant.FP32_REPEAT_SIZE
        self.tik_inst.vmuls(Constant.MASK_FP32, x_sum_ub, x_sum_ub, self.mean_coeff,
                            mean_repeat_num, 1, 1, 8, 8)
        self.tik_inst.vmuls(Constant.MASK_FP32, xx_sum_ub, xx_sum_ub, self.mean_coeff,
                            mean_repeat_num, 1, 1, 8, 8)
        # init x fp32
        x_ub_shape = (self.k_1, m1_num_ln, self.m_0, self.k_0)
        element_x = get_element_num(x_ub_shape)
        self.tik_func.start_tik_compute(element_x, self.fp32_type, self.tik_func.tik_vconv,
                                        args=(x_fp32_ub, x_ub))

        mad_shape = (m1_num_ln, m1_num_ln, self.m_0, self.m_0)
        element_mad_ub = functools.reduce(lambda a, b: a * b, mad_shape)
        # mean^2
        mean_repeat_num = element_mad_ub // Constant.FP32_REPEAT_SIZE
        mask = Constant.MASK_FP32
        self.tik_inst.vmul(mask, squared_mean_ub, x_sum_ub, x_sum_ub, mean_repeat_num, 1, 1, 1, 8, 8, 8)
        # variance is x^2 - mean^2
        self.tik_inst.vsub(mask, var_ub, xx_sum_ub, squared_mean_ub, mean_repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vabs(mask, var_ub, var_ub, mean_repeat_num, 1, 1, 8, 8)
        # variance + epsilon
        self.tik_inst.vadds(mask, squared_mean_ub, var_ub, self.epsilon, mean_repeat_num, 1, 1, 8, 8)
        # rsqrt of variance + epsilon
        self.tik_inst.vln(mask, squared_mean_ub, squared_mean_ub, mean_repeat_num, 1, 1, 8, 8)
        self.tik_inst.vmuls(mask, squared_mean_ub, squared_mean_ub, -0.5, mean_repeat_num, 1, 1, 8, 8)
        self.tik_inst.vexp(mask, squared_mean_ub, squared_mean_ub, mean_repeat_num, 1, 1, 8, 8)
        # norm
        with self.tik_inst.for_range(0, self.m_0 * self.m_0 // Constant.FP32_REPEAT_SIZE) as sub_mul_idx:
            m_offset = Constant.FP32_REPEAT_SIZE // self.m_0 * sub_mul_idx
            # x - mean
            self.tik_inst.vsub(mask, x_fp32_ub[0, 0, m_offset, 0], x_fp32_ub[0, 0, m_offset, 0],
                               x_sum_ub[0, 0, m_offset, 0], self.k_1, 1, 1, 1,
                               Constant.FP32_REPEAT_STRIDE, Constant.FP32_REPEAT_STRIDE, 0)
            # norm is x - mean divides sqrt of variance + epsilon
            self.tik_inst.vmul(mask, x_fp32_ub[0, 0, m_offset, 0], x_fp32_ub[0, 0, m_offset, 0],
                               squared_mean_ub[0, 0, m_offset, 0], self.k_1, 1, 1, 1,
                               Constant.FP32_REPEAT_STRIDE, Constant.FP32_REPEAT_STRIDE, 0)
        # weight bias
        with self.tik_inst.for_range(0, self.m_0 // Constant.NUM_FP32_PER_BLOCK) as block_index:  # 2
            with self.tik_inst.for_range(0, self.m_0 // Constant.BLOCK_PER_REPEAT) as repeat_index:  # 2
                # gamma muls norm
                self.tik_inst.vmul(mask, x_fp32_ub[0, 0, 8 * repeat_index, 8 * block_index],
                                   x_fp32_ub[0, 0, 8 * repeat_index, 8 * block_index],
                                   gamma_fp32_ub[8 * block_index], self.k_1,
                                   Constant.FP32_BLOCK_STRIDE, Constant.FP32_BLOCK_STRIDE, 0,
                                   Constant.FP32_REPEAT_STRIDE, Constant.FP32_REPEAT_STRIDE,
                                   Constant.FP32_BLOCK_STRIDE)

                self.tik_inst.vadd(mask, x_fp32_ub[0, 0, 8 * repeat_index, 8 * block_index],
                                   x_fp32_ub[0, 0, 8 * repeat_index, 8 * block_index],
                                   beta_fp32_ub[8 * block_index], self.k_1,
                                   Constant.FP32_BLOCK_STRIDE, Constant.FP32_BLOCK_STRIDE, 0,
                                   Constant.FP32_REPEAT_STRIDE, Constant.FP32_REPEAT_STRIDE,
                                   Constant.FP32_BLOCK_STRIDE)
        # vconv
        self.tik_func.start_tik_compute(self.k_1 * m1_num_ln * self.m_0 * self.m_0, self.fp32_type,
                                        self.tik_func.tik_vconv,
                                        args=(x_ub, x_fp32_ub))

    def compute_ln_move_ln_result(self, ln_res_l1, ln_res_ub, m1_index_l1, m1_num_l1, m1_num_ln):
        l1_shape = (self.k_1, m1_num_l1, self.m_0, self.k_0)
        start_index_l1 = (0, m1_index_l1)
        start_index_l1_flatten = get_start_index(start_index_l1, l1_shape)
        nburst = self.k_1
        burst = m1_num_ln * self.m_0 * self.k_0 // self.num_fp16_per_block
        dst_gap = (m1_num_l1 - m1_num_ln) * self.m_0 * self.k_0 // self.num_fp16_per_block
        src_gap = (m1_num_ln - m1_num_ln) * self.m_0 * self.k_0 // self.num_fp16_per_block
        self.tik_inst.data_move(ln_res_l1[start_index_l1_flatten:], ln_res_ub,
                                0, nburst, burst, src_gap, dst_gap)

    def start_compute_split_n1(self, core_index, ln_res_l1, weight_l1, bias_fp32_l1,
                               result_lc, result_ub,
                               batch_index, m1_index_start, m1_num):
        result_num = 3
        n_num_result = self.n_1 // result_num
        if self.split_weight:
            n_num_result = self.n_1 // result_num // self.core_num
        each_loop_n1_num = self.each_loop_n1_num
        n1_loop_times, last_loop_n1_num = get_loop_info(n_num_result, each_loop_n1_num)
        for result_index in range(result_num):
            result_gm = self.outputs[result_index]
            if last_loop_n1_num == each_loop_n1_num:
                with self.tik_inst.for_range(0, n1_loop_times) as n1_loop_index:
                    n1_index_loop = n1_loop_index * each_loop_n1_num
                    if self.split_weight:
                        weight = weight_l1[result_index]
                        n_num_result = self.n_1 // 3 // self.core_num
                        n1_index = n_num_result * self.core_num * result_index + core_index * n_num_result
                        with self.tik_inst.for_range(0, m1_num) as m1_index:
                            lc_start_index = m1_index * self.m_0 * self.n_0
                            self.tik_inst.broadcast_ub_to_l0c(result_lc[lc_start_index:],
                                                              bias_fp32_l1,
                                                              each_loop_n1_num, 1, 0, m1_num - 1)
                    else:
                        n1_index = n1_index_loop + n_num_result * result_index
                        weight = weight_l1
                        bias_fp32_ub = self.tensor(self.fp32_type, (each_loop_n1_num, self.n_0), tik.scope_ubuf,
                                                   "bias_fp32_ub")
                        self._init_lc_data(result_lc, bias_fp32_l1, bias_fp32_ub, m1_num, each_loop_n1_num, n1_index)                          
                    if self.m_unalign:
                        each_loop_m1_num = self.each_loop_m1_num
                        each_loop_n1_num = self.each_loop_n1_num
                        result_ub_temp_shape = (each_loop_n1_num, each_loop_m1_num, self.m_0, self.n_0)
                        result_ub_temp = self.tensor(self.data_type, result_ub_temp_shape, \
                                                     tik.scope_ubuf, "result_ub_temp")
                    else:
                        result_ub_temp = self.tensor(self.data_type, result_ub.shape, tik.scope_ubuf, "result_ub_temp")
                    self.compute_matmul_split_k1(result_lc, ln_res_l1, weight,
                                                 n1_index, m1_num, each_loop_n1_num)
                    self.tik_inst.tensor_mov(result_ub_temp, result_lc, 'm', 1, each_loop_n1_num * m1_num, 0, 0)
                    self.compute_matmul_move_out(core_index, result_gm, result_ub, result_ub_temp,
                                                 batch_index, m1_index_start, n1_index_loop, m1_num,
                                                 each_loop_n1_num)

    def compute_matmul_split_k1(self, result_lc, tensor_a_l1, tensor_b_l1,
                                n1_index_start, m1_num, n1_num):
        each_loop_k1_num = self.each_loop_k1_num
        k1_loop_times, last_loop_k1_num = get_loop_info(self.k_1, each_loop_k1_num)
        if last_loop_k1_num == each_loop_k1_num:
            with self.tik_inst.for_range(0, k1_loop_times) as k1_loop_index:
                k1_index_start = k1_loop_index * each_loop_k1_num
                if not self.split_weight:
                    self._init_l1_tensor(tensor_b_l1, self.weight_gm,
                                        n1_index_start, k1_index_start,
                                        n1_num, each_loop_k1_num)
                self.compute_matmul_start(result_lc, tensor_a_l1, tensor_b_l1,
                                          m1_num, self.k_1, n1_num, each_loop_k1_num,
                                          k1_index_start, 0,
                                          each_loop_k1_num, self.each_mul_k_num)

    def compute_matmul_start(self, result_lc, tensor_a_l1, tensor_b_l1,
                             m1_num, k1_num_a, n1_num, k1_num_b,
                             k1_index_start_a_l1, k1_index_start_b_l1,
                             k1_num, each_loop_k1_num,
                             thread_num=2):
        k1_loop_times, last_loop_k1_num = get_loop_info(k1_num, each_loop_k1_num)
        ping_pong_loop_times = ceil_div(k1_loop_times, thread_num) - 1
        ping_pong_last_times = k1_loop_times - ping_pong_loop_times * thread_num
        with self.tik_inst.new_stmt_scope():
            x_l0a_all, weight_l0b_all = self.compute_matmul_init_tensor(m1_num, each_loop_k1_num, n1_num, thread_num)
            if ping_pong_loop_times > 0:
                with self.tik_inst.for_range(0, ping_pong_loop_times) as ping_pong_index:
                    for thread_index in range(thread_num):
                        loop_index = ping_pong_index * thread_num + thread_index
                        k1_index = loop_index * each_loop_k1_num
                        self.compute_matmul_loop(result_lc, x_l0a_all[thread_index], weight_l0b_all[thread_index],
                                                 tensor_a_l1, tensor_b_l1,
                                                 m1_num, k1_num_a, n1_num, k1_num_b,
                                                 k1_index_start_a_l1 + k1_index, k1_index_start_b_l1 + k1_index,
                                                 each_loop_k1_num)

            for thread_index in range(0, ping_pong_last_times):
                loop_index = ping_pong_loop_times * thread_num + thread_index
                k1_index = loop_index * each_loop_k1_num
                loop_k1_num = each_loop_k1_num if loop_index < k1_loop_times - 1 else last_loop_k1_num
                self.compute_matmul_loop(result_lc, x_l0a_all[thread_index], weight_l0b_all[thread_index],
                                         tensor_a_l1, tensor_b_l1,
                                         m1_num, k1_num_a, n1_num, k1_num_b,
                                         k1_index_start_a_l1 + k1_index, k1_index_start_b_l1 + k1_index,
                                         loop_k1_num)

    def compute_matmul_init_tensor(self, m1_num, k1_num, n1_num, thread_num):
        data_type = "float16"
        x_l0a_shape = (m1_num, k1_num, self.m_0, self.k_0)
        x_l0a_all = [None] * thread_num
        for thread_index in range(thread_num):
            tensor_name = "x_l0a_{}".format(thread_index)
            x_l0a_all[thread_index] = self.tensor(data_type, x_l0a_shape, tik.scope_ca, tensor_name)
        weight_l0b_shape = (k1_num, n1_num, self.n_0, self.k_0)
        weight_l0b_all = [None] * thread_num
        for thread_index in range(thread_num):
            tensor_name = "weight_l0b_{}".format(thread_index)
            weight_l0b_all[thread_index] = self.tensor(data_type, weight_l0b_shape, tik.scope_cb, tensor_name)
        return x_l0a_all, weight_l0b_all

    def compute_matmul_loop(self, result_lc, tensor_l0a, tensor_l0b, tensor_a_l1, tensor_b_l1,
                            m1_num, k1_num_a, n1_num, k1_num_b,
                            k1_index_start_a_l1, k1_index_start_b_l1, loop_k1_num):
        tensor_l0a_shape = (m1_num, loop_k1_num, self.m_0, self.k_0)
        tensor_a_l1_shape = (k1_num_a, m1_num, self.m_0, self.k_0)
        with self.tik_inst.for_range(0, loop_k1_num) as k1_index:
            # x_l1[k1_index + k1_index_start_a_l1, 0, 0, 0] -> tensor_l0a[0, k1_index, 0, 0]
            dst_index_a = (0, k1_index)
            dst_index_a_flatten = get_start_index(dst_index_a, tensor_l0a_shape)
            src_index_a = (k1_index + k1_index_start_a_l1, 0)
            src_index_a_flatten = get_start_index(src_index_a, tensor_a_l1_shape)
            repeat_times = m1_num
            dst_gap = loop_k1_num - 1
            src_stride = 1
            if_transpose = False
            self.tik_inst.load2dv2(tensor_l0a[dst_index_a_flatten:], tensor_a_l1[src_index_a_flatten:],
                                   0, repeat_times, dst_gap, src_stride, 0, if_transpose)

        tensor_l0b_shape = (loop_k1_num, n1_num, self.n_0, self.k_0)
        tensor_b_l1_shape = (n1_num, k1_num_b, self.k_0, self.n_0)
        with self.tik_inst.for_range(0, loop_k1_num) as k1_index:
            # weight_l1[0, k1_index + k1_index_start_b_l1, 0, 0] -> tensor_l0b[k_index, 0, 0, 0]
            dst_index_b = (k1_index, 0)
            dst_index_b_flatten = get_start_index(dst_index_b, tensor_l0b_shape)
            src_index_b = (0, k1_index + k1_index_start_b_l1)
            src_index_b_flatten = get_start_index(src_index_b, tensor_b_l1_shape)
            repeat_times = n1_num
            dst_gap = 0
            src_stride = k1_num_b
            if_transpose = True
            self.tik_inst.load2dv2(tensor_l0b[dst_index_b_flatten:], tensor_b_l1[src_index_b_flatten:],
                                   0, repeat_times, dst_gap, src_stride, 0, if_transpose)

        self.tik_inst.mmad(result_lc, tensor_l0a, tensor_l0b,
                           m1_num * self.m_0, loop_k1_num * self.k_0, n1_num * self.n_0, 1)

    def compute_matmul_move_out(self, core_index, result_gm, result_ub, result_ub_temp,
                                batch_index, m1_index_start, n1_index_start,
                                m1_num, n1_num):
        m_num = m1_num * self.m_0
        ub_temp_shape = (n1_num, m_num, self.n_0)
        m2_num_0, m2_num_1, m3_num_0, m3_num_1 = 1, 1, 1, 1
        if m_num <= self.m_3:
            m3_num_1 = m_num
        elif m_num <= self.m_2 * self.m_3:
            m3_num_1 = self.m_3
            m2_num_1 = m_num // self.m_3
        elif m_num <= self.m_3 * self.m_2 * self.m_3:
            m3_num_1 = self.m_3
            m2_num_1 = self.m_2
            m3_num_0 = m_num // (self.m_2 * self.m_3)
        else:
            m3_num_1 = self.m_3
            m2_num_1 = self.m_2
            m3_num_0 = self.m_3
            m2_num_0 = m_num // (self.m_3 * self.m_2 * self.m_3)

        ub_shape = (m2_num_0, m2_num_1, n1_num, m3_num_0, m3_num_1, self.n_0)
        if self.m_unalign:
            m_loop_times = self.m_2 * self.m_2 * self.m_3
        else:
            m_loop_times = m_num // self.m_3

        # m_2_1: index on the previous m_2 dimension of the output shape
        # m_2_0: index on the last m_2 dimension of the output shape
        # m_3_1: index on the previous m_3 dimension of the output shape
        # m_3_0: index on the last m_3 dimension of the output shape
        for m_loop_index_times in range(m_loop_times):
            m_index_loop = m_loop_index_times * self.m_3
            m_2_1 = m_index_loop // self.m_3 % self.m_2
            m_2_0 = m_index_loop // (self.m_3 * self.m_2 * self.m_3)
            m_3_1 = m_index_loop % self.m_3
            m_3_0 = m_index_loop // (self.m_3 * self.m_2) % self.m_3

            ub_temp_start_index = get_start_index((0, m_index_loop), ub_temp_shape)
            ub_start_index = get_start_index((m_2_0, m_2_1, 0, m_3_0, m_3_1), ub_shape)
            nburst = n1_num
            burst = self.m_3 * self.k_0 // self.num_fp16_per_block
            src_gap = (m_num - self.m_3) * self.k_0 // self.num_fp16_per_block
            dst_gap = (m3_num_0 * m3_num_1 - self.m_3) * self.k_0 // self.num_fp16_per_block
            self.tik_inst.data_move(result_ub[ub_start_index:], result_ub_temp[ub_temp_start_index:],
                                    0, nburst, burst, src_gap, dst_gap)

        m_index_start = m1_index_start * self.m_0
        m_2_1 = m_index_start // self.m_3 % self.m_2
        m_2_0 = m_index_start // (self.m_3 * self.m_2 * self.m_3)
        m_3_1 = m_index_start % self.m_3
        m_3_0 = m_index_start // (self.m_3 * self.m_2) % self.m_3
        burst = m3_num_0 * m3_num_1 * self.n_0 // self.num_fp16_per_block
        src_gap = 0
        if self.unalign:
            dst_gap = (self.result_shape[3] * self.result_shape[4] - self.m_3 * self.m_3) * \
                       self.n_0 // self.num_fp16_per_block
        else:
            dst_gap = (self.m_3 * self.m_3 - m3_num_0 * m3_num_1) * self.k_0 // self.num_fp16_per_block
        if n1_num == self.n_1 // 3:
            nburst = m2_num_0 * m2_num_1 * n1_num
            self.tik_inst.data_move(result_gm[batch_index, m_2_0, m_2_1, n1_index_start, m_3_0, m_3_1, 0], result_ub,
                                    0, nburst, burst, src_gap, dst_gap)
        else:
            with self.tik_inst.for_range(0, m2_num_0) as m2_num_0_index:
                with self.tik_inst.for_range(0, m2_num_1) as m2_num_1_index:
                    nburst = n1_num
                    if self.split_weight:
                        n1_index_start = n1_index_start + core_index * self.each_loop_n1_num
                    self.tik_inst.data_move(result_gm[
                                                batch_index, m_2_0 + m2_num_0_index, m_2_1 + m2_num_1_index,
                                                n1_index_start, m_3_0, m_3_1, 0],
                                            result_ub[
                                                get_start_index((m2_num_0_index, m2_num_1_index, 0, 0, 0), ub_shape)],
                                            0, nburst, burst, src_gap, dst_gap)

    def _init_lc_data(self, tensor_lc, bias_fp32_l1, bias_fp32_ub, m1_num, n1_num, n1_start_index):
        with self.tik_inst.new_stmt_scope():
            bias_shape = (n1_num, self.n_0)
            bias_element_num = get_element_num(bias_shape)
            bias_l1_shape = (self.n_1, self.n_0)
            bias_l1_index = get_start_index((n1_start_index, 0), bias_l1_shape)
            bias_block_num = bias_element_num // self.num_fp32_per_block
            self.tik_inst.data_move(bias_fp32_ub, bias_fp32_l1[bias_l1_index], 0, 1, bias_block_num, 0, 0)
            with self.tik_inst.for_range(0, m1_num) as m1_index:
                lc_start_index = m1_index * self.m_0 * self.n_0
                self.tik_inst.broadcast_ub_to_l0c(tensor_lc[lc_start_index:],
                                                  bias_fp32_ub,
                                                  n1_num, 1, 0, m1_num - 1)

    def _init_l1_tensor(self, tensor_l1, tensor_gm, n1_index_start, k1_index_start, n1_num, k1_num):
        weight_block_num = k1_num * self.k_0 * self.n_0 // self.num_fp16_per_block
        src_gap = (self.k_1 - k1_num) * self.k_0 * self.n_0 // self.num_fp16_per_block
        self.tik_inst.data_move(tensor_l1,
                                tensor_gm[n1_index_start, k1_index_start, 0, 0], 0,
                                n1_num, weight_block_num,
                                src_gap, 0)


# 'pylint: disable = unused-argument
# 'pylint: disable=too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def swin_transformer_ln_qkv_v2(x, gamma, beta, weight, bias,
                               query_output, key_output, value_output,
                               head_num, head_dim, seq_length, shifts=(), epsilon=1e-7,
                               kernel_name="swin_transformer_ln_qkv"):
    """
    Parameters
    ----------
    x: dict
        shape and dtype of input x, only support float16
    gamma: dict
        shape and dtype of input kernel_query, only support float16
    beta: dict
        shape and dtype of input kernel_key, only support float16
    weight: dict
        shape and dtype of input kernel_value, only support float16
    bias: dict
        shape and dtype of input gamma, only support float16
    query_output: dict
        shape and dtype of output, only support float16
    key_output: dict
        shape and dtype of output, only support float16
    value_output: dict
        shape and dtype of output, only support float16
    head_num: int
        Head num of swin transformer Attention
    head_dim: int
        Head dim of transformer Attention
    seq_length: int
        seq length of transformer Attention
    shifts: list_int
        roll dim of transformer Attention
    epsilon: float
        Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "attention_ln_qkv"

    Returns
    -------
    None
    """
    m_0, k_0, n_0 = 16, 16, 16
    batch_num, k_1, m_1 = x.get("shape")[:3]
    core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    n_1 = weight.get("shape")[0]
    result_shape = query_output.get("shape")
    m_2 = int(math.sqrt(result_shape[0] // batch_num))
    n_2 = int(head_num)
    n_3 = int(head_dim // n_0)
    m_3 = int(math.sqrt(seq_length))
    dtype = x.get("dtype")
    if m_1 * 16 % seq_length != 0:
        m_unalign = True
    else:
        m_unalign = False
    if shifts:
        m_split = shifts[1]
    else:
        m_split = 0
    if batch_num >= 8 and (m_1 * batch_num < k_1 or (batch_num % core_num != 0 and m_unalign)):
        split_weight = True
    else:
        split_weight = False
    if m_3 != math.sqrt(result_shape[3] * m_0):
        unalign = True
    else:
        unalign = False
    obj = SwinTransformLnMatmul(batch_num, m_0, m_1, m_2, m_3, k_0, k_1, n_0, n_1, n_2, n_3, result_shape,
                                dtype, m_split, split_weight, unalign, m_unalign, epsilon, kernel_name)
    obj.start_compute()
