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
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import PlatformApi
from impl.swin_transformer_ln_qkv_v2 import check_supported as check_supported_v2
from impl.swin_transformer_ln_qkv_v2 import check_input_shape
from impl.swin_transformer_ln_qkv_v2 import swin_transformer_ln_qkv_v2
from impl.vit_transformer_ln_qkv import check_supported as check_supported_vit
from impl.vit_transformer_ln_qkv import check_input_shape
from impl.vit_transformer_ln_qkv import vit_transformer_ln_qkv


# 'pylint: disable=unused-argument,unused-variable,too-many-arguments,too-many-locals
def check_supported(x, gamma, beta, weight, bias,
                    query_output, key_output, value_output,
                    head_num, head_dim, seq_length, shifts=(), epsilon=1e-7,
                    kernel_name="swin_transformer_ln_qkv"):
    """
    check_supported
    """
    soc_version = PlatformApi.get_soc_spec(PlatformApi.SHORT_SOC_VERSION)
    support_version = ("Ascend310P",)
    if soc_version not in support_version:
        return False, "not support short soc version"
    support_dtype = ("float16", "float32")
    input_params_all = (x, gamma, beta, weight, bias, query_output, key_output, value_output)
    for input_params in input_params_all:
        if input_params.get("dtype").lower() not in support_dtype:
            return False, "not support data dtype"
    if len(shifts) == 4:
        roll_num = shifts[1]
        if tuple(shifts) != (0, roll_num, roll_num, 0):
            return False, "attr shifts not support"
    elif shifts:
        return False, "attr shifts not support"
    if epsilon < 0:
        return False, "not support eps"
    x_shape = tuple(x.get("ori_shape"))
    gamma_shape = tuple(gamma.get("ori_shape"))
    beta_shape = tuple(beta.get("ori_shape"))
    weight_shape = tuple(weight.get("ori_shape"))
    bias_shape = tuple(bias.get("ori_shape"))
    query_output_shape = tuple(query_output.get("ori_shape"))
    key_output_shape = tuple(key_output.get("ori_shape"))
    value_output_shape = tuple(value_output.get("ori_shape"))
    not_support_shape = (
        (1, 3136, 96),
        (1, 784, 192),
        (1, 196, 384)
    )
    if len(x_shape) != 3 or len(weight_shape) != 2 or (x_shape in not_support_shape):
        return False, "input shape not support"
    batch_num, m_num, k_num = x_shape
    support_attr_all = (
        (144, 32),
        (256, 32),
        (64, 32),
        (49, 32)
    )
    support_shape_all = ((9216, 128), (2304, 256), (576, 512), \
                         (9216, 192), (2304, 384), (576, 768), \
                         (9216, 96), (2304, 192), (576, 384), \
                         (1024, 96), (256, 192), (64, 384), (16, 768), \
                         (4096, 192), (1024, 384), (256, 768), \
                         (4096, 96), (1024, 192), (256, 384), \
                         (4096, 128), (1024, 256), (256, 512), \
                         (3136, 96), (784, 192), (196, 384), \
                         (3136, 128), (784, 256), \
                         (3136, 192), (784, 384), (65536, 192), (65536, 128),
                         )
    if (seq_length, head_dim) in support_attr_all and (m_num, k_num) in support_shape_all:
        if check_input_shape(batch_num, m_num, k_num, head_num, head_dim, seq_length,
                             x_shape, gamma_shape, beta_shape, weight_shape, bias_shape,
                             query_output_shape, key_output_shape, value_output_shape):
            return True, ""
    if check_supported_v2(x, gamma, beta, weight, bias,
                          query_output, key_output, value_output,
                          head_num, head_dim, seq_length, shifts, epsilon,
                          kernel_name):
        return True, ""
    if check_supported_vit(x, gamma, beta, weight, bias,
                           query_output, key_output, value_output,
                           head_num, head_dim, seq_length, shifts, epsilon,
                           kernel_name):
        return True, ""
    return False, "input shape not support"


def ceil_div(dividend, divisor):
    return (dividend + divisor - 1) // divisor


def get_align_num(data_num, align_num):
    return ceil_div(data_num, align_num) * align_num


def get_start_index(ori_index, ori_shape):
    start_index = 0
    ori_shape = tuple(ori_shape) + (1,)
    for dim_num, dim_index in enumerate(ori_index):
        start_index += dim_index * functools.reduce(lambda a, b: a * b, ori_shape[dim_num + 1:])
    return start_index


def get_loop_info(all_data_num, each_loop_num):
    """
    get loop info
    """
    loop_times = ceil_div(all_data_num, each_loop_num)
    last_loop_num = all_data_num - each_loop_num * (loop_times - 1)
    return loop_times, last_loop_num


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

    def __init__(self, batch_num, m_0, m_1, m_2, m_3, k_0, k_1, n_0, n_1, n_2, n_3, m_3_pad, dtype, m_split, unalign):
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
        self.m_3_pad = m_3_pad
        self.data_type = dtype
        self.fp32_type = "float32"

        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = PlatformApi.get_soc_spec(PlatformApi.UB_SIZE) - 8 * 1024
        self.l1_size = PlatformApi.get_soc_spec(PlatformApi.L1_SIZE)
        self.l0a_size = PlatformApi.get_soc_spec(PlatformApi.L0A_SIZE)
        self.l0b_size = PlatformApi.get_soc_spec(PlatformApi.L0B_SIZE)
        self.l0c_size = PlatformApi.get_soc_spec(PlatformApi.L0C_SIZE)

        # m_1_num_ln_loop const must be 1
        if unalign:
            self.m_1_num_ln_loop = 1
            self.m_1_num_unit = self.m_3 * self.m_3
            self.batch_m_1_num = self.batch_num * self.m_1
            self.m_1_unit_all = self.batch_m_1_num // self.m_1_num_unit
            self.each_core_m1_unit = ceil_div(self.m_1_unit_all, self.core_num)
        else:
            self.m_1_num_ln_loop = 1
            self.m_1_num_unit = self.m_0 * self.m_3 // math.gcd(self.m_0, self.m_3) // self.m_0
            self.batch_m_1_num = self.batch_num * self.m_1
            self.m_1_unit_all = self.batch_m_1_num // self.m_1_num_unit
            self.each_core_m1_unit = ceil_div(self.m_1_unit_all, self.core_num)

        if self.each_core_m1_unit > self.m_1 // self.m_1_num_unit:
            self.each_core_m1_unit = get_align_num(self.each_core_m1_unit, self.m_1 // self.m_1_num_unit)
        else:
            while (self.m_1 // self.m_1_num_unit) % self.each_core_m1_unit != 0:
                self.each_core_m1_unit += 1

        self.core_num, self.last_core_m1_unit = get_loop_info(self.m_1_unit_all, self.each_core_m1_unit)
        self.batch_m1_num_each_core = self.each_core_m1_unit * self.m_1_num_unit
        self.batch_m1_num_last_core = self.last_core_m1_unit * self.m_1_num_unit

        self.m_1_num_loop_l1, self.n_1_num_loop_l1, self.k_1_num_mul = \
            self.get_split_dim(self.m_1_num_unit)
        batch_m_num_each_core = self.batch_m1_num_each_core * self.m_0
        for m_1_num_loop_l1 in range(self.m_1_num_unit,
                                     (self.each_core_m1_unit + 1) * self.m_1_num_unit,
                                     self.m_1_num_unit):
            m_num = m_1_num_loop_l1 * self.m_0
            if self.judge_div(m_num) and batch_m_num_each_core % m_num == 0:
                m_1_num_loop_l1, n_1_num_loop_l1, k_1_num_mul = self.get_split_dim(m_1_num_loop_l1)
                if m_1_num_loop_l1 > n_1_num_loop_l1:
                    break
                else:
                    self.m_1_num_loop_l1, self.n_1_num_loop_l1, self.k_1_num_mul = \
                        m_1_num_loop_l1, n_1_num_loop_l1, k_1_num_mul
        if self.batch_num == 8 and self.m_1 == 36:
            self.m_1_num_loop_l1, self.n_1_num_loop_l1, self.k_1_num_mul = self.get_split_dim(18)
        self.k_1_num_loop_l1 = self.k_1_num_mul

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

    def get_split_dim(self, m_1_num_loop_l1):
        fp32_size = 4
        fp16_size = 2
        thread_num = 2
        ub_dims = (self.ub_size - self.k_0 * self.k_1 * fp32_size * 2) // self.m_0 // self.n_0 // fp16_size // 2
        l0a_dims = self.l0a_size // thread_num // fp16_size // self.m_0 // self.k_0
        l0b_dims = self.l0b_size // thread_num // fp16_size // self.n_0 // self.k_0
        l0c_dims = self.l0c_size // fp32_size // self.m_0 // self.n_0
        n_1_num_loop_l1_max = min(ub_dims // m_1_num_loop_l1, l0c_dims // m_1_num_loop_l1)
        n_1_num_loop_l1 = n_1_num_loop_l1_max
        for n_1_num_loop_l1 in range(n_1_num_loop_l1_max, 0, -1):
            if self.n_1 % n_1_num_loop_l1 == 0 and n_1_num_loop_l1 <= l0b_dims:
                break
        k_1_num_mul = min(l0b_dims // n_1_num_loop_l1, l0a_dims // m_1_num_loop_l1)
        return m_1_num_loop_l1, n_1_num_loop_l1, k_1_num_mul


class SwinTransformLnMatmul(TilingArgs):

    def __init__(self, batch_num, m_0, m_1, m_2, m_3, k_0, k_1, n_0, n_1, n_2, n_3, m_3_pad, dtype,
                 m_split, unalign, epsilon, kernel_name):
        # init tiling param
        super().__init__(batch_num, m_0, m_1, m_2, m_3, k_0, k_1, n_0, n_1, n_2, n_3, m_3_pad, dtype, m_split, unalign)
        # init attr
        self.epsilon = epsilon
        self.m_split = m_split
        self.unalign = unalign
        self.kernel_name = kernel_name
        self.mean_coeff = 1.0 / (self.k_1 * self.k_0)
        self.m_num = self.m_1 * self.m_0
        self.m_sqrt = int(math.sqrt(self.m_num))
        self.tik_inst = tik.Tik()
        # init tik fun
        self.tik_func = TikComputeInterface(self.tik_inst)
        # gm shape
        x_shape = (self.batch_num, self.k_1, self.m_1, self.m_0, self.k_0)
        gamma_shape = (self.k_1 * self.k_0,)
        weight_shape = (self.n_1, self.k_1, self.k_0, self.n_0)
        bias_shape = (self.n_1 * self.n_0,)
        if self.unalign:
            output_shape = (self.batch_num, self.m_2, self.m_2, self.n_2 * self.n_3, \
                            self.m_3_pad, self.m_3_pad, self.n_0)
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
        self.num_fp16_per_block = 16

    def start_compute(self):
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as block_index:
            with self.tik_inst.if_scope(block_index < self.core_num - 1):
                self.start_compute_each_core(block_index, self.each_core_m1_unit * self.m_1_num_unit)
            with self.tik_inst.else_scope():
                self.start_compute_each_core(block_index, self.last_core_m1_unit * self.m_1_num_unit)
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=self.inputs,
                               outputs=self.outputs
                               )

    def start_compute_each_core(self, block_index, m_1_num):
        ln_one_l1, gamma_fp32_ub, beta_fp32_ub = self.init_ln_tensor()
        m_1_num_loop_l1 = self.m_1_num_loop_l1
        m_1_loop_times, m_1_num_last_loop_l1 = get_loop_info(m_1_num, m_1_num_loop_l1)
        with self.tik_inst.for_range(0, m_1_loop_times) as m_1_loop_index:
            self.start_compute_each_loop(block_index, m_1_loop_index, m_1_num_loop_l1,
                                         ln_one_l1, gamma_fp32_ub, beta_fp32_ub)

    def init_ln_tensor(self, ):
        # init ln_one_l1 tensor
        m_1_num_ln_loop = self.m_1_num_ln_loop
        ln_one_l1_shape = (self.k_1, m_1_num_ln_loop, self.m_0, self.k_0)
        element_ln_one_l1 = functools.reduce(lambda a, b: a * b, ln_one_l1_shape)
        ln_one_l1 = self.tik_inst.Tensor(self.data_type, ln_one_l1_shape, tik.scope_cbuf, "ln_one_l1")

        # init gamma_fp32_ub and beta_fp32_ub tensor
        element_num_gamma = self.k_1 * self.k_0
        gamma_shape = (element_num_gamma,)
        gamma_fp32_ub = self.tik_inst.Tensor(self.fp32_type, gamma_shape, tik.scope_ubuf, "gamma_fp32_ub")
        beta_fp32_ub = self.tik_inst.Tensor(self.fp32_type, gamma_shape, tik.scope_ubuf, "beta_fp32_ub")

        with self.tik_inst.new_stmt_scope():
            # vector dup ln_one_l1
            one_l0b_ub = self.tik_inst.Tensor(self.data_type, ln_one_l1_shape, tik.scope_ubuf, "one_l0b_ub")
            self.tik_func.start_tik_compute(element_ln_one_l1, self.data_type, self.tik_func.tik_dup, (one_l0b_ub, 1))
            self.tik_inst.data_move(ln_one_l1, one_l0b_ub, 0, 1, element_ln_one_l1 // Constant.NUM_FP16_PER_BLOCK, 0, 0)

            # init fp32 gamma beta
            gamma_ub = self.tik_inst.Tensor(self.data_type, gamma_shape, name="gamma_ub", scope=tik.scope_ubuf)
            beta_ub = self.tik_inst.Tensor(self.data_type, gamma_shape, name="beta_ub", scope=tik.scope_ubuf)
            gamma_block_num = element_num_gamma // Constant.NUM_FP16_PER_BLOCK
            self.tik_inst.data_move(gamma_ub, self.gamma_gm, 0, 1, gamma_block_num, 0, 0)
            self.tik_func.start_tik_compute(element_num_gamma, self.fp32_type, self.tik_func.tik_vconv,
                                            args=(gamma_fp32_ub, gamma_ub))
            self.tik_inst.data_move(beta_ub, self.beta_gm, 0, 1, gamma_block_num, 0, 0)
            self.tik_func.start_tik_compute(element_num_gamma, self.fp32_type, self.tik_func.tik_vconv,
                                            args=(beta_fp32_ub, beta_ub))
        return ln_one_l1, gamma_fp32_ub, beta_fp32_ub

    def start_compute_each_loop(self, block_index, m_1_loop_index, m_1_num_loop,
                                ln_one_l1, gamma_fp32_ub, beta_fp32_ub):
        ln_res_l1_shape = (m_1_num_loop, self.k_1, self.m_0, self.k_0)
        ln_res_l1 = self.tik_inst.Tensor(self.data_type, ln_res_l1_shape, tik.scope_cbuf, "ln_res_l1")
        self.start_compute_ln(ln_res_l1, block_index, m_1_loop_index,
                              m_1_num_loop, ln_one_l1, gamma_fp32_ub, beta_fp32_ub)
        self.start_compute_mul(ln_res_l1, m_1_num_loop, block_index, m_1_loop_index)

    def start_compute_ln(self, ln_res_l1, block_index, m_1_loop_index, m_1_num_loop,
                         ln_one_l1, gamma_fp32_ub, beta_fp32_ub):
        m_1_num_ln_loop = self.m_1_num_ln_loop
        with self.tik_inst.for_range(0, m_1_num_loop) as m_1_index_loop:
            element_each_loop = m_1_num_ln_loop * self.k_1 * Constant.FRAC_SIZE
            x_l1, x_fp32_ub = self.compute_ln_init_data(block_index, m_1_loop_index, m_1_index_loop, m_1_num_ln_loop)
            xx_sum_ub, x_sum_ub = self.compute_ln_mad_compute(x_l1, ln_one_l1, m_1_num_ln_loop)
            ln_res_ub = self.compute_ln_norm(xx_sum_ub, x_sum_ub, x_fp32_ub,
                                             gamma_fp32_ub, beta_fp32_ub, m_1_num_ln_loop)
            self.tik_inst.data_move(ln_res_l1[m_1_index_loop, 0, 0, 0], ln_res_ub, 0,
                                    1, element_each_loop // Constant.NUM_FP16_PER_BLOCK, 0, 0)

    def compute_ln_init_data(self, block_index, m_1_loop_index, m_1_index_loop, m_1_num_ln):
        batch_index = (block_index * self.batch_m1_num_each_core
                       + m_1_loop_index * self.m_1_num_loop_l1 + m_1_index_loop) // self.m_1
        m_1_index = (block_index * self.batch_m1_num_each_core
                     + m_1_loop_index * self.m_1_num_loop_l1 + m_1_index_loop) % self.m_1
        x_l1_shape = (self.k_1, m_1_num_ln, self.m_0, self.k_0)
        element_x_l1 = functools.reduce(lambda a, b: a * b, x_l1_shape)
        x_l1 = self.tik_inst.Tensor(self.data_type, x_l1_shape, tik.scope_cbuf, "x_l1")
        x_fp32_ub = self.tik_inst.Tensor(self.fp32_type, x_l1_shape, tik.scope_ubuf, "x_fp32_ub")
        # init x l1
        m_sqrt = int(math.sqrt(self.m_1 * self.m_0))
        if self.m_split == 0 or self.m_split == m_sqrt:
            src_stride = (self.m_1 - m_1_num_ln) * Constant.FRAC_SIZE // Constant.NUM_FP16_PER_BLOCK
            self.tik_inst.data_move(x_l1, self.x_gm[batch_index, 0, m_1_index, 0, 0], 0,
                                    self.k_1,
                                    m_1_num_ln * Constant.FRAC_SIZE // Constant.NUM_FP16_PER_BLOCK,
                                    src_stride, 0)
        else:
            self.split_data_move_in(x_l1, batch_index, m_1_index, m_1_num_ln)
        with self.tik_inst.new_stmt_scope():
            x_ub = self.tik_inst.Tensor(self.data_type, x_l1_shape, tik.scope_ubuf, "x_ub")
            self.tik_inst.data_move(x_ub, x_l1, 0, 1, element_x_l1 // Constant.NUM_FP16_PER_BLOCK, 0, 0)
            self.tik_func.start_tik_compute(element_x_l1, self.fp32_type,
                                            self.tik_func.tik_vconv, args=(x_fp32_ub, x_ub))
        return x_l1, x_fp32_ub

    def split_data_move_in(self, x_l1, batch_index, m_1_index, m_1_num_ln):
        x_gm_reshape = self.x_gm.reshape((self.batch_num, self.k_1, self.m_sqrt, self.m_sqrt, self.k_0))
        x_l1_reshape = x_l1.reshape((self.k_1, m_1_num_ln * self.m_0, self.k_0))
        # ub index
        m_num = m_1_num_ln * self.m_0
        m_index_ub = self.tik_inst.Scalar("int64", init_value=0)
        m_index_ub_end = m_1_num_ln * self.m_0
        # gm index
        m_index_start = m_1_index * self.m_0
        m_split_0_index = m_index_start // self.m_sqrt

        m_split_1_start = self.tik_inst.Scalar("int64", init_value=m_index_start % self.m_sqrt)
        m_split_1_end = self.tik_inst.Scalar("int64", init_value=self.m_sqrt)

        m_num_first_move = self.tik_inst.Scalar("int64", init_value=self.m_sqrt - m_split_1_start)

        with self.tik_inst.if_scope(m_num_first_move > m_num):
            m_num_first_move.set_as(m_num)
            m_split_1_end.set_as(m_split_1_start + m_num)
        self.split_data_move_in_loop(x_l1_reshape, x_gm_reshape, batch_index, m_split_0_index, m_split_1_start,
                                     m_split_1_end, m_index_ub, m_num)

        m_index_ub.set_as(m_index_ub + m_num_first_move)

        move_in_loop_times = (m_index_ub_end - m_index_ub) // self.m_sqrt

        with self.tik_inst.if_scope(move_in_loop_times > 0):
            with self.tik_inst.for_range(0, move_in_loop_times) as move_in_index:
                m_split_0 = (m_split_0_index + 1 + move_in_index + self.m_split) % self.m_sqrt
                m_num_split_1 = self.m_sqrt - self.m_split
                self.tik_inst.data_move(x_l1_reshape[0, m_index_ub, 0],
                                        x_gm_reshape[batch_index, 0, m_split_0, self.m_split, 0],
                                        0,
                                        self.k_1,
                                        m_num_split_1 * self.k_0 // Constant.NUM_FP16_PER_BLOCK,
                                        (self.m_num - m_num_split_1) * self.k_0 // Constant.NUM_FP16_PER_BLOCK,
                                        max(0, (m_num - m_num_split_1) * self.k_0 // Constant.NUM_FP16_PER_BLOCK))
                self.tik_inst.data_move(x_l1_reshape[0, m_index_ub + m_num_split_1, 0],
                                        x_gm_reshape[batch_index, 0, m_split_0, 0, 0],
                                        0,
                                        self.k_1,
                                        self.m_split * self.k_0 // Constant.NUM_FP16_PER_BLOCK,
                                        (self.m_num - self.m_split) * self.k_0 // Constant.NUM_FP16_PER_BLOCK,
                                        max(0, (m_num - self.m_split) * self.k_0 // Constant.NUM_FP16_PER_BLOCK))
                m_index_ub.set_as(m_index_ub + self.m_sqrt)

        with self.tik_inst.if_scope(m_index_ub < m_index_ub_end):
            m_split_1_start.set_as(0)
            m_split_1_end.set_as((m_index_start + m_num) % self.m_sqrt)
            m_split_0 = (m_index_start + m_num) // self.m_sqrt
            self.split_data_move_in_loop(x_l1_reshape, x_gm_reshape, batch_index, m_split_0, m_split_1_start,
                                         m_split_1_end, m_index_ub, m_num)

    def split_data_move_in_loop(self, x_l1, x_gm, batch_index, m_split_0, m_split_1_start, m_split_1_end, m_index_ub,
                                m_num_ub):
        m_split_0 = (m_split_0 + self.m_split) % self.m_sqrt
        m_num_split_1 = self.m_sqrt - self.m_split
        with self.tik_inst.if_scope(m_split_1_end <= m_num_split_1):
            m_num_move = m_split_1_end - m_split_1_start
            self.tik_inst.data_move(x_l1[0, m_index_ub, 0],
                                    x_gm[batch_index, 0, m_split_0, self.m_split + m_split_1_start, 0],
                                    0,
                                    self.k_1,
                                    m_num_move * self.k_0 // Constant.NUM_FP16_PER_BLOCK,
                                    (self.m_num - m_num_move) * self.k_0 // Constant.NUM_FP16_PER_BLOCK,
                                    (m_num_ub - m_num_move) * self.k_0 // Constant.NUM_FP16_PER_BLOCK)
        with self.tik_inst.elif_scope(m_split_1_start >= m_num_split_1):
            m_num_move = m_split_1_end - m_split_1_start
            self.tik_inst.data_move(x_l1[0, m_index_ub, 0],
                                    x_gm[batch_index, 0, m_split_0, m_split_1_start - m_num_split_1, 0],
                                    0,
                                    self.k_1,
                                    m_num_move * self.k_0 // Constant.NUM_FP16_PER_BLOCK,
                                    (self.m_num - m_num_move) * self.k_0 // Constant.NUM_FP16_PER_BLOCK,
                                    (m_num_ub - m_num_move) * self.k_0 // Constant.NUM_FP16_PER_BLOCK)
        with self.tik_inst.else_scope():
            m_num_move_0 = m_num_split_1 - m_split_1_start
            self.tik_inst.data_move(x_l1[0, m_index_ub, 0],
                                    x_gm[batch_index, 0, m_split_0, m_split_1_start + self.m_split, 0],
                                    0,
                                    self.k_1,
                                    m_num_move_0 * self.k_0 // Constant.NUM_FP16_PER_BLOCK,
                                    (self.m_num - m_num_move_0) * self.k_0 // Constant.NUM_FP16_PER_BLOCK,
                                    (m_num_ub - m_num_move_0) * self.k_0 // Constant.NUM_FP16_PER_BLOCK)
            m_num_move_1 = m_split_1_end - m_num_split_1
            self.tik_inst.data_move(x_l1[0, m_index_ub + m_num_move_0, 0],
                                    x_gm[batch_index, 0, m_split_0, 0, 0],
                                    0,
                                    self.k_1,
                                    m_num_move_1 * self.k_0 // Constant.NUM_FP16_PER_BLOCK,
                                    (self.m_num - m_num_move_1) * self.k_0 // Constant.NUM_FP16_PER_BLOCK,
                                    (m_num_ub - m_num_move_1) * self.k_0 // Constant.NUM_FP16_PER_BLOCK)

    def compute_ln_mad_compute(self, x_l1, ln_one_l1, m_1_num_ln):
        result_lc_shape = (m_1_num_ln, m_1_num_ln, self.m_0, self.m_0)
        element_result_lc = functools.reduce(lambda a, b: a * b, result_lc_shape)
        x_sum_ub = self.tik_inst.Tensor(self.fp32_type, result_lc_shape, tik.scope_ubuf, "x_sum_ub")
        xx_sum_ub = self.tik_inst.Tensor(self.fp32_type, result_lc_shape, tik.scope_ubuf, "xx_sum_ub")
        with self.tik_inst.new_stmt_scope():
            dst_l0c = self.tik_inst.Tensor(self.fp32_type, result_lc_shape, tik.scope_cc, "dst_l0c")
            l0a_shape = (m_1_num_ln, self.k_1, self.m_0, self.k_0)
            l0b_shape = (self.k_1, m_1_num_ln, self.m_0, self.k_0)
            mat_l0a = self.tik_inst.Tensor(self.data_type, l0a_shape, tik.scope_ca, "mat_l0a")
            mat_l0b = self.tik_inst.Tensor(self.data_type, l0b_shape, tik.scope_cb, "mat_l0b")
            # start_index, repeat, src_stride, sid, is_transpose
            with self.tik_inst.for_range(0, m_1_num_ln) as m_1_index:
                self._load_2d(mat_l0a[m_1_index, 0, 0, 0], x_l1[0, m_1_index, 0, 0],
                              [0, self.k_1, m_1_num_ln, 0, False])
            self._load_2d(mat_l0b, ln_one_l1, [0, self.k_1 * m_1_num_ln, 1, 0, False])
            # dst_fm, src_fm, src_filter, matrix_m, matrix_k, matrix_n, is_bias
            self.tik_inst.mmad(dst_l0c, mat_l0a, mat_l0b,
                               m_1_num_ln * self.m_0,
                               self.k_1 * self.k_0,
                               m_1_num_ln * self.m_0, 0)
            self.tik_inst.data_move(x_sum_ub, dst_l0c, 0, 1, m_1_num_ln, 0, 0)

            self._load_2d(mat_l0b, x_l1, [0, self.k_1 * m_1_num_ln, 1, 0, False])
            self.tik_inst.mmad(dst_l0c, mat_l0a, mat_l0b,
                               m_1_num_ln * self.m_0,
                               self.k_1 * self.k_0,
                               m_1_num_ln * self.m_0, 0)
            self.tik_inst.data_move(xx_sum_ub, dst_l0c, 0, 1, m_1_num_ln, 0, 0)

            # use diagonal element fill the row to remove invalid entry in fractal_matrix
            with self.tik_inst.for_range(0, self.m_0) as brc_idx:
                var_scalar = self.tik_inst.Scalar(self.fp32_type)
                var_scalar.set_as(xx_sum_ub[0, 0, brc_idx, brc_idx])
                # set vector mask as self.m_0 to avoid vector_dup erase next value
                self.tik_inst.vector_dup(self.m_0, xx_sum_ub[0, 0, brc_idx, 0], var_scalar, 1, 1, 0)
        mean_repeat_num = element_result_lc // Constant.FP32_REPEAT_SIZE
        self.tik_inst.vmuls(Constant.MASK_FP32, x_sum_ub, x_sum_ub, self.mean_coeff,
                            mean_repeat_num, 1, 1, 8, 8)
        self.tik_inst.vmuls(Constant.MASK_FP32, xx_sum_ub, xx_sum_ub, self.mean_coeff,
                            mean_repeat_num, 1, 1, 8, 8)
        return xx_sum_ub, x_sum_ub

    def compute_ln_norm(self, xx_sum_ub, x_sum_ub, x_fp32_ub, gamma_fp32_ub, beta_fp32_ub, m_1_num_ln):
        x_l1_shape = (self.k_1, m_1_num_ln, self.m_0, self.k_0)
        ln_res_ub = self.tik_inst.Tensor(self.data_type, x_l1_shape, name="ln_res_ub", scope=tik.scope_ubuf)
        with self.tik_inst.new_stmt_scope():
            mad_shape = (m_1_num_ln, m_1_num_ln, self.m_0, self.m_0)
            element_mad_ub = functools.reduce(lambda a, b: a * b, mad_shape)
            squared_mean_ub = self.tik_inst.Tensor(self.fp32_type, mad_shape, tik.scope_ubuf, "squared_mean_ub")
            var_ub = self.tik_inst.Tensor(self.fp32_type, mad_shape, tik.scope_ubuf, "var_ub")
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
            self.tik_func.start_tik_compute(self.k_1 * m_1_num_ln * self.m_0 * self.m_0, self.fp32_type,
                                            self.tik_func.tik_vconv,
                                            args=(ln_res_ub, x_fp32_ub))
        return ln_res_ub

    def start_compute_mul(self, ln_res_l1, m_1_num_loop, block_index, m_1_loop_index):
        n_1_num_loop_l1 = self.n_1_num_loop_l1
        result_num = 3
        n_num_result = self.n_1 // result_num
        n_1_loop_times, n_1_num_last_loop = get_loop_info(n_num_result, n_1_num_loop_l1)
        for result_index in range(result_num):
            result_gm = self.outputs[result_index]
            with self.tik_inst.for_range(0, n_1_loop_times) as n_1_loop_index:
                n_1_index = n_1_loop_index * n_1_num_loop_l1 + n_num_result * result_index
                with self.tik_inst.if_scope(n_1_loop_index < n_1_loop_times - 1):
                    self.start_compute_mul_loop(result_gm, ln_res_l1, block_index, m_1_loop_index,
                                                n_1_index, m_1_num_loop, n_1_num_loop_l1)
                with self.tik_inst.else_scope():
                    self.start_compute_mul_loop(result_gm, ln_res_l1, block_index, m_1_loop_index,
                                                n_1_index, m_1_num_loop, n_1_num_last_loop)

    def start_compute_mul_loop(self, result_gm, ln_res_l1, block_index, m_1_loop_index,
                               n_1_index, m_1_num_loop, n_1_num_loop):
        loc_shape = (n_1_num_loop, m_1_num_loop, self.m_0, self.n_0)
        res_lc = self.tik_inst.Tensor(self.fp32_type, loc_shape, name="l0c", scope=tik.scope_cbuf_out)
        with self.tik_inst.new_stmt_scope():
            bias_fp32_ub = self.start_compute_mul_bias_init(n_1_index, n_1_num_loop)
            with self.tik_inst.for_range(0, m_1_num_loop) as brc_idx:
                self.tik_inst.broadcast_ub_to_l0c(res_lc[0, brc_idx, 0, 0], bias_fp32_ub,
                                                  n_1_num_loop, 1, 0, m_1_num_loop - 1)
        k_1_num_loop_l1 = self.k_1_num_loop_l1
        k_1_loop_times, k_1_num_last_loop = get_loop_info(self.k_1, k_1_num_loop_l1)
        thread_num = min(2, k_1_loop_times)
        with self.tik_inst.for_range(0, k_1_loop_times, thread_num=thread_num) as kl1_idx:
            k_1_index = kl1_idx * k_1_num_loop_l1
            with self.tik_inst.if_scope(kl1_idx < k_1_loop_times - 1):
                self._matmul_l0c_compute(res_lc, ln_res_l1, n_1_index, k_1_index, m_1_num_loop, k_1_num_loop_l1,
                                         n_1_num_loop)
            with self.tik_inst.else_scope():
                self._matmul_l0c_compute(res_lc, ln_res_l1, n_1_index, k_1_index, m_1_num_loop, k_1_num_last_loop,
                                         n_1_num_loop)

        c_ub = self.tik_inst.Tensor(self.data_type, loc_shape, name="c_ub", scope=tik.scope_ubuf)
        self.tik_inst.tensor_mov(c_ub, res_lc, 'm', 1, n_1_num_loop * m_1_num_loop, 0, 0)
        if self.unalign:
            self.data_move_out_unalign(result_gm, c_ub, block_index, m_1_loop_index, n_1_index, m_1_num_loop,
                                       n_1_num_loop)
        else:
            self.data_move_out(result_gm, c_ub, block_index, m_1_loop_index, n_1_index, m_1_num_loop, n_1_num_loop)

    def start_compute_mul_bias_init(self, n_1_index, n_1_num_loop):
        n_num_loop = n_1_num_loop * self.n_0
        bias_shape = (n_num_loop,)
        bias_fp32_ub = self.tik_inst.Tensor(self.fp32_type, bias_shape, name="bias_fp32_ub", scope=tik.scope_ubuf)
        with self.tik_inst.new_stmt_scope():
            bias_ub = self.tik_inst.Tensor(self.data_type, bias_shape, name="bias_ub", scope=tik.scope_ubuf)
            self.tik_inst.data_move(bias_ub, self.bias_gm[n_1_index * self.n_0], 0, 1,
                                    n_num_loop // Constant.NUM_FP16_PER_BLOCK, 0, 0)
            self.tik_func.start_tik_compute(n_num_loop, self.fp32_type,
                                            self.tik_func.tik_vconv,
                                            args=(bias_fp32_ub, bias_ub))
        return bias_fp32_ub

    def data_move_out(self, result_gm, result_ub, block_index, m_1_loop_index,
                      n_1_index, m_1_num_loop, n_1_num_each_loop):
        # c_ub shape (n_1, m_1, self.m_0, self.n_0)
        batch_index = (block_index * self.batch_m1_num_each_core + m_1_loop_index * self.m_1_num_loop_l1) // self.m_1
        m_1_index = (block_index * self.batch_m1_num_each_core + m_1_loop_index * self.m_1_num_loop_l1) % self.m_1
        result_ub = result_ub.reshape((n_1_num_each_loop, self.m_1_num_loop_l1 * self.m_0, self.n_0))
        m_3_times = self.m_1_num_loop_l1 * self.m_0 // self.m_3
        if m_3_times > self.m_2:
            m_3_0 = m_3_times // self.m_2
            result_trans_shape = (self.m_2, n_1_num_each_loop, m_3_0, self.m_3, self.n_0)
            result_trans_ub = self.tik_inst.Tensor(self.data_type, result_trans_shape,
                                                   name="result_trans_ub", scope=tik.scope_ubuf)
            for i in range(self.m_2):
                for j in range(m_3_0):
                    self.tik_inst.data_move(result_trans_ub[i, 0, j, 0, 0],
                                            result_ub[0, (j * self.m_2 + i) * self.m_3, 0],
                                            0,
                                            n_1_num_each_loop, self.m_3,
                                            m_1_num_loop * self.m_0 - self.m_3, (m_3_0 - 1) * self.m_3)

            if self.unalign:
                m_index_loop = (m_1_index // self.m_1_num_loop_l1 * self.m_3_pad * self.m_3_pad + \
                                m_1_index % self.m_1_num_loop_l1) * self.m_0
                m_2_0 = m_index_loop // self.m_3_pad % self.m_2
                m_3_1 = m_index_loop // (self.m_3_pad * self.m_2) % self.m_3_pad
                m_2_1 = m_index_loop // (self.m_3_pad * self.m_2 * self.m_3_pad)
                m_3_0 = m_index_loop % self.m_3_pad
            else:
                m_index_loop = m_1_index * self.m_0
                m_2_0 = m_index_loop // self.m_3 % self.m_2
                m_3_1 = m_index_loop // (self.m_3 * self.m_2) % self.m_3
                m_2_1 = m_index_loop // (self.m_3 * self.m_2 * self.m_3)

            if n_1_num_each_loop == self.n_2 * self.n_3:
                self.tik_inst.data_move(result_gm[batch_index, m_2_1, m_2_0, 0, m_3_1, 0, 0],
                                        result_trans_ub, 0,
                                        self.m_2 * n_1_num_each_loop, m_3_0 * self.m_3,
                                        0, self.m_3 * self.m_3 - m_3_0 * self.m_3)
            else:
                if self.unalign:
                    result_gm = result_gm.reshape((self.batch_num, self.m_2 * self.m_2,
                                                   self.n_2 * self.n_3, self.m_3_pad * self.m_3_pad, self.n_0))
                    nburst = n_1_num_each_loop
                    burst = self.m_3 * self.m_3
                    src_gap = 0
                    dst_gap = self.m_3_pad * self.m_3_pad - self.m_3 * self.m_3
                    n_2_3_index = n_1_index % (self.n_2 * self.n_3)
                    for i in range(self.m_2):
                        self.tik_inst.data_move(result_gm[batch_index, m_2_1 * self.m_2 + m_2_0 + i,
                                                          n_2_3_index, m_3_1, m_3_0, 0],
                                                result_trans_ub[i, 0, 0, 0, 0], 0,
                                                nburst, burst,
                                                src_gap, dst_gap)
                else:
                    result_gm = result_gm.reshape((self.batch_num, self.m_2 * self.m_2,
                                                   self.n_2 * self.n_3, self.m_3, self.m_3, self.n_0))
                    nburst = n_1_num_each_loop
                    burst = m_3_0 * self.m_3
                    src_gap = 0
                    dst_gap = self.m_3 * self.m_3 - m_3_0 * self.m_3
                    n_2_3_index = n_1_index % (self.n_2 * self.n_3)
                    for i in range(self.m_2):
                        self.tik_inst.data_move(result_gm[batch_index, m_2_1 * self.m_2 + m_2_0 + i,
                                                          n_2_3_index, m_3_1, 0, 0],
                                                result_trans_ub[i, 0, 0, 0, 0], 0,
                                                nburst, burst,
                                                src_gap, dst_gap)



        else:
            result_trans_shape = (m_3_times, n_1_num_each_loop, self.m_3, self.n_0)
            result_trans_ub = self.tik_inst.Tensor(self.data_type, result_trans_shape,
                                                   name="result_trans_ub", scope=tik.scope_ubuf)
            for i in range(m_3_times):
                self.tik_inst.data_move(result_trans_ub[i, 0, 0, 0], result_ub[0, i * self.m_3, 0], 0,
                                        n_1_num_each_loop, self.m_3,
                                        m_1_num_loop * self.m_0 - self.m_3, 0)
            m_index_loop = m_1_index * self.m_0
            m_2_0 = m_index_loop // self.m_3 % self.m_2
            m_3_1 = m_index_loop // (self.m_3 * self.m_2) % self.m_3
            m_2_1 = m_index_loop // (self.m_3 * self.m_2 * self.m_3)
            self.tik_inst.data_move(result_gm[batch_index, m_2_1, m_2_0, 0, m_3_1, 0, 0],
                                    result_trans_ub, 0,
                                    m_3_times * n_1_num_each_loop, self.m_3,
                                    0, self.m_3 * self.m_3 - self.m_3)

    def data_move_out_unalign(self, result_gm, result_ub, block_index, m_1_loop_index,
                              n_1_index, m_1_num_loop, n_1_num_each_loop):
        # c_ub shape (n_1, m_1, self.m_0, self.n_0
        batch_index = (block_index * self.batch_m1_num_each_core + m_1_loop_index * self.m_1_num_loop_l1) // self.m_1
        m1_index_start = (block_index * self.batch_m1_num_each_core + m_1_loop_index * self.m_1_num_loop_l1) % self.m_1
        m_3_times = self.m_1_num_loop_l1 * self.m_0 // self.m_3

        result_trans_shape = (n_1_num_each_loop, self.m_1_num_loop_l1, self.m_0, self.n_0)
        result_trans_ub = self.tik_inst.Tensor(self.data_type, result_trans_shape,
                                               name="result_trans_ub", scope=tik.scope_ubuf)
        m_num = self.m_1_num_loop_l1 * self.m_0
        n1_num = n_1_num_each_loop
        ub_temp_shape = (n1_num, m_num, self.n_0)
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
        m_loop_times = m_num // self.m_3
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
            self.tik_inst.data_move(result_trans_ub[ub_start_index:], result_ub[ub_temp_start_index:],
                                    0, nburst, burst, src_gap, dst_gap)
        m_index_start = m1_index_start // self.m_1_num_loop_l1 * self.m_3_pad * self.m_3_pad * self.m_0
        m_2_1 = m_index_start // self.m_3_pad % self.m_2
        m_2_0 = m_index_start // (self.m_3_pad * self.m_2 * self.m_3_pad)
        m_3_1 = m_index_start % self.m_3_pad
        m_3_0 = m_index_start // (self.m_3_pad * self.m_2) % self.m_3_pad
        burst = m3_num_0 * m3_num_1 * self.n_0 // self.num_fp16_per_block
        src_gap = 0
        dst_gap = (self.m_3_pad * self.m_3_pad - self.m_3 * self.m_3) * self.n_0 // self.num_fp16_per_block
        with self.tik_inst.for_range(0, m2_num_0) as m2_num_0_index:
            with self.tik_inst.for_range(0, m2_num_1) as m2_num_1_index:
                nburst = n1_num
                self.tik_inst.data_move(result_gm[
                                            batch_index, m_2_0 + m2_num_0_index, m_2_1 + m2_num_1_index,
                                            n_1_index, m_3_0, m_3_1, 0],
                                        result_ub[
                                            get_start_index((m2_num_0_index, m2_num_1_index, 0, 0, 0), ub_shape)],
                                        0, nburst, burst, src_gap, dst_gap)
        if n_1_num_each_loop == self.n_2 * self.n_3:
            self.tik_inst.data_move(result_gm[batch_index, m_2_1, m_2_0, 0, m_3_1, m_3_0, 0],
                                    result_trans_ub, 0,
                                    self.m_2 * n_1_num_each_loop, m_3_0 * self.m_3,
                                    0, self.m_3 * self.m_3 - m_3_0 * self.m_3)
        else:
            nburst = n_1_num_each_loop
            burst = self.m_3 * self.m_3
            src_gap = 0
            dst_gap = self.m_3_pad * self.m_3_pad - self.m_3 * self.m_3
            n_2_3_index = n_1_index % (self.n_2 * self.n_3)
            ub_shape = (m2_num_0, m2_num_1, n1_num, m3_num_0, m3_num_1, self.n_0)
            result_trans_ub = result_trans_ub.reshape(ub_shape)
            with self.tik_inst.for_range(0, m2_num_0) as m2_num_0_index:
                with self.tik_inst.for_range(0, m2_num_1) as m2_num_1_index:
                    self.tik_inst.data_move(result_gm[
                                            batch_index, m_2_0 + m2_num_0_index, m_2_1 + m2_num_1_index,
                                            n_2_3_index, m_3_0, m_3_1, 0],
                                            result_trans_ub[
                                            get_start_index((m2_num_0_index, m2_num_1_index, 0, 0, 0), ub_shape)],
                                            0, nburst, burst, src_gap, dst_gap)

    def _matmul_l0c_compute(self, res_lc, ln_res_l1, n_1_index, k_1_index, m_1_num_loop, k_1_num_loop, n_1_num_loop):
        """
        matmul_l0c_compute
        """
        # bl1 process
        weight_l1_shape = (n_1_num_loop, k_1_num_loop, self.k_0, self.n_0)
        weight_l1 = self.tik_inst.Tensor(self.data_type, weight_l1_shape, name="weight_l1", scope=tik.scope_cbuf)
        self.tik_inst.data_move(weight_l1, self.weight_gm[n_1_index, k_1_index, 0, 0], 0,
                                n_1_num_loop, k_1_num_loop * Constant.FRAC_SIZE // Constant.NUM_FP16_PER_BLOCK,
                                (self.k_1 - k_1_num_loop) * Constant.FRAC_SIZE // Constant.NUM_FP16_PER_BLOCK, 0)

        # al0 process
        mul_l0a_shape = (m_1_num_loop, k_1_num_loop, self.m_0, self.k_0)
        mul_l0a = self.tik_inst.Tensor(self.data_type, mul_l0a_shape, name="mul_l0a", scope=tik.scope_ca)
        # start_index, repeat, src_stride, sid, is_transpose
        with self.tik_inst.for_range(0, m_1_num_loop) as ml0_idx:
            self._load_2d(mul_l0a[ml0_idx, 0, 0, 0], ln_res_l1[ml0_idx, k_1_index, 0, 0],
                          [0, k_1_num_loop, 1, 0, False])

        # bl0 process
        mul_l0b_shape = (k_1_num_loop, n_1_num_loop, self.n_0, self.k_0)
        mul_l0b = self.tik_inst.Tensor(self.data_type, mul_l0b_shape, name="mul_l0b", scope=tik.scope_cb)

        with self.tik_inst.for_range(0, k_1_num_loop) as kl0_idx:
            self._load_2d(mul_l0b[kl0_idx, 0, 0, 0], weight_l1[0, kl0_idx, 0, 0],
                          [0, n_1_num_loop, k_1_num_loop, 0, True])
        self.tik_inst.mmad(res_lc, mul_l0a, mul_l0b, m_1_num_loop * self.m_0, k_1_num_loop * self.k_0,
                           n_1_num_loop * self.n_0, 1)

    def _load_2d(self, dst, src, instr_params):
        start_index, repeat, repeat_stride, sid, is_transpose = instr_params
        if tbe_platform.api_check_support("tik.load2dv2"):
            self.tik_inst.load2dv2(dst, src, start_index, repeat, 0, repeat_stride, sid, is_transpose)
        elif tbe_platform.api_check_support("tik.load2dv1"):
            self.tik_inst.load2dv1(dst, src, start_index, repeat, repeat_stride, sid, is_transpose)
        else:
            error_manager_cube.raise_err_specific_user("attention_ln_qkv",
                                                       "load2d instr unsupported.")


# 'pylint: disable = unused-argument
# 'pylint: disable=too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def swin_transformer_ln_qkv(x, gamma, beta, weight, bias,
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
    if check_supported_v2(x, gamma, beta, weight, bias,
                          query_output, key_output, value_output,
                          head_num, head_dim, seq_length, shifts, epsilon,
                          kernel_name):
        swin_transformer_ln_qkv_v2(x, gamma, beta, weight, bias,
                                   query_output, key_output, value_output,
                                   head_num, head_dim, seq_length, shifts, epsilon,
                                   kernel_name)
    elif check_supported_vit(x, gamma, beta, weight, bias,
                             query_output, key_output, value_output,
                             head_num, head_dim, seq_length, shifts, epsilon,
                             kernel_name):
        vit_transformer_ln_qkv(x, gamma, beta, weight, bias,
                                   query_output, key_output, value_output,
                                   head_num, head_dim, seq_length, shifts, epsilon,
                                   kernel_name)
    elif check_supported(x, gamma, beta, weight, bias,
                         query_output, key_output, value_output,
                         head_num, head_dim, seq_length, shifts, epsilon,
                         kernel_name):
        m_0, k_0, n_0 = 16, 16, 16
        batch_num, k_1, m_1 = x.get("shape")[:3]
        n_1 = weight.get("shape")[0]
        result_shape = query_output.get("shape")
        m_2 = int(math.sqrt(result_shape[0] // batch_num))
        n_2 = int(head_num)
        n_3 = int(head_dim // n_0)
        m_3 = int(math.sqrt(seq_length))
        dtype = x.get("dtype")
        if shifts:
            m_split = shifts[1]
        else:
            m_split = 0
        if m_3 != math.sqrt(result_shape[3] * m_0):
            unalign = True
            m_3_pad = int(math.sqrt(result_shape[3] * m_0))
        else:
            unalign = False
            m_3_pad = m_3
        obj = SwinTransformLnMatmul(batch_num, m_0, m_1, m_2, m_3, k_0, k_1, n_0, n_1, n_2, n_3, m_3_pad, dtype,
                                    m_split, unalign, epsilon, kernel_name)
        obj.start_compute()
