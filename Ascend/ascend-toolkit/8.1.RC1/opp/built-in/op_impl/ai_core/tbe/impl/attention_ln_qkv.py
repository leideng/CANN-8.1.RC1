#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
attention_ln_qkv
"""
from __future__ import absolute_import
import math
from functools import reduce
from impl.util.attention_qkv_util import Constant
from impl.util.attention_qkv_util import vconv
from impl.util.attention_qkv_util import matmul_l0c_process
from impl.util.attention_qkv_util import check_equal_shape
from impl.util.attention_qkv_util import check_dtype
from impl.util.attention_qkv_util import check_format
from impl.util.attention_qkv_util import check_trans_flag
from impl.util.attention_qkv_util import get_unit, get_tiling_special_core
from impl.util.attention_qkv_util import get_unit_muti_core
from impl.util.attention_qkv_util import get_factor
from impl.util.attention_qkv_util import load_2d
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from tbe.common.utils import shape_util


class AttentionLnQKV:
    '''
    AttentionLnQKV
    '''

    def __init__(self, params):
        self.dtype = params.get("dtype")
        self.x_shape = params.get("input_x_shape")
        self.kernel_shape = params.get("kernel_shape")
        self.gamma_shape = params.get("gamma_shape")
        self.beta_shape = params.get("beta_shape")
        self.bias_flag = params.get("bias_flag")
        self.bias_shape = params.get("bias_shape")
        self.out_shape = params.get("out_shape")
        self.norm_shape = params.get("norm_shape")
        self.kernel_ori_shape = params.get("kernel_ori_shape")
        self.trans_b = params.get("trans_b")
        self.k1_shape = self.x_shape[0]
        self.m1_shape = self.x_shape[1]
        self.n1_shape = self.kernel_shape[0] if not self.trans_b else self.kernel_shape[1]
        self.n_ori_shape = self.kernel_ori_shape[1] if not self.trans_b else self.kernel_ori_shape[0]
        self.unaligned_n_flag = False if self.n_ori_shape % Constant.N0 == 0 else True
        if self.unaligned_n_flag and not self.trans_b:
            error_manager_cube.raise_err_specific_user("attention_ln_qkv", "unsupported n shape for matmul.")
        self.ori_bs = self.m1_shape * Constant.M0
        self.ori_s = self.ori_bs // self.out_shape[0]
        self.pad_s = self.out_shape[Constant.M_INNER_INDEX] * Constant.M0
        self.unaligned_s_flag = True if self.pad_s > self.ori_s else False
        self.mean_coeff = (self.k1_shape * Constant.C0) ** (-1)
        self.mv_out_shape = (self.m1_shape * Constant.M0,)
        self.epsilon = params.get("epsilon")
        self.kernel_name = params.get("kernel_name")
        self.tik_instance = tik.Tik()
        # only support core_num be 32 or 8
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self._tiling_args_compute()
        self._init_gm_tensor(self.tik_instance)
        self._init_ln_tensors(self.tik_instance)

    def attention_ln_qkv_compute(self):
        '''
        attention_ln_qkv_compute
        '''
        tik_instance = self.tik_instance
        one_l0b_ub = tik_instance.Tensor(self.dtype, (self.ln_k_al1, 1, Constant.M0, Constant.C0), name="one_l0b_ub",
                                         scope=tik.scope_ubuf)
        tik_instance.vector_dup(Constant.MASK_FP16, one_l0b_ub, 1, self.ln_k_al1 * Constant.FRAC_SIZE // \
                                Constant.FP16_REPEAT_SIZE, 1, 8)
        tik_instance.data_move(self.one_l0b_bl1, one_l0b_ub, 0, 1, self.ln_k_al1 * Constant.M0, 0, 0)
        zero_ub = self._get_zero_ub(tik_instance)
        with tik_instance.for_range(0, self.block_m * self.block_n, block_num=self.block_m * self.block_n) as blk_idx:
            # do not split reduce_axis when load data to l1
            blk_m_idx = blk_idx % self.block_m
            blk_n_idx = blk_idx // self.block_m
            gamma_ub = tik_instance.Tensor(self.dtype, self.gamma_shape, name="gamma_ub", scope=tik.scope_ubuf)
            beta_ub = tik_instance.Tensor(self.dtype, self.beta_shape, name="beta_ub", scope=tik.scope_ubuf)
            tik_instance.data_move(gamma_ub, self.gamma_gm, 0, 1, self.k1_shape, 0, 0)
            vconv(tik_instance, gamma_ub, self.gamma_cast, self.k1_shape * Constant.C0 // Constant.FP32_REPEAT_SIZE,
                  True)
            tik_instance.data_move(beta_ub, self.beta_gm, 0, 1, self.k1_shape, 0, 0)
            vconv(tik_instance, beta_ub, self.beta_cast, self.k1_shape * Constant.C0 // Constant.FP32_REPEAT_SIZE,
                  True)
            with self.tik_instance.if_scope(tik.all(blk_idx >= self.common_core_num, self.special_core is True)):
                self._matmul_compute_template(tik_instance, [blk_m_idx, blk_n_idx, blk_idx, True, zero_ub])
            with self.tik_instance.else_scope():
                self._matmul_compute_template(tik_instance, [blk_m_idx, blk_n_idx, blk_idx, False, zero_ub])
        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=self.inputs, outputs=self.outputs)

    def _matmul_compute_template(self, tik_instance, params):
        blk_m_idx, blk_n_idx, blk_idx, special_core, zero_ub = params
        matmul_m_l0, m_single_core, ln_mal1_times = self.matmul_m_l0, self.m_single_core, self.ln_mal1_times
        if special_core:
            matmul_m_l0, m_single_core, ln_mal1_times = \
                self.matmul_m_l0_last, self.m_single_core_last, self.ln_mal1_times_last
        with tik_instance.for_range(0, m_single_core) as m_single_core_idx:
            with tik_instance.for_range(0, ln_mal1_times) as mal1_times_idx:
                ln_m_idx = (blk_m_idx * self.m_single_core + m_single_core_idx) * self.ln_mal1_times + \
                           mal1_times_idx
                if special_core:
                    ln_m_idx = self.common_core_num * self.m_single_core * self.ln_mal1_times + \
                               (blk_m_idx - self.common_core_num) * self.m_single_core_last * \
                               self.ln_mal1_times_last + m_single_core_idx * self.ln_mal1_times_last + \
                               mal1_times_idx
                self._ln_compute(tik_instance, ln_m_idx, mal1_times_idx, special_core)
            matmul_m_idx = (blk_m_idx * self.m_single_core + m_single_core_idx) * self.matmul_m_al1
            if special_core:
                matmul_m_idx = (self.common_core_num * self.m_single_core) * self.matmul_m_al1 + \
                               (blk_m_idx - self.common_core_num) * self.m_single_core_last * \
                               self.matmul_m_al1_last + m_single_core_idx * self.matmul_m_al1_last
            l0c = tik_instance.Tensor(Constant.FP32_DTYPE,
                                      (self.matmul_n_l0, matmul_m_l0, Constant.M0,
                                       Constant.N0), name="l0c", scope=tik.scope_cbuf_out)
            for i in range(Constant.KERNEL_NUM):
                self._matmul_compute(tik_instance, [blk_n_idx, matmul_m_idx, special_core],
                                     [l0c, self.inputs[1 + i], self.outputs[1 + i],
                                      self.inputs[len(self.inputs) + i - Constant.KERNEL_NUM], zero_ub])

    def _tiling_args_compute(self):
        '''
        tiling args setting
        '''
        # in layer_norm cube, ln_m_al0 is same as ln_n_bl0, which must be set as 1
        self.ln_m_al1 = 1
        self.ln_m_al0 = 1
        self.ln_k_al1 = self.k1_shape
        self.ln_k_al0 = self.k1_shape

        self.block_m = self.core_num
        self.block_n = 1

        self.matmul_n_l0 = Constant.CANDIDATE_TILING_N
        if self.unaligned_n_flag and self.trans_b:
            # add support for H=504 scenarios, n_l0 can be 12
            self.matmul_n_l0 = Constant.CANDIDATE_TILING_N2
        matmul_k_max = tbe_platform.get_soc_spec("L0B_SIZE") // self.matmul_n_l0 // Constant.FRAC_SIZE // \
            Constant.DTYPE_SIZE.get(self.dtype) // Constant.DOUBLE_BUFFER
        matmul_m_max = tbe_platform.get_soc_spec("L0A_SIZE") // matmul_k_max // Constant.FRAC_SIZE // \
            Constant.DTYPE_SIZE.get(self.dtype) // Constant.DOUBLE_BUFFER
        # matmul_m_l0 can only be 12 or 8
        m_inner = self.out_shape[Constant.M_INNER_INDEX]

        self.special_core = False if self.m1_shape % self.block_m == 0 else True
        if self.special_core is False and (self.m1_shape // self.block_m) % Constant.CANDIDATE_TILING_M2 == 0 \
                and (m_inner % Constant.CANDIDATE_TILING_M2 == 0 or Constant.CANDIDATE_TILING_M2 % m_inner == 0):
            self.matmul_m_l0 = Constant.CANDIDATE_TILING_M2
        elif self.special_core is False and (self.m1_shape // self.block_m) % Constant.CANDIDATE_TILING_M1 == 0 \
                and (m_inner % Constant.CANDIDATE_TILING_M1 == 0 or Constant.CANDIDATE_TILING_M1 % m_inner == 0):
            self.matmul_m_l0 = Constant.CANDIDATE_TILING_M1
        elif self.special_core:
            self.matmul_m_l0 = get_unit_muti_core(self.m1_shape, self.block_m, matmul_m_max)
        else:
            # add support for S=50 scenarios, find the divisible m_l0 value within the range of [1,12]
            need_other_check = False if self.unaligned_s_flag else True
            self.matmul_m_l0 = get_factor(self.m1_shape // self.block_m, Constant.CANDIDATE_TILING_M1,
                                          m_inner, need_other_check)
        # restrict matmul_k_l0 by L0A_SIZE && matmul_m_l0 / L0B_SIZE && matmul_n_l0
        self.matmul_k_l0 = min(
            tbe_platform.get_soc_spec("L0A_SIZE") // self.matmul_m_l0 // Constant.FRAC_SIZE // \
            Constant.DTYPE_SIZE.get(self.dtype) // Constant.DOUBLE_BUFFER,
            tbe_platform.get_soc_spec("L0B_SIZE") // self.matmul_n_l0 // Constant.FRAC_SIZE // \
            Constant.DTYPE_SIZE.get(self.dtype) // Constant.DOUBLE_BUFFER)
        # non-aligned scenarios, k_l0 may not be divisible, here re-check to obtain the divisible value
        self.matmul_k_l0 = get_factor(self.k1_shape, self.matmul_k_l0, 0, False)
        self.matmul_m_al1 = self.matmul_m_l0
        self.matmul_n_l1 = self.matmul_n_l0
        self.matmul_k_al1 = self.matmul_k_l0
        self.matmul_k_bl1 = self.matmul_k_l0
        self.ln_mal1_times = self.matmul_m_al1 // self.ln_m_al1
        self.m_single_core = math.ceil(self.m1_shape / self.block_m) // self.matmul_m_al1
        self.n_single_core = self.n1_shape // self.block_n // self.matmul_n_l1
        if self.unaligned_n_flag and self.trans_b:
            self.n_single_core = Constant.CANDIDATE_TILING_N_SINGLE_CORE
        self.one_core_data, self.common_core_num, self.one_core_data_last, self.last_core_num = \
            get_tiling_special_core(self.m1_shape, self.block_m)
        self.matmul_m_l0_last = self.matmul_m_l0
        if self.special_core:
            self.matmul_m_l0_last = get_unit(self.one_core_data_last, 1, matmul_m_max)
        self.matmul_m_al1_last = self.matmul_m_l0_last
        self.m_single_core_last = self.one_core_data_last // self.matmul_m_al1_last
        self.ln_mal1_times_last = self.matmul_m_al1_last // self.ln_m_al1

    def _init_gm_tensor(self, tik_instance):
        '''
        init gm tensors
        '''
        # init input_gm tensor
        self.x_gm = tik_instance.Tensor(self.dtype, self.x_shape, name="x_gm",
                                        scope=tik.scope_gm)
        self.kernel_query_gm = tik_instance.Tensor(self.dtype, self.kernel_shape, name="kernel_query_gm",
                                                   scope=tik.scope_gm)
        self.kernel_key_gm = tik_instance.Tensor(self.dtype, self.kernel_shape, name="kernel_key_gm",
                                                 scope=tik.scope_gm)
        self.kernel_value_gm = tik_instance.Tensor(self.dtype, self.kernel_shape, name="kernel_value_gm",
                                                   scope=tik.scope_gm)
        self.gamma_gm = tik_instance.Tensor(self.dtype, self.gamma_shape, name="gamma_gm",
                                            scope=tik.scope_gm)
        self.beta_gm = tik_instance.Tensor(self.dtype, self.beta_shape, name="beta_gm",
                                           scope=tik.scope_gm)
        self.inputs = [self.x_gm, self.kernel_query_gm, self.kernel_key_gm,
                       self.kernel_value_gm, self.gamma_gm, self.beta_gm]
        self.bias_query_gm = self.bias_key_gm = self.bias_value_gm = None
        if self.bias_flag:
            self.bias_query_gm = tik_instance.Tensor(self.dtype, self.bias_shape, name="bias_query_gm",
                                                     scope=tik.scope_gm)
            self.bias_key_gm = tik_instance.Tensor(self.dtype, self.bias_shape, name="bias_key_gm",
                                                   scope=tik.scope_gm)
            self.bias_value_gm = tik_instance.Tensor(self.dtype, self.bias_shape, name="bias_value_gm",
                                                     scope=tik.scope_gm)
            self.inputs.extend([self.bias_query_gm, self.bias_key_gm, self.bias_value_gm])
        # init output_gm tensor
        self.norm_gm = tik_instance.Tensor(self.dtype, self.norm_shape, name="norm_gm",
                                           scope=tik.scope_gm)
        self.query_output_gm = tik_instance.Tensor(self.dtype, self.out_shape, name="query_output_gm",
                                                   scope=tik.scope_gm)
        self.key_output_gm = tik_instance.Tensor(self.dtype, self.out_shape, name="key_output_gm",
                                                 scope=tik.scope_gm)
        self.value_output_gm = tik_instance.Tensor(self.dtype, self.out_shape, name="value_output_gm",
                                                   scope=tik.scope_gm)
        self.mean_out_gm = tik_instance.Tensor(self.dtype, self.mv_out_shape, name="mean_out_gm",
                                               scope=tik.scope_gm)
        self.var_out_gm = tik_instance.Tensor(self.dtype, self.mv_out_shape, name="var_out_gm",
                                              scope=tik.scope_gm)
        self.outputs = [self.norm_gm, self.query_output_gm, self.key_output_gm,
                        self.value_output_gm, self.mean_out_gm, self.var_out_gm]

    def _init_ln_tensors(self, tik_instance):
        '''
        init layer_norm tensors
        '''
        self.mad_shape = (self.ln_m_al0, self.ln_m_al0, Constant.M0, Constant.C0)
        one_l0b_bl1_shape = (self.ln_k_al1, 1, Constant.M0, Constant.C0)
        self.one_l0b_bl1 = tik_instance.Tensor(self.dtype, one_l0b_bl1_shape, name="one_l0b_bl1",
                                               scope=tik.scope_cbuf)
        self.gamma_cast = tik_instance.Tensor(Constant.FP32_DTYPE, self.gamma_shape, name="gamma_cast",
                                              scope=tik.scope_ubuf)
        self.beta_cast = tik_instance.Tensor(Constant.FP32_DTYPE, self.beta_shape, name="beta_cast",
                                             scope=tik.scope_ubuf)
        res_l1_shape = (self.ln_k_al1, self.matmul_m_al1, Constant.M0, Constant.C0)
        self.ln_res_l1 = tik_instance.Tensor(self.dtype, res_l1_shape, name="ln_res_l1", scope=tik.scope_cbuf)
        res_l1_shape_last = (self.ln_k_al1, self.matmul_m_al1_last, Constant.M0, Constant.C0)
        self.ln_res_l1_last = tik_instance.Tensor(self.dtype, res_l1_shape_last, name="ln_res_l1_last",
                                                  scope=tik.scope_cbuf)
        self.x_l1_shape = (self.ln_k_al1, self.ln_m_al1, Constant.M0, Constant.C0)
        self.x_l1 = tik_instance.Tensor(self.dtype, self.x_l1_shape, name="x_l1", scope=tik.scope_cbuf)
        x_l0a_shape = (self.ln_m_al0, self.ln_k_al0, Constant.M0, Constant.C0)
        self.x_l0a = tik_instance.Tensor(self.dtype, x_l0a_shape, name="x_l0a", scope=tik.scope_ca)

    def _ln_compute(self, tik_instance, ln_m_idx, mal1_times_idx, special_core):
        '''
        ln_compute
        '''
        # Tik do not support set_2d, the ones tensor should be dumped from UB
        tik_instance.data_move(self.x_l1, self.x_gm[ln_m_idx * self.ln_m_al1 * Constant.M0 * Constant.C0:],
                               0, self.ln_k_al1, Constant.M0, (self.m1_shape - 1) * Constant.M0, 0)
        x_ub = tik_instance.Tensor(self.dtype, self.x_l1_shape, name="x_ub", scope=tik.scope_ubuf)
        cast_x_ub = tik_instance.Tensor(Constant.FP32_DTYPE, self.x_l1_shape, name="cast_x_ub", scope=tik.scope_ubuf)
        tik_instance.data_move(x_ub, self.x_l1, 0, 1, self.ln_k_al1 * Constant.M0, 0, 0)
        vconv(tik_instance, x_ub, cast_x_ub, self.ln_k_al1 * Constant.C0 * Constant.M0 // Constant.FP32_REPEAT_SIZE,
              True)
        # process mean
        x_sum_ub = self._mad_compute(tik_instance, 1, is_mean_mad=True)
        # process variance
        xx_sum_ub = self._mad_compute(tik_instance, self.ln_m_al0)
        squared_mean_ub = tik_instance.Tensor(Constant.FP32_DTYPE, self.mad_shape, name="squared_mean_ub",
                                              scope=tik.scope_ubuf)
        var_ub = tik_instance.Tensor(Constant.FP32_DTYPE, self.mad_shape, name="var_ub", scope=tik.scope_ubuf)
        # mean^2
        tik_instance.vmul(Constant.MASK_FP32, squared_mean_ub, x_sum_ub, x_sum_ub, Constant.FRAC_REPEAT_NUM,
                          1, 1, 1, 8, 8, 8)
        mean_cast_ub = tik_instance.Tensor(self.dtype, self.mad_shape, name="mean_cast_ub", scope=tik.scope_ubuf)
        mean_trans_ub = tik_instance.Tensor(self.dtype, self.mad_shape, name="mean_trans_ub", scope=tik.scope_ubuf)
        # move x_sum_ub to mean_gm
        self._nz_to_nd_out(tik_instance, [x_sum_ub, mean_cast_ub, mean_trans_ub], self.mean_out_gm, ln_m_idx)
        # variance is x^2 - mean^2
        tik_instance.vsub(Constant.MASK_FP32, var_ub, xx_sum_ub, squared_mean_ub, Constant.FRAC_REPEAT_NUM,
                          1, 1, 1, 8, 8, 8)
        # variance + epsilon
        tik_instance.vadds(Constant.MASK_FP32, squared_mean_ub, var_ub, self.epsilon, Constant.FRAC_REPEAT_NUM,
                           1, 1, 8, 8)
        var_cast_ub = tik_instance.Tensor(self.dtype, self.mad_shape, name="var_cast_ub", scope=tik.scope_ubuf)
        var_trans_ub = tik_instance.Tensor(self.dtype, self.mad_shape, name="var_trans_ub", scope=tik.scope_ubuf)
        # move xx_sum_ub to variance_gm
        self._nz_to_nd_out(tik_instance, [var_ub, var_cast_ub, var_trans_ub], self.var_out_gm, ln_m_idx)
        # rsqrt of variance + epsilon
        tik_instance.vln(Constant.MASK_FP32, squared_mean_ub, squared_mean_ub, Constant.FRAC_REPEAT_NUM, 1, 1, 8, 8)
        tik_instance.vmuls(Constant.MASK_FP32, squared_mean_ub, squared_mean_ub, Constant.SQUARE_ROOT,
                           Constant.FRAC_REPEAT_NUM, 1, 1, 8, 8)
        tik_instance.vexp(Constant.MASK_FP32, squared_mean_ub, squared_mean_ub, Constant.FRAC_REPEAT_NUM, 1, 1, 8, 8)
        self._ln_scale(tik_instance, [x_sum_ub, squared_mean_ub, cast_x_ub], [ln_m_idx, mal1_times_idx], special_core)

    def _ln_scale(self, tik_instance, ub_tensor_list, idx_list, special_core):
        '''
        substract mean && variance division
        '''
        ln_res_l1, matmul_m_al1 = self.ln_res_l1, self.matmul_m_al1
        if special_core:
            ln_res_l1, matmul_m_al1 = self.ln_res_l1_last, self.matmul_m_al1_last
        x_sum_ub, squared_mean_ub, cast_x_ub = ub_tensor_list
        ln_m_idx, mal1_times_idx = idx_list
        with tik_instance.for_range(0, Constant.M0 * Constant.C0 // Constant.FP32_REPEAT_SIZE) as sub_mul_idx:
            vsub_offset = Constant.FP32_REPEAT_SIZE * sub_mul_idx
            # x - mean
            tik_instance.vsub(Constant.MASK_FP32, cast_x_ub[vsub_offset:], cast_x_ub[vsub_offset:],
                              x_sum_ub[vsub_offset:], self.ln_k_al1, 1, 1, 1, Constant.FP32_REPEAT_STRIDE,
                              Constant.FP32_REPEAT_STRIDE, 0)
        with tik_instance.for_range(0, Constant.M0 * Constant.C0 // Constant.FP32_REPEAT_SIZE) as sub_mul_idx:
            vsub_offset = Constant.FP32_REPEAT_SIZE * sub_mul_idx
            # norm is x - mean divides sqrt of variance + epsilon
            tik_instance.vmul(Constant.MASK_FP32, cast_x_ub[vsub_offset:], cast_x_ub[vsub_offset:],
                              squared_mean_ub[vsub_offset:], self.ln_k_al1, 1, 1, 1, Constant.FP32_REPEAT_STRIDE,
                              Constant.FP32_REPEAT_STRIDE, 0)
        with tik_instance.for_range(0, Constant.FP32_BLOCK_STRIDE) as outer_idx:
            with tik_instance.for_range(0, Constant.FRAC_SIZE // Constant.C0 // 8) as inner_idx:
                # gamma muls norm
                tik_instance.vmul(Constant.MASK_FP32, cast_x_ub[8 * Constant.C0 * inner_idx + 8 * outer_idx:],
                                  cast_x_ub[8 * Constant.C0 * inner_idx + 8 * outer_idx:],
                                  self.gamma_cast[8 * outer_idx:],
                                  self.ln_k_al1, Constant.FP32_BLOCK_STRIDE, Constant.FP32_BLOCK_STRIDE, 0,
                                  Constant.FP32_REPEAT_STRIDE, Constant.FP32_REPEAT_STRIDE, Constant.FP32_BLOCK_STRIDE)
        with tik_instance.for_range(0, Constant.FP32_BLOCK_STRIDE) as outer_idx:
            with tik_instance.for_range(0, Constant.FRAC_SIZE // Constant.C0 // 8) as inner_idx:
                # gamma muls norm add beta
                tik_instance.vadd(Constant.MASK_FP32, cast_x_ub[8 * Constant.C0 * inner_idx + 8 * outer_idx:],
                                  cast_x_ub[8 * Constant.C0 * inner_idx + 8 * outer_idx:],
                                  self.beta_cast[8 * outer_idx:],
                                  self.ln_k_al1, Constant.FP32_BLOCK_STRIDE, Constant.FP32_BLOCK_STRIDE, 0,
                                  Constant.FP32_REPEAT_STRIDE, Constant.FP32_REPEAT_STRIDE, Constant.FP32_BLOCK_STRIDE)
        cast_ln_res = tik_instance.Tensor(self.dtype, self.x_l1_shape, name="cast_ln_res", scope=tik.scope_ubuf)
        vconv(tik_instance, cast_x_ub, cast_ln_res, self.ln_k_al1 * Constant.C0 * Constant.M0 // \
              Constant.FP32_REPEAT_SIZE, False)
        # use cast_ln_res as x_input of matmul_qkv
        tik_instance.data_move(ln_res_l1[mal1_times_idx * Constant.M0 * Constant.C0:], cast_ln_res, 0,
                               self.ln_k_al1, Constant.M0, 0, (matmul_m_al1 - self.ln_m_al1) * Constant.M0)
        tik_instance.data_move(self.norm_gm[ln_m_idx * Constant.M0 * Constant.C0:], cast_ln_res, 0, self.ln_k_al1,
                               Constant.M0, 0, (self.m1_shape - self.ln_m_al1) * Constant.M0)

    def _mad_compute(self, tik_instance, mad_n, is_mean_mad=False):
        '''
        ln_mad_compute
        '''
        l0b_tensor_name = "x_l0b"
        ub_tensor_name = "xx_sum_ub"
        if is_mean_mad:
            l0b_tensor_name = "one_l0b"
            ub_tensor_name = "x_sum_ub"
        x_sum_ub = tik_instance.Tensor(Constant.FP32_DTYPE, self.mad_shape, name=ub_tensor_name, scope=tik.scope_ubuf)
        dst_l0c = tik_instance.Tensor(Constant.FP32_DTYPE, self.mad_shape, name="dst_l0c", scope=tik.scope_cc)
        with tik_instance.for_range(0, self.ln_k_al1 // self.ln_k_al0) as kl1_factor_idx:
            l0b_shape = (self.ln_k_al0, self.ln_m_al0, Constant.M0, Constant.C0)
            one_l0b = tik_instance.Tensor(self.dtype, l0b_shape, name=l0b_tensor_name, scope=tik.scope_cb)
            if is_mean_mad:
                # in mean process, the calculation is sum(x)
                # al0 process
                with tik_instance.for_range(0, self.ln_m_al0) as mal0_idx:
                    load_2d(self, self.x_l0a, self.x_l1[mal0_idx * Constant.M0 * Constant.C0:],
                            [0, self.ln_k_al0, self.ln_m_al0, 0, False])
                # bl0 process
                load_2d(self, one_l0b, self.one_l0b_bl1, [0, self.ln_k_al0 * mad_n, 1, 0, False])
            else:
                # in variance process, the al0 can be reused; the calculation is (1,x) * (x,1) = x^2
                # bl0 process
                load_2d(self, one_l0b, self.x_l1, [0, self.ln_k_al0 * mad_n, 1, 0, False])
            # l0c process
            with tik_instance.if_scope(kl1_factor_idx == 0):
                tik_instance.mmad(dst_l0c, self.x_l0a, one_l0b, self.ln_m_al0 * Constant.M0,
                                  self.ln_k_al0 * Constant.C0, mad_n * Constant.C0, 0)
            with tik_instance.else_scope():
                tik_instance.mmad(dst_l0c, self.x_l0a, one_l0b, self.ln_m_al0 * Constant.M0,
                                  self.ln_k_al0 * Constant.C0, mad_n * Constant.C0, 1)
        tik_instance.data_move(x_sum_ub, dst_l0c, 0, 1, self.ln_m_al0, 0, 0)
        if not is_mean_mad:
            # use diagonal element fill the row to remove invalid entry in fractal_matrix
            with tik_instance.for_range(0, Constant.M0) as brc_idx:
                var_scalar = tik_instance.Scalar(Constant.FP32_DTYPE)
                var_scalar.set_as(x_sum_ub[0, 0, (brc_idx * (Constant.C0 + 1)) // Constant.C0,
                                           (brc_idx * (Constant.C0 + 1)) % Constant.C0])
                # set vector mask as Constant.C0 to avoid vector_dup erase next value
                tik_instance.vector_dup(Constant.C0, x_sum_ub[brc_idx * Constant.C0:], var_scalar, 1, 1, 0)
        tik_instance.vmuls(Constant.MASK_FP32, x_sum_ub, x_sum_ub, self.mean_coeff, self.ln_m_al0 * \
                           Constant.FRAC_REPEAT_NUM, 1, 1, 8, 8)
        return x_sum_ub

    def _nz_to_nd_out(self, tik_instance, ub_tensor_list, out_tensor, idx):
        '''
        data move n1mn0 to mn
        '''
        if tbe_platform.api_check_support("tik.load2dv1"):
            src_ub, cast_ub, trans_ub = ub_tensor_list
            vconv(tik_instance, src_ub, cast_ub, Constant.FRAC_REPEAT_NUM, False)
            tik_instance.vtranspose(trans_ub, cast_ub)
            # after transpose, output is the first row
            tik_instance.data_move(out_tensor[idx * Constant.M0:], trans_ub, 0, self.ln_m_al0, 1, 0, 0)

    def _get_unaligned_h_info(self):
        """
        get information about h
        """
        h_inner = self.out_shape[Constant.H_INNER_INDEX]
        h_outer = self.matmul_n_l1 // h_inner
        h_ori = self.n_ori_shape // self.out_shape[1]
        return h_inner, h_outer, h_ori

    def _get_bias(self, tik_instance, bias_gm, matmul_n_idx):
        """
        get bias tensor
        """
        bias_shape = (self.matmul_n_l0 * Constant.N0,)
        bias_ub = tik_instance.Tensor(self.dtype, bias_shape, name="bias_ub", scope=tik.scope_ubuf)
        cast_bias_ub = tik_instance.Tensor(Constant.FP32_DTYPE, bias_shape, name="cast_bias_ub", scope=tik.scope_ubuf)
        if self.unaligned_n_flag:
            h_inner, h_outer, h_ori = self._get_unaligned_h_info()
            with tik_instance.for_range(0, h_outer) as h_idx:
                # [h_ori,h_ori,h_ori,...] -> [h_inner*16,h_inner*16,h_inner*16,...]
                tik_instance.data_move(bias_ub[h_idx * h_inner * Constant.N0],
                                       bias_gm[matmul_n_idx + h_idx * h_ori], 0, 1, h_inner, 0, 0)
        else:
            tik_instance.data_move(bias_ub, bias_gm[matmul_n_idx * Constant.N0:], 0, 1, self.matmul_n_l0, 0, 0)
        vconv(tik_instance, bias_ub, cast_bias_ub, self.matmul_n_l0 * Constant.N0 // Constant.FP32_REPEAT_SIZE, True)
        return cast_bias_ub

    def _get_bl1_offset(self, idx_list, ori_offset, ping_pong=0):
        """
        get the starting coordinate position of bl1
        """
        blk_n_idx, n_single_core_idx, matmul_n_idx, kl1_idx = idx_list
        if ping_pong == 0:
            if not self.trans_b:
                bl1_src_offset = (blk_n_idx * self.n_single_core + n_single_core_idx) * self.matmul_n_l1 * \
                                 self.k1_shape * Constant.C0 * Constant.N0 + Constant.DOUBLE_BUFFER * kl1_idx * \
                                 self.matmul_k_bl1 * Constant.C0 * Constant.N0
            else:
                if self.unaligned_n_flag:
                    bl1_src_offset = Constant.DOUBLE_BUFFER * kl1_idx * self.matmul_k_bl1 * \
                                     self.n1_shape * Constant.FRAC_SIZE + \
                                     matmul_n_idx * Constant.C0
                else:
                    bl1_src_offset = Constant.DOUBLE_BUFFER * kl1_idx * self.matmul_k_bl1 * \
                                     self.n1_shape * Constant.FRAC_SIZE + \
                                     (blk_n_idx * self.n_single_core + n_single_core_idx) * \
                                     self.matmul_n_l1 * Constant.FRAC_SIZE
        else:
            if not self.trans_b:
                bl1_src_offset = ori_offset + self.matmul_k_bl1 * Constant.FRAC_SIZE
            else:
                bl1_src_offset = ori_offset + self.matmul_k_bl1 * self.n1_shape * Constant.FRAC_SIZE
        return bl1_src_offset

    def _get_bl1(self, tik_instance, ping_pong_suffix, kernel_gm, bl1_src_offset):
        """
        get bl1 tensor
        """
        bl1 = tik_instance.Tensor(self.dtype, (self.matmul_n_l1 * self.matmul_k_bl1 * Constant.C0 * Constant.N0,),
                                  name="bl1_" + ping_pong_suffix, scope=tik.scope_cbuf)
        if self.trans_b:
            if self.unaligned_n_flag:
                h_inner, h_outer, h_ori = self._get_unaligned_h_info()
                with tik_instance.for_range(0, self.matmul_k_bl1) as k_idx:
                    tik_instance.data_move(bl1[k_idx * self.matmul_n_l1 * Constant.FRAC_SIZE],
                                           kernel_gm[bl1_src_offset + k_idx * self.n1_shape * Constant.FRAC_SIZE:],
                                           0, h_outer, h_ori, 0, h_inner * Constant.N0 - h_ori)
            else:
                tik_instance.data_move(bl1, kernel_gm[bl1_src_offset:], 0, self.matmul_k_bl1,
                                       self.matmul_n_l1 * Constant.N0,
                                       (self.n1_shape - self.matmul_n_l1) * Constant.N0, 0)
        else:
            tik_instance.data_move(bl1, kernel_gm[bl1_src_offset:], 0, self.matmul_n_l1,
                                   self.matmul_k_bl1 * Constant.C0,
                                   (self.k1_shape - self.matmul_k_bl1) * Constant.C0, 0)
        return bl1

    def _get_bl0(self, tik_instance, bl1, ping_pong_suffix):
        """
        get bl0 tensor
        """
        bl0 = tik_instance.Tensor(self.dtype, (self.matmul_k_l0 * self.matmul_n_l0 * Constant.N0 * Constant.C0,),
                                  name="bl0_" + ping_pong_suffix, scope=tik.scope_cb)
        if not self.trans_b:
            with tik_instance.for_range(0, self.matmul_k_l0) as kl0_idx:
                load_2d(self, bl0[kl0_idx * self.matmul_n_l0 * Constant.N0 * Constant.C0],
                        bl1[kl0_idx * Constant.C0 * Constant.N0],
                        [0, self.matmul_n_l0, self.matmul_k_l0, 0, True])
        else:
            load_2d(self, bl0, bl1, [0, self.matmul_n_l0 * self.matmul_k_l0, 1, 0, False])
        return bl0

    def _unaligned_n_transpose(self, tik_instance, idx_list, tensor_list):
        """
        processing the transpose process in the H/N non-aligned scenario
        """
        matmul_m_idx, matmul_n_idx, special_core = idx_list
        matmul_m_l0 = self.matmul_m_l0
        if special_core:
            matmul_m_l0 = self.matmul_m_l0_last
        out_gm, c_ub = tensor_list
        h_inner, h_outer, h_ori = self._get_unaligned_h_info()
        h_tail = (h_inner * Constant.N0 - h_ori) * Constant.M0
        c_ub_trans = tik_instance.Tensor(self.dtype, (self.matmul_n_l0, matmul_m_l0, Constant.N0, Constant.M0),
                                         name="c_ub_trans", scope=tik.scope_ubuf)
        # n1,m1,m0,n0 -> n1,m1,n0,m0
        dst_list = [c_ub_trans[Constant.N0 * i] for i in range(Constant.N0)]
        src_list = [c_ub[Constant.N0 * i] for i in range(Constant.N0)]
        tik_instance.vnchwconv(False, False, dst_list, src_list, matmul_m_l0 * self.matmul_n_l0, Constant.N0,
                               Constant.N0)
        # pad 0
        with tik_instance.for_range(0, h_outer) as h_idx:
            tik_instance.vector_dup(h_tail,
                                    c_ub_trans[h_idx * h_inner * matmul_m_l0 * Constant.FRAC_SIZE +
                                               (h_inner - 1) * matmul_m_l0 * Constant.FRAC_SIZE +
                                               (h_ori - (h_inner - 1) * Constant.N0) * Constant.M0:],
                                    0, matmul_m_l0, 1, Constant.M0)
        # n1,m1,n0,m0 -> n1,m1,m0,n0
        tik_instance.vnchwconv(False, False, src_list, dst_list, matmul_m_l0 * self.matmul_n_l0, Constant.N0,
                               Constant.N0)
        m_inner = self.out_shape[Constant.M_INNER_INDEX]
        m_out_size = math.gcd(matmul_m_l0, m_inner)
        with tik_instance.for_range(0, matmul_m_l0 // m_out_size) as m_out_idx:
            real_m_idx = matmul_m_idx + m_out_idx * m_out_size
            out_offset = (real_m_idx % m_inner) * Constant.FRAC_SIZE + \
                         matmul_n_idx // h_ori * h_inner * m_inner * Constant.FRAC_SIZE + \
                         real_m_idx // m_inner * self.out_shape[1] * h_inner * m_inner * Constant.FRAC_SIZE
            src_offset = m_out_idx * m_out_size * Constant.FRAC_SIZE
            tik_instance.data_move(out_gm[out_offset:], c_ub[src_offset:], 0, self.matmul_n_l0,
                                   m_out_size * Constant.M0, (matmul_m_l0 - m_out_size) * Constant.M0,
                                   (m_inner - m_out_size) * Constant.M0)

    def _get_zero_ub(self, tik_instance):
        """
        get zero tensor
        """
        zero_ub = None
        if self.unaligned_s_flag:
            zero_num = self.pad_s - self.ori_s
            zero_ub = tik_instance.Tensor(self.dtype, (zero_num * Constant.N0,), name="zero_ub", scope=tik.scope_ubuf)
            repeat = zero_num * Constant.N0 // Constant.MASK_FP16
            tail = zero_num * Constant.N0 % Constant.MASK_FP16
            if repeat > 0:
                tik_instance.vector_dup(Constant.MASK_FP16, zero_ub, 0, repeat, 1, Constant.BLOCK_PER_REPEAT)
            if tail > 0:
                tik_instance.vector_dup(tail, zero_ub[repeat * Constant.MASK_FP16], 0, 1, 1, Constant.BLOCK_PER_REPEAT)
        return zero_ub

    def _unaligned_m_transpose(self, tik_instance, idx_list, tensor_list):
        """
        processing the transpose process in the S non-aligned scenario
        """
        matmul_m_idx, matmul_n_idx, special_core = idx_list
        matmul_m_l0 = self.matmul_m_l0
        if special_core:
            matmul_m_l0 = self.matmul_m_l0_last
        out_gm, c_ub, zero_ub = tensor_list
        zero_num = self.pad_s - self.ori_s
        real_m_idx = matmul_m_idx * Constant.M0
        pre_tail = self.ori_s - real_m_idx % self.ori_s
        ori_s_num = (matmul_m_l0 * Constant.M0 - pre_tail) // self.ori_s
        cur_tail = (matmul_m_l0 * Constant.M0 - pre_tail) % self.ori_s
        with tik_instance.if_scope(pre_tail > 0):
            # (pre_tail,ori_s,...) -> (pre_tail,zero,ori_s,zero,...)
            m_idx = real_m_idx
            out_offset = m_idx % self.ori_s * Constant.N0 + \
                         matmul_n_idx * self.pad_s * Constant.N0 + \
                         m_idx // self.ori_s * self.n1_shape * self.pad_s * Constant.N0
            tik_instance.data_move(out_gm[out_offset:], c_ub, 0, self.matmul_n_l0, pre_tail,
                                   matmul_m_l0 * Constant.M0 - pre_tail,
                                   self.pad_s - pre_tail)
            out_offset += pre_tail * Constant.N0
            tik_instance.data_move(out_gm[out_offset:], zero_ub, 0, 1, zero_num, 0, 0)
        with tik_instance.if_scope(ori_s_num > 0):
            # (ori_s,ori_s,ori_s,...) -> (ori_s,zero,ori_s,zero,ori_s,zero,...)
            with tik_instance.for_range(0, ori_s_num) as m_out_idx:
                m_idx = real_m_idx + pre_tail + m_out_idx * self.ori_s
                out_offset = m_idx % self.ori_s * Constant.N0 + \
                             matmul_n_idx * self.pad_s * Constant.N0 + \
                             m_idx // self.ori_s * self.n1_shape * self.pad_s * Constant.N0
                src_offset = (pre_tail + m_out_idx * self.ori_s) * Constant.N0
                tik_instance.data_move(out_gm[out_offset:], c_ub[src_offset:], 0, self.matmul_n_l0, self.ori_s,
                                       matmul_m_l0 * Constant.M0 - self.ori_s,
                                       self.pad_s - self.ori_s)
                out_offset += self.ori_s * Constant.N0
                tik_instance.data_move(out_gm[out_offset:], zero_ub, 0, 1, zero_num, 0, 0)
        with tik_instance.if_scope(cur_tail > 0):
            # (...,ori_s,cur_tail) -> (...,ori_s,zero,cur_tail)
            m_idx = real_m_idx + pre_tail + ori_s_num * self.ori_s
            out_offset = m_idx % self.ori_s * Constant.N0 + \
                         matmul_n_idx * self.pad_s * Constant.N0 + \
                         m_idx // self.ori_s * self.n1_shape * self.pad_s * Constant.N0
            src_offset = (pre_tail + ori_s_num * self.ori_s) * Constant.N0
            tik_instance.data_move(out_gm[out_offset:], c_ub[src_offset:], 0, self.matmul_n_l0, cur_tail,
                                   matmul_m_l0 * Constant.M0 - cur_tail,
                                   self.pad_s - cur_tail)

    def _matmul_compute(self, tik_instance, matmul_params, matmul_tensor_list):
        '''
        matmul qkv compute
        '''
        blk_n_idx, matmul_m_idx, special_core = matmul_params
        matmul_m_l0 = self.matmul_m_l0
        if special_core:
            matmul_m_l0 = self.matmul_m_l0_last
        l0c, kernel_gm, out_gm, bias_gm, zero_ub = matmul_tensor_list
        with tik_instance.for_range(0, self.n_single_core) as n_single_core_idx:
            matmul_n_idx = (blk_n_idx * self.n_single_core + n_single_core_idx) * self.matmul_n_l1
            if self.unaligned_n_flag and self.trans_b:
                _, h_outer, h_ori = self._get_unaligned_h_info()
                matmul_n_idx = (blk_n_idx * self.n_single_core + n_single_core_idx) * h_outer * h_ori
            if self.bias_flag:
                cast_bias_ub = self._get_bias(tik_instance, bias_gm, matmul_n_idx)
                with tik_instance.for_range(0, matmul_m_l0) as brc_idx:
                    tik_instance.broadcast_ub_to_l0c(l0c[brc_idx * Constant.FRAC_SIZE:], cast_bias_ub,
                                                     self.matmul_n_l0, 1, 0, matmul_m_l0 - 1)
            with tik_instance.for_range(0, self.k1_shape // self.matmul_k_l0 // Constant.DOUBLE_BUFFER) as kl1_idx:
                bl1_src_offset = self._get_bl1_offset([blk_n_idx, n_single_core_idx, matmul_n_idx, kl1_idx], 0, 0)
                # ping
                self._matmul_l0c_compute(tik_instance, kernel_gm, l0c, [bl1_src_offset, kl1_idx, 0, special_core])
                # pong
                bl1_src_offset = self._get_bl1_offset([blk_n_idx, n_single_core_idx, matmul_n_idx, kl1_idx],
                                                      bl1_src_offset, 1)
                self._matmul_l0c_compute(tik_instance, kernel_gm, l0c, [bl1_src_offset, kl1_idx, 1, special_core])
            # tensor_mov
            c_ub = tik_instance.Tensor(self.dtype, (self.matmul_n_l0, matmul_m_l0, Constant.M0, Constant.N0),
                                       name="c_ub", scope=tik.scope_ubuf)
            tik_instance.tensor_mov(c_ub, l0c, 'm', 1, self.matmul_n_l0 * matmul_m_l0, 0, 0)
            if self.unaligned_s_flag:
                self._unaligned_m_transpose(tik_instance, 
                                            [matmul_m_idx, matmul_n_idx, special_core], [out_gm, c_ub, zero_ub])
                return
            elif self.unaligned_n_flag and self.trans_b:
                self._unaligned_n_transpose(tik_instance, 
                                            [matmul_m_idx, matmul_n_idx, special_core], [out_gm, c_ub])
                return

            m_inner = self.out_shape[Constant.M_INNER_INDEX]
            out_offset = (matmul_m_idx % m_inner) * Constant.FRAC_SIZE + matmul_n_idx * m_inner * Constant.FRAC_SIZE + \
                         matmul_m_idx // m_inner * self.n1_shape * m_inner * Constant.FRAC_SIZE
            if m_inner % matmul_m_l0 == 0:
                tik_instance.data_move(out_gm[out_offset:], c_ub, 0, self.matmul_n_l0, matmul_m_l0 * Constant.M0,
                                       0, (m_inner - matmul_m_l0) * Constant.M0)
            elif matmul_m_l0 % m_inner == 0:
                with tik_instance.for_range(0, matmul_m_l0 // m_inner) as m_inner_idx:
                    out_offset += m_inner_idx * self.n1_shape * m_inner * Constant.FRAC_SIZE
                    tik_instance.data_move(out_gm[out_offset:], c_ub[m_inner_idx * m_inner * Constant.FRAC_SIZE], 0,
                                           self.matmul_n_l0, m_inner * Constant.M0,
                                           (matmul_m_l0 - m_inner) * Constant.M0, 0)
            else:
                m_out_size = math.gcd(matmul_m_l0, m_inner)
                with tik_instance.for_range(0, matmul_m_l0 // m_out_size) as m_out_idx:
                    real_m_idx = matmul_m_idx + m_out_idx * m_out_size
                    out_offset = (real_m_idx % m_inner) * Constant.FRAC_SIZE + matmul_n_idx * m_inner * \
                                 Constant.FRAC_SIZE + real_m_idx // m_inner * self.n1_shape * m_inner * \
                                 Constant.FRAC_SIZE
                    src_offset = m_out_idx * m_out_size * Constant.FRAC_SIZE
                    tik_instance.data_move(out_gm[out_offset:], c_ub[src_offset:], 0, self.matmul_n_l0,
                                           m_out_size * Constant.M0, (matmul_m_l0 - m_out_size) * Constant.M0,
                                           (m_inner - m_out_size) * Constant.M0)

    def _matmul_l0c_compute(self, tik_instance, kernel_gm, l0c, ping_pong_params):
        '''
        matmul_l0c_compute
        '''
        bl1_src_offset, kl1_factor_idx, ping_pong, special_core = ping_pong_params
        ping_pong_suffix = "ping" if ping_pong == 0 else "pong"
        matmul_m_al1, matmul_m_l0, ln_res_l1 = self.matmul_m_al1, self.matmul_m_l0, self.ln_res_l1
        if special_core:
            matmul_m_al1, matmul_m_l0, ln_res_l1 = self.matmul_m_al1_last, self.matmul_m_l0_last, self.ln_res_l1_last
        # bl1 process
        bl1 = self._get_bl1(tik_instance, ping_pong_suffix, kernel_gm, bl1_src_offset)
        with tik_instance.for_range(0, matmul_m_al1 // matmul_m_l0):
            # al0 process, al0_ping reuse x_l0a manually
            if ping_pong == 0:
                al0 = self.x_l0a
            else:
                al0 = tik_instance.Tensor(self.dtype, (matmul_m_l0, self.matmul_k_l0, Constant.M0, Constant.C0),
                                          name="al0_" + ping_pong_suffix, scope=tik.scope_ca)
            with tik_instance.for_range(0, matmul_m_l0) as mal0_idx:
                al1_offset = (Constant.DOUBLE_BUFFER * kl1_factor_idx + ping_pong) * self.matmul_k_al1 * \
                             matmul_m_al1 * Constant.M0 * Constant.C0 + mal0_idx * Constant.M0 * Constant.C0
                load_2d(self, al0[mal0_idx * self.matmul_k_l0 * Constant.M0 * Constant.C0:],
                        ln_res_l1[al1_offset:], [0, self.matmul_k_l0, matmul_m_l0, 0, False])
            with tik_instance.for_range(0, self.matmul_n_l1 // self.matmul_n_l0):
                # bl0 process
                bl0 = self._get_bl0(tik_instance, bl1, ping_pong_suffix)
                # l0c process
                cond_params = [ping_pong, kl1_factor_idx, self.bias_flag]
                mad_tensors = [al0, bl0, l0c]
                mad_size = [matmul_m_l0, self.matmul_k_l0, self.matmul_n_l0]
                matmul_l0c_process(tik_instance, cond_params, mad_tensors, mad_size)


def _check_shape_and_dtype(x, kernels, outputs):
    '''
    shape and dtype check of attention_ln_qkv
    '''
    kernel_query, kernel_key, kernel_value = kernels
    query_output, key_output, value_output, mean, variance = outputs
    input_x_shape = shape_util.shape_to_list(x.get("shape"))
    input_x_ori_shape = shape_util.shape_to_list(x.get("ori_shape"))
    kernel_query_shape = shape_util.shape_to_list(kernel_query.get("shape"))
    kernel_key_shape = shape_util.shape_to_list(kernel_key.get("shape"))
    kernel_value_shape = shape_util.shape_to_list(kernel_value.get("shape"))
    query_out_shape = shape_util.shape_to_list(query_output.get("shape"))
    key_out_shape = shape_util.shape_to_list(key_output.get("shape"))
    value_out_shape = shape_util.shape_to_list(value_output.get("shape"))
    mean_shape = shape_util.shape_to_list(mean.get("shape"))
    var_shape = shape_util.shape_to_list(variance.get("shape"))
    k1_shape = input_x_shape[0]
    data_type = x.get("dtype")
    check_dtype("attention_ln_qkv", data_type)
    check_equal_shape("attention_ln_qkv", [kernel_query_shape, kernel_key_shape, kernel_value_shape],
                      "kernel_shape is inconsistant for matmul_qkv.")
    check_equal_shape("attention_ln_qkv", [query_out_shape, key_out_shape, value_out_shape],
                      "output_shape is inconsistant for matmul_qkv.")
    # restrict k_shape with L0A_SIZE since layer_norm cube only support load k once
    if k1_shape > tbe_platform.get_soc_spec("L0A_SIZE") // (Constant.C0 * Constant.M0 *
                                                            Constant.DTYPE_SIZE.get(data_type)):
        error_manager_cube.raise_err_specific_user("attention_ln_qkv",
                                                   "k1_shape is too large to load once in layer_norm calculation.")
    if isinstance(mean_shape, list) and len(mean_shape) > 0 and isinstance(var_shape, list) and len(var_shape) > 0:
        check_equal_shape("attention_ln_qkv", [mean_shape[0], input_x_ori_shape[0], var_shape[0]],
                          "invalid mean_out_shape/variance_out_shape.")
    input_x_format = x.get("format").upper()
    kernel_format = kernel_query.get("format").upper()
    check_format("attention_ln_qkv", input_x_format, kernel_format)


@register_operator("AttentionLnQKV")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def attention_ln_qkv(x, kernel_query, kernel_key, kernel_value, gamma, beta, bias_query, bias_key, bias_value,
                     norm, query_output, key_output, value_output, mean, variance, epsilon=1e-7,
                     trans_a=False, trans_b=False, kernel_name="attention_ln_qkv"):
    """
    Parameters
    ----------
    x: dict
        shape and dtype of input x, only support float16
    kernel_query: dict
        shape and dtype of input kernel_query, only support float16
    kernel_key: dict
        shape and dtype of input kernel_key, only support float16
    kernel_value: dict
        shape and dtype of input kernel_value, only support float16
    gamma: dict
        shape and dtype of input gamma, only support float16
    beta: dict
        shape and dtype of input beta, only support float16
    bias_query: dict
        shape and dtype of input bias_query, only support float16
    bias_key: dict
        shape and dtype of input bias_key, only support float16
    bias_value: dict
        shape and dtype of input bias_value, only support float16
    norm: dict
        shape and dtype of output, only support float16
    query_output: dict
        shape and dtype of output, only support float16
    key_output: dict
        shape and dtype of output, only support float16
    value_output: dict
        shape and dtype of output, only support float16
    mean: dict
        shape and dtype of output, only support float16
    variance: dict
        shape and dtype of output, only support float16
    epsilon: float
        Minimum positive number greater than 0
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: bool
        If True, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "attention_ln_qkv"

    Returns
    -------
    None
    """
    input_x_shape = shape_util.shape_to_list(x.get("shape"))
    kernel_shape = shape_util.shape_to_list(kernel_query.get("shape"))
    gamma_shape = shape_util.shape_to_list(gamma.get("shape"))
    beta_shape = shape_util.shape_to_list(beta.get("shape"))
    norm_shape = shape_util.shape_to_list(norm.get("shape"))
    out_shape = shape_util.shape_to_list(query_output.get("shape"))
    kernel_ori_shape = shape_util.shape_to_list(kernel_query.get("ori_shape"))
    # check bias
    if bias_query and bias_key and bias_value:
        bias_flag = True
        bias_shape = shape_util.shape_to_list(bias_query.get("shape"))
        bias_shape_real = (reduce(lambda x, y: x * y, list(bias_shape)),)
    elif not bias_query and not bias_key and not bias_value:
        bias_shape_real = ()
        bias_flag = False
    else:
        error_manager_cube.raise_err_specific_user("attention_ln_qkv",
                                                   "bias_flag is inconsistant for matmul_qkv.")
    if trans_a:
        error_manager_cube.raise_err_specific_user("attention_ln_qkv", "unsupported transpose flag for matmul.")
    kernels = [kernel_query, kernel_key, kernel_value]
    outputs = [query_output, key_output, value_output, mean, variance]
    _check_shape_and_dtype(x, kernels, outputs)
    params = {
        "dtype": x.get("dtype"),
        "input_x_shape": input_x_shape,
        "kernel_shape": kernel_shape,
        "gamma_shape": gamma_shape,
        "beta_shape": beta_shape,
        "out_shape": out_shape,
        "bias_flag": bias_flag,
        "bias_shape": bias_shape_real,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "norm_shape": norm_shape,
        "kernel_ori_shape": kernel_ori_shape,
        "epsilon": epsilon,
        "kernel_name": kernel_name
    }
    obj = AttentionLnQKV(params)
    obj.attention_ln_qkv_compute()
