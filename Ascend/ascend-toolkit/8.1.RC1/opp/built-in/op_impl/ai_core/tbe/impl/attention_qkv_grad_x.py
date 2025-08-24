#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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
attention_qkv_grad_x
"""
from impl.util.attention_qkv_util import Constant
from impl.util.attention_qkv_util import vconv
from impl.util.attention_qkv_util import check_equal_shape
from impl.util.attention_qkv_util import check_dtype
from impl.util.attention_qkv_util import check_format
from impl.util.attention_qkv_util import check_trans_flag
from impl.util.attention_qkv_util import get_unit, get_tiling_special_core
from impl.util.attention_qkv_util import load_2d
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from tbe.common.utils import shape_util


class AttentionQKVGradX:
    '''
    AttentionQKVGradX
    '''

    def __init__(self, params):
        self.ln_dx_required = params.get("ln_dx_shape") and True
        self.dtype = params.get("dtype")
        self.ln_dx_shape = None
        if self.ln_dx_required:
            self.ln_dx_shape = params.get("ln_dx_shape")
        self.dy_input_shape = params.get("dy_input_shape")
        self.kernel_shape = params.get("kernel_shape")
        self.out_shape = params.get("out_shape")
        self.trans_a = params.get("trans_a")
        self.trans_b = params.get("trans_b")
        self.m1_shape = self.dy_input_shape[1]
        self.k1_shape = self.dy_input_shape[0]
        self.n1_shape = self.kernel_shape[1] if self.trans_b else self.kernel_shape[0]
        self.kernel_name = params.get("kernel_name")
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self._init_gm_tensor(self.tik_instance)
        self._tiling_args_compute()

    def attention_qkv_gradx_compute(self):
        '''
        attention_qkv_gradx_compute
        '''
        tik_instance = self.tik_instance
        m_single_core = self.one_core_data // self.matmul_m_l1
        m_single_core_last = self.one_core_data_last // self.matmul_m_l1_last
        n_single_core = self.n1_shape // self.block_n // self.matmul_n_l1
        with tik_instance.for_range(0, self.block_m * self.block_n, block_num=self.block_m * self.block_n) as blk_idx:
            blk_m_idx = blk_idx % self.block_m
            blk_n_idx = blk_idx // self.block_m
            with self.tik_instance.if_scope(tik.any(blk_m_idx < self.common_core_num, self.special_core is False)):
                with tik_instance.for_range(0, m_single_core) as m_single_core_idx:
                    matmul_m_idx = (blk_m_idx * m_single_core + m_single_core_idx) * self.matmul_m_l1
                    with tik_instance.for_range(0, n_single_core) as n_single_core_idx:
                        matmul_n_idx = (blk_n_idx * n_single_core + n_single_core_idx) * self.matmul_n_l1
                        self._single_core_process(tik_instance, matmul_m_idx, matmul_n_idx)
            with self.tik_instance.else_scope():
                with tik_instance.for_range(0, m_single_core_last) as m_single_core_idx:
                    matmul_m_idx = self.common_core_num * m_single_core * self.matmul_m_l1 + \
                                   ((blk_m_idx - self.common_core_num) * m_single_core_last + m_single_core_idx) \
                                   * self.matmul_m_l1_last
                    with tik_instance.for_range(0, n_single_core) as n_single_core_idx:
                        matmul_n_idx = (blk_n_idx * n_single_core + n_single_core_idx) * self.matmul_n_l1
                        self._single_core_process(tik_instance, matmul_m_idx, matmul_n_idx, True)
        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=self.inputs, outputs=self.outputs)

    def _tiling_args_compute(self):
        '''
        tiling args setting
        '''
        self.block_m = Constant.CANDIDATE_TILING_N
        self.block_n = self.core_num // self.block_m if (self.core_num // self.block_m) != 0 else 1

        self.special_core = False
        if self.core_num % self.block_m != 0:
            self.special_core = True
        self.matmul_m_l0 = get_unit(self.m1_shape, self.block_m, Constant.CANDIDATE_TILING_M2)
        if self.special_core:
            self.block_n = 2
            self.block_m = self.core_num // self.block_n
            self.one_core_data, self.common_core_num, self.one_core_data_last, self.last_core_num = \
                get_tiling_special_core(self.m1_shape, self.block_m)
            self.matmul_m_l0 = get_unit(self.one_core_data, 1, Constant.CANDIDATE_TILING_M2)
            self.matmul_m_l0_last = get_unit(self.one_core_data_last, 1, Constant.CANDIDATE_TILING_M2)
        else:
            self.one_core_data, self.common_core_num, self.one_core_data_last, self.last_core_num = \
                get_tiling_special_core(self.m1_shape, self.block_m)
            self.matmul_m_l0 = get_unit(self.one_core_data, 1, Constant.CANDIDATE_TILING_M2)
            self.matmul_m_l0_last = get_unit(self.one_core_data_last, 1, Constant.CANDIDATE_TILING_M2)
        self.matmul_m_l1 = self.matmul_m_l0
        self.matmul_m_l1_last = self.matmul_m_l0_last

        self.matmul_n_l0 = get_unit(self.n1_shape, self.block_n, Constant.CANDIDATE_TILING_N, True)
        self.matmul_n_l1 = self.matmul_n_l0
        # restrict matmul_k_l0 by L0A_SIZE && matmul_m_l0 / L0B_SIZE && matmul_n_l0
        self.matmul_k_l0 = min(
            tbe_platform.get_soc_spec("L0A_SIZE") // self.matmul_m_l0 // Constant.FRAC_SIZE // \
            Constant.DTYPE_SIZE.get(self.dtype) // Constant.DOUBLE_BUFFER,
            tbe_platform.get_soc_spec("L0B_SIZE") // self.matmul_n_l0 // Constant.FRAC_SIZE // \
            Constant.DTYPE_SIZE.get(self.dtype) // Constant.DOUBLE_BUFFER)
        self.matmul_k_al1 = self.matmul_k_l0
        self.matmul_k_bl1 = self.matmul_k_l0

    def _init_gm_tensor(self, tik_instance):
        '''
        init gm tensors
        '''
        self.query_dx_gm = tik_instance.Tensor(self.dtype, self.dy_input_shape, name="query_dx_gm",
                                               scope=tik.scope_gm)
        self.key_dw_gm = tik_instance.Tensor(self.dtype, self.dy_input_shape, name="key_dw_gm",
                                             scope=tik.scope_gm)
        self.value_dw_gm = tik_instance.Tensor(self.dtype, self.dy_input_shape, name="value_dw_gm",
                                               scope=tik.scope_gm)
        self.kernel_query_gm = tik_instance.Tensor(self.dtype, self.kernel_shape, name="kernel_query_gm",
                                                   scope=tik.scope_gm)
        self.kernel_key_gm = tik_instance.Tensor(self.dtype, self.kernel_shape, name="kernel_key_gm",
                                                 scope=tik.scope_gm)
        self.kernel_value_gm = tik_instance.Tensor(self.dtype, self.kernel_shape, name="kernel_value_gm",
                                                   scope=tik.scope_gm)
        self.inputs = [self.query_dx_gm, self.key_dw_gm, self.value_dw_gm,
                       self.kernel_query_gm, self.kernel_key_gm, self.kernel_value_gm]
        if self.ln_dx_required:
            self.ln_dx_gm = tik_instance.Tensor(self.dtype, self.ln_dx_shape, name="ln_input_gm",
                                                scope=tik.scope_gm)
            self.inputs.insert(0, self.ln_dx_gm)

        self.out_gm = tik_instance.Tensor(self.dtype, self.out_shape, name="out_gm",
                                          scope=tik.scope_gm)
        self.outputs = [self.out_gm]

    def _single_core_process(self, tik_instance, matmul_m_idx, matmul_n_idx, special_core=False):
        '''
        single core process
        '''
        matmul_m_l0 = self.matmul_m_l0
        if special_core:
            matmul_m_l0 = self.matmul_m_l0_last
        cl0_shape = (self.matmul_n_l0, matmul_m_l0, Constant.M0, Constant.N0)
        l0c = tik_instance.Tensor(Constant.FP32_DTYPE, cl0_shape, name="l0c", scope=tik.scope_cbuf_out)
        ub_shape = (self.matmul_n_l0, matmul_m_l0, Constant.M0, Constant.N0)
        add_ub = tik_instance.Tensor(self.dtype, ub_shape, name="add_ub", scope=tik.scope_ubuf)
        add_ub_cast = tik_instance.Tensor(Constant.FP32_DTYPE, ub_shape, name="add_ub_cast", scope=tik.scope_ubuf)
        self._add_ub_process(tik_instance, [matmul_m_idx, matmul_n_idx], [add_ub, add_ub_cast, l0c], special_core)
        for i in range(Constant.KERNEL_NUM):
            if self.ln_dx_required:
                gm_tensor_list = [self.inputs[i + 1], self.inputs[i + Constant.KERNEL_NUM + 1], l0c]
            else:
                gm_tensor_list = [self.inputs[i], self.inputs[i + Constant.KERNEL_NUM], l0c]
            with tik_instance.for_range(0, self.k1_shape // self.matmul_k_al1 // \
                                           Constant.DOUBLE_BUFFER) as kl1_factor_idx:
                idx_list = [matmul_m_idx, matmul_n_idx, kl1_factor_idx]
                # l0c_compute witj ping-pong
                for ping_pong in range(Constant.DOUBLE_BUFFER):
                    self._matmul_l0c_compute(tik_instance, gm_tensor_list, idx_list, [ping_pong, special_core])
        c_ub = tik_instance.Tensor(self.dtype, ub_shape, name="c_ub", scope=tik.scope_ubuf)
        tik_instance.tensor_mov(c_ub, l0c, 'm', 1, self.matmul_n_l0 * matmul_m_l0, 0, 0)
        cub_burst_len = matmul_m_l0 * Constant.M0
        dst_stride = (self.m1_shape - matmul_m_l0) * Constant.M0
        out_offset = matmul_m_idx * Constant.FRAC_SIZE + matmul_n_idx * self.m1_shape * Constant.FRAC_SIZE
        tik_instance.data_move(self.out_gm[out_offset:], c_ub, 0, self.matmul_n_l0, cub_burst_len, 0, dst_stride)

    def _add_ub_process(self, tik_instance, idx_list, tensor_list, special_core=False):
        '''
        add_ub process
        '''
        matmul_m_l0 = self.matmul_m_l0
        if special_core:
            matmul_m_l0 = self.matmul_m_l0_last
        matmul_m_idx, matmul_n_idx = idx_list
        add_ub, add_ub_cast, l0c = tensor_list
        ln_src_offset = matmul_m_idx * Constant.FRAC_SIZE + matmul_n_idx * self.m1_shape * Constant.FRAC_SIZE
        if self.ln_dx_required:
            tik_instance.data_move(add_ub, self.ln_dx_gm[ln_src_offset:], 0, self.matmul_n_l0,
                                   matmul_m_l0 * Constant.M0, (self.m1_shape - matmul_m_l0) * Constant.M0, 0)
        else:
            dup_times = matmul_m_l0 * self.matmul_n_l0 * Constant.DTYPE_SIZE.get(self.dtype)
            dup_loops = dup_times // Constant.REPEAT_SIZE_MAX
            for i in range(dup_loops):
                offset = i * Constant.FP16_REPEAT_SIZE * Constant.REPEAT_SIZE_MAX
                tik_instance.vector_dup(Constant.MASK_FP16, add_ub[offset:], 0, Constant.REPEAT_SIZE_MAX, 1, 8)
            offset = dup_loops * Constant.FP16_REPEAT_SIZE * Constant.REPEAT_SIZE_MAX
            dup_times = dup_times - dup_loops * Constant.REPEAT_SIZE_MAX
            tik_instance.vector_dup(Constant.MASK_FP16, add_ub[offset:], 0, dup_times, 1, 8)
        vconv_repeat_size = self.matmul_n_l0 * matmul_m_l0 * Constant.FRAC_SIZE // Constant.FP32_REPEAT_SIZE
        vconv(tik_instance, add_ub, add_ub_cast, vconv_repeat_size, True)
        tik_instance.data_move(l0c, add_ub_cast, 0, 1, self.matmul_n_l0 * matmul_m_l0, 0, 0)

    def _matmul_l0c_compute(self, tik_instance, tensor_list, idx_list, ping_pong_params):
        '''
        matmul_l0c_compute
        '''
        ping_pong, special_core = ping_pong_params
        matmul_m_l0, matmul_m_l1 = self.matmul_m_l0, self.matmul_m_l1
        if special_core:
            matmul_m_l0, matmul_m_l1 = self.matmul_m_l0_last, self.matmul_m_l1_last
        al1_shape = (self.matmul_k_al1, matmul_m_l1, Constant.M0, Constant.C0)
        bl1_shape = (self.matmul_k_bl1, self.matmul_n_l1, Constant.N0, Constant.C0)
        al0_shape = (matmul_m_l0, self.matmul_k_l0, Constant.M0, Constant.C0)
        bl0_shape = (self.matmul_k_l0, self.matmul_n_l0, Constant.N0, Constant.C0)
        matmul_m_idx, matmul_n_idx, kl1_factor_idx = idx_list
        x_gm, kernel_gm, l0c = tensor_list
        ping_pong_suffix = "ping" if ping_pong == 0 else "pong"
        # al1_process
        al1 = tik_instance.Tensor(self.dtype, al1_shape, name="al1_" + ping_pong_suffix, scope=tik.scope_cbuf)
        a_src_offset = (Constant.DOUBLE_BUFFER * kl1_factor_idx + ping_pong) * self.matmul_k_al1 * self.m1_shape * \
                       Constant.FRAC_SIZE + matmul_m_idx * Constant.FRAC_SIZE
        tik_instance.data_move(al1, x_gm[a_src_offset:], 0, self.matmul_k_al1, matmul_m_l1 * Constant.M0,
                               (self.m1_shape - matmul_m_l1) * Constant.M0, 0)
        # bl1_process
        bl1 = tik_instance.Tensor(self.dtype, bl1_shape, name="bl1_" + ping_pong_suffix, scope=tik.scope_cbuf)
        if self.trans_b:
            b_src_offset = (Constant.DOUBLE_BUFFER * kl1_factor_idx + ping_pong) * self.matmul_k_bl1 * self.n1_shape * \
                           Constant.FRAC_SIZE + matmul_n_idx * Constant.FRAC_SIZE
            b_nburst = self.matmul_k_bl1
            b_burst = self.matmul_n_l1 * Constant.N0
            b_src_stride = (self.n1_shape - self.matmul_n_l1) * Constant.N0
        else:
            b_src_offset = (Constant.DOUBLE_BUFFER * kl1_factor_idx + ping_pong) * self.matmul_k_bl1 * \
                           Constant.FRAC_SIZE + matmul_n_idx * Constant.FRAC_SIZE * self.k1_shape
            b_nburst = self.matmul_n_l1
            b_burst = self.matmul_k_bl1 * Constant.N0
            b_src_stride = (self.k1_shape - self.matmul_k_bl1) * Constant.N0
        tik_instance.data_move(bl1, kernel_gm[b_src_offset:], 0, b_nburst,
                               b_burst, b_src_stride, 0)
        with tik_instance.for_range(0, matmul_m_l1 // matmul_m_l0):
            al0 = tik_instance.Tensor(self.dtype, al0_shape, name="al0_" + ping_pong_suffix, scope=tik.scope_ca)
            # al0 process
            with tik_instance.for_range(0, matmul_m_l0) as mal0_idx:
                al0_offset = mal0_idx * self.matmul_k_l0 * Constant.M0 * Constant.C0
                load_2d(self, al0[al0_offset:], al1[mal0_idx * Constant.FRAC_SIZE:], [0, self.matmul_k_l0,
                                                                                      matmul_m_l0, 0, False])
            # bl0 process
            bl0 = tik_instance.Tensor(self.dtype, bl0_shape, name="bl0_" + ping_pong_suffix, scope=tik.scope_cb)
            with tik_instance.for_range(0, self.matmul_n_l1 // self.matmul_n_l0):
                if self.trans_b:
                    load_2d(self, bl0, bl1, [0, self.matmul_n_l0 * self.matmul_k_l0, 1, 0, False])
                else:
                    with tik_instance.for_range(0, self.matmul_k_l0) as kl0_idx:
                        load_2d(self, bl0[kl0_idx * self.matmul_n_l0 * Constant.N0 * Constant.C0:],
                                bl1[kl0_idx * Constant.C0 * Constant.N0:],
                                [0, self.matmul_n_l0, self.matmul_k_l0, 0, True])
                tik_instance.mmad(l0c, al0, bl0, matmul_m_l0 * Constant.M0, self.matmul_k_l0 * Constant.C0,
                                  self.matmul_n_l0 * Constant.N0, 1)


@register_operator("AttentionQKVGradX")
@para_check.check_op_params(para_check.OPTION_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def attention_qkv_grad_x(ln_dx, query_dx, key_dw, value_dw, kernel_query, kernel_key, kernel_value,
                         dx, trans_a=False, trans_b=True, kernel_name="attention_qkv_grad_x"):
    """
    Parameters
    ----------
    ln_dx: dict
        shape and dtype of input ln_dx, only support float16
    query_dx: dict
        shape and dtype of input query_dx, only support float16
    key_dw: dict
        shape and dtype of input key_dw, only support float16
    value_dw: dict
        shape and dtype of input value_dw, only support float16
    kernel_query: dict
        shape and dtype of input kernel_query, only support float16
    kernel_key: dict
        shape and dtype of input kernel_key, only support float16
    kernel_value: dict
        shape and dtype of input kernel_value, only support float16
    dx: dict
        shape and dtype of output dx, only support float16
    trans_a: bool
        If True, shape_a == transposed before multiplication, default to be false
    trans_b: bool
        If True, the shape in input_x2 must be transposed before multiplication, default to be true
    kernel_name: str
        cce kernel name, default value is "attention_qkv_grad_x"

    Returns
    -------
    None
    """
    query_dx_shape = shape_util.shape_to_list(query_dx.get("shape"))
    key_dw_shape = shape_util.shape_to_list(key_dw.get("shape"))
    value_dw_shape = shape_util.shape_to_list(value_dw.get("shape"))
    kernel_query_shape = shape_util.shape_to_list(kernel_query.get("shape"))
    kernel_key_shape = shape_util.shape_to_list(kernel_key.get("shape"))
    kernel_value_shape = shape_util.shape_to_list(kernel_value.get("shape"))
    out_shape = shape_util.shape_to_list(dx.get("shape"))
    data_type = query_dx.get("dtype")
    check_dtype("attention_qkv_grad_x", data_type)
    input_x_format = query_dx.get("format").upper()
    kernel_format = kernel_query.get("format").upper()
    check_format("attention_qkv_grad_x", input_x_format, kernel_format)
    check_equal_shape("attention_qkv_grad_x", [query_dx_shape, key_dw_shape, value_dw_shape],
                      "matmul_dx_qkv input_shape should be equal.")
    check_equal_shape("attention_qkv_grad_x", [kernel_query_shape, kernel_key_shape, kernel_value_shape],
                      "matmul_dx_qkv kernel_shape should be equal.")
    params = {
        "dtype": query_dx.get("dtype"),
        "dy_input_shape": query_dx_shape,
        "kernel_shape": kernel_query_shape,
        "out_shape": out_shape,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "kernel_name": kernel_name
    }
    ln_dx_required = ln_dx and True
    if ln_dx_required:
        ln_dx_shape = shape_util.shape_to_list(ln_dx.get("shape"))
        check_equal_shape("attention_qkv_grad_x", [ln_dx_shape, out_shape, out_shape],
                          "ln_dx_shape should be equal with out_shape.")
        params["ln_dx_shape"] = ln_dx_shape
        check_equal_shape("attention_qkv_grad_x", [ln_dx_shape, out_shape, out_shape],
                          "ln_dx_shape should be equal with out_shape.")
    obj = AttentionQKVGradX(params)
    obj.attention_qkv_gradx_compute()
