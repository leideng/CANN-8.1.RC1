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
max_pool_grad_with_argmax_v2_resnet50
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl import common_util_v1
from impl import constant_util_v1 as constant


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """

    def __init__(self):
        pass

    # size of vector calc one repeat
    ONE_REPEAT = 256
    # max repeat of vector calc
    V_MAX_REPEAT = 255
    # C0
    C_ZERO = 16
    # ksize for ResNet50
    RESNET50_KSIZE = 3
    # stride for ResNet50
    RESNET50_STRIDE = 2
    # pad for ResNet50
    RESNET50_PAD = 1
    # dilation for ResNet50
    RESNET50_DILATION = 1
    # grad shape for ResNet50
    RESNET50_IN_GRAD = 56
    # output shape for ResNet50
    RESNET50_OUT = 112


# 'pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals,unused-variable
# 'pylint: disable=missing-function-docstring,too-many-statements,invalid-name,too-many-boolean-expressions

class MaxpoolGradV2Resnet50():
    """
    parameter for max_pool_grad_with_pool
    """

    def __init__(self, dtype, data_input_origin, data_input, data_mask, data_output, tik_instance):
        """
        :param dtype:
        :param data_input_origin:
        :param data_input:
        :param data_mask:
        :param data_output:
        :param tik_instance:
        """
        self.blocknum = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

        self.dtype = dtype
        self.tik_instance = tik_instance
        self.ksize_h = Constant.RESNET50_KSIZE
        self.ksize_w = Constant.RESNET50_KSIZE
        self.stride_h = Constant.RESNET50_STRIDE
        self.stride_w = Constant.RESNET50_STRIDE
        self.pad_t = Constant.RESNET50_PAD
        self.pad_b = Constant.RESNET50_PAD
        self.pad_l = Constant.RESNET50_PAD
        self.pad_r = Constant.RESNET50_PAD
        self.ceil_mode = False
        self.dilation = Constant.RESNET50_DILATION

        self.grad_h = Constant.RESNET50_IN_GRAD
        self.grad_w = Constant.RESNET50_IN_GRAD
        self.out_h = Constant.RESNET50_OUT
        self.out_w = Constant.RESNET50_OUT
        self.c0 = Constant.C_ZERO

        self.data_input_origin = data_input_origin
        self.data_input = data_input
        self.data_mask = data_mask
        self.data_output = data_output
        self.nc1_idx = None
        self.kernel_size = None
        self.alg_output_w = None
        self.block_h = None
        self.loop_h = None
        self.input_block_size = None
        self.output_block_size = None
        self.output_block_line = None
        self.output_block_algn_len = None
        self.mask_size = None
        self.dxh_address_offset = None
        self.grad_address_offset = None
        self.dxh_address_res = None
        self.mask_one_window = None
        self.fm_f32_tail_ub_size = None
        self.ub_grad_buf0 = None
        self.ub_grad_buf1 = None
        self.ub_select_fp16_buf = None
        self.maxpool_ub_input_buf0 = None
        self.maxpool_ub_input_buf1 = None
        self.ub_loc_mask_buf0 = None
        self.ub_loc_mask_buf1 = None
        self.ub_zero_buf = None
        self.ub_select_fp32_buf = None
        self.ub_fm_fp32_buf = None
        self.ub_fm_fp32_tail_buf = None

    def _variable_init(self):
        """
        init scalars and variables
        """
        self.nc1_idx = self.tik_instance.Scalar("int32", name="nc1_idx")
        self.kernel_size = self.ksize_h * self.ksize_w
        self.alg_output_w = ((self.out_w + self.pad_l + self.pad_r + 3) // 4) * 4
        self.block_h = 2
        self.loop_h = self.grad_h // self.block_h
        self.input_block_size = self.block_h * self.grad_w * self.c0
        self.output_block_size = self.block_h * 2 * self.out_w * self.c0  # 8
        self.output_block_line = self.block_h * 2 * self.alg_output_w * self.c0  # 8
        self.output_block_algn_len = (self.block_h * 2 + self.pad_t) * self.alg_output_w * self.c0
        self.mask_size = self.input_block_size // self.c0
        self.dxh_address_offset = (self.pad_t * self.alg_output_w + self.pad_l) * self.c0
        self.grad_address_offset = self.pad_l * self.c0
        self.dxh_address_res = self.pad_t * self.out_w * self.c0
        self.mask_one_window = ((self.grad_h * self.grad_w + 15) // self.c0 + 1) * self.c0
        self.fm_f32_tail_ub_size = ((self.alg_output_w * self.c0 + 127) // constant.MASK128) * constant.MASK128

    def _tensor_init(self):
        """
        init tensor
        """
        grad_ub_size = (4 * 2 + self.pad_t + 1) * self.alg_output_w * self.c0  # 9 x 112 x 16
        self.ub_grad_buf0 = self.tik_instance.Tensor(self.dtype, (grad_ub_size,),
                                                     name="ub_grad_buf0",
                                                     scope=tik.scope_ubuf)
        self.ub_grad_buf1 = self.tik_instance.Tensor(self.dtype, (grad_ub_size,),
                                                     name="ub_grad_buf1",
                                                     scope=tik.scope_ubuf)

        select_ub_size = self.input_block_size  # 4 x 56 x 16
        self.ub_select_fp16_buf = self.tik_instance.Tensor(self.dtype, (select_ub_size,),
                                                           name="ub_select_fp16_buf",
                                                           scope=tik.scope_ubuf)

        max_pool_ub_size = self.input_block_size  # 4 x 56 x 16
        self.maxpool_ub_input_buf0 = self.tik_instance.Tensor(self.dtype,
                                                              (max_pool_ub_size,),
                                                              name="maxpool_ub0",
                                                              scope=tik.scope_ubuf)
        self.maxpool_ub_input_buf1 = self.tik_instance.Tensor(self.dtype,
                                                              (max_pool_ub_size,),
                                                              name="maxpool_ub1",
                                                              scope=tik.scope_ubuf)

        mask_ub_size = 9 * 4 * 56 * self.c0 // self.c0  # 9 x 4 x 56
        self.ub_loc_mask_buf0 = self.tik_instance.Tensor("uint16", (mask_ub_size,),
                                                         name="ub_loc_mask_buf0",
                                                         scope=tik.scope_ubuf)
        self.ub_loc_mask_buf1 = self.tik_instance.Tensor("uint16", (mask_ub_size,),
                                                         name="ub_loc_mask_buf1",
                                                         scope=tik.scope_ubuf)

        self.ub_zero_buf = self.tik_instance.Tensor(self.dtype, (constant.MASK128,),
                                                    name="ub_zero_buf",
                                                    scope=tik.scope_ubuf)

        self.ub_select_fp32_buf = self.tik_instance.Tensor("float32",
                                                           (select_ub_size,),
                                                           name="ub_select_fp32_buf",
                                                           scope=tik.scope_ubuf)

        fm_f32_ub_size = (4 * 2 + self.pad_t + 1) * self.alg_output_w * self.c0
        self.ub_fm_fp32_buf = self.tik_instance.Tensor("float32", (fm_f32_ub_size,),
                                                       name="ub_fm_fp32_buf",
                                                       scope=tik.scope_ubuf)

        self.ub_fm_fp32_tail_buf = self.tik_instance.Tensor("float32",
                                                            (self.fm_f32_tail_ub_size,),
                                                            name="ub_fm_fp32_tail",
                                                            scope=tik.scope_ubuf)

        self.tik_instance.data_move(self.ub_zero_buf[0],
                                    self.data_input_origin[0],
                                    constant.SID,
                                    constant.DEFAULT_NBURST,
                                    constant.DEFAULT_BURST_LEN,
                                    constant.STRIDE_ZERO,
                                    constant.STRIDE_ZERO)
        self.tik_instance.vector_dup(constant.MASK128, self.ub_zero_buf, 0.0, 1, 1, 8)

    def _clean_fp32_multi_repeat(self, data_vmul_ub_col2img_fp32, dtype_size):
        """
        The fun just for clean ub
        """
        v_rep_clear_time = data_vmul_ub_col2img_fp32.shape[0] * dtype_size // Constant.ONE_REPEAT
        v_rep_clear_cycle = v_rep_clear_time // Constant.V_MAX_REPEAT
        v_rep_clear_last = v_rep_clear_time % Constant.V_MAX_REPEAT
        data_clean_scalar = self.tik_instance.Scalar("float32")
        data_clean_scalar.set_as(0)
        if v_rep_clear_cycle > 0:
            with self.tik_instance.for_range(0, v_rep_clear_cycle, thread_num=1) as cycle:
                self.tik_instance.vector_dup(constant.MASK64,
                                             data_vmul_ub_col2img_fp32[cycle * Constant.V_MAX_REPEAT
                                                                       * constant.MASK64],
                                             data_clean_scalar,
                                             Constant.V_MAX_REPEAT,
                                             constant.STRIDE_ONE,
                                             constant.REPEAT_STRIDE_EIGHT)
        if v_rep_clear_last != 0:
            self.tik_instance.vector_dup(constant.MASK64,
                                         data_vmul_ub_col2img_fp32[v_rep_clear_cycle *
                                                                   Constant.V_MAX_REPEAT * constant.MASK64],
                                         data_clean_scalar, v_rep_clear_last,
                                         constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)

    def _vconv_fp32(self, data_grad_ub, ub_fm_fp32_buf, align_len):
        """
        function for convert fp16 ub buffer to fp32
        """
        repeat_time = align_len // constant.MASK64
        if repeat_time > Constant.V_MAX_REPEAT:
            res_repeat_time = repeat_time - Constant.V_MAX_REPEAT
            length = Constant.V_MAX_REPEAT * constant.MASK64
            self.tik_instance.vconv(constant.MASK64, "", data_grad_ub,
                                    ub_fm_fp32_buf,
                                    Constant.V_MAX_REPEAT,
                                    1, 1, 4, 8)
            self.tik_instance.vconv(constant.MASK64, "", data_grad_ub[length],
                                    ub_fm_fp32_buf[length],
                                    res_repeat_time,
                                    1, 1, 4, 8)
        else:
            self.tik_instance.vconv(constant.MASK64, "", data_grad_ub,
                                    ub_fm_fp32_buf,
                                    repeat_time,
                                    1, 1, 4, 8)

    def _caculate_kernel(self, data_max_pool_input_ub, data_mask_ub, data_grad_ub):
        """
        :param data_max_pool_input_ub:
        :param data_mask_ub:
        :param data_grad_ub:
        :return:
        """
        with self.tik_instance.for_range(0, self.kernel_size) as flt_idx:  # 9
            with self.tik_instance.for_range(0, self.input_block_size // constant.MASK128) as r_idx:
                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                    data_mask_ub[flt_idx * self.mask_size + r_idx * 8])
                self.tik_instance.vsel(constant.MASK128, 0,
                                       self.ub_select_fp16_buf[r_idx * constant.MASK128],
                                       cmpmask,
                                       data_max_pool_input_ub[r_idx * constant.MASK128],
                                       self.ub_zero_buf[0],
                                       1, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vconv(constant.MASK64, "",
                                    self.ub_select_fp32_buf,
                                    self.ub_select_fp16_buf,
                                    self.input_block_size // constant.MASK64, 1, 1, 8, 4)

            with self.tik_instance.for_range(0, self.block_h) as h_idx:
                fm_ub_idx = flt_idx // 3 * self.alg_output_w * self.c0 + \
                            flt_idx % 3 * self.c0 + \
                            self.alg_output_w * self.c0 * 2 * h_idx
                select_ub_idx = self.grad_w * self.c0 * h_idx
                self.tik_instance.vadd(constant.MASK64,
                                       self.ub_fm_fp32_buf[fm_ub_idx],
                                       self.ub_fm_fp32_buf[fm_ub_idx],
                                       self.ub_select_fp32_buf[select_ub_idx],
                                       7, self.stride_w * 2, self.stride_w * 2, 2,
                                       self.stride_w * self.c0, self.stride_w * self.c0, self.c0)
                self.tik_instance.vadd(constant.MASK64,
                                       self.ub_fm_fp32_buf[fm_ub_idx + 8],
                                       self.ub_fm_fp32_buf[fm_ub_idx + 8],
                                       self.ub_select_fp32_buf[select_ub_idx + 8],
                                       7, self.stride_w * 2, self.stride_w * 2, 2,
                                       self.stride_w * self.c0, self.stride_w * self.c0, self.c0)
        self._vconv_fp32(data_grad_ub, self.ub_fm_fp32_buf, self.output_block_algn_len)

    def maxpool_resnet50_ping(self, loop_h_idx, nc1_idx):
        """
        :param loop_h_idx:
        :param nc1_idx:
        :return:
        """
        data_max_pool_input_ub = self.maxpool_ub_input_buf0
        data_mask_ub = self.ub_loc_mask_buf0
        data_grad_ub = self.ub_grad_buf0

        self.tik_instance.data_move(data_max_pool_input_ub[0],
                                    self.data_input[nc1_idx * self.grad_h * self.grad_w * self.c0 +
                                                    loop_h_idx * 2 * self.input_block_size],
                                    0, 1, self.input_block_size // self.c0, 0, 0)
        self.tik_instance.data_move(data_mask_ub[0],
                                    self.data_mask[nc1_idx * self.mask_one_window * self.kernel_size +
                                                   loop_h_idx * 2 * self.mask_size],
                                    0, self.kernel_size, self.input_block_size // (8 * 32),
                                    (self.mask_one_window * self.c0 - self.input_block_size) // (8 * 32), 0)

        self._clean_fp32_multi_repeat(self.ub_fm_fp32_buf, 4)

        with self.tik_instance.if_scope(loop_h_idx > 0):
            self.tik_instance.vmuls(constant.MASK64,
                                    self.ub_fm_fp32_buf[0],
                                    self.ub_fm_fp32_tail_buf[0], 1.0, self.fm_f32_tail_ub_size // constant.MASK128,
                                    2, 2, self.c0, self.c0)
            self.tik_instance.vmuls(constant.MASK64,
                                    self.ub_fm_fp32_buf[8],
                                    self.ub_fm_fp32_tail_buf[8], 1.0, self.fm_f32_tail_ub_size // constant.MASK128,
                                    2, 2, self.c0, self.c0)

        self._caculate_kernel(data_max_pool_input_ub, data_mask_ub, data_grad_ub)

        data_output_idx = nc1_idx * self.out_h * self.out_w * self.c0 + self.output_block_size * loop_h_idx * 2
        with self.tik_instance.if_scope(loop_h_idx == 0):
            with self.tik_instance.for_range(0, self.block_h * 2) as loop_i:
                self.tik_instance.data_move(self.data_output[data_output_idx + loop_i * self.out_w * self.c0],
                                            data_grad_ub[self.dxh_address_offset + loop_i *
                                                         self.alg_output_w * self.c0],
                                            0, 1, self.out_w, 0, 0)
        with self.tik_instance.else_scope():
            data_output_idx = data_output_idx - self.dxh_address_res
            with self.tik_instance.for_range(0, self.block_h * 2 + 1) as loop_i:
                self.tik_instance.data_move(self.data_output[data_output_idx + loop_i * self.out_w * self.c0],
                                            data_grad_ub[self.grad_address_offset + loop_i *
                                                         self.alg_output_w * self.c0],
                                            0, 1, self.out_w, 0, 0)

        self.tik_instance.vmuls(constant.MASK64,
                                self.ub_fm_fp32_tail_buf,
                                self.ub_fm_fp32_buf[self.output_block_line],
                                1.0, self.fm_f32_tail_ub_size // constant.MASK128, 2, 2, self.c0, self.c0)
        self.tik_instance.vmuls(constant.MASK64,
                                self.ub_fm_fp32_tail_buf[8],
                                self.ub_fm_fp32_buf[self.output_block_line + 8],
                                1.0, self.fm_f32_tail_ub_size // constant.MASK128, 2, 2, self.c0, self.c0)

    def maxpool_resnet50_pong(self, loop_h_idx, nc1_idx):
        """
        :param loop_h_idx:
        :param nc1_idx:
        :return:
        """
        data_max_pool_input_ub = self.maxpool_ub_input_buf1
        data_mask_ub = self.ub_loc_mask_buf1
        data_grad_ub = self.ub_grad_buf1

        self.tik_instance.data_move(data_max_pool_input_ub[0],
                                    self.data_input[nc1_idx * self.grad_h * self.grad_w * self.c0 +
                                                    (loop_h_idx * 2 + 1) * self.input_block_size],
                                    0, 1, self.input_block_size // self.c0, 0, 0)
        self.tik_instance.data_move(data_mask_ub[0],
                                    self.data_mask[nc1_idx * self.mask_one_window * self.kernel_size +
                                                   (loop_h_idx * 2 + 1) * self.mask_size],
                                    0, self.kernel_size, self.input_block_size // (8 * 32),
                                    (self.mask_one_window * self.c0 - self.input_block_size) // (8 * 32), 0)

        self._clean_fp32_multi_repeat(self.ub_fm_fp32_buf, 4)

        self.tik_instance.vmuls(constant.MASK64, self.ub_fm_fp32_buf[0],
                                self.ub_fm_fp32_tail_buf[0], 1.0, self.fm_f32_tail_ub_size // constant.MASK128,
                                2, 2, self.c0, self.c0)
        self.tik_instance.vmuls(constant.MASK64, self.ub_fm_fp32_buf[8],
                                self.ub_fm_fp32_tail_buf[8], 1.0, self.fm_f32_tail_ub_size // constant.MASK128,
                                2, 2, self.c0, self.c0)

        self._caculate_kernel(data_max_pool_input_ub, data_mask_ub, data_grad_ub)

        data_output_idx = nc1_idx * self.out_h * self.out_w * self.c0 + self.output_block_size * (loop_h_idx * 2 + 1)
        data_output_idx = data_output_idx - self.dxh_address_res
        with self.tik_instance.for_range(0, self.block_h * 2 + 1) as loop_i:
            self.tik_instance.data_move(self.data_output[data_output_idx + loop_i * self.out_w * self.c0],
                                        data_grad_ub[
                                            self.grad_address_offset + loop_i * self.alg_output_w * self.c0],
                                        0, 1, self.out_w, 0, 0)

        with self.tik_instance.if_scope(loop_h_idx < self.loop_h // 2 - 1):
            self.tik_instance.vmuls(constant.MASK64,
                                    self.ub_fm_fp32_tail_buf,
                                    self.ub_fm_fp32_buf[self.output_block_line], 1.0,
                                    self.fm_f32_tail_ub_size // constant.MASK128, 2, 2, self.c0, self.c0)
            self.tik_instance.vmuls(constant.MASK64,
                                    self.ub_fm_fp32_tail_buf[8],
                                    self.ub_fm_fp32_buf[self.output_block_line + 8], 1.0,
                                    self.fm_f32_tail_ub_size // constant.MASK128, 2, 2, self.c0, self.c0)

    def maxpool_resnet50(self, core_idx, loop_num, one_core_loop):
        """
        function for max_pool_grad_with_pool calc for normal shape
        """
        self._variable_init()
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.nc1_idx.set_as(core_idx * one_core_loop + loop_idx)
            self._tensor_init()
            with self.tik_instance.for_range(0, self.loop_h // 2) as loop_h_idx:
                # ping
                self.maxpool_resnet50_ping(loop_h_idx, self.nc1_idx)
                # pong
                self.maxpool_resnet50_pong(loop_h_idx, self.nc1_idx)
