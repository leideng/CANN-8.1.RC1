#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
fused_bn2_reluv2_conv2d_bn1
"""

import tbe
from tbe import tik
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils.errormgr import error_manager_cube as err_man


class Bn2ReluV2Conv2dBn1NoRedundant:
    """
    class of Bn2ReluV2Conv2dBn1NoRedundant
    """

    def __init__(self, fmap_ori_shape, filters_ori_shape, padding, stride, dilation, groups, factor, epsilon, tiling,
                 kernel_name="bn2reluv2conv2dbn1noredundant"):
        self.kernel_name = kernel_name
        self.tiling = tiling
        soc_version = get_soc_spec("SHORT_SOC_VERSION")
        status = tbe.common.platform.set_current_compile_soc_info(soc_version)
        if status != "success":
            err_man.raise_err_specific("fused_bn2_reluv2_conv2d_bn1", "set soc_version failed, please check!!!")
        self.tik_instance = tik.Tik()
        self.fmap_ori_shape = fmap_ori_shape
        self.filters_ori_shape = filters_ori_shape

        self.factor = factor
        self.epsilon = epsilon
        self.stride_h, self.stride_w = stride
        self.pad_t, self.pad_b, self.pad_l, self.pad_r = padding
        self.dilation_h, self.dilation_w = dilation
        self.groups = groups

        self.batch, self.channel_in, self.height_in, self.width_in = fmap_ori_shape

        self.channel_out, _, self.kernel_h, self.kernel_w = filters_ori_shape
        self.height_out = (self.height_in + self.pad_t + self.pad_b - self.kernel_h) // self.stride_h + 1
        self.width_out = (self.width_in + self.pad_l + self.pad_r - self.kernel_w) // self.stride_w + 1

        self.channel_in0 = 16
        self.channel_out0 = 16
        self.hw_out0 = 16
        self.channel_in1 = self.int_ceil_div(self.channel_in, self.channel_in0)
        self.channel_out1 = self.int_ceil_div(self.channel_out, self.channel_out0)
        self.hw_out1 = self.int_ceil_div(self.height_out * self.width_out, self.hw_out0)

        self.reduce_k0 = self.channel_in0
        self.reduce_k1 = self.channel_in1 * self.kernel_h * self.kernel_w

        self._parse_tiling_params(tiling)
        self.n_redundant = self.tik_instance.Scalar("int64", name="n_redundant", init_value=0)

        self.fmap_5hd_shape = [self.batch, self.channel_in1, self.height_in, self.width_in, self.channel_in0]
        self.filters_5hd_shape = [self.reduce_k1, self.channel_out1, self.channel_out0, self.reduce_k0]
        self.channel_wise_shape_in = [1, self.channel_in1, 1, 1, self.channel_in0]
        self.reluv2_mask_shape = [self.batch, self.channel_in1, self.height_in, self.width_in]
        self.channel_wise_shape_out = [1, self.channel_out1, 1, 1, self.channel_out0]
        self.output_5hd_shape = [self.batch, self.channel_out1, self.height_out * self.width_out, self.channel_out0]
        self.filters_ddr_size = self.reduce_k1 * self.channel_out1 * self.channel_out0 * self.reduce_k0 * 2
        self.set_orign_input_tensor()
        self.set_final_output_tensors()
        self._init_scalar_params()

        """
        init params with none
        """
        self.fmap_l0c_pong = None
        self.ub_fm1_workbuf = None
        self.ub_fm1_workbuf2 = None
        self.fmap_f16_ub_pong = None
        self.input_x_fp16_ub_pong = None
        self.tem_ub = None
        self.tem_ub_pong = None
        self.scale_ub_64 = None
        self.offset_ub_64 = None
        self.input_x_mask_ub_pong = None
        self.fmap_f16_ub_ping = None
        self.input_x_fp16_ub = None
        self.square_mul_ub = None
        self.last_sum_output_ub = None
        self.last_square_sum_output_ub = None
        self.zeros_const_ub = None
        self.zeros_const_ub_16 = None
        self.input_x_fp32_ub = None
        self.save_ub = None
        self.save_ub_pong = None
        self.y_scale_ub = None
        self.y_offset_ub = None
        self.y_offset_ub_start_addr = None
        self.sum_input_ub = None
        self.square_sum_input_ub = None
        self.mean_ub = None
        self.scale_ub = None
        self.square_sum_div_ub = None
        self.square_mean_ub = None
        self.varience_ub = None
        self.y_scale_add_ub = None
        self.y_scale_sqrt_ub = None
        self.offset_ub = None
        self.y_offset_mul_ub = None
        self.pre_moving_mean_ub = None
        self.mean_mul_ub = None
        self.pre_moving_variance_ub = None
        self.fmap_l1 = None
        self.filters_l1 = None
        self.filters_l1_size = None
        self.fmap_l0a_ping = None
        self.filters_l0b_ping = None
        self.fmap_l0c_ping = None
        self.fmap_l0a_pong = None
        self.filters_l0b_pong = None

        """
        init params with none
        """
        self.mean_mul_rev_ub = None
        self.moving_mean_ub = None
        self.variance_batch_ub = None
        self.variance_mul_ub = None
        self.variance_mul_rev_ub = None
        self.moving_variance_ub = None
        self.input_x_mask_ub = None
        self.input_x_ub = None
        self.fmap_ub = None
        self.output_sum_inub = None
        self.output_square_sum_inub = None

    @staticmethod
    def get_gcd(a_num, b_num):
        """
        great common divisor， 获取最大公约数
        """
        if a_num < b_num:
            a_num, b_num = b_num, a_num
        for i in range(b_num, 0, -1):
            if (a_num % i == 0) and (b_num % i == 0):
                return i
        return 1

    @staticmethod
    def int_ceil_div(num_a, num_b):
        """
        upper division
        """
        if num_b == 0:
            err_man.raise_err_specific("fused_bn2_reluv2_conv2d_bn1", "division by zero")
        return (num_a + num_b - 1) // num_b

    @staticmethod
    def calu_loop_time(square_size):
        """
        calu loop time
        """
        fp32_loop_times = square_size // (255 * 64)
        fp32_loop_redundant = square_size % (255 * 64)
        fp32_repeat_times = fp32_loop_redundant // 64
        fp32_redundant = fp32_loop_redundant % 64
        return fp32_loop_times, fp32_redundant, fp32_repeat_times

    @staticmethod
    def get_start_addr(start_addr, length):
        """
        calu the start addr
        """
        return start_addr + length

    def res_fmap_convertf16(self, fmap_f16_ub, fmap_ub, loop_time):
        """
        convert result from fp32 to fp 16
        """
        fp32_loop_times = loop_time.get("fp32_loop_times", 0)
        fp32_repeat_times = loop_time.get("fp32_repeat_times", 0)
        fp32_redundant = loop_time.get("fp32_redundant", 0)
        if fp32_loop_times > 0:
            with self.tik_instance.for_range(0, fp32_loop_times,
                                             name="fp32_loop_time") as fp32_loop_time:
                loop_dst_offset = fp32_loop_time * 255 * 64
                loop_src_offset = fp32_loop_time * 255 * 64
                self.tik_instance.vconv(
                    64,
                    "none",
                    fmap_f16_ub[loop_dst_offset],
                    fmap_ub[loop_src_offset],
                    255,
                    1, 1, 4, 8)
        if fp32_repeat_times > 0:
            repeat_src_offset = fp32_loop_times * 255 * 64
            repeat_dst_offset = repeat_src_offset
            self.tik_instance.vconv(
                64,
                "none",
                fmap_f16_ub[repeat_dst_offset],
                fmap_ub[repeat_src_offset],
                fp32_repeat_times,
                1, 1, 4, 8)
        if fp32_redundant > 0:
            redundant_src_offset = fp32_loop_times * 255 * 64 + \
                                   fp32_repeat_times * 64
            redundant_dst_offset = redundant_src_offset
            self.tik_instance.vconv(
                fp32_redundant,
                "none",
                fmap_f16_ub[redundant_src_offset],
                fmap_ub[redundant_dst_offset],
                1,
                1, 1, 4, 8)

    def calu_square_sum(self, fmap_ub, fp32_loop_times, fp32_redundant, fp32_repeat_times):
        """
        calc the square sum
        """
        if fp32_loop_times > 0:
            with self.tik_instance.for_range(0, fp32_loop_times,
                                             name="fp32_loop_time") as fp32_loop_time:
                loop_src_offset = fp32_loop_time * 255 * 64
                loop_dst_offset = loop_src_offset
                self.tik_instance.vmul(
                    64,
                    self.square_mul_ub[loop_dst_offset],
                    fmap_ub[loop_src_offset],
                    fmap_ub[loop_src_offset],
                    255,
                    1, 1, 1, 8, 8, 8)
        if fp32_repeat_times > 0:
            repeat_src_offset = fp32_loop_times * 255 * 64
            repeat_dst_offset = repeat_src_offset
            self.tik_instance.vmul(
                64,
                self.square_mul_ub[repeat_dst_offset],
                fmap_ub[repeat_src_offset],
                fmap_ub[repeat_src_offset],
                fp32_repeat_times,
                1, 1, 1, 8, 8, 8)
        if fp32_redundant > 0:
            redundant_src_offset = fp32_loop_times * 255 * 64 + \
                                   fp32_repeat_times * 64
            redundant_dst_offset = redundant_src_offset
            self.tik_instance.vmul(
                fp32_redundant,
                self.square_mul_ub[redundant_dst_offset],
                fmap_ub[redundant_src_offset],
                fmap_ub[redundant_src_offset],
                1, 1, 1, 1,
                8, 8, 8)

    def set_orign_input_tensor(self):
        """
        set input tensors
        """
        self.input_tensors = []
        self.fmap_input = self.tik_instance.Tensor(dtype="float16", shape=self.fmap_5hd_shape, name="fmap_gm",
                                                   scope=tik.scope_gm)
        self.input_tensors.append(self.fmap_input)
        self.sum_input = self.tik_instance.Tensor(dtype="float32", shape=self.channel_wise_shape_in,
                                                  scope=tik.scope_gm, name="sum_gm")
        self.input_tensors.append(self.sum_input)
        self.square_sum_input = self.tik_instance.Tensor(dtype="float32", shape=self.channel_wise_shape_in,
                                                         scope=tik.scope_gm, name="square_sum_gm")
        self.input_tensors.append(self.square_sum_input)
        self.scale_input = self.tik_instance.Tensor(dtype="float32", shape=self.channel_wise_shape_in,
                                                    scope=tik.scope_gm, name="scale_gm")
        self.input_tensors.append(self.scale_input)
        self.offset_input = self.tik_instance.Tensor(dtype="float32", shape=self.channel_wise_shape_in,
                                                     scope=tik.scope_gm, name="offset_gm")
        self.input_tensors.append(self.offset_input)
        self.pre_moving_mean_input = self.tik_instance.Tensor(dtype="float32", shape=self.channel_wise_shape_in,
                                                              scope=tik.scope_gm, name="pre_moving_mean_gm")
        self.input_tensors.append(self.pre_moving_mean_input)
        self.pre_moving_variance_input = self.tik_instance.Tensor(dtype="float32", shape=self.channel_wise_shape_in,
                                                                  scope=tik.scope_gm, name="pre_moving_variance_gm")
        self.input_tensors.append(self.pre_moving_variance_input)

        self.filters_input = self.tik_instance.Tensor(dtype="float16", shape=self.filters_5hd_shape,
                                                      scope=tik.scope_gm, name="filters_gm")
        self.input_tensors.append(self.filters_input)

    def set_final_output_tensors(self):
        """
        set output tensors
        """
        self.output_tensors = []

        self.output_moving_mean = self.tik_instance.Tensor(dtype="float32", shape=self.channel_wise_shape_in,
                                                           scope=tik.scope_gm, name="output_moving_mean_gm")
        self.output_tensors.append(self.output_moving_mean)

        self.output_moving_varience = self.tik_instance.Tensor(dtype="float32", shape=self.channel_wise_shape_in,
                                                               scope=tik.scope_gm, name="output_moving_varience_gm")
        self.output_tensors.append(self.output_moving_varience)

        self.output_mean = self.tik_instance.Tensor(dtype="float32", shape=self.channel_wise_shape_in,
                                                    scope=tik.scope_gm, name="output_mean_gm")
        self.output_tensors.append(self.output_mean)

        self.output_varience = self.tik_instance.Tensor(dtype="float32", shape=self.channel_wise_shape_in,
                                                        scope=tik.scope_gm, name="output_varience_gm")
        self.output_tensors.append(self.output_varience)

        self.output_relu = self.tik_instance.Tensor("float16", self.fmap_5hd_shape, scope=tik.scope_gm,
                                                    name="output_relu_gm")
        self.output_tensors.append(self.output_relu)

        self.output_reluv2_mask = self.tik_instance.Tensor(dtype="uint16", shape=self.reluv2_mask_shape,
                                                           scope=tik.scope_gm, name="output_reluv2_mask_gm")
        self.output_tensors.append(self.output_reluv2_mask)

        self.output_convolution = self.tik_instance.Tensor(dtype="float16", shape=self.output_5hd_shape,
                                                           scope=tik.scope_gm, name="output_convolution_gm")
        self.output_tensors.append(self.output_convolution)

        self.output_sum = self.tik_instance.Tensor(dtype="float32", shape=self.channel_wise_shape_out,
                                                   scope=tik.scope_gm, name="output_sum_gm", is_atomic_add=True)

        self.output_tensors.append(self.output_sum)

        self.output_square_sum = self.tik_instance.Tensor(dtype="float32", shape=self.channel_wise_shape_out,
                                                          scope=tik.scope_gm, name="output_square_sum_gm",
                                                          is_atomic_add=True)
        self.output_tensors.append(self.output_square_sum)

    def get_conv_h_length(self):
        """
        rely on tiling calculate conv h length
        """
        return (self.m_l0a_factor * 16 // self.width_out - 1) * self.stride_h + \
               self.kernel_h

    def calc_dst_b_n_m_offset(self, m_l0, n_l0):
        """
        calculate tbe batch, n_dim, m_dim of result
        """
        dst_b_offset = self.front_batch
        dst_n_offset = ((self.front_n_redundant * self.n_l1a_factor + self.front_n_l1) * \
                        self.n_l0a_factor + n_l0)
        dst_m_offset = (self.front_m * self.m_l0a_factor + m_l0) * self.hw_out0
        dst_n_offset_scalar = self.tik_instance.Scalar(name="dst_n_offset", init_value=dst_n_offset)
        dst_m_offset_scalar = self.tik_instance.Scalar(name="dst_m_offset", init_value=dst_m_offset)
        return dst_b_offset, dst_m_offset_scalar, dst_n_offset_scalar

    def get_max_h_length(self):
        """
        calu the max real h length
        """
        return self.get_conv_h_length() - self.pad_t

    def _init_scalar_params(self):
        """
        init scalar params in tik
        """
        self.already_load_l1_h_length = self.tik_instance.Scalar(dtype="int32", name="already_load_l1_h_length",
                                                                 init_value=0)
        self.preload_already_load_l1_h_length = self.tik_instance.Scalar(dtype="int32",
                                                                         name="preload_already_load_l1_h_length",
                                                                         init_value=0)
        self.next_hin_l1_start_pos = self.tik_instance.Scalar(dtype="int32", name="next_hin_l1_start_pos", init_value=0)
        self.preload_next_hin_l1_start_pos = self.tik_instance.Scalar(dtype="int32",
                                                                      name="preload_next_hin_l1_start_pos",
                                                                      init_value=0)
        self.next_batch = self.tik_instance.Scalar(dtype="int32", name="next_batch", init_value=0)
        self.hin_l1_start_pos = self.tik_instance.Scalar(dtype="int32", name="hin_l1_start_pos",
                                                         init_value=(-self.pad_t))
        self.front_batch = self.tik_instance.Scalar(dtype="int32", name="front_batch", init_value=0)
        self.front_m = self.tik_instance.Scalar(dtype="int32", name="front_m", init_value=0)
        self.front_n_redundant = self.tik_instance.Scalar(dtype="int32", name="front_n_redundant", init_value=0)
        self.front_n_l1 = self.tik_instance.Scalar(dtype="int32", name="front_n_l1", init_value=0)
        self.bn_update_lastub_output_pingpong = self.tik_instance.Scalar(dtype="int32",
                                                                         name="bn_update_lastub_output_pingpong",
                                                                         init_value=0)
        self.l0a_pingpong = self.tik_instance.Scalar(dtype="int32", name="l0a_pingpong",
                                                     init_value=0)
        self.l0b_pingpong = self.tik_instance.Scalar(dtype="int32", name="l0b_pingpong",
                                                     init_value=0)
        self.l0c_pingpong = self.tik_instance.Scalar(dtype="int32", name="l0c_pingpong",
                                                     init_value=0)
        self.bn_reduce_ub_output_pingpong = self.tik_instance.Scalar(dtype="int32",
                                                                     name="bn_reduce_ub_output_pingpong", init_value=0)
        self.this_time_calu_length = self.tik_instance.Scalar(name="this_time_calu_length", init_value=0)
        self.preload_this_time_calu_length = self.tik_instance.Scalar(name="preload_this_time_calu_length",
                                                                      init_value=0)
        self.fm_ddr2ub_burst_length = self.tik_instance.Scalar(name="fm_ddr2ub_burst_length", init_value=0)
        self.fm_ddr2ub_src_stride = self.tik_instance.Scalar(name="fm_ddr2ub_src_stride", init_value=0)
        self.preload_fm_ddr2ub_burst_length = self.tik_instance.Scalar(name="preload_fm_ddr2ub_burst_length",
                                                                       init_value=0)
        self.preload_fm_ddr2ub_src_stride = self.tik_instance.Scalar(name="preload_fm_ddr2ub_src_stride", init_value=0)
        self.preload_next_batch = self.tik_instance.Scalar(name="preload_next_batch", init_value=0)

    def _parse_tiling_params(self, tiling):
        """
        parse tiling params
        """
        self.m_l0a_factor = tiling["AL0"][0]
        self.n_l0a_factor = tiling["BL0"][1]
        self.k_l0a_factor = tiling["AL0"][1]
        self.m_redundant_factor = self.int_ceil_div(self.hw_out1, self.m_l0a_factor)
        self.b_block_factor = tiling["BLOCK_DIM"][0]
        self.b_inner_factor = tiling["BLOCK_INNER"]
        self.b_redundant_factor = self.batch // (self.b_block_factor * self.b_inner_factor)
        self.k_redundant_factor = self.reduce_k1 // self.k_l0a_factor
        self.filters_load_from_l1_flag = False
        if tiling.get("BL1", None):
            self.k_l1a_factor = tiling["BL1"][0]
            self.n_l1a_factor = tiling["BL1"][1]
            self.n_redundant_factor = self.channel_out1 // (self.n_l0a_factor * self.n_l1a_factor)
            self.filters_load_from_l1_flag = True
        else:
            self.k_l1a_factor = 1
            # n_redundant_factor 统一计算逻辑
            self.n_redundant_factor = 1
            self.n_l1a_factor = self.channel_out1 // self.n_l0a_factor

    def _allocate_ub_in_area_one_sub(self, elem_num, square_sum_input_ub_start_addr):
        mean_ub_starr_addr = self.get_start_addr(square_sum_input_ub_start_addr, elem_num * 4)
        self.mean_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="mean_ub",
            scope=tik.scope_ubuf,
            start_addr=mean_ub_starr_addr)
        scale_ub_start_addr = self.get_start_addr(mean_ub_starr_addr, elem_num * 4)
        self.scale_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="scale_ub",
            scope=tik.scope_ubuf,
            start_addr=scale_ub_start_addr)
        square_sum_div_ub_start_addr = self.get_start_addr(scale_ub_start_addr, elem_num * 4)
        self.square_sum_div_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="square_sum_div_ub",
            scope=tik.scope_ubuf,
            start_addr=square_sum_div_ub_start_addr)
        square_mean_ub_start_addr = self.get_start_addr(square_sum_div_ub_start_addr, elem_num * 4)
        self.square_mean_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="square_mean_ub",
            scope=tik.scope_ubuf,
            start_addr=square_mean_ub_start_addr)
        varience_ub_start_addr = self.get_start_addr(square_mean_ub_start_addr, elem_num * 4)
        self.varience_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="varience_ub",
            scope=tik.scope_ubuf,
            start_addr=varience_ub_start_addr)
        y_scale_add_ub_start_addr = self.get_start_addr(varience_ub_start_addr, elem_num * 4)
        self.y_scale_add_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="y_scale_add_ub",
            scope=tik.scope_ubuf,
            start_addr=y_scale_add_ub_start_addr)
        y_scale_sqrt_ub_start_addr = self.get_start_addr(y_scale_add_ub_start_addr, elem_num * 4)
        self.y_scale_sqrt_ub = self.tik_instance.Tensor(
            "float32",
            [elem_num],
            name="y_scale_sqrt_ub",
            scope=tik.scope_ubuf,
            start_addr=y_scale_sqrt_ub_start_addr)
        return y_scale_sqrt_ub_start_addr

    def _mad_l0c_pong(self, dst_offset, k_redundant, src_offset):
        """
        mad l0c pong
        """
        with self.tik_instance.if_scope(self.l0b_pingpong == 0):
            with self.tik_instance.if_scope(self.l0a_pingpong == 0):
                # loa ping,lob ping, loc pong
                with self.tik_instance.if_scope(k_redundant * self.k_l0a_factor == 0):
                    self.tik_instance.mmad(
                        self.fmap_l0c_pong[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_ping[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_ping,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        0)
                with self.tik_instance.else_scope():
                    self.tik_instance.mmad(
                        self.fmap_l0c_pong[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_ping[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_ping,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        1)
            with self.tik_instance.else_scope():
                # loa pong,lob ping, loc pong
                with self.tik_instance.if_scope(k_redundant * self.k_l0a_factor == 0):
                    self.tik_instance.mmad(
                        self.fmap_l0c_pong[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_pong[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_ping,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        0)
                with self.tik_instance.else_scope():
                    self.tik_instance.mmad(
                        self.fmap_l0c_pong[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_pong[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_ping,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        1)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.l0a_pingpong == 0):
                # loa ping,lob pong, loc pong
                with self.tik_instance.if_scope(k_redundant * self.k_l0a_factor == 0):
                    self.tik_instance.mmad(
                        self.fmap_l0c_pong[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_ping[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_pong,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        0)
                with self.tik_instance.else_scope():
                    self.tik_instance.mmad(
                        self.fmap_l0c_pong[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_ping[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_pong,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        1)
            with self.tik_instance.else_scope():
                # loa pong,lob pong, loc pong
                with self.tik_instance.if_scope(k_redundant * self.k_l0a_factor == 0):
                    self.tik_instance.mmad(
                        self.fmap_l0c_pong[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_pong[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_pong,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        0)
                with self.tik_instance.else_scope():
                    self.tik_instance.mmad(
                        self.fmap_l0c_pong[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_pong[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_pong,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        1)

    def _mad_l0c_ping(self, dst_offset, k_redundant, src_offset):
        """
        mad l0c ping
        """
        with self.tik_instance.if_scope(self.l0b_pingpong == 0):
            with self.tik_instance.if_scope(self.l0a_pingpong == 0):
                # loa ping,lob ping, loc ping
                with self.tik_instance.if_scope(k_redundant * self.k_l0a_factor == 0):
                    self.tik_instance.mmad(
                        self.fmap_l0c_ping[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_ping[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_ping,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        0)
                with self.tik_instance.else_scope():
                    self.tik_instance.mmad(
                        self.fmap_l0c_ping[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_ping[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_ping,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        1)
            with self.tik_instance.else_scope():
                # loa pong,lob ping, loc ping
                with self.tik_instance.if_scope(k_redundant * self.k_l0a_factor == 0):
                    self.tik_instance.mmad(
                        self.fmap_l0c_ping[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_pong[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_ping,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        0)
                with self.tik_instance.else_scope():
                    self.tik_instance.mmad(
                        self.fmap_l0c_ping[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_pong[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_ping,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        1)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.l0a_pingpong == 0):
                # loa ping,lob pong, loc ping
                with self.tik_instance.if_scope(k_redundant * self.k_l0a_factor == 0):
                    self.tik_instance.mmad(
                        self.fmap_l0c_ping[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_ping[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_pong,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        0)
                with self.tik_instance.else_scope():
                    self.tik_instance.mmad(
                        self.fmap_l0c_ping[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_ping[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_pong,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        1)
            with self.tik_instance.else_scope():
                # loa pong,lob pong, loc ping
                with self.tik_instance.if_scope(k_redundant * self.k_l0a_factor == 0):
                    self.tik_instance.mmad(
                        self.fmap_l0c_ping[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_pong[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_pong,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        0)
                with self.tik_instance.else_scope():
                    self.tik_instance.mmad(
                        self.fmap_l0c_ping[dst_offset, 0, 0, 0, 0],
                        self.fmap_l0a_pong[src_offset, 0, 0, 0, 0],
                        self.filters_l0b_pong,
                        self.m_l0a_factor * self.hw_out0,
                        self.k_l0a_factor * self.channel_in0,
                        self.n_l0a_factor * self.channel_out0,
                        1)
    
    def _third_part_bn_mean_varience_process_sub(self, burst_length, redundant, repeat, start_pos):
        """
        third part of calculate mean and varience
        """
        # 计算均值的滑动平均 输出到ddr
        factor_reverse = 1.0 - self.factor
        if repeat >= 1:
            self.tik_instance.vmuls(
                64,
                self.mean_mul_ub,
                self.mean_ub,  # float32 一个repeat 64个元素
                self.factor,
                repeat,
                1, 1, 8, 8)
        if redundant >= 1:
            self.tik_instance.vmuls(
                redundant,
                self.mean_mul_ub[repeat * 64],
                self.mean_ub[repeat * 64],
                self.factor,
                1,
                1, 1, 8, 8)
        self.tik_instance.data_move(
            self.pre_moving_variance_ub,
            self.pre_moving_variance_input[start_pos],
            0,
            1, burst_length,
            0, 0)
        self.tik_instance.set_flag("PIPE_MTE2", "PIPE_V", 1)
        self.tik_instance.wait_flag("PIPE_MTE2", "PIPE_V", 0)
        if repeat >= 1:
            self.tik_instance.vmuls(
                64,
                self.mean_mul_rev_ub,
                self.pre_moving_mean_ub,
                factor_reverse,
                repeat,
                1, 1, 8, 8)  # float32 一个repeat 64个元素
        if redundant >= 1:
            self.tik_instance.vmuls(
                redundant,
                self.mean_mul_rev_ub[repeat * 64],
                self.pre_moving_mean_ub[repeat * 64],
                factor_reverse,
                1,
                1, 1, 8, 8)
        self.tik_instance.pipe_barrier("PIPE_V")
        if repeat >= 1:
            self.tik_instance.vadd(
                64,
                self.moving_mean_ub,
                self.mean_mul_ub,
                self.mean_mul_rev_ub,  # float32 一个repeat 64个元素
                repeat,
                1, 1, 1, 8, 8, 8)
        if redundant >= 1:
            self.tik_instance.vadd(
                redundant,
                self.moving_mean_ub[repeat * 64],
                self.mean_mul_ub[repeat * 64],
                self.mean_mul_rev_ub[repeat * 64],
                1,
                1, 1, 1, 8, 8, 8)
        self.tik_instance.set_flag("PIPE_V", "PIPE_MTE3", 2)
        return factor_reverse
