#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
quant_update_scatter
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl import common_util
from impl import constant_util as constant
from impl.util.util_common import ceil_div_scalar as ceil_div


class Constant:
    """
    The class for constant
    """
    TILING_ARG_NUM = 36
    MAX_INT64 = 2**63 - 1
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    deqscale = 1.0


def get_dtype_size(dtype):
    """
    get byte size of dtype
    """
    dtype_dict = {"float32": 4, "float16": 2, "int8": 1, "int32": 4, "int64": 8, "bfloat16": 2}
    return dtype_dict.get(dtype)


def is_ascend310p():
    if tbe_platform.api_check_support("tik.vbi") and not tbe_platform.api_check_support("tik.data_move_pad"):
        return True
    else:
        return False


class QuantUpdateScatterDynImpl():

    def __init__(self, obj):
        # pass
        self.element_obj = obj

        self.tik_inst = self.element_obj.tik_instance
        self.init_scalars()
        tiling_ub = self.element_obj.tiling_ub
        self.out_burst_len.set_as(tiling_ub[2])
        self.each_core_bs_num.set_as(tiling_ub[3])
        self.last_core_bs_num.set_as(tiling_ub[4])
        self.update_axis_shape.set_as(tiling_ub[5])
        self.index_burst_len.set_as(tiling_ub[6])
        self.update_burst_len.set_as(tiling_ub[7])
        self.src_bs_stride.set_as(tiling_ub[8])
        self.dst_bs_stride.set_as(tiling_ub[9])
        self.index_elements.set_as(tiling_ub[10])
        self.num_head.set_as(tiling_ub[11])
        self.size_per_head.set_as(tiling_ub[12])
        self.data_axis_shape.set_as(tiling_ub[13])
        self.scatter_stride.set_as(tiling_ub[14])
        self.num_one_block.set_as(tiling_ub[15])
        self.inner_loop_ele.set_as(tiling_ub[16])
        self.inner_loop_times.set_as(tiling_ub[17])
        self.inner_loop_tail.set_as(tiling_ub[18])
        self.indices_shape_rank.set_as(tiling_ub[19])
        self.quant_scales_elements.set_as(tiling_ub[20])
        self.quant_zero_points_elements.set_as(tiling_ub[21])
        self.quant_rept_num.set_as(tiling_ub[22])
        self.scales_rpt_len.set_as(tiling_ub[23])
        self.zero_points_rpt_len.set_as(tiling_ub[24])
        with self.tik_inst.if_scope(self.element_obj.tiling_mode != 100):
            self.inner_loop_times_lastcore.set_as(tiling_ub[25])
            self.inner_loop_tail_lastcore.set_as(tiling_ub[26])
            self.inner_loop_full_rpt.set_as(tiling_ub[27])
            self.inner_loop_tail_rpt_each_core.set_as(tiling_ub[28])
            self.inner_loop_tail_rpt_last_core.set_as(tiling_ub[29])
            self.src_fir_bs_stide.set_as(tiling_ub[30])
            self.dst_fir_bs_stide.set_as(tiling_ub[31])
            self.update_dim_0.set_as(tiling_ub[32])
            self.update_dim_1.set_as(tiling_ub[33])

        self.updates_ub_f32 = None
        self.quant_scales_ub_f32 = None
        self.quant_zero_points_ub_f32 = None

    def init_scalars(self):
        self.out_burst_len = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='out_burst_len')
        self.each_core_bs_num = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='each_core_bs_num')
        self.core_num = self.element_obj.used_aicore_num
        self.last_core_bs_num = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='last_core_bs_num')
        self.update_axis_shape = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='update_axis_shape')
        self.index_burst_len = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='index_burst_len')
        self.update_burst_len = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='update_burst_len')
        self.src_bs_stride = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='src_bs_stride')
        self.dst_bs_stride = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='dst_bs_stride')
        self.index_elements = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='index_elements')
        self.num_head = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='num_head')
        self.size_per_head = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='size_per_head')
        self.data_axis_shape = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='data_axis_shape')
        self.scatter_stride = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='scatter_stride')
        self.num_one_block = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='num_one_block')
        self.inner_loop_ele = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='inner_loop_ele')
        self.inner_loop_times = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='inner_loop_times')
        self.inner_loop_tail = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='inner_loop_tail')
        self.indices_shape_rank = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='indices_shape_rank')
        self.quant_scales_elements = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype,
                                                          name='quant_scales_elements')
        self.quant_zero_points_elements = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype,
                                                               name='quant_zero_points_elements')
        self.quant_rept_num = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='quant_rept_num')
        self.scales_rpt_len = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='scales_rpt_len')
        self.zero_points_rpt_len = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='zero_points_rpt_len')
        self.inner_loop_times_lastcore = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype,
                                                              name='inner_loop_times_lastcore')
        self.inner_loop_tail_lastcore = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype,
                                                             name='inner_loop_tail_lastcore')
        self.inner_loop_full_rpt = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='inner_loop_full_rpt')
        self.inner_loop_tail_rpt_each_core = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype,
                                                                  name='inner_loop_tail_rpt_each_core')
        self.inner_loop_tail_rpt_last_core = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype,
                                                                  name='inner_loop_tail_rpt_last_core')
        self.src_fir_bs_stide = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='src_fir_bs_stide')
        self.dst_fir_bs_stide = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='dst_fir_bs_stide')
        self.update_dim_0 = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='update_dim_0')
        self.update_dim_1 = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='update_dim_1')
        self.valid_idx = self.tik_inst.Scalar(dtype="int64")
        self.bs_idx = self.tik_inst.Scalar(dtype="int64")
        self.dst_offset = self.tik_inst.Scalar(dtype="int64")
        self.dst_offset_base = self.tik_inst.Scalar(dtype="int64")
        self.src_offset_base = self.tik_inst.Scalar(dtype="int64")

    # 'pylint: disable=too-many-arguments
    def compute_each_core(self, core_idx, core_bs_num):
        updates_ub_elements = core_bs_num * self.src_bs_stride
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (ceil_div(self.index_elements, 8) * 8,),
                                        name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_updates, (updates_ub_elements + 64,),
                                          name="updates_ub",
                                          scope=tik.scope_ubuf)
        if self.element_obj.dtype_updates != Constant.FLOAT32 and is_ascend310p():
            updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_updates, ((updates_ub_elements + 64) * 2,),
                                              name="updates_ub",
                                              scope=tik.scope_ubuf)
        updates_quant_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (updates_ub_elements + 64,),
                                                name="updates_quant_ub",
                                                scope=tik.scope_ubuf)
        self.tik_inst.data_move(index_ub, self.element_obj.indices_gm, 0, 1, self.index_burst_len, 0, 0)
        self.tik_inst.data_move(updates_ub,
                                self.element_obj.updates_gm[core_idx * self.each_core_bs_num * self.src_bs_stride:], 0,
                                1, (updates_ub_elements * get_dtype_size(self.element_obj.dtype_updates) + 31) // 32, 0,
                                0)
        self.quant_update(updates_ub, updates_quant_ub, core_bs_num)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            src_offset = each_core_bs_idx * self.src_bs_stride
            self._set_dst_offset_neg2(core_idx, each_core_bs_idx, index_ub, self.bs_idx, self.valid_idx,
                                      self.dst_offset)
            self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_quant_ub[src_offset], 0, 1,
                                    self.out_burst_len, 0, 0)

    def compute_each_core_large_batch(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (ceil_div(self.index_elements, 8) * 8,),
                                        name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_updates, (self.src_bs_stride + 64,),
                                          name="updates_ub",
                                          scope=tik.scope_ubuf)
        if self.element_obj.dtype_updates != Constant.FLOAT32 and is_ascend310p():
            updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_updates, ((self.src_bs_stride + 64) * 2,),
                                              name="updates_ub",
                                              scope=tik.scope_ubuf)
        updates_quant_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.src_bs_stride + 64,),
                                                name="updates_quant_ub",
                                                scope=tik.scope_ubuf)
        self.tik_inst.data_move(index_ub, self.element_obj.indices_gm, 0, 1, self.index_burst_len, 0, 0)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride
            update_len = (self.src_bs_stride * get_dtype_size(self.element_obj.dtype_updates) + 31) // 32
            self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset:], 0, 1, update_len, 0, 0)
            self.quant_update(updates_ub, updates_quant_ub, 1)
            self._set_dst_offset_neg2(core_idx, each_core_bs_idx, index_ub, self.bs_idx, self.valid_idx,
                                      self.dst_offset)
            update_out_len = (self.src_bs_stride * get_dtype_size(self.element_obj.dtype_data) + 31) // 32
            self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_quant_ub, 0, 1, update_out_len,
                                    0, 0)

    # 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments, too-many-lines
    def quant_update_for_large_ele_little_quant(self, updates_ub, updates_quant_ub, quant_scales_ub_f32,
                                                number, core_bs_num):
        updates_ub_f32 = updates_ub
        if self.element_obj.dtype_updates != Constant.FLOAT32:
            updates_ub_f32 = self.updates_ub_f32
            self.quant_conv_ub_to_f32(updates_ub[0:], updates_ub_f32[0:], number)

        repeat_time = self.quant_scales_elements // constant.MASK64
        mod = self.quant_scales_elements % constant.MASK64
        with self.tik_inst.if_scope(tik.all(repeat_time < core_bs_num, mod == 0)):
            with self.tik_inst.for_range(0, repeat_time) as quant_idx:
                offset = quant_idx * constant.MASK64
                mul_num = self.size_per_head / self.quant_scales_elements
                compute_num = core_bs_num * mul_num
                self.quant_calc_div_1(updates_ub_f32[offset:], quant_scales_ub_f32[offset:], compute_num, repeat_time)
                if self.element_obj.quant_zero_points:
                    self.quant_calc_add_1(updates_ub_f32[offset:], self.quant_zero_points_ub_f32[offset:],
                                          compute_num, repeat_time)
        with self.tik_inst.else_scope():
            mul_num = self.size_per_head / self.quant_scales_elements
            compute_num = core_bs_num * mul_num
            with self.tik_inst.for_range(0, compute_num) as quant_idx:
                offset = quant_idx * self.quant_scales_elements
                self.quant_calc_div(updates_ub_f32[offset:], quant_scales_ub_f32[0:], self.quant_scales_elements)
                if self.element_obj.quant_zero_points:
                    self.quant_calc_add(updates_ub_f32[offset:], self.quant_zero_points_ub_f32[0:],
                                        self.quant_scales_elements)

        if is_ascend310p():
            updates_ub_i32 = updates_ub.reinterpret_cast_to("int32")
            common_util.conv_f32_to_s8(self.tik_inst, updates_ub_i32[0:], updates_ub_f32[0:], number, "round")
            updates_ub_f32_to_f16 = updates_ub_f32.reinterpret_cast_to("float16")
            common_util.conv_s8_to_s4(self.tik_inst, updates_ub_f32_to_f16[0:], updates_ub_i32[0:],
                                    number, "", Constant.deqscale)
            common_util.conv_s2_to_i1(self.tik_inst, updates_quant_ub[0:], updates_ub_f32_to_f16[0:], number)
        elif updates_ub.dtype != Constant.FLOAT32:
            updates_ub_f16 = updates_ub.reinterpret_cast_to("int16")
            common_util.conv_s8_to_s4(self.tik_inst, updates_ub_f16[0:], updates_ub_f32[0:], number, "round")
            updates_ub_f32_to_f16 = updates_ub_f32.reinterpret_cast_to("float16")
            common_util.conv_s4_to_i4(self.tik_inst, updates_ub_f32_to_f16[0:], updates_ub_f16[0:], number, "")
            common_util.conv_s2_to_i1(self.tik_inst, updates_quant_ub[0:], updates_ub_f32_to_f16[0:], number)
        else:
            common_util.conv_i8_to_s8(self.tik_inst, updates_quant_ub[0:], updates_ub_f32[0:], number)

    # 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments, too-many-lines
    def quant_update_for_large_ele_large_quant(self, updates_ub, updates_quant_ub, quant_scales_ub,
                                               quant_zero_points_ub, number):
        updates_ub_f32 = updates_ub
        if self.element_obj.dtype_updates != Constant.FLOAT32:
            updates_ub_f32 = self.updates_ub_f32
            self.quant_conv_ub_to_f32(updates_ub[0:], updates_ub_f32[0:], number)

        if self.element_obj.dtype_quant_scales != Constant.FLOAT32:
            self.quant_conv_ub_to_f32(quant_scales_ub[0:], self.quant_scales_ub_f32[0:], number)
        quant_scales_ub_f32 =\
            quant_scales_ub if self.element_obj.dtype_quant_scales == Constant.FLOAT32 else self.quant_scales_ub_f32

        if self.element_obj.quant_zero_points:
            self.quant_conv_ub_to_f32(quant_zero_points_ub[0:], self.quant_zero_points_ub_f32[0:], number)

        self.quant_calc_div(updates_ub_f32[0:], quant_scales_ub_f32[0:], number)
        if self.element_obj.quant_zero_points:
            self.quant_calc_add(updates_ub_f32[0:], self.quant_zero_points_ub_f32[0:], number)
        if is_ascend310p():
            updates_ub_i32 = updates_ub.reinterpret_cast_to("int32")
            common_util.conv_f32_to_s8(self.tik_inst, updates_ub_i32[0:], updates_ub_f32[0:], number, "round")
            updates_ub_f32_to_f16 = updates_ub_f32.reinterpret_cast_to("float16")
            common_util.conv_s8_to_s4(self.tik_inst, updates_ub_f32_to_f16[0:], updates_ub_i32[0:],
                                    number, "", Constant.deqscale)
            common_util.conv_s2_to_i1(self.tik_inst, updates_quant_ub[0:], updates_ub_f32_to_f16[0:], number)
        elif updates_ub.dtype != Constant.FLOAT32:
            updates_ub_f16 = updates_ub.reinterpret_cast_to("int16")
            common_util.conv_s8_to_s4(self.tik_inst, updates_ub_f16[0:], updates_ub_f32[0:], number, "round")
            updates_ub_f32_to_f16 = updates_ub_f32.reinterpret_cast_to("float16")
            common_util.conv_s4_to_i4(self.tik_inst, updates_ub_f32_to_f16[0:], updates_ub_f16[0:], number, "")
            common_util.conv_s2_to_i1(self.tik_inst, updates_quant_ub[0:], updates_ub_f32_to_f16[0:], number)
        else:
            common_util.conv_i8_to_s8(self.tik_inst, updates_quant_ub[0:], updates_ub_f32[0:], number)

    # 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
    def compute_each_core_large_ele_litte_quant_calc(self, core_idx, is_last_core, updates_ub, updates_quant_ub,
                                                     quant_scales_ub_f32, index_ub):
        update_len = (self.inner_loop_ele * get_dtype_size(self.element_obj.dtype_updates) + 31) // 32
        update_out_len = (self.inner_loop_ele * get_dtype_size(self.element_obj.dtype_data) + 31) // 32
        if not is_last_core:
            full_loop = self.inner_loop_times
            tail_rpt = self.inner_loop_tail_rpt_each_core
            inner_loop_tail_number = self.inner_loop_tail
        else:
            full_loop = self.inner_loop_times_lastcore
            tail_rpt = self.inner_loop_tail_rpt_last_core
            inner_loop_tail_number = self.inner_loop_tail_lastcore

        with self.tik_inst.for_range(0, self.update_dim_0) as update_dim0_idx:
            with self.tik_inst.for_range(0, self.update_dim_1) as update_dim1_idx:
                with self.tik_inst.for_range(0, full_loop) as inner_loop_idx:
                    src_offset = update_dim0_idx * self.src_fir_bs_stide + update_dim1_idx * self.src_bs_stride + (
                        core_idx *
                        self.each_core_bs_num) * self.size_per_head + inner_loop_idx * self.inner_loop_ele
                    self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset:], 0, 1, update_len, 0,
                                            0)
                    self.quant_update_for_large_ele_little_quant(updates_ub, updates_quant_ub, quant_scales_ub_f32,
                                                                 self.inner_loop_ele, self.inner_loop_full_rpt)
                    self._set_dst_offset_neg2_large_ele(core_idx, self.each_core_bs_num, index_ub, self.bs_idx,
                                                        self.valid_idx, update_dim0_idx, update_dim1_idx,
                                                        self.dst_offset)
                    self.dst_offset.set_as(self.dst_offset + inner_loop_idx * self.inner_loop_ele)
                    self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_quant_ub, 0, 1,
                                            update_out_len, 0, 0)

                with self.tik_inst.if_scope(tail_rpt != 0):
                    update_len = (inner_loop_tail_number * get_dtype_size(self.element_obj.dtype_updates) + 31) // 32
                    update_out_len = (inner_loop_tail_number * get_dtype_size(self.element_obj.dtype_data) + 31) // 32
                    src_offset = update_dim0_idx * self.src_fir_bs_stide + update_dim1_idx * self.src_bs_stride + (
                        core_idx * self.each_core_bs_num) * self.size_per_head + full_loop * self.inner_loop_ele
                    self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset:], 0, 1, update_len, 0,
                                            0)
                    self.quant_update_for_large_ele_little_quant(updates_ub, updates_quant_ub, quant_scales_ub_f32,
                                                                 inner_loop_tail_number, tail_rpt)
                    self._set_dst_offset_neg2_large_ele(core_idx, self.each_core_bs_num, index_ub, self.bs_idx,
                                                        self.valid_idx, update_dim0_idx, update_dim1_idx,
                                                        self.dst_offset)
                    self.dst_offset.set_as(self.dst_offset + full_loop * self.inner_loop_ele)
                    self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_quant_ub, 0, 1,
                                            update_out_len, 0, 0)

    def compute_each_core_large_ele_litte_quant(self, core_idx, is_last_core):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (ceil_div(self.index_elements, 8) * 8,),
                                        name="index_ub", scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_updates, (self.inner_loop_ele,),
                                          name="updates_ub", scope=tik.scope_ubuf)
        if is_ascend310p():
            updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_updates, (self.inner_loop_ele * 2,),
                                              name="updates_ub",
                                              scope=tik.scope_ubuf)
        updates_quant_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.inner_loop_ele,),
                                                name="updates_quant_ub",
                                                scope=tik.scope_ubuf)
        quant_scales_ub = self.tik_inst.Tensor(self.element_obj.dtype_quant_scales, (self.quant_scales_elements,),
                                               name="quant_scales_ub",
                                               scope=tik.scope_ubuf)
        if self.element_obj.dtype_updates != Constant.FLOAT32:
            self.updates_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.inner_loop_ele,),
                                                       name="updates_ub_f32",
                                                       scope=tik.scope_ubuf)
        if self.element_obj.dtype_quant_scales != Constant.FLOAT32:
            self.quant_scales_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.quant_scales_elements,),
                                                            name="quant_calc_ub_f32",
                                                            scope=tik.scope_ubuf)
        quant_zero_points_ub = None
        if self.element_obj.quant_zero_points:
            quant_zero_points_ub = self.tik_inst.Tensor(self.element_obj.dtype_quant_zero_points,
                                                        (self.quant_zero_points_elements,),
                                                        name="quant_zero_points_ub",
                                                        scope=tik.scope_ubuf)
            self.quant_zero_points_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32,
                                                                 (self.quant_zero_points_elements,),
                                                                 name="quant_zero_points_ub_f32",
                                                                 scope=tik.scope_ubuf)
            quant_zero_points_len = (
                self.quant_zero_points_elements * get_dtype_size(self.element_obj.dtype_quant_zero_points) + 31) // 32
            self.tik_inst.data_move(quant_zero_points_ub, self.element_obj.quant_zero_points_gm[0:], 0, 1,
                                    quant_zero_points_len, 0, 0)

        self.tik_inst.data_move(index_ub, self.element_obj.indices_gm, 0, 1, self.index_burst_len, 0, 0)
        quant_scales_len = (self.quant_scales_elements * get_dtype_size(self.element_obj.dtype_quant_scales) +
                            31) // 32
        self.tik_inst.data_move(quant_scales_ub, self.element_obj.quant_scales_gm[0:], 0, 1, quant_scales_len, 0, 0)

        if self.element_obj.dtype_quant_scales != Constant.FLOAT32:
            self.quant_conv_ub_to_f32(quant_scales_ub[0:], self.quant_scales_ub_f32[0:], self.quant_scales_elements)
        quant_scales_ub_f32 = \
            quant_scales_ub if self.element_obj.dtype_quant_scales == Constant.FLOAT32 else self.quant_scales_ub_f32

        if self.element_obj.quant_zero_points:
            self.quant_conv_ub_to_f32(quant_zero_points_ub[0:], self.quant_zero_points_ub_f32[0:],
                                      self.quant_scales_elements)

        self.compute_each_core_large_ele_litte_quant_calc(core_idx, is_last_core, updates_ub, updates_quant_ub,
                                                          quant_scales_ub_f32, index_ub)

    # 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments, too-many-lines
    def compute_each_core_large_batch_litte_quant_calc(self, core_idx, core_bs_num, updates_ub, updates_quant_ub,
                                                       quant_scales_ub_f32, index_ub):
        update_len = (self.inner_loop_ele * get_dtype_size(self.element_obj.dtype_updates) + 31) // 32
        update_out_len = (self.inner_loop_ele * get_dtype_size(self.element_obj.dtype_data) + 31) // 32

        full_loop = self.inner_loop_times
        tail_rpt = self.inner_loop_tail_rpt_each_core
        inner_loop_tail_number = self.inner_loop_tail

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            with self.tik_inst.for_range(0, full_loop) as inner_loop_idx:
                src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride + \
                            inner_loop_idx * self.inner_loop_ele
                self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset:], 0, 1, update_len, 0, 0)
                self.quant_update_for_large_ele_little_quant(updates_ub, updates_quant_ub, quant_scales_ub_f32,
                                                             self.inner_loop_ele, self.inner_loop_full_rpt)
                self._set_dst_offset_neg2(core_idx, each_core_bs_idx, index_ub, self.bs_idx, self.valid_idx,
                                          self.dst_offset)
                self.dst_offset.set_as(self.dst_offset + inner_loop_idx * self.inner_loop_ele)
                self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_quant_ub, 0, 1,
                                        update_out_len, 0, 0)

            with self.tik_inst.if_scope(tail_rpt != 0):
                update_len = (inner_loop_tail_number * get_dtype_size(self.element_obj.dtype_updates) + 31) // 32
                update_out_len = (inner_loop_tail_number * get_dtype_size(self.element_obj.dtype_data) + 31) // 32
                src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride + \
                            full_loop * self.inner_loop_ele
                self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset:], 0, 1, update_len, 0, 0)
                self.quant_update_for_large_ele_little_quant(updates_ub, updates_quant_ub, quant_scales_ub_f32,
                                                             inner_loop_tail_number, tail_rpt)
                self._set_dst_offset_neg2(core_idx, each_core_bs_idx, index_ub, self.bs_idx, self.valid_idx,
                                          self.dst_offset)
                self.dst_offset.set_as(self.dst_offset + full_loop * self.inner_loop_ele)
                self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_quant_ub, 0, 1,
                                        update_out_len, 0, 0)

    def compute_each_core_large_batch_litte_quant(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (ceil_div(self.index_elements, 8) * 8,),
                                        name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_updates, (self.inner_loop_ele + 128,),
                                          name="updates_ub",
                                          scope=tik.scope_ubuf)
        if is_ascend310p():
            updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_updates, ((self.inner_loop_ele + 128) * 2,),
                                              name="updates_ub",
                                              scope=tik.scope_ubuf)
        updates_quant_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.inner_loop_ele + 128,),
                                                name="updates_quant_ub",
                                                scope=tik.scope_ubuf)
        quant_scales_ub = self.tik_inst.Tensor(self.element_obj.dtype_quant_scales, (self.quant_scales_elements + 64,),
                                               name="quant_scales_ub",
                                               scope=tik.scope_ubuf)
        if self.element_obj.dtype_updates != Constant.FLOAT32:
            self.updates_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.inner_loop_ele + 128,),
                                                       name="updates_ub_f32",
                                                       scope=tik.scope_ubuf)
        if self.element_obj.dtype_quant_scales != Constant.FLOAT32:
            self.quant_scales_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.quant_scales_elements + 64,),
                                                            name="quant_calc_ub_f32",
                                                            scope=tik.scope_ubuf)
        quant_zero_points_ub = None
        if self.element_obj.quant_zero_points:
            quant_zero_points_ub = self.tik_inst.Tensor(self.element_obj.dtype_quant_zero_points,
                                                        (self.quant_zero_points_elements + 64,),
                                                        name="quant_zero_points_ub",
                                                        scope=tik.scope_ubuf)
            self.quant_zero_points_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.quant_scales_elements + 64,),
                                                                 name="quant_zero_points_ub_f32",
                                                                 scope=tik.scope_ubuf)
            quant_zero_points_len = (
                self.quant_zero_points_elements * get_dtype_size(self.element_obj.dtype_quant_zero_points) + 31) // 32
            self.tik_inst.data_move(quant_zero_points_ub, self.element_obj.quant_zero_points_gm[0:], 0, 1,
                                    quant_zero_points_len, 0, 0)

        self.tik_inst.data_move(index_ub, self.element_obj.indices_gm, 0, 1, self.index_burst_len, 0, 0)

        quant_scales_len = (self.quant_scales_elements * get_dtype_size(self.element_obj.dtype_quant_scales) +
                            31) // 32
        self.tik_inst.data_move(quant_scales_ub, self.element_obj.quant_scales_gm[0:], 0, 1, quant_scales_len, 0, 0)
        if self.element_obj.dtype_quant_scales != Constant.FLOAT32:
            self.quant_conv_ub_to_f32(quant_scales_ub[0:], self.quant_scales_ub_f32[0:], self.quant_scales_elements)
        quant_scales_ub_f32 = \
            quant_scales_ub if self.element_obj.dtype_quant_scales == Constant.FLOAT32 else self.quant_scales_ub_f32

        if self.element_obj.quant_zero_points:
            self.quant_conv_ub_to_f32(quant_zero_points_ub[0:], self.quant_zero_points_ub_f32[0:],
                                      self.quant_scales_elements)

        self.compute_each_core_large_batch_litte_quant_calc(core_idx, core_bs_num, updates_ub, updates_quant_ub,
                                                            quant_scales_ub_f32, index_ub)

    # 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
    def compute_each_core_large_ele_large_quant_calc(self, updates_ub, updates_quant_ub, quant_scales_ub,
                                                     quant_zero_points_ub, src_offset_base, dst_offset_base,
                                                     quant_zero_points_len):
        update_len = (self.inner_loop_ele * get_dtype_size(self.element_obj.dtype_updates) + 31) // 32
        update_out_len = (self.inner_loop_ele * get_dtype_size(self.element_obj.dtype_data) + 31) // 32
        quant_scales_len = (self.inner_loop_ele * get_dtype_size(self.element_obj.dtype_quant_scales) + 31) // 32

        with self.tik_inst.for_range(0, self.inner_loop_times) as inner_idx:
            src_offset = src_offset_base + inner_idx * self.inner_loop_ele
            self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset:], 0, 1, update_len, 0, 0)
            quant_offset = inner_idx * self.inner_loop_ele
            self.tik_inst.data_move(quant_scales_ub, self.element_obj.quant_scales_gm[quant_offset:], 0, 1,
                                    quant_scales_len, 0, 0)
            if self.element_obj.quant_zero_points:
                self.tik_inst.data_move(quant_zero_points_ub, self.element_obj.quant_zero_points_gm[quant_offset:], 0,
                                        1, quant_zero_points_len, 0, 0)
            self.quant_update_for_large_ele_large_quant(updates_ub, updates_quant_ub, quant_scales_ub,
                                                        quant_zero_points_ub, self.inner_loop_ele)

            self.dst_offset.set_as(dst_offset_base + inner_idx * self.inner_loop_ele)
            self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_quant_ub, 0, 1, update_out_len,
                                    0, 0)

        with self.tik_inst.if_scope(self.inner_loop_tail != 0):
            update_len = (self.inner_loop_tail * get_dtype_size(self.element_obj.dtype_updates) + 31) // 32
            update_out_len = (self.inner_loop_tail * get_dtype_size(self.element_obj.dtype_data) + 31) // 32
            quant_scales_len = (self.inner_loop_tail * get_dtype_size(self.element_obj.dtype_quant_scales) + 31) // 32
            src_offset = src_offset_base + self.inner_loop_times * self.inner_loop_ele
            self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset:], 0, 1, update_len, 0, 0)
            quant_offset = self.inner_loop_times * self.inner_loop_ele
            self.tik_inst.data_move(quant_scales_ub, self.element_obj.quant_scales_gm[quant_offset:], 0, 1,
                                    quant_scales_len, 0, 0)
            if self.element_obj.quant_zero_points:
                quant_zero_points_len = (
                    self.inner_loop_tail * get_dtype_size(self.element_obj.dtype_quant_zero_points) + 31) // 32
                self.tik_inst.data_move(quant_zero_points_ub, self.element_obj.quant_zero_points_gm[quant_offset:], 0,
                                        1, quant_zero_points_len, 0, 0)
            self.quant_update_for_large_ele_large_quant(updates_ub, updates_quant_ub, quant_scales_ub,
                                                        quant_zero_points_ub, self.inner_loop_tail)
            self.dst_offset.set_as(dst_offset_base + self.inner_loop_times * self.inner_loop_ele)
            self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_quant_ub, 0, 1, update_out_len,
                                    0, 0)

    def compute_each_core_large_ele_large_quant(self, core_idx, is_last_core):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (ceil_div(self.index_elements, 8) * 8,),
                                        name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_updates, (self.inner_loop_ele + 128,),
                                          name="updates_ub",
                                          scope=tik.scope_ubuf)
        updates_quant_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.inner_loop_ele + 128,),
                                                name="updates_quant_ub",
                                                scope=tik.scope_ubuf)
        quant_scales_ub = self.tik_inst.Tensor(self.element_obj.dtype_quant_scales, (self.inner_loop_ele + 64,),
                                               name="quant_scales_ub",
                                               scope=tik.scope_ubuf)
        if self.element_obj.dtype_updates != Constant.FLOAT32:
            self.updates_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.inner_loop_ele + 128,),
                                                       name="updates_ub_f32",
                                                       scope=tik.scope_ubuf)
        if self.element_obj.dtype_quant_scales != Constant.FLOAT32:
            self.quant_scales_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.inner_loop_ele + 64,),
                                                            name="quant_calc_ub_f32",
                                                            scope=tik.scope_ubuf)
        quant_zero_points_ub, quant_zero_points_len = None, 0
        if self.element_obj.quant_zero_points:
            quant_zero_points_ub = self.tik_inst.Tensor(self.element_obj.dtype_quant_zero_points,
                                                        (self.inner_loop_ele + 64,),
                                                        name="quant_zero_points_ub",
                                                        scope=tik.scope_ubuf)
            self.quant_zero_points_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.inner_loop_ele + 64,),
                                                                 name="quant_zero_points_ub_f32",
                                                                 scope=tik.scope_ubuf)
            quant_zero_points_len = (self.inner_loop_ele * get_dtype_size(self.element_obj.dtype_quant_zero_points) +
                                     31) // 32

        self.tik_inst.data_move(index_ub, self.element_obj.indices_gm, 0, 1, self.index_burst_len, 0, 0)

        core_bs_num = self.each_core_bs_num if not is_last_core else self.last_core_bs_num
        mul_num = self.size_per_head / self.quant_scales_elements
        compute_num = core_bs_num * mul_num
        
        with self.tik_inst.for_range(0, self.update_dim_0) as update_dim0_idx:
            with self.tik_inst.for_range(0, self.update_dim_1) as update_dim1_idx:
                with self.tik_inst.for_range(0, compute_num) as loop_bs_idx:
                    self.src_offset_base.set_as(update_dim0_idx * self.src_fir_bs_stide +
                                                update_dim1_idx * self.src_bs_stride +
                                                (core_idx * self.each_core_bs_num * mul_num + loop_bs_idx) *
                                                self.quant_scales_elements)
                    self._set_dst_offset_neg2_large_ele(core_idx, self.each_core_bs_num, index_ub, self.bs_idx,
                                                        self.valid_idx, update_dim0_idx, update_dim1_idx,
                                                        self.dst_offset_base)
                    self.dst_offset_base.set_as(self.dst_offset_base + loop_bs_idx * self.quant_scales_elements)
                    self.compute_each_core_large_ele_large_quant_calc(updates_ub, updates_quant_ub, quant_scales_ub,
                                                                      quant_zero_points_ub, self.src_offset_base,
                                                                      self.dst_offset_base, quant_zero_points_len)

    # 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
    def compute_each_core_large_batch_large_quant_calc(self, updates_ub, updates_quant_ub, quant_scales_ub,
                                                       quant_zero_points_ub, src_offset_base, dst_offset_base,
                                                       quant_zero_points_len):
        update_len = (self.inner_loop_ele * get_dtype_size(self.element_obj.dtype_updates) + 31) // 32
        update_out_len = (self.inner_loop_ele * get_dtype_size(self.element_obj.dtype_data) + 31) // 32
        quant_scales_len = (self.inner_loop_ele * get_dtype_size(self.element_obj.dtype_quant_scales) + 31) // 32

        with self.tik_inst.for_range(0, self.inner_loop_times) as inner_idx:
            src_offset = src_offset_base + inner_idx * self.inner_loop_ele
            self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset:], 0, 1, update_len, 0, 0)
            quant_offset = inner_idx * self.inner_loop_ele
            self.tik_inst.data_move(quant_scales_ub, self.element_obj.quant_scales_gm[quant_offset:], 0, 1,
                                    quant_scales_len, 0, 0)
            if self.element_obj.quant_zero_points:
                self.tik_inst.data_move(quant_zero_points_ub, self.element_obj.quant_zero_points_gm[quant_offset:], 0,
                                        1, quant_zero_points_len, 0, 0)
            self.quant_update_for_large_ele_large_quant(updates_ub, updates_quant_ub, quant_scales_ub,
                                                        quant_zero_points_ub, self.inner_loop_ele)

            self.dst_offset.set_as(dst_offset_base + inner_idx * self.inner_loop_ele)
            self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_quant_ub, 0, 1, update_out_len,
                                    0, 0)

        with self.tik_inst.if_scope(self.inner_loop_tail != 0):
            update_len = (self.inner_loop_tail * get_dtype_size(self.element_obj.dtype_updates) + 31) // 32
            update_out_len = (self.inner_loop_tail * get_dtype_size(self.element_obj.dtype_data) + 31) // 32
            quant_scales_len = (self.inner_loop_tail * get_dtype_size(self.element_obj.dtype_quant_scales) + 31) // 32
            src_offset = src_offset_base + self.inner_loop_times * self.inner_loop_ele
            self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset:], 0, 1, update_len, 0, 0)
            quant_offset = self.inner_loop_times * self.inner_loop_ele
            self.tik_inst.data_move(quant_scales_ub, self.element_obj.quant_scales_gm[quant_offset:], 0, 1,
                                    quant_scales_len, 0, 0)
            if self.element_obj.quant_zero_points:
                quant_zero_points_len = (
                    self.inner_loop_tail * get_dtype_size(self.element_obj.dtype_quant_zero_points) + 31) // 32
                self.tik_inst.data_move(quant_zero_points_ub, self.element_obj.quant_zero_points_gm[quant_offset:], 0,
                                        1, quant_zero_points_len, 0, 0)
            self.quant_update_for_large_ele_large_quant(updates_ub, updates_quant_ub, quant_scales_ub,
                                                        quant_zero_points_ub, self.inner_loop_tail)
            self.dst_offset.set_as(dst_offset_base + self.inner_loop_times * self.inner_loop_ele)
            self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_quant_ub, 0, 1, update_out_len,
                                    0, 0)

    def compute_each_core_large_batch_large_quant(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (ceil_div(self.index_elements, 8) * 8,),
                                        name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_updates, (self.inner_loop_ele + 128,),
                                          name="updates_ub",
                                          scope=tik.scope_ubuf)
        updates_quant_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.inner_loop_ele + 128,),
                                                name="updates_quant_ub",
                                                scope=tik.scope_ubuf)
        quant_scales_ub = self.tik_inst.Tensor(self.element_obj.dtype_quant_scales, (self.inner_loop_ele + 64,),
                                               name="quant_scales_ub",
                                               scope=tik.scope_ubuf)
        if self.element_obj.dtype_updates != Constant.FLOAT32:
            self.updates_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.inner_loop_ele + 128,),
                                                       name="updates_ub_f32",
                                                       scope=tik.scope_ubuf)
        if self.element_obj.dtype_quant_scales != Constant.FLOAT32:
            self.quant_scales_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.inner_loop_ele + 64,),
                                                            name="quant_calc_ub_f32",
                                                            scope=tik.scope_ubuf)
        quant_zero_points_ub = None
        quant_zero_points_len = 0
        if self.element_obj.quant_zero_points:
            quant_zero_points_ub = self.tik_inst.Tensor(self.element_obj.dtype_quant_zero_points,
                                                        (self.inner_loop_ele + 64,),
                                                        name="quant_zero_points_ub",
                                                        scope=tik.scope_ubuf)
            self.quant_zero_points_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.inner_loop_ele + 64,),
                                                                 name="quant_zero_points_ub_f32",
                                                                 scope=tik.scope_ubuf)
            quant_zero_points_len = (self.inner_loop_ele * get_dtype_size(self.element_obj.dtype_quant_zero_points) +
                                     31) // 32
        self.tik_inst.data_move(index_ub, self.element_obj.indices_gm, 0, 1, self.index_burst_len, 0, 0)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            with self.tik_inst.for_range(0, self.update_axis_shape) as update_idx:
                self.src_offset_base.set_as((core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride +
                                            update_idx * self.quant_scales_elements)
                self._set_dst_offset_neg2(core_idx, each_core_bs_idx, index_ub, self.bs_idx, self.valid_idx,
                                          self.dst_offset_base)
                self.dst_offset_base.set_as(self.dst_offset_base + update_idx * self.quant_scales_elements)
                self.compute_each_core_large_batch_large_quant_calc(updates_ub, updates_quant_ub, quant_scales_ub,
                                                                    quant_zero_points_ub, self.src_offset_base,
                                                                    self.dst_offset_base, quant_zero_points_len)

    def quant_calc_div(self, updates_ub_f32, quant_scales_ub_f32, number):
        repeat_time = (number) // constant.MASK64
        if self.element_obj.reciprocal_scale:
            self.tik_inst.vmul(constant.MASK64, updates_ub_f32[0:], updates_ub_f32[0:], quant_scales_ub_f32[0:],
                            repeat_time, 1, 1, 1, 8, 8, 8)
            mod = number % constant.MASK64
            with self.tik_inst.if_scope(mod != 0):
                offset = repeat_time * constant.MASK64
                self.tik_inst.vmul(mod, updates_ub_f32[offset:], updates_ub_f32[offset:], quant_scales_ub_f32[offset:],
                                   1, 1, 1, 1, 8, 8, 8)
        else:
            self.tik_inst.vdiv(constant.MASK64, updates_ub_f32[0:], updates_ub_f32[0:], quant_scales_ub_f32[0:],
                            repeat_time, 1, 1, 1, 8, 8, 8)
            mod = number % constant.MASK64
            with self.tik_inst.if_scope(mod != 0):
                offset = repeat_time * constant.MASK64
                self.tik_inst.vdiv(mod, updates_ub_f32[offset:], updates_ub_f32[offset:], quant_scales_ub_f32[offset:],
                                   1, 1, 1, 1, 8, 8, 8)

    def quant_calc_add(self, updates_ub_f32, quant_zero_points_ub_f32, number):
        repeat_time = (number) // constant.MASK64
        self.tik_inst.vadd(constant.MASK64, updates_ub_f32[0:], updates_ub_f32[0:], quant_zero_points_ub_f32[0:],
                           repeat_time, 1, 1, 1, 8, 8, 8)
        mod = number % constant.MASK64
        with self.tik_inst.if_scope(mod != 0):
            offset = repeat_time * constant.MASK64
            self.tik_inst.vadd(mod, updates_ub_f32[offset:], updates_ub_f32[offset:], quant_zero_points_ub_f32[offset:],
                               1, 1, 1, 1, 8, 8, 8)

    def quant_calc_div_1(self, updates_ub_f32, quant_scales_ub_f32, core_bs_num, repeat_time):
        stride_num = repeat_time * 8
        loop = core_bs_num // constant.MAX_REPEAT_TIMES
        with self.tik_inst.if_scope(loop > 0):
            with self.tik_inst.for_range(0, loop) as index:
                compute_offsets = index * constant.MASK64 * constant.MAX_REPEAT_TIMES
                if self.element_obj.reciprocal_scale:
                    self.tik_inst.vmul(constant.MASK64, updates_ub_f32[compute_offsets:],
                                       updates_ub_f32[compute_offsets:], quant_scales_ub_f32[0:],
                                       constant.MAX_REPEAT_TIMES, 1, 1, 1, stride_num, stride_num, 0)
                else:
                    self.tik_inst.vdiv(constant.MASK64, updates_ub_f32[compute_offsets:],
                                       updates_ub_f32[compute_offsets:], quant_scales_ub_f32[0:],
                                       constant.MAX_REPEAT_TIMES, 1, 1, 1, stride_num, stride_num, 0)
        tail_offset = loop * constant.MASK64 * constant.MAX_REPEAT_TIMES
        tail_num = core_bs_num % constant.MAX_REPEAT_TIMES
        with self.tik_inst.if_scope(tail_num > 0):
            if self.element_obj.reciprocal_scale:
                self.tik_inst.vmul(constant.MASK64, updates_ub_f32[tail_offset:], updates_ub_f32[tail_offset:],
                                    quant_scales_ub_f32[0:], tail_num, 1, 1, 1,
                                    stride_num, stride_num, 0)
            else:
                self.tik_inst.vdiv(constant.MASK64, updates_ub_f32[tail_offset:], updates_ub_f32[tail_offset:],
                                    quant_scales_ub_f32[0:], tail_num, 1, 1, 1,
                                    stride_num, stride_num, 0)

    def quant_calc_add_1(self, updates_ub_f32, quant_zero_points_ub_f32, core_bs_num, repeat_time):
        stride_num = repeat_time * 8
        loop = core_bs_num // constant.MAX_REPEAT_TIMES
        with self.tik_inst.if_scope(loop > 0):
            with self.tik_inst.for_range(0, loop) as index:
                compute_offsets = index * constant.MASK64 * constant.MAX_REPEAT_TIMES
                self.tik_inst.vadd(constant.MASK64, updates_ub_f32[compute_offsets:], updates_ub_f32[compute_offsets:],
                                   quant_zero_points_ub_f32[0:], constant.MAX_REPEAT_TIMES, 1, 1, 1,
                                   stride_num, stride_num, 0)
        tail_offset = loop * constant.MASK64 * constant.MAX_REPEAT_TIMES
        tail_num = core_bs_num % constant.MAX_REPEAT_TIMES
        with self.tik_inst.if_scope(tail_num > 0):
            self.tik_inst.vadd(constant.MASK64, updates_ub_f32[tail_offset:], updates_ub_f32[tail_offset:],
                                   quant_zero_points_ub_f32[0:], tail_num, 1, 1, 1,
                                   stride_num, stride_num, 0)

    def quant_conv_ub_to_f32(self, input_ub, out_ub, number):
        with self.tik_inst.new_stmt_scope():
            if input_ub.dtype == Constant.FLOAT32:
                out_ub = input_ub
            elif input_ub.dtype == "int32":
                common_util.conv_i8_to_s8(self.tik_inst, out_ub[0:], input_ub[0:], number)
            elif input_ub.dtype == "bfloat16" or input_ub.dtype == "float16":
                common_util.conv_s4_to_s8(self.tik_inst, out_ub[0:], input_ub[0:], number)
            else:
                temp_ub_f16 = self.tik_inst.Tensor(Constant.FLOAT16, (number,),
                                                   name="temp_ub_f16",
                                                   scope=tik.scope_ubuf)
                common_util.conv_i1_to_s2(self.tik_inst, temp_ub_f16[0:], input_ub[0:], number)
                common_util.conv_s4_to_s8(self.tik_inst, out_ub[0:], temp_ub_f16[0:], number)

    def quant_update(self, updates_ub, updates_quant_ub, core_bs_num):
        updates_ub_f32 = updates_ub
        if self.element_obj.dtype_updates != Constant.FLOAT32:
            updates_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (core_bs_num * self.src_bs_stride + 64,),
                                                  name="updates_ub_f32", scope=tik.scope_ubuf)
            self.quant_conv_ub_to_f32(updates_ub[0:], updates_ub_f32[0:], core_bs_num * self.src_bs_stride)

        quant_scales_ub = self.tik_inst.Tensor(self.element_obj.dtype_quant_scales, (self.quant_scales_elements + 64,),
                                               name="quant_scales_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(quant_scales_ub, self.element_obj.quant_scales_gm, 0, 1, self.scales_rpt_len, 0, 0)
        if self.element_obj.dtype_quant_scales != Constant.FLOAT32:
            quant_calc_ub_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.quant_scales_elements + 64,),
                                                     name="quant_calc_ub_f32", scope=tik.scope_ubuf)
            self.quant_conv_ub_to_f32(quant_scales_ub[0:], quant_calc_ub_f32[0:], self.quant_scales_elements)

        quant_scales_ub_f32 = \
            quant_scales_ub if self.element_obj.dtype_quant_scales == Constant.FLOAT32 else quant_calc_ub_f32
        total_loop = self.quant_rept_num * core_bs_num
        with self.tik_inst.for_range(0, total_loop) as quant_idx:
            offset = quant_idx * self.quant_scales_elements
            self.quant_calc_div(updates_ub_f32[offset:], quant_scales_ub_f32[0:], self.quant_scales_elements)

        if self.element_obj.quant_zero_points:
            quant_zero_points_ub = self.tik_inst.Tensor(self.element_obj.dtype_quant_zero_points,
                                                        (self.quant_zero_points_elements + 64,),
                                                        name="quant_zero_points_ub",
                                                        scope=tik.scope_ubuf)
            quant_zero_points_f32 = self.tik_inst.Tensor(Constant.FLOAT32, (self.quant_scales_elements + 64,),
                                                         name="quant_zero_points_f32",
                                                         scope=tik.scope_ubuf)
            self.tik_inst.data_move(quant_zero_points_ub, self.element_obj.quant_zero_points_gm, 0, 1,
                                    self.zero_points_rpt_len, 0, 0)
            self.quant_conv_ub_to_f32(quant_zero_points_ub[0:], quant_zero_points_f32[0:],
                                      self.quant_zero_points_elements)
            with self.tik_inst.for_range(0, total_loop) as quant_idx:
                offset = quant_idx * self.quant_scales_elements
                self.quant_calc_add(updates_ub_f32[offset:], quant_zero_points_f32[0:], self.quant_scales_elements)
        updates_ub_i32 = updates_ub.reinterpret_cast_to("int32")
        quant_number = core_bs_num * self.src_bs_stride
        if is_ascend310p():
            common_util.conv_f32_to_s8(self.tik_inst, updates_ub_i32[0:], updates_ub_f32[0:], quant_number, "round")
            updates_ub_f32_to_f16 = updates_ub_f32.reinterpret_cast_to("float16")
            common_util.conv_s8_to_s4(self.tik_inst, updates_ub_f32_to_f16[0:], updates_ub_i32[0:],
                                    quant_number, "", Constant.deqscale)
            common_util.conv_s2_to_i1(self.tik_inst, updates_quant_ub[0:], updates_ub_f32_to_f16[0:], quant_number)
        elif updates_quant_ub.dtype != Constant.FLOAT32:
            updates_ub_f16 = updates_ub.reinterpret_cast_to("int16")
            common_util.conv_s8_to_s4(self.tik_inst, updates_ub_f16[0:], updates_ub_f32[0:], quant_number, "round")
            updates_ub_f32_to_f16 = updates_ub_f32.reinterpret_cast_to("float16")
            common_util.conv_s4_to_i4(self.tik_inst, updates_ub_f32_to_f16[0:], updates_ub_f16[0:], quant_number)
            common_util.conv_s2_to_i1(self.tik_inst, updates_quant_ub[0:], updates_ub_f32_to_f16[0:], quant_number)
        else:
            common_util.conv_i8_to_s8(self.tik_inst, updates_quant_ub[0:], updates_ub_f32[0:], quant_number)

    def compute(self, core_index):
        with self.tik_inst.if_scope(self.element_obj.tiling_mode == 100):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 101):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_large_batch(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_large_batch(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 102):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_large_ele_litte_quant(core_idx=core_index, is_last_core=False)
            with self.tik_inst.else_scope():
                self.compute_each_core_large_ele_litte_quant(core_idx=core_index, is_last_core=True)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 103):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_large_ele_large_quant(core_idx=core_index, is_last_core=False)
            with self.tik_inst.else_scope():
                self.compute_each_core_large_ele_large_quant(core_idx=core_index, is_last_core=True)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 104):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_large_batch_litte_quant(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_large_batch_litte_quant(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 105):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_large_batch_large_quant(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_large_batch_large_quant(core_idx=core_index, core_bs_num=self.last_core_bs_num)

    # 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
    def _set_dst_offset_neg2(self, core_idx, each_core_bs_idx, index_ub, bs_idx, valid_idx, dst_offset):
        with self.tik_inst.if_scope(self.indices_shape_rank == 2):
            index_idx = (core_idx * self.each_core_bs_num + each_core_bs_idx) // self.num_head
            bs_idx.set_as(index_ub[2 * index_idx])
            valid_idx.set_as(index_ub[2 * index_idx + 1])
            actual_bs_idx = bs_idx * self.num_head + (core_idx * self.each_core_bs_num +
                                                      each_core_bs_idx) % self.num_head
            dst_offset.set_as(actual_bs_idx * self.dst_bs_stride + valid_idx * self.size_per_head)
        with self.tik_inst.else_scope():
            index_idx = core_idx * self.each_core_bs_num + each_core_bs_idx
            bs_idx.set_as(index_idx)
            valid_idx.set_as(index_ub[index_idx // self.num_head])
            dst_offset.set_as(bs_idx * self.dst_bs_stride + valid_idx * self.size_per_head)

    # 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
    def _set_dst_offset_neg2_large_ele(self, core_idx, each_core_bs, index_ub, bs_idx, valid_idx, update_dim0_idx,
                                       update_dim1_idx, dst_offset):
        with self.tik_inst.if_scope(self.indices_shape_rank == 2):
            index_idx = update_dim0_idx
            bs_idx.set_as(index_ub[2 * index_idx])
            valid_idx.set_as(index_ub[2 * index_idx + 1])
            actual_bs_idx = bs_idx * self.dst_fir_bs_stide + update_dim1_idx * self.dst_bs_stride
            dst_offset.set_as(actual_bs_idx + (valid_idx + each_core_bs * core_idx) * self.size_per_head)
        with self.tik_inst.else_scope():
            index_idx = update_dim0_idx
            bs_idx.set_as(index_ub[index_idx])
            actual_bs_idx = index_idx * self.dst_fir_bs_stide + update_dim1_idx * self.dst_bs_stride
            dst_offset.set_as(actual_bs_idx + (bs_idx + each_core_bs * core_idx) * self.size_per_head)


class QuantUpdateScatter:
    """
    The class for scatter
    """

    def __init__(self, data, indices, updates, quant_scales, quant_zero_points, result, axis, reduction, quant_axis,
                 reciprocal_scale, kernel_name) -> None:
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.tiling_param_dtype = "int64"
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_param_dtype, (Constant.TILING_ARG_NUM,),
                                                  name='tiling_gm',
                                                  scope=tik.scope_gm)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.total_core_number = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_data = data.get("dtype").lower()
        self.dtype_indices = indices.get("dtype").lower()
        self.dtype_updates = updates.get("dtype").lower()
        self.dtype_out = result.get("dtype").lower()
        if self.dtype_data == "bfloat16":
            self.dtype_data = "float16"
            self.dtype_updates = "float16"
            self.dtype_out = "float16"
        self.dtype_quant_scales = quant_scales.get("dtype").lower()
        self.quant_zero_points = quant_zero_points
        self.reciprocal_scale = reciprocal_scale

        self.input_gm_list = []
        self.data_gm = self.tik_instance.Tensor(self.dtype_data, [Constant.MAX_INT64],
                                                name="data_gm",
                                                scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.dtype_indices, [Constant.MAX_INT64],
                                                   name="indices_gm",
                                                   scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(self.dtype_updates, [Constant.MAX_INT64],
                                                   name="updates_gm",
                                                   scope=tik.scope_gm)
        self.quant_scales_gm = self.tik_instance.Tensor(self.dtype_quant_scales, [Constant.MAX_INT64],
                                                        name="quant_scales_gm",
                                                        scope=tik.scope_gm)
        self.input_gm_list.append(self.data_gm)
        self.input_gm_list.append(self.indices_gm)
        self.input_gm_list.append(self.updates_gm)
        self.input_gm_list.append(self.quant_scales_gm)
        if self.quant_zero_points is not None:
            self.dtype_quant_zero_points = quant_zero_points.get("dtype").lower()
            self.quant_zero_points_gm = self.tik_instance.Tensor(self.dtype_quant_zero_points, [Constant.MAX_INT64],
                                                                 name="quant_zero_points_gm",
                                                                 scope=tik.scope_gm)
            self.input_gm_list.append(self.quant_zero_points_gm)
        self.result_gm = self.tik_instance.Tensor(self.dtype_out, [Constant.MAX_INT64],
                                                  name="result_gm",
                                                  scope=tik.scope_gm)
        self.tiling_ub = self.tik_instance.Tensor(self.tiling_param_dtype, (Constant.TILING_ARG_NUM,),
                                                  name='tiling_ub',
                                                  scope=tik.scope_ubuf)
        self.tiling_mode = self.tik_instance.Scalar(self.tiling_param_dtype, name='tiling_mode')
        self.used_aicore_num = self.tik_instance.Scalar(self.tiling_param_dtype, name='used_aicore_num')

    def get_tilings(self):
        """
        get_tilings
        """
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, (Constant.TILING_ARG_NUM + 3) // 4, 0,
                                    0)  # 4 for int64
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.used_aicore_num.set_as(self.tiling_ub[1])

    def compute(self):
        self.get_tilings()
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            obj_quant_update_scatter = QuantUpdateScatterDynImpl(self)
            obj_quant_update_scatter.compute(i)

        tbe_context.get_context().add_compile_info('vars', {'ub_size': self.ub_size_bytes})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.input_gm_list,
                                   outputs=[self.result_gm],
                                   flowtable=[self.tiling_gm])


# 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
@register_operator("QuantUpdateScatter")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def quant_update_scatter(var,
                         indices,
                         updates,
                         quant_scales,
                         quant_zeros_point,
                         var_out,
                         reduce,
                         axis=0,
                         quant_axis=1,
                         reciprocal_scale=False,
                         kernel_name="quant_update_scatter"):
    """
    scatter_mul interface

    Parameters
    ----------
    var_dict: input var shape, dtype and range
    indices_dict: input indices shape, dtype and range
    updates_dict: input updates shape, dtype and range
    var_out_dict: output shape, dtype and range
    reduce: type of scatter op, support "update", "add", "mul"
    kernel_name: kernel name of scatter op
    impl_mode: high_precision or high_performance
    Returns
    -------
    compile info
    """
    obj = QuantUpdateScatter(var, indices, updates, quant_scales, quant_zeros_point, var_out, axis, reduce, quant_axis,
                             reciprocal_scale, kernel_name)
    return obj.compute()
