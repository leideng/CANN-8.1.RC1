#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

as_strided
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import para_check
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods, too-many-lines
class TilingParams:
    """
    The class for getting tiling parameters
    """

    def __init__(self, tik_inst, tiling_gm, tiling_ub, tiling_params):
        self.tik_inst = tik_inst
        self.tiling_gm = tiling_gm
        self.tiling_ub = tiling_ub
        self.tiling_params = tiling_params
        dtype_bytes = get_bit_len(tiling_ub.dtype) // AsStrided.BITS_PER_BYTE
        self.ele_per_block = AsStrided.BYTES_PER_BLOCK // dtype_bytes

    def get_tiling_params(self):
        """
        method of getting tiling parameters
        """
        self.tik_inst.data_move(self.tiling_ub, self.tiling_gm,
                                0, 1, AsStrided.TILING_SPACE[1] // self.ele_per_block, 0, 0)
        for idx, reg in enumerate(self.tiling_params):
            reg.set_as(self.tiling_ub[idx])


# 'pylint: disable=too-many-instance-attributes
class AsStrided:
    """
    The class of AsStrided
    """
    BYTES_PER_BLOCK = 32
    BITS_PER_BYTE = 8
    PER_BLOCK_16_ELEMS = 16
    DUP_MASK = 128
    B32_MASK = 64
    REPEAT_LIMIT = 255
    STRIDE_LIMIT = 65535
    MTE_PAD_STRIDE_LIMIT = 2**32 - 1
    MTE_REPEAT_LIMIT = 4095
    MAX_INT64_VALUE = 2**64 - 1
    TILING_SPACE = ("int64", 88, 128 * 8)
    TILING_LAST_STRIDE_IS_ONE = 3000
    TILING_LAST_DIM_IS_LARGE = 3001
    TILING_LAST_DIM_IS_SMALL = 3002
    TILING_INPUT_OR_OUTPUT_IS_ALL_IN = 3003
    TILING_LAST_LARGE_DIM_LARGE_STRIDE = 3004
    TILING_LAST_SMALL_DIM_LARGE_STRIDE = 3005
    TILING_LAST_TWO_DIM_IS_LARGE = 3006
    TILING_LAST_STRIDE_IS_ZERO_SIZE_IS_LARGE = 3007
    TILING_LAST_STRIDE_IS_ZERO_SIZE_IS_SMALL = 3008
    TILING_FIRST_STRIDE_IS_SMALL = 3009
    VNC_ROWS = 16
    DIM_STRIDE = 3
    B16_BYTES = 2
    B32_BYTES = 4
    B64_BYTES = 8
    ADDR_INDEX = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    SCALAR_INDEX = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)

    # 'pylint: disable=too-many-statements
    def __init__(self, tik_inst, core_num, x_dtype, tensor_list):
        data_in, data_out, tiling_gm, tiling_ub, tiling_params = tensor_list
        self.tik_inst = tik_inst
        self.core_num = core_num
        self.x_dtype = x_dtype
        self.dtype_bytes = get_bit_len(self.x_dtype) // AsStrided.BITS_PER_BYTE
        self.max_elem_cnt = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) -
                             AsStrided.TILING_SPACE[2]) // self.dtype_bytes
        self.ele_per_block = AsStrided.BYTES_PER_BLOCK // self.dtype_bytes
        self.data_in = data_in
        self.data_out = data_out
        self.tiling_gm = tiling_gm
        self.data_ub = None
        self.tiling_ub = tiling_ub
        self.tiling_params = tiling_params
        self.tiling_mode, self.used_core_cnt, self.out_ub_offset, self.vnc_col_size = tiling_params[:4]
        self.axis_1_burst_unit, self.axis_1_lp_unit, self.axis_0_lp_unit = tiling_params[4:7]
        self.mc_pos, self.core_step_in = tiling_params[7:9]
        self.nlc_axis_1_lp_cnt, self.nlc_axis_1_lp_left = tiling_params[9:11]
        self.lc_axis_1_lp_cnt, self.lc_axis_1_lp_left = tiling_params[11:13]
        self.nlc_axis_0_lp_cnt, self.nlc_axis_0_lp_left = tiling_params[13:15]
        self.lc_axis_0_lp_cnt, self.lc_axis_0_lp_left = tiling_params[15:17]
        self.storage_offset, self.last_dim_size, self.last_dim_stride = tiling_params[17:20]
        self.rsecond_dim_size, self.rsecond_dim_stride, self.out_lp_step = tiling_params[20:23]
        self.nfirst_cnt_per_row, self.dim_num = tiling_params[23:25]
        self.dim_info_beg = 25
        self.rsize_reg = self.tik_inst.Scalar(name="rsize_reg")
        self.size_reg = self.tik_inst.Scalar(name="size_reg")
        self.stride_reg = self.tik_inst.Scalar(name="stride_reg")
        self.axis_0_plp_size = self.tik_inst.Scalar(name="axis_0_plp_size")
        self.axis_1_plp_size = self.tik_inst.Scalar(name="axis_1_plp_size")
        self.axis_1_plp_burst = self.tik_inst.Scalar(name="axis_1_plp_burst")
        self.axis_0_input_idx = self.tik_inst.Scalar(name="axis_0_input_idx")
        self.axis_0_cur_idx = self.tik_inst.Scalar(name="axis_0_cur_idx")
        self.axis_1_cur_idx = self.tik_inst.Scalar(name="axis_1_cur_idx")
        self.burst_len = self._ceil_div(self.axis_1_plp_burst, self.ele_per_block)
        self.burst_elems = self.burst_len * self.ele_per_block
        self.vnc_src_stride = self.tik_inst.Scalar(name="vnc_src_stride")
        self.vnc_dst_stride = self.tik_inst.Scalar(name="vnc_dst_stride")
        self.dtype_factor = self._ceil_div(AsStrided.PER_BLOCK_16_ELEMS, self.ele_per_block)
        self.scalar_reg = [self.tik_inst.Scalar(dtype=data_in.dtype) for i in AsStrided.SCALAR_INDEX]
        self.reorder_lp_cnt = self.tik_inst.Scalar(name="reorder_lp_cnt")
        self.burst_valid_elems = self.tik_inst.Scalar(name="burst_valid_elems")
        self.reorder_src_stride = self.tik_inst.Scalar(name="reorder_src_stride")
        self.reorder_gap = self.tik_inst.Scalar(name="reorder_gap")
        self.is_axis_0_back = self.tik_inst.Scalar(name="is_axis_0_back", init_value=0)
        self.axis_0_backend = self.tik_inst.Scalar(name="axis_0_backend", init_value=0)
        self.is_axis_1_back = self.tik_inst.Scalar(name="is_axis_1_back", init_value=0)
        self.axis_1_backend = self.tik_inst.Scalar(name="axis_1_backend", init_value=0)
        self.branch_5_stride = self.tik_inst.Scalar(name="branch_5_stride", init_value=0)
        self.r2nd_head = self.tik_inst.Scalar(name="r2nd_head", init_value=0)
        self.last_dim_in_axis_0 = self.tik_inst.Scalar(name="last_dim_in_axis_0")
        self.last_stride_in_axis_0 = self.tik_inst.Scalar(name="last_stride_in_axis_0")
        self.last_dim_in_axis_0.set_as(self.tiling_ub[self.dim_info_beg - 1 + self.dim_num * self.DIM_STRIDE - 1])
        self.last_stride_in_axis_0.set_as(self.tiling_ub[self.dim_info_beg - 1 + self.dim_num * self.DIM_STRIDE])
        self.vg_offset_tensor = None
        self.vg_factor = self.B32_BYTES // self.dtype_bytes
        self.vg_ub_beg_pos = self.vnc_col_size // self.vg_factor
        self.vg_lp_unit = self.B32_MASK * self.vg_factor
        self.vg_rep_stride = self.vg_lp_unit * self.dtype_bytes // self.BYTES_PER_BLOCK
        self.vg_cond = (tbe_platform.api_check_support("tik.vgather")
                        and self.dtype_bytes in (self.B16_BYTES, self.B32_BYTES))
        self.mte_pad_cond = tbe_platform.api_check_support("tik.data_move_pad")
        self.m_data_in = self.data_in
        self.m_data_out = self.data_out
        self.m_data_ub = None
        self.m_offset_factor = 1

    def _get_axis_0_idx(self, block_idx, axis_0_lp_idx):
        """
        get output axis 0 current position
        """
        with self.tik_inst.if_scope(self.mc_pos == 0):
            self.axis_0_cur_idx.set_as(axis_0_lp_idx * self.axis_0_lp_unit - self.is_axis_0_back * self.axis_0_backend +
                                       block_idx * self.core_step_in)
        with self.tik_inst.else_scope():
            self.axis_0_cur_idx.set_as(axis_0_lp_idx * self.axis_0_lp_unit - self.is_axis_0_back * self.axis_0_backend)

    def _get_axis_1_idx(self, block_idx, axis_1_lp_idx):
        """
        get output axis 1 current position
        """
        with self.tik_inst.if_scope(self.mc_pos == 1):
            self.axis_1_cur_idx.set_as(axis_1_lp_idx * self.axis_1_lp_unit - self.is_axis_1_back * self.axis_1_backend +
                                       block_idx * self.core_step_in)
        with self.tik_inst.else_scope():
            self.axis_1_cur_idx.set_as(axis_1_lp_idx * self.axis_1_lp_unit - self.is_axis_1_back * self.axis_1_backend)

    # 'pylint: disable=too-many-locals, too-many-statements
    def _get_axis_0_elem_idx(self, axis_0_cur_idx):
        """
        get index in input shape for each axis 0 elements
        """
        # the factors order is: rsize0, size0, stride0, rsize1, size1, stride1, ...
        idx_0_offset_list = []
        gap = AsStrided.DIM_STRIDE
        for val in AsStrided.SCALAR_INDEX[2:10]:
            tmp_offset = 0
            for idx in reversed(AsStrided.SCALAR_INDEX[1:val]):
                tmp_offset += (axis_0_cur_idx // self.tiling_params[self.dim_info_beg + idx * gap] %
                               self.tiling_params[self.dim_info_beg + idx * gap + 1] *
                               self.tiling_params[self.dim_info_beg + idx * gap + gap - 1])
            tmp_offset += (axis_0_cur_idx // self.tiling_params[self.dim_info_beg] *
                           self.tiling_params[self.dim_info_beg + gap - 1])
            idx_0_offset_list.append(tmp_offset + self.storage_offset)

        # now suppose the max dimension is 10
        with self.tik_inst.if_scope(self.dim_num == AsStrided.SCALAR_INDEX[1]):
            self.axis_0_input_idx.set_as(axis_0_cur_idx * self.tiling_params[self.dim_info_beg + gap - 1] +
                                         self.storage_offset)
        with self.tik_inst.if_scope(self.dim_num == AsStrided.SCALAR_INDEX[2]):
            self.axis_0_input_idx.set_as(idx_0_offset_list[0])
        with self.tik_inst.if_scope(self.dim_num == AsStrided.SCALAR_INDEX[3]):
            self.axis_0_input_idx.set_as(idx_0_offset_list[1])
        with self.tik_inst.if_scope(self.dim_num == AsStrided.SCALAR_INDEX[4]):
            self.axis_0_input_idx.set_as(idx_0_offset_list[2])
        with self.tik_inst.if_scope(self.dim_num == AsStrided.SCALAR_INDEX[5]):
            self.axis_0_input_idx.set_as(idx_0_offset_list[3])
        with self.tik_inst.if_scope(self.dim_num == AsStrided.SCALAR_INDEX[6]):
            self.axis_0_input_idx.set_as(idx_0_offset_list[4])
        with self.tik_inst.if_scope(self.dim_num == AsStrided.SCALAR_INDEX[7]):
            self.axis_0_input_idx.set_as(idx_0_offset_list[5])
        with self.tik_inst.if_scope(self.dim_num == AsStrided.SCALAR_INDEX[8]):
            self.axis_0_input_idx.set_as(idx_0_offset_list[6])
        with self.tik_inst.if_scope(self.dim_num == AsStrided.SCALAR_INDEX[9]):
            self.axis_0_input_idx.set_as(idx_0_offset_list[7])

    @staticmethod
    def _ceil_div(val_x, val_y):
        """
        ceiling division
        """
        return (val_x + val_y - 1) // val_y

    @staticmethod
    def _floor_div(val_x, val_y):
        """
        floor division
        """
        return val_x // val_y

    def _compute_branch_func(self, args, block_idx):
        """
        compute function branch
        """
        with self.tik_inst.if_scope(self.tiling_mode == AsStrided.TILING_LAST_STRIDE_IS_ONE):
            self._compute_1_func(args, block_idx)
        with self.tik_inst.if_scope(self.tiling_mode == AsStrided.TILING_LAST_DIM_IS_LARGE):
            self._compute_2_func(args, block_idx)
        with self.tik_inst.if_scope(self.tiling_mode == AsStrided.TILING_LAST_DIM_IS_SMALL):
            self._compute_3_func(args, block_idx)
        with self.tik_inst.if_scope(self.tiling_mode == AsStrided.TILING_INPUT_OR_OUTPUT_IS_ALL_IN):
            self._compute_4_func(args, block_idx)
        with self.tik_inst.if_scope(self.tiling_mode == AsStrided.TILING_LAST_LARGE_DIM_LARGE_STRIDE):
            self._compute_5_func(args, block_idx)
        with self.tik_inst.if_scope(self.tiling_mode == AsStrided.TILING_LAST_SMALL_DIM_LARGE_STRIDE):
            self._compute_6_func(args, block_idx)
        with self.tik_inst.if_scope(self.tiling_mode == AsStrided.TILING_LAST_TWO_DIM_IS_LARGE):
            self._compute_7_func(args, block_idx)
        with self.tik_inst.if_scope(self.tiling_mode == AsStrided.TILING_LAST_STRIDE_IS_ZERO_SIZE_IS_LARGE):
            self._compute_8_func(args, block_idx)
        with self.tik_inst.if_scope(self.tiling_mode == AsStrided.TILING_LAST_STRIDE_IS_ZERO_SIZE_IS_SMALL):
            self._compute_9_func(args, block_idx)
        with self.tik_inst.if_scope(self.tiling_mode == AsStrided.TILING_FIRST_STRIDE_IS_SMALL):
            self._compute_10_func(args, block_idx)

    def _compute_func(self, block_idx):
        """
        compute function entrance
        """
        axis_0_lp_cnt_reg = self.tik_inst.Scalar(name="axis_0_lp_cnt_reg")
        axis_0_lp_left_reg = self.tik_inst.Scalar(name="axis_0_lp_left_reg")
        axis_1_lp_cnt_reg = self.tik_inst.Scalar(name="axis_1_lp_cnt_reg")
        axis_1_lp_left_reg = self.tik_inst.Scalar(name="axis_1_lp_left_reg")
        with self.tik_inst.if_scope(block_idx != self.used_core_cnt - 1):
            axis_0_lp_cnt_reg.set_as(self.nlc_axis_0_lp_cnt)
            axis_0_lp_left_reg.set_as(self.nlc_axis_0_lp_left)
            axis_1_lp_cnt_reg.set_as(self.nlc_axis_1_lp_cnt)
            axis_1_lp_left_reg.set_as(self.nlc_axis_1_lp_left)
        with self.tik_inst.else_scope():
            axis_0_lp_cnt_reg.set_as(self.lc_axis_0_lp_cnt)
            axis_0_lp_left_reg.set_as(self.lc_axis_0_lp_left)
            axis_1_lp_cnt_reg.set_as(self.lc_axis_1_lp_cnt)
            axis_1_lp_left_reg.set_as(self.lc_axis_1_lp_left)

        axis_args = (axis_0_lp_cnt_reg, axis_0_lp_left_reg,
                     axis_1_lp_cnt_reg, axis_1_lp_left_reg)
        self.data_ub = self.tik_inst.Tensor(self.x_dtype, (self.max_elem_cnt,), tik.scope_ubuf, "data_ub")
        self.m_data_ub = self.data_ub
        self.vg_offset_tensor = self.data_ub.reinterpret_cast_to("int32")

        self._compute_branch_func(axis_args, block_idx)

    def _compute_data_move_in(self, ub_step_offset, gm_step_offset):
        """
        copy data from gm to ub for common branch
        """

        def _inner_process(lp_idx, lp_unit, head=0):
            axis_0_cur_idx = lp_idx * self.last_dim_in_axis_0 + head + self.axis_0_cur_idx
            self._get_axis_0_elem_idx(axis_0_cur_idx)
            ub_stride = (ub_step_offset - self.burst_elems) // self.ele_per_block
            if self.mte_pad_cond:
                if self.dtype_bytes == self.B64_BYTES:
                    self.m_data_in = self.data_in.reinterpret_cast_to("int8")
                    self.m_data_ub = self.data_ub.reinterpret_cast_to("int8")
                    self.m_offset_factor = self.B64_BYTES
                gm_stride = (self.last_stride_in_axis_0 - self.axis_1_plp_burst) * self.dtype_bytes
                with self.tik_inst.if_scope(
                        tik.all(gm_stride > 0, gm_stride <= self.MTE_PAD_STRIDE_LIMIT,
                                ub_stride >= 0, lp_unit <= self.MTE_REPEAT_LIMIT)):
                    self.tik_inst.data_move_pad(
                        self.m_data_ub[(lp_idx * self.last_dim_in_axis_0 + head) * ub_step_offset *
                                       self.m_offset_factor],
                        self.m_data_in[(self.axis_0_input_idx + self.axis_1_cur_idx * gm_step_offset) *
                                       self.m_offset_factor], lp_unit, self.axis_1_plp_burst * self.dtype_bytes,
                        ub_stride, gm_stride)
                with self.tik_inst.else_scope():
                    with self.tik_inst.for_range(0, lp_unit) as ori_r2nd_idx:
                        self.tik_inst.data_move_pad(
                            self.m_data_ub[(lp_idx * self.last_dim_in_axis_0 + ori_r2nd_idx + head)
                                           * ub_step_offset * self.m_offset_factor],
                            self.m_data_in[(self.axis_0_input_idx + ori_r2nd_idx * self.last_stride_in_axis_0 +
                                            self.axis_1_cur_idx * gm_step_offset) * self.m_offset_factor],
                            1, self.axis_1_plp_burst * self.dtype_bytes, 0, 0)
            else:
                gm_stride = (self.last_stride_in_axis_0 - self.burst_elems) // self.ele_per_block
                with self.tik_inst.if_scope(
                        tik.all(gm_stride > 0, self.last_stride_in_axis_0 % self.ele_per_block == 0,
                                ub_stride >= 0, lp_unit <= self.MTE_REPEAT_LIMIT, gm_stride <= self.STRIDE_LIMIT)):
                    self.tik_inst.data_move(self.data_ub[(lp_idx * self.last_dim_in_axis_0 + head) * ub_step_offset],
                                            self.data_in[self.axis_0_input_idx + self.axis_1_cur_idx * gm_step_offset],
                                            0, lp_unit, self.burst_len, gm_stride, ub_stride)
                with self.tik_inst.else_scope():
                    with self.tik_inst.for_range(0, lp_unit) as ori_r2nd_idx:
                        self.tik_inst.data_move(
                            self.data_ub[(lp_idx * self.last_dim_in_axis_0 + ori_r2nd_idx + head) * ub_step_offset],
                            self.data_in[self.axis_0_input_idx + ori_r2nd_idx * self.last_stride_in_axis_0 +
                                         self.axis_1_cur_idx * gm_step_offset], 0, 1, self.burst_len, 0, 0)

        with self.tik_inst.if_scope(self.axis_0_lp_unit > self.last_dim_in_axis_0):
            self.r2nd_head.set_as(self.last_dim_in_axis_0 - self.axis_0_cur_idx % self.last_dim_in_axis_0)
            ori_r2nd_dim_lp_cnt = (self.axis_0_plp_size - self.r2nd_head) // self.last_dim_in_axis_0
            ori_r2nd_plp_left_size = (self.axis_0_plp_size - self.r2nd_head) % self.last_dim_in_axis_0
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.if_scope(self.r2nd_head > 0):
                    _inner_process(0, self.r2nd_head)
                with self.tik_inst.for_range(0, ori_r2nd_dim_lp_cnt) as lp_idx:
                    _inner_process(lp_idx, self.last_dim_in_axis_0, self.r2nd_head)
                with self.tik_inst.if_scope(ori_r2nd_plp_left_size > 0):
                    _inner_process(ori_r2nd_dim_lp_cnt, ori_r2nd_plp_left_size, self.r2nd_head)
        with self.tik_inst.else_scope():
            self.r2nd_head.set_as(self.axis_0_cur_idx % self.last_dim_in_axis_0)
            ori_r2nd_plp_left_size = (self.axis_0_plp_size + self.r2nd_head) % self.last_dim_in_axis_0
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.if_scope(self.axis_0_plp_size + self.r2nd_head <= self.last_dim_in_axis_0):
                    _inner_process(0, self.axis_0_plp_size)
                with self.tik_inst.else_scope():
                    _inner_process(0, self.last_dim_in_axis_0 - self.r2nd_head)
                    _inner_process(0, ori_r2nd_plp_left_size, self.last_dim_in_axis_0 - self.r2nd_head)

    def _compute_large_stride_data_move_in(self, ub_step_offset):
        with self.tik_inst.if_scope(self.last_dim_stride != 0):
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, self.axis_0_plp_size) as axis_0_idx:
                    axis_0_cur_idx = axis_0_idx + self.axis_0_cur_idx
                    self._get_axis_0_elem_idx(axis_0_cur_idx)
                    if self.mte_pad_cond:
                        if self.dtype_bytes == self.B64_BYTES:
                            self.m_data_in = self.data_in.reinterpret_cast_to("int8")
                            self.m_data_ub = self.data_ub.reinterpret_cast_to("int8")
                            self.m_offset_factor = self.B64_BYTES
                        src_stride = (self.last_dim_stride - 1) * self.dtype_bytes
                        with self.tik_inst.if_scope(src_stride <= self.MTE_PAD_STRIDE_LIMIT):
                            gm_in_offset = (self.axis_0_input_idx + self.axis_1_cur_idx * self.last_dim_stride)
                            ub_in_offset = axis_0_idx * ub_step_offset
                            self.tik_inst.data_move_pad(self.m_data_ub[ub_in_offset * self.m_offset_factor],
                                                        self.m_data_in[gm_in_offset * self.m_offset_factor],
                                                        self.axis_1_plp_size, self.dtype_bytes, 0, src_stride)
                    else:
                        src_stride = self.last_dim_stride // self.ele_per_block - 1
                        with self.tik_inst.if_scope(
                                tik.all(self.last_dim_stride % self.ele_per_block == 0,
                                        src_stride <= AsStrided.STRIDE_LIMIT)):
                            gm_in_offset = self.axis_0_input_idx + self.axis_1_cur_idx * self.last_dim_stride
                            ub_in_offset = axis_0_idx * ub_step_offset
                            self.tik_inst.data_move(self.data_ub[ub_in_offset], self.data_in[gm_in_offset], 0,
                                                    self.axis_1_plp_size, self.burst_len, src_stride, 0)
                    with self.tik_inst.else_scope():
                        with self.tik_inst.for_range(0, self.axis_1_plp_size) as axis_1_idx:
                            gm_in_offset_1 = axis_1_idx * self.last_dim_stride + gm_in_offset
                            ub_in_offset_1 = axis_1_idx * self.burst_elems + ub_in_offset
                            self.tik_inst.data_move(self.data_ub[ub_in_offset_1], self.data_in[gm_in_offset_1], 0, 1,
                                                    self.burst_len, 0, 0)
        with self.tik_inst.else_scope():
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, self.axis_0_plp_size) as axis_0_idx:
                    axis_0_cur_idx = axis_0_idx + self.axis_0_cur_idx
                    self._get_axis_0_elem_idx(axis_0_cur_idx)
                    gm_in_offset = self.axis_0_input_idx
                    ub_in_offset = axis_0_idx * ub_step_offset
                    self.tik_inst.data_move(self.data_ub[ub_in_offset], self.data_in[gm_in_offset], 0, 1,
                                            self.burst_len, 0, 0)
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                data_ub = self.data_ub.reinterpret_cast_to("int16")
                with self.tik_inst.for_range(0, self.axis_0_plp_size) as axis_0_idx:
                    ub_in_offset = axis_0_idx * ub_step_offset // self.ele_per_block * self.PER_BLOCK_16_ELEMS
                    ub_in_offset_1 = self.burst_len * self.PER_BLOCK_16_ELEMS + ub_in_offset
                    self.tik_inst.vor(self.PER_BLOCK_16_ELEMS, data_ub[ub_in_offset_1], data_ub[ub_in_offset],
                                      data_ub[ub_in_offset], self.axis_1_plp_size - 1, 1, 1, 1, self.burst_len, 0, 0)

    def _compute_first_stride_small_data_move_in(self):
        """
        copy data from gm to ub for compute 10
        """
        nfirst_lp_cnt = self.axis_0_plp_size // self.nfirst_cnt_per_row
        nfirst_left = self.axis_0_plp_size % self.nfirst_cnt_per_row

        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, nfirst_lp_cnt) as lp_idx:
                with self.tik_inst.for_range(0, self.nfirst_cnt_per_row) as axis_0_idx:
                    axis_0_cur_idx = axis_0_idx + self.axis_0_cur_idx + lp_idx * self.nfirst_cnt_per_row
                    self._get_axis_0_elem_idx(axis_0_cur_idx)
                    ub_offset = axis_0_idx * self.burst_elems + lp_idx * self.vnc_col_size
                    gm_offset = self.axis_0_input_idx + self.axis_1_cur_idx * self.last_dim_stride
                    self.tik_inst.data_move(self.data_ub[ub_offset], self.data_in[gm_offset],
                                            0, 1, self.burst_len, 0, 0)

        with self.tik_inst.if_scope(nfirst_left > 0):
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, nfirst_left) as axis_0_idx:
                    axis_0_cur_idx = axis_0_idx + self.axis_0_cur_idx + nfirst_lp_cnt * self.nfirst_cnt_per_row
                    self._get_axis_0_elem_idx(axis_0_cur_idx)
                    ub_offset = axis_0_idx * self.burst_elems + nfirst_lp_cnt * self.vnc_col_size
                    gm_offset = self.axis_0_input_idx + self.axis_1_cur_idx * self.last_dim_stride
                    self.tik_inst.data_move(self.data_ub[ub_offset], self.data_in[gm_offset],
                                            0, 1, self.burst_len, 0, 0)
    
    def _compute_first_stride_small_data_move_in_with_dmp(self):
        """
        copy data from gm to ub for compute 10 with data_move_pad
        """
        nfirst_lp_cnt = self.axis_0_plp_size // self.nfirst_cnt_per_row
        nfirst_left = self.axis_0_plp_size % self.nfirst_cnt_per_row
        if self.dtype_bytes == self.B64_BYTES:
            self.m_data_in = self.data_in.reinterpret_cast_to("int8")
            self.m_data_ub = self.data_ub.reinterpret_cast_to("int8")
            self.m_offset_factor = self.B64_BYTES

        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, nfirst_lp_cnt) as lp_idx:
                with self.tik_inst.for_range(0, self.nfirst_cnt_per_row) as axis_0_idx:
                    axis_0_cur_idx = axis_0_idx + self.axis_0_cur_idx + lp_idx * self.nfirst_cnt_per_row
                    self._get_axis_0_elem_idx(axis_0_cur_idx)
                    ub_offset = (axis_0_idx * self.burst_elems + lp_idx * self.vnc_col_size) * self.m_offset_factor
                    gm_offset = (self.axis_0_input_idx + self.axis_1_cur_idx *
                                 self.last_dim_stride) * self.m_offset_factor
                    self.tik_inst.data_move_pad(self.m_data_ub[ub_offset], self.m_data_in[gm_offset],
                                                1, self.axis_1_plp_burst * self.dtype_bytes, 0, 0)

        with self.tik_inst.if_scope(nfirst_left > 0):
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, nfirst_left) as axis_0_idx:
                    axis_0_cur_idx = axis_0_idx + self.axis_0_cur_idx + nfirst_lp_cnt * self.nfirst_cnt_per_row
                    self._get_axis_0_elem_idx(axis_0_cur_idx)
                    ub_offset = (axis_0_idx * self.burst_elems + nfirst_lp_cnt *
                                 self.vnc_col_size) * self.m_offset_factor
                    gm_offset = (self.axis_0_input_idx + self.axis_1_cur_idx *
                                 self.last_dim_stride) * self.m_offset_factor
                    self.tik_inst.data_move_pad(self.m_data_ub[ub_offset], self.m_data_in[gm_offset],
                                                1, self.axis_1_plp_burst * self.dtype_bytes, 0, 0)

    # 'pylint: disable=too-many-arguments
    def _compute_data_move_out(self, axis_0_lp, valid_elems_cnt, data_gap, axis_0_cur_idx, axis_1_cur_idx):
        """
        copy data from ub to gm for all compute branch
        """
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            elems_mod = valid_elems_cnt % self.ele_per_block
            elems_ceil_blocks = self._ceil_div(valid_elems_cnt, self.ele_per_block)
            elems_floor_blocks = self._floor_div(valid_elems_cnt, self.ele_per_block)

            with self.tik_inst.if_scope(
                    tik.all(axis_0_lp > 1, axis_0_lp <= self.MTE_REPEAT_LIMIT, elems_mod == 0,
                            self.out_lp_step % self.ele_per_block == 0,
                            (self.out_lp_step - valid_elems_cnt) // self.ele_per_block <= self.STRIDE_LIMIT)):
                self.tik_inst.data_move(self.data_out[axis_0_cur_idx * self.out_lp_step + axis_1_cur_idx],
                                        self.data_ub[self.out_ub_offset], 0, axis_0_lp, elems_ceil_blocks,
                                        (data_gap - valid_elems_cnt) // self.ele_per_block,
                                        (self.out_lp_step - valid_elems_cnt) // self.ele_per_block)
            with self.tik_inst.else_scope():
                with self.tik_inst.for_range(0, axis_0_lp) as axis_0_idx:
                    gm_out_offset = (axis_0_idx + axis_0_cur_idx) * self.out_lp_step + axis_1_cur_idx
                    ub_out_offset = self.out_ub_offset + axis_0_idx * data_gap
                    if self.mte_pad_cond:
                        if self.dtype_bytes == self.B64_BYTES:
                            self.m_data_out = self.data_out.reinterpret_cast_to("int8")
                            self.m_data_ub = self.data_ub.reinterpret_cast_to("int8")
                            self.m_offset_factor = self.B64_BYTES
                        self.tik_inst.data_move_pad(self.m_data_out[gm_out_offset * self.m_offset_factor],
                                                    self.m_data_ub[ub_out_offset * self.m_offset_factor],
                                                    1, valid_elems_cnt * self.dtype_bytes, 0, 0)
                    else:
                        with self.tik_inst.if_scope(tik.any(valid_elems_cnt <= self.ele_per_block, elems_mod == 0)):
                            self.tik_inst.data_move(self.data_out[gm_out_offset], self.data_ub[ub_out_offset],
                                                    0, 1, elems_ceil_blocks, 0, 0)
                        with self.tik_inst.else_scope():
                            self.tik_inst.data_move(self.data_out[gm_out_offset], self.data_ub[ub_out_offset],
                                                    0, 1, elems_floor_blocks, 0, 0)
                            gm_out_offset_1 = gm_out_offset + valid_elems_cnt - self.ele_per_block
                            ub_out_offset_1 = ub_out_offset + elems_floor_blocks * self.ele_per_block
                            self.tik_inst.data_move(self.data_out[gm_out_offset_1], self.data_ub[ub_out_offset_1],
                                                    0, 1, 1, 0, 0)

    def _get_axis_0_plp_size(self, block_idx, axis_0_lp_idx, axis_0_lp_cnt, axis_0_lp_left):
        """
        get axis 0 per loop parameters
        """
        with self.tik_inst.if_scope(tik.any(axis_0_lp_idx != axis_0_lp_cnt - 1, axis_0_lp_left == 0)):
            with self.tik_inst.if_scope(tik.all(tik.any(tik.all(axis_0_lp_left > 0, axis_0_lp_idx == axis_0_lp_cnt - 2),
                                                        tik.all(block_idx == self.used_core_cnt - 2,
                                                                axis_0_lp_idx == axis_0_lp_cnt - 1,
                                                                self.lc_axis_0_lp_cnt == 1,
                                                                self.lc_axis_0_lp_left > 0)),
                                                self.out_lp_step < self.ele_per_block,
                                                self.lc_axis_0_lp_left * self.out_lp_step < self.ele_per_block)):
                self.axis_0_plp_size.set_as(self.axis_0_lp_unit - self.axis_0_backend)
            with self.tik_inst.else_scope():
                self.axis_0_plp_size.set_as(self.axis_0_lp_unit)
            self.is_axis_0_back.set_as(0)

        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(tik.all(tik.any(self.used_core_cnt > 1, axis_0_lp_cnt > 1),
                                                self.out_lp_step < self.ele_per_block,
                                                axis_0_lp_left * self.out_lp_step < self.ele_per_block)):
                self.axis_0_plp_size.set_as(axis_0_lp_left + self.axis_0_backend)
                self.is_axis_0_back.set_as(1)
            with self.tik_inst.else_scope():
                self.axis_0_plp_size.set_as(axis_0_lp_left)
                self.is_axis_0_back.set_as(0)

    def _get_axis_1_plp_size(self, block_idx, axis_1_lp_idx, axis_1_lp_cnt, axis_1_lp_left):
        """
        get axis 1 per loop parameters
        """
        with self.tik_inst.if_scope(tik.any(axis_1_lp_idx != axis_1_lp_cnt - 1, axis_1_lp_left == 0)):
            with self.tik_inst.if_scope(tik.all(tik.any(tik.all(axis_1_lp_idx == axis_1_lp_cnt - 2, axis_1_lp_left > 0),
                                                        tik.all(block_idx == self.used_core_cnt - 2,
                                                                axis_1_lp_idx == axis_1_lp_cnt - 1,
                                                                self.lc_axis_1_lp_cnt == 1,
                                                                self.lc_axis_1_lp_left > 0)),
                                                self.lc_axis_1_lp_left < self.ele_per_block)):
                self.axis_1_plp_size.set_as(self.axis_1_lp_unit - self.axis_1_backend)
            with self.tik_inst.else_scope():
                self.axis_1_plp_size.set_as(self.axis_1_lp_unit)
            self.is_axis_1_back.set_as(0)

        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(tik.all(tik.any(self.used_core_cnt > 1, axis_1_lp_cnt > 1),
                                                axis_1_lp_left > 0, axis_1_lp_left < self.ele_per_block)):
                self.axis_1_plp_size.set_as(axis_1_lp_left + self.axis_1_backend)
                self.is_axis_1_back.set_as(1)
            with self.tik_inst.else_scope():
                self.axis_1_plp_size.set_as(axis_1_lp_left)
                self.is_axis_1_back.set_as(0)

    def _compute_9_get_axis_1_plp_size(self, block_idx, axis_1_lp_idx, axis_1_lp_cnt, axis_1_lp_left):
        """
        get axis 1 per loop parameters for compute 9
        """
        with self.tik_inst.if_scope(tik.any(axis_1_lp_idx != axis_1_lp_cnt - 1, axis_1_lp_left == 0)):
            with self.tik_inst.if_scope(tik.all(tik.any(tik.all(axis_1_lp_idx == axis_1_lp_cnt - 2, axis_1_lp_left > 0),
                                                        tik.all(block_idx == self.used_core_cnt - 2,
                                                                axis_1_lp_idx == axis_1_lp_cnt - 1,
                                                                self.lc_axis_1_lp_cnt == 1,
                                                                self.lc_axis_1_lp_left > 0)),
                                                self.lc_axis_1_lp_left * self.last_dim_size < self.ele_per_block)):
                self.axis_1_plp_size.set_as(self.axis_1_lp_unit - self.axis_1_backend)
            with self.tik_inst.else_scope():
                self.axis_1_plp_size.set_as(self.axis_1_lp_unit)
            self.is_axis_1_back.set_as(0)

        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(tik.all(tik.any(self.used_core_cnt > 1, axis_1_lp_cnt > 1),
                                                axis_1_lp_left > 0,
                                                axis_1_lp_left * self.last_dim_size < self.ele_per_block)):
                self.axis_1_plp_size.set_as(axis_1_lp_left + self.axis_1_backend)
                self.is_axis_1_back.set_as(1)
            with self.tik_inst.else_scope():
                self.axis_1_plp_size.set_as(axis_1_lp_left)
                self.is_axis_1_back.set_as(0)

    def _compute_10_get_axis_0_plp_size(self, block_idx, axis_0_lp_idx, axis_0_lp_cnt, axis_0_lp_left):
        """
        get axis 1 per loop parameters
        """
        with self.tik_inst.if_scope(tik.any(axis_0_lp_idx != axis_0_lp_cnt - 1, axis_0_lp_left == 0)):
            with self.tik_inst.if_scope(tik.all(tik.any(tik.all(axis_0_lp_idx == axis_0_lp_cnt - 2, axis_0_lp_left > 0),
                                                        tik.all(block_idx == self.used_core_cnt - 2,
                                                                axis_0_lp_idx == axis_0_lp_cnt - 1,
                                                                self.lc_axis_0_lp_cnt == 1,
                                                                self.lc_axis_0_lp_left > 0)),
                                                self.lc_axis_0_lp_left < self.ele_per_block)):
                self.axis_0_plp_size.set_as(self.axis_0_lp_unit - self.axis_0_backend)
            with self.tik_inst.else_scope():
                self.axis_0_plp_size.set_as(self.axis_0_lp_unit)
            self.is_axis_0_back.set_as(0)

        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(tik.all(tik.any(self.used_core_cnt > 1, axis_0_lp_cnt > 1),
                                                axis_0_lp_left > 0, axis_0_lp_left < self.ele_per_block)):
                self.axis_0_plp_size.set_as(axis_0_lp_left + self.axis_0_backend)
                self.is_axis_0_back.set_as(1)
            with self.tik_inst.else_scope():
                self.axis_0_plp_size.set_as(axis_0_lp_left)
                self.is_axis_0_back.set_as(0)

    def _adjust_tail_elems(self, loop_cnt, data_len, data_gap):
        """
        adjust tail elements to block align
        """
        if self.mte_pad_cond is False:
            data_block_align = self._floor_div(data_len, self.ele_per_block) * self.ele_per_block
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.if_scope(tik.all(data_len % self.ele_per_block > 0, data_len > self.ele_per_block)):
                    with self.tik_inst.for_range(0, loop_cnt) as lp_idx:
                        data_offset = lp_idx * data_gap + self.out_ub_offset
                        for i in AsStrided.SCALAR_INDEX[:self.ele_per_block]:
                            self.scalar_reg[i].set_as(self.data_ub[data_offset + data_len - self.ele_per_block + i])
                        for j in AsStrided.SCALAR_INDEX[:self.ele_per_block]:
                            self.data_ub[data_offset + data_block_align + j].set_as(self.scalar_reg[j])

    def _set_vnchwconv_stride(self, repeat_cnt, src_val, dst_val):
        """
        set source and target stride for vnchwconv
        """
        with self.tik_inst.if_scope(repeat_cnt == 1):
            self.vnc_src_stride.set_as(0)
            self.vnc_dst_stride.set_as(0)
        with self.tik_inst.else_scope():
            self.vnc_src_stride.set_as(src_val)
            self.vnc_dst_stride.set_as(dst_val)

    def _move_to_target_layout(self):
        """
        move elements to target layout
        """
        with self.tik_inst.if_scope(self.last_dim_stride != 0):
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, self.reorder_lp_cnt) as burst_idx:
                    target_offset = burst_idx * self.burst_valid_elems * self.dtype_factor * self.ele_per_block
                    source_offset = (self.out_ub_offset +
                                     burst_idx * self.reorder_gap * self.dtype_factor * self.ele_per_block)
                    self.tik_inst.data_move(self.data_ub[target_offset], self.data_ub[source_offset],
                                            0, self.burst_valid_elems, self.dtype_factor,
                                            self.reorder_src_stride * self.dtype_factor, 0)
        with self.tik_inst.else_scope():
            data_ub = self.data_ub.reinterpret_cast_to("int16")
            lp_cnt = self.burst_valid_elems // AsStrided.REPEAT_LIMIT
            lp_left = self.burst_valid_elems % AsStrided.REPEAT_LIMIT
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, self.reorder_lp_cnt) as burst_idx:
                    target_offset = (burst_idx * self.burst_valid_elems *
                                     self.dtype_factor * AsStrided.PER_BLOCK_16_ELEMS)
                    source_offset = (self.out_ub_offset * AsStrided.PER_BLOCK_16_ELEMS // self.ele_per_block +
                                     burst_idx * self.reorder_gap * self.dtype_factor * AsStrided.PER_BLOCK_16_ELEMS)
                    with self.tik_inst.for_range(0, lp_cnt) as lp_idx:
                        target_offset_1 = (target_offset + lp_idx * AsStrided.REPEAT_LIMIT *
                                           self.dtype_factor * AsStrided.PER_BLOCK_16_ELEMS)
                        self.tik_inst.vor(self.dtype_factor * AsStrided.PER_BLOCK_16_ELEMS, data_ub[target_offset_1],
                                          data_ub[source_offset], data_ub[source_offset], AsStrided.REPEAT_LIMIT,
                                          1, 1, 1, self.dtype_factor, self.branch_5_stride, self.branch_5_stride)
                    with self.tik_inst.if_scope(lp_left > 0):
                        target_offset_2 = (target_offset + lp_cnt * AsStrided.REPEAT_LIMIT *
                                           self.dtype_factor * AsStrided.PER_BLOCK_16_ELEMS)
                        self.tik_inst.vor(self.dtype_factor * AsStrided.PER_BLOCK_16_ELEMS, data_ub[target_offset_2],
                                          data_ub[source_offset], data_ub[source_offset], lp_left,
                                          1, 1, 1, self.dtype_factor, self.branch_5_stride, self.branch_5_stride)

    def _transpose_by_vnchwconv_b16(self, data_len):
        """
        transpose two axises by vnchwconv for b16 dtype
        """
        data_ub_fp16 = self.data_ub.reinterpret_cast_to("float16")
        src_addr_list = [data_ub_fp16[self.vnc_col_size * self.dtype_factor * i] for i in AsStrided.ADDR_INDEX]
        dst_addr_list = [data_ub_fp16[AsStrided.VNC_ROWS * i + self.out_ub_offset * self.dtype_factor]
                         for i in AsStrided.ADDR_INDEX]
        repeat_cnt = self._ceil_div(data_len * self.dtype_factor, AsStrided.VNC_ROWS)
        self._set_vnchwconv_stride(repeat_cnt, 1, 16)
        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list,
                                repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)

    def _transpose_by_vnchwconv_b8(self, data_len):
        """
        transpose two axises by vnchwconv for b8 dtype
        """
        src_addr_list = [self.data_ub[self.vnc_col_size * i] for i in AsStrided.ADDR_INDEX]
        dst_addr_list = [self.data_ub[self.ele_per_block * i + self.out_ub_offset]
                         for i in AsStrided.ADDR_INDEX]
        repeat_cnt = self._ceil_div(data_len, self.ele_per_block)
        self._set_vnchwconv_stride(repeat_cnt, 1, 32)
        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list,
                                repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)
        dst_addr_list = [self.data_ub[self.ele_per_block * (i + AsStrided.VNC_ROWS) + self.out_ub_offset]
                         for i in AsStrided.ADDR_INDEX]
        self.tik_inst.vnchwconv(False, True, dst_addr_list, src_addr_list,
                                repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)

    def _transpose_back_by_vnchwconv_b16(self, valid_data_len):
        """
        transpose two axises back by vnchwconv for b16 dtype
        """
        data_ub_fp16 = self.data_ub.reinterpret_cast_to("float16")
        src_addr_list = [data_ub_fp16[AsStrided.VNC_ROWS * i] for i in AsStrided.ADDR_INDEX]
        dst_addr_list = [
            data_ub_fp16[self.vnc_col_size * self.dtype_factor * i + self.out_ub_offset * self.dtype_factor]
            for i in AsStrided.ADDR_INDEX]
        repeat_cnt = self._ceil_div(valid_data_len * self.dtype_factor, AsStrided.VNC_ROWS)
        self._set_vnchwconv_stride(repeat_cnt, 16, 1)
        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list,
                                repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)

    def _transpose_back_by_vnchwconv_b8(self, valid_data_len):
        """
        transpose two axises back by vnchwconv for b8 dtype
        """
        src_addr_list = [self.data_ub[self.ele_per_block * i] for i in AsStrided.ADDR_INDEX]
        dst_addr_list = [self.data_ub[self.vnc_col_size * i + self.out_ub_offset]
                         for i in AsStrided.ADDR_INDEX]
        repeat_cnt = self._ceil_div(valid_data_len, self.ele_per_block)
        self._set_vnchwconv_stride(repeat_cnt, 32, 1)
        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list,
                                repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)
        src_addr_list = [self.data_ub[self.ele_per_block * (i + AsStrided.VNC_ROWS)] for i in AsStrided.ADDR_INDEX]
        self.tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list,
                                repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)

    def _compute_10_transpose_back_by_vnchwconv_b16(self, valid_data_len):
        """
        transpose two axises back by vnchwconv for compute 10 which dtype is b16
        """
        data_ub_fp16 = self.data_ub.reinterpret_cast_to("float16")
        repeat_cnt = self._ceil_div(valid_data_len * self.dtype_factor, AsStrided.VNC_ROWS)

        with self.tik_inst.for_range(0, self.axis_1_plp_size) as axis_1_idx:
            lp_step = axis_1_idx * self.nfirst_cnt_per_row * self.dtype_factor * AsStrided.VNC_ROWS
            src_addr_list = [data_ub_fp16[i * AsStrided.VNC_ROWS + lp_step] for i in AsStrided.ADDR_INDEX]
            dst_addr_list = [
                data_ub_fp16[(self.nfirst_cnt_per_row * i + self.out_ub_offset) * self.dtype_factor + lp_step]
                for i in AsStrided.ADDR_INDEX]
            self._set_vnchwconv_stride(repeat_cnt, 16, 1)
            self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list,
                                    repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)

    def _compute_10_transpose_back_by_vnchwconv_b8(self, valid_data_len):
        """
        transpose two axises back by vnchwconv for compute 10 which dtype is b8
        """
        repeat_cnt = self._ceil_div(valid_data_len, self.ele_per_block)

        with self.tik_inst.for_range(0, self.axis_1_plp_size) as axis_1_idx:
            src_lp_step = axis_1_idx * self.nfirst_cnt_per_row * self.ele_per_block
            dst_lp_step = axis_1_idx * self.nfirst_cnt_per_row * AsStrided.VNC_ROWS
            src_addr_list = [self.data_ub[self.ele_per_block * i + src_lp_step] for i in AsStrided.ADDR_INDEX]
            dst_addr_list = [self.data_ub[self.nfirst_cnt_per_row * i + self.out_ub_offset + dst_lp_step]
                             for i in AsStrided.ADDR_INDEX]

            self._set_vnchwconv_stride(repeat_cnt, 32, 1)
            self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list,
                                    repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)
            src_addr_list = [self.data_ub[self.ele_per_block * (i + AsStrided.VNC_ROWS) + src_lp_step]
                             for i in AsStrided.ADDR_INDEX]
            self.tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list,
                                    repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)

    def _vnchwconv_scheme(self, data_len, valid_data_len):
        """
        reorder elements by vnchwconv
        """
        if self.ele_per_block != AsStrided.BYTES_PER_BLOCK:
            self._transpose_by_vnchwconv_b16(data_len)
            self._move_to_target_layout()
            self._transpose_back_by_vnchwconv_b16(valid_data_len)
        else:
            self._transpose_by_vnchwconv_b8(data_len)
            self._move_to_target_layout()
            self._transpose_back_by_vnchwconv_b8(valid_data_len)

    def _compute_10_vnchwconv_scheme(self, data_len, valid_data_len):
        """
        reorder elements by vnchwconv for compute 10
        """
        if self.ele_per_block != AsStrided.BYTES_PER_BLOCK:
            self._transpose_by_vnchwconv_b16(data_len)
            self._move_to_target_layout()
            self._compute_10_transpose_back_by_vnchwconv_b16(valid_data_len)
        else:
            self._transpose_by_vnchwconv_b8(data_len)
            self._move_to_target_layout()
            self._compute_10_transpose_back_by_vnchwconv_b8(valid_data_len)

    def _scalar_scheme(self):
        """
        reorder elements by scalar
        """
        with self.tik_inst.for_range(0, self.axis_0_plp_size) as axis_0_idx:
            axis_0_cur_idx = axis_0_idx + self.axis_0_cur_idx
            self._get_axis_0_elem_idx(axis_0_cur_idx)
            with self.tik_inst.if_scope(
                    tik.all(self.last_dim_stride == 0,
                            tik.any(self.axis_1_plp_size % self.ele_per_block == 0, self.axis_0_plp_size == 1))):
                in_offset = self.axis_0_input_idx
                out_offset = axis_0_idx * self.axis_1_plp_size + self.out_ub_offset
                self.scalar_reg[0].set_as(self.data_ub[in_offset])
                for idx in self.SCALAR_INDEX[:self.ele_per_block]:
                    self.data_ub[out_offset + idx].set_as(self.scalar_reg[0])
                out_offset_int16 = out_offset // self.ele_per_block * self.PER_BLOCK_16_ELEMS
                data_ub_int16 = self.data_ub.reinterpret_cast_to("int16")
                axis_1_plp_size_int16 = (self._ceil_div(self.axis_1_plp_size, self.ele_per_block) -
                                         1) * self.PER_BLOCK_16_ELEMS
                loop_cnt = axis_1_plp_size_int16 // self.DUP_MASK
                loop_left = axis_1_plp_size_int16 % self.DUP_MASK
                with self.tik_inst.if_scope(loop_cnt > 0):
                    self.tik_inst.vor(self.DUP_MASK, data_ub_int16[out_offset_int16 + self.PER_BLOCK_16_ELEMS],
                                      data_ub_int16[out_offset_int16], data_ub_int16[out_offset_int16], loop_cnt, 1, 0,
                                      0, 8, 0, 0)
                with self.tik_inst.if_scope(loop_left > 0):
                    out_offset_1 = out_offset_int16 + self.PER_BLOCK_16_ELEMS + loop_cnt * self.DUP_MASK
                    self.tik_inst.vor(loop_left, data_ub_int16[out_offset_1], data_ub_int16[out_offset_int16],
                                      data_ub_int16[out_offset_int16], 1, 1, 0, 0, 8, 0, 0)
            with self.tik_inst.else_scope():
                with self.tik_inst.for_range(0, self.axis_1_plp_size) as axis_1_idx:
                    in_offset = self.axis_0_input_idx + (self.axis_1_cur_idx + axis_1_idx) * self.last_dim_stride
                    out_offset = axis_1_idx + axis_0_idx * self.axis_1_plp_size + self.out_ub_offset
                    self.scalar_reg[0].set_as(self.data_ub[in_offset])
                    self.data_ub[out_offset].set_as(self.scalar_reg[0])

    # 'pylint: disable=too-many-locals
    def _reorder_and_move_out_data(self):
        """
        reorder elements by vector_dup and move out
        """
        with self.tik_inst.for_range(0, self.axis_1_plp_size) as axis_1_idx:
            in_offset = axis_1_idx * self.rsecond_dim_stride
            if tbe_platform.api_check_support("tik.vector_dup", self.data_in.dtype):
                mask_value = AsStrided.DUP_MASK // self.dtype_factor
                repeat_cnt = self.vnc_col_size // mask_value
                zero_data_len = repeat_cnt * mask_value
                self.scalar_reg[0].set_as(self.data_ub[in_offset])
                self.tik_inst.vector_dup(mask_value, self.data_ub[self.out_ub_offset], self.scalar_reg[0],
                                         repeat_cnt, 1, 8)
            else:
                for j in AsStrided.SCALAR_INDEX[:self.ele_per_block]:
                    self.scalar_reg[j].set_as(self.data_ub[in_offset])
                for j in AsStrided.SCALAR_INDEX[:self.ele_per_block]:
                    self.data_ub[self.out_ub_offset + j].set_as(self.scalar_reg[j])
                data_ub = self.data_ub.reinterpret_cast_to("int16")
                mask_value = AsStrided.DUP_MASK
                if self.ele_per_block == AsStrided.BYTES_PER_BLOCK:
                    repeat_cnt = self.vnc_col_size // 2 // mask_value
                    zero_data_len = repeat_cnt * mask_value * 2
                    self.tik_inst.vor(mask_value,
                                      data_ub[self.out_ub_offset // 2 + AsStrided.PER_BLOCK_16_ELEMS],
                                      data_ub[self.out_ub_offset // 2], data_ub[self.out_ub_offset // 2],
                                      repeat_cnt, 1, 0, 0, 8, 0, 0)
                else:
                    repeat_cnt = self.dtype_factor * self.vnc_col_size // mask_value
                    zero_data_len = repeat_cnt * mask_value // self.dtype_factor
                    self.tik_inst.vor(mask_value,
                                      data_ub[self.out_ub_offset * self.dtype_factor + AsStrided.PER_BLOCK_16_ELEMS],
                                      data_ub[self.out_ub_offset * self.dtype_factor],
                                      data_ub[self.out_ub_offset * self.dtype_factor],
                                      repeat_cnt, 1, 0, 0, 8, 0, 0)

            # move data out
            last_dim_lp_cnt = self.last_dim_size // zero_data_len
            last_dim_left = self.last_dim_size % zero_data_len
            block_mod = last_dim_left % self.ele_per_block
            out_offset = self.axis_0_cur_idx * self.out_lp_step + (
                        self.axis_1_cur_idx + axis_1_idx) * self.last_dim_size
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, last_dim_lp_cnt) as last_lp_idx:
                    out_gm_offset = last_lp_idx * zero_data_len + out_offset
                    self.tik_inst.data_move(self.data_out[out_gm_offset], self.data_ub[self.out_ub_offset],
                                            0, 1, zero_data_len // self.ele_per_block, 0, 0)

            with self.tik_inst.if_scope(tik.all(last_dim_left > 0, block_mod == 0)):
                out_gm_offset_1 = last_dim_lp_cnt * zero_data_len + out_offset
                self.tik_inst.data_move(self.data_out[out_gm_offset_1], self.data_ub[self.out_ub_offset],
                                        0, 1, last_dim_left // self.ele_per_block, 0, 0)
            with self.tik_inst.elif_scope(tik.all(last_dim_left > self.ele_per_block, block_mod > 0)):
                out_gm_offset_1 = last_dim_lp_cnt * zero_data_len + out_offset
                self.tik_inst.data_move(self.data_out[out_gm_offset_1], self.data_ub[self.out_ub_offset],
                                        0, 1, last_dim_left // self.ele_per_block, 0, 0)
                out_gm_offset_2 = self.last_dim_size - self.ele_per_block + out_offset
                self.tik_inst.data_move(self.data_out[out_gm_offset_2], self.data_ub[self.out_ub_offset],
                                        0, 1, 1, 0, 0)
            with self.tik_inst.elif_scope(tik.all(last_dim_left < self.ele_per_block, block_mod > 0)):
                out_gm_offset_2 = self.last_dim_size - self.ele_per_block + out_offset
                self.tik_inst.data_move(self.data_out[out_gm_offset_2], self.data_ub[self.out_ub_offset],
                                        0, 1, 1, 0, 0)

    def _compute_1_func(self, args, block_idx):
        """
        case for last dimension stride is 1, using data move scheme
        """
        axis_0_lp_cnt, axis_0_lp_left, axis_1_lp_cnt, axis_1_lp_left = args
        self.axis_0_backend.set_as(self._ceil_div(self.ele_per_block, self.out_lp_step) - self.lc_axis_0_lp_left)
        self.axis_1_backend.set_as(self.ele_per_block)

        with self.tik_inst.for_range(0, axis_0_lp_cnt) as axis_0_lp_idx:
            self._get_axis_0_plp_size(block_idx, axis_0_lp_idx, axis_0_lp_cnt, axis_0_lp_left)

            with self.tik_inst.for_range(0, axis_1_lp_cnt) as axis_1_lp_idx:
                self._get_axis_1_plp_size(block_idx, axis_1_lp_idx, axis_1_lp_cnt, axis_1_lp_left)
                self.axis_1_plp_burst.set_as((self.axis_1_plp_size - 1) * self.last_dim_stride + 1)

                self._get_axis_0_idx(block_idx, axis_0_lp_idx)
                self._get_axis_1_idx(block_idx, axis_1_lp_idx)

                self._compute_data_move_in(self.burst_elems, self.last_dim_stride)
                self._adjust_tail_elems(self.axis_0_plp_size, self.axis_1_plp_size, self.burst_elems)
                self._compute_data_move_out(self.axis_0_plp_size, self.axis_1_plp_size, self.burst_elems,
                                            self.axis_0_cur_idx, self.axis_1_cur_idx)

    def _compute_2_func(self, args, block_idx):
        """
        case for last dimension size is bigger than four blocks
        and can move in four blocks valid elements one time, using full vnchwconv scheme
        """
        axis_0_lp_cnt, axis_0_lp_left, axis_1_lp_cnt, axis_1_lp_left = args
        self.axis_0_backend.set_as(self._ceil_div(self.ele_per_block, self.out_lp_step) - self.lc_axis_0_lp_left)
        self.axis_1_backend.set_as(self.ele_per_block)

        with self.tik_inst.for_range(0, axis_0_lp_cnt) as axis_0_lp_idx:
            self._get_axis_0_plp_size(block_idx, axis_0_lp_idx, axis_0_lp_cnt, axis_0_lp_left)

            with self.tik_inst.for_range(0, axis_1_lp_cnt) as axis_1_lp_idx:
                self._get_axis_1_plp_size(block_idx, axis_1_lp_idx, axis_1_lp_cnt, axis_1_lp_left)
                self.axis_1_plp_burst.set_as((self.axis_1_plp_size - 1) * self.last_dim_stride + 1)

                self._get_axis_0_idx(block_idx, axis_0_lp_idx)
                self._get_axis_1_idx(block_idx, axis_1_lp_idx)

                self._compute_data_move_in(self.vnc_col_size, self.last_dim_stride)

                self.burst_valid_elems.set_as(self.axis_1_plp_size)
                self.reorder_src_stride.set_as(self.last_dim_stride - 1)
                self.reorder_lp_cnt.set_as(1)
                self.reorder_gap.set_as(self.burst_elems)
                self._vnchwconv_scheme(self.burst_elems, self.axis_1_plp_size)

                self._adjust_tail_elems(self.axis_0_plp_size, self.axis_1_plp_size, self.vnc_col_size)
                self._compute_data_move_out(self.axis_0_plp_size, self.axis_1_plp_size, self.vnc_col_size,
                                            self.axis_0_cur_idx, self.axis_1_cur_idx)

    def _compute_3_func(self, args, block_idx):
        """
        case for last dimension size is smaller than four blocks
        and can move last dimension in one time, using single row vnchwconv scheme
        """
        axis_0_lp_cnt, axis_0_lp_left, axis_1_lp_cnt, axis_1_lp_left = args
        self.axis_0_backend.set_as(self._ceil_div(self.ele_per_block, self.out_lp_step) - self.lc_axis_0_lp_left)
        self.axis_1_backend.set_as(self.ele_per_block)

        if self.vg_cond:
            round_gap = self._ceil_div(
                (self.axis_1_lp_unit - 1) * self.last_dim_stride + 1, self.ele_per_block) * self.BYTES_PER_BLOCK
            parallel_cnt = 2
            inner_lp_cnt = self.axis_1_lp_unit // parallel_cnt
            inner_lp_left = self.axis_1_lp_unit % parallel_cnt
            with self.tik_inst.for_range(0, self.axis_0_lp_unit) as i:
                with self.tik_inst.for_range(0, inner_lp_cnt) as j:
                    for k in self.ADDR_INDEX[:parallel_cnt]:
                        self.vg_offset_tensor[self.vg_ub_beg_pos + parallel_cnt * j + k +
                                              self.axis_1_lp_unit * i].set_as(self.last_dim_stride * self.dtype_bytes *
                                                                              (parallel_cnt * j + k) + round_gap * i)
                with self.tik_inst.for_range(0, inner_lp_left) as k:
                    self.vg_offset_tensor[self.vg_ub_beg_pos + parallel_cnt * inner_lp_cnt + k +
                                          self.axis_1_lp_unit * i].set_as(self.last_dim_stride * self.dtype_bytes *
                                                                          (parallel_cnt * inner_lp_cnt + k) +
                                                                          round_gap * i)

        with self.tik_inst.for_range(0, axis_0_lp_cnt) as axis_0_lp_idx:
            self._get_axis_0_plp_size(block_idx, axis_0_lp_idx, axis_0_lp_cnt, axis_0_lp_left)

            with self.tik_inst.for_range(0, axis_1_lp_cnt) as axis_1_lp_idx:
                self._get_axis_1_plp_size(block_idx, axis_1_lp_idx, axis_1_lp_cnt, axis_1_lp_left)
                self.axis_1_plp_burst.set_as((self.axis_1_plp_size - 1) * self.last_dim_stride + 1)

                self._get_axis_0_idx(block_idx, axis_0_lp_idx)
                self._get_axis_1_idx(block_idx, axis_1_lp_idx)

                self._compute_data_move_in(self.burst_elems, self.last_dim_stride)

                data_len = self.axis_1_plp_size * self.axis_0_plp_size
                data_gap = self._ceil_div(data_len, self.ele_per_block) * self.ele_per_block
                if self.vg_cond:
                    repeat = self.axis_0_plp_size * self.axis_1_plp_size // self.vg_lp_unit
                    rp_left = self.axis_0_plp_size * self.axis_1_plp_size % self.vg_lp_unit
                    with self.tik_inst.if_scope(repeat > 0):
                        self.tik_inst.vgather(self.vg_lp_unit, self.data_ub[self.out_ub_offset], self.data_ub,
                                              self.vg_offset_tensor[self.vg_ub_beg_pos], repeat, self.vg_rep_stride)
                    with self.tik_inst.if_scope(rp_left > 0):
                        base_offset = repeat * self.vg_lp_unit
                        self.tik_inst.vgather(rp_left, self.data_ub[self.out_ub_offset + base_offset], self.data_ub,
                                              self.vg_offset_tensor[self.vg_ub_beg_pos + base_offset], 1,
                                              self.vg_rep_stride)
                else:
                    self.burst_valid_elems.set_as(self.axis_1_plp_size)
                    self.reorder_src_stride.set_as(self.last_dim_stride - 1)
                    self.reorder_lp_cnt.set_as(self.axis_0_plp_size)
                    self.reorder_gap.set_as(self.burst_elems)
                    self._vnchwconv_scheme(self.burst_elems * self.axis_0_plp_size, data_len)

                self._adjust_tail_elems(1, data_len, data_gap)
                self._compute_data_move_out(1, data_len, data_gap, self.axis_0_cur_idx, self.axis_1_cur_idx)

    def _compute_4_func(self, args, block_idx):
        """
        case for input or output elements can move all in one time, using scalar scheme
        """
        axis_0_lp_cnt, axis_0_lp_left, axis_1_lp_cnt, axis_1_lp_left = args
        self.axis_1_plp_burst.set_as(self.axis_1_burst_unit)
        if self.mte_pad_cond:
            if self.dtype_bytes == self.B64_BYTES:
                self.m_data_in = self.data_in.reinterpret_cast_to("int8")
                self.m_data_ub = self.data_ub.reinterpret_cast_to("int8")
            self.tik_inst.data_move_pad(self.m_data_ub, self.m_data_in, 1,
                                        self.axis_1_plp_burst * self.dtype_bytes, 0, 0)
        else:
            self.tik_inst.data_move(self.data_ub, self.data_in, 0, 1, self.burst_len, 0, 0)
        self.axis_0_backend.set_as(self._ceil_div(self.ele_per_block, self.out_lp_step) - self.lc_axis_0_lp_left)
        self.axis_1_backend.set_as(self.ele_per_block)

        with self.tik_inst.for_range(0, axis_0_lp_cnt) as axis_0_lp_idx:
            self._get_axis_0_plp_size(block_idx, axis_0_lp_idx, axis_0_lp_cnt, axis_0_lp_left)

            with self.tik_inst.for_range(0, axis_1_lp_cnt) as axis_1_lp_idx:
                self._get_axis_1_plp_size(block_idx, axis_1_lp_idx, axis_1_lp_cnt, axis_1_lp_left)

                self._get_axis_0_idx(block_idx, axis_0_lp_idx)
                self._get_axis_1_idx(block_idx, axis_1_lp_idx)

                self._scalar_scheme()
                data_len = self.axis_1_plp_size * self.axis_0_plp_size
                data_gap = self._ceil_div(data_len, self.ele_per_block) * self.ele_per_block
                self._adjust_tail_elems(1, data_len, data_gap)
                self._compute_data_move_out(1, data_len, data_gap, self.axis_0_cur_idx, self.axis_1_cur_idx)

    def _compute_5_func(self, args, block_idx):
        """
        case for last dimension size and stride is large, using full vnchwconv scheme
        """
        axis_0_lp_cnt, axis_0_lp_left, axis_1_lp_cnt, axis_1_lp_left = args
        self.axis_1_plp_burst.set_as(self.axis_1_burst_unit)
        self.axis_0_backend.set_as(self._ceil_div(self.ele_per_block, self.out_lp_step) - self.lc_axis_0_lp_left)
        self.axis_1_backend.set_as(self.ele_per_block)

        with self.tik_inst.for_range(0, axis_0_lp_cnt) as axis_0_lp_idx:
            self._get_axis_0_plp_size(block_idx, axis_0_lp_idx, axis_0_lp_cnt, axis_0_lp_left)

            with self.tik_inst.for_range(0, axis_1_lp_cnt) as axis_1_lp_idx:
                self._get_axis_1_plp_size(block_idx, axis_1_lp_idx, axis_1_lp_cnt, axis_1_lp_left)

                self._get_axis_0_idx(block_idx, axis_0_lp_idx)
                self._get_axis_1_idx(block_idx, axis_1_lp_idx)

                self._compute_large_stride_data_move_in(self.vnc_col_size)

                self.burst_valid_elems.set_as(self.axis_1_plp_size)
                self.reorder_src_stride.set_as(self.ele_per_block - 1)
                self.reorder_lp_cnt.set_as(1)
                self.reorder_gap.set_as(self.burst_elems)
                self._vnchwconv_scheme(self.burst_elems * self.axis_1_plp_size, self.axis_1_plp_size)

                self._adjust_tail_elems(self.axis_0_plp_size, self.axis_1_plp_size, self.vnc_col_size)
                self._compute_data_move_out(self.axis_0_plp_size, self.axis_1_plp_size, self.vnc_col_size,
                                            self.axis_0_cur_idx, self.axis_1_cur_idx)

    def _compute_6_func(self, args, block_idx):
        """
        case for last dimension stride is big and size is small, using single vnchwconv scheme
        """
        axis_0_lp_cnt, axis_0_lp_left, axis_1_lp_cnt, axis_1_lp_left = args
        self.axis_1_plp_burst.set_as(self.axis_1_burst_unit)
        self.axis_0_backend.set_as(self._ceil_div(self.ele_per_block, self.out_lp_step) - self.lc_axis_0_lp_left)
        self.axis_1_backend.set_as(self.ele_per_block)

        if self.vg_cond:
            mask = len(self.ADDR_INDEX)
            for i in self.ADDR_INDEX:
                self.vg_offset_tensor[self.vg_ub_beg_pos + i].set_as(i * self.BYTES_PER_BLOCK)
            lp_cnt = self._ceil_div(self.axis_0_lp_unit * self.axis_1_lp_unit,
                                    self.vg_lp_unit) * self.vg_lp_unit // mask - 1
            with self.tik_inst.for_range(0, lp_cnt) as j:
                self.tik_inst.vadds(mask, self.vg_offset_tensor[self.vg_ub_beg_pos + mask * (j + 1)],
                                    self.vg_offset_tensor[self.vg_ub_beg_pos + mask * j], mask * self.BYTES_PER_BLOCK,
                                    1, 1, 1, 1, 1)

        with self.tik_inst.for_range(0, axis_0_lp_cnt) as axis_0_lp_idx:
            self._get_axis_0_plp_size(block_idx, axis_0_lp_idx, axis_0_lp_cnt, axis_0_lp_left)

            with self.tik_inst.for_range(0, axis_1_lp_cnt) as axis_1_lp_idx:
                self._get_axis_1_plp_size(block_idx, axis_1_lp_idx, axis_1_lp_cnt, axis_1_lp_left)

                self._get_axis_0_idx(block_idx, axis_0_lp_idx)
                self._get_axis_1_idx(block_idx, axis_1_lp_idx)

                self._compute_large_stride_data_move_in(self.burst_elems * self.axis_1_plp_size)

                data_len = self.axis_1_plp_size * self.axis_0_plp_size
                data_gap = self._ceil_div(data_len, self.ele_per_block) * self.ele_per_block
                if self.vg_cond:
                    repeat = self._ceil_div(self.axis_0_plp_size * self.axis_1_plp_size, self.vg_lp_unit)
                    self.tik_inst.vgather(self.vg_lp_unit, self.data_ub[self.out_ub_offset], self.data_ub,
                                          self.vg_offset_tensor[self.vg_ub_beg_pos], repeat, self.vg_rep_stride)
                else:
                    self.burst_valid_elems.set_as(data_len)
                    self.reorder_src_stride.set_as(self.ele_per_block - 1)
                    self.reorder_lp_cnt.set_as(1)
                    self.reorder_gap.set_as(self.burst_elems)
                    self.branch_5_stride.set_as(self.dtype_factor * self.ele_per_block)
                    self._vnchwconv_scheme(self.burst_elems * data_len, data_len)

                self._adjust_tail_elems(1, data_len, data_gap)
                self._compute_data_move_out(1, data_len, data_gap, self.axis_0_cur_idx, self.axis_1_cur_idx)

    def _compute_7_func(self, args, block_idx):
        """
        case for last two dimension size is bigger than four blocks
        and can move in four blocks valid elements one time, using full vnchwconv scheme
        """
        axis_0_lp_cnt, axis_0_lp_left, axis_1_lp_cnt, axis_1_lp_left = args
        self.axis_1_plp_burst.set_as(self.axis_1_burst_unit)

        with self.tik_inst.for_range(0, axis_0_lp_cnt) as axis_0_lp_idx:
            with self.tik_inst.if_scope(tik.any(axis_0_lp_idx != axis_0_lp_cnt - 1, axis_0_lp_left == 0)):
                self.axis_0_plp_size.set_as(self.axis_0_lp_unit)
            with self.tik_inst.else_scope():
                self.axis_0_plp_size.set_as(axis_0_lp_left)

            with self.tik_inst.for_range(0, axis_1_lp_cnt) as axis_1_lp_idx:
                with self.tik_inst.if_scope(tik.any(axis_1_lp_idx != axis_1_lp_cnt - 1, axis_1_lp_left == 0)):
                    self.axis_1_plp_size.set_as(self.axis_1_lp_unit)
                with self.tik_inst.else_scope():
                    self.axis_1_plp_size.set_as(axis_1_lp_left)

                self._get_axis_0_idx(block_idx, axis_0_lp_idx)
                self._get_axis_1_idx(block_idx, axis_1_lp_idx)

                self._compute_data_move_in(self.vnc_col_size, self.last_dim_stride)

                self.burst_valid_elems.set_as(self.last_dim_size)
                self.reorder_src_stride.set_as(self.last_dim_stride - 1)
                self.reorder_lp_cnt.set_as(self.rsecond_dim_size)
                self.reorder_gap.set_as(self.rsecond_dim_stride)
                self._vnchwconv_scheme(self.burst_elems, self.axis_1_plp_size)

                self._adjust_tail_elems(self.axis_0_plp_size, self.axis_1_plp_size, self.vnc_col_size)
                self._compute_data_move_out(self.axis_0_plp_size, self.axis_1_plp_size, self.vnc_col_size,
                                            self.axis_0_cur_idx, self.axis_1_cur_idx)

    def _compute_8_func(self, args, block_idx):
        """
        case for last dimension stride is zero and size is large, using vector_dup scheme
        """
        axis_0_lp_cnt, axis_0_lp_left, axis_1_lp_cnt, axis_1_lp_left = args

        with self.tik_inst.for_range(0, axis_0_lp_cnt) as axis_0_lp_idx:
            with self.tik_inst.if_scope(tik.any(axis_0_lp_idx != axis_0_lp_cnt - 1, axis_0_lp_left == 0)):
                self.axis_0_plp_size.set_as(self.axis_0_lp_unit)
            with self.tik_inst.else_scope():
                self.axis_0_plp_size.set_as(axis_0_lp_left)

            with self.tik_inst.for_range(0, axis_1_lp_cnt) as axis_1_lp_idx:
                with self.tik_inst.if_scope(tik.any(axis_1_lp_idx != axis_1_lp_cnt - 1, axis_1_lp_left == 0)):
                    self.axis_1_plp_size.set_as(self.axis_1_lp_unit)
                with self.tik_inst.else_scope():
                    self.axis_1_plp_size.set_as(axis_1_lp_left)
                self.axis_1_plp_burst.set_as((self.axis_1_plp_size - 1) * self.rsecond_dim_stride + 1)

                self._get_axis_0_idx(block_idx, axis_0_lp_idx)
                self._get_axis_1_idx(block_idx, axis_1_lp_idx)

                self._compute_data_move_in(0, self.rsecond_dim_stride)

                self._reorder_and_move_out_data()

    def _compute_9_func(self, args, block_idx):
        """
        case for last dimension stride is zero and size is small, using full vnchwconv scheme
        """
        axis_0_lp_cnt, axis_0_lp_left, axis_1_lp_cnt, axis_1_lp_left = args
        self.axis_1_backend.set_as(self._ceil_div(self.ele_per_block, self.last_dim_size) - self.lc_axis_1_lp_left)

        with self.tik_inst.for_range(0, axis_0_lp_cnt) as axis_0_lp_idx:
            with self.tik_inst.if_scope(tik.any(axis_0_lp_idx != axis_0_lp_cnt - 1, axis_0_lp_left == 0)):
                self.axis_0_plp_size.set_as(self.axis_0_lp_unit)
            with self.tik_inst.else_scope():
                self.axis_0_plp_size.set_as(axis_0_lp_left)

            with self.tik_inst.for_range(0, axis_1_lp_cnt) as axis_1_lp_idx:
                self._compute_9_get_axis_1_plp_size(block_idx, axis_1_lp_idx, axis_1_lp_cnt, axis_1_lp_left)
                self.axis_1_plp_burst.set_as((self.axis_1_plp_size - 1) * self.rsecond_dim_stride + 1)

                self._get_axis_0_idx(block_idx, axis_0_lp_idx)
                self._get_axis_1_idx(block_idx, axis_1_lp_idx)

                self._compute_data_move_in(self.vnc_col_size, self.rsecond_dim_stride)

                self.burst_valid_elems.set_as(self.last_dim_size)
                self.reorder_src_stride.set_as(0)
                self.reorder_lp_cnt.set_as(self.axis_1_plp_size)
                self.reorder_gap.set_as(self.rsecond_dim_stride)
                data_len = self.axis_1_plp_size * self.last_dim_size
                self._vnchwconv_scheme(self.burst_elems, data_len)

                self._adjust_tail_elems(self.axis_0_plp_size, data_len, self.vnc_col_size)
                self._compute_data_move_out(self.axis_0_plp_size, data_len, self.vnc_col_size,
                                            self.axis_0_cur_idx, self.axis_1_cur_idx * self.last_dim_size)

    def _compute_10_func(self, args, block_idx):
        """
        case for first dimension is small, using vnchwconv scheme
        """
        axis_0_lp_cnt, axis_0_lp_left, axis_1_lp_cnt, axis_1_lp_left = args
        self.axis_0_backend.set_as(self.ele_per_block)

        with self.tik_inst.for_range(0, axis_0_lp_cnt) as axis_0_lp_idx:
            self._compute_10_get_axis_0_plp_size(block_idx, axis_0_lp_idx, axis_0_lp_cnt, axis_0_lp_left)

            with self.tik_inst.for_range(0, axis_1_lp_cnt) as axis_1_lp_idx:
                with self.tik_inst.if_scope(tik.any(axis_1_lp_idx != axis_1_lp_cnt - 1, axis_1_lp_left == 0)):
                    self.axis_1_plp_size.set_as(self.axis_1_lp_unit)
                with self.tik_inst.else_scope():
                    self.axis_1_plp_size.set_as(axis_1_lp_left)
                self.axis_1_plp_burst.set_as((self.axis_1_plp_size - 1) * self.last_dim_stride + 1)

                self._get_axis_0_idx(block_idx, axis_0_lp_idx)
                self._get_axis_1_idx(block_idx, axis_1_lp_idx)

                if self.mte_pad_cond is False:
                    self._compute_first_stride_small_data_move_in()
                else:
                    self._compute_first_stride_small_data_move_in_with_dmp()

                self.burst_valid_elems.set_as(self.nfirst_cnt_per_row)
                self.reorder_src_stride.set_as(self.burst_elems - 1)
                self.reorder_lp_cnt.set_as(self.axis_1_plp_size)
                self.reorder_gap.set_as(self.last_dim_stride)
                self._compute_10_vnchwconv_scheme(self.burst_elems * self.nfirst_cnt_per_row, self.nfirst_cnt_per_row)

                data_len = self.nfirst_cnt_per_row * AsStrided.VNC_ROWS
                self._adjust_tail_elems(self.axis_1_plp_size, self.axis_0_plp_size, data_len)
                self._compute_data_move_out(self.axis_1_plp_size, self.axis_0_plp_size, data_len,
                                            self.axis_1_cur_idx, self.axis_0_cur_idx)

    def _compute_tiling(self):
        """
        execution function
        """
        with self.tik_inst.for_range(0, self.used_core_cnt, block_num=self.used_core_cnt) as block_idx:
            self._compute_func(block_idx)

    def compute(self, input_list, kernel_name):
        """
        entrance function
        """
        tbe_context.get_context().add_compile_info("vars", {
            "max_elem_cnt": self.max_elem_cnt,
            "core_num": self.core_num
        })

        self._compute_tiling()
        self.tik_inst.BuildCCE(kernel_name=kernel_name,
                               inputs=input_list,
                               outputs=[self.data_out],
                               flowtable=[self.tiling_gm], config={"enable_const_fold": True})


# 'pylint: disable=invalid-name, too-many-arguments, too-many-locals, unused-argument
@register_operator("AsStrided")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def as_strided(x, size, stride, storage_offset, y, kernel_name="as_strided"):
    """
    Generate contiguous memory with the given shape, strides and storage_offset.

    Parameters
    ----------
    x : the input tensor
    size : the shape of output tensor
    stride: the stride of output tensor
    storage_offset : the offset in the underlying storage of the output tensor
    y : the output tensor
    kernel_name: operator name, default value is "as_strided"
    Returns
    -------
    None
    """

    tik_inst = tik.Tik()
    x_dtype = x.get("dtype").lower() if x.get("dtype").lower() != "bool" else "int8"
    x_dtype = x_dtype if x_dtype != "bfloat16" else "float16"
    x_dtype = x_dtype if x_dtype != "complex32" else "int32"
    x_dtype = x_dtype if x_dtype != "complex64" else "int64"
    size_dtype = size.get("dtype").lower()
    stride_dtype = stride.get("dtype").lower()
    if storage_offset is None:
        storage_offset_dtype = stride_dtype
    else:
        storage_offset_dtype = storage_offset.get("dtype").lower()

    data_in = tik_inst.Tensor(x_dtype, (AsStrided.MAX_INT64_VALUE,), tik.scope_gm, "x")
    size = tik_inst.Tensor(size_dtype, (AsStrided.MAX_INT64_VALUE,), tik.scope_gm, "size")
    stride = tik_inst.Tensor(stride_dtype, (AsStrided.MAX_INT64_VALUE,), tik.scope_gm, "stride")
    storage_offset = tik_inst.Tensor(storage_offset_dtype, (AsStrided.MAX_INT64_VALUE, ), tik.scope_gm,
                                     "storage_offset")
    data_out = tik_inst.Tensor(x_dtype, (AsStrided.MAX_INT64_VALUE,), tik.scope_gm, "y")

    # get tiling params
    tiling_gm = tik_inst.Tensor(AsStrided.TILING_SPACE[0], (AsStrided.TILING_SPACE[1],), tik.scope_gm, "tiling_gm")
    tiling_ub = tik_inst.Tensor(AsStrided.TILING_SPACE[0], (AsStrided.TILING_SPACE[1],), tik.scope_ubuf, "tiling_ub")
    tiling_params = [tik_inst.Scalar(AsStrided.TILING_SPACE[0]) for i in range(AsStrided.TILING_SPACE[1])]
    tiling_inst = TilingParams(tik_inst, tiling_gm, tiling_ub, tiling_params)
    tiling_inst.get_tiling_params()

    input_list = [data_in, size, stride, storage_offset]
    tensor_list = [data_in, data_out, tiling_gm, tiling_ub, tiling_params]
    core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    as_strided_instance = AsStrided(tik_inst, core_num, x_dtype, tensor_list)
    as_strided_instance.compute(input_list, kernel_name)
