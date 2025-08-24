#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

trans_data_fractal_z_g_2_fractal_z
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.util_common import is_unknown
from impl.trans_data_common_func import clean_ubuf
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods
class TilingParams:
    """
    The class for getting tiling parameters
    """

    def __init__(self, tik_inst, tiling_gm, tiling_ub, tiling_params):
        self.tik_inst = tik_inst
        self.tiling_gm = tiling_gm
        self.tiling_ub = tiling_ub
        self.tiling_params = tiling_params

    def get_tiling_params(self):
        """
        method of getting tiling parameters
        """
        self.tik_inst.data_move(self.tiling_ub, self.tiling_gm, 0, 1, FzgToFz.TILING_CFG[1] // FzgToFz.TILING_CFG[2], 0,
                                0)
        for idx, reg in enumerate(self.tiling_params):
            reg.set_as(self.tiling_ub[idx])


# 'pylint: disable=too-many-instance-attributes
class FzgToFz:
    """
    The class of FzgToFz
    """
    BYTES_PER_BLOCK = 32
    BITS_PER_BYTE = 8
    STRIDE_LIMIT = 65535
    MAX_INT64_VALUE = 2**63 - 1
    TILING_CFG = ("int64", 8 * 4, 4, 128 * 8)
    C_IS_C0_ALIGN_HW = 0
    C_IS_C0_ALIGN_C1OUT = 1
    C_IS_C0_ALIGN_D = 2
    C_IS_C0_UNALIGN_HW = 3
    C_IS_C0_UNALIGN_D = 4
    C_IS_C0_UNALIGN_HW_1 = 5
    C_IS_C0_UNALIGN_D_1 = 6
    VNC_ROWS = 16
    NI = 16
    G_POS = 0
    D_POS = 1
    C1_OUT_POS = 2
    HW_POS = 3
    N_NG_POS = 4
    ADDR_IDX = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

    # 'pylint: disable=invalid-name
    def __init__(self, src, kernel_name):
        self.src = src
        self.kernel_name = kernel_name
        self.tik_inst = tik.Tik()

        self.tiling_gm = self.tik_inst.Tensor(self.TILING_CFG[0], (self.TILING_CFG[1], ), tik.scope_gm, "tile_gm")
        self.tiling_ub = self.tik_inst.Tensor(self.TILING_CFG[0], (self.TILING_CFG[1], ), tik.scope_ubuf, "tile_ub")
        self.tiling_params = [self.tik_inst.Scalar(self.TILING_CFG[0]) for _ in range(self.TILING_CFG[1])]
        tiling_inst = TilingParams(self.tik_inst, self.tiling_gm, self.tiling_ub, self.tiling_params)
        tiling_inst.get_tiling_params()

        self.inner_dtype = FzgToFz._get_inner_dtype(self.src.get("dtype"))
        self.gm_in = self.tik_inst.Tensor(self.inner_dtype, (self.MAX_INT64_VALUE, ), tik.scope_gm, "src")
        self.gm_out = self.tik_inst.Tensor(self.inner_dtype, (self.MAX_INT64_VALUE, ), tik.scope_gm, "dst")

        self.dtype_bytes = get_bit_len(self.inner_dtype) // self.BITS_PER_BYTE
        self.b16_bytes = 2
        self.elem_per_block = self.BYTES_PER_BLOCK // self.dtype_bytes
        self.max_elem_cnt = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - self.TILING_CFG[3]) // self.dtype_bytes
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.data_ub = self.tik_inst.Tensor(self.inner_dtype, (self.max_elem_cnt, ), tik.scope_ubuf, "data_ub")
        self.data_ub_b16 = self.data_ub.reinterpret_cast_to("float16")

        (self.tiling_key, self.ub_offset, self.used_core_cnt, self.core_step_in, self.core_step_out,
         self.max_n_per_row) = self.tiling_params[:6]
        (self.g, self.d, self.c1_in, self.c1_out, self.c, self.c0, self.e, self.n_ng, self.n_e_align, self.n_gp_align,
         self.hw) = self.tiling_params[6:17]
        (self.mc_pos, self.nlc_g_lp_cnt, self.nlc_d_lp_cnt, self.nlc_c1_out_lp_cnt, self.nlc_hw_lp_cnt,
         self.nlc_n_ng_lp_cnt, self.lc_g_lp_cnt, self.lc_d_lp_cnt, self.lc_c1_out_lp_cnt, self.lc_hw_lp_cnt,
         self.lc_n_ng_lp_cnt, self.hw_lp_unit, self.n_ng_lp_unit, self.c1_out_lp_unit,
         self.d_lp_unit) = self.tiling_params[17:32]
        # G D C1 E HW N loop control
        self.g_lp_cnt = self.tik_inst.Scalar(name="g_lp_cnt")
        self.d_lp_cnt = self.tik_inst.Scalar(name="d_lp_cnt")
        self.c1_out_lp_cnt = self.tik_inst.Scalar(name="c1_out_lp_cnt")
        self.hw_lp_cnt = self.tik_inst.Scalar(name="hw_lp_cnt")
        self.n_ng_lp_cnt = self.tik_inst.Scalar(name="n_ng_lp_cnt")
        self.cur_hw_lp_unit = self.tik_inst.Scalar(name="cur_hw_lp_unit")
        self.cur_n_ng_lp_unit = self.tik_inst.Scalar(name="cur_n_ng_lp_unit")
        self.cur_d_lp_unit = self.tik_inst.Scalar(name="cur_d_lp_unit")
        self.cur_c1_out_lp_unit = self.tik_inst.Scalar(name="cur_c1_out_lp_unit")
        self.c0_parts = self.tik_inst.Scalar(name="c0_parts", init_value=1)
        self.vnc_src_stride = self.tik_inst.Scalar(name="vnc_src_stride")
        self.vnc_dst_stride = self.tik_inst.Scalar(name="vnc_dst_stride")
        self.e_in_offset = self.tik_inst.Scalar(name="e_in_offset")
        self.out_ub_offset = self.tik_inst.Scalar(name="out_ub_offset")

    @staticmethod
    def _ceil_div(val_x, val_y):
        """
        ceiling division
        """
        return (val_x + val_y - 1) // val_y

    @staticmethod
    def _get_inner_dtype(in_dtype):
        """
        get inner dtype
        """
        low_case_dtype = in_dtype.lower()
        if low_case_dtype == "bfloat16":
            return "float16"
        if low_case_dtype == "hfloat32":
            return "float32"
        return low_case_dtype

    def _tiling_branch_func(self, block_idx):
        """
        tiling branch function
        """
        # The input layout is: G, D, E, C1_in, HW, N, C0
        # The output layout is: D, C1_out, HW, G, E, N, C0
        with self.tik_inst.for_range(0, self.g_lp_cnt) as g_lp_idx:
            # never bind core on E
            with self.tik_inst.for_range(0, self.e) as e_lp_idx:
                with self.tik_inst.if_scope(e_lp_idx == 0):
                    self.e_in_offset.set_as(0)
                with self.tik_inst.else_scope():
                    self.e_in_offset.set_as(self.e_in_offset + (
                        (e_lp_idx - 1) * self.c % self.c0 + self.c) // self.c0 * self.hw * self.n_e_align * self.c0)
                # The G*E can be bigger than groups, so making the judge to avoid aicore error
                with self.tik_inst.if_scope(
                        tik.any(
                            tik.all(self.mc_pos == self.G_POS,
                                    (block_idx * self.nlc_g_lp_cnt + g_lp_idx) * self.e + e_lp_idx < self.g),
                            tik.all(self.mc_pos != self.G_POS, g_lp_idx * self.e + e_lp_idx < self.g))):
                    with self.tik_inst.for_range(0, self.n_ng_lp_cnt) as n_ng_lp_idx:
                        self._get_current_n_ng_unit(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        with self.tik_inst.if_scope(self.tiling_key == self.C_IS_C0_ALIGN_HW):
                            self._c_is_c0_align_hw(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        with self.tik_inst.if_scope(self.tiling_key == self.C_IS_C0_ALIGN_C1OUT):
                            self._c_is_c0_align_c1out(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        with self.tik_inst.if_scope(self.tiling_key == self.C_IS_C0_ALIGN_D):
                            self._c_is_c0_align_d(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        with self.tik_inst.if_scope(self.tiling_key == self.C_IS_C0_UNALIGN_HW):
                            self._c_is_c0_unalign_hw(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        with self.tik_inst.if_scope(self.tiling_key == self.C_IS_C0_UNALIGN_D):
                            self._c_is_c0_unalign_d(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        with self.tik_inst.if_scope(self.tiling_key == self.C_IS_C0_UNALIGN_HW_1):
                            self._c_is_c0_unalign_hw_1(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        with self.tik_inst.if_scope(self.tiling_key == self.C_IS_C0_UNALIGN_D_1):
                            self._c_is_c0_unalign_d_1(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)

    def _get_current_hw_unit(self, block_idx, hw_lp_idx):
        with self.tik_inst.if_scope(tik.all(hw_lp_idx == self.hw_lp_cnt - 1, self.hw % self.hw_lp_unit > 0)):
            with self.tik_inst.if_scope(
                    tik.any(self.mc_pos != self.HW_POS,
                            tik.all(block_idx == self.used_core_cnt - 1, self.mc_pos == self.HW_POS))):
                self.cur_hw_lp_unit.set_as(self.hw % self.hw_lp_unit)
            with self.tik_inst.else_scope():
                self.cur_hw_lp_unit.set_as(self.hw_lp_unit)
        with self.tik_inst.else_scope():
            self.cur_hw_lp_unit.set_as(self.hw_lp_unit)

    def _get_current_c1_out_unit(self, block_idx, c1_out_lp_idx):
        with self.tik_inst.if_scope(
                tik.all(c1_out_lp_idx == self.c1_out_lp_cnt - 1, self.c1_out % self.c1_out_lp_unit > 0)):
            with self.tik_inst.if_scope(
                    tik.any(self.mc_pos != self.C1_OUT_POS,
                            tik.all(block_idx == self.used_core_cnt - 1, self.mc_pos == self.C1_OUT_POS))):
                self.cur_c1_out_lp_unit.set_as(self.c1_out % self.c1_out_lp_unit)
            with self.tik_inst.else_scope():
                self.cur_c1_out_lp_unit.set_as(self.c1_out_lp_unit)
        with self.tik_inst.else_scope():
            self.cur_c1_out_lp_unit.set_as(self.c1_out_lp_unit)

    def _get_current_d_unit(self, block_idx, d_lp_idx):
        with self.tik_inst.if_scope(tik.all(d_lp_idx == self.d_lp_cnt - 1, self.d % self.d_lp_unit > 0)):
            with self.tik_inst.if_scope(
                    tik.any(self.mc_pos != self.D_POS,
                            tik.all(block_idx == self.used_core_cnt - 1, self.mc_pos == self.D_POS))):
                self.cur_d_lp_unit.set_as(self.d % self.d_lp_unit)
            with self.tik_inst.else_scope():
                self.cur_d_lp_unit.set_as(self.d_lp_unit)
        with self.tik_inst.else_scope():
            self.cur_d_lp_unit.set_as(self.d_lp_unit)

    def _get_current_n_ng_unit(self, block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx):
        with self.tik_inst.if_scope(tik.all(n_ng_lp_idx == self.n_ng_lp_cnt - 1, self.n_ng % self.n_ng_lp_unit > 0)):
            with self.tik_inst.if_scope(
                    tik.any(self.mc_pos != self.N_NG_POS,
                            tik.all(block_idx == self.used_core_cnt - 1, self.mc_pos == self.N_NG_POS))):
                self.cur_n_ng_lp_unit.set_as(self.n_ng % self.n_ng_lp_unit)
            with self.tik_inst.else_scope():
                self.cur_n_ng_lp_unit.set_as(self.n_ng_lp_unit)
        with self.tik_inst.else_scope():
            self.cur_n_ng_lp_unit.set_as(self.n_ng_lp_unit)

        with self.tik_inst.if_scope(
                tik.all(
                    self.g * self.n_ng % self.NI > 0,
                    tik.all(
                        n_ng_lp_idx == self.n_ng_lp_cnt - 1,
                        tik.any(self.mc_pos != self.N_NG_POS,
                                tik.all(block_idx == self.used_core_cnt - 1, self.mc_pos == self.N_NG_POS))),
                    tik.any(
                        tik.all(self.mc_pos == self.G_POS,
                                (block_idx * self.nlc_g_lp_cnt + g_lp_idx) * self.e + e_lp_idx == self.g - 1),
                        tik.all(self.mc_pos != self.G_POS, g_lp_idx * self.e + e_lp_idx == self.g - 1)))):
            self.cur_n_ng_lp_unit.set_as(self.cur_n_ng_lp_unit + self.NI - self.g * self.n_ng % self.NI)

    def _move_data_in(self, in_offset, repeat, src_stride_offset, dst_stride_offset):
        """
        The layout is: H0, NC0_1, (NC0_2,) ...
                       H1, NC0_1, (NC0_2,) ...
                       . , .    , (.    ,) ...
                       H15,NC0_1, (NC0_2,) ...
        """
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, self.c0_parts) as c0_idx:
                c0_part_gm_offset = c0_idx * self.hw * self.n_e_align * self.c0
                c0_part_ub_offset = c0_idx * self.cur_n_ng_lp_unit * self.c0
                src_stride = (src_stride_offset - self.cur_n_ng_lp_unit * self.c0) // self.elem_per_block
                dst_stride = (dst_stride_offset - self.cur_n_ng_lp_unit * self.c0) // self.elem_per_block
                with self.tik_inst.if_scope(src_stride <= self.STRIDE_LIMIT):
                    self.tik_inst.data_move(self.data_ub[c0_part_ub_offset], self.gm_in[in_offset + c0_part_gm_offset],
                                            0, repeat, self.cur_n_ng_lp_unit * self.c0 // self.elem_per_block,
                                            src_stride, dst_stride)
                with self.tik_inst.else_scope():
                    with self.tik_inst.for_range(0, repeat) as hw_idx:
                        self.tik_inst.data_move(self.data_ub[c0_part_ub_offset + hw_idx * dst_stride_offset],
                                                self.gm_in[in_offset + c0_part_gm_offset + hw_idx * src_stride_offset],
                                                0, 1, self.cur_n_ng_lp_unit * self.c0 // self.elem_per_block, 0, 0)

    def _move_data_out(self, out_offset, repeat, src_stride_offset, dst_stride_offset):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            src_stride = (src_stride_offset - self.cur_n_ng_lp_unit * self.c0) // self.elem_per_block
            dst_stride = (dst_stride_offset - self.cur_n_ng_lp_unit * self.c0) // self.elem_per_block
            with self.tik_inst.if_scope(dst_stride <= self.STRIDE_LIMIT):
                self.tik_inst.data_move(self.gm_out[out_offset], self.data_ub[self.out_ub_offset], 0, repeat,
                                        self.cur_n_ng_lp_unit * self.c0 // self.elem_per_block, src_stride, dst_stride)
            with self.tik_inst.else_scope():
                with self.tik_inst.for_range(0, repeat) as hw_idx:
                    self.tik_inst.data_move(self.gm_out[out_offset + hw_idx * dst_stride_offset],
                                            self.data_ub[self.out_ub_offset + hw_idx * src_stride_offset], 0, 1,
                                            self.cur_n_ng_lp_unit * self.c0 // self.elem_per_block, 0, 0)

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

    def _move_to_target_layout(self, e_lp_idx):
        """
        move elements to target layout
        """
        with self.tik_inst.if_scope(self.c0_parts == 1):
            clean_ubuf(self.tik_inst, self.data_ub_b16, 0, self.ub_offset * self.dtype_bytes // self.b16_bytes)

        with self.tik_inst.new_stmt_scope(disable_sync=True):
            cp2_len = e_lp_idx * self.c % self.c0 * self.dtype_bytes // self.b16_bytes
            cp1_len = self.c0 * self.dtype_bytes // self.b16_bytes - cp2_len
            self.tik_inst.data_move(
                self.data_ub_b16,
                self.data_ub_b16[self.ub_offset * self.dtype_bytes // self.b16_bytes + cp2_len * self.VNC_ROWS], 0,
                self.cur_n_ng_lp_unit, cp1_len, cp2_len, cp2_len)
            with self.tik_inst.if_scope(self.c0_parts > 1):
                target_offset = cp1_len * self.VNC_ROWS
                source_offset = (self.ub_offset * self.dtype_bytes // self.b16_bytes +
                                 self.cur_n_ng_lp_unit * self.c0 * self.dtype_bytes // self.b16_bytes * self.VNC_ROWS)
                self.tik_inst.data_move(self.data_ub_b16[target_offset], self.data_ub_b16[source_offset], 0,
                                        self.cur_n_ng_lp_unit, cp2_len, cp1_len, cp1_len)

    def _move_to_target_layout_4_one_row(self, e_lp_idx, n_cube_cnt):
        """
        move elements to target layout when all valid data is in one row
        """
        with self.tik_inst.if_scope(self.c0_parts == 1):
            clean_ubuf(self.tik_inst, self.data_ub_b16, 0, self.ub_offset * self.dtype_bytes // self.b16_bytes)

        with self.tik_inst.new_stmt_scope(disable_sync=True):
            cp2_len = e_lp_idx * self.c % self.c0 * self.dtype_bytes // self.b16_bytes
            cp1_len = self.c0 * self.dtype_bytes // self.b16_bytes - cp2_len
            with self.tik_inst.for_range(0, self.cur_n_ng_lp_unit) as n_ng_idx:
                src_stride = (self.cur_n_ng_lp_unit * self.c0 * self.dtype_bytes // self.b16_bytes * self.c0_parts -
                              cp1_len)
                dst_stride = (self.cur_n_ng_lp_unit * self.c0 * self.dtype_bytes // self.b16_bytes - cp1_len)
                src_offset = (self.ub_offset * self.dtype_bytes // self.b16_bytes +
                              (cp2_len + n_ng_idx * self.c0 * self.dtype_bytes // self.b16_bytes) * self.VNC_ROWS)
                dst_offset = (n_ng_idx * self.c0 * self.dtype_bytes // self.b16_bytes * self.VNC_ROWS)
                self.tik_inst.data_move(self.data_ub_b16[dst_offset], self.data_ub_b16[src_offset], 0, n_cube_cnt,
                                        cp1_len, src_stride, dst_stride)
            with self.tik_inst.if_scope(self.c0_parts > 1):
                with self.tik_inst.for_range(0, self.cur_n_ng_lp_unit) as n_ng_idx_1:
                    src_stride = (self.cur_n_ng_lp_unit * self.c0 * self.dtype_bytes // self.b16_bytes * self.c0_parts -
                                  cp2_len)
                    dst_stride = (self.cur_n_ng_lp_unit * self.c0 * self.dtype_bytes // self.b16_bytes - cp2_len)
                    src_offset = (self.ub_offset * self.dtype_bytes // self.b16_bytes +
                                  (self.cur_n_ng_lp_unit * self.c0 * self.dtype_bytes // self.b16_bytes +
                                   n_ng_idx_1 * self.c0 * self.dtype_bytes // self.b16_bytes) * self.VNC_ROWS)
                    dst_offset = (n_ng_idx_1 * self.c0 * self.dtype_bytes // self.b16_bytes + cp1_len) * self.VNC_ROWS
                    self.tik_inst.data_move(self.data_ub_b16[dst_offset], self.data_ub_b16[src_offset], 0, n_cube_cnt,
                                            cp2_len, src_stride, dst_stride)

    def _move_to_target_layout_b8(self, e_lp_idx):
        """
        move elements to target layout for b8
        """
        with self.tik_inst.if_scope(self.c0_parts == 1):
            clean_ubuf(self.tik_inst, self.data_ub_b16, 0, self.ub_offset * self.dtype_bytes // self.b16_bytes)

        with self.tik_inst.new_stmt_scope(disable_sync=True):
            cp2_len = e_lp_idx * self.c % self.c0
            cp1_len = self.c0 - cp2_len
            col_factor = 2
            self.tik_inst.data_move(self.data_ub, self.data_ub[self.ub_offset + cp2_len * self.VNC_ROWS * col_factor],
                                    0, self.cur_n_ng_lp_unit, cp1_len, cp2_len, cp2_len)
            with self.tik_inst.if_scope(self.c0_parts > 1):
                target_offset = cp1_len * self.VNC_ROWS * col_factor
                source_offset = (self.ub_offset + self.cur_n_ng_lp_unit * self.c0 * self.VNC_ROWS * col_factor)
                self.tik_inst.data_move(self.data_ub[target_offset], self.data_ub[source_offset], 0,
                                        self.cur_n_ng_lp_unit, cp2_len, cp1_len, cp1_len)

    def _move_to_target_layout_b8_4_one_row(self, e_lp_idx, n_cube_cnt):
        """
        move elements to target layout for b8 when all valid data is in one row
        """
        with self.tik_inst.if_scope(self.c0_parts == 1):
            clean_ubuf(self.tik_inst, self.data_ub_b16, 0, self.ub_offset * self.dtype_bytes // self.b16_bytes)

        with self.tik_inst.new_stmt_scope(disable_sync=True):
            cp2_len = e_lp_idx * self.c % self.c0
            cp1_len = self.c0 - cp2_len
            col_factor = 2
            with self.tik_inst.for_range(0, self.cur_n_ng_lp_unit) as n_ng_idx:
                src_stride = (self.cur_n_ng_lp_unit * self.c0 * self.c0_parts - cp1_len)
                dst_stride = (self.cur_n_ng_lp_unit * self.c0 - cp1_len)
                src_offset = (self.ub_offset + (cp2_len + n_ng_idx * self.c0) * self.VNC_ROWS * col_factor)
                dst_offset = (n_ng_idx * self.c0 * self.VNC_ROWS * col_factor)
                self.tik_inst.data_move(self.data_ub[dst_offset], self.data_ub[src_offset], 0, n_cube_cnt, cp1_len,
                                        src_stride, dst_stride)
            with self.tik_inst.if_scope(self.c0_parts > 1):
                with self.tik_inst.for_range(0, self.cur_n_ng_lp_unit) as n_ng_idx_1:
                    src_stride = (self.cur_n_ng_lp_unit * self.c0 * self.c0_parts - cp2_len)
                    dst_stride = (self.cur_n_ng_lp_unit * self.c0 - cp2_len)
                    src_offset = (self.ub_offset +
                                  (self.cur_n_ng_lp_unit * self.c0 + n_ng_idx_1 * self.c0) * self.VNC_ROWS * col_factor)
                    dst_offset = (n_ng_idx_1 * self.c0 + cp1_len) * self.VNC_ROWS * col_factor
                    self.tik_inst.data_move(self.data_ub[dst_offset], self.data_ub[src_offset], 0, n_cube_cnt, cp2_len,
                                            src_stride, dst_stride)

    def _transpose_by_vnchwconv_b16(self, n_cube_cnt):
        """
        transpose two axises by vnchwconv for b16 dtype
        """
        src_addrs = [
            self.data_ub_b16[self.max_n_per_row * self.c0 * self.dtype_bytes // self.b16_bytes * i]
            for i in self.ADDR_IDX
        ]
        dst_addrs = [
            self.data_ub_b16[self.VNC_ROWS * i + self.ub_offset * self.dtype_bytes // self.b16_bytes]
            for i in self.ADDR_IDX
        ]
        repeat_cnt = self._ceil_div(
            n_cube_cnt * self.cur_n_ng_lp_unit * self.c0 * self.dtype_bytes // self.b16_bytes * self.c0_parts,
            self.VNC_ROWS)
        self._set_vnchwconv_stride(repeat_cnt, 1, self.VNC_ROWS)
        self.tik_inst.vnchwconv(False, False, dst_addrs, src_addrs, repeat_cnt, self.vnc_dst_stride,
                                self.vnc_src_stride)

    def _transpose_by_vnchwconv_b8(self, n_cube_cnt):
        """
        transpose two axises by vnchwconv for b8 dtype
        """
        src_addrs = [self.data_ub[self.max_n_per_row * self.c0 * i] for i in self.ADDR_IDX]
        dst_addrs = [self.data_ub[self.elem_per_block * i + self.ub_offset] for i in self.ADDR_IDX]
        repeat_cnt = self._ceil_div(n_cube_cnt * self.cur_n_ng_lp_unit * self.c0 * self.c0_parts, self.elem_per_block)
        self._set_vnchwconv_stride(repeat_cnt, 1, self.elem_per_block)
        self.tik_inst.vnchwconv(False, False, dst_addrs, src_addrs, repeat_cnt, self.vnc_dst_stride,
                                self.vnc_src_stride)
        dst_addrs = [self.data_ub[self.elem_per_block * (i + self.VNC_ROWS) + self.ub_offset] for i in self.ADDR_IDX]
        self.tik_inst.vnchwconv(False, True, dst_addrs, src_addrs, repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)

    def _transpose_back_by_vnchwconv_b16(self, n_cube_cnt):
        """
        transpose two axises back by vnchwconv for b16 dtype
        """
        src_addrs = [self.data_ub_b16[self.VNC_ROWS * i] for i in self.ADDR_IDX]
        dst_addrs = [
            self.data_ub_b16[(self.max_n_per_row * self.c0 * i + self.ub_offset) * self.dtype_bytes // self.b16_bytes]
            for i in self.ADDR_IDX
        ]
        repeat_cnt = self._ceil_div(n_cube_cnt * self.cur_n_ng_lp_unit * self.c0 * self.dtype_bytes // self.b16_bytes,
                                    self.VNC_ROWS)
        self._set_vnchwconv_stride(repeat_cnt, self.VNC_ROWS, 1)
        self.tik_inst.vnchwconv(False, False, dst_addrs, src_addrs, repeat_cnt, self.vnc_dst_stride,
                                self.vnc_src_stride)

    def _transpose_back_by_vnchwconv_b8(self, n_cube_cnt):
        """
        transpose two axises back by vnchwconv for b8 dtype
        """
        src_addrs = [self.data_ub[self.elem_per_block * i] for i in self.ADDR_IDX]
        dst_addrs = [self.data_ub[self.max_n_per_row * self.c0 * i + self.ub_offset] for i in self.ADDR_IDX]
        repeat_cnt = self._ceil_div(n_cube_cnt * self.cur_n_ng_lp_unit * self.c0, self.elem_per_block)
        self._set_vnchwconv_stride(repeat_cnt, self.elem_per_block, 1)
        self.tik_inst.vnchwconv(False, False, dst_addrs, src_addrs, repeat_cnt, self.vnc_dst_stride,
                                self.vnc_src_stride)
        src_addrs = [self.data_ub[self.elem_per_block * (i + self.VNC_ROWS)] for i in self.ADDR_IDX]
        self.tik_inst.vnchwconv(True, False, dst_addrs, src_addrs, repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)

    def _transform_by_vnchwconv(self, e_lp_idx, n_cube_cnt):
        """
        reorder elements by vnchwconv
        """
        if self.elem_per_block != self.BYTES_PER_BLOCK:
            self._transpose_by_vnchwconv_b16(n_cube_cnt)
            with self.tik_inst.if_scope(
                    tik.any(self.tiling_key == self.C_IS_C0_UNALIGN_HW, self.tiling_key == self.C_IS_C0_UNALIGN_D)):
                self._move_to_target_layout(e_lp_idx)
            with self.tik_inst.else_scope():
                self._move_to_target_layout_4_one_row(e_lp_idx, n_cube_cnt)
            self._transpose_back_by_vnchwconv_b16(n_cube_cnt)
        else:
            with self.tik_inst.if_scope(e_lp_idx * self.c % self.c0 % 2 != 0):
                self._transpose_by_vnchwconv_b8(n_cube_cnt)
                with self.tik_inst.if_scope(
                        tik.any(self.tiling_key == self.C_IS_C0_UNALIGN_HW, self.tiling_key == self.C_IS_C0_UNALIGN_D)):
                    self._move_to_target_layout_b8(e_lp_idx)
                with self.tik_inst.else_scope():
                    self._move_to_target_layout_b8_4_one_row(e_lp_idx, n_cube_cnt)
                self._transpose_back_by_vnchwconv_b8(n_cube_cnt)
            with self.tik_inst.else_scope():
                self._transpose_by_vnchwconv_b16(n_cube_cnt)
                with self.tik_inst.if_scope(
                        tik.any(self.tiling_key == self.C_IS_C0_UNALIGN_HW, self.tiling_key == self.C_IS_C0_UNALIGN_D)):
                    self._move_to_target_layout(e_lp_idx)
                with self.tik_inst.else_scope():
                    self._move_to_target_layout_4_one_row(e_lp_idx, n_cube_cnt)
                self._transpose_back_by_vnchwconv_b16(n_cube_cnt)

    def _get_c0_parts(self, block_idx, e_lp_idx, c1_out_lp_idx):
        with self.tik_inst.if_scope(
                tik.any(
                    tik.all(
                        self.mc_pos != self.C1_OUT_POS,
                        tik.any(
                            tik.all(c1_out_lp_idx < self.c1_out - 1, e_lp_idx * self.c % self.c0 > 0),
                            tik.all(c1_out_lp_idx == self.c1_out - 1,
                                    e_lp_idx * self.c % self.c0 + self.c % self.c0 > self.c0))),
                    tik.all(
                        self.mc_pos == self.C1_OUT_POS,
                        tik.any(
                            tik.all(block_idx * self.nlc_c1_out_lp_cnt + c1_out_lp_idx < self.c1_out - 1,
                                    e_lp_idx * self.c % self.c0 > 0),
                            tik.all(block_idx * self.nlc_c1_out_lp_cnt + c1_out_lp_idx == self.c1_out - 1,
                                    e_lp_idx * self.c % self.c0 + self.c % self.c0 > self.c0))))):
            self.c0_parts.set_as(2)
        with self.tik_inst.else_scope():
            self.c0_parts.set_as(1)

    # 'pylint: disable=too-many-locals
    def _c_is_c0_align_hw(self, block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx):
        """
        c is aligned with c0 and hw loop unit is larger than 1
        """
        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
        e_out_offset = e_lp_idx * self.n_ng * self.c0
        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0
        self.out_ub_offset.set_as(0)

        with self.tik_inst.for_range(0, self.d_lp_cnt) as d_lp_idx:
            d_in_offset = d_lp_idx * self.c1_in * self.hw * self.n_e_align * self.c0
            d_out_offset = d_lp_idx * self.c1_out * self.hw * self.n_gp_align * self.c0
            with self.tik_inst.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                c1_in_offset = c1_out_lp_idx * self.hw * self.n_e_align * self.c0
                c1_out_offset = c1_out_lp_idx * self.hw * self.n_gp_align * self.c0
                with self.tik_inst.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                    hw_in_offset = hw_lp_idx * self.hw_lp_unit * self.n_e_align * self.c0
                    hw_out_offset = hw_lp_idx * self.hw_lp_unit * self.n_gp_align * self.c0
                    self._get_current_hw_unit(block_idx, hw_lp_idx)

                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset + n_in_offset +
                                 hw_in_offset + block_idx * self.core_step_in)
                    in_src_stride = self.n_e_align * self.c0
                    in_dst_stride = self.max_n_per_row * self.c0
                    self._move_data_in(in_offset, self.cur_hw_lp_unit, in_src_stride, in_dst_stride)
                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset + n_out_offset +
                                  hw_out_offset + block_idx * self.core_step_out)
                    out_src_stride = in_dst_stride
                    out_dst_stride = self.n_gp_align * self.c0
                    self._move_data_out(out_offset, self.cur_hw_lp_unit, out_src_stride, out_dst_stride)

    def _c_is_c0_align_c1out(self, block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx):
        """
        c is aligned with c0 and c1_out loop unit is larger than 1
        """
        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
        e_out_offset = e_lp_idx * self.n_ng * self.c0
        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0
        self.out_ub_offset.set_as(0)

        with self.tik_inst.for_range(0, self.d_lp_cnt) as d_lp_idx:
            d_in_offset = d_lp_idx * self.c1_in * self.hw * self.n_e_align * self.c0
            d_out_offset = d_lp_idx * self.c1_out * self.hw * self.n_gp_align * self.c0
            with self.tik_inst.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                c1_in_offset = c1_out_lp_idx * self.c1_out_lp_unit * self.hw * self.n_e_align * self.c0
                c1_out_offset = c1_out_lp_idx * self.c1_out_lp_unit * self.hw * self.n_gp_align * self.c0
                self._get_current_c1_out_unit(block_idx, c1_out_lp_idx)
                with self.tik_inst.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                    hw_in_offset = hw_lp_idx * self.n_e_align * self.c0
                    hw_out_offset = hw_lp_idx * self.n_gp_align * self.c0

                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset + n_in_offset +
                                 hw_in_offset + block_idx * self.core_step_in)
                    in_src_stride = self.hw * self.n_e_align * self.c0
                    in_dst_stride = self.max_n_per_row * self.c0
                    self._move_data_in(in_offset, self.cur_c1_out_lp_unit, in_src_stride, in_dst_stride)
                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset + n_out_offset +
                                  hw_out_offset + block_idx * self.core_step_out)
                    out_src_stride = in_dst_stride
                    out_dst_stride = self.hw * self.n_gp_align * self.c0
                    self._move_data_out(out_offset, self.cur_c1_out_lp_unit, out_src_stride, out_dst_stride)

    def _c_is_c0_align_d(self, block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx):
        """
        c is aligned with c0 and d loop unit is larger than 1
        """
        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
        e_out_offset = e_lp_idx * self.n_ng * self.c0
        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0
        self.out_ub_offset.set_as(0)

        with self.tik_inst.for_range(0, self.d_lp_cnt) as d_lp_idx:
            d_in_offset = d_lp_idx * self.d_lp_unit * self.c1_in * self.hw * self.n_e_align * self.c0
            d_out_offset = d_lp_idx * self.d_lp_unit * self.c1_out * self.hw * self.n_gp_align * self.c0
            self._get_current_d_unit(block_idx, d_lp_idx)
            with self.tik_inst.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                c1_in_offset = c1_out_lp_idx * self.hw * self.n_e_align * self.c0
                c1_out_offset = c1_out_lp_idx * self.hw * self.n_gp_align * self.c0
                with self.tik_inst.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                    hw_in_offset = hw_lp_idx * self.n_e_align * self.c0
                    hw_out_offset = hw_lp_idx * self.n_gp_align * self.c0

                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset + n_in_offset +
                                 hw_in_offset + block_idx * self.core_step_in)
                    in_src_stride = self.c1_in * self.hw * self.n_e_align * self.c0
                    in_dst_stride = self.max_n_per_row * self.c0
                    self._move_data_in(in_offset, self.cur_d_lp_unit, in_src_stride, in_dst_stride)
                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset + n_out_offset +
                                  hw_out_offset + block_idx * self.core_step_out)
                    out_src_stride = in_dst_stride
                    out_dst_stride = self.c1_out * self.hw * self.n_gp_align * self.c0
                    self._move_data_out(out_offset, self.cur_d_lp_unit, out_src_stride, out_dst_stride)

    def _c_is_c0_unalign_hw(self, block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx):
        """
        c is not aligned with c0 and hw loop unit is larger than 1
        """
        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
        e_out_offset = e_lp_idx * self.n_ng * self.c0
        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0

        with self.tik_inst.for_range(0, self.d_lp_cnt) as d_lp_idx:
            d_in_offset = d_lp_idx * self.c1_in * self.hw * self.n_e_align * self.c0
            d_out_offset = d_lp_idx * self.c1_out * self.hw * self.n_gp_align * self.c0
            with self.tik_inst.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                c1_in_offset = c1_out_lp_idx * self.hw * self.n_e_align * self.c0
                c1_out_offset = c1_out_lp_idx * self.hw * self.n_gp_align * self.c0
                self._get_c0_parts(block_idx, e_lp_idx, c1_out_lp_idx)
                with self.tik_inst.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                    hw_in_offset = hw_lp_idx * self.hw_lp_unit * self.n_e_align * self.c0
                    hw_out_offset = hw_lp_idx * self.hw_lp_unit * self.n_gp_align * self.c0
                    self._get_current_hw_unit(block_idx, hw_lp_idx)

                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset + n_in_offset +
                                 hw_in_offset + block_idx * self.core_step_in)
                    in_src_stride = self.n_e_align * self.c0
                    in_dst_stride = self.max_n_per_row * self.c0
                    self._move_data_in(in_offset, self.cur_hw_lp_unit, in_src_stride, in_dst_stride)

                    with self.tik_inst.if_scope(e_lp_idx * self.c % self.c0 > 0):
                        self._transform_by_vnchwconv(e_lp_idx, 1)
                        self.out_ub_offset.set_as(self.ub_offset)
                    with self.tik_inst.else_scope():
                        self.out_ub_offset.set_as(0)

                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset + n_out_offset +
                                  hw_out_offset + block_idx * self.core_step_out)
                    out_src_stride = in_dst_stride
                    out_dst_stride = self.n_gp_align * self.c0
                    self._move_data_out(out_offset, self.cur_hw_lp_unit, out_src_stride, out_dst_stride)

    def _c_is_c0_unalign_d(self, block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx):
        """
        c is not aligned with c0 and d loop unit is larger than 1
        """
        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
        e_out_offset = e_lp_idx * self.n_ng * self.c0
        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0

        with self.tik_inst.for_range(0, self.d_lp_cnt) as d_lp_idx:
            d_in_offset = d_lp_idx * self.d_lp_unit * self.c1_in * self.hw * self.n_e_align * self.c0
            d_out_offset = d_lp_idx * self.d_lp_unit * self.c1_out * self.hw * self.n_gp_align * self.c0
            self._get_current_d_unit(block_idx, d_lp_idx)
            with self.tik_inst.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                c1_in_offset = c1_out_lp_idx * self.hw * self.n_e_align * self.c0
                c1_out_offset = c1_out_lp_idx * self.hw * self.n_gp_align * self.c0
                self._get_c0_parts(block_idx, e_lp_idx, c1_out_lp_idx)
                with self.tik_inst.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                    hw_in_offset = hw_lp_idx * self.n_e_align * self.c0
                    hw_out_offset = hw_lp_idx * self.n_gp_align * self.c0

                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset + n_in_offset +
                                 hw_in_offset + block_idx * self.core_step_in)
                    in_src_stride = self.c1_in * self.hw * self.n_e_align * self.c0
                    in_dst_stride = self.max_n_per_row * self.c0
                    self._move_data_in(in_offset, self.cur_d_lp_unit, in_src_stride, in_dst_stride)

                    with self.tik_inst.if_scope(e_lp_idx * self.c % self.c0 > 0):
                        self._transform_by_vnchwconv(e_lp_idx, 1)
                        self.out_ub_offset.set_as(self.ub_offset)
                    with self.tik_inst.else_scope():
                        self.out_ub_offset.set_as(0)

                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset + n_out_offset +
                                  hw_out_offset + block_idx * self.core_step_out)
                    out_src_stride = in_dst_stride
                    out_dst_stride = self.c1_out * self.hw * self.n_gp_align * self.c0
                    self._move_data_out(out_offset, self.cur_d_lp_unit, out_src_stride, out_dst_stride)

    def _c_is_c0_unalign_hw_1(self, block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx):
        """
        c is not aligned with c0, hw loop unit is larger than 1 and n loop unit is small
        """
        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
        e_out_offset = e_lp_idx * self.n_ng * self.c0
        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0

        with self.tik_inst.for_range(0, self.d_lp_cnt) as d_lp_idx:
            d_in_offset = d_lp_idx * self.c1_in * self.hw * self.n_e_align * self.c0
            d_out_offset = d_lp_idx * self.c1_out * self.hw * self.n_gp_align * self.c0
            with self.tik_inst.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                c1_in_offset = c1_out_lp_idx * self.hw * self.n_e_align * self.c0
                c1_out_offset = c1_out_lp_idx * self.hw * self.n_gp_align * self.c0
                self._get_c0_parts(block_idx, e_lp_idx, c1_out_lp_idx)
                with self.tik_inst.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                    hw_in_offset = hw_lp_idx * self.hw_lp_unit * self.n_e_align * self.c0
                    hw_out_offset = hw_lp_idx * self.hw_lp_unit * self.n_gp_align * self.c0
                    self._get_current_hw_unit(block_idx, hw_lp_idx)

                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset + n_in_offset +
                                 hw_in_offset + block_idx * self.core_step_in)
                    in_src_stride = self.n_e_align * self.c0
                    in_dst_stride = self.c0_parts * self.cur_n_ng_lp_unit * self.c0
                    self._move_data_in(in_offset, self.cur_hw_lp_unit, in_src_stride, in_dst_stride)

                    with self.tik_inst.if_scope(e_lp_idx * self.c % self.c0 > 0):
                        self._transform_by_vnchwconv(e_lp_idx, self.cur_hw_lp_unit)
                        self.out_ub_offset.set_as(self.ub_offset)
                    with self.tik_inst.else_scope():
                        self.out_ub_offset.set_as(0)

                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset + n_out_offset +
                                  hw_out_offset + block_idx * self.core_step_out)
                    out_src_stride = self.cur_n_ng_lp_unit * self.c0
                    out_dst_stride = self.n_gp_align * self.c0
                    self._move_data_out(out_offset, self.cur_hw_lp_unit, out_src_stride, out_dst_stride)

    def _c_is_c0_unalign_d_1(self, block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx):
        """
        c is not aligned with c0, d loop unit is larger than 1 and n loop unit is small
        """
        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
        e_out_offset = e_lp_idx * self.n_ng * self.c0
        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0

        with self.tik_inst.for_range(0, self.d_lp_cnt) as d_lp_idx:
            d_in_offset = d_lp_idx * self.d_lp_unit * self.c1_in * self.hw * self.n_e_align * self.c0
            d_out_offset = d_lp_idx * self.d_lp_unit * self.c1_out * self.hw * self.n_gp_align * self.c0
            self._get_current_d_unit(block_idx, d_lp_idx)
            with self.tik_inst.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                c1_in_offset = c1_out_lp_idx * self.hw * self.n_e_align * self.c0
                c1_out_offset = c1_out_lp_idx * self.hw * self.n_gp_align * self.c0
                self._get_c0_parts(block_idx, e_lp_idx, c1_out_lp_idx)
                with self.tik_inst.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                    hw_in_offset = hw_lp_idx * self.n_e_align * self.c0
                    hw_out_offset = hw_lp_idx * self.n_gp_align * self.c0

                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset + n_in_offset +
                                 hw_in_offset + block_idx * self.core_step_in)
                    in_src_stride = self.c1_in * self.hw * self.n_e_align * self.c0
                    in_dst_stride = self.c0_parts * self.cur_n_ng_lp_unit * self.c0
                    self._move_data_in(in_offset, self.cur_d_lp_unit, in_src_stride, in_dst_stride)

                    with self.tik_inst.if_scope(e_lp_idx * self.c % self.c0 > 0):
                        self._transform_by_vnchwconv(e_lp_idx, self.cur_d_lp_unit)
                        self.out_ub_offset.set_as(self.ub_offset)
                    with self.tik_inst.else_scope():
                        self.out_ub_offset.set_as(0)

                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset + n_out_offset +
                                  hw_out_offset + block_idx * self.core_step_out)
                    out_src_stride = self.cur_n_ng_lp_unit * self.c0
                    out_dst_stride = self.c1_out * self.hw * self.n_gp_align * self.c0
                    self._move_data_out(out_offset, self.cur_d_lp_unit, out_src_stride, out_dst_stride)

    def _compute_tiling(self):
        """
        tiling entrance function
        """
        with self.tik_inst.for_range(0, self.used_core_cnt, block_num=self.used_core_cnt) as block_idx:
            with self.tik_inst.if_scope(block_idx != self.used_core_cnt - 1):
                self.g_lp_cnt.set_as(self.nlc_g_lp_cnt)
                self.d_lp_cnt.set_as(self.nlc_d_lp_cnt)
                self.c1_out_lp_cnt.set_as(self.nlc_c1_out_lp_cnt)
                self.hw_lp_cnt.set_as(self.nlc_hw_lp_cnt)
                self.n_ng_lp_cnt.set_as(self.nlc_n_ng_lp_cnt)
            with self.tik_inst.else_scope():
                self.g_lp_cnt.set_as(self.lc_g_lp_cnt)
                self.d_lp_cnt.set_as(self.lc_d_lp_cnt)
                self.c1_out_lp_cnt.set_as(self.lc_c1_out_lp_cnt)
                self.hw_lp_cnt.set_as(self.lc_hw_lp_cnt)
                self.n_ng_lp_cnt.set_as(self.lc_n_ng_lp_cnt)
            self._tiling_branch_func(block_idx)

    def compute(self):
        """
        entrance function
        """
        tbe_context.get_context().add_compile_info("is_fzg2fz", True)
        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.core_num,
            "max_elem_cnt": self.max_elem_cnt,
            "elem_per_block": self.elem_per_block
        })

        self._compute_tiling()

        opt_config = {"enable_const_fold": True}
        is_dynamic = is_unknown([self.src])
        if is_dynamic:
            tiling_map = {
                "tiling_key": [self.tiling_key],
                "tiling_key_value": [[self.C_IS_C0_ALIGN_HW], [self.C_IS_C0_ALIGN_C1OUT], [self.C_IS_C0_ALIGN_D],
                                     [self.C_IS_C0_UNALIGN_HW], [self.C_IS_C0_UNALIGN_D], [self.C_IS_C0_UNALIGN_HW_1],
                                     [self.C_IS_C0_UNALIGN_D_1]]
            }
            self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.gm_in, ),
                                   outputs=(self.gm_out, ),
                                   flowtable=(self.tiling_gm, ),
                                   config=opt_config,
                                   extend_params={"build_multi_kernels": tiling_map})
        else:
            self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.gm_in, ),
                                   outputs=(self.gm_out, ),
                                   flowtable=(self.tiling_gm, ),
                                   config=opt_config)


def fzg_2_fz(src, kernel_name="fzg_2_fz"):
    """
    Do transform from groups fractal_z to fractal_z.

    Parameters
    ----------
    src : the input tensor
    kernel_name: operator name, default value is "fzg_2_fz"
    Returns
    -------
    None
    """

    fzg2fz_instance = FzgToFz(src, kernel_name)
    fzg2fz_instance.compute()
