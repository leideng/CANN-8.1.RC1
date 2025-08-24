#!/usr/env/bin python
# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
transpose_d_014253_big
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl import trans_data_common_func as tdc


# used for scalar
PAD_IDX_LIST = (0, 1, 2)
# frame up levels
FRAME_LEVEL = 3


# 'pylint: disable=too-many-locals,inconsistent-return-statements,too-many-lines,too-many-return-statements
# 'pylint: disable=too-many-statements,unused-argument
def _renew_input_output_perm(in_shape, out_shape, perm):
    """
    renew shape and perm to adapt tiling process
    """

    new_perm = [0, 3, 1, 4, 2]
    new_in_shape = [tdc.get_shape_size(in_shape[:2])] + [in_shape[2]] + [in_shape[3]] + [in_shape[4]] + [in_shape[5]]
    new_out_shape = [tdc.get_shape_size(out_shape[:2])] + [out_shape[2]] + [out_shape[3]] + \
                    [out_shape[4]] + [out_shape[5]]

    return [new_in_shape, new_out_shape, new_perm]


# 'pylint: disable=too-many-statements,unused-variable
def _get_mc_info_transpose(axis_lp_args, tp_info):
    """
    get multiple core axis position for transpose transform
    """

    dst_r_lp_cnt, dst_r_left, dst_c_lp_cnt, dst_c_left, dst_l_lp_cnt, dst_l_left = axis_lp_args

    tmp_full_lp_cnt_r = tdc.CORE_DIM_NUM if tdc.floor_div(dst_r_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_r = dst_r_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_r == 0 and dst_r_left > tp_info.get("tp_dst_r_lp_unit") // 2:
        tmp_full_lp_cnt_r += tdc.CORE_DIM_NUM
    full_lp_cnt_r = tmp_full_lp_cnt_r + reminder_lp_cnt_r

    tmp_full_lp_cnt_c = tdc.CORE_DIM_NUM if tdc.floor_div(dst_c_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_c = dst_c_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_c == 0:
        tmp_full_lp_cnt_c += tdc.CORE_DIM_NUM
    full_lp_cnt_c = tmp_full_lp_cnt_c + reminder_lp_cnt_c

    tmp_full_lp_cnt_l = tdc.CORE_DIM_NUM if tdc.floor_div(dst_l_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_l = dst_l_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_l == 0:
        tmp_full_lp_cnt_l += tdc.CORE_DIM_NUM
    full_lp_cnt_l = tmp_full_lp_cnt_l + reminder_lp_cnt_l

    lp_cnt_list = (full_lp_cnt_l, full_lp_cnt_r, full_lp_cnt_c)
    if lp_cnt_list.index(max(lp_cnt_list)) == 0:
        tp_info["tp_mc_pos"] = 0
        tp_info["tp_used_core_cnt"] = tdc.ceil_div(dst_l_lp_cnt, tdc.ceil_div(dst_l_lp_cnt, tdc.CORE_DIM_NUM))
        tp_info["tp_nlc_l_lp_cnt"] = tdc.ceil_div(dst_l_lp_cnt, tp_info["tp_used_core_cnt"])
        tp_info["tp_lc_l_lp_cnt"] = dst_l_lp_cnt - tp_info["tp_nlc_l_lp_cnt"] * (tp_info.get("tp_used_core_cnt") - 1)
        tp_info["tp_core_step_in"] = tp_info["tp_nlc_l_lp_cnt"] * tp_info["tp_dst_l_lp_step_in"]
        tp_info["tp_core_step_out"] = tp_info["tp_nlc_l_lp_cnt"] * tp_info["tp_dst_l_lp_step_out"]
        tp_info["tp_nlc_l_left"] = 0
        tp_info["tp_lc_l_left"] = dst_l_left
        tp_info["tp_nlc_c_lp_cnt"] = dst_c_lp_cnt
        tp_info["tp_lc_c_lp_cnt"] = dst_c_lp_cnt
        tp_info["tp_nlc_c_left"] = dst_c_left
        tp_info["tp_lc_c_left"] = dst_c_left
        tp_info["tp_nlc_r_lp_cnt"] = dst_r_lp_cnt
        tp_info["tp_lc_r_lp_cnt"] = dst_r_lp_cnt
        tp_info["tp_nlc_r_left"] = dst_r_left
        tp_info["tp_lc_r_left"] = dst_r_left
    elif lp_cnt_list.index(max(lp_cnt_list)) == 1:
        tp_info["tp_mc_pos"] = 1
        tp_info["tp_used_core_cnt"] = tdc.ceil_div(dst_r_lp_cnt, tdc.ceil_div(dst_r_lp_cnt, tdc.CORE_DIM_NUM))
        tp_info["tp_nlc_r_lp_cnt"] = tdc.ceil_div(dst_r_lp_cnt, tp_info["tp_used_core_cnt"])
        tp_info["tp_lc_r_lp_cnt"] = dst_r_lp_cnt - tp_info["tp_nlc_r_lp_cnt"] * (tp_info.get("tp_used_core_cnt") - 1)
        tp_info["tp_core_step_in"] = tp_info["tp_nlc_r_lp_cnt"] * tp_info["tp_dst_r_lp_step_in"]
        tp_info["tp_core_step_out"] = tp_info["tp_nlc_r_lp_cnt"] * tp_info["tp_dst_r_lp_step_out"]
        tp_info["tp_nlc_r_left"] = 0
        tp_info["tp_lc_r_left"] = dst_r_left
        tp_info["tp_nlc_c_lp_cnt"] = dst_c_lp_cnt
        tp_info["tp_lc_c_lp_cnt"] = dst_c_lp_cnt
        tp_info["tp_nlc_c_left"] = dst_c_left
        tp_info["tp_lc_c_left"] = dst_c_left
        tp_info["tp_nlc_l_lp_cnt"] = dst_l_lp_cnt
        tp_info["tp_lc_l_lp_cnt"] = dst_l_lp_cnt
        tp_info["tp_nlc_l_left"] = dst_l_left
        tp_info["tp_lc_l_left"] = dst_l_left
    else:
        tp_info["tp_mc_pos"] = 2
        tp_info["tp_used_core_cnt"] = tdc.ceil_div(dst_c_lp_cnt, tdc.ceil_div(dst_c_lp_cnt, tdc.CORE_DIM_NUM))
        tp_info["tp_nlc_c_lp_cnt"] = tdc.ceil_div(dst_c_lp_cnt, tp_info["tp_used_core_cnt"])
        tp_info["tp_lc_c_lp_cnt"] = dst_c_lp_cnt - tp_info["tp_nlc_c_lp_cnt"] * (tp_info.get("tp_used_core_cnt") - 1)
        tp_info["tp_core_step_in"] = tp_info["tp_nlc_c_lp_cnt"] * tp_info["tp_dst_c_lp_step_in"]
        tp_info["tp_core_step_out"] = tp_info["tp_nlc_c_lp_cnt"] * tp_info["tp_dst_c_lp_step_out"]
        tp_info["tp_nlc_c_left"] = 0
        tp_info["tp_lc_c_left"] = dst_c_left
        tp_info["tp_nlc_r_lp_cnt"] = dst_r_lp_cnt
        tp_info["tp_lc_r_lp_cnt"] = dst_r_lp_cnt
        tp_info["tp_nlc_r_left"] = dst_r_left
        tp_info["tp_lc_r_left"] = dst_r_left
        tp_info["tp_nlc_l_lp_cnt"] = dst_l_lp_cnt
        tp_info["tp_lc_l_lp_cnt"] = dst_l_lp_cnt
        tp_info["tp_nlc_l_left"] = dst_l_left
        tp_info["tp_lc_l_left"] = dst_l_left


# 'pylint: disable=redefined-builtin,unbalanced-tuple-unpacking,too-many-branches
def _tiling_params_transpose(args):
    """
    calculate real tiling params for transpose transform and last axis of target format is not c
    """

    in_shape, out_shape, perm, block_elem_cnt, ub_size, in_dtype, tp_info = args

    # ub layout tiling parameters
    half_ub_size = ub_size // 2 // block_elem_cnt * block_elem_cnt
    tp_info["tp_vnc_col_size"] = half_ub_size // tdc.VNC_LINES // block_elem_cnt * block_elem_cnt
    tp_info["tp_ub_offset"] = half_ub_size

    # axises in 3 parts
    axis_dst_l_size = out_shape[0]
    axis_dst_c_size = tdc.get_shape_size(out_shape[1:-1])
    axis_dst_r_size = out_shape[-1]

    # dst axis right tiling parameters
    axis_c_gate = 2 * block_elem_cnt
    tmp_dst_r_lp_unit = tp_info["tp_vnc_col_size"] // axis_c_gate // block_elem_cnt * block_elem_cnt
    tp_info["tp_dst_r_lp_unit"] = tmp_dst_r_lp_unit if axis_dst_r_size > tmp_dst_r_lp_unit else axis_dst_r_size
    dst_r_lp_cnt = tdc.ceil_div(axis_dst_r_size, tp_info.get("tp_dst_r_lp_unit"))
    dst_r_left = axis_dst_r_size % tp_info.get("tp_dst_r_lp_unit")
    tp_info["tp_dst_r_step_in"] = tdc.get_shape_size(in_shape[perm[-1] + 1:])
    tp_info["tp_dst_r_step_out"] = 1
    tp_info["tp_dst_r_lp_step_in"] = tp_info.get("tp_dst_r_lp_unit") * tp_info.get("tp_dst_r_step_in")
    tp_info["tp_dst_r_lp_step_out"] = tp_info.get("tp_dst_r_lp_unit") * tp_info.get("tp_dst_r_step_out")

    # dst axis center tiling parameters
    tmp_dst_c_lp_unit = tp_info["tp_vnc_col_size"] // tp_info["tp_dst_r_lp_unit"]
    src_last_axis_align_block_size = tdc.ceil_fill(in_shape[-1], block_elem_cnt)
    if tmp_dst_c_lp_unit > src_last_axis_align_block_size:
        tmp_dst_c_lp_unit = tmp_dst_c_lp_unit // src_last_axis_align_block_size * src_last_axis_align_block_size
    else:
        # can only has one gap between source last axis
        tmp_dst_c_lp_unit = tmp_dst_c_lp_unit - 2 * block_elem_cnt

    tp_info["tp_dst_c_lp_unit"] = tmp_dst_c_lp_unit if axis_dst_c_size > tmp_dst_c_lp_unit else axis_dst_c_size
    dst_c_lp_cnt = tdc.ceil_div(axis_dst_c_size, tp_info["tp_dst_c_lp_unit"])
    dst_c_left = axis_dst_c_size % tp_info["tp_dst_c_lp_unit"]
    if len(perm[1:-1]) == 1:
        tp_info["tp_dst_c_step_in"] = tdc.get_shape_size(in_shape[perm[1] + 1:])
    else:
        tp_info["tp_dst_c_step_in"] = 0
    tp_info["tp_dst_c_step_out"] = tdc.get_shape_size(out_shape[-1:])
    tp_info["tp_dst_c_lp_step_in"] = tp_info.get("tp_dst_c_lp_unit") * tp_info.get("tp_dst_c_step_in")
    tp_info["tp_dst_c_lp_step_out"] = tp_info.get("tp_dst_c_lp_unit") * tp_info.get("tp_dst_c_step_out")

    # `count method: r_idx/dst_rsize%size*dst_asize`
    dst_c_shape = out_shape[1:-1]
    tmp_dst_c_shape = dst_c_shape[:]
    tmp_dst_c_shape.append(1)
    in_shape.append(1)
    for idx, val in enumerate(reversed(dst_c_shape)):
        tp_info["tp_c_in_idx_" + str(idx) + "_size"] = val
        tp_info["tp_c_in_idx_" + str(idx) + "_dst_rsize"] = tdc.get_shape_size(tmp_dst_c_shape[-1 * (idx + 1):])
        tp_info["tp_c_in_idx_" + str(idx) + "_src_asize"] = tdc.get_shape_size(in_shape[perm[-1 - (idx + 1)] + 1:])
    # suppose there are 3 axises
    pad_axis_cnt = FRAME_LEVEL - len(tmp_dst_c_shape)
    if pad_axis_cnt:
        for _, idx in enumerate(PAD_IDX_LIST[len(tmp_dst_c_shape):]):
            tp_info["tp_c_in_idx_" + str(idx) + "_size"] = 1
            tp_info["tp_c_in_idx_" + str(idx) + "_dst_rsize"] = 1
            tp_info["tp_c_in_idx_" + str(idx) + "_src_asize"] = 0

    # dst axis left tiling parameters
    tmp_dst_l_lp_unit = tdc.VNC_LINES
    tp_info["tp_dst_l_lp_unit"] = tmp_dst_l_lp_unit if axis_dst_l_size > tmp_dst_l_lp_unit else axis_dst_l_size
    dst_l_lp_cnt = tdc.ceil_div(axis_dst_l_size, tp_info["tp_dst_l_lp_unit"])
    dst_l_left = axis_dst_l_size % tp_info["tp_dst_l_lp_unit"]
    tp_info["tp_dst_l_step_in"] = tdc.get_shape_size(in_shape[perm[0] + 1:])
    tp_info["tp_dst_l_step_out"] = tdc.get_shape_size(out_shape[1:])
    tp_info["tp_dst_l_lp_step_in"] = tp_info.get("tp_dst_l_lp_unit") * tp_info.get("tp_dst_l_step_in")
    tp_info["tp_dst_l_lp_step_out"] = tp_info.get("tp_dst_l_lp_unit") * tp_info.get("tp_dst_l_step_out")

    # mulitple core parameters
    axis_lp_args = (dst_r_lp_cnt, dst_r_left, dst_c_lp_cnt, dst_c_left, dst_l_lp_cnt, dst_l_left)
    _get_mc_info_transpose(axis_lp_args, tp_info)


def _get_tiling_params_func(args):
    """
    get tiling parameters function
    """

    in_shape, out_shape, perm, block_elem_cnt, ub_size, in_dtype, tp_info = args

    (in_shape_new, out_shape_new, perm_new) = _renew_input_output_perm(in_shape, out_shape, perm)
    args_get_tp = in_shape_new, out_shape_new, perm_new, block_elem_cnt, ub_size, in_dtype, tp_info
    _tiling_params_transpose(args_get_tp)


def _twice_vnchwconv_invert(args):
    """
    do nch to nhc transform by twice vnchwconv
    """

    (tik_inst, src_ub, ub_offset, vnc_col_size, r_pln_size, c_plp_size,
     left_dims_size, mid_lp_cnt, right_dims_size, c_in_idx_0_size, ele_per_block) = args
    left_block_align = tdc.ceil_fill(left_dims_size, ele_per_block)
    c_0_block_align = tdc.ceil_fill(c_in_idx_0_size, ele_per_block)
    right_block_align = tdc.ceil_fill(right_dims_size, ele_per_block)
    actual_c_size = (left_block_align + mid_lp_cnt * c_0_block_align + right_block_align)

    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar(dtype="int32")
        dst_stride = tik_inst.Scalar(dtype="int32")
        c_0_size = tik_inst.Scalar(name="c_0_size")
        with tik_inst.if_scope(c_0_size > tdc.STRIDE_LIMIT_MTE):
            c_0_size.set_as(1)
        with tik_inst.else_scope():
            c_0_size.set_as(c_in_idx_0_size)

        # do nch -> chn
        repeat_cnt = vnc_col_size // tdc.NI_16
        src_addr_list = [src_ub[vnc_col_size * i] for i in tdc.ADDR_IDX_LIST]
        dst_addr_list = [src_ub[ub_offset + tdc.NI_16 * i] for i in tdc.ADDR_IDX_LIST]
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(1)
            dst_stride.set_as(16)
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

        # reduce gap in c_plp_size
        with tik_inst.if_scope(tik.all(left_dims_size > 0, mid_lp_cnt * c_0_block_align + right_dims_size > 0)):
            with tik_inst.for_range(0, r_pln_size) as r_idx:
                tik_inst.data_move(src_ub[ub_offset + (left_dims_size + r_idx * actual_c_size) * ele_per_block],
                                   src_ub[ub_offset + (left_block_align + r_idx * actual_c_size) * ele_per_block],
                                   0, 1, (mid_lp_cnt * c_0_block_align + right_dims_size), 0, 0)
        with tik_inst.if_scope(mid_lp_cnt > 1):
            with tik_inst.for_range(0, r_pln_size) as r_idx:
                tik_inst.data_move(src_ub[ub_offset + (left_dims_size + c_in_idx_0_size +
                                                       r_idx * actual_c_size) * ele_per_block],
                                   src_ub[ub_offset + (left_dims_size + c_0_block_align +
                                                       r_idx * actual_c_size) * ele_per_block],
                                   0, mid_lp_cnt - 1, c_0_size, c_0_block_align - c_in_idx_0_size, 0)
        with tik_inst.if_scope(right_dims_size > 0):
            with tik_inst.for_range(0, r_pln_size) as r_idx:
                tik_inst.data_move(src_ub[ub_offset + (left_dims_size + mid_lp_cnt * c_in_idx_0_size +
                                          r_idx * actual_c_size) * ele_per_block],
                                   src_ub[ub_offset + (left_dims_size + mid_lp_cnt * c_0_block_align +
                                          r_idx * actual_c_size) * ele_per_block],
                                   0, 1, right_dims_size, 0, 0)
        # do chn -> hcn
        with tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, r_pln_size) as r_idx:
                tik_inst.data_move(src_ub[r_idx * ele_per_block],
                                   src_ub[ub_offset + r_idx * actual_c_size * ele_per_block],
                                   0, c_plp_size, 1, 0, r_pln_size - 1)

        # do c1thn -> nc1th
        repeat_cnt = tdc.ceil_div(r_pln_size * c_plp_size, tdc.NI_16)
        src_addr_list = [src_ub[tdc.NI_16 * i] for i in tdc.ADDR_IDX_LIST]
        dst_addr_list = [src_ub[ub_offset + vnc_col_size * i] for i in tdc.ADDR_IDX_LIST]
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(16)
            dst_stride.set_as(1)
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


def _update_input_offset_three_c_dims(args, c_beg):
    """
    count input gm offset for c-right has two dimensions
    """

    (l_lp_idx, dst_l_lp_step_in, dst_l_step_in, r_lp_idx, dst_r_lp_step_in, dst_r_step_in, r_backend, core_step_in,
     c_in_idx_0_size, c_in_idx_0_dst_rsize, c_in_idx_0_src_asize, c_in_idx_1_size, c_in_idx_1_dst_rsize,
     c_in_idx_1_src_asize, c_in_idx_2_size, c_in_idx_2_dst_rsize, c_in_idx_2_src_asize) = args

    in_offset = (l_lp_idx * dst_l_lp_step_in +
                 r_lp_idx * dst_r_lp_step_in - r_backend * dst_r_step_in +
                 c_beg // c_in_idx_0_dst_rsize % c_in_idx_0_size * c_in_idx_0_src_asize +
                 c_beg // c_in_idx_1_dst_rsize % c_in_idx_1_size * c_in_idx_1_src_asize +
                 c_beg // c_in_idx_2_dst_rsize % c_in_idx_2_size * c_in_idx_2_src_asize + core_step_in)

    return in_offset


def _update_output_offset(args):
    """
    count output gm offset
    """

    (l_lp_idx, dst_l_lp_step_out, c_lp_idx, dst_c_lp_step_out, c_backend, dst_c_step_out,
     r_lp_idx, dst_r_lp_step_out, r_backend, dst_r_step_out, core_step_out) = args

    out_offset = (l_lp_idx * dst_l_lp_step_out +
                  r_lp_idx * dst_r_lp_step_out - r_backend * dst_r_step_out +
                  c_lp_idx * dst_c_lp_step_out - c_backend * dst_c_step_out + core_step_out)

    return out_offset


def _split_center(args):
    """
    split c-right dimensions into three parts when it has three dimensions
    """

    tik_inst, c_in_idx_0_size, c_beg, left_dims_size, mid_lp_cnt, right_dims_size, c_plp_size = args
    next_r_gap = c_in_idx_0_size - c_beg % c_in_idx_0_size
    with tik_inst.if_scope(next_r_gap == c_in_idx_0_size):
        left_dims_size.set_as(0)
    with tik_inst.else_scope():
        with tik_inst.if_scope(next_r_gap <= c_plp_size):
            left_dims_size.set_as(next_r_gap)
        with tik_inst.else_scope():
            left_dims_size.set_as(c_plp_size)
    mid_lp_cnt.set_as((c_plp_size - left_dims_size) // c_in_idx_0_size)
    right_dims_size.set_as(c_plp_size - left_dims_size - mid_lp_cnt * c_in_idx_0_size)


def _move_dst_c_in_for_three_dims(data_in_args, offset_args, c_beg):
    """
    move target center in first in process when target center has three dimensions
    """

    (tik_inst, src_in_gm, src_ub, c_in_idx_0_size, left_dims_size, mid_lp_cnt, right_dims_size, dst_l_step_in,
     l_plp_size, c_plp_size, r_pln_size, dst_r_step_in, base_in_offset, ele_per_block, vnc_col_size) = data_in_args
    left_dims_block_align_size = tdc.ceil_fill(left_dims_size, ele_per_block)
    c_0_block_align_size = tdc.ceil_fill(c_in_idx_0_size, ele_per_block)
    right_dims_block_align_size = tdc.ceil_fill(right_dims_size, ele_per_block)
    r_ub_offset = (left_dims_block_align_size + c_0_block_align_size * mid_lp_cnt + right_dims_block_align_size)

    with tik_inst.if_scope(left_dims_size > 0):
        with tik_inst.for_range(0, l_plp_size) as l_idx:
            with tik_inst.for_range(0, r_pln_size) as r_idx:
                left_dims_gm_offset = (l_idx * dst_l_step_in + r_idx * dst_r_step_in + base_in_offset)
                left_dims_ub_offset = (l_idx * vnc_col_size + r_idx * r_ub_offset)
                tik_inst.data_move(src_ub[left_dims_ub_offset], src_in_gm[left_dims_gm_offset],
                                   0, 1, left_dims_block_align_size // ele_per_block, 0, 0)

    with tik_inst.if_scope(mid_lp_cnt > 0):
        # to avoid compile error
        c_0_size = tik_inst.Scalar(name="c_0_size")
        with tik_inst.if_scope(c_0_size > tdc.STRIDE_LIMIT_MTE):
            c_0_size.set_as(ele_per_block)
        with tik_inst.else_scope():
            c_0_size.set_as(c_0_block_align_size)

        with tik_inst.for_range(0, l_plp_size) as l_idx:
            with tik_inst.for_range(0, r_pln_size) as r_idx:
                with tik_inst.for_range(0, mid_lp_cnt) as mid_idx:
                    mid_c_beg = c_beg + left_dims_size + mid_idx * c_in_idx_0_size
                    mid_lp_gm_offset = (l_idx * dst_l_step_in + r_idx * dst_r_step_in +
                                        _update_input_offset_three_c_dims(offset_args, mid_c_beg))
                    mid_lp_ub_offset = (l_idx * vnc_col_size + r_idx * r_ub_offset + left_dims_block_align_size)
                    tik_inst.data_move(src_ub[mid_lp_ub_offset + mid_idx * c_0_block_align_size],
                                       src_in_gm[mid_lp_gm_offset],
                                       0, 1, c_0_size // ele_per_block, 0, 0)

    right_c_beg = c_beg + left_dims_size + mid_lp_cnt * c_in_idx_0_size
    right_base_in_offset = _update_input_offset_three_c_dims(offset_args, right_c_beg)
    with tik_inst.if_scope(right_dims_size > 0):
        with tik_inst.for_range(0, l_plp_size) as l_idx:
            with tik_inst.for_range(0, r_pln_size) as r_idx:
                right_dims_gm_offset = (l_idx * dst_l_step_in + r_idx * dst_r_step_in + right_base_in_offset)
                right_dims_ub_offset = (l_idx * vnc_col_size + r_idx * r_ub_offset +
                                        left_dims_block_align_size + mid_lp_cnt * c_0_block_align_size)
                tik_inst.data_move(src_ub[right_dims_ub_offset], src_in_gm[right_dims_gm_offset],
                                   0, 1, right_dims_block_align_size // ele_per_block, 0, 0)


# 'pylint: disable=unused-variable
def _copy_data_in(in_offset_args, tik_args):
    """
    copy data from gm to ub for transpose
    """

    (r_lp_idx, dst_r_lp_step_in, c_lp_idx, dst_c_lp_step_in, l_lp_idx, dst_l_lp_step_in, core_step_in,
     c_in_idx_0_size, c_in_idx_0_dst_rsize, c_in_idx_0_src_asize, c_in_idx_1_size, c_in_idx_1_dst_rsize,
     c_in_idx_1_src_asize, c_in_idx_2_size, c_in_idx_2_dst_rsize, c_in_idx_2_src_asize) = in_offset_args
    (tik_inst, src_in_gm, src_ub, dst_r_step_in, r_pln_size, dst_r_lp_unit, r_backend, dst_c_step_in, c_plp_size,
     c_backend, dst_l_step_in, l_plp_size, nlc_c_lp_cnt, dst_c_lp_unit, block_idx, mc_pos, ele_per_block,
     left_dims_size, mid_lp_cnt, right_dims_size, vnc_col_size) = tik_args

    with tik_inst.new_stmt_scope():
        c_beg = tik_inst.Scalar(name="r_beg")
        with tik_inst.if_scope(mc_pos == 2):
            c_beg.set_as(c_lp_idx * dst_c_lp_unit + block_idx * nlc_c_lp_cnt * dst_c_lp_unit - c_backend)
        with tik_inst.else_scope():
            c_beg.set_as(c_lp_idx * dst_c_lp_unit - c_backend)

        offset_args = (l_lp_idx, dst_l_lp_step_in, dst_l_step_in, r_lp_idx, dst_r_lp_step_in,
                       dst_r_step_in, r_backend, core_step_in, c_in_idx_0_size, c_in_idx_0_dst_rsize,
                       c_in_idx_0_src_asize, c_in_idx_1_size, c_in_idx_1_dst_rsize, c_in_idx_1_src_asize,
                       c_in_idx_2_size, c_in_idx_2_dst_rsize, c_in_idx_2_src_asize)
        base_in_offset = _update_input_offset_three_c_dims(offset_args, c_beg)

        # split target center into three parts
        split_args = (tik_inst, c_in_idx_0_size, c_beg, left_dims_size, mid_lp_cnt, right_dims_size, c_plp_size)
        _split_center(split_args)

        # move data in
        data_in_args = (tik_inst, src_in_gm, src_ub, c_in_idx_0_size, left_dims_size, mid_lp_cnt, right_dims_size,
                        dst_l_step_in, l_plp_size, c_plp_size, r_pln_size, dst_r_step_in, base_in_offset,
                        ele_per_block, vnc_col_size)
        with tik_inst.new_stmt_scope(disable_sync=True):
            _move_dst_c_in_for_three_dims(data_in_args, offset_args, c_beg)


# 'pylint: disable=unused-variable
def _copy_data_out(copy_out_args):
    """
    copy data from ub to gm
    """

    (tik_inst, dst_out_gm, src_ub, dst_l_step_out, vnc_col_size,
     l_plp_size, r_pln_size, c_plp_size, ele_per_block) = copy_out_args

    with tik_inst.new_stmt_scope():
        tmp_reg = [tik_inst.Scalar(dtype=src_ub.dtype) for i in tdc.REG_IDX_LIST[:ele_per_block]]
        with tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, l_plp_size) as l_idx:
                tmp_ub_offset = l_idx * vnc_col_size
                tmp_gm_offset = l_idx * dst_l_step_out
                tik_inst.data_move(dst_out_gm[tmp_gm_offset], src_ub[tmp_ub_offset],
                                   0, 1, c_plp_size * r_pln_size // ele_per_block, 0, 0)
        with tik_inst.if_scope((c_plp_size * r_pln_size) % ele_per_block > 0):
            with tik_inst.for_range(0, l_plp_size) as l_idx:
                tmp_ub_offset = l_idx * vnc_col_size + c_plp_size * r_pln_size - ele_per_block
                tmp_gm_offset = l_idx * dst_l_step_out + c_plp_size * r_pln_size - ele_per_block
                for i in tdc.REG_IDX_LIST[:ele_per_block]:
                    tmp_reg[i].set_as(src_ub[tmp_ub_offset + i])
                for i in tdc.REG_IDX_LIST[:ele_per_block]:
                    src_ub[i:].set_as(tmp_reg[i])
                tik_inst.data_move(dst_out_gm[tmp_gm_offset], src_ub, 0, 1, 1, 0, 0)


def _func_transform(tensor_args, tp_args):
    """
    transform function for transpose
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, ele_per_block, in_dtype = tensor_args
    mc_pos = tp_args.get("tp_mc_pos")
    vnc_col_size = tp_args.get("tp_vnc_col_size")
    ub_offset = tp_args.get("tp_ub_offset")
    used_core_cnt = tp_args.get("tp_used_core_cnt")
    core_step_in = tp_args.get("tp_core_step_in")
    core_step_out = tp_args.get("tp_core_step_out")
    nlc_l_lp_cnt = tp_args.get("tp_nlc_l_lp_cnt")
    lc_l_lp_cnt = tp_args.get("tp_lc_l_lp_cnt")
    nlc_l_left = tp_args.get("tp_nlc_l_left")
    lc_l_left = tp_args.get("tp_lc_l_left")
    dst_l_lp_unit = tp_args.get("tp_dst_l_lp_unit")
    dst_l_step_in = tp_args.get("tp_dst_l_step_in")
    dst_l_step_out = tp_args.get("tp_dst_l_step_out")
    dst_l_lp_step_in = tp_args.get("tp_dst_l_lp_step_in")
    dst_l_lp_step_out = tp_args.get("tp_dst_l_lp_step_out")
    nlc_r_lp_cnt = tp_args.get("tp_nlc_r_lp_cnt")
    lc_r_lp_cnt = tp_args.get("tp_lc_r_lp_cnt")
    nlc_r_left = tp_args.get("tp_nlc_r_left")
    lc_r_left = tp_args.get("tp_lc_r_left")
    dst_r_lp_unit = tp_args.get("tp_dst_r_lp_unit")
    dst_r_step_in = tp_args.get("tp_dst_r_step_in")
    dst_r_step_out = tp_args.get("tp_dst_r_step_out")
    dst_r_lp_step_in = tp_args.get("tp_dst_r_lp_step_in")
    dst_r_lp_step_out = tp_args.get("tp_dst_r_lp_step_out")
    nlc_c_lp_cnt = tp_args.get("tp_nlc_c_lp_cnt")
    lc_c_lp_cnt = tp_args.get("tp_lc_c_lp_cnt")
    nlc_c_left = tp_args.get("tp_nlc_c_left")
    lc_c_left = tp_args.get("tp_lc_c_left")
    dst_c_lp_unit = tp_args.get("tp_dst_c_lp_unit")
    dst_c_step_in = tp_args.get("tp_dst_c_step_in")
    dst_c_step_out = tp_args.get("tp_dst_c_step_out")
    dst_c_lp_step_in = tp_args.get("tp_dst_c_lp_step_in")
    dst_c_lp_step_out = tp_args.get("tp_dst_c_lp_step_out")
    c_in_idx_0_size = tp_args.get("tp_c_in_idx_0_size")
    c_in_idx_0_dst_rsize = tp_args.get("tp_c_in_idx_0_dst_rsize")
    c_in_idx_0_src_asize = tp_args.get("tp_c_in_idx_0_src_asize")
    c_in_idx_1_size = tp_args.get("tp_c_in_idx_1_size")
    c_in_idx_1_dst_rsize = tp_args.get("tp_c_in_idx_1_dst_rsize")
    c_in_idx_1_src_asize = tp_args.get("tp_c_in_idx_1_src_asize")
    c_in_idx_2_size = tp_args.get("tp_c_in_idx_2_size")
    c_in_idx_2_dst_rsize = tp_args.get("tp_c_in_idx_2_dst_rsize")
    c_in_idx_2_src_asize = tp_args.get("tp_c_in_idx_2_src_asize")

    def _inner_func(tiling_args):
        r_lp_cnt, r_left, c_lp_cnt, c_left, l_lp_cnt, l_left = tiling_args
        r_pln_size = tik_inst.Scalar(name="r_pln_size")
        c_plp_size = tik_inst.Scalar(name="c_plp_size")
        l_plp_size = tik_inst.Scalar(name="l_plp_size")
        is_r_back = tik_inst.Scalar(name="is_r_back")
        is_c_back = tik_inst.Scalar(name="is_c_back")
        left_dims_size = tik_inst.Scalar(name="left_dims_size")
        mid_lp_cnt = tik_inst.Scalar(name="mid_lp_cnt")
        right_dims_size = tik_inst.Scalar(name="right_dims_size")

        with tik_inst.for_range(0, r_lp_cnt) as r_lp_idx:  # axis dst last
            with tik_inst.if_scope(tik.any(r_lp_idx != r_lp_cnt - 1, r_left == 0)):
                # r_lp_idx for last second core and lc_r_left for last core
                with tik_inst.if_scope(tik.any(tik.all(block_idx == used_core_cnt - 1, r_lp_idx == r_lp_cnt - 2,
                                                       mc_pos == 1, lc_r_left > 0,
                                                       lc_r_left < ele_per_block),
                                               tik.all(block_idx == used_core_cnt - 2, r_lp_idx == r_lp_cnt - 1,
                                                       lc_r_lp_cnt == 1, mc_pos == 1, lc_r_left > 0,
                                                       lc_r_left < ele_per_block),
                                               tik.all(r_lp_idx == r_lp_cnt - 2,
                                                       mc_pos != 1, lc_r_left > 0, lc_r_left < ele_per_block))):
                    r_pln_size.set_as(dst_r_lp_unit - ele_per_block)
                with tik_inst.else_scope():
                    r_pln_size.set_as(dst_r_lp_unit)
                is_r_back.set_as(0)
            with tik_inst.else_scope():
                with tik_inst.if_scope(tik.all(used_core_cnt > 1, r_left > 0, r_left < ele_per_block)):
                    r_pln_size.set_as(r_left + ele_per_block)
                    is_r_back.set_as(1)
                with tik_inst.else_scope():
                    r_pln_size.set_as(r_left)
                    is_r_back.set_as(0)
            r_backend = is_r_back * ele_per_block

            with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:  # axis dst center
                with tik_inst.if_scope(tik.any(c_lp_idx != c_lp_cnt - 1, c_left == 0)):
                    with tik_inst.if_scope(tik.any(tik.all(block_idx == used_core_cnt - 1, c_lp_idx == c_lp_cnt - 2,
                                                           mc_pos == 2, lc_c_left > 0,
                                                           lc_c_left * r_pln_size < ele_per_block),
                                                   tik.all(block_idx == used_core_cnt - 2, c_lp_idx == c_lp_cnt - 1,
                                                           lc_c_lp_cnt == 1, mc_pos == 2, lc_c_left > 0,
                                                           lc_c_left * r_pln_size < ele_per_block),
                                                   tik.all(c_lp_idx == c_lp_cnt - 2,
                                                           mc_pos != 2, lc_c_left > 0,
                                                           lc_c_left * r_pln_size < ele_per_block))):
                        c_plp_size.set_as(dst_c_lp_unit - ele_per_block)
                    with tik_inst.else_scope():
                        c_plp_size.set_as(dst_c_lp_unit)
                    is_c_back.set_as(0)
                with tik_inst.else_scope():
                    with tik_inst.if_scope(tik.all(used_core_cnt > 1, lc_c_left > 0,
                                                   lc_c_left * r_pln_size < ele_per_block)):
                        c_plp_size.set_as(lc_c_left + ele_per_block)
                        is_c_back.set_as(1)
                    with tik_inst.else_scope():
                        c_plp_size.set_as(lc_c_left)
                        is_c_back.set_as(0)
                c_backend = is_c_back * ele_per_block

                with tik_inst.for_range(0, l_lp_cnt) as l_lp_idx:  # axis dst first
                    with tik_inst.if_scope(tik.any(l_lp_idx != l_lp_cnt - 1, l_left == 0)):
                        l_plp_size.set_as(dst_l_lp_unit)
                    with tik_inst.else_scope():
                        l_plp_size.set_as(l_left)

                    in_offset_args = (r_lp_idx, dst_r_lp_step_in, c_lp_idx, dst_c_lp_step_in, l_lp_idx,
                                      dst_l_lp_step_in, block_idx * core_step_in,
                                      c_in_idx_0_size, c_in_idx_0_dst_rsize, c_in_idx_0_src_asize,
                                      c_in_idx_1_size, c_in_idx_1_dst_rsize, c_in_idx_1_src_asize,
                                      c_in_idx_2_size, c_in_idx_2_dst_rsize, c_in_idx_2_src_asize)
                    copy_in_args = (tik_inst, src_in_gm, src_ub, dst_r_step_in, r_pln_size, dst_r_lp_unit, r_backend,
                                    dst_c_step_in, c_plp_size, c_backend, dst_l_step_in, l_plp_size, nlc_c_lp_cnt,
                                    dst_c_lp_unit, block_idx, mc_pos, ele_per_block, left_dims_size, mid_lp_cnt,
                                    right_dims_size, vnc_col_size)
                    _copy_data_in(in_offset_args, copy_in_args)

                    reorder_args = (tik_inst, src_ub, ub_offset, vnc_col_size, r_pln_size, c_plp_size,
                                    left_dims_size, mid_lp_cnt, right_dims_size, c_in_idx_0_size, ele_per_block)
                    _twice_vnchwconv_invert(reorder_args)
                    out_gm_args = (l_lp_idx, dst_l_lp_step_out, c_lp_idx, dst_c_lp_step_out, c_backend, dst_c_step_out,
                                   r_lp_idx, dst_r_lp_step_out, r_backend, dst_r_step_out, block_idx * core_step_out)
                    out_gm_offset = _update_output_offset(out_gm_args)
                    copy_out_args = (tik_inst, dst_out_gm[out_gm_offset], src_ub[ub_offset], dst_l_step_out,
                                     vnc_col_size, l_plp_size, r_pln_size, c_plp_size, ele_per_block)
                    _copy_data_out(copy_out_args)

    with tik_inst.if_scope(block_idx != used_core_cnt - 1):
        nlc_args = (nlc_r_lp_cnt, nlc_r_left, nlc_c_lp_cnt, nlc_c_left, nlc_l_lp_cnt, nlc_l_left)
        _inner_func(nlc_args)
    with tik_inst.else_scope():
        lc_args = (lc_r_lp_cnt, lc_r_left, lc_c_lp_cnt, lc_c_left, lc_l_lp_cnt, lc_l_left)
        _inner_func(lc_args)


def transpose_d_014253_big(input_x, output_y, perm, kernel_name="transpose_d_014253_big"):
    """
    axis transpose for perm is (0, 1, 4, 2, 5, 3)

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        dtype of output should be same type as input
    perm: list or tuple
        same length with input_x shape
    kernel_name : str
        kernel name, default value is "transpose_d_014253_big"

    Returns
    -------
    None
    """

    in_shape = list(input_x.get("shape"))
    out_shape = list(output_y.get("shape"))
    in_dtype = input_x.get("dtype").lower()
    ub_size = tdc.get_max_element_in_ub(in_dtype, 1)
    block_elem_cnt = tdc.BLOCK_BYTE_SIZE // tdc.get_dtype_len(in_dtype)
    tp_info = {}

    # get tiling parameters
    args = in_shape, out_shape, perm, block_elem_cnt, ub_size, in_dtype, tp_info
    _get_tiling_params_func(args)

    tik_inst = tik.Tik()
    src_in_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tbe_platform.scope_gm, "src_in_gm")
    dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tbe_platform.scope_gm, "dst_out_gm")
    src_ub = tik_inst.Tensor(in_dtype, (ub_size,), tbe_platform.scope_ubuf, "total_ub")

    used_core_cnt = tp_info.get("tp_used_core_cnt")
    with tik_inst.for_range(0, tdc.CORE_DIM_NUM, block_num=tdc.CORE_DIM_NUM) as block_idx:
        with tik_inst.if_scope(block_idx < used_core_cnt):
            tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, block_elem_cnt, in_dtype]
            _func_transform(tensor_args, tp_info)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[src_in_gm], outputs=[dst_out_gm])
