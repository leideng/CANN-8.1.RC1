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
trans_data_rnn_negative_target
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl import trans_data_common_func as tdc
from impl.util.platform_adapter import error_manager_vector

# used for scalar
PAD_IDX_LIST = (0, 1)
# frame up levels
FRAME_LEVEL = 2
INT8_DTYPES = ("int8", "uint8")
NEED_CAST_DTYPES = ("float32", "int32")
VNC_SUPPORT_DTYPES = ("int8", "uint8", "float16")


# 'pylint: disable=too-many-locals, inconsistent-return-statements, too-many-lines, too-many-return-statements,
# 'pylint: disable=too-many-statements
def _renew_input_output_shape_format(in_shape, out_shape):
    """
    renew shape and format to adapt tiling process
    """
    in_format_new = "HCNT"
    out_format_new = "HCN"
    axis_h, axis_c, axis_n = 1, out_shape[0], out_shape[1]
    axis_c0 = in_shape[3]
    axis_ni = in_shape[2]
    axis_no = in_shape[1]
    axis_c1 = in_shape[0]
    in_shape_new = [axis_h] + [axis_c1] + [axis_no * axis_ni] + [axis_c0]
    out_shape_new = [axis_h] + [axis_c] + [axis_n]
    new_params_nd = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

    return new_params_nd


# 'pylint: disable=too-many-statements, unused-variable
def _get_mc_info_negative(cr_args, c_args, cl_args):
    """
    get multiple core axis position for negative transform
    """

    (dst_cr_lp_cnt, dst_cr_left, dst_cr_lp_unit, tp_200_dst_cr_lp_step_in, tp_200_dst_cr_lp_step_out,
     tp_200_dst_cr_dims) = cr_args
    src_c_lp_cnt, src_c_left, tp_200_src_c_lp_step_in, tp_200_src_c_lp_step_out = c_args
    dst_cl_lp_cnt, dst_cl_left, tp_200_dst_cl_lp_step_in, tp_200_dst_cl_lp_step_out, tp_200_dst_cr_dims = cl_args

    tmp_full_lp_cnt_cr = tdc.get_core_num() if tdc.floor_div(dst_cr_lp_cnt, tdc.get_core_num()) > 0 else 0
    reminder_lp_cnt_cr = dst_cr_lp_cnt % tdc.get_core_num()
    if reminder_lp_cnt_cr == 0 and dst_cr_left > dst_cr_lp_unit // 2:
        tmp_full_lp_cnt_cr += tdc.get_core_num()
    full_lp_cnt_cr = tmp_full_lp_cnt_cr + reminder_lp_cnt_cr

    tmp_full_lp_cnt_c = tdc.get_core_num() if tdc.floor_div(src_c_lp_cnt, tdc.get_core_num()) > 0 else 0
    reminder_lp_cnt_c = src_c_lp_cnt % tdc.get_core_num()
    if reminder_lp_cnt_c == 0:
        tmp_full_lp_cnt_c += tdc.get_core_num()
    full_lp_cnt_c = tmp_full_lp_cnt_c + reminder_lp_cnt_c

    tmp_full_lp_cnt_cl = tdc.get_core_num() if tdc.floor_div(dst_cl_lp_cnt, tdc.get_core_num()) > 0 else 0
    reminder_lp_cnt_cl = dst_cl_lp_cnt % tdc.get_core_num()
    if reminder_lp_cnt_cl == 0:
        tmp_full_lp_cnt_cl += tdc.get_core_num()
    full_lp_cnt_cl = tmp_full_lp_cnt_cl + reminder_lp_cnt_cl

    lp_cnt_list = (full_lp_cnt_cl, full_lp_cnt_c, full_lp_cnt_cr)
    if lp_cnt_list.index(max(lp_cnt_list)) == 0:
        tp_200_mc_pos = 0
        tp_200_is_mc_cl = 1
        tp_200_is_mc_cr = 0
        tp_200_used_core_cnt = tdc.ceil_div(dst_cl_lp_cnt, tdc.ceil_div(dst_cl_lp_cnt, tdc.get_core_num()))
        tp_200_nlc_cl_lp_cnt = tdc.ceil_div(dst_cl_lp_cnt, tp_200_used_core_cnt)
        tp_200_lc_cl_lp_cnt = dst_cl_lp_cnt - tp_200_nlc_cl_lp_cnt * (tp_200_used_core_cnt - 1)
        tp_200_core_step_in = tp_200_nlc_cl_lp_cnt * tp_200_dst_cl_lp_step_in
        tp_200_core_step_out = tp_200_nlc_cl_lp_cnt * tp_200_dst_cl_lp_step_out
        tp_200_nlc_cl_left = 0
        tp_200_lc_cl_left = dst_cl_left
        tp_200_nlc_c_lp_cnt = src_c_lp_cnt
        tp_200_lc_c_lp_cnt = src_c_lp_cnt
        tp_200_nlc_c_left = src_c_left
        tp_200_lc_c_left = src_c_left
        tp_200_nlc_cr_lp_cnt = dst_cr_lp_cnt
        tp_200_lc_cr_lp_cnt = dst_cr_lp_cnt
        tp_200_nlc_cr_left = dst_cr_left
        tp_200_lc_cr_left = dst_cr_left
    elif lp_cnt_list.index(max(lp_cnt_list)) == 1:
        tp_200_mc_pos = 1
        tp_200_is_mc_cl = 0
        tp_200_is_mc_cr = 0
        tp_200_used_core_cnt = tdc.ceil_div(src_c_lp_cnt, tdc.ceil_div(src_c_lp_cnt, tdc.get_core_num()))
        tp_200_nlc_c_lp_cnt = tdc.ceil_div(src_c_lp_cnt, tp_200_used_core_cnt)
        tp_200_lc_c_lp_cnt = src_c_lp_cnt - (tp_200_used_core_cnt - 1) * tp_200_nlc_c_lp_cnt
        tp_200_nlc_c_left = 0
        tp_200_lc_c_left = src_c_left
        tp_200_core_step_in = tp_200_nlc_c_lp_cnt * tp_200_src_c_lp_step_in
        tp_200_core_step_out = tp_200_nlc_c_lp_cnt * tp_200_src_c_lp_step_out
        tp_200_nlc_cr_lp_cnt = dst_cr_lp_cnt
        tp_200_lc_cr_lp_cnt = dst_cr_lp_cnt
        tp_200_nlc_cr_left = dst_cr_left
        tp_200_lc_cr_left = dst_cr_left
        tp_200_nlc_cl_lp_cnt = dst_cl_lp_cnt
        tp_200_lc_cl_lp_cnt = dst_cl_lp_cnt
        tp_200_nlc_cl_left = dst_cl_left
        tp_200_lc_cl_left = dst_cl_left
    else:
        tp_200_mc_pos = 2
        tp_200_is_mc_cl = 0
        tp_200_is_mc_cr = 1
        tp_200_used_core_cnt = tdc.ceil_div(dst_cr_lp_cnt, tdc.ceil_div(dst_cr_lp_cnt, tdc.get_core_num()))
        tp_200_nlc_cr_lp_cnt = tdc.ceil_div(dst_cr_lp_cnt, tp_200_used_core_cnt)
        tp_200_lc_cr_lp_cnt = dst_cr_lp_cnt - (tp_200_used_core_cnt - 1) * tp_200_nlc_cr_lp_cnt
        tp_200_nlc_cr_left = 0
        tp_200_lc_cr_left = dst_cr_left
        tp_200_core_step_in = tp_200_nlc_cr_lp_cnt * tp_200_dst_cr_lp_step_in
        tp_200_core_step_out = tp_200_nlc_cr_lp_cnt * tp_200_dst_cr_lp_step_out
        tp_200_nlc_c_lp_cnt = src_c_lp_cnt
        tp_200_lc_c_lp_cnt = src_c_lp_cnt
        tp_200_nlc_c_left = src_c_left
        tp_200_lc_c_left = src_c_left
        tp_200_nlc_cl_lp_cnt = dst_cl_lp_cnt
        tp_200_lc_cl_lp_cnt = dst_cl_lp_cnt
        tp_200_nlc_cl_left = dst_cl_left
        tp_200_lc_cl_left = dst_cl_left

    tiling_params = [tp_200_mc_pos, tp_200_is_mc_cr, tp_200_is_mc_cl] + \
                    [tp_200_used_core_cnt, tp_200_core_step_in, tp_200_core_step_out] + \
                    [tp_200_nlc_cl_lp_cnt, tp_200_nlc_c_lp_cnt, tp_200_nlc_cr_lp_cnt,
                     tp_200_nlc_cl_left, tp_200_nlc_c_left, tp_200_nlc_cr_left] + \
                    [tp_200_lc_cl_lp_cnt, tp_200_lc_c_lp_cnt, tp_200_lc_cr_lp_cnt,
                     tp_200_lc_cl_left, tp_200_lc_c_left, tp_200_lc_cr_left]

    return tiling_params


# 'pylint: disable=redefined-builtin, unbalanced-tuple-unpacking, too-many-branches
def _tiling_params_negative(args):
    """
    calculate real tiling params for negative transform and last axis of target format is not c
    """

    in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size, in_dtype, hidden_size = args
    c0_len = in_shape[-1]  # axis c0
    tp_200_c0_len = c0_len
    tp_names = locals()

    # ub layout tiling parameters
    if src_format[-2] == dst_format[-1]:  # such as NC1HWC0 -> NCHW
        tp_200_src_r2nd_dst_r1st_same = 1
    else:  # such as C1HWNoNiC0 -> NCHW
        tp_200_src_r2nd_dst_r1st_same = 0

    half_ub_size = ub_size // 2 // block_elem_cnt * block_elem_cnt
    vnc_col_size = half_ub_size // tdc.VNC_LINES // block_elem_cnt * block_elem_cnt
    tp_200_ub_offset = half_ub_size

    # dst axis C-RIGHT tiling parameters
    tp_200_dst_cr_dims = 2
    axis_dst_cr_size = tdc.get_shape_size(out_shape[dst_format.index("C") + 1:])
    axis_src_cr_size = tdc.get_shape_size(in_shape[src_format.index("C") + 1:]) // c0_len
    tp_200_hidden_cnt = axis_dst_cr_size // hidden_size
    cr_gate = 3 * c0_len  # select different tiling mode
    # once vnchwconv flow
    if axis_src_cr_size >= cr_gate:
        tmp_dst_cr_lp_unit = half_ub_size // c0_len // block_elem_cnt * block_elem_cnt  # block align
    else:  # twice vnchwconv flow
        if in_dtype in INT8_DTYPES:
            tmp_dst_cr_lp_unit = vnc_col_size // 2 // c0_len // block_elem_cnt * block_elem_cnt
        else:
            tmp_dst_cr_lp_unit = vnc_col_size // c0_len // block_elem_cnt * block_elem_cnt
    tp_200_dst_cr_lp_unit = tmp_dst_cr_lp_unit if axis_src_cr_size > tmp_dst_cr_lp_unit else axis_src_cr_size
    dst_cr_lp_cnt = tdc.ceil_div(axis_src_cr_size, tp_200_dst_cr_lp_unit)
    dst_cr_left = axis_src_cr_size % tp_200_dst_cr_lp_unit
    tmp_dst_cr_format = dst_format[dst_format.index("C") + 1:]
    tmp_dst_cr_shape = out_shape[dst_format.index("C") + 1:]
    # count method: cr_idx/dst_rsize%size*dst_asize
    tmp_dst_cr_shape.append(1)
    for idx, char in enumerate(reversed(tmp_dst_cr_format)):
        src_chr_pos = src_format.index(char)
        tp_names["tp_200_cr_in_idx_" + str(idx) + "_size"] = out_shape[dst_format.index(char)]
        tp_names["tp_200_cr_in_idx_" + str(idx) + "_dst_rsize"] = tdc.get_shape_size(tmp_dst_cr_shape[-1 - idx:])
        tp_names["tp_200_cr_in_idx_" + str(idx) + "_src_asize"] = tdc.get_shape_size(in_shape[src_chr_pos + 1:])
    # suppose there are 2 axises
    pad_axis_cnt = FRAME_LEVEL - len(tmp_dst_cr_format)
    if pad_axis_cnt:
        tp_200_dst_cr_dims = 1
        for _, idx in enumerate(PAD_IDX_LIST[len(tmp_dst_cr_format):]):
            tp_names["tp_200_cr_in_idx_" + str(idx) + "_size"] = 1
            tp_names["tp_200_cr_in_idx_" + str(idx) + "_dst_rsize"] = 1
            tp_names["tp_200_cr_in_idx_" + str(idx) + "_src_asize"] = 0
    tp_200_dst_cr_step_out = 1
    tp_200_dst_cr_lp_step_out = tdc.get_shape_size([tp_200_dst_cr_lp_unit, tp_200_dst_cr_step_out])
    if tp_200_dst_cr_dims == 2:
        tp_200_dst_cr_step_in = 0
    else:
        dst_cr_chr = dst_format[-1]
        tp_200_dst_cr_step_in = tdc.get_shape_size(in_shape[src_format.index(dst_cr_chr) + 1:])
    tp_200_dst_cr_lp_step_in = tp_200_dst_cr_lp_unit * tp_200_dst_cr_step_in

    # axis C tiling parameters
    axis_src_c_size = in_shape[src_format.index("C")]
    axis_dst_c_size = out_shape[dst_format.index("C")]
    if dst_cr_lp_cnt > 1 or axis_src_c_size == 1:
        tp_200_src_c_lp_unit = 1
    else:
        tmp_src_c_lp_unit = tmp_dst_cr_lp_unit // tdc.ceil_fill(axis_src_cr_size, block_elem_cnt)
        tp_200_src_c_lp_unit = tmp_src_c_lp_unit if axis_src_c_size > tmp_src_c_lp_unit else axis_src_c_size
    src_c_lp_cnt = tdc.ceil_div(axis_src_c_size, tp_200_src_c_lp_unit)
    src_c_left = axis_src_c_size % tp_200_src_c_lp_unit
    tp_200_src_c_step_in = tdc.get_shape_size(in_shape[src_format.index("C") + 1:])
    tp_200_src_c_step_out = tdc.get_shape_size(out_shape[dst_format.index("C") + 1:])
    tp_200_src_c_lp_step_in = tp_200_src_c_lp_unit * tp_200_src_c_step_in
    tp_200_src_c_lp_step_out = tp_200_src_c_lp_unit * c0_len * tp_200_src_c_step_out
    tp_200_c_mod_c0 = axis_dst_c_size % c0_len
    tp_200_dst_c_size = axis_dst_c_size

    # dst axis C-LEFT tiling parameters
    tp_200_dst_cl_dims = 2
    axis_dst_cl_size = tdc.get_shape_size(out_shape[:dst_format.index("C")])
    src_c_dst_cr_size = axis_src_c_size * axis_src_cr_size
    if axis_src_cr_size >= cr_gate:
        tp_200_tiling_mode = 2001
        tmp_dst_cl_lp_unit = half_ub_size // (tp_200_src_c_lp_unit *
                                              tdc.ceil_fill(tp_200_dst_cr_lp_unit, block_elem_cnt) * c0_len)
        tp_200_dst_cl_lp_unit = tmp_dst_cl_lp_unit if axis_dst_cl_size > tmp_dst_cl_lp_unit else axis_dst_cl_size
    # c and c-right cannot move out one time or one vnc line cannot save c0_size c * c-right
    elif axis_src_c_size > tp_200_src_c_lp_unit or src_c_dst_cr_size > tmp_dst_cr_lp_unit:
        tp_200_tiling_mode = 2002
        tp_200_dst_cl_lp_unit = tdc.VNC_LINES if axis_dst_cl_size > tdc.VNC_LINES else axis_dst_cl_size
        tp_200_dst_cr_lp_step_out = tp_200_src_c_step_out
    else:
        tp_200_tiling_mode = 2003
        supposed_lp_unit = 4 * block_elem_cnt
        tmp_dst_cl_lp_unit = tmp_dst_cr_lp_unit // (tp_200_src_c_lp_unit * tp_200_dst_cr_lp_unit)
        tp_200_dst_cl_lp_unit = tmp_dst_cl_lp_unit if tmp_dst_cl_lp_unit < supposed_lp_unit else supposed_lp_unit
        tp_200_dst_cr_lp_step_out = tp_200_src_c_step_out
    dst_cl_lp_cnt = tdc.ceil_div(axis_dst_cl_size, tp_200_dst_cl_lp_unit)
    dst_cl_left = axis_dst_cl_size % tp_200_dst_cl_lp_unit
    tp_200_left_cl_c_cr_size = dst_cl_left * axis_dst_c_size * axis_src_cr_size  # for tiling mode 2003
    tmp_dst_cl_format = dst_format[:dst_format.index("C")]
    tmp_c_left_shape = out_shape[:dst_format.index("C")]
    tmp_c_left_shape.append(1)
    # count method: left_axis_size/dst_rsize%size*asize
    for idx, char in enumerate(reversed(tmp_dst_cl_format)):
        src_pos = src_format.index(char)
        tp_names["tp_200_cl_in_idx_" + str(idx) + "_size"] = out_shape[dst_format.index(char)]
        tp_names["tp_200_cl_in_idx_" + str(idx) + "_dst_rsize"] = tdc.get_shape_size(tmp_c_left_shape[-1 - idx:])
        tp_names["tp_200_cl_in_idx_" + str(idx) + "_src_asize"] = tdc.get_shape_size(in_shape[src_pos + 1:])
    # suppose there are 2 axises
    pad_axis_cnt = FRAME_LEVEL - len(tmp_dst_cl_format)
    if pad_axis_cnt:
        tp_200_dst_cl_dims = 1
        for _, idx in enumerate(PAD_IDX_LIST[len(tmp_dst_cl_format):]):
            tp_names["tp_200_cl_in_idx_" + str(idx) + "_size"] = 1
            tp_names["tp_200_cl_in_idx_" + str(idx) + "_dst_rsize"] = 1
            tp_names["tp_200_cl_in_idx_" + str(idx) + "_src_asize"] = 0

    tp_200_dst_cl_step_out = tdc.get_shape_size(out_shape[dst_format.index("C"):])
    tp_200_dst_cl_lp_step_out = tp_200_dst_cl_lp_unit * tp_200_dst_cl_step_out
    if tp_200_dst_cl_dims == 2:
        tp_200_dst_cl_step_in = 0
    else:
        dst_cl_chr = dst_format[0]
        tp_200_dst_cl_step_in = tdc.get_shape_size(in_shape[src_format.index(dst_cl_chr) + 1:])
    tp_200_dst_cl_lp_step_in = tp_200_dst_cl_lp_unit * tp_200_dst_cl_step_in

    # mulitple core parameters
    cr_args = (dst_cr_lp_cnt, dst_cr_left, tp_200_dst_cr_lp_unit, tp_200_dst_cr_lp_step_in, tp_200_dst_cr_lp_step_out,
               tp_200_dst_cr_dims)
    c_args = src_c_lp_cnt, src_c_left, tp_200_src_c_lp_step_in, tp_200_src_c_lp_step_out
    cl_args = dst_cl_lp_cnt, dst_cl_left, tp_200_dst_cl_lp_step_in, tp_200_dst_cl_lp_step_out, tp_200_dst_cr_dims
    (tp_200_mc_pos, tp_200_is_mc_cr, tp_200_is_mc_cl, tp_200_used_core_cnt, tp_200_core_step_in, tp_200_core_step_out,
     tp_200_nlc_cl_lp_cnt, tp_200_nlc_c_lp_cnt, tp_200_nlc_cr_lp_cnt, tp_200_nlc_cl_left, tp_200_nlc_c_left,
     tp_200_nlc_cr_left, tp_200_lc_cl_lp_cnt, tp_200_lc_c_lp_cnt, tp_200_lc_cr_lp_cnt, tp_200_lc_cl_left,
     tp_200_lc_c_left, tp_200_lc_cr_left) = _get_mc_info_negative(cr_args, c_args, cl_args)

    sub_tiling_params = [
        tp_200_tiling_mode, tp_200_ub_offset, tp_200_mc_pos, tp_200_used_core_cnt, tp_200_hidden_cnt, tp_200_c0_len,
        tp_200_core_step_in, tp_200_core_step_out, tp_200_nlc_cr_lp_cnt, tp_200_nlc_c_lp_cnt, tp_200_nlc_cl_lp_cnt,
        tp_200_nlc_cr_left, tp_200_nlc_c_left, tp_200_nlc_cl_left, tp_200_lc_cr_lp_cnt, tp_200_lc_c_lp_cnt,
        tp_200_lc_cl_lp_cnt, tp_200_lc_cr_left, tp_200_lc_c_left, tp_200_lc_cl_left, tp_200_dst_cr_lp_unit,
        tp_200_src_c_lp_unit, tp_200_dst_cl_lp_unit, tp_200_dst_cr_step_in, tp_200_dst_cr_step_out,
        tp_200_dst_cr_lp_step_in, tp_200_dst_cr_lp_step_out, tp_200_dst_c_size, tp_200_src_c_step_in,
        tp_200_src_c_step_out, tp_200_src_c_lp_step_in, tp_200_src_c_lp_step_out, tp_200_dst_cl_step_in,
        tp_200_dst_cl_step_out, tp_200_dst_cl_lp_step_in, tp_200_dst_cl_lp_step_out, tp_200_c_mod_c0,
        tp_200_dst_cr_dims, tp_200_dst_cl_dims, tp_200_is_mc_cr, tp_200_is_mc_cl, tp_200_src_r2nd_dst_r1st_same,
        tp_200_left_cl_c_cr_size, tp_names["tp_200_cl_in_idx_0_size"], tp_names["tp_200_cl_in_idx_0_dst_rsize"],
        tp_names["tp_200_cl_in_idx_0_src_asize"], tp_names["tp_200_cl_in_idx_1_size"],
        tp_names["tp_200_cl_in_idx_1_dst_rsize"], tp_names["tp_200_cl_in_idx_1_src_asize"],
        tp_names["tp_200_cr_in_idx_0_size"], tp_names["tp_200_cr_in_idx_0_dst_rsize"],
        tp_names["tp_200_cr_in_idx_0_src_asize"], tp_names["tp_200_cr_in_idx_1_size"],
        tp_names["tp_200_cr_in_idx_1_dst_rsize"], tp_names["tp_200_cr_in_idx_1_src_asize"]
    ]

    return sub_tiling_params


def _get_tiling_params_func(args):
    """
    get tiling parameters function
    """

    in_shape, out_shape, block_elem_cnt, ub_size, in_dtype, hidden_size = args

    (in_shape_new, out_shape_new, in_format_new, out_format_new) = _renew_input_output_shape_format(in_shape, out_shape)
    args_get_tp = in_shape_new, out_shape_new, in_format_new, out_format_new, \
                  block_elem_cnt, ub_size, in_dtype, hidden_size
    tiling_params = _tiling_params_negative(args_get_tp)

    return tiling_params


def _twice_vnchwconv_invert(args):
    """
    do nc1ht to nc1th transform by twice vnchwconv
    """

    (tik_inst, src_ub, ub_offset, in_dtype, cl_plp_size, dst_c_size, c_plp_size,
     cr_lp_cnt, hidden_output, cr_pln_size, is_mc_cr, c0_len, ele_per_block, tiling_mode, c0_pad_size) = args
    size_factor = tdc.get_dtype_factor(in_dtype)
    if in_dtype in NEED_CAST_DTYPES:
        src_ub_casted = src_ub.reinterpret_cast_to("float16")
    else:
        src_ub_casted = src_ub
    ub_offset_casted = ub_offset * size_factor

    def _do_nc1ht_2_nc1th(src_cl_offset, dst_cl_offset):
        with tik_inst.for_range(0, c_plp_size) as c1_idx:
            with tik_inst.for_range(0, c0_len) as c0_idx:
                tik_inst.data_move(src_ub[(c1_idx * c0_len + c0_idx) * cr_align_block_size *
                                          ele_per_block * size_factor + dst_cl_offset],
                                   src_ub[ub_offset + (c1_idx * hidden_output * c0_len + c0_idx) *
                                          ele_per_block * size_factor + src_cl_offset],
                                   0, hidden_output, size_factor, (c0_len - 1) * size_factor, 0)

    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar(dtype="int32")
        dst_stride = tik_inst.Scalar(dtype="int32")
        cr_align_block_size = tik_inst.Scalar(name="cr_align_block_size")
        vnc_col_len = tik_inst.Scalar(name="vnc_col_len")
        with tik_inst.if_scope(tik.all(cr_lp_cnt == 1, is_mc_cr == 0)):
            cr_align_block_size.set_as(hidden_output)
        with tik_inst.else_scope():
            cr_align_block_size.set_as(tdc.ceil_fill(hidden_output, ele_per_block))

        # do nc1ht -> c1htn
        with tik_inst.if_scope(tiling_mode == 2002):  # using 16 lines
            vnc_col_len.set_as(c_plp_size * hidden_output * c0_len * size_factor)
        with tik_inst.else_scope():  # using 1 line
            vnc_col_len.set_as(cl_plp_size * c_plp_size * hidden_output * c0_len * size_factor)
        repeat_cnt = vnc_col_len // c0_len
        src_addr_list = [src_ub_casted[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
        dst_addr_list = [src_ub_casted[ub_offset_casted + c0_len * i] for i in tdc.ADDR_IDX_LIST]
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(1)
            dst_stride.set_as(16)
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        # do c1htn -> c1thn
        with tik_inst.if_scope(tiling_mode == 2002):
            src_cl_offset_2002 = 0
            dst_cl_offset_2002 = 0
            _do_nc1ht_2_nc1th(src_cl_offset_2002, dst_cl_offset_2002)
            vnc_col_len.set_as(c_plp_size * cr_align_block_size * c0_len * size_factor)
        with tik_inst.else_scope():
            with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                src_cl_offset_2003 = cl_idx * c_plp_size * hidden_output * c0_len * ele_per_block
                dst_cl_offset_2003 = cl_idx * dst_c_size * hidden_output * ele_per_block
                _do_nc1ht_2_nc1th(src_cl_offset_2003, dst_cl_offset_2003)
            vnc_col_len.set_as(cl_plp_size * c_plp_size * cr_align_block_size * c0_len * size_factor)
        # do c1thn -> nc1th
        repeat_cnt = vnc_col_len // c0_len
        src_addr_list = [src_ub_casted[tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
        dst_addr_list = [src_ub_casted[ub_offset_casted + vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(16)
            dst_stride.set_as(1)
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


def _once_vnchwconv_invert(args):
    """
    do nc1ht to nc1th transform by once vnchwconv
    """

    tik_inst, src_ub, ub_offset, in_dtype, cl_plp_size, c_plp_size, cr_lp_idx, dst_cr_lp_unit, cr_pln_size, c0_len, \
    ele_per_block, hidden_output, hidden_size, core_step_in = args

    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar(dtype="int64")
        dst_stride = tik_inst.Scalar(dtype="int64")
        hidden_ceil = tik_inst.Scalar(dtype="int64")
        hidden_left = tik_inst.Scalar(dtype="int64")
        hidden_head_len = tik_inst.Scalar(dtype="int64")
        hidden_mid_start = tik_inst.Scalar(dtype="int64")
        hidden_mid_cnt = tik_inst.Scalar(dtype="int64")
        hidden_tail_start = tik_inst.Scalar(dtype="int64")
        hidden_tail_len = tik_inst.Scalar(dtype="int64")
        hidden_ceil.set_as(tdc.ceil_fill(hidden_size, c0_len))
        hidden_left.set_as((core_step_in + dst_cr_lp_unit * cr_lp_idx) % hidden_ceil)
        clean_len = c_plp_size * cr_pln_size * c0_len
        tdc.clean_ubuf(tik_inst, src_ub[ub_offset], 0, clean_len)
        with tik_inst.if_scope(cr_pln_size > hidden_ceil - hidden_left):
            with tik_inst.if_scope(hidden_left < hidden_size):
                with tik_inst.if_scope(hidden_left == 0):
                    hidden_head_len.set_as(0)
                    hidden_mid_start.set_as(0)
                with tik_inst.else_scope():
                    hidden_head_len.set_as(hidden_size - hidden_left)
                    hidden_mid_start.set_as(hidden_ceil - hidden_left)
            with tik_inst.else_scope():
                hidden_head_len.set_as(0)
                hidden_mid_start.set_as(hidden_ceil - hidden_left)
            with tik_inst.if_scope(cr_pln_size > hidden_mid_start):
                hidden_mid_cnt.set_as((cr_pln_size - hidden_mid_start) // hidden_ceil)
                hidden_tail_start.set_as(hidden_mid_start + hidden_mid_cnt * hidden_ceil)
                hidden_tail_len.set_as(cr_pln_size - hidden_tail_start)
                with tik_inst.if_scope(hidden_tail_len > hidden_size):
                    hidden_tail_len.set_as(hidden_size)
            with tik_inst.else_scope():
                hidden_mid_cnt.set_as(0)
                hidden_tail_len.set_as(0)
            hidden_output.set_as(hidden_head_len + hidden_mid_cnt * hidden_size + hidden_tail_len)
            #head
            with tik_inst.if_scope(hidden_head_len > 0):
                src_stride.set_as(cr_pln_size - hidden_head_len)
                dst_stride.set_as(hidden_output - hidden_head_len)
                tik_inst.data_move(src_ub[ub_offset], src_ub, 0, c_plp_size, hidden_head_len, src_stride, dst_stride)
            #middle
            src_stride.set_as(cr_pln_size - hidden_size)
            dst_stride.set_as(hidden_output - hidden_size)
            with tik_inst.for_range(0, hidden_mid_cnt) as hidden_mid_idx:
                tik_inst.data_move(src_ub[ub_offset + (hidden_head_len + hidden_mid_idx * hidden_size) * c0_len],
                                   src_ub[(hidden_mid_start + hidden_mid_idx * hidden_ceil) * c0_len], 0, c_plp_size,
                                   hidden_size, src_stride, dst_stride)
            #tail
            with tik_inst.if_scope(hidden_tail_len > 0):
                src_stride.set_as(cr_pln_size - hidden_tail_len)
                dst_stride.set_as(hidden_output - hidden_tail_len)
                tik_inst.data_move(src_ub[ub_offset + (hidden_head_len + hidden_mid_cnt * hidden_size) * c0_len],
                                   src_ub[hidden_tail_start * c0_len], 0, c_plp_size, hidden_tail_len, src_stride,
                                   dst_stride)
        with tik_inst.else_scope():
            with tik_inst.if_scope(hidden_left < hidden_size):
                hidden_output.set_as(cr_pln_size)
                with tik_inst.if_scope(cr_pln_size > hidden_size - hidden_left):
                    hidden_output.set_as(hidden_size - hidden_left)
                src_stride.set_as(cr_pln_size - hidden_output)
                dst_stride.set_as(0)
                tik_inst.data_move(src_ub[ub_offset], src_ub, 0, c_plp_size, hidden_output, src_stride, dst_stride)
            with tik_inst.else_scope():
                hidden_output.set_as(0)

        with tik_inst.if_scope(hidden_output > 0):
            #reload
            tdc.clean_ubuf(tik_inst, src_ub, 0, ub_offset)
            tik_inst.data_move(src_ub, src_ub[ub_offset], 0, 1, c_plp_size * hidden_output, 0, 0)

            #vnc
            cr_align_block_size = tdc.ceil_fill(hidden_output, ele_per_block)
            repeat_cnt = tdc.ceil_div(hidden_output, ele_per_block)
            with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                with tik_inst.for_range(0, c_plp_size) as c_idx:
                    src_offset = (cl_idx * c_plp_size + c_idx) * hidden_output * c0_len
                    dst_offset = (cl_idx * c_plp_size + c_idx) * cr_align_block_size * c0_len + ub_offset
                    # do c1ht -> c1th
                    src_addr_list = [src_ub[c0_len * i + src_offset] for i in tdc.ADDR_IDX_LIST]
                    dst_addr_list = [src_ub[cr_align_block_size * i + dst_offset] for i in tdc.ADDR_IDX_LIST]
                    with tik_inst.if_scope(repeat_cnt == 1):
                        src_stride.set_as(0)
                        dst_stride.set_as(0)
                    with tik_inst.else_scope():
                        src_stride.set_as(16)
                        dst_stride.set_as(1)
                    tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


def _reorder_data(args, tiling_mode):
    """
    reorder data from ncht to ncth
    """

    (tik_inst, src_ub, ub_offset, in_dtype, cl_plp_size, dst_c_size, c_plp_size, cr_lp_cnt, cr_lp_idx, dst_cr_lp_unit,
     cr_pln_size, is_mc_cr, c0_len, ele_per_block, c0_pad_size, hidden_cnt, hidden_output, hidden_size,
     core_step_in) = args

    # dtype is float16, int8, uint8 and c-right >= c0_size
    with tik_inst.if_scope(tiling_mode == 2001):
        vnc_args = (tik_inst, src_ub, ub_offset, in_dtype, cl_plp_size, c_plp_size, cr_lp_idx, dst_cr_lp_unit,
                    cr_pln_size, c0_len, ele_per_block, hidden_output, hidden_size, core_step_in)
        _once_vnchwconv_invert(vnc_args)
    with tik_inst.else_scope():
        hidden_output.set_as(hidden_cnt * hidden_size)
        vnc_args = (tik_inst, src_ub, ub_offset, in_dtype, cl_plp_size, dst_c_size, c_plp_size, cr_lp_cnt,
                    hidden_output, cr_pln_size, is_mc_cr, c0_len, ele_per_block, tiling_mode, c0_pad_size)
        _twice_vnchwconv_invert(vnc_args)


def _update_input_offset_all_dims_one(args):
    """
    count input gm offset for c-left and c-right only have one dimension
    """

    (cr_lp_idx, dst_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, cl_lp_idx, dst_cl_lp_step_in, core_step_in, cr_backend,
     dst_cr_step_in, cl_backend, dst_cl_step_in, c_backend, src_c_step_in) = args

    in_offset = (cr_lp_idx * dst_cr_lp_step_in + c_lp_idx * src_c_lp_step_in + cl_lp_idx * dst_cl_lp_step_in +
                 core_step_in - (cr_backend * dst_cr_step_in + cl_backend * dst_cl_step_in + c_backend * src_c_step_in))

    return in_offset


def _update_output_offset(args):
    """
    count output gm offset
    """

    (cl_lp_idx, dst_cl_lp_step_out, cl_backend, dst_cl_step_out, c_lp_idx, src_c_lp_step_out, cr_lp_idx,
     hidden_dst_cr_lp_step_out, cr_backend, dst_cr_step_out, hidden_core_step_out, c_backend, src_c_step_out,
     block_idx) = args

    out_offset = (cl_lp_idx * dst_cl_lp_step_out - cl_backend * dst_cl_step_out + hidden_dst_cr_lp_step_out -
                  cr_backend * dst_cr_step_out + c_lp_idx * src_c_lp_step_out - c_backend * src_c_step_out +
                  hidden_core_step_out)

    return out_offset


def _move_data_in_cr_cl_one_dims(args):
    """
    move data in process when c-right or c-right only has one dimensions
    """

    (tik_inst, src_in_gm, cl_cr_dims_gm_offset, src_ub, cl_cr_dims_ub_offset, c0_pad_size, c_cr_block_mod,
     c_plp_size, src_c_step_in, cr_pln_size, c_cr_gap, c0_len, ele_per_block) = args

    with tik_inst.new_stmt_scope(disable_sync=True):
        with tik_inst.if_scope(tik.all(c_cr_gap <= tdc.STRIDE_LIMIT_MTE, c_cr_block_mod == 0, c0_pad_size == 0)):
            tik_inst.data_move(src_ub[cl_cr_dims_ub_offset], src_in_gm[cl_cr_dims_gm_offset],
                               0, c_plp_size, cr_pln_size * c0_len // ele_per_block, c_cr_gap, 0)
        with tik_inst.else_scope():
            with tik_inst.for_range(0, c_plp_size) as c_idx:
                cl_cr_dims_one_ub_offset = c_idx * (cr_pln_size * c0_len + c0_pad_size) + cl_cr_dims_ub_offset
                cl_cr_dims_one_gm_offset = c_idx * src_c_step_in + cl_cr_dims_gm_offset
                tik_inst.data_move(src_ub[cl_cr_dims_one_ub_offset], src_in_gm[cl_cr_dims_one_gm_offset],
                                   0, 1, tdc.ceil_div(cr_pln_size * c0_len, ele_per_block), 0, 0)


# 'pylint: disable=unused-variable
def _copy_data_in_0(in_offset_args, tik_args):
    """
    copy data from gm to ub for transform such as nc1hwc0 -> nchw
    """

    (tiling_mode, cr_lp_idx, dst_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, cl_lp_idx, dst_cl_lp_step_in, core_step_in,
     cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize, cr_in_idx_1_size, cr_in_idx_1_dst_rsize,
     cr_in_idx_1_src_asize, cl_in_idx_0_size, cl_in_idx_0_dst_rsize, cl_in_idx_0_src_asize, cl_in_idx_1_size,
     cl_in_idx_1_dst_rsize, cl_in_idx_1_src_asize) = in_offset_args
    (tik_inst, src_in_gm, src_ub, hidden_cnt, c0_pad_size, dst_cr_step_in, cr_pln_size, nlc_cr_lp_cnt, dst_cr_lp_unit,
     cr_backend, dst_cr_dims, src_c_step_in, c_plp_size, c_backend, dst_cl_step_in, cl_plp_size, nlc_cl_lp_cnt,
     dst_cl_lp_unit, cl_backend, dst_cl_dims, ele_per_block, c0_len, is_mc_cl, is_mc_cr, hidden_size,
     block_idx) = tik_args

    with tik_inst.new_stmt_scope():
        offset_args = (cr_lp_idx, dst_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, cl_lp_idx, dst_cl_lp_step_in,
                       core_step_in, cr_backend, dst_cr_step_in, cl_backend, dst_cl_step_in, c_backend, src_c_step_in)
        base_gm_offset = _update_input_offset_all_dims_one(offset_args)
        with tik_inst.if_scope(tiling_mode == 2001):
            c_cr_gap = (src_c_step_in - cr_pln_size * dst_cr_step_in) // ele_per_block
            c_cr_block_mod = (src_c_step_in - cr_pln_size * dst_cr_step_in) % ele_per_block
            with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                cl_cr_dims_ub_offset = cl_idx * c_plp_size * (cr_pln_size * c0_len + c0_pad_size)
                cl_cr_dims_gm_offset = cl_idx * dst_cl_step_in + base_gm_offset
                data_in_args = (tik_inst, src_in_gm, cl_cr_dims_gm_offset, src_ub, cl_cr_dims_ub_offset, c0_pad_size,
                                c_cr_block_mod, c_plp_size, src_c_step_in, cr_pln_size, c_cr_gap, c0_len, ele_per_block)
                _move_data_in_cr_cl_one_dims(data_in_args)
        with tik_inst.else_scope():
            tik_inst.data_move(src_ub, src_in_gm[base_gm_offset], 0, c_plp_size * hidden_cnt,
                               hidden_size * c0_len // ele_per_block, c0_len - hidden_size % c0_len, 0)


def copy_nonalign_data(input_args):
    """
    copy non align data from ub to gm with address fallback
    """
    tmp_reg, ele_per_block, tmp_ub_offset, tmp_gm_offset, hidden_output, src_ub, dst_out_gm, tik_inst = input_args
    for i in tdc.REG_IDX_LIST[:ele_per_block]:
        tmp_reg[i].set_as(src_ub[tmp_ub_offset + hidden_output - ele_per_block + i])
    for i in tdc.REG_IDX_LIST[:ele_per_block]:
        src_ub[i:].set_as(tmp_reg[i])
    tik_inst.data_move(dst_out_gm[tmp_gm_offset + hidden_output - ele_per_block], src_ub, 0, 1, 1, 0, 0)
                    
                    
# 'pylint: disable=unused-variable
def _copy_data_out(copy_out_args):
    """
    copy data from ub to gm
    """

    (tik_inst, dst_out_gm, src_ub, tiling_mode, src_c_step_out, dst_cl_step_out, cl_plp_size, hidden_output, c_plp_size,
     cr_lp_cnt, is_mc_cr, ele_per_block, is_last_c1, c0_len, c_mod_c0) = copy_out_args

    cr_block_align_size = tdc.ceil_fill(hidden_output, ele_per_block)
    support_atomic = tbe_platform.api_check_support("tik.set_atomic_add", dst_out_gm.dtype)
    with tik_inst.new_stmt_scope():
        tmp_reg = [tik_inst.Scalar(dtype=src_ub.dtype) for i in tdc.REG_IDX_LIST[:ele_per_block]]
        sub_c_size = tik_inst.Scalar(dtype="int64", name="sub_c_size")
        with tik_inst.if_scope(tik.all(c_mod_c0 > 0, is_last_c1 > 0)):
            sub_c_size.set_as((c_plp_size - 1) * c0_len + c_mod_c0)
        with tik_inst.else_scope():
            sub_c_size.set_as(c_plp_size * c0_len)

        with tik_inst.if_scope(tiling_mode == 2003):
            burst_len = cl_plp_size * sub_c_size * hidden_output
            with tik_inst.if_scope(burst_len // ele_per_block > 0):
                tik_inst.data_move(dst_out_gm, src_ub, 0, 1, burst_len // ele_per_block, 0, 0)
                with tik_inst.if_scope(burst_len % ele_per_block > 0):
                    for i in tdc.REG_IDX_LIST[:ele_per_block]:
                        tmp_reg[i].set_as(src_ub[burst_len - ele_per_block + i])
                    for i in tdc.REG_IDX_LIST[:ele_per_block]:
                        src_ub[i:].set_as(tmp_reg[i])
                    tik_inst.data_move(dst_out_gm[burst_len - ele_per_block], src_ub, 0, 1, 1, 0, 0)
            with tik_inst.else_scope():  # data less than 1 block size
                tik_inst.data_move(dst_out_gm, src_ub, 0, 1, 1, 0, 0)

        with tik_inst.else_scope():
            with tik_inst.if_scope(tik.all(tiling_mode == 2002, cr_lp_cnt == 1, is_mc_cr == 0)):
                c_cr_size = sub_c_size * hidden_output
                with tik_inst.new_stmt_scope(disable_sync=True):
                    with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                        tmp_ub_offset = cl_idx * c_plp_size * c0_len * hidden_output
                        tmp_gm_offset = cl_idx * dst_cl_step_out
                        tik_inst.data_move(dst_out_gm[tmp_gm_offset], src_ub[tmp_ub_offset],
                                           0, 1, c_cr_size // ele_per_block, 0, 0)
                with tik_inst.if_scope(c_cr_size % ele_per_block > 0):
                    with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                        tmp_ub_offset = cl_idx * c_plp_size * c0_len * hidden_output
                        tmp_gm_offset = cl_idx * dst_cl_step_out
                        for i in tdc.REG_IDX_LIST[:ele_per_block]:
                            tmp_reg[i].set_as(src_ub[tmp_ub_offset + c_cr_size - ele_per_block + i])
                        for i in tdc.REG_IDX_LIST[:ele_per_block]:
                            src_ub[i:].set_as(tmp_reg[i])
                        tik_inst.data_move(dst_out_gm[tmp_gm_offset + c_cr_size - ele_per_block], src_ub,
                                           0, 1, 1, 0, 0)
            with tik_inst.else_scope():
                with tik_inst.if_scope(hidden_output > 0):
                    with tik_inst.new_stmt_scope(disable_sync=True):
                        with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                            with tik_inst.for_range(0, sub_c_size) as c_idx:
                                tmp_ub_offset = (cl_idx * c_plp_size * c0_len + c_idx) * cr_block_align_size
                                tmp_gm_offset = cl_idx * dst_cl_step_out + c_idx * src_c_step_out
                                with tik_inst.if_scope(hidden_output // ele_per_block > 0):
                                    tik_inst.data_move(dst_out_gm[tmp_gm_offset], src_ub[tmp_ub_offset], 0, 1,
                                                       hidden_output // ele_per_block, 0, 0)
                    with tik_inst.if_scope(hidden_output % ele_per_block > 0):
                        with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                            with tik_inst.for_range(0, sub_c_size) as c_idx:
                                tmp_ub_offset = (cl_idx * c_plp_size * c0_len + c_idx) * cr_block_align_size
                                tmp_gm_offset = cl_idx * dst_cl_step_out + c_idx * src_c_step_out
                                input_args = (tmp_reg, ele_per_block, tmp_ub_offset, tmp_gm_offset, hidden_output,
                                              src_ub, dst_out_gm, tik_inst)
                                if support_atomic:
                                    if tbe_platform.api_check_support("tik.data_move_pad"):
                                        left_data_size = tik_inst.Scalar(dtype="int64", name="left_data_size")
                                        left_num = hidden_output % ele_per_block
                                        left_data_size.set_as(tdc.get_dtype_len(dst_out_gm.dtype) * left_num)
                                        tik_inst.data_move_pad(dst_out_gm[tmp_gm_offset + hidden_output - left_num],
                                                               src_ub[tmp_ub_offset + hidden_output - left_num], 1,
                                                               left_data_size, 0, 0)
                                    else:
                                        with tik_inst.if_scope(hidden_output < ele_per_block):
                                            tik_inst.set_atomic_add(dst_out_gm.dtype)
                                            copy_nonalign_data(input_args)
                                            tdc.clean_ubuf(tik_inst, src_ub, 0, ele_per_block)
                                            tik_inst.set_atomic_add(0)
                                        with tik_inst.else_scope():
                                            copy_nonalign_data(input_args)
                                else:
                                    copy_nonalign_data(input_args)


def _copy_data_out_with_offset(copy_out_args):
    """
    copy data from ub to gm
    """

    (tik_inst, dst_out_gm, src_ub, src_c_step_out, dst_cl_step_out, cl_plp_size, hidden_output, c_plp_size,
    ele_per_block, is_last_c1, c0_len, c_mod_c0) = copy_out_args

    cr_block_align_size = tdc.ceil_fill(hidden_output, ele_per_block)
    with tik_inst.new_stmt_scope():
        tmp_reg_output = [tik_inst.Scalar(dtype=src_ub.dtype) for i in tdc.REG_IDX_LIST[:ele_per_block]]
        sub_c_size = tik_inst.Scalar(dtype="int64")
        with tik_inst.if_scope(tik.all(c_mod_c0 > 0, is_last_c1 > 0)):
            sub_c_size.set_as((c_plp_size - 1) * c0_len + c_mod_c0)
        with tik_inst.else_scope():
            sub_c_size.set_as(c_plp_size * c0_len)

        with tik_inst.if_scope(hidden_output % ele_per_block > 0):
            with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                with tik_inst.for_range(0, sub_c_size) as c_idx:
                    tmp_gm_offset = cl_idx * dst_cl_step_out + c_idx * src_c_step_out
                    tmp_ub_offset = (cl_idx * c_plp_size * c0_len + c_idx) * cr_block_align_size
                    
                    for i in tdc.REG_IDX_LIST[:ele_per_block]:
                        tmp_reg_output[i].set_as(src_ub[tmp_ub_offset + hidden_output - ele_per_block + i])
                    for i in tdc.REG_IDX_LIST[:ele_per_block]:
                        src_ub[i:].set_as(tmp_reg_output[i])
                    tik_inst.data_move(dst_out_gm[tmp_gm_offset + hidden_output - ele_per_block - c0_len], src_ub, 0, 1,
                                       1, 0, 0)


def calc_cr_offset(step_out, in_idx, hidden_size, c0_size, cr_offset):
    hidden_ceil = tdc.ceil_fill(hidden_size, c0_size)
    cr_offset_cnt = (step_out * in_idx) // hidden_ceil
    cr_offset.set_as(cr_offset_cnt * (hidden_ceil - hidden_size))


def calc_cr_offset_with_core_offset(offset_args):
    step_out, in_idx, hidden_size, c0_size, core_step_out, cr_offset = offset_args
    hidden_ceil = tdc.ceil_fill(hidden_size, c0_size)
    cr_offset_cnt = (core_step_out + step_out * in_idx) // hidden_ceil
    cr_offset.set_as(cr_offset_cnt * (hidden_ceil - hidden_size))


def _once_vnchwconv_with_offset(vnchwconv_args):
    """
    vnchwconv data with 16*16 offset, repeat_cnt is at least 2 times
    """
    (tik_inst, src_ub, ub_offset, hidden_output_with_offset, ele_per_block, cl_plp_size, c_plp_size,
     c0_len) = vnchwconv_args
    src_stride = tik_inst.Scalar(name="src_stride")
    dst_stride = tik_inst.Scalar(name="dst_stride")
    cr_align_block_size = tdc.ceil_fill(hidden_output_with_offset, ele_per_block)
    repeat_cnt = tdc.ceil_div(hidden_output_with_offset, ele_per_block)
    with tik_inst.for_range(0, cl_plp_size) as cl_idx:
        with tik_inst.for_range(0, c_plp_size) as c_idx:
            src_offset = (cl_idx * c_plp_size + c_idx) * hidden_output_with_offset * c0_len
            dst_offset = (cl_idx * c_plp_size + c_idx) * cr_align_block_size * c0_len + ub_offset
            src_addr_list = [src_ub[c0_len * i + src_offset] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [src_ub[cr_align_block_size * i + dst_offset] for i in tdc.ADDR_IDX_LIST]
            src_stride.set_as(16)
            dst_stride.set_as(1)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


def process_unsupport_atomic(tensor_args, tp_args, process_args):
    """
    When the soc does not support atomic, the data with 16*16 offset will be read forward in the following two cases
        1. Under the last loop of mode 2, when the loop data number is 16
        2. When the mode is not 2 and the last kernel cycle data number is 16
    """
    (tiling_mode, ub_offset, mc_pos, used_core_cnt, hidden_cnt, c0_len, core_step_in, core_step_out, nlc_cr_lp_cnt,
     nlc_c_lp_cnt, nlc_cl_lp_cnt, nlc_cr_left, nlc_c_left, nlc_cl_left, lc_cr_lp_cnt, lc_c_lp_cnt, lc_cl_lp_cnt,
     lc_cr_left, lc_c_left, lc_cl_left, dst_cr_lp_unit, src_c_lp_unit, dst_cl_lp_unit, dst_cr_step_in, dst_cr_step_out,
     dst_cr_lp_step_in, dst_cr_lp_step_out, dst_c_size, src_c_step_in, src_c_step_out, src_c_lp_step_in,
     src_c_lp_step_out, dst_cl_step_in, dst_cl_step_out, dst_cl_lp_step_in, dst_cl_lp_step_out, c_mod_c0, dst_cr_dims,
     dst_cl_dims, is_mc_cr, is_mc_cl, src_r2nd_dst_r1st_same, left_cl_c_cr_size, cl_in_idx_0_size,
     cl_in_idx_0_dst_rsize, cl_in_idx_0_src_asize, cl_in_idx_1_size, cl_in_idx_1_dst_rsize, cl_in_idx_1_src_asize,
     cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize, cr_in_idx_1_size, cr_in_idx_1_dst_rsize,
     cr_in_idx_1_src_asize) = tp_args
    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, ele_per_block, hidden_size, in_dtype = tensor_args
    (cr_lp_idx, c_lp_idx, cl_lp_idx, cr_backend, cl_backend, c_backend, cl_plp_size, cr_pln_size, c_plp_size,
     c0_pad_size, is_last_c1) = process_args

    hidden_left = tik_inst.Scalar(name="hidden_left")
    hidden_ceil = tik_inst.Scalar(name="hidden_ceil")
    hidden_output = tik_inst.Scalar(name="hidden_output")
    hidden_core_step_out = tik_inst.Scalar(name="hidden_core_step_out")
    hidden_dst_cr_lp_step_out = tik_inst.Scalar(name="hidden_dst_cr_lp_step_out")
    cr_lp_offset = tik_inst.Scalar(name="cr_lp_offset")
    hidden_output.set_as(0)
    hidden_ceil.set_as(tdc.ceil_fill(hidden_size, c0_len))

    with tik_inst.new_stmt_scope():
        offset_args = (cr_lp_idx, dst_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, cl_lp_idx, dst_cl_lp_step_in,
                       block_idx * core_step_in, cr_backend, dst_cr_step_in, cl_backend, dst_cl_step_in, c_backend,
                       src_c_step_in)

        # Copy data to ub, offset forward more 16*16, src_stride should also be reduced by 16
        base_gm_offset = _update_input_offset_all_dims_one(offset_args) - c0_len * c0_len
        c_cr_gap = (src_c_step_in - cr_pln_size * dst_cr_step_in) // ele_per_block - c0_len
        c_cr_block_mod = (src_c_step_in - cr_pln_size * dst_cr_step_in) % ele_per_block

        with tik_inst.for_range(0, cl_plp_size) as cl_idx:
            cl_cr_dims_ub_offset = cl_idx * c_plp_size * (cr_pln_size * c0_len + c0_pad_size)
            cl_cr_dims_gm_offset = cl_idx * dst_cl_step_in + base_gm_offset
            data_in_args = (tik_inst, src_in_gm, cl_cr_dims_gm_offset, src_ub, cl_cr_dims_ub_offset, c0_pad_size,
                            c_cr_block_mod, c_plp_size, src_c_step_in, cr_pln_size + c0_len, c_cr_gap, c0_len,
                            ele_per_block)
            _move_data_in_cr_cl_one_dims(data_in_args)

        with tik_inst.if_scope(mc_pos == 2):
            offset_args = (dst_cr_lp_step_out, cr_lp_idx, hidden_size, c0_len, block_idx * core_step_out, cr_lp_offset)
            calc_cr_offset_with_core_offset(offset_args)
            hidden_dst_cr_lp_step_out.set_as(cr_lp_idx * dst_cr_lp_step_out)
            hidden_core_step_out.set_as(block_idx * core_step_out - cr_lp_offset)
            hidden_left.set_as((block_idx * core_step_out + dst_cr_lp_unit * cr_lp_idx) % hidden_ceil)
            hidden_output.set_as(hidden_size - hidden_left)
            tik_inst.data_move(src_ub[ub_offset], src_ub, 0, c_plp_size, hidden_output + c0_len,
                               cr_pln_size - hidden_output, 0)
        with tik_inst.else_scope():
            calc_cr_offset(dst_cr_lp_step_out, cr_lp_idx, hidden_size, c0_len, cr_lp_offset)
            hidden_dst_cr_lp_step_out.set_as(cr_lp_idx * dst_cr_lp_step_out - cr_lp_offset)
            hidden_core_step_out.set_as(block_idx * core_step_out)
            hidden_left.set_as((block_idx * core_step_in + dst_cr_lp_unit * cr_lp_idx) % hidden_ceil)
            hidden_output.set_as(hidden_size - hidden_left)
            tik_inst.data_move(src_ub[ub_offset], src_ub, 0, c_plp_size, hidden_output + c0_len,
                               cr_pln_size - hidden_output, 0)

        # Clear the data before ub[offset] and move the data after ub_offset back in preparation for vnchwconv
        tdc.clean_ubuf(tik_inst, src_ub, 0, ub_offset)
        hidden_output.set_as(hidden_output + c0_len)
        tik_inst.data_move(src_ub, src_ub[ub_offset], 0, 1, c_plp_size * hidden_output, 0, 0)
        vnchwconv_args = (tik_inst, src_ub, ub_offset, hidden_output, ele_per_block, cl_plp_size, c_plp_size, c0_len)
        _once_vnchwconv_with_offset(vnchwconv_args)

        out_gm_args = (cl_lp_idx, dst_cl_lp_step_out, cl_backend, dst_cl_step_out, c_lp_idx, src_c_lp_step_out,
                       cr_lp_idx, hidden_dst_cr_lp_step_out, cr_backend, dst_cr_step_out, hidden_core_step_out,
                       c_backend * c0_len, src_c_step_out, block_idx)
        out_gm_offset = _update_output_offset(out_gm_args)
        copy_out_args = (tik_inst, dst_out_gm[out_gm_offset:], src_ub[ub_offset:], src_c_step_out, dst_cl_step_out,
                         cl_plp_size, hidden_output, c_plp_size, ele_per_block, is_last_c1, c0_len, c_mod_c0)
        _copy_data_out_with_offset(copy_out_args)


def _func_transform_200(tensor_args, tp_args):
    """
    transform function for tiling mode 200
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, ele_per_block, hidden_size, in_dtype = tensor_args
    (tiling_mode, ub_offset, mc_pos, used_core_cnt, hidden_cnt, c0_len, core_step_in, core_step_out, nlc_cr_lp_cnt,
     nlc_c_lp_cnt, nlc_cl_lp_cnt, nlc_cr_left, nlc_c_left, nlc_cl_left, lc_cr_lp_cnt, lc_c_lp_cnt, lc_cl_lp_cnt,
     lc_cr_left, lc_c_left, lc_cl_left, dst_cr_lp_unit, src_c_lp_unit, dst_cl_lp_unit, dst_cr_step_in, dst_cr_step_out,
     dst_cr_lp_step_in, dst_cr_lp_step_out, dst_c_size, src_c_step_in, src_c_step_out, src_c_lp_step_in,
     src_c_lp_step_out, dst_cl_step_in, dst_cl_step_out, dst_cl_lp_step_in, dst_cl_lp_step_out, c_mod_c0, dst_cr_dims,
     dst_cl_dims, is_mc_cr, is_mc_cl, src_r2nd_dst_r1st_same, left_cl_c_cr_size, cl_in_idx_0_size,
     cl_in_idx_0_dst_rsize, cl_in_idx_0_src_asize, cl_in_idx_1_size, cl_in_idx_1_dst_rsize, cl_in_idx_1_src_asize,
     cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize, cr_in_idx_1_size, cr_in_idx_1_dst_rsize,
     cr_in_idx_1_src_asize) = tp_args

    def _inner_func(tiling_args):
        cr_lp_cnt, cr_left, c_lp_cnt, c_left, cl_lp_cnt, cl_left = tiling_args
        cr_pln_size = tik_inst.Scalar(name="cr_pln_size")
        c_plp_size = tik_inst.Scalar(name="c_plp_size")
        cl_plp_size = tik_inst.Scalar(name="cl_plp_size")
        is_last_c1 = tik_inst.Scalar(name="is_last_c1")
        is_cr_back = tik_inst.Scalar(name="is_cr_back")
        is_cl_back = tik_inst.Scalar(name="is_cl_back")
        is_c_back = tik_inst.Scalar(name="is_c_back")
        c0_pad_size = tik_inst.Scalar(name="c0_pad_size")
        hidden_len = tik_inst.Scalar(name="hidden_len")
        hidden_output = tik_inst.Scalar(name="hidden_output")
        hidden_core_step_out = tik_inst.Scalar(name="hidden_core_step_out")
        hidden_dst_cr_lp_step_out = tik_inst.Scalar(name="hidden_dst_cr_lp_step_out")
        cr_core_offset = tik_inst.Scalar(name="cr_core_offset")
        cr_lp_offset = tik_inst.Scalar(name="cr_lp_offset")
        hidden_output.set_as(0)
        hidden_len.set_as(hidden_size)
        not_support_atomic = not(tbe_platform.api_check_support("tik.set_atomic_add", dst_out_gm.dtype))

        with tik_inst.for_range(0, cr_lp_cnt) as cr_lp_idx:  # axis C-RIGHT
            with tik_inst.if_scope(tik.any(cr_lp_idx != cr_lp_cnt - 1, cr_left == 0)):
                # cr_lp_idx for last second core and lc_cr_left for last core
                with tik_inst.if_scope(tik.any(tik.all(block_idx == used_core_cnt - 1, cr_lp_idx == cr_lp_cnt - 2,
                                                       is_mc_cr == 1, lc_cr_left > 0,
                                                       lc_cr_left < ele_per_block),
                                               tik.all(block_idx == used_core_cnt - 2, cr_lp_idx == cr_lp_cnt - 1,
                                                       lc_cr_lp_cnt == 1, is_mc_cr == 1, lc_cr_left > 0,
                                                       lc_cr_left < ele_per_block),
                                               tik.all(cr_lp_idx == cr_lp_cnt - 2,
                                                       is_mc_cr != 1, lc_cr_left > 0, lc_cr_left < ele_per_block))):
                    cr_pln_size.set_as(dst_cr_lp_unit - ele_per_block)
                with tik_inst.else_scope():
                    cr_pln_size.set_as(dst_cr_lp_unit)
                is_cr_back.set_as(0)
            with tik_inst.else_scope():
                with tik_inst.if_scope(tik.all(used_core_cnt > 1, cr_left > 0, cr_left < ele_per_block)):
                    cr_pln_size.set_as(cr_left + ele_per_block)
                    is_cr_back.set_as(1)
                with tik_inst.else_scope():
                    cr_pln_size.set_as(cr_left)
                    is_cr_back.set_as(0)
            cr_backend = is_cr_back * ele_per_block

            with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:  # axis C
                with tik_inst.if_scope(tik.any(c_lp_idx != c_lp_cnt - 1, c_left == 0)):
                    with tik_inst.if_scope(tik.any(tik.all(block_idx == used_core_cnt - 1, c_lp_idx == c_lp_cnt - 2,
                                                           mc_pos == 1, lc_c_left == 1, c_mod_c0 > 0,
                                                           c_mod_c0 * cr_pln_size < ele_per_block),
                                                   tik.all(block_idx == used_core_cnt - 2, c_lp_idx == c_lp_cnt - 1,
                                                           lc_c_lp_cnt == 1, mc_pos == 1, lc_c_left == 1, c_mod_c0 > 0,
                                                           c_mod_c0 * cr_pln_size < ele_per_block),
                                                   tik.all(c_lp_idx == c_lp_cnt - 2,
                                                           mc_pos != 1, lc_c_left == 1, c_mod_c0 > 0,
                                                           c_mod_c0 * cr_pln_size < ele_per_block))):
                        c_plp_size.set_as(src_c_lp_unit - 1)
                    with tik_inst.else_scope():
                        c_plp_size.set_as(src_c_lp_unit)
                    is_c_back.set_as(0)
                    with tik_inst.if_scope(tik.any(tik.all(c_lp_idx == c_lp_cnt - 1, mc_pos != 1),
                                                   tik.all(c_lp_idx == c_lp_cnt - 1, mc_pos == 1,
                                                           block_idx == used_core_cnt - 1))):
                        is_last_c1.set_as(1)
                    with tik_inst.else_scope():
                        is_last_c1.set_as(0)
                with tik_inst.else_scope():
                    with tik_inst.if_scope(tik.all(used_core_cnt > 1, lc_c_left == 1, c_mod_c0 > 0,
                                                   c_mod_c0 * cr_pln_size < ele_per_block)):
                        c_plp_size.set_as(lc_c_left + 1)
                        is_c_back.set_as(1)
                    with tik_inst.else_scope():
                        c_plp_size.set_as(lc_c_left)
                        is_c_back.set_as(0)
                    is_last_c1.set_as(1)
                c_backend = is_c_back

                with tik_inst.for_range(0, cl_lp_cnt) as cl_lp_idx:  # axis C-LEFT
                    with tik_inst.if_scope(tik.any(cl_lp_idx != cl_lp_cnt - 1, cl_left == 0)):
                        with tik_inst.if_scope(tik.any(tik.all(block_idx == used_core_cnt - 1,
                                                               cl_lp_idx == cl_lp_cnt - 2, is_mc_cl == 1,
                                                               left_cl_c_cr_size > 0,
                                                               left_cl_c_cr_size < ele_per_block),
                                                       tik.all(block_idx == used_core_cnt - 2,
                                                               cl_lp_idx == cl_lp_cnt - 1,
                                                               lc_cl_lp_cnt == 1, is_mc_cl == 1,
                                                               left_cl_c_cr_size > 0,
                                                               left_cl_c_cr_size < ele_per_block),
                                                       tik.all(cl_lp_idx == cl_lp_cnt - 2, is_mc_cl != 1,
                                                               left_cl_c_cr_size > 0,
                                                               left_cl_c_cr_size < ele_per_block))):
                            cl_plp_size.set_as(dst_cl_lp_unit - ele_per_block)
                        with tik_inst.else_scope():
                            cl_plp_size.set_as(dst_cl_lp_unit)
                        is_cl_back.set_as(0)
                    with tik_inst.else_scope():
                        with tik_inst.if_scope(tik.all(used_core_cnt > 1,
                                                       left_cl_c_cr_size > 0, left_cl_c_cr_size < ele_per_block)):
                            cl_plp_size.set_as(cl_left + ele_per_block)
                            is_cl_back.set_as(1)
                        with tik_inst.else_scope():
                            cl_plp_size.set_as(cl_left)
                            is_cl_back.set_as(0)
                    cl_backend = is_cl_back * ele_per_block
                    # for NC1HWC0 -> NCHW, bool dtype and c0_size is 16
                    with tik_inst.if_scope(tik.all(in_dtype == "int8", c0_len == tdc.C0_16)):
                        with tik_inst.if_scope(src_r2nd_dst_r1st_same == 1):
                            c0_pad_size.set_as(cr_pln_size * c0_len // ele_per_block)
                        with tik_inst.else_scope():
                            c0_pad_size.set_as(cl_plp_size * c0_len // ele_per_block)
                    with tik_inst.else_scope():
                        c0_pad_size.set_as(0)

                    last_loop_cnt_less16 = tik.all(cr_lp_cnt > 1, cr_lp_idx == cr_lp_cnt - 1, cr_pln_size == c0_len,
                                                   hidden_len % c0_len != 0, mc_pos != 2, not_support_atomic)
                    last_block_cnt_less16 = tik.all(used_core_cnt > 1, block_idx == used_core_cnt - 1,
                                                    cr_pln_size == c0_len, hidden_len % c0_len != 0, mc_pos == 2,
                                                    not_support_atomic)
                    with tik_inst.if_scope(tik.any(last_loop_cnt_less16, last_block_cnt_less16)):
                        process_args = (cr_lp_idx, c_lp_idx, cl_lp_idx, cr_backend, cl_backend, c_backend, cl_plp_size,
                                        cr_pln_size, c_plp_size, c0_pad_size, is_last_c1)
                        process_unsupport_atomic(tensor_args, tp_args, process_args)
                    with tik_inst.else_scope():
                        with tik_inst.if_scope(src_r2nd_dst_r1st_same == 1):  # such as NC1HWC0 -> NCHW
                            in_offset_args = (tiling_mode, cr_lp_idx, dst_cr_lp_step_in, c_lp_idx, src_c_lp_step_in,
                                              cl_lp_idx, dst_cl_lp_step_in, block_idx * core_step_in, cr_in_idx_0_size,
                                              cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize, cr_in_idx_1_size,
                                              cr_in_idx_1_dst_rsize, cr_in_idx_1_src_asize, cl_in_idx_0_size,
                                              cl_in_idx_0_dst_rsize, cl_in_idx_0_src_asize, cl_in_idx_1_size,
                                              cl_in_idx_1_dst_rsize, cl_in_idx_1_src_asize)
                            copy_in_args = (tik_inst, src_in_gm, src_ub, hidden_cnt, c0_pad_size, dst_cr_step_in,
                                            cr_pln_size, nlc_cr_lp_cnt, dst_cr_lp_unit, cr_backend, dst_cr_dims,
                                            src_c_step_in, c_plp_size, c_backend, dst_cl_step_in, cl_plp_size,
                                            nlc_cl_lp_cnt, dst_cl_lp_unit, cl_backend, dst_cl_dims, ele_per_block,
                                            c0_len, is_mc_cl, is_mc_cr, hidden_len, block_idx)
                            _copy_data_in_0(in_offset_args, copy_in_args)
                        with tik_inst.if_scope(mc_pos == 2):
                            offset_args = (dst_cr_lp_step_out, cr_lp_idx, hidden_size, c0_len,
                                           block_idx * core_step_out, cr_lp_offset)
                            calc_cr_offset_with_core_offset(offset_args)
                            hidden_dst_cr_lp_step_out.set_as(cr_lp_idx * dst_cr_lp_step_out)
                            hidden_core_step_out.set_as(block_idx * core_step_out - cr_lp_offset)
                            reorder_args = (tik_inst, src_ub, ub_offset, in_dtype, cl_plp_size, dst_c_size, c_plp_size,
                                            cr_lp_cnt, cr_lp_idx, dst_cr_lp_unit, cr_pln_size, is_mc_cr, c0_len,
                                            ele_per_block, c0_pad_size, hidden_cnt, hidden_output, hidden_len,
                                            block_idx * core_step_out)
                            _reorder_data(reorder_args, tiling_mode)
                        with tik_inst.else_scope():
                            calc_cr_offset(dst_cr_lp_step_out, cr_lp_idx, hidden_size, c0_len, cr_lp_offset)
                            hidden_dst_cr_lp_step_out.set_as(cr_lp_idx * dst_cr_lp_step_out - cr_lp_offset)
                            hidden_core_step_out.set_as(block_idx * core_step_out)
                            reorder_args = (tik_inst, src_ub, ub_offset, in_dtype, cl_plp_size, dst_c_size, c_plp_size,
                                            cr_lp_cnt, cr_lp_idx, dst_cr_lp_unit, cr_pln_size, is_mc_cr, c0_len,
                                            ele_per_block, c0_pad_size, hidden_cnt, hidden_output, hidden_len,
                                            block_idx * core_step_in)
                            _reorder_data(reorder_args, tiling_mode)

                        out_gm_args = (cl_lp_idx, dst_cl_lp_step_out, cl_backend, dst_cl_step_out, c_lp_idx,
                                       src_c_lp_step_out, cr_lp_idx, hidden_dst_cr_lp_step_out, cr_backend,
                                       dst_cr_step_out, hidden_core_step_out, c_backend * c0_len, src_c_step_out,
                                       block_idx)
                        out_gm_offset = _update_output_offset(out_gm_args)
                        copy_out_args = (tik_inst, dst_out_gm[out_gm_offset:], src_ub[ub_offset:], tiling_mode,
                                         src_c_step_out, dst_cl_step_out, cl_plp_size, hidden_output, c_plp_size,
                                         cr_lp_cnt, is_mc_cr, ele_per_block, is_last_c1, c0_len, c_mod_c0)
                        _copy_data_out(copy_out_args)

    with tik_inst.if_scope(block_idx != used_core_cnt - 1):
        nlc_args = (nlc_cr_lp_cnt, nlc_cr_left, nlc_c_lp_cnt, nlc_c_left, nlc_cl_lp_cnt, nlc_cl_left)
        _inner_func(nlc_args)
    with tik_inst.else_scope():
        lc_args = (lc_cr_lp_cnt, lc_cr_left, lc_c_lp_cnt, lc_c_left, lc_cl_lp_cnt, lc_cl_left)
        _inner_func(lc_args)


def trans_data_rnn_negative_target(src, dst, input_size, hidden_size, kernel_name="trans_data_rnn_negative_target"):
    """

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst : dict
        dtype of output should be same type as input
    input_size: int
    hidden_size: int
    kernel_name : str
        kernel name, default value is "trans_data_rnn_negative_target"

    Returns
    -------
    None
    """

    in_shape = list(src.get("shape"))
    out_shape = list(dst.get("shape"))
    in_dtype = src.get("dtype").lower()
    ub_size = tdc.get_max_element_in_ub(in_dtype, 1)
    block_elem_cnt = tdc.BLOCK_BYTE_SIZE // tdc.get_dtype_len(in_dtype)
    tik_inst = tik.Tik()
    src_in_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tbe_platform.scope_gm, "src_in_gm")
    dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,),
                                 tbe_platform.scope_gm,
                                 "dst_out_gm",
                                 is_atomic_add=True)
    src_ub = tik_inst.Tensor(in_dtype, (ub_size,), tbe_platform.scope_ubuf, "src_ub")

    if out_shape[0] == input_size or out_shape[0] == hidden_size:
        # get tiling parameters
        args = in_shape, out_shape, block_elem_cnt, ub_size, in_dtype, hidden_size
        tiling_params = _get_tiling_params_func(args)
        used_core_cnt = tiling_params[3]
        with tik_inst.for_range(0, tdc.get_core_num(), block_num=tdc.get_core_num()) as block_idx:
            with tik_inst.if_scope(block_idx < used_core_cnt):
                tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, block_elem_cnt, hidden_size,
                               in_dtype]
                tp_args = tiling_params
                _func_transform_200(tensor_args, tp_args)
    else:
        if out_shape[0] != input_size + hidden_size:
            error_detail = "the shape of out_shape[0] is not satisfied."
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "out_shape", error_detail)
        output0_shape = [input_size, out_shape[1]]
        input0_size = tdc.ceil_div(input_size, tdc.C0_16)
        input0_shape = [input0_size, in_shape[1], in_shape[2], in_shape[3]]
        args = input0_shape, output0_shape, block_elem_cnt, ub_size, in_dtype, hidden_size
        tiling_params0 = _get_tiling_params_func(args)
        used_core_cnt0 = tiling_params0[3]
        output1_shape = [hidden_size, out_shape[1]]
        input1_size = tdc.ceil_div(hidden_size, tdc.C0_16)
        input1_shape = [input1_size, in_shape[1], in_shape[2], in_shape[3]]
        args = input1_shape, output1_shape, block_elem_cnt, ub_size, in_dtype, hidden_size
        tiling_params1 = _get_tiling_params_func(args)
        used_core_cnt1 = tiling_params1[3]
        with tik_inst.for_range(0, tdc.get_core_num(), block_num=tdc.get_core_num()) as block_idx:
            with tik_inst.if_scope(block_idx < used_core_cnt0):
                tensor_args0 = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, block_elem_cnt, hidden_size,
                                in_dtype]
                tp_args0 = tiling_params0
                _func_transform_200(tensor_args0, tp_args0)
            with tik_inst.if_scope(block_idx < used_core_cnt1):
                tensor_args1 = [tik_inst, block_idx,
                               src_in_gm[input0_size * in_shape[1] * in_shape[2] * in_shape[3]],
                               dst_out_gm[input_size * out_shape[1]],
                               src_ub, block_elem_cnt, hidden_size, in_dtype]
                tp_args1 = tiling_params1
                _func_transform_200(tensor_args1, tp_args1)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[src_in_gm], outputs=[dst_out_gm])
