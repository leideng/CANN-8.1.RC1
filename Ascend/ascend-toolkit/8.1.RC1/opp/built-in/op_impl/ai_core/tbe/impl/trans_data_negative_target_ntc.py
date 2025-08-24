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
trans_data_negative_target_ntc
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform as cce
from impl import trans_data_common_func as tdc


# used for scalar
PAD_IDX_LIST = (0, 1)
# frame up levels
FRAME_LEVEL = 2
# burst limit
BURST_LIMIT = 65535
INT8_DTYPES = ("int8", "uint8")
NEED_CAST_DTYPES = ("float32", "int32", "uint32")
VNC_SUPPORT_DTYPES = ("int8", "uint8", "float16")


# 'pylint: disable=too-many-locals, inconsistent-return-statements, too-many-lines, too-many-return-statements,
# 'pylint: disable=too-many-statements
def _renew_input_output_shape_format(in_shape, out_shape, in_format, out_format):
    """
    renew shape and format to adapt tiling process
    """

    in_format_upper = in_format.upper()
    out_format_upper = out_format.upper()

    if in_format_upper == "NDC1HWC0" and out_format_upper == "NCDHW":
        in_format_new = "NDCHT"
        out_format_new = "NCDH"
        axis_n, axis_c, axis_d, axis_h, axis_w = out_shape
        axis_c0 = in_shape[-1]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        in_shape_new = [axis_n] + [axis_d] + [axis_c1] + [axis_h * axis_w] + [axis_c0]
        out_shape_new = [axis_n] + [axis_c] + [axis_d] + [axis_h * axis_w]
        new_params_ncdhw = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_ncdhw

    if in_format_upper == "NC1HWC0" and out_format_upper == "NCHW":
        in_format_new = "NCHT"
        out_format_new = "NCH"
        axis_n, axis_c, axis_h, axis_w = out_shape
        axis_c0 = in_shape[-1]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        in_shape_new = [axis_n] + [axis_c1] + [axis_h * axis_w] + [axis_c0]
        out_shape_new = [axis_n] + [axis_c] + [axis_h * axis_w]
        new_params_nchw = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_nchw

    if in_format_upper in ("FRACTAL_Z", "FRACTAL_ZN") and out_format_upper == "HWCN":
        in_format_new = "CHNT"
        out_format_new = "HCN"
        axis_h, axis_w, axis_c, axis_n = out_shape
        axis_c0 = in_shape[-1]
        axis_ni = in_shape[-2]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, axis_ni)
        in_shape_new = [axis_c1] + [axis_h * axis_w] + [axis_no * axis_ni] + [axis_c0]
        out_shape_new = [axis_h * axis_w] + [axis_c] + [axis_n]
        new_params_hwcn = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_hwcn

    if in_format_upper == "FRACTAL_Z_3D" and out_format_upper == "DHWCN":
        in_format_new = "DCHNT"
        out_format_new = "DHCN"
        axis_d, axis_h, axis_w, axis_c, axis_n = out_shape
        axis_c0 = in_shape[-1]
        axis_ni = in_shape[-2]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, axis_ni)
        in_shape_new = [axis_d] + [axis_c1] + [axis_h * axis_w] + [axis_no * axis_ni] + [axis_c0]
        out_shape_new = [axis_d] + [axis_h * axis_w] + [axis_c] + [axis_n]
        new_params_dhwcn = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_dhwcn

    if in_format_upper in ("FRACTAL_Z", "FRACTAL_ZN") and out_format_upper == "NCHW":
        in_format_new = "CHNT"
        out_format_new = "NCH"
        axis_n, axis_c, axis_h, axis_w = out_shape
        axis_c0 = in_shape[-1]
        axis_ni = in_shape[-2]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, axis_ni)
        in_shape_new = [axis_c1] + [axis_h * axis_w] + [axis_no * axis_ni] + [axis_c0]
        out_shape_new = [axis_n] + [axis_c] + [axis_h * axis_w]
        new_params_nchw = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_nchw

    if in_format_upper == "FRACTAL_Z_3D" and out_format_upper == "NCDHW":
        in_format_new = "DCHNT"
        out_format_new = "NCDH"
        axis_n, axis_c, axis_d, axis_h, axis_w = out_shape
        axis_c0 = in_shape[-1]
        axis_ni = in_shape[-2]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, axis_ni)
        in_shape_new = [axis_d] + [axis_c1] + [axis_h * axis_w] + [axis_no * axis_ni] + [axis_c0]
        out_shape_new = [axis_n] + [axis_c] + [axis_d] + [axis_h * axis_w]

        new_params_ncdhw = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_ncdhw

    if in_format_upper in ("FRACTAL_Z", "FRACTAL_ZN") and out_format_upper == "ND":
        in_format_new = "HCNT"
        out_format_new = "HCN"
        out_shape_len = len(out_shape)
        if out_shape_len == 1:
            axis_h, axis_c, axis_n = 1, 1, out_shape[0]
        elif out_shape_len == 2:
            axis_h, axis_c, axis_n = 1, out_shape[0], out_shape[1]
        else:
            axis_h, axis_c, axis_n = tdc.get_shape_size(out_shape[:-2]), out_shape[-2], out_shape[-1]
        axis_c0 = in_shape[-1]
        axis_ni = in_shape[-2]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, axis_ni)
        in_shape_new = [axis_h] + [axis_c1] + [axis_no * axis_ni] + [axis_c0]
        out_shape_new = [axis_h] + [axis_c] + [axis_n]
        new_params_nd = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_nd

    return [in_shape, out_shape] + [in_format, out_format]


# 'pylint: disable=too-many-statements, unused-variable
def _get_mc_info_negative(cr_args, c_args, cl_args):
    """
    get multiple core axis position for negative transform
    """

    (dst_cr_lp_cnt, dst_cr_left, dst_cr_lp_unit,
     tp_200_dst_cr_lp_step_in, tp_200_dst_cr_lp_step_out, tp_200_dst_cr_dims) = cr_args
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

    in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size, in_dtype = args
    c0_len = in_shape[-1]  # axis c0
    tp_200_c0_len = c0_len
    tp_names = locals()

    # ub layout tiling parameters
    if src_format[-2] == dst_format[-1]:  # such as NC1HWC0 -> NCHW
        tp_200_src_r2nd_dst_r1st_same = 1
    else:  # such as C1HWNoNiC0 -> NCHW
        tp_200_src_r2nd_dst_r1st_same = 0

    half_ub_size = ub_size // 2 // block_elem_cnt * block_elem_cnt
    tp_200_ub_offset = half_ub_size
    vnc_col_block_size = tdc.floor_div(tp_200_ub_offset // tdc.VNC_LINES, block_elem_cnt)
    if vnc_col_block_size % 2 == 0:
        vnc_col_block_size -= 1
    vnc_col_size = vnc_col_block_size * block_elem_cnt
    tp_200_vnc_col_size = vnc_col_size

    # dst axis C-RIGHT tiling parameters
    axis_src_c_size = in_shape[src_format.index("C")]
    tp_200_dst_cr_dims = 2
    axis_dst_cr_size = tdc.get_shape_size(out_shape[dst_format.index("C") + 1:])
    cr_per_vnc_line = half_ub_size // c0_len // c0_len * c0_len
    # select different tiling mode
    if axis_dst_cr_size % c0_len == 0:
        cr_gate = 2 * tdc.VNC_LINES
    elif cr_per_vnc_line // tdc.ceil_fill(axis_dst_cr_size, c0_len) <= axis_src_c_size:
        cr_gate = 8 * tdc.VNC_LINES
    else:
        cr_gate = 15 * tdc.VNC_LINES
    # once vnchwconv flow
    # 'pylint: disable=too-many-boolean-expressions
    if (in_dtype == "float16" or (in_dtype in INT8_DTYPES and c0_len == tdc.C0_32) or
        (in_dtype in NEED_CAST_DTYPES and cce.api_check_support("tik.vmins"))) and \
            axis_dst_cr_size >= cr_gate:
        tmp_dst_cr_lp_unit = half_ub_size // c0_len // c0_len * c0_len  # block align
    else:  # twice vnchwconv flow
        if in_dtype in INT8_DTYPES:
            tmp_dst_cr_lp_unit = vnc_col_size // 2 // c0_len // block_elem_cnt * block_elem_cnt
        else:
            tmp_dst_cr_lp_unit = vnc_col_size // c0_len // block_elem_cnt * block_elem_cnt
    tp_200_dst_cr_lp_unit = tmp_dst_cr_lp_unit if axis_dst_cr_size > tmp_dst_cr_lp_unit else axis_dst_cr_size
    dst_cr_lp_cnt = tdc.ceil_div(axis_dst_cr_size, tp_200_dst_cr_lp_unit)
    dst_cr_left = axis_dst_cr_size % tp_200_dst_cr_lp_unit
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
    tp_200_dst_cr_all_in = 1 if dst_cr_lp_cnt == 1 else 0

    # axis C tiling parameters
    axis_dst_c_size = out_shape[dst_format.index("C")]
    if dst_cr_lp_cnt > 1 or axis_src_c_size == 1:
        tp_200_src_c_lp_unit = 1
    # 'pylint: disable=too-many-boolean-expressions
    elif (in_dtype == "float16" or (in_dtype in INT8_DTYPES and c0_len == tdc.C0_32) or
        (in_dtype in NEED_CAST_DTYPES and cce.api_check_support("tik.vmins"))) and \
            axis_dst_cr_size >= cr_gate:
        tmp_src_c_lp_unit = tmp_dst_cr_lp_unit // tdc.ceil_fill(tp_200_dst_cr_lp_unit, c0_len)
        tp_200_src_c_lp_unit = tmp_src_c_lp_unit if axis_src_c_size > tmp_src_c_lp_unit else axis_src_c_size
    else:
        tmp_src_c_lp_unit = tmp_dst_cr_lp_unit // tdc.ceil_fill(tp_200_dst_cr_lp_unit, block_elem_cnt)
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
    dst_c_dst_cr_size = axis_dst_c_size * axis_dst_cr_size
    # one vnchwconv flow
    if (in_dtype == "float16" or (in_dtype in INT8_DTYPES and c0_len == tdc.C0_32) or
        (in_dtype in NEED_CAST_DTYPES and cce.api_check_support("tik.vmins"))) and \
            axis_dst_cr_size >= cr_gate:
        tp_200_tiling_mode = 2001
        tmp_dst_cl_lp_unit = half_ub_size // (tp_200_src_c_lp_unit *
                                              tdc.ceil_fill(tp_200_dst_cr_lp_unit, c0_len) * c0_len)
        tp_200_dst_cl_lp_unit = tmp_dst_cl_lp_unit if axis_dst_cl_size > tmp_dst_cl_lp_unit else axis_dst_cl_size
    # 1/16 two vnchwconv flow
    elif dst_c_dst_cr_size < 54 * block_elem_cnt and dst_cr_lp_cnt == 1 and src_c_lp_cnt == 1:
        tp_200_tiling_mode = 2003
        supposed_lp_unit = 4 * block_elem_cnt
        tmp_dst_cl_lp_unit = tmp_dst_cr_lp_unit // (tp_200_src_c_lp_unit * tp_200_dst_cr_lp_unit)
        tp_200_dst_cl_lp_unit = tmp_dst_cl_lp_unit if tmp_dst_cl_lp_unit < supposed_lp_unit else supposed_lp_unit
    # full two vnchwconv flow
    else:
        tp_200_tiling_mode = 2002
        tp_200_dst_cl_lp_unit = tdc.VNC_LINES if axis_dst_cl_size > tdc.VNC_LINES else axis_dst_cl_size
    dst_cl_lp_cnt = tdc.ceil_div(axis_dst_cl_size, tp_200_dst_cl_lp_unit)
    dst_cl_left = axis_dst_cl_size % tp_200_dst_cl_lp_unit
    tp_200_left_cl_c_cr_size = dst_cl_left * axis_dst_c_size * axis_dst_cr_size  # for tiling mode 2003
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
    cr_args = (dst_cr_lp_cnt, dst_cr_left, tp_200_dst_cr_lp_unit,
               tp_200_dst_cr_lp_step_in, tp_200_dst_cr_lp_step_out, tp_200_dst_cr_dims)
    c_args = src_c_lp_cnt, src_c_left, tp_200_src_c_lp_step_in, tp_200_src_c_lp_step_out
    cl_args = dst_cl_lp_cnt, dst_cl_left, tp_200_dst_cl_lp_step_in, tp_200_dst_cl_lp_step_out, tp_200_dst_cr_dims
    (tp_200_mc_pos, tp_200_is_mc_cr, tp_200_is_mc_cl, tp_200_used_core_cnt, tp_200_core_step_in, tp_200_core_step_out,
     tp_200_nlc_cl_lp_cnt, tp_200_nlc_c_lp_cnt, tp_200_nlc_cr_lp_cnt, tp_200_nlc_cl_left, tp_200_nlc_c_left,
     tp_200_nlc_cr_left, tp_200_lc_cl_lp_cnt, tp_200_lc_c_lp_cnt, tp_200_lc_cr_lp_cnt,
     tp_200_lc_cl_left, tp_200_lc_c_left, tp_200_lc_cr_left) = _get_mc_info_negative(cr_args, c_args, cl_args)

    sub_tiling_params = [tp_200_tiling_mode, tp_200_ub_offset, tp_200_mc_pos, tp_200_used_core_cnt, tp_200_c0_len,
                         tp_200_core_step_in, tp_200_core_step_out, tp_200_nlc_cr_lp_cnt, tp_200_nlc_c_lp_cnt,
                         tp_200_nlc_cl_lp_cnt, tp_200_nlc_cr_left, tp_200_nlc_c_left, tp_200_nlc_cl_left,
                         tp_200_lc_cr_lp_cnt, tp_200_lc_c_lp_cnt, tp_200_lc_cl_lp_cnt, tp_200_lc_cr_left,
                         tp_200_lc_c_left, tp_200_lc_cl_left, tp_200_dst_cr_lp_unit, tp_200_src_c_lp_unit,
                         tp_200_dst_cl_lp_unit, tp_200_vnc_col_size, tp_200_dst_cr_step_in, tp_200_dst_cr_step_out,
                         tp_200_dst_cr_lp_step_in, tp_200_dst_cr_lp_step_out, tp_200_dst_c_size, tp_200_src_c_step_in,
                         tp_200_src_c_step_out, tp_200_src_c_lp_step_in, tp_200_src_c_lp_step_out,
                         tp_200_dst_cr_all_in, tp_200_dst_cl_step_in, tp_200_dst_cl_step_out, tp_200_dst_cl_lp_step_in,
                         tp_200_dst_cl_lp_step_out, tp_200_c_mod_c0, tp_200_dst_cr_dims, tp_200_dst_cl_dims,
                         tp_200_is_mc_cr, tp_200_is_mc_cl, tp_200_src_r2nd_dst_r1st_same, tp_200_left_cl_c_cr_size,
                         tp_names["tp_200_cl_in_idx_0_size"], tp_names["tp_200_cl_in_idx_0_dst_rsize"],
                         tp_names["tp_200_cl_in_idx_0_src_asize"], tp_names["tp_200_cl_in_idx_1_size"],
                         tp_names["tp_200_cl_in_idx_1_dst_rsize"], tp_names["tp_200_cl_in_idx_1_src_asize"],
                         tp_names["tp_200_cr_in_idx_0_size"], tp_names["tp_200_cr_in_idx_0_dst_rsize"],
                         tp_names["tp_200_cr_in_idx_0_src_asize"], tp_names["tp_200_cr_in_idx_1_size"],
                         tp_names["tp_200_cr_in_idx_1_dst_rsize"], tp_names["tp_200_cr_in_idx_1_src_asize"]]

    return sub_tiling_params


def _get_tiling_params_func(args):
    """
    get tiling parameters function
    """

    in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size, in_dtype = args

    (in_shape_new, out_shape_new,
     in_format_new, out_format_new) = _renew_input_output_shape_format(in_shape, out_shape, src_format, dst_format)
    args_get_tp = in_shape_new, out_shape_new, in_format_new, out_format_new, block_elem_cnt, ub_size, in_dtype
    tiling_params = _tiling_params_negative(args_get_tp)

    return tiling_params


def _twice_vnchwconv_invert(args):
    """
    do nc1ht to nc1th transform by twice vnchwconv
    """

    (tik_inst, src_ub, dst_ub, in_dtype, cl_plp_size, dst_c_size, c_plp_size, cr_lp_cnt, cr_pln_size,
     is_mc_cr, c0_len, ele_per_block, tiling_mode, vnc_col_len, c_mod_c0, is_last_c1) = args
    size_factor = tdc.get_dtype_factor(in_dtype)
    if in_dtype in NEED_CAST_DTYPES:
        src_ub_casted = src_ub.reinterpret_cast_to("float16")
        dst_ub_casted = dst_ub.reinterpret_cast_to("float16")
    else:
        src_ub_casted = src_ub
        dst_ub_casted = dst_ub

    def _do_nc1ht_2_nc1th(src_cl_offset, dst_cl_offset):
        with tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.if_scope(tik.all(c_mod_c0 > 0, is_last_c1 > 0)):
                with tik_inst.for_range(0, c_plp_size-1) as c1_idx:
                    with tik_inst.for_range(0, c0_len) as c0_idx:
                        tik_inst.data_move(src_ub[(c1_idx * c0_len + c0_idx) * cr_align_block_size *
                                                  ele_per_block * size_factor + dst_cl_offset],
                                           dst_ub[(c1_idx * cr_pln_size * c0_len + c0_idx) *
                                                  ele_per_block * size_factor + src_cl_offset],
                                           0, cr_pln_size, size_factor, (c0_len - 1) * size_factor, 0)
                with tik_inst.for_range(0, c_mod_c0) as c0_idx:
                    tik_inst.data_move(src_ub[((c_plp_size - 1)*c0_len + c0_idx) * cr_align_block_size *
                                              ele_per_block * size_factor + dst_cl_offset],
                                       dst_ub[((c_plp_size - 1)*cr_pln_size * c0_len + c0_idx) *
                                              ele_per_block * size_factor + src_cl_offset],
                                       0, cr_pln_size, size_factor, (c0_len - 1) * size_factor, 0)
            with tik_inst.else_scope():
                with tik_inst.for_range(0, c_plp_size) as c1_idx:
                    with tik_inst.for_range(0, c0_len) as c0_idx:
                        tik_inst.data_move(src_ub[(c1_idx * c0_len + c0_idx) * cr_align_block_size *
                                                  ele_per_block * size_factor + dst_cl_offset],
                                           dst_ub[(c1_idx * cr_pln_size * c0_len + c0_idx) *
                                                  ele_per_block * size_factor + src_cl_offset],
                                           0, cr_pln_size, size_factor, (c0_len - 1) * size_factor, 0)

    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar(dtype="int32")
        dst_stride = tik_inst.Scalar(dtype="int32")
        cr_align_block_size = tik_inst.Scalar(name="cr_align_block_size")
        vnc_col_size = tik_inst.Scalar(name="vnc_col_size")
        with tik_inst.if_scope(tik.all(cr_lp_cnt == 1, is_mc_cr == 0)):
            cr_align_block_size.set_as(cr_pln_size)
        with tik_inst.else_scope():
            cr_align_block_size.set_as(tdc.ceil_fill(cr_pln_size, ele_per_block))

        if in_dtype not in INT8_DTYPES:
            # do nc1ht -> c1htn
            with tik_inst.if_scope(tiling_mode == 2002):  # using 16 lines
                vnc_col_size.set_as(c_plp_size * cr_pln_size * c0_len * size_factor)
            with tik_inst.else_scope():  # using 1 line
                vnc_col_size.set_as(cl_plp_size * c_plp_size * cr_pln_size * c0_len * size_factor)
            repeat_cnt = vnc_col_size // c0_len
            src_addr_list = [src_ub_casted[vnc_col_len * size_factor * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub_casted[c0_len * i] for i in tdc.ADDR_IDX_LIST]
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
                vnc_col_size.set_as(c_plp_size * cr_align_block_size * c0_len * size_factor)
            with tik_inst.else_scope():
                with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                    src_cl_offset_2003 = cl_idx * c_plp_size * cr_pln_size * c0_len * ele_per_block * size_factor
                    dst_cl_offset_2003 = cl_idx * dst_c_size * cr_pln_size * ele_per_block * size_factor
                    _do_nc1ht_2_nc1th(src_cl_offset_2003, dst_cl_offset_2003)
                vnc_col_size.set_as(cl_plp_size * c_plp_size * cr_align_block_size * c0_len * size_factor)
            # do c1thn -> nc1th
            repeat_cnt = vnc_col_size // c0_len
            src_addr_list = [src_ub_casted[tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub_casted[vnc_col_len * size_factor * i] for i in tdc.ADDR_IDX_LIST]
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(16)
                dst_stride.set_as(1)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        else:
            # do nc1ht -> c1htn
            with tik_inst.if_scope(tiling_mode == 2002):
                vnc_col_size.set_as(c_plp_size * cr_pln_size * c0_len)
            with tik_inst.else_scope():
                vnc_col_size.set_as(cl_plp_size * c_plp_size * cr_pln_size * c0_len)
            repeat_cnt = vnc_col_size // ele_per_block
            src_addr_list = [src_ub[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub[ele_per_block * i] for i in tdc.ADDR_IDX_LIST]
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(1)
                dst_stride.set_as(32)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            dst_addr_list = [dst_ub[ele_per_block * (i + tdc.VNC_LINES)] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, True, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            # do c1htn -> c1thn
            with tik_inst.if_scope(tiling_mode == 2002):
                # unpad c0
                src_cl_offset_2002 = 0
                dst_cl_offset_2002 = 0
                _do_nc1ht_2_nc1th(src_cl_offset_2002, dst_cl_offset_2002)
                vnc_col_size.set_as(c_plp_size * cr_align_block_size * c0_len)
            with tik_inst.else_scope():
                # unpad c0
                with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                    src_cl_offset_2003 = cl_idx * c_plp_size * cr_pln_size * c0_len * ele_per_block
                    dst_cl_offset_2003 = cl_idx * dst_c_size * cr_pln_size * ele_per_block
                    _do_nc1ht_2_nc1th(src_cl_offset_2003, dst_cl_offset_2003)
                vnc_col_size.set_as(cl_plp_size * c_plp_size * cr_align_block_size * c0_len)
            # do c1thn -> nc1th
            repeat_cnt = vnc_col_size // ele_per_block
            src_addr_list = [src_ub[ele_per_block * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(32)
                dst_stride.set_as(1)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            src_addr_list = [src_ub[ele_per_block * (i + tdc.VNC_LINES)] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


def _once_vnchwconv_invert(args):
    """
    do nc1ht to nc1th transform by once vnchwconv
    """

    tik_inst, src_ub, dst_ub, in_dtype, cl_plp_size, c_plp_size, cr_pln_size, c0_len, ele_per_block = args

    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar(dtype="int32")
        dst_stride = tik_inst.Scalar(dtype="int32")
        cr_align_block_size = tdc.ceil_fill(cr_pln_size, c0_len)
        repeat_cnt = tdc.ceil_div(cr_pln_size, c0_len)
        if (src_ub.dtype.lower() in NEED_CAST_DTYPES and not cce.api_check_support("tik.vmins")):
            src_ub = src_ub.reinterpret_cast_to("float16")
            dst_ub = dst_ub.reinterpret_cast_to("float16")

        with tik_inst.for_range(0, cl_plp_size) as cl_idx:
            with tik_inst.for_range(0, c_plp_size) as c_idx:
                src_offset = (cl_idx * c_plp_size + c_idx) * cr_pln_size * c0_len
                dst_offset = (cl_idx * c_plp_size + c_idx) * cr_align_block_size * c0_len
                if in_dtype in ("float16", "int16", "uint16"):  # for b16
                    # do c1ht -> c1th
                    src_addr_list = [src_ub[c0_len * i + src_offset] for i in tdc.ADDR_IDX_LIST]
                    dst_addr_list = [dst_ub[cr_align_block_size * i + dst_offset] for i in tdc.ADDR_IDX_LIST]
                    with tik_inst.if_scope(repeat_cnt == 1):
                        src_stride.set_as(0)
                        dst_stride.set_as(0)
                    with tik_inst.else_scope():
                        src_stride.set_as(16)
                        dst_stride.set_as(1)
                    tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
                elif in_dtype in NEED_CAST_DTYPES and cce.api_check_support("tik.vmins"):
                    with tik_inst.for_range(0, c0_len // 8) as c0_idx:  # process 16*8 once
                        c0_offset = c0_idx * 8 * cr_align_block_size
                        src_addr_list = [src_ub[c0_len * i + c0_idx * 8 + src_offset] for i in tdc.ADDR_IDX_LIST]
                        dst_addr_list = [dst_ub[c0_offset + dst_offset], dst_ub[c0_offset + 8 + dst_offset],
                                         dst_ub[cr_align_block_size + c0_offset + dst_offset],
                                         dst_ub[cr_align_block_size + c0_offset + 8 + dst_offset],
                                         dst_ub[2 * cr_align_block_size + c0_offset + dst_offset],
                                         dst_ub[2 * cr_align_block_size + c0_offset + 8 + dst_offset],
                                         dst_ub[3 * cr_align_block_size + c0_offset + dst_offset],
                                         dst_ub[3 * cr_align_block_size + c0_offset + 8 + dst_offset],
                                         dst_ub[4 * cr_align_block_size + c0_offset + dst_offset],
                                         dst_ub[4 * cr_align_block_size + c0_offset + 8 + dst_offset],
                                         dst_ub[5 * cr_align_block_size + c0_offset + dst_offset],
                                         dst_ub[5 * cr_align_block_size + c0_offset + 8 + dst_offset],
                                         dst_ub[6 * cr_align_block_size + c0_offset + dst_offset],
                                         dst_ub[6 * cr_align_block_size + c0_offset + 8 + dst_offset],
                                         dst_ub[7 * cr_align_block_size + c0_offset + dst_offset],
                                         dst_ub[7 * cr_align_block_size + c0_offset + 8 + dst_offset],
                                         ]
                        with tik_inst.if_scope(repeat_cnt == 1):
                            src_stride.set_as(0)
                            dst_stride.set_as(0)
                        with tik_inst.else_scope():
                            src_stride.set_as(32)
                            dst_stride.set_as(2)
                        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list,
                                           repeat_cnt, dst_stride, src_stride)
                else:  # for int8, uint8
                    src_addr_list = [src_ub[c0_len * i + src_offset] for i in tdc.ADDR_IDX_LIST]
                    dst_addr_list = [dst_ub[cr_align_block_size * i + dst_offset] for i in tdc.ADDR_IDX_LIST]
                    with tik_inst.if_scope(repeat_cnt == 1):
                        src_stride.set_as(0)
                        dst_stride.set_as(0)
                    with tik_inst.else_scope():
                        src_stride.set_as(32)
                        dst_stride.set_as(1)
                    tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
                    src_addr_list = [src_ub[c0_len * (tdc.VNC_LINES + i) + src_offset] for i in tdc.ADDR_IDX_LIST]
                    tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
                    src_addr_list = [src_ub[c0_len * i + src_offset] for i in tdc.ADDR_IDX_LIST]
                    dst_addr_list = [dst_ub[cr_align_block_size * (tdc.VNC_LINES + i) + dst_offset]
                                     for i in tdc.ADDR_IDX_LIST]
                    tik_inst.vnchwconv(False, True, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
                    src_addr_list = [src_ub[c0_len * (tdc.VNC_LINES + i) + src_offset] for i in tdc.ADDR_IDX_LIST]
                    tik_inst.vnchwconv(True, True, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


def _reorder_data(args, tiling_mode):
    """
    reorder data from ncht to ncth
    """

    (tik_inst, src_ub, dst_ub, in_dtype, cl_plp_size, dst_c_size, c_plp_size, cr_lp_cnt, cr_pln_size, is_mc_cr,
     c0_len, ele_per_block, vnc_col_len, c_mod_c0, is_last_c1, sub_c_size, cl_lp_cnt_2003, dst_cr_all_in,
     dmp_flag) = args

    # data type is float16, int8, uint8 and c-right bigger than c0 size
    with tik_inst.if_scope(tiling_mode == 2001):
        vnc_args = (tik_inst, src_ub, dst_ub, in_dtype, cl_plp_size, c_plp_size,
                    cr_pln_size, c0_len, ele_per_block)
        _once_vnchwconv_invert(vnc_args)
    with tik_inst.else_scope():
        vnc_args = (tik_inst, src_ub, dst_ub, in_dtype, cl_plp_size, dst_c_size, c_plp_size, cr_lp_cnt, cr_pln_size,
                    is_mc_cr, c0_len, ele_per_block, tiling_mode, vnc_col_len, c_mod_c0, is_last_c1)
        _twice_vnchwconv_invert(vnc_args)

    def _adjust_tail_block(base_offset, col_size):
        for i in tdc.REG_IDX_LIST[:ele_per_block]:
            tmp_reg[i].set_as(dst_ub[base_offset + col_size - ele_per_block + i])
        for i in tdc.REG_IDX_LIST[:ele_per_block]:
            dst_ub[base_offset + col_size // ele_per_block * ele_per_block + i:].set_as(tmp_reg[i])

    if dmp_flag is False:
        with tik_inst.new_stmt_scope(disable_sync=True):
            tmp_reg = [tik_inst.Scalar(dtype=dst_ub.dtype) for i in tdc.REG_IDX_LIST[:ele_per_block]]
            cl_c_cr_size = cl_plp_size * sub_c_size * cr_pln_size
            c_cr_size = sub_c_size * cr_pln_size
            cl_c_size = cl_plp_size * c_plp_size * c0_len

            with tik_inst.if_scope(tik.all(tiling_mode == 2003,
                                           cl_c_cr_size > ele_per_block, cl_c_cr_size % ele_per_block > 0)):
                with tik_inst.if_scope(cl_lp_cnt_2003 > 1):
                    _adjust_tail_block((cl_lp_cnt_2003 - 1) * vnc_col_len, cl_c_cr_size)
                with tik_inst.else_scope():
                    _adjust_tail_block(0, cl_c_cr_size)
            with tik_inst.elif_scope(tik.all(tiling_mode == 2002,
                                             dst_cr_all_in == 1, c_cr_size % ele_per_block > 0)):
                with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                    _adjust_tail_block(cl_idx * vnc_col_len, c_cr_size)
            with tik_inst.elif_scope(tik.all(tiling_mode == 2002,
                                             dst_cr_all_in == 0, cr_pln_size % ele_per_block > 0)):
                with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                    with tik_inst.for_range(0, sub_c_size) as c_idx:
                        cl_c_ub_offset = c_idx * tdc.ceil_fill(cr_pln_size, ele_per_block) + cl_idx * vnc_col_len
                        _adjust_tail_block(cl_c_ub_offset, cr_pln_size)
            with tik_inst.elif_scope(tik.all(tiling_mode == 2001, cr_pln_size % ele_per_block > 0)):
                with tik_inst.for_range(0, cl_c_size) as clc_idx:
                    _adjust_tail_block(clc_idx * tdc.ceil_fill(cr_pln_size, c0_len), cr_pln_size)


def _update_input_offset_all_dims_one(args):
    """
    count input gm offset for c-left and c-right only have one dimension
    """

    (cr_lp_idx, dst_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, cl_lp_idx, dst_cl_lp_step_in,
     core_step_in, cr_backend, dst_cr_step_in, cl_backend, dst_cl_step_in, c_backend, src_c_step_in) = args

    in_offset = (cr_lp_idx * dst_cr_lp_step_in + c_lp_idx * src_c_lp_step_in +
                 cl_lp_idx * dst_cl_lp_step_in + core_step_in -
                 (cr_backend * dst_cr_step_in + cl_backend * dst_cl_step_in + c_backend * src_c_step_in))

    return in_offset


def _update_input_offset_cl_dims_two(args, cl_beg):
    """
    count input gm offset for c-left has two dimensions
    """

    (cr_lp_idx, dst_cr_lp_step_in, cr_backend, dst_cr_step_in, c_lp_idx, src_c_lp_step_in, c_backend, src_c_step_in,
     core_step_in, cl_in_idx_0_size, cl_in_idx_0_dst_rsize, cl_in_idx_0_src_asize,
     cl_in_idx_1_size, cl_in_idx_1_dst_rsize, cl_in_idx_1_src_asize) = args

    in_offset = (cr_lp_idx * dst_cr_lp_step_in - cr_backend * dst_cr_step_in +
                 c_lp_idx * src_c_lp_step_in - c_backend * src_c_step_in +
                 cl_beg // cl_in_idx_0_dst_rsize % cl_in_idx_0_size * cl_in_idx_0_src_asize +
                 cl_beg // cl_in_idx_1_dst_rsize % cl_in_idx_1_size * cl_in_idx_1_src_asize + core_step_in)

    return in_offset


def _update_input_offset_cr_dims_two(args, cr_beg):
    """
    count input gm offset for c-right has two dimensions
    """

    (cl_lp_idx, dst_cl_lp_step_in, cl_backend, dst_cl_step_in, c_lp_idx, src_c_lp_step_in, c_backend, src_c_step_in,
     core_step_in, cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize,
     cr_in_idx_1_size, cr_in_idx_1_dst_rsize, cr_in_idx_1_src_asize) = args

    in_offset = (cl_lp_idx * dst_cl_lp_step_in - cl_backend * dst_cl_step_in +
                 c_lp_idx * src_c_lp_step_in - c_backend * src_c_step_in +
                 cr_beg // cr_in_idx_0_dst_rsize % cr_in_idx_0_size * cr_in_idx_0_src_asize +
                 cr_beg // cr_in_idx_1_dst_rsize % cr_in_idx_1_size * cr_in_idx_1_src_asize + core_step_in)

    return in_offset


def _update_output_offset(args):
    """
    count output gm offset
    """

    (cl_lp_idx, dst_cl_lp_step_out, cl_backend, dst_cl_step_out, c_lp_idx, src_c_lp_step_out, cr_lp_idx,
     dst_cr_lp_step_out, cr_backend, dst_cr_step_out, core_step_out, c_backend, src_c_step_out, block_idx) = args

    out_offset = (cl_lp_idx * dst_cl_lp_step_out - cl_backend * dst_cl_step_out +
                  cr_lp_idx * dst_cr_lp_step_out - cr_backend * dst_cr_step_out +
                  c_lp_idx * src_c_lp_step_out - c_backend * src_c_step_out + block_idx * core_step_out)

    return out_offset


def _move_data_in_cr_cl_one_dims(args):
    """
    move data in process when c-right or c-right only has one dimensions
    """

    (tik_inst, src_in_gm, cl_cr_dims_gm_offset, src_ub, cl_cr_dims_ub_offset,
     c_plp_size, src_c_step_in, cr_pln_size, c_cr_gap, c0_len, ele_per_block) = args

    with tik_inst.new_stmt_scope(disable_sync=True):
        with tik_inst.if_scope(c_cr_gap <= tdc.STRIDE_LIMIT_MTE):
            tik_inst.data_move(src_ub[cl_cr_dims_ub_offset], src_in_gm[cl_cr_dims_gm_offset],
                               0, c_plp_size, cr_pln_size * c0_len // ele_per_block, c_cr_gap, 0)
        with tik_inst.else_scope():
            with tik_inst.for_range(0, c_plp_size) as c_idx:
                cl_cr_dims_one_ub_offset = c_idx * cr_pln_size * c0_len + cl_cr_dims_ub_offset
                cl_cr_dims_one_gm_offset = c_idx * src_c_step_in + cl_cr_dims_gm_offset
                tik_inst.data_move(src_ub[cl_cr_dims_one_ub_offset], src_in_gm[cl_cr_dims_one_gm_offset],
                                   0, 1, cr_pln_size * c0_len // ele_per_block, 0, 0)


def _move_data_in_c1_is_one(args):
    """
    move data in process when c1 is one
    """

    (tik_inst, src_in_gm, base_gm_offset, src_ub, cl_plp_size, dst_cl_step_in,
     dst_cr_step_in, cr_pln_size, c0_len, ele_per_block, vnc_col_size, cl_cr_gap) = args
    cl_crc_gap = (vnc_col_size - cr_pln_size*dst_cr_step_in) // ele_per_block

    with tik_inst.new_stmt_scope(disable_sync=True):
        with tik_inst.if_scope(cl_cr_gap <= tdc.STRIDE_LIMIT_MTE):
            tik_inst.data_move(src_ub, src_in_gm[base_gm_offset],
                               0, cl_plp_size, cr_pln_size * c0_len // ele_per_block, cl_cr_gap, cl_crc_gap)
        with tik_inst.else_scope():
            with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                cl_cr_dims_one_ub_offset = cl_idx * vnc_col_size
                cl_cr_dims_one_gm_offset = cl_idx * dst_cl_step_in + base_gm_offset
                tik_inst.data_move(src_ub[cl_cr_dims_one_ub_offset], src_in_gm[cl_cr_dims_one_gm_offset],
                                   0, 1, cr_pln_size * c0_len // ele_per_block, 0, 0)


def _split_cr(args):
    """
    split c-right dimensions into three parts when it has two dimensions
    """

    tik_inst, cr_in_idx_0_size, cr_beg, left_dims_size, mid_lp_cnt, right_dims_size, cr_pln_size = args
    next_cr_gap = cr_in_idx_0_size - cr_beg % cr_in_idx_0_size
    with tik_inst.if_scope(next_cr_gap == cr_in_idx_0_size):
        left_dims_size.set_as(0)
    with tik_inst.else_scope():
        with tik_inst.if_scope(next_cr_gap <= cr_pln_size):
            left_dims_size.set_as(next_cr_gap)
        with tik_inst.else_scope():
            left_dims_size.set_as(cr_pln_size)
    mid_lp_cnt.set_as((cr_pln_size - left_dims_size) // cr_in_idx_0_size)
    right_dims_size.set_as(cr_pln_size - left_dims_size - mid_lp_cnt * cr_in_idx_0_size)


def _move_cr_in_for_two_cr_dims(args):
    """
    move c-right in first in process when c-right has two dimensions
    """

    (tik_inst, src_in_gm, src_ub, left_dims_size, mid_lp_cnt, cr1_cr0_gap, right_dims_size,
     dst_cl_step_in, cl_plp_size, c_plp_size, src_c_step_in, cr_pln_size, base_in_offset,
     cr_in_idx_0_size, cr_in_idx_1_src_asize, c0_len, ele_per_block, is_left_dims_nz, vnc_col_size) = args

    with tik_inst.if_scope(left_dims_size > 0):
        is_left_dims_nz.set_as(1)  # read data from next cr_1
        left_c_cr_gap = (src_c_step_in - left_dims_size * c0_len) // ele_per_block
        left_cr_gap = (cr_pln_size - left_dims_size) * c0_len // ele_per_block
        with tik_inst.for_range(0, cl_plp_size) as cl_idx:
            with tik_inst.if_scope(left_c_cr_gap > tdc.STRIDE_LIMIT_MTE):
                with tik_inst.for_range(0, c_plp_size) as c_idx:
                    left_dims_gm_offset = (cl_idx * dst_cl_step_in + c_idx * src_c_step_in + base_in_offset)
                    left_dims_ub_offset = cl_idx * vnc_col_size + c_idx * cr_pln_size * c0_len
                    tik_inst.data_move(src_ub[left_dims_ub_offset], src_in_gm[left_dims_gm_offset],
                                       0, 1, left_dims_size * c0_len // ele_per_block, 0, 0)
            with tik_inst.else_scope():
                left_dims_gm_offset = (cl_idx * dst_cl_step_in + base_in_offset)
                left_dims_ub_offset = cl_idx * vnc_col_size
                tik_inst.data_move(src_ub[left_dims_ub_offset], src_in_gm[left_dims_gm_offset],
                                   0, c_plp_size, left_dims_size * c0_len // ele_per_block, left_c_cr_gap, left_cr_gap)
    with tik_inst.else_scope():
        is_left_dims_nz.set_as(0)
    left_gm_offset = is_left_dims_nz * (left_dims_size * c0_len - cr_in_idx_0_size * c0_len + cr_in_idx_1_src_asize)

    bust_len_scalar = tik_inst.Scalar(name="bust_len_scalar")
    with tik_inst.if_scope(cr_in_idx_0_size * c0_len // ele_per_block > BURST_LIMIT):
        bust_len_scalar.set_as(1)
    with tik_inst.else_scope():
        bust_len_scalar.set_as(cr_in_idx_0_size * c0_len // ele_per_block)

    with tik_inst.if_scope(mid_lp_cnt > 0):
        cr1_cr0_gap.set_as((cr_in_idx_1_src_asize - cr_in_idx_0_size * c0_len) // ele_per_block)
        with tik_inst.if_scope(cr1_cr0_gap < 0):  # to avoid compile error
            cr1_cr0_gap.set_as(0)

        with tik_inst.for_range(0, cl_plp_size) as cl_idx:
            with tik_inst.for_range(0, c_plp_size) as c_idx:
                mid_lp_gm_offset = (cl_idx * dst_cl_step_in + c_idx * src_c_step_in + left_gm_offset + base_in_offset)
                mid_lp_ub_offset = (cl_idx * vnc_col_size + (c_idx * cr_pln_size + left_dims_size) * c0_len)
                with tik_inst.if_scope(cr1_cr0_gap > tdc.STRIDE_LIMIT_MTE):
                    with tik_inst.for_range(0, mid_lp_cnt) as mid_idx:
                        tik_inst.data_move(src_ub[mid_lp_ub_offset + mid_idx * cr_in_idx_0_size * c0_len],
                                           src_in_gm[mid_lp_gm_offset + mid_idx * cr_in_idx_1_src_asize],
                                           0, 1, bust_len_scalar, 0, 0)
                with tik_inst.else_scope():
                    tik_inst.data_move(src_ub[mid_lp_ub_offset], src_in_gm[mid_lp_gm_offset],
                                       0, mid_lp_cnt, bust_len_scalar, cr1_cr0_gap, 0)

    with tik_inst.if_scope(right_dims_size > 0):
        right_c_cr_gap = (src_c_step_in - right_dims_size * c0_len) // ele_per_block
        right_cr_gap = (cr_pln_size - right_dims_size) * c0_len // ele_per_block
        with tik_inst.for_range(0, cl_plp_size) as cl_idx:
            with tik_inst.if_scope(right_c_cr_gap > tdc.STRIDE_LIMIT_MTE):
                with tik_inst.for_range(0, c_plp_size) as c_idx:
                    right_dims_gm_offset = (cl_idx * dst_cl_step_in + c_idx * src_c_step_in +
                                            left_gm_offset + mid_lp_cnt * cr_in_idx_1_src_asize + base_in_offset)
                    right_dims_ub_offset = (cl_idx * vnc_col_size + (c_idx * cr_pln_size +
                                            left_dims_size + mid_lp_cnt * cr_in_idx_0_size) * c0_len)
                    tik_inst.data_move(src_ub[right_dims_ub_offset], src_in_gm[right_dims_gm_offset], 0, 1,
                                       right_dims_size * c0_len // ele_per_block, 0, 0)
            with tik_inst.else_scope():
                right_dims_gm_offset = (cl_idx*dst_cl_step_in + left_gm_offset +
                                        mid_lp_cnt*cr_in_idx_1_src_asize + base_in_offset)
                right_dims_ub_offset = cl_idx*vnc_col_size + (left_dims_size + mid_lp_cnt*cr_in_idx_0_size)*c0_len
                tik_inst.data_move(src_ub[right_dims_ub_offset], src_in_gm[right_dims_gm_offset], 0, c_plp_size,
                                   right_dims_size * c0_len // ele_per_block, right_c_cr_gap, right_cr_gap)


# 'pylint: disable=unused-variable
def _move_cl_in_for_two_cr_dims(args):
    """
    move c-left in first in process when c-right has two dimensions
    """

    (tik_inst, src_in_gm, src_ub, dst_ub, left_dims_size, mid_lp_cnt, right_dims_size, dst_cl_step_in, cl_plp_size,
     c_plp_size, src_c_step_in, cr_pln_size, base_in_offset, cr_in_idx_0_size, cr_in_idx_0_src_asize,
     cr_in_idx_1_src_asize, c0_len, ele_per_block, tmp_cr_size, is_left_dims_nz, vnc_col_size) = args
    cr_cl_gap = (cr_in_idx_0_src_asize - cl_plp_size * c0_len) // ele_per_block

    def _inner_process(inner_args):
        sub_cr_size, gm_base_offset, ub_base_offset = inner_args
        with tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.if_scope(cr_cl_gap > tdc.STRIDE_LIMIT_MTE):
                with tik_inst.for_range(0, sub_cr_size) as cr_idx:
                    dims_gm_offset = cr_idx * cr_in_idx_0_src_asize + gm_base_offset
                    dims_ub_offset = cr_idx * cl_plp_size * c0_len + ub_base_offset
                    tik_inst.data_move(dst_ub[dims_ub_offset], src_in_gm[dims_gm_offset],
                                       0, 1, cl_plp_size * c0_len // ele_per_block, 0, 0)
            with tik_inst.else_scope():
                dims_gm_offset = gm_base_offset
                dims_ub_offset = ub_base_offset
                tik_inst.data_move(dst_ub[dims_ub_offset], src_in_gm[dims_gm_offset],
                                   0, sub_cr_size, cl_plp_size * c0_len // ele_per_block, cr_cl_gap, 0)

    with tik_inst.if_scope(left_dims_size > 0):
        is_left_dims_nz.set_as(1)  # read data from next cr_1
        with tik_inst.for_range(0, c_plp_size) as c_idx:
            left_gm_offset = c_idx * src_c_step_in + base_in_offset
            left_ub_offset = c_idx * cr_pln_size * cl_plp_size * c0_len
            left_args = (left_dims_size, left_gm_offset, left_ub_offset)
            _inner_process(left_args)
    with tik_inst.else_scope():
        is_left_dims_nz.set_as(0)
    left_gm_offset = is_left_dims_nz * ((left_dims_size - cr_in_idx_0_size) * cr_in_idx_0_src_asize +
                                        cr_in_idx_1_src_asize)

    with tik_inst.if_scope(mid_lp_cnt > 0):
        with tik_inst.if_scope(cr_in_idx_0_size > tdc.REPEAT_LIMIT_MTE):  # to avoid compile error
            tmp_cr_size.set_as(1)
        with tik_inst.else_scope():
            tmp_cr_size.set_as(cr_in_idx_0_size)
        with tik_inst.for_range(0, c_plp_size) as c_idx:
            with tik_inst.for_range(0, mid_lp_cnt) as mid_idx:
                mid_gm_offset = (mid_idx * cr_in_idx_1_src_asize + left_gm_offset +
                                 c_idx * src_c_step_in + base_in_offset)
                mid_ub_offset = ((c_idx * cr_pln_size + mid_idx * cr_in_idx_0_size +
                                 left_dims_size) * cl_plp_size * c0_len)
                mid_args = (tmp_cr_size, mid_gm_offset, mid_ub_offset)
                _inner_process(mid_args)
    with tik_inst.if_scope(right_dims_size > 0):
        with tik_inst.for_range(0, c_plp_size) as c_idx:
            right_gm_offset = (mid_lp_cnt * cr_in_idx_1_src_asize + left_gm_offset +
                               c_idx * src_c_step_in + base_in_offset)
            right_ub_offset = ((c_idx * cr_pln_size + mid_lp_cnt * cr_in_idx_0_size +
                               left_dims_size) * cl_plp_size * c0_len)
            right_args = (right_dims_size, right_gm_offset, right_ub_offset)
            _inner_process(right_args)

    # reorder chnt -> ncht
    sub_c_cr_size = c_plp_size * cr_pln_size
    with tik_inst.new_stmt_scope(disable_sync=True):
        with tik_inst.for_range(0, cl_plp_size) as cl_idx:
            tik_inst.data_move(src_ub[cl_idx * vnc_col_size], dst_ub[cl_idx * c0_len], 0,
                               sub_c_cr_size, c0_len // ele_per_block, (cl_plp_size - 1) * c0_len // ele_per_block, 0)


# 'pylint: disable=unused-variable
def _copy_data_in_0(in_offset_args, tik_args):
    """
    copy data from gm to ub for transform such as nc1hwc0 -> nchw
    """

    (cr_lp_idx, dst_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, cl_lp_idx, dst_cl_lp_step_in, core_step_in,
     cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize, cr_in_idx_1_size, cr_in_idx_1_dst_rsize,
     cr_in_idx_1_src_asize, cl_in_idx_0_size, cl_in_idx_0_dst_rsize, cl_in_idx_0_src_asize,
     cl_in_idx_1_size, cl_in_idx_1_dst_rsize, cl_in_idx_1_src_asize) = in_offset_args
    (tik_inst, src_in_gm, src_ub, dst_cr_step_in, cr_pln_size, nlc_cr_lp_cnt, dst_cr_lp_unit, cr_backend,
     dst_cr_dims, src_c_step_in, c_plp_size, c_backend, dst_cl_step_in, cl_plp_size, nlc_cl_lp_cnt, dst_cl_lp_unit,
     cl_backend, dst_cl_dims, ele_per_block, c0_len, is_mc_cl, is_mc_cr, block_idx, vnc_col_size, c_lp_cnt, mc_pos,
     tiling_mode) = tik_args

    with tik_inst.new_stmt_scope():
        cl_beg = tik_inst.Scalar(name="cl_beg")
        cr_beg = tik_inst.Scalar(name="cr_beg")
        is_left_dims_nz = tik_inst.Scalar(name="is_left_dims_nz")
        left_dims_size = tik_inst.Scalar(name="left_dims_size")
        mid_lp_cnt = tik_inst.Scalar(name="mid_lp_cnt")
        right_dims_size = tik_inst.Scalar(name="right_dims_size")
        cr1_cr0_gap = tik_inst.Scalar(name="cr1_cr0_gap")
        c_cr_gap = (src_c_step_in - cr_pln_size * dst_cr_step_in) // ele_per_block

        with tik_inst.if_scope(tik.all(dst_cr_dims == 1, dst_cl_dims == 1)):  # such as NC1HWC0 -> NCHW
            offset_args = (cr_lp_idx, dst_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, cl_lp_idx,
                           dst_cl_lp_step_in, core_step_in, cr_backend, dst_cr_step_in, cl_backend, dst_cl_step_in,
                           c_backend, src_c_step_in)
            base_gm_offset = _update_input_offset_all_dims_one(offset_args)

            with tik_inst.if_scope(tik.all(c_plp_size == 1, c_lp_cnt == 1, mc_pos != 1)):  # c1 is 1
                cl_cr_gap = (dst_cl_step_in - cr_pln_size*dst_cr_step_in) // ele_per_block
                data_in_args = (tik_inst, src_in_gm, base_gm_offset, src_ub, cl_plp_size, dst_cl_step_in,
                                dst_cr_step_in, cr_pln_size, c0_len, ele_per_block, vnc_col_size, cl_cr_gap)
                _move_data_in_c1_is_one(data_in_args)
            with tik_inst.else_scope():  # c is not 1
                with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                    cl_cr_dims_ub_offset = cl_idx * vnc_col_size
                    cl_cr_dims_gm_offset = cl_idx * dst_cl_step_in + base_gm_offset
                    data_in_args = (tik_inst, src_in_gm, cl_cr_dims_gm_offset, src_ub, cl_cr_dims_ub_offset,
                                    c_plp_size, src_c_step_in, cr_pln_size, c_cr_gap, c0_len, ele_per_block)
                    _move_data_in_cr_cl_one_dims(data_in_args)

        with tik_inst.else_scope():
            with tik_inst.if_scope(dst_cl_dims == 2):  # dst_cr_dims is 1 and dst_cl_dims is 2
                with tik_inst.if_scope(is_mc_cl == 1):
                    cl_beg.set_as(cl_lp_idx * dst_cl_lp_unit + block_idx * nlc_cl_lp_cnt * dst_cl_lp_unit - cl_backend)
                with tik_inst.else_scope():
                    cl_beg.set_as(cl_lp_idx * dst_cl_lp_unit - cl_backend)
                offset_args = (cr_lp_idx, dst_cr_lp_step_in, cr_backend, dst_cr_step_in, c_lp_idx, src_c_lp_step_in,
                               c_backend, src_c_step_in, core_step_in, cl_in_idx_0_size, cl_in_idx_0_dst_rsize,
                               cl_in_idx_0_src_asize, cl_in_idx_1_size, cl_in_idx_1_dst_rsize, cl_in_idx_1_src_asize)

                with tik_inst.if_scope(tik.all(c_plp_size == 1, c_lp_cnt == 1, mc_pos != 1)):  # c1 is 1
                    cl_0_cr_gap = (cl_in_idx_0_src_asize - cr_pln_size*dst_cr_step_in) // ele_per_block
                    cl_cr_dims_gm_offset = _update_input_offset_cl_dims_two(offset_args, cl_beg)
                    data_in_args = (tik_inst, src_in_gm, cl_cr_dims_gm_offset, src_ub, cl_plp_size,
                                    cl_in_idx_0_src_asize, dst_cr_step_in, cr_pln_size, c0_len, ele_per_block,
                                    vnc_col_size, cl_0_cr_gap)
                    _move_data_in_c1_is_one(data_in_args)
                with tik_inst.else_scope():  # c1 is not 1
                    with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                        cl_cr_dims_ub_offset = cl_idx * vnc_col_size
                        cl_cr_dims_gm_offset = _update_input_offset_cl_dims_two(offset_args, cl_beg + cl_idx)
                        data_in_args = (tik_inst, src_in_gm, cl_cr_dims_gm_offset, src_ub, cl_cr_dims_ub_offset,
                                        c_plp_size, src_c_step_in, cr_pln_size, c_cr_gap, c0_len, ele_per_block)
                        _move_data_in_cr_cl_one_dims(data_in_args)

            with tik_inst.else_scope():  # dst_cr_dims is 2 and dst_cl_dims is 1
                with tik_inst.if_scope(is_mc_cr == 1):
                    cr_beg.set_as(cr_lp_idx * dst_cr_lp_unit + block_idx * nlc_cr_lp_cnt * dst_cr_lp_unit - cr_backend)
                with tik_inst.else_scope():
                    cr_beg.set_as(cr_lp_idx * dst_cr_lp_unit - cr_backend)
                offset_args = (cl_lp_idx, dst_cl_lp_step_in, cl_backend, dst_cl_step_in, c_lp_idx, src_c_lp_step_in,
                               c_backend, src_c_step_in, core_step_in, cr_in_idx_0_size, cr_in_idx_0_dst_rsize,
                               cr_in_idx_0_src_asize, cr_in_idx_1_size, cr_in_idx_1_dst_rsize, cr_in_idx_1_src_asize)
                base_in_offset = _update_input_offset_cr_dims_two(offset_args, cr_beg)
                # split c-right into three parts
                split_args = (tik_inst, cr_in_idx_0_size, cr_beg,
                              left_dims_size, mid_lp_cnt, right_dims_size, cr_pln_size)
                _split_cr(split_args)
                # move data in
                data_in_args = (tik_inst, src_in_gm, src_ub, left_dims_size, mid_lp_cnt, cr1_cr0_gap, right_dims_size,
                                dst_cl_step_in, cl_plp_size, c_plp_size, src_c_step_in, cr_pln_size, base_in_offset,
                                cr_in_idx_0_size, cr_in_idx_1_src_asize, c0_len, ele_per_block, is_left_dims_nz,
                                vnc_col_size)
                with tik_inst.new_stmt_scope(disable_sync=True):
                    _move_cr_in_for_two_cr_dims(data_in_args)


# 'pylint: disable=unused-variable
def _copy_data_in_1(in_offset_args, tik_args):
    """
    copy data from gm to ub for transform such as fractal_z -> nchw
    """

    (cr_lp_idx, dst_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, cl_lp_idx, dst_cl_lp_step_in,
     core_step_in, cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize,
     cr_in_idx_1_size, cr_in_idx_1_dst_rsize, cr_in_idx_1_src_asize) = in_offset_args
    (tik_inst, src_in_gm, src_ub, dst_ub, dst_cr_step_in, cr_pln_size, nlc_cr_lp_cnt,
     dst_cr_lp_unit, cr_backend, dst_cr_dims, src_c_step_in, c_plp_size, c_backend, dst_cl_step_in,
     cl_plp_size, cl_backend, dst_cl_dims, ele_per_block, c0_len, is_mc_cr, block_idx, vnc_col_size) = tik_args

    with tik_inst.new_stmt_scope():
        cr_beg = tik_inst.Scalar(name="cr_beg")
        is_left_dims_nz = tik_inst.Scalar(name="is_left_dims_nz")
        left_dims_size = tik_inst.Scalar(name="left_dims_size")
        mid_lp_cnt = tik_inst.Scalar(name="mid_lp_cnt")
        right_dims_size = tik_inst.Scalar(name="right_dims_size")
        tmp_cr_size = tik_inst.Scalar()
        cr_cl_gap = (cr_in_idx_0_src_asize - cl_plp_size * dst_cl_step_in) // ele_per_block

        with tik_inst.if_scope(tik.all(dst_cr_dims == 1, dst_cl_dims == 1)):  # such as FRACTAL_Z -> NCHW
            offset_args = (cr_lp_idx, dst_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, cl_lp_idx,
                           dst_cl_lp_step_in, core_step_in, cr_backend, dst_cr_step_in, cl_backend, dst_cl_step_in,
                           c_backend, src_c_step_in)
            base_gm_offset = _update_input_offset_all_dims_one(offset_args)
            with tik_inst.for_range(0, c_plp_size) as c_idx:
                cl_cr_dims_ub_offset = c_idx * cr_pln_size * cl_plp_size * c0_len
                cl_cr_dims_gm_offset = c_idx * src_c_step_in + base_gm_offset
                data_in_args = (tik_inst, src_in_gm, cl_cr_dims_gm_offset, dst_ub, cl_cr_dims_ub_offset,
                                cr_pln_size, dst_cr_step_in, cl_plp_size, cr_cl_gap, c0_len, ele_per_block)
                _move_data_in_cr_cl_one_dims(data_in_args)

            with tik_inst.new_stmt_scope(disable_sync=True):  # do chnt -> ncht
                sub_c_cr_size = c_plp_size * cr_pln_size
                with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                    tik_inst.data_move(src_ub[cl_idx * vnc_col_size],
                                       dst_ub[cl_idx * c0_len], 0, sub_c_cr_size,
                                       c0_len // ele_per_block, (cl_plp_size - 1) * c0_len // ele_per_block, 0)

        with tik_inst.else_scope():  # dst_cr_dims is 2 and dst_cl_dims is 1
            with tik_inst.if_scope(dst_cr_dims == 2):
                with tik_inst.if_scope(is_mc_cr == 1):
                    cr_beg.set_as(cr_lp_idx * dst_cr_lp_unit + block_idx * nlc_cr_lp_cnt * dst_cr_lp_unit - cr_backend)
                with tik_inst.else_scope():
                    cr_beg.set_as(cr_lp_idx * dst_cr_lp_unit - cr_backend)
                offset_args = (cl_lp_idx, dst_cl_lp_step_in, cl_backend, dst_cl_step_in, c_lp_idx, src_c_lp_step_in,
                               c_backend, src_c_step_in, core_step_in, cr_in_idx_0_size, cr_in_idx_0_dst_rsize,
                               cr_in_idx_0_src_asize, cr_in_idx_1_size, cr_in_idx_1_dst_rsize, cr_in_idx_1_src_asize)
                base_in_offset = _update_input_offset_cr_dims_two(offset_args, cr_beg)
                # split c-right into three parts
                split_args = (tik_inst, cr_in_idx_0_size, cr_beg,
                              left_dims_size, mid_lp_cnt, right_dims_size, cr_pln_size)
                _split_cr(split_args)
                # move data in
                data_in_args = (tik_inst, src_in_gm, src_ub, dst_ub, left_dims_size, mid_lp_cnt, right_dims_size,
                                dst_cl_step_in, cl_plp_size, c_plp_size, src_c_step_in, cr_pln_size, base_in_offset,
                                cr_in_idx_0_size, cr_in_idx_0_src_asize, cr_in_idx_1_src_asize, c0_len, ele_per_block,
                                tmp_cr_size, is_left_dims_nz, vnc_col_size)
                _move_cl_in_for_two_cr_dims(data_in_args)


# 'pylint: disable=unused-variable
def _copy_data_out(copy_out_args):
    """
    copy data from ub to gm
    """

    (tik_inst, dst_out_gm, dst_ub, tiling_mode, src_c_step_out, dst_cl_step_out, cl_plp_size, cr_pln_size, c_plp_size,
     dst_cr_all_in, ele_per_block, c0_len, sub_c_size, vnc_col_size, vnc_col_len, cl_lp_cnt_2003) = copy_out_args

    with tik_inst.new_stmt_scope():
        cr_block_align_size = tik_inst.Scalar(name="cr_block_align_size")
        cl_ub_step = tik_inst.Scalar(name="cl_ub_step")

        with tik_inst.if_scope(tiling_mode == 2003):
            burst_len = cl_plp_size * sub_c_size * cr_pln_size
            burst_len_block = burst_len // ele_per_block
            with tik_inst.if_scope(tik.all(burst_len_block > 0, burst_len % ele_per_block > 0)):
                with tik_inst.if_scope(cl_lp_cnt_2003 > 1):
                    with tik_inst.for_range(0, cl_lp_cnt_2003 - 1) as cl_idx_2003:
                        tik_inst.data_move(dst_out_gm[cl_idx_2003 * burst_len], dst_ub[cl_idx_2003 * vnc_col_len],
                                           0, 1, tdc.ceil_div(burst_len, ele_per_block), 0, 0)
                    tik_inst.data_move(dst_out_gm[(cl_lp_cnt_2003 - 1)*burst_len],
                                       dst_ub[(cl_lp_cnt_2003 - 1)*vnc_col_len], 0, 1, burst_len_block, 0, 0)
                    tik_inst.data_move(dst_out_gm[cl_lp_cnt_2003*burst_len - ele_per_block],
                                       dst_ub[(cl_lp_cnt_2003 - 1)*vnc_col_len + burst_len_block*ele_per_block],
                                       0, 1, 1, 0, 0)
                with tik_inst.else_scope():
                    tik_inst.data_move(dst_out_gm, dst_ub, 0, 1, burst_len_block, 0, 0)
                    tik_inst.data_move(dst_out_gm[burst_len - ele_per_block], dst_ub[burst_len_block * ele_per_block],
                                       0, 1, 1, 0, 0)
            with tik_inst.else_scope():
                with tik_inst.if_scope(cl_lp_cnt_2003 > 1):
                    tik_inst.data_move(dst_out_gm, dst_ub, 0, cl_lp_cnt_2003, tdc.ceil_div(burst_len, ele_per_block),
                                       (vnc_col_len - burst_len) // ele_per_block, 0)
                with tik_inst.else_scope():
                    tik_inst.data_move(dst_out_gm, dst_ub, 0, 1, tdc.ceil_div(burst_len, ele_per_block), 0, 0)

        with tik_inst.else_scope():
            with tik_inst.if_scope(tiling_mode == 2001):
                cl_ub_step.set_as(c_plp_size * c0_len * tdc.ceil_fill(cr_pln_size, c0_len))
                cr_block_align_size.set_as(tdc.ceil_fill(cr_pln_size, c0_len))
            with tik_inst.else_scope():
                cl_ub_step.set_as(vnc_col_size)
                cr_block_align_size.set_as(tdc.ceil_fill(cr_pln_size, ele_per_block))

            with tik_inst.if_scope(tik.all(dst_cr_all_in == 1,
                                           tik.any(tiling_mode == 2002, cr_pln_size % c0_len == 0))):
                c_cr_size = sub_c_size * cr_pln_size
                cl_crc_gap = (dst_cl_step_out - c_cr_size) // ele_per_block
                with tik_inst.if_scope(c_cr_size % ele_per_block == 0):
                    with tik_inst.new_stmt_scope(disable_sync=True):
                        with tik_inst.if_scope(tik.all(dst_cl_step_out % ele_per_block == 0,
                                                       cl_crc_gap <= tdc.STRIDE_LIMIT_MTE)):
                            tik_inst.data_move(dst_out_gm, dst_ub, 0, cl_plp_size, c_cr_size // ele_per_block,
                                               (cl_ub_step - c_cr_size) // ele_per_block, cl_crc_gap)
                        with tik_inst.else_scope():
                            with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                                tik_inst.data_move(dst_out_gm[cl_idx * dst_cl_step_out], dst_ub[cl_idx * cl_ub_step],
                                                   0, 1, c_cr_size // ele_per_block, 0, 0)
                with tik_inst.else_scope():
                    with tik_inst.new_stmt_scope(disable_sync=True):
                        with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                            tmp_ub_offset = cl_idx * cl_ub_step
                            tmp_gm_offset = cl_idx * dst_cl_step_out
                            tik_inst.data_move(dst_out_gm[tmp_gm_offset], dst_ub[tmp_ub_offset],
                                               0, 1, c_cr_size // ele_per_block, 0, 0)
                    with tik_inst.new_stmt_scope(disable_sync=True):
                        with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                            tmp_ub_offset = cl_idx * cl_ub_step
                            tmp_gm_offset = cl_idx * dst_cl_step_out
                            tik_inst.data_move(dst_out_gm[tmp_gm_offset + c_cr_size - ele_per_block],
                                               dst_ub[tmp_ub_offset + c_cr_size // ele_per_block * ele_per_block],
                                               0, 1, 1, 0, 0)

            with tik_inst.else_scope():
                with tik_inst.new_stmt_scope(disable_sync=True):
                    with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                        with tik_inst.for_range(0, sub_c_size) as c_idx:
                            tmp_ub_offset = cl_idx * cl_ub_step + c_idx * cr_block_align_size
                            tmp_gm_offset = cl_idx * dst_cl_step_out + c_idx * src_c_step_out
                            tik_inst.data_move(dst_out_gm[tmp_gm_offset], dst_ub[tmp_ub_offset],
                                               0, 1, cr_pln_size // ele_per_block, 0, 0)
                with tik_inst.new_stmt_scope(disable_sync=True):
                    with tik_inst.if_scope(cr_pln_size % ele_per_block > 0):
                        with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                            with tik_inst.for_range(0, sub_c_size) as c_idx:
                                tmp_ub_offset = cl_idx * cl_ub_step + c_idx * cr_block_align_size
                                tmp_gm_offset = cl_idx * dst_cl_step_out + c_idx * src_c_step_out
                                tik_inst.data_move(dst_out_gm[tmp_gm_offset + cr_pln_size - ele_per_block],
                                                   dst_ub[tmp_ub_offset + cr_pln_size // ele_per_block * ele_per_block],
                                                   0, 1, 1, 0, 0)


# 'pylint: disable=unused-variable
def _copy_data_out_with_dmp(copy_out_args):
    """
    copy data from ub to gm with data_move_pad
    """

    (tik_inst, dst_out_gm, dst_ub, tiling_mode, src_c_step_out, dst_cl_step_out, cl_plp_size, cr_pln_size, c_plp_size,
     dst_cr_all_in, ele_per_block, c0_len, sub_c_size, vnc_col_size, vnc_col_len, cl_lp_cnt_2003) = copy_out_args
    b8_times = tdc.BLOCK_BYTE_SIZE // ele_per_block
    max_dst_strided = 2**31 - 1

    with tik_inst.new_stmt_scope(disable_sync=True):
        with tik_inst.if_scope(tiling_mode == 2003):
            burst_len = cl_plp_size * sub_c_size * cr_pln_size
            burst_len_align = tdc.ceil_fill(burst_len, ele_per_block)
            with tik_inst.if_scope(cl_lp_cnt_2003 > 1):
                tik_inst.data_move_pad(dst_out_gm, dst_ub, cl_lp_cnt_2003, burst_len * b8_times,
                                       0, (vnc_col_len - burst_len_align) // ele_per_block)
            with tik_inst.else_scope():
                tik_inst.data_move_pad(dst_out_gm, dst_ub, 1, burst_len * b8_times, 0, 0)
        with tik_inst.else_scope():
            cr_block_align_size = tik_inst.Scalar(name="cr_block_align_size")
            cl_ub_step = tik_inst.Scalar(name="cl_ub_step")
            with tik_inst.if_scope(tiling_mode == 2001):
                cr_align_c0_size = tdc.ceil_fill(cr_pln_size, c0_len)
                cl_ub_step.set_as(c_plp_size * c0_len * cr_align_c0_size)
                cr_block_align_size.set_as(cr_align_c0_size)
            with tik_inst.else_scope():
                cl_ub_step.set_as(vnc_col_size)
                cr_block_align_size.set_as(tdc.ceil_fill(cr_pln_size, ele_per_block))

            with tik_inst.if_scope(tik.all(dst_cr_all_in == 1,
                                           tik.any(tiling_mode == 2002, cr_pln_size % c0_len == 0))):
                c_cr_size = sub_c_size * cr_pln_size
                cl_crc_gap = (dst_cl_step_out - c_cr_size) * b8_times
                src_ub_stride = (cl_ub_step - tdc.ceil_fill(c_cr_size, ele_per_block)) // ele_per_block
                with tik_inst.if_scope(cl_crc_gap <= max_dst_strided):
                    tik_inst.data_move_pad(dst_out_gm, dst_ub, cl_plp_size, c_cr_size * b8_times,
                                           cl_crc_gap, src_ub_stride)
                with tik_inst.else_scope():
                    with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                        tik_inst.data_move_pad(dst_out_gm[cl_idx * dst_cl_step_out], dst_ub[cl_idx * cl_ub_step],
                                               1, c_cr_size * b8_times, 0, 0)
            with tik_inst.else_scope():
                src_ub_stride = (cr_block_align_size - tdc.ceil_fill(cr_pln_size, ele_per_block)) // ele_per_block
                dst_gm_stride = (src_c_step_out - cr_pln_size) * b8_times
                with tik_inst.if_scope(dst_gm_stride <= max_dst_strided):
                    with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                        tmp_ub_offset = cl_idx * cl_ub_step
                        tmp_gm_offset = cl_idx * dst_cl_step_out
                        tik_inst.data_move_pad(dst_out_gm[tmp_gm_offset], dst_ub[tmp_ub_offset],
                                               sub_c_size, cr_pln_size * b8_times, dst_gm_stride, src_ub_stride)
                with tik_inst.else_scope():
                    with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                        with tik_inst.for_range(0, sub_c_size) as c_idx:
                            tmp_ub_offset = cl_idx * cl_ub_step + c_idx * cr_block_align_size
                            tmp_gm_offset = cl_idx * dst_cl_step_out + c_idx * src_c_step_out
                            tik_inst.data_move_pad(dst_out_gm[tmp_gm_offset], dst_ub[tmp_ub_offset],
                                                   1, cr_pln_size * b8_times, 0, 0)


def _get_backend_idx(args):
    """
    get backend index for each axis
    """

    (tik_inst, used_core_cnt, block_idx, mc_pos, is_mc_cr, lc_cr_lp_cnt, lc_cr_left, cr_lp_cnt, cr_left,
     cr_backend_idx, lc_c_lp_cnt, lc_c_left, c_lp_cnt, c_left, c_backend_idx, is_mc_cl, lc_cl_lp_cnt,
     left_cl_c_cr_size, cl_lp_cnt, cl_left, cl_backend_idx, src_c_step_out, c_mod_c0, ele_per_block) = args

    with tik_inst.if_scope(tik.all(block_idx == used_core_cnt - 2, lc_cr_lp_cnt == 1, is_mc_cr == 1,
                                   lc_cr_left > 0, lc_cr_left < ele_per_block)):
        cr_backend_idx.set_as(cr_lp_cnt - 1)
    with tik_inst.elif_scope(tik.any(tik.all(block_idx == used_core_cnt - 1, is_mc_cr == 1,
                                             lc_cr_left > 0, lc_cr_left < ele_per_block),
                                     tik.all(is_mc_cr != 1, lc_cr_left > 0, lc_cr_left < ele_per_block))):
        cr_backend_idx.set_as(cr_lp_cnt - 2)
    with tik_inst.elif_scope(cr_left > 0):
        cr_backend_idx.set_as(cr_lp_cnt - 1)
    with tik_inst.else_scope():
        cr_backend_idx.set_as(cr_lp_cnt)

    with tik_inst.if_scope(tik.all(block_idx == used_core_cnt - 2, lc_c_lp_cnt == 1, mc_pos == 1,
                                   lc_c_left == 1, c_mod_c0 > 0, c_mod_c0 * src_c_step_out < ele_per_block)):
        c_backend_idx.set_as(c_lp_cnt - 1)
    with tik_inst.elif_scope(tik.any(tik.all(block_idx == used_core_cnt - 1, mc_pos == 1, lc_c_left == 1,
                                             c_mod_c0 > 0, c_mod_c0 * src_c_step_out < ele_per_block),
                                     tik.all(mc_pos != 1, lc_c_left == 1, c_mod_c0 > 0,
                                             c_mod_c0 * src_c_step_out < ele_per_block))):
        c_backend_idx.set_as(c_lp_cnt - 2)
    with tik_inst.elif_scope(c_left > 0):
        c_backend_idx.set_as(c_lp_cnt - 1)
    with tik_inst.else_scope():
        c_backend_idx.set_as(c_lp_cnt)

    with tik_inst.if_scope(tik.all(block_idx == used_core_cnt - 2, lc_cl_lp_cnt == 1, is_mc_cl == 1,
                                   left_cl_c_cr_size > 0, left_cl_c_cr_size < ele_per_block)):
        cl_backend_idx.set_as(cl_lp_cnt - 1)
    with tik_inst.elif_scope(tik.any(tik.all(block_idx == used_core_cnt - 1, is_mc_cl == 1,
                                             left_cl_c_cr_size > 0, left_cl_c_cr_size < ele_per_block),
                                     tik.all(is_mc_cl != 1, left_cl_c_cr_size > 0,
                                             left_cl_c_cr_size < ele_per_block))):
        cl_backend_idx.set_as(cl_lp_cnt - 2)
    with tik_inst.elif_scope(cl_left > 0):
        cl_backend_idx.set_as(cl_lp_cnt - 1)
    with tik_inst.else_scope():
        cl_backend_idx.set_as(cl_lp_cnt)


def _set_main_cr_pln_size(args):
    (tik_inst, cr_lp_idx, cr_backend_idx, cr_pln_size, dst_cr_lp_unit, ele_per_block, is_cr_back, dmp_flag) = args
    if dmp_flag is False:
        with tik_inst.if_scope(cr_lp_idx == cr_backend_idx):
            cr_pln_size.set_as(dst_cr_lp_unit - ele_per_block)
        with tik_inst.else_scope():
            cr_pln_size.set_as(dst_cr_lp_unit)
    else:
        cr_pln_size.set_as(dst_cr_lp_unit)
    is_cr_back.set_as(0)


def _set_tail_cr_pln_size(args):
    (tik_inst, used_core_cnt, cr_lp_cnt, cr_left, ele_per_block, cr_pln_size, is_cr_back, dmp_flag) = args
    if dmp_flag is False:
        with tik_inst.if_scope(tik.all(tik.any(used_core_cnt > 1, cr_lp_cnt > 1),
                                       cr_left > 0, cr_left < ele_per_block)):
            cr_pln_size.set_as(cr_left + ele_per_block)
            is_cr_back.set_as(1)
        with tik_inst.else_scope():
            cr_pln_size.set_as(cr_left)
            is_cr_back.set_as(0)
    else:
        cr_pln_size.set_as(cr_left)


def _set_main_c_plp_size(args):
    (tik_inst, c_lp_idx, c_backend_idx, c_plp_size, src_c_lp_unit, is_c_back, dmp_flag) = args
    if dmp_flag is False:
        with tik_inst.if_scope(c_lp_idx == c_backend_idx):
            c_plp_size.set_as(src_c_lp_unit - 1)
        with tik_inst.else_scope():
            c_plp_size.set_as(src_c_lp_unit)
    else:
        c_plp_size.set_as(src_c_lp_unit)
    is_c_back.set_as(0)


def _set_tail_c_plp_size(args):
    (tik_inst, c_left, c_mod_c0, cr_pln_size, ele_per_block, c_plp_size, lc_c_left, is_c_back, dmp_flag) = args
    if dmp_flag is False:
        with tik_inst.if_scope(tik.all(c_left == 1, c_mod_c0 > 0,
                                       c_mod_c0 * cr_pln_size < ele_per_block)):
            c_plp_size.set_as(lc_c_left + 1)
            is_c_back.set_as(1)
        with tik_inst.else_scope():
            c_plp_size.set_as(lc_c_left)
            is_c_back.set_as(0)
    else:
        c_plp_size.set_as(lc_c_left)


def _set_main_cl_plp_size(args):
    (tik_inst, cl_lp_idx, cl_backend_idx, cl_plp_size, dst_cl_lp_unit, left_back, is_cl_back, dmp_flag) = args
    if dmp_flag is False:
        with tik_inst.if_scope(cl_lp_idx == cl_backend_idx):
            cl_plp_size.set_as(dst_cl_lp_unit - left_back)
        with tik_inst.else_scope():
            cl_plp_size.set_as(dst_cl_lp_unit)
    else:
        cl_plp_size.set_as(dst_cl_lp_unit)
    is_cl_back.set_as(0)


def _set_tail_cl_plp_size(args):
    (tik_inst, used_core_cnt, cl_lp_cnt, left_cl_c_cr_size, ele_per_block,
     cl_plp_size, cl_left, left_back, is_cl_back, dmp_flag) = args
    if dmp_flag is False:
        with tik_inst.if_scope(tik.all(tik.any(used_core_cnt > 1, cl_lp_cnt > 1),
                                       left_cl_c_cr_size > 0, left_cl_c_cr_size < ele_per_block)):
            cl_plp_size.set_as(cl_left + left_back)
            is_cl_back.set_as(1)
        with tik_inst.else_scope():
            cl_plp_size.set_as(cl_left)
            is_cl_back.set_as(0)
    else:
        cl_plp_size.set_as(cl_left)


def _get_cl_backend_idx(args):
    (tik_inst, cl_left, cl_lp_cnt, cl_backend_idx) = args
    with tik_inst.if_scope(cl_left > 0):
        cl_backend_idx.set_as(cl_lp_cnt - 1)
    with tik_inst.else_scope():
        cl_backend_idx.set_as(cl_lp_cnt)


def _func_transform_200(tensor_args, tp_args):
    """
    transform function for tiling mode 200
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, dst_ub, ele_per_block, in_dtype = tensor_args
    (tiling_mode, ub_offset, mc_pos, used_core_cnt, c0_len, core_step_in, core_step_out, nlc_cr_lp_cnt, nlc_c_lp_cnt,
     nlc_cl_lp_cnt, nlc_cr_left, nlc_c_left, nlc_cl_left, lc_cr_lp_cnt, lc_c_lp_cnt, lc_cl_lp_cnt, lc_cr_left,
     lc_c_left, lc_cl_left, dst_cr_lp_unit, src_c_lp_unit, dst_cl_lp_unit, vnc_col_len, dst_cr_step_in, dst_cr_step_out,
     dst_cr_lp_step_in, dst_cr_lp_step_out, dst_c_size, src_c_step_in, src_c_step_out, src_c_lp_step_in,
     src_c_lp_step_out, dst_cr_all_in, dst_cl_step_in, dst_cl_step_out, dst_cl_lp_step_in, dst_cl_lp_step_out, c_mod_c0,
     dst_cr_dims, dst_cl_dims, is_mc_cr, is_mc_cl, src_r2nd_dst_r1st_same, left_cl_c_cr_size, cl_in_idx_0_size,
     cl_in_idx_0_dst_rsize, cl_in_idx_0_src_asize, cl_in_idx_1_size, cl_in_idx_1_dst_rsize, cl_in_idx_1_src_asize,
     cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize, cr_in_idx_1_size, cr_in_idx_1_dst_rsize,
     cr_in_idx_1_src_asize) = tp_args

    def _inner_func(tiling_args):
        cr_lp_cnt, cr_left, c_lp_cnt, c_left, cl_lp_cnt, cl_left = tiling_args
        cr_pln_size = tik_inst.Scalar(name="cr_pln_size")
        c_plp_size = tik_inst.Scalar(name="c_plp_size")
        cl_plp_size = tik_inst.Scalar(name="cl_plp_size")
        is_last_c1 = tik_inst.Scalar(name="is_last_c1")
        is_cr_back = tik_inst.Scalar(name="is_cr_back", init_value=0)
        is_cl_back = tik_inst.Scalar(name="is_cl_back", init_value=0)
        is_c_back = tik_inst.Scalar(name="is_c_back", init_value=0)
        vnc_col_size = tik_inst.Scalar(name="vnc_col_size")
        sub_c_size = tik_inst.Scalar(name="sub_c_size")
        src_ub_offset = tik_inst.Scalar(name="src_ub_offset")
        cl_lp_cnt_2003 = tik_inst.Scalar(name="cl_lp_cnt_2003")
        cur_cl_lp_idx = tik_inst.Scalar(name="cur_cl_lp_idx")
        cr_backend_idx = tik_inst.Scalar(name="cr_backend_idx")
        c_backend_idx = tik_inst.Scalar(name="c_backend_idx")
        cl_backend_idx = tik_inst.Scalar(name="cl_backend_idx")
        dmp_flag = cce.api_check_support("tik.data_move_pad", in_dtype)

        if dmp_flag is False:
            backend_args = (tik_inst, used_core_cnt, block_idx, mc_pos, is_mc_cr, lc_cr_lp_cnt, lc_cr_left,
                            cr_lp_cnt, cr_left, cr_backend_idx, lc_c_lp_cnt, lc_c_left, c_lp_cnt, c_left,
                            c_backend_idx, is_mc_cl, lc_cl_lp_cnt, left_cl_c_cr_size, cl_lp_cnt, cl_left,
                            cl_backend_idx, src_c_step_out, c_mod_c0, ele_per_block)
            _get_backend_idx(backend_args)
        else:
            backend_args = (tik_inst, cl_left, cl_lp_cnt, cl_backend_idx)
            _get_cl_backend_idx(backend_args)

        with tik_inst.for_range(0, cr_lp_cnt) as cr_lp_idx:  # axis C-RIGHT
            with tik_inst.if_scope(tik.any(cr_lp_idx != cr_lp_cnt - 1, cr_left == 0)):
                # cr_lp_idx for last second core and lc_cr_left for last core
                cr_args = (tik_inst, cr_lp_idx, cr_backend_idx, cr_pln_size,
                           dst_cr_lp_unit, ele_per_block, is_cr_back, dmp_flag)
                _set_main_cr_pln_size(cr_args)
            with tik_inst.else_scope():
                cr_args = (tik_inst, used_core_cnt, cr_lp_cnt, cr_left, ele_per_block,
                           cr_pln_size, is_cr_back, dmp_flag)
                _set_tail_cr_pln_size(cr_args)
            cr_backend = is_cr_back * ele_per_block

            with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:  # axis C
                with tik_inst.if_scope(tik.any(c_lp_idx != c_lp_cnt - 1, c_left == 0)):
                    c_args = (tik_inst, c_lp_idx, c_backend_idx, c_plp_size, src_c_lp_unit, is_c_back, dmp_flag)
                    _set_main_c_plp_size(c_args)
                    with tik_inst.if_scope(tik.any(tik.all(c_lp_idx == c_lp_cnt - 1, mc_pos != 1),
                                                   tik.all(c_lp_idx == c_lp_cnt - 1, mc_pos == 1,
                                                           block_idx == used_core_cnt - 1))):
                        is_last_c1.set_as(1)
                    with tik_inst.else_scope():
                        is_last_c1.set_as(0)
                with tik_inst.else_scope():
                    c_args = (tik_inst, c_left, c_mod_c0, cr_pln_size, ele_per_block,
                              c_plp_size, lc_c_left, is_c_back, dmp_flag)
                    _set_tail_c_plp_size(c_args)
                    is_last_c1.set_as(1)
                c_backend = is_c_back
                with tik_inst.if_scope(tik.all(c_mod_c0 > 0, is_last_c1 > 0)):
                    sub_c_size.set_as((c_plp_size - 1) * c0_len + c_mod_c0)
                with tik_inst.else_scope():
                    sub_c_size.set_as(c_plp_size * c0_len)

                r2nd_c_size = cr_in_idx_0_size * cr_in_idx_1_size * sub_c_size
                left_back = tdc.ceil_div(ele_per_block, r2nd_c_size) - lc_cl_left
                with tik_inst.for_range(0, cl_lp_cnt) as cl_lp_idx:  # axis C-LEFT
                    with tik_inst.if_scope(tik.any(cl_lp_idx != cl_lp_cnt - 1, cl_left == 0)):
                        cl_args = (tik_inst, cl_lp_idx, cl_backend_idx, cl_plp_size, dst_cl_lp_unit,
                                   left_back, is_cl_back, dmp_flag)
                        _set_main_cl_plp_size(cl_args)
                    with tik_inst.else_scope():
                        cl_args = (tik_inst, used_core_cnt, cl_lp_cnt, left_cl_c_cr_size, ele_per_block,
                                   cl_plp_size, cl_left, left_back, is_cl_back, dmp_flag)
                        _set_tail_cl_plp_size(cl_args)
                    cl_backend = is_cl_back * left_back

                    with tik_inst.if_scope(tiling_mode == 2002):
                        vnc_col_size.set_as(vnc_col_len)
                    with tik_inst.else_scope():
                        vnc_col_size.set_as(c_plp_size * cr_pln_size * c0_len)

                    # in order to use 1-15 address for tiling mode 2003
                    with tik_inst.if_scope(tik.all(tiling_mode == 2003, cl_lp_idx < cl_backend_idx)):
                        src_ub_offset.set_as(cl_lp_idx % tdc.VNC_LINES * vnc_col_len)
                        cl_lp_cnt_2003.set_as(cl_lp_idx % tdc.VNC_LINES + 1)
                        cur_cl_lp_idx.set_as(cl_lp_idx - cl_lp_cnt_2003 + 1)
                    with tik_inst.else_scope():
                        src_ub_offset.set_as(0)
                        cl_lp_cnt_2003.set_as(0)
                        cur_cl_lp_idx.set_as(cl_lp_idx)

                    with tik_inst.if_scope(src_r2nd_dst_r1st_same == 1):  # such as NC1HWC0 -> NCHW
                        in_offset_args = (cr_lp_idx, dst_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, cl_lp_idx,
                                          dst_cl_lp_step_in, block_idx * core_step_in,
                                          cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize,
                                          cr_in_idx_1_size, cr_in_idx_1_dst_rsize, cr_in_idx_1_src_asize,
                                          cl_in_idx_0_size, cl_in_idx_0_dst_rsize, cl_in_idx_0_src_asize,
                                          cl_in_idx_1_size, cl_in_idx_1_dst_rsize, cl_in_idx_1_src_asize)
                        copy_in_args = (tik_inst, src_in_gm, src_ub[src_ub_offset:], dst_cr_step_in, cr_pln_size,
                                        nlc_cr_lp_cnt, dst_cr_lp_unit, cr_backend, dst_cr_dims, src_c_step_in,
                                        c_plp_size, c_backend, dst_cl_step_in, cl_plp_size, nlc_cl_lp_cnt,
                                        dst_cl_lp_unit, cl_backend, dst_cl_dims, ele_per_block, c0_len,
                                        is_mc_cl, is_mc_cr, block_idx, vnc_col_size, c_lp_cnt, mc_pos, tiling_mode)
                        _copy_data_in_0(in_offset_args, copy_in_args)
                    with tik_inst.else_scope():  # such as FRACTAL_Z -> NCHW
                        in_offset_args = (cr_lp_idx, dst_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, cl_lp_idx,
                                          dst_cl_lp_step_in, block_idx * core_step_in,
                                          cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize,
                                          cr_in_idx_1_size, cr_in_idx_1_dst_rsize, cr_in_idx_1_src_asize)
                        copy_in_args = (tik_inst, src_in_gm, src_ub[src_ub_offset:], dst_ub, dst_cr_step_in,
                                        cr_pln_size,
                                        nlc_cr_lp_cnt, dst_cr_lp_unit, cr_backend, dst_cr_dims, src_c_step_in,
                                        c_plp_size, c_backend, dst_cl_step_in, cl_plp_size, cl_backend, dst_cl_dims,
                                        ele_per_block, c0_len, is_mc_cr, block_idx, vnc_col_size)
                        _copy_data_in_1(in_offset_args, copy_in_args)

                    with tik_inst.if_scope(tik.any(cl_lp_idx == cl_backend_idx - 1,
                                                   cl_lp_cnt_2003 == 0, cl_lp_cnt_2003 == tdc.VNC_LINES)):
                        reorder_args = (tik_inst, src_ub, dst_ub, in_dtype, cl_plp_size, dst_c_size, c_plp_size,
                                        cr_lp_cnt, cr_pln_size, is_mc_cr, c0_len, ele_per_block, vnc_col_len,
                                        c_mod_c0, is_last_c1, sub_c_size, cl_lp_cnt_2003, dst_cr_all_in, dmp_flag)
                        _reorder_data(reorder_args, tiling_mode)
                        out_gm_args = (cur_cl_lp_idx, dst_cl_lp_step_out, cl_backend, dst_cl_step_out, c_lp_idx,
                                       src_c_lp_step_out, cr_lp_idx, dst_cr_lp_step_out, cr_backend, dst_cr_step_out,
                                       core_step_out, c_backend * c0_len, src_c_step_out, block_idx)
                        out_gm_offset = _update_output_offset(out_gm_args)
                        copy_out_args = (tik_inst, dst_out_gm[out_gm_offset], dst_ub, tiling_mode,
                                         src_c_step_out, dst_cl_step_out, cl_plp_size, cr_pln_size, c_plp_size,
                                         dst_cr_all_in, ele_per_block, c0_len, sub_c_size, vnc_col_size,
                                         vnc_col_len, cl_lp_cnt_2003)
                        if dmp_flag is False:
                            _copy_data_out(copy_out_args)
                        else:
                            _copy_data_out_with_dmp(copy_out_args)

    with tik_inst.if_scope(block_idx != used_core_cnt - 1):
        nlc_args = (nlc_cr_lp_cnt, nlc_cr_left, nlc_c_lp_cnt, nlc_c_left, nlc_cl_lp_cnt, nlc_cl_left)
        _inner_func(nlc_args)
    with tik_inst.else_scope():
        lc_args = (lc_cr_lp_cnt, lc_cr_left, lc_c_lp_cnt, lc_c_left, lc_cl_lp_cnt, lc_cl_left)
        _inner_func(lc_args)


def trans_data_negative_target_ntc(src, dst, src_format, dst_format, kernel_name="trans_data_negative_target_ntc"):
    """
    negative transform for last dimension of target format is c

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst : dict
        dtype of output should be same type as input
    src_format: str
        source data format, can be ND, NC1HWC0 etc.
    dst_format: str
        target data format, can be NHWC etc.
    groups: int
        groups count for conv case, default value is 1
    kernel_name : str
        kernel name, default value is "trans_data_negative_target_ntc"

    Returns
    -------
    None
    """

    src_format = src_format.upper()
    dst_format = dst_format.upper()
    in_shape = list(src.get("shape"))
    out_shape = list(dst.get("shape"))
    in_dtype = src.get("dtype").lower() if src.get("dtype").lower() != "bfloat16" else "float16"
    ub_size = tdc.get_max_element_in_ub(in_dtype, 1)
    block_elem_cnt = tdc.BLOCK_BYTE_SIZE // tdc.get_dtype_len(in_dtype)

    # get tiling parameters
    args = in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size, in_dtype
    tiling_params = _get_tiling_params_func(args)

    tik_inst = tik.Tik()
    src_in_gm = tik_inst.Tensor(in_dtype, in_shape, tik.scope_gm, "src_in_gm")
    dst_out_gm = tik_inst.Tensor(in_dtype, out_shape, tik.scope_gm, "dst_out_gm")
    half_ub = ub_size // 2
    src_ub = tik_inst.Tensor(in_dtype, (half_ub,), tik.scope_ubuf, "src_ub")
    dst_ub = tik_inst.Tensor(in_dtype, (half_ub,), tik.scope_ubuf, "dst_ub")

    used_core_cnt = tiling_params[3]
    with tik_inst.for_range(0, tdc.get_core_num(), block_num=tdc.get_core_num()) as block_idx:
        with tik_inst.if_scope(block_idx < used_core_cnt):
            tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, dst_ub, block_elem_cnt, in_dtype]
            tp_args = tiling_params
            _func_transform_200(tensor_args, tp_args)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[src_in_gm], outputs=[dst_out_gm],
                      config={"dynamic_tik": True, "out_of_bound_sync_check": True, "enable_s64_to_s32": True})
