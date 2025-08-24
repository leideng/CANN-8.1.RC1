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
trans_data_positive_source_ntc
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl import trans_data_common_func as tdc


# used for scalar
PAD_IDX_LIST = (0, 1)
# frame up levels
FRAME_LEVEL = 2


# 'pylint: disable=too-many-locals, inconsistent-return-statements, too-many-return-statements, too-many-statements
def _renew_input_output_shape_format(in_shape, out_shape, in_format, out_format):
    """
    renew shape and format to adapt tiling process
    """

    in_format_upper = in_format.upper()
    out_format_upper = out_format.upper()

    if in_format_upper == "NCDHW" and out_format_upper == "NDC1HWC0":
        in_format_new = "NCDH"
        out_format_new = "NDCHT"
        axis_n, axis_c, axis_d, axis_h, axis_w = in_shape
        axis_c0 = out_shape[-1]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        in_shape_new = [axis_n] + [axis_c] + [axis_d] + [axis_h * axis_w]
        out_shape_new = [axis_n] + [axis_d] + [axis_c1] + [axis_h * axis_w] + [axis_c0]
        new_params_ndc1hwc0 = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_ndc1hwc0

    if in_format_upper == "NCHW" and out_format_upper == "NC1HWC0":
        in_format_new = "NCH"
        out_format_new = "NCHT"
        axis_n, axis_c, axis_h, axis_w = in_shape
        axis_c0 = out_shape[-1]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        in_shape_new = [axis_n] + [axis_c] + [axis_h * axis_w]
        out_shape_new = [axis_n] + [axis_c1] + [axis_h * axis_w] + [axis_c0]
        new_params_nc1hwc0 = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_nc1hwc0

    if in_format_upper == "HWCN" and out_format_upper == "FRACTAL_Z":
        in_format_new = "HCN"
        out_format_new = "CHNT"
        axis_h, axis_w, axis_c, axis_n = in_shape
        axis_c0 = out_shape[-1]
        axis_ni = tdc.NI_16
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, axis_ni)
        in_shape_new = [axis_h * axis_w] + [axis_c] + [axis_n]
        out_shape_new = [axis_c1] + [axis_h * axis_w] + [axis_no * axis_ni] + [axis_c0]
        new_params_fz = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_fz

    if in_format_upper == "DHWCN" and out_format_upper == "FRACTAL_Z_3D":
        in_format_new = "DHCN"
        out_format_new = "DCHNT"
        axis_d, axis_h, axis_w, axis_c, axis_n = in_shape
        axis_c0 = out_shape[-1]
        axis_ni = tdc.NI_16
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, axis_ni)
        in_shape_new = [axis_d] + [axis_h * axis_w] + [axis_c] + [axis_n]
        out_shape_new = [axis_d] + [axis_c1] + [axis_h * axis_w] + [axis_no * axis_ni] + [axis_c0]
        new_params_fz = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_fz

    if in_format_upper == "ND" and out_format_upper == "FRACTAL_Z":
        in_format_new = "HCN"
        out_format_new = "HCNT"
        in_shape_len = len(in_shape)
        if in_shape_len == 1:
            axis_h, axis_c, axis_n = 1, 1, in_shape[0]
        elif in_shape_len == 2:
            axis_h, axis_c, axis_n = 1, in_shape[0], in_shape[1]
        else:
            axis_h, axis_c, axis_n = tdc.get_shape_size(in_shape[:-2]), in_shape[-2], in_shape[-1]
        axis_c0 = out_shape[-1]
        axis_ni = tdc.NI_16
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, axis_ni)
        in_shape_new = [axis_h] + [axis_c] + [axis_n]
        out_shape_new = [axis_h] + [axis_c1] + [axis_no * axis_ni] + [axis_c0]
        new_params_zn = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_zn

    if in_format_upper == "NCHW" and out_format_upper == "FRACTAL_Z":
        in_format_new = "NCH"
        out_format_new = "CHNT"
        axis_n, axis_c, axis_h, axis_w = in_shape
        axis_c0 = out_shape[-1]
        axis_ni = tdc.NI_16
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, axis_ni)
        in_shape_new = [axis_n] + [axis_c] + [axis_h * axis_w]
        out_shape_new = [axis_c1] + [axis_h * axis_w] + [axis_no * axis_ni] + [axis_c0]
        new_params_fz = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_fz

    if in_format_upper == "NCDHW" and out_format_upper == "FRACTAL_Z_3D":
        in_format_new = "NCDH"
        out_format_new = "DCHNT"
        axis_n, axis_c, axis_d, axis_h, axis_w = in_shape
        axis_c0 = out_shape[-1]
        axis_ni = tdc.NI_16
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, axis_ni)
        in_shape_new = [axis_n] + [axis_c] + [axis_d] + [axis_h * axis_w]
        out_shape_new = [axis_d] + [axis_c1] + [axis_h * axis_w] + [axis_no * axis_ni] + [axis_c0]
        new_params_z3d = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_z3d

    return [in_shape, out_shape] + [in_format, out_format]


# 'pylint: disable=too-many-locals, too-many-statements
def _get_mc_info_positive(cr_args, c_args, cl_args):
    """
    get multiple core axis position for positive transform
    """

    src_cr_lp_cnt, src_cr_size, src_cr_lp_unit, src_cr_lp_step_in, src_cr_lp_step_out = cr_args
    src_c_lp_cnt, src_c_size, src_c_lp_unit, src_c_lp_step_in, src_c_lp_step_out = c_args
    src_cl_lp_cnt, src_cl_size, src_cl_lp_unit, src_cl_lp_step_in, src_cl_lp_step_out = cl_args

    tmp_full_lp_cnt_cr = tdc.get_core_num() if tdc.floor_div(src_cr_lp_cnt, tdc.get_core_num()) > 0 else 0
    reminder_lp_cnt_cr = src_cr_lp_cnt % tdc.get_core_num()
    if reminder_lp_cnt_cr == 0:
        tmp_full_lp_cnt_cr += tdc.get_core_num()
    full_lp_cnt_cr = tmp_full_lp_cnt_cr + reminder_lp_cnt_cr

    tmp_full_lp_cnt_c = tdc.get_core_num() if tdc.floor_div(src_c_lp_cnt, tdc.get_core_num()) > 0 else 0
    reminder_lp_cnt_c = src_c_lp_cnt % tdc.get_core_num()
    if reminder_lp_cnt_c == 0:
        tmp_full_lp_cnt_c += tdc.get_core_num()
    full_lp_cnt_c = tmp_full_lp_cnt_c + reminder_lp_cnt_c

    tmp_full_lp_cnt_left = tdc.get_core_num() if tdc.floor_div(src_cl_lp_cnt, tdc.get_core_num()) > 0 else 0
    reminder_lp_cnt_left = src_cl_lp_cnt % tdc.get_core_num()
    if reminder_lp_cnt_left == 0:
        tmp_full_lp_cnt_left += tdc.get_core_num()
    full_lp_cnt_left = tmp_full_lp_cnt_left + reminder_lp_cnt_left

    lp_cnt_list = (full_lp_cnt_left, full_lp_cnt_c, full_lp_cnt_cr)
    if lp_cnt_list.index(max(lp_cnt_list)) == 0:
        tp_100_mc_pos = 0
        tp_100_used_core_cnt = tdc.ceil_div(src_cl_lp_cnt, tdc.ceil_div(src_cl_lp_cnt, tdc.get_core_num()))
        tp_100_nlc_cl_lp_cnt = tdc.ceil_div(src_cl_lp_cnt, tp_100_used_core_cnt)
        tp_100_lc_cl_lp_cnt = src_cl_lp_cnt - tp_100_nlc_cl_lp_cnt * (tp_100_used_core_cnt - 1)
        tp_100_core_step_in = tp_100_nlc_cl_lp_cnt * src_cl_lp_step_in
        tp_100_core_step_out = tp_100_nlc_cl_lp_cnt * src_cl_lp_step_out
        tp_100_nlc_cl_left = 0
        tp_100_lc_cl_left = src_cl_size % src_cl_lp_unit
        tp_100_nlc_c_lp_cnt = src_c_lp_cnt
        tp_100_lc_c_lp_cnt = src_c_lp_cnt
        tp_100_nlc_c_left = src_c_size % src_c_lp_unit
        tp_100_lc_c_left = src_c_size % src_c_lp_unit
        tp_100_nlc_cr_lp_cnt = src_cr_lp_cnt
        tp_100_lc_cr_lp_cnt = src_cr_lp_cnt
        tp_100_nlc_cr_left = src_cr_size % src_cr_lp_unit
        tp_100_lc_cr_left = src_cr_size % src_cr_lp_unit
    elif lp_cnt_list.index(max(lp_cnt_list)) == 1:
        tp_100_mc_pos = 1
        tp_100_used_core_cnt = tdc.ceil_div(src_c_lp_cnt, tdc.ceil_div(src_c_lp_cnt, tdc.get_core_num()))
        tp_100_nlc_c_lp_cnt = tdc.ceil_div(src_c_lp_cnt, tp_100_used_core_cnt)
        tp_100_lc_c_lp_cnt = src_c_lp_cnt - (tp_100_used_core_cnt - 1) * tp_100_nlc_c_lp_cnt
        tp_100_nlc_c_left = 0
        tp_100_lc_c_left = src_c_size % src_c_lp_unit
        tp_100_core_step_in = tp_100_nlc_c_lp_cnt * src_c_lp_step_in
        tp_100_core_step_out = tp_100_nlc_c_lp_cnt * src_c_lp_step_out
        tp_100_nlc_cr_lp_cnt = src_cr_lp_cnt
        tp_100_lc_cr_lp_cnt = src_cr_lp_cnt
        tp_100_nlc_cr_left = src_cr_size % src_cr_lp_unit
        tp_100_lc_cr_left = src_cr_size % src_cr_lp_unit
        tp_100_nlc_cl_lp_cnt = src_cl_lp_cnt
        tp_100_lc_cl_lp_cnt = src_cl_lp_cnt
        tp_100_nlc_cl_left = src_cl_size % src_cl_lp_unit
        tp_100_lc_cl_left = src_cl_size % src_cl_lp_unit
    else:
        tp_100_mc_pos = 2
        tp_100_used_core_cnt = tdc.ceil_div(src_cr_lp_cnt, tdc.ceil_div(src_cr_lp_cnt, tdc.get_core_num()))
        tp_100_nlc_cr_lp_cnt = tdc.ceil_div(src_cr_lp_cnt, tp_100_used_core_cnt)
        tp_100_lc_cr_lp_cnt = src_cr_lp_cnt - (tp_100_used_core_cnt - 1) * tp_100_nlc_cr_lp_cnt
        tp_100_nlc_cr_left = 0
        tp_100_lc_cr_left = src_cr_size % src_cr_lp_unit
        tp_100_core_step_in = tp_100_nlc_cr_lp_cnt * src_cr_lp_step_in
        tp_100_core_step_out = tp_100_nlc_cr_lp_cnt * src_cr_lp_step_out
        tp_100_nlc_c_lp_cnt = src_c_lp_cnt
        tp_100_lc_c_lp_cnt = src_c_lp_cnt
        tp_100_nlc_c_left = src_c_size % src_c_lp_unit
        tp_100_lc_c_left = src_c_size % src_c_lp_unit
        tp_100_nlc_cl_lp_cnt = src_cl_lp_cnt
        tp_100_lc_cl_lp_cnt = src_cl_lp_cnt
        tp_100_nlc_cl_left = src_cl_size % src_cl_lp_unit
        tp_100_lc_cl_left = src_cl_size % src_cl_lp_unit
    tiling_params = [tp_100_mc_pos, tp_100_used_core_cnt, tp_100_core_step_in, tp_100_core_step_out] + \
                    [tp_100_nlc_cl_lp_cnt, tp_100_nlc_cl_left, tp_100_nlc_c_lp_cnt,
                     tp_100_nlc_c_left, tp_100_nlc_cr_lp_cnt, tp_100_nlc_cr_left] + \
                    [tp_100_lc_cl_lp_cnt, tp_100_lc_cl_left, tp_100_lc_c_lp_cnt,
                     tp_100_lc_c_left, tp_100_lc_cr_lp_cnt, tp_100_lc_cr_left]

    return tiling_params


# 'pylint: disable=redefined-builtin, too-many-statements, too-many-branches, unbalanced-tuple-unpacking
def _tiling_params_positive(args):
    """
    calculate real tiling params for positive transform and last axis of source format is not c
    """

    in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size, in_dtype = args
    c0_len = out_shape[-1]  # axis c0
    tp_names = locals()

    # get tiling params for using vnchwconv
    half_ub_size = ub_size // 2 if c0_len == tdc.C0_16 else ub_size // 4  # for int8, c0 is 32
    one_vnc_line_size = half_ub_size // tdc.VNC_LINES // block_elem_cnt * block_elem_cnt
    tmp_ub_offset = one_vnc_line_size * tdc.VNC_LINES
    tp_100_ub_offset = tmp_ub_offset if c0_len == tdc.C0_16 else tmp_ub_offset * 2
    tp_100_vnc_line_size = one_vnc_line_size
    tp_100_c0_size = c0_len

    # axis c-right tiling parameters
    tp_100_cr_dims = FRAME_LEVEL
    tp_100_r1st_src_r2nd_dst_same = 1  # such as HWCN -> C1HWNoNiC0
    axis_src_cr_size = tdc.get_shape_size(in_shape[src_format.index("C") + 1:])
    tmp_src_cr_lp_unit = tp_100_vnc_line_size // c0_len // block_elem_cnt * block_elem_cnt
    if axis_src_cr_size < 2 * block_elem_cnt or in_dtype in ("float32", "int32", "uint32"):
        tp_100_tiling_mode = 1000
        tp_100_src_cr_lp_unit = tmp_src_cr_lp_unit if axis_src_cr_size > tmp_src_cr_lp_unit else axis_src_cr_size
    else:
        tp_100_tiling_mode = 1001
        tp_100_src_cr_lp_unit = tp_100_vnc_line_size if axis_src_cr_size > tp_100_vnc_line_size else axis_src_cr_size
    # count method: cr_idx/dst_rsize%size*dst_asize
    tmp_src_cr_format = src_format[src_format.index("C") + 1:]
    tmp_src_cr_shape = in_shape[src_format.index("C") + 1:]
    tmp_src_cr_shape.append(1)
    for idx, char in enumerate(reversed(tmp_src_cr_format)):
        tmp_src_idx = src_format.index(char)
        tmp_dst_idx = dst_format.index(char)
        tp_names["tp_100_cr_out_idx_" + str(idx) + "_size"] = in_shape[tmp_src_idx]
        tp_names["tp_100_cr_out_idx_" + str(idx) + "_dst_rsize"] = tdc.get_shape_size(tmp_src_cr_shape[-1 - idx:])
        tp_names["tp_100_cr_out_idx_" + str(idx) + "_dst_asize"] = tdc.get_shape_size(out_shape[tmp_dst_idx + 1:])
    # suppose there are 2 axises
    pad_axis_cnt = FRAME_LEVEL - len(tmp_src_cr_format)
    if pad_axis_cnt:
        tp_100_cr_dims = 1
        for _, idx in enumerate(PAD_IDX_LIST[len(tmp_src_cr_format):]):
            tp_names["tp_100_cr_out_idx_" + str(idx) + "_size"] = 1
            tp_names["tp_100_cr_out_idx_" + str(idx) + "_dst_rsize"] = 1
            tp_names["tp_100_cr_out_idx_" + str(idx) + "_dst_asize"] = 0
    if src_format[-1] != dst_format[-2]:
        tp_100_r1st_src_r2nd_dst_same = 0
    src_cr_lp_cnt = tdc.ceil_div(axis_src_cr_size, tp_100_src_cr_lp_unit)
    tp_100_src_cr_step_in = 1
    tp_100_src_cr_lp_step_in = tdc.get_shape_size([tp_100_src_cr_step_in, tp_100_src_cr_lp_unit])
    if tp_100_cr_dims == 2:
        tp_100_src_cr_step_out = 0
        tp_100_src_cr_lp_step_out = 0
    else:
        cr_char = src_format[-1]
        tp_100_src_cr_step_out = tdc.get_shape_size(out_shape[dst_format.index(cr_char) + 1:])
        tp_100_src_cr_lp_step_out = tdc.get_shape_size([tp_100_src_cr_step_out, tp_100_src_cr_lp_unit])

    # axis c tiling parameters
    src_c_idx = src_format.index("C")
    axis_src_c_size = in_shape[src_format.index("C")]
    tp_100_src_c_lp_unit = c0_len
    src_c_lp_cnt = tdc.ceil_div(axis_src_c_size, tp_100_src_c_lp_unit)
    tp_100_src_c_step_in = tdc.get_shape_size(in_shape[src_c_idx + 1:])
    tp_100_src_c_lp_step_in = tdc.get_shape_size([tp_100_src_c_lp_unit] + in_shape[src_c_idx + 1:])
    tp_100_src_c_lp_step_out = tdc.get_shape_size(out_shape[dst_format.index("C") + 1:])
    tp_100_src_c_step_in = tdc.get_shape_size(in_shape[src_c_idx + 1:])
    tp_100_c_mod_c0 = axis_src_c_size % c0_len

    # axis left parameters
    tp_100_cl_dims = FRAME_LEVEL
    axis_src_cl_size = tdc.get_shape_size(in_shape[:src_format.index("C")])
    if tp_100_tiling_mode == 1000:
        tmp_src_cl_lp_unit = tdc.NI_16
    elif tp_100_r1st_src_r2nd_dst_same == 0 and tp_100_tiling_mode == 1001 and axis_src_cl_size > tdc.get_core_num():
        tmp_src_cl_lp_unit = tp_100_vnc_line_size // tdc.ceil_fill(tp_100_src_cr_lp_unit, c0_len)
    else:
        tmp_src_cl_lp_unit = 1
    tp_100_src_cl_lp_unit = tmp_src_cl_lp_unit if axis_src_cl_size > tmp_src_cl_lp_unit else axis_src_cl_size
    src_cl_lp_cnt = tdc.ceil_div(axis_src_cl_size, tp_100_src_cl_lp_unit)
    # count method: left_axis_size/dst_rsize%size*asize
    tmp_src_cl_format = src_format[:src_format.index("C")]
    tmp_src_cl_shape = in_shape[:src_format.index("C")]
    tmp_src_cl_shape.append(1)
    for idx, char in enumerate(reversed(tmp_src_cl_format)):
        tmp_src_chr = src_format.index(char)
        tmp_dst_chr = dst_format.index(char)
        tp_names["tp_100_cl_out_idx_" + str(idx) + "_size"] = in_shape[tmp_src_chr]
        tp_names["tp_100_cl_out_idx_" + str(idx) + "_dst_rsize"] = tdc.get_shape_size(tmp_src_cl_shape[-1 - idx:])
        tp_names["tp_100_cl_out_idx_" + str(idx) + "_dst_asize"] = tdc.get_shape_size(out_shape[tmp_dst_chr + 1:])
    # suppose there are 2 axises
    pad_axis_cnt = FRAME_LEVEL - len(tmp_src_cl_format)
    if pad_axis_cnt:
        tp_100_cl_dims = 1
        for _, idx in enumerate(PAD_IDX_LIST[len(tmp_src_cl_format):]):
            tp_names["tp_100_cl_out_idx_" + str(idx) + "_size"] = 1
            tp_names["tp_100_cl_out_idx_" + str(idx) + "_dst_rsize"] = 1
            tp_names["tp_100_cl_out_idx_" + str(idx) + "_dst_asize"] = 0

    tp_100_src_cl_step_in = tdc.get_shape_size(in_shape[src_c_idx:])
    tp_100_src_cl_lp_step_in = tdc.get_shape_size([tp_100_src_cl_step_in, tp_100_src_cl_lp_unit])
    if tp_100_cl_dims == 2:
        tp_100_src_cl_step_out = 0
        tp_100_src_cl_lp_step_out = 0
    else:
        cl_char = src_format[0]
        tp_100_src_cl_step_out = tdc.get_shape_size(out_shape[dst_format.index(cl_char) + 1:])
        tp_100_src_cl_lp_step_out = tdc.get_shape_size([tp_100_src_cl_step_out, tp_100_src_cl_lp_unit])

    # mulitple core parameters
    cr_args = (src_cr_lp_cnt, axis_src_cr_size, tp_100_src_cr_lp_unit,
               tp_100_src_cr_lp_step_in, tp_100_src_cr_lp_step_out)
    c_args = src_c_lp_cnt, axis_src_c_size, tp_100_src_c_lp_unit, tp_100_src_c_lp_step_in, tp_100_src_c_lp_step_out
    cl_args = (src_cl_lp_cnt, axis_src_cl_size, tp_100_src_cl_lp_unit,
               tp_100_src_cl_lp_step_in, tp_100_src_cl_lp_step_out)
    (tp_100_mc_pos, tp_100_used_core_cnt, tp_100_core_step_in, tp_100_core_step_out, tp_100_nlc_cl_lp_cnt,
     tp_100_nlc_cl_left, tp_100_nlc_c_lp_cnt, tp_100_nlc_c_left, tp_100_nlc_cr_lp_cnt, tp_100_nlc_cr_left,
     tp_100_lc_cl_lp_cnt, tp_100_lc_cl_left, tp_100_lc_c_lp_cnt, tp_100_lc_c_left, tp_100_lc_cr_lp_cnt,
     tp_100_lc_cr_left) = _get_mc_info_positive(cr_args, c_args, cl_args)

    sub_tiling_params = [tp_100_tiling_mode, tp_100_ub_offset, tp_100_mc_pos, tp_100_used_core_cnt, tp_100_core_step_in,
                         tp_100_core_step_out, tp_100_vnc_line_size, tp_100_c_mod_c0, tp_100_c0_size, tp_100_cl_dims,
                         tp_100_cr_dims, tp_100_r1st_src_r2nd_dst_same,
                         tp_100_src_cl_step_in, tp_100_src_cl_step_out, tp_100_src_cl_lp_unit, tp_100_src_cl_lp_step_in,
                         tp_100_src_cl_lp_step_out, tp_100_src_c_step_in, tp_100_src_c_lp_unit, tp_100_src_c_lp_step_in,
                         tp_100_src_c_lp_step_out, tp_100_src_cr_step_in, tp_100_src_cr_step_out, tp_100_src_cr_lp_unit,
                         tp_100_src_cr_lp_step_in, tp_100_src_cr_lp_step_out,
                         tp_100_nlc_cl_lp_cnt, tp_100_nlc_cl_left, tp_100_nlc_c_lp_cnt, tp_100_nlc_c_left,
                         tp_100_nlc_cr_lp_cnt, tp_100_nlc_cr_left, tp_100_lc_cl_lp_cnt, tp_100_lc_cl_left,
                         tp_100_lc_c_lp_cnt, tp_100_lc_c_left, tp_100_lc_cr_lp_cnt, tp_100_lc_cr_left,
                         tp_names["tp_100_cl_out_idx_0_size"], tp_names["tp_100_cl_out_idx_0_dst_rsize"],
                         tp_names["tp_100_cl_out_idx_0_dst_asize"], tp_names["tp_100_cl_out_idx_1_size"],
                         tp_names["tp_100_cl_out_idx_1_dst_rsize"], tp_names["tp_100_cl_out_idx_1_dst_asize"],
                         tp_names["tp_100_cr_out_idx_0_size"], tp_names["tp_100_cr_out_idx_0_dst_rsize"],
                         tp_names["tp_100_cr_out_idx_0_dst_asize"], tp_names["tp_100_cr_out_idx_1_size"],
                         tp_names["tp_100_cr_out_idx_1_dst_rsize"], tp_names["tp_100_cr_out_idx_1_dst_asize"]]

    return sub_tiling_params


def _get_tiling_params_func(args):
    """
    get tiling parameters function
    """

    in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size, in_dtype = args

    (in_shape_new, out_shape_new,
     in_format_new, out_format_new) = _renew_input_output_shape_format(in_shape, out_shape, src_format, dst_format)
    args_get_tp = in_shape_new, out_shape_new, in_format_new, out_format_new, block_elem_cnt, ub_size, in_dtype
    tiling_params = _tiling_params_positive(args_get_tp)

    return tiling_params


# 'pylint: disable=unused-variable
def _twice_vnchwconv_invert(args):
    """
    do ncdh to ndhc transform by twice vnchwconv
    """

    (tik_inst, src_ub, mc_pos, dst_ub, dst_ub_offset, vnc_col_size, plp_c_size, r1st_src_r2nd_dst_same,
     cl_lp_idx, plp_cl_size, cr_lp_cnt, plp_cr_size, c_mod_c0, c0_size, ele_per_block) = args
    tensor_dtype = src_ub.dtype.lower()
    size_factor = tdc.get_dtype_factor(tensor_dtype)
    vnc_col_len = vnc_col_size * size_factor
    dst_ub_offset_casted = dst_ub_offset * size_factor
    if tensor_dtype in ("float32", "int32", "uint32"):
        src_ub_casted = src_ub.reinterpret_cast_to("float16")
        dst_ub_casted = dst_ub.reinterpret_cast_to("float16")
    else:
        src_ub_casted = src_ub
        dst_ub_casted = dst_ub

    # do ncdh -> cdhn
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        plp_cr_block_align_size = tik_inst.Scalar(name="plp_cr_block_align_size")
        with tik_inst.if_scope(tik.all(cr_lp_cnt == 1, mc_pos != 2)):
            plp_cr_block_align_size.set_as(plp_cr_size)
        with tik_inst.else_scope():
            plp_cr_block_align_size.set_as(tdc.ceil_fill(plp_cr_size, ele_per_block))

        repeat_cnt = tdc.ceil_div(plp_c_size * plp_cr_block_align_size * size_factor, c0_size)
        if tensor_dtype not in ("int8", "uint8"):
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(1)
                dst_stride.set_as(16)
            src_addr_list = [src_ub_casted[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub_casted[dst_ub_offset_casted + c0_size * i] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        else:
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(1)
                dst_stride.set_as(32)
            src_addr_list = [src_ub[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub[dst_ub_offset + c0_size * i] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            dst_addr_list = [dst_ub[dst_ub_offset + (tdc.C0_16 + i) * c0_size] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, True, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

        # padding zero
        clean_len = plp_cr_size * tdc.ceil_fill(plp_c_size, c0_size) * c0_size
        with tik_inst.if_scope(plp_c_size < c0_size):
            if tensor_dtype in ("int8", "uint8"):
                src_ub_int32 = src_ub.reinterpret_cast_to("int32")
                tdc.clean_ubuf(tik_inst, src_ub_int32, 0, tdc.ceil_div(clean_len, 4))
            else:
                tdc.clean_ubuf(tik_inst, src_ub, 0, clean_len)
        # do cdhn -> dhcn
        with tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, plp_c_size) as c_idx:
                tik_inst.data_move(src_ub[c_idx * c0_size],
                                   dst_ub[dst_ub_offset + c_idx*plp_cr_block_align_size*c0_size],
                                   0, plp_cr_size, 1 * size_factor, 0, (c0_size - 1) * size_factor)

        # do dhcn -> ndhc or dhcn -> dhnc
        if tensor_dtype not in ("int8", "uint8"):
            with tik_inst.if_scope(r1st_src_r2nd_dst_same == 0):  # for NCHW -> C1HWNoNiC0
                with tik_inst.for_range(0, size_factor) as factor_idx:
                    src_addr_list = [src_ub_casted[tdc.C0_16 * (i + c0_size * factor_idx)] for i in tdc.ADDR_IDX_LIST]
                    dst_addr_list = [dst_ub_casted[dst_ub_offset_casted + (size_factor * i + factor_idx) * c0_size]
                                     for i in tdc.ADDR_IDX_LIST]
                    repeat_cnt = plp_cr_size
                    with tik_inst.if_scope(repeat_cnt == 1):
                        src_stride.set_as(0)
                        dst_stride.set_as(0)
                    with tik_inst.else_scope():
                        src_stride.set_as(16 * size_factor)
                        dst_stride.set_as(plp_cl_size * size_factor)
                    tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            with tik_inst.else_scope():  # for NCHW -> NC1HWC0
                vnc_row_size = plp_cr_size * size_factor * c0_size
                src_addr_list = [src_ub_casted[tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
                dst_addr_list = [dst_ub_casted[dst_ub_offset_casted + vnc_row_size * i] for i in tdc.ADDR_IDX_LIST]
                repeat_cnt = plp_cr_size * size_factor
                with tik_inst.if_scope(repeat_cnt == 1):
                    src_stride.set_as(0)
                    dst_stride.set_as(0)
                with tik_inst.else_scope():
                    src_stride.set_as(16)
                    dst_stride.set_as(1)
                tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        else:
            repeat_cnt = plp_cr_size
            with tik_inst.if_scope(r1st_src_r2nd_dst_same == 0):  # for NCHW -> C1HWNoNiC0
                dst_addr_list = [dst_ub[dst_ub_offset + c0_size * i] for i in tdc.ADDR_IDX_LIST]
                with tik_inst.if_scope(repeat_cnt == 1):
                    src_stride.set_as(0)
                    dst_stride.set_as(0)
                with tik_inst.else_scope():
                    src_stride.set_as(32)
                    dst_stride.set_as(plp_cl_size)
                src_addr_list = [src_ub[c0_size * i] for i in tdc.ADDR_IDX_LIST]
                tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
                src_addr_list = [src_ub[c0_size * (i + tdc.C0_16)] for i in tdc.ADDR_IDX_LIST]
                tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            with tik_inst.else_scope():  # for NCHW -> NC1HWC0
                vnc_row_size = plp_cr_size * c0_size
                dst_addr_list = [dst_ub[dst_ub_offset + vnc_row_size * i] for i in tdc.ADDR_IDX_LIST]
                with tik_inst.if_scope(repeat_cnt == 1):
                    src_stride.set_as(0)
                    dst_stride.set_as(0)
                with tik_inst.else_scope():
                    src_stride.set_as(32)
                    dst_stride.set_as(1)
                src_addr_list = [src_ub[c0_size * i] for i in tdc.ADDR_IDX_LIST]
                tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
                src_addr_list = [src_ub[c0_size * (i + tdc.C0_16)] for i in tdc.ADDR_IDX_LIST]
                tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


# 'pylint: disable=unused-variable
def _once_vnchwconv_invert(args):
    """
    do cdh to dhc transform by once vnchwconv
    """

    (tik_inst, src_ub, mc_pos, dst_ub, dst_ub_offset, vnc_col_size, plp_c_size, r1st_src_r2nd_dst_same,
     cl_lp_idx, plp_cl_size, cr_lp_cnt, plp_cr_size, c_mod_c0, c0_size, ele_per_block) = args
    tensor_dtype = src_ub.dtype.lower()
    size_factor = tdc.get_dtype_factor(tensor_dtype)
    repeat_cnt = tdc.ceil_div(plp_cr_size, c0_size)
    if tensor_dtype in ("float32", "int32", "uint32"):  # to avoid compile error
        src_ub = src_ub.reinterpret_cast_to("float16")
        dst_ub = dst_ub.reinterpret_cast_to("float16")
        if vnc_col_size % 32 > 0:  # to avoid compile error
            vnc_col_size = 32

    # do cdh -> dhc
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        dst_gap = tik_inst.Scalar()
        if tensor_dtype not in ("int8", "uint8"):
            with tik_inst.if_scope(tik.all(plp_c_size % c0_size > 0, cl_lp_idx == 0)):  # padding zero
                tdc.clean_ubuf(tik_inst, src_ub, c_mod_c0 * vnc_col_size, (c0_size - c_mod_c0) * vnc_col_size)

            with tik_inst.if_scope(r1st_src_r2nd_dst_same == 1):  # for NCHW -> NC1HWC0
                dst_gap.set_as(c0_size)
                dst_stride.set_as(16)
            with tik_inst.else_scope():  # for NCHW -> C1HWNoNiC0
                dst_gap.set_as(plp_cl_size * c0_size)
                dst_stride.set_as(plp_cl_size * 16)

            src_addr_list = [src_ub[vnc_col_size * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub[dst_ub_offset + dst_gap * size_factor * i] for i in tdc.ADDR_IDX_LIST]
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(1)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        else:
            with tik_inst.if_scope(tik.all(plp_c_size % c0_size > 0, cl_lp_idx == 0)):  # padding zero
                src_ub_int32 = src_ub.reinterpret_cast_to("int32")
                tdc.clean_ubuf(tik_inst, src_ub_int32, c_mod_c0 * vnc_col_size // 4,
                               (c0_size - c_mod_c0) * vnc_col_size // 4)

            with tik_inst.if_scope(r1st_src_r2nd_dst_same == 1):  # for NCHW -> NC1HWC0
                dst_gap.set_as(c0_size)
                dst_stride.set_as(32)
            with tik_inst.else_scope():  # for NCHW -> C1HWNoNiC0
                dst_gap.set_as(plp_cl_size * c0_size)
                dst_stride.set_as(plp_cl_size * 32)

            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(1)

            # for target low half
            src_addr_list = [src_ub[vnc_col_size * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub[dst_ub_offset + dst_gap * i] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            dst_addr_list = [dst_ub[dst_ub_offset + dst_gap * (i + tdc.C0_16)] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, True, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            # for target high half
            src_addr_offset = tdc.NI_16 * vnc_col_size
            src_addr_list = [src_ub[src_addr_offset + vnc_col_size * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub[dst_ub_offset + dst_gap * i] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            dst_addr_list = [dst_ub[dst_ub_offset + dst_gap * (i + tdc.C0_16)] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(True, True, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


def _update_input_offset(args):
    """
    count input gm offset
    """

    cl_lp_idx, cl_lp_step_in, c_lp_idx, c_lp_step_in, cr_lp_idx, cr_lp_step_in, core_step_in = args

    in_offset = (cl_lp_idx * cl_lp_step_in + c_lp_idx * c_lp_step_in + cr_lp_idx * cr_lp_step_in + core_step_in)

    return in_offset


def _copy_data_in(args):
    """
    copy data from gm to ub
    """

    (tik_inst, src_in_gm, src_ub, mc_pos, in_gm_offset, vnc_col_size, tiling_mode, plp_cl_size,
     cl_step_in, plp_c_size, c_step_in, cr_lp_cnt, plp_cr_size, ele_per_block) = args
    cr_block_align_size = tdc.ceil_fill(plp_cr_size, ele_per_block)
    cr_blocks = tdc.ceil_div(plp_cr_size, ele_per_block)
    cr_gm_stride = (c_step_in - plp_cr_size) // ele_per_block
    cr_ub_stride = (vnc_col_size - plp_cr_size) // ele_per_block

    with tik_inst.new_stmt_scope(disable_sync=True):
        with tik_inst.if_scope(tiling_mode == 1000):  # for two times vnchwconv case
            with tik_inst.if_scope(tik.all(cr_lp_cnt == 1, mc_pos != 2)):
                with tik_inst.for_range(0, plp_cl_size) as cl_idx:
                    tik_inst.data_move(src_ub[cl_idx * vnc_col_size], src_in_gm[in_gm_offset + cl_idx * cl_step_in],
                                       0, 1, tdc.ceil_div(plp_c_size * plp_cr_size, ele_per_block), 0, 0)
            with tik_inst.elif_scope(tik.all(c_step_in % ele_per_block == 0, cr_gm_stride <= tdc.STRIDE_LIMIT_MTE)):
                with tik_inst.for_range(0, plp_cl_size) as cl_idx:
                    tik_inst.data_move(src_ub[cl_idx * vnc_col_size], src_in_gm[in_gm_offset + cl_idx * cl_step_in],
                                       0, plp_c_size, cr_blocks, cr_gm_stride, 0)
            with tik_inst.else_scope():
                with tik_inst.for_range(0, plp_cl_size) as cl_idx:
                    with tik_inst.for_range(0, plp_c_size) as c_idx:
                        tik_inst.data_move(src_ub[cl_idx * vnc_col_size + c_idx * cr_block_align_size],
                                           src_in_gm[in_gm_offset + cl_idx * cl_step_in + c_idx * c_step_in],
                                           0, 1, cr_blocks, 0, 0)
        with tik_inst.else_scope():  # for one time vnchwconv case
            with tik_inst.if_scope(tik.all(c_step_in % ele_per_block == 0, cr_gm_stride <= tdc.STRIDE_LIMIT_MTE)):
                tik_inst.data_move(src_ub, src_in_gm[in_gm_offset],
                                   0, plp_c_size, cr_blocks, cr_gm_stride, cr_ub_stride)
            with tik_inst.else_scope():
                with tik_inst.for_range(0, plp_c_size) as c_idx_1:
                    tik_inst.data_move(src_ub[c_idx_1 * vnc_col_size], src_in_gm[in_gm_offset + c_idx_1 * c_step_in],
                                       0, 1, cr_blocks, 0, 0)


def _copy_data_in_with_dmp_for_two_times_transpose(args):
    (tik_inst, src_ub, src_in_gm, in_gm_offset, cl_step_in, c_step_in, cr_lp_cnt, mc_pos, plp_cl_size, plp_c_size,
     plp_cr_size, cr_block_align_size, vnc_col_size, cr_gm_stride, dmp_max_stride, b8_times) = args
    with tik_inst.if_scope(tik.all(cr_lp_cnt == 1, mc_pos != 2)):
        with tik_inst.for_range(0, plp_cl_size) as cl_idx:
            tik_inst.data_move_pad(src_ub[cl_idx * vnc_col_size],
                                   src_in_gm[in_gm_offset + cl_idx * cl_step_in],
                                   1, plp_c_size * plp_cr_size * b8_times, 0, 0)
    with tik_inst.elif_scope(cr_gm_stride <= dmp_max_stride):
        with tik_inst.for_range(0, plp_cl_size) as cl_idx:
            tik_inst.data_move_pad(src_ub[cl_idx * vnc_col_size],
                                   src_in_gm[in_gm_offset + cl_idx * cl_step_in],
                                   plp_c_size, plp_cr_size * b8_times, 0, cr_gm_stride)
    with tik_inst.else_scope():
        with tik_inst.for_range(0, plp_cl_size) as cl_idx:
            with tik_inst.for_range(0, plp_c_size) as c_idx:
                tik_inst.data_move_pad(src_ub[cl_idx * vnc_col_size + c_idx * cr_block_align_size],
                                       src_in_gm[in_gm_offset + cl_idx * cl_step_in + c_idx * c_step_in],
                                       1, plp_cr_size * b8_times, 0, 0)


def _copy_data_in_with_dmp_for_one_time_transpose(args):
    (tik_inst, src_ub, src_in_gm, in_gm_offset, plp_c_size, plp_cr_size, vnc_col_size, c_step_in, cr_ub_stride,
     cr_gm_stride, dmp_max_stride, b8_times) = args
    with tik_inst.if_scope(cr_gm_stride <= dmp_max_stride):
        tik_inst.data_move_pad(src_ub, src_in_gm[in_gm_offset],
                               plp_c_size, plp_cr_size * b8_times, cr_ub_stride, cr_gm_stride)
    with tik_inst.else_scope():
        with tik_inst.for_range(0, plp_c_size) as c_idx_1:
            tik_inst.data_move_pad(src_ub[c_idx_1 * vnc_col_size],
                                   src_in_gm[in_gm_offset + c_idx_1 * c_step_in],
                                   1, plp_cr_size * b8_times, 0, 0)


def _copy_data_in_with_dmp(args):
    """
    copy data from gm to ub with data_move_pad
    """

    (tik_inst, src_in_gm, src_ub, mc_pos, in_gm_offset, vnc_col_size, tiling_mode, plp_cl_size,
     cl_step_in, plp_c_size, c_step_in, cr_lp_cnt, plp_cr_size, ele_per_block) = args
    dmp_max_stride = 2**32 - 1
    b8_times = tdc.BLOCK_BYTE_SIZE // ele_per_block
    cr_block_align_size = tdc.ceil_fill(plp_cr_size, ele_per_block)
    cr_blocks = tdc.ceil_div(plp_cr_size, ele_per_block)
    cr_gm_stride = (c_step_in - plp_cr_size) * b8_times
    cr_ub_stride = vnc_col_size // ele_per_block - cr_blocks

    with tik_inst.new_stmt_scope(disable_sync=True):
        with tik_inst.if_scope(tiling_mode == 1000):  # for two times vnchwconv case
            two_times_transpose_args = (tik_inst, src_ub, src_in_gm, in_gm_offset, cl_step_in, c_step_in, cr_lp_cnt,
                                        mc_pos, plp_cl_size, plp_c_size, plp_cr_size, cr_block_align_size,
                                        vnc_col_size, cr_gm_stride, dmp_max_stride, b8_times)
            _copy_data_in_with_dmp_for_two_times_transpose(two_times_transpose_args)
        with tik_inst.else_scope():  # for one time vnchwconv case
            one_time_transpose_args = (tik_inst, src_ub, src_in_gm, in_gm_offset, plp_c_size, plp_cr_size,
                                       vnc_col_size, c_step_in, cr_ub_stride, cr_gm_stride, dmp_max_stride, b8_times)
            _copy_data_in_with_dmp_for_one_time_transpose(one_time_transpose_args)


def _update_out_offset_cl(args):
    """
    update c-left out offset
    """

    (cl_offset, cl_base, cl_out_idx_0_size, cl_out_idx_0_dst_rsize, cl_out_idx_0_dst_asize,
     cl_out_idx_1_size, cl_out_idx_1_dst_rsize, cl_out_idx_1_dst_asize) = args

    cl_offset.set_as(cl_base // cl_out_idx_0_dst_rsize % cl_out_idx_0_size * cl_out_idx_0_dst_asize +
                     cl_base // cl_out_idx_1_dst_rsize % cl_out_idx_1_size * cl_out_idx_1_dst_asize)


def _move_data_out(args):
    (tik_inst, dst_out_gm, out_gm_offset, dst_ub, out_ub_offset,
     loop_cnt, burst_len, gm_step, ub_step, ele_per_block) = args
    out_gm_gap = (gm_step - burst_len) // ele_per_block
    out_ub_gap = (ub_step - burst_len) // ele_per_block
    with tik_inst.if_scope(out_gm_gap <= tdc.STRIDE_LIMIT_MTE):
        tik_inst.data_move(dst_out_gm[out_gm_offset], dst_ub[out_ub_offset],
                           0, loop_cnt, burst_len // ele_per_block, out_ub_gap, out_gm_gap)
    with tik_inst.else_scope():
        with tik_inst.for_range(0, loop_cnt) as last_cl_idx:
            tik_inst.data_move(dst_out_gm[out_gm_offset + last_cl_idx*gm_step],
                               dst_ub[out_ub_offset + last_cl_idx * ub_step], 0, 1, burst_len // ele_per_block, 0, 0)


def _count_out_cr_idx(args):
    """
    get c-right index
    """

    (tik_inst, block_idx, mc_pos, cr_index, cr_idx, cr_lp_idx, nlc_cr_lp_cnt, cr_lp_unit) = args
    with tik_inst.if_scope(mc_pos == 2):
        cr_index.set_as(cr_idx + (cr_lp_idx + block_idx * nlc_cr_lp_cnt) * cr_lp_unit)
    with tik_inst.else_scope():
        cr_index.set_as(cr_idx + cr_lp_idx * cr_lp_unit)


def _update_out_offset_cr(args):
    """
    update c-right out offset
    """

    (cr_offset, cr_index, cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
     cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize) = args

    cr_offset.set_as(cr_index // cr_out_idx_0_dst_rsize % cr_out_idx_0_size * cr_out_idx_0_dst_asize +
                     cr_index // cr_out_idx_1_dst_rsize % cr_out_idx_1_size * cr_out_idx_1_dst_asize)


def _inner_move_data_out_target_two_cl(args):
    """
    move data out process for target format has two c-left dims
    """
    (tik_inst, dst_out_gm, gm_out_offset, dst_ub, ub_out_offset, last_cr_cl_size, is_last_cr_cl_nz, mid_lp_cnt,
     cur_cr_cl_size, burst_len, idx_0_dim_asize, idx_0_dim_size, idx_1_dim_asize,
     renew_idx_0_dim_size, ele_per_block) = args
    with tik_inst.if_scope(last_cr_cl_size > 0):
        last_cl_1_cr_2_args = (tik_inst, dst_out_gm, gm_out_offset, dst_ub, ub_out_offset, last_cr_cl_size,
                               burst_len, idx_0_dim_asize, burst_len, ele_per_block)
        _move_data_out(last_cl_1_cr_2_args)
        is_last_cr_cl_nz.set_as(1)
    with tik_inst.else_scope():
        is_last_cr_cl_nz.set_as(0)

    last_gm_end_offset = is_last_cr_cl_nz * ((last_cr_cl_size - idx_0_dim_size)*idx_0_dim_asize + idx_1_dim_asize)
    last_ub_end_offset = last_cr_cl_size * burst_len + ub_out_offset
    with tik_inst.if_scope(mid_lp_cnt > 0):
        with tik_inst.for_range(0, mid_lp_cnt) as mid_cl_idx:
            mid_cl_gm_offset = mid_cl_idx * idx_1_dim_asize + gm_out_offset + last_gm_end_offset
            mid_cl_ub_offset = mid_cl_idx * renew_idx_0_dim_size * burst_len + last_ub_end_offset
            mid_cl_1_cr_2_args = (tik_inst, dst_out_gm, mid_cl_gm_offset, dst_ub, mid_cl_ub_offset,
                                  renew_idx_0_dim_size, burst_len, idx_0_dim_asize, burst_len, ele_per_block)
            _move_data_out(mid_cl_1_cr_2_args)

    mid_gm_end_offset = mid_lp_cnt * idx_1_dim_asize + last_gm_end_offset + gm_out_offset
    mid_ub_end_offset = mid_lp_cnt * renew_idx_0_dim_size * burst_len + last_ub_end_offset
    with tik_inst.if_scope(cur_cr_cl_size > 0):
        cur_cl_1_cr_2_args = (tik_inst, dst_out_gm, mid_gm_end_offset, dst_ub, mid_ub_end_offset, cur_cr_cl_size,
                              burst_len, idx_0_dim_asize, burst_len, ele_per_block)
        _move_data_out(cur_cl_1_cr_2_args)


# 'pylint: disable=unused-variable
def _copy_data_out_1st_src_r2nd_dst_not_same(out_offset_args, copy_out_args):
    """
    copy data from ub to gm for source 1st and target r2nd is not same
    """

    (cl_lp_idx, cl_lp_step_out, c_lp_idx, c_lp_step_out, cr_lp_unit, nlc_cr_lp_cnt, cr_lp_idx,
     cr_lp_step_out, core_step_out, cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
     cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize) = out_offset_args
    (tik_inst, dst_out_gm, dst_ub, mc_pos, block_idx,
     cl_dims, cr_dims, plp_cl_size, plp_cr_size, cr_step_out, c0_size, ele_per_block) = copy_out_args
    offset_base = (cl_lp_idx * cl_lp_step_out + c_lp_idx * c_lp_step_out + cr_lp_idx * cr_lp_step_out + core_step_out)

    with tik_inst.new_stmt_scope(disable_sync=True):
        cr_offset = tik_inst.Scalar(name="cr_offset")
        cr_index = tik_inst.Scalar(name="cr_index")
        last_cr_size = tik_inst.Scalar(name="last_cr_size")
        mid_lp_cnt = tik_inst.Scalar(name="mid_lp_cnt")
        cur_cr_size = tik_inst.Scalar(name="cur_cr_size")
        is_last_cr_nz = tik_inst.Scalar(name="is_last_cr_nz")
        renew_cr_0_size = tik_inst.Scalar(name="renew_cr_0_size")

        # to avoid compile failed
        with tik_inst.if_scope(cr_out_idx_0_size <= tdc.REPEAT_LIMIT_MTE):
            renew_cr_0_size.set_as(cr_out_idx_0_size)
        with tik_inst.else_scope():
            renew_cr_0_size.set_as(1)

        with tik_inst.if_scope(cr_dims == 2):
            cr_idx_args = (tik_inst, block_idx, mc_pos, cr_index, 0, cr_lp_idx, nlc_cr_lp_cnt, cr_lp_unit)
            _count_out_cr_idx(cr_idx_args)
            cr_args = (cr_offset, cr_index, cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
                       cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize)
            _update_out_offset_cr(cr_args)

            split_cr_args = (tik_inst, cr_out_idx_0_size, last_cr_size,
                             mid_lp_cnt, cur_cr_size, cr_index, plp_cr_size)
            _split_dims(split_cr_args)
            two_cr_args = (tik_inst, dst_out_gm, offset_base + cr_offset, dst_ub, 0, last_cr_size,
                           is_last_cr_nz, mid_lp_cnt, cur_cr_size, plp_cl_size * c0_size,
                           cr_out_idx_0_dst_asize, cr_out_idx_0_size, cr_out_idx_1_dst_asize,
                           renew_cr_0_size, ele_per_block)
            _inner_move_data_out_target_two_cl(two_cr_args)
        with tik_inst.else_scope():
            cr_1_cl_1_args = (tik_inst, dst_out_gm, offset_base, dst_ub, 0, plp_cr_size,
                              plp_cl_size * c0_size, cr_step_out, plp_cl_size * c0_size, ele_per_block)
            _move_data_out(cr_1_cl_1_args)


def _split_dims(args):
    """
    split two dimensions into three parts
    """

    tik_inst, cr_out_idx_0_size, last_cr_size, mid_lp_cnt, cur_cr_size, cr_begin, plp_cr_size = args
    next_cr_gap = cr_out_idx_0_size - cr_begin % cr_out_idx_0_size
    with tik_inst.if_scope(next_cr_gap == cr_out_idx_0_size):
        last_cr_size.set_as(0)
    with tik_inst.else_scope():
        with tik_inst.if_scope(next_cr_gap <= plp_cr_size):
            last_cr_size.set_as(next_cr_gap)
        with tik_inst.else_scope():
            last_cr_size.set_as(plp_cr_size)
    mid_lp_cnt.set_as((plp_cr_size - last_cr_size) // cr_out_idx_0_size)
    cur_cr_size.set_as(plp_cr_size - last_cr_size - mid_lp_cnt * cr_out_idx_0_size)


# 'pylint: disable=unused-variable
def _copy_data_out_1st_src_r2nd_dst_same(out_offset_args, copy_out_args):
    """
    copy data from ub to gm for source 1st and target r2nd is same
    """

    (cl_lp_unit, nlc_cl_lp_cnt, cl_lp_idx, cl_lp_step_out, c_lp_idx, c_lp_step_out,
     cr_lp_unit, nlc_cr_lp_cnt, cr_lp_idx, cr_lp_step_out, core_step_out,
     cl_out_idx_0_size, cl_out_idx_0_dst_rsize, cl_out_idx_0_dst_asize,
     cl_out_idx_1_size, cl_out_idx_1_dst_rsize, cl_out_idx_1_dst_asize,
     cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
     cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize) = out_offset_args
    (tik_inst, dst_out_gm, dst_ub, mc_pos, block_idx, cl_dims, cr_dims, plp_cl_size,
     cl_step_out, plp_cr_size, cr_step_out, c0_size, ele_per_block) = copy_out_args
    offset_base = (cl_lp_idx * cl_lp_step_out + c_lp_idx * c_lp_step_out + cr_lp_idx * cr_lp_step_out + core_step_out)

    with tik_inst.new_stmt_scope(disable_sync=True):
        cl_offset = tik_inst.Scalar(name="cl_offset")
        cr_offset = tik_inst.Scalar(name="cr_offset")
        cr_begin = tik_inst.Scalar(name="cr_begin")
        last_cr_cl_size = tik_inst.Scalar(name="last_cr_cl_size")
        mid_lp_cnt = tik_inst.Scalar(name="mid_lp_cnt")
        cur_cr_cl_size = tik_inst.Scalar(name="cur_cr_cl_size")
        is_last_cr_cl_nz = tik_inst.Scalar(name="is_last_cr_cl_nz")
        renew_cr_0_size = tik_inst.Scalar(name="renew_cr_0_size")
        renew_cl_0_size = tik_inst.Scalar(name="renew_cl_0_size")
        cl_base = tik_inst.Scalar(name="cl_base")

        # to avoid compile failed
        with tik_inst.if_scope(cr_out_idx_0_size <= tdc.REPEAT_LIMIT_MTE):
            renew_cr_0_size.set_as(cr_out_idx_0_size)
        with tik_inst.else_scope():
            renew_cr_0_size.set_as(1)
        with tik_inst.if_scope(cl_out_idx_0_size <= tdc.REPEAT_LIMIT_MTE):
            renew_cl_0_size.set_as(cl_out_idx_0_size)
        with tik_inst.else_scope():
            renew_cl_0_size.set_as(1)

        with tik_inst.if_scope(mc_pos == 0):
            cl_base.set_as((cl_lp_idx + block_idx * nlc_cl_lp_cnt) * cl_lp_unit)
        with tik_inst.else_scope():
            cl_base.set_as(cl_lp_idx * cl_lp_unit)

        def _inner_move_data_out_srcr1st_dstr2nd_same():
            with tik_inst.if_scope(last_cr_cl_size > 0):
                last_cl_1_cr_2_args = (tik_inst, dst_out_gm, offset_base + cr_offset, dst_ub, 0, plp_cl_size,
                                       last_cr_cl_size * c0_size, cl_step_out, plp_cr_size * c0_size, ele_per_block)
                _move_data_out(last_cl_1_cr_2_args)
                is_last_cr_cl_nz.set_as(1)
            with tik_inst.else_scope():
                is_last_cr_cl_nz.set_as(0)

            last_gm_end_offset = is_last_cr_cl_nz * ((last_cr_cl_size - cr_out_idx_0_size)*cr_out_idx_0_dst_asize +
                                                     cr_out_idx_1_dst_asize)
            last_ub_end_offset = last_cr_cl_size * c0_size
            with tik_inst.if_scope(mid_lp_cnt > 0):
                with tik_inst.for_range(0, plp_cl_size) as mid_cl_idx:
                    mid_cl_gm_offset = mid_cl_idx * cl_step_out + offset_base + cr_offset + last_gm_end_offset
                    mid_cl_ub_offset = mid_cl_idx * plp_cr_size * c0_size + last_ub_end_offset
                    mid_cl_1_cr_2_args = (tik_inst, dst_out_gm, mid_cl_gm_offset, dst_ub, mid_cl_ub_offset, mid_lp_cnt,
                                          renew_cr_0_size * c0_size, cr_out_idx_1_dst_asize, renew_cr_0_size * c0_size,
                                          ele_per_block)
                    _move_data_out(mid_cl_1_cr_2_args)

            mid_gm_end_offset = mid_lp_cnt * cr_out_idx_1_dst_asize + last_gm_end_offset
            mid_ub_end_offset = mid_lp_cnt * renew_cr_0_size * c0_size + last_ub_end_offset
            with tik_inst.if_scope(cur_cr_cl_size > 0):
                cur_cl_1_cr_2_args = (tik_inst, dst_out_gm, offset_base + cr_offset + mid_gm_end_offset, dst_ub,
                                      mid_ub_end_offset, plp_cl_size, cur_cr_cl_size * c0_size, cl_step_out,
                                      plp_cr_size * c0_size, ele_per_block)
                _move_data_out(cur_cl_1_cr_2_args)

        with tik_inst.if_scope(tik.all(cl_dims == 1, cr_dims == 1)):
            cl_1_cr_1_args = (tik_inst, dst_out_gm, offset_base, dst_ub, 0, plp_cl_size,
                              plp_cr_size * c0_size, cl_step_out, plp_cr_size * c0_size, ele_per_block)
            _move_data_out(cl_1_cr_1_args)

        with tik_inst.elif_scope(tik.all(cl_dims == 2, cr_dims == 1)):
            cl_args = (cl_offset, cl_base, cl_out_idx_0_size, cl_out_idx_0_dst_rsize, cl_out_idx_0_dst_asize,
                       cl_out_idx_1_size, cl_out_idx_1_dst_rsize, cl_out_idx_1_dst_asize)
            _update_out_offset_cl(cl_args)

            split_cl_args = (tik_inst, cl_out_idx_0_size, last_cr_cl_size,
                             mid_lp_cnt, cur_cr_cl_size, cl_base, plp_cl_size)
            _split_dims(split_cl_args)
            two_cl_args = (tik_inst, dst_out_gm, offset_base + cl_offset, dst_ub, 0, last_cr_cl_size,
                           is_last_cr_cl_nz, mid_lp_cnt, cur_cr_cl_size, plp_cr_size * c0_size,
                           cl_out_idx_0_dst_asize, cl_out_idx_0_size, cl_out_idx_1_dst_asize,
                           renew_cl_0_size, ele_per_block)
            _inner_move_data_out_target_two_cl(two_cl_args)

        with tik_inst.elif_scope(tik.all(cl_dims == 1, cr_dims == 2)):
            cr_idx_args = (tik_inst, block_idx, mc_pos, cr_begin, 0, cr_lp_idx, nlc_cr_lp_cnt, cr_lp_unit)
            _count_out_cr_idx(cr_idx_args)
            cr_args = (cr_offset, cr_begin, cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
                       cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize)
            _update_out_offset_cr(cr_args)

            split_cr_args = (tik_inst, cr_out_idx_0_size, last_cr_cl_size,
                             mid_lp_cnt, cur_cr_cl_size, cr_begin, plp_cr_size)
            _split_dims(split_cr_args)
            _inner_move_data_out_srcr1st_dstr2nd_same()


def _func_transform_100(tensor_args, tp_args):
    """
    transform function for tiling mode 100
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, dst_ub, ele_per_block, in_dtype = tensor_args
    (tiling_mode, ub_offset, mc_pos, used_core_cnt, core_step_in, core_step_out, vnc_col_size, c_mod_c0, c0_size,
     cl_dims, cr_dims, r1st_src_r2nd_dst_same, cl_step_in, cl_step_out, cl_lp_unit, cl_lp_step_in, cl_lp_step_out,
     c_step_in, c_lp_unit, c_lp_step_in, c_lp_step_out, cr_step_in, cr_step_out, cr_lp_unit, cr_lp_step_in,
     cr_lp_step_out, nlc_cl_lp_cnt, nlc_cl_left, nlc_c_lp_cnt, nlc_c_left, nlc_cr_lp_cnt, nlc_cr_left, lc_cl_lp_cnt,
     lc_cl_left, lc_c_lp_cnt, lc_c_left, lc_cr_lp_cnt, lc_cr_left,
     cl_out_idx_0_size, cl_out_idx_0_dst_rsize, cl_out_idx_0_dst_asize, cl_out_idx_1_size,
     cl_out_idx_1_dst_rsize, cl_out_idx_1_dst_asize, cr_out_idx_0_size, cr_out_idx_0_dst_rsize,
     cr_out_idx_0_dst_asize, cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize) = tp_args

    def _inner_func(args):
        cl_lp_cnt, cl_left, c_lp_cnt, c_left, cr_lp_cnt, cr_left = args
        plp_cl_size = tik_inst.Scalar(name="plp_cl_size")
        plp_c_size = tik_inst.Scalar(name="plp_c_size")
        plp_cr_size = tik_inst.Scalar(name="pln_cr_size")
        nout_lp_cnt = tik_inst.Scalar(name="nout_lp_cnt")
        dmp_flag = tbe_platform.api_check_support("tik.data_move_pad", in_dtype)

        with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:
            with tik_inst.if_scope(tik.any(c_lp_idx != c_lp_cnt - 1, c_left == 0)):
                plp_c_size.set_as(c_lp_unit)
            with tik_inst.else_scope():
                plp_c_size.set_as(c_left)

            with tik_inst.for_range(0, cl_lp_cnt) as cl_lp_idx:
                with tik_inst.if_scope(tik.any(cl_lp_idx != cl_lp_cnt - 1, cl_left == 0)):
                    plp_cl_size.set_as(cl_lp_unit)
                with tik_inst.else_scope():
                    plp_cl_size.set_as(cl_left)
                with tik_inst.if_scope(tik.all(tiling_mode == 1001, r1st_src_r2nd_dst_same == 0)):
                    nout_lp_cnt.set_as(plp_cl_size)  # for NCHW -> C1HWNoNic0
                with tik_inst.else_scope():
                    nout_lp_cnt.set_as(1)  # for NCHW -> NC1HWC0

                with tik_inst.for_range(0, cr_lp_cnt) as cr_lp_idx:
                    with tik_inst.if_scope(tik.any(cr_lp_idx != cr_lp_cnt - 1, cr_left == 0)):
                        plp_cr_size.set_as(cr_lp_unit)
                    with tik_inst.else_scope():
                        plp_cr_size.set_as(cr_left)

                    in_offset_args = (cl_lp_idx, cl_lp_step_in, c_lp_idx, c_lp_step_in,
                                      cr_lp_idx, cr_lp_step_in, block_idx * core_step_in)
                    in_gm_offset = _update_input_offset(in_offset_args)
                    with tik_inst.for_range(0, nout_lp_cnt) as nout_lp_idx:
                        in_gm_offset = in_gm_offset + nout_lp_idx * cl_step_in
                        copy_in_args = (tik_inst, src_in_gm, src_ub, mc_pos, in_gm_offset, vnc_col_size,
                                        tiling_mode, plp_cl_size, cl_step_in, plp_c_size, c_step_in,
                                        cr_lp_cnt, plp_cr_size, ele_per_block)
                        if dmp_flag is False:
                            _copy_data_in(copy_in_args)
                        else:
                            _copy_data_in_with_dmp(copy_in_args)
                        vnc_args = (tik_inst, src_ub, mc_pos, dst_ub, nout_lp_idx * c0_size, vnc_col_size,
                                    plp_c_size, r1st_src_r2nd_dst_same, cl_lp_idx, plp_cl_size,
                                    cr_lp_cnt, plp_cr_size, c_mod_c0, c0_size, ele_per_block)
                        with tik_inst.if_scope(tiling_mode == 1000):
                            _twice_vnchwconv_invert(vnc_args)
                        with tik_inst.else_scope():
                            _once_vnchwconv_invert(vnc_args)

                    with tik_inst.if_scope(r1st_src_r2nd_dst_same == 1):
                        out_gm_args = (cl_lp_unit, nlc_cl_lp_cnt, cl_lp_idx, cl_lp_step_out, c_lp_idx, c_lp_step_out,
                                       cr_lp_unit, nlc_cr_lp_cnt, cr_lp_idx, cr_lp_step_out, block_idx * core_step_out,
                                       cl_out_idx_0_size, cl_out_idx_0_dst_rsize, cl_out_idx_0_dst_asize,
                                       cl_out_idx_1_size, cl_out_idx_1_dst_rsize, cl_out_idx_1_dst_asize,
                                       cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
                                       cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize)
                        copy_out_args = (tik_inst, dst_out_gm, dst_ub, mc_pos, block_idx, cl_dims, cr_dims,
                                         plp_cl_size, cl_step_out, plp_cr_size, cr_step_out, c0_size, ele_per_block)
                        _copy_data_out_1st_src_r2nd_dst_same(out_gm_args, copy_out_args)
                    with tik_inst.else_scope():
                        out_gm_args = (cl_lp_idx, cl_lp_step_out, c_lp_idx, c_lp_step_out, cr_lp_unit,
                                       nlc_cr_lp_cnt, cr_lp_idx, cr_lp_step_out, block_idx * core_step_out,
                                       cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
                                       cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize)
                        copy_out_args = (tik_inst, dst_out_gm, dst_ub, mc_pos, block_idx, cl_dims, cr_dims,
                                         plp_cl_size, plp_cr_size, cr_step_out, c0_size, ele_per_block)
                        _copy_data_out_1st_src_r2nd_dst_not_same(out_gm_args, copy_out_args)

    with tik_inst.if_scope(block_idx != used_core_cnt - 1):
        nlc_args = nlc_cl_lp_cnt, nlc_cl_left, nlc_c_lp_cnt, nlc_c_left, nlc_cr_lp_cnt, nlc_cr_left
        _inner_func(nlc_args)
    with tik_inst.else_scope():
        lc_args = lc_cl_lp_cnt, lc_cl_left, lc_c_lp_cnt, lc_c_left, lc_cr_lp_cnt, lc_cr_left
        _inner_func(lc_args)


def _check_input_output_n_same(in_shape, in_format, out_format):
    """
    check the axis n of input and output is same or not
    """

    in_format_upper = in_format.upper()
    out_format_upper = out_format.upper()
    axis_n = 1

    if (in_format_upper, out_format_upper) in [("NCHW", "NC1HWC0"), ("NCDHW", "NDC1HWC0")]:
        axis_n = tdc.NI_16
    elif (in_format_upper, out_format_upper) in [("ND", "FRACTAL_Z"), ("HWCN", "FRACTAL_Z"), ("DHWCN", "FRACTAL_Z_3D")]:
        axis_n = in_shape[-1]
    elif (in_format_upper, out_format_upper) in [("NCDHW", "FRACTAL_Z_3D"), ("NCHW", "FRACTAL_Z")]:
        axis_n = in_shape[0]

    is_not_padding_n = axis_n % tdc.NI_16 == 0

    return is_not_padding_n


def trans_data_positive_source_ntc(src, dst, src_format, dst_format, kernel_name="trans_data_positive_source_ntc"):
    """
    positive transform for last dimension of source format is not c

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst : dict
        dtype of output should be same type as input
    src_format: str
        source data format, can be ND, NCHW etc.
    dst_format: str
        target data format, can be NC1HWC0, FRACTAL_Z etc.
    kernel_name : str
        kernel name, default value is "trans_data_positive_source_ntc"

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
    src_in_gm = tik_inst.Tensor(in_dtype, in_shape, tbe_platform.scope_gm, "src_in_gm")
    if _check_input_output_n_same(in_shape, src_format, dst_format):
        dst_out_gm = tik_inst.Tensor(
            in_dtype, out_shape, tbe_platform.scope_gm, "dst_out_gm", is_atomic_add=False)
    else:
        dst_out_gm = tik_inst.Tensor(
            in_dtype, out_shape, tbe_platform.scope_gm, "dst_out_gm", is_atomic_add=True)
    half_ub = ub_size // 2
    src_ub = tik_inst.Tensor(in_dtype, (half_ub,), tbe_platform.scope_ubuf, "src_ub")
    dst_ub = tik_inst.Tensor(in_dtype, (half_ub,), tbe_platform.scope_ubuf, "dst_ub")
    used_core_cnt = tiling_params[3]
    with tik_inst.for_range(0, tdc.get_core_num(), block_num=tdc.get_core_num()) as block_idx:
        with tik_inst.if_scope(block_idx < used_core_cnt):
            tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, dst_ub, block_elem_cnt, in_dtype]
            tp_args = tiling_params
            _func_transform_100(tensor_args, tp_args)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[src_in_gm], outputs=[dst_out_gm],
                      config={"dynamic_tik": True, "out_of_bound_sync_check": True, "enable_s64_to_s32": True})
