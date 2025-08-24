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
trans_data_rnn_positive_source
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl import trans_data_common_func as tdc
from impl.util.platform_adapter import error_manager_vector

# used for scalar
PAD_IDX_LIST = (0, 1)
# frame up levels
FRAME_LEVEL = 2


# 'pylint: disable=too-many-locals, inconsistent-return-statements, too-many-return-statements, too-many-statements
def _renew_input_output_shape_format(in_shape, out_shape, hidden_size):
    """
    renew shape and format to adapt tiling process
    """

    in_format_new = "HCN"
    out_format_new = "HCNT"
    axis_h, axis_c, axis_n = 1, in_shape[0], in_shape[1]
    hidden_cnt = axis_n // hidden_size
    axis_c0 = out_shape[-1]
    axis_ni = tdc.NI_16
    axis_c1 = tdc.ceil_div(axis_c, axis_c0)
    axis_no = tdc.ceil_div(hidden_cnt * tdc.ceil_fill(hidden_size, axis_ni), axis_ni)
    in_shape_new = [axis_h] + [axis_c] + [axis_n]
    out_shape_new = [axis_h] + [axis_c1] + [axis_no * axis_ni] + [axis_c0]
    new_params_zn = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

    return new_params_zn


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

    in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size, in_dtype, hidden_size = args
    c0_len = out_shape[-1]  # axis c0
    tp_names = locals()

    # get tiling params for using vnchwconv
    half_ub_size = ub_size // 2 if c0_len == tdc.C0_16 else ub_size // 4  # for int8, c0 is 32
    one_vnc_line_size = half_ub_size // tdc.VNC_LINES // block_elem_cnt * block_elem_cnt
    tmp_ub_offset = one_vnc_line_size * tdc.VNC_LINES
    tp_100_ub_offset = tmp_ub_offset if c0_len == tdc.C0_16 else tmp_ub_offset * 2
    tp_100_vnc_line_size = one_vnc_line_size - 64
    tp_100_c0_size = c0_len

    # axis c-right tiling parameters
    tp_100_cr_dims = FRAME_LEVEL
    tp_100_r1st_src_r2nd_dst_same = 1  # such as HWCN -> C1HWNoNiC0
    axis_src_cr_size = tdc.get_shape_size(in_shape[src_format.index("C") + 1:])
    tp_100_hidden_cnt = axis_src_cr_size // hidden_size
    axis_dst_cr_size = tdc.get_shape_size(out_shape[dst_format.index("C") + 1:]) // c0_len
    tmp_src_cr_lp_unit = tp_100_vnc_line_size // c0_len // block_elem_cnt * block_elem_cnt
    if tmp_src_cr_lp_unit >= axis_dst_cr_size:
        tp_100_tiling_mode = 1000
        tp_100_src_cr_lp_unit = axis_src_cr_size
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
            tp_names["tp_100_cr_out_idx_" + str(idx) + "_size"] = 0
            tp_names["tp_100_cr_out_idx_" + str(idx) + "_dst_rsize"] = 0
            tp_names["tp_100_cr_out_idx_" + str(idx) + "_dst_asize"] = 0
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
    axis_src_c_size = in_shape[src_c_idx]
    tp_100_src_c_lp_unit = c0_len
    src_c_lp_cnt = tdc.ceil_div(axis_src_c_size, tp_100_src_c_lp_unit)
    tp_100_src_c_step_in = tdc.get_shape_size(in_shape[src_c_idx + 1:])
    tp_100_src_c_lp_step_in = tdc.get_shape_size([tp_100_src_c_lp_unit] + in_shape[src_c_idx + 1:])
    tp_100_src_c_lp_step_out = tdc.get_shape_size(out_shape[dst_format.index("C") + 1:])
    tp_100_c_mod_c0 = axis_src_c_size % c0_len

    # axis left parameters
    tp_100_cl_dims = FRAME_LEVEL
    axis_src_cl_size = tdc.get_shape_size(in_shape[:src_format.index("C")])
    if tp_100_tiling_mode == 1000 or (tp_100_r1st_src_r2nd_dst_same == 0 and
                                      tp_100_tiling_mode == 1001 and tp_100_cr_dims != 1):
        tp_100_src_cl_lp_unit = tdc.NI_16 if axis_src_cl_size > tdc.NI_16 else axis_src_cl_size
    else:
        tp_100_src_cl_lp_unit = 1
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
            tp_names["tp_100_cl_out_idx_" + str(idx) + "_size"] = 0
            tp_names["tp_100_cl_out_idx_" + str(idx) + "_dst_rsize"] = 0
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
                         tp_100_cr_dims, tp_100_r1st_src_r2nd_dst_same, tp_100_hidden_cnt,
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

    in_shape, out_shape, hidden_size, block_elem_cnt, ub_size, in_dtype = args

    (in_shape_new, out_shape_new,
     in_format_new, out_format_new) = _renew_input_output_shape_format(in_shape, out_shape, hidden_size)
    args_get_tp = (in_shape_new, out_shape_new, in_format_new, out_format_new, block_elem_cnt, ub_size, in_dtype,
                   hidden_size)
    tiling_params = _tiling_params_positive(args_get_tp)

    return tiling_params


# 'pylint: disable=unused-variable
def _twice_vnchwconv_invert(args):
    """
    do ncdh to ndhc transform by twice vnchwconv
    """

    (tik_inst, src_ub, mc_pos, ub_offset, vnc_col_size, hidden_cnt, hidden_size, plp_c_size,
     r1st_src_r2nd_dst_same, plp_cl_size, cr_lp_cnt, cr_lp_idx, cr_lp_unit, plp_cr_size, c_mod_c0, c0_size,
     ele_per_block, hidden_output, core_step_in) = args
    tensor_dtype = src_ub.dtype.lower()
    size_factor = tdc.get_dtype_factor(tensor_dtype)
    src_ub_casted = src_ub
    ub_offset_casted = ub_offset * size_factor
    vnc_col_len = vnc_col_size * size_factor

    # do ncdh -> cdhn
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        hidden_ceil = tik_inst.Scalar()
        hidden_ceil.set_as(tdc.ceil_fill(hidden_size, c0_size))
        plp_cr_block_align_size = tik_inst.Scalar(name="plp_cr_block_align_size")
        with tik_inst.if_scope(tik.all(cr_lp_cnt == 1, mc_pos != 2)):
            plp_cr_block_align_size.set_as(plp_cr_size)
        with tik_inst.else_scope():
            plp_cr_block_align_size.set_as(tdc.ceil_fill(plp_cr_size, ele_per_block))

        repeat_cnt = tdc.ceil_div(plp_c_size * plp_cr_block_align_size * size_factor, c0_size)
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(1)
            dst_stride.set_as(16)
        src_addr_list = [src_ub_casted[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
        dst_addr_list = [src_ub_casted[ub_offset_casted + c0_size * i] for i in tdc.ADDR_IDX_LIST]
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

        # padding zero
        clean_len = hidden_cnt * tdc.ceil_fill(hidden_size, c0_size) * tdc.ceil_fill(plp_c_size, c0_size) * c0_size
        tdc.clean_ubuf(tik_inst, src_ub, 0, clean_len)
        # do cdhn -> dhcn
        with tik_inst.for_range(0, plp_c_size) as c_idx:
            with tik_inst.for_range(0, hidden_cnt) as hidden_idx:
                tik_inst.data_move(src_ub[(hidden_ceil * hidden_idx * c0_size + c_idx) * c0_size],
                                   src_ub[ub_offset + hidden_size * (hidden_cnt * c_idx + hidden_idx) * c0_size],
                                   0, hidden_size, 1 * size_factor, 0, (c0_size - 1) * size_factor)

        # do dhcn -> ndhc or dhcn -> dhnc
        repeat_cnt = hidden_cnt * hidden_ceil * size_factor
        vnc_row_size = repeat_cnt * c0_size
        src_addr_list = [src_ub_casted[tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
        dst_addr_list = [src_ub_casted[ub_offset_casted + vnc_row_size * i] for i in tdc.ADDR_IDX_LIST]

        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(16)
            dst_stride.set_as(1)
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        hidden_output.set_as(vnc_row_size)


# 'pylint: disable=unused-variable
def _once_vnchwconv_invert(args):
    """
    do cdh to dhc transform by once vnchwconv
    """

    (tik_inst, src_ub, mc_pos, ub_offset, vnc_col_size, hidden_cnt, hidden_size, plp_c_size,
     r1st_src_r2nd_dst_same, plp_cl_size, cr_lp_cnt, cr_lp_idx, cr_lp_unit, plp_cr_size, c_mod_c0, c0_size,
     ele_per_block, hidden_output, core_step_in) = args
    repeat_cnt = tdc.ceil_div(plp_cr_size, c0_size)

    # do cdh -> dhc
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        dst_gap = tik_inst.Scalar()
        with tik_inst.if_scope(plp_c_size % c0_size > 0):  # padding zero
            tdc.clean_ubuf(tik_inst, src_ub, c_mod_c0 * vnc_col_size, (c0_size - c_mod_c0) * vnc_col_size)

        with tik_inst.if_scope(r1st_src_r2nd_dst_same == 1):  # for NCHW -> NC1HWC0
            dst_gap.set_as(c0_size)
            dst_stride.set_as(16)
        with tik_inst.else_scope():  # for NCHW -> C1HWNoNiC0
            dst_gap.set_as(plp_cl_size * c0_size)
            dst_stride.set_as(plp_cl_size * 16)

        src_addr_list = [src_ub[vnc_col_size * i] for i in tdc.ADDR_IDX_LIST]
        dst_addr_list = [src_ub[ub_offset + dst_gap * i] for i in tdc.ADDR_IDX_LIST]
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(1)
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        #move for noni
        hidden_left = tik_inst.Scalar(dtype="int64")
        hidden_offset = tik_inst.Scalar(dtype="int64")
        hidden_ceil = tik_inst.Scalar(dtype="int64")
        hidden_head_len = tik_inst.Scalar(dtype="int64")
        hidden_head_output = tik_inst.Scalar(dtype="int64")
        hidden_middle_output = tik_inst.Scalar(dtype="int64")
        hidden_tail_output = tik_inst.Scalar(dtype="int64")
        hidden_mid_start = tik_inst.Scalar(dtype="int64")
        hidden_mid_cnt = tik_inst.Scalar(dtype="int64")
        hidden_tail_start = tik_inst.Scalar(dtype="int64")
        hidden_tail_end = tik_inst.Scalar(dtype="int64")
        hidden_cr_output = tik_inst.Scalar(dtype="int64")
        hidden_left.set_as((core_step_in + cr_lp_unit * cr_lp_idx) % hidden_size)
        hidden_offset.set_as(0)
        with tik_inst.if_scope(hidden_size % c0_size != 0):
            hidden_offset.set_as(c0_size - hidden_size % c0_size)
        hidden_ceil.set_as(tdc.ceil_fill(hidden_size, c0_size))
        with tik_inst.if_scope((hidden_size - hidden_left) > plp_cr_size):
            hidden_output.set_as(plp_cr_size * c0_size)
        with tik_inst.else_scope():
            with tik_inst.if_scope((hidden_size - hidden_left) < plp_cr_size):
                hidden_head_len.set_as(0)
                hidden_head_output.set_as(0)
                hidden_mid_cnt.set_as(0)
                hidden_tail_end.set_as(0)
                with tik_inst.if_scope(hidden_left > 0):
                    hidden_head_len.set_as(hidden_size - hidden_left)
                with tik_inst.if_scope(hidden_head_len > 0):
                    hidden_head_output.set_as(hidden_head_len + hidden_offset)
                hidden_mid_start.set_as(hidden_head_len)
                with tik_inst.if_scope(plp_cr_size - hidden_mid_start >= hidden_size):
                    hidden_mid_cnt.set_as((plp_cr_size - hidden_mid_start) // hidden_size)
                hidden_middle_output.set_as(hidden_mid_cnt * hidden_ceil)
                hidden_tail_start.set_as(hidden_mid_start + hidden_mid_cnt * hidden_size)
                with tik_inst.if_scope(plp_cr_size > hidden_tail_start):
                    hidden_tail_end.set_as(plp_cr_size - hidden_tail_start)
                hidden_tail_output.set_as(0)
                with tik_inst.if_scope(hidden_tail_end > 0):
                    hidden_tail_output.set_as(hidden_tail_end)
                hidden_cr_output.set_as(hidden_head_output + hidden_middle_output + hidden_tail_output)
                clean_len = hidden_cr_output * c0_size
                tdc.clean_ubuf(tik_inst, src_ub, 0, clean_len)
                #head
                with tik_inst.if_scope(hidden_head_len > 0):
                    tik_inst.data_move(src_ub, src_ub[ub_offset], 0, 1, hidden_head_len, 0, 0)
                #middle
                with tik_inst.for_range(0, hidden_mid_cnt) as hidden_mid_idx:
                    tik_inst.data_move(src_ub[(hidden_head_output + hidden_mid_idx * hidden_ceil) * c0_size],
                                       src_ub[ub_offset + (hidden_mid_start + hidden_mid_idx * hidden_size) * c0_size],
                                       0, 1, hidden_size, 0, 0)
                #tail
                with tik_inst.if_scope(hidden_tail_end > 0):
                    tik_inst.data_move(src_ub[(hidden_head_output + hidden_mid_cnt * hidden_ceil) * c0_size],
                                       src_ub[ub_offset + hidden_tail_start * c0_size],
                                       0, 1, hidden_tail_end, 0, 0)
                hidden_output.set_as(hidden_cr_output * c0_size)
            with tik_inst.else_scope():
                hidden_cr_output.set_as(plp_cr_size + hidden_offset)
                clean_len = hidden_cr_output * c0_size
                tdc.clean_ubuf(tik_inst, src_ub, 0, clean_len)
                tik_inst.data_move(src_ub, src_ub[ub_offset], 0, 1, plp_cr_size, 0, 0)
                hidden_output.set_as(hidden_cr_output * c0_size)
            #remove
            tik_inst.data_move(src_ub[ub_offset], src_ub, 0, 1, hidden_output // ele_per_block, 0, 0)


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

    (tik_inst, src_in_gm, src_ub, mc_pos, in_gm_offset, vnc_col_size,
     tiling_mode, plp_cl_size, cl_step_in, plp_c_size, c_step_in, cr_lp_cnt, plp_cr_size, ele_per_block) = args
    cr_block_align_size = tdc.ceil_fill(plp_cr_size, ele_per_block)

    with tik_inst.if_scope(tiling_mode == 1000):  # for two times vnchwconv case
        with tik_inst.for_range(0, plp_cl_size) as cl_idx:
            with tik_inst.if_scope(tik.all(cr_lp_cnt == 1, mc_pos != 2)):
                tik_inst.data_move(src_ub[cl_idx * vnc_col_size], src_in_gm[in_gm_offset + cl_idx * cl_step_in],
                                   0, 1, tdc.ceil_div(plp_c_size * plp_cr_size, ele_per_block), 0, 0)
            with tik_inst.else_scope():
                with tik_inst.for_range(0, plp_c_size) as c_idx:
                    tik_inst.data_move(src_ub[cl_idx * vnc_col_size + c_idx * cr_block_align_size],
                                       src_in_gm[in_gm_offset + cl_idx * cl_step_in + c_idx * c_step_in],
                                       0, 1, tdc.ceil_div(plp_cr_size, ele_per_block), 0, 0)
    with tik_inst.else_scope():  # for one time vnchwconv case
        with tik_inst.for_range(0, plp_c_size) as c_idx_1:
            tik_inst.data_move(src_ub[c_idx_1 * vnc_col_size], src_in_gm[in_gm_offset + c_idx_1 * c_step_in],
                               0, 1, tdc.ceil_div(plp_cr_size, ele_per_block), 0, 0)


def calc_cr_offset(step_in, in_idx, hidden_size, c0_size, cr_offset):
    hidden_ceil = tdc.ceil_fill(hidden_size, c0_size)
    cr_offset_cnt = (step_in * in_idx) // hidden_size
    cr_offset.set_as(cr_offset_cnt * (hidden_ceil - hidden_size) * c0_size)


def calc_cr_offset_with_core_offset(offset_args):
    step_in, in_idx, hidden_size, c0_size, core_step_in, cr_offset = offset_args
    hidden_ceil = tdc.ceil_fill(hidden_size, c0_size)
    cr_offset_cnt = (core_step_in + step_in * in_idx) // hidden_size
    cr_offset.set_as(cr_offset_cnt * (hidden_ceil - hidden_size) * c0_size)


# 'pylint: disable=unused-variable
def _copy_data_out(out_offset_args, copy_out_args):
    """
    copy data from ub to gm
    """

    (cl_lp_unit, nlc_cl_lp_cnt, cl_lp_idx, cl_lp_step_out, c_lp_idx, c_lp_step_out,
     nlc_cr_lp_cnt, cr_lp_idx, cr_lp_step_out, core_step_out,
     cl_out_idx_0_size, cl_out_idx_0_dst_rsize, cl_out_idx_0_dst_asize,
     cl_out_idx_1_size, cl_out_idx_1_dst_rsize, cl_out_idx_1_dst_asize,
     cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
     cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize) = out_offset_args
    (tik_inst, dst_out_gm, src_ub, mc_pos, block_idx,
     r1st_src_r2nd_dst_same, cl_dims, cr_dims, plp_cl_size,
     cl_step_out, plp_cr_size, cr_step_out, c0_size, ele_per_block, hidden_output) = copy_out_args
    offset_base = (cl_lp_idx * cl_lp_step_out + c_lp_idx * c_lp_step_out + cr_lp_step_out + core_step_out)

    with tik_inst.new_stmt_scope(disable_sync=True):
        tik_inst.data_move(dst_out_gm[offset_base], src_ub, 0, 1, hidden_output // ele_per_block, 0, 0)


def _func_transform_100(tensor_args, tp_args):
    """
    transform function for tiling mode 100
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, ele_per_block, hidden_size, in_dtype = tensor_args
    (tiling_mode, ub_offset, mc_pos, used_core_cnt, core_step_in, core_step_out, vnc_col_size, c_mod_c0, c0_size,
     cl_dims, cr_dims, r1st_src_r2nd_dst_same, hidden_cnt, cl_step_in,
     cl_step_out, cl_lp_unit, cl_lp_step_in, cl_lp_step_out,
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
        hidden_len = tik_inst.Scalar(name="hidden_len")
        hidden_output = tik_inst.Scalar(name="hidden_output")
        hidden_core_step_out = tik_inst.Scalar(name="hidden_core_step_out")
        hidden_dst_cr_lp_step_out = tik_inst.Scalar(name="hidden_dst_cr_lp_step_out")
        cr_core_offset = tik_inst.Scalar(name="cr_core_offset")
        cr_lp_offset = tik_inst.Scalar(name="cr_lp_offset")
        hidden_output.set_as(0)
        hidden_len.set_as(hidden_size)
        calc_cr_offset(core_step_in, block_idx, hidden_size, c0_size, cr_core_offset)

        with tik_inst.for_range(0, cl_lp_cnt) as cl_lp_idx:
            with tik_inst.if_scope(tik.any(cl_lp_idx != cl_lp_cnt - 1, cl_left == 0)):
                plp_cl_size.set_as(cl_lp_unit)
            with tik_inst.else_scope():
                plp_cl_size.set_as(cl_left)
            with tik_inst.if_scope(tik.all(tiling_mode == 1001, r1st_src_r2nd_dst_same == 0)):
                nout_lp_cnt.set_as(plp_cl_size)  # for NCHW -> C1HWNoNic0
            with tik_inst.else_scope():
                nout_lp_cnt.set_as(1)  # for NCHW -> NC1HWC0

            with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:
                with tik_inst.if_scope(tik.any(c_lp_idx != c_lp_cnt - 1, c_left == 0)):
                    plp_c_size.set_as(c_lp_unit)
                with tik_inst.else_scope():
                    plp_c_size.set_as(c_left)

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
                        with tik_inst.new_stmt_scope(disable_sync=True):
                            _copy_data_in(copy_in_args)
                        with tik_inst.if_scope(mc_pos == 2):
                            offset_args = (cr_lp_step_in, cr_lp_idx, hidden_size, c0_size, block_idx * core_step_in,
                                           cr_lp_offset)
                            calc_cr_offset_with_core_offset(offset_args)
                            hidden_dst_cr_lp_step_out.set_as(cr_lp_idx * cr_lp_step_out + cr_lp_offset)
                            hidden_core_step_out.set_as(block_idx * core_step_out)
                        with tik_inst.else_scope():
                            calc_cr_offset(cr_lp_step_in, cr_lp_idx, hidden_size, c0_size, cr_lp_offset)
                            hidden_dst_cr_lp_step_out.set_as(cr_lp_idx * cr_lp_step_out + cr_lp_offset)
                            hidden_core_step_out.set_as(block_idx * core_step_out)
                        vnc_args = (tik_inst, src_ub, mc_pos, ub_offset + nout_lp_idx * c0_size,
                                    vnc_col_size, hidden_cnt, hidden_len, plp_c_size, r1st_src_r2nd_dst_same,
                                    plp_cl_size, cr_lp_cnt, cr_lp_idx, cr_lp_unit, plp_cr_size, c_mod_c0, c0_size,
                                    ele_per_block, hidden_output, block_idx * core_step_in)
                        with tik_inst.if_scope(tiling_mode == 1000):
                            _twice_vnchwconv_invert(vnc_args)
                        with tik_inst.else_scope():
                            _once_vnchwconv_invert(vnc_args)

                    out_gm_args = (cl_lp_unit, nlc_cl_lp_cnt, cl_lp_idx, cl_lp_step_out, c_lp_idx, c_lp_step_out,
                                   nlc_cr_lp_cnt, cr_lp_idx, hidden_dst_cr_lp_step_out, hidden_core_step_out,
                                   cl_out_idx_0_size, cl_out_idx_0_dst_rsize, cl_out_idx_0_dst_asize,
                                   cl_out_idx_1_size, cl_out_idx_1_dst_rsize, cl_out_idx_1_dst_asize,
                                   cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
                                   cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize)
                    copy_out_args = (tik_inst, dst_out_gm, src_ub[ub_offset], mc_pos, block_idx,
                                     r1st_src_r2nd_dst_same, cl_dims, cr_dims, plp_cl_size,
                                     cl_step_out, plp_cr_size, cr_step_out, c0_size, ele_per_block, hidden_output)
                    _copy_data_out(out_gm_args, copy_out_args)

    with tik_inst.if_scope(block_idx != used_core_cnt - 1):
        nlc_args = nlc_cl_lp_cnt, nlc_cl_left, nlc_c_lp_cnt, nlc_c_left, nlc_cr_lp_cnt, nlc_cr_left
        _inner_func(nlc_args)
    with tik_inst.else_scope():
        lc_args = lc_cl_lp_cnt, lc_cl_left, lc_c_lp_cnt, lc_c_left, lc_cr_lp_cnt, lc_cr_left
        _inner_func(lc_args)


def trans_data_rnn_positive_source(src, dst, input_size, hidden_size, kernel_name="trans_data_rnn_positive_source"):
    """

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst : dict
        dtype of output should be same type as input
    input_size: int
    hiddden_size: int
    kernel_name : str
        kernel name, default value is "trans_data_positive_source_ntc"

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
    dst_out_gm = tik_inst.Tensor(
        in_dtype, (tdc.MAX_INT64_VALUE,), tbe_platform.scope_gm, "dst_out_gm", is_atomic_add=True)
    src_ub = tik_inst.Tensor(in_dtype, (ub_size,), tbe_platform.scope_ubuf, "src_ub")
    if in_shape[0] == input_size or in_shape[0] == hidden_size:
        args = in_shape, out_shape, hidden_size, block_elem_cnt, ub_size, in_dtype
        # get tiling parameters
        tiling_params = _get_tiling_params_func(args)
        used_core_cnt = tiling_params[3]
        with tik_inst.for_range(0, tdc.get_core_num(), block_num=tdc.get_core_num()) as block_idx:
            with tik_inst.if_scope(block_idx < used_core_cnt):
                tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, block_elem_cnt, hidden_size,
                               in_dtype]
                tp_args = tiling_params
                _func_transform_100(tensor_args, tp_args)
    else:
        if in_shape[0] != input_size + hidden_size:
            error_detail = "the shape of in_shape[0] is not satisfied."
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "in_shape", error_detail)
        input0_shape = [input_size, in_shape[1]]
        output0_size = tdc.ceil_div(input_size, tdc.C0_16)
        output0_shape = [output0_size, out_shape[1], out_shape[2], out_shape[3]]
        args = input0_shape, output0_shape, hidden_size, block_elem_cnt, ub_size, in_dtype
        tiling_params0 = _get_tiling_params_func(args)
        used_core_cnt0 = tiling_params0[3]
        input1_shape = [hidden_size, in_shape[1]]
        output1_size = tdc.ceil_div(hidden_size, tdc.C0_16)
        output1_shape = [output1_size, out_shape[1], out_shape[2], out_shape[3]]
        args = input1_shape, output1_shape, hidden_size, block_elem_cnt, ub_size, in_dtype
        tiling_params1 = _get_tiling_params_func(args)
        used_core_cnt1 = tiling_params1[3]
        with tik_inst.for_range(0, tdc.get_core_num(), block_num=tdc.get_core_num()) as block_idx:
            with tik_inst.if_scope(block_idx < used_core_cnt0):
                tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, block_elem_cnt, hidden_size,
                               in_dtype]
                tp_args = tiling_params0
                _func_transform_100(tensor_args, tp_args)
            with tik_inst.if_scope(block_idx < used_core_cnt1):
                tensor_args = [tik_inst, block_idx,
                               src_in_gm[input_size * in_shape[1]],
                               dst_out_gm[output0_size * out_shape[1] * out_shape[2] * out_shape[3]],
                               src_ub, block_elem_cnt, hidden_size, in_dtype]
                tp_args = tiling_params1
                _func_transform_100(tensor_args, tp_args)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[src_in_gm], outputs=[dst_out_gm])
