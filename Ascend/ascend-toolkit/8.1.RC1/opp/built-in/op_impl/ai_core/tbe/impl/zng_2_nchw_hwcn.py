#!/usr/bin/python
# -*- coding: utf-8 -*-
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
zng_2_nchw_hwcn
"""

import functools
import math
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik

# UB size by byte
UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
# AICORE count
CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
# C0 length except int8 and uint8
C0_16 = 16
# C0 length int8 and uint8
C0_32 = 32
# Ni length
NI_16 = 16
# bytes in one block
BLOCK_BYTE_SIZE = 32
# repeat up limit for vector command
REPEAT_LIMIT_VECT = 255
# repeat up limit for mte
REPEAT_LIMIT_MTE = 4095
# strides up limit for mte
STRIDE_LIMIT_MTE = 65535
# mask value for float32
MASK_64 = 64
# mask value for float16
MASK_128 = 128
# max int64 value
MAX_INT64_VALUE = 2 ** 63 - 1
# float type list
TYPE_FLOAT_LIST = ("float16", "float32")
# used for scalar
REG_IDX_LIST = (0, 1, 2, 3, 4, 5, 6, 7)
# used for vnchwconv
ADDR_IDX_LIST = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
# vnchwconv line count
VNC_LINES = 16


# 'pylint: disable=too-many-locals,invalid-name,unused-variable,too-many-statements,too-many-arguments
def _ceil_div(value_x, value_y):
    """
    do ceil division
    """

    result = (value_x + value_y - 1) // value_y

    return result


def _ceil_fill(value_x, value_y):
    """
    do ceil filling
    """

    result = (value_x + value_y - 1) // value_y * value_y

    return result


def _floor_div(value_x, value_y):
    """
    do floor division
    """

    result = value_x // value_y

    return result


def _lcm(value_x, value_y):
    """
    count lcm
    """

    result = value_x * value_y // math.gcd(value_x, value_y)

    return result


def _get_c0_len(dtype):
    """
    get c0 length according to dtype
    """

    c0_len = C0_32 if dtype.lower() in ("int8", "uint8", "bool") else C0_16

    return c0_len


def _get_dtype_len(in_dtype):
    """
    get the byte count of certain dtype
    """

    temp_dtype = in_dtype.lower()

    if temp_dtype in ("int8", "uint8"):
        byte_len = 1
    elif temp_dtype in ("float16", "int16", "uint16"):
        byte_len = 2
    elif temp_dtype in ("float32", "int32", "uint32"):
        byte_len = 4
    elif temp_dtype in ("int64", "uint64"):
        byte_len = 8

    return byte_len


def _get_max_element_in_ub(in_dtype, ub_part):
    """
    get the up limit elements in UB
    """

    byte_len = _get_dtype_len(in_dtype)
    ub_upper_limit = 248 * 1024 // ub_part if UB_SIZE > 248 * 1024 else UB_SIZE // ub_part  # the unit is Byte
    element_size = ub_upper_limit // byte_len

    return element_size


def _get_shape_size(sub_shape):
    """
    return shape size
    """

    shape_size = functools.reduce(lambda x, y: x * y, sub_shape)

    return shape_size


def get_tp_zng_2_nchw(args):
    """
    get tiling parameters for fractal_z_g to nchw

    Parameters
    ----------
    args : list
        args for get tiling parameters

    Returns
    -------
    tiling parameters
    """

    in_shape, out_shape, src_format, dst_format, c0_len, block_elem_cnt, ub_size, groups = args
    c_out, c, h, w = out_shape
    gc1hw, no, ni, c0 = in_shape
    n = c_out // groups
    vnc_line_size = (ub_size - NI_16 * c0) // 2 // VNC_LINES // block_elem_cnt * block_elem_cnt

    # get E tiling parameters
    tmp_e_size = _lcm(_lcm(c, c0) // c, _lcm(n, NI_16) // n)
    e_size = tmp_e_size if tmp_e_size <= groups else groups
    e_factor_1 = _get_shape_size([h, w, no, ni, c0])
    e_factor_2 = _get_shape_size([n, c0])

    # get hw tiling parameters
    hw_size = _get_shape_size([h, w])  # h, w confused as one dimension
    hw_step_in = _get_shape_size([no, ni, c0])
    tmp_hw_gate = (ub_size - NI_16 * c0) // 2 // (2 * c0 * c0)  # half UB, move in 2 * c0 * c0 one time
    hw_gate = tmp_hw_gate // block_elem_cnt * block_elem_cnt if hw_size > tmp_hw_gate else hw_size
    # for hw cannot move in one time, only support hw block align case now
    if hw_size > hw_gate and hw_size % block_elem_cnt > 0:
        error_manager_vector.raise_err_check_params_rules("zng_2_nchw", 'hw_size is not block align',
                                                          "hw_size", hw_size)
    hw_lp_cnt = _ceil_div(hw_size, hw_gate)
    hw_left = hw_size % hw_gate

    # get multiple core parameters
    g_size = _ceil_div(groups, e_size)
    used_core = _ceil_div(g_size, _ceil_div(g_size, CORE_NUM))
    nlc_g_cnt = _ceil_div(g_size, used_core)
    lc_g_cnt = g_size - nlc_g_cnt * (used_core - 1)
    g_step_in = _get_shape_size([gc1hw // g_size, no, ni, c0])
    g_step_out = _get_shape_size([e_size, n, c, h, w])
    core_step_in = g_step_in * nlc_g_cnt
    core_step_out = g_step_out * nlc_g_cnt

    # get n tiling parameters
    n_cube_size = VNC_LINES  # max value is 16
    n_loop_cnt = _ceil_div(n, n_cube_size)
    n_left_size = n % n_cube_size
    n_step_in = n_cube_size * c0
    n_step_out = n_cube_size * _get_shape_size(out_shape[1:])

    # get output tiling parameters
    tiling_params = [used_core, core_step_in, core_step_out, vnc_line_size] + \
                    [nlc_g_cnt, lc_g_cnt, g_step_in, g_step_out] + \
                    [e_size, e_factor_1, e_factor_2] + [c, c0, n, hw_size, hw_step_in, hw_lp_cnt, hw_gate, hw_left] + \
                    [n_loop_cnt, n_step_in, n_step_out, n_cube_size, n_left_size]

    return tiling_params


def _twice_vnchwconv_2_nchw(args):
    """
    do nhwc to nchw by two vnchwconv operations
    """

    tik_inst, src_ub, ub_in, ub_out, vnc_line_size, hw_size, c0_cnt, c0_size, sub_c_size, head_gap, dtype_factor = args
    src_ub_fp16 = src_ub.reinterpret_cast_to("float16")

    # do nhwc -> hwcn
    src_addr_list = [src_ub_fp16[(ub_in + vnc_line_size * i) * dtype_factor] for i in ADDR_IDX_LIST]
    dst_addr_list = [src_ub_fp16[ub_out * dtype_factor + C0_16 * i] for i in ADDR_IDX_LIST]
    repeat_cnt = _ceil_div(hw_size * c0_cnt * c0_size * dtype_factor, C0_16)
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(1)
            dst_stride.set_as(16)
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list,
                           repeat_cnt, dst_stride, src_stride)

    # do hwcn -> chwn
    with tik_inst.for_range(0, hw_size) as hw_idx_1:
        tik_inst.data_move(src_ub_fp16[(ub_in + hw_idx_1 * VNC_LINES) * dtype_factor],
                           src_ub_fp16[(ub_out + (hw_idx_1 * c0_cnt * c0_size + head_gap) * VNC_LINES) * dtype_factor],
                           0, sub_c_size, 1 * dtype_factor, 0, (hw_size - 1) * dtype_factor)

    # do chwn -> nchw
    repeat_cnt = _ceil_div(hw_size * sub_c_size * dtype_factor, C0_16)
    src_addr_list = [src_ub_fp16[ub_in * dtype_factor + C0_16 * i] for i in ADDR_IDX_LIST]
    dst_addr_list = [src_ub_fp16[ub_out * dtype_factor + repeat_cnt * C0_16 * i] for i in ADDR_IDX_LIST]
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(16)
            dst_stride.set_as(1)
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list,
                           repeat_cnt, dst_stride, src_stride)


def _move_data_in_nchw(args):
    """
    move data in from gm for fz to nchw
    """

    (tik_inst, src_gm, src_ub, ub_in, hw_size, level2_in_offset,
     hw_step_in, c0_cnt, c0_size, e_factor_1, in_cube, in_dst_stride, dtype_factor) = args

    with tik_inst.for_range(0, hw_size) as hw_idx:
        level1_in_offset = level2_in_offset + hw_idx * hw_step_in
        level1_ub_offset = hw_idx * c0_cnt * c0_size + ub_in
        with tik_inst.for_range(0, c0_cnt) as c0_idx:
            level0_in_offset = level1_in_offset + c0_idx * e_factor_1
            level0_ub_offset = level1_ub_offset + c0_idx * c0_size
            tik_inst.data_move(src_ub, src_gm[level0_in_offset], 0, 1, in_cube * dtype_factor, 0, 0)
            # move data to each vnchwconv line
            tik_inst.data_move(src_ub[level0_ub_offset], src_ub, 0, in_cube, 1 * dtype_factor, 0, in_dst_stride)


def _move_data_out_nchw(args):
    """
    move data out to gm for fz to nchw
    """

    (tik_inst, dst_gm, level2_out_offset, src_ub, ub_out,
     hw_size, sub_c_size, in_cube, c_lp_idx, c_lp_cnt, c_size, dtype, ele_per_block) = args

    burst_len = _ceil_div(hw_size * sub_c_size, ele_per_block)
    with tik_inst.if_scope(hw_size * sub_c_size % ele_per_block != 0):
        with tik_inst.for_range(0, in_cube) as n_idx:
            # to reduce scalar operation
            with tik_inst.if_scope(c_lp_idx != c_lp_cnt - 1):
                tik_inst.data_move(dst_gm[level2_out_offset + n_idx * c_size * hw_size],
                                   src_ub[ub_out + n_idx * burst_len * ele_per_block],
                                   0, 1, burst_len, 0, 0)
            with tik_inst.else_scope():
                chw_block_align = sub_c_size * hw_size // ele_per_block
                backward_len = sub_c_size * hw_size - ele_per_block
                tik_inst.data_move(dst_gm[level2_out_offset + n_idx * c_size * hw_size],
                                   src_ub[ub_out + n_idx * burst_len * ele_per_block],
                                   0, 1, chw_block_align, 0, 0)
                with tik_inst.new_stmt_scope():
                    tmp_reg = tik_inst.Scalar(name="tmp_reg", dtype=dtype)
                    with tik_inst.for_range(0, ele_per_block) as tmp_idx:
                        tmp_reg.set_as(src_ub[ub_out + n_idx * burst_len * ele_per_block +
                                              backward_len + tmp_idx])
                        src_ub[ub_out + tmp_idx].set_as(tmp_reg)
                    tik_inst.data_move(dst_gm[level2_out_offset +
                                              n_idx * c_size * hw_size + backward_len],
                                       src_ub[ub_out],
                                       0, 1, 1, 0, 0)

    with tik_inst.else_scope():
        with tik_inst.if_scope(sub_c_size != c_size):
            chw_size = c_size * hw_size
            with tik_inst.if_scope(chw_size % ele_per_block != 0):
                with tik_inst.for_range(0, in_cube) as n_idx_1:
                    tik_inst.data_move(dst_gm[level2_out_offset + n_idx_1 * c_size * hw_size],
                                       src_ub[ub_out + n_idx_1 * burst_len * ele_per_block],
                                       0, 1, burst_len, 0, 0)
            with tik_inst.else_scope():
                tik_inst.data_move(dst_gm[level2_out_offset], src_ub[ub_out],
                                   0, in_cube, burst_len, 0, chw_size // ele_per_block - burst_len)
        with tik_inst.else_scope():
            tik_inst.data_move(dst_gm[level2_out_offset], src_ub[ub_out],
                               0, 1, in_cube * burst_len, 0, 0)


def _move_data_out_nchw_hw_block_align(args):
    """
    move data out to gm for fz to nchw when hw block align
    """

    (tik_inst, dst_gm, level2_out_offset, src_ub, ub_out,
     hw_size, per_lp_hw, sub_c_size, in_cube, c_size, ele_per_block) = args

    with tik_inst.for_range(0, in_cube) as n_idx_1:
        tik_inst.data_move(dst_gm[level2_out_offset + n_idx_1 * c_size * hw_size],
                           src_ub[ub_out + n_idx_1 * sub_c_size * per_lp_hw],
                           0, sub_c_size, per_lp_hw // ele_per_block, 0, (hw_size - per_lp_hw) // ele_per_block)


def zng_nchw_transform(tensor_args, tiling_args):
    """
    do fractal_z_g to nchw transform

    Parameters
    ----------
    tensor_args : list
        tensor related parameters
    tiling_args: list
        tiling parameters

    Returns
    -------
    None
    """

    tik_inst, src_gm, dst_gm, src_ub, c_lp_cnt, c0_cnt, e_in_offset, \
    in_cube, sub_c_size, block_idx, dtype, head_gap, c_mod_c0, ele_per_block = tensor_args
    used_core, core_step_in, core_step_out, vnc_line_size, nlc_g_cnt, lc_g_cnt, g_step_in, g_step_out, \
    e_size, e_factor_1, e_factor_2, c_size, c0_size, n_size, hw_size, hw_step_in, hw_lp_cnt, hw_gate, hw_left, \
    n_loop_cnt, n_step_in, n_step_out, n_cube_size, n_left_size = tiling_args
    ub_in = NI_16 * c0_size  # start to save move in data here
    ub_out = vnc_line_size * VNC_LINES + ub_in
    dtype_factor = C0_16 // ele_per_block
    in_dst_stride = vnc_line_size // ele_per_block - 1 * dtype_factor
    per_lp_hw = tik_inst.Scalar(name="per_lp_hw")

    def _inner_transformer(g_cnt):
        """
        the transform process
        """

        with tik_inst.for_range(0, g_cnt) as g_idx:
            level5_in_offset = g_idx * g_step_in + block_idx * core_step_in
            level5_out_offset = g_idx * g_step_out + block_idx * core_step_out

            with tik_inst.for_range(0, e_size) as e_idx:
                e_in_offset.set_as(_floor_div(e_idx * c_size, c0_size) * e_factor_1 + e_idx * e_factor_2)
                c1_cnt = _ceil_div(e_idx * c_size % c0_size + c_size, c0_size)
                with tik_inst.if_scope(c1_cnt > 1):
                    c0_cnt.set_as(2)
                    c_lp_cnt.set_as(_ceil_div(c1_cnt, 2))
                with tik_inst.else_scope():
                    c0_cnt.set_as(1)
                    c_lp_cnt.set_as(c1_cnt)
                level4_in_offset = level5_in_offset + e_in_offset
                level4_out_offset = level5_out_offset + e_idx * n_size * c_size * hw_size

                with tik_inst.for_range(0, hw_lp_cnt) as hw_lp_idx:
                    level4_in_offset = level4_in_offset + hw_lp_idx * hw_step_in * hw_gate
                    level4_out_offset = level4_out_offset + hw_lp_idx * hw_gate
                    with tik_inst.if_scope(tik.any(hw_lp_idx != hw_lp_cnt - 1, hw_left == 0)):
                        per_lp_hw.set_as(hw_gate)
                    with tik_inst.else_scope():
                        per_lp_hw.set_as(hw_left)

                    with tik_inst.for_range(0, n_loop_cnt) as n_lp_idx:
                        head_gap.set_as(e_idx * c_size % c0_size)  # used for ubuf_to_ubuf after first vnchwconv
                        c_mod_c0.set_as((head_gap + c_size) % c0_size)
                        level3_in_offset = level4_in_offset + n_lp_idx * n_step_in
                        level3_out_offset = level4_out_offset + n_lp_idx * n_step_out
                        with tik_inst.if_scope(tik.any(n_lp_idx < n_loop_cnt - 1, n_left_size == 0)):
                            in_cube.set_as(n_cube_size)
                        with tik_inst.else_scope():
                            in_cube.set_as(n_left_size)

                        level2_in_offset = tik_inst.Scalar(name="level2_in_offset")
                        level2_out_offset = tik_inst.Scalar(name="level2_out_offset")
                        level2_out_offset.set_as(level3_out_offset)
                        with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:
                            level2_in_offset.set_as(level3_in_offset + c_lp_idx * c0_cnt * e_factor_1)
                            with tik_inst.if_scope(c1_cnt <= 2):
                                sub_c_size.set_as(c_size)
                            with tik_inst.else_scope():
                                with tik_inst.if_scope(c_lp_idx == 0):
                                    sub_c_size.set_as(2 * c0_size - head_gap)
                                with tik_inst.else_scope():
                                    head_gap.set_as(0)
                                    with tik_inst.if_scope(tik.any(c_lp_idx != c_lp_cnt - 1, c1_cnt % 2 == 0)):
                                        with tik_inst.if_scope(tik.all(c_lp_idx == c_lp_cnt - 1, c_mod_c0 > 0)):
                                            sub_c_size.set_as(c0_size + c_mod_c0)
                                        with tik_inst.else_scope():
                                            sub_c_size.set_as(2 * c0_size)

                                    with tik_inst.else_scope():
                                        with tik_inst.if_scope(c_mod_c0 == 0):
                                            sub_c_size.set_as(c0_size)
                                        with tik_inst.else_scope():  # roll back c0
                                            level2_in_offset.set_as(level2_in_offset - e_factor_1)
                                            level2_out_offset.set_as(level2_out_offset - c0_size * hw_size)
                                            sub_c_size.set_as(c0_size + c_mod_c0)

                            # move data in
                            mv_in_args = (tik_inst, src_gm, src_ub, ub_in, per_lp_hw, level2_in_offset,
                                          hw_step_in, c0_cnt, c0_size, e_factor_1, in_cube, in_dst_stride, dtype_factor)
                            _move_data_in_nchw(mv_in_args)
                            # do nhwc to nchw
                            vnc_args = (tik_inst, src_ub, ub_in, ub_out, vnc_line_size,
                                        per_lp_hw, c0_cnt, c0_size, sub_c_size, head_gap, dtype_factor)
                            _twice_vnchwconv_2_nchw(vnc_args)
                            # move data out
                            with tik_inst.if_scope(hw_size == hw_gate):
                                mv_out_args = (tik_inst, dst_gm, level2_out_offset, src_ub, ub_out, hw_size,
                                               sub_c_size, in_cube, c_lp_idx, c_lp_cnt, c_size, dtype, ele_per_block)
                                _move_data_out_nchw(mv_out_args)
                            with tik_inst.else_scope():
                                mv_out_args = (tik_inst, dst_gm, level2_out_offset, src_ub, ub_out,
                                               hw_size, per_lp_hw, sub_c_size, in_cube, c_size, ele_per_block)
                                _move_data_out_nchw_hw_block_align(mv_out_args)
                            # update output offset
                            level2_out_offset.set_as(level2_out_offset + sub_c_size * hw_size)

    with tik_inst.if_scope(block_idx < used_core):
        with tik_inst.if_scope(block_idx < used_core - 1):
            _inner_transformer(nlc_g_cnt)
        with tik_inst.else_scope():
            _inner_transformer(lc_g_cnt)


def get_tp_zng_2_hwcn(args):
    """
    get tiling parameters for fractal_z_g to hwcn

    Parameters
    ----------
    args : list
        args for get tiling parameters

    Returns
    -------
    tiling parameters
    """

    in_shape, out_shape, src_format, dst_format, c0_len, block_elem_cnt, ub_size, groups = args
    if len(out_shape) == 4:
        out_shape.insert(0, 1)
    d, h, w, c, c_out = out_shape
    gdc1hw, no, ni, c0 = in_shape
    n = c_out // groups

    # get multiple core parameters
    hw_size = _get_shape_size([h, w])  # h, w confused as one dimension
    used_core = _ceil_div(hw_size, _ceil_div(hw_size, CORE_NUM))
    nlc_hw_cnt = _ceil_div(hw_size, used_core)
    lc_hw_cnt = hw_size - nlc_hw_cnt * (used_core - 1)
    hw_step_in = _get_shape_size([no, ni, c0])
    hw_step_out = _get_shape_size([c, c_out])
    core_step_in = hw_step_in * nlc_hw_cnt
    core_step_out = hw_step_out * nlc_hw_cnt

    # get c tiling parameters
    c_lp_cnt = _ceil_div(c, c0)
    c_left_size = c % c0
    c_per_lp_size = c0
    c_lp_step_in = _get_shape_size([hw_size, no, ni, c0])
    c_lp_step_out = _get_shape_size([c0, c_out])

    # get E tiling parameters
    tmp_e_size = _lcm(_lcm(c, c0) // c, _lcm(n, NI_16) // n)
    e_size = tmp_e_size if tmp_e_size <= groups else groups
    e_factor_1 = _get_shape_size([d, h, w, no, ni, c0])
    e_factor_2 = _get_shape_size([n, c0])

    # get d tiling parameters
    c1_size = _ceil_div(e_size * c, c0)
    d_size = d
    d_step_in = _get_shape_size([c1_size, hw_size, no, ni, c0])
    d_step_out = _get_shape_size([hw_size, c, c_out])

    # get n tiling parameters
    vnc_line_size = ub_size // 2 // VNC_LINES // block_elem_cnt * block_elem_cnt
    if not (c0 % c == 0 or c % c0 == 0):  # c0 will be split
        n_per_lp_size = vnc_line_size // 2 // c0 // block_elem_cnt * block_elem_cnt
    else:
        n_per_lp_size = vnc_line_size // c0 // block_elem_cnt * block_elem_cnt
    n_lp_cnt = _ceil_div(c_out, n_per_lp_size)
    n_left = c_out % n_per_lp_size

    # get g tiling parameters
    if not (c0 % c == 0, c % c0 == 0):  # c will not be split
        valid_e_cnt = 1
    else:
        valid_e_cnt = _ceil_div(c0, c)

    # get output tiling parameters
    tiling_params = [used_core, core_step_in, core_step_out] + \
                    [nlc_hw_cnt, lc_hw_cnt, hw_step_in, hw_step_out] + \
                    [c_lp_cnt, c_left_size, c_per_lp_size, c_lp_step_in, c_lp_step_out] + \
                    [e_size, e_factor_1, e_factor_2, valid_e_cnt] + [c, c0, n, c_out] + \
                    [d_size, d_step_in, d_step_out] + [vnc_line_size, n_per_lp_size, n_lp_cnt, n_left]

    return tiling_params


def _move_data_in_hwcn(args):
    """
    move data in from gm for hwcn transform
    """

    (tik_inst, src_gm, src_ub, n_lp_idx, n_per_lp_size, n_lp_size, dtype_factor, level1_in_offset,
     c_size, c0_size, e_factor_1, e_size, e_factor_2, valid_e_cnt, n_size, rollback_size, c_out_beg) = args

    with tik_inst.new_stmt_scope():
        level0_in_offset = tik_inst.Scalar(name="level0_in_offset")
        level0_ub_offset = tik_inst.Scalar(name="level0_ub_offset")
        cube_e_cnt = tik_inst.Scalar(name="cube_e_cnt")
        dup_size = tik_inst.Scalar(name="dup_size")
        level0_ub_offset.set_as(0)

        # per n_size is one group
        n_beg_idx = (n_lp_idx * n_per_lp_size - rollback_size)
        # e count in one cube can be: _ceil_div(c0_size - head_gap, c_size))
        cube_e_cnt.set_as(1)
        with tik_inst.if_scope(tik.any(c0_size % c_size == 0, c_size % c0_size == 0)):
            c_out_beg.set_as(0)

        with tik_inst.for_range(0, n_lp_size) as c_out_idx:
            n_value = n_beg_idx + c_out_idx
            g_value = n_value // n_size
            with tik_inst.if_scope(tik.any(c0_size % c_size == 0, c_size % c0_size == 0)):  # c0 will not be split
                with tik_inst.if_scope(tik.any(n_value % (valid_e_cnt * n_size) == 0, c_out_idx == 0)):
                    level0_ub_offset.set_as(level0_ub_offset + (c_out_idx - c_out_beg) * c0_size)
                    c_out_beg.set_as(c_out_idx)
                    level0_in_offset.set_as(level1_in_offset + g_value * c_size // c0_size * e_factor_1 +
                                            g_value % e_size * e_factor_2 + n_value % n_size * c0_size)
                with tik_inst.if_scope(tik.any((n_value + 1) % (valid_e_cnt * n_size) == 0,
                                               c_out_idx == n_lp_size - 1)):
                    tik_inst.data_move(src_ub[level0_ub_offset], src_gm[level0_in_offset],
                                       0, 1, (c_out_idx + 1 - c_out_beg) * dtype_factor, 0, 0)

            with tik_inst.else_scope():
                with tik_inst.if_scope(tik.any((n_value - c_out_beg) % (cube_e_cnt * n_size) == 0, c_out_idx == 0)):
                    level0_in_offset.set_as(level1_in_offset + g_value * c_size // c0_size * e_factor_1 +
                                            g_value % e_size * e_factor_2 + n_value % n_size * c0_size)
                with tik_inst.if_scope(tik.any((n_value + 1 - c_out_beg) % (cube_e_cnt * n_size) == 0,
                                               c_out_idx == n_lp_size - 1)):
                    tik_inst.data_move(src_ub[level0_ub_offset], src_gm[level0_in_offset],
                                       0, 1, (n_value + 1 - c_out_beg) * dtype_factor, 0, 0)
                    level0_ub_offset.set_as(level0_ub_offset + (n_value + 1 - c_out_beg) * c0_size)

                    with tik_inst.if_scope(g_value * c_size % c0_size + c_size > c0_size):  # when c0 is split
                        with tik_inst.if_scope((n_value + 1 - c_out_beg) % n_size > 0):
                            dup_size.set_as((n_value + 1 - c_out_beg) % n_size)
                        with tik_inst.else_scope():
                            dup_size.set_as(n_size)
                        level0_in_offset.set_as(level0_in_offset + e_factor_1)
                        tik_inst.data_move(src_ub[level0_ub_offset], src_gm[level0_in_offset],
                                           0, 1, dup_size * dtype_factor, 0, 0)
                        level0_ub_offset.set_as(level0_ub_offset + dup_size * c0_size)
                    c_out_beg.set_as(n_value + 1)


def _vnchwconv_2_hwcn(args):
    """
    vnchwconv operations for hwcn transform
    """

    (tik_inst, src_ub, vnc_line_size, n_lp_cnt, n_lp_idx, n_per_lp_size, n_lp_size,
     ele_per_block, c0_size, c_size, n_size, dtype_factor, c0_cnt, rollback_size, head_gap) = args

    with tik_inst.new_stmt_scope():
        ub_offset = tik_inst.Scalar(name="ub_offset")
        dst_ub_offset = tik_inst.Scalar(name="dst_ub_offset")
        n_size_block_align = tik_inst.Scalar(name="n_size_block_align")
        c_out_beg_v = tik_inst.Scalar(name="c_out_beg_v")
        c_out_beg_v.set_as(0)

        # do gnc to ncg
        src_ub_fp16 = src_ub.reinterpret_cast_to("float16")
        ub_offset.set_as(vnc_line_size * dtype_factor * VNC_LINES)
        repeat_cnt = _ceil_div(vnc_line_size * dtype_factor, C0_16)
        src_addr_list = [src_ub_fp16[vnc_line_size * dtype_factor * i] for i in ADDR_IDX_LIST]
        dst_addr_list = [src_ub_fp16[ub_offset + C0_16 * i] for i in ADDR_IDX_LIST]
        with tik_inst.new_stmt_scope():
            src_stride = tik_inst.Scalar()
            dst_stride = tik_inst.Scalar()
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(1)
                dst_stride.set_as(16)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

            # delete gap between each e cube
            with tik_inst.if_scope(c_size % c0_size == 0):  # no gap
                tik_inst.data_move(src_ub_fp16, src_ub_fp16[ub_offset], 0, 1, n_lp_size * c0_size * dtype_factor, 0, 0)

            with tik_inst.else_scope():  # gap is random
                with tik_inst.for_range(0, n_lp_size) as c_out_idx:
                    n_value = n_lp_idx * n_per_lp_size + c_out_idx - rollback_size
                    g_value = n_value // n_size
                    with tik_inst.if_scope(tik.any(n_value % n_size == 0, c_out_idx == 0)):
                        ub_offset.set_as(ub_offset + (c_out_idx - c_out_beg_v) * c0_size * dtype_factor * VNC_LINES)
                        head_gap.set_as(g_value * c_size % c0_size * dtype_factor)
                        dst_ub_offset.set_as(c_out_idx * c0_size * dtype_factor * VNC_LINES)
                        c_out_beg_v.set_as(c_out_idx)
                    with tik_inst.if_scope(tik.any((n_value + 1) % n_size == 0, c_out_idx == n_lp_size - 1)):
                        tmp_n_lp_step = ((c_out_idx + 1 - c_out_beg_v) * c0_size) * dtype_factor - head_gap
                        tik_inst.data_move(src_ub_fp16[dst_ub_offset], src_ub_fp16[ub_offset + head_gap * VNC_LINES],
                                           0, 1, tmp_n_lp_step, 0, 0)
                        with tik_inst.if_scope(head_gap // dtype_factor + c_size > c0_size):
                            front_part = c0_size * dtype_factor - head_gap
                            tik_inst.data_move(src_ub_fp16[dst_ub_offset + front_part * VNC_LINES],
                                               src_ub_fp16[ub_offset + (tmp_n_lp_step + head_gap) * VNC_LINES],
                                               0, (c_out_idx + 1 - c_out_beg_v), head_gap,
                                               front_part, front_part)
                            ub_offset.set_as(ub_offset + (tmp_n_lp_step + head_gap) * VNC_LINES)

            # do ncg to gnc
            ub_offset.set_as(vnc_line_size * dtype_factor * VNC_LINES)
            with tik_inst.if_scope(n_lp_cnt > 1):
                n_size_block_align.set_as(_ceil_fill(n_lp_size, ele_per_block))
            with tik_inst.else_scope():
                n_size_block_align.set_as(n_lp_size)
            with tik_inst.for_range(0, c0_cnt) as c0_idx:
                tik_inst.data_move(src_ub_fp16[ub_offset + c0_idx * n_size_block_align * dtype_factor * VNC_LINES],
                                   src_ub_fp16[c0_idx * dtype_factor * VNC_LINES],
                                   0, n_lp_size, 1 * dtype_factor, (c0_size - 1) * dtype_factor, 0)

            # do gnc to cgn
            repeat_cnt = _ceil_div(n_size_block_align * dtype_factor * c0_size, C0_16)
            src_addr_list = [src_ub_fp16[ub_offset + C0_16 * i] for i in ADDR_IDX_LIST]
            dst_addr_list = [src_ub_fp16[repeat_cnt * C0_16 * i] for i in ADDR_IDX_LIST]
            with tik_inst.new_stmt_scope():
                src_stride = tik_inst.Scalar()
                dst_stride = tik_inst.Scalar()
                with tik_inst.if_scope(repeat_cnt == 1):
                    src_stride.set_as(0)
                    dst_stride.set_as(0)
                with tik_inst.else_scope():
                    src_stride.set_as(16)
                    dst_stride.set_as(1)
                tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list,
                                   repeat_cnt, dst_stride, src_stride)


def _move_data_out_hwcn(args):
    """
    move data to gm for hwcn transform
    """

    (tik_inst, dst_gm, src_ub, level1_out_offset, n_lp_cnt,
     c0_cnt, n_lp_size, ele_per_block, dtype, c_out) = args

    with tik_inst.if_scope(n_lp_cnt == 1):  # c_out load in by once
        out_size = c0_cnt * n_lp_size
        tik_inst.data_move(dst_gm[level1_out_offset], src_ub,
                           0, 1, out_size // ele_per_block, 0, 0)
        with tik_inst.if_scope(out_size % ele_per_block > 0):
            with tik_inst.new_stmt_scope():
                tmp_offset = out_size - ele_per_block
                tmp_reg = tik_inst.Scalar(dtype=dtype)
                with tik_inst.for_range(0, ele_per_block) as tmp_idx:
                    tmp_reg.set_as(src_ub[tmp_offset + tmp_idx])
                    src_ub[tmp_idx].set_as(tmp_reg)
                tik_inst.data_move(dst_gm[level1_out_offset + tmp_offset], src_ub, 0, 1, 1, 0, 0)

    with tik_inst.else_scope():
        out_size = n_lp_size
        out_size_1 = _ceil_fill(n_lp_size, ele_per_block)
        with tik_inst.for_range(0, c0_cnt) as c0_idx:
            tik_inst.data_move(dst_gm[level1_out_offset + c0_idx * c_out],
                               src_ub[c0_idx * out_size_1],
                               0, 1, out_size // ele_per_block, 0, 0)
            with tik_inst.if_scope(out_size % ele_per_block > 0):
                with tik_inst.new_stmt_scope():
                    tmp_offset = c0_idx * out_size_1 + out_size - ele_per_block
                    tmp_reg = tik_inst.Scalar(dtype=dtype)
                    with tik_inst.for_range(0, ele_per_block) as tmp_idx:
                        tmp_reg.set_as(src_ub[tmp_offset + tmp_idx])
                        src_ub[tmp_idx].set_as(tmp_reg)
                    tik_inst.data_move(dst_gm[level1_out_offset + c0_idx * c_out + out_size - ele_per_block],
                                       src_ub, 0, 1, 1, 0, 0)


def zng_hwcn_transform(tensor_args, tiling_args):
    """
    do fractal_z_g to nchw transform

    Parameters
    ----------
    tensor_args : list
        tensor related parameters
    tiling_args: list
        tiling parameters

    Returns
    -------
    None
    """

    (tik_inst, src_gm, dst_gm, src_ub, c_lp_cnt, c0_cnt,
     e_in_offset, in_cube, sub_c_size, block_idx, dtype, head_gap, c_mod_c0, ele_per_block) = tensor_args
    (used_core, core_step_in, core_step_out, nlc_hw_cnt, lc_hw_cnt, hw_step_in, hw_step_out, c_lp_cnt, c_left_size,
     c_per_lp_size, c_lp_step_in, c_lp_step_out, e_size, e_factor_1, e_factor_2, valid_e_cnt, c_size, c0_size, n_size,
     c_out, d_size, d_step_in, d_step_out, vnc_line_size, n_per_lp_size, n_lp_cnt, n_left) = tiling_args
    dtype_factor = C0_16 // ele_per_block

    def _inner_hwcn_transformer(hw_cnt):
        """
        the transform process
        """

        n_lp_size = tik_inst.Scalar(name="n_lp_size")
        rollback_size = tik_inst.Scalar(name="rollback_size")
        c_out_beg = tik_inst.Scalar(name="c_out_beg")

        with tik_inst.for_range(0, d_size) as d_idx:
            with tik_inst.for_range(0, hw_cnt) as hw_idx:
                level3_in_offset = hw_idx * hw_step_in + block_idx * core_step_in + d_idx * d_step_in
                level3_out_offset = hw_idx * hw_step_out + block_idx * core_step_out + d_idx * d_step_out

                with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:
                    level2_in_offset = level3_in_offset + c_lp_idx * c_lp_step_in
                    level2_out_offset = level3_out_offset + c_lp_idx * c_lp_step_out
                    with tik_inst.if_scope(tik.any(c_lp_idx != c_lp_cnt - 1, c_left_size == 0)):
                        c0_cnt.set_as(c_per_lp_size)
                    with tik_inst.else_scope():
                        c0_cnt.set_as(c_left_size)

                    c_out_beg.set_as(0)
                    with tik_inst.for_range(0, n_lp_cnt) as n_lp_idx:
                        with tik_inst.if_scope(tik.any(n_lp_idx != n_lp_cnt - 1, n_left == 0)):
                            n_lp_size.set_as(n_per_lp_size)
                            rollback_size.set_as(0)
                        with tik_inst.else_scope():
                            with tik_inst.if_scope(tik.all(n_lp_cnt > 1, n_left < ele_per_block)):
                                rollback_size.set_as(ele_per_block - n_left)
                                n_lp_size.set_as(ele_per_block)
                            with tik_inst.else_scope():
                                n_lp_size.set_as(n_left)
                                rollback_size.set_as(0)
                        level1_in_offset = level2_in_offset
                        level1_out_offset = level2_out_offset + n_lp_idx * n_per_lp_size - rollback_size

                        # move data in
                        mv_in_args = (tik_inst, src_gm, src_ub, n_lp_idx, n_per_lp_size, n_lp_size, dtype_factor,
                                      level1_in_offset, c_size, c0_size, e_factor_1, e_size, e_factor_2, valid_e_cnt,
                                      n_size, rollback_size, c_out_beg)
                        _move_data_in_hwcn(mv_in_args)
                        # make e line align
                        vnc_args = (tik_inst, src_ub, vnc_line_size, n_lp_cnt, n_lp_idx, n_per_lp_size, n_lp_size,
                                    ele_per_block, c0_size, c_size, n_size, dtype_factor, c0_cnt, rollback_size,
                                    head_gap)
                        _vnchwconv_2_hwcn(vnc_args)
                        # move data out
                        mv_out_args = (tik_inst, dst_gm, src_ub, level1_out_offset, n_lp_cnt,
                                       c0_cnt, n_lp_size, ele_per_block, dtype, c_out)
                        _move_data_out_hwcn(mv_out_args)

    with tik_inst.if_scope(block_idx < used_core):
        with tik_inst.if_scope(block_idx < used_core - 1):
            _inner_hwcn_transformer(nlc_hw_cnt)
        with tik_inst.else_scope():
            _inner_hwcn_transformer(lc_hw_cnt)


def zng_2_nchw_hwcn(src, dst, src_format, dst_format, groups=1, kernel_name="zng_2_nchw_hwcn"):
    """
    algorithm: format_transfer
    do fractal_z to nchw and hwcn transform for groups > 1.

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    src_format: str
        source data format, can be NHWC, NCHW, FRACTAL_Zn etc.
    dst_format: str
        target data format, can be NC1HWC0, NCHW, FRACTAL_Zn etc.
    groups: int
        default 1
    kernel_name: str
        kernel name, default value is "zng_2_nchw_hwcn"

    Returns
    -------
    None
    """

    src_format = src_format.upper()
    dst_format = dst_format.upper()
    in_shape = list(src.get("shape"))
    out_shape = list(dst.get("shape"))
    in_dtype = src.get("dtype").lower()
    ub_size = _get_max_element_in_ub(in_dtype, 1)
    c0_len = _get_c0_len(in_dtype)
    block_elem_cnt = BLOCK_BYTE_SIZE // _get_dtype_len(in_dtype)

    tik_inst = tik.Tik()
    src_in_gm = tik_inst.Tensor(in_dtype, (MAX_INT64_VALUE,), tbe_platform.scope_gm, "src_in_gm")
    dst_out_gm = tik_inst.Tensor(in_dtype, (MAX_INT64_VALUE,), tbe_platform.scope_gm, "dst_out_gm")
    src_ub = tik_inst.Tensor(in_dtype, (ub_size,), tbe_platform.scope_ubuf, "total_ub")
    c0_cnt = tik_inst.Scalar(name="c0_cnt")
    c_lp_cnt = tik_inst.Scalar(name="c_lp_cnt")
    e_in_offset = tik_inst.Scalar(name="e_in_offset")
    in_cube = tik_inst.Scalar(name="in_cube")
    sub_c_size = tik_inst.Scalar(name="sub_c_size")
    head_gap = tik_inst.Scalar(name="head_gap")
    c_mod_c0 = tik_inst.Scalar(name="c_mod_c0")

    with tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
        args = in_shape, out_shape, src_format, dst_format, c0_len, block_elem_cnt, ub_size, groups
        tensor_args = tik_inst, src_in_gm, dst_out_gm, src_ub, c_lp_cnt, c0_cnt, e_in_offset, in_cube, \
                      sub_c_size, block_idx, in_dtype, head_gap, c_mod_c0, block_elem_cnt
        if dst_format == "NCHW":
            tiling_params = get_tp_zng_2_nchw(args)
            zng_nchw_transform(tensor_args, tiling_params)
        else:
            # for hwcn
            tiling_params = get_tp_zng_2_hwcn(args)
            zng_hwcn_transform(tensor_args, tiling_params)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[src_in_gm], outputs=[dst_out_gm])
