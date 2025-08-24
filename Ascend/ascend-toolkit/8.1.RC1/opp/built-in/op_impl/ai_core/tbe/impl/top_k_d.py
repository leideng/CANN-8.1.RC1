#!/usr/bin/env python
# coding: utf-8
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
top_k_d
"""
import functools
from enum import Enum
from enum import unique

from impl.dynamic.top_k_d import top_k_d as top_k_template
from impl.top_k_large import top_k_large
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info

FP16_MINIMUM = -65520
FP16_MAXMUM = 65520
MAX_INT32 = 2 ** 31 - 1
INDICES_NUM = MAX_INT32
DTYPE_INT32 = "int32"
TILING_PARAMS_NUM = 8
MAX_SHAPE_SIZE = MAX_INT32
TILING_PARAM_DTYPE = DTYPE_INT32

# byte of one block
BYTE_BLOCK = 32
FULL_MASK_FP16 = 128
FULL_MASK_INT32 = 64
FULL_MASK_INT64 = 32


# 'pylint: disable=unused-argument,redefined-builtin
# 'pylint: disable=too-many-arguments
def get_op_support_info(input_tensor,
                        indices_tensor,
                        out_tensor,
                        out_indices_tensor,
                        k,
                        sorted=True,
                        dim=-1,
                        largest=True,
                        kernel_name='top_k'):
    """
    get top_k slice info
    """
    format_x = input_tensor.get("format")
    dims_x = len(input_tensor.get("shape"))
    if format_x == "ND":
        axis_split_matrix = []
        for i in range(dims_x - 1):
            split_info = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]], [1, [i]])]
            axis_split_matrix.append(split_info)
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)

    return op_cal_info_in_json


# 'pylint: disable=invalid-name
@unique
class Mode(Enum):
    """Mode for Region proposal"""
    X1 = 0
    Y1 = 1
    X2 = 2
    Y2 = 3
    Score = 4


# 'pylint: disable=too-many-public-methods
class GlobalVar:
    """GlobalVar Class Defination"""

    def __init__(self):
        """"
        __init__
        """
        self.data_gm = None
        self.data_gm_out = None
        self.indices_gm = None
        self.indices_gm_out = None
        self.indices_ub = None
        self.indices_out_fp16_ub = None
        self.indices_out_int32_ub = None
        self.data_tail_block_ub = None
        self.indices_tail_block_ub = None
        self.region_k_ub = None
        self.region_k2_ub = None
        self.data_ub = None
        self.region_ub = None
        self.region_sorted_ub = None
        self.reg_min_number = None
        self._max_part_num = None
        self.indices_out_final_ub = None
        self.offset_ub = None
        self.offset_fp16_ub = None
        self.offset_int32_ub = None
        self._index_reg = None
        self.offset_gm = None

    def set_data_gm(self, data_gm):
        """"
        set_data_gm
        """
        self.data_gm = data_gm

    def get_data_gm(self):
        """"
        get_data_gm
        """
        return self.data_gm

    def set_data_gm_out(self, data_gm_out):
        """"
        set_data_gm_out
        """
        self.data_gm_out = data_gm_out

    def get_data_gm_out(self):
        """"
        data_gm_out
        """
        return self.data_gm_out

    def set_indices_gm(self, indices_gm):
        """"
        set_indices_gm
        """
        self.indices_gm = indices_gm

    def get_indices_gm(self):
        """"
        get_indices_gm
        """
        return self.indices_gm

    def set_indices_gm_out(self, indices_gm_out):
        """"
        set_indices_gm_out
        """
        self.indices_gm_out = indices_gm_out

    def get_indices_gm_out(self):
        """"
        get_indices_gm_out
        """
        return self.indices_gm_out

    def set_indices_ub(self, indices_ub):
        """"
        set_indices_ub
        """
        self.indices_ub = indices_ub

    def get_indices_ub(self):
        """"
        get_indices_ub
        """
        return self.indices_ub

    def set_indices_out_fp16_ub(self, indices_out_fp16_ub):
        """"
        set_indices_out_fp16_ub
        """
        self.indices_out_fp16_ub = indices_out_fp16_ub

    def get_indices_out_fp16_ub(self):
        """"
        get_indices_out_fp16_ub
        """
        return self.indices_out_fp16_ub

    def set_indices_out_int32_ub(self, indices_out_int32_ub):
        """"
        set_indices_out_int32_ub
        """
        self.indices_out_int32_ub = indices_out_int32_ub

    def get_indices_out_int32_ub(self):
        """"
        get_indices_out_int32_ub
        """
        return self.indices_out_int32_ub

    def set_data_tail_block_ub(self, data_tail_block_ub):
        """"
        set_data_tail_block_ub
        """
        self.data_tail_block_ub = data_tail_block_ub

    def get_data_tail_block_ub(self):
        """"
        get_data_tail_block_ub
        """
        return self.data_tail_block_ub

    def set_indices_tail_block_ub(self, indices_tail_block_ub):
        """"
        set_indices_tail_block_ub
        """
        self.indices_tail_block_ub = indices_tail_block_ub

    def get_indices_tail_block_ub(self):
        """"
        get_indices_tail_block_ub
        """
        return self.indices_tail_block_ub

    def set_region_k2_ub(self, region_k2_ub):
        """"
        set_region_k2_ub
        """
        self.region_k2_ub = region_k2_ub

    def get_region_k2_ub(self):
        """"
        get_region_k2_ub
        """
        return self.region_k2_ub

    def set_data_ub(self, data_ub):
        """"
        set_data_ub
        """
        self.data_ub = data_ub

    def get_data_ub(self):
        """"
        get_data_ub
        """
        return self.data_ub

    def set_region_ub(self, region_ub):
        """"
        set_region_ub
        """
        self.region_ub = region_ub

    def get_region_ub(self):
        """"
        get_region_ub
        """
        return self.region_ub

    def set_region_sorted_ub(self, region_sorted_ub):
        """"
        set_region_sorted_ub
        """
        self.region_sorted_ub = region_sorted_ub

    def get_region_sorted_ub(self):
        """"
        get_region_sorted_ub
        """
        return self.region_sorted_ub

    def set_region_k_ub(self, region_k_ub):
        """"
        set_region_k_ub
        """
        self.region_k_ub = region_k_ub

    def get_region_k_ub(self):
        """"
        get_region_k_ub
        """
        return self.region_k_ub

    def set_reg_min_number(self, reg_min_number):
        """"
        set_reg_min_number
        """
        self.reg_min_number = reg_min_number

    def get_reg_min_number(self):
        """"
        get_reg_min_number
        """
        return self.reg_min_number

    @property
    def max_part_num(self):
        """
        get max_part_num
        """
        return self._max_part_num

    @max_part_num.setter
    def max_part_num(self, max_part_num):
        """
        set max_part_num
        """
        self._max_part_num = max_part_num

    def set_indices_out_final_ub(self, indices_out_final_ub):
        """
        set_indices_out_final_ub
        """
        self.indices_out_final_ub = indices_out_final_ub

    def get_indices_out_final_ub(self):
        """
        get_indices_out_final_ub
        """
        return self.indices_out_final_ub

    def set_offset_ub(self, offset_ub):
        """
        set_offset_ub
        """
        self.offset_ub = offset_ub

    def get_offset_ub(self):
        """
        get_offset_ub
        """
        return self.offset_ub

    def set_offset_fp16_ub(self, offset_fp16_ub):
        """
        set_offset_fp16_ub
        """
        self.offset_fp16_ub = offset_fp16_ub

    def get_offset_fp16_ub(self):
        """
        get_offset_fp16_ub
        """
        return self.offset_fp16_ub

    def set_offset_int32_ub(self, offset_int32_ub):
        """
        set_offset_int32_ub
        """
        self.offset_int32_ub = offset_int32_ub

    def get_offset_int32_ub(self):
        """
        get_offset_int32_ub
        """
        return self.offset_int32_ub

    @property
    def index_reg(self):
        """
        get index_reg
        """
        return self._index_reg

    @index_reg.setter
    def index_reg(self, index_reg):
        """
        set index_reg
        """
        self._index_reg = index_reg

    def set_offset_gm(self, offset_gm):
        """
        set_offset_gm
        """
        self.offset_gm = offset_gm

    def get_offset_gm(self):
        """
        get_offset_gm
        """
        return self.offset_gm


GLOBAL_VAR = GlobalVar()


# 'pylint: disable=too-many-arguments
def _emit_copy_ubuf_to_gm(tik_instance,
                          dtype,
                          dst,
                          src,
                          nburst,
                          burstlen,
                          srcstride,
                          dststride,
                          dst_offset=0,
                          src_offset=0):
    """
    _emit_copy_ubuf_to_gm
    """
    tik_instance.data_move(dst[dst_offset], src[src_offset], 0, nburst, burstlen, srcstride, dststride)


def _copy_ubuf_to_gm_k_less_16(tik_instance,
                               dtype,
                               dst,
                               src,
                               num_rows,
                               cols_padding,
                               k,
                               tail_block_ub,
                               gm_offset=0,
                               multi_core=False):
    """
    _copy_ubuf_to_gm
    """
    if dtype == 'float16':
        blocklen = 16
        zero_length = blocklen - k
        zero_scalar = tik_instance.Scalar(dtype=dtype, name="zero_scalar", init_value=0)
        tik_instance.set_atomic_add(2)
        with tik_instance.for_range(0, num_rows - 1, name='ub2gmi0') as i:
            for zero_pad in range(zero_length):
                src[cols_padding * i + k + zero_pad].set_as(zero_scalar)
            _emit_copy_ubuf_to_gm(tik_instance,
                                  dtype,
                                  dst,
                                  src,
                                  1,
                                  1,
                                  0,
                                  0,
                                  dst_offset=k * i + gm_offset,
                                  src_offset=cols_padding * i)
        for i in range(blocklen):
            tail_block_ub[i].set_as(src[cols_padding * (num_rows - 1) + i])
        for zero_pad in range(zero_length):
            tail_block_ub[k + zero_pad].set_as(zero_scalar)

        _emit_copy_ubuf_to_gm(tik_instance,
                              dtype,
                              dst,
                              tail_block_ub,
                              1,
                              1,
                              0,
                              0,
                              dst_offset=k * (num_rows - 1) + gm_offset,
                              src_offset=0)
        tik_instance.set_atomic_add(0)
    elif dtype == 'int32':
        blocklen = 8
        with tik_instance.for_range(0, num_rows - 1, name='ub2gmi0') as i:
            _emit_copy_ubuf_to_gm(tik_instance,
                                  dtype,
                                  dst,
                                  src,
                                  1,
                                  2,
                                  0,
                                  0,
                                  dst_offset=k * i + gm_offset,
                                  src_offset=cols_padding * i)

        _emit_copy_ubuf_to_gm(tik_instance,
                              dtype,
                              dst,
                              src,
                              1,
                              1,
                              0,
                              0,
                              dst_offset=k * (num_rows - 1) + gm_offset,
                              src_offset=cols_padding * (num_rows - 1))

        for i in range(blocklen):
            tail_block_ub[i].set_as(src[cols_padding * (num_rows - 1) + k - blocklen + i])

        _emit_copy_ubuf_to_gm(tik_instance,
                              dtype,
                              dst,
                              tail_block_ub,
                              1,
                              1,
                              0,
                              0,
                              dst_offset=k * (num_rows - 1) + gm_offset + k - blocklen,
                              src_offset=0)


# 'pylint: disable=too-many-arguments
def _copy_ubuf_to_gm(tik_instance,
                     dtype,
                     dst,
                     src,
                     num_rows,
                     cols_padding,
                     k,
                     tail_block_ub,
                     gm_offset=0,
                     multi_core=False):
    """
    _copy_ubuf_to_gm
    """
    if dtype == 'float16':
        burstlen = (k * 2 + 31) // 32
        blocklen = 16
    elif dtype == 'int32':
        burstlen = (k * 4 + 31) // 32
        blocklen = 8

    with tik_instance.for_range(0, num_rows - 1, name='ub2gmi0') as i:
        _emit_copy_ubuf_to_gm(tik_instance,
                              dtype,
                              dst,
                              src,
                              1,
                              burstlen,
                              0,
                              0,
                              dst_offset=k * i + gm_offset,
                              src_offset=cols_padding * i)
    if multi_core and k > 16:
        _emit_copy_ubuf_to_gm(tik_instance,
                              dtype,
                              dst,
                              src,
                              1,
                              burstlen - 1,
                              0,
                              0,
                              dst_offset=k * (num_rows - 1) + gm_offset,
                              src_offset=cols_padding * (num_rows - 1))

        for i in range(blocklen):
            tail_block_ub[i].set_as(src[cols_padding * (num_rows - 1) + k - blocklen + i])

        _emit_copy_ubuf_to_gm(tik_instance,
                              dtype,
                              dst,
                              tail_block_ub,
                              1,
                              1,
                              0,
                              0,
                              dst_offset=k * (num_rows - 1) + gm_offset + k - blocklen,
                              src_offset=0)
    else:
        _emit_copy_ubuf_to_gm(tik_instance,
                              dtype,
                              dst,
                              src,
                              1,
                              burstlen,
                              0,
                              0,
                              dst_offset=k * (num_rows - 1) + gm_offset,
                              src_offset=cols_padding * (num_rows - 1))


# 'pylint: disable=too-many-arguments
# 'pylint: disable=too-many-locals
def _add(tik_instance, dst, src1, src2, rows, cols_padding):
    # process 256B data per repeat for vsub
    vadd_len = 64
    repeat = (rows * cols_padding) // vadd_len
    remain = (rows * cols_padding) % vadd_len
    if repeat > 0:
        tik_instance.vadd(FULL_MASK_INT32, dst, src1, src2, repeat, 1, 1, 1, 8, 8, 8)
    if remain > 0:
        tik_instance.vadd(remain, dst[repeat * vadd_len], src1[repeat * vadd_len], src2[repeat * vadd_len], 1, 1, 1, 1,
                          8, 8, 8)


# 'pylint: disable=too-many-arguments
def _emit_vmul(tik_instance, dtype, dst, src1, src2, cnt):
    """
    _emit_vmul
    """
    # Vector instr process data bytes in a cycle
    vector_process_bytes = 256
    dtype_bytes_size = tbe_platform.get_bit_len(dtype) // 8
    calc_num_each_times = vector_process_bytes // dtype_bytes_size
    repeat_255 = cnt // calc_num_each_times
    repeat_remain = cnt % calc_num_each_times
    times = (repeat_255 + 254) // 255
    if repeat_255 > 0:
        with tik_instance.for_range(0, times, name='vmul_i0') as i:
            vmul_src0_scalar = tik_instance.Scalar(dtype="int64",
                                                   name='vmul_src0_scalar',
                                                   init_value=repeat_255 - i * 255)
            vmul_src1_scalar = tik_instance.Scalar(dtype="int64", name='vmul_src1_scalar', init_value=255)
            vmul_times_len = tik_instance.Scalar(dtype="int64", name='vmul_dst_scalar')
            tik_instance.scalar_min(vmul_times_len, vmul_src0_scalar, vmul_src1_scalar)
            tik_instance.vmul(FULL_MASK_INT32, dst[i * calc_num_each_times * 255], src1[i * calc_num_each_times * 255],
                              src2, vmul_times_len, 1, 1, 0, 8, 8, 0)

    if repeat_remain > 0:
        tik_instance.vmul(repeat_remain, dst[repeat_255 * calc_num_each_times], src1[repeat_255 * calc_num_each_times],
                          src2, 1, 1, 1, 0, 8, 8, 0)


# 'pylint: disable=too-many-arguments
def _conv_fp162s32(tik_instance, s32ub, s32ub_offset, fp16ub, fp16ub_offset, num):
    """
    fp16 to int32
    """
    repeat = (num) // 64
    remain = (num) % 64
    if repeat > 0:
        tik_instance.vconv(64, "round", s32ub[s32ub_offset], fp16ub[fp16ub_offset], repeat, 1, 1, 8, 4)
    if remain > 0:
        tik_instance.vconv(remain, "round", s32ub[s32ub_offset + repeat * 64], fp16ub[fp16ub_offset + repeat * 64], 1,
                           1, 1, 8, 4)


# 'pylint: disable=too-many-arguments
def _emit_vextract(tik_instance, dst, src, mode, cnt, dst_offset=0, src_offset=0):
    """
    _emit_vextract
    """
    repeat_255 = cnt // (16 * 255)
    repeat_remain = (cnt - repeat_255 * 16 * 255) // 16
    if repeat_255 > 0:
        with tik_instance.for_range(0, repeat_255, name='i0') as i:
            tik_instance.vextract(dst[dst_offset + i * 255 * 16], src[src_offset + i * 255 * 16 * 8], 255, mode)

    if repeat_remain > 0:
        tik_instance.vextract(dst[dst_offset + 255 * 16 * repeat_255], src[src_offset + 255 * 16 * 8 * repeat_255],
                              repeat_remain, mode)


# 'pylint: disable=too-many-arguments
def _merge_two_sorted_region(tik_instance, dst, src_region_k, src_region_sorted, len_region_k, len_region_sorted,
                             len_region_k_limit, merge_num):
    """
    _merge_two_sorted_region
    """
    if len_region_k < len_region_k_limit:
        merge_n0 = len_region_k
        merge_n1 = len_region_sorted
        src_list = [src_region_k[0], src_region_sorted[0], src_region_k[16], src_region_k[16]]
        tik_instance.vmrgsort4(dst, src_list, (merge_n0, merge_n1, 16, 16), False, 3, 1)

    elif len_region_k >= len_region_k_limit:
        merge_n0 = merge_num
        merge_n1 = merge_num
        merge_n2 = len_region_sorted
        src_list = [src_region_k[0], src_region_k[(merge_num) * 8], src_region_sorted[0], src_region_k[16]]
        tik_instance.vmrgsort4(dst, src_list, (merge_n0, merge_n1, merge_n2, 16), False, 7, 1)


def _copy_region(tik_instance, dst, src, num, dst_offset=0):
    """
    _copy_region
    """
    burstlen = (num * 2 * 8 + 31) // 32
    tik_instance.data_move(dst[dst_offset], src, 0, 1, burstlen, 0, 0)


# 'pylint: disable=too-many-arguments
# 'pylint: disable=too-many-locals
# 'pylint: disable=too-many-statements
def _merge_recur(tik_instance, src_ub, dst_ub, last_dim, total_region_list, level, region_offset=0):
    """
    _merge_recur
    merge multi sorted region proposal list to one sorted region proposal list
    """
    # vmrgsort4 can merger at most 4 sorted region list
    loops = total_region_list // 4
    remain = total_region_list % 4

    merge_n0 = 16 * (4 ** (level - 1))
    merge_n1 = merge_n0
    merge_n2 = merge_n0
    merge_n3 = merge_n0
    merge_repeat = loops
    need_tail_process = False
    if loops > 0 and remain == 0:
        if merge_n0 * 4 * loops > last_dim:
            merge_repeat = loops - 1
            n012 = merge_n0 + merge_n1 + merge_n2
            merge_left = last_dim - ((merge_n0 * 4 * (loops - 1)) + n012)
            need_tail_process = True
    if merge_repeat > 0:
        src_list = [
            src_ub[region_offset], src_ub[region_offset + merge_n0 * 8],
            src_ub[region_offset + merge_n0 * 8 + merge_n1 * 8],
            src_ub[region_offset + merge_n0 * 8 + merge_n1 * 8 + merge_n2 * 8]
        ]
        tik_instance.vmrgsort4(dst_ub[region_offset], src_list, (merge_n0, merge_n1, merge_n2, merge_n3), False, 15,
                               merge_repeat)

    if need_tail_process:
        tail_offset = 4 * merge_n0 * merge_repeat * 8
        src_list = [
            src_ub[region_offset + tail_offset], src_ub[region_offset + tail_offset + merge_n0 * 8],
            src_ub[region_offset + tail_offset + merge_n0 * 8 + merge_n1 * 8],
            src_ub[region_offset + tail_offset + merge_n0 * 8 + merge_n1 * 8 + merge_n2 * 8]
        ]
        tik_instance.vmrgsort4(dst_ub[region_offset + tail_offset], src_list,
                               (merge_n0, merge_n1, merge_n2, merge_left), False, 15, 1)

    if loops > 0:
        offset = 4 * loops * 16 * (4 ** (level - 1))
    else:
        offset = 0

    if remain == 3:
        merge_n0 = 16 * (4 ** (level - 1))
        merge_n1 = merge_n0
        merge_n2 = last_dim - (offset + merge_n0 + merge_n1)

        src_list = [
            src_ub[region_offset + offset * 8], src_ub[region_offset + offset * 8 + merge_n0 * 8],
            src_ub[region_offset + offset * 8 + merge_n0 * 8 + merge_n1 * 8], src_ub[0]
        ]

        tik_instance.vmrgsort4(dst_ub[region_offset + offset * 8], src_list, (merge_n0, merge_n1, merge_n2, 16), False,
                               7, 1)

    elif remain == 2:
        merge_n0 = 16 * (4 ** (level - 1))
        merge_n1 = last_dim - (offset + merge_n0)

        src_list = [src_ub[region_offset + offset * 8], src_ub[region_offset + offset * 8 + merge_n0 * 8]]
        tik_instance.vmrgsort4(dst_ub[region_offset + offset * 8], src_list + [src_ub[0], src_ub[0]],
                               (merge_n0, merge_n1, 16, 16), False, 3, 1)

    elif remain == 1:
        merge_n0 = last_dim - offset
        num_blocks_write = (merge_n0 * 16 + 31) // 32
        tik_instance.data_move(dst_ub[region_offset + offset * 8], src_ub[region_offset + offset * 8], 0, 1,
                               num_blocks_write, 0, 0)

    next_total_region_list = (total_region_list + 3) // 4

    if next_total_region_list <= 1:
        return dst_ub
    return _merge_recur(tik_instance, dst_ub, src_ub, last_dim, next_total_region_list, level + 1, region_offset)


def _merge_region(tik_instance, dst, src, rows, cols):
    """
    _merge_region
    """
    cols_padding = ((cols + 15) // 16) * 16
    with tik_instance.for_range(0, rows, name='merge_i0') as i:
        result_ub = _merge_recur(tik_instance, src, dst, cols, (cols + 15) // 16, 1, region_offset=i * cols_padding * 8)
    return result_ub


# 'pylint: disable=too-many-arguments
def _emit_vbitsort(tik_instance, dst, src, cnt, dst_offset=0, src_offset=0):
    """
    _emit_vbitsort
    """
    repeat_255 = cnt // (16 * 255)
    repeat_remain = (cnt - repeat_255 * 16 * 255) // 16
    if repeat_255 > 0:
        with tik_instance.for_range(0, repeat_255, name='i0') as i:
            tik_instance.vrpsort16(dst[dst_offset + i * 255 * 16 * 8], src[src_offset + i * 255 * 16 * 8], 255)

    if repeat_remain > 0:
        tik_instance.vrpsort16(dst[dst_offset + 255 * 16 * 8 * repeat_255], src[src_offset + 255 * 16 * 8 * repeat_255],
                               repeat_remain)


def _sort_region(tik_instance, dst, src, rows, cols):
    """
    _sort_region
    """
    _emit_vbitsort(tik_instance, dst, src, cnt=rows * cols)
    if cols > 16:
        result_ub = _merge_region(tik_instance, dst=src, src=dst, rows=rows, cols=cols)
    else:
        result_ub = dst
    return result_ub


# 'pylint: disable=too-many-arguments
def _emit_vconcat(tik_instance, dst, src, mode, cnt, dst_offset=0, src_offset=0):
    """
    _emit_vconcat
    """
    repeat_255 = cnt // (16 * 255)
    repeat_remain = (cnt - repeat_255 * 16 * 255) // 16
    with tik_instance.if_scope(repeat_255 > 0):
        with tik_instance.for_range(0, repeat_255, name='vconcat_i0') as i:
            tik_instance.vconcat(dst[dst_offset + i * 255 * 16 * 8], src[src_offset + i * 255 * 16], 255, mode)
    if repeat_remain > 0:
        tik_instance.vconcat(dst[dst_offset + 255 * 16 * 8 * repeat_255], src[src_offset + 255 * 16 * repeat_255],
                             repeat_remain, mode)


# 'pylint: disable=too-many-arguments
def _emit_copy_gm_to_ubuf(tik_instance,
                          dtype,
                          dst,
                          src,
                          nburst,
                          burstlen,
                          srcstride,
                          dststride,
                          dst_offset=0,
                          src_offset=0):
    """
    _emit_copy_gm_to_ubuf
    """
    tik_instance.data_move(dst[dst_offset], src[src_offset], 0, nburst, burstlen, srcstride, dststride)


def _emit_vmuls(tik_instance, dst, src, cnt):
    """
    _emit_vmuls
    """
    repeat_255 = cnt // 128
    repeat_remain = cnt % 128
    times = (repeat_255 + 254) // 255
    if repeat_255 > 0:
        with tik_instance.for_range(0, times, name='vmuls_i0') as i:
            src0_scalar = tik_instance.Scalar(dtype="int64", name='src0_scalar', init_value=repeat_255 - i * 255)
            src1_scalar = tik_instance.Scalar(dtype="int64", name='src1_scalar', init_value=255)
            times_len = tik_instance.Scalar(dtype="int64", name='dst_scalar')
            tik_instance.scalar_min(times_len, src0_scalar, src1_scalar)
            tik_instance.vmuls(FULL_MASK_FP16, dst[i * 128 * 255], src[i * 128 * 255], -1, times_len, 1, 1, 8, 8)

    if repeat_remain > 0:
        tik_instance.vmuls(repeat_remain, dst[repeat_255 * 128], src[repeat_255 * 128], -1, 1, 1, 1, 8, 8)


# 'pylint: disable=too-many-arguments
def _copy_gm_to_ubuf_func(tik_instance, dst, src, num_rows, cols, col_start, gm_offset, largest):
    """
    _copy_gm_to_ubuf copy data from gm to ubuf
    """
    cols_padding = ((cols + 15) // 16) * 16
    burstlen = (cols * 2 + 31) // 32
    reg_min_number = GLOBAL_VAR.get_reg_min_number()
    with tik_instance.for_range(0, num_rows, name='gm2ub_i0') as i:
        _emit_copy_gm_to_ubuf(tik_instance,
                              'float16',
                              dst,
                              src,
                              1,
                              burstlen,
                              0,
                              0,
                              dst_offset=cols_padding * i,
                              src_offset=cols * i + col_start + gm_offset)
    if not largest:
        with tik_instance.for_range(0, num_rows, name='gm2ub_i0') as i:
            with tik_instance.for_range(0, cols_padding - cols) as j:
                dst[cols_padding * i + cols + j].set_as(FP16_MAXMUM)
        _emit_vmuls(tik_instance, dst, dst, cnt=num_rows * cols_padding)
    else:
        with tik_instance.for_range(0, num_rows, name='gm2ub_i0') as i:
            with tik_instance.for_range(0, cols_padding - cols) as j:
                dst[cols_padding * i + cols + j].set_as(reg_min_number)


# 'pylint: disable=too-many-arguments
def _copy_gm_to_ubuf(tik_instance, dst, src, num_rows, cols, col_start, gm_offset):
    """
    _copy_gm_to_ubuf copy data from gm to ubuf
    """
    cols_padding = ((cols + 15) // 16) * 16
    burstlen = (cols * 2 + 31) // 32
    with tik_instance.for_range(0, num_rows, name='gm2ub_i0') as i:
        _emit_copy_gm_to_ubuf(tik_instance,
                              'float16',
                              dst,
                              src,
                              1,
                              burstlen,
                              0,
                              0,
                              dst_offset=cols_padding * i,
                              src_offset=cols * i + col_start + gm_offset)


# 'pylint: disable=too-many-arguments
# 'pylint: disable=too-many-locals
# 'pylint: disable=too-many-statements
def _topk_a_row_by_part(tik_instance, row_start_in_core, cols, k, core_rows_start, multi_core, largest, is_lhisi):
    """
    _topk_a_row_by_part
    """
    data_gm = GLOBAL_VAR.get_data_gm()
    data_ub = GLOBAL_VAR.get_data_ub()
    data_gm_out = GLOBAL_VAR.get_data_gm_out()
    indices_gm = GLOBAL_VAR.get_indices_gm()
    indices_gm_out = GLOBAL_VAR.get_indices_gm_out()
    indices_ub = GLOBAL_VAR.get_indices_ub()
    indices_out_int32_ub = GLOBAL_VAR.get_indices_out_int32_ub()
    region_ub = GLOBAL_VAR.get_region_ub()
    region_sorted_ub = GLOBAL_VAR.get_region_sorted_ub()
    region_k_ub = GLOBAL_VAR.get_region_k_ub()
    region_k2_ub = GLOBAL_VAR.get_region_k2_ub()
    data_tail_block_ub = GLOBAL_VAR.get_data_tail_block_ub()
    indices_tail_block_ub = GLOBAL_VAR.get_indices_tail_block_ub()
    max_part_num = GLOBAL_VAR.max_part_num

    offset_ub = GLOBAL_VAR.get_offset_ub()
    offset_int32_ub = GLOBAL_VAR.get_offset_int32_ub()
    indices_out_final_ub = GLOBAL_VAR.get_indices_out_final_ub()
    index_reg = GLOBAL_VAR.index_reg

    # set lhisi version
    if is_lhisi:
        cols_per_part = 512
        len_region_k_limit = 2048
        merge_num = 1024
        region_k_num = 2048
        cnt_num = 512
    else:
        cols_per_part = 1024
        len_region_k_limit = 4096
        merge_num = 2048
        region_k_num = 4096
        cnt_num = 1024

    k_padding = ((k + 15) // 16) * 16
    cols_padding = ((cols + 15) // 16) * 16
    part_cnt = (cols + cols_per_part - 1) // cols_per_part
    last_part_cols = cols - ((part_cnt - 1) * cols_per_part)
    last_part_cols_padding = ((last_part_cols + 15) // 16) * 16
    gm_offset = row_start_in_core * cols + core_rows_start * cols
    data_dtype = "float16"
    # Vector instr process data bytes in a cycle
    vector_process_bytes = 256
    dtype_bytes_size = tbe_platform.get_bit_len(data_dtype) // 8
    data_num_per_process = vector_process_bytes // dtype_bytes_size
    repeat_times = cols_per_part // data_num_per_process

    _copy_gm_to_ubuf_func(tik_instance,
                          data_ub,
                          data_gm,
                          num_rows=1,
                          cols=cols_per_part,
                          col_start=0,
                          gm_offset=gm_offset,
                          largest=largest)

    tik_instance.vector_dup(128, indices_ub, 0.0, repeat_times, 1, 8)
    _emit_copy_gm_to_ubuf(tik_instance, data_dtype, offset_ub, indices_gm, 1, max_part_num // 16, 0, 0)
    _emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Score.value, cnt=cols_per_part)
    # for Ascend310 can't support extract score
    _emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Y2.value, cnt=cols_per_part)
    _emit_vconcat(tik_instance, region_ub, indices_ub, mode=Mode.X1.value, cnt=cols_per_part)
    _emit_vconcat(tik_instance, region_ub, offset_ub, mode=Mode.Y1.value, cnt=cols_per_part)
    result_ub = _sort_region(tik_instance, region_sorted_ub, region_ub, 1, cols_per_part)
    _copy_region(tik_instance, dst=region_k_ub, src=result_ub, num=cols_per_part)

    with tik_instance.for_range(0, part_cnt - 2, name='topk_i0') as i:

        index_reg.set_as(offset_ub[i + 1])

        tik_instance.vector_dup(128, indices_ub, index_reg, repeat_times, 1, 8)

        _copy_gm_to_ubuf_func(tik_instance,
                              data_ub,
                              data_gm,
                              num_rows=1,
                              cols=cols_per_part,
                              col_start=cols_per_part * (i + 1),
                              gm_offset=gm_offset,
                              largest=largest)
        _emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Score.value, cnt=cols_per_part)
        # for Ascend310 can't support extract score
        _emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Y2.value, cnt=cols_per_part)
        _emit_vconcat(tik_instance, region_ub, indices_ub, mode=Mode.X1.value, cnt=cols_per_part)
        _emit_vconcat(tik_instance, region_ub, offset_ub, mode=Mode.Y1.value, cnt=cols_per_part)
        result_ub = _sort_region(tik_instance, region_sorted_ub, region_ub, 1, cols_per_part)

        with tik_instance.if_scope(i == 0):
            _merge_two_sorted_region(tik_instance,
                                     dst=region_k2_ub,
                                     src_region_k=region_k_ub,
                                     src_region_sorted=result_ub,
                                     len_region_k=cols_per_part,
                                     len_region_sorted=cols_per_part,
                                     len_region_k_limit=len_region_k_limit,
                                     merge_num=merge_num)
            _copy_region(tik_instance, dst=region_k_ub, src=region_k2_ub, num=cols_per_part * 2)
        with tik_instance.if_scope(i == 1):
            _merge_two_sorted_region(tik_instance,
                                     dst=region_k2_ub,
                                     src_region_k=region_k_ub,
                                     src_region_sorted=result_ub,
                                     len_region_k=cols_per_part * 2,
                                     len_region_sorted=cols_per_part,
                                     len_region_k_limit=len_region_k_limit,
                                     merge_num=merge_num)
            _copy_region(tik_instance, dst=region_k_ub, src=region_k2_ub, num=cols_per_part * 3)
        with tik_instance.if_scope(i == 2):
            _merge_two_sorted_region(tik_instance,
                                     dst=region_k2_ub,
                                     src_region_k=region_k_ub,
                                     src_region_sorted=result_ub,
                                     len_region_k=cols_per_part * 3,
                                     len_region_sorted=cols_per_part,
                                     len_region_k_limit=len_region_k_limit,
                                     merge_num=merge_num)
            _copy_region(tik_instance, dst=region_k_ub, src=region_k2_ub, num=cols_per_part * 4)
        with tik_instance.if_scope(i >= 3):
            _merge_two_sorted_region(tik_instance,
                                     dst=region_k2_ub,
                                     src_region_k=region_k_ub,
                                     src_region_sorted=result_ub,
                                     len_region_k=cols_per_part * 4,
                                     len_region_sorted=cols_per_part,
                                     len_region_k_limit=len_region_k_limit,
                                     merge_num=merge_num)

            _copy_region(tik_instance, dst=region_k_ub, src=region_k2_ub, num=cols_per_part * 5)

    _copy_gm_to_ubuf_func(tik_instance,
                          data_ub,
                          data_gm,
                          num_rows=1,
                          cols=last_part_cols,
                          col_start=(part_cnt - 1) * cols_per_part,
                          gm_offset=gm_offset,
                          largest=largest)

    index_reg.set_as(offset_ub[part_cnt - 1])
    tik_instance.vector_dup(128, indices_ub, index_reg, repeat_times, 1, 8)
    _emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Score.value, cnt=last_part_cols_padding)
    # for Ascend310 can't support extract score
    _emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Y2.value, cnt=last_part_cols_padding)
    _emit_vconcat(tik_instance, region_ub, indices_ub, mode=Mode.X1.value, cnt=last_part_cols_padding)
    _emit_vconcat(tik_instance, region_ub, offset_ub, mode=Mode.Y1.value, cnt=last_part_cols_padding)
    result_ub = _sort_region(tik_instance, region_sorted_ub, region_ub, 1, last_part_cols_padding)
    _merge_two_sorted_region(tik_instance,
                             dst=region_k2_ub,
                             src_region_k=region_k_ub,
                             src_region_sorted=result_ub,
                             len_region_k=region_k_num,
                             len_region_sorted=last_part_cols_padding,
                             len_region_k_limit=len_region_k_limit,
                             merge_num=merge_num)

    # for Ascend310 can't support extract score
    _emit_vextract(tik_instance, region_k_ub, region_k2_ub, mode=Mode.Y2.value, cnt=k_padding)
    _emit_vextract(tik_instance, region_k_ub, region_k2_ub, mode=Mode.X1.value, cnt=k_padding, dst_offset=k_padding)
    _emit_vextract(tik_instance, region_k_ub, region_k2_ub, mode=Mode.Y1.value, cnt=k_padding, dst_offset=k_padding * 2)
    if not largest:
        _emit_vmuls(tik_instance, region_k_ub, region_k_ub, cnt=k_padding)
    _conv_fp162s32(tik_instance, indices_out_int32_ub, 0, region_k_ub, k_padding, k_padding)
    _conv_fp162s32(tik_instance, offset_int32_ub, 0, region_k_ub, k_padding * 2, k_padding)

    # fix cnt 512
    tik_instance.vector_dup(8, indices_tail_block_ub, cnt_num, 1, 0, 0)
    _emit_vmul(tik_instance, 'int32', indices_out_int32_ub, indices_out_int32_ub, indices_tail_block_ub, cnt=k_padding)
    _add(tik_instance, indices_out_final_ub, indices_out_int32_ub, offset_int32_ub, 1, k_padding)

    if k >= 8 and k < 16 and tbe_platform.api_check_support("tik.set_atomic_add", "float16"):
        _copy_ubuf_to_gm_k_less_16(tik_instance,
                                   'float16',
                                   data_gm_out,
                                   region_k_ub,
                                   num_rows=1,
                                   cols_padding=cols_padding,
                                   k=k,
                                   tail_block_ub=data_tail_block_ub,
                                   gm_offset=row_start_in_core * k + core_rows_start * k,
                                   multi_core=multi_core)
        _copy_ubuf_to_gm_k_less_16(tik_instance,
                                   'int32',
                                   indices_gm_out,
                                   indices_out_final_ub,
                                   1,
                                   cols_padding,
                                   k,
                                   tail_block_ub=indices_tail_block_ub,
                                   gm_offset=row_start_in_core * k + core_rows_start * k,
                                   multi_core=multi_core)
    else:
        _copy_ubuf_to_gm(tik_instance,
                         'float16',
                         data_gm_out,
                         region_k_ub,
                         num_rows=1,
                         cols_padding=cols_padding,
                         k=k,
                         tail_block_ub=data_tail_block_ub,
                         gm_offset=row_start_in_core * k + core_rows_start * k,
                         multi_core=multi_core)
        _copy_ubuf_to_gm(tik_instance,
                         'int32',
                         indices_gm_out,
                         indices_out_final_ub,
                         1,
                         cols_padding,
                         k,
                         tail_block_ub=indices_tail_block_ub,
                         gm_offset=row_start_in_core * k + core_rows_start * k,
                         multi_core=multi_core)


# 'pylint: disable=too-many-arguments
# 'pylint: disable=too-many-locals
def _topk_rows(tik_instance, row_start_in_core, rows, cols, k, core_rows_start, multi_core, largest):
    """
    _topk_rows do topk action muilti rows
    """
    data_gm = GLOBAL_VAR.get_data_gm()
    data_gm_out = GLOBAL_VAR.get_data_gm_out()
    indices_gm = GLOBAL_VAR.get_indices_gm()
    indices_gm_out = GLOBAL_VAR.get_indices_gm_out()
    indices_ub = GLOBAL_VAR.get_indices_ub()
    indices_out_fp16_ub = GLOBAL_VAR.get_indices_out_fp16_ub()
    indices_out_int32_ub = GLOBAL_VAR.get_indices_out_int32_ub()
    data_ub = GLOBAL_VAR.get_data_ub()
    region_ub = GLOBAL_VAR.get_region_ub()
    region_sorted_ub = GLOBAL_VAR.get_region_sorted_ub()
    data_tail_block_ub = GLOBAL_VAR.get_data_tail_block_ub()
    indices_tail_block_ub = GLOBAL_VAR.get_indices_tail_block_ub()
    offset_ub = GLOBAL_VAR.get_offset_ub()
    offset_fp16_ub = GLOBAL_VAR.get_offset_fp16_ub()
    offset_int32_ub = GLOBAL_VAR.get_offset_int32_ub()
    indices_out_final_ub = GLOBAL_VAR.get_indices_out_final_ub()
    cols_padding = ((cols + 15) // 16) * 16
    _copy_gm_to_ubuf_func(tik_instance,
                          data_ub,
                          data_gm,
                          num_rows=rows,
                          cols=cols,
                          col_start=0,
                          gm_offset=row_start_in_core * cols + core_rows_start * cols,
                          largest=largest)
    _copy_gm_to_ubuf(tik_instance, indices_ub, indices_gm, num_rows=1, cols=cols, col_start=0, gm_offset=0)
    _copy_gm_to_ubuf(tik_instance, offset_ub, indices_gm, num_rows=1, cols=cols, col_start=4096, gm_offset=0)
    _emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Score.value, cnt=rows * cols_padding)
    # for Ascend310 can't support extract score
    _emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Y2.value, cnt=rows * cols_padding)
    with tik_instance.for_range(0, rows, name='i0') as i:
        _emit_vconcat(tik_instance,
                      region_ub,
                      indices_ub,
                      mode=Mode.X1.value,
                      cnt=cols_padding,
                      dst_offset=i * cols_padding * 8,
                      src_offset=0)
        _emit_vconcat(tik_instance,
                      region_ub,
                      offset_ub,
                      mode=Mode.Y1.value,
                      cnt=cols_padding,
                      dst_offset=i * cols_padding * 8,
                      src_offset=0)
    result_ub = _sort_region(tik_instance, region_sorted_ub, region_ub, rows, cols_padding)
    # for Ascend310 can't support extract score
    _emit_vextract(tik_instance, data_ub, result_ub, mode=Mode.Y2.value, cnt=rows * cols_padding)
    with tik_instance.for_range(0, rows, name='i0') as i:
        _emit_vextract(tik_instance,
                       indices_out_fp16_ub,
                       result_ub,
                       mode=Mode.X1.value,
                       cnt=cols_padding,
                       dst_offset=i * cols_padding,
                       src_offset=i * cols_padding * 8)
        _emit_vextract(tik_instance,
                       offset_fp16_ub,
                       result_ub,
                       mode=Mode.Y1.value,
                       cnt=cols_padding,
                       dst_offset=i * cols_padding,
                       src_offset=i * cols_padding * 8)
    if not largest:
        _emit_vmuls(tik_instance, data_ub, data_ub, cnt=rows * cols_padding)
    with tik_instance.for_range(0, rows, name='i0') as i:
        _conv_fp162s32(tik_instance, indices_out_int32_ub, i * cols_padding, indices_out_fp16_ub, i * cols_padding,
                       cols_padding)
        _conv_fp162s32(tik_instance, offset_int32_ub, i * cols_padding, offset_fp16_ub, i * cols_padding, cols_padding)
    _add(tik_instance, indices_out_final_ub, indices_out_int32_ub, offset_int32_ub, rows, cols_padding)

    if k >= 8 and k < 16 and tbe_platform.api_check_support("tik.set_atomic_add", "float16"):
        _copy_ubuf_to_gm_k_less_16(tik_instance,
                                   'float16',
                                   data_gm_out,
                                   data_ub,
                                   rows,
                                   cols_padding,
                                   k,
                                   tail_block_ub=data_tail_block_ub,
                                   gm_offset=row_start_in_core * k + core_rows_start * k,
                                   multi_core=multi_core)

        _copy_ubuf_to_gm_k_less_16(tik_instance,
                                   'int32',
                                   indices_gm_out,
                                   indices_out_final_ub,
                                   rows,
                                   cols_padding,
                                   k,
                                   tail_block_ub=indices_tail_block_ub,
                                   gm_offset=row_start_in_core * k + core_rows_start * k,
                                   multi_core=multi_core)
    else:
        _copy_ubuf_to_gm(tik_instance,
                         'float16',
                         data_gm_out,
                         data_ub,
                         rows,
                         cols_padding,
                         k,
                         tail_block_ub=data_tail_block_ub,
                         gm_offset=row_start_in_core * k + core_rows_start * k,
                         multi_core=multi_core)
        _copy_ubuf_to_gm(tik_instance,
                         'int32',
                         indices_gm_out,
                         indices_out_final_ub,
                         rows,
                         cols_padding,
                         k,
                         tail_block_ub=indices_tail_block_ub,
                         gm_offset=row_start_in_core * k + core_rows_start * k,
                         multi_core=multi_core)


def _tiling(rows, cols, cols_limit):
    """
    Funcion for _tiling
    """
    ret = []  # rows for each core

    num_cores = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    if rows < num_cores:
        for i in range(rows):
            ret.append(1)
        return ret, rows, 1
    ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    # the following three of 32 represent UB align of data tail block,
    # indices tail block and vmrgsort4 addr block
    ub_bytes = ub_size_bytes - 32 - 32 - 32
    cols_padding = ((cols + 15) // 16) * 16

    remain = rows % num_cores
    for i in range(num_cores):
        ret.append(rows // num_cores)
    for i in range(remain):
        ret[i] = ret[i] + 1

    if cols <= cols_limit:
        ub_bytes = ub_bytes - cols_padding * 4  # indices_ub
        row_bytes = cols_padding * 50
        batch = ub_bytes // row_bytes
    else:
        batch = 1
    turning = remain
    if remain == 0:
        turning = num_cores

    return ret, turning, batch


# 'pylint: disable=too-many-locals
# 'pylint: disable=too-many-statements
# 'pylint: disable=too-many-branches
def _kernel_ir(tik_instance, ins, outs, k, largest, is_lhisi):
    """
    Funtion for common process in top_k op
    """
    input_a = ins[0]
    indices = ins[1]
    output = outs[0]
    indices_out = outs[1]
    shape = list(input_a.shape)
    cols = int(shape[-1])
    cols_padding = ((cols + 15) // 16) * 16
    rows = 1
    # max_part_num must be a multiple of 16
    if is_lhisi:
        max_part_num = 2080
        cols_limit = 2048
        cols_per_part = 512
    else:
        max_part_num = 1424
        cols_limit = 4096
        cols_per_part = 1024

    for i in range(len(shape) - 1):
        rows = rows * int(shape[i])
    multi_core = True
    rows_cores, turn, batch = _tiling(rows, cols, cols_limit)

    if k < 16 and not tbe_platform.api_check_support("tik.set_atomic_add", "float16"):
        rows_cores = [rows]
        turn = 1
        multi_core = False
    elif k < 8:
        rows_cores = [rows]
        turn = 1
        multi_core = False

    if len(rows_cores) <= 1:
        multi_core = False

    if cols > cols_limit:
        data_ub = tik_instance.Tensor("float16", (cols_per_part,), name="data_ub", scope=tbe_platform.scope_ubuf)
        indices_ub = tik_instance.Tensor("float16", (cols_per_part,), name="indices_ub", scope=tbe_platform.scope_ubuf)
        indices_out_fp16_ub = indices_ub
        indices_out_int32_ub = tik_instance.Tensor("int32", (1, 4096),
                                                   name="indices_out_int32_ub",
                                                   scope=tbe_platform.scope_ubuf)
        indices_out_final_ub = indices_out_int32_ub
        offset_ub = tik_instance.Tensor("float16", (max_part_num,), name="offset_ub", scope=tbe_platform.scope_ubuf)
        offset_fp16_ub = offset_ub
        offset_int32_ub = tik_instance.Tensor("int32", (1, 4096), name="offset_int32_ub", scope=tbe_platform.scope_ubuf)

        region_ub = tik_instance.Tensor("float16", (1, cols_per_part * 8), name="region_ub",
                                        scope=tbe_platform.scope_ubuf)
        region_sorted_ub = tik_instance.Tensor("float16", (1, cols_per_part * 8),
                                               name="region_sorted_ub",
                                               scope=tbe_platform.scope_ubuf)

        # merge sort need cols_per_part * 5 * 8
        region_k_ub = tik_instance.Tensor("float16", (1, cols_per_part * 5 * 8),
                                          name="region_k_ub",
                                          scope=tbe_platform.scope_ubuf)
        region_k2_ub = tik_instance.Tensor("float16", (1, cols_per_part * 5 * 8),
                                           name="region_k2_ub",
                                           scope=tbe_platform.scope_ubuf)

        index_reg = tik_instance.Scalar(dtype="float16", name="index_reg", init_value=1)
    else:
        data_ub = tik_instance.Tensor("float16", (batch, cols_padding), name="data_ub", scope=tbe_platform.scope_ubuf)
        indices_ub = tik_instance.Tensor("float16", (cols_padding,), name="indices_ub", scope=tbe_platform.scope_ubuf)
        indices_out_fp16_ub = tik_instance.Tensor("float16", (batch, cols_padding),
                                                  name="indices_out_fp16_ub",
                                                  scope=tbe_platform.scope_ubuf)
        indices_out_int32_ub = tik_instance.Tensor("int32", (batch, cols_padding),
                                                   name="indices_out_int32_ub",
                                                   scope=tbe_platform.scope_ubuf)
        indices_out_final_ub = tik_instance.Tensor("int32", (batch, cols_padding),
                                                   name="indices_out_final_ub",
                                                   scope=tbe_platform.scope_ubuf)
        offset_ub = tik_instance.Tensor("float16", (cols_padding,), name="offset_ub", scope=tbe_platform.scope_ubuf)
        offset_fp16_ub = tik_instance.Tensor("float16", (batch, cols_padding),
                                             name="offset_fp16_ub",
                                             scope=tbe_platform.scope_ubuf)
        offset_int32_ub = tik_instance.Tensor("int32", (batch, cols_padding),
                                              name="offset_int32_ub",
                                              scope=tbe_platform.scope_ubuf)
        region_ub = tik_instance.Tensor("float16", (batch, cols_padding * 8), name="region_ub",
                                        scope=tbe_platform.scope_ubuf)
        region_sorted_ub = tik_instance.Tensor("float16", (batch, cols_padding * 8),
                                               name="region_sorted_ub",
                                               scope=tbe_platform.scope_ubuf)
        region_k_ub = None
        region_k2_ub = None
        index_reg = None

    data_tail_block_ub = tik_instance.Tensor("float16", (16,), name="data_tail_block_ub", scope=tbe_platform.scope_ubuf)
    indices_tail_block_ub = tik_instance.Tensor("int32", (8,), name="indices_tail_block_ub",
                                                scope=tbe_platform.scope_ubuf)

    reg_min_number = tik_instance.Scalar(dtype="float16", name="reg_min_number", init_value=FP16_MINIMUM)

    GLOBAL_VAR.set_data_gm_out(output)
    GLOBAL_VAR.set_data_ub(data_ub)
    GLOBAL_VAR.set_region_ub(region_ub)
    GLOBAL_VAR.set_region_sorted_ub(region_sorted_ub)
    GLOBAL_VAR.set_region_k_ub(region_k_ub)
    GLOBAL_VAR.set_reg_min_number(reg_min_number)
    GLOBAL_VAR.set_indices_ub(indices_ub)
    GLOBAL_VAR.set_indices_out_fp16_ub(indices_out_fp16_ub)
    GLOBAL_VAR.set_indices_out_int32_ub(indices_out_int32_ub)
    GLOBAL_VAR.set_indices_gm_out(indices_out)
    GLOBAL_VAR.set_data_gm(input_a)
    GLOBAL_VAR.set_indices_gm(indices)
    GLOBAL_VAR.set_region_k2_ub(region_k2_ub)
    GLOBAL_VAR.set_data_tail_block_ub(data_tail_block_ub)
    GLOBAL_VAR.set_indices_tail_block_ub(indices_tail_block_ub)
    GLOBAL_VAR.set_offset_ub(offset_ub)
    GLOBAL_VAR.set_offset_fp16_ub(offset_fp16_ub)
    GLOBAL_VAR.set_offset_int32_ub(offset_int32_ub)
    GLOBAL_VAR.set_indices_out_final_ub(indices_out_final_ub)
    GLOBAL_VAR.index_reg = index_reg
    GLOBAL_VAR.max_part_num = max_part_num

    blocks = len(rows_cores)
    rows_per_core1 = rows_cores[0]
    rows_per_core2 = rows_cores[0] - 1

    loops1 = rows_per_core1 // batch
    loops2 = rows_per_core2 // batch

    remain1 = rows_per_core1 % batch
    remain2 = rows_per_core2 % batch

    with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
        with tik_instance.if_scope(block_index < turn):
            core_rows_start = rows_per_core1 * block_index
            if cols > cols_limit:
                with tik_instance.for_range(0, loops1, name='i0') as i:
                    _topk_a_row_by_part(tik_instance,
                                        row_start_in_core=i,
                                        cols=cols,
                                        k=k,
                                        core_rows_start=core_rows_start,
                                        multi_core=multi_core,
                                        largest=largest,
                                        is_lhisi=is_lhisi)
            else:
                with tik_instance.for_range(0, loops1, name='i0') as i:
                    _topk_rows(tik_instance,
                               row_start_in_core=i * batch,
                               rows=batch,
                               cols=cols,
                               k=k,
                               core_rows_start=core_rows_start,
                               multi_core=multi_core,
                               largest=largest)
                if remain1 > 0:
                    _topk_rows(tik_instance,
                               row_start_in_core=loops1 * batch,
                               rows=remain1,
                               cols=cols,
                               k=k,
                               core_rows_start=core_rows_start,
                               multi_core=multi_core,
                               largest=largest)

        with tik_instance.if_scope(block_index >= turn):
            core_rows_start = (rows_per_core1 * turn) + (rows_per_core2) * (block_index - turn)
            if cols > cols_limit:
                with tik_instance.for_range(0, loops2, name='i0') as i:
                    _topk_a_row_by_part(tik_instance,
                                        row_start_in_core=i,
                                        cols=cols,
                                        k=k,
                                        core_rows_start=core_rows_start,
                                        multi_core=multi_core,
                                        largest=largest,
                                        is_lhisi=is_lhisi)
            else:
                with tik_instance.for_range(0, loops2, name='i0') as i:
                    _topk_rows(tik_instance,
                               row_start_in_core=i * batch,
                               rows=batch,
                               cols=cols,
                               k=k,
                               core_rows_start=core_rows_start,
                               multi_core=multi_core,
                               largest=largest)
                if remain2 > 0:
                    _topk_rows(tik_instance,
                               row_start_in_core=loops2 * batch,
                               rows=remain2,
                               cols=cols,
                               k=k,
                               core_rows_start=core_rows_start,
                               multi_core=multi_core,
                               largest=largest)


# 'pylint: disable=unused-argument,redefined-builtin
def check_supported(input_tensor,
                    indices_tensor,
                    out_tensor,
                    out_indices_tensor,
                    k,
                    sorted=True,
                    dim=-1,
                    largest=True,
                    kernel_name='top_k'):
    """
    check whether ai_core is supported
    max last dim should exist and max last dim of input_tensor should <= 1024 * 2048 and k
    should <= 4096 and last dim of input_tensor should <= 1458176 and k should <= 5120
    sorted should == True
    input size > 32768 and k > 0 and k < 16 three conditions cannot be met at the same time
    """
    shape = input_tensor.get("ori_shape")
    input_size = functools.reduce(lambda x, y: x * y, shape)

    # Special adaptation to pytorch ("sorted" is false indicates the pytorch operator)
    if sorted is not True:
        return True, ""
    # When input_size > 32768, k < 8 or (k < 16 and not support set_stomic_add with fp16),
    # the AICPU performance is better than the AICore performance.
    # k = 0 is set in fe pass when top_k is version two, top_k_v2 cannot check k value in compile phase.
    if input_size > 32768:
        if k > 0 and k < 8:
            reason = "input_size is too big(> 32768), and k is in (0-8), input_size:%s, k:%s" \
                     % (input_size, k)
            return False, reason
        if k > 0 and k < 16 and not tbe_platform.api_check_support("tik.set_atomic_add", "float16"):
            reason = "input_size is too big(> 32768), and k is in (0-16), input_size:%s, k:%s" \
                     % (input_size, k)
            return False, reason

    if input_tensor.get("dtype").lower() in ["float32", "bfloat16"] and shape[dim] == 1:
        reason = "static topkd does not support fp32 and data of sort axis is 1"
        return False, reason
    return True, ""


# 'pylint: disable=too-many-arguments
# 'pylint: disable=too-many-local-variables
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def top_k_d(input_tensor,
            indices_tensor,
            out_tensor,
            out_indices_tensor,
            k,
            sorted=True,
            dim=-1,
            largest=True,
            kernel_name='top_k'):
    """
    Select top K elements from  last dimension
    Parameters
    ----------
    input_tensor: dict.
        Shape and dtype of input to be reversed.
    indices_tensor: dict
    out_tensor: dict.
        Shape and dtype of output.
    out_indices_tensor: dict
    k: int.
        Number of largest elements to be select
    sorted : bool
        if is sorted
    kernel_name : str
        cce kernel name, default value is "cce_topk"
    Returns
    -------
    None
    """
    shape = input_tensor.get("shape")
    input_dtype = input_tensor.get("dtype").lower()
    indices_shape = indices_tensor.get("shape")
    input_indices_dtype = indices_tensor.get("dtype").lower()
    out_shape = out_tensor.get("shape")
    out_dtype = out_tensor.get("dtype").lower()
    out_indices_shape = out_indices_tensor.get("shape")
    out_indices_dtype = out_indices_tensor.get("dtype")
    para_check.check_dtype(input_dtype, ("float16",), param_name='input_tensor')
    para_check.check_dtype(input_indices_dtype, ("float16", "int32"), param_name='indices_tensor')
    para_check.check_dtype(out_dtype, ("float16",), param_name='out_tensor')
    para_check.check_dtype(out_indices_dtype, ("int32",), param_name='out_indices_tensor')
    para_check.check_shape(shape, param_name='input_tensor')
    para_check.check_shape(indices_shape, param_name='indices_tensor')
    para_check.check_shape(out_shape, param_name='out_tensor')
    para_check.check_shape(out_indices_shape, param_name='out_indices_tensor')

    shape_dim = len(shape)
    out_shape_dim = len(out_shape)

    if dim not in (-1, shape_dim - 1):
        error_manager_vector.raise_err_check_params_rules(kernel_name,
                                                          "Dim should equal last dim of shape, actural dim is %d" % dim)

    if shape_dim != out_shape_dim:
        error_manager_vector.raise_err_check_params_rules(
            kernel_name, "input_tensor and out_tensor dim not equal, the dim of input_tensor is %d" % shape_dim,
            'out_tensor', out_shape_dim)
    if list(shape[:-1]) != list(out_shape[:-1]):
        error_manager_vector.raise_err_check_params_rules(
            kernel_name,
            "input_tensor's shape must be same as out_tensor's shape expect the last dim, input_tensor's shape is %s" %
            shape, 'out_tensor', out_shape)
    if out_shape[-1] != k:
        error_manager_vector.raise_err_check_params_rules(kernel_name,
                                                          "output tensor last dim must equal to k, k is %d" % k,
                                                          'out_tensor', out_shape)

    if k < 1 or k > shape[-1]:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, 'k', 1, shape[-1], k)

    soc_version = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
    is_lhisi = soc_version in ("Hi3796CV300CS", "Hi3796CV300ES", "SD3403")
    if is_lhisi:
        # 3000 indicates max size of the last dimension for better performance
        # 2048 indicates max k_num due to ub memory limitation
        max_last_size = 3000
        max_k_num = 2048
    else:
        # 5000 indicates max size of the last dimension for better performance
        # 4096 indicates max k_num due to ub memory limitation
        max_last_size = 5000
        max_k_num = 4096
    if int(shape[-1]) > max_last_size or k > max_k_num:
        return top_k_large(shape, indices_shape, out_shape, k, input_dtype, input_indices_dtype,
                           out_indices_dtype, largest, kernel_name)

    if tbe_platform.api_check_support("tik.vbitsort32"):
        return top_k_template(input_tensor,
                              indices_tensor,
                              out_tensor,
                              out_indices_tensor,
                              k,
                              sorted=sorted,
                              dim=dim,
                              largest=largest,
                              kernel_name=kernel_name,
                              mode="static")

    tik_instance = tik.Tik()
    data_input = tik_instance.Tensor(input_dtype, shape, name='data_a', scope=tbe_platform.scope_gm)
    indices = tik_instance.Tensor(input_indices_dtype, indices_shape, name='indices', scope=tbe_platform.scope_gm)
    if k >= 8 and k < 16 and tbe_platform.api_check_support("tik.set_atomic_add", "float16"):
        res = tik_instance.Tensor(input_dtype, out_shape, name='res', scope=tbe_platform.scope_gm, is_atomic_add=True)
    else:
        res = tik_instance.Tensor(input_dtype, out_shape, name='res', scope=tbe_platform.scope_gm)
    indices_out = tik_instance.Tensor(out_indices_dtype, out_shape, name='indices_out', scope=tbe_platform.scope_gm)

    ins = [data_input, indices]
    outs = [res, indices_out]

    _kernel_ir(tik_instance, ins, outs, k, largest, is_lhisi)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=ins, outputs=outs, enable_l2=True)
