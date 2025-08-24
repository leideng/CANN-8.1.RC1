#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
max_pool_grad
"""
# 'pylint: disable=too-many-lines
import math
import te.platform as tbe_platform_check
from impl.max_pool_grad import op_select_format as op_select_format_static
from impl.util import util_common
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len
from tbe import tvm
import tbe
from tbe.common.utils import shape_util
from tbe.dsl.compute.pooling.max_pool_grad import max_pool_grad as max_pool_grad_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for Constant
    """
    # vector_repeat
    MAX_REPEAT = 255
    # block_size
    MINI_UNIT = 32
    # mini value of fp16
    MIN_VALUE_FP16 = -65535.0
    # vector fp16 size
    VECTOR_FP16_SIZE = 128
    # vector fp32 size
    VECTOR_FP32_SIZE = 64
    # vconv mask
    MASK64_VALUE = 64
    # maximum dma_copy stride
    MAX_STRIDE = 65535
    # UB SIZE
    SIZE_UB = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    # L1 SIZE
    SIZE_L1 = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
    # BLOCK NUMS
    MAX_CORE = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    # VALID MASK BITS FOR 128
    MASK128_VALUE = 128
    # SRC0STRIDEM1
    SRC0STRIDEM1 = 8
    # SRC1STRIDEM1
    SRC1STRIDEM1 = 8
    # max block stride for vector
    VEC_MAX_STRIDE = 255
    #reserve ub size
    RESERVE_UB_SIZE = 288
    #reserve ub size for 1971
    RESERVE_UB_SIZE_Milan = 3072

    TILING_PARAM_DTYPE = "int64"
    TILING_PARAMS_NUM = 136
    MAX_INT32 = 2 ** 31 - 1
    C0 = 16

    CASE_ZERO = 0
    CASE_ONE = 1
    CASE_TWO = 2
    CASE_THREE = 3
    CASE_FORE = 4
    CASE_FIVE = 5
    CASE_SIX = 6
    CASE_SEVEN = 7


# 'pylint: disable=too-few-public-methods,too-many-statements,too-many-branches,no-self-use,huawei-too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-many-lines
# 'pylint: disable=too-many-lines,too-many-locals,too-many-statements,unused-variable,too-many-arguments
def op_select_format(x1, x2, grad, y, ksize, strides,
                     padding, data_format="NHWC", kernel_name="maxpoolgrad"):
    """
    Operator output under dynamic shape only supports float32,
    fixed shape output only supports float16
    """
    return op_select_format_static(x1, x2, grad, y, ksize, strides, padding,
                                   data_format=data_format, kernel_name=kernel_name)


# 'pylint: disable=too-many-locals,too-many-arguments,invalid-name,too-many-locals,no-self-use,too-few-public-methods
# 'pylint: disable=too-many-statements,unused-variable,too-many-branches,too-many-instance-attributes,unused-argument
def _init_coordinate_dy(tik_instance, pad_x_top, xi_coordinate):
    """
    init_coordinate
    """
    xi_coord = tik_instance.Scalar(dtype='int64', name='xi_coord')
    with tik_instance.if_scope(pad_x_top != 0):
        with tik_instance.if_scope(xi_coordinate < 0):
            xi_coord.set_as(0)
        with tik_instance.else_scope():
            xi_coord.set_as(xi_coordinate)
    with tik_instance.else_scope():
        xi_coord.set_as(xi_coordinate)

    return xi_coord


def _calc_pad_dy(tik_instance, pad_top, pad_bottom, xi_coord, xi_value, boundary):
    """
    calc_pad
    """
    # return pad_value in different axis
    top = tik_instance.Scalar(dtype='int64', name='top_data')
    bottom = tik_instance.Scalar(dtype='int64', name='bottom_data')
    top.set_as(pad_top)
    bottom.set_as(pad_bottom)

    with tik_instance.if_scope(pad_top != 0):
        with tik_instance.if_scope(xi_coord < 0):
            top.set_as(0 - xi_coord)
        with tik_instance.else_scope():
            top.set_as(0)
    with tik_instance.if_scope(pad_bottom != 0):
        with tik_instance.if_scope(xi_coord + xi_value > boundary):
            bottom.set_as(xi_coord + xi_value - boundary)
        with tik_instance.else_scope():
            bottom.set_as(0)
    return top, bottom


def _calc_pad_dy1(tik_instance, pad_top, pad_bottom, xi_coord, xi_value, boundary):
    """
    calc_pad
    """
    top1 = tik_instance.Scalar(dtype='int64', name='top1')
    bottom1 = tik_instance.Scalar(dtype='int64', name='bottom1')
    top1.set_as(pad_top)
    bottom1.set_as(pad_bottom)

    with tik_instance.if_scope(pad_top != 0):
        with tik_instance.if_scope(xi_coord < 0):
            top1.set_as(0 - xi_coord)
        with tik_instance.else_scope():
            top1.set_as(0)
    with tik_instance.if_scope(pad_bottom != 0):
        with tik_instance.if_scope(xi_coord + xi_value > boundary):
            bottom1.set_as(xi_coord + xi_value - boundary)
        with tik_instance.else_scope():
            bottom1.set_as(0)
    return top1, bottom1


class ParamsUB:
    """
    Function: use to store concat base parameters
    """

    def __init__(self, sh, sw, kh, kw, check_load3d_support):
        self.cur_ub_size = Constant.SIZE_UB
        if not check_load3d_support:
            self.cur_ub_size -= Constant.RESERVE_UB_SIZE_Milan
        if check_load3d_support:
            self.l1_in_size = Constant.SIZE_L1 // 2
        else:
            self.l1_in_size = 0
        self.col_in_size = ((self.cur_ub_size - 256) // (198 + 64 * sh * sw)) * Constant.C0
        if self.col_in_size < 256:
            self.col_in_size = 256
        self.forward_ou_size = self.col_in_size
        self.mask_size = (self.cur_ub_size - 256) // (198 + 64 * sh * sw)
        if self.mask_size < 128:
            self.mask_size = 128
        self.grad_size = self.col_in_size
        self.zero_size = 128
        self.grad_sel_fp16_size = self.col_in_size
        self.grad_sel_fp32_size = self.col_in_size
        used_ub_byte = (self.col_in_size + self.forward_ou_size + self.mask_size * 3 +
                        self.grad_size + self.zero_size +
                        self.grad_sel_fp16_size) * 2
        if check_load3d_support:
            self.ori_in_size = 0
            self.f_map_fp32_size = (self.cur_ub_size - Constant.RESERVE_UB_SIZE - used_ub_byte) // 4 - \
                                   self.grad_sel_fp32_size
        else:
            self.f_map_fp32_size = ((self.cur_ub_size - Constant.RESERVE_UB_SIZE - \
                                    used_ub_byte) // 4 -
                                    self.grad_sel_fp32_size) // 2
            self.ori_in_size = self.f_map_fp32_size * 2
        if self.f_map_fp32_size < 16:
            error_manager_vector.raise_err_specific_reson("max_pool_grad", "ub is error")


def _ultimate_data_move(tik_instance, src_buf, dst_buf, in_list, num_bit):
    """
    move ub to gm
    """
    src_idx, dst_idx = in_list[-2], in_list[-1]
    n_burst, burst_len = in_list[0], in_list[1]
    src_stride, dst_stride = in_list[2], in_list[3]

    with tik_instance.for_range(0, n_burst) as i:
        src_idx += i * (src_stride + burst_len) * Constant.MINI_UNIT // num_bit
        dst_idx += i * (dst_stride + burst_len) * Constant.MINI_UNIT // num_bit

        tik_instance.data_move(dst_buf[dst_idx], src_buf[src_idx], 0, 1, burst_len, 0, 0)


def _ultimate_data_move_dy(tik_instance, src_buf, dst_buf, in_list, num_bit):
    """
    dynamic move ub to gm
    """
    src_idx, dst_idx = in_list[-2], in_list[-1]
    n_burst, burst_len = in_list[0], in_list[1]
    src_stride, dst_stride = in_list[2], in_list[3]

    tik_instance.data_move(dst_buf[dst_idx], src_buf[src_idx], 0, 1, burst_len, 0, 0)


def _norm_data_move(tik_instance, src_buf, dst_buf, in_list):
    """
    _norm_data_move
    """
    src_idx, dst_idx = in_list[-2], in_list[-1]
    n_burst, burst_len = in_list[0], in_list[1]
    src_stride, dst_stride = in_list[2], in_list[3]

    tik_instance.data_move(dst_buf[dst_idx], src_buf[src_idx], 0, n_burst, burst_len, src_stride, dst_stride)


def _set_vector_dup_zero(tik_instance, dst, idx, number, dtype):
    """
    set_vector_dup
    """
    if dtype == "float16":
        mask = 128
    else:
        mask = 64
    dst_blk_stride = 1
    dst_rep_stride = 8
    repeats = 1
    tik_instance.vector_dup(mask, dst[idx], number, repeats, dst_blk_stride, dst_rep_stride)


def _set_vector_dup_dy(tik_instance, dst, idx, number, dtype, dup_repeat_merchant, dup_repeat_remainder,
                       dup_remainder, repeats, dst_size):
    """
    set_vector_dup
    """
    if dtype == "float16":
        mask = 128
    else:
        mask = 64
    dup_psm = Constant.MAX_REPEAT * mask
    dup_repeat_merchant_g = dst_size // dup_psm

    dst_blk_stride = 1
    dst_rep_stride = 8
    if dup_repeat_merchant_g > 0:
        with tik_instance.for_range(0, dup_repeat_merchant) as i:
            tik_instance.vector_dup(mask, dst[idx + i * dup_psm], number, Constant.MAX_REPEAT, dst_blk_stride,
                                    dst_rep_stride)

    with tik_instance.if_scope(dup_repeat_remainder != 0):
        if dst_size >= mask:
            with tik_instance.if_scope(repeats != 0):
                tik_instance.vector_dup(mask, dst[idx + dup_repeat_merchant * dup_psm], number, repeats,
                                        dst_blk_stride, dst_rep_stride)
        with tik_instance.if_scope(dup_remainder != 0):
            tik_instance.vector_dup(dup_remainder, dst[idx + dup_repeat_merchant * dup_psm + repeats * mask],
                                    number, 1, dst_blk_stride, dst_rep_stride)


def _vconv_dy(tik_instance, src, src_start, dst, dst_start, src_dtype, repeat_max_time, remain_repeat_time,
              remain_ele, dstsize):
    """
    vconv
    """
    mask_value = Constant.VECTOR_FP32_SIZE

    if src_dtype == 'float16':
        src_stride, dst_stride = 4, 8
        if dstsize >= Constant.MASK64_VALUE * Constant.MAX_REPEAT:
            with tik_instance.if_scope(repeat_max_time > 0):
                with tik_instance.for_range(0, repeat_max_time) as loop1:
                    tik_instance.vconv(Constant.MASK64_VALUE, "",
                                       dst[dst_start + loop1 * Constant.MAX_REPEAT * mask_value],
                                       src[src_start + loop1 * Constant.MAX_REPEAT * mask_value],
                                       Constant.MAX_REPEAT, 1, 1, dst_stride, src_stride)
        if dstsize >= Constant.MASK64_VALUE:
            with tik_instance.if_scope(remain_repeat_time > 0):
                tik_instance.vconv(Constant.MASK64_VALUE, "",
                                   dst[dst_start + repeat_max_time * Constant.MAX_REPEAT * mask_value],
                                   src[src_start + repeat_max_time * Constant.MAX_REPEAT * mask_value],
                                   remain_repeat_time, 1, 1, dst_stride, src_stride)
        with tik_instance.if_scope(remain_ele > 0):
            tik_instance.vconv(
                remain_ele, "",
                dst[dst_start + repeat_max_time * Constant.MAX_REPEAT * mask_value +
                    remain_repeat_time * mask_value],
                src[src_start + repeat_max_time * Constant.MAX_REPEAT * mask_value +
                    remain_repeat_time * mask_value],
                1, 1, 1, dst_stride, src_stride)

    else:
        src_stride, dst_stride = 8, 4
        with tik_instance.if_scope(repeat_max_time > 0):
            with tik_instance.for_range(0, repeat_max_time) as loop1:
                tik_instance.vconv(Constant.MASK64_VALUE, "",
                                   dst[dst_start + loop1 * Constant.MAX_REPEAT * mask_value],
                                   src[src_start + loop1 * Constant.MAX_REPEAT * mask_value],
                                   Constant.MAX_REPEAT, 1, 1, dst_stride, src_stride)
        with tik_instance.if_scope(remain_repeat_time > 0):
            tik_instance.vconv(Constant.MASK64_VALUE, "",
                               dst[dst_start + repeat_max_time * Constant.MAX_REPEAT * mask_value],
                               src[src_start + repeat_max_time * Constant.MAX_REPEAT * mask_value],
                               remain_repeat_time, 1, 1, dst_stride, src_stride)
        with tik_instance.if_scope(remain_ele > 0):
            tik_instance.vconv(
                remain_ele, "",
                dst[dst_start + repeat_max_time * Constant.MAX_REPEAT * mask_value +
                    remain_repeat_time * mask_value],
                src[src_start + repeat_max_time * Constant.MAX_REPEAT * mask_value +
                    remain_repeat_time * mask_value],
                1, 1, 1, dst_stride, src_stride)


def _vector_op_dy(tik_instance, operator, src1, src2, dst, dtype, stride_config, repeat_max_loop,
                  remain_max_loop, remain_ele, src1_size, src2_size, dst_size):
    """
    vadd
    """
    stride_config = list(stride_config)
    if dtype == "float16":
        mask = Constant.VECTOR_FP16_SIZE
    else:
        mask = Constant.VECTOR_FP32_SIZE

    config_num = stride_config[0] <= 255 and stride_config[1] <= 255 and stride_config[2] <= 255 and stride_config[
        3] <= 255 and stride_config[4] <= 255 and stride_config[5] <= 255
    config_num_3 = stride_config[0] <= 255 and stride_config[1] <= 255 and stride_config[2] <= 255

    if operator == "vadd":
        if stride_config is None:
            stride_config = 1, 1, 1, 8, 8, 8
        dst_offset = 0
        src1_offset = 0
        src2_offset = 0
        dst_less = (stride_config[0] * 8 + stride_config[3] * Constant.MAX_REPEAT) * 8
        src1_less = (stride_config[1] * 8 + stride_config[4] * Constant.MAX_REPEAT) * 8
        src2_size_less = (stride_config[2] * 8 + stride_config[5] * Constant.MAX_REPEAT) * 8

        dst_less1 = (stride_config[0] * 8 + stride_config[3]) * 8
        src1_less1 = (stride_config[1] * 8 + stride_config[4]) * 8
        src2_size1_less = (stride_config[2] * 8 + stride_config[5]) * 8

        if dst_size >= dst_less and src1_size >= src1_less and src2_size >= src2_size_less and config_num:
            with tik_instance.if_scope(repeat_max_loop > 0):
                tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset], src2[src2_offset], Constant.MAX_REPEAT,
                                  stride_config[0], stride_config[1], stride_config[2], stride_config[3],
                                  stride_config[4], stride_config[5])
                dst_offset += Constant.MINI_UNIT // (get_bit_len(dst.dtype.lower()) //
                                            8) * stride_config[3] * 255
                src1_offset += Constant.MINI_UNIT // (get_bit_len(src1.dtype.lower()) //
                                             8) * stride_config[4] * 255
                src2_offset += Constant.MINI_UNIT // (get_bit_len(src2.dtype.lower()) //
                                             8) * stride_config[5] * 255

        with tik_instance.if_scope(remain_max_loop > 0):
            if config_num_3 and dst_size >= stride_config[0] * mask and src1_size >= stride_config[1] * mask \
                    and src2_size >= stride_config[2] * mask:
                with tik_instance.if_scope(remain_max_loop == 1):
                    tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset], src2[src2_offset], remain_max_loop,
                                      stride_config[0], stride_config[1], stride_config[2], 0, 0, 0)
            if config_num and dst_size >= dst_less1 and src1_size >= src1_less1 and src2_size >= src2_size1_less:
                with tik_instance.if_scope(remain_max_loop != 1):
                    tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset], src2[src2_offset], remain_max_loop,
                                      stride_config[0], stride_config[1], stride_config[2], stride_config[3],
                                      stride_config[4], stride_config[5])
            dst_offset += Constant.MINI_UNIT // (get_bit_len(dst.dtype.lower()) //
                                                 8) * stride_config[3] * remain_max_loop
            src1_offset += Constant.MINI_UNIT // (get_bit_len(src1.dtype.lower()) //
                                                  8) * stride_config[4] * remain_max_loop
            src2_offset += Constant.MINI_UNIT // (get_bit_len(src2.dtype.lower()) //
                                                  8) * stride_config[5] * remain_max_loop
        with tik_instance.if_scope(remain_ele > 0):
            stride_config[3] = stride_config[4] = stride_config[5] = 0
            tik_instance.vadd(remain_ele, dst[dst_offset], src1[src1_offset], src2[src2_offset], 1,
                              stride_config[0], stride_config[1], stride_config[2], stride_config[3],
                              stride_config[4], stride_config[5])


def _rewrite_fmap_dy(tik_instance, operator, src1, src2, dst, dtype, repeat_times, shape_map, shape_grad,
                     config, num_instr_loop_h, num_instr_loop_w, remain_mask, remain_repeat, num_instr_loop_w_1,
                     num_instr_loop_h_1):
    """
    vadd
    """
    mask = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="mask_params")
    rep = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="rep_params")

    h, w, c0 = shape_map[0], shape_map[1], 16
    ho, wo = shape_grad[0], shape_grad[1]
    if dtype == "float16":
        max_mask = 128
        num_block = 8
        block_size = 16
    else:
        max_mask = 64
        num_block = 8
        block_size = 8

    with tik_instance.for_range(0, num_instr_loop_h) as idx_h:
        with tik_instance.for_range(0, num_instr_loop_w) as idx_w:
            src1_offset = idx_w * num_block * config[1] * block_size + \
                          idx_h * Constant.MAX_REPEAT * w * c0
            src2_offset = idx_w * num_block * config[2] * block_size + \
                          idx_h * Constant.MAX_REPEAT * wo * c0
            dst_offset = idx_w * num_block * config[0] * block_size + \
                         idx_h * Constant.MAX_REPEAT * w * c0

            with tik_instance.if_scope(idx_w < num_instr_loop_w_1):
                mask.set_as(max_mask)
            with tik_instance.else_scope():
                mask.set_as(remain_mask)
            with tik_instance.if_scope(idx_h < num_instr_loop_h_1):
                rep.set_as(Constant.MAX_REPEAT)

            with tik_instance.else_scope():
                rep.set_as(remain_repeat)

            tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset], src2[src2_offset], rep, config[0],
                              config[1], config[2], config[3], config[4], config[5])


class MaxPoolGradCompute:
    """
    Function: use to store concat base parameters
    """

    def __init__(self, params1):
        self.core_ou_shape = []
        self.core_in_shape = []
        self.c0 = Constant.C0
        self.ksize = params1[0]
        self.strides = params1[1]
        self.pads = params1[2]
        self.dtype = params1[3]
        self.kernel_name = params1[4]
        if params1[5] == "NHWC":
            self.kh = self.ksize[1]
            self.kw = self.ksize[2]
            self.sh = self.strides[1]
            self.sw = self.strides[2]
        else:
            self.kh = self.ksize[2]
            self.kw = self.ksize[3]
            self.sh = self.strides[2]
            self.sw = self.strides[3]

        self.num_bit = 2
        self.num_bit_fp32 = 4
        self.mask_fp16 = 128
        self.mask_fp32 = 64
        self.tik_instance = tik.Tik()
        self.check_load3d_support = tbe_platform_check.cce_conf.api_check_support("tik.load3dv1")
        self.cur_ub_size = Constant.SIZE_UB if self.check_load3d_support else \
            Constant.SIZE_UB - Constant.RESERVE_UB_SIZE_Milan
        self.ub_maxsize = self.cur_ub_size // self.num_bit
        self.L1_maxsize = Constant.SIZE_L1 // self.num_bit
        self.orig_x_gm = None
        self.orig_y_gm = None
        self.grads_gm = None
        self.ou_y_gm = None

        class GmTensor():
            """
            Function: use to store concat base parameters
            Modify : 2020-12-9
            """

            def __init__(self, tik_instance):
                """
                constructor of class GmTensor

                Parameters
                ----------
                tik_instance: tik_instance
                input_dtype: x dtype
                ids_dtype: ids dtype
                num_segments_dtype: num_segments dtype

                Returns
                -------
                None
                """
                self.tiling_gm = tik_instance.Tensor(Constant.TILING_PARAM_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                     name="tiling_gm",
                                                     scope=tik.scope_gm)

        # scalar of tiling params
        class CommonScalar():
            """
            Function: use to store concat base parameters
            """

            def __init__(self, tik_instance, pads):
                """
                constructor of class CommonScalar

                Parameters
                ----------
                tik_instance: tik_instance
                num_segments_dtype: num_segments dtype
                ids_dtype: ids dtype

                Returns
                -------
                None
                """
                self.select_key = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="select_key")
                self.n = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="forward_in_shape_n")
                self.c1 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="forward_in_shape_c1")
                self.h = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="h")
                self.w = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="w")
                self.ho = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="grad_shape_h")
                self.wo = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="grad_shape_w")
                self.pad_hw_top = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="pad_hw_top")
                self.pad_hw_bottom = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="pad_hw_bottom")
                self.pad_hw_left = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="pad_hw_left")
                self.pad_hw_right = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="pad_hw_right")
                self.overlap_h = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="overlap_h")
                self.overlap_w = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="overlap_w")
                self.hi_invalid = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="hi_invalid")
                self.wi_invalid = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="wi_invalid")
                self.total_num = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="total_num")
                self.core_num = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="real_core_num")
                self.forward_in_shape = [self.n, self.c1, self.h, Constant.C0]
                self.grad_shape = [self.n, self.c1, self.ho, self.wo, Constant.C0]
                self.forward_ou_shape = self.grad_shape
                self.ou_shape = self.forward_in_shape
                self.pad = []
                if pads == "VALID":
                    self.pad = [[0, 0], [0, 0]]
                else:
                    self.pad = [[self.pad_hw_top, self.pad_hw_bottom], [self.pad_hw_left, self.pad_hw_right]]
                self.core_ou_shape_h = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="core_ou_shape_h")
                self.core_ou_shape_w = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="core_ou_shape_w")
                self.core_in_shape_h = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="core_in_shape_h")
                self.core_in_shape_w = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="core_in_shape_w")
                self.new_ho = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="new_ho")
                self.new_wo = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="new_wo")
                self.total_num_div_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="total_num_div_core")
                self.total_num_div_core_1 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                name="total_num_div_core")
                self.core_loop_params = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="core_loop_params")
                self.core_loop_params1 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                             name="core_loop_params1")
                self.core_ou_shape = [self.core_ou_shape_h, self.core_ou_shape_w, Constant.C0]
                self.core_in_shape = [self.core_in_shape_h, self.core_in_shape_w, Constant.C0]
                self.hi_batch = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="hi_batch")
                self.wi_batch = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="wi_batch")
                self.wi_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="wi_tail")
                self.wo_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="wo_tail")
                self.loop_ho = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="loop_ho")
                self.loop_wo = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="loop_ho")
                self.dup_repeat_merchant_f_map_fp32 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                          name="dup_repeat_merchant_f_map_fp32")
                self.dup_repeat_remainder_f_map_fp32 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                           name="dup_repeat_remainder_f_map_fp32")
                self.dup_remainder_f_map_fp32 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                    name="dup_remainder_f_map_fp32")
                self.repeats_f_map_fp32 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="repeats_f_map_fp32")
                self.forward_in_shape_h_w_c0 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="forward_in_shape_h_w_c0")
                self.forward_ou_shape_h_w_c0 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="forward_ou_shape_h_w_c0")
                self.hi_val = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="hi_val")
                self.wi_val = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="wi_val")
                self.burst_len = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="burst_len")
                self.src_stride = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="src_stride")
                self.burst_len_src_orig_y = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                name="burst_len_src_orig_y")
                self.src_stride_src_orig_y = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                 name="src_stride_src_orig_y")
                self.repeat_times = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="repeat_times")
                self.howo_co_ver = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="howo_co_ver")
                self.mask_size_16 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="mask_size_16")
                self.mask_size_ver = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="mask_size_ver")
                self.repeat_max_time_grad_sel = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                    name="repeat_max_time_grad_sel")
                self.remain_repeat_time_grad_sel = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                       name="remain_repeat_time_grad_sel")
                self.remain_ele_grad_sel = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                               name="remain_ele_grad_sel")
                self.repeat_max_loop_vadd = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                name="repeat_max_loop_vadd")
                self.remain_max_loop_vadd = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                name="remain_max_loop_vadd")
                self.remain_ele_vadd = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="remain_ele_vadd")
                self.src_stride_ub_2_gm = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="src_stride_ub_2_gm")
                self.dst_stride_ub_2_gm = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="dst_stride_ub_2_gm")
                self.repeat_max_loop_f_map_fp32 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                      name="repeat_max_loop_f_map_fp32")
                self.remain_max_loop_f_map_fp32 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                      name="remain_max_loop_f_map_fp32")
                self.remain_ele_f_map_fp32 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                 name="remain_ele_f_map_fp32")
                self.wi_val_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="wi_val_tail")
                self.burst_len_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="burst_len_tail")
                self.src_stride_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="src_stride_tail")
                self.pad_hw_top_neg = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="pad_hw_top_neg")
                self.pad_hw_left_neg = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="pad_hw_left_neg")
                self.forward_in_shape_h_w_2 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="forward_in_shape_h_w_2")
                self.burst_len_src_orig_y_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                     name="burst_len_src_orig_y_tail")
                self.src_stride_src_orig_y_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                      name="src_stride_src_orig_y_tail")
                self.repeat_times_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                             name="repeat_times_tail")
                self.howo_co_ver_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="howo_co_ver_tail")
                self.repeat_max_loop_vadd_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                     name="repeat_max_loop_vadd_tail")
                self.remain_max_loop_vadd_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                     name="remain_max_loop_vadd_tail")
                self.remain_ele_vadd_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                name="remain_ele_vadd_tail")
                self.src_stride_ub_2_gm_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="src_stride_ub_2_gm_tail")
                self.dst_stride_ub_2_gm_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="dst_stride_ub_2_gm_tail")
                self.core_ho_times = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="core_ho_times")
                self.core_wo_times = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="core_wo_times")
                self.map_hi = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="map_hi")
                self.map_wi = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="map_wi")

                self.config = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="config")
                self.sh_wi_2 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="sh_wi_2")
                self.num_instr_loop_h = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="num_instr_loop_h")
                self.num_instr_loop_w = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="num_instr_loop_w")
                self.remain_mask = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="remain_mask")
                self.remain_repeat = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="remain_repeat")
                self.num_instr_loop_w_1 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="num_instr_loop_w_1")
                self.num_instr_loop_h_1 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="num_instr_loop_h_1")
                self.ho_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="ho_tail")
                self.hi_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="hi_tail")
                self.dst_stride_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="dst_stride_tail")
                self.wo_2 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="wo_2")
                self.boundary_h = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="boundary_h")
                self.burst_len_ub_2_gm = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                             name="burst_len_ub_2_gm")
                self.non_overlap_1 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="non_overlap_1")
                self.overlap_1 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="poverlap_1")
                self.burst_len_over = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="burst_len_over")
                self.src_stride_over = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="src_stride_over")
                self.dst_stride_over = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="dst_stride_over")
                self.dup_repeat_merchant_non_overlap = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                           name="dup_repeat_merchant_non_overlap")
                self.dup_repeat_remainder_non_overlap = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                            name="dup_repeat_remainder_non_overlap")
                self.dup_remainder_non_overlap = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                     name="dup_remainder_non_overlap")
                self.repeats_non_overlap = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                               name="repeats_non_overlap")
                self.burst_len_ub2gm_2 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                             name="burst_len_ub2gm_2")
                self.src_stride_ub2gm_2 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="src_stride_ub2gm_2")
                self.dst_stride_ub2gm_2 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="dst_stride_ub2gm_2")
                self.burst_len_ub2gm_3 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                             name="burst_len_ub2gm_3")
                self.src_stride_ub2gm_3 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="src_stride_ub2gm_3")
                self.dst_stride_ub2gm_3 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="dst_stride_ub2gm_3")
                self.hi_val_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="hi_val_tail")
                self.burst_len_val = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="hi_val_tail")
                self.src_stride_val = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="src_stride_val")
                self.dst_stride_val = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="dst_stride_val")
                self.burst_len_val_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="burst_len_val_tail")
                self.src_stride_val_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                               name="src_stride_val_tail")
                self.dst_stride_val_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                               name="dst_stride_val_tail")
                self.num_instr_loop_h_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                 name="num_instr_loop_h_tail")
                self.remain_repeat_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="remain_repeat_tail")
                self.num_instr_loop_h_1_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="remain_repeat_tail")
                self.burst_len_ub_2_gm_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="burst_len_ub_2_gm_tail")
                self.non_overlap_1_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="non_overlap_1_tail")
                self.src_stride_over_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                name="src_stride_over_tail")
                self.dst_stride_over_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                name="dst_stride_over_tail")
                self.dup_repeat_merchant_non_overlap_tail = tik_instance.Scalar(
                    dtype=Constant.TILING_PARAM_DTYPE, name="dup_repeat_merchant_non_overlap_tail")
                self.dup_repeat_remainder_non_overlap_tail = tik_instance.Scalar(
                    dtype=Constant.TILING_PARAM_DTYPE, name="dup_repeat_remainder_non_overlap_tail")
                self.dup_remainder_non_overlap_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                          name="dup_remainder_non_overlap_tail")
                self.repeats_non_overlap_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                    name="repeats_non_overlap_tail")
                self.burst_len_ub2gm_2_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="burst_len_ub2gm_2_tail")
                self.src_stride_ub2gm_2_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="src_stride_ub2gm_2_tail")
                self.dst_stride_ub2gm_2_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="dst_stride_ub2gm_2_tail")
                self.burst_len_ub2gm_3_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="burst_len_ub2gm_3_tail")
                self.src_stride_ub2gm_3_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="src_stride_ub2gm_3_tail")
                self.dst_stride_ub2gm_3_tail = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="dst_stride_ub2gm_3_tail")
                self.forward_in_shape_w_c0 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                 name="forward_in_shape_w_c0")
                self.dst_stride = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="dst_stride")


        self.params = CommonScalar(self.tik_instance, self.pads)
        self.gm = GmTensor(self.tik_instance)
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor(Constant.TILING_PARAM_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                 name="tiling_ub",
                                                 scope=tik.scope_ubuf)
            # mov tiling params from gm to ub
            self.tik_instance.data_move(tiling_ub, self.gm.tiling_gm, 0, 1, Constant.TILING_PARAMS_NUM * 8 // 32, 0, 0)
            # input scalar in flowtable
            i = 0
            # common params
            self.params.select_key.set_as(tiling_ub[i])
            i = i + 1
            self.params.n.set_as(tiling_ub[i])
            i = i + 1
            self.params.c1.set_as(tiling_ub[i])
            i = i + 1
            self.params.h.set_as(tiling_ub[i])
            i = i + 1
            self.params.w.set_as(tiling_ub[i])
            i = i + 1
            self.params.ho.set_as(tiling_ub[i])
            i = i + 1
            self.params.wo.set_as(tiling_ub[i])
            i = i + 1
            self.params.pad_hw_top.set_as(tiling_ub[i])
            i = i + 1
            self.params.pad_hw_bottom.set_as(tiling_ub[i])
            i = i + 1
            self.params.pad_hw_left.set_as(tiling_ub[i])
            i = i + 1
            self.params.pad_hw_right.set_as(tiling_ub[i])
            i = i + 1
            self.params.overlap_h.set_as(tiling_ub[i])
            i = i + 1
            self.params.overlap_w.set_as(tiling_ub[i])
            i = i + 1
            self.params.hi_invalid.set_as(tiling_ub[i])
            i = i + 1
            self.params.wi_invalid.set_as(tiling_ub[i])
            i = i + 1
            self.params.total_num.set_as(tiling_ub[i])
            i = i + 1
            self.params.core_num.set_as(tiling_ub[i])
            i = i + 1
            self.params.core_ou_shape_h.set_as(tiling_ub[i])
            i = i + 1
            self.params.core_ou_shape_w.set_as(tiling_ub[i])
            i = i + 1
            self.params.core_in_shape_h.set_as(tiling_ub[i])
            i = i + 1
            self.params.core_in_shape_w.set_as(tiling_ub[i])
            i = i + 1
            self.params.new_ho.set_as(tiling_ub[i])
            i = i + 1
            self.params.new_wo.set_as(tiling_ub[i])
            i = i + 1
            self.params.total_num_div_core.set_as(tiling_ub[i])
            i = i + 1
            self.params.total_num_div_core_1.set_as(tiling_ub[i])
            i = i + 1
            self.params.core_loop_params.set_as(tiling_ub[i])
            i = i + 1
            self.params.core_loop_params1.set_as(tiling_ub[i])
            i = i + 1
            self.params.hi_batch.set_as(tiling_ub[i])
            i = i + 1
            self.params.wi_batch.set_as(tiling_ub[i])
            i = i + 1
            self.params.wi_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.wo_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.loop_ho.set_as(tiling_ub[i])
            i = i + 1
            self.params.loop_wo.set_as(tiling_ub[i])
            i = i + 1
            self.params.dup_repeat_merchant_f_map_fp32.set_as(tiling_ub[i])
            i = i + 1
            self.params.dup_repeat_remainder_f_map_fp32.set_as(tiling_ub[i])
            i = i + 1
            self.params.dup_remainder_f_map_fp32.set_as(tiling_ub[i])
            i = i + 1
            self.params.repeats_f_map_fp32.set_as(tiling_ub[i])
            i = i + 1
            self.params.forward_in_shape_h_w_c0.set_as(tiling_ub[i])
            i = i + 1
            self.params.forward_ou_shape_h_w_c0.set_as(tiling_ub[i])
            i = i + 1
            self.params.hi_val.set_as(tiling_ub[i])
            i = i + 1
            self.params.wi_val.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len_src_orig_y.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_src_orig_y.set_as(tiling_ub[i])
            i = i + 1
            self.params.repeat_times.set_as(tiling_ub[i])
            i = i + 1
            self.params.howo_co_ver.set_as(tiling_ub[i])
            i = i + 1
            self.params.mask_size_16.set_as(tiling_ub[i])
            i = i + 1
            self.params.mask_size_ver.set_as(tiling_ub[i])
            i = i + 1
            self.params.repeat_max_time_grad_sel.set_as(tiling_ub[i])
            i = i + 1
            self.params.remain_repeat_time_grad_sel.set_as(tiling_ub[i])
            i = i + 1
            self.params.remain_ele_grad_sel.set_as(tiling_ub[i])
            i = i + 1
            self.params.repeat_max_loop_vadd.set_as(tiling_ub[i])
            i = i + 1
            self.params.remain_max_loop_vadd.set_as(tiling_ub[i])
            i = i + 1
            self.params.remain_ele_vadd.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_ub_2_gm.set_as(tiling_ub[i])
            i = i + 1
            self.params.dst_stride_ub_2_gm.set_as(tiling_ub[i])
            i = i + 1
            self.params.repeat_max_loop_f_map_fp32.set_as(tiling_ub[i])
            i = i + 1
            self.params.remain_max_loop_f_map_fp32.set_as(tiling_ub[i])
            i = i + 1
            self.params.remain_ele_f_map_fp32.set_as(tiling_ub[i])
            i = i + 1
            self.params.wi_val_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.pad_hw_top_neg.set_as(tiling_ub[i])
            i = i + 1
            self.params.pad_hw_left_neg.set_as(tiling_ub[i])
            i = i + 1
            self.params.forward_in_shape_h_w_2.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len_src_orig_y_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_src_orig_y_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.repeat_times_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.howo_co_ver_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.repeat_max_loop_vadd_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.remain_max_loop_vadd_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.remain_ele_vadd_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_ub_2_gm_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.dst_stride_ub_2_gm_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.core_ho_times.set_as(tiling_ub[i])
            i = i + 1
            self.params.core_wo_times.set_as(tiling_ub[i])
            i = i + 1
            self.params.map_hi.set_as(tiling_ub[i])
            i = i + 1
            self.params.map_wi.set_as(tiling_ub[i])

            i = i + 1
            self.params.config.set_as(tiling_ub[i])
            i = i + 1
            self.params.sh_wi_2.set_as(tiling_ub[i])
            i = i + 1
            self.params.num_instr_loop_h.set_as(tiling_ub[i])
            i = i + 1
            self.params.num_instr_loop_w.set_as(tiling_ub[i])
            i = i + 1
            self.params.remain_mask.set_as(tiling_ub[i])
            i = i + 1
            self.params.remain_repeat.set_as(tiling_ub[i])
            i = i + 1
            self.params.num_instr_loop_w_1.set_as(tiling_ub[i])
            i = i + 1
            self.params.num_instr_loop_h_1.set_as(tiling_ub[i])
            i = i + 1
            self.params.ho_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.hi_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.dst_stride_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.wo_2.set_as(tiling_ub[i])
            i = i + 1
            self.params.boundary_h.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len_ub_2_gm.set_as(tiling_ub[i])
            i = i + 1
            self.params.non_overlap_1.set_as(tiling_ub[i])
            i = i + 1
            self.params.overlap_1.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len_over.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_over.set_as(tiling_ub[i])
            i = i + 1
            self.params.dst_stride_over.set_as(tiling_ub[i])
            i = i + 1
            self.params.dup_repeat_merchant_non_overlap.set_as(tiling_ub[i])
            i = i + 1
            self.params.dup_repeat_remainder_non_overlap.set_as(tiling_ub[i])
            i = i + 1
            self.params.dup_remainder_non_overlap.set_as(tiling_ub[i])
            i = i + 1
            self.params.repeats_non_overlap.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len_ub2gm_2.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_ub2gm_2.set_as(tiling_ub[i])
            i = i + 1
            self.params.dst_stride_ub2gm_2.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len_ub2gm_3.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_ub2gm_3.set_as(tiling_ub[i])
            i = i + 1
            self.params.dst_stride_ub2gm_3.set_as(tiling_ub[i])
            i = i + 1
            self.params.hi_val_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len_val.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_val.set_as(tiling_ub[i])
            i = i + 1
            self.params.dst_stride_val.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len_val_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_val_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.dst_stride_val_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.num_instr_loop_h_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.remain_repeat_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.num_instr_loop_h_1_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len_ub_2_gm_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.non_overlap_1_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_over_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.dst_stride_over_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.dup_repeat_merchant_non_overlap_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.dup_repeat_remainder_non_overlap_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.dup_remainder_non_overlap_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.repeats_non_overlap_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len_ub2gm_2_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_ub2gm_2_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.dst_stride_ub2gm_2_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.burst_len_ub2gm_3_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.src_stride_ub2gm_3_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.dst_stride_ub2gm_3_tail.set_as(tiling_ub[i])
            i = i + 1
            self.params.forward_in_shape_w_c0.set_as(tiling_ub[i])
            i = i + 1
            self.params.dst_stride.set_as(tiling_ub[i])

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self._compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.orig_x_gm, self.orig_y_gm, self.grads_gm],
                              outputs=[self.ou_y_gm],
                              flowtable=[self.gm.tiling_gm])

        return tik_instance

    def _set_tik_instance(self, tik_instance):
        """
        set tik_instance
        """
        self._set_src_dst_tensor(tik_instance)

        return tik_instance

    def _set_src_dst_tensor(self, tik_instance):
        """
        set input and output tensor
        """
        self.orig_x_gm = tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,), name="orig_x_gm", scope=tik.scope_gm)

        self.orig_y_gm = tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,), name="orig_y_gm", scope=tik.scope_gm)

        self.grads_gm = tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,), name="grads_gm", scope=tik.scope_gm)

        self.ou_y_gm = tik_instance.Tensor("float32", (Constant.MAX_INT32,),
                                           name="ou_y_gm",
                                           scope=tik.scope_gm,
                                           is_atomic_add=True)

    def _copy_gm_to_dst_buf_dy(self, tik_instance, dst_buf, src_idx, dst_idx, in_shape, burst_len, src_stride):
        """
        copy gm to l1
        """
        dst_stride = 0
        in_list = [1, burst_len, src_stride, dst_stride, src_idx, dst_idx]

        with tik_instance.if_scope(src_stride > Constant.MAX_STRIDE):
            _ultimate_data_move_dy(tik_instance, self.orig_x_gm, dst_buf, in_list, self.num_bit)
        with tik_instance.else_scope():
            _norm_data_move(tik_instance, self.orig_x_gm, dst_buf, in_list)

    def _gm2dst_tiling_do_ho_dy(self, tik_instance, dst_buf, src_idx, dst_idx, in_shape, hi_batch, burst_len,
                                src_stride, dst_stride):
        """
        copy gm to l1
        """
        in_list = [1, burst_len, src_stride, dst_stride, src_idx, dst_idx]

        with tik_instance.if_scope(tik.any(src_stride > Constant.MAX_STRIDE, dst_stride > Constant.MAX_STRIDE)):
            _ultimate_data_move_dy(tik_instance, self.orig_x_gm, dst_buf, in_list, self.num_bit)
        with tik_instance.else_scope():
            _norm_data_move(tik_instance, self.orig_x_gm, dst_buf, in_list)

    def _gm2l1_tiling_do_ho_wo_dy(self, tik_instance, l1_buf, src_idx, dst_idx, input0, input1, burst_len, src_stride):
        """
        copy gm to l1
        """

        hi_val, wi_val = input0[0], input0[1]
        c0 = self.c0
        in_shape = [hi_val, wi_val, c0]
        n_burst = in_shape[0]

        dst_stride = 0

        in_list = [n_burst, burst_len, src_stride, dst_stride, src_idx, dst_idx]

        with tik_instance.if_scope(src_stride > Constant.MAX_STRIDE):
            _ultimate_data_move(tik_instance, self.orig_x_gm, l1_buf, in_list, self.num_bit)

        with tik_instance.else_scope():
            _norm_data_move(tik_instance, self.orig_x_gm, l1_buf, in_list)

    def _gm2l1_tiling_do_ho_wo_dy1(self, tik_instance, l1_buf, src_idx, dst_idx, input0, input1):
        """
        copy gm to l1
        """
        hi_val, wi_val = input0[0], input0[1]

        c0 = self.c0
        in_shape = [hi_val, wi_val, c0]
        n_burst = in_shape[0]
        burst_len = wi_val * 16 * self.num_bit // Constant.MINI_UNIT
        src_stride = (self.params.w - wi_val) * c0 * self.num_bit // Constant.MINI_UNIT
        dst_stride = 0

        in_list = [n_burst, burst_len, src_stride, dst_stride, src_idx, dst_idx]

        with tik_instance.if_scope(src_stride > Constant.MAX_STRIDE):
            _ultimate_data_move(tik_instance, self.orig_x_gm, l1_buf, in_list, self.num_bit)

        with tik_instance.else_scope():
            _norm_data_move(tik_instance, self.orig_x_gm, l1_buf, in_list)

    def _copy_ub_to_gm_dy(self, tik_instance, src_buf, src_idx, dst_buf, dst_idx, in_shape, burst_len, dst_stride):
        """
        copy ub to gm
        """
        src_stride = 0
        in_list = [1, burst_len, src_stride, dst_stride, src_idx, dst_idx]

        with tik_instance.if_scope(dst_stride > Constant.MAX_STRIDE):
            _ultimate_data_move_dy(tik_instance, src_buf, dst_buf, in_list, self.num_bit_fp32)
        with tik_instance.else_scope():
            _norm_data_move(tik_instance, src_buf, dst_buf, in_list)

    def _ub2gm_split_do_ho_2_dy(self, tik_instance, src, src_idx, dst, dst_idx, burst_len, src_stride, dst_stride):
        """
        copy ub to gm
        """
        in_list = [1, burst_len, src_stride, dst_stride, src_idx, dst_idx]

        with tik_instance.if_scope(tik.any(src_stride > Constant.MAX_STRIDE, dst_stride > Constant.MAX_STRIDE)):
            _ultimate_data_move_dy(tik_instance, src, dst, in_list, self.num_bit_fp32)

        with tik_instance.else_scope():
            _norm_data_move(tik_instance, src, dst, in_list)

    def _ub2gm_split_do_ho_dy(self, tik_instance, src_buf, src_idx, dst_buf, dst_idx, burst_len, src_stride,
                              dst_stride):
        """
        copy ub to gm
        """
        in_list = [1, burst_len, src_stride, dst_stride, src_idx, dst_idx]

        with tik_instance.if_scope(tik.any(src_stride > Constant.MAX_STRIDE, dst_stride > Constant.MAX_STRIDE)):
            _ultimate_data_move(tik_instance, src_buf, dst_buf, in_list, self.num_bit_fp32)

        with tik_instance.else_scope():
            _norm_data_move(tik_instance, src_buf, dst_buf, in_list)

    def _ub2gm_split_do_ho_wo_dy(self, tik_instance, src, src_idx, dst, dst_idx, in_shape, hi_batch, wi_batch,
                                 burst_len, src_stride, dst_stride):
        """
        copy ub to gm
        """
        num_bit = self.num_bit_fp32

        n_burst = in_shape[1]

        dst_idx_new = dst_idx
        src_idx_new = src_idx

        in_list = [n_burst, burst_len, src_stride, dst_stride, src_idx_new, dst_idx_new]

        with tik_instance.if_scope(tik.any(src_stride > Constant.MAX_STRIDE, dst_stride > Constant.MAX_STRIDE)):
            _ultimate_data_move(tik_instance, src, dst, in_list, num_bit)
        with tik_instance.else_scope():
            _norm_data_move(tik_instance, src, dst, in_list)

    def _copy_gm_to_ub_dy(self, tik_instance, dst_buf, src_buf, src_idx, in_shape, burst_len, src_stride):
        """
        copy gm to ub
        """
        n_burst = 1
        dst_stride = 0
        with tik_instance.if_scope(src_stride > Constant.MAX_STRIDE):
            in_list = [n_burst, burst_len, src_stride, dst_stride, src_idx, 0]
            _ultimate_data_move_dy(tik_instance, src_buf, dst_buf, in_list, self.num_bit)

        with tik_instance.else_scope():
            tik_instance.data_move(dst_buf[0], src_buf[src_idx], 0, n_burst, burst_len, src_stride, dst_stride)

    def _set_buf_tensor(self, tik_instance, param, is_vadds=False):
        """
        apply ub and l1
        """
        l1_in_buf = None
        if self.check_load3d_support:
            l1_in_buf = tik_instance.Tensor(self.dtype, [param.l1_in_size, ], name="l1_in_buf", scope=tik.scope_cbuf)
        forward_ou_buf = tik_instance.Tensor(self.dtype, [param.forward_ou_size, ],
                                             name="forward_ou_buf",
                                             scope=tik.scope_ubuf)
        grad_buf = tik_instance.Tensor(self.dtype, [param.grad_size, ], name="grad_buf", scope=tik.scope_ubuf)
        col_in_buf = None
        if self.check_load3d_support or is_vadds:
            col_in_buf = tik_instance.Tensor(self.dtype, [param.col_in_size, ], name="col_in_buf", scope=tik.scope_ubuf)
        ori_in_buf = None
        mask_temp_buf = None
        mask_temp_not_buf = None
        if not self.check_load3d_support:
            ori_in_buf = tik_instance.Tensor(self.dtype, [param.ori_in_size, ], name="ori_in_buf", scope=tik.scope_ubuf)
            mask_temp_buf = tik_instance.Tensor("uint16", [param.mask_size, ],
                                                name='mask_temp_buf', scope=tik.scope_ubuf)
            mask_temp_not_buf = tik_instance.Tensor("uint16", [param.mask_size, ],
                                                    name='mask_temp_not_buf', scope=tik.scope_ubuf)

        mask_buf = tik_instance.Tensor("uint16", [param.mask_size, ], name='mask_buf', scope=tik.scope_ubuf)
        mask_or_buf = tik_instance.Tensor("uint16", [param.mask_size, ], name='mask_or_buf', scope=tik.scope_ubuf)
        mask_not_buf = tik_instance.Tensor("uint16", [param.mask_size, ], name='mask_not_buf', scope=tik.scope_ubuf)
        zero_buf = tik_instance.Tensor(self.dtype, [param.zero_size, ], name='zero_buf', scope=tik.scope_ubuf)

        grad_sel_fp16_buf = tik_instance.Tensor(self.dtype, [param.grad_sel_fp16_size, ],
                                                name='grad_sel_fp16_buf',
                                                scope=tik.scope_ubuf)
        grad_sel_fp32_buf = tik_instance.Tensor("float32", [param.grad_sel_fp32_size, ],
                                                name='grad_sel_fp32_buf',
                                                scope=tik.scope_ubuf)
        f_map_fp32_buf = tik_instance.Tensor("float32", [param.f_map_fp32_size, ],
                                             name='f_map_fp32_buf',
                                             scope=tik.scope_ubuf)

        buf_list = [
            l1_in_buf, forward_ou_buf, grad_buf, col_in_buf, mask_buf, mask_or_buf, mask_not_buf, zero_buf,
            grad_sel_fp16_buf, grad_sel_fp32_buf, f_map_fp32_buf, ori_in_buf, mask_temp_buf, mask_temp_not_buf
        ]

        return buf_list

    def _calc_mask_dy(self, tik_instance, buf_list, param, idx_list, const_list, howo_co_ver, mask_size_16,
                      mask_size_ver, ori_in_buf=None, wi=-1, is_vadds=False):
        """
        calculate mask
        """
        forward_ou_buf = buf_list[1]
        col_in_buf = buf_list[3]
        mask_buf = buf_list[4]
        mask_or_buf = buf_list[5]
        mask_not_buf = buf_list[6]
        mask_temp_buf = buf_list[12]
        mask_temp_not_buf = buf_list[13]
        idx_h = idx_list[0]
        idx_w = idx_list[1]
        ho, wo, c0 = const_list

        with tik_instance.if_scope(tik.all(idx_h == 0, idx_w == 0)):
            if self.check_load3d_support or is_vadds:
                tik_instance.vcmpv_eq(mask_buf[0], forward_ou_buf[0], col_in_buf[0], howo_co_ver, 1, 1, 8, 8)
            else:
                if Constant.SRC1STRIDEM1 * self.sw > Constant.VEC_MAX_STRIDE:
                    with tik_instance.if_scope(howo_co_ver > 1):
                        error_manager_vector.raise_err_specific_reson(
                            "maxpoolgrad", "stride_w exceed limit")
                    with tik_instance.else_scope():
                        tik_instance.vcmpv_eq(mask_buf[0], forward_ou_buf[0],
                                              ori_in_buf[idx_h * wi * c0 + idx_w * c0],
                                              1, 1, self.sw, 0, 0)
                else:
                    tik_instance.vcmpv_eq(mask_buf[0], forward_ou_buf[0], ori_in_buf[idx_h * wi * c0 + idx_w * c0],
                                          howo_co_ver, 1, self.sw,
                                          Constant.SRC1STRIDEM1, Constant.SRC1STRIDEM1 * self.sw)

            tik_instance.data_move(mask_or_buf[0], mask_buf[0], 0, 1, mask_size_16, 0, 0)
            tik_instance.vnot(self.mask_fp16, mask_not_buf, mask_or_buf, mask_size_ver, 1, 1, 8, 8)

        with tik_instance.else_scope():
            if self.check_load3d_support or is_vadds:
                tik_instance.vcmpv_eq(
                    mask_buf[0], forward_ou_buf[0], col_in_buf[0], howo_co_ver, 1, 1, 8, 8)
            else:
                if Constant.SRC1STRIDEM1 * self.sw > Constant.VEC_MAX_STRIDE:
                    with tik_instance.if_scope(howo_co_ver > 1):
                        error_manager_vector.raise_err_specific_reson(
                            "maxpoolgrad", "stride_w exceed limit")
                    with tik_instance.else_scope():
                        tik_instance.vcmpv_eq(mask_buf[0], forward_ou_buf[0],
                                              ori_in_buf[idx_h * wi * c0 + idx_w * c0],
                                              1, 1, self.sw, 0, 0)
                else:
                    tik_instance.vcmpv_eq(mask_buf[0], forward_ou_buf[0], ori_in_buf[idx_h * wi * c0 + idx_w * c0],
                                          howo_co_ver, 1, self.sw,
                                          Constant.SRC1STRIDEM1, Constant.SRC1STRIDEM1 * self.sw)

            tik_instance.vand(self.mask_fp16, mask_buf, mask_not_buf, mask_buf, mask_size_ver, 1, 1, 1, 8, 8, 8)

            tik_instance.vor(self.mask_fp16, mask_or_buf, mask_or_buf, mask_buf, mask_size_ver, 1, 1, 1, 8, 8, 8)

            tik_instance.vnot(self.mask_fp16, mask_not_buf, mask_or_buf, mask_size_ver, 1, 1, 8, 8)

    def _sel_dy(self, tik_instance, buf_list, idx_list, const_list, howo_co_ver):
        """
        sel
        """
        mask_buf = buf_list[4]
        zero_buf = buf_list[7]
        grad_buf = buf_list[2]
        grad_sel_fp16_buf = buf_list[8]

        with tik_instance.for_range(0, howo_co_ver) as serial:
            grad_sel_offset = serial * 128
            grad_offset = serial * 128
            mask_offset = serial * 8
            cmp_mask = tik_instance.mov_tensor_to_cmpmask(mask_buf[mask_offset])
            tik_instance.vsel(self.mask_fp16, 0, grad_sel_fp16_buf[grad_sel_offset], cmp_mask, grad_buf[grad_offset],
                              zero_buf, 1, 1, 1, 1, 8, 8, 0)

    def _img2col(self, tik_instance, ori_in_buf, col_in_buf, in_shape, const_list, window_list, param,
                 pad_hw_left=0, pad_hw_right=0):
        hi, wi, c0 = in_shape[0], in_shape[1], in_shape[2]
        ho, wo = const_list[0], const_list[1]
        idx_h, idx_w = window_list[0], window_list[1]

        dup_psm_col_in_buf = Constant.VEC_MAX_STRIDE * 128
        dup_repeat_merchant_col_in_buf = param.col_in_size // dup_psm_col_in_buf
        dup_repeat_remainder_col_in_buf = param.col_in_size % dup_psm_col_in_buf
        repeats_col_in_buf = dup_repeat_remainder_col_in_buf // 128
        dup_remainder_col_in_buf = dup_repeat_remainder_col_in_buf % 128
        _set_vector_dup_dy(tik_instance, col_in_buf, 0, Constant.MIN_VALUE_FP16, "float16",
                           dup_repeat_merchant_col_in_buf, dup_repeat_remainder_col_in_buf,
                           dup_remainder_col_in_buf, repeats_col_in_buf, param.col_in_size)
        output_nums = wo * c0
        repeat = output_nums // Constant.MASK128_VALUE
        remain = output_nums % Constant.MASK128_VALUE
        with tik_instance.for_range(0, ho) as ho_idx:
            with tik_instance.if_scope(repeat > 0):
                if Constant.SRC1STRIDEM1 * self.sw > Constant.VEC_MAX_STRIDE:
                    with tik_instance.for_range(0, repeat) as idx:
                        tik_instance.vadds(Constant.MASK128_VALUE,
                                           col_in_buf[output_nums * ho_idx + Constant.MASK128_VALUE * idx],
                                           ori_in_buf[(
                                               wi + pad_hw_left + pad_hw_right) * c0 * (ho_idx * self.sh + idx_h) +
                                               idx_w * c0 + Constant.MASK128_VALUE * idx * self.sw],
                                           tik_instance.Scalar(
                                               dtype="float16", init_value=0),
                                           1, 1, self.sw, 0, 0)
                else:
                    tik_instance.vadds(Constant.MASK128_VALUE, col_in_buf[output_nums * ho_idx],
                                       ori_in_buf[(
                                           wi + pad_hw_left + pad_hw_right) * c0 * (ho_idx * self.sh + idx_h) +
                                           idx_w * c0],
                                       tik_instance.Scalar(
                                           dtype="float16", init_value=0),
                                       repeat, 1, self.sw, Constant.SRC1STRIDEM1, Constant.SRC1STRIDEM1 * self.sw)
            with tik_instance.if_scope(remain > 0):
                tik_instance.vadds(remain, col_in_buf[output_nums * ho_idx + repeat * Constant.MASK128_VALUE],
                                   ori_in_buf[(
                                       wi + pad_hw_left + pad_hw_right) * c0 * (ho_idx * self.sh + idx_h) + idx_w * c0 +
                                   Constant.MASK128_VALUE * repeat * self.sw],
                                   tik_instance.Scalar(
                                       dtype="float16", init_value=0),
                                   1, 1, self.sw, 0, 0)

    def _not_tiling_main_dy(self, tik_instance, core_loop, sum_core, model, param):
        """
        not_tiling_main
        """
        ho = model[0]
        wo = model[1]
        hi = self.params.hi_batch
        wi = self.params.wi_batch
        c0 = self.c0
        c1 = self.params.c1

        buf_list = self._set_buf_tensor(tik_instance, param, is_vadds=True)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        mask_buf = buf_list[4]
        mask_or_buf = buf_list[5]
        mask_not_buf = buf_list[6]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        ori_in_buf = buf_list[11]
        mask_temp_buf = buf_list[12]
        mask_temp_not_buf = buf_list[13]
        l1_in_size = param.l1_in_size
        col_in_size = param.col_in_size
        ori_in_size = param.ori_in_size
        forward_ou_size = param.forward_ou_size
        mask_size = param.mask_size
        grad_size = param.grad_size
        zero_size = 128
        grad_sel_fp16_size = param.grad_sel_fp16_size
        grad_sel_fp32_size = param.grad_sel_fp32_size
        f_map_fp32_size = param.f_map_fp32_size
        _set_vector_dup_zero(tik_instance, zero_buf, 0, 0, self.dtype)

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                               self.params.dup_repeat_merchant_f_map_fp32,
                               self.params.dup_repeat_remainder_f_map_fp32, self.params.dup_remainder_f_map_fp32,
                               self.params.repeats_f_map_fp32, f_map_fp32_size)
            merchant = (sum_core + num_core_loop) // c1
            remainder = (sum_core + num_core_loop) % c1

            # ----COPY_GM_2_L1_BUF----
            src_orig_x_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                remainder * hi * wi * c0
            gm2dst_shape = [hi, wi, c0]
            if self.check_load3d_support:
                self._copy_gm_to_dst_buf_dy(tik_instance, l1_in_buf, src_orig_x_gm, 0, gm2dst_shape,
                                            self.params.burst_len, self.params.src_stride)
            else:
                dup_psm_ori_in_buf = Constant.VEC_MAX_STRIDE * 128
                dup_repeat_merchant_ori_in_buf = param.ori_in_size // dup_psm_ori_in_buf
                dup_repeat_remainder_ori_in_buf = param.ori_in_size % dup_psm_ori_in_buf
                repeats_ori_in_buf = dup_repeat_remainder_ori_in_buf // 128
                dup_remainder_ori_in_buf = dup_repeat_remainder_ori_in_buf % 128
                _set_vector_dup_dy(tik_instance, ori_in_buf, 0, Constant.MIN_VALUE_FP16, "float16",
                                   dup_repeat_merchant_ori_in_buf, dup_repeat_remainder_ori_in_buf,
                                   dup_remainder_ori_in_buf, repeats_ori_in_buf, param.ori_in_size)
                self._copy_gm_to_dst_buf_dy(tik_instance, ori_in_buf, src_orig_x_gm, 0, gm2dst_shape,
                                            self.params.burst_len, self.params.src_stride)

            # ----COPY_ORI_OUTPUT_2_FORWARD_OU_BUF----
            # ----COPY_GRAD_2_GRAD_BUF----
            src_orig_y_gm = merchant * self.params.forward_ou_shape_h_w_c0 + \
                remainder * ho * wo * c0
            gm2ub_data_shape = [1, ho, wo, c0]
            self._copy_gm_to_ub_dy(tik_instance, forward_ou_buf, self.orig_y_gm, src_orig_y_gm, gm2ub_data_shape,
                                   self.params.burst_len_src_orig_y, self.params.src_stride_src_orig_y)

            src_grad_gm = src_orig_y_gm
            self._copy_gm_to_ub_dy(tik_instance, grad_buf, self.grads_gm, src_grad_gm, gm2ub_data_shape,
                                   self.params.burst_len_src_orig_y, self.params.src_stride_src_orig_y)

            # ---load3d l1 to col_in_buffer---
            with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                    if self.check_load3d_support:
                        src_l1 = 0
                        tik_instance.load3dv1(col_in_buf[0], l1_in_buf[src_l1], [0, 0, 0, 0], hi, wi, 0, idx_w, idx_h,
                                              0, 0, self.sw, self.sh, self.kw, self.kh, 1, 1, 1, 1,
                                              self.params.repeat_times, 0, Constant.MIN_VALUE_FP16)
                    else:
                        self._img2col(tik_instance, ori_in_buf, col_in_buf, [hi, wi, c0], [ho, wo],
                                      [idx_h, idx_w], param)

                    # ---calculate mask---
                    with tik_instance.if_scope(tik.all(idx_h == 0, idx_w == 0)):
                        tik_instance.vcmpv_eq(mask_buf[0], forward_ou_buf[0], col_in_buf[0], self.params.howo_co_ver,
                                              1, 1, 8, 8)
                        if not self.check_load3d_support:
                            tik_instance.vcmpv_eq(mask_temp_buf[0], col_in_buf[0], col_in_buf[0],
                                                  self.params.howo_co_ver,
                                                  1, 1, 8, 8)
                            tik_instance.vnot(self.mask_fp16, mask_temp_not_buf, mask_temp_buf,
                                              self.params.mask_size_ver, 1, 1, 8, 8)
                            tik_instance.vor(self.mask_fp16, mask_buf, mask_temp_not_buf, mask_buf,
                                             self.params.mask_size_ver, 1, 1, 1, 8, 8, 8)
                        tik_instance.data_move(mask_or_buf[0], mask_buf[0], 0, 1, self.params.mask_size_16, 0, 0)

                        tik_instance.vnot(self.mask_fp16, mask_not_buf, mask_or_buf, self.params.mask_size_ver,
                                          1, 1, 8, 8)

                    with tik_instance.else_scope():
                        tik_instance.vcmpv_eq(mask_buf[0], forward_ou_buf[0], col_in_buf[0], self.params.howo_co_ver,
                                              1, 1, 8, 8)
                        if not self.check_load3d_support:
                            tik_instance.vcmpv_eq(mask_temp_buf[0], col_in_buf[0], col_in_buf[0],
                                                  self.params.howo_co_ver,
                                                  1, 1, 8, 8)
                            tik_instance.vnot(self.mask_fp16, mask_temp_not_buf, mask_temp_buf,
                                              self.params.mask_size_ver, 1, 1, 8, 8)
                            tik_instance.vor(self.mask_fp16, mask_buf, mask_temp_not_buf, mask_buf,
                                             self.params.mask_size_ver, 1, 1, 1, 8, 8, 8)

                        tik_instance.vand(self.mask_fp16, mask_buf, mask_not_buf, mask_buf, self.params.mask_size_ver,
                                          1, 1, 1, 8, 8, 8)

                        tik_instance.vor(self.mask_fp16, mask_or_buf, mask_or_buf, mask_buf, self.params.mask_size_ver,
                                         1, 1, 1, 8, 8, 8)

                        tik_instance.vnot(self.mask_fp16, mask_not_buf, mask_or_buf, self.params.mask_size_ver,
                                          1, 1, 8, 8)

                    # ---vsel(grad,zero,mask)---
                    with tik_instance.for_range(0, self.params.howo_co_ver) as serial:
                        grad_sel_offset = serial * 128
                        grad_offset = serial * 128
                        mask_offset = serial * 8
                        cmp_mask = tik_instance.mov_tensor_to_cmpmask(mask_buf[mask_offset])
                        tik_instance.vsel(self.mask_fp16, 0, grad_sel_fp16_buf[grad_sel_offset], cmp_mask,
                                          grad_buf[grad_offset], zero_buf, 1, 1, 1, 1, 8, 8, 0)

                    # ---vconv grad_sel_fp16 to fp32---
                    _vconv_dy(tik_instance, grad_sel_fp16_buf, 0, grad_sel_fp32_buf, 0, "float16",
                              self.params.repeat_max_time_grad_sel, self.params.remain_repeat_time_grad_sel,
                              self.params.remain_ele_grad_sel, grad_size)

                    # ---rewrite grad_sel_fp32 to f_map_fp32
                    config = [self.sw * 2, self.sw * 2, 2, self.params.sh_wi_2, self.params.sh_wi_2, self.params.wo_2]

                    with tik_instance.if_scope(self.params.config == 1):
                        map_index = (idx_h * self.params.wi_batch * c0 + idx_w * c0)
                        mask_index = 0
                        shape_map_hw = [hi, wi, c0]
                        shape_grad = [ho, wo, c0]

                        _rewrite_fmap_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                         grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:],
                                         "float32", ho, shape_map_hw, shape_grad, config,
                                         self.params.num_instr_loop_h,
                                         self.params.num_instr_loop_w, self.params.remain_mask,
                                         self.params.remain_repeat, self.params.num_instr_loop_w_1,
                                         self.params.num_instr_loop_h_1)

                        _rewrite_fmap_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                         grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                         "float32", ho, shape_map_hw, shape_grad, config,
                                         self.params.num_instr_loop_h, self.params.num_instr_loop_w,
                                         self.params.remain_mask, self.params.remain_repeat,
                                         self.params.num_instr_loop_w_1, self.params.num_instr_loop_h_1)

                    with tik_instance.else_scope():
                        # map_index has three part: which hwc0 in
                        # which window, begin_index of kernel,
                        # begin_index of child kernel
                        with tik_instance.for_range(0, ho) as ho_idx:
                            map_index = (ho_idx * self.sh * wi * c0) + (idx_h * wi * c0 + idx_w * c0)
                            mask_index = wo * ho_idx * c0

                            _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                          grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:], "float32",
                                          (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                          self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                                          self.params.remain_ele_vadd, f_map_fp32_size, grad_sel_fp32_size,
                                          f_map_fp32_size)
                            _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                          grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                          "float32",
                                          (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                          self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                                          self.params.remain_ele_vadd, f_map_fp32_size, grad_sel_fp32_size,
                                          f_map_fp32_size)

            # ---mov_out---
            dst_ou_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                remainder * hi * wi * c0
            ub2gm_shape = [1, hi, wi, c0]
            self._copy_ub_to_gm_dy(tik_instance, f_map_fp32_buf, 0, self.ou_y_gm, dst_ou_gm, ub2gm_shape,
                                   self.params.burst_len_tail, self.params.dst_stride_ub_2_gm)

    def _tiling_ho_main_dy(self, tik_instance, core_loop, sum_core, model, param):
        """
        Just only split ho
        """
        ho_batch = model[0]
        wo = model[1]
        c0 = self.c0
        c1 = self.params.c1
        l1_in_size = param.l1_in_size
        col_in_size = param.col_in_size
        ori_in_size = param.ori_in_size
        forward_ou_size = param.forward_ou_size
        mask_size = param.mask_size
        grad_size = param.grad_size
        zero_size = 128
        grad_sel_fp16_size = param.grad_sel_fp16_size
        grad_sel_fp32_size = param.grad_sel_fp32_size
        f_map_fp32_size = param.f_map_fp32_size

        buf_list = self._set_buf_tensor(tik_instance, param, is_vadds=True)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        ori_in_buf = buf_list[11]
        _set_vector_dup_zero(tik_instance, zero_buf, 0, 0, self.dtype)
        _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                           self.params.dup_repeat_merchant_f_map_fp32, self.params.dup_repeat_remainder_f_map_fp32,
                           self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32, f_map_fp32_size)

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            merchant = (sum_core + num_core_loop) // c1
            remainder = (sum_core + num_core_loop) % c1

            def _main(loop_ho_idx, hi, ho, burst_len, src_stride, dst_stride, hi_val, burst_len_val, src_stride_val,
                      dst_stride_val, burst_len_src_orig_y, src_stride_src_orig_y, repeat_times, howo_co_ver,
                      mask_size_16, mask_size_ver, repeat_max_time, remain_repeat_time, remain_ele, num_instr_loop_h,
                      num_instr_loop_w, remain_mask, remain_repeat, num_instr_loop_w_1, num_instr_loop_h_1,
                      config_params, repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd, burst_len_ub_2_gm,
                      src_stride_ub_2_gm, dst_stride_ub_2_gm, non_overlap_1, overlap_1,
                      burst_len_over, src_stride_over, dst_stride_over, dup_repeat_merchant_non_overlap,
                      dup_repeat_remainder_non_overlap, dup_remainder_non_overlap, repeats_non_overlap,
                      burst_len_ub2gm_2, src_stride_ub2gm_2, dst_stride_ub2gm_2, burst_len_ub2gm_3,
                      src_stride_ub2gm_3, dst_stride_ub2gm_3):
                # Init_Begin_Idx
                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * \
                        (self.params.hi_batch - self.params.overlap_h)
                else:
                    hi_coordinate = loop_ho_idx * self.params.hi_batch

                do_coordinate = 0
                ho_coordinate = loop_ho_idx * ho_batch

                src_orig_x_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                    remainder * self.params.h * self.params.wi_batch * c0 + \
                    hi_coordinate * self.params.wi_batch * c0
                src_orig_y_gm = merchant * self.params.forward_ou_shape_h_w_c0 + \
                    remainder * self.params.ho * wo * c0 + \
                    ho_coordinate * wo * c0
                src_grad_gm = src_orig_y_gm
                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds
                # ================================
                if self.check_load3d_support:
                    with tik_instance.if_scope(hi_coordinate + hi <= self.params.h):
                        self._gm2dst_tiling_do_ho_dy(tik_instance, l1_in_buf, src_orig_x_gm, 0,
                                                     [1, hi, self.params.wi_batch, c0], self.params.hi_batch, burst_len,
                                                     src_stride, dst_stride)
                    with tik_instance.else_scope():
                        with tik_instance.if_scope(self.params.overlap_h < 0):
                            self._gm2dst_tiling_do_ho_dy(tik_instance, l1_in_buf, src_orig_x_gm, 0,
                                                         [1, hi_val, self.params.wi_batch, c0], self.params.hi_batch,
                                                         burst_len_val, src_stride_val, dst_stride_val)
                else:
                    dup_psm_ori_in_buf = Constant.VEC_MAX_STRIDE * 128
                    dup_repeat_merchant_ori_in_buf = param.ori_in_size // dup_psm_ori_in_buf
                    dup_repeat_remainder_ori_in_buf = param.ori_in_size % dup_psm_ori_in_buf
                    repeats_ori_in_buf = dup_repeat_remainder_ori_in_buf // 128
                    dup_remainder_ori_in_buf = dup_repeat_remainder_ori_in_buf % 128
                    _set_vector_dup_dy(tik_instance, ori_in_buf, 0, Constant.MIN_VALUE_FP16, "float16",
                                       dup_repeat_merchant_ori_in_buf, dup_repeat_remainder_ori_in_buf,
                                       dup_remainder_ori_in_buf, repeats_ori_in_buf, param.ori_in_size)
                    with tik_instance.if_scope(hi_coordinate + hi <= self.params.h):
                        self._gm2dst_tiling_do_ho_dy(tik_instance, ori_in_buf, src_orig_x_gm, 0,
                                                     [1, hi, self.params.wi_batch, c0], self.params.hi_batch, burst_len,
                                                     src_stride, dst_stride)
                    with tik_instance.else_scope():
                        with tik_instance.if_scope(self.params.overlap_h < 0):
                            self._gm2dst_tiling_do_ho_dy(tik_instance, ori_in_buf, src_orig_x_gm, 0,
                                                         [1, hi_val, self.params.wi_batch, c0], self.params.hi_batch,
                                                         burst_len_val, src_stride_val, dst_stride_val)
                    # ================================
                    # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                    # COPY_GRAD_2_GRAD_BUF
                    # ================================
                self._copy_gm_to_ub_dy(tik_instance, forward_ou_buf, self.orig_y_gm, src_orig_y_gm, [1, ho, wo, c0],
                                       burst_len_src_orig_y, src_stride_src_orig_y)

                self._copy_gm_to_ub_dy(tik_instance, grad_buf, self.grads_gm, src_grad_gm, [1, ho, wo, c0],
                                       burst_len_src_orig_y, src_stride_src_orig_y)
                # load3d l1 to col_in_buffer
                with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                    with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                        if self.check_load3d_support:
                            src_l1 = 0
                            tik_instance.load3dv1(col_in_buf[0], l1_in_buf[src_l1], [0, 0, 0, 0], hi,
                                                  self.params.wi_batch, 0, idx_w, idx_h, 0, 0, self.sw, self.sh,
                                                  self.kw, self.kh, 1, 1, 1, 1,
                                                  repeat_times, 0, Constant.MIN_VALUE_FP16)
                        else:
                            self._img2col(tik_instance, ori_in_buf, col_in_buf, [hi_val, self.params.wi_batch, c0],
                                          [ho, wo], [idx_h, idx_w], param)

                        # ---calculate mask---
                        idx_list = [idx_h, idx_w]
                        const_list = [ho, wo, c0]
                        self._calc_mask_dy(tik_instance, buf_list, param, idx_list, const_list, howo_co_ver,
                                           mask_size_16, mask_size_ver, is_vadds=True)

                        # ---sel(grad,zero,mask)---
                        self._sel_dy(tik_instance, buf_list, idx_list, const_list, howo_co_ver)

                        # ---vconv grad_sel_fp16 to fp32---
                        _vconv_dy(tik_instance, grad_sel_fp16_buf, 0, grad_sel_fp32_buf, 0, "float16",
                                  repeat_max_time, remain_repeat_time, remain_ele, grad_size)

                        # ---rewrite grad_sel_fp32 to f_map_fp32
                        config_1 = (self.sw * 2, self.sw * 2)
                        config = (self.sw * 2, self.sw * 2, 2, self.params.sh_wi_2, self.params.sh_wi_2,
                                  self.params.wo_2)
                        with tik_instance.if_scope(config_params == 1):
                            map_index = (idx_h * self.params.wi_batch * c0 + idx_w * c0)
                            mask_index = 0
                            shape_map_hw = [self.params.hi_batch, self.params.wi_batch, c0]
                            shape_grad = [ho, wo, c0]

                            _rewrite_fmap_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                             grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:],
                                             "float32", ho, shape_map_hw, shape_grad, config, num_instr_loop_h,
                                             num_instr_loop_w, remain_mask, remain_repeat, num_instr_loop_w_1,
                                             num_instr_loop_h_1)

                            _rewrite_fmap_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                             grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                             "float32", ho, shape_map_hw, shape_grad, config, num_instr_loop_h,
                                             num_instr_loop_w, remain_mask, remain_repeat, num_instr_loop_w_1,
                                             num_instr_loop_h_1)

                        with tik_instance.else_scope():
                            # map_index has three part: which hwc0 in
                            # which window, begin_index of kernel,
                            # begin_index of child kernel
                            with tik_instance.for_range(0, ho) as ho_idx:
                                map_index = (ho_idx * self.sh * self.params.wi_batch * c0) + \
                                            (idx_h * self.params.wi_batch * c0 + idx_w * c0)
                                mask_index = wo * ho_idx * c0

                                _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                              grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:],
                                              "float32",
                                              (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                              repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd,
                                              f_map_fp32_size, grad_sel_fp32_size, f_map_fp32_size)
                                _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                              grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                              "float32",
                                              (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                              repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd,
                                              f_map_fp32_size, grad_sel_fp32_size, f_map_fp32_size)

                # mov_out
                dst_ou_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                    remainder * self.params.h * self.params.wi_batch * c0 + \
                    hi_coordinate * self.params.wi_batch * c0

                def mov_atomic(dst, dst_idx, src_idx):
                    """
                    move atomic
                    """
                    if self.kh > self.sh:
                        with tik_instance.if_scope(hi_coordinate + hi < self.params.boundary_h):
                            # ==============================
                            # move accumulated data to gm
                            # ==============================
                            self._ub2gm_split_do_ho_dy(tik_instance, f_map_fp32_buf, src_idx, dst, dst_idx,
                                                       burst_len_ub_2_gm, src_stride_ub_2_gm, dst_stride_ub_2_gm)

                            n_burst = 1
                            tik_instance.data_move(f_map_fp32_buf[src_idx], f_map_fp32_buf[src_idx + non_overlap_1], 0,
                                                   n_burst, burst_len_over, src_stride_over, dst_stride_over)

                            dst_vec_idx = src_idx + overlap_1
                            _set_vector_dup_dy(tik_instance, f_map_fp32_buf, dst_vec_idx, 0, "float32",
                                               dup_repeat_merchant_non_overlap, dup_repeat_remainder_non_overlap,
                                               dup_remainder_non_overlap, repeats_non_overlap, f_map_fp32_size)

                        with tik_instance.else_scope():
                            self._ub2gm_split_do_ho_2_dy(tik_instance, f_map_fp32_buf, src_idx, dst, dst_idx,
                                                         burst_len_ub2gm_2, src_stride_ub2gm_2, dst_stride_ub2gm_2)

                            _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                                               self.params.dup_repeat_merchant_f_map_fp32,
                                               self.params.dup_repeat_remainder_f_map_fp32,
                                               self.params.dup_remainder_f_map_fp32,
                                               self.params.repeats_f_map_fp32, f_map_fp32_size)

                    elif self.kh == self.sh:
                        self._ub2gm_split_do_ho_2_dy(tik_instance, f_map_fp32_buf, src_idx, dst, dst_idx,
                                                     burst_len_ub2gm_2, src_stride_ub2gm_2, dst_stride_ub2gm_2)

                        _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                                           self.params.dup_repeat_merchant_f_map_fp32,
                                           self.params.dup_repeat_remainder_f_map_fp32,
                                           self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32,
                                           f_map_fp32_size)

                    else:
                        with tik_instance.if_scope(self.params.hi_invalid >= 0):
                            self._ub2gm_split_do_ho_2_dy(tik_instance, f_map_fp32_buf, src_idx, dst, dst_idx,
                                                         burst_len_ub2gm_2, src_stride_ub2gm_2, dst_stride_ub2gm_2)

                            _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                                               self.params.dup_repeat_merchant_f_map_fp32,
                                               self.params.dup_repeat_remainder_f_map_fp32,
                                               self.params.dup_remainder_f_map_fp32,
                                               self.params.repeats_f_map_fp32, f_map_fp32_size)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(hi_coordinate + hi < self.params.boundary_h):
                                self._ub2gm_split_do_ho_2_dy(tik_instance, f_map_fp32_buf, src_idx, dst, dst_idx,
                                                             burst_len_ub2gm_2, src_stride_ub2gm_2, dst_stride_ub2gm_2)

                                _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                                                   self.params.dup_repeat_merchant_f_map_fp32,
                                                   self.params.dup_repeat_remainder_f_map_fp32,
                                                   self.params.dup_remainder_f_map_fp32,
                                                   self.params.repeats_f_map_fp32, f_map_fp32_size)
                            with tik_instance.else_scope():
                                self._ub2gm_split_do_ho_2_dy(tik_instance, f_map_fp32_buf, src_idx, dst, dst_idx,
                                                             burst_len_ub2gm_3, src_stride_ub2gm_3, dst_stride_ub2gm_3)
                                _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                                                   self.params.dup_repeat_merchant_f_map_fp32,
                                                   self.params.dup_repeat_remainder_f_map_fp32,
                                                   self.params.dup_remainder_f_map_fp32,
                                                   self.params.repeats_f_map_fp32, f_map_fp32_size)

                tik_instance.set_atomic_add(1)
                mov_atomic(self.ou_y_gm, dst_ou_gm, 0)
                tik_instance.set_atomic_add(0)

            with tik_instance.if_scope(self.params.ho_tail != 0):
                with tik_instance.for_range(0, self.params.loop_ho) as ho_idx:
                    _main(ho_idx, self.params.hi_batch, ho_batch, self.params.burst_len, self.params.src_stride, 0,
                          self.params.hi_val, self.params.burst_len_val, self.params.src_stride_val,
                          self.params.dst_stride_val, self.params.burst_len_src_orig_y,
                          self.params.src_stride_src_orig_y, self.params.repeat_times, self.params.howo_co_ver,
                          self.params.mask_size_16, self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                          self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                          self.params.num_instr_loop_h, self.params.num_instr_loop_w, self.params.remain_mask,
                          self.params.remain_repeat, self.params.num_instr_loop_w_1, self.params.num_instr_loop_h_1,
                          self.params.config, self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                          self.params.remain_ele_vadd, self.params.burst_len_ub_2_gm, self.params.src_stride_ub_2_gm,
                          self.params.dst_stride_ub_2_gm, self.params.non_overlap_1, self.params.overlap_1,
                          self.params.burst_len_over, self.params.src_stride_over, self.params.dst_stride_over,
                          self.params.dup_repeat_merchant_non_overlap, self.params.dup_repeat_remainder_non_overlap,
                          self.params.dup_remainder_non_overlap, self.params.repeats_non_overlap,
                          self.params.burst_len_ub2gm_2, 0, self.params.dst_stride_ub2gm_2,
                          self.params.burst_len_ub2gm_3, self.params.src_stride_ub2gm_3,
                          self.params.dst_stride_ub2gm_3)

                _main(
                    self.params.loop_ho, self.params.hi_tail, self.params.ho_tail, self.params.burst_len_tail,
                    self.params.src_stride_tail, self.params.dst_stride_tail, self.params.hi_val_tail,
                    self.params.burst_len_val_tail, self.params.src_stride_val_tail, self.params.dst_stride_val_tail,
                    self.params.burst_len_src_orig_y_tail, self.params.src_stride_src_orig_y_tail,
                    self.params.repeat_times_tail, self.params.howo_co_ver_tail, self.params.mask_size_16,
                    self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                    self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                    self.params.num_instr_loop_h_tail, self.params.num_instr_loop_w, self.params.remain_mask,
                    self.params.remain_repeat_tail, self.params.num_instr_loop_w_1,
                    self.params.num_instr_loop_h_1_tail,
                    self.params.config, self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                    self.params.remain_ele_vadd, self.params.burst_len_ub_2_gm_tail, self.params.src_stride_ub_2_gm,
                    self.params.dst_stride_ub_2_gm_tail, self.params.non_overlap_1_tail, self.params.overlap_1,
                    self.params.burst_len_over, self.params.src_stride_over_tail, self.params.dst_stride_over_tail,
                    self.params.dup_repeat_merchant_non_overlap_tail,
                    self.params.dup_repeat_remainder_non_overlap_tail,
                    self.params.dup_remainder_non_overlap_tail, self.params.repeats_non_overlap_tail,
                    self.params.burst_len_ub2gm_2_tail, self.params.src_stride_ub2gm_2_tail,
                    self.params.dst_stride_ub2gm_2_tail, self.params.burst_len_ub2gm_3_tail,
                    self.params.src_stride_ub2gm_3_tail, self.params.dst_stride_ub2gm_3_tail)

            with tik_instance.else_scope():
                with tik_instance.for_range(0, self.params.loop_ho) as ho_idx:
                    _main(ho_idx, self.params.hi_batch, ho_batch, self.params.burst_len, self.params.src_stride, 0,
                          self.params.hi_val, self.params.burst_len_val, self.params.src_stride_val,
                          self.params.dst_stride_val, self.params.burst_len_src_orig_y,
                          self.params.src_stride_src_orig_y, self.params.repeat_times, self.params.howo_co_ver,
                          self.params.mask_size_16, self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                          self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                          self.params.num_instr_loop_h, self.params.num_instr_loop_w, self.params.remain_mask,
                          self.params.remain_repeat, self.params.num_instr_loop_w_1, self.params.num_instr_loop_h_1,
                          self.params.config, self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                          self.params.remain_ele_vadd, self.params.burst_len_ub_2_gm, self.params.src_stride_ub_2_gm,
                          self.params.dst_stride_ub_2_gm, self.params.non_overlap_1, self.params.overlap_1,
                          self.params.burst_len_over, self.params.src_stride_over, self.params.dst_stride_over,
                          self.params.dup_repeat_merchant_non_overlap, self.params.dup_repeat_remainder_non_overlap,
                          self.params.dup_remainder_non_overlap, self.params.repeats_non_overlap,
                          self.params.burst_len_ub2gm_2, 0, self.params.dst_stride_ub2gm_2,
                          self.params.burst_len_ub2gm_3, self.params.src_stride_ub2gm_3,
                          self.params.dst_stride_ub2gm_3)

    def _tiling_ho_wo_main_dy(self, tik_instance, core_loop, sum_core, model, param):
        """
        Just split ho, wo
        """
        ho_batch = model[0]
        wo_batch = model[1]
        c0 = self.c0
        c1 = self.params.c1
        l1_in_size = param.l1_in_size
        col_in_size = param.col_in_size
        ori_in_size = param.ori_in_size
        forward_ou_size = param.forward_ou_size
        mask_size = param.mask_size
        grad_size = param.grad_size
        zero_size = 128
        grad_sel_fp16_size = param.grad_sel_fp16_size
        grad_sel_fp32_size = param.grad_sel_fp32_size
        f_map_fp32_size = param.f_map_fp32_size
        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        ori_in_buf = buf_list[11]
        _set_vector_dup_zero(tik_instance, zero_buf, 0, 0, self.dtype)
        _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                           self.params.dup_repeat_merchant_f_map_fp32, self.params.dup_repeat_remainder_f_map_fp32,
                           self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32, f_map_fp32_size)

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            merchant = (sum_core + num_core_loop) // c1
            remainder = (sum_core + num_core_loop) % c1

            def _main_dy(loop_ho_idx, loop_wo_idx, hi, ho, wi, wo, hi_val, wi_val, burst_len, src_stride,
                         burst_len_src_orig_y, src_stride_src_orig_y, repeat_times, howo_co_ver, mask_size_16,
                         mask_size_ver, repeat_max_time, remain_repeat_time, remain_ele, repeat_max_loop_add,
                         remain_max_loop_add, remain_ele_vadd, burst_len_ub_2_gm, src_stride_ub_2_gm,
                         dst_stride_ub_2_gm, num_h, num_w):
                # ==========================
                # Init_Begin_Idx
                # ==========================

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * \
                        (self.params.hi_batch - self.params.overlap_h)
                else:
                    hi_coordinate = loop_ho_idx * self.params.hi_batch

                if self.kw >= self.sw:
                    wi_coordinate = loop_wo_idx * \
                        (self.params.wi_batch - self.params.overlap_w)
                else:
                    wi_coordinate = loop_wo_idx * self.params.wi_batch

                ho_coordinate = loop_ho_idx * ho_batch
                wo_coordinate = loop_wo_idx * wo_batch

                src_orig_x_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                    remainder * self.params.h * self.params.w * c0 + \
                    hi_coordinate * self.params.w * c0 + \
                    wi_coordinate * c0
                src_orig_y_gm = merchant * self.params.forward_ou_shape_h_w_c0 + \
                    remainder * self.params.ho * self.params.wo * c0 + \
                    ho_coordinate * self.params.wo * c0 + \
                    wo_coordinate * c0
                src_grad_gm = src_orig_y_gm

                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds
                if self.check_load3d_support:
                    input0 = [hi_val, wi_val]
                    input1 = [self.params.hi_batch, self.params.wi_batch]
                    self._gm2l1_tiling_do_ho_wo_dy(tik_instance, l1_in_buf, src_orig_x_gm, 0, input0, input1, burst_len,
                                                   src_stride)
                else:
                    dup_psm_ori_in_buf = Constant.VEC_MAX_STRIDE * 128
                    dup_repeat_merchant_ori_in_buf = param.ori_in_size // dup_psm_ori_in_buf
                    dup_repeat_remainder_ori_in_buf = param.ori_in_size % dup_psm_ori_in_buf
                    repeats_ori_in_buf = dup_repeat_remainder_ori_in_buf // 128
                    dup_remainder_ori_in_buf = dup_repeat_remainder_ori_in_buf % 128
                    _set_vector_dup_dy(tik_instance, ori_in_buf, 0, Constant.MIN_VALUE_FP16, "float16",
                                       dup_repeat_merchant_ori_in_buf, dup_repeat_remainder_ori_in_buf,
                                       dup_remainder_ori_in_buf, repeats_ori_in_buf, param.ori_in_size)
                    tik_instance.data_move(
                        ori_in_buf, self.orig_x_gm[src_orig_x_gm], 0, hi_val, wi_val, src_stride, 0)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # in the branch, do and ho are 1.
                self._copy_gm_to_ub_dy(tik_instance, forward_ou_buf, self.orig_y_gm, src_orig_y_gm, [ho, wo, c0],
                                       burst_len_src_orig_y, src_stride_src_orig_y)

                self._copy_gm_to_ub_dy(tik_instance, grad_buf, self.grads_gm, src_grad_gm, [ho, wo, c0],
                                       burst_len_src_orig_y, src_stride_src_orig_y)

                # load3d l1 to col_in_buffer
                with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                    with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                        if self.check_load3d_support:
                            src_l1 = 0
                            tik_instance.load3dv1(col_in_buf[0], l1_in_buf[src_l1], [0, 0, 0, 0], hi_val, wi_val,
                                                  0, idx_w, idx_h, 0, 0, self.sw, self.sh, self.kw, self.kh,
                                                  1, 1, 1, 1, repeat_times, 0, Constant.MIN_VALUE_FP16)

                        # ---calculate mask---
                        idx_list = [idx_h, idx_w]
                        const_list = [ho, wo, c0]

                        self._calc_mask_dy(tik_instance, buf_list, param, idx_list, const_list, howo_co_ver,
                                           mask_size_16, mask_size_ver, ori_in_buf, wi_val)

                        # ---sel(grad,zero,mask)---
                        self._sel_dy(tik_instance, buf_list, idx_list, const_list, howo_co_ver)
                        _vconv_dy(tik_instance, grad_sel_fp16_buf, 0, grad_sel_fp32_buf, 0, "float16",
                                  repeat_max_time, remain_repeat_time, remain_ele, grad_size)

                        map_index = idx_h * self.params.wi_batch * c0 + idx_w * c0
                        mask_index = 0
                        _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                      grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:], "float32",
                                      (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                      repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd, f_map_fp32_size,
                                      grad_sel_fp32_size, f_map_fp32_size)
                        _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                      grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                      "float32", (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                      repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd, f_map_fp32_size,
                                      grad_sel_fp32_size, f_map_fp32_size)

                # mov_out
                dst_ou_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                    remainder * self.params.h * self.params.w * c0 + \
                    hi_coordinate * self.params.w * c0 + \
                    wi_coordinate * c0

                def mov_atomic(num_d, dst, dst_idx, src_idx):
                    """
                    move atomic
                    """
                    in_shape = [1, num_h, num_w, c0]
                    self._ub2gm_split_do_ho_wo_dy(tik_instance, f_map_fp32_buf, src_idx, dst, dst_idx, in_shape,
                                                  self.params.hi_batch, self.params.wi_batch, burst_len_ub_2_gm,
                                                  src_stride_ub_2_gm, dst_stride_ub_2_gm)

                    _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                                       self.params.dup_repeat_merchant_f_map_fp32,
                                       self.params.dup_repeat_remainder_f_map_fp32,
                                       self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32,
                                       f_map_fp32_size)

                tik_instance.set_atomic_add(1)
                mov_atomic(1, self.ou_y_gm, dst_ou_gm, 0)
                tik_instance.set_atomic_add(0)

            with tik_instance.if_scope(self.params.wo_tail != 0):
                with tik_instance.for_range(0, self.params.loop_ho) as ho_idx:
                    with tik_instance.for_range(0, self.params.loop_wo) as wo_idx:
                        _main_dy(ho_idx, wo_idx, self.params.hi_batch, ho_batch, self.params.wi_batch, wo_batch,
                                 self.params.hi_val, self.params.wi_val, self.params.burst_len, self.params.src_stride,
                                 self.params.burst_len_src_orig_y, self.params.src_stride_src_orig_y,
                                 self.params.repeat_times, self.params.howo_co_ver, self.params.mask_size_16,
                                 self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                                 self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                                 self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                                 self.params.remain_ele_vadd, self.params.burst_len_ub_2_gm,
                                 self.params.src_stride_ub_2_gm, self.params.dst_stride_ub_2_gm, self.params.hi_val,
                                 self.params.wi_val)
                    _main_dy(ho_idx, self.params.loop_wo, self.params.hi_batch, ho_batch, self.params.wi_tail,
                             self.params.wo_tail, self.params.hi_val, self.params.wi_val_tail,
                             self.params.burst_len_tail, self.params.src_stride_tail,
                             self.params.burst_len_src_orig_y_tail, self.params.src_stride_src_orig_y_tail,
                             self.params.repeat_times_tail, self.params.howo_co_ver_tail, self.params.mask_size_16,
                             self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                             self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                             self.params.repeat_max_loop_vadd_tail, self.params.remain_max_loop_vadd_tail,
                             self.params.remain_ele_vadd_tail, self.params.burst_len_ub_2_gm_tail,
                             self.params.src_stride_ub_2_gm_tail, self.params.dst_stride_ub_2_gm_tail,
                             self.params.hi_val, self.params.wi_val_tail)

            with tik_instance.else_scope():
                with tik_instance.for_range(0, self.params.loop_ho) as ho_idx:
                    with tik_instance.for_range(0, self.params.loop_wo) as wo_idx:
                        _main_dy(ho_idx, wo_idx, self.params.hi_batch, ho_batch, self.params.wi_batch, wo_batch,
                                 self.params.hi_val, self.params.wi_val, self.params.burst_len, self.params.src_stride,
                                 self.params.burst_len_src_orig_y, self.params.src_stride_src_orig_y,
                                 self.params.repeat_times, self.params.howo_co_ver, self.params.mask_size_16,
                                 self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                                 self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                                 self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                                 self.params.remain_ele_vadd, self.params.burst_len_ub_2_gm,
                                 self.params.src_stride_ub_2_gm, self.params.dst_stride_ub_2_gm, self.params.hi_val,
                                 self.params.wi_val)

    def _pure_atomic_tiling_ho_dy(self, tik_instance, core_loop, sum_core, model, param):
        """
        ===================================================
        In the case, do must be split as part of core_axis,
        ho may be split as part of core_axis.
        Solution:
        0: split do as core, tiling_do_ho: do_batch is 1,
        1: split do_ho as core, not_tiling: do_batch is 1,
        2: split do_ho as core, tiling_do: do_batch is 1,
        3: split do_ho as core, tiling_do_ho: do_batch is 1,
        result:
        Only have ho_tail, do_tail is not existed.
        ===================================================
        """
        ho_batch = model[0]
        wo_batch = model[1]
        c0 = self.c0
        c1 = self.params.c1
        l1_in_size = param.l1_in_size
        col_in_size = param.col_in_size
        forward_ou_size = param.forward_ou_size
        mask_size = param.mask_size
        grad_size = param.grad_size
        zero_size = 128
        grad_sel_fp16_size = param.grad_sel_fp16_size
        grad_sel_fp32_size = param.grad_sel_fp32_size
        f_map_fp32_size = param.f_map_fp32_size

        buf_list = self._set_buf_tensor(tik_instance, param, is_vadds=True)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        ori_in_buf = buf_list[11]
        _set_vector_dup_zero(tik_instance, zero_buf, 0, 0, self.dtype)
        _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                           self.params.dup_repeat_merchant_f_map_fp32, self.params.dup_repeat_remainder_f_map_fp32,
                           self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32, f_map_fp32_size)

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            core_ho = self.params.core_ou_shape_h

            core_hi = self.params.core_in_shape_h

            merchant = (sum_core + num_core_loop) // (c1 * self.params.core_ho_times)
            remainder = (sum_core + num_core_loop) % (c1 * self.params.core_ho_times)

            merchant_c1 = remainder // self.params.core_ho_times
            remainder_c1 = remainder % self.params.core_ho_times

            merchant_d = remainder_c1 // self.params.core_ho_times
            remainder_d = remainder_c1 % self.params.core_ho_times

            def _main(loop_ho_idx, hi, ho, hi_val, burst_len, src_stride, dst_stride, burst_len_src_orig_y,
                      src_stride_src_orig_y, repeat_times, howo_co_ver, mask_size_16, mask_size_ver,
                      repeat_max_time_grad_sel, remain_repeat_time_grad_sel, remain_ele_grad_sel, config_params,
                      num_instr_loop_h, num_instr_loop_w, remain_mask, remain_repeat, num_instr_loop_w_1,
                      num_instr_loop_h_1, repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd, burst_len_ub2gm_2,
                      src_stride_ub2gm_2, dst_stride_ub2gm_2):
                # ==========================
                # Init_Begin_Idx
                # ==========================

                di_coordinate = merchant_d

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * (self.params.hi_batch - self.params.overlap_h) + \
                                    remainder_d * (core_hi - self.params.overlap_h)
                else:
                    hi_coordinate = loop_ho_idx * self.params.hi_batch + \
                                    remainder_d * core_hi

                do_coordinate = merchant_d
                ho_coordinate = loop_ho_idx * ho_batch + remainder_d * core_ho

                src_orig_x_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                                merchant_c1 * self.params.h * self.params.w * c0 + \
                                di_coordinate * c1 * self.params.h * self.params.w * c0 + \
                                hi_coordinate * self.params.w * c0
                src_orig_y_gm = merchant * self.params.forward_ou_shape_h_w_c0 + \
                                merchant_c1 * self.params.ho * self.params.wo * c0 + \
                                do_coordinate * c1 * self.params.ho * self.params.wo * c0 + \
                                ho_coordinate * self.params.wo * c0
                src_grad_gm = src_orig_y_gm

                in_shape = [hi_val, self.params.wi_batch, c0]
                if self.check_load3d_support:
                    self._gm2dst_tiling_do_ho_dy(tik_instance, l1_in_buf, src_orig_x_gm, 0, in_shape,
                                                 self.params.hi_batch, burst_len, src_stride, dst_stride)
                else:
                    dup_psm_ori_in_buf = Constant.VEC_MAX_STRIDE * 128
                    dup_repeat_merchant_ori_in_buf = param.ori_in_size // dup_psm_ori_in_buf
                    dup_repeat_remainder_ori_in_buf = param.ori_in_size % dup_psm_ori_in_buf
                    repeats_ori_in_buf = dup_repeat_remainder_ori_in_buf // 128
                    dup_remainder_ori_in_buf = dup_repeat_remainder_ori_in_buf % 128
                    _set_vector_dup_dy(tik_instance, ori_in_buf, 0, Constant.MIN_VALUE_FP16, "float16",
                                       dup_repeat_merchant_ori_in_buf, dup_repeat_remainder_ori_in_buf,
                                       dup_remainder_ori_in_buf, repeats_ori_in_buf, param.ori_in_size)
                    self._gm2dst_tiling_do_ho_dy(tik_instance, ori_in_buf, src_orig_x_gm, 0, in_shape,
                                                 self.params.hi_batch, burst_len, src_stride, dst_stride)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # ================================
                self._copy_gm_to_ub_dy(tik_instance, forward_ou_buf, self.orig_y_gm, src_orig_y_gm,
                                       [1, ho, wo_batch, c0], burst_len_src_orig_y, src_stride_src_orig_y)

                self._copy_gm_to_ub_dy(tik_instance, grad_buf, self.grads_gm, src_grad_gm, [ho, wo_batch, c0],
                                       burst_len_src_orig_y, src_stride_src_orig_y)
                # load3d l1 to col_in_buffer
                with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                    with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                        if self.check_load3d_support:
                            src_l1 = 0
                            tik_instance.load3dv1(col_in_buf[0], l1_in_buf[src_l1], [0, 0, 0, 0], hi,
                                                  self.params.wi_batch, 0, idx_w, idx_h, 0, 0, self.sw, self.sh,
                                                  self.kw, self.kh, 1, 1, 1, 1,
                                                  repeat_times, 0, Constant.MIN_VALUE_FP16)
                        else:
                            self._img2col(tik_instance, ori_in_buf, col_in_buf, [
                                          hi_val, self.params.wi_batch, c0], [ho, wo_batch], [idx_h, idx_w], param)

                        # ---calculate mask---
                        idx_list = [idx_h, idx_w]
                        const_list = [ho, wo_batch, c0]
                        self._calc_mask_dy(tik_instance, buf_list, param, idx_list, const_list, howo_co_ver,
                                           mask_size_16, mask_size_ver, is_vadds=True)

                        # ---sel(grad,zero,mask)---
                        self._sel_dy(tik_instance, buf_list, idx_list, const_list, howo_co_ver)

                        # ---vconv grad_sel_fp16 to fp32---
                        _vconv_dy(tik_instance, grad_sel_fp16_buf, 0, grad_sel_fp32_buf, 0, "float16",
                                  repeat_max_time_grad_sel, remain_repeat_time_grad_sel, remain_ele_grad_sel,
                                  grad_size)

                        # ---rewrite grad_sel_fp32 to f_map_fp32
                        config = (self.sw * 2, self.sw * 2, 2, self.params.sh_wi_2, self.params.sh_wi_2,
                                  self.params.wo_2)

                        with tik_instance.if_scope(config_params == 1):
                            map_index = (idx_h * self.params.wi_batch * c0 + idx_w * c0)
                            mask_index = 0
                            shape_map_hw = [self.params.hi_batch, self.params.wi_batch, c0]
                            shape_grad = [ho, wo_batch, c0]

                            _rewrite_fmap_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                             grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:],
                                             "float32",
                                             ho, shape_map_hw, shape_grad, config, num_instr_loop_h,
                                             num_instr_loop_w, remain_mask, remain_repeat, num_instr_loop_w_1,
                                             num_instr_loop_h_1)

                            _rewrite_fmap_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                             grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                             "float32", ho, shape_map_hw, shape_grad, config, num_instr_loop_h,
                                             num_instr_loop_w, remain_mask, remain_repeat, num_instr_loop_w_1,
                                             num_instr_loop_h_1)
                        with tik_instance.else_scope():
                            # map_index has three part: which hwc0 in
                            # which window, begin_index of kernel,
                            # begin_index of child kernel
                            with tik_instance.for_range(0, ho) as ho_idx:
                                map_index = (ho_idx * self.sh * self.params.wi_batch * c0) + \
                                            (idx_h * self.params.wi_batch * c0 + idx_w * c0)
                                mask_index = wo_batch * ho_idx * c0

                                _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                              grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:],
                                              "float32",
                                              (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                              repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd,
                                              f_map_fp32_size, grad_sel_fp32_size, f_map_fp32_size)
                                _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                              grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                              "float32",
                                              (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                              repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd,
                                              f_map_fp32_size, grad_sel_fp32_size, f_map_fp32_size)

                # mov_out
                dst_ou_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                    merchant_c1 * self.params.h * self.params.w * c0 + \
                    di_coordinate * c1 * self.params.h * self.params.w * c0 + \
                    hi_coordinate * self.params.w * c0

                def mov_atomic(dst, dst_idx, src_idx):
                    """
                    move atomic
                    """
                    num_h = hi_val

                    ub2gm_shape = [1, num_h, self.params.wi_batch, c0]
                    self._ub2gm_split_do_ho_2_dy(tik_instance, f_map_fp32_buf, src_idx, dst, dst_idx,
                                                 burst_len_ub2gm_2, src_stride_ub2gm_2, dst_stride_ub2gm_2)

                    _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                                       self.params.dup_repeat_merchant_f_map_fp32,
                                       self.params.dup_repeat_remainder_f_map_fp32,
                                       self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32,
                                       f_map_fp32_size)

                tik_instance.set_atomic_add(1)
                mov_atomic(self.ou_y_gm, dst_ou_gm, 0)
                tik_instance.set_atomic_add(0)

            with tik_instance.if_scope(self.params.ho_tail != 0):
                with tik_instance.for_range(0, self.params.loop_ho) as ho_idx:
                    _main(ho_idx, self.params.hi_batch, ho_batch, self.params.hi_val, self.params.burst_len,
                          self.params.src_stride, self.params.dst_stride, self.params.burst_len_src_orig_y,
                          self.params.src_stride_src_orig_y, self.params.repeat_times, self.params.howo_co_ver,
                          self.params.mask_size_16, self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                          self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel, self.params.config,
                          self.params.num_instr_loop_h, self.params.num_instr_loop_w, self.params.remain_mask,
                          self.params.remain_repeat, self.params.num_instr_loop_w_1, self.params.num_instr_loop_h_1,
                          self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                          self.params.remain_ele_vadd, self.params.burst_len_ub_2_gm, self.params.src_stride_ub_2_gm,
                          self.params.dst_stride_ub_2_gm)

                _main(self.params.loop_ho, self.params.hi_tail, self.params.ho_tail, self.params.hi_val_tail,
                      self.params.burst_len_tail, self.params.src_stride_tail, self.params.dst_stride_tail,
                      self.params.burst_len_src_orig_y_tail, self.params.src_stride_src_orig_y_tail,
                      self.params.repeat_times_tail, self.params.howo_co_ver_tail, self.params.mask_size_16,
                      self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                      self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel, self.params.config,
                      self.params.num_instr_loop_h_tail, self.params.num_instr_loop_w, self.params.remain_mask,
                      self.params.remain_repeat_tail, self.params.num_instr_loop_w_1,
                      self.params.num_instr_loop_h_1_tail, self.params.repeat_max_loop_vadd,
                      self.params.remain_max_loop_vadd, self.params.remain_ele_vadd,
                      self.params.burst_len_ub_2_gm_tail,
                      self.params.src_stride_ub_2_gm_tail, self.params.dst_stride_ub_2_gm_tail)

            with tik_instance.else_scope():
                with tik_instance.for_range(0, self.params.loop_ho) as ho_idx:
                    _main(ho_idx, self.params.hi_batch, ho_batch, self.params.hi_val, self.params.burst_len,
                          self.params.src_stride, self.params.dst_stride, self.params.burst_len_src_orig_y,
                          self.params.src_stride_src_orig_y, self.params.repeat_times, self.params.howo_co_ver,
                          self.params.mask_size_16, self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                          self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel, self.params.config,
                          self.params.num_instr_loop_h, self.params.num_instr_loop_w, self.params.remain_mask,
                          self.params.remain_repeat, self.params.num_instr_loop_w_1, self.params.num_instr_loop_h_1,
                          self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                          self.params.remain_ele_vadd, self.params.burst_len_ub_2_gm, self.params.src_stride_ub_2_gm,
                          self.params.dst_stride_ub_2_gm)

    def _pure_atomic_tiling_ho_wo_dy(self, tik_instance, core_loop, sum_core, model, param):
        """
        ===================================================
        In the case, do must be split as part of core_axis,
        ho, wo may be split as part of core_axis.
        Solution:
        0: split do as core, tiling_do_ho_wo: do_batch, ho_batch is 1
        1: split do_ho as core, tiling_do_ho_wo: do_batch, ho_batch is 1
        2: split do_ho_wo as core, not_tiling: do_batch, ho_batch is 1
        3: split do_ho_wo as core, tiling_do: do_batch, ho_batch is 1
        4: split do_ho_wo as core, tiling_do_ho: do_batch, ho_batch, is 1
        5: split do_ho_wo as core, tiling_do_ho_wo: do_batch, ho_batch, is 1
        result:
        Only have wo_tail, do_tail ho_tail are not existed.
        ===================================================
        """
        ho_batch = model[0]
        wo_batch = model[1]
        c0 = self.c0
        c1 = self.params.c1
        l1_in_size = param.l1_in_size
        col_in_size = param.col_in_size
        forward_ou_size = param.forward_ou_size
        mask_size = param.mask_size
        grad_size = param.grad_size
        zero_size = 128
        grad_sel_fp16_size = param.grad_sel_fp16_size
        grad_sel_fp32_size = param.grad_sel_fp32_size
        f_map_fp32_size = param.f_map_fp32_size

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        ori_in_buf = buf_list[11]
        _set_vector_dup_zero(tik_instance, zero_buf, 0, 0, self.dtype)
        _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                           self.params.dup_repeat_merchant_f_map_fp32, self.params.dup_repeat_remainder_f_map_fp32,
                           self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32, f_map_fp32_size)

        with tik_instance.for_range(0, core_loop) as num_core_loop:

            core_ho = self.params.core_ou_shape_h
            core_ho_times = self.params.core_ho_times
            core_hi = self.params.core_in_shape_h

            core_wo = self.params.core_ou_shape_w
            core_wo_times = self.params.core_wo_times
            core_wi = self.params.core_in_shape_w

            merchant = (sum_core + num_core_loop) // \
                       (c1 * core_ho_times * core_wo_times)
            remainder = (sum_core + num_core_loop) % \
                        (c1 * core_ho_times * core_wo_times)

            merchant_c1 = remainder // (core_ho_times * core_wo_times)
            remainder_c1 = remainder % (core_ho_times * core_wo_times)

            merchant_d = remainder_c1 // (core_ho_times * core_wo_times)
            remainder_d = remainder_c1 % (core_ho_times * core_wo_times)

            merchant_h = remainder_d // core_wo_times
            remainder_h = remainder_d % core_wo_times

            def _main(loop_ho_idx, loop_wo_idx, hi, ho, wi, wo, hi_val, wi_val, burst_len, src_stride,
                      burst_len_src_orig_y, src_stride_src_orig_y, repeat_times, howo_co_ver, mask_size_16,
                      mask_size_ver, repeat_max_time, remain_repeat_time, remain_ele, repeat_max_loop_add,
                      remain_max_loop_add, remain_ele_vadd, burst_len_ub2gm_2, src_stride_ub_2_gm, dst_stride_ub_2_gm,
                      num_h, num_w, repeat_max_loop_f_map_fp32, remain_max_loop_f_map_fp32, remain_ele_f_map_fp32):
                # ==========================
                # Init_Begin_Idx
                # ==========================

                di_coordinate = merchant_d

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * (self.params.hi_batch - self.params.overlap_h) + \
                                    merchant_h * (core_hi - self.params.overlap_h)
                else:
                    hi_coordinate = loop_ho_idx * self.params.hi_batch + \
                                    merchant_h * core_hi

                if self.kw >= self.sw:
                    wi_coordinate = loop_wo_idx * (self.params.wi_batch - self.params.overlap_w) + \
                                    remainder_h * (core_wi - self.params.overlap_w)
                else:
                    wi_coordinate = loop_wo_idx * self.params.wi_batch + \
                                    remainder_h * core_wi

                do_coordinate = merchant_d
                ho_coordinate = loop_ho_idx * ho_batch + merchant_h * core_ho
                wo_coordinate = loop_wo_idx * wo_batch + remainder_h * core_wo

                src_orig_x_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                                merchant_c1 * self.params.h * self.params.w * c0 + \
                                di_coordinate * c1 * self.params.h * self.params.w * c0 + \
                                hi_coordinate * self.params.w * c0 + \
                                wi_coordinate * c0
                src_orig_y_gm = merchant * self.params.forward_ou_shape_h_w_c0 + \
                                merchant_c1 * self.params.ho * self.params.wo * c0 + \
                                do_coordinate * c1 * self.params.ho * self.params.wo * c0 + \
                                ho_coordinate * self.params.wo * c0 + \
                                wo_coordinate * c0
                src_grad_gm = src_orig_y_gm

                if self.check_load3d_support:
                    input0 = [hi_val, wi_val]
                    input1 = [self.params.hi_batch, self.params.wi_batch]
                    self._gm2l1_tiling_do_ho_wo_dy(tik_instance, l1_in_buf, src_orig_x_gm, 0, input0, input1, burst_len,
                                                   src_stride)
                else:
                    dup_psm_ori_in_buf = Constant.VEC_MAX_STRIDE * 128
                    dup_repeat_merchant_ori_in_buf = param.ori_in_size // dup_psm_ori_in_buf
                    dup_repeat_remainder_ori_in_buf = param.ori_in_size % dup_psm_ori_in_buf
                    repeats_ori_in_buf = dup_repeat_remainder_ori_in_buf // 128
                    dup_remainder_ori_in_buf = dup_repeat_remainder_ori_in_buf % 128
                    _set_vector_dup_dy(tik_instance, ori_in_buf, 0, Constant.MIN_VALUE_FP16, "float16",
                                       dup_repeat_merchant_ori_in_buf, dup_repeat_remainder_ori_in_buf,
                                       dup_remainder_ori_in_buf, repeats_ori_in_buf, param.ori_in_size)
                    tik_instance.data_move(
                        ori_in_buf, self.orig_x_gm[src_orig_x_gm], 0, hi_val, wi_val, src_stride, 0)

                self._copy_gm_to_ub_dy(tik_instance, forward_ou_buf, self.orig_y_gm, src_orig_y_gm, [ho, wo, c0],
                                       burst_len_src_orig_y, src_stride_src_orig_y)

                self._copy_gm_to_ub_dy(tik_instance, grad_buf, self.grads_gm, src_grad_gm, [ho, wo, c0],
                                       burst_len_src_orig_y, src_stride_src_orig_y)

                # load3d l1 to col_in_buffer
                with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                    with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                        if self.check_load3d_support:
                            src_l1 = 0
                            tik_instance.load3dv1(col_in_buf[0], l1_in_buf[src_l1], [0, 0, 0, 0], hi_val, wi_val,
                                                  0, idx_w, idx_h, 0, 0, self.sw, self.sh, self.kw, self.kh,
                                                  1, 1, 1, 1, repeat_times, 0, Constant.MIN_VALUE_FP16)

                        # ---calculate mask---
                        idx_list = [idx_h, idx_w]
                        const_list = [ho, wo, c0]
                        self._calc_mask_dy(tik_instance, buf_list, param, idx_list, const_list, howo_co_ver,
                                           mask_size_16, mask_size_ver, ori_in_buf, wi_val)

                        # ---sel(grad,zero,mask)---
                        self._sel_dy(tik_instance, buf_list, idx_list, const_list, howo_co_ver)

                        _vconv_dy(tik_instance, grad_sel_fp16_buf, 0, grad_sel_fp32_buf, 0, "float16",
                                  repeat_max_time, remain_repeat_time, remain_ele, grad_size)

                        map_index = idx_h * self.params.wi_batch * c0 + idx_w * c0
                        mask_index = 0

                        _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                      grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:], "float32",
                                      (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                      repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd, f_map_fp32_size,
                                      grad_sel_fp32_size, f_map_fp32_size)
                        _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                      grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                      "float32", (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                      repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd, f_map_fp32_size,
                                      grad_sel_fp32_size, f_map_fp32_size)

                # mov_out
                dst_ou_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                    merchant_c1 * self.params.h * self.params.w * c0 + \
                    di_coordinate * c1 * self.params.h * self.params.w * c0 + \
                    hi_coordinate * self.params.w * c0 + \
                    wi_coordinate * c0

                def mov_atomic(dst, dst_idx, src_idx):
                    """
                    move atomic
                    """
                    in_shape = [1, num_h, num_w, c0]

                    self._ub2gm_split_do_ho_wo_dy(tik_instance, f_map_fp32_buf, src_idx, dst, dst_idx, in_shape,
                                                  self.params.hi_batch, self.params.wi_batch, burst_len_ub2gm_2,
                                                  src_stride_ub_2_gm, dst_stride_ub_2_gm)

                    _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32", repeat_max_loop_f_map_fp32,
                                       remain_max_loop_f_map_fp32, remain_ele_f_map_fp32,
                                       self.params.repeats_f_map_fp32, f_map_fp32_size)

                tik_instance.set_atomic_add(1)
                mov_atomic(self.ou_y_gm, dst_ou_gm, 0)
                tik_instance.set_atomic_add(0)

            with tik_instance.if_scope(self.params.wo_tail != 0):
                with tik_instance.for_range(0, self.params.loop_ho) as ho_idx:
                    with tik_instance.for_range(0, self.params.loop_wo) as wo_idx:
                        _main(ho_idx, wo_idx, self.params.hi_batch, ho_batch, self.params.wi_batch, wo_batch,
                              self.params.hi_val, self.params.wi_val, self.params.burst_len, self.params.src_stride,
                              self.params.burst_len_src_orig_y, self.params.src_stride_src_orig_y,
                              self.params.repeat_times, self.params.howo_co_ver, self.params.mask_size_16,
                              self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                              self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                              self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                              self.params.remain_ele_vadd, self.params.burst_len_ub2gm_2,
                              self.params.src_stride_ub_2_gm, self.params.dst_stride_ub_2_gm, self.params.hi_val,
                              self.params.wi_val, self.params.repeat_max_loop_f_map_fp32,
                              self.params.remain_max_loop_f_map_fp32, self.params.remain_ele_f_map_fp32)
                    _main(ho_idx, self.params.loop_wo, self.params.hi_tail, ho_batch, self.params.wi_tail,
                          self.params.wo_tail, self.params.hi_val_tail, self.params.wi_val_tail,
                          self.params.burst_len_tail, self.params.src_stride_tail,
                          self.params.burst_len_src_orig_y_tail, self.params.src_stride_src_orig_y_tail,
                          self.params.repeat_times_tail, self.params.howo_co_ver_tail, self.params.mask_size_16,
                          self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                          self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                          self.params.repeat_max_loop_vadd_tail, self.params.remain_max_loop_vadd_tail,
                          self.params.remain_ele_vadd_tail, self.params.burst_len_ub2gm_2_tail,
                          self.params.src_stride_ub_2_gm_tail, self.params.dst_stride_ub_2_gm_tail, self.params.hi_val,
                          self.params.wi_val_tail, self.params.repeat_max_loop_f_map_fp32,
                          self.params.remain_max_loop_f_map_fp32, self.params.remain_ele_f_map_fp32)

            with tik_instance.else_scope():
                with tik_instance.for_range(0, self.params.loop_ho) as ho_idx:
                    with tik_instance.for_range(0, self.params.loop_wo) as wo_idx:
                        _main(ho_idx, wo_idx, self.params.hi_batch, ho_batch, self.params.wi_batch, wo_batch,
                              self.params.hi_val, self.params.wi_val, self.params.burst_len, self.params.src_stride,
                              self.params.burst_len_src_orig_y, self.params.src_stride_src_orig_y,
                              self.params.repeat_times, self.params.howo_co_ver, self.params.mask_size_16,
                              self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                              self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                              self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                              self.params.remain_ele_vadd, self.params.burst_len_ub2gm_2,
                              self.params.src_stride_ub_2_gm, self.params.dst_stride_ub_2_gm, self.params.hi_val,
                              self.params.wi_val, self.params.repeat_max_loop_f_map_fp32,
                              self.params.remain_max_loop_f_map_fp32, self.params.remain_ele_f_map_fp32)

    def _same_pure_atomic_tiling_dy(self, tik_instance, core_loop, sum_core, model, param):
        """
        ==============================================================
        In the case, [do,ho,wo] will be infer return
        [di_batch,hi_batch,wi_batch] and [map_di, map_hi, map_wi].
        xi_batch: size of input_data which restored in l1_in_buf.
        map_xi: size of feature_map which restored in f_map_fp32_buf.
        ==============================================================
        """
        ho_batch = model[0]
        wo_batch = model[1]
        c0 = self.c0
        c1 = self.params.c1
        l1_in_size = param.l1_in_size
        col_in_size = param.col_in_size
        forward_ou_size = param.forward_ou_size
        mask_size = param.mask_size
        grad_size = param.grad_size
        zero_size = 128
        grad_sel_fp16_size = param.grad_sel_fp16_size
        grad_sel_fp32_size = param.grad_sel_fp32_size
        f_map_fp32_size = param.f_map_fp32_size

        pad_hw_top, pad_hw_bottom = self.params.pad[0][0], self.params.pad[0][1]
        pad_hw_left, pad_hw_right = self.params.pad[1][0], self.params.pad[1][1]
        pad_hw_list = [pad_hw_left, pad_hw_right, pad_hw_top, pad_hw_bottom]

        buf_list = self._set_buf_tensor(tik_instance, param, is_vadds=True)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        ori_in_buf = buf_list[11]
        _set_vector_dup_zero(tik_instance, zero_buf, 0, 0, self.dtype)
        _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                           self.params.dup_repeat_merchant_f_map_fp32, self.params.dup_repeat_remainder_f_map_fp32,
                           self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32, f_map_fp32_size)

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            merchant = (sum_core + num_core_loop) // c1
            remainder = (sum_core + num_core_loop) % c1
            merchant_c1 = remainder
            remainder_c1 = 0

            src_orig_x_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                merchant_c1 * self.params.h * self.params.w * c0
            src_orig_y_gm = merchant * self.params.forward_ou_shape_h_w_c0 + \
                merchant_c1 * self.params.ho * self.params.wo * c0
            src_grad_gm = src_orig_y_gm

            in_shape = [1, self.params.hi_batch, self.params.wi_batch, c0]
            if self.check_load3d_support:
                self._copy_gm_to_dst_buf_dy(tik_instance, l1_in_buf, src_orig_x_gm, 0, in_shape, self.params.burst_len,
                                            self.params.src_stride)
            else:
                dup_psm_ori_in_buf = Constant.VEC_MAX_STRIDE * 128
                dup_repeat_merchant_ori_in_buf = param.ori_in_size // dup_psm_ori_in_buf
                dup_repeat_remainder_ori_in_buf = param.ori_in_size % dup_psm_ori_in_buf
                repeats_ori_in_buf = dup_repeat_remainder_ori_in_buf // 128
                dup_remainder_ori_in_buf = dup_repeat_remainder_ori_in_buf % 128
                _set_vector_dup_dy(tik_instance, ori_in_buf, 0, Constant.MIN_VALUE_FP16, "float16",
                                   dup_repeat_merchant_ori_in_buf, dup_repeat_remainder_ori_in_buf,
                                   dup_remainder_ori_in_buf, repeats_ori_in_buf, param.ori_in_size)
                tik_instance.data_move(ori_in_buf[pad_hw_top * (self.params.wi_batch + pad_hw_left +
                                                                pad_hw_right) * c0 +
                                                  pad_hw_left * c0],
                                       self.orig_x_gm[src_orig_x_gm], 0, self.params.hi_batch, self.params.wi_batch,
                                       0, pad_hw_left + pad_hw_right)

            # ----COPY_ORI_OUTPUT_2_FORWARD_OU_BUF----
            self._copy_gm_to_ub_dy(tik_instance, forward_ou_buf, self.orig_y_gm, src_orig_y_gm,
                                   [1, ho_batch, wo_batch, c0], self.params.burst_len_src_orig_y,
                                   self.params.src_stride_src_orig_y)

            # ----COPY_GRAD_2_GRAD_BUF----
            self._copy_gm_to_ub_dy(tik_instance, grad_buf, self.grads_gm, src_grad_gm, [1, ho_batch, wo_batch, c0],
                                   self.params.burst_len_src_orig_y, self.params.src_stride_src_orig_y)

            # ---load3d l1 to col_in_buffer---

            with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                    if self.check_load3d_support:
                        src_l1 = 0
                        tik_instance.load3dv1(col_in_buf[0], l1_in_buf[src_l1], pad_hw_list, self.params.hi_batch,
                                              self.params.wi_batch, 0, idx_w, idx_h, self.params.pad_hw_left_neg,
                                              self.params.pad_hw_top_neg, self.sw, self.sh, self.kw, self.kh,
                                              1, 1, 1, 1, self.params.repeat_times, 0, Constant.MIN_VALUE_FP16)
                    else:
                        self._img2col(tik_instance, ori_in_buf, col_in_buf,
                                      [self.params.hi_batch, self.params.wi_batch, c0],
                                      [ho_batch, wo_batch], [idx_h, idx_w],
                                      param, pad_hw_left, pad_hw_right)

                    # ---calculate mask---
                    idx_list = [idx_h, idx_w]
                    const_list = [ho_batch, wo_batch, c0]
                    self._calc_mask_dy(tik_instance, buf_list, param, idx_list, const_list, self.params.howo_co_ver,
                                       self.params.mask_size_16, self.params.mask_size_ver, is_vadds=True)

                    # ---sel(grad,zero,mask)---
                    self._sel_dy(tik_instance, buf_list, idx_list, const_list, self.params.howo_co_ver)

                    # ---vconv grad_sel_fp16 to fp32---
                    _vconv_dy(tik_instance, grad_sel_fp16_buf, 0, grad_sel_fp32_buf, 0, "float16",
                              self.params.repeat_max_time_grad_sel, self.params.remain_repeat_time_grad_sel,
                              self.params.remain_ele_grad_sel, grad_size)

                    # ---rewrite grad_sel_fp32 to f_map_fp32
                    config = (self.sw * 2, self.sw * 2, 2, self.params.sh_wi_2, self.params.sh_wi_2, self.params.wo_2)
                    with tik_instance.if_scope(self.params.config == 1):
                        map_index = idx_h * self.params.map_wi * c0 + idx_w * c0
                        mask_index = 0
                        shape_map_hw = [self.params.map_hi, self.params.map_wi, c0]
                        shape_grad = [ho_batch, wo_batch, c0]

                        _rewrite_fmap_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                         grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:], "float32",
                                         ho_batch, shape_map_hw, shape_grad, config, self.params.num_instr_loop_h,
                                         self.params.num_instr_loop_w, self.params.remain_mask,
                                         self.params.remain_repeat, self.params.num_instr_loop_w_1,
                                         self.params.num_instr_loop_h_1)

                        _rewrite_fmap_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                         grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                         "float32", ho_batch, shape_map_hw, shape_grad, config,
                                         self.params.num_instr_loop_h, self.params.num_instr_loop_w,
                                         self.params.remain_mask, self.params.remain_repeat,
                                         self.params.num_instr_loop_w_1, self.params.num_instr_loop_h_1)
                    with tik_instance.else_scope():
                        # map_index has three part: which hwc0 in
                        # which window, begin_index of kernel,
                        # begin_index of child kernel
                        with tik_instance.for_range(0, ho_batch) as ho_idx:
                            map_index = (ho_idx * self.sh * self.params.map_wi * c0) + \
                                        (idx_h * self.params.map_wi * c0 + idx_w * c0)
                            mask_index = wo_batch * ho_idx * c0

                            _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                          grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:], "float32",
                                          (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                          self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                                          self.params.remain_ele_vadd, f_map_fp32_size, grad_sel_fp32_size,
                                          f_map_fp32_size)
                            _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                          grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                          "float32",
                                          (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                          self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                                          self.params.remain_ele_vadd, f_map_fp32_size, grad_sel_fp32_size,
                                          f_map_fp32_size)

            # ---mov_out---
            dst_ou_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                merchant_c1 * self.params.h * self.params.w * c0

            def mov_atomic(dst, dst_idx, src_idx):
                """
                move atomic
                """
                ub2gm_shape = [1, self.params.hi_batch, self.params.wi_batch, c0]
                src_idx = (pad_hw_top * self.params.map_wi + pad_hw_left) * c0

                num_bit = self.num_bit_fp32
                n_burst = ub2gm_shape[1]
                burst_len = self.params.burst_len_over
                src_stride = self.params.src_stride_val
                dst_stride = 0

                src_idx_new = src_idx
                dst_idx_new = dst_idx

                in_list = [n_burst, burst_len, src_stride, dst_stride, src_idx_new, dst_idx_new]
                with tik_instance.if_scope(src_stride > Constant.MAX_STRIDE):
                    _ultimate_data_move(tik_instance, f_map_fp32_buf, dst, in_list, num_bit)
                with tik_instance.else_scope():
                    _norm_data_move(tik_instance, f_map_fp32_buf, dst, in_list)

                _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                                   self.params.dup_repeat_merchant_f_map_fp32,
                                   self.params.dup_repeat_remainder_f_map_fp32,
                                   self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32,
                                   f_map_fp32_size)

            tik_instance.set_atomic_add(1)
            mov_atomic(self.ou_y_gm, dst_ou_gm, 0)
            tik_instance.set_atomic_add(0)

    def _same_pure_atomic_tiling_ho_dy(self, tik_instance, core_loop, sum_core, model, param):
        """
        ===================================================
        In the case, hi will be split/tiling.Due to load3d
        has the ability to fill h*w, l1_in_buf will save
        factual data(h*w).
        ===================================================
        """
        ho_batch = model[0]
        wo_batch = model[1]
        c0 = self.c0
        c1 = self.params.c1
        l1_in_size = param.l1_in_size
        col_in_size = param.col_in_size
        forward_ou_size = param.forward_ou_size
        mask_size = param.mask_size
        grad_size = param.grad_size
        zero_size = 128
        grad_sel_fp16_size = param.grad_sel_fp16_size
        grad_sel_fp32_size = param.grad_sel_fp32_size
        f_map_fp32_size = param.f_map_fp32_size

        pad_hw_top, pad_hw_bottom = self.params.pad[0][0], self.params.pad[0][1]
        pad_hw_left, pad_hw_right = self.params.pad[1][0], self.params.pad[1][1]
        pad_hw_list = [pad_hw_left, pad_hw_right, pad_hw_top, pad_hw_bottom]

        buf_list = self._set_buf_tensor(tik_instance, param, is_vadds=True)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        ori_in_buf = buf_list[11]
        _set_vector_dup_zero(tik_instance, zero_buf, 0, 0, self.dtype)
        _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                           self.params.dup_repeat_merchant_f_map_fp32, self.params.dup_repeat_remainder_f_map_fp32,
                           self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32, f_map_fp32_size)

        with tik_instance.for_range(0, core_loop) as num_core_loop:

            core_ho = self.params.core_ou_shape_h
            core_ho_times = self.params.core_ho_times
            core_hi = self.params.core_in_shape_h

            merchant = (sum_core + num_core_loop) // (c1 * core_ho_times)
            remainder = (sum_core + num_core_loop) % (c1 * core_ho_times)

            merchant_c1 = remainder // core_ho_times
            remainder_c1 = remainder % core_ho_times

            merchant_d = remainder_c1 // core_ho_times
            remainder_d = remainder_c1 % core_ho_times

            def _main(loop_ho_idx, hi, ho, hi_val1, burst_len_src_orig_y, src_stride_src_orig_y, repeat_times,
                      howo_co_ver, mask_size_16, mask_size_ver, repeat_max_time_grad_sel, remain_repeat_time_grad_sel,
                      remain_ele_grad_sel, num_instr_loop_h, num_instr_loop_w, remain_mask, remain_repeat,
                      num_instr_loop_w_1, num_instr_loop_h_1, repeat_max_loop_add, remain_max_loop_add,
                      remain_ele_vadd):
                # ==========================
                # Init_Begin_Idx
                # ==========================
                di_coordinate = merchant_d

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * (self.params.hi_batch - self.params.overlap_h) + \
                        remainder_d * (core_hi - self.params.overlap_h) - \
                        pad_hw_top
                else:
                    hi_coordinate = loop_ho_idx * self.params.hi_batch + \
                        remainder_d * core_hi - \
                        pad_hw_top

                do_coordinate = merchant_d
                ho_coordinate = loop_ho_idx * ho_batch + remainder_d * core_ho

                # init begin coordinate of di,hi.
                di_coord = di_coordinate
                hi_coord = _init_coordinate_dy(tik_instance, pad_hw_top, hi_coordinate)

                src_orig_x_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                    merchant_c1 * self.params.h * self.params.w * c0 + \
                    hi_coord * self.params.w * c0 + di_coord * c1 * self.params.h * self.params.w * c0
                src_orig_y_gm = merchant * self.params.forward_ou_shape_h_w_c0 + \
                    merchant_c1 * self.params.ho * self.params.wo * c0 + \
                    ho_coordinate * self.params.wo * c0 + \
                    do_coordinate * c1 * self.params.ho * self.params.wo * c0
                src_grad_gm = src_orig_y_gm

                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds.
                # ================================
                hi_value = hi_val1

                hi_val = hi_value
                l1_idx = 0

                h_top, h_bottom = _calc_pad_dy(tik_instance, pad_hw_top, pad_hw_bottom, hi_coordinate, hi_value,
                                               self.params.h)
                pad_hw_list[-1] = h_bottom
                pad_hw_list[-2] = h_top

                with tik_instance.if_scope(pad_hw_top != 0):

                    hi_val -= h_top
                with tik_instance.if_scope(pad_hw_bottom != 0):

                    hi_val -= h_bottom

                in_shape = [hi_val, self.params.wi_batch, c0]
                burst_len = hi_val * self.params.wi_batch * 16 * self.num_bit // Constant.MINI_UNIT
                src_stride = (self.params.forward_in_shape_h_w_2 + self.params.forward_in_shape_w_c0 *
                              (self.params.h - hi_val)) * self.num_bit // Constant.MINI_UNIT
                dst_stride = (self.params.hi_batch - hi_val) * self.params.w * self.c0 * \
                    self.num_bit // Constant.MINI_UNIT
                if self.check_load3d_support:
                    self._gm2dst_tiling_do_ho_dy(tik_instance, l1_in_buf, src_orig_x_gm,
                                                 l1_idx * self.params.hi_batch * self.params.wi_batch * c0, in_shape,
                                                 self.params.hi_batch, burst_len, src_stride, dst_stride)
                else:
                    dup_psm_ori_in_buf = Constant.VEC_MAX_STRIDE * 128
                    dup_repeat_merchant_ori_in_buf = param.ori_in_size // dup_psm_ori_in_buf
                    dup_repeat_remainder_ori_in_buf = param.ori_in_size % dup_psm_ori_in_buf
                    repeats_ori_in_buf = dup_repeat_remainder_ori_in_buf // 128
                    dup_remainder_ori_in_buf = dup_repeat_remainder_ori_in_buf % 128
                    _set_vector_dup_dy(tik_instance, ori_in_buf, 0, Constant.MIN_VALUE_FP16, "float16",
                                       dup_repeat_merchant_ori_in_buf, dup_repeat_remainder_ori_in_buf,
                                       dup_remainder_ori_in_buf, repeats_ori_in_buf, param.ori_in_size)
                    tik_instance.data_move(ori_in_buf[h_top * (self.params.wi_batch + pad_hw_left + pad_hw_right) * c0 +
                                                      pad_hw_left * c0],
                                           self.orig_x_gm[src_orig_x_gm], 0, hi_val, self.params.wi_batch,
                                           0, pad_hw_left + pad_hw_right)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # ================================
                self._copy_gm_to_ub_dy(tik_instance, forward_ou_buf, self.orig_y_gm, src_orig_y_gm, [ho, wo_batch, c0],
                                       burst_len_src_orig_y, src_stride_src_orig_y)

                self._copy_gm_to_ub_dy(tik_instance, grad_buf, self.grads_gm, src_grad_gm, [ho, wo_batch, c0],
                                       burst_len_src_orig_y, src_stride_src_orig_y)

                # ================================
                # load3d l1 to col_in_buffer
                with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                    with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                        if self.check_load3d_support:
                            src_l1 = 0
                            tik_instance.load3dv1(col_in_buf[0], l1_in_buf[src_l1], pad_hw_list, hi_val,
                                                  self.params.wi_batch, 0, idx_w, idx_h, self.params.pad_hw_left_neg,
                                                  -h_top, self.sw, self.sh, self.kw, self.kh, 1, 1, 1, 1,
                                                  repeat_times, 0, Constant.MIN_VALUE_FP16)
                        else:
                            self._img2col(tik_instance, ori_in_buf, col_in_buf,
                                          [self.params.hi_batch, self.params.wi_batch, c0],
                                          [ho_batch, wo_batch], [idx_h, idx_w],
                                          param, pad_hw_left, pad_hw_right)

                        # ---calculate mask---
                        idx_list = [idx_h, idx_w]
                        const_list = [ho, wo_batch, c0]
                        self._calc_mask_dy(tik_instance, buf_list, param, idx_list, const_list, howo_co_ver,
                                           mask_size_16, mask_size_ver, is_vadds=True)

                        # ---sel(grad,zero,mask)---
                        self._sel_dy(tik_instance, buf_list, idx_list, const_list, howo_co_ver)

                        # ---vconv grad_sel_fp16 to fp32---
                        _vconv_dy(tik_instance, grad_sel_fp16_buf, 0, grad_sel_fp32_buf, 0, "float16",
                                  repeat_max_time_grad_sel, remain_repeat_time_grad_sel, remain_ele_grad_sel,
                                  grad_size)

                        # ---rewrite grad_sel_fp32 to f_map_fp32
                        config = (self.sw * 2, self.sw * 2, 2, self.params.sh_wi_2, self.params.sh_wi_2,
                                  self.params.wo_2)
                        with tik_instance.if_scope(self.params.config == 1):
                            map_index = idx_h * self.params.map_wi * c0 + idx_w * c0
                            mask_index = 0
                            shape_map_hw = [self.params.map_hi, self.params.map_wi, c0]
                            shape_grad = [ho, wo_batch, c0]

                            _rewrite_fmap_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                             grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:],
                                             "float32", ho, shape_map_hw, shape_grad, config, num_instr_loop_h,
                                             num_instr_loop_w, remain_mask, remain_repeat, num_instr_loop_w_1,
                                             num_instr_loop_h_1)

                            _rewrite_fmap_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                             grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                             "float32", ho, shape_map_hw, shape_grad, config, num_instr_loop_h,
                                             num_instr_loop_w, remain_mask, remain_repeat, num_instr_loop_w_1,
                                             num_instr_loop_h_1)
                        with tik_instance.else_scope():
                            with tik_instance.for_range(0, ho) as ho_idx:
                                map_index = (ho_idx * self.sh * self.params.map_wi * c0) + \
                                            (idx_h * self.params.map_wi * c0 + idx_w * c0)
                                mask_index = wo_batch * ho_idx * c0

                                _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                              grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:],
                                              "float32",
                                              (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                              repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd,
                                              f_map_fp32_size, grad_sel_fp32_size, f_map_fp32_size)
                                _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                              grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                              "float32",
                                              (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                              repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd,
                                              f_map_fp32_size, grad_sel_fp32_size, f_map_fp32_size)

                # mov_out
                dst_ou_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                    merchant_c1 * self.params.h * self.params.w * c0 + \
                    di_coord * c1 * self.params.h * self.params.w * c0 + \
                    hi_coord * self.params.w * c0

                def mov_atomic(num_h, dst, dst_idx, src_idx):
                    """
                    move atomic
                    """
                    ub2gm_shape = [1, num_h, self.params.wi_batch, c0]
                    src_idx += (h_top * self.params.map_wi + pad_hw_left) * c0

                    num_bit = self.num_bit_fp32
                    n_burst = ub2gm_shape[1]
                    burst_len = self.params.burst_len_ub_2_gm
                    src_stride = self.params.src_stride_ub_2_gm
                    dst_stride = 0

                    in_list = [n_burst, burst_len, src_stride, dst_stride, src_idx, dst_idx]
                    with tik_instance.if_scope(src_stride > Constant.MAX_STRIDE):
                        _ultimate_data_move(tik_instance, f_map_fp32_buf, dst, in_list, num_bit)
                    with tik_instance.else_scope():
                        _norm_data_move(tik_instance, f_map_fp32_buf, dst, in_list)

                    # vec_dup
                    _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                                       self.params.dup_repeat_merchant_f_map_fp32,
                                       self.params.dup_repeat_remainder_f_map_fp32,
                                       self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32,
                                       f_map_fp32_size)

                tik_instance.set_atomic_add(1)
                mov_atomic(hi_val, self.ou_y_gm, dst_ou_gm, l1_idx * self.params.map_hi * self.params.map_wi * c0)
                tik_instance.set_atomic_add(0)

            with tik_instance.if_scope(self.params.ho_tail != 0):
                with tik_instance.for_range(0, self.params.loop_ho) as ho_idx:
                    _main(ho_idx, self.params.hi_batch, ho_batch, self.params.hi_val, self.params.burst_len_src_orig_y,
                          self.params.src_stride_src_orig_y, self.params.repeat_times, self.params.howo_co_ver,
                          self.params.mask_size_16, self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                          self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                          self.params.num_instr_loop_h, self.params.num_instr_loop_w, self.params.remain_mask,
                          self.params.remain_repeat, self.params.num_instr_loop_w_1, self.params.num_instr_loop_h_1,
                          self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                          self.params.remain_ele_vadd)

                _main(self.params.loop_ho, self.params.hi_tail, self.params.ho_tail, self.params.hi_val_tail,
                      self.params.burst_len_src_orig_y_tail, self.params.src_stride_src_orig_y_tail,
                      self.params.repeat_times_tail, self.params.howo_co_ver_tail, self.params.mask_size_16,
                      self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                      self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                      self.params.num_instr_loop_h_tail, self.params.num_instr_loop_w, self.params.remain_mask,
                      self.params.remain_repeat_tail, self.params.num_instr_loop_w_1,
                      self.params.num_instr_loop_h_1_tail, self.params.repeat_max_loop_vadd,
                      self.params.remain_max_loop_vadd, self.params.remain_ele_vadd)

            with tik_instance.else_scope():
                with tik_instance.for_range(0, self.params.loop_ho) as ho_idx:
                    _main(ho_idx, self.params.hi_batch, ho_batch, self.params.hi_val, self.params.burst_len_src_orig_y,
                          self.params.src_stride_src_orig_y, self.params.repeat_times, self.params.howo_co_ver,
                          self.params.mask_size_16, self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                          self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                          self.params.num_instr_loop_h, self.params.num_instr_loop_w, self.params.remain_mask,
                          self.params.remain_repeat, self.params.num_instr_loop_w_1, self.params.num_instr_loop_h_1,
                          self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                          self.params.remain_ele_vadd)

    def _same_pure_atomic_tiling_ho_wo_dy(self, tik_instance, core_loop, sum_core, model, param):
        """
        ===================================================
        In the case, do,ho,wo will be split/tiling.So,need
        to assure pad_value of different axis.
        ===================================================
        """
        ho_batch = model[0]
        wo_batch = model[1]
        c0 = self.c0
        c1 = self.params.c1
        l1_in_size = param.l1_in_size
        col_in_size = param.col_in_size
        forward_ou_size = param.forward_ou_size
        mask_size = param.mask_size
        grad_size = param.grad_size
        zero_size = 128
        grad_sel_fp16_size = param.grad_sel_fp16_size
        grad_sel_fp32_size = param.grad_sel_fp32_size
        f_map_fp32_size = param.f_map_fp32_size

        pad_hw_top, pad_hw_bottom = self.params.pad[0][0], self.params.pad[0][1]
        pad_hw_left, pad_hw_right = self.params.pad[1][0], self.params.pad[1][1]
        pad_hw_list = [pad_hw_left, pad_hw_right, pad_hw_top, pad_hw_bottom]

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        ori_in_buf = buf_list[11]
        _set_vector_dup_zero(tik_instance, zero_buf, 0, 0, self.dtype)
        _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                           self.params.dup_repeat_merchant_f_map_fp32, self.params.dup_repeat_remainder_f_map_fp32,
                           self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32, f_map_fp32_size)

        with tik_instance.for_range(0, core_loop) as num_core_loop:

            core_ho = self.params.core_ou_shape_h
            core_ho_times = self.params.core_ho_times
            core_hi = self.params.core_in_shape_h
            core_wo = self.params.core_ou_shape_w
            core_wo_times = self.params.core_wo_times
            core_wi = self.params.core_in_shape_w

            merchant = (sum_core + num_core_loop) // \
                       (c1 * core_ho_times * core_wo_times)
            remainder = (sum_core + num_core_loop) % \
                        (c1 * core_ho_times * core_wo_times)

            merchant_c1 = remainder // (core_ho_times * core_wo_times)
            remainder_c1 = remainder % (core_ho_times * core_wo_times)

            merchant_d = remainder_c1 // (core_ho_times * core_wo_times)
            remainder_d = remainder_c1 % (core_ho_times * core_wo_times)

            merchant_h = remainder_d // core_wo_times
            remainder_h = remainder_d % core_wo_times

            def _main(loop_ho_idx, loop_wo_idx, hi, ho, wi, wo, hi_value, wi_value, burst_len_src_orig_y,
                      src_stride_src_orig_y, repeat_times, howo_co_ver, mask_size_16, mask_size_ver, repeat_max_time,
                      remain_repeat_time, remain_ele, repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd):
                # ==========================
                # Init_Begin_Idx
                # ==========================

                di_coordinate = merchant_d

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * (self.params.hi_batch - self.params.overlap_h) + \
                        merchant_h * (core_hi - self.params.overlap_h) - \
                        pad_hw_top
                else:
                    hi_coordinate = loop_ho_idx * self.params.hi_batch + \
                        merchant_h * core_hi - pad_hw_top

                if self.kw >= self.sw:
                    wi_coordinate = loop_wo_idx * (self.params.wi_batch - self.params.overlap_w) + \
                        remainder_h * (core_wi - self.params.overlap_w) - pad_hw_left
                else:
                    wi_coordinate = loop_wo_idx * self.params.wi_batch + \
                        remainder_h * core_wi - pad_hw_left

                do_coordinate = merchant_d
                ho_coordinate = loop_ho_idx * ho_batch + merchant_h * core_ho
                wo_coordinate = loop_wo_idx * wo_batch + remainder_h * core_wo

                di_coord = di_coordinate
                hi_coord = _init_coordinate_dy(tik_instance, pad_hw_top, hi_coordinate)
                wi_coord = _init_coordinate_dy(tik_instance, pad_hw_left, wi_coordinate)

                src_orig_x_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                    merchant_c1 * self.params.h * self.params.w * c0 + \
                    di_coord * c1 * self.params.h * self.params.w * c0 + \
                    hi_coord * self.params.w * c0 + \
                    wi_coord * c0
                src_orig_y_gm = merchant * self.params.forward_ou_shape_h_w_c0 + \
                    merchant_c1 * self.params.ho * self.params.wo * c0 + \
                    do_coordinate * c1 * self.params.ho * self.params.wo * c0 + \
                    ho_coordinate * self.params.wo * c0 + \
                    wo_coordinate * c0
                src_grad_gm = src_orig_y_gm

                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds

                h_top, h_bottom = _calc_pad_dy(tik_instance, pad_hw_top, pad_hw_bottom, hi_coordinate, hi_value,
                                               self.params.h)
                w_top, w_bottom = _calc_pad_dy1(tik_instance, pad_hw_left, pad_hw_right, wi_coordinate, wi_value,
                                                self.params.w)
                pad_hw_list[-1], pad_hw_list[-2] = h_bottom, h_top
                pad_hw_list[-3], pad_hw_list[-4] = w_bottom, w_top

                # gm2l1: filled regions don't move except d
                hi_val = hi_value
                wi_val = wi_value

                hi_val = hi_val - h_top - h_bottom
                wi_val = wi_val - w_top - w_bottom
                l1_idx = 0

                input0 = [hi_val, wi_val]
                input1 = [self.params.hi_batch, self.params.wi_batch]
                if self.check_load3d_support:
                    self._gm2l1_tiling_do_ho_wo_dy1(
                        tik_instance, l1_in_buf, src_orig_x_gm, 0, input0, input1)
                else:
                    dup_psm_ori_in_buf = Constant.VEC_MAX_STRIDE * 128
                    dup_repeat_merchant_ori_in_buf = param.ori_in_size // dup_psm_ori_in_buf
                    dup_repeat_remainder_ori_in_buf = param.ori_in_size % dup_psm_ori_in_buf
                    repeats_ori_in_buf = dup_repeat_remainder_ori_in_buf // 128
                    dup_remainder_ori_in_buf = dup_repeat_remainder_ori_in_buf % 128
                    _set_vector_dup_dy(tik_instance, ori_in_buf, 0, Constant.MIN_VALUE_FP16, "float16",
                                       dup_repeat_merchant_ori_in_buf, dup_repeat_remainder_ori_in_buf,
                                       dup_remainder_ori_in_buf, repeats_ori_in_buf, param.ori_in_size)
                    tik_instance.data_move(ori_in_buf[h_top * (wi_val + w_top + w_bottom) * c0 + w_top * c0],
                                           self.orig_x_gm[src_orig_x_gm], 0, hi_val, wi_val,
                                           self.params.w - wi_val, w_top + w_bottom)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # in the branch, do and ho are 1.
                # ================================
                self._copy_gm_to_ub_dy(tik_instance, forward_ou_buf, self.orig_y_gm, src_orig_y_gm, [ho, wo, c0],
                                       burst_len_src_orig_y, src_stride_src_orig_y)

                self._copy_gm_to_ub_dy(tik_instance, grad_buf, self.grads_gm, src_grad_gm, [ho, wo, c0],
                                       burst_len_src_orig_y, src_stride_src_orig_y)

                # ================================
                # load3d l1 to col_in_buffer
                with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                    with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                        if self.check_load3d_support:
                            src_l1 = 0
                            tik_instance.load3dv1(col_in_buf[0], l1_in_buf[src_l1], pad_hw_list, hi_val, wi_val,
                                                  0, idx_w, idx_h, -w_top, -h_top, self.sw, self.sh,
                                                  self.kw, self.kh, 1, 1, 1, 1,
                                                  repeat_times, 0, Constant.MIN_VALUE_FP16)

                        # ---calculate mask---
                        idx_list = [idx_h, idx_w]
                        const_list = [ho, wo, c0]
                        self._calc_mask_dy(tik_instance, buf_list, param, idx_list, const_list, howo_co_ver,
                                           mask_size_16, mask_size_ver, ori_in_buf, wi_val + w_top + w_bottom)

                        self._sel_dy(tik_instance, buf_list, idx_list, const_list, howo_co_ver)

                        _vconv_dy(tik_instance, grad_sel_fp16_buf, 0, grad_sel_fp32_buf, 0, "float16",
                                  repeat_max_time, remain_repeat_time, remain_ele, grad_size)

                        # ---rewrite grad_sel_fp32 to f_map_fp32
                        # `do = 1, ho = 1`
                        # map_index has two part: begin_index of kernel,
                        # begin_index of child kernel
                        # must use tik variable as index of grad_sel_fp32_buf,
                        # python variable is not work in grad_sel_fp32_buf[mask_index],
                        # `while x = grad_sel_fp32_buf[mask_index], y = x[n].`
                        map_index = idx_h * self.params.map_wi * c0 + idx_w * c0
                        mask_index = 0

                        _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index:],
                                      grad_sel_fp32_buf[mask_index:], f_map_fp32_buf[map_index:], "float32",
                                      (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                      repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd, f_map_fp32_size,
                                      grad_sel_fp32_size, f_map_fp32_size)
                        _vector_op_dy(tik_instance, "vadd", f_map_fp32_buf[map_index + 8:],
                                      grad_sel_fp32_buf[mask_index + 8:], f_map_fp32_buf[map_index + 8:],
                                      "float32", (self.sw * 2, self.sw * 2, 2, self.sw * 16, self.sw * 16, 16),
                                      repeat_max_loop_add, remain_max_loop_add, remain_ele_vadd, f_map_fp32_size,
                                      grad_sel_fp32_size, f_map_fp32_size)

                # mov_out
                dst_ou_gm = merchant * self.params.forward_in_shape_h_w_c0 + \
                    merchant_c1 * self.params.h * self.params.w * c0 + \
                    di_coord * c1 * self.params.h * self.params.w * c0 + \
                    hi_coord * self.params.w * c0 + \
                    wi_coord * c0

                def mov_atomic(num_d, num_h, num_w, dst, dst_idx, src_idx):
                    """
                    move atomic
                    """
                    ub2gm_shape = [num_d, num_h, num_w, c0]
                    src_idx += (h_top * self.params.map_wi + w_top) * c0

                    num_bit = self.num_bit_fp32
                    n_burst = ub2gm_shape[1]
                    burst_len = ub2gm_shape[2] * ub2gm_shape[3] * num_bit // Constant.MINI_UNIT

                    src_stride = (self.params.map_wi - num_w) * 2
                    dst_stride = (self.params.w - num_w) * 2

                    in_list = [n_burst, burst_len, src_stride, dst_stride, src_idx, dst_idx]

                    with tik_instance.if_scope(
                            tik.any(src_stride > Constant.MAX_STRIDE, dst_stride > Constant.MAX_STRIDE)):
                        _ultimate_data_move(tik_instance, f_map_fp32_buf, dst, in_list, num_bit)

                    with tik_instance.else_scope():
                        _norm_data_move(tik_instance, f_map_fp32_buf, dst, in_list)

                    _set_vector_dup_dy(tik_instance, f_map_fp32_buf, 0, 0, "float32",
                                       self.params.dup_repeat_merchant_f_map_fp32,
                                       self.params.dup_repeat_remainder_f_map_fp32,
                                       self.params.dup_remainder_f_map_fp32, self.params.repeats_f_map_fp32,
                                       f_map_fp32_size)

                tik_instance.set_atomic_add(1)
                mov_atomic(1, hi_val, wi_val, self.ou_y_gm, dst_ou_gm, 0)
                tik_instance.set_atomic_add(0)

            with tik_instance.if_scope(self.params.wo_tail != 0):
                with tik_instance.for_range(0, self.params.loop_ho) as ho_idx:
                    with tik_instance.for_range(0, self.params.loop_wo) as wo_idx:
                        _main(ho_idx, wo_idx, self.params.hi_batch, ho_batch, self.params.wi_batch, wo_batch,
                              self.params.hi_val, self.params.wi_val, self.params.burst_len_src_orig_y,
                              self.params.src_stride_src_orig_y, self.params.repeat_times, self.params.howo_co_ver,
                              self.params.mask_size_16, self.params.mask_size_ver,
                              self.params.repeat_max_time_grad_sel,
                              self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                              self.params.repeat_max_loop_vadd, self.params.remain_max_loop_vadd,
                              self.params.remain_ele_vadd)
                    _main(ho_idx, self.params.loop_wo, self.params.hi_tail, ho_batch, self.params.wi_tail,
                          self.params.wo_tail, self.params.hi_val_tail, self.params.wi_val_tail,
                          self.params.burst_len_src_orig_y_tail, self.params.src_stride_src_orig_y_tail,
                          self.params.repeat_times_tail, self.params.howo_co_ver_tail, self.params.mask_size_16,
                          self.params.mask_size_ver, self.params.repeat_max_time_grad_sel,
                          self.params.remain_repeat_time_grad_sel, self.params.remain_ele_grad_sel,
                          self.params.repeat_max_loop_vadd_tail, self.params.remain_max_loop_vadd_tail,
                          self.params.remain_ele_vadd_tail)

            with tik_instance.else_scope():
                with tik_instance.for_range(0, self.params.loop_ho) as ho_idx:
                    with tik_instance.for_range(0, self.params.loop_wo) as wo_idx:
                        _main(ho_idx, wo_idx, self.params.hi_batch, ho_batch, self.params.wi_batch, wo_batch,
                              self.params.hi_val, self.params.wi_val, self.params.burst_len_src_orig_y,
                              self.params.src_stride_src_orig_y, self.params.repeat_times, self.params.howo_co_ver,
                              self.params.mask_size_16, self.params.mask_size_ver,
                              self.params.repeat_max_time_grad_sel, self.params.remain_repeat_time_grad_sel,
                              self.params.remain_ele_grad_sel, self.params.repeat_max_loop_vadd,
                              self.params.remain_max_loop_vadd, self.params.remain_ele_vadd)

    def _compute(self):
        """
        the overall data move process
        """
        tik_instance = self._set_tik_instance(self.tik_instance)

        self.core_ou_shape = self.params.core_ou_shape
        self.core_in_shape = self.params.core_in_shape
        split_model = [self.params.new_ho, self.params.new_wo]

        param_ub = ParamsUB(self.sh, self.sw, self.kh, self.kw, self.check_load3d_support)
        with tik_instance.for_range(0, Constant.MAX_CORE, block_num=Constant.MAX_CORE) as blk_idx:
            with tik_instance.if_scope(blk_idx < self.params.core_num):
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.params.select_key == Constant.CASE_TWO):
                        core_loop = tik_instance.Scalar("int64")
                        sum_core = tik_instance.Scalar("int64")

                        with tik_instance.if_scope(self.params.total_num_div_core == 0):
                            core_loop.set_as(self.params.core_loop_params)
                            sum_core.set_as(core_loop * blk_idx)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(blk_idx < self.params.total_num_div_core_1):
                                core_loop.set_as(self.params.core_loop_params1)
                                sum_core.set_as(core_loop * blk_idx)
                            with tik_instance.else_scope():
                                core_loop.set_as(self.params.core_loop_params)
                                sum_core.set_as((core_loop + 1) * self.params.total_num_div_core_1 + core_loop *
                                                (blk_idx - self.params.total_num_div_core_1))
                        self._tiling_ho_wo_main_dy(tik_instance, core_loop, sum_core, split_model, param_ub)

                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.params.select_key == Constant.CASE_FORE):
                        core_loop = tik_instance.Scalar("int64")
                        sum_core = tik_instance.Scalar("int64")

                        with tik_instance.if_scope(self.params.total_num_div_core == 0):
                            core_loop.set_as(self.params.core_loop_params)
                            sum_core.set_as(core_loop * blk_idx)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(blk_idx < self.params.total_num_div_core_1):
                                core_loop.set_as(self.params.core_loop_params1)
                                sum_core.set_as(core_loop * blk_idx)
                            with tik_instance.else_scope():
                                core_loop.set_as(self.params.core_loop_params)
                                sum_core.set_as((core_loop + 1) * self.params.total_num_div_core_1 + core_loop *
                                                (blk_idx - self.params.total_num_div_core_1))
                        self._pure_atomic_tiling_ho_wo_dy(tik_instance, core_loop, sum_core, split_model, param_ub)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.params.select_key == Constant.CASE_SEVEN):
                        core_loop = tik_instance.Scalar("int64")
                        sum_core = tik_instance.Scalar("int64")

                        with tik_instance.if_scope(self.params.total_num_div_core == 0):
                            core_loop.set_as(self.params.core_loop_params)
                            sum_core.set_as(core_loop * blk_idx)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(blk_idx < self.params.total_num_div_core_1):
                                core_loop.set_as(self.params.core_loop_params1)
                                sum_core.set_as(core_loop * blk_idx)
                            with tik_instance.else_scope():
                                core_loop.set_as(self.params.core_loop_params)
                                sum_core.set_as((core_loop + 1) * self.params.total_num_div_core_1 + core_loop *
                                                (blk_idx - self.params.total_num_div_core_1))

                        self._same_pure_atomic_tiling_ho_wo_dy(tik_instance, core_loop, sum_core, split_model, param_ub)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.params.select_key == Constant.CASE_ZERO):
                        core_loop = tik_instance.Scalar("int64")
                        sum_core = tik_instance.Scalar("int64")

                        with tik_instance.if_scope(self.params.total_num_div_core == 0):
                            core_loop.set_as(self.params.core_loop_params)
                            sum_core.set_as(core_loop * blk_idx)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(blk_idx < self.params.total_num_div_core_1):
                                core_loop.set_as(self.params.core_loop_params1)
                                sum_core.set_as(core_loop * blk_idx)
                            with tik_instance.else_scope():
                                core_loop.set_as(self.params.core_loop_params)
                                sum_core.set_as((core_loop + 1) * self.params.total_num_div_core_1 + core_loop *
                                                (blk_idx - self.params.total_num_div_core_1))
                        self._not_tiling_main_dy(tik_instance, core_loop, sum_core, split_model, param_ub)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.params.select_key == Constant.CASE_ONE):
                        core_loop = tik_instance.Scalar("int64")
                        sum_core = tik_instance.Scalar("int64")

                        with tik_instance.if_scope(self.params.total_num_div_core == 0):
                            core_loop.set_as(self.params.core_loop_params)
                            sum_core.set_as(core_loop * blk_idx)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(blk_idx < self.params.total_num_div_core_1):
                                core_loop.set_as(self.params.core_loop_params1)
                                sum_core.set_as(core_loop * blk_idx)
                            with tik_instance.else_scope():
                                core_loop.set_as(self.params.core_loop_params)
                                sum_core.set_as((core_loop + 1) * self.params.total_num_div_core_1 + core_loop *
                                                (blk_idx - self.params.total_num_div_core_1))
                        self._tiling_ho_main_dy(tik_instance, core_loop, sum_core, split_model, param_ub)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.params.select_key == Constant.CASE_THREE):
                        core_loop = tik_instance.Scalar("int64")
                        sum_core = tik_instance.Scalar("int64")

                        with tik_instance.if_scope(self.params.total_num_div_core == 0):
                            core_loop.set_as(self.params.core_loop_params)
                            sum_core.set_as(core_loop * blk_idx)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(blk_idx < self.params.total_num_div_core_1):
                                core_loop.set_as(self.params.core_loop_params1)
                                sum_core.set_as(core_loop * blk_idx)
                            with tik_instance.else_scope():
                                core_loop.set_as(self.params.core_loop_params)
                                sum_core.set_as((core_loop + 1) * self.params.total_num_div_core_1 + core_loop *
                                                (blk_idx - self.params.total_num_div_core_1))
                        self._pure_atomic_tiling_ho_dy(tik_instance, core_loop, sum_core, split_model, param_ub)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.params.select_key == Constant.CASE_FIVE):
                        core_loop = tik_instance.Scalar("int64")
                        sum_core = tik_instance.Scalar("int64")

                        with tik_instance.if_scope(self.params.total_num_div_core == 0):
                            core_loop.set_as(self.params.core_loop_params)
                            sum_core.set_as(core_loop * blk_idx)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(blk_idx < self.params.total_num_div_core_1):
                                core_loop.set_as(self.params.core_loop_params1)
                                sum_core.set_as(core_loop * blk_idx)
                            with tik_instance.else_scope():
                                core_loop.set_as(self.params.core_loop_params)
                                sum_core.set_as((core_loop + 1) * self.params.total_num_div_core_1 + core_loop *
                                                (blk_idx - self.params.total_num_div_core_1))
                        self._same_pure_atomic_tiling_dy(tik_instance, core_loop, sum_core, split_model, param_ub)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.params.select_key == Constant.CASE_SIX):
                        core_loop = tik_instance.Scalar("int64")
                        sum_core = tik_instance.Scalar("int64")

                        with tik_instance.if_scope(self.params.total_num_div_core == 0):
                            core_loop.set_as(self.params.core_loop_params)
                            sum_core.set_as(core_loop * blk_idx)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(blk_idx < self.params.total_num_div_core_1):
                                core_loop.set_as(self.params.core_loop_params1)
                                sum_core.set_as(core_loop * blk_idx)
                            with tik_instance.else_scope():
                                core_loop.set_as(self.params.core_loop_params)
                                sum_core.set_as((core_loop + 1) * self.params.total_num_div_core_1 + core_loop *
                                                (blk_idx - self.params.total_num_div_core_1))
                        self._same_pure_atomic_tiling_ho_dy(tik_instance, core_loop, sum_core, split_model, param_ub)

        return tik_instance


def _check_param(ksize, strides, padding, data_format):
    """
    check parameters, if one is invalid, then raise error

    Parameters
    ----------
    ksize: list or tuple
    strides: list or tuple
    padding: str
    data_format: str

    Returns
    -------
    None
    """
    if padding not in ("VALID", "SAME"):
        error_manager_vector.raise_err_specific_reson(
            "max_pool_grad", "padding must be VALID or SAME")

    if len(ksize) != 4 or len(strides) != 4:
        error_manager_vector.raise_err_specific_reson(
            "max_pool_grad", "len ksieze and stride nust be 4")

    if data_format == "NHWC":
        if ksize[0] != 1 or ksize[3] != 1:
            error_manager_vector.raise_err_specific_reson(
                "max_pool_grad", "ksize[0] and ksize[3] must be 1")
        if strides[0] != 1 or strides[3] != 1:
            error_manager_vector.raise_err_specific_reson(
                "max_pool_grad", "strides[0] and strides[3] must be 1")
    elif data_format == "NCHW":
        if ksize[0] != 1 or ksize[1] != 1:
            error_manager_vector.raise_err_specific_reson(
                "max_pool_grad", "ksize[0] and ksize[1] must be 1")

        if strides[0] != 1 or strides[1] != 1:
            error_manager_vector.raise_err_specific_reson(
                "max_pool_grad", "strides[0] and strides[1] must be 1")
    else:
        error_manager_vector.raise_err_specific_reson(
            "max_pool_grad", "data_format must be NHWC or NCHW")


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,huawei-too-many-arguments
def max_pool_grad_tik(orig_x, orig_y, grads, y, ksize, strides, padding,
                      data_format="NHWC", kernel_name="max_pool_grad"):
    dtype = orig_x.get("dtype")
    ksize = list(ksize)
    strides = list(strides)
    _check_param(ksize, strides, padding, data_format)
    if data_format == "NHWC":
        kh = ksize[1]
        kw = ksize[2]
        sh = strides[1]
        sw = strides[2]
    else:
        kh = ksize[2]
        kw = ksize[3]
        sh = strides[2]
        sw = strides[3]

    if padding == "VALID":
        padding_int = 0
    else:
        padding_int = 1

    params = [ksize, strides, padding, dtype, kernel_name, data_format]
    result = MaxPoolGradCompute(params)

    result.get_tik_instance()
    check_load3d_support = tbe_platform_check.cce_conf.api_check_support("tik.load3dv1")

    l1_size = Constant.SIZE_L1 if check_load3d_support else 0

    cur_ub_size = Constant.SIZE_UB if check_load3d_support else Constant.SIZE_UB - Constant.RESERVE_UB_SIZE_Milan

    tbe_context.get_context().add_compile_info(
        "vars", {
            "ub_size": cur_ub_size,
            "l1_size": l1_size,
            "core_num": Constant.MAX_CORE,
            "kh": kh,
            "kw": kw,
            "sh": sh,
            "sw": sw,
            "padding": padding_int
        })
    tbe_context.get_context().add_compile_info("is_tik", 1)


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,huawei-too-many-arguments
def max_pool_grad_dsl(orig_x, orig_y, grads, y, ksize, strides, padding,
                    data_format="NHWC", kernel_name="max_pool_grad"):
    input_format, ori_format = orig_x["format"], orig_x["ori_format"]
    x_dtype = orig_x["dtype"]

    window_axes, normalized_ksize, normalized_strides = [], [], []
    for i, (i_axis, i_ksize, i_stride) in enumerate(zip(ori_format, ksize, strides)):
        if i_axis in ('D', 'H', 'W'):
            window_axes.append(i)
            if orig_x["shape"] == (-2, ):
                i_ksize, i_stride = -1, -1
            normalized_ksize.append(i_ksize)
            normalized_strides.append(i_stride)

    if input_format == "NC1HWC0":
        window_axes = [2, 3]
    elif input_format == "NDC1HWC0":
        window_axes = [1, 3, 4]

    ins = tbe.dsl.classify([orig_x, orig_y, grads, window_axes], "PoolGrad")
    schedules, tensors = [], []
    for (x_c, y_c, dy_c, window_axes) in ins:
        with tbe.dsl.compute():
            x_v, y_v, dy_v, ksize_v, strides_v, pads_v = shape_util.variable_shape([x_c, y_c, dy_c,
                    window_axes, normalized_ksize, normalized_strides, []], op_mode="PoolGrad")

            ph_x = tvm.placeholder(x_v, dtype=x_dtype, name="x")
            ph_y = tvm.placeholder(y_v, dtype=x_dtype, name="y")
            ph_dy = tvm.placeholder(dy_v, dtype=x_dtype, name="dy")

            dx = max_pool_grad_compute(ph_x, ph_y, ph_dy,
                                       window_axes,
                                       ksize_v,
                                       strides_v,
                                       padding_mode=padding)

            tensors.append((ph_x, ph_y, ph_dy, dx))

        with tvm.target.cce():
            schedule = tbe.dsl.auto_schedule(dx)
            schedules.append(schedule)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,huawei-too-many-arguments
def check_supported(orig_x, orig_y, grads, y, ksize, strides, padding,
                    data_format="NHWC", kernel_name="max_pool_grad"):
    """
    check whether ai_core is supported
    """
    if util_common.is_unknown([orig_x, orig_y]):
        return True, ""

    input_dtype = orig_x.get("dtype").lower()
    if input_dtype in ("float32",):
        return True, ""

    return False, ""


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,huawei-too-many-arguments
@register_operator("MaxPoolGrad", pattern="PoolGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def max_pool_grad(orig_x, orig_y, grads, y, ksize, strides, padding,
                  data_format="NHWC", kernel_name="max_pool_grad"):
    """
    main function of max_pool_grad

    Parameters
    ----------
    orig_x: dict
        shape and data type of max_pool's forward_input
    orig_y: dict
        shape and data type of max_pool's forward_output
    grads: dict
        shape and data type of grads
    y: dict
        shape and data type of y
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    padding: string
        VALID or SAME
    data_format: str
        value from `NHWC`, `NCHW`
    kernel_name: str

    Returns
    -------
    return the tik api function
    """
    if grads.get("dtype").lower() == "float32":
        max_pool_grad_dsl(orig_x, orig_y, grads, y, ksize, strides, padding,
                          data_format=data_format, kernel_name=kernel_name)
    else:
        max_pool_grad_tik(orig_x, orig_y, grads, y, ksize, strides, padding,
                          data_format=data_format, kernel_name=kernel_name)
