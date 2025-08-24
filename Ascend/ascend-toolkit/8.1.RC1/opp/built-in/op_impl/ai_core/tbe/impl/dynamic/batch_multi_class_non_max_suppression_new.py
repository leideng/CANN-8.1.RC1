#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
batch_multi_class_non_max_suppression
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len
from impl import common_util
from impl import constant_util


# 'pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements
# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals,too-many-branches
# 'pylint: disable=invalid-name
# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant
    """
    # scaling factor
    DOWN_FACTOR = 0.10
    # RPN compute 16 proposals per iteration
    RPN_PROPOSAL_NUM = 16
    # define the positive min value in fp16
    MIN_SCALAR_FP16 = 2 ** (-24)
    # define a fp16 value = 2**12
    TMP_SCALAR_FP16 = 2 ** 12
    # define repeat elements every time for vsrot32
    REPEAT_ELE = 32
    # every loop process 4096 units
    PER_LOOP_UNIT = 4096
    # location elements, [x1, y1, x2, y2]
    FOUR_DIRECTION = 4
    # b16 elements num of every block also uesed as b16 elements num of mask
    BLOCK_ELE = 16
    # b32 elements num of every block
    BLOCK_ELE_B32 = 8
    # different size of different dtype
    INT8_SIZE = 1
    FP16_SIZE = 2
    FP32_SIZE = 4
    B16_BIT = 16
    B32_BIT = 32
    # the socres_index contains four elements also marked as the class num processed every cycle
    UNIT_ELE = 4
    TILING_ARG_NUM = 16
    REPEAT_TIMES_MAX = 255
    # 0b0001 0001 0001 0001 is equals to type 3
    PATTERN_TYPE = 3
    FP16_MINS = -65504
    MAX_INT32 = 2 ** 30
    MAX_BOX = 60000
    SCALAR_TENSOR_SIZE = 16
    BLOCK_SIZE = 32
    CYCLE_NUM = 8
    I8_SIZE = 1
    B16_SIZE = 2
    B32_SIZE = 4
    I64_SIZE = 8
    BATCH_LOOP_UNIT = 1024
    MASK = 128


def tik_func_vector(tik_instance, _ub, value, dup_len):
    """
    tik_func_vector

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    _ub: ub
        vcetor ub
    value: value
        vcetor value
    dup_len: int
        vcetor data len

    Returns
    -------
    None
    """
    repeat = dup_len // 128
    repeat_tail = dup_len % 128
    offset = 0
    while repeat > 255:
        tik_instance.vector_dup(128, _ub[offset], value, 255, 1, 8)
        repeat = repeat - 255
        offset = offset + 128 * 255
    if repeat > 0:
        tik_instance.vector_dup(128, _ub[offset], value, repeat, 1, 8)
        offset = offset + 128 * repeat
    if repeat_tail > 0:
        tik_instance.vector_dup(repeat_tail, _ub[offset], value, 1, 1, 8)


def tik_func_packaged(tik_instance, function, out_dst, src0, src1, copy_num,
                      dst_blk=1, src0_blk=1, src1_blk=1, dst_rep=8, src0_rep=8,
                      src1_rep=8):
    """
    the packaged tik function, which can Support vmin, vmax, vmul, vadd, vsub function
    """
    do_dtype = out_dst.dtype
    if do_dtype in ("float16",):
        block_num = 16
    else:
        block_num = 8
    vector_num = block_num * 8
    loop_num = copy_num // (vector_num * 255)
    repeat_time = copy_num % (vector_num * 255) // vector_num
    repeat_tail = copy_num % (vector_num * 255) % vector_num
    tik_fun = None
    ori_offset_dst = ub_offset(out_dst)
    ori_offset_src0 = ub_offset(src0)
    ori_offset_src1 = ub_offset(src1)
    if function == "vmin":
        tik_fun = tik_instance.vmin
    elif function == "vmax":
        tik_fun = tik_instance.vmax
    elif function == "vmul":
        tik_fun = tik_instance.vmul
    elif function == "vadd":
        tik_fun = tik_instance.vadd
    elif function == "vsub":
        tik_fun = tik_instance.vsub

    with tik_instance.if_scope(loop_num > 0):
        with tik_instance.for_range(0, loop_num) as i:
            tik_fun(vector_num, out_dst[ori_offset_dst + i * 255 * block_num * dst_rep],
                    src0[ori_offset_src0 + i * 255 * block_num * src0_rep],
                    src1[ori_offset_src1 + i * 255 * block_num * src1_rep], 255,
                    dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)

    with tik_instance.if_scope(repeat_time > 0):
        ori_offset_dst = ori_offset_dst + loop_num * block_num * dst_rep * 255
        ori_offset_src0 = ori_offset_src0 + loop_num * 255 * block_num * src0_rep
        ori_offset_src1 = ori_offset_src1 + loop_num * block_num * src1_rep * 255
        tik_fun(vector_num, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1],
                repeat_time, dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)

    with tik_instance.if_scope(repeat_tail > 0):
        ori_offset_dst = ori_offset_dst + repeat_time * block_num * dst_rep + loop_num * block_num * dst_rep * 255
        ori_offset_src0 = ori_offset_src0 + repeat_time * block_num * src0_rep + loop_num * 255 * block_num * src0_rep
        ori_offset_src1 = ori_offset_src1 + repeat_time * block_num * src1_rep + loop_num * block_num * src1_rep * 255
        tik_fun(repeat_tail, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1],
                1, dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)


def tik_func_available(tik_instance, tik_func_vmode, out_dst, src0, src1, copy_num,
                       dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep):
    """
    the tik func which is available
    """
    do_dtype = out_dst.dtype
    if do_dtype in ("float16"):
        block_num = 16
    else:
        block_num = 8
    vector_num = block_num * 8
    repeat_time = tik_instance.Scalar(dtype='int32', name='repeat_time')
    repeat_time.set_as(copy_num // vector_num)
    repeat_tail = tik_instance.Scalar(dtype='int32', name='repeat_tail')
    repeat_tail.set_as(copy_num % vector_num)
    tik_fun = None
    if tik_func_vmode == "vmin":
        tik_fun = tik_instance.vmin

    if tik_func_vmode == "vmax":
        tik_fun = tik_instance.vmax

    if tik_func_vmode == "vsub":
        tik_fun = tik_instance.vsub

    if tik_func_vmode == "vmul":
        tik_fun = tik_instance.vmul

    with tik_instance.if_scope(repeat_time > 0):
        tik_fun(vector_num, out_dst, src0[0], src1[0],
                repeat_time,
                dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)

    with tik_instance.if_scope(repeat_tail > 0):
        offset = tik_instance.Scalar(dtype='int32', name='offset')
        offset.set_as(repeat_time * vector_num)
        tik_fun(repeat_tail, out_dst[offset], src0[offset], src1[0], 1, dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)


def ceil_div(divisor, dividend):
    res = (divisor - 1) // dividend + 1
    return res


# 'pylint: disable=too-many-instance-attributes,too-many-arguments
def get_input_gm_list(tik_instance, need_valid_num, need_clip_window,
                      boxes_gm, scores_gm, batch, box_dtype):
    """
    get_input_gm_list
    """
    clip_window_gm = None
    valid_num_gm = None
    if need_valid_num:
        valid_num_gm = tik_instance.Tensor("int32", (batch,), name="valid_num_gm", scope=tik.scope_gm)
    if need_clip_window:
        clip_window_gm = tik_instance.Tensor(box_dtype, (batch, 4), name="clip_window_gm",
                                             scope=tik.scope_gm)
    if need_valid_num and need_clip_window:
        input_gm_list = [boxes_gm, scores_gm, clip_window_gm, valid_num_gm]
    elif need_clip_window:
        input_gm_list = [boxes_gm, scores_gm, clip_window_gm]
    elif need_valid_num:
        input_gm_list = [boxes_gm, scores_gm, valid_num_gm]
    else:
        input_gm_list = [boxes_gm, scores_gm]
    return input_gm_list


# fill index for sort ub
def init_index(tik_instance, src, index, offset, do_lens=Constant.PER_LOOP_UNIT):
    """
    initialize index for tik commond vsort32

    Parameters
    ----------
    tik_instance : tik instance
    src : tensor
        index tensor in gm, dtype must be uint32
    index : tensor
        index tensor in ub, dtype must be uint32
    offset : int
        data lens (16 aligned)
    do_lens: int
        init value set as 4096

    Returns
    -------
    None
    """
    burst_lens = do_lens // Constant.BLOCK_ELE_B32
    tik_instance.data_move(index, src[offset], 0, 1, burst_lens, 0, 0)


def gm2ub_for_vsort32(tik_instance, src, idx_list, dst, do_lens):
    """
    move data from gm to ub for get_tik_func_vsort32

    Parameters
    ----------
    tik_instance : tik instance
    src : tensor
        scores tensor in ub(3D)
    idx_list : list
        batch_idx, class_idx, box_idx
    dst : tensor
        scores tensor in gm(1D)
    do_lens : int
        data lens

    Returns
    -------
    None
    """
    dtype_size = common_util.get_data_size(src.dtype)
    block_element = constant_util.BLOCK_SIZE // dtype_size
    burst_lens = do_lens // block_element
    tail_num = do_lens % block_element
    batch_idx, class_idx, box_idx = idx_list
    with tik_instance.if_scope(burst_lens > 0):
        tik_instance.data_move(dst, src[batch_idx, class_idx, box_idx], 0, 1, burst_lens, 0, 0)
    with tik_instance.for_range(0, tail_num) as idx:
        dst[burst_lens * block_element + idx].set_as(src[batch_idx, class_idx,
                                                         box_idx + burst_lens * block_element + idx])


def ub2ub(tik_instance, dst, src, count, tail_overlap=True):
    """
    move data from ub to ub
    :param tik_instance: tik instance
    :param dst: dst ub
    :param src: src ub
    :param count: count to move
    :param tail_overlap: when count is not 32 bytes align, set to allow write overlap the tail count of dst from src.
            For example, to move 5 count fof float32 data, which is not 32 bytes align,
            the tail count is 3 (32 // sizeof(float32) - (5 % (32 // sizeof(float32)))),
            if tail_overlap is `True`, will write more 3 count data at dst to make better performance
    :return: None
    """
    dtype_size = common_util.get_data_size(src.dtype)
    block_element = constant_util.BLOCK_SIZE // dtype_size
    if tail_overlap:
        burst = ceil_div(count, block_element)
        tik_instance.data_move(dst, src, 0, 1, burst, 0, 0)
    else:
        burst = count // block_element
        with tik_instance.if_scope(burst != 0):
            tik_instance.data_move(dst, src, 0, 1, burst, 0, 0)
        new_index = block_element * burst
        with tik_instance.for_range(new_index, count) as index:
            dst[index] = src[index]


def sort_scalar(tik_instance, score_idx, score_idx_out, do_lens, index):
    """
    sort_scalar
    """
    compute_lens = tik_instance.Scalar("int32", name="compute_lens", init_value=1)
    data_lens = tik_instance.Scalar("int32", name="data_lens")
    whole_lens = tik_instance.Scalar("int32", name="whole_lens")
    loop_num = tik_instance.Scalar("int32", name="loop_num")
    with tik_instance.for_range(0, index) as i:
        compute_lens.set_as(Constant.FOUR_DIRECTION * compute_lens)
    compute_lens.set_as(Constant.REPEAT_ELE * compute_lens)
    data_lens.set_as(compute_lens // Constant.FOUR_DIRECTION)
    whole_lens.set_as(compute_lens * Constant.FOUR_DIRECTION)
    loop_num.set_as(do_lens // whole_lens)
    offset = tik_instance.Scalar("uint32", init_value=0)
    with tik_instance.if_scope(loop_num > 0):
        with tik_instance.for_range(0, loop_num) as idx:
            src_list = [score_idx[offset:offset + compute_lens],
                        score_idx[offset + compute_lens:offset + compute_lens * 2],
                        score_idx[offset + compute_lens * 2:offset + compute_lens * 3],
                        score_idx[offset + compute_lens * 3:offset + compute_lens * 4]]
            count_list = [data_lens, data_lens, data_lens, data_lens]
            tik_instance.vmrgsort(score_idx_out[whole_lens * idx], src_list, count_list, False, 1)
            offset.set_as(offset + whole_lens)
    with tik_instance.else_scope():
        compute_lens.set_as(do_lens // Constant.FOUR_DIRECTION)
        whole_lens.set_as(do_lens)
        data_lens.set_as(compute_lens // Constant.FOUR_DIRECTION)
        src_list = [score_idx[0:compute_lens],
                    score_idx[compute_lens:compute_lens * 2],
                    score_idx[compute_lens * 2:compute_lens * 3],
                    score_idx[compute_lens * 3:compute_lens * 4]]
        count_list = [data_lens, data_lens, data_lens, data_lens]
        tik_instance.vmrgsort(score_idx_out, src_list, count_list, False, 1)


def cur_sort_score_idx_scalar(tik_instance, score_idx, score_idx_out, do_lens, level):
    """
    Sort score_index data with vmrgsort
    Parameters
    ----------
    tik_instance : tik instance
    score_idx : tensor
        temp score_index tensor in ub
    score_idx_out : tensor
        score_index tensor in ub
    do_lens : int/scalar
    level : scalar

    Returns
    -------
    score_idx_out:
    """

    with tik_instance.for_range(1, level + 1) as index:
        with tik_instance.if_scope(index % 2 != 0):
            sort_scalar(tik_instance, score_idx, score_idx_out, do_lens, index)
        with tik_instance.else_scope():
            sort_scalar(tik_instance, score_idx_out, score_idx, do_lens, index)
    with tik_instance.if_scope(level % 2 == 0):
        ub2ub(tik_instance, score_idx_out, score_idx, do_lens)


def cur_sort_score_idx(tik_instance, score_idx, score_idx_out, do_lens, uint_ele, level=1):
    """
    Sort score_index data with vmrgsort
    Parameters
    ----------
    tik_instance : tik instance
    score_idx : tensor
    temp score_index tensor in ub
    score_idx_out : tensor
    score_index tensor in ub
    do_lens : int
    level : cur_level

    Returns
    -------
    score_idx_out:
    """
    compute_lens = Constant.REPEAT_ELE * Constant.FOUR_DIRECTION ** level
    data_lens = compute_lens // Constant.FOUR_DIRECTION
    whole_data_lens = data_lens * uint_ele
    whole_lens = compute_lens * uint_ele
    loop_num = do_lens // whole_lens
    tail = do_lens % whole_lens
    offset = tik_instance.Scalar("uint32", init_value=0)
    if loop_num > 0:
        with tik_instance.for_range(0, loop_num) as idx:
            src_list = [score_idx[offset:offset + whole_data_lens],
                        score_idx[offset + whole_data_lens:offset + whole_data_lens * 2],
                        score_idx[offset + whole_data_lens * 2:offset + whole_data_lens * 3],
                        score_idx[offset + whole_data_lens * 3:offset + whole_data_lens * 4]]
            count_list = [data_lens, data_lens, data_lens, data_lens]
            tik_instance.vmrgsort(score_idx_out[whole_lens * idx], src_list, count_list, False, 1)
            offset.set_as(offset + whole_lens)
    if tail != 0:
        tail_mode = tail // whole_data_lens
        if tail_mode == 1:
            ub2ub(tik_instance, score_idx_out[offset], score_idx[offset], tail)
        elif tail_mode == 2:
            src_list = [score_idx[offset:offset + whole_data_lens],
                        score_idx[offset + whole_data_lens:offset + whole_data_lens * 2]]
            count_list = [data_lens, data_lens]
            tik_instance.vmrgsort(score_idx_out[offset], src_list, count_list, False, 1)
        else:
            src_list = [score_idx[offset:offset + whole_data_lens],
                        score_idx[offset + whole_data_lens:offset + whole_data_lens * 2],
                        score_idx[offset + whole_data_lens * 2:offset + whole_data_lens * 3]]
            count_list = [data_lens, data_lens, data_lens]
            tik_instance.vmrgsort(score_idx_out[offset], src_list, count_list, False, 1)

    if whole_lens >= do_lens:
        if level % 2 == 0:
            return ub2ub(tik_instance, score_idx, score_idx_out, do_lens)
    level += 1
    return cur_sort_score_idx(tik_instance, score_idx_out, score_idx, do_lens, uint_ele, level)


def get_mask_rep_stride(src):
    """
    get mask value

    Parameters
    ----------
    src : tensor
        source data in ub

    Returns
    -------
    mask : int
        mask value
    """
    if src.dtype in ["float16", "int16", "uint16"]:
        mask = Constant.BLOCK_SIZE * Constant.CYCLE_NUM // Constant.B16_SIZE
        rep_stride = mask // (Constant.BLOCK_SIZE // Constant.B16_SIZE)
    elif src.dtype in ["float32", "int32", "uint32"]:
        mask = Constant.BLOCK_SIZE * Constant.CYCLE_NUM // Constant.B32_SIZE
        rep_stride = mask // (Constant.BLOCK_SIZE // Constant.B32_SIZE)
    elif src.dtype in ["int8"]:
        mask = Constant.BLOCK_SIZE * Constant.CYCLE_NUM // Constant.I8_SIZE
        rep_stride = mask // (Constant.BLOCK_SIZE // Constant.I8_SIZE)
    elif src.dtype in ["int64", "uint64"]:
        mask = Constant.BLOCK_SIZE * Constant.CYCLE_NUM // Constant.I64_SIZE
        rep_stride = mask // (Constant.BLOCK_SIZE // Constant.I64_SIZE)
    else:
        raise RuntimeError("src.dtype can't be recognized")

    return mask, rep_stride


def ub_offset(input_ub):
    """
    get ub offset
    when ub.shape is 1D tensor offset = 0
    when ub.shape is not 1D tensor change offset = 1D
    ex:
       ub.shape = [2,2,2]
       ub1 = ub[1,:,:]
       ub_offset(ub1) = 2*2 = 4 for ub
    """
    ub_shape = input_ub.shape
    if len(ub_shape) in (0, 1):
        return 0

    return input_ub.offset


def _tik_func_single_input_with_scalar(tik_api, dst_ub, src_ub, value, do_len,
                                       dst_blk=1, src_blk=1, dst_rep=8, src_rep=8):
    """
    _tik_func_single
    """
    vmuls_type = dst_ub.dtype
    byte_num_one = common_util.get_data_size(vmuls_type)
    block_num = constant_util.BLOCK_SIZE // byte_num_one
    vector_num = block_num * constant_util.REPEAT_STRIDE_EIGHT
    repeat = do_len // vector_num
    repeat_tail = do_len % vector_num
    dst_offset = ub_offset(dst_ub)
    src_offset = ub_offset(src_ub)
    while repeat > Constant.REPEAT_TIMES_MAX:
        tik_api(vector_num, dst_ub[dst_offset], src_ub[src_offset], value,
                Constant.REPEAT_TIMES_MAX, dst_blk, src_blk, dst_rep, src_rep)
        repeat = repeat - Constant.REPEAT_TIMES_MAX
        dst_offset = dst_offset + block_num * Constant.REPEAT_TIMES_MAX * dst_rep
        src_offset = src_offset + block_num * Constant.REPEAT_TIMES_MAX * src_rep
    if repeat > 0:
        tik_api(vector_num, dst_ub[dst_offset], src_ub[src_offset], value,
                repeat, dst_blk, src_blk, dst_rep, src_rep)
        dst_offset = dst_offset + block_num * repeat * dst_rep
        src_offset = src_offset + block_num * repeat * src_rep
    if repeat_tail > 0:
        tik_api(repeat_tail, dst_ub[dst_offset], src_ub[src_offset], value,
                1, dst_blk, src_blk, dst_rep, src_rep)


def tik_func_vmins(tik_instance, dst_ub, src_ub, value, do_len,
                   dst_blk=1, src_blk=1, dst_rep=8, src_rep=8):
    """
    tik_func_vmins
    """
    _tik_func_single_input_with_scalar(tik_instance.vmins, dst_ub, src_ub, value, do_len,
                                       dst_blk, src_blk, dst_rep, src_rep)


def tik_func_vmaxs(tik_instance, dst_ub, src_ub, value, do_len,
                   dst_blk=1, src_blk=1, dst_rep=8, src_rep=8):
    """
    tik_func_vmaxs
    """
    _tik_func_single_input_with_scalar(tik_instance.vmaxs, dst_ub, src_ub, value, do_len,
                                       dst_blk, src_blk, dst_rep, src_rep)


def tik_fuc_vrec_newton(tik_instance, vrec_ub, origin_ub, do_len, newton_iteration=2, block_num=16):
    """tik_fuc_vrec_newton
    """
    with tik_instance.new_stmt_scope():
        vrec_newton_1 = tik_instance.Tensor(
            vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
            name="vrec_newton_1", scope=tik.scope_ubuf)
        vrec_newton_2 = tik_instance.Tensor(
            vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
            name="vrec_newton_2", scope=tik.scope_ubuf)

        def _one_newton():
            tik_instance.vmul(2, vrec_newton_1, vrec_ub, origin_ub, 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vmuls(2, vrec_newton_2, vrec_newton_1, -1, 1, 1, 1, 8, 8)
            tik_instance.vadds(2, vrec_newton_1, vrec_newton_2, 2, 1, 1, 1, 8, 8)
            tik_instance.vmul(2, vrec_ub, vrec_newton_1, vrec_ub, 1, 1, 1, 1, 8, 8, 8)

        for _ in range(newton_iteration):
            _one_newton()


def tik_func_vmuls(tik_instance, dst_ub, src_ub, value, do_len):
    """tik_func_vmuls
    """
    mask = 128
    cur_dtype = dst_ub.dtype
    if cur_dtype == "float32":
        mask = 64
    repeat = do_len // mask
    repeat_tail = do_len % mask
    offset = dst_ub.offset
    with tik_instance.if_scope(repeat > 0):
        tik_instance.vmuls(mask, dst_ub[offset], src_ub[offset], value,
                           repeat, 1, 1, 8, 8)
        offset = offset + mask * repeat
    with tik_instance.if_scope(repeat_tail > 0):
        tik_instance.vmuls(repeat_tail, dst_ub[offset], src_ub[offset], value,
                           1, 1, 1, 8, 8)


def change_coordinate_frame_compute(tik_instance, clip_window_value_list, ub_tmp_boxes, do_num):
    """change_coordinate_frame_compute
    """
    if clip_window_value_list is None:
        # no need to do clip_window and change_coordinate_frame
        return
    is_need_change_coordinate_frame = False
    if len(clip_window_value_list) == 6:
        is_need_change_coordinate_frame = True

    if is_need_change_coordinate_frame:
        h_scale_scale = clip_window_value_list[4]
        w_scale_scale = clip_window_value_list[5]
        tik_func_vmuls(tik_instance, ub_tmp_boxes[0], ub_tmp_boxes[0], h_scale_scale, do_num)
        tik_func_vmuls(tik_instance, ub_tmp_boxes[1], ub_tmp_boxes[1], w_scale_scale, do_num)
        tik_func_vmuls(tik_instance, ub_tmp_boxes[2], ub_tmp_boxes[2], h_scale_scale, do_num)
        tik_func_vmuls(tik_instance, ub_tmp_boxes[3], ub_tmp_boxes[3], w_scale_scale, do_num)


class BMCNMS:
    """
    a brand new compute flow, temporarily support fp16
    step 1- initialize, get core_used by batches with the tiling file
    step 2- sorted the scores and get the corresponding index
    step 3- select the indexes based on the score_thresh with dichotomous ordering
    step 4- according to the selected indexes, move the top 4096 scores boxes for iou selection
    step 5- do nms for each class in each batch use top 4094 proposals
    step 6- sorted the scores of every batches to get the top max_size boxes
    step 7- move the data out to gm
    """

    def __init__(self, boxes, scores, clip_window, num_valid_boxes, score_thresh, iou_thresh, max_size_per_class,
                 max_total_size, change_coordinate_frame, transpose_box, kernel_name):
        """
        Init BMCNMS base parameters
        """
        self.boxes_type = boxes.get("dtype")
        self.scores_type = scores.get("dtype")
        # when input have no class dim, will extend 1 for input shape
        self.need_clip_window = False
        self.clip_window = clip_window
        self.num_valid_boxes = num_valid_boxes
        self.change_coordinate_frame = change_coordinate_frame
        self.transpose_box = transpose_box
        if self.boxes_type == "float16":
            self.per_loop_unit = 4096
            self.unit_ele = 4
            self.dtype_byte = 2
            self.block = Constant.BLOCK_ELE
        elif self.boxes_type == "float32":
            self.per_loop_unit = 2048
            self.unit_ele = 2
            self.dtype_byte = 4
            self.block = Constant.BLOCK_ELE_B32

        self.need_valid_num = False
        self.tiling_dtype = 'int32'
        self.dtype_bytes_size_tiling = get_bit_len(self.tiling_dtype) // 8
        block_bite_size = 32
        self.tiling_each_block = block_bite_size // self.dtype_bytes_size_tiling

        # whether down the boxes to avoid fp16 overflow
        self.down_flag = True

        self.is_second_nms = False
        self.kernel_name = kernel_name
        # tiling attrs
        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.gm_output_shape = (100000,)
        self.gm_work_size = (1600000,)
        self.gm_work_boxes_size = (3200000,)
        self.proposal_topk_k = self.ub_size // 4 // 16
        self.proposal_topk_k = min(self.proposal_topk_k, 255 * 16)
        self.core_used_c = 1
        self.class_per_core = 1
        self.class_last_core = 1

        self.core_used_b = 1
        self.batch_per_core = 1
        self.batch_last_core = 1

        self.loop_num = 1
        self.last_loop_size = 0

        self.input_gm_list = []
        self.output_gm_list = []
        self.bacth = None
        self.align_boxes = self.tik_instance.Scalar("int32", name="align_boxes")
        self.cal_mode = self.tik_instance.Scalar(dtype='int32')
        self.core_used_b = self.tik_instance.Scalar(dtype='int32')
        self.batch_per_core = self.tik_instance.Scalar(dtype='int32')
        self.batch_last_core = self.tik_instance.Scalar(dtype='int32')
        self.batch = self.tik_instance.Scalar(dtype='int32')
        self.classes = self.tik_instance.Scalar(dtype='int32')
        self.boxes_num = self.tik_instance.Scalar(dtype='int32')
        self.classes_box = self.tik_instance.Scalar(dtype='int32')
        self.score_thresh = self.tik_instance.Scalar(dtype="float32")
        self.iou_thresh = self.tik_instance.Scalar(dtype="float32")
        self.max_size_per_class = self.tik_instance.Scalar(dtype="int32")
        self.max_total_size = self.tik_instance.Scalar(dtype="int32")
        if self.clip_window:
            self.need_clip_window = True
            self.per_loop_unit = 2048
        if self.num_valid_boxes:
            self.need_valid_num = True
        self.clip_window_value_list = None
        self.init_gm_tensor()

    def init_gm_tensor(self):
        """
        init_gm_tensor
        """
        if self.need_clip_window:
            if self.change_coordinate_frame:
                self.down_flag = False
                self.clip_window_value_list = [self.tik_instance.Scalar(dtype=self.boxes_type) for _ in range(6)]
            else:
                self.clip_window_value_list = [self.tik_instance.Scalar(dtype=self.boxes_type) for _ in range(4)]
        else:
            self.clip_window_value_list = None

        idx_size = ceil_div(Constant.MAX_BOX, 4096) * 4096
        idx_init = [i for i in range(idx_size)]
        self.idx_gm = self.tik_instance.Tensor("uint32",
                                               [idx_size, ],
                                               name="idx_gm",
                                               scope=tik.scope_gm, init_value=idx_init)

        # workspace
        self.workspace_score_idx = self.tik_instance.Tensor(self.boxes_type,
                                                            self.gm_work_boxes_size,
                                                            name="workspace_score_idx",
                                                            scope=tik.scope_gm, is_workspace=True)

        self.workspace_boxes = self.tik_instance.Tensor(self.boxes_type,
                                                        self.gm_work_size,
                                                        name="workspace_boxes",
                                                        scope=tik.scope_gm, is_workspace=True)

        self.workspace_scores = self.tik_instance.Tensor(self.boxes_type,
                                                         self.gm_work_size,
                                                         name="workspace_scores",
                                                         scope=tik.scope_gm, is_workspace=True)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,), name='tiling_gm',
                                                  scope=tik.scope_gm)

    def get_tiling_args(self):
        """
        get_tiling_args
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor(
                self.tiling_dtype, (Constant.TILING_ARG_NUM,), name='tiling_ub', scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, Constant.SCALAR_TENSOR_SIZE // \
                                        self.tiling_each_block, 0, 0)
            self.cal_mode.set_as(tiling_ub[0])
            self.core_used_b.set_as(tiling_ub[1])
            self.batch_per_core.set_as(tiling_ub[2])
            self.batch_last_core.set_as(tiling_ub[3])
            self.batch.set_as(tiling_ub[4])
            self.classes.set_as(tiling_ub[5])
            self.boxes_num.set_as(tiling_ub[6])
            self.classes_box.set_as(tiling_ub[9])
            self.score_thresh.set_as(tiling_ub[10])
            self.iou_thresh.set_as(tiling_ub[11])
            self.iou_thresh.set_as(self.iou_thresh / (1 + self.iou_thresh))
            self.max_size_per_class.set_as(tiling_ub[12])
            self.max_total_size.set_as(tiling_ub[13])

    def bmcnms_compute(self):
        """
        compute of batch_multi_class_non_max_suppression

        Parameters
        ----------
        None

        Returns
        -------
        compile info
        """
        self.get_tiling_args()
        boxes_gm = self.tik_instance.Tensor(self.boxes_type, (self.batch, self.classes_box, 4, self.boxes_num),
                                            name="boxes_gm", scope=tik.scope_gm)
        scores_gm = self.tik_instance.Tensor(self.boxes_type, (self.batch, self.classes, self.boxes_num),
                                             name="scores_gm", scope=tik.scope_gm)

        # init gm output
        nmsed_boxes_gm = self.tik_instance.Tensor(self.boxes_type, self.gm_output_shape,
                                                  name="nmsed_boxes_gm", scope=tik.scope_gm)
        nmsed_scores_gm = self.tik_instance.Tensor(self.boxes_type, self.gm_output_shape,
                                                   name="nmsed_scores_gm", scope=tik.scope_gm)
        nmsed_classes_gm = self.tik_instance.Tensor(self.boxes_type, self.gm_output_shape,
                                                    name="nmsed_classes_gm", scope=tik.scope_gm)
        nmsed_num_gm = self.tik_instance.Tensor("int32", self.gm_output_shape,
                                                name="nmsed_num_gm", scope=tik.scope_gm)
        self.output_gm_list = [nmsed_boxes_gm, nmsed_scores_gm, nmsed_classes_gm, nmsed_num_gm]

        self.input_gm_list = get_input_gm_list(self.tik_instance, self.need_valid_num,
                                               self.need_clip_window, boxes_gm,
                                               scores_gm, self.batch, self.boxes_type)

        self.class_nms_compute(boxes_gm, scores_gm)

        opt_config = {
            "enable_const_fold": True
        }
        tbe_context.get_context().add_compile_info('vars',
                                                   {'aicore_num': self.aicore_num,
                                                    'proposal_topk_k': self.proposal_topk_k,
                                                    "dtype_byte": self.dtype_byte, "is_old": 0,
                                                    "is_second_nms": 0})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.input_gm_list,
                                   outputs=self.output_gm_list,
                                   flowtable=[self.tiling_gm],
                                   output_files_path=None,
                                   enable_l2=False, config=opt_config)
        return self.tik_instance

    def gm2ub_for_vsort32_workspace(self, src, idx_list, dst, do_lens):
        """
        move data from gm to ub for get_tik_func_vsort32

        Parameters
        ----------
        src : tensor
            scores tensor in ub(3D)
        idx_list : list
            batch_idx, class_idx, box_idx
        dst : tensor
            scores tensor in gm(1D)
        do_lens : int
            data lens

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        dtype_size = common_util.get_data_size(src.dtype)
        block_element = constant_util.BLOCK_SIZE // dtype_size
        burst_lens = do_lens // block_element
        tail_num = do_lens % block_element
        batch_idx, class_idx, box_idx = idx_list
        offset = tik_instance.Scalar("int32")
        offset.set_as(batch_idx * self.classes * self.max_size_per_class * 1 + class_idx * self.max_size_per_class * 1
                      + box_idx)
        with tik_instance.if_scope(burst_lens > 0):
            tik_instance.data_move(dst, src[offset], 0, 1, burst_lens, 0, 0)
        with tik_instance.for_range(0, tail_num) as idx:
            dst[burst_lens * block_element + idx].set_as(src[burst_lens * block_element + offset + idx])

    def class_nms_compute(self, boxes, scores):
        """
        main compute cycle

        Parameters
        ----------
        boxes : tensor
            input location data in gm
        scores : tensor
            input scores data in gm

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        self.align_boxes.set_as(((self.boxes_num * 4 - 1) // 32 + 1) * 32)
        # do nms with multi cores
        with tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as core_index:
            with tik_instance.if_scope(core_index < self.core_used_b):
                with self.tik_instance.if_scope(tik.any(self.core_used_b == 1,
                                                        self.batch_per_core == self.batch_last_core)):
                    with tik_instance.for_range(0, self.batch_last_core) as index:
                        batch_index = core_index * self.batch_last_core + index
                        with tik_instance.for_range(0, self.classes) as class_index:
                            with tik_instance.new_stmt_scope():
                                self.do_nms(batch_index, class_index, boxes, scores)
                        with self.tik_instance.if_scope(self.classes != 1):
                            self.sort_class_per_batch(batch_index)

                with self.tik_instance.else_scope():
                    with tik_instance.if_scope(core_index < self.core_used_b - 1):
                        with tik_instance.for_range(0, self.batch_per_core) as index:
                            batch_index = core_index * self.batch_per_core + index
                            with tik_instance.for_range(0, self.classes) as class_index:
                                with tik_instance.new_stmt_scope():
                                    self.do_nms(batch_index, class_index, boxes, scores)
                            with self.tik_instance.if_scope(self.classes != 1):
                                self.sort_class_per_batch(batch_index)

                    with tik_instance.else_scope():
                        with tik_instance.for_range(0, self.batch_last_core) as index:
                            batch_index = core_index * self.batch_per_core + index
                            with tik_instance.for_range(0, self.classes) as class_index:
                                with tik_instance.new_stmt_scope():
                                    self.do_nms(batch_index, class_index, boxes, scores)
                            with self.tik_instance.if_scope(self.classes != 1):
                                self.sort_class_per_batch(batch_index)

    def do_nms(self, real_batch_idx, class_idx, boxes, scores):
        """
        deal with class nms compute

        Parameters
        ----------
        real_batch_idx : int
            batch index
        class_idx : int
            class index
        boxes : tensor
            input location data in gm
        scores: tensor
            input scores data in gm
        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        # set as 4096 or 2048
        shape_aligned = self.per_loop_unit
        vector_mask, _ = get_mask_rep_stride(boxes)
        repeat_time = shape_aligned // vector_mask
        eff_size = tik_instance.Scalar(dtype="uint32", name="eff_size", init_value=self.per_loop_unit)
        eff_lens = tik_instance.Scalar(dtype="uint32", name="eff_lens", init_value=self.per_loop_unit)
        pre_eff_lens = tik_instance.Scalar(dtype="uint32", name="pre_eff_lens", init_value=0)
        scalar_window = self.clip_window_value_list

        x1_ub = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="x1_ub", scope=tik.scope_ubuf)
        x2_ub = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="x2_ub", scope=tik.scope_ubuf)
        y1_ub = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="y1_ub", scope=tik.scope_ubuf)
        y2_ub = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="y2_ub", scope=tik.scope_ubuf)
        scores_ub = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="scores_ub", scope=tik.scope_ubuf)
        input_boxes_ub_lis = [x1_ub, y1_ub, x2_ub, y2_ub]

        with tik_instance.new_stmt_scope():
            scores_idx_out = tik_instance.Tensor(self.boxes_type, [self.per_loop_unit * self.unit_ele * 2, ],
                                                 name="scores_idx_out", scope=tik.scope_ubuf)
            # first round, process top 4096 units
            self.gen_score_index(real_batch_idx, class_idx, scores, scores_idx_out, input_boxes_ub_lis)
            self.select_threshold(scores_idx_out, eff_size, gate_value=self.score_thresh)
            with tik_instance.if_scope(eff_size > self.boxes_num):
                eff_size.set_as(self.boxes_num)

            with tik_instance.if_scope(tik.all(self.classes_box != self.classes, self.classes_box == 1)):
                self.get_boxes_after_score_thresh(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, real_batch_idx, 0, boxes,
                                                  scores_idx_out, eff_size)
            with tik_instance.else_scope():
                self.get_boxes_after_score_thresh(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, real_batch_idx,
                                                  class_idx, boxes, scores_idx_out, eff_size)
            input_boxes_ub_list = [y1_ub, x1_ub, y2_ub, x2_ub]
            if self.need_clip_window:
                self.clip_window_compute_final(tik_instance, scores_ub, input_boxes_ub_list, scalar_window,
                                               eff_size, data_each_block=self.block)
            with tik_instance.new_stmt_scope():
                change_coordinate_frame_compute(tik_instance, scalar_window, input_boxes_ub_list, eff_size)
            if self.down_flag:
                tik_instance.vmuls(vector_mask, x1_ub, x1_ub, 0.1, repeat_time, 1, 1, 8, 8)
                tik_instance.vmuls(vector_mask, x2_ub, x2_ub, 0.1, repeat_time, 1, 1, 8, 8)
                tik_instance.vmuls(vector_mask, y1_ub, y1_ub, 0.1, repeat_time, 1, 1, 8, 8)
                tik_instance.vmuls(vector_mask, y2_ub, y2_ub, 0.1, repeat_time, 1, 1, 8, 8)
            self.iou_selection(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, eff_size)
            pre_eff_lens.set_as(eff_lens)

        with tik_instance.if_scope(
                tik.any(eff_lens >= self.max_size_per_class, self.per_loop_unit >= self.boxes_num)):
            with tik_instance.if_scope(self.classes == 1):
                self.sort_single_class_per_batch(real_batch_idx, eff_lens, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub)
            with self.tik_instance.else_scope():
                self.store_data(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, real_batch_idx, class_idx)

        with tik_instance.else_scope():
            # to process the second round
            with tik_instance.new_stmt_scope():
                scores_idx_out = tik_instance.Tensor(self.boxes_type, [self.per_loop_unit * self.unit_ele * 2, ],
                                                     name="scores_idx_out", scope=tik.scope_ubuf)
                self.sort_second_round_data(real_batch_idx, class_idx, scores_idx_out)
                self.select_threshold(scores_idx_out, eff_size, gate_value=self.score_thresh)
                with tik_instance.if_scope(tik.all(self.boxes_num > self.per_loop_unit,
                                                   eff_size > (self.boxes_num - self.per_loop_unit))):
                    eff_size.set_as(self.boxes_num - self.per_loop_unit)
                eff_lens.set_as(self.per_loop_unit)
                with tik_instance.if_scope(eff_size > 0):
                    with tik_instance.if_scope(tik.all(self.classes_box != self.classes, self.classes_box == 1)):
                        self.get_boxes_after_score_thresh(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, real_batch_idx, 0,
                                                          boxes, scores_idx_out, eff_size, pre_eff_lens)
                    with tik_instance.else_scope():
                        self.get_boxes_after_score_thresh(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                                                          real_batch_idx, class_idx, boxes,
                                                          scores_idx_out, eff_size, pre_eff_lens)
                    input_boxes_ub_list = [y1_ub, x1_ub, y2_ub, x2_ub]
                    if self.need_clip_window:
                        self.clip_window_compute_final(tik_instance, scores_ub, input_boxes_ub_list, scalar_window,
                                                       eff_size, data_each_block=self.block)
                    with tik_instance.new_stmt_scope():
                        change_coordinate_frame_compute(tik_instance, scalar_window, input_boxes_ub_list, eff_size)
                    self.iou_selection(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, eff_lens)
                    cur = tik_instance.Scalar(dtype="int32")
                    with tik_instance.if_scope(self.max_total_size > self.max_size_per_class):
                        cur.set_as(self.max_total_size)
                    with tik_instance.else_scope():
                        cur.set_as(self.max_size_per_class)
                    with tik_instance.if_scope(eff_lens < cur):
                        self.set_default_to_scores(scores_ub, eff_lens, self.per_loop_unit)
            with tik_instance.if_scope(self.classes == 1):
                self.sort_single_class_per_batch(real_batch_idx, eff_lens, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub)
            with tik_instance.else_scope():
                self.store_data(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, real_batch_idx, class_idx)

    def set_default_to_scores(self, scores, data_size, whole_size=Constant.PER_LOOP_UNIT):
        """
        when data size is less than 4096, set default value(-65504) to scores[data_size:whole_size]

        Parameters
        ----------
        scores : tensor
            score tensor in ub
        data_size : scalar
            effective data lens
        whole_size : int
            shape lens

        Returns
        -------
        None
        """
        vector_mask, rep_stride = get_mask_rep_stride(scores)

        tik_instance = self.tik_instance
        size = tik_instance.Scalar("uint32")
        offset = tik_instance.Scalar("uint32")
        repeat_times = tik_instance.Scalar("uint32")
        tail = tik_instance.Scalar("uint32")

        size.set_as(whole_size - data_size)
        repeat_times.set_as(size // vector_mask)
        tail.set_as(size % vector_mask)

        with tik_instance.if_scope(tail != 0):
            offset.set_as(((data_size - 1) // vector_mask + 1) * vector_mask)
            with tik_instance.for_range(data_size, offset) as _idx:
                scores[_idx].set_as(0)
            tik_instance.vec_dup(vector_mask, scores[offset], 0, repeat_times, rep_stride)
        with tik_instance.else_scope():
            offset.set_as(data_size)
            tik_instance.vec_dup(vector_mask, scores[offset], 0, repeat_times, rep_stride)

    def sort_second_round_data(self, real_batch_idx, class_idx, dst):
        """
        when second round, get top 4096 scores and index from workspace

        Parameters
        ----------
        real_batch_idx : int
            batch index
        class_idx : int
            class index
        dst : tensor
            scores_idx_out

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        per_loop_ele = self.per_loop_unit
        score_idx_lens = per_loop_ele * self.unit_ele
        # as the first cycle takes 4096 units
        loop_num = self.tik_instance.Scalar("int32", name="loop_num")
        tail = self.tik_instance.Scalar("int32", name="tail")
        left_size = self.tik_instance.Scalar("int32", name="left_size")
        burst_lens = self.tik_instance.Scalar("int32", name="burst_lens")
        tail_left = self.tik_instance.Scalar("int32", name="tail_left")
        offset = self.tik_instance.Scalar("int32", name="offset")
        offset.set_as(real_batch_idx * self.classes * self.align_boxes + class_idx * self.align_boxes)
        loop_num.set_as((self.boxes_num - self.per_loop_unit) // per_loop_ele)
        tail.set_as(self.boxes_num - self.per_loop_unit - loop_num * per_loop_ele)
        left_size.set_as(self.boxes_num - self.per_loop_unit)

        self.init_tensor(dst, score_idx_lens * 2, 0)

        with tik_instance.new_stmt_scope():
            # define the tmp tensor, as 32 bytes aligned required
            scores_idx_ub = tik_instance.Tensor(self.boxes_type, [score_idx_lens * 2, ],
                                                name="scores_idx_ub", scope=tik.scope_ubuf)
            with tik_instance.if_scope(loop_num > 0):
                # the first 4096 units
                self.init_tensor(scores_idx_ub, score_idx_lens * 2)
                burst_lens_base = score_idx_lens // self.block
                tik_instance.data_move(dst, self.workspace_score_idx[offset], 0, 1, burst_lens_base, 0, 0)

                with tik_instance.for_range(1, loop_num) as loop_idx:
                    # set value for index
                    tik_instance.data_move(scores_idx_ub, self.workspace_score_idx[offset + score_idx_lens * loop_idx],
                                           0, 1, burst_lens_base, 0, 0)
                    tik_instance.data_move(scores_idx_ub[score_idx_lens], dst, 0, 1, burst_lens_base, 0, 0)
                    cur_sort_score_idx(tik_instance, scores_idx_ub, dst, score_idx_lens * 2, self.unit_ele)

                with tik_instance.if_scope(tail > 0):
                    # init scores_ub & scores_idx_ub in order to clear the pre data
                    self.init_tensor(scores_idx_ub, score_idx_lens)

                    burst_lens.set_as((tail * self.unit_ele) // self.block)
                    tail_left.set_as((tail * self.unit_ele) % self.block)
                    with tik_instance.if_scope(burst_lens > 0):
                        tik_instance.data_move(scores_idx_ub, self.workspace_score_idx[
                            offset + left_size - tail], 0, 1, burst_lens, 0, 0)
                    with tik_instance.for_range(0, tail_left) as _idx:
                        scores_idx_ub[burst_lens * self.block + _idx].set_as(
                            self.workspace_score_idx[offset + left_size - tail_left + _idx])

                    tik_instance.data_move(scores_idx_ub[score_idx_lens], dst, 0, 1, burst_lens_base, 0, 0)
                    self.init_tensor(dst, score_idx_lens * 2, 0)
                    cur_sort_score_idx(tik_instance, scores_idx_ub, dst, score_idx_lens * 2, self.unit_ele)

            with tik_instance.else_scope():
                self.init_tensor(scores_idx_ub, score_idx_lens * 2)
                burst_lens.set_as(left_size * self.unit_ele // self.block)
                tail_left.set_as(left_size * self.unit_ele % self.block)
                with tik_instance.if_scope(burst_lens > 0):
                    tik_instance.data_move(scores_idx_ub, self.workspace_score_idx[offset], 0, 1,
                                           burst_lens, 0, 0)
                with tik_instance.for_range(0, tail_left) as _idx:
                    scores_idx_ub[burst_lens * self.block + _idx].set_as(
                        self.workspace_score_idx[offset + left_size - tail_left + _idx])
                cur_sort_score_idx(tik_instance, scores_idx_ub, dst, score_idx_lens, self.unit_ele)

    def init_tensor(self, src, size=Constant.PER_LOOP_UNIT, init_value=0):
        """
        initialize the input tensor, set as init value

        Parameters
        ----------
        src: tensor
            target tensor in ub
        size: int
            data size, default as 4096
        init_value: int
            initial value

        Returns
        -------
        None
        """
        vector_mask, rep_stride = get_mask_rep_stride(src)

        max_lens = Constant.REPEAT_TIMES_MAX * vector_mask
        loop_num = size // max_lens
        tail = size % max_lens
        repeat_times = tail // vector_mask
        tail_aligned = tail % vector_mask

        tik_instance = self.tik_instance

        off = tik_instance.Scalar("uint32")
        with tik_instance.for_range(0, loop_num) as idx:
            off.set_as(vector_mask * Constant.REPEAT_TIMES_MAX * idx)
            tik_instance.vec_dup(vector_mask, src[off], init_value, Constant.REPEAT_TIMES_MAX, rep_stride)
        if tail != 0 and repeat_times > 0:
            offset = size - tail
            tik_instance.vec_dup(vector_mask, src[offset], init_value, repeat_times, rep_stride)
        if tail_aligned != 0:
            with tik_instance.for_range(0, tail_aligned) as i:
                src[size - tail_aligned + i].set_as(init_value)

    def init_tensor_scalar(self, src, size=Constant.PER_LOOP_UNIT, init_value=0):
        """
        initialize the input tensor, set as init value

        Parameters
        ----------
        src: tensor
            target tensor in ub
        size: int
            data size, default as 4096
        init_value: int
            initial value

        Returns
        -------
        None
        """
        vector_mask, rep_stride = get_mask_rep_stride(src)

        max_lens = Constant.REPEAT_TIMES_MAX * vector_mask
        loop_num = size // max_lens
        tail = size % max_lens
        repeat_times = tail // vector_mask
        tail_aligned = tail % vector_mask

        tik_instance = self.tik_instance

        off = tik_instance.Scalar("uint32")
        offset = tik_instance.Scalar("uint32")
        with tik_instance.for_range(0, loop_num) as idx:
            off.set_as(vector_mask * Constant.REPEAT_TIMES_MAX * idx)
            tik_instance.vec_dup(vector_mask, src[off], init_value, Constant.REPEAT_TIMES_MAX, rep_stride)
        with tik_instance.if_scope(tik.all(tail != 0, repeat_times > 0)):
            offset.set_as(size - tail)
            tik_instance.vec_dup(vector_mask, src[offset], init_value, repeat_times, rep_stride)
        with tik_instance.if_scope(tail_aligned != 0):
            with tik_instance.for_range(0, tail_aligned) as i:
                src[size - tail_aligned + i].set_as(init_value)

    def cal_params(self, batch_idx):
        tik_instance = self.tik_instance
        per_loop_ele = self.per_loop_unit
        score_idx_lens = per_loop_ele * self.unit_ele
        burst_len = score_idx_lens // self.block

        loop_num = tik_instance.Scalar('int32', name='loop_num')
        tail = tik_instance.Scalar("int32", name="tail")
        cur = tik_instance.Scalar("int32", name="cur")
        if self.need_valid_num:
            with self.tik_instance.new_stmt_scope():
                valid_num_ub = self.tik_instance.Tensor(
                    "int32", (Constant.BLOCK_SIZE,), name='valid_num_ub', scope=tik.scope_ubuf)
                self.tik_instance.data_move(valid_num_ub, self.input_gm_list[-1], 0, 1,
                                            (self.batch - 1) // self.tiling_each_block + 1, 0, 0)
                cur.set_as(valid_num_ub[batch_idx])
            with tik_instance.if_scope(self.boxes_num > cur):
                self.boxes_num.set_as(cur)
        loop_num.set_as(self.boxes_num // per_loop_ele)
        tail.set_as(self.boxes_num - loop_num * per_loop_ele)
        repeat_times = per_loop_ele // Constant.REPEAT_ELE
        res = [tik_instance, per_loop_ele, score_idx_lens, burst_len, loop_num, tail, repeat_times]
        return res

    def gen_score_index(self, batch_idx, class_idx, score_gm, scores_idx_out, input_boxes_ub_lis):
        """
        construct the tensor(score_index) for vsort32 and vmrgsort command
        get top 4096 scores and index, others stored in workspace

        Parameters
        ----------
        batch_idx : int
            batch index
        class_idx : int
            class index
        score_gm : tensor
            input scores data in gm
        scores_idx_out : tensor
            scores_idx_out

        Returns
        -------
        None
        """
        boxes_num = self.boxes_num
        tik_instance, per_loop_ele, score_idx_lens, burst_len, loop_num, tail, \
        repeat_times = self.cal_params(batch_idx)
        offset = tik_instance.Scalar("int32", name="offset")
        offset.set_as(batch_idx * self.classes * self.align_boxes + class_idx * self.align_boxes)
        # repeat_times for every vsort32 cycle
        self.init_tensor(scores_idx_out, score_idx_lens * 2, 0)
        input_boxes_gm = self.input_gm_list[0]
        if self.need_clip_window:
            self.read_window_compute(self.input_gm_list[2], [batch_idx, 0], self.clip_window_value_list)
        scalar_window = self.clip_window_value_list
        with tik_instance.new_stmt_scope():
            # define the tmp tensor, as 32 bytes aligned required
            index = tik_instance.Tensor("uint32", [per_loop_ele, ], name="idx_ub", scope=tik.scope_ubuf)
            init_index(tik_instance, self.idx_gm, index, 0, self.per_loop_unit)
            scores_ub = tik_instance.Tensor(self.boxes_type, [per_loop_ele, ], name="scores_ub", scope=tik.scope_ubuf)
            scores_idx_ub = tik_instance.Tensor(self.boxes_type, [score_idx_lens * 2, ],
                                                name="scores_idx_ub", scope=tik.scope_ubuf)

            with tik_instance.if_scope(loop_num > 0):
                # the first 4096 units
                burst_lens_base = per_loop_ele // self.block
                tik_instance.data_move(scores_ub, score_gm[batch_idx, class_idx, 0], 0, 1, burst_lens_base, 0, 0)
                if self.need_clip_window:
                    gm_offset = [batch_idx, class_idx, 0, 0]
                    self.clip_window_compute(tik_instance, input_boxes_gm, scores_ub,
                                             input_boxes_ub_lis, gm_offset, scalar_window, 
                                             boxes_num, copy_num=self.per_loop_unit, data_each_block=self.block)
                tik_instance.vsort32(scores_idx_out, scores_ub, index, repeat_times)

                with tik_instance.for_range(1, loop_num) as loop_idx:
                    # set value for index
                    init_index(tik_instance, self.idx_gm, index, loop_idx * per_loop_ele, self.per_loop_unit)

                    gm2ub_for_vsort32(tik_instance, score_gm, [batch_idx, class_idx, per_loop_ele * loop_idx],
                                      scores_ub, per_loop_ele)
                    if self.need_clip_window:
                        gm_offset = [batch_idx, class_idx, 0, loop_idx * per_loop_ele]
                        self.clip_window_compute(tik_instance, input_boxes_gm, scores_ub, input_boxes_ub_lis,
                                                 gm_offset, scalar_window, boxes_num, copy_num=self.per_loop_unit,
                                                 data_each_block=self.block)

                    tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)
                    tik_instance.data_move(scores_idx_ub[score_idx_lens], scores_idx_out, 0, 1, burst_len, 0, 0)
                    cur_sort_score_idx(tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens * 2, self.unit_ele)

                    # move last 4096 scores_index uints to workspace
                    tik_instance.data_move(self.workspace_score_idx[offset + score_idx_lens * (loop_idx - 1)],
                                           scores_idx_out[score_idx_lens], 0, 1, burst_len, 0, 0)

                with tik_instance.if_scope(tail > 0):
                    init_index(tik_instance, self.idx_gm, index, loop_num * per_loop_ele, self.per_loop_unit)
                    # init scores_ub & scores_idx_ub in order to clear the pre data
                    self.init_tensor(scores_ub, per_loop_ele, 0)
                    self.init_tensor(scores_idx_ub, score_idx_lens * 2)

                    gm2ub_for_vsort32(tik_instance, score_gm, [batch_idx, class_idx, self.boxes_num - tail], scores_ub,
                                      tail)
                    if self.need_clip_window:
                        gm_offset = [batch_idx, class_idx, 0, loop_num * per_loop_ele]
                        self.clip_window_compute(tik_instance, input_boxes_gm, scores_ub, input_boxes_ub_lis,
                                                 gm_offset, scalar_window, boxes_num, tail, data_each_block=self.block)

                    tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)
                    tik_instance.data_move(scores_idx_ub[score_idx_lens], scores_idx_out, 0, 1, burst_len, 0, 0)
                    self.init_tensor(scores_idx_out, score_idx_lens * 2, 0)
                    cur_sort_score_idx(tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens * 2, self.unit_ele)

                    # move last 4096 scores_idx in to workspace
                    tik_instance.data_move(self.workspace_score_idx[offset + score_idx_lens * (loop_num - 1)],
                                           scores_idx_out[score_idx_lens], 0, 1, burst_len, 0, 0)

            with tik_instance.else_scope():
                # init tensor
                self.init_tensor(scores_ub, per_loop_ele, 0)
                gm2ub_for_vsort32(tik_instance, score_gm, [batch_idx, class_idx, 0], scores_ub, tail)
                if self.need_clip_window:
                    gm_offset = [batch_idx, class_idx, 0, 0]
                    self.clip_window_compute(tik_instance, input_boxes_gm, scores_ub, input_boxes_ub_lis,
                                             gm_offset, scalar_window, tail, copy_num=self.per_loop_unit,
                                             data_each_block=self.block)
                tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)
                cur_sort_score_idx(tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens, self.unit_ele)

    def select_threshold(self, scores_index, eff_size, shape_size=Constant.PER_LOOP_UNIT, gate_value=0):
        """
        compute of index of effective scores based on the gate_value

        Parameters
        ----------
        scores_index : tensor
            scores_index in ub
        eff_size : scalar
            effective data size
        shape_size : int
            shape size of scores, must be 16 aligned
        gate_value : int
            threshold

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        mask_type = "uint16"
        mask_shape = (shape_size // Constant.B16_BIT,)
        direction = 4
        patten = Constant.PATTERN_TYPE

        if self.boxes_type == "float32":
            mask_type = "uint32"
            direction = 2
            patten = 1
            if shape_size == Constant.PER_LOOP_UNIT:
                shape_size = self.per_loop_unit
                mask_shape = (shape_size // Constant.B32_BIT,)

        if self.need_clip_window and shape_size == Constant.PER_LOOP_UNIT:
            shape_size = self.per_loop_unit
        shape = (shape_size,)

        with tik_instance.new_stmt_scope():
            scores_tmp = tik_instance.Tensor(self.boxes_type, shape,
                                             name="scores_tmp", scope=tik.scope_ubuf)
            scores_thresh = tik_instance.Tensor(self.boxes_type, shape,
                                                name="scores_thresh", scope=tik.scope_ubuf)
            # gen scores_thresh tensor
            value = gate_value
            if self.boxes_type == "float16":
                gate_value_fp16 = tik_instance.Scalar(dtype="float16")
                tik_instance.scalar_conv("", gate_value_fp16, gate_value)
                value = gate_value_fp16
            
            self.init_tensor_scalar(scores_thresh, shape_size, value)

            mask_uint = tik_instance.Tensor(mask_type, mask_shape, name="mask_uint",
                                            scope=tik.scope_ubuf)

            # move scores data from scores_index to scores_tmp
            mask, _ = get_mask_rep_stride(scores_thresh)
            repeat_times = shape_size * direction // mask
            tik_instance.vreducev2(None, scores_tmp, scores_index, patten, repeat_times, 1, 8, 0)

            # gen mask and then get the effective data lens
            self.gen_mask(scores_thresh, scores_tmp, mask_uint, shape_size)
            tik_instance.vreducev2(shape_size, scores_thresh, scores_tmp, mask_uint, 1, 1, 8, 1, rsvd_scalar=eff_size,
                                   mask_mode="counter")

    def read_window_compute(
            self,
            input_window_gm,
            offset,
            scalar_list):
        """
        read_window_compute
        """
        tik_instance = self.tik_instance
        with tik_instance.new_stmt_scope():
            input_window_ub = tik_instance.Tensor(input_window_gm.dtype, (self.block * 2,), name="input_window_ub",
                                                  scope=tik.scope_ubuf)
            tik_instance.data_move(input_window_ub, input_window_gm[offset],
                                   0, 1, 1, 0, 0)

            [index_win_y_min, index_win_x_min, index_win_y_max, index_win_x_max] = [scalar_list[0],
                                                                                    scalar_list[1], scalar_list[2],
                                                                                    scalar_list[3]]
            index_win_y_min.set_as(input_window_ub[0])
            index_win_x_min.set_as(input_window_ub[1])
            index_win_y_max.set_as(input_window_ub[2])
            index_win_x_max.set_as(input_window_ub[3])

            if self.change_coordinate_frame:
                [_, _, _, _, scale_h, scale_w] = scalar_list
                last_offset = offset[-1] + 2
                offset[-1] = last_offset
                tik_instance.data_move(input_window_ub[self.block],
                                       input_window_gm[offset],
                                       0, 1, 1, 0, 0)
                tik_func_packaged(tik_instance, "vsub", input_window_ub, input_window_ub[self.block],
                    input_window_ub, self.block)
                # do rec in mini
                tik_instance.vrec(2, input_window_ub[self.block], input_window_ub, 1, 1, 1, 8, 8)
                tik_fuc_vrec_newton(tik_instance, input_window_ub[self.block], input_window_ub, 2, self.block)
                scale_h.set_as(input_window_ub[self.block])
                scale_w.set_as(input_window_ub[self.block + 1])

    def select_threshold_scalar(self, scores_index, eff_size, shape_size=Constant.PER_LOOP_UNIT, gate_value=0):
        """
        compute of index of effective scores based on the gate_value

        Parameters
        ----------
        scores_index : tensor
            scores_index in ub
        eff_size : scalar
            effective data size
        shape_size : int
            shape size of scores, must be 16 aligned
        gate_value : int
            threshold

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        mask_type = "uint16"
        mask_shape = (shape_size // Constant.B16_BIT,)
        direction = 4
        patten = Constant.PATTERN_TYPE

        if self.boxes_type == "float32":
            mask_type = "uint32"
            direction = 2
            patten = 1
            with tik_instance.if_scope(shape_size == Constant.PER_LOOP_UNIT):
                shape_size = self.per_loop_unit
                mask_shape = (shape_size // Constant.B32_BIT,)

        if self.need_clip_window:
            with tik_instance.if_scope(shape_size == Constant.PER_LOOP_UNIT):
                shape_size = self.per_loop_unit
        shape = (shape_size,)

        with tik_instance.new_stmt_scope():
            scores_tmp = tik_instance.Tensor(self.boxes_type, shape,
                                             name="scores_tmp", scope=tik.scope_ubuf)
            scores_thresh = tik_instance.Tensor(self.boxes_type, shape,
                                                name="scores_thresh", scope=tik.scope_ubuf)
            # gen scores_thresh tensor
            value = gate_value
            if self.boxes_type == "float16":
                gate_value_fp16 = tik_instance.Scalar(dtype="float16")
                tik_instance.scalar_conv("", gate_value_fp16, gate_value)
                value = gate_value_fp16
            
            self.init_tensor_scalar(scores_thresh, shape_size, value)

            mask_uint = tik_instance.Tensor(mask_type, mask_shape, name="mask_uint",
                                            scope=tik.scope_ubuf)

            # move scores data from scores_index to scores_tmp
            mask, _ = get_mask_rep_stride(scores_thresh)
            repeat_times = shape_size * direction // mask
            tik_instance.vreducev2(None, scores_tmp, scores_index, patten, repeat_times, 1, 8, 0)

            # gen mask and then get the effective data lens
            self.gen_mask_scalar(scores_thresh, scores_tmp, mask_uint, shape_size)
            tik_instance.vreducev2(shape_size, scores_thresh, scores_tmp, mask_uint, 1, 1, 8, 1, rsvd_scalar=eff_size,
                                   mask_mode="counter")
    
    def clip_window_compute(self, tik_instance, input_boxes_gm, input_scorces_ub,
                            input_boxes_ub_list, gm_offset, scalar_window, boxes_num,
                            copy_num=4096, data_each_block=16):
        """
        clip_window_compute
        """
        input_num_boxes = self.per_loop_unit

        dtype = input_scorces_ub.dtype
        with self.tik_instance.new_stmt_scope():
            win_y_min = tik_instance.Tensor(dtype, (data_each_block,), name="win_y_min", scope=tik.scope_ubuf)
            win_x_min = tik_instance.Tensor(dtype, (data_each_block,), name="win_x_min", scope=tik.scope_ubuf)
            win_y_max = tik_instance.Tensor(dtype, (data_each_block,), name="win_y_man", scope=tik.scope_ubuf)
            win_x_max = tik_instance.Tensor(dtype, (data_each_block,), name="win_x_max", scope=tik.scope_ubuf)
            zero_tensor = tik_instance.Tensor(dtype, (data_each_block,), name="zero", scope=tik.scope_ubuf)
            # min float16 value
            min_fp16 = tik_instance.Tensor(dtype, (data_each_block,), name="min_fp16", scope=tik.scope_ubuf)
            concst24 = tik_instance.Tensor(dtype, (data_each_block,), name="concst24", scope=tik.scope_ubuf)
            sub1 = tik_instance.Tensor(dtype, (input_num_boxes,), name="sub1", scope=tik.scope_ubuf)
            sub2 = tik_instance.Tensor(dtype, (input_num_boxes,), name="sub2", scope=tik.scope_ubuf)
            nburst = 1
            burse_len = tik_instance.Scalar(dtype='int32', name='burst_len')
            burse_len.set_as((copy_num - 1) // data_each_block + 1)
            class_offset = self.tik_instance.Scalar(dtype='int32')
            with tik_instance.if_scope(self.classes_box == 1):
                class_offset.set_as(0)
            with tik_instance.else_scope():
                class_offset.set_as(gm_offset[1])
            batch_offset = tik_instance.Scalar(dtype='int32', name='batch_offset')
            batch_offset.set_as(gm_offset[0] * self.classes_box * 4 * boxes_num + class_offset * 4 * boxes_num)
            for i in range(4):
                tik_instance.data_move(input_boxes_ub_list[i],
                                       input_boxes_gm[batch_offset + i * boxes_num +
                                                      gm_offset[3]], 0, nburst, burse_len, 0, 0)
            if scalar_window is None:
                # no need to do clip to window, return directly
                return
            index_win_y_min = scalar_window[0]
            index_win_x_min = scalar_window[1]
            index_win_y_max = scalar_window[2]
            index_win_x_max = scalar_window[3]
            tik_instance.vector_dup(data_each_block, win_y_min, index_win_y_min, 1, 1, 1)
            tik_instance.vector_dup(data_each_block, win_x_min, index_win_x_min, 1, 1, 1)
            tik_instance.vector_dup(data_each_block, win_y_max, index_win_y_max, 1, 1, 1)
            tik_instance.vector_dup(data_each_block, win_x_max, index_win_x_max, 1, 1, 1)

            y_min_input = input_boxes_ub_list[1]
            x_min_input = input_boxes_ub_list[0]
            y_max_input = input_boxes_ub_list[3]
            x_max_input = input_boxes_ub_list[2]
            y_min_out = input_boxes_ub_list[1]
            x_min_out = input_boxes_ub_list[0]
            y_max_out = input_boxes_ub_list[3]
            x_max_out = input_boxes_ub_list[2]

            tik_func_available(tik_instance, "vmin", y_min_out, y_min_input, win_y_max,
                               copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_available(tik_instance, "vmin", y_max_out, y_max_input, win_y_max,
                               copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_available(tik_instance, "vmin", x_min_out, x_min_input, win_x_max,
                               copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_available(tik_instance, "vmin", x_max_out, x_max_input, win_x_max,
                               copy_num, 1, 1, 0, 8, 8, 0)

            tik_func_available(tik_instance, "vmax", y_min_out, y_min_out, win_y_min,
                               copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_available(tik_instance, "vmax", y_max_out, y_max_out, win_y_min,
                               copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_available(tik_instance, "vmax", x_min_out, x_min_out, win_x_min,
                               copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_available(tik_instance, "vmax", x_max_out, x_max_out, win_x_min,
                               copy_num, 1, 1, 0, 8, 8, 0)

            tik_func_available(tik_instance, "vsub", sub1, y_max_out, y_min_out,
                               copy_num, 1, 1, 1, 8, 8, 8)
            tik_func_available(tik_instance, "vsub", sub2, x_max_out, x_min_out,
                               copy_num, 1, 1, 1, 8, 8, 8)
            tik_func_available(tik_instance, "vmul", sub1, sub1, sub2,
                               copy_num, 1, 1, 1, 8, 8, 8)

            tik_func_vector(tik_instance, zero_tensor, 0.0, data_each_block)
            tik_func_vector(tik_instance, min_fp16, Constant.MIN_SCALAR_FP16, data_each_block)
            tik_func_vector(tik_instance, concst24, Constant.TMP_SCALAR_FP16, data_each_block)

            tik_func_packaged(tik_instance, "vmin", sub1, sub1, min_fp16,
                              copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_packaged(tik_instance, "vmax", sub1, sub1, zero_tensor,
                              copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_packaged(tik_instance, "vmul", sub1, sub1, concst24,
                              copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_packaged(tik_instance, "vmul", sub1, sub1, concst24,
                              copy_num, 1, 1, 0, 8, 8, 0)

            # modify score = 0 when area <= 0
            tik_func_available(tik_instance, "vmul", input_scorces_ub,
                               input_scorces_ub, sub1,
                               copy_num, 1, 1, 1, 8, 8, 8)

    def clip_window_compute_final(
            self,
            tik_instance,
            input_scorces_ub,
            input_boxes_ub_list,
            scalar_window,
            copy_num=4096,
            data_each_block=16):
        """
        clip_window_compute_final
        """
        input_num_boxes = self.per_loop_unit

        dtype = input_scorces_ub.dtype
        with self.tik_instance.new_stmt_scope():
            win_y_min = tik_instance.Tensor(dtype, (data_each_block,), name="win_y_min", scope=tik.scope_ubuf)
            win_x_min = tik_instance.Tensor(dtype, (data_each_block,), name="win_x_min", scope=tik.scope_ubuf)
            win_y_max = tik_instance.Tensor(dtype, (data_each_block,), name="win_y_man", scope=tik.scope_ubuf)

            win_x_max = tik_instance.Tensor(dtype, (data_each_block,), name="win_x_max", scope=tik.scope_ubuf)
            zero_tensor = tik_instance.Tensor(dtype, (data_each_block,), name="zero", scope=tik.scope_ubuf)
            # min float16 value
            min_fp16 = tik_instance.Tensor(dtype, (data_each_block,), name="min_fp16", scope=tik.scope_ubuf)
            concst24 = tik_instance.Tensor(dtype, (data_each_block,), name="concst24", scope=tik.scope_ubuf)
            sub1 = tik_instance.Tensor(dtype, (input_num_boxes,), name="sub1", scope=tik.scope_ubuf)
            sub2 = tik_instance.Tensor(dtype, (input_num_boxes,), name="sub2", scope=tik.scope_ubuf)

            if scalar_window is None:
                # no need to do clip to window, return directly
                return
            index_win_y_min = scalar_window[0]
            index_win_x_min = scalar_window[1]
            index_win_y_max = scalar_window[2]
            index_win_x_max = scalar_window[3]
            tik_instance.vector_dup(data_each_block, win_y_min, index_win_y_min, 1, 1, 1)
            tik_instance.vector_dup(data_each_block, win_x_min, index_win_x_min, 1, 1, 1)
            tik_instance.vector_dup(data_each_block, win_y_max, index_win_y_max, 1, 1, 1)
            tik_instance.vector_dup(data_each_block, win_x_max, index_win_x_max, 1, 1, 1)

            y_min_input = input_boxes_ub_list[0]
            x_min_input = input_boxes_ub_list[1]
            y_max_input = input_boxes_ub_list[2]
            x_max_input = input_boxes_ub_list[3]

            tik_func_available(tik_instance, "vmin", y_min_input, y_min_input, win_y_max,
                               copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_available(tik_instance, "vmin", y_max_input, y_max_input, win_y_max,
                               copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_available(tik_instance, "vmin", x_min_input, x_min_input, win_x_max,
                               copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_available(tik_instance, "vmin", x_max_input, x_max_input, win_x_max,
                               copy_num, 1, 1, 0, 8, 8, 0)

            tik_func_available(tik_instance, "vmax", y_min_input, y_min_input, win_y_min,
                               copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_available(tik_instance, "vmax", y_max_input, y_max_input, win_y_min,
                               copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_available(tik_instance, "vmax", x_min_input, x_min_input, win_x_min,
                               copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_available(tik_instance, "vmax", x_max_input, x_max_input, win_x_min,
                               copy_num, 1, 1, 0, 8, 8, 0)

            tik_func_available(tik_instance, "vsub", sub1, y_max_input, y_min_input,
                               copy_num, 1, 1, 1, 8, 8, 8)
            tik_func_available(tik_instance, "vsub", sub2, x_max_input, x_min_input,
                               copy_num, 1, 1, 1, 8, 8, 8)
            tik_func_available(tik_instance, "vmul", sub1, sub1, sub2,
                               copy_num, 1, 1, 1, 8, 8, 8)

            tik_func_vector(tik_instance, zero_tensor, 0.0, data_each_block)
            tik_func_vector(tik_instance, min_fp16, Constant.MIN_SCALAR_FP16, data_each_block)
            tik_func_vector(tik_instance, concst24, Constant.TMP_SCALAR_FP16, data_each_block)

            tik_func_packaged(tik_instance, "vmin", sub1, sub1, min_fp16,
                              copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_packaged(tik_instance, "vmax", sub1, sub1, zero_tensor,
                              copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_packaged(tik_instance, "vmul", sub1, sub1, concst24,
                              copy_num, 1, 1, 0, 8, 8, 0)
            tik_func_packaged(tik_instance, "vmul", sub1, sub1, concst24,
                              copy_num, 1, 1, 0, 8, 8, 0)

            # modify score = 0 when area <= 0
            tik_func_available(tik_instance, "vmul", input_scorces_ub,
                               input_scorces_ub, sub1,
                               copy_num, 1, 1, 1, 8, 8, 8)

    def get_boxes_after_score_thresh(self, xx1, xx2, yy1, yy2, scores_ub, batch_idx, class_idx, boxes,
                                     scores_index, size=4096, offset=0):
        """
        move boxes_gm to boxes_ub according to the sorting index

        Parameters
        ----------
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        scores_ub : tensor
            scores data in ub
        batch_idx : int
            batch index
        class_idx : int
            class index
        boxes : tensor
            input location data in gm
        scores_index : tensor
            scores_index in ub
        size : int/Scalar
            valid num default as 4096
        offset : int/Scalar
            pre valid boxes num default as 0
        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        idx_aligned = self.per_loop_unit
        with tik_instance.if_scope(offset == 0):
            self.init_tensor(xx1, idx_aligned)
            self.init_tensor(xx2, idx_aligned)
            self.init_tensor(yy1, idx_aligned)
            self.init_tensor(yy2, idx_aligned)
            self.init_tensor(scores_ub, idx_aligned, 0)

        with tik_instance.if_scope(tik.all(offset > 0, size + offset > self.per_loop_unit)):
            size.set_as(self.per_loop_unit - offset)

        # define the location index, the distance from the begin of class_begin
        loc_index = tik_instance.Scalar("uint32")
        # boxes is set as [4 * 4096], then the x1/x2/y1/y2 is stored in [0/1/2/3, 4096]
        with tik_instance.for_range(0, size) as idx:
            scores_index_offset = idx * self.unit_ele
            if self.boxes_type == "float16":
                loc_index.set_as(
                    scores_index[scores_index_offset + 2:scores_index_offset + 4].reinterpret_cast_to("uint32"))
            elif self.boxes_type == "float32":
                loc_index.set_as(
                    scores_index[scores_index_offset + 1].reinterpret_cast_to("uint32"))
            xx1[offset + idx].set_as(boxes[batch_idx, class_idx, 0, loc_index])
            yy1[offset + idx].set_as(boxes[batch_idx, class_idx, 1, loc_index])
            xx2[offset + idx].set_as(boxes[batch_idx, class_idx, 2, loc_index])
            yy2[offset + idx].set_as(boxes[batch_idx, class_idx, 3, loc_index])
            scores_ub[offset + idx].set_as(scores_index[scores_index_offset])

    def iou_selection(self, xx1, xx2, yy1, yy2, scores, eff_lens):
        """
        calculate the overlap of multi boxes, sieve out target boxes with  iou_thresh

        Parameters
        ----------
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        scores : tensor
            scores data in ub
        eff_lens : Scalar
            effect data lens
        pre_lens : Scalar
            pre cycle effective data lens

        Returns
        -------
        eff_lens : int
            valid boxes num
        """
        tik_instance = self.tik_instance
        shape_aligned = self.per_loop_unit

        with tik_instance.new_stmt_scope():
            single_area = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="single_area",
                                              scope=tik.scope_ubuf)
            iou = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="iou",
                                      scope=tik.scope_ubuf)
            mask_shape_lens = self.per_loop_unit // Constant.B16_BIT
            mask_type = "uint16"
            if self.boxes_type == "float32":
                mask_type = "uint32"
                mask_shape_lens = self.per_loop_unit // Constant.B32_BIT

            mask_uint = tik_instance.Tensor(mask_type, [mask_shape_lens, ], name="mask_uint16",
                                            scope=tik.scope_ubuf)

            self.init_tensor(iou, self.per_loop_unit)
            self.init_tensor(mask_uint, mask_shape_lens)

            # get area of every window
            self.get_rectangle_area(xx1, xx2, yy1, yy2, single_area)

            # calculate the iou, end up when the output windows is more than max_size_per_class
            overlap = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="overlap",
                                          scope=tik.scope_ubuf)
            # define tmp tensor for following use, to reduce the cycle of apply/release memory
            tmp1 = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="tmp1",
                                       scope=tik.scope_ubuf)
            tmp2 = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="tmp2",
                                       scope=tik.scope_ubuf)

            with tik_instance.for_range(0, self.max_size_per_class) as idx:
                with tik_instance.if_scope(idx < eff_lens):
                    # get overlap of windows_idx and the followings
                    self.get_overlap(xx1, xx2, yy1, yy2, overlap, tmp1, tmp2, idx)
                    # get overlap of windows_idx and the followings
                    self.cal_iou(single_area, iou, tmp2, idx, self.per_loop_unit)
                    self.gen_mask(overlap, iou, mask_uint, self.per_loop_unit)
                    self.update_input(xx1, xx2, yy1, yy2, scores, single_area, eff_lens, tmp1, tmp2, mask_uint)

    def update_input(self, xx1, xx2, yy1, yy2, scores, single_area, size, tmp1, tmp2, cmpmask_ub):
        """
        update the location and scores according to cmpmask

        Parameters
        ----------
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        scores : tensor
            scores data in ub
        single_area : tensor
            boxes area data in ub
        size : scalar
            data size
        cmpmask_ub : tensor
            mask pattern

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        mask = self.per_loop_unit
        burst_lens = self.per_loop_unit // self.block
        self.init_tensor(tmp1, self.per_loop_unit)
        self.init_tensor(tmp2, self.per_loop_unit)

        tik_instance.vreducev2(mask, tmp1, xx1, cmpmask_ub, 1, 1, 8, 1, rsvd_scalar=size, mask_mode="counter")
        tik_instance.data_move(xx1, tmp1, 0, 1, burst_lens, 0, 0)

        tik_instance.vreducev2(mask, tmp2, xx2, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
        tik_instance.data_move(xx2, tmp2, 0, 1, burst_lens, 0, 0)

        tik_instance.vreducev2(mask, tmp1, yy1, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
        tik_instance.data_move(yy1, tmp1, 0, 1, burst_lens, 0, 0)

        tik_instance.vreducev2(mask, tmp2, yy2, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
        tik_instance.data_move(yy2, tmp2, 0, 1, burst_lens, 0, 0)

        self.init_tensor(tmp1, mask, 0)
        tik_instance.vreducev2(mask, tmp1, scores, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
        tik_instance.data_move(scores, tmp1, 0, 1, burst_lens, 0, 0)

        tik_instance.vreducev2(mask, tmp2, single_area, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
        tik_instance.data_move(single_area, tmp2, 0, 1, burst_lens, 0, 0)

    def get_rectangle_area(self, xx1, xx2, yy1, yy2, dst):
        """
        area = (x2-x1) * (y2-y1), this is vector computing
        area can be reused in loops

        Parameters
        ----------
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        dst : tensor
            rectangle_area data in ub

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        shape_aligned = self.per_loop_unit

        with tik_instance.new_stmt_scope():
            tmp1 = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="tmp1", scope=tik.scope_ubuf)
            tmp2 = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="tmp2", scope=tik.scope_ubuf)

            mask, _ = get_mask_rep_stride(xx2)
            repeat_times = shape_aligned // mask

            tik_instance.vsub(mask, tmp1, xx2, xx1, repeat_times, 1, 1, 1, 8, 8, 8)
            tik_instance.vsub(mask, tmp2, yy2, yy1, repeat_times, 1, 1, 1, 8, 8, 8)

            tik_instance.vmul(mask, dst, tmp1, tmp2, repeat_times, 1, 1, 1, 8, 8, 8)

    def get_overlap(self, xx1, xx2, yy1, yy2, overlap, tmp1, yyy1, offset):
        """
        get overlap area of x1 and the following others, the pre units mask the overlap 0

        Parameters
        ----------
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        overlap : tensor
            overlap data in ub
        tmp1 : tensor
            tmp tensor
        yyy1 : tensor
            tmp tensor
        offset : scalar
            location index

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        shape_aligned = self.per_loop_unit

        with tik_instance.new_stmt_scope():
            tmp = tik_instance.Tensor(self.boxes_type, [shape_aligned, ], name="tmp", scope=tik.scope_ubuf)

            x1 = tik_instance.Scalar(self.boxes_type, init_value=xx1[offset])
            x2 = tik_instance.Scalar(self.boxes_type, init_value=xx2[offset])
            y1 = tik_instance.Scalar(self.boxes_type, init_value=yy1[offset])
            y2 = tik_instance.Scalar(self.boxes_type, init_value=yy2[offset])

            # `tmp = max(xx1[i], xx1[1:]), overlap=min(xx2[i], xx2[1:])
            tik_func_vmaxs(tik_instance, tmp, xx1, x1, shape_aligned, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)
            tik_func_vmins(tik_instance, overlap, xx2, x2, shape_aligned, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)

            mask, _ = get_mask_rep_stride(xx1)
            repeat_times = shape_aligned // mask
            # `w = max(0, xx2-xx1+offset), offset=1 here, sorted in tmp`
            tik_instance.vsub(mask, tmp, overlap, tmp, repeat_times, 1, 1, 1, 8, 8, 8)

            tik_func_vmaxs(tik_instance, tmp1, tmp, 0, shape_aligned, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)
            # `yyy1 = max(yy1[i], yy1[1:]), overlap = min(yy2[i], yy2[1:])`
            tik_func_vmaxs(tik_instance, yyy1, yy1, y1, shape_aligned, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)
            tik_func_vmins(tik_instance, overlap, yy2, y2, shape_aligned, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)

            # `h = max(0, yy2 - yy1 + offset), offset=1 here, sorted in yyy1`
            tik_instance.vsub(mask, yyy1, overlap, yyy1, repeat_times, 1, 1, 1, 8, 8, 8)

            tik_func_vmaxs(tik_instance, tmp, yyy1, 0, shape_aligned, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)
            tik_instance.vmul(mask, overlap, tmp1, tmp, repeat_times, 1, 1, 1, 8, 8, 8)

            # the overlap of the fixed boxes and itself default as 0
            overlap[offset].set_as(0)

    def cal_iou(self, src0, dst, tmp, offset, size=Constant.PER_LOOP_UNIT):
        """
        to calculate the related areas based on iou_thresh

        Parameters
        ----------
        src0 : tensor
            area of every window
        dst : tensor
            iou data
        tmp : tensor
            tmp tensor in ub
        offset : int
            the start window offset from the beginning
        size : int
            valid num

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        # calculate the sum of area1 and area2
        area1 = tik_instance.Scalar(self.boxes_type, init_value=src0[offset])

        mask, _ = get_mask_rep_stride(src0)
        repeat_times = size // mask
        tik_instance.vadds(mask, tmp, src0, area1, repeat_times, 1, 1, 8, 8)
        value = self.iou_thresh
        if self.boxes_type == "float16":
            iou_thresh_fp16 = tik_instance.Scalar(dtype="float16")
            tik_instance.scalar_conv("", iou_thresh_fp16, self.iou_thresh)
            value = iou_thresh_fp16
        tik_instance.vmuls(mask, dst, tmp, value, repeat_times, 1, 1, 8, 8)

    def gen_mask(self, overlap, iou, mask, size=Constant.PER_LOOP_UNIT):
        """
        gen mask

        Parameters
        ----------
        overlap : tensor
            overlap data in ub
        iou : tensor
            iou data in ub
        mask : tensor
            mask tensor
        size: total size of proposals

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        vector_mask, _ = get_mask_rep_stride(overlap)
        per_loop_num = Constant.REPEAT_TIMES_MAX * vector_mask
        loops = size // per_loop_num
        offset = tik_instance.Scalar("int32", init_value=0)

        # step1: max. mask * max. repeat  * loops times
        if loops > 0:
            with tik_instance.for_range(0, loops) as idx:
                # vec_cmpv_lt deal with 255 * 128 fp16 elements once
                tik_instance.vec_cmpv_lt(mask[offset],
                                         overlap[offset],
                                         iou[offset],
                                         Constant.REPEAT_TIMES_MAX,
                                         8, 8)
                offset.set_as(per_loop_num * (idx + 1))

        # step2: last num
        repeat_times = (size % per_loop_num) // vector_mask
        if repeat_times > 0:
            tik_instance.vec_cmpv_lt(mask[offset],
                                     overlap[offset],
                                     iou[offset],
                                     repeat_times,
                                     8, 8)

    def store_data(self, xx1, xx2, yy1, yy2, scores, batch_idx, class_idx, offset=0):
        """
        sort data in workspace for batch sorting

        Parameters
        ----------
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        scores : tensor
            scores data in ub
        batch_idx : int
            batch index
        class_idx : int
            class index
        offset : int
            offset from the beginning

        Returns
        -------
        None
        """

        tik_instance = self.tik_instance
        boxes_offset = tik_instance.Scalar("int32", name="boxes_offset")
        scores_offset = tik_instance.Scalar("int32", name="scores_offset")
        uint = ceil_div(self.max_size_per_class, self.block) * self.block
        boxes_offset.set_as(batch_idx * self.classes * 4 * uint +
                            class_idx * 4 * uint)
        scores_offset.set_as(batch_idx * self.classes * 1 * self.max_size_per_class +
                             class_idx * 1 * self.max_size_per_class)
        size = self.max_size_per_class
        burst_lens = size // self.block
        tail = size % self.block
        # move ub data to workspace
        with tik_instance.if_scope(burst_lens > 0):
            tik_instance.data_move(self.workspace_boxes[boxes_offset + offset], xx1, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(self.workspace_boxes[boxes_offset + 1 * uint + offset],
                                   yy1, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(self.workspace_boxes[boxes_offset + 2 * uint + offset],
                                   xx2, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(self.workspace_boxes[boxes_offset + 3 * uint + offset],
                                   yy2, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(self.workspace_scores[scores_offset + offset], scores, 0, 1, burst_lens, 0, 0)

        with tik_instance.if_scope(tail > 0):
            block_ub0 = tik_instance.Tensor(self.boxes_type, (self.block,), name="block_ub0", scope=tik.scope_ubuf)
            block_ub1 = tik_instance.Tensor(self.boxes_type, (self.block,), name="block_ub1", scope=tik.scope_ubuf)

            remain = tik_instance.Scalar(dtype="int32")
            with tik_instance.if_scope(size - self.block > 0):
                remain.set_as(size - self.block)
            with tik_instance.else_scope():
                remain.set_as(0)
            last_offset = offset + remain
            with tik_instance.for_range(0, self.block) as idx0:
                block_ub0[idx0].set_as(xx1[last_offset + idx0])
                block_ub1[idx0].set_as(yy1[last_offset + idx0])
            tik_instance.data_move(self.workspace_boxes[boxes_offset + last_offset], block_ub0, 0, 1, 1, 0, 0)
            tik_instance.data_move(self.workspace_boxes[boxes_offset + 1 * uint + last_offset],
                                   block_ub1, 0, 1, 1, 0, 0)
            with tik_instance.for_range(0, self.block) as idx2:
                block_ub0[idx2].set_as(xx2[last_offset + idx2])
                block_ub1[idx2].set_as(yy2[last_offset + idx2])
            tik_instance.data_move(self.workspace_boxes[boxes_offset + 2 * uint + last_offset],
                                   block_ub0, 0, 1, 1, 0, 0)
            tik_instance.data_move(self.workspace_boxes[boxes_offset + 3 * uint + last_offset],
                                   block_ub1, 0, 1, 1, 0, 0)
            with tik_instance.for_range(0, self.block) as idx3:
                block_ub0[idx3].set_as(scores[last_offset + idx3])
            tik_instance.data_move(self.workspace_scores[scores_offset + last_offset], block_ub0, 0, 1, 1, 0, 0)

    def gen_mask_scalar(self, overlap, iou, mask, size=Constant.PER_LOOP_UNIT):
        """
        gen mask

        Parameters
        ----------
        overlap : tensor
            overlap data in ub
        iou : tensor
            iou data in ub
        mask : tensor
            mask tensor
        size: total size of proposals

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        vector_mask, _ = get_mask_rep_stride(overlap)
        per_loop_num = Constant.REPEAT_TIMES_MAX * vector_mask
        loops = size // per_loop_num
        offset = tik_instance.Scalar("int32", init_value=0)

        # step1: max. mask * max. repeat  * loops times
        with tik_instance.if_scope(loops > 0):
            with tik_instance.for_range(0, loops) as idx:
                # vec_cmpv_lt deal with 255 * 128 fp16 elements once
                tik_instance.vec_cmpv_lt(mask[offset],
                                         overlap[offset],
                                         iou[offset],
                                         Constant.REPEAT_TIMES_MAX,
                                         8, 8)
                offset.set_as(per_loop_num * (idx + 1))

        # step2: last num
        repeat_times = (size % per_loop_num) // vector_mask
        with tik_instance.if_scope(repeat_times > 0):
            tik_instance.vec_cmpv_lt(mask[offset],
                                     overlap[offset],
                                     iou[offset],
                                     repeat_times,
                                     8, 8)
    
    def get_batch_scores_idx(self, src, real_batch_idx):
        """
        sort data in workspace for batch sorting

        Parameters
        ----------
        src : tensor
            scores data in workspace
        real_batch_idx : int
            batch index

        Returns
        -------
        score_idx_out : tensor
            score_index in ub
        """
        tik_instance = self.tik_instance
        mask, _ = get_mask_rep_stride(src)
        size = Constant.BATCH_LOOP_UNIT
        offset = self.tik_instance.Scalar("int32", name="offset")

        size_out = size * self.unit_ele
        score_idx_out = tik_instance.Tensor(self.boxes_type, [size_out * self.unit_ele * 2, ], name="score_idx_out",
                                            scope=tik.scope_ubuf)

        # when the class_num * max_size_per_class is less than 4096, the data can be processed in one loop in ub
        # otherwise the sorting should be cycled
        with tik_instance.if_scope(self.classes * self.max_size_per_class < size):
            with tik_instance.new_stmt_scope():
                index = tik_instance.Tensor("uint32", [size, ], name="index", scope=tik.scope_ubuf)
                score_tmp = tik_instance.Tensor(self.boxes_type, [size, ], name="score_tmp", scope=tik.scope_ubuf)
                score_idx_sort = tik_instance.Tensor(self.boxes_type, [size * self.unit_ele, ], name="score_idx_sort",
                                                     scope=tik.scope_ubuf)

                init_index(tik_instance, self.idx_gm, index, 0, size)
                self.init_tensor(score_tmp, size, 0)
                self.init_tensor(score_idx_sort, size * self.unit_ele, 0)
                self.init_tensor(score_idx_out, size_out * self.unit_ele * 2, 0)

                # move scores in workspcae to score_tmp
                burst_lens = self.max_size_per_class // self.block
                tail = self.max_size_per_class % self.block
                uint_lens = ceil_div(self.max_size_per_class, self.block) * self.block
                with tik_instance.for_range(0, self.classes) as i:
                    offset.set_as(real_batch_idx * self.classes * self.max_size_per_class * 1 + i *
                                  self.max_size_per_class * 1)
                    with tik_instance.if_scope(burst_lens > 0):
                        tik_instance.data_move(score_tmp[uint_lens * i], src[offset], 0, 1,
                                               burst_lens, 0, 0)
                    with tik_instance.for_range(0, tail) as _idx:
                        score_tmp[uint_lens * i + self.max_size_per_class - tail + _idx].set_as(
                            src[offset + self.max_size_per_class - tail + _idx])
                repeat_times = size // Constant.REPEAT_ELE
                tik_instance.vsort32(score_idx_sort, score_tmp, index, repeat_times)
                do_lens = size * self.unit_ele
                cur_sort_score_idx(tik_instance, score_idx_sort, score_idx_out, do_lens, self.unit_ele)

        with tik_instance.else_scope():
            self.gen_batch_score_index(real_batch_idx, src, score_idx_out)

        return score_idx_out

    def gen_batch_score_index(self, batch_idx, src, scores_idx_out):
        """
        construct the tensor(score_index) for vsort32 and vmrgsort command
        get top max_total_size scores and index

        Parameters
        ----------
        batch_idx : int
            batch index
        src : tensor
            input scores data in workspace
        scores_idx_out : tensor
            scores_idx_out

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        mask, _ = get_mask_rep_stride(src)
        uint_lens = ceil_div(self.max_size_per_class, self.block) * self.block
        # to accelerate the process, every cycle deal with data of 4 classes
        per_loop_ele = Constant.BATCH_LOOP_UNIT * Constant.UNIT_ELE
        score_idx_lens = per_loop_ele * self.unit_ele
        loop_num = tik_instance.Scalar("int32", name="loop_num")
        loop_num.set_as((self.classes - 1) // Constant.UNIT_ELE + 1)

        with tik_instance.new_stmt_scope():
            index = tik_instance.Tensor("uint32", [per_loop_ele, ], name="index", scope=tik.scope_ubuf)
            init_index(tik_instance, self.idx_gm, index, 0, uint_lens * 4)

            scores_ub = tik_instance.Tensor(self.boxes_type, [per_loop_ele, ], name="scores_ub", scope=tik.scope_ubuf)
            scores_idx_ub = tik_instance.Tensor(self.boxes_type, [score_idx_lens * 2, ],
                                                name="scores_idx_ub", scope=tik.scope_ubuf)
            self.init_tensor(scores_ub, per_loop_ele, 0)
            self.init_tensor(scores_idx_ub, score_idx_lens * 2, 0)

            repeat_times = ceil_div(per_loop_ele, Constant.REPEAT_ELE)

            with tik_instance.if_scope(loop_num > 1):
                burst_lens = score_idx_lens // self.block
                # the first part units
                with tik_instance.for_range(0, Constant.UNIT_ELE) as class_idx:
                    self.gm2ub_for_vsort32_workspace(src, [batch_idx, class_idx, 0], scores_ub[uint_lens * class_idx],
                                                     self.max_size_per_class)

                tik_instance.vsort32(scores_idx_out, scores_ub, index, repeat_times)

                with tik_instance.for_range(1, loop_num) as loop_idx:
                    # set value for index
                    with tik_instance.if_scope(loop_idx == loop_num - 1):
                        # init scores_ub & scores_idx_ub in order to clear the pre data
                        self.init_tensor(scores_ub, per_loop_ele, 0)

                    init_index(tik_instance, self.idx_gm, index, loop_idx * uint_lens * 4, uint_lens * 4)
                    self.init_tensor(scores_ub, per_loop_ele, 0)
                    with tik_instance.for_range(0, Constant.UNIT_ELE) as class_idx:
                        real_class_idx = loop_idx * Constant.UNIT_ELE + class_idx
                        with tik_instance.if_scope(real_class_idx < self.classes):
                            self.gm2ub_for_vsort32_workspace(src, [batch_idx, real_class_idx, 0],
                                                             scores_ub[uint_lens * class_idx], self.max_size_per_class)
                    tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)
                    tik_instance.data_move(scores_idx_ub[score_idx_lens], scores_idx_out, 0, 1, burst_lens, 0, 0)
                    cur_sort_score_idx(tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens * 2, self.unit_ele)

            with tik_instance.else_scope():
                with tik_instance.for_range(0, Constant.UNIT_ELE) as class_idx:
                    self.gm2ub_for_vsort32_workspace(src, [batch_idx, class_idx, 0], scores_ub[uint_lens * class_idx],
                                                     self.max_size_per_class)
                tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)
                cur_sort_score_idx(tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens, self.unit_ele)

    def sort_single_class_per_batch(self, batch_idx, data_lens, xx1, xx2, yy1, yy2, scores_ub):
        """
        batch(single class) sorting then move data out

        Parameters
        ----------
        batch_idx : int
            batch index
        data_lens : Scalar
            valid detection boxes num
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        scores_ub : tensor
            scores data in ub

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        class_size = tik_instance.Scalar(dtype="int32")
        with tik_instance.if_scope(self.max_total_size > self.block):
            class_size.set_as(self.max_total_size)
        with tik_instance.else_scope():
            class_size.set_as(self.block)
        classes_out = tik_instance.Tensor(self.boxes_type, [class_size, ], name="classes_out", scope=tik.scope_ubuf)
        valid_detection = tik_instance.Tensor("int32", [Constant.BLOCK_ELE_B32, ], name="valid_detection",
                                              scope=tik.scope_ubuf)
        self.init_tensor_scalar(classes_out, class_size, 0)
        self.init_tensor(valid_detection, Constant.BLOCK_ELE_B32, 0)
        with tik_instance.for_range(0, data_lens) as idx:
            classes_out[idx].set_as(0)

        with tik_instance.if_scope(data_lens > self.max_total_size):
            data_lens.set_as(self.max_total_size)
        valid_detection[0].set_as(data_lens)

        self.move_data_out_continuous(xx1, xx2, yy1, yy2, scores_ub, classes_out, valid_detection, batch_idx)

    def init_ub_tensor_scalar(self, lis, date_lens):
        """
        init ub tensor with scalar
        """
        x1, x2, y1, y2, scores_ub, classes_out, classes_tmp, valid_detection = lis
        self.init_tensor_scalar(x1, date_lens, 0)
        self.init_tensor_scalar(x2, date_lens, 0)
        self.init_tensor_scalar(y1, date_lens, 0)
        self.init_tensor_scalar(y2, date_lens, 0)
        self.init_tensor_scalar(scores_ub, date_lens, 0)
        self.init_tensor_scalar(classes_out, date_lens, 0)
        self.init_tensor_scalar(classes_tmp, date_lens, 0)
        self.init_tensor(valid_detection, Constant.BLOCK_ELE_B32, 0)

    def sort_class_per_batch(self, batch_idx):
        """
        batch sorting then move data out

        Parameters
        ----------
        batch_idx : int
            batch index

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        mask, _ = get_mask_rep_stride(self.workspace_scores)

        date_lens = ceil_div(self.max_total_size, mask) * mask
        init_shape = (date_lens,)

        eff_size = tik_instance.Scalar("uint32")
        offset = tik_instance.Scalar("int32")

        x1 = tik_instance.Tensor(self.boxes_type, init_shape, name="x1", scope=tik.scope_ubuf)
        x2 = tik_instance.Tensor(self.boxes_type, init_shape, name="x2", scope=tik.scope_ubuf)
        y1 = tik_instance.Tensor(self.boxes_type, init_shape, name="y1", scope=tik.scope_ubuf)
        y2 = tik_instance.Tensor(self.boxes_type, init_shape, name="y2", scope=tik.scope_ubuf)
        scores_ub = tik_instance.Tensor(self.boxes_type, init_shape, name="scores_ub", scope=tik.scope_ubuf)
        classes_out = tik_instance.Tensor(self.boxes_type, init_shape, name="classes_out", scope=tik.scope_ubuf)
        classes_tmp = tik_instance.Tensor("uint32", init_shape, name="classes_tmp", scope=tik.scope_ubuf)
        valid_detection = tik_instance.Tensor("int32", [Constant.BLOCK_ELE_B32, ], name="valid_detection",
                                              scope=tik.scope_ubuf)

        # set default value to x1/x2/y1/y2/scores/class_out/valid_detection
        tensor_lis = [x1, x2, y1, y2, scores_ub, classes_out, classes_tmp, valid_detection]
        self.init_ub_tensor_scalar(tensor_lis, date_lens)

        scores_idx = self.get_batch_scores_idx(self.workspace_scores, batch_idx)
        self.select_threshold_scalar(scores_idx, eff_size, date_lens, self.score_thresh)

        with tik_instance.if_scope(eff_size > self.max_total_size):
            eff_size.set_as(self.max_total_size)

        valid_detection[0].set_as(eff_size)

        # define the location index, the distance from the begin of class_begin
        tmp = tik_instance.Scalar("uint32")
        lo_index = tik_instance.Scalar("uint32")
        class_idx = tik_instance.Scalar("uint32")

        uint_lens = ceil_div(self.max_size_per_class, self.block) * self.block
        with tik_instance.new_stmt_scope():
            with tik_instance.for_range(0, eff_size) as idx:
                scores_index_offset = idx * self.unit_ele
                if self.boxes_type == "float16":
                    tmp.set_as(
                        scores_idx[scores_index_offset + 2:scores_index_offset + 4].reinterpret_cast_to("uint32"))
                elif self.boxes_type == "float32":
                    tmp.set_as(
                        scores_idx[scores_index_offset + 1].reinterpret_cast_to("uint32"))
                class_idx.set_as(tmp // uint_lens)
                lo_index.set_as(tmp % uint_lens)
                offset.set_as(batch_idx * self.classes * 4 * uint_lens + class_idx *
                              4 * uint_lens)
                x1[idx].set_as(self.workspace_boxes[offset + lo_index])
                y1[idx].set_as(self.workspace_boxes[offset + 1 * uint_lens + lo_index])
                x2[idx].set_as(self.workspace_boxes[offset + 2 * uint_lens + lo_index])
                y2[idx].set_as(self.workspace_boxes[offset + 3 * uint_lens + lo_index])
                classes_tmp[idx].set_as(class_idx)
                scores_ub[idx].set_as(scores_idx[idx * self.unit_ele])

            # conv the dtype of classes_tmp to int32, then move to classes_out, meanwhile the dtype is changed to fp16
            data_b = classes_tmp.reinterpret_cast_to("int32")
            mask, _ = get_mask_rep_stride(data_b)
            repeat_times = date_lens // mask
            if self.boxes_type == "float16":
                scale = 1.0
                stride = 4
            elif self.boxes_type == "float32":
                scale = None
                stride = 8
            tik_instance.vec_conv(mask, "none", classes_out, data_b, repeat_times, stride, 8, deqscale=scale)
        self.move_data_out_continuous(x1, x2, y1, y2, scores_ub, classes_out, valid_detection, batch_idx)

    def move_data_out(self, x1, x2, y1, y2, scores_ub, classes_out, valid_detection, batch_idx):
        """
        sort data in workspace for batch sorting

        Parameters
        ----------
        x1 : tensor
            x1 data in ub
        x2 : tensor
            x2 data in ub
        y1 : tensor
            y1 data in ub
        y2 : tensor
            y2 data in ub
        scores_ub : tensor
            scores data in ub
        classes_out : tensor
            classes data in ub
        valid_detection : tensor
            valid_detection data in ub
        batch_idx : int
            batch_index

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        [nmsed_boxes_gm, nmsed_scores_gm, nmsed_classes_gm, nmsed_num_gm] = self.output_gm_list

        data_size = self.max_total_size
        burst_lens = data_size // self.block

        with tik_instance.if_scope(burst_lens > 0):
            tik_instance.data_move(nmsed_boxes_gm[batch_idx, 0, 0], x1, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_boxes_gm[batch_idx, 1, 0], y1, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_boxes_gm[batch_idx, 2, 0], x2, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_boxes_gm[batch_idx, 3, 0], y2, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_scores_gm[batch_idx, 0], scores_ub, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_classes_gm[batch_idx, 0], classes_out, 0, 1, burst_lens, 0, 0)

        block_ub0 = tik_instance.Tensor(self.boxes_type, (self.block,), name="block_ub0", scope=tik.scope_ubuf)
        block_ub1 = tik_instance.Tensor(self.boxes_type, (self.block,), name="block_ub1", scope=tik.scope_ubuf)

        # move tail data to gm
        last_offset = tik_instance.Scalar(dtype="int32")
        with tik_instance.if_scope(data_size - self.block > 0):
            last_offset.set_as(data_size - self.block)
        with tik_instance.else_scope():
            last_offset.set_as(0)
        with tik_instance.for_range(0, self.block) as idx0:
            block_ub0[idx0].set_as(x1[last_offset + idx0])
            block_ub1[idx0].set_as(y1[last_offset + idx0])
        tik_instance.data_move(nmsed_boxes_gm[batch_idx, 0, last_offset], block_ub0, 0, 1, 1, 0, 0)
        tik_instance.data_move(nmsed_boxes_gm[batch_idx, 1, last_offset], block_ub1, 0, 1, 1, 0, 0)
        with tik_instance.for_range(0, self.block) as idx1:
            block_ub0[idx1].set_as(x2[last_offset + idx1])
            block_ub1[idx1].set_as(y2[last_offset + idx1])
        tik_instance.data_move(nmsed_boxes_gm[batch_idx, 2, last_offset], block_ub0, 0, 1, 1, 0, 0)
        tik_instance.data_move(nmsed_boxes_gm[batch_idx, 3, last_offset], block_ub1, 0, 1, 1, 0, 0)

        with tik_instance.for_range(0, self.block) as idx2:
            block_ub0[idx2].set_as(scores_ub[last_offset + idx2])
            block_ub1[idx2].set_as(classes_out[last_offset + idx2])
        tik_instance.data_move(nmsed_scores_gm[batch_idx, last_offset], block_ub0, 0, 1, 1, 0, 0)
        tik_instance.data_move(nmsed_classes_gm[batch_idx, last_offset], block_ub1, 0, 1,
                               1, 0, 0)
        tik_instance.data_move(nmsed_num_gm[batch_idx, 0], valid_detection, 0, 1, 1, 0, 0)

    def move_data_out_continuous(self, x1, x2, y1, y2, scores_ub, classes_out, valid_detection, batch_idx):
        """
        sort data in workspace for batch sorting

        Parameters
        ----------
        x1 : tensor
            x1 data in ub
        x2 : tensor
            x2 data in ub
        y1 : tensor
            y1 data in ub
        y2 : tensor
            y2 data in ub
        scores_ub : tensor
            scores data in ub
        classes_out : tensor
            classes data in ub
        valid_detection : tensor
            valid_detection data in ub
        batch_idx : int
            batch_index

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        [nmsed_boxes_gm, nmsed_scores_gm, nmsed_classes_gm, nmsed_num_gm] = self.output_gm_list

        data_size = self.max_total_size
        vector_mask, _ = get_mask_rep_stride(scores_ub)
        burst_lens = data_size // self.block
        if self.down_flag:
            repeat_time = data_size // vector_mask
            remain = data_size % vector_mask
            with tik_instance.if_scope(repeat_time > 0):
                tik_instance.vmuls(vector_mask, x1, x1, 10, repeat_time, 1, 1, 8, 8)
                tik_instance.vmuls(vector_mask, y1, y1, 10, repeat_time, 1, 1, 8, 8)
                tik_instance.vmuls(vector_mask, x2, x2, 10, repeat_time, 1, 1, 8, 8)
                tik_instance.vmuls(vector_mask, y2, y2, 10, repeat_time, 1, 1, 8, 8)
            with tik_instance.if_scope(remain > 0):
                tik_instance.vmuls(remain, x1[vector_mask * repeat_time], x1[vector_mask * repeat_time],
                                   10, 1, 1, 1, 8, 8)
                tik_instance.vmuls(remain, y1[vector_mask * repeat_time], y1[vector_mask * repeat_time],
                                   10, 1, 1, 1, 8, 8)
                tik_instance.vmuls(remain, x2[vector_mask * repeat_time], x2[vector_mask * repeat_time],
                                   10, 1, 1, 1, 8, 8)
                tik_instance.vmuls(remain, y2[vector_mask * repeat_time], y2[vector_mask * repeat_time],
                                   10, 1, 1, 1, 8, 8)

        with tik_instance.if_scope(burst_lens > 0):
            tik_instance.data_move(nmsed_boxes_gm[batch_idx * 4 * self.max_total_size],
                                   x1, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_boxes_gm[batch_idx * 4 * self.max_total_size +
                                                  self.max_total_size], y1, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_boxes_gm[batch_idx * 4 * self.max_total_size + 2 *
                                                  self.max_total_size], x2, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_boxes_gm[batch_idx * 4 * self.max_total_size +
                                                  self.max_total_size * 3], y2, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_scores_gm[batch_idx * self.max_total_size], scores_ub, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_classes_gm[batch_idx * self.max_total_size],
                                   classes_out, 0, 1, burst_lens, 0, 0)

        block_ub0 = tik_instance.Tensor(self.boxes_type, (self.block,), name="block_ub0", scope=tik.scope_ubuf)
        block_ub1 = tik_instance.Tensor(self.boxes_type, (self.block,), name="block_ub1", scope=tik.scope_ubuf)

        # move tail data to gm
        last_offset = tik_instance.Scalar(dtype="int32")
        with tik_instance.if_scope(data_size - self.block > 0):
            last_offset.set_as(data_size - self.block)
        with tik_instance.else_scope():
            last_offset.set_as(0)
        with tik_instance.for_range(0, self.block) as idx0:
            block_ub0[idx0].set_as(x1[last_offset + idx0])
            block_ub1[idx0].set_as(y1[last_offset + idx0])
        tik_instance.data_move(nmsed_boxes_gm[batch_idx * 4 * self.max_total_size + last_offset],
                               block_ub0, 0, 1, 1, 0, 0)
        tik_instance.data_move(nmsed_boxes_gm[batch_idx * 4 * self.max_total_size + self.max_total_size +
                                              last_offset], block_ub1, 0, 1, 1, 0, 0)
        with tik_instance.for_range(0, self.block) as idx1:
            block_ub0[idx1].set_as(x2[last_offset + idx1])
            block_ub1[idx1].set_as(y2[last_offset + idx1])
        tik_instance.data_move(nmsed_boxes_gm[batch_idx * 4 * self.max_total_size + 2 * self.max_total_size +
                                              last_offset], block_ub0, 0, 1, 1, 0, 0)
        tik_instance.data_move(nmsed_boxes_gm[batch_idx * 4 * self.max_total_size + 3 * self.max_total_size +
                                              last_offset], block_ub1, 0, 1, 1, 0, 0)

        with tik_instance.for_range(0, self.block) as idx2:
            block_ub0[idx2].set_as(scores_ub[last_offset + idx2])
            block_ub1[idx2].set_as(classes_out[last_offset + idx2])
        tik_instance.data_move(nmsed_scores_gm[batch_idx * self.max_total_size + last_offset], block_ub0, 0, 1, 1, 0, 0)
        tik_instance.data_move(nmsed_classes_gm[batch_idx * self.max_total_size + last_offset], block_ub1,
                               0, 1, 1, 0, 0)
        tik_instance.data_move(nmsed_num_gm[batch_idx * 8], valid_detection, 0, 1, 1, 0, 0)
