#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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
cnms_yolov3_ver
"""
from impl.util.util_tik_comm_func import tik
from impl.util.util_tik_comm_func import tik_func_vmins
from impl.util.util_tik_comm_func import tik_func_vmaxs
from impl.util.util_tik_comm_func import sort_score_idx_by_desc
from impl.util.util_tik_comm_func import init_index
from impl.util.util_tik_comm_func import gm2ub_for_vsort32
from impl.util.util_common import get_mask_rep_stride
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
    # define repeat elements every time for vsrot32
    REPEAT_ELE = 32
    # location elements, [x1, y1, x2, y2]
    FOUR_DIRECTION = 4
    # b16 elements num of every block also uesed as b16 elements num of mask
    BLOCK_ELE = 16
    # b32 elements num of every block
    BLOCK_ELE_B32 = 8
    # the socres_index contains four elements also marked as the class num processed every cycle
    UNIT_ELE = 4

    PER_LOOP_UNIT = 4096

    REPEAT_TIMES_MAX = 255
    # 0b0001 0001 0001 0001 is equals to type 3
    PATTERN_TYPE = 3
    FP16_MINS = -65504

    # PIECES
    FOUR_PIECE = 4
    TWO_PIECE = 2


def process_nms_mode_v300(tik_instance, boxes_info, threshold_info, 
                          max_size_info, other_calc_info):
    """
    deal with class nms compute

    """
    box_type, boxes_num = boxes_info
    score_thresh, iou_thresh = threshold_info
    max_size_per_class, max_total_size = max_size_info
    idx_gm, boxes, scores, bbox_out_list = other_calc_info

    real_boxes_num = boxes_num

    # every loop process 4096 units
    pre_loop_unit = 4096
    # optimize the compute data lens
    if real_boxes_num <= pre_loop_unit // Constant.FOUR_PIECE:
        pre_loop_unit = pre_loop_unit // Constant.FOUR_PIECE
    elif real_boxes_num <= pre_loop_unit // Constant.TWO_PIECE:
        pre_loop_unit = pre_loop_unit // Constant.TWO_PIECE

    real_iou_thresh = iou_thresh / (1 + iou_thresh)

    eff_size = tik_instance.Scalar(dtype="uint32", name="eff_size")
    eff_lens = tik_instance.Scalar(dtype="uint32", name="eff_lens", init_value=pre_loop_unit)

    x1_ub = tik_instance.Tensor(box_type, [pre_loop_unit, ], name="x1_ub", scope=tik.scope_ubuf)
    x2_ub = tik_instance.Tensor(box_type, [pre_loop_unit, ], name="x2_ub", scope=tik.scope_ubuf)
    y1_ub = tik_instance.Tensor(box_type, [pre_loop_unit, ], name="y1_ub", scope=tik.scope_ubuf)
    y2_ub = tik_instance.Tensor(box_type, [pre_loop_unit, ], name="y2_ub", scope=tik.scope_ubuf)
    scores_ub = tik_instance.Tensor(box_type, [pre_loop_unit, ], name="scores_ub", scope=tik.scope_ubuf)
    proposal_data = [x1_ub, x2_ub, y1_ub, y2_ub, scores_ub]

    with tik_instance.new_stmt_scope():
        scores_idx_out = tik_instance.Tensor(box_type, [pre_loop_unit * Constant.UNIT_ELE * 2, ],
                                             name="scores_idx_out", scope=tik.scope_ubuf)
        gen_score_index_v300(tik_instance, [scores, scores_idx_out], real_boxes_num, idx_gm, pre_loop_unit)
        select_threshold_v300(tik_instance, scores_idx_out, eff_size, pre_loop_unit, gate_value=score_thresh)
        get_boxes_after_score_thresh_v300(tik_instance, proposal_data, [boxes, real_boxes_num,
                                          scores_idx_out, pre_loop_unit], eff_size)
        iou_selection_v300(tik_instance, proposal_data, [eff_lens, pre_loop_unit],
                           max_size_per_class, real_iou_thresh)
    with tik_instance.if_scope(
            tik.any(eff_lens >= max_size_per_class, pre_loop_unit >= boxes_num)):
        move_data_out_v300(tik_instance, proposal_data, eff_lens, 
                           bbox_out_list, [max_total_size, real_boxes_num])


def init_tensor(tik_instance, src, size, init_value=0):
    """
    initialize the input tensor, set as init value

    """
    vector_mask, rep_stride = get_mask_rep_stride(src)

    max_lens = Constant.REPEAT_TIMES_MAX * vector_mask
    loop_num = size // max_lens
    tail = size % max_lens
    repeat_times = tail // vector_mask
    tail_aligned = tail % vector_mask

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


def gen_mask(tik_instance, overlap, iou, mask, size=4096):
    """
    gen mask

    """
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

    # step3: last num
    repeat_times = (size % per_loop_num) // vector_mask
    if repeat_times > 0:
        tik_instance.vec_cmpv_lt(mask[offset],
                                 overlap[offset],
                                 iou[offset],
                                 repeat_times,
                                 8, 8)


def gm2ub_for_vsort32_v300(tik_instance, src, box_idx, dst, do_lens):
    """
    move data from gm to ub for get_tik_func_vsort32

    Parameters
    ----------
    tik_instance : tik instance
    src : tensor
        scores tensor in ub(3D)
    box_idx : box_idx
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
    with tik_instance.if_scope(burst_lens > 0):
        tik_instance.data_move(dst, src[box_idx], 0, 1, burst_lens, 0, 0)
    with tik_instance.for_range(0, tail_num) as idx:
        dst[burst_lens * block_element + idx].set_as(src[box_idx + idx])


def gen_score_index_v300(tik_instance, score_val_list, boxes_num, idx_gm, per_loop_ele):
    """
    construct the tensor(score_index) for vsort32 and vmrgsort command

    """
    score_gm, scores_idx_out = score_val_list

    score_idx_lens = per_loop_ele * Constant.UNIT_ELE
    burst_lens_idx = score_idx_lens // Constant.BLOCK_ELE

    loop_num = boxes_num // per_loop_ele
    if loop_num == 1:
        tail = boxes_num
    else:
        tail = boxes_num - loop_num * per_loop_ele
    # repeat_times for every vsort32 cycle
    repeat_times = per_loop_ele // Constant.REPEAT_ELE

    init_tensor(tik_instance, scores_idx_out, score_idx_lens * 2, Constant.FP16_MINS)
    with tik_instance.new_stmt_scope():
        # define the tmp tensor, as 32 bytes aligned required
        index = tik_instance.Tensor("uint32", [per_loop_ele, ], name="idx_ub", scope=tik.scope_ubuf)
        if per_loop_ele > Constant.PER_LOOP_UNIT:
            burst_lens = per_loop_ele // Constant.BLOCK_ELE_B32
            tik_instance.data_move(index, idx_gm, 0, 1, burst_lens, 0, 0)
        else:
            init_index(tik_instance, idx_gm, index, 0, per_loop_ele)
        scores_ub = tik_instance.Tensor("float16", [per_loop_ele, ], name="scores_ub", scope=tik.scope_ubuf)
        scores_idx_ub = tik_instance.Tensor("float16", [score_idx_lens * 2, ],
                                            name="scores_idx_ub", scope=tik.scope_ubuf)

        if loop_num > 1:
            # the first 4096 units
            burst_lens_base = per_loop_ele // Constant.BLOCK_ELE
            tik_instance.data_move(scores_ub, score_gm, 0, 1, burst_lens_base, 0, 0)
            tik_instance.vsort32(scores_idx_out, scores_ub, index, repeat_times)

            with tik_instance.for_range(1, loop_num) as loop_idx:
                # set value for index
                init_index(tik_instance, idx_gm, index, loop_idx * per_loop_ele, per_loop_ele)

                gm2ub_for_vsort32_v300(tik_instance, score_gm, per_loop_ele * loop_idx, scores_ub, per_loop_ele)

                tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)
                tik_instance.data_move(scores_idx_ub[score_idx_lens], scores_idx_out, 0, 1, burst_lens_idx, 0, 0)
                sort_score_idx_by_desc(tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens * 2)

            with tik_instance.if_scope(tail > 0):
                tail_block = (tail // Constant.BLOCK_ELE_B32) * Constant.BLOCK_ELE_B32
                init_index(tik_instance, idx_gm, index, loop_num * per_loop_ele, tail_block)
                # init scores_ub & scores_idx_ub in order to clear the pre data
                init_tensor(tik_instance, scores_ub, per_loop_ele, Constant.FP16_MINS)
                init_tensor(tik_instance, scores_idx_ub, score_idx_lens * 2)

                gm2ub_for_vsort32_v300(tik_instance, score_gm, boxes_num - tail, scores_ub, tail)

                tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)
                tik_instance.data_move(scores_idx_ub[score_idx_lens], scores_idx_out, 0, 1, burst_lens_idx, 0, 0)
                init_tensor(tik_instance, scores_idx_out, score_idx_lens * 2, Constant.FP16_MINS)
                sort_score_idx_by_desc(tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens * 2)

        else:
            # init tensor
            init_tensor(tik_instance, scores_ub, per_loop_ele, Constant.FP16_MINS)
            gm2ub_for_vsort32_v300(tik_instance, score_gm, 0, scores_ub, tail)
            tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)
            sort_score_idx_by_desc(tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens)


def select_threshold_v300(tik_instance, scores_index, eff_size, shape_size, gate_value=0):
    """
    compute of index of effective scores based on the gate_value

    """
    shape = (shape_size,)
    mask_shape = (shape_size // Constant.BLOCK_ELE,)

    if gate_value == 0:
        eff_size.set_as(shape_size)
    else:
        with tik_instance.new_stmt_scope():
            scores_tmp = tik_instance.Tensor("float16", shape, name="scores_tmp", scope=tik.scope_ubuf)
            scores_thresh = tik_instance.Tensor("float16", shape, name="scores_thresh", scope=tik.scope_ubuf)
            # gen scores_thresh tensor
            init_tensor(tik_instance, scores_thresh, shape_size, gate_value)

            mask_uint16 = tik_instance.Tensor("uint16", mask_shape, name="mask_int8", scope=tik.scope_ubuf)

            # move scores data from scores_index to scores_tmp
            mask, _ = get_mask_rep_stride(scores_thresh)
            repeat_times = shape_size * Constant.FOUR_DIRECTION // mask
            tik_instance.vreducev2(None, scores_tmp, scores_index, Constant.PATTERN_TYPE, repeat_times, 1, 8, 0)

            # gen mask and then get the effective data lens
            gen_mask(tik_instance, scores_thresh, scores_tmp, mask_uint16, shape_size)
            tik_instance.vreducev2(shape_size, scores_thresh, scores_tmp, mask_uint16, 1, 1, 8, 1,
                                   rsvd_scalar=eff_size, mask_mode="counter")


def exchange_coordinate(tik_instance, coord_info, pre_loop_unit):
    """
    exchange coordinate, and update the data in tensor_list
    release the temp dst

    """
    xx1, xx2, yy1, yy2 = coord_info

    with tik_instance.new_stmt_scope():
        tem_x0 = tik_instance.Tensor("float16", [pre_loop_unit, ], name="tem_x0", scope=tik.scope_ubuf)
        tem_x1 = tik_instance.Tensor("float16", [pre_loop_unit, ], name="tem_x1", scope=tik.scope_ubuf)

        mask, _ = get_mask_rep_stride(xx1)
        repeat_times = pre_loop_unit // mask
        burst_lens = pre_loop_unit // Constant.BLOCK_ELE

        tik_instance.vmin(mask, tem_x0, xx1, xx2, repeat_times, 1, 1, 1, 8, 8, 8)
        tik_instance.vmax(mask, tem_x1, xx1, xx2, repeat_times, 1, 1, 1, 8, 8, 8)

        # update x1 and x2
        tik_instance.data_move(xx1, tem_x0, 0, 1, burst_lens, 0, 0)
        tik_instance.data_move(xx2, tem_x1, 0, 1, burst_lens, 0, 0)

        tik_instance.vmin(mask, tem_x0, yy1, yy2, repeat_times, 1, 1, 1, 8, 8, 8)
        tik_instance.vmax(mask, tem_x1, yy1, yy2, repeat_times, 1, 1, 1, 8, 8, 8)

        # update x1 and x2
        tik_instance.data_move(yy1, tem_x0, 0, 1, burst_lens, 0, 0)
        tik_instance.data_move(yy2, tem_x1, 0, 1, burst_lens, 0, 0)


def get_boxes_after_score_thresh_v300(tik_instance, proposal_info, calc_info, size=4096, offset=0):
    """
    move boxes_gm to boxes_ub according to the sorting index

    """
    xx1, xx2, yy1, yy2, scores_ub = proposal_info
    boxes, boxes_num, scores_index, pre_loop_unit = calc_info
    
    with tik_instance.if_scope(offset == 0):
        init_tensor(tik_instance, xx1, pre_loop_unit)
        init_tensor(tik_instance, xx2, pre_loop_unit)
        init_tensor(tik_instance, yy1, pre_loop_unit)
        init_tensor(tik_instance, yy2, pre_loop_unit)
        init_tensor(tik_instance, scores_ub, pre_loop_unit)

    with tik_instance.if_scope(tik.all(offset > 0, size + offset > pre_loop_unit)):
        size.set_as(pre_loop_unit - offset)

    # define the location index, the distance from the begin of class_begin
    lo_index = tik_instance.Scalar("uint32")
    # boxes is set as [4 * 1024], then the x1/x2/y1/y2 is stored in [0/1/2/3, 1024]
    with tik_instance.for_range(0, size) as idx:
        scores_index_offset = idx * Constant.UNIT_ELE
        lo_index.set_as(
            scores_index[scores_index_offset + 2:scores_index_offset + 4].reinterpret_cast_to("uint32"))
        xx1[offset + idx].set_as(boxes[0 * boxes_num + lo_index])
        yy1[offset + idx].set_as(boxes[1 * boxes_num + lo_index])
        xx2[offset + idx].set_as(boxes[2 * boxes_num + lo_index])
        yy2[offset + idx].set_as(boxes[3 * boxes_num + lo_index])

        scores_ub[offset + idx].set_as(scores_index[scores_index_offset])
    # yolov3 x1 values larger than x2, y1 values larger than y2, 
    # therefore exchange coordinate before calculate iou_selection
    exchange_coordinate(tik_instance, [xx1, xx2, yy1, yy2], pre_loop_unit)


def get_rectangle_area(tik_instance, coords_info, dst, pre_loop_unit):
    """
    area = (x2-x1) * (y2-y1), this is vector computing
    area can be reused in loops

    """
    xx1, xx2, yy1, yy2 = coords_info

    with tik_instance.new_stmt_scope():
        tmp1 = tik_instance.Tensor("float16", [pre_loop_unit, ], name="tmp1", scope=tik.scope_ubuf)
        tmp2 = tik_instance.Tensor("float16", [pre_loop_unit, ], name="y_diff", scope=tik.scope_ubuf)

        mask, _ = get_mask_rep_stride(xx2)
        repeat_times = pre_loop_unit // mask

        tik_instance.vsub(mask, tmp1, xx2, xx1, repeat_times, 1, 1, 1, 8, 8, 8)
        tik_instance.vsub(mask, tmp2, yy2, yy1, repeat_times, 1, 1, 1, 8, 8, 8)

        tik_instance.vmul(mask, dst, tmp1, tmp2, repeat_times, 1, 1, 1, 8, 8, 8)


def get_overlap(tik_instance, coords_info, calc_info, offset, pre_loop_unit):
    """
    get overlap area of x1 and the following others, the pre units mask the overlap 0

    """
    xx1, xx2, yy1, yy2 = coords_info
    overlap, tmp1, yyy1 = calc_info

    with tik_instance.new_stmt_scope():
        tmp = tik_instance.Tensor("float16", [pre_loop_unit, ], name="tmp", scope=tik.scope_ubuf)

        x1 = tik_instance.Scalar("float16", init_value=xx1[offset])
        x2 = tik_instance.Scalar("float16", init_value=xx2[offset])
        y1 = tik_instance.Scalar("float16", init_value=yy1[offset])
        y2 = tik_instance.Scalar("float16", init_value=yy2[offset])

        # `tmp = max(xx1[i], xx1[1:]), overlap=min(xx2[i], xx2[1:])
        tik_func_vmaxs(tik_instance, tmp, xx1, x1, pre_loop_unit, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)
        tik_func_vmins(tik_instance, overlap, xx2, x2, pre_loop_unit, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)
        mask, _ = get_mask_rep_stride(xx1)
        repeat_times = pre_loop_unit // mask
        # `w = max(0, xx2-xx1+offset), offset=0 here, sorted in tmp1`
        tik_instance.vsub(mask, tmp, overlap, tmp, repeat_times, 1, 1, 1, 8, 8, 8)
        tik_func_vmaxs(tik_instance, tmp1, tmp, 0, pre_loop_unit, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)

        # `yyy1 = max(yy1[i], yy1[1:]), overlap = min(yy2[i], yy2[1:])`
        tik_func_vmaxs(tik_instance, yyy1, yy1, y1, pre_loop_unit, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)
        tik_func_vmins(tik_instance, overlap, yy2, y2, pre_loop_unit, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)

        # `h = max(0, yy2 - yy1 + offset), offset=0 here, sorted in tmp`
        tik_instance.vsub(mask, yyy1, overlap, yyy1, repeat_times, 1, 1, 1, 8, 8, 8)
        tik_func_vmaxs(tik_instance, tmp, yyy1, 0, pre_loop_unit, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)

        tik_instance.vmul(mask, overlap, tmp1, tmp, repeat_times, 1, 1, 1, 8, 8, 8)

        # the overlap of the fixed boxes and itself default as 0
        overlap[offset].set_as(0)


def cal_iou(tik_instance, ub_info, offset, size, iou_thresh):
    """
    to calculate the related areas based on iou_thresh

    """
    src0, dst, tmp = ub_info

    # calculate the sum of area1 and area2
    area1 = tik_instance.Scalar("float16", init_value=src0[offset])

    mask, _ = get_mask_rep_stride(src0)
    repeat_times = size // mask
    tik_instance.vadds(mask, tmp, src0, area1, repeat_times, 1, 1, 8, 8)
    tik_instance.vmuls(mask, dst, tmp, iou_thresh, repeat_times, 1, 1, 8, 8)


def update_input_v300(tik_instance, propsal_info, single_area, size, param_info):
    """
    update the location and scores according to cmpmask

    """
    xx1, xx2, yy1, yy2, scores = propsal_info
    tmp1, tmp2, cmpmask_ub, pre_loop_unit = param_info

    mask = pre_loop_unit
    burst_lens = pre_loop_unit // Constant.BLOCK_ELE
    init_tensor(tik_instance, tmp1, pre_loop_unit)
    init_tensor(tik_instance, tmp2, pre_loop_unit)

    tik_instance.vreducev2(mask, tmp1, xx1, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
    tik_instance.data_move(xx1, tmp1, 0, 1, burst_lens, 0, 0)

    tik_instance.vreducev2(mask, tmp2, xx2, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
    tik_instance.data_move(xx2, tmp2, 0, 1, burst_lens, 0, 0)

    tik_instance.vreducev2(mask, tmp1, yy1, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
    tik_instance.data_move(yy1, tmp1, 0, 1, burst_lens, 0, 0)

    tik_instance.vreducev2(mask, tmp2, yy2, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
    tik_instance.data_move(yy2, tmp2, 0, 1, burst_lens, 0, 0)

    tik_instance.vreducev2(mask, tmp1, scores, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
    tik_instance.data_move(scores, tmp1, 0, 1, burst_lens, 0, 0)

    tik_instance.vreducev2(mask, tmp2, single_area, cmpmask_ub, 1, 1, 8, 1, rsvd_scalar=size, mask_mode="counter")
    tik_instance.data_move(single_area, tmp2, 0, 1, burst_lens, 0, 0)


def vreduce_rsvd_scalar(tik_instance, rsvd_scalar, shape, count, result, src, mask, repeat):
    """
    vrdeuce and deal with rsvd_scalar parameter

    """
    rsvd_scalar.set_as(0)
    vector_mask, _ = get_mask_rep_stride(result)
    repeat = (shape + vector_mask - 1) // vector_mask

    tik_instance.vec_dup(vector_mask, result, -1.0, repeat, 8)
    tik_instance.vreduce(count, result, src, mask, repeat, 1, 8, 1, 0, mask_mode="counter")

    with tik_instance.new_stmt_scope():
        vsel_score_ub = tik_instance.Tensor("float16", (shape,), name="vsel_score_ub", scope=tik.scope_ubuf)
        ones_ub = tik_instance.Tensor("float16", (shape,), name="ones_ub", scope=tik.scope_ubuf)
        zeros_ub = tik_instance.Tensor("float16", (shape,), name="zeros_ub", scope=tik.scope_ubuf)
        
        tik_instance.vec_dup(vector_mask, ones_ub, 1.0, repeat, 8)
        tik_instance.vec_dup(vector_mask, zeros_ub, 0.0, repeat, 8)

        with tik_instance.new_stmt_scope():
            mask_uint16 = tik_instance.Tensor("uint16", (shape,), name="mask_uint16", scope=tik.scope_ubuf)
        
            tik_instance.vec_cmpv_gt(mask_uint16, result, zeros_ub, repeat, 8, 8)
            tik_instance.vsel(vector_mask, 2, vsel_score_ub, mask_uint16, ones_ub, zeros_ub, repeat, 1, 1, 1, 8, 8, 8)

        rsvd_dst_ub = tik_instance.Tensor("int32", (8,), name="rsvd_dst_ub", scope=tik.scope_ubuf)
        rsvd_src_ub = tik_instance.Tensor("float16", (16,), name="rsvd_src_ub", scope=tik.scope_ubuf)
        vec_reduce_add_ub = tik_instance.Tensor("float16", (shape,), name="vec_reduce_add_ub", scope=tik.scope_ubuf)

        tik_instance.vec_reduce_add(vector_mask, rsvd_src_ub, vsel_score_ub, vec_reduce_add_ub, repeat, 8)
        tik_instance.vconv(1, "round", rsvd_dst_ub, rsvd_src_ub, 1, 1, 1, 8, 4)
        rsvd_scalar.set_as(rsvd_dst_ub[0])


def update_input_v300_resvd(tik_instance, xx1, xx2, yy1, yy2, scores, 
                            single_area, size, tmp1, tmp2, cmpmask_ub):
    """
    update the location and scores according to cmpmask

    """
    mask = Constant.PER_LOOP_UNIT
    burst_lens = Constant.PER_LOOP_UNIT // Constant.BLOCK_ELE
    init_tensor(tik_instance, tmp1, Constant.PER_LOOP_UNIT)
    init_tensor(tik_instance, tmp2, Constant.PER_LOOP_UNIT)

    vreduce_rsvd_scalar(tik_instance, size, burst_lens, mask, tmp1, xx1, cmpmask_ub, 1)
    tik_instance.data_move(xx1, tmp1, 0, 1, burst_lens, 0, 0)

    tik_instance.vreducev2(mask, tmp2, xx2, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
    tik_instance.data_move(xx2, tmp2, 0, 1, burst_lens, 0, 0)

    tik_instance.vreducev2(mask, tmp1, yy1, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
    tik_instance.data_move(yy1, tmp1, 0, 1, burst_lens, 0, 0)

    tik_instance.vreducev2(mask, tmp2, yy2, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
    tik_instance.data_move(yy2, tmp2, 0, 1, burst_lens, 0, 0)

    tik_instance.vreducev2(mask, tmp1, scores, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
    tik_instance.data_move(scores, tmp1, 0, 1, burst_lens, 0, 0)

    tik_instance.vreducev2(mask, tmp2, single_area, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
    tik_instance.data_move(single_area, tmp2, 0, 1, burst_lens, 0, 0)


def iou_selection_v300(tik_instance, proposal_data, size_data, max_size_per_class, iou_thresh):
    """
    calculate the overlap of multi boxes, sieve out target boxes with  iou_thresh

    """
    xx1, xx2, yy1, yy2, scores = proposal_data
    eff_lens, pre_loop_unit = size_data

    with tik_instance.new_stmt_scope():
        single_area = tik_instance.Tensor("float16", [pre_loop_unit, ], name="single_area",
                                          scope=tik.scope_ubuf)
        iou = tik_instance.Tensor("float16", [pre_loop_unit, ], name="iou",
                                  scope=tik.scope_ubuf)
        mask_shape_lens = pre_loop_unit // Constant.BLOCK_ELE
        mask_uint16 = tik_instance.Tensor("uint16", [mask_shape_lens, ], name="mask_int8",
                                          scope=tik.scope_ubuf)

        init_tensor(tik_instance, iou, pre_loop_unit)
        init_tensor(tik_instance, mask_uint16, mask_shape_lens)

        # get area of every window
        get_rectangle_area(tik_instance, [xx1, xx2, yy1, yy2], single_area, pre_loop_unit)

        # calculate the iou, end up when the output windows is more than max_size_per_class
        overlap = tik_instance.Tensor("float16", [pre_loop_unit, ], name="overlap", scope=tik.scope_ubuf)
        # define tmp tensor for following use, to reduce the cycle of apply/release memory
        tmp1 = tik_instance.Tensor("float16", [pre_loop_unit, ], name="tmp1", scope=tik.scope_ubuf)
        tmp2 = tik_instance.Tensor("float16", [pre_loop_unit, ], name="tmp2", scope=tik.scope_ubuf)

        with tik_instance.for_range(0, max_size_per_class) as idx:
            with tik_instance.if_scope(idx < eff_lens):
                # get overlap of windows_idx and the followings
                get_overlap(tik_instance, [xx1, xx2, yy1, yy2], [overlap, tmp1, tmp2], idx, pre_loop_unit)
                # get overlap of windows_idx and the followings
                cal_iou(tik_instance, [single_area, iou, tmp2], idx, pre_loop_unit, iou_thresh)
                gen_mask(tik_instance, overlap, iou, mask_uint16, pre_loop_unit)
                update_input_v300(tik_instance, [xx1, xx2, yy1, yy2, scores], 
                                  single_area, eff_lens, [tmp1, tmp2, mask_uint16, pre_loop_unit])
            with tik_instance.else_scope():
                tik_instance.tik_break()


def move_data_out_v300(tik_instance, proposal_data, eff_lens, bbox_out_list, size_data):
    """
    store results

    """
    x1, x2, y1, y2, scores_ub = proposal_data
    ret_ub, selected_count = bbox_out_list
    max_total_size, boxes_num = size_data
    
    data_size = max_total_size
    burst_lens = data_size // Constant.BLOCK_ELE

    with tik_instance.if_scope(burst_lens > 0):
        tik_instance.data_move(ret_ub[0 * boxes_num], x1, 0, 1, burst_lens, 0, 0)
        tik_instance.data_move(ret_ub[1 * boxes_num], y1, 0, 1, burst_lens, 0, 0)
        tik_instance.data_move(ret_ub[2 * boxes_num], x2, 0, 1, burst_lens, 0, 0)
        tik_instance.data_move(ret_ub[3 * boxes_num], y2, 0, 1, burst_lens, 0, 0)
        tik_instance.data_move(ret_ub[4 * boxes_num], scores_ub, 0, 1, burst_lens, 0, 0)

    block_ub0 = tik_instance.Tensor("float16", (Constant.BLOCK_ELE,), name="block_ub0", scope=tik.scope_ubuf)
    block_ub1 = tik_instance.Tensor("float16", (Constant.BLOCK_ELE,), name="block_ub1", scope=tik.scope_ubuf)

    # move tail data to gm
    last_offset = max(0, data_size - Constant.BLOCK_ELE)
    with tik_instance.for_range(0, Constant.BLOCK_ELE) as idx0:
        block_ub0[idx0].set_as(x1[last_offset + idx0])
    tik_instance.data_move(ret_ub[0 * boxes_num + last_offset], block_ub0, 0, 1, 1, 0, 0)
    with tik_instance.for_range(0, Constant.BLOCK_ELE) as idx1:
        block_ub1[idx1].set_as(y1[last_offset + idx1])
    tik_instance.data_move(ret_ub[1 * boxes_num + last_offset], block_ub1, 0, 1, 1, 0, 0)
    with tik_instance.for_range(0, Constant.BLOCK_ELE) as idx2:
        block_ub0[idx2].set_as(x2[last_offset + idx2])
    tik_instance.data_move(ret_ub[2 * boxes_num + last_offset], block_ub0, 0, 1, 1, 0, 0)
    with tik_instance.for_range(0, Constant.BLOCK_ELE) as idx3:
        block_ub1[idx3].set_as(y2[last_offset + idx3])
    tik_instance.data_move(ret_ub[3 * boxes_num + last_offset], block_ub1, 0, 1, 1, 0, 0)

    with tik_instance.for_range(0, Constant.BLOCK_ELE) as idx4:
        block_ub0[idx4].set_as(scores_ub[last_offset + idx4])
    tik_instance.data_move(ret_ub[4 * boxes_num + last_offset], block_ub0, 0, 1, 1, 0, 0)

    selected_count.set_as(eff_lens)
    
