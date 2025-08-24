#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
fsr_detection_output
"""
# 'pylint: disable=too-many-lines

import math
from impl import topk
from impl import nms
from impl import common_util
from impl.util import util_select_op_base
from impl.util.platform_adapter import tik
from impl import constant_util as constant
from impl import cnms_yolov3_ver as cnms_yolo
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.common_util import get_vector_repeat_times
from impl.util.util_common import get_mask_rep_stride
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_tik_comm_func import sort_score_idx_by_desc


# 'pylint: disable=R0913
# 'pylint: disable=R0914
# 'pylint: disable=R0915
# 'pylint: disable=W0201
# 'pylint: disable=W0134
# 'pylint: disable=C0111
# 'pylint: disable=C0121


# 'pylint: disable=too-few-public-methods,invalid-name
class Constant:
    """
    The class for constant.
    """
    MAX_REPEAT_TIME = 255
    FP16_ALIGN_NUM = 16
    TO_ALIGN_NUM = 15
    FP16_SIZE = 2
    FP16_MASK = 128
    FP16_RATIO = 1
    FP32_SIZE = 4
    FP32_MASK = 64
    INT32_MASK = 64
    FP32_RATIO = 2
    BLOCK_SIZE = 32
    VECTOR_BLOCK_SIZE = 256
    DATA_EIGHT = 8
    DATA_ONE = 1
    # every loop process 4096 units
    PER_LOOP_UNIT = 4096
    # location elements, [x1, y1, x2, y2]
    FOUR_DIRECTION = 4
    # b16 elements num of every block also uesed as b16 elements num of mask
    BLOCK_ELE = 16
    # define repeat elements every time for vsrot32
    REPEAT_ELE = 32
    # the socres_index contains four elements also marked as the class num processed every cycle
    UNIT_ELE = 4
    REPEAT_TIMES_MAX = 255
    # 0b0001 0001 0001 0001 is equals to type 3
    PATTERN_TYPE = 3
    DATALEN_4K = 4096
    DATALEN_2K = 2048
    DATALEN_1K = 1024
    DATALEN_128 = 128
    FACTOR_1 = 0.01
    FACTOR_2 = 100


# 'pylint: disable=too-many-lines,unused-argument,too-many-instance-attributes,too-few-public-methods
def get_op_support_info(rois_dic, bbox_delta_dic, score_dic, im_info_dic,
                        actual_rois_num_dic, actual_bbox_num_dic, box_dic,
                        num_classes, score_threshold, iou_threshold, batch_rois=1,
                        kernel_name="fsr_detection_output"):
    """
    get split info
    rois: [batch, 5, max_rois_num], 5 means (batchId, x1, y1, x2, y2)
    bbox_delta: `[total_rois, num_class*4], total_rois = batch * max_rois_num`
    score: [total_rois, num_class]
    img_info
    actual_rois_num: [batch_rois, 8]
    actual_bbox_num: [batch, num_classes]
    box: [batch, numBoxes, 8]
    """
    return util_select_op_base.get_op_cal_info(None, None, 0, 0)


def get_params(dtype):
    """
    :param dtype:
    :return:
    """
    if dtype == "float16":
        size = Constant.FP16_SIZE
        mask = Constant.FP16_MASK
        ratio = Constant.FP16_RATIO
    elif dtype == "float32":
        size = Constant.FP32_SIZE
        mask = Constant.FP32_MASK
        ratio = Constant.FP32_RATIO
    return size, mask, ratio


def vec_dup(inputs, ub_to_dup, const=0):
    """
    :param inputs:
    :param ub_to_dup:
    :param const:
    :return:
    """
    tik_instance = inputs[0]
    cur_process_num = inputs[1]
    input_dtype = inputs[2]

    size, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//mask)//Constant.MAX_REPEAT_TIME
    with tik_instance.if_scope(cur_process_num//mask > Constant.MAX_REPEAT_TIME):
        with tik_instance.for_range(
                0, repeat) as i:
            tik_instance.vector_dup(
                mask, ub_to_dup[Constant.MAX_REPEAT_TIME*mask*i], const,
                Constant.MAX_REPEAT_TIME, Constant.DATA_ONE, Constant.DATA_EIGHT)
    tail = cur_process_num % (Constant.MAX_REPEAT_TIME*mask)
    tail_n = tail//mask
    if tail_n != 0:
        tik_instance.vector_dup(mask, ub_to_dup[Constant.MAX_REPEAT_TIME*mask*repeat],
                                const, tail_n, Constant.DATA_ONE,
                                Constant.DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail != 0:
        tik_instance.vector_dup(
            tail_tail, ub_to_dup[Constant.MAX_REPEAT_TIME*mask*repeat+tail_n*mask],
            const, Constant.DATA_ONE, Constant.DATA_ONE,
            tail_tail//(Constant.BLOCK_SIZE//size))


def vec_muls(inputs, dst, const, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param const:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src = inputs[1]
    input_dtype = inputs[2]

    size, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//mask)//Constant.MAX_REPEAT_TIME
    if cur_process_num//mask > Constant.MAX_REPEAT_TIME:
        for i in range(repeat):
            tik_instance.vmuls(mask, dst[Constant.MAX_REPEAT_TIME*mask*i],
                               src[Constant.MAX_REPEAT_TIME*mask*i],
                               const, Constant.MAX_REPEAT_TIME,
                               Constant.DATA_ONE, Constant.DATA_ONE,
                               Constant.DATA_EIGHT,
                               Constant.DATA_EIGHT)
    tail = cur_process_num % (mask*Constant.MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vmuls(mask, dst[Constant.MAX_REPEAT_TIME*mask*repeat],
                           src[Constant.MAX_REPEAT_TIME*mask*repeat],
                           const, tail_n,
                           Constant.DATA_ONE, Constant.DATA_ONE,
                           Constant.DATA_EIGHT,
                           Constant.DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail > 0:
        tik_instance.vmuls(tail_tail,
                           dst[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                           src[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                           const, Constant.DATA_ONE, Constant.DATA_ONE, Constant.DATA_ONE,
                           tail_tail//(Constant.BLOCK_SIZE//size),
                           tail_tail//(Constant.BLOCK_SIZE//size))


def vec_sub(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src1 = inputs[1]
    src2 = inputs[2]
    input_dtype = inputs[3]

    size, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//mask)//Constant.MAX_REPEAT_TIME
    if cur_process_num//mask > Constant.MAX_REPEAT_TIME:
        for i in range(repeat):
            tik_instance.vsub(mask, dst[Constant.MAX_REPEAT_TIME*mask*i],
                              src1[Constant.MAX_REPEAT_TIME*mask*i],
                              src2[Constant.MAX_REPEAT_TIME*mask*i],
                              Constant.MAX_REPEAT_TIME, Constant.DATA_ONE,
                              Constant.DATA_ONE, Constant.DATA_ONE,
                              Constant.DATA_EIGHT,
                              Constant.DATA_EIGHT,
                              Constant.DATA_EIGHT)
    tail = cur_process_num % (mask*Constant.MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vsub(mask, dst[Constant.MAX_REPEAT_TIME*mask*repeat],
                          src1[Constant.MAX_REPEAT_TIME*mask*repeat],
                          src2[Constant.MAX_REPEAT_TIME*mask*repeat],
                          tail_n, Constant.DATA_ONE,
                          Constant.DATA_ONE, Constant.DATA_ONE,
                          Constant.DATA_EIGHT,
                          Constant.DATA_EIGHT,
                          Constant.DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail > 0:
        tik_instance.vsub(tail_tail,
                          dst[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src1[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src2[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n], Constant.DATA_ONE,
                          Constant.DATA_ONE, Constant.DATA_ONE,
                          Constant.DATA_ONE,
                          tail_tail//(Constant.BLOCK_SIZE//size),
                          tail_tail//(Constant.BLOCK_SIZE//size),
                          tail_tail//(Constant.BLOCK_SIZE//size))


def vec_adds(inputs, dst, const, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param const:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src = inputs[1]
    input_dtype = inputs[2]

    size, mask, _ = get_params(input_dtype)
    repeat = (cur_process_num//mask)//Constant.MAX_REPEAT_TIME
    if cur_process_num//mask > Constant.MAX_REPEAT_TIME:
        for i in range(repeat):
            tik_instance.vadds(mask, dst[Constant.MAX_REPEAT_TIME*mask*i],
                               src[Constant.MAX_REPEAT_TIME*mask*i],
                               const, Constant.MAX_REPEAT_TIME,
                               Constant.DATA_ONE, Constant.DATA_ONE,
                               Constant.DATA_EIGHT,
                               Constant.DATA_EIGHT)
    tail = cur_process_num % (mask*Constant.MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vadds(mask, dst[Constant.MAX_REPEAT_TIME*mask*repeat],
                           src[Constant.MAX_REPEAT_TIME*mask*repeat],
                           const, tail_n,
                           Constant.DATA_ONE, Constant.DATA_ONE,
                           Constant.DATA_EIGHT,
                           Constant.DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail > 0:
        tik_instance.vadds(tail_tail,
                           dst[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                           src[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                           const, Constant.DATA_ONE, Constant.DATA_ONE, Constant.DATA_ONE,
                           tail_tail//(Constant.BLOCK_SIZE//size),
                           tail_tail//(Constant.BLOCK_SIZE//size))


def vec_add(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src1 = inputs[1]
    src2 = inputs[2]
    input_dtype = inputs[3]

    size, mask, _ = get_params(input_dtype)
    repeat = (cur_process_num//mask)//Constant.MAX_REPEAT_TIME
    if cur_process_num//mask > Constant.MAX_REPEAT_TIME:
        for i in range(repeat):
            tik_instance.vadd(mask, dst[Constant.MAX_REPEAT_TIME*mask*i],
                              src1[Constant.MAX_REPEAT_TIME*mask*i],
                              src2[Constant.MAX_REPEAT_TIME*mask*i],
                              Constant.MAX_REPEAT_TIME,
                              Constant.DATA_ONE, Constant.DATA_ONE,
                              Constant.DATA_ONE, Constant.DATA_EIGHT,
                              Constant.DATA_EIGHT,
                              Constant.DATA_EIGHT)
    tail = cur_process_num % (mask*Constant.MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vadd(mask, dst[Constant.MAX_REPEAT_TIME*mask*repeat],
                          src1[Constant.MAX_REPEAT_TIME*mask*repeat],
                          src2[Constant.MAX_REPEAT_TIME*mask*repeat],
                          tail_n, Constant.DATA_ONE,
                          Constant.DATA_ONE, Constant.DATA_ONE,
                          Constant.DATA_EIGHT,
                          Constant.DATA_EIGHT,
                          Constant.DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail > 0:
        tik_instance.vadd(tail_tail,
                          dst[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src1[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src2[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n], Constant.DATA_ONE,
                          Constant.DATA_ONE,
                          Constant.DATA_ONE,
                          Constant.DATA_ONE,
                          tail_tail//(Constant.BLOCK_SIZE//size),
                          tail_tail//(Constant.BLOCK_SIZE//size),
                          tail_tail//(Constant.BLOCK_SIZE//size))


def vec_mla(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src1 = inputs[1]
    src2 = inputs[2]
    input_dtype = inputs[3]

    size, mask, _ = get_params(input_dtype)
    repeat = (cur_process_num//mask)//Constant.MAX_REPEAT_TIME
    if cur_process_num//mask > Constant.MAX_REPEAT_TIME:
        for i in range(repeat):
            tik_instance.vmla(mask, dst[Constant.MAX_REPEAT_TIME*mask*i],
                              src1[Constant.MAX_REPEAT_TIME*mask*i],
                              src2[Constant.MAX_REPEAT_TIME*mask*i], Constant.MAX_REPEAT_TIME,
                              Constant.DATA_ONE, Constant.DATA_ONE,
                              Constant.DATA_ONE, Constant.DATA_EIGHT,
                              Constant.DATA_EIGHT,
                              Constant.DATA_EIGHT)
    tail = cur_process_num % (mask*Constant.MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vmla(mask, dst[Constant.MAX_REPEAT_TIME*mask*repeat],
                          src1[Constant.MAX_REPEAT_TIME*mask*repeat],
                          src2[Constant.MAX_REPEAT_TIME*mask*repeat], tail_n,
                          Constant.DATA_ONE, Constant.DATA_ONE,
                          Constant.DATA_ONE,
                          Constant.DATA_EIGHT,
                          Constant.DATA_EIGHT,
                          Constant.DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail > 0:
        tik_instance.vmla(tail_tail,
                          dst[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src1[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src2[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n], Constant.DATA_ONE,
                          Constant.DATA_ONE,
                          Constant.DATA_ONE, Constant.DATA_ONE,
                          tail_tail//(Constant.BLOCK_SIZE//size),
                          tail_tail//(Constant.BLOCK_SIZE//size),
                          tail_tail//(Constant.BLOCK_SIZE//size))


def vec_exp(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src = inputs[1]
    input_dtype = inputs[2]

    size, mask, _ = get_params(input_dtype)
    repeat = (cur_process_num//mask)//Constant.MAX_REPEAT_TIME
    if cur_process_num//mask > Constant.MAX_REPEAT_TIME:
        for i in range((cur_process_num//mask)//Constant.MAX_REPEAT_TIME):
            tik_instance.vexp(mask, dst[Constant.MAX_REPEAT_TIME*mask*i],
                              src[Constant.MAX_REPEAT_TIME*mask*i], Constant.MAX_REPEAT_TIME,
                              Constant.DATA_ONE, Constant.DATA_ONE,
                              Constant.DATA_EIGHT,
                              Constant.DATA_EIGHT)
    tail = cur_process_num % (mask*Constant.MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vexp(mask, dst[Constant.MAX_REPEAT_TIME*mask*repeat],
                          src[Constant.MAX_REPEAT_TIME*mask*repeat],
                          tail_n, Constant.DATA_ONE, Constant.DATA_ONE,
                          Constant.DATA_EIGHT,
                          Constant.DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail > 0:
        tik_instance.vexp(tail_tail,
                          dst[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          Constant.DATA_ONE, Constant.DATA_ONE, Constant.DATA_ONE,
                          tail_tail//(Constant.BLOCK_SIZE//size),
                          tail_tail//(Constant.BLOCK_SIZE//size))


def vec_mul(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src1 = inputs[1]
    src2 = inputs[2]
    input_dtype = inputs[3]
    size, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//mask)//Constant.MAX_REPEAT_TIME
    if cur_process_num//mask > Constant.MAX_REPEAT_TIME:
        for i in range((cur_process_num//mask)//Constant.MAX_REPEAT_TIME):
            tik_instance.vmul(mask, dst[Constant.MAX_REPEAT_TIME*mask*i],
                              src1[Constant.MAX_REPEAT_TIME*mask*i],
                              src2[Constant.MAX_REPEAT_TIME*mask*i], Constant.MAX_REPEAT_TIME,
                              Constant.DATA_ONE, Constant.DATA_ONE,
                              Constant.DATA_ONE,
                              Constant.DATA_EIGHT,
                              Constant.DATA_EIGHT,
                              Constant.DATA_EIGHT)
    tail = cur_process_num % (mask*Constant.MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vmul(mask, dst[Constant.MAX_REPEAT_TIME*mask*repeat],
                          src1[Constant.MAX_REPEAT_TIME*mask*repeat],
                          src2[Constant.MAX_REPEAT_TIME*mask*repeat], tail_n,
                          Constant.DATA_ONE, Constant.DATA_ONE,
                          Constant.DATA_ONE,
                          Constant.DATA_EIGHT,
                          Constant.DATA_EIGHT,
                          Constant.DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail > 0:
        tik_instance.vmul(tail_tail,
                          dst[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src1[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src2[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n], Constant.DATA_ONE,
                          Constant.DATA_ONE,
                          Constant.DATA_ONE, Constant.DATA_ONE,
                          tail_tail//(Constant.BLOCK_SIZE//size),
                          tail_tail//(Constant.BLOCK_SIZE//size),
                          tail_tail//(Constant.BLOCK_SIZE//size))


def vec_min(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src1 = inputs[1]
    src2 = inputs[2]
    input_dtype = inputs[3]
    size, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//mask)//Constant.MAX_REPEAT_TIME
    if cur_process_num//mask > Constant.MAX_REPEAT_TIME:
        for i in range(repeat):
            tik_instance.vmin(mask, dst[Constant.MAX_REPEAT_TIME*mask*i],
                              src1[Constant.MAX_REPEAT_TIME*mask*i],
                              src2[Constant.MAX_REPEAT_TIME*mask*i], Constant.MAX_REPEAT_TIME,
                              Constant.DATA_ONE, Constant.DATA_ONE,
                              Constant.DATA_ONE, Constant.DATA_EIGHT,
                              Constant.DATA_EIGHT,
                              Constant.DATA_EIGHT)
    tail = cur_process_num % (mask*Constant.MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vmin(mask, dst[Constant.MAX_REPEAT_TIME*mask*repeat],
                          src1[Constant.MAX_REPEAT_TIME*mask*repeat],
                          src2[Constant.MAX_REPEAT_TIME*mask*repeat], tail_n,
                          Constant.DATA_ONE, Constant.DATA_ONE,
                          Constant.DATA_ONE,
                          Constant.DATA_EIGHT,
                          Constant.DATA_EIGHT,
                          Constant.DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail > 0:
        tik_instance.vmin(tail_tail,
                          dst[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src1[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                          src2[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n], Constant.DATA_ONE,
                          Constant.DATA_ONE,
                          Constant.DATA_ONE, Constant.DATA_ONE,
                          tail_tail//(Constant.BLOCK_SIZE//size),
                          tail_tail//(Constant.BLOCK_SIZE//size),
                          tail_tail//(Constant.BLOCK_SIZE//size))


def vec_relu(inputs, dst, cur_process_num):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :return:
    """
    tik_instance = inputs[0]
    src = inputs[1]
    input_dtype = inputs[2]
    size, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//mask)//Constant.MAX_REPEAT_TIME
    if cur_process_num//mask > Constant.MAX_REPEAT_TIME:
        for i in range((cur_process_num//mask)//Constant.MAX_REPEAT_TIME):
            tik_instance.vrelu(mask, dst[Constant.MAX_REPEAT_TIME*mask*i],
                               src[Constant.MAX_REPEAT_TIME*mask*i],
                               Constant.MAX_REPEAT_TIME,
                               Constant.DATA_ONE, Constant.DATA_ONE,
                               Constant.DATA_EIGHT,
                               Constant.DATA_EIGHT)
    tail = cur_process_num % (mask*Constant.MAX_REPEAT_TIME)
    tail_n = tail // mask
    if tail_n > 0:
        tik_instance.vrelu(mask, dst[Constant.MAX_REPEAT_TIME*mask*repeat],
                           src[Constant.MAX_REPEAT_TIME*mask*repeat],
                           tail_n,
                           Constant.DATA_ONE, Constant.DATA_ONE,
                           Constant.DATA_EIGHT,
                           Constant.DATA_EIGHT)
    tail_tail = tail % mask
    if tail_tail > 0:
        tik_instance.vrelu(tail_tail,
                           dst[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                           src[Constant.MAX_REPEAT_TIME*mask*repeat+mask*tail_n],
                           Constant.DATA_ONE,
                           Constant.DATA_ONE, Constant.DATA_ONE,
                           tail_tail//(Constant.BLOCK_SIZE//size),
                           tail_tail//(Constant.BLOCK_SIZE//size))


def vec_concat(inputs, dst, cur_process_num, const):
    """
    :param inputs:
    :param dst:
    :param cur_process_num:
    :param const:
    :return:
    """
    tik_instance = inputs[0]
    src = inputs[1]
    input_dtype = inputs[2]
    _, mask, _ = get_params(input_dtype)

    repeat = (cur_process_num//Constant.FP16_ALIGN_NUM)//Constant.MAX_REPEAT_TIME
    if cur_process_num//Constant.FP16_ALIGN_NUM > Constant.MAX_REPEAT_TIME:
        for i in range((cur_process_num//Constant.FP16_ALIGN_NUM)//Constant.MAX_REPEAT_TIME):
            tik_instance.vconcat(dst[Constant.MAX_REPEAT_TIME*mask*i],
                                 src[Constant.MAX_REPEAT_TIME*Constant.FP16_ALIGN_NUM*i],
                                 Constant.MAX_REPEAT_TIME, const)
    tail = cur_process_num % (Constant.FP16_ALIGN_NUM*Constant.MAX_REPEAT_TIME)
    if tail > 0:
        tik_instance.vconcat(dst[Constant.MAX_REPEAT_TIME*mask*repeat],
                             src[Constant.MAX_REPEAT_TIME*Constant.FP16_ALIGN_NUM*repeat],
                             tail//Constant.FP16_ALIGN_NUM, const)


def get_ub_size():
    """
    :return:
    """
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    if ub_size <= 0:
        error_manager_vector.raise_err_specific_reson("fsr_detection_output", "The value of the UB_SIZE is illegal!")
    return ub_size


def filter_device_core(batch):
    """
    :param batch:
    :return:
    """
    device_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    if device_core_num <= 0:
        error_manager_vector.raise_err_specific_reson("fsr_detection_output", "The value of the CORE_NUM is illegal!")
    if batch >= device_core_num:
        batch_factor = batch//device_core_num
        batch_factor_tail = batch - batch_factor*device_core_num
    else:
        batch_factor = batch
        batch_factor_tail = 0
        device_core_num = Constant.DATA_ONE

    return device_core_num, batch_factor, batch_factor_tail


def call_topk_sort(tik_instance, input_topk, output):
    """
    :param tik_instance:
    :param input_topk:
    :param output:
    :return:
    """
    max_rois_num = input_topk[0]
    score_threshold = input_topk[1]
    pre_nms_topn = input_topk[2]
    output_box = input_topk[3]
    mem_swap = input_topk[4]

    batch_id = output[0]
    regions_sorted = output[1]
    proposal_actual_num = output[2]

    k = pre_nms_topn

    topk_input = {
        "proposal_num": max_rois_num,
        "k": k,
        "score_threshold": score_threshold,
        "regions_orig": output_box,
        "mem_swap": mem_swap,
        # "batch_offset": batch_offset
    }

    topk_out = {
        "batch_id": batch_id,
        "regions_sorted": regions_sorted,
        "proposal_actual_num": proposal_actual_num,
    }

    topk.tik_topk(tik_instance, topk_input, topk_out)
    

def check_soc_version():
    """
    Check devices whether is rpn supported
    each of these devices only needs coordinate data and scores data

    """
    is_rpn_supported = True
    if tbe_platform.api_check_support("tik.vcopy"):
        is_rpn_supported = False
    return is_rpn_supported


class DecodeRois:
    """
    Update Decode
    """
    def __init__(self, tik_instance, input_data, tiling_flage):
        """
        :param tik_instance:
        :param input_data:
        """
        self.tik_instance = tik_instance
        self.cur_process_num = input_data[0]
        self.input_dtype = input_data[1]
        self.image_info = input_data[2]
        self.num_class = input_data[3]
        if tiling_flage:
            shape = (self.cur_process_num//Constant.FP16_ALIGN_NUM, Constant.FP16_ALIGN_NUM)
            self.output_region_proposal_ub = tik_instance.Tensor(
                self.input_dtype, (self.cur_process_num//Constant.FP16_ALIGN_NUM,
                                   Constant.FP16_ALIGN_NUM, constant.REPEAT_STRIDE_EIGHT),
                name="output_region_proposal_ub", scope=tik.scope_ubuf)
            vec_dup((tik_instance,
                     self.cur_process_num*Constant.DATA_EIGHT,
                     self.input_dtype),
                    self.output_region_proposal_ub)
        else:
            shape = (self.cur_process_num//Constant.FP16_ALIGN_NUM*self.num_class, Constant.FP16_ALIGN_NUM)
            self.output_region_proposal_ub = tik_instance.Tensor(
                self.input_dtype, (self.num_class, self.cur_process_num//Constant.FP16_ALIGN_NUM,
                                   Constant.FP16_ALIGN_NUM, Constant.DATA_EIGHT),
                name="output_region_proposal_ub", scope=tik.scope_ubuf)

            vec_dup((tik_instance,
                     self.cur_process_num*Constant.DATA_EIGHT*self.num_class,
                     self.input_dtype),
                    self.output_region_proposal_ub)

        self.size, self.mask, self.ratio = get_params(self.input_dtype)
        self.im_info_ub = tik_instance.Tensor(self.input_dtype,
                                              (Constant.FP16_ALIGN_NUM/self.ratio,),
                                              name="im_info_ub", scope=tik.scope_ubuf)

        self.x1_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="x1_ubaddr",
            scope=tik.scope_ubuf)
        self.y1_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="y1_ubaddr",
            scope=tik.scope_ubuf)
        self.x2_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="x2_ubaddr",
            scope=tik.scope_ubuf)
        self.y2_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="y2_ubaddr",
            scope=tik.scope_ubuf)
        self.dx_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="dx_ubaddr",
            scope=tik.scope_ubuf)
        self.dy_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="dy_ubaddr",
            scope=tik.scope_ubuf)
        self.dw_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="dw_ubaddr",
            scope=tik.scope_ubuf)
        self.dh_ubaddr = tik_instance.Tensor(
            self.input_dtype, shape, name="dh_ubaddr",
            scope=tik.scope_ubuf)
        self.ubaddr0 = tik_instance.Tensor(
            self.input_dtype, shape, name="ubaddr0",
            scope=tik.scope_ubuf)
        self.ubaddr1 = tik_instance.Tensor(
            self.input_dtype, shape, name="ubaddr1",
            scope=tik.scope_ubuf)

    def generate_rois(self, input_list, cur_batch_index, output_region_proposal):
        """
        :param input_list:
        :param batchID:
        :param output_region_proposal:
        :return:
        """
        cur_process_num = input_list[0]
        rois_offset = input_list[1]
        prior_offset = input_list[2]
        score_offset = input_list[3]
        score_gm = input_list[4]
        prior_box_gm = input_list[5]
        rois = input_list[6]
        max_rois_num = input_list[7]
        x1y1x2y2_gm = input_list[8]
        scores_gm = input_list[9]

        self.tik_instance.data_move(self.im_info_ub,
                                    self.image_info[cur_batch_index, 0],
                                    0, Constant.DATA_ONE, Constant.DATA_ONE, 0, 0, 0)

        with self.tik_instance.new_scope():
            self.tik_instance.data_move(self.x1_ubaddr[0], rois[rois_offset],
                                        0, cur_process_num//Constant.FP16_ALIGN_NUM,
                                        self.ratio, 0, 0)
            self.tik_instance.data_move(self.y1_ubaddr[0],
                                        rois[rois_offset+max_rois_num], 0,
                                        cur_process_num//Constant.FP16_ALIGN_NUM,
                                        self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.x2_ubaddr[0], rois[rois_offset+max_rois_num*2], 0,
                cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.y2_ubaddr[0], rois[rois_offset+max_rois_num*3], 0,
                cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)

            with self.tik_instance.for_range(1, self.num_class) as class_index:
                self.tik_instance.data_move(
                    self.x1_ubaddr[cur_process_num*class_index], self.x1_ubaddr,
                    0, cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)

                self.tik_instance.data_move(
                    self.y1_ubaddr[cur_process_num*class_index], self.y1_ubaddr,
                    0, cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)

                self.tik_instance.data_move(
                    self.x2_ubaddr[cur_process_num*class_index], self.x2_ubaddr,
                    0, cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)

                self.tik_instance.data_move(
                    self.y2_ubaddr[cur_process_num*class_index], self.y2_ubaddr,
                    0, cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dx_ubaddr[0], prior_box_gm[prior_offset], 0,
                cur_process_num//Constant.FP16_ALIGN_NUM*self.num_class, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dy_ubaddr[0],
                prior_box_gm[prior_offset + max_rois_num*self.num_class],
                0, cur_process_num//Constant.FP16_ALIGN_NUM*self.num_class, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dw_ubaddr[0],
                prior_box_gm[prior_offset + max_rois_num*2*self.num_class],
                0, cur_process_num//Constant.FP16_ALIGN_NUM*self.num_class, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dh_ubaddr[0], prior_box_gm[
                    prior_offset + max_rois_num * 3*self.num_class], 0,
                cur_process_num//Constant.FP16_ALIGN_NUM*self.num_class, self.ratio, 0, 0)

            vec_sub((self.tik_instance, self.x2_ubaddr, self.x1_ubaddr,
                     self.input_dtype), self.ubaddr0,
                    cur_process_num*self.num_class)

            vec_sub((self.tik_instance, self.y2_ubaddr, self.y1_ubaddr,
                     self.input_dtype), self.ubaddr1,
                    cur_process_num*self.num_class)

            temp = Constant.DATA_ONE
            vec_adds((self.tik_instance, self.ubaddr0, self.input_dtype),
                     self.x2_ubaddr, temp, cur_process_num*self.num_class)

            temp = Constant.DATA_ONE
            vec_adds((self.tik_instance, self.ubaddr1, self.input_dtype),
                     self.y2_ubaddr, temp, cur_process_num*self.num_class)

            temp = 0.5
            vec_muls((self.tik_instance, self.x2_ubaddr, self.input_dtype),
                     self.ubaddr0, temp, cur_process_num*self.num_class)

            temp = 0.5
            vec_muls((self.tik_instance, self.y2_ubaddr, self.input_dtype),
                     self.ubaddr1, temp, cur_process_num*self.num_class)

            vec_add((self.tik_instance, self.ubaddr0, self.x1_ubaddr,
                     self.input_dtype), self.x1_ubaddr,
                    cur_process_num*self.num_class)

            vec_add((self.tik_instance, self.ubaddr1, self.y1_ubaddr,
                     self.input_dtype), self.y1_ubaddr,
                    cur_process_num*self.num_class)

            vec_mla((self.tik_instance, self.dx_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.x1_ubaddr,
                    cur_process_num*self.num_class)

            vec_mla((self.tik_instance, self.dy_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.y1_ubaddr,
                    cur_process_num*self.num_class)

            vec_exp((self.tik_instance, self.dw_ubaddr, self.input_dtype),
                    self.dx_ubaddr, cur_process_num*self.num_class)

            vec_exp((self.tik_instance, self.dh_ubaddr, self.input_dtype),
                    self.dy_ubaddr, cur_process_num*self.num_class)

            vec_mul((self.tik_instance, self.dx_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.x2_ubaddr,
                    cur_process_num*self.num_class)

            vec_mul((self.tik_instance, self.dy_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.y2_ubaddr,
                    cur_process_num*self.num_class)

            temp = 0.5
            vec_muls((self.tik_instance, self.x2_ubaddr, self.input_dtype),
                     self.x2_ubaddr, temp, cur_process_num*self.num_class)

            temp = 0.5
            vec_muls((self.tik_instance, self.y2_ubaddr, self.input_dtype),
                     self.y2_ubaddr, temp, cur_process_num*self.num_class)

            vec_sub((self.tik_instance, self.x1_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.dx_ubaddr,
                    cur_process_num*self.num_class)

            vec_sub((self.tik_instance, self.y1_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.dy_ubaddr,
                    cur_process_num*self.num_class)

            vec_add((self.tik_instance, self.x1_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.dw_ubaddr,
                    cur_process_num*self.num_class)

            vec_add((self.tik_instance, self.y1_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.dh_ubaddr,
                    cur_process_num*self.num_class)
            self.tik_instance.vadds(Constant.FP16_ALIGN_NUM//self.ratio, self.im_info_ub,
                                    self.im_info_ub, -1, Constant.DATA_ONE, Constant.DATA_ONE, Constant.DATA_ONE, 0, 0)

            #clip

            im_scalar = self.tik_instance.Scalar(dtype=self.input_dtype)
            im_scalar.set_as(self.im_info_ub[1])

            vec_dup((self.tik_instance, cur_process_num*self.num_class,
                     self.input_dtype), self.ubaddr0, im_scalar)
            im_scalar.set_as(self.im_info_ub[0])
            vec_dup((self.tik_instance, cur_process_num*self.num_class,
                     self.input_dtype), self.ubaddr1, im_scalar)

            vec_min((self.tik_instance, self.dx_ubaddr, self.ubaddr0,
                     self.input_dtype), self.dx_ubaddr,
                    cur_process_num*self.num_class)

            vec_min((self.tik_instance, self.dy_ubaddr, self.ubaddr1,
                     self.input_dtype), self.dy_ubaddr,
                    cur_process_num*self.num_class)

            vec_min((self.tik_instance, self.dw_ubaddr, self.ubaddr0,
                     self.input_dtype), self.dw_ubaddr,
                    cur_process_num*self.num_class)

            vec_min((self.tik_instance, self.dh_ubaddr, self.ubaddr1,
                     self.input_dtype), self.dh_ubaddr,
                    cur_process_num*self.num_class)

            vec_relu((self.tik_instance, self.dx_ubaddr, self.input_dtype),
                     self.x1_ubaddr, cur_process_num*self.num_class)

            vec_relu((self.tik_instance, self.dy_ubaddr, self.input_dtype),
                     self.y1_ubaddr, cur_process_num*self.num_class)

            vec_relu((self.tik_instance, self.dw_ubaddr, self.input_dtype),
                     self.x2_ubaddr, cur_process_num*self.num_class)

            vec_relu((self.tik_instance, self.dh_ubaddr, self.input_dtype),
                     self.y2_ubaddr, cur_process_num*self.num_class)
            self.tik_instance.data_move(
                self.ubaddr0, score_gm[score_offset], 0,
                cur_process_num//Constant.FP16_ALIGN_NUM*self.num_class, self.ratio, 0, 0)

            is_rpn_supported = check_soc_version()
            if is_rpn_supported:

                vec_concat((self.tik_instance, self.x1_ubaddr, self.input_dtype),
                        self.output_region_proposal_ub,
                        cur_process_num*self.num_class, 0)
                vec_concat((self.tik_instance, self.y1_ubaddr, self.input_dtype),
                        self.output_region_proposal_ub,
                        cur_process_num*self.num_class, 1)
                vec_concat((self.tik_instance, self.x2_ubaddr, self.input_dtype),
                        self.output_region_proposal_ub,
                        cur_process_num*self.num_class, 2)
                vec_concat((self.tik_instance, self.y2_ubaddr, self.input_dtype),
                        self.output_region_proposal_ub,
                        cur_process_num*self.num_class, 3)
                vec_concat((self.tik_instance, self.ubaddr0, self.input_dtype),
                        self.output_region_proposal_ub,
                        cur_process_num*self.num_class, 4)
                
                self.tik_instance.data_move(
                output_region_proposal[score_offset*Constant.DATA_EIGHT],
                self.output_region_proposal_ub, 0,
                cur_process_num*Constant.DATA_EIGHT//Constant.FP16_ALIGN_NUM*self.num_class,
                self.ratio, 0, 0)

            else:
                with self.tik_instance.for_range(0, self.num_class) as class_index:
                    self.tik_instance.data_move(scores_gm[cur_batch_index, class_index, 0], 
                                                self.ubaddr0[class_index*cur_process_num], 
                                                0, cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)
                    self.tik_instance.data_move(x1y1x2y2_gm[cur_batch_index, 0, class_index, 0], 
                                                self.x1_ubaddr[class_index*cur_process_num], 0, 
                                                cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)
                    self.tik_instance.data_move(x1y1x2y2_gm[cur_batch_index, 1, class_index, 0], 
                                                self.y1_ubaddr[class_index*cur_process_num], 0, 
                                                cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)
                    self.tik_instance.data_move(x1y1x2y2_gm[cur_batch_index, 2, class_index, 0], 
                                                self.x2_ubaddr[class_index*cur_process_num], 0, 
                                                cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)
                    self.tik_instance.data_move(x1y1x2y2_gm[cur_batch_index, 3, class_index, 0], 
                                                self.y2_ubaddr[class_index*cur_process_num], 0, 
                                                cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)


    def tiling_generate_rois(self, input_list, cur_batch_index, output_region_proposal):
        """
        :param input_list:
        :param batchID:
        :param output_region_proposal:
        :return:
        """
        cur_process_num = input_list[0]
        rois_offset = input_list[1]
        prior_offset = input_list[2]
        score_offset = input_list[3]
        score_gm = input_list[4]
        prior_box_gm = input_list[5]
        rois = input_list[6]
        max_rois_num = input_list[7]
        x1y1x2y2_gm = input_list[8]
        scores_gm = input_list[9]
        class_idx = input_list[10]

        self.tik_instance.data_move(self.im_info_ub,
                                    self.image_info[cur_batch_index, 0],
                                    0, Constant.DATA_ONE, Constant.DATA_ONE, 0, 0, 0)

        with self.tik_instance.new_scope():
            self.tik_instance.data_move(self.x1_ubaddr[0], rois[rois_offset],
                                        0, cur_process_num//Constant.FP16_ALIGN_NUM,
                                        self.ratio, 0, 0)
            self.tik_instance.data_move(self.y1_ubaddr[0],
                                        rois[rois_offset+max_rois_num], 0,
                                        cur_process_num//Constant.FP16_ALIGN_NUM,
                                        self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.x2_ubaddr[0], rois[rois_offset+max_rois_num*2], 0,
                cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.y2_ubaddr[0], rois[rois_offset+max_rois_num*3], 0,
                cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dx_ubaddr[0], prior_box_gm[prior_offset], 0,
                cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dy_ubaddr[0], prior_box_gm[prior_offset + max_rois_num*self.num_class],
                0, cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(
                self.dw_ubaddr[0], prior_box_gm[prior_offset + max_rois_num*2*self.num_class],
                0, cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)

            self.tik_instance.data_move(self.dh_ubaddr[0],
                                        prior_box_gm[prior_offset + max_rois_num
                                                     * 3*self.num_class], 0,
                                        cur_process_num//Constant.FP16_ALIGN_NUM,
                                        self.ratio, 0, 0)

            vec_sub((self.tik_instance, self.x2_ubaddr, self.x1_ubaddr,
                     self.input_dtype), self.ubaddr0, cur_process_num)

            vec_sub((self.tik_instance, self.y2_ubaddr, self.y1_ubaddr,
                     self.input_dtype), self.ubaddr1, cur_process_num)

            temp = Constant.DATA_ONE
            vec_adds((self.tik_instance, self.ubaddr0, self.input_dtype),
                     self.x2_ubaddr, temp, cur_process_num)

            temp = Constant.DATA_ONE
            vec_adds((self.tik_instance, self.ubaddr1, self.input_dtype),
                     self.y2_ubaddr, temp, cur_process_num)

            temp = 0.5
            vec_muls((self.tik_instance, self.x2_ubaddr, self.input_dtype),
                     self.ubaddr0, temp, cur_process_num)

            temp = 0.5
            vec_muls((self.tik_instance, self.y2_ubaddr, self.input_dtype),
                     self.ubaddr1, temp, cur_process_num)

            vec_add((self.tik_instance, self.ubaddr0, self.x1_ubaddr,
                     self.input_dtype), self.x1_ubaddr, cur_process_num)

            vec_add((self.tik_instance, self.ubaddr1, self.y1_ubaddr,
                     self.input_dtype), self.y1_ubaddr, cur_process_num)

            vec_mla((self.tik_instance, self.dx_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.x1_ubaddr, cur_process_num)

            vec_mla((self.tik_instance, self.dy_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.y1_ubaddr, cur_process_num)

            vec_exp((self.tik_instance, self.dw_ubaddr, self.input_dtype),
                    self.dx_ubaddr, cur_process_num)

            vec_exp((self.tik_instance, self.dh_ubaddr, self.input_dtype),
                    self.dy_ubaddr, cur_process_num)

            vec_mul((self.tik_instance, self.dx_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.x2_ubaddr, cur_process_num)

            vec_mul((self.tik_instance, self.dy_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.y2_ubaddr, cur_process_num)

            temp = 0.5
            vec_muls((self.tik_instance, self.x2_ubaddr, self.input_dtype),
                     self.x2_ubaddr, temp, cur_process_num)

            temp = 0.5
            vec_muls((self.tik_instance, self.y2_ubaddr, self.input_dtype),
                     self.y2_ubaddr, temp, cur_process_num)

            vec_sub((self.tik_instance, self.x1_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.dx_ubaddr, cur_process_num)

            vec_sub((self.tik_instance, self.y1_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.dy_ubaddr, cur_process_num)

            vec_add((self.tik_instance, self.x1_ubaddr, self.x2_ubaddr,
                     self.input_dtype), self.dw_ubaddr, cur_process_num)

            vec_add((self.tik_instance, self.y1_ubaddr, self.y2_ubaddr,
                     self.input_dtype), self.dh_ubaddr, cur_process_num)

            self.tik_instance.vadds(16//self.ratio, self.im_info_ub,
                                    self.im_info_ub, -1, Constant.DATA_ONE, Constant.DATA_ONE, Constant.DATA_ONE, 0, 0)

            #clip

            im_scalar = self.tik_instance.Scalar(dtype=self.input_dtype)
            im_scalar.set_as(self.im_info_ub[1])

            vec_dup((self.tik_instance, cur_process_num,
                     self.input_dtype), self.ubaddr0, im_scalar)
            im_scalar.set_as(self.im_info_ub[0])
            vec_dup((self.tik_instance, cur_process_num,
                     self.input_dtype), self.ubaddr1, im_scalar)

            vec_min((self.tik_instance, self.dx_ubaddr, self.ubaddr0,
                     self.input_dtype), self.dx_ubaddr, cur_process_num)

            vec_min((self.tik_instance, self.dy_ubaddr, self.ubaddr1,
                     self.input_dtype), self.dy_ubaddr, cur_process_num)

            vec_min((self.tik_instance, self.dw_ubaddr, self.ubaddr0,
                     self.input_dtype), self.dw_ubaddr, cur_process_num)

            vec_min((self.tik_instance, self.dh_ubaddr, self.ubaddr1,
                     self.input_dtype), self.dh_ubaddr, cur_process_num)

            vec_relu((self.tik_instance, self.dx_ubaddr, self.input_dtype),
                     self.x1_ubaddr, cur_process_num)

            vec_relu((self.tik_instance, self.dy_ubaddr, self.input_dtype),
                     self.y1_ubaddr, cur_process_num)

            vec_relu((self.tik_instance, self.dw_ubaddr, self.input_dtype),
                     self.x2_ubaddr, cur_process_num)

            vec_relu((self.tik_instance, self.dh_ubaddr, self.input_dtype),
                     self.y2_ubaddr, cur_process_num)

            self.tik_instance.data_move(self.ubaddr0, score_gm[score_offset],
                                        0, cur_process_num//Constant.FP16_ALIGN_NUM,
                                        self.ratio, 0, 0)

            is_rpn_supported = check_soc_version()
            if is_rpn_supported:

                vec_concat((self.tik_instance, self.x1_ubaddr, self.input_dtype),
                        self.output_region_proposal_ub, cur_process_num, 0)
                vec_concat((self.tik_instance, self.y1_ubaddr, self.input_dtype),
                        self.output_region_proposal_ub, cur_process_num, 1)
                vec_concat((self.tik_instance, self.x2_ubaddr, self.input_dtype),
                        self.output_region_proposal_ub, cur_process_num, 2)
                vec_concat((self.tik_instance, self.y2_ubaddr, self.input_dtype),
                        self.output_region_proposal_ub, cur_process_num, 3)
                vec_concat((self.tik_instance, self.ubaddr0, self.input_dtype),
                        self.output_region_proposal_ub, cur_process_num, 4)

                self.tik_instance.data_move(
                    output_region_proposal[score_offset*constant.DATA_SIZE_EIGHT],
                    self.output_region_proposal_ub, 0,
                    cur_process_num*constant.DATA_SIZE_EIGHT//Constant.FP16_ALIGN_NUM,
                    self.ratio, 0, 0)
            
            else:
                self.tik_instance.data_move(scores_gm[cur_batch_index, class_idx, 0], self.ubaddr0, 0, 
                                            cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)
                self.tik_instance.data_move(x1y1x2y2_gm[cur_batch_index, 0, class_idx, 0], self.x1_ubaddr, 0, 
                                            cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)
                self.tik_instance.data_move(x1y1x2y2_gm[cur_batch_index, 1, class_idx, 0], self.y1_ubaddr, 0, 
                                            cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)
                self.tik_instance.data_move(x1y1x2y2_gm[cur_batch_index, 2, class_idx, 0], self.x2_ubaddr, 0, 
                                            cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)
                self.tik_instance.data_move(x1y1x2y2_gm[cur_batch_index, 3, class_idx, 0], self.y2_ubaddr, 0, 
                                            cur_process_num//Constant.FP16_ALIGN_NUM, self.ratio, 0, 0)



class OneCoreProcess:
    """
    One Core Process
    """
    def __init__(self, input_data):
        """
        :param input_data:
        """
        self.tik_instance = input_data[0]
        self.input_dtype = input_data[1]
        self.input_shape = input_data[2]
        self.size = input_data[3]
        self.batch_rois = input_data[4]
        self.image_info = input_data[5]
        self.num_classes = input_data[6]
        self.max_rois_num = input_data[7]

        self.total_num = input_data[8]
        self.device_cor_num = input_data[9]
        self.batch_factor = input_data[10]
        self.block_id = input_data[11]
        self.ub_size = input_data[12]
        self.total_num = self.input_shape[0]
        self.is_rpn_supported = check_soc_version()

        self.x1y1x2y2_gm = input_data[13]
        self.scores_gm = input_data[14]

        cal_var_num = 18
        tmp_size = self.batch_rois * Constant.DATA_EIGHT * Constant.DATA_EIGHT
        self.total_size = cal_var_num*self.max_rois_num*self.size*self.num_classes+tmp_size
        self.one_batch_one_class_size = cal_var_num*self.max_rois_num*self.size
        # one time need space

        self.reserved_ub_size = cal_var_num*Constant.FP16_ALIGN_NUM*self.size

    def get_offset(self, batch_index):
        """
        :param input_offset:
        :return:
        """
        rois_offset = batch_index*self.max_rois_num*5+self.max_rois_num
        prior_offset = self.num_classes*batch_index*self.max_rois_num*4
        score_offset = self.num_classes*batch_index*self.max_rois_num
        return rois_offset, prior_offset, score_offset

    def get_tiling_branch1_offset(self, batch_index, class_index):
        """
        :param batch_index:
        :param class_index:
        :return:
        """
        rois_offset = batch_index * self.max_rois_num * 5 + self.max_rois_num
        prior_offset = class_index*self.max_rois_num + \
                       batch_index*self.max_rois_num*4*self.num_classes
        score_offset = class_index*self.max_rois_num + \
                       batch_index*self.max_rois_num*self.num_classes
        return rois_offset, prior_offset, score_offset

    def get_tiling_branch2_offset(self, input_list):
        """
        :param input_list:
        :return:
        """
        batch_index = input_list[0]
        class_index = input_list[1]
        tiling_process = input_list[2]
        tiling_loop = input_list[3]
        rois_offset = batch_index*self.max_rois_num*5 + \
                      self.max_rois_num+tiling_process*Constant.FP16_ALIGN_NUM*tiling_loop
        prior_offset = self.num_classes*batch_index * \
                       self.max_rois_num*4+class_index*self.max_rois_num*4 + \
                       tiling_process*Constant.FP16_ALIGN_NUM*tiling_loop
        score_offset = self.num_classes*batch_index * \
                       self.max_rois_num+class_index*self.max_rois_num+tiling_process * \
                       Constant.FP16_ALIGN_NUM*tiling_loop
        return rois_offset, prior_offset, score_offset

    def get_tiling_tail_offset(self, input_list):
        """
        :param input_list:
        :return:
        """
        batch_index = input_list[0]
        class_index = input_list[1]
        tiling_process = input_list[2]
        tiling_num = input_list[3]
        rois_offset = batch_index*self.max_rois_num*5+self.max_rois_num + \
                      tiling_process*Constant.FP16_ALIGN_NUM*tiling_num
        prior_offset = self.num_classes*batch_index*self.max_rois_num*4 + \
                       class_index*self.max_rois_num*4+tiling_process*Constant.FP16_ALIGN_NUM*tiling_num
        score_offset = self.num_classes*batch_index*self.max_rois_num + \
                       class_index*self.max_rois_num+tiling_process*Constant.FP16_ALIGN_NUM*tiling_num
        return rois_offset, prior_offset, score_offset

    def one_core_process_decode_rois(self, input_list, output_box):
        """
        :param block_id:
        :param class_index:
        :param batch_index:
        :param rois:
        :param actual_rois_num:
        :param prior_box_gm:
        :param score_gm:
        :param output_box:
        :return:
        """

        batch_index = input_list[0]
        rois = input_list[1]
        input_tensor = input_list[2]
        self.actual_rois_num_effect = input_tensor[0]
        prior_box_gm = input_list[3]
        score_gm = input_list[4]
        cur_batch_index = batch_index
        if self.actual_rois_num_effect:
            actual_rois_num = input_tensor[1]
            actual_rois_num_ub = self.tik_instance.Tensor(
                "int32", (self.batch_rois, Constant.DATA_EIGHT),
                name="actual_rois_num_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(actual_rois_num_ub, actual_rois_num, 0,
                                        self.batch_rois, Constant.DATA_ONE, 0, 0)

            cur_batch_num = self.tik_instance.Scalar("int32")
            cur_batch_num.set_as(
                actual_rois_num_ub[cur_batch_index*Constant.DATA_EIGHT])
        else:
            cur_batch_num = self.max_rois_num

        if self.ub_size >= self.total_size:
            rois_offset, prior_offset, score_offset = \
                self.get_offset(batch_index)
            decode_rois_object = DecodeRois(
                self.tik_instance, (self.max_rois_num, self.input_dtype,
                                    self.image_info, self.num_classes), False)

            decode_rois_object.generate_rois(
                (self.max_rois_num, rois_offset, prior_offset,
                 score_offset, score_gm, prior_box_gm,
                 rois, self.max_rois_num, self.x1y1x2y2_gm, self.scores_gm), cur_batch_index,
                output_box)

        else:
            with self.tik_instance.for_range(Constant.DATA_ONE, self.num_classes) \
                    as class_index:
                if self.ub_size >= self.one_batch_one_class_size:
                    rois_offset, prior_offset, score_offset = \
                        self.get_tiling_branch1_offset(batch_index, class_index)
                    decode_rois_object = DecodeRois(
                        self.tik_instance, (self.max_rois_num, self.input_dtype,
                                            self.image_info, self.num_classes), True)
                    decode_rois_object.tiling_generate_rois(
                        (self.max_rois_num, rois_offset, prior_offset,
                         score_offset, score_gm, prior_box_gm,
                         rois, self.max_rois_num,
                         self.x1y1x2y2_gm, self.scores_gm, class_index), cur_batch_index,
                        output_box)
                else:
                    reserved_space = self.batch_rois*constant.DATA_SIZE_EIGHT*4
                    tiling_process = \
                        (self.ub_size-reserved_space) // self.reserved_ub_size
                    tiling_num = self.max_rois_num//(tiling_process*Constant.FP16_ALIGN_NUM)

                    tiling_tail = self.max_rois_num - \
                                  tiling_num*tiling_process*Constant.FP16_ALIGN_NUM

                    with self.tik_instance.for_range(0, tiling_num) as tiling_loop:
                        with self.tik_instance.if_scope(cur_batch_num >= tiling_process
                                                        * Constant.FP16_ALIGN_NUM * tiling_loop):

                            rois_offset, prior_offset, score_offset = \
                                self.get_tiling_branch2_offset((batch_index, class_index,
                                                                tiling_process, tiling_loop))
                            decode_rois_object = DecodeRois(
                                self.tik_instance, (tiling_process*Constant.FP16_ALIGN_NUM,
                                                    self.input_dtype, self.image_info,
                                                    self.num_classes), True)
                            decode_rois_object.tiling_generate_rois(
                                (tiling_process*Constant.FP16_ALIGN_NUM, rois_offset,
                                 prior_offset, score_offset, score_gm,
                                 prior_box_gm, rois, self.max_rois_num,
                                 self.x1y1x2y2_gm, self.scores_gm, class_index),
                                cur_batch_index, output_box)
                    with self.tik_instance.if_scope(tiling_tail > 0):
                        with self.tik_instance.if_scope(cur_batch_num >= tiling_process
                                                        * Constant.FP16_ALIGN_NUM * tiling_num):
                            rois_offset, prior_offset, score_offset = \
                                self.get_tiling_tail_offset(
                                    (batch_index, class_index, tiling_process,
                                     tiling_num))
                            decode_rois_object = DecodeRois(
                                self.tik_instance, (tiling_tail, self.input_dtype,
                                                    self.image_info, self.num_classes), True)
                            decode_rois_object.tiling_generate_rois((tiling_tail,
                                                                     rois_offset, prior_offset,
                                                                     score_offset,
                                                                     score_gm, prior_box_gm,
                                                                     rois, self.max_rois_num,
                                                                     self.x1y1x2y2_gm, self.scores_gm, class_index),
                                                                    cur_batch_index,
                                                                    output_box)


class PreProcess:
    """
    PreProcess
    """
    def __init__(self, input_data, input_tensor):
        """
        :param input_data:
        """
        self.tik_instance = input_data[0]
        self.max_rois_num = input_data[1]
        self.num_classes = input_data[2]
        self.batch_rois = input_data[3]
        self.input_shape = input_data[4]
        self.input_dtype = input_data[5]

        self.size, self.mask, self.ratio = get_params(self.input_dtype)
        self.actual_rois_num_effect = input_tensor[0]
        if self.actual_rois_num_effect:
            actual_rois_num = input_tensor[1]
            self.actual_rois_num_ub = self.tik_instance.Tensor(
                "int32", (self.batch_rois, Constant.DATA_EIGHT),
                name="actual_rois_num_ub", scope=tik.scope_ubuf)
            rois_act_dup_times = self.batch_rois//Constant.MAX_REPEAT_TIME
            rois_act_tail = self.batch_rois - rois_act_dup_times*Constant.MAX_REPEAT_TIME
            with self.tik_instance.for_range(0, rois_act_dup_times) \
                    as rois_act_loop:
                self.tik_instance.vector_dup(
                    Constant.DATA_EIGHT,
                    self.actual_rois_num_ub[Constant.MAX_REPEAT_TIME *
                                            rois_act_loop*Constant.DATA_EIGHT],
                    0, Constant.MAX_REPEAT_TIME, 0, Constant.DATA_ONE)
            if rois_act_tail != 0:
                self.tik_instance.vector_dup(
                    Constant.DATA_EIGHT,
                    self.actual_rois_num_ub[
                        Constant.MAX_REPEAT_TIME*Constant.DATA_EIGHT *
                        rois_act_dup_times], 0, rois_act_tail, 0, Constant.DATA_ONE)

            self.tik_instance.data_move(self.actual_rois_num_ub, actual_rois_num, 0,
                                        self.batch_rois, Constant.DATA_ONE, 0, 0)
            self.actual_sum_num = self.tik_instance.Tensor(
                "int32", (self.batch_rois, Constant.DATA_EIGHT),
                name="actual_sum_num", scope=tik.scope_ubuf)

            vec_dup((self.tik_instance,
                     self.batch_rois*Constant.DATA_EIGHT, "float32"),
                    self.actual_sum_num)

            with self.tik_instance.for_range(Constant.DATA_ONE, self.batch_rois) as batch_index:
                self.tik_instance.vadd(Constant.BLOCK_SIZE//4,
                                       self.actual_sum_num[
                                           batch_index*Constant.DATA_EIGHT],
                                       self.actual_sum_num[
                                           (batch_index-Constant.DATA_ONE) *
                                           Constant.DATA_EIGHT],
                                       self.actual_rois_num_ub[
                                           (batch_index-Constant.DATA_ONE) *
                                           Constant.DATA_EIGHT],
                                       Constant.DATA_ONE, Constant.DATA_ONE, Constant.DATA_ONE,
                                       Constant.DATA_ONE, Constant.DATA_ONE,
                                       Constant.DATA_ONE, Constant.DATA_ONE,)

    def trans(self, src_ub, dst_ub, length):
        """
        :param src_ub:
        :param dst_ub:
        :param length:
        :return:
        """
        if self.input_dtype == "float16":
            vnch_loop_times = ((length*Constant.FP16_ALIGN_NUM) //
                               Constant.VECTOR_BLOCK_SIZE)//Constant.MAX_REPEAT_TIME
            tail_loop_times = ((length*Constant.FP16_ALIGN_NUM) //
                               Constant.VECTOR_BLOCK_SIZE) % Constant.MAX_REPEAT_TIME
            with self.tik_instance.for_range(0, vnch_loop_times) as vnch_loop:
                src_list = \
                    [src_ub[Constant.FP16_ALIGN_NUM*i+vnch_loop *
                            Constant.MAX_REPEAT_TIME*Constant.VECTOR_BLOCK_SIZE]
                     for i in range(Constant.FP16_ALIGN_NUM)]
                dst_list = [dst_ub[Constant.FP16_ALIGN_NUM*i+vnch_loop *
                                   Constant.MAX_REPEAT_TIME*Constant.VECTOR_BLOCK_SIZE]
                            for i in range(Constant.FP16_ALIGN_NUM)]
                self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                            Constant.MAX_REPEAT_TIME, Constant.FP16_ALIGN_NUM,
                                            Constant.FP16_ALIGN_NUM)
            src_list = [src_ub[Constant.FP16_ALIGN_NUM*i+vnch_loop_times *
                               Constant.MAX_REPEAT_TIME*Constant.VECTOR_BLOCK_SIZE]
                        for i in range(Constant.FP16_ALIGN_NUM)]
            dst_list = [dst_ub[Constant.FP16_ALIGN_NUM*i+vnch_loop_times *
                               Constant.MAX_REPEAT_TIME*Constant.VECTOR_BLOCK_SIZE]
                        for i in range(Constant.FP16_ALIGN_NUM)]
            if tail_loop_times == Constant.DATA_ONE:
                self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                            tail_loop_times, 0, 0)
            else:
                self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                            tail_loop_times, Constant.FP16_ALIGN_NUM,
                                            Constant.FP16_ALIGN_NUM)
        elif self.input_dtype == "float32":
            src_list = [src_ub[i*Constant.FP16_ALIGN_NUM] for i in range(Constant.FP16_ALIGN_NUM)]
            dst_list = [dst_ub[i//2*Constant.FP16_ALIGN_NUM +
                               (i % 2) *
                               Constant.DATA_EIGHT]
                        for i in range(Constant.FP16_ALIGN_NUM)]
            if (length*Constant.FP16_ALIGN_NUM) // Constant.VECTOR_BLOCK_SIZE == Constant.DATA_ONE:
                self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                            (length*Constant.FP16_ALIGN_NUM) //
                                            Constant.VECTOR_BLOCK_SIZE,
                                            0, 0)
            else:
                self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                            (length*Constant.FP16_ALIGN_NUM) //
                                            Constant.VECTOR_BLOCK_SIZE,
                                            Constant.BLOCK_SIZE,
                                            Constant.BLOCK_SIZE)
            src_list = [src_ub[i*Constant.FP16_ALIGN_NUM+Constant.DATA_EIGHT]
                        for i in range(Constant.FP16_ALIGN_NUM)]
            dst_list = [dst_ub[i//2*Constant.FP16_ALIGN_NUM +
                               (i % 2) *
                               Constant.DATA_EIGHT +
                               Constant.DATA_EIGHT*Constant.FP16_ALIGN_NUM]
                        for i in range(Constant.FP16_ALIGN_NUM)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                        (length*Constant.FP16_ALIGN_NUM) //
                                        Constant.VECTOR_BLOCK_SIZE,
                                        Constant.BLOCK_SIZE,
                                        Constant.BLOCK_SIZE)

    def no_tiling_trans_score(self, input_list, score, score_gm):
        shape = input_list[0]
        sum_addr = input_list[1]
        cur_batch_index = input_list[2]
        cur_batch_num = input_list[3]
        align_num_times = shape[0]
        score_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                            name="score_ub",
                                            scope=tik.scope_ubuf)
        new_score_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                                name="new_score_ub",
                                                scope=tik.scope_ubuf)

        vec_dup((self.tik_instance, align_num_times * Constant.FP16_ALIGN_NUM * self.max_rois_num,
                 self.input_dtype), new_score_ub)

        self.tik_instance.data_move(
            new_score_ub[0],
            score[sum_addr*align_num_times*Constant.FP16_ALIGN_NUM],
            0, cur_batch_num*align_num_times, self.ratio,
            0, 0)

        vec_dup((self.tik_instance, align_num_times * Constant.FP16_ALIGN_NUM * self.max_rois_num,
                 self.input_dtype), score_ub)

        with self.tik_instance.for_range(0, align_num_times) as loop:
            self.tik_instance.data_move(
                score_ub[loop*cur_batch_num*Constant.FP16_ALIGN_NUM],
                new_score_ub[loop*Constant.FP16_ALIGN_NUM], 0, cur_batch_num, self.ratio,
                (align_num_times - Constant.DATA_ONE) * self.ratio, 0)

        self.trans(score_ub, new_score_ub,
                   score_ub.shape[0]*score_ub.shape[2])

        with self.tik_instance.for_range(0, self.num_classes // Constant.FP16_ALIGN_NUM) as loop:
            with self.tik_instance.for_range(0, Constant.FP16_ALIGN_NUM) as inner_loop:
                self.tik_instance.data_move(
                    score_ub[loop*self.max_rois_num*Constant.FP16_ALIGN_NUM +
                             inner_loop*self.max_rois_num],
                    new_score_ub[inner_loop*Constant.FP16_ALIGN_NUM +
                                 loop*self.max_rois_num*Constant.FP16_ALIGN_NUM], 0,
                    self.max_rois_num//Constant.FP16_ALIGN_NUM, self.ratio,
                    15*self.ratio, 0)

        loop_times = self.num_classes % Constant.FP16_ALIGN_NUM
        with self.tik_instance.for_range(0, loop_times) as inner_loop:
            self.tik_instance.data_move(
                score_ub[(self.num_classes//Constant.FP16_ALIGN_NUM)*self.max_rois_num*Constant.FP16_ALIGN_NUM +
                         inner_loop*self.max_rois_num],
                new_score_ub[inner_loop*Constant.FP16_ALIGN_NUM +
                             (self.num_classes//Constant.FP16_ALIGN_NUM)*self.max_rois_num*Constant.FP16_ALIGN_NUM],
                0, self.max_rois_num//Constant.FP16_ALIGN_NUM, self.ratio,
                15*self.ratio, 0)
        self.tik_instance.data_move(
            score_gm[cur_batch_index*self.max_rois_num*self.num_classes],
            score_ub, 0,
            self.max_rois_num//Constant.FP16_ALIGN_NUM*self.num_classes, self.ratio,
            0, 0)

    def tiling_trans_score_branch1(self, input_list, score, score_gm):
        shape = input_list[0]
        sum_addr = input_list[1]
        class_loop = input_list[2]
        cur_batch_index = input_list[3]
        cur_batch_num = input_list[4]

        score_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                            name="score_ub",
                                            scope=tik.scope_ubuf)
        new_score_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                                name="new_score_ub",
                                                scope=tik.scope_ubuf)

        vec_dup((self.tik_instance, shape[0]*shape[1],
                 self.input_dtype), score_ub)
        vec_dup((self.tik_instance, shape[0]*shape[1],
                 self.input_dtype), new_score_ub)

        self.tik_instance.data_move(
            score_ub[0],
            score[sum_addr*((self.num_classes+Constant.TO_ALIGN_NUM) //
                            Constant.FP16_ALIGN_NUM)*Constant.FP16_ALIGN_NUM +
                  ((class_loop+Constant.TO_ALIGN_NUM)//Constant.FP16_ALIGN_NUM) *
                  Constant.FP16_ALIGN_NUM],
            0, cur_batch_num, self.ratio,
            ((self.num_classes+Constant.TO_ALIGN_NUM) //
             Constant.FP16_ALIGN_NUM-Constant.DATA_ONE)*self.ratio, 0)
        self.trans(score_ub, new_score_ub, shape[0])
        vec_dup((self.tik_instance, shape[0]*shape[1],
                 self.input_dtype), score_ub)
        with self.tik_instance.for_range(0, Constant.FP16_ALIGN_NUM) as loop:
            with self.tik_instance.if_scope(
                    (class_loop*Constant.FP16_ALIGN_NUM+loop) < self.num_classes):
                self.tik_instance.data_move(
                    score_ub[loop*self.max_rois_num],
                    new_score_ub[loop*Constant.FP16_ALIGN_NUM], 0,
                    self.max_rois_num//Constant.FP16_ALIGN_NUM, self.ratio,
                    15*self.ratio, 0)
        with self.tik_instance.for_range(0, Constant.FP16_ALIGN_NUM) as loop:
            with self.tik_instance.if_scope(
                    class_loop*Constant.FP16_ALIGN_NUM+loop < self.num_classes):
                with self.tik_instance.if_scope(
                        (class_loop*Constant.FP16_ALIGN_NUM+loop) <
                        self.num_classes):
                    score_gm_offset = \
                        ((class_loop+Constant.TO_ALIGN_NUM)//Constant.FP16_ALIGN_NUM) * \
                        Constant.FP16_ALIGN_NUM * \
                        self.max_rois_num+(class_loop -
                                           (class_loop+Constant.TO_ALIGN_NUM) //
                                           Constant.FP16_ALIGN_NUM) * \
                        self.max_rois_num + \
                        cur_batch_index*self.max_rois_num*self.num_classes+loop * \
                        self.max_rois_num

                    self.tik_instance.data_move(
                        score_gm[score_gm_offset],
                        score_ub[loop*self.max_rois_num], 0,
                        self.max_rois_num//Constant.FP16_ALIGN_NUM, self.ratio,
                        0, 0)

    def tiling_trans_score_branch2(self, input_list, score, score_gm):
        shape = input_list[0]
        sum_addr = input_list[1]
        class_loop = input_list[2]
        cur_batch_index = input_list[3]
        cur_batch_num = input_list[4]
        one_tiling_process = input_list[5]
        score_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                            name="score_ub",
                                            scope=tik.scope_ubuf)
        new_score_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                                name="new_score_ub",
                                                scope=tik.scope_ubuf)
        vec_dup((self.tik_instance, shape[0]*shape[1],
                 self.input_dtype), score_ub)
        vec_dup((self.tik_instance, shape[0]*shape[1],
                 self.input_dtype), new_score_ub)

        one_batch_loop_time = cur_batch_num // \
                              (one_tiling_process//Constant.FP16_ALIGN_NUM)
        with self.tik_instance.for_range(0, one_batch_loop_time) \
                as inner_batch_loop:
            with self.tik_instance.if_scope(
                    cur_batch_num >= (inner_batch_loop *
                                      (one_tiling_process // Constant.FP16_ALIGN_NUM))):
                vec_dup((self.tik_instance, shape[0]*shape[1],
                         self.input_dtype), score_ub)
                vec_dup((self.tik_instance, shape[0]*shape[1],
                         self.input_dtype), new_score_ub)
                self.tik_instance.data_move(
                    score_ub,
                    score[sum_addr+inner_batch_loop *
                          (one_tiling_process//Constant.FP16_ALIGN_NUM),
                          class_loop, 0, 0, 0],
                    0, one_tiling_process//Constant.FP16_ALIGN_NUM, self.ratio,
                    ((self.num_classes+Constant.TO_ALIGN_NUM)//Constant.FP16_ALIGN_NUM-Constant.DATA_ONE) *
                    self.ratio, 0)
                self.trans(score_ub, new_score_ub, shape[0])
                vec_dup((self.tik_instance, shape[0]*shape[1],
                         self.input_dtype), score_ub)
                with self.tik_instance.for_range(0, Constant.FP16_ALIGN_NUM) as loop:
                    self.tik_instance.data_move(
                        score_ub[loop*(one_tiling_process//Constant.FP16_ALIGN_NUM)],
                        new_score_ub[loop*Constant.FP16_ALIGN_NUM], 0,
                        (one_tiling_process//Constant.FP16_ALIGN_NUM)//Constant.FP16_ALIGN_NUM,
                        self.ratio, 15*self.ratio, 0)
                with self.tik_instance.for_range(0, Constant.FP16_ALIGN_NUM) as loop:
                    with self.tik_instance.if_scope(
                            class_loop*Constant.FP16_ALIGN_NUM+loop < self.num_classes):
                        self.tik_instance.data_move(
                            score_gm[cur_batch_index, class_loop*Constant.FP16_ALIGN_NUM +
                                     loop,
                                     inner_batch_loop *
                                     ((one_tiling_process//Constant.FP16_ALIGN_NUM) //
                                      Constant.FP16_ALIGN_NUM),
                                     0, 0],
                            score_ub[loop*(one_tiling_process//Constant.FP16_ALIGN_NUM)],
                            0, (one_tiling_process//Constant.FP16_ALIGN_NUM) //
                            Constant.FP16_ALIGN_NUM, self.ratio,
                            0, 0)
        with self.tik_instance.if_scope(
                cur_batch_num-one_batch_loop_time*(one_tiling_process //
                                                   Constant.FP16_ALIGN_NUM) > 0):
            shape = [one_tiling_process//Constant.FP16_ALIGN_NUM, Constant.FP16_ALIGN_NUM]
            self.tik_instance.data_move(
                score_ub[0],
                score[sum_addr+one_batch_loop_time *
                      (one_tiling_process//Constant.FP16_ALIGN_NUM),
                      class_loop, 0, 0, 0],
                0, cur_batch_num-one_batch_loop_time*(one_tiling_process //
                                                      Constant.FP16_ALIGN_NUM),
                self.ratio,
                ((self.num_classes+Constant.TO_ALIGN_NUM)//Constant.FP16_ALIGN_NUM-Constant.DATA_ONE)*self.ratio,
                0)
            self.trans(score_ub, new_score_ub, shape[0])
            vec_dup((self.tik_instance, shape[0]*shape[1],
                     self.input_dtype), score_ub)
            with self.tik_instance.for_range(0, Constant.FP16_ALIGN_NUM) as loop:
                self.tik_instance.data_move(
                    score_ub[loop*(one_tiling_process//Constant.FP16_ALIGN_NUM)],
                    new_score_ub[loop*Constant.FP16_ALIGN_NUM], 0,
                    one_tiling_process//(Constant.FP16_ALIGN_NUM*Constant.FP16_ALIGN_NUM),
                    self.ratio, 15*self.ratio, 0)
            with self.tik_instance.for_range(0, Constant.FP16_ALIGN_NUM) as loop:
                with self.tik_instance.if_scope(
                        class_loop*Constant.FP16_ALIGN_NUM+loop < self.num_classes):
                    self.tik_instance.data_move(
                        score_gm[cur_batch_index,
                                 class_loop*Constant.FP16_ALIGN_NUM+loop,
                                 one_batch_loop_time *
                                 (one_tiling_process //
                                  (Constant.FP16_ALIGN_NUM*Constant.FP16_ALIGN_NUM)), 0, 0],
                        score_ub[loop*(one_tiling_process//Constant.FP16_ALIGN_NUM)],
                        0, (cur_batch_num-one_batch_loop_time *
                            (one_tiling_process//Constant.FP16_ALIGN_NUM) +
                            Constant.TO_ALIGN_NUM)//Constant.FP16_ALIGN_NUM, self.ratio,
                        0, 0)

    def trans_score(self, score, score_gm, cur_batch_index):
        """
        :param score:
        :param score_gm:
        :param cur_batch_index:
        :return:
        """
        align_num_times = (self.num_classes+15) // Constant.FP16_ALIGN_NUM
        ub_size = get_ub_size()

        one_batch_size = self.max_rois_num * Constant.FP16_ALIGN_NUM * self.size * \
                         align_num_times * 2 + self.batch_rois * Constant.DATA_EIGHT * 4 * 2

        if self.max_rois_num*self.batch_rois != score.shape[0]:
            if self.actual_rois_num_effect == False:
                error_manager_vector.raise_err_inputs_shape_not_equal("fsr_detection_output",
                                                                      "self.max_rois_num*self.batch_rois",
                                                                      "score.shape[0]",
                                                                      self.max_rois_num*self.batch_rois,
                                                                      score.shape[0],
                                                                      score.shape[0])
            sum_addr = self.tik_instance.Scalar("int32")
            sum_addr.set_as(0)
            cur_batch_num = self.tik_instance.Scalar("int32")
            cur_batch_num.set_as(0)
            sum_addr.set_as(self.actual_sum_num[
                                cur_batch_index*Constant.DATA_EIGHT])
            cur_batch_num.set_as(
                self.actual_rois_num_ub[
                    cur_batch_index*Constant.DATA_EIGHT])
        else:
            cur_batch_num = self.max_rois_num
            sum_addr = cur_batch_index*self.max_rois_num

        if one_batch_size < ub_size:
            shape = [align_num_times, Constant.FP16_ALIGN_NUM, self.max_rois_num]
            self.no_tiling_trans_score((shape, sum_addr,
                                        cur_batch_index, cur_batch_num),
                                       score, score_gm)
        else:
            num_piece_of_space = \
                ((ub_size-self.batch_rois*Constant.DATA_EIGHT*4*2) //
                 2) // ((Constant.FP16_ALIGN_NUM*Constant.FP16_ALIGN_NUM)*self.size)
            one_tiling_process = num_piece_of_space*Constant.FP16_ALIGN_NUM*Constant.FP16_ALIGN_NUM
            if one_tiling_process//Constant.FP16_ALIGN_NUM < self.max_rois_num:
                shape = [one_tiling_process//Constant.FP16_ALIGN_NUM, Constant.FP16_ALIGN_NUM]
            else:
                shape = [self.max_rois_num, Constant.FP16_ALIGN_NUM]
            with self.tik_instance.for_range(0,
                                             (self.num_classes+Constant.FP16_ALIGN_NUM) //
                                             Constant.FP16_ALIGN_NUM) as class_loop:
                if one_tiling_process >= self.max_rois_num*Constant.FP16_ALIGN_NUM:
                    self.tiling_trans_score_branch1((shape, sum_addr, class_loop,
                                                     cur_batch_index, cur_batch_num),
                                                    score, score_gm)
                else:
                    self.tiling_trans_score_branch2((shape, sum_addr, class_loop,
                                                     cur_batch_index, cur_batch_num,
                                                     one_tiling_process), score, score_gm)

    def prior_ub_move(self, loop, length, dst_ub, src_ub):
        """
        :param loop:
        :param length:
        :param dst_ub:
        :param src_ub:
        :return:
        """
        self.tik_instance.data_move(dst_ub[loop*length*4],
                                    src_ub[loop*Constant.FP16_ALIGN_NUM*4], 0,
                                    length//Constant.FP16_ALIGN_NUM, self.ratio, 15*self.ratio, 0)
        self.tik_instance.data_move(
            dst_ub[loop*length*4+length],
            src_ub[loop*Constant.FP16_ALIGN_NUM*4+Constant.FP16_ALIGN_NUM], 0,
            length//Constant.FP16_ALIGN_NUM, self.ratio, 15*self.ratio, 0)
        self.tik_instance.data_move(
            dst_ub[loop*length*4+length*2],
            src_ub[loop*Constant.FP16_ALIGN_NUM*4+Constant.FP16_ALIGN_NUM*2],
            0, length//Constant.FP16_ALIGN_NUM, self.ratio, 15*self.ratio, 0)
        self.tik_instance.data_move(
            dst_ub[loop*length*4+length*3],
            src_ub[loop*Constant.FP16_ALIGN_NUM*4+Constant.FP16_ALIGN_NUM*3], 0,
            length//Constant.FP16_ALIGN_NUM, self.ratio, 15*self.ratio, 0)

    def prior_gm_move(self, dst_list, src_list, repeat_times):
        """
        :param dst_list:
        :param src_list:
        :param repeat_times:
        :return:
        """
        prior_gm_offset = dst_list[0]
        prior_box_gm = dst_list[1]
        base_offset = dst_list[2]
        prior_ub_offset = src_list[0]
        prior_ub = src_list[1]
        ub_base_offset = src_list[2]
        self.tik_instance.data_move(prior_box_gm[prior_gm_offset],
                                    prior_ub[prior_ub_offset], 0,
                                    repeat_times, self.ratio, 0, 0)
        self.tik_instance.data_move(prior_box_gm[prior_gm_offset+base_offset],
                                    prior_ub[prior_ub_offset+ub_base_offset],
                                    0, repeat_times, self.ratio, 0, 0)
        self.tik_instance.data_move(
            prior_box_gm[prior_gm_offset+base_offset*2],
            prior_ub[prior_ub_offset+ub_base_offset*2],
            0, repeat_times, self.ratio, 0, 0)
        self.tik_instance.data_move(prior_box_gm[prior_gm_offset+base_offset*3],
                                    prior_ub[prior_ub_offset+ub_base_offset*3],
                                    0, repeat_times, self.ratio, 0, 0)

    def no_tiling_trans_prior_box(self, input_list, prior, prior_box_gm):
        shape = input_list[0]
        sum_addr = input_list[1]
        cur_batch_index = input_list[2]
        cur_batch_num = input_list[3]
        prior_box_align = 4
        align_times = (self.num_classes * 4 + 15) // Constant.FP16_ALIGN_NUM
        prior_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                            name="prior_ub",
                                            scope=tik.scope_ubuf)
        new_prior_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                                name="prior_ub",
                                                scope=tik.scope_ubuf)
        self.tik_instance.data_move(
            new_prior_ub[0],
            prior[sum_addr*align_times*Constant.FP16_ALIGN_NUM], 0,
            cur_batch_num*align_times, self.ratio,
            0, 0)

        with self.tik_instance.for_range(0, align_times) as loop:
            self.tik_instance.data_move(
                prior_ub[loop*cur_batch_num*Constant.FP16_ALIGN_NUM],
                new_prior_ub[loop*Constant.FP16_ALIGN_NUM], 0, cur_batch_num,
                self.ratio, (align_times - Constant.DATA_ONE) * self.ratio, 0)

        self.trans(prior_ub, new_prior_ub,
                   prior_ub.shape[0]*prior_ub.shape[2])
        vec_dup((self.tik_instance, shape[0]*shape[Constant.DATA_ONE]*shape[2],
                 self.input_dtype), prior_ub)
        with self.tik_instance.for_range(
                0, self.num_classes*prior_box_align // Constant.FP16_ALIGN_NUM) \
                as out_loop:
            with self.tik_instance.for_range(0, prior_box_align) as loop:
                with self.tik_instance.for_range(
                        0, prior_box_align) as inner_loop:
                    self.tik_instance.data_move(
                        prior_ub[
                            loop*self.max_rois_num+inner_loop *
                            self.max_rois_num*self.num_classes+out_loop *
                            self.max_rois_num*prior_box_align],
                        new_prior_ub[out_loop*Constant.FP16_ALIGN_NUM *
                                     self.max_rois_num + inner_loop *
                                     Constant.FP16_ALIGN_NUM +
                                     (loop*prior_box_align)*Constant.FP16_ALIGN_NUM],
                        0, self.max_rois_num//Constant.FP16_ALIGN_NUM, self.ratio,
                        15*self.ratio, 0)

        loop_times = self.num_classes % 4
        with self.tik_instance.for_range(0, loop_times) as loop:
            with self.tik_instance.for_range(0,
                                             prior_box_align) as inner_loop:
                self.tik_instance.data_move(
                    prior_ub[
                        loop*self.max_rois_num+inner_loop *
                        self.max_rois_num*self.num_classes +
                        (self.num_classes*prior_box_align) //
                        Constant.FP16_ALIGN_NUM *
                        self.max_rois_num*prior_box_align],
                    new_prior_ub[(self.num_classes*prior_box_align) //
                                 Constant.FP16_ALIGN_NUM * Constant.FP16_ALIGN_NUM *
                                 self.max_rois_num+inner_loop *
                                 Constant.FP16_ALIGN_NUM+(loop*prior_box_align) *
                                 Constant.FP16_ALIGN_NUM], 0,
                    self.max_rois_num//Constant.FP16_ALIGN_NUM, self.ratio,
                    15*self.ratio, 0)

        prior_gm_offset = cur_batch_index*self.max_rois_num * \
                          self.num_classes*prior_box_align

        self.tik_instance.data_move(
            prior_box_gm[prior_gm_offset],
            prior_ub, 0,
            self.max_rois_num//Constant.FP16_ALIGN_NUM *
            self.num_classes*prior_box_align,
            self.ratio, 0, 0)

    def tiling_trans_prior_box_branch1(self, input_list, prior, prior_box_gm):
        shape = input_list[0]
        sum_addr = input_list[1]
        class_loop = input_list[2]
        cur_batch_index = input_list[3]
        cur_batch_num = input_list[4]
        prior_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                            name="prior_ub",
                                            scope=tik.scope_ubuf)
        new_prior_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                                name="prior_ub",
                                                scope=tik.scope_ubuf)
        self.tik_instance.data_move(
            prior_ub[0],
            prior[sum_addr*((self.num_classes*4+Constant.TO_ALIGN_NUM) //
                            Constant.FP16_ALIGN_NUM)*Constant.FP16_ALIGN_NUM +
                  ((class_loop*4+Constant.TO_ALIGN_NUM)//Constant.FP16_ALIGN_NUM) *
                  Constant.FP16_ALIGN_NUM], 0, cur_batch_num, self.ratio,
            ((self.num_classes*4+Constant.TO_ALIGN_NUM)//Constant.FP16_ALIGN_NUM-Constant.DATA_ONE
             )*self.ratio, 0)
        self.trans(prior_ub, new_prior_ub, shape[0])
        vec_dup((self.tik_instance, shape[0]*shape[1],
                 self.input_dtype), prior_ub)

        with self.tik_instance.for_range(0, 4) as loop:
            with self.tik_instance.if_scope(class_loop*4+loop <
                                            self.num_classes):
                self.prior_ub_move(loop, shape[0], prior_ub,
                                   new_prior_ub)
        with self.tik_instance.for_range(0, 4) as loop:
            with self.tik_instance.if_scope(
                    class_loop*4+loop < self.num_classes):
                prior_gm_offset = \
                    ((class_loop*4+Constant.TO_ALIGN_NUM)//Constant.FP16_ALIGN_NUM
                     )*Constant.FP16_ALIGN_NUM * \
                    self.max_rois_num+(class_loop*4 -
                                       ((class_loop*4+Constant.TO_ALIGN_NUM) //
                                        Constant.FP16_ALIGN_NUM) *
                                       Constant.FP16_ALIGN_NUM) * \
                    self.max_rois_num + \
                    cur_batch_index*self.max_rois_num*self.num_classes*4+loop * \
                    self.max_rois_num
                self.prior_gm_move((prior_gm_offset, prior_box_gm,
                                    self.max_rois_num*self.num_classes),
                                   (loop*self.max_rois_num*4, prior_ub,
                                    self.max_rois_num),
                                   self.max_rois_num//Constant.FP16_ALIGN_NUM)

    def tiling_trans_prior_box_branch2(self, input_list, prior, prior_box_gm):
        shape = input_list[0]
        sum_addr = input_list[1]
        class_loop = input_list[2]
        cur_batch_index = input_list[3]
        cur_batch_num = input_list[4]
        one_tiling_process = input_list[5]
        prior_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                            name="prior_ub",
                                            scope=tik.scope_ubuf)
        new_prior_ub = self.tik_instance.Tensor(self.input_dtype, shape,
                                                name="prior_ub",
                                                scope=tik.scope_ubuf)
        with self.tik_instance.for_range(
                0, cur_batch_num//(one_tiling_process // Constant.FP16_ALIGN_NUM)) as \
                inner_batch_loop:
            with self.tik_instance.if_scope(
                    cur_batch_num >=
                    (inner_batch_loop*one_tiling_process //
                     Constant.FP16_ALIGN_NUM)):
                self.tik_instance.data_move(
                    prior_ub[0],
                    prior[sum_addr+inner_batch_loop *
                          (one_tiling_process//Constant.FP16_ALIGN_NUM),
                          class_loop, 0, 0, 0], 0,
                    one_tiling_process//Constant.FP16_ALIGN_NUM, self.ratio,
                    ((self.num_classes*4+Constant.TO_ALIGN_NUM) //
                     Constant.FP16_ALIGN_NUM-Constant.DATA_ONE)*self.ratio, 0)

                self.trans(prior_ub, new_prior_ub, shape[0])
                vec_dup((self.tik_instance, shape[0]*shape[1],
                         self.input_dtype), prior_ub)

                with self.tik_instance.for_range(0, 4) as loop:
                    self.prior_ub_move(loop, shape[0], prior_ub,
                                       new_prior_ub)
                with self.tik_instance.for_range(0, 4) as loop:
                    with self.tik_instance.if_scope(
                            class_loop*4+loop < self.num_classes):
                        prior_gm_offset = \
                            (class_loop*4+loop)*self.batch_rois * \
                            self.max_rois_num*4+cur_batch_index * \
                            self.max_rois_num*4 + \
                            (one_tiling_process//Constant.FP16_ALIGN_NUM) * \
                            inner_batch_loop
                        self.prior_gm_move(
                            (prior_gm_offset, prior_box_gm,
                             self.max_rois_num),
                            (loop*(one_tiling_process //
                                   Constant.FP16_ALIGN_NUM)*4, prior_ub,
                             one_tiling_process//Constant.FP16_ALIGN_NUM),
                            (one_tiling_process//Constant.FP16_ALIGN_NUM) //
                            Constant.FP16_ALIGN_NUM)

        with self.tik_instance.if_scope(
                cur_batch_num -
                (cur_batch_num//(one_tiling_process//Constant.FP16_ALIGN_NUM)) *
                (one_tiling_process//Constant.FP16_ALIGN_NUM) > 0):

            self.tik_instance.data_move(
                prior_ub[0],
                prior[sum_addr+cur_batch_num//(one_tiling_process //
                                               Constant.FP16_ALIGN_NUM) *
                      (one_tiling_process//Constant.FP16_ALIGN_NUM),
                      class_loop, 0, 0, 0], 0,
                cur_batch_num -
                (cur_batch_num//(one_tiling_process//Constant.FP16_ALIGN_NUM)) *
                (one_tiling_process//Constant.FP16_ALIGN_NUM), self.ratio,
                ((self.num_classes*4+Constant.TO_ALIGN_NUM)//Constant.FP16_ALIGN_NUM-Constant.DATA_ONE) *
                self.ratio, 0)
            self.trans(prior_ub, new_prior_ub, shape[0])
            vec_dup((self.tik_instance, shape[0]*shape[1],
                     self.input_dtype), prior_ub)

            with self.tik_instance.for_range(0, 4) as loop:
                self.prior_ub_move(loop, shape[0], prior_ub,
                                   new_prior_ub)
            with self.tik_instance.for_range(0, 4) as loop:
                with self.tik_instance.if_scope(
                        class_loop*4+loop < self.num_classes):
                    prior_gm_offset = \
                        class_loop*4*self.batch_rois * \
                        self.max_rois_num*4 + \
                        cur_batch_index*self.max_rois_num*4 + \
                        loop*self.batch_rois*self.max_rois_num*4 + \
                        (one_tiling_process//Constant.FP16_ALIGN_NUM)*(
                            cur_batch_num //
                            (one_tiling_process//Constant.FP16_ALIGN_NUM))

                    self.prior_gm_move(
                        (prior_gm_offset, prior_box_gm,
                         self.max_rois_num), (
                             (loop*one_tiling_process //
                              Constant.FP16_ALIGN_NUM)*4,
                             prior_ub,
                             (one_tiling_process//Constant.FP16_ALIGN_NUM)),
                        (cur_batch_num -
                         (cur_batch_num //
                          (one_tiling_process//Constant.FP16_ALIGN_NUM)) *
                         (one_tiling_process//Constant.FP16_ALIGN_NUM) +
                         Constant.TO_ALIGN_NUM)//Constant.FP16_ALIGN_NUM)

    def trans_prior_box(self, prior, prior_box_gm, cur_batch_index):
        """
        :param prior:
        :param prior_box_gm:
        :param cur_batch_index:
        :return:
        """
        ub_size = get_ub_size()
        align_times = (self.num_classes * 4 + 15) // Constant.FP16_ALIGN_NUM
        one_batch_size = self.max_rois_num * Constant.FP16_ALIGN_NUM * self.size * \
                         align_times * 2 + self.batch_rois * Constant.DATA_EIGHT * 4 * 2

        if self.max_rois_num*self.batch_rois != prior.shape[0]:
            sum_addr = self.tik_instance.Scalar("int32")
            sum_addr.set_as(0)
            cur_batch_num = self.tik_instance.Scalar("int32")
            cur_batch_num.set_as(0)
            sum_addr.set_as(self.actual_sum_num[
                cur_batch_index*Constant.DATA_EIGHT])
            cur_batch_num.set_as(
                self.actual_rois_num_ub[
                    cur_batch_index*Constant.DATA_EIGHT])
        else:
            cur_batch_num = self.max_rois_num
            sum_addr = cur_batch_index*self.max_rois_num

        if one_batch_size < ub_size:
            shape = [align_times, Constant.FP16_ALIGN_NUM, self.max_rois_num]
            self.no_tiling_trans_prior_box((shape, sum_addr,
                                            cur_batch_index, cur_batch_num),
                                           prior, prior_box_gm)
        else:
            num_piece_of_space = \
                ((ub_size-self.batch_rois*constant.DATA_SIZE_EIGHT*4*2) //
                 constant.DATA_SIZE_TWO)//((Constant.FP16_ALIGN_NUM*Constant.FP16_ALIGN_NUM) *
                                           self.size)
            one_tiling_process = num_piece_of_space*Constant.FP16_ALIGN_NUM*Constant.FP16_ALIGN_NUM
            if one_tiling_process//Constant.FP16_ALIGN_NUM < self.max_rois_num:
                shape = [one_tiling_process//Constant.FP16_ALIGN_NUM, Constant.FP16_ALIGN_NUM]
            else:
                shape = [self.max_rois_num, Constant.FP16_ALIGN_NUM]
            with self.tik_instance.for_range(
                    0, (self.num_classes*4+Constant.TO_ALIGN_NUM)//Constant.FP16_ALIGN_NUM) \
                    as class_loop:
                if one_tiling_process >= self.max_rois_num*Constant.FP16_ALIGN_NUM:
                    self.tiling_trans_prior_box_branch1((shape, sum_addr,
                                                         class_loop, cur_batch_index,
                                                         cur_batch_num), prior, prior_box_gm)
                else:
                    self.tiling_trans_prior_box_branch2((shape, sum_addr, class_loop,
                                                         cur_batch_index, cur_batch_num,
                                                         one_tiling_process), prior, prior_box_gm)


class FsrProcess:
    """
    Faster Process
    """
    def __init__(self, tik_instance, input_fsr, attr_fsr):
        """
        :param tik_instance:
        :param input_fsr:
        :param attr_fsr:
        """
        rois_dic = input_fsr[0]
        bbox_delta_dic = input_fsr[1]
        score_dic = input_fsr[2]
        im_info_dic = input_fsr[3]
        actual_bbox_num_dic = input_fsr[4]
        box_dic = input_fsr[5]
        if len(input_fsr) == 7:
            actual_rois_num_dic = input_fsr[6]

        self.batch_rois = attr_fsr[0]
        self.num_classes = attr_fsr[1]
        self.score_threshlod = attr_fsr[2]
        self.nms_threshold = attr_fsr[3]

        self.max_rois_num = rois_dic.get("shape")[2]
        if len(input_fsr) == 7:
            self.actual_rois_num_effect = True
        else:
            self.actual_rois_num_effect = False

        self.total_num = bbox_delta_dic.get("shape")[0]

        if self.max_rois_num >= 1024:
            self.post_nms_topn = 1024
        else:
            self.post_nms_topn = self.max_rois_num

        self.input_dtype = rois_dic.get('dtype')
        self.input_shape = bbox_delta_dic.get('shape')

        self.tik_instance = tik_instance

        self.size, _, _ = get_params(self.input_dtype)
        self.rois = self.tik_instance.Tensor(
            self.input_dtype, rois_dic.get('shape'),
            name="rois", scope=tik.scope_gm)

        self.bbox_delta = self.tik_instance.Tensor(
            self.input_dtype, self.input_shape, name="bbox_delta", scope=tik.scope_gm)

        self.score = self.tik_instance.Tensor(
            self.input_dtype, score_dic.get('shape'), name="score", scope=tik.scope_gm)

        self.im_info = self.tik_instance.Tensor(self.input_dtype,
                                                im_info_dic.get("shape"),
                                                name="im_info",
                                                scope=tik.scope_gm)

        self.score_gm = self.tik_instance.Tensor(
            self.input_dtype, (self.batch_rois, self.num_classes,
                               self.max_rois_num//Constant.FP16_ALIGN_NUM, Constant.DATA_ONE, Constant.FP16_ALIGN_NUM),
            name="score_gm", is_workspace=True, scope=tik.scope_gm)

        self.prior_box_gm = self.tik_instance.Tensor(
            self.input_dtype, (self.batch_rois, 4, self.num_classes,
                               self.max_rois_num//Constant.FP16_ALIGN_NUM, Constant.FP16_ALIGN_NUM),
            scope=tik.scope_gm, is_workspace=True, name="prior_box_gm")

        if self.actual_rois_num_effect:
            self.actual_rois_num = self.tik_instance.Tensor(
                "int32", actual_rois_num_dic.get("shape"),
                name="actual_rois_num", scope=tik.scope_gm)

        self.output_box = self.tik_instance.Tensor(
            self.input_dtype, (self.num_classes*self.batch_rois, self.max_rois_num,
                               Constant.DATA_EIGHT),
            name="output_box", scope=tik.scope_gm, is_workspace=True)

        self.mem_swap = self.tik_instance.Tensor(
            self.input_dtype, (self.num_classes*self.batch_rois, self.max_rois_num,
                               Constant.DATA_EIGHT),
            name="mem_swap", scope=tik.scope_gm, is_workspace=True)

        if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
            self.pre_nms_topn = 3000
        else:
            if self.input_dtype == "float32":
                self.pre_nms_topn = 3000
            else:
                self.pre_nms_topn = 6000
        if self.max_rois_num < self.pre_nms_topn:
            self.pre_nms_topn = self.max_rois_num

        self.topk_output_proposal = self.tik_instance.Tensor(
            self.input_dtype, (self.num_classes*self.batch_rois,
                               ((self.pre_nms_topn+15)//Constant.FP16_ALIGN_NUM) *
                               Constant.FP16_ALIGN_NUM+4, Constant.DATA_EIGHT),
            name="topk_output_proposal", is_workspace=True, scope=tik.scope_gm)

        self.temp_proposal_out = self.tik_instance.Tensor(
            self.input_dtype, (self.num_classes*self.batch_rois,
                               (self.post_nms_topn+15)//Constant.FP16_ALIGN_NUM*Constant.FP16_ALIGN_NUM,
                               Constant.DATA_EIGHT),
            name="temp_proposal_out",
            is_workspace=True, scope=tik.scope_gm)

        self.is_rpn_supported = check_soc_version()

        self.box = self.tik_instance.Tensor(
            self.input_dtype, box_dic.get("shape"),
            name="box", scope=tik.scope_gm)

        self.actual_bbox_num = self.tik_instance.Tensor(
            "int32", actual_bbox_num_dic.get("shape"),
            name="actual_bbox_num", scope=tik.scope_gm)    
        
        self.dsize = common_util.get_data_size(self.input_dtype)
        
        self.burnest_len = Constant.BLOCK_SIZE // self.dsize
        
        self.max_total_size = self.max_rois_num * self.num_classes   # N*Num_classes
        
        loc_coord_num_align = math.ceil(self.max_rois_num / 16) * 16 + self.burnest_len
        
        self.x1y1x2y2_gm = self.tik_instance.Tensor(
            self.input_dtype,
            (self.batch_rois, Constant.FOUR_DIRECTION, self.num_classes, loc_coord_num_align),
            name="x1y1x2y2_gm",
            is_workspace=True,
            scope=tbe_platform.scope_gm)
        
        self.scores_gm = self.tik_instance.Tensor(
            self.input_dtype,
            (self.batch_rois, self.num_classes, loc_coord_num_align),
            name="scores_gm",
            is_workspace=True,
            scope=tbe_platform.scope_gm)

        self._per_loop_unit = Constant.PER_LOOP_UNIT
        if self.pre_nms_topn <= Constant.PER_LOOP_UNIT // 4:
            self._per_loop_unit = Constant.PER_LOOP_UNIT // 4
        elif self.pre_nms_topn <= Constant.PER_LOOP_UNIT // 2:
            self._per_loop_unit = Constant.PER_LOOP_UNIT // 2
        
        self.nmsed_boxes_gm = self.tik_instance.Tensor(self.input_dtype, (self.batch_rois, 4, self.max_total_size),
                                                name="nmsed_boxes_gm",
                                                is_workspace=True,
                                                scope=tbe_platform.scope_gm)
        self.nmsed_scores_gm = self.tik_instance.Tensor(self.input_dtype, (self.batch_rois, self.max_total_size),
                                                    name="nmsed_scores_gm",
                                                    is_workspace=True,
                                                    scope=tbe_platform.scope_gm)
        self.nmsed_classes_gm = self.tik_instance.Tensor(self.input_dtype, (self.batch_rois, self.max_total_size),
                                                    name="nmsed_classes_gm",
                                                    is_workspace=True,
                                                    scope=tbe_platform.scope_gm)
        self.nmsed_num_gm = self.tik_instance.Tensor("int32", (self.batch_rois, 8),
                                                name="nmsed_num_gm",
                                                is_workspace=True,
                                                is_atomic_add=True,
                                                scope=tbe_platform.scope_gm)
        
        self.boxes_num = self.max_rois_num
        idx_size = self.ceil_div(self.boxes_num, Constant.DATALEN_4K) * Constant.DATALEN_4K
        idx_init = [i for i in range(idx_size)]
        self.idx_gm = self.tik_instance.Tensor("uint32",
                                        [idx_size, ],
                                        name="idx_gm",
                                        scope=tbe_platform.scope_gm,
                                        init_value=idx_init)
        self.workspace_ub_list = [None, None, None, None, None]  # xx1, yy1, xx2, yy2, scores
        
        out_box_len = self.ceil_div(self.post_nms_topn, Constant.DATALEN_128) * Constant.DATALEN_128
        if self.post_nms_topn <= 0:
            out_box_len = math.ceil(Constant.DATALEN_1K / Constant.DATALEN_128) * Constant.DATALEN_128
        self.out_box_len = out_box_len


    @staticmethod
    def ceil_div(value, factor):
        """
        Compute the smallest integer value that is greater than
        or equal to value/factor
        """
        result = (value + (factor - 1)) // factor
        return result
        
    def pre_topk_selection_class(self, batch_idx, class_idx, bbox_cnms_gm, scores_gm):
        """
        topk selection for each class
        :param batch_idx:
        :param class_idx:
        :param bbox_cnms_gm:
        :param scores_gm:
        :return:
        """

        shape_aligned = self._per_loop_unit
        x1_ub = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], 
                                         name="x1_ub", scope=tbe_platform.scope_ubuf)
        y1_ub = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="y1_ub",
                                         scope=tbe_platform.scope_ubuf)
        x2_ub = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="x2_ub",
                                         scope=tbe_platform.scope_ubuf)
        y2_ub = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="y2_ub",
                                         scope=tbe_platform.scope_ubuf)
        scores_ub = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="scores_ub",
                                         scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.tik_instance, x1_ub, shape_aligned)
        cnms_yolo.init_tensor(self.tik_instance, y1_ub, shape_aligned)
        cnms_yolo.init_tensor(self.tik_instance, x2_ub, shape_aligned)
        cnms_yolo.init_tensor(self.tik_instance, y2_ub, shape_aligned)
        cnms_yolo.init_tensor(self.tik_instance, scores_ub, shape_aligned)

        with self.tik_instance.new_stmt_scope():
            handle_length = Constant.DATALEN_1K
            topk_length = self.ceil_div(self.pre_nms_topn, Constant.REPEAT_ELE) * Constant.REPEAT_ELE
            topk_x1_ub = self.tik_instance.Tensor(self.input_dtype, [topk_length, ], name="topk_x1_ub",
                                            scope=tbe_platform.scope_ubuf)
            topk_y1_ub = self.tik_instance.Tensor(self.input_dtype, [topk_length, ], name="topk_y1_ub",
                                            scope=tbe_platform.scope_ubuf)
            topk_x2_ub = self.tik_instance.Tensor(self.input_dtype, [topk_length, ], name="topk_x2_ub",
                                            scope=tbe_platform.scope_ubuf)
            topk_y2_ub = self.tik_instance.Tensor(self.input_dtype, [topk_length, ], name="topk_y2_ub",
                                            scope=tbe_platform.scope_ubuf)
            topk_scores_ub = self.tik_instance.Tensor(self.input_dtype, [topk_length, ], name="topk_scores_ub",
                                                scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.tik_instance, topk_x1_ub, topk_length)
            cnms_yolo.init_tensor(self.tik_instance, topk_y1_ub, topk_length)
            cnms_yolo.init_tensor(self.tik_instance, topk_x2_ub, topk_length)
            cnms_yolo.init_tensor(self.tik_instance, topk_y2_ub, topk_length)
            cnms_yolo.init_tensor(self.tik_instance, topk_scores_ub, topk_length)

            loc_repeat_times = scores_gm.shape[2] // handle_length
            loc_tails_length = scores_gm.shape[2] % handle_length
            with self.tik_instance.for_range(0, loc_repeat_times) as idx:
                self.pre_topk_select_data([x1_ub, y1_ub, x2_ub, y2_ub, scores_ub],
                                        [bbox_cnms_gm[batch_idx, 0, class_idx, (idx * handle_length):],
                                        bbox_cnms_gm[batch_idx, 1, class_idx, (idx * handle_length):],
                                        bbox_cnms_gm[batch_idx, 2, class_idx, (idx * handle_length):],
                                        bbox_cnms_gm[batch_idx, 3, class_idx, (idx * handle_length):],
                                        scores_gm[batch_idx, class_idx, (idx * handle_length):]],
                                        [topk_x1_ub, topk_y1_ub, topk_x2_ub, topk_y2_ub, topk_scores_ub],
                                        handle_length, self.pre_nms_topn)

            with self.tik_instance.if_scope(loc_tails_length > 0):
                self.pre_topk_select_data([x1_ub, y1_ub, x2_ub, y2_ub, scores_ub],
                                        [bbox_cnms_gm[batch_idx, 0, class_idx, (loc_repeat_times * handle_length):],
                                        bbox_cnms_gm[batch_idx, 1, class_idx, (loc_repeat_times * handle_length):],
                                        bbox_cnms_gm[batch_idx, 2, class_idx, (loc_repeat_times * handle_length):],
                                        bbox_cnms_gm[batch_idx, 3, class_idx, (loc_repeat_times * handle_length):],
                                        scores_gm[batch_idx, class_idx, (loc_repeat_times * handle_length):]],
                                        [topk_x1_ub, topk_y1_ub, topk_x2_ub, topk_y2_ub, topk_scores_ub],
                                        loc_tails_length, self.pre_nms_topn)
            cnms_yolo.init_tensor(self.tik_instance, x1_ub, shape_aligned)
            cnms_yolo.init_tensor(self.tik_instance, y1_ub, shape_aligned)
            cnms_yolo.init_tensor(self.tik_instance, x2_ub, shape_aligned)
            cnms_yolo.init_tensor(self.tik_instance, y2_ub, shape_aligned)
            cnms_yolo.init_tensor(self.tik_instance, scores_ub, shape_aligned)
            self.data_move(x1_ub, topk_x1_ub, topk_length)
            self.data_move(y1_ub, topk_y1_ub, topk_length)
            self.data_move(x2_ub, topk_x2_ub, topk_length)
            self.data_move(y2_ub, topk_y2_ub, topk_length)
            self.data_move(scores_ub, topk_scores_ub, topk_length)
        cnms_yolo.exchange_coordinate(self.tik_instance, [x1_ub, x2_ub, y1_ub, y2_ub], shape_aligned)
        self.workspace_ub_list[0] = x1_ub
        self.workspace_ub_list[1] = y1_ub
        self.workspace_ub_list[2] = x2_ub
        self.workspace_ub_list[3] = y2_ub
        self.workspace_ub_list[4] = scores_ub
        
    def pre_topk_select_data(self, ub_list, loc_gm_list, topk_ub_list, handle_length, top_k):
        """
        execute topk select for per block data
        :param ub_list:
        :param loc_gm_list:
        :param topk_ub_list:
        :param handle_length:
        :param top_k:
        :return:
        """
        topk_length = self.ceil_div(top_k, Constant.REPEAT_ELE) * Constant.REPEAT_ELE
        x1_ub, y1_ub, x2_ub, y2_ub, scores_ub = ub_list
        x1_gm, y1_gm, x2_gm, y2_gm, scores_gm = loc_gm_list
        topk_x1_ub, topk_y1_ub, topk_x2_ub, topk_y2_ub, topk_scores_ub = topk_ub_list

        # step1: prepare data from topk selection
        cnms_yolo.init_tensor(self.tik_instance, x1_ub, self._per_loop_unit)
        cnms_yolo.init_tensor(self.tik_instance, y1_ub, self._per_loop_unit)
        cnms_yolo.init_tensor(self.tik_instance, x2_ub, self._per_loop_unit)
        cnms_yolo.init_tensor(self.tik_instance, y2_ub, self._per_loop_unit)
        cnms_yolo.init_tensor(self.tik_instance, scores_ub, self._per_loop_unit)
        self.data_move(x1_ub, topk_x1_ub, topk_length)
        self.data_move(y1_ub, topk_y1_ub, topk_length)
        self.data_move(x2_ub, topk_x2_ub, topk_length)
        self.data_move(y2_ub, topk_y2_ub, topk_length)
        self.data_move(scores_ub, topk_scores_ub, topk_length)

        self.data_move(x1_ub[topk_length:], x1_gm, handle_length)
        self.data_move(y1_ub[topk_length:], y1_gm, handle_length)
        self.data_move(x2_ub[topk_length:], x2_gm, handle_length)
        self.data_move(y2_ub[topk_length:], y2_gm, handle_length)
        self.data_move(scores_ub[topk_length:], scores_gm, handle_length)

        cnms_yolo.init_tensor(self.tik_instance, topk_x1_ub, topk_length)
        cnms_yolo.init_tensor(self.tik_instance, topk_y1_ub, topk_length)
        cnms_yolo.init_tensor(self.tik_instance, topk_x2_ub, topk_length)
        cnms_yolo.init_tensor(self.tik_instance, topk_y2_ub, topk_length)
        cnms_yolo.init_tensor(self.tik_instance, topk_scores_ub, topk_length)

        if self._per_loop_unit ==  Constant.DATALEN_1K:
            total_data_length = Constant.DATALEN_1K
        else:
            total_data_length = Constant.DATALEN_2K
        score_idx_lens = total_data_length * Constant.UNIT_ELE
        scores_idx_out = self.tik_instance.Tensor(self.input_dtype, [score_idx_lens * 2], name="scores_idx_out",
                                            scope=tbe_platform.scope_ubuf)
        scores_idx_ub = self.tik_instance.Tensor(self.input_dtype, [score_idx_lens * 2], name="scores_idx_ub",
                                            scope=tbe_platform.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            index_ub = self.tik_instance.Tensor("uint32", [total_data_length, ], name="index_ub",
                                            scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_index(self.tik_instance, self.idx_gm, index_ub, 0, total_data_length)
            repeat_times = total_data_length // Constant.REPEAT_ELE
            self.tik_instance.vsort32(scores_idx_ub, scores_ub, index_ub, repeat_times)
            sort_score_idx_by_desc(self.tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens)
            cnms_yolo.init_tensor(self.tik_instance, scores_idx_ub, total_data_length)
            mask, _ = get_mask_rep_stride(scores_idx_ub)
            repeat_times = total_data_length * Constant.FOUR_DIRECTION // mask
            self.tik_instance.vreducev2(None, scores_idx_ub, scores_idx_out, 
                                        Constant.PATTERN_TYPE, repeat_times, 1, 8, 0)

        if tbe_platform.api_check_support("tik.vgather"):
            lo_index = self.tik_instance.Scalar("int32", init_value=2)
            mask_length = self.ceil_div(top_k, Constant.DATALEN_128) * Constant.DATALEN_128
            topk_index_ub = self.tik_instance.Tensor("int32", [mask_length, ], name="topk_index_ub",
                                                     scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.tik_instance, topk_index_ub, mask_length)
            with self.tik_instance.for_range(0, top_k) as idx:
                topk_scores_ub[idx].set_as(scores_idx_ub[idx])
                scores_index_offset = idx * Constant.UNIT_ELE
                topk_index_ub[idx].set_as(
                    scores_idx_out[scores_index_offset + 2: scores_index_offset + 4].reinterpret_cast_to("uint32"))
            mul_loop = mask_length // Constant.INT32_MASK
            self.tik_instance.vmuls(Constant.INT32_MASK, topk_index_ub, topk_index_ub, lo_index, mul_loop, 1, 1, 8, 8)
            loop_len = top_k // Constant.FP16_MASK
            loop_tail = top_k % Constant.FP16_MASK
            with self.tik_instance.if_scope(loop_len > 0):
                self.tik_instance.vgather(Constant.FP16_MASK, topk_x1_ub, x1_ub, topk_index_ub, loop_len,
                                          8, 0, 0, mask_mode="normal")
                self.tik_instance.vgather(Constant.FP16_MASK, topk_y1_ub, y1_ub, topk_index_ub, loop_len,
                                          8, 0, 0, mask_mode="normal")
                self.tik_instance.vgather(Constant.FP16_MASK, topk_x2_ub, x2_ub, topk_index_ub, loop_len,
                                          8, 0, 0, mask_mode="normal")
                self.tik_instance.vgather(Constant.FP16_MASK, topk_y2_ub, y2_ub, topk_index_ub, loop_len,
                                          8, 0, 0, mask_mode="normal")
            with self.tik_instance.if_scope(loop_tail > 0):
                self.tik_instance.vgather(loop_tail, topk_x1_ub[loop_len * Constant.FP16_MASK],
                                          x1_ub, topk_index_ub[loop_len * Constant.FP16_MASK],
                                          1, 8, 0, mask_mode="normal")
                self.tik_instance.vgather(loop_tail, topk_y1_ub[loop_len * Constant.FP16_MASK],
                                          y1_ub, topk_index_ub[loop_len * Constant.FP16_MASK],
                                          1, 8, 0, mask_mode="normal")
                self.tik_instance.vgather(loop_tail, topk_x2_ub[loop_len * Constant.FP16_MASK],
                                          x2_ub, topk_index_ub[loop_len * Constant.FP16_MASK],
                                          1, 8, 0, mask_mode="normal")
                self.tik_instance.vgather(loop_tail, topk_y2_ub[loop_len * Constant.FP16_MASK],
                                          y2_ub, topk_index_ub[loop_len * Constant.FP16_MASK],
                                          1, 8, 0, mask_mode="normal")
        else:
            lo_index = self.tik_instance.Scalar("uint32", init_value=0)
            with self.tik_instance.for_range(0, top_k) as idx:
                topk_scores_ub[idx].set_as(scores_idx_ub[idx])
                scores_index_offset = idx * Constant.UNIT_ELE
                lo_index.set_as(
                    scores_idx_out[scores_index_offset + 2: scores_index_offset + 4].reinterpret_cast_to("uint32"))
                topk_x1_ub[idx].set_as(x1_ub[lo_index])
                topk_y1_ub[idx].set_as(y1_ub[lo_index])
                topk_x2_ub[idx].set_as(x2_ub[lo_index])
                topk_y2_ub[idx].set_as(y2_ub[lo_index])
            
            
    def data_move(self, dst, src, length):
        """
        move data beteen gm and ub
        :param dst: memory space in UB or GM, if in GM, src must in UB
        :param src: memory space in UB or GM, if in GM, dst must in UB
        :param length: Data length which is Aligned by 32 Bytes
        :return:
        """
        burst_len = length // Constant.BLOCK_ELE
        with self.tik_instance.if_scope(burst_len > 0):
            self.tik_instance.data_move(dst, src, 0, 1, burst_len, 0, 0)
        
    def cnms_calcation_class(self, batch_idx, class_idx, bbox_out_list, im_info):
        """
        execute cnms calculation for per classes
        :param batch_idx:
        :param class_idx:
        :param bbox_out_list:
        :return:
        """
        shape_aligned = self._per_loop_unit

        x1_ub = self.workspace_ub_list[0]
        y1_ub = self.workspace_ub_list[1]
        x2_ub = self.workspace_ub_list[2]
        y2_ub = self.workspace_ub_list[3]
        scores_ub = self.workspace_ub_list[4]

        # select by scores_threshold
        eff_lens = self.tik_instance.Scalar("uint32", "eff_lens", 0)
        eff_lens.set_as(shape_aligned)
        self.scores_threshold_selection_class(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, self.score_threshlod, eff_lens)

        with self.tik_instance.if_scope(eff_lens > 0):
            # do iou selection
            iou_thresh = self.nms_threshold / (1 + self.nms_threshold)

            repeats = get_vector_repeat_times(self.tik_instance, eff_lens * self.dsize)

            self.tik_instance.vec_muls(128, x1_ub, x1_ub,
                                Constant.FACTOR_1, repeats, constant.REPEAT_STRIDE_EIGHT,
                                constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vec_muls(128, y1_ub, y1_ub,
                                Constant.FACTOR_1, repeats, constant.REPEAT_STRIDE_EIGHT,
                                constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vec_muls(128, x2_ub, x2_ub,
                                Constant.FACTOR_1, repeats, constant.REPEAT_STRIDE_EIGHT,
                                constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vec_muls(128, y2_ub, y2_ub,
                                Constant.FACTOR_1, repeats, constant.REPEAT_STRIDE_EIGHT,
                                constant.REPEAT_STRIDE_EIGHT)

            self.ssd_iou_selection(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, eff_lens)

            self.tik_instance.vec_muls(128, x1_ub, x1_ub,
                                Constant.FACTOR_2, repeats, constant.REPEAT_STRIDE_EIGHT,
                                constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vec_muls(128, y1_ub, y1_ub,
                                Constant.FACTOR_2, repeats, constant.REPEAT_STRIDE_EIGHT,
                                constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vec_muls(128, x2_ub, x2_ub,
                                Constant.FACTOR_2, repeats, constant.REPEAT_STRIDE_EIGHT,
                                constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vec_muls(128, y2_ub, y2_ub,
                                Constant.FACTOR_2, repeats, constant.REPEAT_STRIDE_EIGHT,
                                constant.REPEAT_STRIDE_EIGHT)
            
            with self.tik_instance.if_scope(eff_lens > self.max_rois_num):
                eff_lens.set_as(self.max_rois_num )

            # store data
            with self.tik_instance.if_scope(eff_lens > 0):
                self.store_cnms_data_per_class(batch_idx, class_idx,
                                            x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                                            bbox_out_list, eff_lens)

    def scores_threshold_selection_class(self,
                                        x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, scores_value, eff_lens):
        """

        :param x1_ub:
        :param x2_ub:
        :param y1_ub:
        :param y2_ub:
        :param scores_ub:
        :param scores_value:
        :param eff_lens:
        :return:
        """
        shape_size = self._per_loop_unit
        shape = (shape_size,)
        mask_shape = (self.ceil_div(shape_size, Constant.BLOCK_ELE),)
        if scores_value == 0:
            eff_lens.set_as(self.pre_nms_topn)
        else:
            with self.tik_instance.new_stmt_scope():
                scores_thresh = self.tik_instance.Tensor(self.input_dtype, shape, name="scores_threshold",
                                                    scope=tbe_platform.scope_ubuf)
                cnms_yolo.init_tensor(self.tik_instance, scores_thresh, shape_size, scores_value)

                tmp1 = self.tik_instance.Tensor(self.input_dtype, [shape_size, ],
                                                name="tmp1", scope=tbe_platform.scope_ubuf)
                tmp2 = self.tik_instance.Tensor(self.input_dtype, [shape_size, ], name="tmp2", 
                                                scope=tbe_platform.scope_ubuf)
                _single_area = self.tik_instance.Tensor(self.input_dtype, [shape_size, ], name="_single_area",
                                                    scope=tbe_platform.scope_ubuf)

                mask_uint16 = self.tik_instance.Tensor("uint16", mask_shape, name="mask_uint16", 
                                                    scope=tbe_platform.scope_ubuf)
                cnms_yolo.init_tensor(self.tik_instance, mask_uint16, self.ceil_div(shape_size, Constant.BLOCK_ELE), 0)
                cnms_yolo.gen_mask(self.tik_instance, scores_thresh, scores_ub, mask_uint16, shape_size)
                cnms_yolo.update_input_v300(self.tik_instance,
                                            [x1_ub, x2_ub, y1_ub, y2_ub, scores_ub], _single_area,
                                            eff_lens, [tmp1, tmp2, mask_uint16, shape_size])
        
    def ssd_iou_selection(self, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, eff_lens):
        """
        execute iou selection,
        :param x1_ub:
        :param x2_ub:
        :param y1_ub:
        :param y2_ub:
        :param scores_ub:
        :param eff_lens:
        :return:
        """
        shape_aligned = self._per_loop_unit
        mask, _ = get_mask_rep_stride(x1_ub)

        # iou Selection for only topk data for per class
        single_area = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="single_area",
                                        scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.tik_instance, single_area, shape_aligned)
        # get area of every windows
        cnms_yolo.get_rectangle_area(self.tik_instance, [x1_ub, x2_ub, y1_ub, y2_ub], single_area, shape_aligned)

        iou = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="iou",
                                scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.tik_instance, iou, shape_aligned)
        # calculate the iou, exit when the output windows is more than eff_lens
        overlap = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="overlap",
                                    scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.tik_instance, overlap, shape_aligned)
        # define tmp tensor for following use, to reduce the cycle of apply/release memory
        tmp1 = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="tmp1", scope=tbe_platform.scope_ubuf)
        tmp2 = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="tmp2", scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.tik_instance, tmp1, shape_aligned)
        cnms_yolo.init_tensor(self.tik_instance, tmp2, shape_aligned)
        mask_shape_lens = self._per_loop_unit // Constant.BLOCK_ELE
        mask_uint16 = self.tik_instance.Tensor("uint16", [mask_shape_lens, ], name="mask_uint16",
                                        scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.tik_instance, mask_uint16, mask_shape_lens)
        iou_thresh = self.nms_threshold / (1 + self.nms_threshold)

        # calculate ioues for every windows
        with self.tik_instance.for_range(0, self.pre_nms_topn) as idx:
            with self.tik_instance.if_scope(idx < eff_lens):
                cnms_yolo.get_overlap(self.tik_instance, [x1_ub, x2_ub, y1_ub, y2_ub],
                                      [overlap, tmp1, tmp2], idx, shape_aligned)
                cnms_yolo.init_tensor(self.tik_instance, tmp2, shape_aligned)
                _aligned_length = self.ceil_div(eff_lens, mask) * mask
                cnms_yolo.cal_iou(self.tik_instance, [single_area, iou, tmp2], idx, _aligned_length, iou_thresh)
                cnms_yolo.gen_mask(self.tik_instance, overlap, iou, mask_uint16, size=self._per_loop_unit)
                cnms_yolo.update_input_v300(self.tik_instance,
                                            [x1_ub, x2_ub, y1_ub, y2_ub, scores_ub], single_area,
                                            eff_lens, [tmp1, tmp2, mask_uint16, shape_aligned])
                
    def partial_init_tensor(self, dst, size, start, init_value=0):
        """
        init party memory of tensor
        :param dst: ub memory
        :param size: tensor size
        :param start: start init address
        :param init_value:
        :return:
        """
        vector_mask, rep_stride = get_mask_rep_stride(dst)
        aligned_start = self.ceil_div(start, 32) * 32
        length = size - aligned_start
        max_lens = Constant.REPEAT_TIMES_MAX * vector_mask
        loop_num = length // max_lens
        tail = length % max_lens
        repeat_times = tail // vector_mask
        tail_aligned = tail % vector_mask

        with self.tik_instance.for_range(start, aligned_start) as idx:
            dst[idx].set_as(init_value)

        off = self.tik_instance.Scalar("uint32")
        with self.tik_instance.for_range(0, loop_num) as idx:
            off.set_as(vector_mask * Constant.REPEAT_TIMES_MAX * idx)
            self.tik_instance.vec_dup(vector_mask,
                                dst[aligned_start + off],
                                init_value,
                                Constant.REPEAT_TIMES_MAX,
                                rep_stride)
        with self.tik_instance.if_scope(tik.all(tail != 0, repeat_times > 0)):
            offset = length - tail
            self.tik_instance.vec_dup(vector_mask,
                                dst[aligned_start + offset],
                                init_value,
                                repeat_times,
                                rep_stride)
        with self.tik_instance.if_scope(tail_aligned != 0):
            with self.tik_instance.for_range(0, tail_aligned) as idx:
                dst[aligned_start + length - tail_aligned + idx].set_as(init_value)

    def post_topk_selection_class(self, eff_lens):
        """
        post topk selection, if post_nms_topn > 0, set eff_lens to post_nms_topn
        :param eff_lens:
        :return:
        """
        if self.post_nms_topn > 0:
            eff_lens.set_as(self.post_nms_topn)
                
    def store_cnms_data_per_class(self, batch_idx, class_idx,
                                xx1, xx2, yy1, yy2, scores_ub, bbox_out_list, eff_lens):
        """
        save data which select by cnms
        :param batch_idx:
        :param class_idx:
        :param xx1:
        :param xx2:
        :param yy1:
        :param yy2:
        :param scores_ub:
        :param bbox_out_list:
        :param eff_lens:
        :return:
        """
        boxes_out, scores_out, class_out, box_num_out = bbox_out_list
        valid_detection = self.tik_instance.Scalar("int32", "valid_detection", 0)
        valid_detection.set_as(box_num_out[batch_idx, 0])

        box_outnum_ub = self.tik_instance.Tensor("int32",
                                            (self.batch_rois, 8),
                                            name="box_outnum_ub",
                                            scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.tik_instance, box_outnum_ub, 8, 0)
        box_outnum_ub[batch_idx, 0].set_as(eff_lens)
        self.tik_instance.data_move(box_num_out, box_outnum_ub, 0, 1, 1, 0, 0)

        with self.tik_instance.new_stmt_scope():
            _cls_idx = self.tik_instance.Scalar(self.input_dtype, "_cls_idx")
            _cls_idx.set_as(class_idx)
            with self.tik_instance.for_range(0, eff_lens) as idx:
                class_out[batch_idx, idx].set_as(_cls_idx)

        repeat_times = self.ceil_div(eff_lens, Constant.BLOCK_ELE)
        with self.tik_instance.if_scope(repeat_times > 0):
            self.tik_instance.data_move(boxes_out[batch_idx, 0, 0], xx1, 0, 1, repeat_times, 0, 0)
            self.tik_instance.data_move(boxes_out[batch_idx, 1, 0], yy1, 0, 1, repeat_times, 0, 0)
            self.tik_instance.data_move(boxes_out[batch_idx, 2, 0], xx2, 0, 1, repeat_times, 0, 0)
            self.tik_instance.data_move(boxes_out[batch_idx, 3, 0], yy2, 0, 1, repeat_times, 0, 0)
            self.tik_instance.data_move(scores_out[batch_idx, 0], scores_ub, 0, 1, repeat_times, 0, 0)
            
    def store_cnms_bbox_num_data_output(self, batch_idx, class_index, bbox_out_list):
        """
        store cnms output data into gm with SSD out formate
        :param batch_idx:
        :param bbox_out_list:
        :return:
        """
        box_num_out = bbox_out_list[3]
        nms_len = self.tik_instance.Scalar("int32", "nms_len", init_value=0)
        nms_len.set_as(box_num_out[batch_idx, 0])

        with self.tik_instance.if_scope(nms_len > 0):
            with self.tik_instance.if_scope(nms_len > self.out_box_len):
                nms_len.set_as(self.out_box_len)
            box_outnum_ub = self.tik_instance.Tensor("int32",
                                                (1, 8),
                                                name="box_outnum_ub",
                                                scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.tik_instance, box_outnum_ub, 8)
            box_outnum_ub[0, 0].set_as(nms_len)
            # the size of box_outnum_ub is 32 Byte(1 Block)
            self.tik_instance.data_move(self.actual_bbox_num[batch_idx, class_index, 0], box_outnum_ub, 0, 1, 1, 0, 0)

    def store_cnms_out_box_data_output(self, batch_idx, class_index, bbox_out_list):
        """
        store cnms output data into gm with SSD out formate
        :param batch_idx:
        :param bbox_out_list:
        :return:
        """
        boxes_out, scores_out, classes_out, box_num_out = bbox_out_list
        nms_len = self.tik_instance.Scalar("int32", "nms_len", init_value=0)
        nms_len.set_as(box_num_out[batch_idx, 0])

        with self.tik_instance.new_stmt_scope():
            out_box_ub = self.tik_instance.Tensor(self.input_dtype,
                                            (self.out_box_len, 8),
                                            name="out_box_ub",
                                            scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.tik_instance, out_box_ub, self.out_box_len * 8)
            with self.tik_instance.for_range(0, nms_len) as idx:
                out_box_ub[idx, 0].set_as(boxes_out[batch_idx, 0, idx])
                out_box_ub[idx, 1].set_as(boxes_out[batch_idx, 1, idx])
                out_box_ub[idx, 2].set_as(boxes_out[batch_idx, 2, idx])
                out_box_ub[idx, 3].set_as(boxes_out[batch_idx, 3, idx])
                out_box_ub[idx, 4].set_as(scores_out[batch_idx, idx])
                out_box_ub[idx, 5].set_as(classes_out[batch_idx, idx])
                out_box_ub[idx, 6].set_as(batch_idx)
                out_box_ub[idx, 7].set_as(0)

            burst_times = (self.out_box_len * 8) // Constant.BLOCK_ELE
            with self.tik_instance.if_scope(burst_times > 0):
                self.tik_instance.data_move(self.box[batch_idx, class_index, 0, 0], out_box_ub, 0, 1, burst_times, 0, 0)

    def cce_fsr(self, kernel_name):
        """
        :param kernel_name:
        :return:
        """
        device_core_num, batch_factor, batch_factor_tail = \
            filter_device_core(self.batch_rois)

        with self.tik_instance.for_range(0, device_core_num,
                                         block_num=device_core_num) as \
                block_id:

            ub_size = get_ub_size()
            one_core_process_object = OneCoreProcess(
                (self.tik_instance, self.input_dtype, self.input_shape,
                 self.size, self.batch_rois, self.im_info,
                 self.num_classes, self.max_rois_num,
                 self.total_num, device_core_num, batch_factor, block_id,
                 ub_size, self.x1y1x2y2_gm, self.scores_gm))

            if self.actual_rois_num_effect:
                input_tensor = [self.actual_rois_num_effect, self.actual_rois_num]
            else:
                input_tensor = [self.actual_rois_num_effect]

            cur_batch_index = self.tik_instance.Scalar("int32", "cur_batch_index", 0)
            output_gm_list = [self.nmsed_boxes_gm[cur_batch_index, :, :],
                    self.nmsed_scores_gm[cur_batch_index, :],
                    self.nmsed_classes_gm[cur_batch_index, :],
                    self.nmsed_num_gm[cur_batch_index, :]]

            with self.tik_instance.for_range(
                    0, batch_factor)  as batch_index:

                cur_batch_index = block_id*batch_factor+batch_index
                with self.tik_instance.new_stmt_scope():
                    pre_object = PreProcess((self.tik_instance,
                                             self.max_rois_num,
                                             self.num_classes,
                                             self.batch_rois,
                                             self.input_shape,
                                             self.input_dtype),
                                            input_tensor)
                    with self.tik_instance.new_stmt_scope():
                        pre_object.trans_score(self.score, self.score_gm,
                                               cur_batch_index)
                    with self.tik_instance.new_stmt_scope():
                        pre_object.trans_prior_box(self.bbox_delta,
                                                   self.prior_box_gm,
                                                   cur_batch_index)
                with self.tik_instance.new_stmt_scope():
                    one_core_process_object.one_core_process_decode_rois(
                        (cur_batch_index, self.rois, input_tensor,
                         self.prior_box_gm, self.score_gm), self.output_box)

                if not self.is_rpn_supported:

                    with self.tik_instance.for_range(0, self.num_classes) \
                            as class_index:

                        self.pre_topk_selection_class(cur_batch_index, class_index, self.x1y1x2y2_gm, self.scores_gm)

                        self.cnms_calcation_class(cur_batch_index, class_index, output_gm_list, self.im_info)

                        self.store_cnms_bbox_num_data_output(cur_batch_index, class_index, output_gm_list)
                    
                        self.store_cnms_out_box_data_output(cur_batch_index, class_index, output_gm_list)

                else:

                    with self.tik_instance.for_range(0, self.num_classes) \
                            as class_index:

                        topk_output_actual_proposal_num = \
                            self.tik_instance.Scalar(dtype="int32")

                        batch_id = cur_batch_index*self.num_classes + class_index

                        with self.tik_instance.new_stmt_scope():
                            call_topk_sort(
                                self.tik_instance, (self.max_rois_num,
                                                    self.score_threshlod,
                                                    self.pre_nms_topn,
                                                    self.output_box, self.mem_swap),
                                (batch_id, self.topk_output_proposal,
                                topk_output_actual_proposal_num))

                        input_offset = \
                            batch_id*(((self.pre_nms_topn+15) //
                                    Constant.FP16_ALIGN_NUM)*Constant.FP16_ALIGN_NUM+4) * \
                            Constant.DATA_EIGHT
                        real_batch_index = cur_batch_index
                        with self.tik_instance.new_stmt_scope():
                            used_in_proposal = False
                            nms.cce_nms(
                                (self.input_dtype, ub_size, self.nms_threshold,
                                batch_id, self.pre_nms_topn, self.post_nms_topn,
                                input_offset, self.im_info, self.tik_instance,
                                self.num_classes, class_index, real_batch_index),
                                self.temp_proposal_out, self.topk_output_proposal,
                                topk_output_actual_proposal_num,
                                self.actual_bbox_num, self.box,
                                used_in_proposal)

                with self.tik_instance.if_scope(block_id
                                                < batch_factor_tail):
                    cur_batch_index = batch_factor*device_core_num+block_id
                    with self.tik_instance.new_stmt_scope():
                        pre_object = PreProcess((self.tik_instance,
                                                 self.max_rois_num,
                                                 self.num_classes,
                                                 self.batch_rois,
                                                 self.input_shape,
                                                 self.input_dtype),
                                                input_tensor)
                        with self.tik_instance.new_stmt_scope():
                            pre_object.trans_score(self.score,
                                                   self.score_gm,
                                                   cur_batch_index)
                        with self.tik_instance.new_stmt_scope():
                            pre_object.trans_prior_box(self.bbox_delta,
                                                       self.prior_box_gm,
                                                       cur_batch_index)
                    with self.tik_instance.new_stmt_scope():
                        one_core_process_object.one_core_process_decode_rois((
                            cur_batch_index, self.rois, input_tensor,
                            self.prior_box_gm, self.score_gm), self.output_box)

                    if not self.is_rpn_supported:

                        with self.tik_instance.for_range(0, self.num_classes) \
                                as class_index:

                            self.pre_topk_selection_class(cur_batch_index, class_index,
                                                          self.x1y1x2y2_gm, self.scores_gm)

                            self.cnms_calcation_class(cur_batch_index, class_index, output_gm_list, self.im_info)

                            self.store_cnms_bbox_num_data_output(cur_batch_index, class_index, output_gm_list)
                    
                            self.store_cnms_out_box_data_output(cur_batch_index, class_index, output_gm_list)

                    else:
                        with self.tik_instance.for_range(0, self.num_classes) \
                                as class_index:
                            topk_output_actual_proposal_num = \
                                self.tik_instance.Scalar(dtype="int32")

                            batch_id = cur_batch_index*self.num_classes+class_index
                            real_batch_index = cur_batch_index
                            with self.tik_instance.new_stmt_scope():
                                call_topk_sort(self.tik_instance,
                                            (self.max_rois_num,
                                                self.score_threshlod,
                                                self.pre_nms_topn, self.output_box,
                                                self.mem_swap),
                                            (batch_id, self.topk_output_proposal,
                                                topk_output_actual_proposal_num))

                            input_offset = \
                                batch_id*(((self.pre_nms_topn+15) //
                                        Constant.FP16_ALIGN_NUM)*Constant.FP16_ALIGN_NUM+4) * \
                                Constant.DATA_EIGHT

                            with self.tik_instance.new_stmt_scope():
                                used_in_proposal = False
                                nms.cce_nms((self.input_dtype, ub_size,
                                            self.nms_threshold, batch_id,
                                            self.pre_nms_topn, self.post_nms_topn,
                                            input_offset, self.im_info,
                                            self.tik_instance,
                                            self.num_classes, class_index,
                                            real_batch_index),
                                            self.temp_proposal_out,
                                            self.topk_output_proposal,
                                            topk_output_actual_proposal_num,
                                            self.actual_bbox_num,
                                            self.box, used_in_proposal)

        if self.actual_rois_num_effect:
            self.tik_instance.BuildCCE(kernel_name,
                                       inputs=[self.rois,
                                               self.bbox_delta, self.score,
                                               self.im_info,
                                               self.actual_rois_num],
                                       outputs=[self.actual_bbox_num,
                                                self.box])
        else:
            self.tik_instance.BuildCCE(kernel_name,
                                       inputs=[self.rois,
                                               self.bbox_delta, self.score,
                                               self.im_info],
                                       outputs=[self.actual_bbox_num,
                                                self.box])

        return self.tik_instance


def check_datatype(tik_name, dtype):
    """
    :param tik_name:
    :param dtype:
    :return:
    """
    if not tbe_platform.api_check_support("te.lang.cce.vrelu", "float32"):
        para_check.check_dtype(dtype.lower(), ["float16"], param_name="roidic")
    else:
        para_check.check_dtype(dtype.lower(), ["float16", "float32"], param_name="roidic")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def fsr_detection_output(rois_dic, bbox_delta_dic, score_dic, im_info_dic,
                         actual_rois_num_dic, actual_bbox_num_dic, box_dic,
                         num_classes, score_threshold, iou_threshold, batch_rois=1,
                         kernel_name="fsr_detection_output"):
    """
    :param rois_dic:
    :param bbox_delta_dic:
    :param score_dic:
    :param im_info_dic:
    :param actual_rois_num_dic:
    :param actual_bbox_num_dic:
    :param box_dic:
    :param num_classes:
    :param score_threshold:
    :param iou_threshold:
    :param batch_rois:
    :param kernel_name:
    :return:
    """

    tik_instance = tik.Tik(tik.Dprofile())
    tik_name = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    input_dtype = rois_dic.get('dtype')
    check_datatype(tik_name, input_dtype)
    batch_rois = rois_dic.get("shape")[0]

    para_check.check_shape(score_dic.get("shape"), min_rank=5, param_name="score_dic")

    if im_info_dic.get("shape")[0] != batch_rois:
        error_manager_vector.raise_err_input_value_invalid("fsr_dection_output", "im_info_dic", str(batch_rois),
                                                           str(im_info_dic.get("shape")[0]))
    if num_classes > score_dic.get("shape")[1] * score_dic.get("shape")[4]:
        error_manager_vector.raise_err_input_param_not_in_range("fsr_dection_output",
                                                                "num_classes",
                                                                "0",
                                                                str(score_dic.get("shape")[1] * \
                                                                    score_dic.get("shape")[4]),
                                                                num_classes)
    if num_classes > bbox_delta_dic.get("shape")[1] * bbox_delta_dic.get("shape")[4] // 4:
        error_manager_vector.raise_err_input_param_not_in_range("fsr_dection_output",
                                                                "num_classes",
                                                                "0",
                                                                str(bbox_delta_dic.get("shape")[1] * \
                                                                    bbox_delta_dic.get("shape")[4] // 4),
                                                                num_classes)

    if iou_threshold <= 0.0 or iou_threshold >= 1.0:
        error_manager_vector.raise_err_input_param_not_in_range("fsr_dection_output",
                                                                "iou_threshold",
                                                                "0.0",
                                                                "1.0",
                                                                iou_threshold)

    if score_threshold < 0.0 or score_threshold > 1.0:
        error_manager_vector.raise_err_input_param_not_in_range("fsr_dection_output", "score_threshold",
                                                                "0.0", "1.0", score_threshold)
    if num_classes < 1:
        error_manager_vector.raise_err_input_param_not_in_range("fsr_dection_output", "score_threshold",
                                                                "1", "inf", num_classes)

    if actual_rois_num_dic:
        input_list = (rois_dic, bbox_delta_dic, score_dic, im_info_dic,
                      actual_bbox_num_dic, box_dic, actual_rois_num_dic)
    else:
        input_list = (rois_dic, bbox_delta_dic, score_dic, im_info_dic,
                      actual_bbox_num_dic, box_dic)
    fsr_result = FsrProcess(tik_instance, input_list,
                            (batch_rois, num_classes, score_threshold,
                             iou_threshold))

    return fsr_result.cce_fsr(kernel_name)
