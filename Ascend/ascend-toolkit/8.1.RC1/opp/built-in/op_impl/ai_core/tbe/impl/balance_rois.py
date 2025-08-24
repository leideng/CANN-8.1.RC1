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
balance_rois
"""

import math
from functools import reduce as functools_reduce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform


# proposal struct contains 8 elements
PROPOSAL_NUM = 8
# use this idx in proposal struct for val
VAL_IDX = 4
# use this idx in proposal struct for idx's merchant part
INT_IDX = 0
# use this idx in proposal struct for idx's remainder part
REM_IDX = 1
# min val in fp16
MIN_VAL = -65504
# max val in fp16
MAX_VAL = 65504
# sorting threshold for normal data volume
BLOCK = 16
# sorting threshold for data volume over 2048
NUM_BLOCK = 2048


def sort_compute(tik_instance, dtype, num, num_16, num_2048, core_idx, used_aicore_num, data_out,
                 data_indices, input_gm, temp, num_gm, descending, cce_product):
    """
    Function: sortcompute in UB.
    Modify : 2020-08-03

    Attention : This way is unstable (can't compare two scalar).
    Init base parameters
    Parameters
    ----------
    dtype, num_16, core_idx, num, distance, shape, big_distance, L : for index compute
    data_out, data_indices, input_gm : for data move
    ----------
    """
    if num <= NUM_BLOCK:
        idx_ub = tik_instance.Tensor(dtype, [NUM_BLOCK], name="idx_ub", scope=tik.scope_ubuf)
        input_ub = tik_instance.Tensor(dtype, [num_16 * PROPOSAL_NUM * 2], name="input_ub", scope=tik.scope_ubuf)

        offset_in = core_idx * num
        offset_out = core_idx * num_16
        dest_pos_ub = num_16 * PROPOSAL_NUM
        n_repeat_total = num_16 // BLOCK
        # 1. Move data from OUT to UB
        tik_instance.data_move(input_ub[dest_pos_ub], input_gm[offset_in], 0, 1, n_repeat_total, 0, 0)
        max_num = tik_instance.Scalar('float16', init_value=MAX_VAL)
        min_num = tik_instance.Scalar('float16', init_value=MIN_VAL)
        # Add ineffective object for 16 alignment
        if descending:
            with tik_instance.for_range(0, num_16 - num) as i:
                input_ub[(num + i) + dest_pos_ub].set_as(min_num)
        else:
            with tik_instance.for_range(0, num_16 - num) as i:
                input_ub[(num + i) + dest_pos_ub].set_as(max_num)

        tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], n_repeat_total, VAL_IDX)

        if cce_product == tbe_platform.ASCEND_310:
            data_out_ub_ = tik_instance.Tensor(dtype, [BLOCK], name="data_out_ub_", scope=tik.scope_ubuf)
            data_indices_ub_int_ = tik_instance.Tensor("int32", [BLOCK], name="data_indices_ub_int_",
                                                       scope=tik.scope_ubuf)
            with tik_instance.for_range(0, num) as i2:
                data_indices_ub_int_.set_as(num - 1 - i2)
                tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
                idx_ub[(num - 1 - i2)].set_as(data_out_ub_[0])
        else:
            idx = tik_instance.Scalar(dtype="float32", init_value=num)
            with tik_instance.for_range(0, num) as i2:
                idx.set_as(idx - 1)
                idx_ub[(num - 1 - i2)].set_as(idx)
        tik_instance.vconcat(input_ub[0], idx_ub[0], n_repeat_total, INT_IDX)

        # 2. vbs16
        tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=n_repeat_total)
        # 3. vms4
        input_ub, dest_pos_ub = vms4(tik_instance, num_16, input_ub, dest_pos_ub)
        # 4. Move Data from UB to OUT
        data_out, data_indices = moveout(tik_instance, num_16, num, data_out, offset_out, input_ub,
                                         dest_pos_ub, data_indices, descending, cce_product)

    else:
        idx_ub = tik_instance.Tensor(dtype, [NUM_BLOCK], name="idx_ub", scope=tik.scope_ubuf)
        tmp_ub = tik_instance.Tensor(dtype, [NUM_BLOCK], name="tmp_ub", scope=tik.scope_ubuf)

        if cce_product == tbe_platform.ASCEND_310:
            data_out_ub_ = tik_instance.Tensor(dtype, [BLOCK], name="data_out_ub_", scope=tik.scope_ubuf)
            data_indices_ub_int_ = tik_instance.Tensor("int32", [BLOCK], name="data_indices_ub_int_",
                                                       scope=tik.scope_ubuf)
            with tik_instance.for_range(0, NUM_BLOCK) as i2:
                data_indices_ub_int_.set_as(i2)
                tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
                idx_ub[i2].set_as(data_out_ub_[0])
        else:
            idx = tik_instance.Scalar(dtype="float32", init_value=0)
            with tik_instance.for_range(0, NUM_BLOCK) as i2:
                idx_ub[i2].set_as(idx)
                idx.set_as(idx + 1)
        offset = (core_idx % used_aicore_num) * num_gm * NUM_BLOCK * PROPOSAL_NUM

        ori_offset = core_idx * num

        input_ub = tik_instance.Tensor(dtype, [NUM_BLOCK * 2 * PROPOSAL_NUM * 2], name="input_ub", scope=tik.scope_ubuf)

        # SORT IN UB
        for i in range(num_gm):
            temp = sort_in_ub(tik_instance, input_ub, idx_ub, tmp_ub, num, i, input_gm, temp, ori_offset, offset,
                              descending)

        # SORT IN GM
        temp = sort_in_gm(tik_instance, temp, num_gm, input_ub, offset)

        # Pick Data from GM to GM
        data_out, data_indices = pick(tik_instance, temp, offset, core_idx, num_2048, data_out, data_indices, input_ub,
                                      num_gm, descending, cce_product)

    return data_out, data_indices


def moveout(tik_instance, num_16, num, data_out, offset_out, input_ub, dest_pos_ub, data_indices, descending,
            cce_product):
    """
    Function: Move UB to GM, and trans y2 from fp16 to int32.
    Modify : 2020-08-03

    Attention : This way is unstable (can't compare two scalar).
    Init base parameters
    Parameters
    ----------
    offset_out, num_16, num, dest_pos_ub : for index compute
    data_out, input_ub, data_indices : for data move
    ----------
    """
    int_list = tik_instance.Tensor("int32", [num_16], name="int_list", scope=tik.scope_ubuf)
    src_pos_ub = num_16 * PROPOSAL_NUM if dest_pos_ub == 0 else 0
    # ascend
    with tik_instance.if_scope(descending is False):
        # data is continuous in GM & gather scattered data together
        with tik_instance.for_range(0, num) as i2:
            input_ub[i2 + src_pos_ub].set_as(input_ub[(num_16 - 1 - i2) * PROPOSAL_NUM + VAL_IDX + dest_pos_ub])
            input_ub[i2 + src_pos_ub + num_16].set_as(input_ub[(num_16 - 1 - i2) * PROPOSAL_NUM + dest_pos_ub])

    # descend
    with tik_instance.else_scope():
        # data is continuous in GM & gather scattered data together
        if cce_product == tbe_platform.ASCEND_310:
            with tik_instance.for_range(0, num) as i2:
                input_ub[i2 + src_pos_ub].set_as(input_ub[i2 * PROPOSAL_NUM + VAL_IDX + dest_pos_ub])
                input_ub[i2 + src_pos_ub + num_16].set_as(input_ub[i2 * PROPOSAL_NUM + dest_pos_ub])
        else:
            tik_instance.vextract(input_ub[src_pos_ub], input_ub[dest_pos_ub], num_16 // BLOCK, VAL_IDX)
            tik_instance.vextract(input_ub[src_pos_ub + num_16], input_ub[dest_pos_ub], num_16 // BLOCK, INT_IDX)

    # conv indices (float16->int32) , and move from UB to GM
    tik_instance.vec_conv(BLOCK, "round", int_list, input_ub[src_pos_ub + num_16], num_16 // BLOCK, 2, 1)

    # move output (float16) from UB to GM
    tik_instance.data_move(data_out[offset_out], input_ub[src_pos_ub], 0, 1, num_16 // BLOCK, 0, 0)
    tik_instance.data_move(data_indices[offset_out], int_list, 0, 1, 2 * num_16 // BLOCK, 0, 0)

    return data_out, data_indices


def sort_in_ub(tik_instance, input_ub, idx_ub, tmp_ub, num, i, input_gm, temp, ori_offset, offset, descending):
    """
    Function: sort in ub.
    Modify : 2020-11-16

    Init base parameters
    Parameters
    ----------
    num, i, ori_offset, offset : for index compute
    input_gm, temp, dtype : for data move
    ----------
    """
    # dest position in UB
    dest_pos_ub = NUM_BLOCK * PROPOSAL_NUM
    repeat_times = NUM_BLOCK // BLOCK
    # 1. Move data from OUT to UB
    tik_instance.data_move(input_ub[dest_pos_ub], input_gm[ori_offset + i * NUM_BLOCK], 0, 1, repeat_times, 0, 0)

    tik_instance.vector_dup(BLOCK, tmp_ub[0], i, repeat_times, 1, 1)

    tik_instance.vconcat(input_ub[0], tmp_ub[0], repeat_times, INT_IDX)
    tik_instance.vconcat(input_ub[0], idx_ub[0], repeat_times, REM_IDX)

    if num < (i + 1) * NUM_BLOCK:
        # aline for NUM_BLOCK
        aline = NUM_BLOCK - num % NUM_BLOCK
        if descending:
            tmp = tik_instance.Scalar('float16', init_value=MIN_VAL)
        # descend
        else:
            tmp = tik_instance.Scalar('float16', init_value=MAX_VAL)
        # Add ineffective object for 16 alignment
        for j in range(aline % BLOCK):
            input_ub[dest_pos_ub + num % NUM_BLOCK + j].set_as(tmp)
        # Add ineffective object for NUM_BLOCK alignment
        if aline > BLOCK - 1:
            tik_instance.vec_dup(BLOCK, input_ub[dest_pos_ub + num % NUM_BLOCK + aline % BLOCK], tmp,
                                 aline // BLOCK, 1)

    tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], repeat_times, VAL_IDX)
    # 2. vrpsort16
    tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=repeat_times)
    # 3. vms4
    input_ub, dest_pos_ub = vms4(tik_instance, NUM_BLOCK, input_ub, dest_pos_ub)
    # 4. Move Data from UB to OUT
    tik_instance.data_move(temp[offset + i * NUM_BLOCK * PROPOSAL_NUM], input_ub[dest_pos_ub], 0, 1,
                           NUM_BLOCK * PROPOSAL_NUM // BLOCK, 0, 0)

    return temp


def vms4(tik_instance, total, input_ub, dest_pos_ub):
    """
    Function: Merge all lists into one.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    total: The number of all object (16 alignment).
    input_ub: UB
    dest_pos_ub: The dest position in UB.
    ----------
    """
    # record the lists info, since overlapping src and dst addresses can lead to perf degradation
    length = total // BLOCK
    num_list = [BLOCK] * length
    src_pos_ub = 0

    while len(num_list) > 1:
        src_pos_ub, dest_pos_ub = dest_pos_ub, src_pos_ub
        index = 0
        offset = 0
        while True:
            res = len(num_list) - index
            if res > 3:
                num_list, input_ub, offset = merge4(tik_instance, num_list, input_ub,
                                                    offset, src_pos_ub, index, dest_pos_ub)
            elif res == 3:
                num_list, input_ub, offset = merge3(tik_instance, num_list, input_ub,
                                                    offset, src_pos_ub, index, dest_pos_ub)
            elif res == 2:
                num_list, input_ub, offset = merge2(tik_instance, num_list, input_ub,
                                                    offset, src_pos_ub, index, dest_pos_ub)
            elif res == 1:
                tik_instance.data_move(input_ub[dest_pos_ub + offset * PROPOSAL_NUM],
                                       input_ub[src_pos_ub + offset * PROPOSAL_NUM], 0, 1,
                                       num_list[index] * PROPOSAL_NUM // BLOCK, 0, 0)
            else:
                break
            index += 1

    return input_ub, dest_pos_ub


def merge4(tik_instance, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
    """
    Function: Merge 4 lists in UB.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    num_list: record the lists info
    offset: used space
    src_pos_ub, dest_pos_ub: position info
    input_ub: UB
    ----------
    """
    src_list = [input_ub[src_pos_ub + offset * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index]) * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index] + num_list[index + 1]) * PROPOSAL_NUM],
                input_ub[
                    src_pos_ub + (offset + num_list[index] + num_list[index + 1] + num_list[index + 2]) * PROPOSAL_NUM]]

    src_list_lengths = [num_list[index], num_list[index + 1], num_list[index + 2], num_list[index + 3]]
    # merge 4 lists
    tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * PROPOSAL_NUM], src_list, src_list_lengths,
                           if_exhausted_suspension=False, valid_bit="1111", repeat_times=1)
    # update the lists info : Merge the four element values and record them in a(num_list)
    num_list[index] = sum(num_list[index:index + 4])
    a = num_list[:index + 1:]
    b = num_list[index + 4::]
    a.extend(b)
    offset += a[index]

    return a, input_ub, offset


def merge3(tik_instance, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
    """
    Function: Merge 3 lists in UB.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    num_list: record the lists info
    offset: used space
    src_pos_ub, dest_pos_ub: position info
    input_ub: UB
    ----------
    """
    src_list = [input_ub[src_pos_ub + offset * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index]) * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index] + num_list[index + 1]) * PROPOSAL_NUM], input_ub[0]]
    src_list_lengths = [num_list[index], num_list[index + 1], num_list[index + 2], 0]
    # merge 3 lists
    tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * PROPOSAL_NUM], src_list, src_list_lengths,
                           if_exhausted_suspension=False, valid_bit="0111", repeat_times=1)
    # update the lists info : Merge the three element values and record them in a(num_list)
    num_list[index] = sum(num_list[index:index + 3])
    a = num_list[:index + 1:]
    b = num_list[index + 3::]
    a.extend(b)
    offset += a[index]

    return a, input_ub, offset


def merge2(tik_instance, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
    """
    Function: Merge 2 lists in UB.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    num_list: record the lists info
    offset: used space
    src_pos_ub, dest_pos_ub: position info
    input_ub: UB
    ----------
    """
    src_list = [input_ub[src_pos_ub + offset * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index]) * PROPOSAL_NUM],
                input_ub[0], input_ub[0]]
    src_list_lengths = [num_list[index], num_list[index + 1], 0, 0]
    # merge 2 lists
    tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * PROPOSAL_NUM], src_list, src_list_lengths,
                           if_exhausted_suspension=False, valid_bit="0011", repeat_times=1)

    # update the lists info : Merge the two element values and record them in num_list
    num_list[index] += num_list[index + 1]
    del num_list[index + 1]
    offset += num_list[index]

    return num_list, input_ub, offset


def sort_in_gm(tik_instance, temp, num_gm, input_ub, offset):
    """
    Function: sort in gm.
    Modify : 2020-11-16

    Init base parameters
    Parameters
    ----------
    num_gm, offset : for index compute
    temp, input_ub : for data move
    ----------
    """
    src_pos_ub = tik_instance.Scalar("int32")
    dest_pos_ub = tik_instance.Scalar("int32")

    with tik_instance.for_range(0, num_gm - 1) as tail:
        src_pos_ub.set_as(0)
        dest_pos_ub.set_as(NUM_BLOCK * 2 * PROPOSAL_NUM)

        tik_instance.data_move(input_ub[src_pos_ub + NUM_BLOCK * PROPOSAL_NUM], temp[offset], 0, 1,
                               (NUM_BLOCK * PROPOSAL_NUM) // BLOCK, 0, 0)
        with tik_instance.for_range(1, num_gm - tail) as i:
            tik_instance.data_move(input_ub[src_pos_ub], temp[offset + NUM_BLOCK * i * PROPOSAL_NUM], 0, 1,
                                   (NUM_BLOCK * PROPOSAL_NUM) // BLOCK, 0, 0)

            tik_instance.vmrgsort4(input_ub[dest_pos_ub],
                                   [input_ub[src_pos_ub], input_ub[src_pos_ub + NUM_BLOCK * PROPOSAL_NUM],
                                    input_ub[0], input_ub[0]], [NUM_BLOCK, NUM_BLOCK, 0, 0],
                                   if_exhausted_suspension=False, valid_bit="0011", repeat_times=1)

            tik_instance.data_move(temp[offset + NUM_BLOCK * (i - 1) * PROPOSAL_NUM], input_ub[dest_pos_ub], 0, 1,
                                   (NUM_BLOCK * PROPOSAL_NUM) // BLOCK, 0, 0)

            dest_pos_ub.set_as(src_pos_ub)
            src_pos_ub.set_as(NUM_BLOCK * 2 * PROPOSAL_NUM - dest_pos_ub)

        # Move Data from UB to GM
        tik_instance.data_move(temp[offset + NUM_BLOCK * (num_gm - tail - 1) * PROPOSAL_NUM],
                               input_ub[src_pos_ub + NUM_BLOCK * PROPOSAL_NUM], 0, 1,
                               (NUM_BLOCK * PROPOSAL_NUM) // BLOCK, 0, 0)

    return temp


def pick(tik_instance, temp, offset, core_idx, num_2048, data_out, data_indices, input_ub, num_gm, descending,
         cce_product):
    """
    Function: pick value from proposal.
    Modify : 2020-11-11
    ----------
    temp, offset, core_idx, num_2048 : for index compute
    data_out, input_ub2, num_gm : for data move
    ----------
    """
    # dest position in UB
    dest_pos_ub = NUM_BLOCK * PROPOSAL_NUM
    repeat_times = NUM_BLOCK // BLOCK

    with tik_instance.for_range(0, num_gm) as i:
        tik_instance.data_move(input_ub[0], temp[offset + NUM_BLOCK * i * PROPOSAL_NUM], 0, 1,
                               (NUM_BLOCK * PROPOSAL_NUM) // BLOCK, 0, 0)
        int_list_1 = tik_instance.Tensor("int32", [NUM_BLOCK], name="int_list_1", scope=tik.scope_ubuf)
        int_list_2 = tik_instance.Tensor("int32", [NUM_BLOCK], name="int_list_2", scope=tik.scope_ubuf)
        int_list_3 = tik_instance.Tensor("int32", [NUM_BLOCK], name="int_list_3", scope=tik.scope_ubuf)
        int_list_4 = tik_instance.Tensor("int32", [NUM_BLOCK], name="int_list_4", scope=tik.scope_ubuf)

        tik_instance.vector_dup(BLOCK, int_list_4, NUM_BLOCK, repeat_times, 1, 2)

        tik_instance.vextract(input_ub[dest_pos_ub], input_ub[0], repeat_times, INT_IDX)
        tik_instance.vextract(input_ub[dest_pos_ub + NUM_BLOCK], input_ub[0], repeat_times, REM_IDX)
        tik_instance.vec_conv(BLOCK, "round", int_list_1, input_ub[dest_pos_ub], repeat_times, 2, 1)
        tik_instance.vec_conv(BLOCK, "round", int_list_2, input_ub[dest_pos_ub + NUM_BLOCK], repeat_times, 2, 1)

        tik_instance.vec_mul(BLOCK, int_list_3, int_list_1, int_list_4, repeat_times, 2, 2, 2)
        tik_instance.vec_add(BLOCK, int_list_1, int_list_2, int_list_3, repeat_times, 2, 2, 2)

        # data is continuous in GM & gather scattered data together

        if cce_product == tbe_platform.ASCEND_310:
            with tik_instance.for_range(0, NUM_BLOCK) as i2:
                input_ub[dest_pos_ub + i2].set_as(input_ub[i2 * PROPOSAL_NUM + VAL_IDX])
        else:
            tik_instance.vextract(input_ub[dest_pos_ub], input_ub[0], repeat_times, VAL_IDX)
        # move output (float16) from UB to GM
        # ascend
        with tik_instance.if_scope(descending is False):
            with tik_instance.for_range(0, NUM_BLOCK) as i2:
                int_list_2[i2].set_as(int_list_1[NUM_BLOCK - i2 - 1])
                input_ub[dest_pos_ub + NUM_BLOCK + i2].set_as(input_ub[dest_pos_ub + NUM_BLOCK - i2 - 1])
            tik_instance.data_move(data_indices[core_idx * num_2048 + NUM_BLOCK * (num_gm - i - 1)], int_list_2, 0, 1,
                                   2 * repeat_times, 0, 0)
            tik_instance.data_move(data_out[core_idx * num_2048 + NUM_BLOCK * (num_gm - i - 1)],
                                   input_ub[dest_pos_ub + NUM_BLOCK], 0, 1, repeat_times, 0, 0)
        # descend
        with tik_instance.else_scope():
            tik_instance.data_move(data_indices[core_idx * num_2048 + NUM_BLOCK * i], int_list_1, 0, 1,
                                   2 * repeat_times, 0, 0)
            tik_instance.data_move(data_out[core_idx * num_2048 + NUM_BLOCK * i], input_ub[dest_pos_ub], 0, 1,
                                   repeat_times, 0, 0)
    return data_out, data_indices


def tune(tik_instance, num, num_16, num_2048, rounds, num_gm, data_out, data_out_, data_indices, data_indices_):
    """
    Function: remove min.
    Modify : 2020-11-17

    Attention : This way is unstable (can't compare two scalar).
    Init base parameters
    Parameters
    ----------
    num, num_2048, rounds, descending : for index compute
    data_out, data_out_ : for data move
    ----------
    """

    if num <= NUM_BLOCK:
        repeat_times = num_16 // BLOCK
        float_ub = tik_instance.Tensor("float16", [num_16], name="float_ub", scope=tik.scope_ubuf)
        int_ub = tik_instance.Tensor("int32", [num_16], name="int_ub", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, rounds) as i:
            tik_instance.data_move(float_ub[0], data_out[i * num_16], 0, 1, repeat_times, 0, 0)
            tik_instance.data_move(data_out_[i * num], float_ub[0], 0, 1, repeat_times, 0, 0)

            tik_instance.data_move(int_ub[0], data_indices[i * num_16], 0, 1, 2 * repeat_times, 0, 0)
            tik_instance.data_move(data_indices_[i * num], int_ub[0], 0, 1, 2 * repeat_times, 0, 0)

    else:
        repeat_times = NUM_BLOCK // BLOCK
        float_ub = tik_instance.Tensor("float16", [NUM_BLOCK], name="float_ub", scope=tik.scope_ubuf)
        int_ub = tik_instance.Tensor("int32", [NUM_BLOCK], name="int_ub", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, rounds) as i:
            with tik_instance.for_range(0, num_gm) as j:
                tik_instance.data_move(float_ub[0], data_out[i * num_2048 + j * NUM_BLOCK], 0, 1,
                                       repeat_times, 0, 0)
                tik_instance.data_move(data_out_[i * num + j * NUM_BLOCK], float_ub[0], 0, 1,
                                       repeat_times, 0, 0)

                tik_instance.data_move(int_ub[0], data_indices[i * num_2048 + j * NUM_BLOCK], 0, 1,
                                       2 * repeat_times, 0, 0)
                tik_instance.data_move(data_indices_[i * num + j * NUM_BLOCK], int_ub[0], 0, 1,
                                       2 * repeat_times, 0, 0)

    return data_out_, data_indices_


class BalanceRoiByArea(object):
    """
    sort rois to be balanced
    """

    def __init__(self, rois, sort_rois, sort_idx, kernel_name):
        self.tik_instance = tik.Tik()
        self.rois_dtype = rois.get("dtype")
        self.rois_shape = rois.get("shape")
        self.idx_dtype = sort_idx.get("dtype")
        self.batch_n = self.rois_shape[0]
        self.proposal_num = self.rois_shape[1]
        self.idx_shape = [self.batch_n]
        self.block_byte_size = 32
        self.block_num = 1
        self.kernel_name = kernel_name
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.loop_num = 2048
        self.loop = math.ceil(self.batch_n / self.loop_num)
        self.tail_num = self.batch_n % self.loop_num

        self.rois_gm = self.tik_instance.Tensor(self.rois_dtype, self.rois_shape, scope=tik.scope_gm, name="rois_gm")
        self.sort_gm = self.tik_instance.Tensor(self.rois_dtype, self.rois_shape, scope=tik.scope_gm, name="sort_gm")
        self.idx_gm = self.tik_instance.Tensor(self.idx_dtype, self.idx_shape, scope=tik.scope_gm, name="idx_gm")
        self.area_gm = self.tik_instance.Tensor(self.rois_dtype, self.idx_shape, scope=tik.scope_gm, name="area_gm",
                                                is_workspace=True)

    def extract_roi(self, roi_ub, x0_ub, y0_ub, x1_ub, y1_ub, loop_idx, num):
        """
        extract roi
        """
        with self.tik_instance.for_range(0, num) as idx:
            x0_ub[loop_idx * self.loop_num + idx].set_as(roi_ub[idx, 1])
            y0_ub[loop_idx * self.loop_num + idx].set_as(roi_ub[idx, 2])
            x1_ub[loop_idx * self.loop_num + idx].set_as(roi_ub[idx, 3])
            y1_ub[loop_idx * self.loop_num + idx].set_as(roi_ub[idx, 4])

    def compute(self):
        """
        compute func
        """
        with self.tik_instance.new_stmt_scope():
            x0_ub = self.tik_instance.Tensor(self.rois_dtype, [self.batch_n], scope=tik.scope_ubuf, name="x0_ub")
            y0_ub = self.tik_instance.Tensor(self.rois_dtype, [self.batch_n], scope=tik.scope_ubuf, name="y0_ub")
            x1_ub = self.tik_instance.Tensor(self.rois_dtype, [self.batch_n], scope=tik.scope_ubuf, name="x1_ub")
            y1_ub = self.tik_instance.Tensor(self.rois_dtype, [self.batch_n], scope=tik.scope_ubuf, name="y1_ub")

            with self.tik_instance.new_stmt_scope():
                roi_ub = self.tik_instance.Tensor(self.rois_dtype, [self.loop_num, self.proposal_num],
                                                  scope=tik.scope_ubuf, name="roi_ub")
                with self.tik_instance.for_range(0, self.loop) as loop_idx:
                    with self.tik_instance.if_scope(loop_idx != self.loop - 1):
                        self.data_move(roi_ub[0, 0], self.rois_gm[loop_idx * self.loop_num, 0],
                                       num=self.loop_num * self.proposal_num)
                        self.extract_roi(roi_ub, x0_ub, y0_ub, x1_ub, y1_ub, loop_idx, self.loop_num)
                    with self.tik_instance.else_scope():
                        self.data_move(roi_ub[0, 0], self.rois_gm[loop_idx * self.loop_num, 0],
                                       num=self.tail_num * self.proposal_num)
                        self.extract_roi(roi_ub, x0_ub, y0_ub, x1_ub, y1_ub, loop_idx, self.tail_num)

            self.data_sub(x1_ub, x1_ub, x0_ub, [0, 0, 0], num=self.batch_n)
            self.data_sub(y1_ub, y1_ub, y0_ub, [0, 0, 0], num=self.batch_n)
            self.data_maxs(x1_ub, x1_ub, 0, [0, 0], self.batch_n)
            self.data_maxs(y1_ub, y1_ub, 0, [0, 0], self.batch_n)
            self.data_sqrt(x1_ub, x1_ub, [0, 0], self.batch_n)
            self.data_sqrt(y1_ub, y1_ub, [0, 0], self.batch_n)
            self.data_mul(x1_ub, x1_ub, y1_ub, [0, 0, 0], num=self.batch_n)

            self.data_move(self.area_gm, x1_ub, num=self.batch_n)

        data_indices = self.sort_area()
        indice_ub = self.tik_instance.Tensor("int32", [self.batch_n], scope=tik.scope_ubuf, name="indice_ub")
        self.data_move(indice_ub, data_indices, num=self.batch_n)
        self.re_rois(indice_ub)

        self.tik_instance.BuildCCE(inputs=[self.rois_gm], outputs=[self.sort_gm, self.idx_gm],
                                   kernel_name=self.kernel_name)

        return self.tik_instance

    def sort_area(self):
        """
        sort rois area
        """
        descending = False
        shape, dtype, num = [self.batch_n], "float16", self.batch_n
        allnum = functools_reduce(lambda x, y: x * y, shape)
        rounds = allnum // num

        num_16 = (num + BLOCK - 1) // BLOCK * BLOCK
        num_2048 = (num + NUM_BLOCK - 1) // NUM_BLOCK * NUM_BLOCK
        num_gm = num_2048 // NUM_BLOCK

        if self.rois_dtype == "float32":
            input_gm = self.tik_instance.Tensor("float16", self.idx_shape, scope=tik.scope_gm, name="input_gm",
                                                is_workspace=True)
            fp32_ub = self.tik_instance.Tensor("float32", self.idx_shape, scope=tik.scope_ubuf, name="fp32_ub")
            fp16_ub = self.tik_instance.Tensor("float16", self.idx_shape, scope=tik.scope_ubuf, name="fp16_ub")

            self.data_move(fp32_ub, self.area_gm, num=self.batch_n)
            self.data_conv(fp16_ub, fp32_ub, [0, 0], mode="", num=self.batch_n, dst_stride=4, src_stride=8)
            self.data_move(input_gm, fp16_ub, num=self.batch_n)
        else:
            input_gm = self.area_gm

        if num <= NUM_BLOCK:
            data_out = self.tik_instance.Tensor(dtype, [rounds * num_16], name="data_out", scope=tik.scope_gm,
                                                is_workspace=True)
            data_indices = self.tik_instance.Tensor("int32", [rounds * num_16], name="data_indices", scope=tik.scope_gm,
                                                    is_workspace=True)
            data_out_ = self.tik_instance.Tensor(dtype, shape, name="data_out_", scope=tik.scope_gm, is_workspace=True)
            data_indices_ = self.tik_instance.Tensor("int32", shape, name="data_indices_", scope=tik.scope_gm,
                                                     is_workspace=True)

        else:
            data_out = self.tik_instance.Tensor(dtype, [rounds * num_2048], name="data_out", scope=tik.scope_gm,
                                                is_workspace=True)
            data_indices = self.tik_instance.Tensor("int32", [rounds * num_2048], name="data_indices",
                                                    scope=tik.scope_gm, is_workspace=True)

            data_out_ = self.tik_instance.Tensor(dtype, shape, name="data_out_", scope=tik.scope_gm, is_workspace=True)
            data_indices_ = self.tik_instance.Tensor("int32", shape, name="data_indices_", scope=tik.scope_gm,
                                                     is_workspace=True)

        cce_product = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
        available_aicore_num = tik.Dprofile().get_aicore_num()
        used_aicore_num = available_aicore_num if rounds > available_aicore_num else rounds

        temp = self.tik_instance.Tensor(dtype, [used_aicore_num * num_2048 * PROPOSAL_NUM], name="temp",
                                        scope=tik.scope_gm, is_workspace=True)

        data_out, data_indices = sort_compute(self.tik_instance, dtype, num, num_16, num_2048, 0,
                                              used_aicore_num, data_out, data_indices, input_gm,
                                              temp, num_gm, descending, cce_product)

        data_out_, data_indices_ = tune(self.tik_instance, num, num_16, num_2048, rounds, num_gm, data_out, data_out_,
                                        data_indices, data_indices_)
        return data_indices_

    def re_rois(self, indice_ub):
        """
        balance rois
        """
        loop = self.batch_n // self.core_num
        tmp_scalar = self.tik_instance.Scalar("int32")
        tmp_roi = self.tik_instance.Tensor(self.rois_dtype, [self.proposal_num], scope=tik.scope_ubuf, name="tmp_roi")
        sort_ub = self.tik_instance.Tensor("int32", [self.batch_n], scope=tik.scope_ubuf, name="sort_ub")

        with self.tik_instance.for_range(0, loop) as idx:
            with self.tik_instance.if_scope(idx % 2 == 0):
                with self.tik_instance.for_range(0, self.core_num) as i:
                    sort_ub[i * loop + idx].set_as(indice_ub[idx * self.core_num + i])

            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, self.core_num) as i:
                    sort_ub[i * loop + idx].set_as(indice_ub[idx * self.core_num + self.core_num - 1 - i])

        with self.tik_instance.for_range(0, self.batch_n) as n:
            tmp_scalar.set_as(sort_ub[n])
            self.data_move(tmp_roi, self.rois_gm[tmp_scalar * self.proposal_num], num=self.proposal_num)
            self.data_move(self.sort_gm[n * self.proposal_num], tmp_roi, num=self.proposal_num)
        self.data_move(self.idx_gm, sort_ub, num=self.batch_n)

    def get_dtype_size(self, dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2}
        return dtype_dict.get(dtype)

    def data_move(self, dst, src, num, src_stride=0, dst_stride=0):
        """
        move data
        """
        sid = 0
        nburst = 1
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        burst_len = (num + data_each_block - 1) // data_each_block
        self.tik_instance.data_move(dst, src, sid, nburst, burst_len, src_stride=src_stride,
                                    dst_stride=dst_stride)

    def single_operator_template(self, op_obj, dst, src, offsets, scalar=None, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        vector_mask_max = 64
        dst_offset, src_offset = offsets

        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        if not dst_stride:
            dst_stride = vector_mask_max // data_each_block
            src_stride = vector_mask_max // data_each_block

        tensor_size = num if num else src.size
        loop = tensor_size // (vector_mask_max * 255)
        dst_blk_stride = 1
        src_blk_stride = 1

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                if scalar is not None:
                    op_obj(vector_mask_max, dst[tmp_dst_offset], src[tmp_src_offset], scalar, 255, dst_blk_stride,
                           src_blk_stride, dst_stride, src_stride)
                else:
                    op_obj(vector_mask_max, dst[tmp_dst_offset], src[tmp_src_offset], 255, dst_blk_stride,
                           src_blk_stride, dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            if scalar is not None:
                op_obj(vector_mask_max, dst[dst_offset], src[src_offset], scalar, repeat_time, dst_blk_stride,
                       src_blk_stride, dst_stride, src_stride)
            else:
                op_obj(vector_mask_max, dst[dst_offset], src[src_offset], repeat_time, dst_blk_stride,
                       src_blk_stride, dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            if scalar is not None:
                op_obj(last_num, dst[dst_offset], src[src_offset], scalar, 1, dst_blk_stride, src_blk_stride,
                       dst_stride, src_stride)
            else:
                op_obj(last_num, dst[dst_offset], src[src_offset], 1, dst_blk_stride, src_blk_stride,
                       dst_stride, src_stride)

    def double_operator_template(self, op_obj, dst, src0, src1, offsets, num=0, dst_stride=None, src0_stride=None,
                                 src1_stride=None):
        """
        tik api template
        """
        vector_mask_max = 64
        dst_offset, src0_offset, src1_offset = offsets

        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        if not dst_stride:
            dst_stride = vector_mask_max // data_each_block
            src0_stride = vector_mask_max // data_each_block
            src1_stride = vector_mask_max // data_each_block

        tensor_size = num if num else src1.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src0_offset = src0_offset + index * vector_mask_max * 255
                tmp_src1_offset = src1_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src0[tmp_src0_offset], src1[tmp_src1_offset], 255,
                       dst_stride, src0_stride, src1_stride)

            dst_offset += loop * vector_mask_max * 255
            src0_offset += loop * vector_mask_max * 255
            src1_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            op_obj(vector_mask_max, dst[dst_offset], src0[src0_offset], src1[src1_offset], repeat_time, dst_stride,
                   src0_stride, src1_stride)
            dst_offset += repeat_time * vector_mask_max
            src0_offset += repeat_time * vector_mask_max
            src1_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            op_obj(last_num, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1, dst_stride, src0_stride,
                   src1_stride)

    def data_sub(self, dst, src0, src1, offsets, num=0, dst_stride=None, src0_stride=None, src1_stride=None):
        """
        tik sub
        """
        self.double_operator_template(self.tik_instance.vec_sub, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_mul(self, dst, src0, src1, offsets, num=0, dst_stride=None, src0_stride=None, src1_stride=None):
        """
        tik mul
        """
        self.double_operator_template(self.tik_instance.vec_mul, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_conv(self, dst, src, offsets, mode="ceil", num=0, dst_stride=8, src_stride=8):
        """
        tik conv
        """
        vector_mask_max = 64
        dst_offset, src_offset = offsets

        tensor_size = num if num else src.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                self.tik_instance.vec_conv(vector_mask_max, mode, dst[tmp_dst_offset], src[tmp_src_offset], 255,
                                           dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            self.tik_instance.vec_conv(vector_mask_max, mode, dst[dst_offset], src[src_offset], repeat_time,
                                       dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)

    def data_sqrt(self, dst, src, offsets, num=0, dst_stride=None, src_stride=None):
        """
        tik sqrt
        """
        self.single_operator_template(self.tik_instance.vsqrt, dst, src, offsets, None, num, dst_stride, src_stride)

    def data_maxs(self, dst, src, scalar, offsets, num=0, dst_stride=None, src_stride=None):
        """
        tik maxs
        """
        self.single_operator_template(self.tik_instance.vmaxs, dst, src, offsets, scalar, num, dst_stride, src_stride)


# 'pylint: disable=unused-argument,too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def balance_rois(rois, sort_rois, sort_idx, kernel_name="balance_rois"):
    """
    op func
    """
    obj = BalanceRoiByArea(rois, sort_rois, sort_idx, kernel_name)
    return obj.compute()
