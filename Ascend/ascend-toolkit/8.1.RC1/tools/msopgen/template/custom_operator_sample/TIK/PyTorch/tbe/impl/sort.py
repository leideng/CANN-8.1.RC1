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
sort
"""

# pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
from functools import reduce as functools_reduce
import tbe.common.register as tbe_register
from tbe import tik
from tbe.common.utils import para_check

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
# sorting limit for data volume
DATA_LIMITE = 100000


def check_supported(x, y1, y2, axis, descending, kernel_name="sort"):
    """
    check the op support situation.
    Go to AICPU when the date in sort axis is over 100K. 
    """
    input_shape = x.get("shape")
    if input_shape[-1] > DATA_LIMITE:
        reason = "The date in sort axis is over 100K."
        return False, reason

    return True, ""


def check(x, y1, y2, axis, kernel_name):
    """
    Function: Check parameters (eg: shape dtype etc).
    Modify : 2020-08-03
    """
    para_check.check_kernel_name(kernel_name)

    shape = y1.get("shape")
    dtype = y1.get("dtype").lower()
    para_check.check_dtype_rule(dtype, ("float16"))
    para_check.check_shape_rule(shape)

    shape = y2.get("shape")
    dtype = y2.get("dtype").lower()
    para_check.check_dtype_rule(dtype, ("int32"))
    para_check.check_shape_rule(shape)

    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    para_check.check_dtype_rule(dtype, ("float16"))
    para_check.check_shape_rule(shape)

    if axis == -1:
        axis = len(shape) - 1

    if axis != len(shape) - 1:
        raise RuntimeError("The dim of sort_op should take the last one.")

    num = shape[axis]

    return shape, dtype, num


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


def moveout(tik_instance, num_16, num, data_out, offset_out, input_ub, dest_pos_ub, data_indices, descending,
            version):
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
        if version == "mini":
            with tik_instance.for_range(0, num) as i2:
                input_ub[i2 + src_pos_ub].set_as(input_ub[i2 * PROPOSAL_NUM + VAL_IDX + dest_pos_ub])
                input_ub[i2 + src_pos_ub + num_16].set_as(input_ub[i2 * PROPOSAL_NUM + dest_pos_ub])
        elif version == "cloud":
            tik_instance.vextract(input_ub[src_pos_ub], input_ub[dest_pos_ub], num_16 // BLOCK, VAL_IDX)
            tik_instance.vextract(input_ub[src_pos_ub + num_16], input_ub[dest_pos_ub], num_16 // BLOCK, INT_IDX)
        else:
            raise RuntimeError("Unexcepted version.")

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
    # index // 2048
    tik_instance.vconcat(input_ub[0], tmp_ub[0], repeat_times, INT_IDX)
    # index % 2048
    tik_instance.vconcat(input_ub[0], idx_ub[0], repeat_times, REM_IDX)

    with tik_instance.if_scope(num < (i + 1) * NUM_BLOCK):
        # aline for NUM_BLOCK
        aline = NUM_BLOCK - num % NUM_BLOCK
        if descending:
            Tmp = tik_instance.Scalar('float16', init_value=MIN_VAL)
        # descend
        else:
            Tmp = tik_instance.Scalar('float16', init_value=MAX_VAL)
        # Add ineffective object for 16 alignment
        with tik_instance.for_range(0, aline % BLOCK) as j:
            input_ub[dest_pos_ub + num % NUM_BLOCK + j].set_as(Tmp)
        # Add ineffective object for NUM_BLOCK alignment
        with tik_instance.if_scope(aline > BLOCK - 1):
            tik_instance.vec_dup(BLOCK, input_ub[dest_pos_ub + num % NUM_BLOCK + aline % BLOCK], Tmp,
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


def pick(tik_instance, temp, offset, core_idx, num_2048, data_out, data_indices, input_ub, num_gm, descending, version):
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
        if version == "cloud":
            tik_instance.vextract(input_ub[dest_pos_ub], input_ub[0], repeat_times, VAL_IDX)
        elif version == "mini":
            with tik_instance.for_range(0, NUM_BLOCK) as i2:
                input_ub[dest_pos_ub + i2].set_as(input_ub[i2 * PROPOSAL_NUM + VAL_IDX])
        else:
            raise RuntimeError("Unexcepted version.")

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
        threadNum = 2 if rounds > 1 else 1
        with tik_instance.for_range(0, rounds, thread_num=threadNum) as i:
            float_ub = tik_instance.Tensor("float16", [num_16], name="float_ub", scope=tik.scope_ubuf)
            int_ub = tik_instance.Tensor("int32", [num_16], name="int_ub", scope=tik.scope_ubuf)

            tik_instance.data_move(float_ub[0], data_out[i * num_16], 0, 1, repeat_times, 0, 0)
            tik_instance.data_move(data_out_[i * num], float_ub[0], 0, 1, repeat_times, 0, 0)

            tik_instance.data_move(int_ub[0], data_indices[i * num_16], 0, 1, 2 * repeat_times, 0, 0)
            tik_instance.data_move(data_indices_[i * num], int_ub[0], 0, 1, 2 * repeat_times, 0, 0)

    else:
        num_res_align = ((num % NUM_BLOCK) + BLOCK - 1) // BLOCK * BLOCK
        repeat_times = NUM_BLOCK // BLOCK
        threadNum = 2 if num_gm > 2 else 1
        with tik_instance.for_range(0, rounds) as i:
            with tik_instance.for_range(0, num_gm - 1, thread_num=threadNum) as j:
                float_ub = tik_instance.Tensor("float16", [NUM_BLOCK], name="float_ub", scope=tik.scope_ubuf)
                int_ub = tik_instance.Tensor("int32", [NUM_BLOCK], name="int_ub", scope=tik.scope_ubuf)
                tik_instance.data_move(float_ub[0], data_out[i * num_2048 + j * NUM_BLOCK], 0, 1,
                                       repeat_times, 0, 0)
                tik_instance.data_move(data_out_[i * num + j * NUM_BLOCK], float_ub[0], 0, 1,
                                       repeat_times, 0, 0)

                tik_instance.data_move(int_ub[0], data_indices[i * num_2048 + j * NUM_BLOCK], 0, 1,
                                       2 * repeat_times, 0, 0)
                tik_instance.data_move(data_indices_[i * num + j * NUM_BLOCK], int_ub[0], 0, 1,
                                       2 * repeat_times, 0, 0)
            # for last block in 32Byte align                  
            float_ub = tik_instance.Tensor("float16", [num_res_align], name="float_ub", scope=tik.scope_ubuf)
            int_ub = tik_instance.Tensor("int32", [num_res_align], name="int_ub", scope=tik.scope_ubuf)
            tik_instance.data_move(float_ub[0], data_out[i * num_2048 + (num_gm - 1) * NUM_BLOCK], 0, 1,
                                   num_res_align // BLOCK, 0, 0)
            tik_instance.data_move(data_out_[i * num + (num_gm - 1) * NUM_BLOCK], float_ub[0], 0, 1,
                                   num_res_align // BLOCK, 0, 0)

            tik_instance.data_move(int_ub[0], data_indices[i * num_2048 + (num_gm - 1) * NUM_BLOCK], 0, 1,
                                   2 * num_res_align // BLOCK, 0, 0)
            tik_instance.data_move(data_indices_[i * num + (num_gm - 1) * NUM_BLOCK], int_ub[0], 0, 1,
                                   2 * num_res_align // BLOCK, 0, 0)

    return data_out_, data_indices_


@tbe_register.register_op_compute("sort")
def sort_compute(tik_instance, dtype, num, num_16, num_2048, core_idx, used_aicore_num, data_out,
                 data_indices, input_gm, temp, num_gm, descending, version):
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
        Max = tik_instance.Scalar('float16', init_value=MAX_VAL)
        Min = tik_instance.Scalar('float16', init_value=MIN_VAL)
        # Add ineffective object for 16 alignment
        if descending:
            with tik_instance.for_range(0, num_16 - num) as i:
                input_ub[(num + i) + dest_pos_ub].set_as(Min)
        else:
            with tik_instance.for_range(0, num_16 - num) as i:
                input_ub[(num + i) + dest_pos_ub].set_as(Max)

        tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], n_repeat_total, VAL_IDX)

        if version == "cloud":
            idx = tik_instance.Scalar(dtype="float32", init_value=num)
            with tik_instance.for_range(0, num) as i2:
                idx.set_as(idx - 1)
                idx_ub[(num - 1 - i2)].set_as(idx)
        elif version == "mini":
            data_out_ub_ = tik_instance.Tensor(dtype, [BLOCK], name="data_out_ub_", scope=tik.scope_ubuf)
            data_indices_ub_int_ = tik_instance.Tensor("int32", [BLOCK], name="data_indices_ub_int_",
                                                       scope=tik.scope_ubuf)
            with tik_instance.for_range(0, num) as i2:
                data_indices_ub_int_.set_as(num - 1 - i2)
                tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
                idx_ub[(num - 1 - i2)].set_as(data_out_ub_[0])
        else:
            raise RuntimeError("Unexcepted version.")

        tik_instance.vconcat(input_ub[0], idx_ub[0], n_repeat_total, INT_IDX)

        # 2. vbs16
        tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=n_repeat_total)
        # 3. vms4
        input_ub, dest_pos_ub = vms4(tik_instance, num_16, input_ub, dest_pos_ub)
        # 4. Move Data from UB to OUT
        data_out, data_indices = moveout(tik_instance, num_16, num, data_out, offset_out, input_ub,
                                         dest_pos_ub, data_indices, descending, version)

    else:
        idx_ub = tik_instance.Tensor(dtype, [NUM_BLOCK], name="idx_ub", scope=tik.scope_ubuf)
        tmp_ub = tik_instance.Tensor(dtype, [NUM_BLOCK], name="tmp_ub", scope=tik.scope_ubuf)
        if version == "cloud":
            idx = tik_instance.Scalar(dtype="float32", init_value=0)
            with tik_instance.for_range(0, NUM_BLOCK) as i2:
                idx_ub[i2].set_as(idx)
                idx.set_as(idx + 1)
        elif version == "mini":
            data_out_ub_ = tik_instance.Tensor(dtype, [BLOCK], name="data_out_ub_", scope=tik.scope_ubuf)
            data_indices_ub_int_ = tik_instance.Tensor("int32", [BLOCK], name="data_indices_ub_int_",
                                                       scope=tik.scope_ubuf)
            with tik_instance.for_range(0, NUM_BLOCK) as i2:
                data_indices_ub_int_.set_as(i2)
                tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
                idx_ub[i2].set_as(data_out_ub_[0])
        else:
            raise RuntimeError("Unexcepted version.")

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
                                      num_gm, descending, version)

    return data_out, data_indices


@tbe_register.register_operator("sort")
@para_check.check_input_type(dict, dict, dict, int, bool, str)
def sort(x, y1, y2, axis=-1, descending=False, kernel_name="sort"):
    """
    Function: Sorts the elements of the input tensor along a given dimension in ascending order by value.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    input(x): dict
        data of input
    output(y1): dict
        data of output
    indices(y2): dict
        data of indices
    dim(axis): int
    descending: bool
    kernel_name: str
        the name of the operator
    ----------
    """
    shape, dtype, num = check(x, y1, y2, axis, kernel_name)
    allnum = functools_reduce(lambda x, y: x * y, shape)
    rounds = allnum // num
    tik_instance = tik.Tik(tik.Dprofile())

    num_16 = (num + BLOCK - 1) // BLOCK * BLOCK
    num_2048 = (num + NUM_BLOCK - 1) // NUM_BLOCK * NUM_BLOCK
    num_gm = num_2048 // NUM_BLOCK

    if num <= NUM_BLOCK:
        input_gm = tik_instance.Tensor(dtype, shape, name="x", scope=tik.scope_gm)
        data_out = tik_instance.Tensor(dtype, [rounds * num_16], name="data_out", scope=tik.scope_gm, is_workspace=True)
        data_indices = tik_instance.Tensor("int32", [rounds * num_16], name="data_indices", scope=tik.scope_gm,
                                           is_workspace=True)
        data_out_ = tik_instance.Tensor(dtype, shape, name="data_out_", scope=tik.scope_gm)
        data_indices_ = tik_instance.Tensor("int32", shape, name="data_indices_", scope=tik.scope_gm)

    else:
        input_gm = tik_instance.Tensor(dtype, shape, name="input_gm", scope=tik.scope_gm)
        data_out = tik_instance.Tensor(dtype, [rounds * num_2048], name="data_out", scope=tik.scope_gm,
                                       is_workspace=True)
        data_indices = tik_instance.Tensor("int32", [rounds * num_2048], name="data_indices", scope=tik.scope_gm,
                                           is_workspace=True)

        data_out_ = tik_instance.Tensor(dtype, shape, name="data_out_", scope=tik.scope_gm)
        data_indices_ = tik_instance.Tensor("int32", shape, name="data_indices_", scope=tik.scope_gm)

    available_aicore_num = tik.Dprofile().get_aicore_num()
    used_aicore_num = available_aicore_num if rounds > available_aicore_num else rounds
    batch_num_per_aicore = rounds // used_aicore_num
    batch_tail = rounds % used_aicore_num

    temp = tik_instance.Tensor(dtype, [used_aicore_num * num_2048 * PROPOSAL_NUM], name="temp",
                               scope=tik.scope_gm, is_workspace=True)

    version = tik.Dprofile().get_product_name()

    with tik_instance.for_range(0, used_aicore_num, block_num=used_aicore_num) as i:
        with tik_instance.for_range(0, batch_num_per_aicore) as j:
            data_out, data_indices = sort_compute(tik_instance, dtype, num, num_16, num_2048, i + j * used_aicore_num,
                                                  used_aicore_num, data_out, data_indices, input_gm,
                                                  temp, num_gm, descending, version)
        with tik_instance.if_scope(i < batch_tail):
            data_out, data_indices = sort_compute(tik_instance, dtype, num, num_16, num_2048,
                                                  batch_num_per_aicore * used_aicore_num + i, used_aicore_num,
                                                  data_out, data_indices, input_gm, temp, num_gm, descending, version)

    data_out_, data_indices_ = tune(tik_instance, num, num_16, num_2048, rounds, num_gm, data_out, data_out_,
                                    data_indices, data_indices_)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_gm], outputs=[data_out_, data_indices_])

    return tik_instance
