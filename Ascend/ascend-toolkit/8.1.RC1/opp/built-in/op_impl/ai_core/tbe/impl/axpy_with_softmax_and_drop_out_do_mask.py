#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
axpy_with_softmax_and_drop_out_do_mask
"""
from functools import reduce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check

SHAPE_SIZE_LIMIT = 1 << 30
UB_NUMBER_BYTE_SIZE = 15
# LINE_THRESHOLD：
# if the UB size is larger than 200k, two rows of data can be processed
# if the UB size is smaller than 200k, one rows of data can be processed
LINE_THRESHOLD = 204800 
BLOCK = 16
VEC_MASK = 128
VEC_MASK_FP32 = 64
VEC_MASK_FP16 = 128
VEC_DUMP_SHAPE = 128
FP32_NUM_PER_BLOCK = 8
FP16_NUM_PER_BLOCK = 16
MAX_REPEAT = 255
DIM_N = 0
DIM_C = 1
DIM_W1 = 2
DIM_W2 = 5
DIM_H1 = 3
DIM_H2 = 4
LEN = 2
LIMIT_LEN = 512


def cal_level(dividend):
    """
    cal_level
    """
    cnt = 0
    while dividend % LEN == 0:
        dividend //= LEN
        cnt += 1
    return cnt, dividend - 1


def total_num(shape):
    """the value of shape_total_num should be returned"""
    shape_total_num = reduce(lambda a, b: a * b, shape)
    return shape_total_num


# 'pylint: disable=too-many-locals,too-many-statements
def data_move_in(offset, params_list, mov_list):
    """
    data_move_in
    """
    _, line, input_shape, tik_instance, _ = params_list
    tensor_input_x1, tensor_input_x2, ub_1, ub_2 = mov_list
    tik_instance.data_move(ub_1, tensor_input_x1[offset], 0, input_shape[DIM_W1],
                           BLOCK * line, (input_shape[DIM_W1] - line) * BLOCK, 0)
    tik_instance.data_move(ub_2, tensor_input_x2[offset], 0, input_shape[DIM_W1],
                           BLOCK * line, (input_shape[DIM_W1] - line) * BLOCK, 0)


def reduce_max_and_sub(params_list, reduce_max_list):
    """
    reduce_max_and_sub
    """
    w_dim, line, input_shape, tik_instance, _ = params_list
    ub_1, ub_2, ub_reducemax, ub_broadcast, ub_dup = reduce_max_list
    cnt, remain = cal_level(input_shape[DIM_W1])
    time = tik_instance.Scalar("int32", name='time', init_value=LEN)
    tik_instance.vmax(VEC_MASK_FP16, ub_2, ub_1,
                      ub_1[w_dim * BLOCK * line // time],
                      w_dim * BLOCK * line // time // VEC_MASK_FP16, 1, 1, 1, 8, 8, 8)
    with tik_instance.for_range(1, cnt) as j:
        time.set_as(time * LEN)
        tik_instance.vmax(VEC_MASK_FP16, ub_2, ub_2,
                            ub_2[w_dim * BLOCK * line // time],
                            w_dim * BLOCK * line // time // VEC_MASK_FP16, 1, 1, 1, 8, 8, 8)
    with tik_instance.if_scope(remain > 0):
        with tik_instance.for_range(1, remain + 1) as j:
            tik_instance.vmax(VEC_MASK_FP16, ub_2[BLOCK * BLOCK * line * (remain - j)],
                              ub_2[BLOCK * BLOCK * line * (remain - j)],
                              ub_2[BLOCK * BLOCK * line * (remain - j + 1)],
                              BLOCK * BLOCK * line // VEC_MASK_FP16, 1, 1, 1, 8, 8, 8)
    tik_instance.vcgmax(VEC_MASK_FP16, ub_reducemax, ub_2, LEN * line, 1, 1, 8)

    tik_instance.vector_dup(VEC_DUMP_SHAPE, ub_dup, tik_instance.Scalar(init_value=0, dtype="uint16"), 1, 1, 8)
    ub_reducemax_int16 = ub_reducemax.reinterpret_cast_to("uint16")
    with tik_instance.for_range(0, line) as j:
        tik_instance.vor(BLOCK, ub_broadcast[BLOCK * BLOCK * j], ub_reducemax_int16[BLOCK * j], ub_dup, BLOCK,
                         1, 1, 0, 1, 0, 0)
    with tik_instance.for_range(0, line) as j:
        tik_instance.vtranspose(ub_broadcast[BLOCK * BLOCK * j], ub_broadcast[BLOCK * BLOCK * j])
    ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")

    with tik_instance.for_range(0, line * BLOCK * BLOCK // VEC_MASK) as idx:
        tik_instance.vsub(VEC_MASK, ub_2[idx * VEC_MASK], ub_1[idx * VEC_MASK],
                          ub_broadcast_fp16[idx * VEC_MASK],
                          input_shape[DIM_W1], 1, 1, 1, line * BLOCK, line * BLOCK, 0)


def conv_and_exp(params_list, exp_list):
    """
    conv_and_exp
    """
    w_dim, line, _, tik_instance, _ = params_list
    ub_1, ub_2, ub_3, ub_cast = exp_list
    repeat_time = line * BLOCK * w_dim // VEC_MASK_FP32
    cnt = repeat_time // MAX_REPEAT
    remain = repeat_time % MAX_REPEAT
    if cnt > 0:
        tik_instance.vconv(VEC_MASK_FP32, "", ub_cast[MAX_REPEAT * VEC_MASK_FP32 * i],
                           ub_2[MAX_REPEAT * VEC_MASK_FP32 * i], MAX_REPEAT, 1, 1, 8, 4)
        tik_instance.vconv(VEC_MASK_FP32, "", ub_cast[MAX_REPEAT * VEC_MASK_FP32 * cnt],
                           ub_2[MAX_REPEAT * VEC_MASK_FP32 * cnt], remain, 1, 1, 8, 4)

        tik_instance.vexp(VEC_MASK_FP32, ub_cast[MAX_REPEAT * VEC_MASK_FP32 * i],
                          ub_cast[MAX_REPEAT * VEC_MASK_FP32 * i], MAX_REPEAT, 1, 1, 8, 8)
        tik_instance.vexp(VEC_MASK_FP32, ub_cast[MAX_REPEAT * VEC_MASK_FP32 * cnt],
                          ub_cast[MAX_REPEAT * VEC_MASK_FP32 * cnt], remain, 1, 1, 8, 8)

    else:
        tik_instance.vconv(VEC_MASK_FP32, "", ub_cast, ub_2, remain, 1, 1, 8, 4)
        tik_instance.vexp(VEC_MASK_FP32, ub_cast, ub_cast, remain, 1, 1, 8, 8)


def reduce_sum_and_div(params_list, reduce_sum_and_div_list):
    """
    reduce_sum_and_div
    """
    w_dim, line, input_shape, tik_instance, _ = params_list
    ub_cast, ub_reduceadd, ub_reduceadd_fp16, ub_broadcast, ub_dup, ub_3, ub_4, ub_dup_fp32, ub_1 = \
        reduce_sum_and_div_list
    cnt, remain = cal_level(input_shape[DIM_W1])
    tik_instance.vmuls(VEC_MASK_FP32, ub_3, ub_cast,
                       tik_instance.Scalar(init_value=1, dtype="float32"),
                       line * BLOCK * w_dim // VEC_MASK_FP32, 1, 1, 8, 8)

    time = tik_instance.Scalar("int32", name='time', init_value=1)
    # reduce_add
    with tik_instance.for_range(0, cnt) as j:
        time.set_as(time * LEN)
        tik_instance.vadd(VEC_MASK_FP32, ub_cast, ub_cast,
                          ub_cast[w_dim * BLOCK * line // time],
                          w_dim * BLOCK * line // time // VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
    with tik_instance.if_scope(remain > 0):
        with tik_instance.for_range(1, remain + 1) as j:
            tik_instance.vadd(VEC_MASK_FP32, ub_cast[BLOCK * BLOCK * line * (remain - j)],
                              ub_cast[BLOCK * BLOCK * line * (remain - j)],
                              ub_cast[BLOCK * BLOCK * line * (remain - j + 1)],
                              BLOCK * BLOCK * line // VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
    tik_instance.vcadd(BLOCK, ub_reduceadd, ub_cast, BLOCK * line, 1, 1, 2)


    # vrec
    tik_instance.vrec(line * BLOCK, ub_reduceadd, ub_reduceadd, 1, 1, 1, 0, 0)

    with tik_instance.for_range(0, line * input_shape[DIM_H2] / 8) as j:
        with tik_instance.for_range(0, 8) as k:
            tik_instance.vector_dup(16, ub_dup_fp32[j * 128 + 16 * k],
                                    tik_instance.Scalar(init_value=ub_reduceadd[j * 8 + k], dtype="float32"), 1, 1, 8)

    with tik_instance.for_range(0, line * BLOCK * BLOCK // VEC_MASK_FP32) as idx:
        tik_instance.vmul(VEC_MASK_FP32, ub_4[idx * VEC_MASK_FP32], ub_3[idx * VEC_MASK_FP32],
                          ub_dup_fp32[idx * VEC_MASK_FP32],
                          input_shape[DIM_W1], 1, 1, 1, line * BLOCK * 2, line * BLOCK * 2, 0)

    tik_instance.vconv(VEC_MASK_FP32, "", ub_1, ub_4,
                       w_dim * line // (VEC_MASK_FP32 // FP16_NUM_PER_BLOCK), 1, 1, 4, 8)



def mul_with_dropout(params_list, ub_2, ub_3, ub_4, mov_list):
    """
    mul_with_dropout
    """
    w_dim, line, input_shape, tik_instance, input_keep_prob = params_list
    mask_input, offset, mask_shape = mov_list
    ub_mask = tik_instance.Tensor("uint8", mask_shape, scope=tbe_platform.scope_ubuf, name="ub_mask")
    ub_mask_fp16 = tik_instance.Tensor("float16", mask_shape, scope=tbe_platform.scope_ubuf, name="ub_mask_fp16")
    tik_instance.data_move(ub_mask, mask_input[offset], 0, input_shape[DIM_W1], BLOCK * line // LEN,
                           (input_shape[DIM_W1] - line) * BLOCK // LEN, 0)
    tik_instance.vconv(VEC_MASK, "", ub_mask_fp16, ub_mask, line * BLOCK * w_dim // VEC_MASK, 1, 1, 8, 4)

    # vmuls and vmul
    tik_instance.vmuls(VEC_MASK_FP32, ub_3, ub_4,
                       tik_instance.Scalar(init_value=1 / input_keep_prob, dtype="float32"),
                       line * BLOCK * w_dim // VEC_MASK_FP32, 1, 1, 8, 8)
    tik_instance.vconv(VEC_MASK_FP32, "", ub_2, ub_3,
                       w_dim * line // (VEC_MASK_FP32 // FP16_NUM_PER_BLOCK), 1, 1, 4, 8)
    tik_instance.vmul(VEC_MASK, ub_2, ub_mask_fp16, ub_2, line * BLOCK * w_dim // VEC_MASK,
                      1, 1, 1, 8, 8, 8)


def mul_with_dropout_bit(params_list, ub_2, ub_3, ub_4, mov_list):
    """
    mul_with_dropout_bit
    """
    w_dim, line, input_shape, tik_instance, input_keep_prob = params_list
    mask_input, offset, sel_mode = mov_list

    # vmuls and vmul
    tik_instance.vmuls(VEC_MASK_FP32, ub_3, ub_4,
                       tik_instance.Scalar(init_value=1 / input_keep_prob, dtype="float32"),
                       line * BLOCK * w_dim // VEC_MASK_FP32, 1, 1, 8, 8)
    tik_instance.vconv(VEC_MASK_FP32, "", ub_2, ub_3,
                       w_dim * line // (VEC_MASK_FP32 // FP16_NUM_PER_BLOCK), 1, 1, 4, 8)
    dup_sel_0 = tik_instance.Tensor("float16", (128,), scope=tik.scope_ubuf, name="dup_sel_0")
    tik_instance.vec_dup(128, dup_sel_0, 0, 1, 0)
    ub_mask_0 = tik_instance.Tensor("uint8", (input_shape[DIM_W1] * line * BLOCK * 2,),
                                    scope=tik.scope_ubuf, name="ub_mask_0")
    tik_instance.data_move(ub_mask_0, mask_input[offset // 8], 0, input_shape[DIM_W1], line,
                           (input_shape[DIM_W1] - line), 0)
    if sel_mode == 0:
        ub_mask_1 = tik_instance.Tensor("uint8", (input_shape[DIM_W1] * line * BLOCK * 2,),
                                        scope=tik.scope_ubuf, name="ub_mask_1")
        tik_instance.data_move(ub_mask_1, mask_input[offset // 8 + 16], 0, input_shape[DIM_W1], line,
                               (input_shape[DIM_W1] - line), 0)
        with tik_instance.for_range(0, input_shape[DIM_W1] * line) as mask_idx:
            tik_instance.vec_sel(128, 0, ub_2[mask_idx*256], ub_mask_0[mask_idx*32], ub_2[mask_idx*256],
                                 dup_sel_0, 1, 8, 8, 0)
            tik_instance.vec_sel(128, 0, ub_2[mask_idx*256 + 128], ub_mask_1[mask_idx*32], ub_2[mask_idx*256 + 128],
                                 dup_sel_0, 1, 8, 8, 0)
    else:
        tik_instance.vec_sel(128, 1, ub_2, ub_mask_0, ub_2, 0, input_shape[DIM_W1] * line * 2, 8, 8, 0)


def move_data_out(offset, params_list, ub_1, ub_2, output1, output2):
    """
    move_data_out
    """
    w_dim, line, input_shape, tik_instance, _ = params_list

    tik_instance.data_move(output1[offset], ub_1, 0, input_shape[DIM_W1],
                           BLOCK * line, 0, (input_shape[DIM_H1] - line) * BLOCK)
    tik_instance.data_move(output2[offset], ub_2, 0, input_shape[DIM_W1],
                           BLOCK * line, 0, (input_shape[DIM_H1] - line) * BLOCK)


def cal_prarms_list(input_tensor, input_keep_prob):
    """
    cal_prarms_list
    """
    input_shape = input_tensor.get("shape")
    tik_instance = tik.Tik(tik.Dprofile(), disable_debug=False)
    aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    batch_per_core = (input_shape[DIM_N] * input_shape[DIM_C] + aicore_num - 1) // aicore_num
    batch_small_per_core = batch_per_core - 1
    batch_large_core_num = input_shape[DIM_N] * input_shape[DIM_C] - batch_small_per_core * aicore_num
    ele_per_batch = input_shape[DIM_W1] * input_shape[DIM_H1] * input_shape[DIM_H2] * input_shape[DIM_W2]
    w_dim = input_shape[DIM_W1] * input_shape[DIM_W2]
    line = 2 if ub_size > LINE_THRESHOLD else 1
    ranges = input_shape[DIM_H1] // line
    shape = (input_shape[DIM_W1], line * BLOCK, input_shape[DIM_W2])
    return [w_dim, line, input_shape, tik_instance, input_keep_prob], \
           [aicore_num, batch_per_core, batch_large_core_num, batch_small_per_core, ele_per_batch, ranges, shape]


# 'pylint: disable=unused-argument,too-many-arguments
# 'pylint: disable=too-many-locals,too-many-statements
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def axpy_with_softmax_and_drop_out_do_mask(x1, x2, mask, y1, y2, alpha, input_keep_prob, axis=-1,
                                           kernel_name="axpy_with_softmax_and_drop_out_do_mask"):
    mask_shape = mask.get("shape")
    tensor_dtype = x1.get("dtype").lower()
    mask_dtype = mask.get("dtype").lower()

    soc_version = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
    if soc_version == "Ascend910B":
        sel_mode = 1
    else:
        sel_mode = 0


    params_list, core_attr_list = cal_prarms_list(x1, input_keep_prob)
    w_dim, line, tensor_shape, tik_instance, _ = params_list
    if w_dim == LIMIT_LEN:
        return axpy_with_softmax_and_drop_out_do_mask_w_large(x1, x2, mask, y1, y2, alpha, input_keep_prob, sel_mode,
                                                              axis, kernel_name)
    else:
        aicore_num, batch_per_core, batch_large_core_num, batch_small_per_core, ele_per_batch, \
            batch_range, shape = core_attr_list

        tensor_input_x1, tensor_input_x2, mask_input, output1, output2 = \
            create_gm_tensor(tik_instance, tensor_shape, tensor_dtype, mask_dtype, mask_shape)
        
        mask_is_bit = False
        if total_num(mask_shape) != total_num(tensor_shape):
            mask_is_bit = True

        with tik_instance.for_range(0, aicore_num, block_num=aicore_num) as core_index:
            ub_1 = tik_instance.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="ub_1")
            ub_2 = tik_instance.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="ub_2")
            ub_cast = tik_instance.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="ub_cast")
            ub_3 = tik_instance.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="ub_3")
            ub_4 = tik_instance.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="ub_4")

            ub_reducemax = tik_instance.Tensor("float16", (line * BLOCK,),
                                            scope=tbe_platform.scope_ubuf, name="ub_reducemax")
            ub_reduceadd = tik_instance.Tensor("float32", (line * BLOCK,),
                                            scope=tbe_platform.scope_ubuf, name="ub_reduceadd")
            ub_reduceadd_fp16 = tik_instance.Tensor("float16", (line * BLOCK,), scope=tbe_platform.scope_ubuf,
                                                    name="ub_reduceadd_fp16")
            ub_dup = tik_instance.Tensor("uint16", (VEC_DUMP_SHAPE,), scope=tbe_platform.scope_ubuf, name="ub_dup")
            ub_dup_fp32 = tik_instance.Tensor("float32", (32, 16), scope=tbe_platform.scope_ubuf, name="ub_dup_fp32")
            ub_broadcast = tik_instance.Tensor("uint16", (line * BLOCK * BLOCK,), scope=tbe_platform.scope_ubuf,
                                            name="ub_broadcast")

            offset = tik_instance.Scalar("int32", name="offset")
            core_offset = tik_instance.Scalar("int32", name="core_offset")
            batch_per_core_truth = tik_instance.Scalar("int32", name="batch_per_core_truth")
            with tik_instance.if_scope(core_index < batch_large_core_num):
                batch_per_core_truth.set_as(batch_per_core)
                core_offset.set_as(core_index * batch_per_core * ele_per_batch)
            with tik_instance.else_scope():
                batch_per_core_truth.set_as(batch_small_per_core)
                core_offset.set_as(batch_large_core_num * batch_per_core * ele_per_batch + \
                                   (core_index - batch_large_core_num) * batch_small_per_core * ele_per_batch)

            with tik_instance.for_range(0, batch_per_core_truth) as index:
                with tik_instance.for_range(0, batch_range) as i:
                    offset.set_as(core_offset + index * ele_per_batch + i * line * BLOCK * BLOCK)
                    data_move_in(offset, params_list,
                                [tensor_input_x1, tensor_input_x2, ub_1, ub_2])
                    tik_instance.vec_axpy(VEC_MASK_FP16, ub_1, ub_2, alpha,
                                          w_dim * line // (VEC_MASK_FP16 // FP16_NUM_PER_BLOCK), 8, 8)
                    reduce_max_and_sub(params_list, [ub_1, ub_2, ub_reducemax, ub_broadcast, ub_dup])
                    conv_and_exp(params_list, [ub_1, ub_2, ub_3, ub_cast])
                    reduce_sum_and_div(params_list, [ub_cast, ub_reduceadd, ub_reduceadd_fp16, ub_broadcast, ub_dup,
                                                    ub_3, ub_4, ub_dup_fp32, ub_1])
                    if mask_is_bit:
                        mul_with_dropout_bit(params_list, ub_2, ub_3, ub_4, [mask_input, offset, sel_mode])
                    else:
                        mul_with_dropout(params_list, ub_2, ub_3, ub_4, [mask_input, offset, shape])
                    move_data_out(offset, params_list, ub_1, ub_2, output1, output2)
        tik_instance.BuildCCE(kernel_name=kernel_name,
                            inputs=[tensor_input_x1, tensor_input_x2, mask_input], outputs=[output1, output2])
        return tik_instance


def axpy_with_softmax_and_drop_out_do_mask_w_large(x1, x2, mask, y1, y2, alpha, input_keep_prob, sel_mode, axis=-1,
                                                   kernel_name="axpy_with_softmax_and_drop_out_do_mask"):
    """
    axpy_v2 + softmax_v2 + drop_out_do_mask_v3_d
    """
    check_input(x1, x2, mask, kernel_name)
    (tensor_shape, tensor_dtype, tensor_mask_dtype, tensor_mask_shape) = get_shape_and_dtype(x1, x2, mask)
    tik_instance = tik.Tik(tik.Dprofile(), disable_debug=False)
    (tensor_input_x1, tensor_input_x2, tensor_mask, tensor_output_y1, tensor_output_y2) = \
        create_gm_tensor(tik_instance, tensor_shape, tensor_dtype, tensor_mask_dtype, tensor_mask_shape)
    aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    block_count_large = (tensor_shape[0] * tensor_shape[1] + aicore_num - 1) // aicore_num
    block_count_small = block_count_large - 1
    batch_large_core_num = tensor_shape[0] * tensor_shape[1] - block_count_small * aicore_num
    h_len, w_len = tensor_shape[2], tensor_shape[3]
    aicore_process_num = h_len * w_len * 16 * 16
    mask_is_bit = False
    if total_num(tensor_mask_shape) != total_num(tensor_shape):
        mask_is_bit = True

    block_count_truth = tik_instance.Scalar("int32", name='block_count_truth')
    core_offset = tik_instance.Scalar("int32", name="core_offset")
    move_offset = tik_instance.Scalar("int32", name="move_offset")

    dup_sel_0 = tik_instance.Tensor("float16", (128,), scope=tik.scope_ubuf, name="dup_sel_0")
    tik_instance.vec_dup(128, dup_sel_0, 0, 1, 0)

    with tik_instance.for_range(0, aicore_num, block_num=aicore_num) as blockid:
        with tik_instance.if_scope(blockid < batch_large_core_num):
            block_count_truth.set_as(block_count_large)
            core_offset.set_as(blockid * block_count_large)
        with tik_instance.else_scope():
            block_count_truth.set_as(block_count_small)
            core_offset.set_as(batch_large_core_num * block_count_large + \
                               (blockid - batch_large_core_num) * block_count_small)
        with tik_instance.for_range(0, block_count_truth) as i:
            with tik_instance.for_range(0, h_len // 2) as j:
                ub_1 = tik_instance.Tensor("float16", (w_len, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_1")
                ub_2 = tik_instance.Tensor("float16", (w_len, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_2")
                
                move_offset.set_as((core_offset + i) * aicore_process_num + j * 512)
                # gm->ub
                tik_instance.data_move(ub_1[0], tensor_input_x1[move_offset], 0, h_len, 32, w_len * 16 - 32, 0)
                tik_instance.data_move(ub_2[0], tensor_input_x2[move_offset], 0, h_len, 32, w_len * 16 - 32, 0)

                # axpyv2
                tik_instance.vmuls(VEC_MASK, ub_2, ub_2, tik_instance.Scalar(init_value=alpha, dtype="float16"),
                                   w_len * 4, 1, 1, 8, 8)
                tik_instance.vadd(VEC_MASK, ub_1, ub_1, ub_2, w_len * 4, 1, 1, 1, 8, 8, 8)

                ub_broadcast = tik_instance.Tensor("uint16", (32 * 16,), scope=tbe_platform.scope_ubuf,
                                                   name="ub_broadcast")
                with tik_instance.new_stmt_scope():
                    ub_reducemax = tik_instance.Tensor("float16", (32,), scope=tbe_platform.scope_ubuf, 
                                                       name="ub_reducemax")
                    ub_dup = tik_instance.Tensor("uint16", (128,), scope=tbe_platform.scope_ubuf, name="ub_dup")
                    ub_cast = tik_instance.Tensor("float32", (w_len, 32, 16), scope=tbe_platform.scope_ubuf,
                                                  name="ub_cast")
                    ub_reduceadd = tik_instance.Tensor("float32", (32,), scope=tbe_platform.scope_ubuf,
                                                       name="ub_reduceadd")
                    ub_reduceadd_high_preci = tik_instance.Tensor("float32", (32,), scope=tbe_platform.scope_ubuf,
                                                                  name="ub_reduceadd_high_preci")
                    work_tensor_ub = tik_instance.Tensor("float32", (64,), scope=tbe_platform.scope_ubuf,
                                                         name="work_tensor_ub")
                    ub_reduceadd_fp16 = tik_instance.Tensor("float16", (32,), scope=tbe_platform.scope_ubuf,
                                                            name="ub_reduceadd_fp16")
                    # get max element
                    get_max_ele(tik_instance, ub_1, ub_2, ub_reducemax, w_len)

                    # ub_reducemax broadcast 32 -> (32, 16)
                    ub_broadcast_fp16 = broadcast(tik_instance, ub_dup, ub_reducemax, ub_broadcast)

                    # x - x_max
                    with tik_instance.for_range(0, 4) as idx:
                        tik_instance.vsub(VEC_MASK, ub_2[idx * 128], ub_1[idx * 128], ub_broadcast_fp16[idx * 128],
                                          32, 1, 1, 1, w_len, w_len, 0)

                    # exp
                    exp(tik_instance, ub_2, ub_1, ub_cast, w_len)

                    # sum exp
                    sum_exp(tik_instance, ub_cast, ub_reduceadd, w_len)

                    # 1 / sum_exp
                    tik_instance.vec_rec_high_preci(32, ub_reduceadd_high_preci[0], ub_reduceadd[0],
                                                    work_tensor_ub[0:], 1, 4, 4)
                    tik_instance.vconv(32, "", ub_reduceadd_fp16[0], ub_reduceadd_high_preci[0], 1, 1, 1, 0, 0)

                    # ub_reduceadd_fp16 broadcast 32 -> (32, 16)
                    ub_broadcast_fp16 = broadcast(tik_instance, ub_dup, ub_reduceadd_fp16, ub_broadcast)

                ub_temp = tik_instance.Tensor("float16", (w_len, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_4")
                # calculate exp * (1 / sum_exp)
                with tik_instance.for_range(0, 4) as idx:
                    tik_instance.vmul(VEC_MASK, ub_temp[idx * 128], ub_1[idx * 128], ub_broadcast_fp16[idx * 128],
                                      32, 1, 1, 1, w_len, w_len, 0)

                # dropoutdomaskv3
                with tik_instance.if_scope(input_keep_prob == 0):
                    tik_instance.vmuls(VEC_MASK, ub_2[0], ub_temp[0],
                                       tik_instance.Scalar(init_value=0.0, dtype="float16"), w_len * 4, 1, 1, 8, 8)
                with tik_instance.else_scope():
                    tik_instance.vmuls(VEC_MASK, ub_2[0], ub_temp[0],
                                       tik_instance.Scalar(init_value=1 / input_keep_prob, dtype="float16"),
                                       w_len * 4, 1, 1, 8, 8)

                if not mask_is_bit:
                    ub_mask = tik_instance.Tensor("uint8", (w_len, 32, 16), scope=tbe_platform.scope_ubuf,
                                                  name="ub_mask")
                    ub_mask_fp16 = tik_instance.Tensor("float16", (w_len, 32, 16), scope=tbe_platform.scope_ubuf,
                                                    name="ub_mask_fp16")
                    tik_instance.data_move(ub_mask[0], tensor_mask[move_offset], 0, h_len, 16, w_len * 8 - 16, 0)
                    tik_instance.vconv(VEC_MASK, "", ub_mask_fp16[0], ub_mask[0], w_len * 4, 1, 1, 8, 4)
                    tik_instance.vmul(VEC_MASK, ub_1[0], ub_mask_fp16[0], ub_2[0], w_len * 4, 1, 1, 1, 8, 8, 8)
                else:
                    large_mask_bit(tik_instance, tensor_mask, move_offset, sel_mode, ub_1, ub_2, dup_sel_0)

                # ub -> gm
                tik_instance.data_move(tensor_output_y1[move_offset], ub_temp[0], 0, h_len, 32, 0, w_len * 16 - 32)
                tik_instance.data_move(tensor_output_y2[move_offset], ub_1[0], 0, h_len, 32, 0, w_len * 16 - 32)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[tensor_input_x1, tensor_input_x2, tensor_mask],
                          outputs=[tensor_output_y1, tensor_output_y2])
    return tik_instance


def check_input(x1, x2, mask, kernel_name):
    para_check.check_kernel_name(kernel_name)
    para_check.check_dtype_rule(x1.get('dtype').lower(), ("float16"))
    para_check.check_dtype_rule(x2.get('dtype').lower(), ("float16"))
    para_check.check_dtype_rule(mask.get('dtype').lower(), ("uint8"))
    para_check.check_shape_rule(x1.get('shape'), max_shape_num=SHAPE_SIZE_LIMIT)
    para_check.check_shape_rule(x2.get('shape'), max_shape_num=SHAPE_SIZE_LIMIT)
    para_check.check_shape_rule(mask.get('shape'), max_shape_num=SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(x1.get('shape'), SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(x2.get('shape'), SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(mask.get('shape'), SHAPE_SIZE_LIMIT)


def get_shape_and_dtype(x1,  x2, mask):
    tensor_shape_x1 = x1.get("shape")
    tensor_shape_x2 = x2.get("shape")
    tensor_mask_shape = mask.get("shape")
    tensor_dtype_x1 = x1.get("dtype").lower()
    tensor_dtype_x2 = x2.get("dtype").lower()
    tensor_mask_dtype = mask.get("dtype").lower()
    tensor_shape = tensor_shape_x1
    tensor_dtype = tensor_dtype_x1
    res_list = (tensor_shape, tensor_dtype, tensor_mask_dtype, tensor_mask_shape)
    return res_list


def create_gm_tensor(tik_instance, tensor_shape, tensor_dtype, tensor_mask_dtype, tensor_mask_shape):
    tensor_input_x1 = tik_instance.Tensor(tensor_dtype,
                                          tensor_shape,
                                          name="tensor_input_x1",
                                          scope=tbe_platform.scope_gm)
    tensor_input_x2 = tik_instance.Tensor(tensor_dtype,
                                          tensor_shape,
                                          name="tensor_input_x2",
                                          scope=tbe_platform.scope_gm)
    tensor_mask = tik_instance.Tensor(tensor_mask_dtype,
                                      tensor_mask_shape,
                                      name="tensor_mask",
                                      scope=tbe_platform.scope_gm)
    tensor_output_y1 = tik_instance.Tensor(tensor_dtype,
                                           tensor_shape,
                                           name="tensor_output_y1",
                                           scope=tbe_platform.scope_gm)
    tensor_output_y2 = tik_instance.Tensor(tensor_dtype,
                                           tensor_shape,
                                           name="tensor_output_y2",
                                           scope=tbe_platform.scope_gm)
    gm_tensor_tuple = (tensor_input_x1, tensor_input_x2, tensor_mask, tensor_output_y1, tensor_output_y2)
    return gm_tensor_tuple


def get_max_ele(tik_instance, ub_1, ub_2, ub_reducemax, w_len):
    tik_instance.vmax(VEC_MASK, ub_2[0], ub_1[0], ub_1[w_len * 32 * 8], w_len * 2, 1, 1, 1, 8, 8, 8)
    tik_instance.vmax(VEC_MASK, ub_2[0], ub_2[0], ub_2[w_len * 32 * 4], w_len, 1, 1, 1, 8, 8, 8)
    tik_instance.vmax(VEC_MASK, ub_2[0], ub_2[0], ub_2[w_len * 32 * 2], 16, 1, 1, 1, 8, 8, 8)
    tik_instance.vmax(VEC_MASK, ub_2[0], ub_2[0], ub_2[w_len * 32], 8, 1, 1, 1, 8, 8, 8)
    tik_instance.vmax(VEC_MASK, ub_2[0], ub_2[0], ub_2[w_len * 16], 4, 1, 1, 1, 8, 8, 8)
    tik_instance.vcgmax(VEC_MASK, ub_reducemax[0], ub_2[0], 4, 1, 1, 8)


def broadcast(tik_instance, ub_dup, ub_need_broadcast, ub_broadcast):
    tik_instance.vector_dup(VEC_MASK, ub_dup[0], tik_instance.Scalar(init_value=0, dtype="uint16"), 1, 1, 8)
    ub_need_broadcast_int16 = ub_need_broadcast.reinterpret_cast_to("uint16")
    tik_instance.vor(16, ub_broadcast[0], ub_need_broadcast_int16[0], ub_dup[0], 16, 1, 1, 0, 1, 0, 0)
    tik_instance.vor(16, ub_broadcast[256], ub_need_broadcast_int16[16], ub_dup[0], 16, 1, 1, 0, 1, 0, 0)

    tik_instance.vtranspose(ub_broadcast[0], ub_broadcast[0])
    tik_instance.vtranspose(ub_broadcast[256], ub_broadcast[256])
    ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")
    return ub_broadcast_fp16


def exp(tik_instance, ub_2, ub_3, ub_cast, w_len):
    tik_instance.vconv(VEC_MASK_FP32, "", ub_cast[0], ub_2[0], MAX_REPEAT, 1, 1, 8, 4)
    tik_instance.vconv(VEC_MASK_FP32, "", ub_cast[VEC_MASK_FP32 * MAX_REPEAT],
                       ub_2[VEC_MASK_FP32 * MAX_REPEAT], 1, 1, 1, 8, 4)
    tik_instance.vexp(VEC_MASK_FP32, ub_cast[0], ub_cast[0], MAX_REPEAT, 1, 1, 8, 8)
    tik_instance.vexp(VEC_MASK_FP32, ub_cast[VEC_MASK_FP32 * MAX_REPEAT],
                      ub_cast[VEC_MASK_FP32 * MAX_REPEAT], 1, 1, 1, 8, 8)
    tik_instance.vconv(VEC_MASK_FP32, "", ub_3[0], ub_cast[0], MAX_REPEAT, 1, 1, 4, 8)
    tik_instance.vconv(VEC_MASK_FP32, "", ub_3[VEC_MASK_FP32 * MAX_REPEAT],
                       ub_cast[VEC_MASK_FP32 * MAX_REPEAT], 1, 1, 1, 4, 8)


def sum_exp(tik_instance, ub_cast, ub_reduceadd, w_len):
    tik_instance.vadd(VEC_MASK_FP32, ub_cast[0], ub_cast[0], ub_cast[w_len * 32 * 8], w_len * 4, 1, 1, 1, 8, 8, 8)
    tik_instance.vadd(VEC_MASK_FP32, ub_cast[0], ub_cast[0], ub_cast[w_len * 32 * 4], w_len * 2, 1, 1, 1, 8, 8, 8)
    tik_instance.vadd(VEC_MASK_FP32, ub_cast[0], ub_cast[0], ub_cast[w_len * 32 * 2], 32, 1, 1, 1, 8, 8, 8)
    tik_instance.vadd(VEC_MASK_FP32, ub_cast[0], ub_cast[0], ub_cast[w_len * 32], 16, 1, 1, 1, 8, 8, 8)
    tik_instance.vadd(VEC_MASK_FP32, ub_cast[0], ub_cast[0], ub_cast[w_len * 16], 8, 1, 1, 1, 8, 8, 8)
    tik_instance.vcadd(16, ub_reduceadd[0], ub_cast[0], 32, 1, 1, 2)

def large_mask_bit(tik_instance, tensor_mask, move_offset, sel_mode, ub_1, ub_2, dup_sel_0):
    ub_mask_0 = tik_instance.Tensor("uint8", (32*32*2,), scope=tik.scope_ubuf, name="ub_mask_0")
    tik_instance.data_move(ub_mask_0, tensor_mask[move_offset // 8], 0, 32, 2, 32 - 2, 0)
    if sel_mode == 0:
        ub_mask_1 = tik_instance.Tensor("uint8", (32*32*2,), scope=tik.scope_ubuf, name="ub_mask_1")
        tik_instance.data_move(ub_mask_1, tensor_mask[move_offset // 8 + 16], 0, 32, 2, 32 - 2, 0)
        for mask_idx in range(0, 64):
            tik_instance.vec_sel(128, 0, ub_1[mask_idx*256], ub_mask_0[mask_idx*32], ub_2[mask_idx*256],
                                    dup_sel_0, 1, 8, 8, 0)
            tik_instance.vec_sel(128, 0, ub_1[mask_idx*256 + 128], ub_mask_1[mask_idx*32],
                                    ub_2[mask_idx*256 + 128], dup_sel_0, 1, 8, 8, 0)
    else:
        tik_instance.vec_sel(VEC_MASK, 1, ub_1, ub_mask_0, ub_2, 0, 128, 8, 8)
