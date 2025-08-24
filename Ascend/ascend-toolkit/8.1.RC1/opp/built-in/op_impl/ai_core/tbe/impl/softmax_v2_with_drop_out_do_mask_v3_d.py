#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
drop_out_do_mask_v3_d
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant.
    """
    SHAPE_SIZE_LIMIT = 1 << 30
    UB_NUMBER_BYTE_SIZE = 20
    BLOCK = 16
    VEC_MASK_FP32 = 64
    VEC_MASK_FP16 = 128
    VEC_DUMP_SHAPE = 128
    MAX_REPEAT = 255
    DIM_N = 0
    DIM_C = 1
    DIM_W1 = 2
    DIM_W2 = 5
    DIM_H1 = 3
    DIM_H2 = 4
    LEN = 2


def cal_level(dividend):
    """
    cal_level
    """
    cnt = 0
    while dividend % Constant.LEN == 0:
        dividend //= Constant.LEN
        cnt += 1
    return cnt, dividend - 1


def special_calc(input_tensor, input_mask, input_keep_prob, kernel_name):
    """
    special_calc
    """
    tensor_shape = input_tensor.get("shape")
    mask_shape = input_mask.get("shape")
    tensor_dtype = input_tensor.get("dtype").lower()
    mask_dtype = input_mask.get("dtype").lower()

    tik_inst = tik.Tik(tik.Dprofile(), disable_debug=False)
    tensor_input = tik_inst.Tensor(tensor_dtype,
                                   tensor_shape,
                                   name="tensor_input",
                                   scope=tbe_platform.scope_gm)
    mask_input = tik_inst.Tensor(mask_dtype,
                                 mask_shape,
                                 name="mask_input",
                                 scope=tbe_platform.scope_gm)
    output1 = tik_inst.Tensor(tensor_dtype,
                              tensor_shape,
                              name="output1",
                              scope=tbe_platform.scope_gm)
    output2 = tik_inst.Tensor(tensor_dtype,
                              tensor_shape,
                              name="output2",
                              scope=tbe_platform.scope_gm)
    aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    block_count = tensor_shape[0] * tensor_shape[1] / aicore_num

    with tik_inst.for_range(0, aicore_num, block_num=aicore_num) as blockid:
        with tik_inst.for_range(0, block_count) as i:
            with tik_inst.for_range(0, 16) as j:
                ub_1 = tik_inst.Tensor("float16", (32, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_1")
                ub_2 = tik_inst.Tensor("float16", (32, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_2")
                ub_cast = tik_inst.Tensor("float32", (32, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_cast")
                ub_3 = tik_inst.Tensor("float16", (32, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_3")
                ub_4 = tik_inst.Tensor("float16", (32, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_4")

                ub_reducemax = tik_inst.Tensor("float16", (32,), scope=tbe_platform.scope_ubuf, name="ub_reducemax")
                ub_reduceadd = tik_inst.Tensor("float32", (32,), scope=tbe_platform.scope_ubuf, name="ub_reduceadd")
                ub_reduceadd_fp16 = tik_inst.Tensor("float16", (32,), scope=tbe_platform.scope_ubuf,
                                                    name="ub_reduceadd_fp16")
                ub_dup = tik_inst.Tensor("uint16", (128,), scope=tbe_platform.scope_ubuf, name="ub_dup")
                ub_broadcast = tik_inst.Tensor("uint16", (32 * 16,), scope=tbe_platform.scope_ubuf,
                                               name="ub_broadcast")

                ub_mask = tik_inst.Tensor("uint8", (32, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_mask")
                ub_mask_fp16 = tik_inst.Tensor("float16", (32, 32, 16), scope=tbe_platform.scope_ubuf,
                                               name="ub_mask_fp16")

                tik_inst.data_move(ub_1[0], tensor_input[blockid * block_count * 262144 + i * 262144 + j * 512],
                                   0, 32, 32, 480, 0)
                tik_inst.data_move(ub_mask[0], mask_input[blockid * block_count * 262144 + i * 262144 + j * 512],
                                   0, 32, 16, 240, 0)
                tik_inst.vconv(128, "", ub_mask_fp16[0], ub_mask[0], 128, 1, 1, 8, 4)

                tik_inst.vmax(128, ub_2[0], ub_1[0], ub_1[8192], 64, 1, 1, 1, 8, 8, 8)
                tik_inst.vmax(128, ub_2[0], ub_2[0], ub_2[4096], 32, 1, 1, 1, 8, 8, 8)
                tik_inst.vmax(128, ub_2[0], ub_2[0], ub_2[2048], 16, 1, 1, 1, 8, 8, 8)
                tik_inst.vmax(128, ub_2[0], ub_2[0], ub_2[1024], 8, 1, 1, 1, 8, 8, 8)
                tik_inst.vmax(128, ub_2[0], ub_2[0], ub_2[512], 4, 1, 1, 1, 8, 8, 8)
                tik_inst.vcgmax(128, ub_reducemax[0], ub_2[0], 4, 1, 1, 8)

                tik_inst.vector_dup(128, ub_dup[0], tik_inst.Scalar(init_value=0, dtype="uint16"), 1, 1, 8)
                ub_reducemax_int16 = ub_reducemax.reinterpret_cast_to("uint16")
                tik_inst.vor(16, ub_broadcast[0], ub_reducemax_int16[0], ub_dup[0], 16, 1, 1, 0, 1, 0, 0)
                tik_inst.vor(16, ub_broadcast[256], ub_reducemax_int16[16], ub_dup[0], 16, 1, 1, 0, 1, 0, 0)

                tik_inst.vtranspose(ub_broadcast[0], ub_broadcast[0])
                tik_inst.vtranspose(ub_broadcast[256], ub_broadcast[256])

                ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")

                with tik_inst.for_range(0, 4) as idx:
                    tik_inst.vsub(128, ub_2[idx * 128], ub_1[idx * 128], ub_broadcast_fp16[idx * 128],
                                  32, 1, 1, 1, 32, 32, 0)

                tik_inst.vconv(64, "", ub_cast[0], ub_2[0], 255, 1, 1, 8, 4)
                tik_inst.vconv(64, "", ub_cast[16320], ub_2[16320], 1, 1, 1, 8, 4)

                tik_inst.vexp(64, ub_cast[0], ub_cast[0], 255, 1, 1, 8, 8)
                tik_inst.vexp(64, ub_cast[16320], ub_cast[16320], 1, 1, 1, 8, 8)

                tik_inst.vconv(64, "", ub_3[0], ub_cast[0], 255, 1, 1, 4, 8)
                tik_inst.vconv(64, "", ub_3[16320], ub_cast[16320], 1, 1, 1, 4, 8)

                tik_inst.vadd(64, ub_cast[0], ub_cast[0], ub_cast[8192], 128, 1, 1, 1, 8, 8, 8)
                tik_inst.vadd(64, ub_cast[0], ub_cast[0], ub_cast[4096], 64, 1, 1, 1, 8, 8, 8)
                tik_inst.vadd(64, ub_cast[0], ub_cast[0], ub_cast[2048], 32, 1, 1, 1, 8, 8, 8)
                tik_inst.vadd(64, ub_cast[0], ub_cast[0], ub_cast[1024], 16, 1, 1, 1, 8, 8, 8)
                tik_inst.vadd(64, ub_cast[0], ub_cast[0], ub_cast[512], 8, 1, 1, 1, 8, 8, 8)
                tik_inst.vcadd(16, ub_reduceadd[0], ub_cast[0], 32, 1, 1, 2)

                tik_inst.vrec(32, ub_reduceadd[0], ub_reduceadd[0], 1, 1, 1, 0, 0)

                tik_inst.vconv(32, "", ub_reduceadd_fp16[0], ub_reduceadd[0], 1, 1, 1, 0, 0)

                tik_inst.vector_dup(128, ub_dup[0], tik_inst.Scalar(init_value=0, dtype="uint16"), 1, 1, 8)

                ub_reduceadd_int16 = ub_reduceadd_fp16.reinterpret_cast_to("uint16")
                tik_inst.vor(16, ub_broadcast[0], ub_reduceadd_int16[0], ub_dup[0], 16, 1, 1, 0, 1, 0, 0)
                tik_inst.vor(16, ub_broadcast[256], ub_reduceadd_int16[16], ub_dup[0], 16, 1, 1, 0, 1, 0, 0)

                tik_inst.vtranspose(ub_broadcast[0], ub_broadcast[0])
                tik_inst.vtranspose(ub_broadcast[256], ub_broadcast[256])

                ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")

                with tik_inst.for_range(0, 4) as idx:
                    tik_inst.vmul(128, ub_4[idx * 128], ub_3[idx * 128], ub_broadcast_fp16[idx * 128],
                                  32, 1, 1, 1, 32, 32, 0)

                tik_inst.vmuls(128, ub_2[0], ub_4[0],
                               tik_inst.Scalar(init_value=1 / input_keep_prob, dtype="float16"), 128, 1, 1, 8, 8)
                tik_inst.vmul(128, ub_3[0], ub_mask_fp16[0], ub_2[0], 128, 1, 1, 1, 8, 8, 8)

                tik_inst.data_move(output1[blockid * block_count * 262144 + i * 262144 + j * 512],
                                   ub_4[0], 0, 32, 32, 0, 480)
                tik_inst.data_move(output2[blockid * block_count * 262144 + i * 262144 + j * 512],
                                   ub_3[0], 0, 32, 32, 0, 480)

    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[tensor_input, mask_input], outputs=[output1, output2])
    return tik_inst


# 'pylint: disable=too-many-locals,too-many-statements
def data_move_in(offset, params_list, move_list):
    """
    data_move_in
    """
    w_dim, line, input_shape, tik_inst, _ = params_list
    tensor_input, mask_input, ub_1, ub_mask, ub_mask_fp16 = move_list

    tik_inst.data_move(ub_1, tensor_input[offset], 0, input_shape[Constant.DIM_W1],
                       Constant.BLOCK * line, (input_shape[Constant.DIM_W1] - line) * Constant.BLOCK, 0)
    tik_inst.data_move(ub_mask, mask_input[offset], 0, input_shape[Constant.DIM_W1],
                       Constant.BLOCK * line // Constant.LEN,
                       (input_shape[Constant.DIM_W1] - line) * Constant.BLOCK // Constant.LEN, 0)
    tik_inst.vconv(Constant.VEC_MASK_FP16, "", ub_mask_fp16,
                   ub_mask, line * Constant.BLOCK * w_dim // Constant.VEC_MASK_FP16, 1, 1, 8, 4)


def reduce_max_and_sub(params_list, reduce_max_list):
    """
    reduce_max_and_sub
    """
    w_dim, line, input_shape, tik_inst, _ = params_list
    ub_1, ub_2, ub_reducemax, ub_broadcast, ub_dup = reduce_max_list
    cnt, remain = cal_level(input_shape[Constant.DIM_W1])
    time = tik_inst.Scalar("int32", name='time', init_value=1)
    with tik_inst.for_range(0, cnt) as j:
        time.set_as(time * Constant.LEN)
        with tik_inst.if_scope(j == 0):
            tik_inst.vmax(Constant.VEC_MASK_FP16, ub_2, ub_1,
                          ub_1[w_dim * Constant.BLOCK * line // time],
                          w_dim * Constant.BLOCK * line // time // Constant.VEC_MASK_FP16, 1, 1, 1, 8, 8, 8)
        with tik_inst.else_scope():
            tik_inst.vmax(Constant.VEC_MASK_FP16, ub_2, ub_2,
                          ub_2[w_dim * Constant.BLOCK * line // time],
                          w_dim * Constant.BLOCK * line // time // Constant.VEC_MASK_FP16, 1, 1, 1, 8, 8, 8)
    with tik_inst.if_scope(remain > 0):
        with tik_inst.for_range(1, remain + 1) as j:
            with tik_inst.if_scope(cnt == 0 and j == 0):
                tik_inst.vmax(Constant.VEC_MASK_FP16, ub_2[Constant.BLOCK * Constant.BLOCK * line * (remain - j)],
                              ub_1[Constant.BLOCK * Constant.BLOCK * line * (remain - j)],
                              ub_1[Constant.BLOCK * Constant.BLOCK * line * (remain - j + 1)],
                              Constant.BLOCK * Constant.BLOCK * line // Constant.VEC_MASK_FP16, 1, 1, 1, 8, 8, 8)
            with tik_inst.else_scope():
                tik_inst.vmax(Constant.VEC_MASK_FP16, ub_2[Constant.BLOCK * Constant.BLOCK * line * (remain - j)],
                              ub_2[Constant.BLOCK * Constant.BLOCK * line * (remain - j)],
                              ub_2[Constant.BLOCK * Constant.BLOCK * line * (remain - j + 1)],
                              Constant.BLOCK * Constant.BLOCK * line // Constant.VEC_MASK_FP16, 1, 1, 1, 8, 8, 8)
    tik_inst.vcgmax(Constant.VEC_MASK_FP16, ub_reducemax, ub_2, Constant.LEN * line, 1, 1, 8)

    tik_inst.vector_dup(Constant.VEC_DUMP_SHAPE, ub_dup, tik_inst.Scalar(init_value=0, dtype="uint16"), 1, 1, 8)
    ub_reducemax_int16 = ub_reducemax.reinterpret_cast_to("uint16")
    with tik_inst.for_range(0, line) as j:
        tik_inst.vor(Constant.BLOCK, ub_broadcast[Constant.BLOCK * Constant.BLOCK * j],
                     ub_reducemax_int16[Constant.BLOCK * j], ub_dup, Constant.BLOCK, 1, 1, 0, 1, 0, 0)
    with tik_inst.for_range(0, line) as j:
        tik_inst.vtranspose(ub_broadcast[Constant.BLOCK * Constant.BLOCK * j],
                            ub_broadcast[Constant.BLOCK * Constant.BLOCK * j])
    ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")

    with tik_inst.for_range(0, line * Constant.BLOCK * Constant.BLOCK // Constant.VEC_MASK_FP16) as idx:
        tik_inst.vsub(Constant.VEC_MASK_FP16, ub_2[idx * Constant.VEC_MASK_FP16], ub_1[idx * Constant.VEC_MASK_FP16],
                      ub_broadcast_fp16[idx * Constant.VEC_MASK_FP16],
                      input_shape[Constant.DIM_W1], 1, 1, 1, line * Constant.BLOCK, line * Constant.BLOCK, 0)


def conv_and_exp(params_list, exp_list):
    """
    conv_and_exp
    """
    w_dim, line, _, tik_inst, _ = params_list
    ub_2, ub_3, ub_cast = exp_list
    repeat_time = line * Constant.BLOCK * w_dim // Constant.VEC_MASK_FP32
    cnt = repeat_time // Constant.MAX_REPEAT
    remain = repeat_time % Constant.MAX_REPEAT
    if cnt > 0:
        with tik_inst.for_range(0, cnt) as i:
            tik_inst.vconv(Constant.VEC_MASK_FP32, "", ub_cast[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * i],
                           ub_2[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * i], Constant.MAX_REPEAT, 1, 1, 8, 4)
        tik_inst.vconv(Constant.VEC_MASK_FP32, "", ub_cast[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * cnt],
                       ub_2[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * cnt], remain, 1, 1, 8, 4)

        with tik_inst.for_range(0, cnt) as i:
            tik_inst.vexp(Constant.VEC_MASK_FP32, ub_cast[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * i],
                          ub_cast[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * i], Constant.MAX_REPEAT, 1, 1, 8, 8)
        tik_inst.vexp(Constant.VEC_MASK_FP32, ub_cast[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * cnt],
                      ub_cast[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * cnt], remain, 1, 1, 8, 8)       
    else:
        tik_inst.vconv(Constant.VEC_MASK_FP32, "", ub_cast, ub_2, remain, 1, 1, 8, 4)
        tik_inst.vexp(Constant.VEC_MASK_FP32, ub_cast, ub_cast, remain, 1, 1, 8, 8)


def reduce_sum_and_div(params_list, reduce_sum_and_div_list):
    """
    reduce_sum_and_div
    """
    w_dim, line, input_shape, tik_inst, _ = params_list
    ub_cast, ub_reduceadd, ub_dup_fp32, ub_1, ub_3, ub_4 = reduce_sum_and_div_list
    cnt, remain = cal_level(input_shape[Constant.DIM_W1])

    time = tik_inst.Scalar("int32", name='time', init_value=1)
    # reduce_add
    with tik_inst.for_range(0, cnt) as j:
        time.set_as(time * Constant.LEN)
        with tik_inst.if_scope(j == 0):
            tik_inst.vadd(Constant.VEC_MASK_FP32, ub_3, ub_cast,
                          ub_cast[w_dim * Constant.BLOCK * line // time],
                          w_dim * Constant.BLOCK * line // time // Constant.VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
        with tik_inst.else_scope():
            tik_inst.vadd(Constant.VEC_MASK_FP32, ub_3, ub_3,
                          ub_3[w_dim * Constant.BLOCK * line // time],
                          w_dim * Constant.BLOCK * line // time // Constant.VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
    with tik_inst.if_scope(remain > 0):
        with tik_inst.for_range(1, remain + 1) as j:
            with tik_inst.if_scope(cnt == 0 and j == 0):
                tik_inst.vadd(Constant.VEC_MASK_FP32, ub_3[Constant.BLOCK * Constant.BLOCK * line * (remain - j)],
                              ub_cast[Constant.BLOCK * Constant.BLOCK * line * (remain - j)],
                              ub_cast[Constant.BLOCK * Constant.BLOCK * line * (remain - j + 1)],
                              Constant.BLOCK * Constant.BLOCK * line // Constant.VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
            with tik_inst.else_scope():
                tik_inst.vadd(Constant.VEC_MASK_FP32, ub_3[Constant.BLOCK * Constant.BLOCK * line * (remain - j)],
                              ub_3[Constant.BLOCK * Constant.BLOCK * line * (remain - j)],
                              ub_3[Constant.BLOCK * Constant.BLOCK * line * (remain - j + 1)],
                              Constant.BLOCK * Constant.BLOCK * line // Constant.VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
    tik_inst.vcadd(Constant.BLOCK, ub_reduceadd, ub_3, Constant.BLOCK * line, 1, 1, 2)

    # vrec
    tik_inst.vrec(line * Constant.BLOCK, ub_reduceadd, ub_reduceadd, 1, 1, 1, 0, 0)

    with tik_inst.for_range(0, line * input_shape[Constant.DIM_H2] // 8) as j:
        tik_inst.vector_dup(16, ub_dup_fp32[j * 128],
                            tik_inst.Scalar(init_value=ub_reduceadd[j * 8], dtype="float32"), 1, 1, 8)
        tik_inst.vector_dup(16, ub_dup_fp32[j * 128 + 16],
                            tik_inst.Scalar(init_value=ub_reduceadd[j * 8 + 1], dtype="float32"), 1, 1, 8)
        tik_inst.vector_dup(16, ub_dup_fp32[j * 128 + 32],
                            tik_inst.Scalar(init_value=ub_reduceadd[j * 8 + 2], dtype="float32"), 1, 1, 8)
        tik_inst.vector_dup(16, ub_dup_fp32[j * 128 + 48],
                            tik_inst.Scalar(init_value=ub_reduceadd[j * 8 + 3], dtype="float32"), 1, 1, 8)
        tik_inst.vector_dup(16, ub_dup_fp32[j * 128 + 64],
                            tik_inst.Scalar(init_value=ub_reduceadd[j * 8 + 4], dtype="float32"), 1, 1, 8)
        tik_inst.vector_dup(16, ub_dup_fp32[j * 128 + 80],
                            tik_inst.Scalar(init_value=ub_reduceadd[j * 8 + 5], dtype="float32"), 1, 1, 8)
        tik_inst.vector_dup(16, ub_dup_fp32[j * 128 + 96],
                            tik_inst.Scalar(init_value=ub_reduceadd[j * 8 + 6], dtype="float32"), 1, 1, 8)
        tik_inst.vector_dup(16, ub_dup_fp32[j * 128 + 112],
                            tik_inst.Scalar(init_value=ub_reduceadd[j * 8 + 7], dtype="float32"), 1, 1, 8)

    with tik_inst.for_range(0, line * Constant.BLOCK * Constant.BLOCK // Constant.VEC_MASK_FP32) as idx:
        tik_inst.vmul(Constant.VEC_MASK_FP32, ub_4[idx * Constant.VEC_MASK_FP32], ub_cast[idx * Constant.VEC_MASK_FP32],
                      ub_dup_fp32[idx * Constant.VEC_MASK_FP32],
                      input_shape[Constant.DIM_W1], 1, 1, 1, line * Constant.BLOCK * Constant.LEN,
                      line * Constant.BLOCK * Constant.LEN, 0)

    tik_inst.vconv(Constant.VEC_MASK_FP32, "", ub_1, ub_4,
                   w_dim * line * Constant.BLOCK // Constant.VEC_MASK_FP32, 1, 1, 4, 8)


def mul_with_dropout(params_list, ub_2, ub_3, ub_4, ub_mask_fp16):
    """
    mul_with_dropout
    """
    w_dim, line, _, tik_inst, input_keep_prob = params_list

    # vmuls and vmul
    tik_inst.vmuls(Constant.VEC_MASK_FP32, ub_3, ub_4,
                   tik_inst.Scalar(init_value=1 / input_keep_prob, dtype="float32"),
                   line * Constant.BLOCK * w_dim // Constant.VEC_MASK_FP32, 1, 1, 8, 8)
    tik_inst.vconv(Constant.VEC_MASK_FP32, "", ub_2, ub_3,
                   w_dim * line * Constant.BLOCK // Constant.VEC_MASK_FP32, 1, 1, 4, 8)
    tik_inst.vmul(Constant.VEC_MASK_FP16, ub_2, ub_mask_fp16, ub_2,
                   w_dim * line * Constant.BLOCK // Constant.VEC_MASK_FP16, 1, 1, 1, 8, 8, 8)


def move_data_out(offset, params_list, ub_1, ub_2, output1, output2):
    """
    move_data_out
    """
    _, line, input_shape, tik_inst, _ = params_list

    tik_inst.data_move(output1[offset], ub_1, 0, input_shape[Constant.DIM_W1],
                       Constant.BLOCK * line, 0, (input_shape[Constant.DIM_H1] - line) * Constant.BLOCK)
    tik_inst.data_move(output2[offset], ub_2, 0, input_shape[Constant.DIM_W1],
                       Constant.BLOCK * line, 0, (input_shape[Constant.DIM_H1] - line) * Constant.BLOCK)


def paras_check(input_tensor, input_mask):
    """
    paras_check
    """
    para_check.check_dtype_rule(input_tensor.get('dtype').lower(), ("float16"))
    para_check.check_dtype_rule(input_mask.get('dtype').lower(), ("uint8"))
    para_check.check_shape_rule(input_tensor.get('shape'), max_shape_num=Constant.SHAPE_SIZE_LIMIT)
    para_check.check_shape_rule(input_mask.get('shape'), max_shape_num=Constant.SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(input_tensor.get('shape'), Constant.SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(input_mask.get('shape'), Constant.SHAPE_SIZE_LIMIT)


def cal_prarms_list(input_tensor, input_keep_prob):
    """
    cal_prarms_list
    """
    input_shape = input_tensor.get("shape")
    tik_inst = tik.Tik(tik.Dprofile(), disable_debug=False)
    aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    batch_per_core = input_shape[Constant.DIM_N] * input_shape[Constant.DIM_C] // aicore_num
    ele_per_core = batch_per_core * input_shape[Constant.DIM_W1] * \
                   input_shape[Constant.DIM_H1] * input_shape[Constant.DIM_H2] * input_shape[Constant.DIM_W2]
    ele_per_batch = input_shape[Constant.DIM_W1] * input_shape[Constant.DIM_H1] * \
                    input_shape[Constant.DIM_H2] * input_shape[Constant.DIM_W2]
    w_dim = input_shape[Constant.DIM_W1] * input_shape[Constant.DIM_W2]
    size_per_ub = ub_size // Constant.UB_NUMBER_BYTE_SIZE
    line = size_per_ub // Constant.BLOCK // Constant.BLOCK // input_shape[Constant.DIM_W1]
    if line > Constant.LEN:
        line = Constant.LEN
    ranges = input_shape[Constant.DIM_H1] // line
    shape = (input_shape[Constant.DIM_W1], line * Constant.BLOCK, input_shape[Constant.DIM_W2])
    params_list = [w_dim, line, input_shape, tik_inst, input_keep_prob]
    core_attr_list = [aicore_num, batch_per_core, ele_per_core, ele_per_batch, ranges, shape]
    return params_list, core_attr_list


# 'pylint: disable=unused-argument,too-many-arguments
# 'pylint: disable=too-many-locals,too-many-statements
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME)
def softmax_v2_with_drop_out_do_mask_v3_d(input_tensor, input_mask, output_1, output_2, input_keep_prob, axis=-1,
                                          kernel_name="softmax_v2_with_drop_out_do_mask_v3_d"):
    """
    softmax_v2 + drop_out_do_mask_v3_d

    Parameters
    ----------
    input_tensor : dict,shape and dtype of input_tensor,only support float16
    input_mask : dict,shape and dtype of input_mask
        shape of mask,1D, dtype == uint8
    input_keep_prob : dict,shape and dtype of input_keep_prob
        shape of keep_prob, only 1 parament and equals to (1)
        prob scale (0.0,1.0] NOTICE: type same as dytpe
    output1 : dict,shape and dtype of output1
    output2 : dict,shape and dtype of output2
    kernel_name : str
        cce kernel name, default value is "softmax_v2_with_drop_out_do_mask_v3_d"

    Returns
    -------
    None
    """
    paras_check(input_tensor, input_mask)

    tensor_shape = input_tensor.get("ori_shape")
    if tensor_shape[-1] == 512 and tensor_shape[-2] == 512:
        return special_calc(input_tensor, input_mask, input_keep_prob, kernel_name)

    tensor_shape = input_tensor.get("shape")
    mask_shape = input_mask.get("shape")
    tensor_dtype = input_tensor.get("dtype").lower()
    mask_dtype = input_mask.get("dtype").lower()

    params_list, core_attr_list = cal_prarms_list(input_tensor, input_keep_prob)
    w_dim, line, input_shape, tik_inst, _ = params_list
    aicore_num, batch_per_core, ele_per_core, ele_per_batch, batch_range, shape = core_attr_list

    tensor_input = tik_inst.Tensor(tensor_dtype,
                                   tensor_shape,
                                   name="tensor_input",
                                   scope=tbe_platform.scope_gm)
    mask_input = tik_inst.Tensor(mask_dtype,
                                 mask_shape,
                                 name="mask_input",
                                 scope=tbe_platform.scope_gm)
    output1 = tik_inst.Tensor(tensor_dtype,
                              tensor_shape,
                              name="output1",
                              scope=tbe_platform.scope_gm)
    output2 = tik_inst.Tensor(tensor_dtype,
                              tensor_shape,
                              name="output2",
                              scope=tbe_platform.scope_gm)

    with tik_inst.for_range(0, aicore_num, block_num=aicore_num) as core_index:
        ub_1 = tik_inst.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="ub_1")
        ub_2 = tik_inst.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="ub_2")
        ub_cast = tik_inst.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="ub_cast")
        ub_3 = tik_inst.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="ub_3")
        ub_4 = tik_inst.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="ub_4")

        ub_reducemax = tik_inst.Tensor("float16", (line * Constant.BLOCK,),
                                       scope=tbe_platform.scope_ubuf, name="ub_reducemax")
        ub_reduceadd = tik_inst.Tensor("float32", (line * Constant.BLOCK,),
                                       scope=tbe_platform.scope_ubuf, name="ub_reduceadd")

        ub_dup = tik_inst.Tensor("uint16", (Constant.VEC_DUMP_SHAPE,), scope=tbe_platform.scope_ubuf, name="ub_dup")
        ub_dup_fp32 = tik_inst.Tensor("float32", (line * Constant.BLOCK * Constant.BLOCK,),
                                      scope=tbe_platform.scope_ubuf, name="ub_dup_fp32")
        ub_broadcast = tik_inst.Tensor("uint16", (line * Constant.BLOCK * Constant.BLOCK,),
                                       scope=tbe_platform.scope_ubuf, name="ub_broadcast")

        ub_mask = tik_inst.Tensor("uint8", shape, scope=tbe_platform.scope_ubuf, name="ub_mask")
        ub_mask_fp16 = tik_inst.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="ub_mask_fp16")
        offset = tik_inst.Scalar("int64", name="offset")

        with tik_inst.for_range(0, batch_per_core) as index:
            with tik_inst.for_range(0, batch_range) as i:
                offset.set_as(core_index * ele_per_core + index * ele_per_batch + \
                              i * line * Constant.BLOCK * Constant.BLOCK)
                move_list = [tensor_input, mask_input, ub_1, ub_mask, ub_mask_fp16]
                data_move_in(offset, params_list, move_list)
                reduce_max_list = [ub_1, ub_2, ub_reducemax, ub_broadcast, ub_dup]
                reduce_max_and_sub(params_list, reduce_max_list)
                exp_list = [ub_2, ub_3, ub_cast]
                conv_and_exp(params_list, exp_list)
                reduce_sum_and_div_list = [ub_cast, ub_reduceadd, ub_dup_fp32, ub_1, ub_3, ub_4]
                reduce_sum_and_div(params_list, reduce_sum_and_div_list)
                mul_with_dropout(params_list, ub_2, ub_3, ub_4, ub_mask_fp16)
                move_data_out(offset, params_list, ub_1, ub_2, output1, output2)
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[tensor_input, mask_input], outputs=[output1, output2])
    return tik_inst