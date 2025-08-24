#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
drop_out_with_muls_and_softmax_grad
"""
import math
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform

SHAPE_SIZE_LIMIT = 1 << 30
UB_NUMBER_BYTE_SIZE = 15
BLOCK = 16
VEC_MASK = 128
VEC_MASK_FP32 = 64
VEC_DUMP_SHAPE = 32
MAX_REPEAT = 255
FULL_LINE = 8
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
    while dividend % LEN == 0:
        dividend //= LEN
        cnt += 1
    return cnt, dividend - 1


# 'pylint: disable=too-many-locals,too-many-statements
def data_move_with_dropout(offset, para_lis, mov_lis):
    """
    data_move_with_dropout
    """
    w_dim, line, grad_shape, tik_inst, input_keep_prob = para_lis
    grad_ub, grad_gm, softmax_ub, softmax_gm, mask_ub, mask_gm, output_ub = mov_lis
    if line == FULL_LINE:
        tik_inst.data_move(grad_ub, grad_gm[offset], 0, 1, BLOCK * line * line, 0, 0)
        tik_inst.data_move(softmax_ub, softmax_gm[offset], 0, 1, BLOCK * line * line, 0, 0)
        tik_inst.data_move(mask_ub, mask_gm[offset], 0, 1, line * line * BLOCK // LEN, 0, 0)
    else:
        tik_inst.data_move(grad_ub, grad_gm[offset], 0, grad_shape[DIM_W1],
                           BLOCK * line, (grad_shape[DIM_W1] - line) * BLOCK, 0)
        tik_inst.data_move(softmax_ub, softmax_gm[offset], 0, grad_shape[DIM_W1],
                           BLOCK * line, (grad_shape[DIM_W1] - line) * BLOCK, 0)
        tik_inst.data_move(mask_ub, mask_gm[offset], 0, grad_shape[DIM_W1], BLOCK * line // LEN, (grad_shape[DIM_W1]
                           - line) * BLOCK // LEN, 0)
    tik_inst.vconv(VEC_MASK, "", output_ub, mask_ub, line * BLOCK * w_dim // VEC_MASK, 1, 1, 8, 4)
    if input_keep_prob != 0 and input_keep_prob != 1:
        tik_inst.vec_mul(VEC_MASK, grad_ub, grad_ub, output_ub, line * BLOCK * w_dim // VEC_MASK, 8, 8, 8)
    if input_keep_prob == 0:
        tik_inst.vec_muls(VEC_MASK, output_ub, grad_ub,
                          tik_inst.Scalar(init_value=0, dtype="float16"),
                          line * BLOCK * w_dim // VEC_MASK, 8, 8)
    else:
        tik_inst.vec_muls(VEC_MASK, output_ub, grad_ub,
                          tik_inst.Scalar(init_value=1 / input_keep_prob, dtype="float16"),
                          line * BLOCK * w_dim // VEC_MASK, 8, 8)


def conv_and_mul(para_lis, output_ub_fp32, output_ub, softmax_ub_fp32, softmax_ub):
    """
    conv_and_mul
    """
    w_dim, line, _, tik_inst, _ = para_lis
    if line * BLOCK * w_dim // VEC_MASK_FP32 > MAX_REPEAT:
        tik_inst.vconv(VEC_MASK_FP32, "", output_ub_fp32, output_ub, MAX_REPEAT, 1, 1, 8, 4)
        tik_inst.vconv(VEC_MASK_FP32, "", output_ub_fp32[MAX_REPEAT * VEC_MASK_FP32],
                       output_ub[MAX_REPEAT * VEC_MASK_FP32], 1, 1, 1, 8, 4)
        tik_inst.vconv(VEC_MASK_FP32, "", softmax_ub_fp32, softmax_ub, MAX_REPEAT, 1, 1, 8, 4)
        tik_inst.vconv(VEC_MASK_FP32, "", softmax_ub_fp32[MAX_REPEAT * VEC_MASK_FP32],
                       softmax_ub[MAX_REPEAT * VEC_MASK_FP32], 1, 1, 1, 8, 4)
        tik_inst.vec_mul(VEC_MASK_FP32, output_ub_fp32, softmax_ub_fp32, output_ub_fp32, MAX_REPEAT,
                         8, 8, 8)
        tik_inst.vec_mul(VEC_MASK_FP32, output_ub_fp32[MAX_REPEAT * VEC_MASK_FP32],
                         softmax_ub_fp32[MAX_REPEAT * VEC_MASK_FP32],
                         output_ub_fp32[MAX_REPEAT * VEC_MASK_FP32], 1, 8, 8, 8)
    else:
        tik_inst.vconv(VEC_MASK_FP32, "", output_ub_fp32, output_ub, line * BLOCK * w_dim
                       // VEC_MASK_FP32, 1, 1, 8, 4)
        tik_inst.vconv(VEC_MASK_FP32, "", softmax_ub_fp32, softmax_ub, line * BLOCK * w_dim
                       // VEC_MASK_FP32, 1, 1, 8, 4)
        tik_inst.vec_mul(VEC_MASK_FP32, output_ub_fp32, softmax_ub_fp32, output_ub_fp32,
                         line * BLOCK * w_dim // VEC_MASK_FP32, 8, 8, 8)


def reduce_sum_and_broadcast(para_lis, output_ub_fp32, ub_reduce_add, ub_reduceadd_fp16, ub_broadcast, ub_dup):
    """
    reduce_sum_and_broadcast
    """
    w_dim, line, grad_shape, tik_inst, _ = para_lis
    cnt, remain = cal_level(grad_shape[DIM_W1])
    time = tik_inst.Scalar("int32", name='time', init_value=1)
    with tik_inst.for_range(0, cnt) as j:
        time.set_as(time * LEN)
        tik_inst.vadd(VEC_MASK_FP32, output_ub_fp32, output_ub_fp32,
                      output_ub_fp32[w_dim * BLOCK * line // time],
                      w_dim * BLOCK * line // time // VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
    with tik_inst.if_scope(remain > 0):
        with tik_inst.for_range(1, remain + 1) as j:
            tik_inst.vadd(VEC_MASK_FP32, output_ub_fp32[BLOCK * BLOCK * line * (remain - j)],
                          output_ub_fp32[BLOCK * BLOCK * line * (remain - j)],
                          output_ub_fp32[BLOCK * BLOCK * line * (remain - j + 1)],
                          BLOCK * BLOCK * line // VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
    tik_inst.vcadd(BLOCK, ub_reduce_add, output_ub_fp32, BLOCK * line, 1, 1, 2)
    if BLOCK * line > VEC_MASK_FP32:
        tik_inst.vconv(VEC_MASK_FP32, "", ub_reduceadd_fp16, ub_reduce_add, BLOCK * line // VEC_MASK_FP32,
                       1, 1, 4, 8)
    else:
        tik_inst.vconv(BLOCK * line, "", ub_reduceadd_fp16, ub_reduce_add, 1, 1, 1, 0, 0)
    tik_inst.vector_dup(VEC_DUMP_SHAPE, ub_dup, tik_inst.Scalar(init_value=0, dtype="int16"), 1, 1, 8)
    ub_reduceadd_int16 = ub_reduceadd_fp16.reinterpret_cast_to("int16")
    with tik_inst.for_range(0, line) as j:
        tik_inst.vor(BLOCK, ub_broadcast[BLOCK * BLOCK * j], ub_reduceadd_int16[BLOCK * j], ub_dup, BLOCK,
                     1, 1, 0, 1, 0, 0)
    with tik_inst.for_range(0, line) as j:
        tik_inst.vtranspose(ub_broadcast[BLOCK * BLOCK * j], ub_broadcast[BLOCK * BLOCK * j])


def sub_with_mul_and_move_data_out(offset, para_lis, ub_broadcast, output_ub, softmax_ub, alpha, y_gm):
    """
    sub_with_mul_and_move_data_out
    """
    w_dim, line, grad_shape, tik_inst, _ = para_lis
    ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")
    with tik_inst.for_range(0, line * BLOCK * BLOCK // VEC_MASK) as idx:
        tik_inst.vsub(VEC_MASK, output_ub[idx * VEC_MASK], output_ub[idx * VEC_MASK],
                      ub_broadcast_fp16[idx * VEC_MASK],
                      grad_shape[DIM_W1], 1, 1, 1, line * BLOCK, line * BLOCK, 0)
    tik_inst.vec_mul(VEC_MASK, output_ub, output_ub, softmax_ub,
                     line * BLOCK * w_dim // VEC_MASK, 8, 8, 8)
    tik_inst.vec_muls(VEC_MASK, output_ub, output_ub,
                      tik_inst.Scalar(init_value=alpha, dtype="float16"),
                      line * BLOCK * w_dim // VEC_MASK, 8, 8)
    if line == FULL_LINE:
        tik_inst.data_move(y_gm[offset], output_ub, 0, 1, BLOCK * line * line, 0, 0)
    else:
        tik_inst.data_move(y_gm[offset], output_ub, 0, grad_shape[DIM_W1],
                           BLOCK * line, 0, (grad_shape[DIM_W1] - line) * BLOCK)


def paras_check(y_grad, mask):
    """
    paras_check
    """
    para_check.check_dtype_rule(y_grad.get('dtype').lower(), ("float16",))
    para_check.check_dtype_rule(mask.get('dtype').lower(), ("uint8",))
    para_check.check_shape_rule(y_grad.get('shape'), max_shape_num=SHAPE_SIZE_LIMIT)
    para_check.check_shape_rule(mask.get('shape'), max_shape_num=SHAPE_SIZE_LIMIT)


def cal_prarms_list(y_grad, input_keep_prob):
    """
    cal_prarms_list
    """
    grad_shape = y_grad.get("shape")
    tik_inst = tik.Tik(tik.Dprofile(), disable_debug=False)
    aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    batch_per_core = math.ceil(grad_shape[DIM_N] * grad_shape[DIM_C] / aicore_num)
    core_num = math.ceil(grad_shape[DIM_N] * grad_shape[DIM_C] / batch_per_core)
    batch_last_core = grad_shape[DIM_N] * grad_shape[DIM_C] - (core_num - 1) * batch_per_core

    ele_per_batch = grad_shape[DIM_W1] * grad_shape[DIM_H1] * grad_shape[DIM_H2] * grad_shape[DIM_W2]
    ele_per_core = batch_per_core * ele_per_batch

    w_dim = grad_shape[DIM_W1] * grad_shape[DIM_W2]
    size_per_ub = ub_size // UB_NUMBER_BYTE_SIZE
    line = size_per_ub // BLOCK // BLOCK // grad_shape[DIM_W1]
    ranges = grad_shape[DIM_H1] // line
    shape = (grad_shape[DIM_W1], line * BLOCK, grad_shape[DIM_W2])
    params_list = [w_dim, line, grad_shape, tik_inst, input_keep_prob]
    core_attr_list = [core_num, batch_per_core, batch_last_core, ele_per_core, ele_per_batch, ranges, shape]
    return params_list, core_attr_list


# 'pylint: disable=unused-argument,too-many-arguments, disable=too-many-locals,too-many-statements
def drop_out_with_muls_and_softmax_grad(y_grad, mask, softmax_output, x_grad, input_keep_prob, alpha, axes=-1,
                                        kernel_name="drop_out_with_muls_and_softmax_grad"):
    """
    drop_out_do_mask_v3_d + softmaxgrad + muls
    """
    paras_check(y_grad, mask)
    grad_dtype = y_grad.get("dtype").lower()
    mask_dtype = mask.get("dtype").lower()
    softmax_output_dtype = softmax_output.get("dtype").lower()
    params_lis, core_attr_lis = cal_prarms_list(y_grad, input_keep_prob)
    w_dim, line, grad_shape, tik_inst, _ = params_lis
    aicore_num, batch_per_core, batch_last_core, ele_per_core, ele_per_batch, batch_range, shape = core_attr_lis
    grad_gm = tik_inst.Tensor(grad_dtype, grad_shape, name="grad_gm", scope=tbe_platform.scope_gm)
    mask_gm = tik_inst.Tensor(mask_dtype, grad_shape, name="mask_gm", scope=tbe_platform.scope_gm)
    softmax_gm = tik_inst.Tensor(softmax_output_dtype, grad_shape, name="softmax_gm", scope=tbe_platform.scope_gm)
    y_gm = tik_inst.Tensor(grad_dtype, grad_shape, name="y_gm", scope=tbe_platform.scope_gm)
    with tik_inst.for_range(0, aicore_num, block_num=aicore_num) as core_index:
        grad_ub = tik_inst.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="grad_ub")
        mask_ub = tik_inst.Tensor("uint8", shape, scope=tbe_platform.scope_ubuf, name="mask_ub")
        softmax_ub = tik_inst.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="softmax_ub")
        output_ub = tik_inst.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="output_ub")
        softmax_ub_fp32 = tik_inst.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="softmax_ub_fp32")
        output_ub_fp32 = tik_inst.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="output_ub_fp32")
        ub_reduce_add = tik_inst.Tensor("float32", (line * BLOCK,), tik.scope_ubuf, "work_tensor_ub")
        ub_reduceadd_fp16 = tik_inst.Tensor("float16", (line * BLOCK,), tik.scope_ubuf, "dst_ub")
        ub_dup = tik_inst.Tensor("int16", (VEC_DUMP_SHAPE,), scope=tbe_platform.scope_ubuf, name="ub_dup")
        ub_broadcast = tik_inst.Tensor("int16", (line * BLOCK, BLOCK), scope=tbe_platform.scope_ubuf,
                                       name="ub_broadcast")
        offset = tik_inst.Scalar("int32", name="offset")
        batch_core = tik_inst.Scalar("int32")

        with tik_inst.if_scope(core_index < aicore_num - 1):
            batch_core.set_as(batch_per_core)
        with tik_inst.else_scope():
            batch_core.set_as(batch_last_core)

        with tik_inst.for_range(0, batch_core) as index:
            with tik_inst.for_range(0, batch_range) as i:
                offset.set_as(core_index * ele_per_core + index * ele_per_batch + i * line * BLOCK * BLOCK)
                move_list = [grad_ub, grad_gm, softmax_ub, softmax_gm, mask_ub, mask_gm, output_ub]
                data_move_with_dropout(offset, params_lis, move_list)
                conv_and_mul(params_lis, output_ub_fp32, output_ub, softmax_ub_fp32, softmax_ub)
                reduce_sum_and_broadcast(params_lis, output_ub_fp32, ub_reduce_add, ub_reduceadd_fp16,
                                         ub_broadcast, ub_dup)
                sub_with_mul_and_move_data_out(offset, params_lis, ub_broadcast, output_ub, softmax_ub, alpha, y_gm)
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[grad_gm, mask_gm, softmax_gm], outputs=[y_gm,])
    return tik_inst
