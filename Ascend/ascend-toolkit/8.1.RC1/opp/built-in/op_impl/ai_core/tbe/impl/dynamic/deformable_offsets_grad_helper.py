"""
Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

deformable_offsets_grad_helper
"""

import math
from impl import common_util
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    # reserved ub size
    RESERVED_UB_SIZE = 4 * 1024
    # vector_repeat
    MAX_REPEAT = 255
    # vector fp32 size
    VECTOR_FP32_SIZE = 64
    # vector fp32 mask size
    MASK64_VALUE = 64
    # block fp32 size
    BLOCK_FP32_SIZE = 8
    # block fp16 size
    BLOCK_FP16_SIZE = 16
    # maximum dma_copy stride
    MAX_STRIDE = 65535
    # uint16 bit size
    UINT16_BIT_SIZE = 8
    # 3 for offsets' w h scale
    W_H_SCALE = 3
    # 220T only has 30 cores
    CORE_NUM_220T = 30
    # byte size
    FP32_BYTESIZE = 4
    BF16_BYTESIZE = 2


def _get_shape_size(shape):
    """
    get the total size of shape
    """
    total = 1
    for i in shape:
        total = total * i
    return total


def _get_index_num(shape, start, end, step, threshold):
    """
    get the index num
    """
    val = 1
    index = end - step
    for i in range(start, end, step):
        val = val * shape[i]
        if val >= threshold:
            index = i
            break
    return index


def _is_immediate(val):
    """
    check is immediate number
    """
    if isinstance(val, (tik.tik_lib.tik_expr.Expr, tik.api.tik_scalar.Scalar)):
        return False
    return True


# 'pylint: disable=too-many-locals
def _get_ub_need_size(grad_size, offsets_size, group_c, input_dtype):
    """
    get the size of the UB corresponding to the shape
    """
    zero_ub_size, h_max_ub_size, w_max_ub_size = Constant.VECTOR_FP32_SIZE, \
                                                 Constant.VECTOR_FP32_SIZE, Constant.VECTOR_FP32_SIZE
    mask_ub_size = 9 * Constant.UINT16_BIT_SIZE // 2
    x_ub_size = 4 * group_c
    fp32_floor, fp32_ceil = 2 * grad_size + Constant.VECTOR_FP32_SIZE, 2 * grad_size + Constant.VECTOR_FP32_SIZE
    int32_floor, int32_ceil = 2 * grad_size, 2 * grad_size
    ceil_sub, sub_floor = max(2 * grad_size, offsets_size), max(2 * grad_size, offsets_size)
    grad_ub_size, grad_scale_ub = grad_size, grad_size
    grad_scale_weight_size = 4 * grad_size
    check_offset = 2 * grad_size
    fp16_size = 0
    if input_dtype in ("float16", "bfloat16"):
        fp16_size = max(2 * grad_size, offsets_size) // 2 + (2 * grad_size + Constant.VECTOR_FP32_SIZE) // 2 + group_c
    all_size = zero_ub_size + h_max_ub_size + w_max_ub_size + mask_ub_size + x_ub_size + fp32_floor + fp32_ceil + \
               int32_floor + int32_ceil + ceil_sub + sub_floor + grad_ub_size + \
               grad_scale_ub + grad_scale_weight_size + check_offset + fp16_size
    return all_size


# 'pylint: disable=too-many-public-methods
class DeformableOffsetsGradHelper:
    """
    use to store DeformableOffset base parameters
    """

    def __init__(self, input_params):
        """
        init shape and format information
        """
        self.input_params = input_params
        self.instance = tik.Tik(tik.Dprofile())
        self.dtype = input_params.get("dtype")
        self.dsize = common_util.get_data_size("float32")
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

        self.grad_shape = input_params.get("grad_shape")
        self.x_shape = input_params.get("x_shape")
        self.offsets_shape = input_params.get("offsets_shape")
        self.grad_x_shape = input_params.get("grad_x_shape")
        self.grad_offsets_shape = input_params.get("grad_offsets_shape")
        self.helper_shape = self.offsets_shape[:]
        self.helper_shape[0] = 1

        self.k_h = input_params.get("k_h")
        self.k_w = input_params.get("k_w")
        self.stride_h = input_params.get("stride_h")
        self.stride_w = input_params.get("stride_w")
        self.dilation_h = input_params.get("dilation_h")
        self.dilation_w = input_params.get("dilation_w")
        self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = input_params.get("pads")

        self.batch = input_params.get("batch")
        self.h_in = input_params.get("h_in")
        self.w_in = input_params.get("w_in")
        self.groups = input_params.get("deformable_groups")
        self.group_c = input_params.get("group_c")
        self.h_out = input_params.get("h_out")
        self.w_out = input_params.get("w_out")
        self.kernel_name = input_params.get("kernel_name")
        self.support_move_pad = tbe_platform.api_check_support("tik.data_move_pad", self.dtype)
        self.tiling_params = {}

        self.grad_gm = self.instance.Tensor(self.dtype, (_get_shape_size(self.grad_shape),), name="grad_gm",
                                            scope=tik.scope_gm)
        self.x_gm = self.instance.Tensor(self.dtype, (_get_shape_size(self.x_shape),), name="x_gm", scope=tik.scope_gm)
        self.offsets_gm = self.instance.Tensor(self.dtype, (_get_shape_size(self.offsets_shape),), name="offsets_gm",
                                               scope=tik.scope_gm)
        
        self.grad_x_gm = self.instance.Tensor("float32", (_get_shape_size(self.grad_x_shape),), name="grad_x_gm",
                                              scope=tik.scope_gm, is_atomic_add=True)

        self.grad_offsets_gm = self.instance.Tensor("float32", (_get_shape_size(self.grad_offsets_shape),),
                                                    name="grad_offsets_gm", scope=tik.scope_gm)

        helper_shape_other = _get_shape_size(self.helper_shape[3:])
        assist_data = [[[0] * helper_shape_other for _ in range(self.helper_shape[2])]
                       for _ in range(self.helper_shape[1])]
        for h in range(self.h_out):
            for w in range(self.w_out):
                for kh in range(self.k_h):
                    for kw in range(self.k_w):
                        for g in range(self.groups):
                            assist_data[h][w][
                            0 * self.groups * self.k_h * self.k_w + g * self.k_h * self.k_w + kh * self.k_w + kw] = \
                            w * self.stride_w - self.pad_left + kw * self.dilation_w

                            assist_data[h][w][
                            1 * self.groups * self.k_h * self.k_w + g * self.k_h * self.k_w + kh * self.k_w + kw] = \
                            h * self.stride_h - self.pad_top + kh * self.dilation_h

        self.helper_gm = self.instance.Tensor("float32", (_get_shape_size(self.helper_shape),), name="helper_gm",
                                              scope=tik.scope_gm, init_value=assist_data)

        self.repeat_max = self.instance.Scalar("uint32", name="repeat_max")
        self.repeat_max.set_as(255)

        self.offset_w_val = self.instance.Scalar("float32", name="offset_w_val")
        self.offset_h_val = self.instance.Scalar("float32", name="offset_h_val")
        self.offset_scale_val = self.instance.Scalar("float32", name="offset_scale_val")

    # 'pylint: disable=too-many-locals
    def init_ub_tensor(self):
        """
        init ub tensors
        """
        grad_size = self.tiling_params.get("grad_size")
        offset_size = self.tiling_params.get("offset_size")

        zero_ub = self.instance.Tensor("float32", (Constant.VECTOR_FP32_SIZE,),
                                       name="zero_ub", scope=tbe_platform.scope_ubuf)
        neg_one_ub = self.instance.Tensor("float32", (Constant.VECTOR_FP32_SIZE,),
                                          name="neg_one_ub", scope=tbe_platform.scope_ubuf)
        h_max_ub = self.instance.Tensor("float32", (Constant.VECTOR_FP32_SIZE,),
                                        name="h_max_ub", scope=tbe_platform.scope_ubuf)
        w_max_ub = self.instance.Tensor("float32", (Constant.VECTOR_FP32_SIZE,),
                                        name="w_max_ub", scope=tbe_platform.scope_ubuf)
        h_max_ub2 = self.instance.Tensor("float32", (Constant.VECTOR_FP32_SIZE,),
                                         name="h_max_ub2", scope=tbe_platform.scope_ubuf)
        w_max_ub2 = self.instance.Tensor("float32", (Constant.VECTOR_FP32_SIZE,),
                                         name="w_max_ub2", scope=tbe_platform.scope_ubuf)

        mask_ub1 = self.instance.Tensor("uint16", (Constant.UINT16_BIT_SIZE,),
                                        name="mask_ub1", scope=tbe_platform.scope_ubuf)
        mask_ub2 = self.instance.Tensor("uint16", (Constant.UINT16_BIT_SIZE,),
                                        name="mask_ub2", scope=tbe_platform.scope_ubuf)
        mask_ub3 = self.instance.Tensor("uint16", (Constant.UINT16_BIT_SIZE,),
                                        name="mask_ub3", scope=tbe_platform.scope_ubuf)
        mask_ub4 = self.instance.Tensor("uint16", (Constant.UINT16_BIT_SIZE,),
                                        name="mask_ub4", scope=tbe_platform.scope_ubuf)
        mask_ub5 = self.instance.Tensor("uint16", (Constant.UINT16_BIT_SIZE,),
                                        name="mask_ub5", scope=tbe_platform.scope_ubuf)
        mask_ub6 = self.instance.Tensor("uint16", (Constant.UINT16_BIT_SIZE,),
                                        name="mask_ub6", scope=tbe_platform.scope_ubuf)
        mask_ub7 = self.instance.Tensor("uint16", (Constant.UINT16_BIT_SIZE,),
                                        name="mask_ub7", scope=tbe_platform.scope_ubuf)
        mask_ub8 = self.instance.Tensor("uint16", (Constant.UINT16_BIT_SIZE,),
                                        name="mask_ub8", scope=tbe_platform.scope_ubuf)
        mask_ub9 = self.instance.Tensor("uint16", (Constant.UINT16_BIT_SIZE,),
                                        name="mask_ub9", scope=tbe_platform.scope_ubuf)

        x_ub_lt = self.instance.Tensor("float32", (self.group_c,), name="x_ub_lt", scope=tbe_platform.scope_ubuf)
        x_ub_lb = self.instance.Tensor("float32", (self.group_c,), name="x_ub_lb", scope=tbe_platform.scope_ubuf)
        x_ub_rt = self.instance.Tensor("float32", (self.group_c,), name="x_ub_rt", scope=tbe_platform.scope_ubuf)
        x_ub_rb = self.instance.Tensor("float32", (self.group_c,), name="x_ub_rb", scope=tbe_platform.scope_ubuf)

        check_offset_ub = self.instance.Tensor("float32", (grad_size * 2,),
                                               name="check_offset_ub", scope=tbe_platform.scope_ubuf)

        grad_ub = self.instance.Tensor("float32", (grad_size,), name="grad_ub", scope=tbe_platform.scope_ubuf)
        grad_scale_ub = self.instance.Tensor("float32", (grad_size,), name="grad_scale_ub",
                                             scope=tbe_platform.scope_ubuf)
        grad_scale_weight_ub = self.instance.Tensor("float32", (grad_size * 4,), name="grad_scale_weight_ub",
                                                    scope=tbe_platform.scope_ubuf)
        size = max(grad_size * 2, offset_size)
        ceil_sub = self.instance.Tensor("float32", (size,), name="ceil_sub", scope=tbe_platform.scope_ubuf)
        sub_floor = self.instance.Tensor("float32", (size,), name="sub_floor", scope=tbe_platform.scope_ubuf)
        fp32_ceil_index = self.instance.Tensor("float32", (grad_size * 2 + Constant.VECTOR_FP32_SIZE,),
                                               name="fp32_ceil_index",
                                               scope=tbe_platform.scope_ubuf)
        fp32_floor_index = self.instance.Tensor("float32", (grad_size * 2 + Constant.VECTOR_FP32_SIZE,),
                                                name="fp32_floor_index",
                                                scope=tbe_platform.scope_ubuf)
        int32_ceil_index = self.instance.Tensor("int32", (grad_size * 2,), name="int32_ceil_index",
                                                scope=tbe_platform.scope_ubuf)
        int32_floor_index = self.instance.Tensor("int32", (grad_size * 2,), name="int32_floor_index",
                                                 scope=tbe_platform.scope_ubuf)
        ceil_sub_fp16 = None
        fp32_ceil_index_fp16 = None
        x_ub_lt_fp16 = None
        x_ub_lb_fp16 = None
        x_ub_rt_fp16 = None
        x_ub_rb_fp16 = None
        if self.dtype in ("float16", "bfloat16"):
            ceil_sub_fp16 = self.instance.Tensor(self.dtype, (size,), name="ceil_sub_fp16",
                                                 scope=tbe_platform.scope_ubuf)

            fp32_ceil_index_fp16 = self.instance.Tensor(self.dtype, (grad_size * 2 + Constant.VECTOR_FP32_SIZE,),
                                                name="fp32_ceil_index_fp16", scope=tbe_platform.scope_ubuf)

            x_ub_lt_fp16 = self.instance.Tensor(self.dtype, (self.group_c,), name="x_ub_lt_fp16",
                                                scope=tbe_platform.scope_ubuf)
            x_ub_lb_fp16 = self.instance.Tensor(self.dtype, (self.group_c,), name="x_ub_lb_fp16",
                                                scope=tbe_platform.scope_ubuf)
            x_ub_rt_fp16 = self.instance.Tensor(self.dtype, (self.group_c,), name="x_ub_rt_fp16",
                                                scope=tbe_platform.scope_ubuf)
            x_ub_rb_fp16 = self.instance.Tensor(self.dtype, (self.group_c,), name="x_ub_rb_fp16",
                                                scope=tbe_platform.scope_ubuf)

        buf_list = [ceil_sub, sub_floor, grad_ub, grad_scale_ub, grad_scale_weight_ub,
                    fp32_ceil_index, int32_ceil_index, fp32_floor_index, int32_floor_index,
                    zero_ub, neg_one_ub, h_max_ub, w_max_ub, h_max_ub2, w_max_ub2,
                    mask_ub1, mask_ub2, mask_ub3, mask_ub4,
                    mask_ub5, mask_ub6, mask_ub7, mask_ub8, mask_ub9,
                    x_ub_lt, x_ub_lb, x_ub_rt, x_ub_rb, check_offset_ub,
                    ceil_sub_fp16, fp32_ceil_index_fp16,
                    x_ub_lt_fp16, x_ub_lb_fp16, x_ub_rt_fp16, x_ub_rb_fp16]
        return buf_list

    # 'pylint: disable=too-many-locals
    def tiling(self, ub_size):
        """
        get tiling information
        """
        tiling_shape = [self.batch, self.h_out, self.w_out, self.k_h, self.k_w, self.groups, self.group_c]
        self.tiling_params["tiling_shape"] = tiling_shape

        # [n, h_out] cut core, block tiling index in [0, 1]
        block_index = _get_index_num(tiling_shape, 0, 2, 1, self.core_num)
        block_len = _get_shape_size(tiling_shape[:block_index + 1])
        block_num = self.core_num
        if block_len < block_num:
            block_num = block_len
        self.tiling_params["block_num"] = block_num
        self.tiling_params["block_index"] = block_index
        self.tiling_params["block_cycle"] = block_len // block_num
        self.tiling_params["block_tail"] = block_len % block_num
        self.tiling_params["block_grad_element"] = _get_shape_size(tiling_shape[block_index + 1:])
        self.tiling_params["block_offset_element"] = _get_shape_size(self.offsets_shape[block_index + 1:])

        flag = False

        # grad offset min size is 3 * k_h * k_w * groups, grad min size  is k_h * k_w * groups * group_c
        # so, ub tiling index in [0, 1, 2]
        mod = 1 if block_len % block_num > 0 else 0
        ub_tiling_shape = [block_len // block_num + mod]
        ub_tiling_shape.extend(tiling_shape[block_index + 1:3])

        min_offset_size = Constant.W_H_SCALE * self.groups * self.k_h * self.k_w
        for index, elem in enumerate(ub_tiling_shape):
            if flag:
                break
            for t_factor in range(elem, 0, -1):
                tmp_shape = [1] * (block_index + index)
                tmp_shape.extend([t_factor])
                tmp_shape.extend(tiling_shape[block_index + index + 1:])
                grad_size = _get_shape_size(tmp_shape)
                offset_size = tmp_shape[0] * tmp_shape[1] * tmp_shape[2] * min_offset_size
                offset_size_align = (offset_size + Constant.BLOCK_FP32_SIZE - 1) // \
                                    Constant.BLOCK_FP32_SIZE * Constant.BLOCK_FP32_SIZE
                size = _get_ub_need_size(grad_size, offset_size_align, self.group_c, self.dtype)
                if size <= ub_size and offset_size >= Constant.BLOCK_FP32_SIZE:
                    flag = True
                    self.tiling_params["ub_index"] = block_index + index
                    self.tiling_params["ub_factor"] = t_factor
                    self.tiling_params["grad_size"] = grad_size
                    self.tiling_params["offset_size"] = offset_size_align
                    self.tiling_params["ub_num"] = _get_shape_size(tmp_shape)
                    if self.tiling_params.get("ub_index") <= 1:
                        if self.tiling_params.get("ub_index") == 1 and block_index == 0:
                            h_grad_size = self.h_out * self.w_out * self.k_h * \
                                            self.k_w * self.groups * self.group_c
                            num = self.tiling_params.get("block_cycle") * \
                                self.tiling_params.get("block_grad_element") // h_grad_size
                            self.tiling_params["thread_num"] = 2 if num >= 2 else 1
                        else:
                            size = self.tiling_params.get("block_cycle") * \
                                    self.tiling_params.get("block_grad_element")
                            db_num = size // self.tiling_params.get("ub_num")
                            self.tiling_params["thread_num"] = 2 if db_num >= 2 else 1
                    else:
                        w_grad_size = self.w_out * self.k_h * self.k_w * self.groups * self.group_c
                        num = self.tiling_params.get("block_cycle") * \
                            self.tiling_params.get("block_grad_element") // w_grad_size
                        self.tiling_params["thread_num"] = 2 if num >= 2 else 1
                    break

        return flag

    # 'pylint: disable=too-many-locals,too-many-statements
    def offsets_grad(self):
        """
        offsets grad compute func
        """
        ub_size = (self.ub_size - Constant.RESERVED_UB_SIZE) // self.dsize
        # try open double buffer
        flag_db = self.tiling(ub_size // 2)
        # try close double buffer
        if not flag_db or self.tiling_params.get("thread_num") == 1:
            flag_no_db = self.tiling(ub_size)
            self.tiling_params["thread_num"] = 1
            if not flag_no_db:
                error_manager_vector.raise_err_specific_reson(self.kernel_name,
                                                              "can not find tiling, grad or offsets shape is too big")
        block_num = self.tiling_params.get("block_num")
        block_index = self.tiling_params.get("block_index")
        block_cycle = self.tiling_params.get("block_cycle")
        block_grad_element = self.tiling_params.get("block_grad_element")
        block_offset_element = self.tiling_params.get("block_offset_element")
        block_tail = self.tiling_params.get("block_tail")
        ub_num = self.tiling_params.get("ub_num")
        thread_num = self.tiling_params.get("thread_num")
        ub_index = self.tiling_params.get("ub_index")
        ub_factor = self.tiling_params.get("ub_factor")

        with self.instance.for_range(0, block_num, block_num=block_num) as block_id:
            each_cycle = self.instance.Scalar("uint32", name="each_cycle")
            each_cycle.set_as(block_cycle)
            if block_tail > 0:
                with self.instance.if_scope(block_id < block_tail):
                    each_cycle.set_as(block_cycle + 1)
            block_offset = self.instance.Scalar("uint32", name="block_offset")
            block_offset.set_as(block_id * each_cycle)
            if block_tail > 0:
                with self.instance.if_scope(block_id >= block_tail):
                    block_offset.set_as(block_id * block_cycle + block_tail)

            if ub_index <= 1:
                grad_size = self.w_out * self.k_h * self.k_w * self.groups * self.group_c
                offset_size = self.w_out * Constant.W_H_SCALE * self.groups * self.k_h * self.k_w
                if ub_index == 1 and block_index == 0:
                    h_grad_size = self.h_out * grad_size
                    h_offset_size = self.h_out * offset_size
                    ub_loop = self.h_out // ub_factor
                    ub_tail = self.h_out % ub_factor
                    loop = each_cycle * block_grad_element // h_grad_size
                    with self.instance.for_range(0, loop, thread_num=thread_num) as loop_id:
                        with self.instance.for_range(0, ub_loop) as u_id:
                            grad_index = block_offset * block_grad_element + \
                                         loop_id * h_grad_size + u_id * ub_factor * grad_size
                            offset_index = block_offset * block_offset_element + \
                                           loop_id * h_offset_size + u_id * ub_factor * offset_size
                            self.compute(grad_index,
                                         offset_index,
                                         ub_factor * grad_size,
                                         ub_factor * offset_size)
                        if ub_tail > 0:
                            grad_index = block_offset * block_grad_element + \
                                         loop_id * h_grad_size + ub_loop * ub_factor * grad_size
                            offset_index = block_offset * block_offset_element + \
                                           loop_id * h_offset_size + ub_loop * ub_factor * offset_size
                            self.compute(grad_index,
                                         offset_index,
                                         ub_tail * grad_size,
                                         ub_tail * offset_size)
                else:
                    ub_loop = each_cycle * block_grad_element // ub_num
                    ub_tail = each_cycle * block_grad_element % ub_num
                    loop_num = ub_num // grad_size
                    tail_num = ub_tail // grad_size
                    with self.instance.for_range(0, ub_loop, thread_num=thread_num) as loop_id:
                        self.compute(block_offset * block_grad_element + loop_id * ub_num,
                                     block_offset * block_offset_element + loop_id * loop_num * offset_size,
                                     ub_num,
                                     loop_num * offset_size)
                    with self.instance.if_scope(ub_tail > 0):
                        self.compute(block_offset * block_grad_element + ub_loop * ub_num,
                                     block_offset * block_offset_element + ub_loop * loop_num * offset_size,
                                     ub_tail,
                                     tail_num * offset_size)
            elif ub_index == 2:
                grad_size = self.k_h * self.k_w * self.groups * self.group_c
                offset_size = Constant.W_H_SCALE * self.groups * self.k_h * self.k_w
                grad_offset = self.k_w * self.groups * self.group_c
                w_grad_size = self.w_out * grad_size
                w_offset_size = self.w_out * offset_size
                loop = each_cycle * block_grad_element // w_grad_size
                ub_loop = self.w_out // ub_factor
                ub_tail = self.w_out % ub_factor
                with self.instance.for_range(0, loop, thread_num=thread_num) as loop_id:
                    with self.instance.for_range(0, ub_loop) as u_id:
                        grad_index = block_offset * block_grad_element + \
                                     loop_id * w_grad_size + u_id * ub_factor * grad_offset
                        offset_index = block_offset * block_offset_element + \
                                       loop_id * w_offset_size + u_id * ub_factor * offset_size
                        self.compute(grad_index,
                                     offset_index,
                                     ub_factor * grad_size,
                                     ub_factor * offset_size)
                    if ub_tail > 0:
                        grad_index = block_offset * block_grad_element + \
                                     loop_id * w_grad_size + \
                                     ub_loop * ub_factor * grad_offset
                        offset_index = block_offset * block_offset_element + \
                                       loop_id * w_offset_size + \
                                       ub_loop * ub_factor * offset_size
                        self.compute(grad_index,
                                     offset_index,
                                     ub_tail * grad_size,
                                     ub_tail * offset_size)

    # 'pylint: disable=too-many-locals
    def compute(self, grad_index, offset_index, grad_size, offset_size):
        """
        calculation process
        """
        buf_list = self.init_ub_tensor()
        ub_index = self.tiling_params["ub_index"]
        n = offset_index // (self.h_out * self.w_out * Constant.W_H_SCALE * self.groups * self.k_h * self.k_w)
        h_out = offset_index % (self.h_out * self.w_out * Constant.W_H_SCALE * self.groups * self.k_h * self.k_w) // \
                (self.w_out * Constant.W_H_SCALE * self.groups * self.k_h * self.k_w)
        w_out = offset_index % (self.h_out * self.w_out * Constant.W_H_SCALE * self.groups * self.k_h * self.k_w) % \
                (self.w_out * Constant.W_H_SCALE * self.groups * self.k_h * self.k_w) // \
                (Constant.W_H_SCALE * self.groups * self.k_h * self.k_w)

        dx_start = n * self.h_in * self.w_in * self.groups * self.group_c
        helper_index = h_out * self.w_out * Constant.W_H_SCALE * self.groups * self.k_h * self.k_w + \
                       w_out * Constant.W_H_SCALE * self.groups * self.k_h * self.k_w

        if ub_index == 0:
            helper_size = self.h_out * self.w_out * Constant.W_H_SCALE * self.groups * self.k_h * self.k_w
        else:
            helper_size = offset_size

        ceil_sub = buf_list[0]
        sub_floor = buf_list[1]
        grad_ub = buf_list[2]
        grad_scale_ub = buf_list[3]
        fp32_ceil_index = buf_list[5]
        check_offset_ub = buf_list[28]
        ceil_sub_fp16 = buf_list[29]
        fp32_ceil_index_fp16 = buf_list[30]

        if ub_index <= 1:
            # grad -> fp32_ceil_index, offset -> ceil_sub, helper -> sub_floor
            self.move_grad_and_offset_to_ub_mode_1([grad_size, offset_size, helper_size],
                                                   [grad_index, offset_index, helper_index],
                                                   [fp32_ceil_index, ceil_sub, sub_floor],
                                                   [fp32_ceil_index_fp16, ceil_sub_fp16])
            # broadcast offset
            offset_base_size = self.groups * self.k_h * self.k_w
            offset_num = offset_size // (Constant.W_H_SCALE * offset_base_size)
            with self.instance.for_range(0, offset_num) as n_i:
                self.broadcast_offset(grad_size,
                                      [n_i * offset_base_size * self.group_c, 
                                       n_i * Constant.W_H_SCALE * offset_base_size],
                                      [sub_floor, ceil_sub, grad_scale_ub, check_offset_ub])
            # transpose grad
            grad_base_size = self.w_out * self.k_h * self.k_w * self.groups * self.group_c
            grad_num = grad_size // grad_base_size
            with self.instance.for_range(0, grad_num) as n_i:
                self.transpose_grad_y(grad_base_size * n_i, [grad_ub, fp32_ceil_index], self.w_out)

            self.compute_index_weight(grad_size, buf_list)

            with self.instance.for_range(0, grad_num) as n_i:
                self.compute_grad_x(grad_size, [grad_base_size * n_i, dx_start], buf_list,
                                    grad_base_size // self.group_c)

            self.compute_grad_offset(grad_size, offset_index, buf_list)

        elif ub_index == 2:
            # grad -> fp32_ceil_index, offset -> ceil_sub helper -> sub_floor
            self.move_grad_and_offset_to_ub_mode_2([grad_size, offset_size, helper_size],
                                                   [grad_index, offset_index, helper_index],
                                                   [fp32_ceil_index, ceil_sub, sub_floor],
                                                   [fp32_ceil_index_fp16, ceil_sub_fp16])
            # broadcast offset
            offset_base_size = self.groups * self.k_h * self.k_w
            offset_num = offset_size // (Constant.W_H_SCALE * offset_base_size)
            with self.instance.for_range(0, offset_num) as n_i:
                self.broadcast_offset(grad_size,
                                      [n_i * offset_base_size * self.group_c, 
                                       n_i * Constant.W_H_SCALE * offset_base_size],
                                      [sub_floor, ceil_sub, grad_scale_ub, check_offset_ub])
            # transpose grad
            grad_base_size = self.k_h * self.k_w * self.groups * self.group_c
            grad_num = grad_size // grad_base_size
            self.transpose_grad_y(0, [grad_ub, fp32_ceil_index], grad_num)

            self.compute_index_weight(grad_size, buf_list)

            self.compute_grad_x(grad_size, [0, dx_start], buf_list, grad_size // self.group_c)

            self.compute_grad_offset(grad_size, offset_index, buf_list)

    def transpose_grad_y(self, start_index, ub_list, w_len):
        """
        transpose grad_y
        """
        base_size = w_len * self.groups * self.k_h * self.k_w
        g_h_w = self.groups * self.k_h * self.k_w
        h_w = self.k_h * self.k_w
        with self.instance.for_range(0, base_size) as b_i:
            w_out = b_i // g_h_w
            groups = b_i % g_h_w // h_w
            k_h = b_i % g_h_w % h_w // self.k_w
            k_w = b_i % g_h_w % h_w % self.k_w
            dst_index = start_index + b_i * self.group_c
            src_index = start_index + \
                        k_h * w_len * self.k_w * self.groups * self.group_c + \
                        w_out * self.k_w * self.groups * self.group_c + \
                        k_w * self.groups * self.group_c + groups * self.group_c
            self.vector_adds(self.group_c, ub_list, 0, dst_index, src_index)

    # 'pylint: disable=too-many-locals
    def broadcast_offset(self, size, start_list, ub_list):
        """
        [...,kh,kw] -> [...,kh,kw,group_c]
        """
        expand_start, offset_start = start_list
        expand_ub, offset_ub, scale_ub, check_ub = ub_list
        offset_w_val = self.instance.Scalar("float32", name="offset_w_val")
        offset_h_val = self.instance.Scalar("float32", name="offset_h_val")
        offset_scale_val = self.instance.Scalar("float32", name="offset_scale_val")
        base_size = self.groups * self.k_h * self.k_w
        with self.instance.for_range(0, base_size) as b_i:
            offset_w = b_i
            offset_h = base_size + b_i
            offset_scale = base_size * 2 + b_i
            offset_w_val.set_as(offset_ub[offset_start + offset_w])
            offset_h_val.set_as(offset_ub[offset_start + offset_h])
            offset_scale_val.set_as(offset_ub[offset_start + offset_scale])
            self.vector_dup(expand_start + b_i * self.group_c, expand_ub, self.group_c, offset_w_val)
            self.vector_dup(expand_start + size + b_i * self.group_c, expand_ub, self.group_c, offset_h_val)
            self.vector_dup(expand_start + b_i * self.group_c, scale_ub, self.group_c, offset_scale_val)

            self.vector_dup(expand_start + b_i * self.group_c, check_ub, self.group_c, offset_w_val)
            self.vector_dup(expand_start + size + b_i * self.group_c, check_ub, self.group_c, offset_h_val)

    # 'pylint: disable=too-many-locals
    def compute_index_weight(self, size, buf_list):
        """
        compute offset index and weight
        """
        ceil_sub = buf_list[0]
        sub_floor = buf_list[1]
        grad_ub = buf_list[2]
        grad_scale_ub = buf_list[3]
        grad_scale_weight_ub = buf_list[4]
        fp32_ceil_index = buf_list[5]
        int32_ceil_index = buf_list[6]
        fp32_floor_index = buf_list[7]
        int32_floor_index = buf_list[8]
        zero_ub = buf_list[9]
        h_max_ub = buf_list[11]
        w_max_ub = buf_list[12]

        # 'fp32 -> int32
        self.vector_conv(size * 2, [int32_floor_index, sub_floor], "floor")
        ones_ub = self.instance.Tensor("int32", (Constant.BLOCK_FP32_SIZE,), 
                                       name="ones_ub", scope=tbe_platform.scope_ubuf)
        self.vector_dup(0, ones_ub, Constant.BLOCK_FP32_SIZE, 1)
        self.vector_binary_op(size * 2, [int32_ceil_index, int32_floor_index, ones_ub], "add",
                              blk_stride_list=[1, 1, 0], rep_stride_list=[8, 8, 0])

        # 'int32 -> fp32
        self.vector_conv(size * 2, [fp32_ceil_index, int32_ceil_index], "")
        self.vector_conv(size * 2, [fp32_floor_index, int32_floor_index], "")

        # 'ceil_sub = 1 + offset_broadcast - fp32_ceil_index
        self.vector_binary_op(size * 2, [ceil_sub, sub_floor, fp32_ceil_index], "sub")
        self.vector_adds(size * 2, [ceil_sub, ceil_sub], 1)

        # 'sub_floor = 1 + fp32_floor_index - offset_broadcast
        self.vector_binary_op(size * 2, [sub_floor, fp32_floor_index, sub_floor], "sub")
        self.vector_adds(size * 2, [sub_floor, sub_floor], 1)

        # 'grad_scale_ub = grad * scale
        self.vector_binary_op(size, [grad_scale_ub, grad_ub, grad_scale_ub], "mul")

        # 'grad_scale_weight_ub[0] = grad * scale * sub_floor[0]
        self.vector_binary_op(size, [grad_scale_weight_ub, grad_scale_ub, sub_floor], "mul", [0, 0, 0])

        # 'grad_scale_weight_ub[3] = grad * scale * ceil_sub[0]
        self.vector_binary_op(size, [grad_scale_weight_ub, grad_scale_ub, ceil_sub], "mul", [size * 3, 0, 0])

        # 'grad_scale_lb = grad * scale * l_b_w * l_b_h -> grad * scale * sub_floor[0] * ceil_sub[1]
        self.vector_binary_op(size, [grad_scale_weight_ub, grad_scale_weight_ub, ceil_sub], "mul", [size, 0, size])

        # 'grad_scale_lt = grad * scale * l_t_w * l_t_h -> grad * scale * sub_floor[0] * sub_floor[1]
        self.vector_binary_op(size, [grad_scale_weight_ub, grad_scale_weight_ub, sub_floor], "mul", [0, 0, size])

        # 'grad_scale_rt = grad * scale * r_t_w * r_t_h -> grad * scale  * ceil_sub[0] * sub_floor[1]
        self.vector_binary_op(size, [grad_scale_weight_ub, grad_scale_weight_ub, sub_floor], "mul",
                              [size * 2, size * 3, size])

        # 'grad_scale_rb = grad * scale * r_b_w * r_b_h -> grad * scale * ceil_sub[0] * ceil_sub[1]
        self.vector_binary_op(size, [grad_scale_weight_ub, grad_scale_weight_ub, ceil_sub], "mul",
                              [size * 3, size * 3, size])

        int32_zero_ub = zero_ub.reinterpret_cast_to("int32")
        int32_h_max_ub = h_max_ub.reinterpret_cast_to("int32")
        int32_w_max_ub = w_max_ub.reinterpret_cast_to("int32")
        self.vector_dup(0, int32_zero_ub, Constant.VECTOR_FP32_SIZE, 0)
        self.vector_dup(0, int32_h_max_ub, Constant.VECTOR_FP32_SIZE, self.h_in - 1)
        self.vector_dup(0, int32_w_max_ub, Constant.VECTOR_FP32_SIZE, self.w_in - 1)

        # 'make index value in [[0,h_in) ,[0, w_in)]
        self.vector_binary_op(size, [int32_floor_index, int32_floor_index, int32_w_max_ub], "min",
                              rep_stride_list=[8, 8, 0])
        self.vector_binary_op(size, [int32_ceil_index, int32_ceil_index, int32_w_max_ub], "min",
                              rep_stride_list=[8, 8, 0])
        self.vector_binary_op(size, [int32_floor_index, int32_floor_index, int32_h_max_ub], "min", [size, size, 0],
                              rep_stride_list=[8, 8, 0])
        self.vector_binary_op(size, [int32_ceil_index, int32_ceil_index, int32_h_max_ub], "min", [size, size, 0],
                              rep_stride_list=[8, 8, 0])
        self.vector_binary_op(size * 2, [int32_floor_index, int32_floor_index, int32_zero_ub], "max",
                              rep_stride_list=[8, 8, 0])
        self.vector_binary_op(size * 2, [int32_ceil_index, int32_ceil_index, int32_zero_ub], "max",
                              rep_stride_list=[8, 8, 0])

    # 'pylint: disable=too-many-locals
    def compute_grad_x(self, size, start_list, buf_list, cycle_size):
        """
        compute grad_x func
        """
        index_start, dx_start = start_list
        int32_ceil_index = buf_list[6]
        int32_floor_index = buf_list[8]
        zero_ub = buf_list[9]
        neg_one_ub = buf_list[10]
        h_max_ub = buf_list[11]
        w_max_ub = buf_list[12]
        h_max_ub2 = buf_list[13]
        w_max_ub2 = buf_list[14]
        x_ub_lt = buf_list[24]
        x_ub_lb = buf_list[25]
        x_ub_rt = buf_list[26]
        x_ub_rb = buf_list[27]
        
        x_ub_lt_fp16 = buf_list[31]
        x_ub_lb_fp16 = buf_list[32]
        x_ub_rt_fp16 = buf_list[33]
        x_ub_rb_fp16 = buf_list[34]

        self.vector_dup(0, zero_ub, Constant.VECTOR_FP32_SIZE, 0)
        self.vector_dup(0, neg_one_ub, Constant.VECTOR_FP32_SIZE, -1)
        self.vector_dup(0, h_max_ub, Constant.VECTOR_FP32_SIZE, self.h_in - 1)
        self.vector_dup(0, h_max_ub2, Constant.VECTOR_FP32_SIZE, self.h_in)
        self.vector_dup(0, w_max_ub, Constant.VECTOR_FP32_SIZE, self.w_in - 1)
        self.vector_dup(0, w_max_ub2, Constant.VECTOR_FP32_SIZE, self.w_in)

        floor_h = self.instance.Scalar("int32", name="floor_h")
        floor_w = self.instance.Scalar("int32", name="floor_w")
        ceil_h = self.instance.Scalar("int32", name="ceil_h")
        ceil_w = self.instance.Scalar("int32", name="ceil_w")
        g_h_w = self.groups * self.k_h * self.k_w
        h_w = self.k_h * self.k_w
        with self.instance.for_range(0, cycle_size) as b_i:
            groups = b_i % g_h_w // h_w
            index = index_start + b_i * self.group_c
            floor_h.set_as(int32_floor_index[size + index])
            floor_w.set_as(int32_floor_index[index])
            ceil_h.set_as(int32_ceil_index[size + index])
            ceil_w.set_as(int32_ceil_index[index])

            dx_lt_index = dx_start + \
                          floor_h * self.w_in * self.groups * self.group_c + \
                          floor_w * self.groups * self.group_c + \
                          groups * self.group_c
            dx_lb_index = dx_start + \
                          ceil_h * self.w_in * self.groups * self.group_c + \
                          floor_w * self.groups * self.group_c + \
                          groups * self.group_c
            dx_rt_index = dx_start + \
                          floor_h * self.w_in * self.groups * self.group_c + \
                          ceil_w * self.groups * self.group_c + \
                          groups * self.group_c
            dx_rb_index = dx_start + \
                          ceil_h * self.w_in * self.groups * self.group_c + \
                          ceil_w * self.groups * self.group_c + \
                          groups * self.group_c

            group_c_burst = self.group_c // Constant.BLOCK_FP32_SIZE
            if self.dtype in ("float16", "bfloat16"):
                group_c_burst_fp16 = self.group_c // Constant.BLOCK_FP16_SIZE
                self.instance.data_move(x_ub_lt_fp16, self.x_gm[dx_lt_index], 0, 1, group_c_burst_fp16, 0, 0)
                self.instance.data_move(x_ub_lb_fp16, self.x_gm[dx_lb_index], 0, 1, group_c_burst_fp16, 0, 0)
                self.instance.data_move(x_ub_rt_fp16, self.x_gm[dx_rt_index], 0, 1, group_c_burst_fp16, 0, 0)
                self.instance.data_move(x_ub_rb_fp16, self.x_gm[dx_rb_index], 0, 1, group_c_burst_fp16, 0, 0)

                self.vector_conv_new(x_ub_lt, x_ub_lt_fp16, self.group_c)
                self.vector_conv_new(x_ub_lb, x_ub_lb_fp16, self.group_c)
                self.vector_conv_new(x_ub_rt, x_ub_rt_fp16, self.group_c)
                self.vector_conv_new(x_ub_rb, x_ub_rb_fp16, self.group_c)

            else:
                self.instance.data_move(x_ub_lt, self.x_gm[dx_lt_index], 0, 1, group_c_burst, 0, 0)
                self.instance.data_move(x_ub_lb, self.x_gm[dx_lb_index], 0, 1, group_c_burst, 0, 0)
                self.instance.data_move(x_ub_rt, self.x_gm[dx_rt_index], 0, 1, group_c_burst, 0, 0)
                self.instance.data_move(x_ub_rb, self.x_gm[dx_rb_index], 0, 1, group_c_burst, 0, 0)

            num = self.group_c // Constant.VECTOR_FP32_SIZE
            tail = self.group_c % Constant.VECTOR_FP32_SIZE
            with self.instance.for_range(0, num) as l_i:
                self.vector_sel(Constant.MASK64_VALUE, size, buf_list,
                                [index + l_i * Constant.VECTOR_FP32_SIZE, l_i * Constant.VECTOR_FP32_SIZE])
            if tail > 0:
                self.vector_sel(tail, size, buf_list,
                                [index + num * Constant.VECTOR_FP32_SIZE, num * Constant.VECTOR_FP32_SIZE])

            self.instance.set_atomic_add(1)
            self.instance.data_move(self.grad_x_gm[dx_lt_index], x_ub_lt, 0, 1, group_c_burst, 0, 0)
            self.instance.data_move(self.grad_x_gm[dx_lb_index], x_ub_lb, 0, 1, group_c_burst, 0, 0)
            self.instance.data_move(self.grad_x_gm[dx_rt_index], x_ub_rt, 0, 1, group_c_burst, 0, 0)
            self.instance.data_move(self.grad_x_gm[dx_rb_index], x_ub_rb, 0, 1, group_c_burst, 0, 0)
            self.instance.set_atomic_add(0)

    # 'pylint: disable=too-many-locals
    def compute_grad_offset(self, size, out_index, buf_list):
        """
        compute grad_offset func
        """
        ceil_sub = buf_list[0]
        sub_floor = buf_list[1]
        grad_ub = buf_list[2]
        grad_scale_ub = buf_list[3]
        grad_scale_weight_ub = buf_list[4]
        fp32_ceil_index = buf_list[5]
        int32_ceil_index = buf_list[6]
        fp32_floor_index = buf_list[7]
        int32_floor_index = buf_list[8]
        fp32_ub1 = int32_ceil_index.reinterpret_cast_to("float32")
        fp32_ub2 = int32_floor_index.reinterpret_cast_to("float32")

        # compute grad_offsets_h
        self.vector_binary_op(size, [grad_scale_weight_ub, fp32_floor_index, sub_floor],
                              "mul", [0, 0, 0])
        self.vector_binary_op(size, [grad_scale_weight_ub, fp32_ceil_index, sub_floor],
                              "mul", [size, 0, 0])
        self.vector_binary_op(size, [grad_scale_weight_ub, fp32_floor_index, ceil_sub],
                              "mul", [size * 2, size, 0])
        self.vector_binary_op(size, [grad_scale_weight_ub, fp32_ceil_index, ceil_sub],
                              "mul", [size * 3, size, 0])

        self.vector_binary_op(size, [fp32_ub2, grad_scale_weight_ub, grad_scale_weight_ub], "sub", [0, size, 0])
        self.vector_binary_op(size, [fp32_ub2, fp32_ub2, grad_scale_weight_ub], "sub", [0, 0, size * 2])
        self.vector_binary_op(size, [fp32_ub2, fp32_ub2, grad_scale_weight_ub], "add", [0, 0, size * 3])
        self.vector_binary_op(size, [fp32_ub2, fp32_ub2, grad_scale_ub], "mul", [0, 0, 0])

        # compute grad_offsets_scale
        self.vector_binary_op(size, [grad_scale_weight_ub, grad_scale_weight_ub, sub_floor],
                              "mul", [0, 0, size])
        self.vector_binary_op(size, [grad_scale_weight_ub, grad_scale_weight_ub, ceil_sub],
                              "mul", [size, size, size])
        self.vector_binary_op(size, [grad_scale_weight_ub, grad_scale_weight_ub, sub_floor],
                              "mul", [size * 2, size * 2, size])
        self.vector_binary_op(size, [grad_scale_weight_ub, grad_scale_weight_ub, ceil_sub],
                              "mul", [size * 3, size * 3, size])

        self.vector_binary_op(size, [fp32_ub1, grad_scale_weight_ub, grad_scale_weight_ub], "add", [0, 0, size])
        self.vector_binary_op(size, [fp32_ub1, fp32_ub1, grad_scale_weight_ub], "add", [0, 0, size * 2])
        self.vector_binary_op(size, [fp32_ub1, fp32_ub1, grad_scale_weight_ub], "add", [0, 0, size * 3])
        self.vector_binary_op(size, [fp32_ub1, fp32_ub1, grad_ub], "mul", [0, 0, 0])

        # compute grad_offsets_w
        self.vector_binary_op(size, [grad_scale_weight_ub, fp32_floor_index, sub_floor],
                              "mul", [0, 0, size])
        self.vector_binary_op(size, [grad_scale_weight_ub, fp32_ceil_index, ceil_sub],
                              "mul", [size, 0, size])
        self.vector_binary_op(size, [grad_scale_weight_ub, fp32_floor_index, sub_floor],
                              "mul", [size * 2, size, size])
        self.vector_binary_op(size, [grad_scale_weight_ub, fp32_ceil_index, ceil_sub],
                              "mul", [size * 3, size, size])

        self.vector_binary_op(size, [grad_ub, grad_scale_weight_ub, grad_scale_weight_ub], "add",
                              [0, size * 2, size * 3])
        self.vector_binary_op(size, [grad_ub, grad_ub, grad_scale_weight_ub], "sub", [0, 0, 0])
        self.vector_binary_op(size, [grad_ub, grad_ub, grad_scale_weight_ub], "sub", [0, 0, size])
        self.vector_binary_op(size, [grad_ub, grad_ub, grad_scale_ub], "mul", [0, 0, 0])

        base_size = self.groups * self.k_h * self.k_w * self.group_c
        num = size // base_size
        with self.instance.for_range(0, num) as n_i:
            index = n_i * Constant.W_H_SCALE * base_size
            self.vector_adds(base_size, [grad_scale_weight_ub, grad_ub], 0, index, n_i * base_size)
            self.vector_adds(base_size, [grad_scale_weight_ub, fp32_ub2], 0, index + base_size, n_i * base_size)
            self.vector_adds(base_size, [grad_scale_weight_ub, fp32_ub1], 0, index + 2 * base_size, n_i * base_size)

        self.reduce_sum(out_index, size * Constant.W_H_SCALE, grad_scale_weight_ub)

    # 'pylint: disable=too-many-locals
    def move_grad_and_offset_to_ub_mode_1(self, size_list, start_list, ub_list, fp16_ub_list):
        """
        move grad and offsets to ub when ub_index in [0, 1]
        """
        grad_size, offset_size, helper_size = size_list
        grad_ub, offset_ub, helper_ub = ub_list
        grad_start, offset_start, helper_start = start_list
        grad_ub_fp16, offset_ub_fp16 = fp16_ub_list
        grad_burst_len = grad_size // Constant.BLOCK_FP32_SIZE
        offset_burst_len = (offset_size + Constant.BLOCK_FP32_SIZE - 1) // Constant.BLOCK_FP32_SIZE
        offset_burst_len_pad = offset_size * Constant.FP32_BYTESIZE
        helper_burst_len = (helper_size + Constant.BLOCK_FP32_SIZE - 1) // Constant.BLOCK_FP32_SIZE
        helper_burst_len_pad = helper_size * Constant.FP32_BYTESIZE
        if self.dtype in ("float16", "bfloat16"):
            grad_burst_len = grad_size // Constant.BLOCK_FP16_SIZE
            offset_burst_len = (offset_size + Constant.BLOCK_FP16_SIZE - 1) // Constant.BLOCK_FP16_SIZE
            offset_burst_len_pad = offset_size * Constant.BF16_BYTESIZE
            self.instance.data_move(grad_ub_fp16, self.grad_gm[grad_start], 0, 1, grad_burst_len, 0, 0)
            if self.support_move_pad:
                self.instance.data_move_pad(offset_ub_fp16, self.offsets_gm[offset_start],
                                            1, offset_burst_len_pad, 0, 0)
            else:
                self.instance.data_move(offset_ub_fp16, self.offsets_gm[offset_start], 0, 1, offset_burst_len, 0, 0)
            self.vector_conv_new(grad_ub, grad_ub_fp16, grad_size)
            self.vector_conv_new(offset_ub, offset_ub_fp16, offset_burst_len * Constant.BLOCK_FP16_SIZE)
        else:
            self.instance.data_move(grad_ub, self.grad_gm[grad_start], 0, 1, grad_burst_len, 0, 0)
            if self.support_move_pad:
                self.instance.data_move_pad(offset_ub, self.offsets_gm[offset_start], 1, offset_burst_len_pad, 0, 0)
            else:
                self.instance.data_move(offset_ub, self.offsets_gm[offset_start], 0, 1, offset_burst_len, 0, 0)
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.instance.data_move_pad(helper_ub, self.helper_gm[helper_start], 1, helper_burst_len_pad, 0, 0)
        else:
            self.instance.data_move(helper_ub, self.helper_gm[helper_start], 0, 1, helper_burst_len, 0, 0)
        # offset + helper
        if self.tiling_params["ub_index"] == 0:
            n_num = offset_size // helper_size
            with self.instance.for_range(0, n_num) as n_i:
                self.vector_binary_op(helper_size, [offset_ub, offset_ub, helper_ub], "add",
                                      [helper_size * n_i, helper_size * n_i, 0])
        else:
            self.vector_binary_op(offset_size, [offset_ub, offset_ub, helper_ub], "add")

    # 'pylint: disable=too-many-locals
    def move_grad_and_offset_to_ub_mode_2(self, size_list, start_list, ub_list, fp16_ub_list):
        """
        move grad and offsets to ub when ub_index in [2,]
        """
        grad_size, offset_size, helper_size = size_list
        grad_ub, offset_ub, helper_ub = ub_list
        grad_start, offset_start, helper_start = start_list
        grad_ub_fp16, offset_ub_fp16 = fp16_ub_list
        gm_base_size = self.w_out * self.k_w * self.groups * self.group_c
        num = grad_size // (self.k_h * self.k_w * self.groups * self.group_c)
        move_base_size = num * self.k_w * self.groups * self.group_c
        grad_burst_len = move_base_size // Constant.BLOCK_FP32_SIZE
        offset_burst_len = (offset_size + Constant.BLOCK_FP32_SIZE - 1) // Constant.BLOCK_FP32_SIZE
        offset_burst_len_pad = offset_size * Constant.FP32_BYTESIZE
        helper_burst_len = (helper_size + Constant.BLOCK_FP32_SIZE - 1) // Constant.BLOCK_FP32_SIZE
        helper_burst_len_pad = helper_size * Constant.FP32_BYTESIZE
        src_stride = (self.w_out - num) * self.k_w * self.groups * self.group_c // Constant.BLOCK_FP32_SIZE

        if self.dtype in ("float16", "bfloat16"):
            grad_burst_len = move_base_size // Constant.BLOCK_FP16_SIZE
            offset_burst_len = (offset_size + Constant.BLOCK_FP16_SIZE - 1) // Constant.BLOCK_FP16_SIZE
            offset_burst_len_pad = offset_size * Constant.BF16_BYTESIZE
            src_stride = (self.w_out - num) * self.k_w * self.groups * self.group_c // Constant.BLOCK_FP16_SIZE
            if src_stride > Constant.MAX_STRIDE:
                with self.instance.for_range(0, self.k_h) as k_i:
                    self.instance.data_move(grad_ub_fp16[k_i * move_base_size],
                                            self.grad_gm[grad_start + k_i * gm_base_size], 0, 1, grad_burst_len, 0, 0)
            else:
                self.instance.data_move(grad_ub_fp16, self.grad_gm[grad_start], 0,
                                        self.k_h, grad_burst_len, src_stride, 0)
            if self.support_move_pad:
                self.instance.data_move_pad(offset_ub_fp16, self.offsets_gm[offset_start],
                                            1, offset_burst_len_pad, 0, 0)
            else:
                self.instance.data_move(offset_ub_fp16, self.offsets_gm[offset_start], 0, 1, offset_burst_len, 0, 0)
            self.vector_conv_new(grad_ub, grad_ub_fp16, self.k_h * move_base_size)
            self.vector_conv_new(offset_ub, offset_ub_fp16, offset_burst_len * Constant.BLOCK_FP16_SIZE)
        else:
            if src_stride > Constant.MAX_STRIDE:
                with self.instance.for_range(0, self.k_h) as k_i:
                    self.instance.data_move(grad_ub[k_i * move_base_size],
                                            self.grad_gm[grad_start + k_i * gm_base_size], 0, 1, grad_burst_len, 0, 0)
            else:
                self.instance.data_move(grad_ub, self.grad_gm[grad_start], 0, self.k_h, grad_burst_len, src_stride, 0)
            if self.support_move_pad:
                self.instance.data_move_pad(offset_ub, self.offsets_gm[offset_start], 1, offset_burst_len_pad, 0, 0)
            else:
                self.instance.data_move(offset_ub, self.offsets_gm[offset_start], 0, 1, offset_burst_len, 0, 0)
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.instance.data_move_pad(helper_ub, self.helper_gm[helper_start], 1, helper_burst_len_pad, 0, 0)
        else:
            self.instance.data_move(helper_ub, self.helper_gm[helper_start], 0, 1, helper_burst_len, 0, 0)
        # offset + helper
        self.vector_binary_op(offset_size, [offset_ub, offset_ub, helper_ub])

    def vector_dup(self, start_index, ub_buf, size, val):
        """
        vector_dup function, set ub_buf to 0
        """
        one_cnt = Constant.VECTOR_FP32_SIZE
        repeat = size // one_cnt
        remainder = size % one_cnt
        loop_repeat = repeat // Constant.MAX_REPEAT
        loop_remainder = repeat % Constant.MAX_REPEAT

        if _is_immediate(size):
            if loop_repeat > 0:
                with self.instance.for_range(0, loop_repeat) as l_i:
                    self.instance.vec_dup(one_cnt,
                                          ub_buf[start_index + l_i * one_cnt * Constant.MAX_REPEAT],
                                          val, Constant.MAX_REPEAT, 8)
            if loop_remainder > 0:
                self.instance.vec_dup(one_cnt,
                                      ub_buf[start_index + loop_repeat * one_cnt * Constant.MAX_REPEAT],
                                      val, loop_remainder, 8)
            if remainder > 0:
                self.instance.vec_dup(remainder,
                                      ub_buf[start_index + repeat * one_cnt],
                                      val, 1, 8)
        else:
            loop_remainder_s = self.instance.Scalar("uint32", name="dup_loop_remainder")
            mask = self.instance.Scalar("uint32", name="dup_mask")
            loop_remainder_s.set_as(loop_remainder)
            mask.set_as(remainder)
            with self.instance.for_range(0, loop_repeat) as l_i:
                self.instance.vec_dup(one_cnt,
                                      ub_buf[start_index + l_i * one_cnt * Constant.MAX_REPEAT],
                                      val, self.repeat_max, 8)
            with self.instance.if_scope(loop_remainder > 0):
                self.instance.vec_dup(one_cnt,
                                      ub_buf[start_index + loop_repeat * one_cnt * Constant.MAX_REPEAT],
                                      val, loop_remainder_s, 8)
            with self.instance.if_scope(remainder > 0):
                self.instance.vec_dup(mask,
                                      ub_buf[start_index + repeat * one_cnt],
                                      val, 1, 8)

    # 'pylint: disable=too-many-locals
    def vector_conv(self, size, ub_list, round_mode, dst_start=0, src_start=0):
        """
        vconv func
        """
        dst_ub, src_ub = ub_list
        one_cnt = Constant.VECTOR_FP32_SIZE
        repeat = size // one_cnt
        remainder = size % one_cnt
        loop_repeat = repeat // Constant.MAX_REPEAT
        loop_remainder = repeat % Constant.MAX_REPEAT

        if _is_immediate(size):
            if loop_repeat > 0:
                with self.instance.for_range(0, loop_repeat) as l_i:
                    self.instance.vconv(one_cnt, round_mode,
                                        dst_ub[dst_start + l_i * one_cnt * Constant.MAX_REPEAT],
                                        src_ub[src_start + l_i * one_cnt * Constant.MAX_REPEAT],
                                        Constant.MAX_REPEAT,
                                        1, 1, 8, 8)
            if loop_remainder > 0:
                self.instance.vconv(one_cnt, round_mode,
                                    dst_ub[dst_start + loop_repeat * one_cnt * Constant.MAX_REPEAT],
                                    src_ub[src_start + loop_repeat * one_cnt * Constant.MAX_REPEAT],
                                    loop_remainder,
                                    1, 1, 8, 8)
            if remainder > 0:
                self.instance.vconv(remainder, round_mode,
                                    dst_ub[dst_start + repeat * one_cnt],
                                    src_ub[src_start + repeat * one_cnt],
                                    1,
                                    1, 1, 8, 8)
        else:
            loop_remainder_s = self.instance.Scalar("uint32", name="vconv_loop_remainder")
            mask = self.instance.Scalar("uint32", name="vconv_mask")
            loop_remainder_s.set_as(loop_remainder)
            mask.set_as(remainder)
            with self.instance.for_range(0, loop_repeat) as l_i:
                self.instance.vconv(one_cnt, round_mode,
                                    dst_ub[dst_start + l_i * one_cnt * Constant.MAX_REPEAT],
                                    src_ub[src_start + l_i * one_cnt * Constant.MAX_REPEAT],
                                    self.repeat_max,
                                    1, 1, 8, 8)
            with self.instance.if_scope(loop_remainder > 0):
                self.instance.vconv(one_cnt, round_mode,
                                    dst_ub[dst_start + loop_repeat * one_cnt * Constant.MAX_REPEAT],
                                    src_ub[src_start + loop_repeat * one_cnt * Constant.MAX_REPEAT],
                                    loop_remainder_s,
                                    1, 1, 8, 8)
            with self.instance.if_scope(remainder > 0):
                self.instance.vconv(mask, round_mode,
                                    dst_ub[dst_start + repeat * one_cnt],
                                    src_ub[src_start + repeat * one_cnt],
                                    1,
                                    1, 1, 8, 8)

    def vector_conv_new(self, dst, src, process_num):
        """
        vector_conv_new operator

        Parameters:
        ----------
        dst: Dst tensor to store the result.
        process_num: The length to operate.
        src: src tensor.
        dtype: Data type.

        Returns:
        -------
        None.
        The result will be stored in dst.
        """
        src_dtype = src.dtype
        dst_dtype = dst.dtype
        if src_dtype in ("float32",) and dst_dtype in ("float16", "bfloat16"):
            dst_repeat_stride, src_repeat_stride = 4, 8
        else:
            dst_repeat_stride, src_repeat_stride = 8, 4
        repeat = process_num // Constant.VECTOR_FP32_SIZE
        remainder = process_num % Constant.VECTOR_FP32_SIZE
        cycles_num = repeat // Constant.MAX_REPEAT
        cycles_tail = repeat % Constant.MAX_REPEAT

        with self.instance.for_range(0, cycles_num) as cycles_index:
            self.instance.vconv(Constant.VECTOR_FP32_SIZE, "",
                                    dst[cycles_index * Constant.MAX_REPEAT * Constant.VECTOR_FP32_SIZE],
                                    src[cycles_index * Constant.MAX_REPEAT * Constant.VECTOR_FP32_SIZE],
                                    Constant.MAX_REPEAT, 1, 1, dst_repeat_stride, src_repeat_stride)

        with self.instance.if_scope(cycles_tail > 0):
            self.instance.vconv(Constant.VECTOR_FP32_SIZE, "",
                                    dst[cycles_num * Constant.MAX_REPEAT * Constant.VECTOR_FP32_SIZE],
                                    src[cycles_num * Constant.MAX_REPEAT * Constant.VECTOR_FP32_SIZE],
                                    cycles_tail, 1, 1,
                                    dst_repeat_stride, src_repeat_stride)

        with self.instance.if_scope(remainder > 0):
            self.instance.vconv(remainder, "", dst[repeat * Constant.VECTOR_FP32_SIZE],
                                    src[repeat * Constant.VECTOR_FP32_SIZE], 1, 1, 1,
                                    dst_repeat_stride, src_repeat_stride)

    # 'pylint: disable=too-many-locals
    def vector_adds(self, size, ub_list, val, dst_start=0, src_start=0):
        """
        vadds func
        """
        dst_ub, src_ub = ub_list
        one_cnt = Constant.VECTOR_FP32_SIZE
        repeat = size // one_cnt
        remainder = size % one_cnt
        loop_repeat = repeat // Constant.MAX_REPEAT
        loop_remainder = repeat % Constant.MAX_REPEAT

        if _is_immediate(size):
            if loop_repeat > 0:
                with self.instance.for_range(0, loop_repeat) as l_i:
                    self.instance.vadds(one_cnt,
                                        dst_ub[dst_start + l_i * one_cnt * Constant.MAX_REPEAT],
                                        src_ub[src_start + l_i * one_cnt * Constant.MAX_REPEAT],
                                        val,
                                        Constant.MAX_REPEAT,
                                        1, 1, 8, 8)
            if loop_remainder > 0:
                self.instance.vadds(one_cnt,
                                    dst_ub[dst_start + loop_repeat * one_cnt * Constant.MAX_REPEAT],
                                    src_ub[src_start + loop_repeat * one_cnt * Constant.MAX_REPEAT],
                                    val,
                                    loop_remainder,
                                    1, 1, 8, 8)
            if remainder > 0:
                self.instance.vadds(remainder,
                                    dst_ub[dst_start + repeat * one_cnt],
                                    src_ub[src_start + repeat * one_cnt],
                                    val,
                                    1,
                                    1, 1, 8, 8)
        else:
            loop_remainder_s = self.instance.Scalar("uint32", name="adds_loop_remainder")
            mask = self.instance.Scalar("uint32", name="adds_mask")
            loop_remainder_s.set_as(loop_remainder)
            mask.set_as(remainder)
            with self.instance.for_range(0, loop_repeat) as l_i:
                self.instance.vadds(one_cnt,
                                    dst_ub[dst_start + l_i * one_cnt * Constant.MAX_REPEAT],
                                    src_ub[src_start + l_i * one_cnt * Constant.MAX_REPEAT],
                                    val,
                                    self.repeat_max,
                                    1, 1, 8, 8)
            with self.instance.if_scope(loop_remainder > 0):
                self.instance.vadds(one_cnt,
                                    dst_ub[dst_start + loop_repeat * one_cnt * Constant.MAX_REPEAT],
                                    src_ub[src_start + loop_repeat * one_cnt * Constant.MAX_REPEAT],
                                    val,
                                    loop_remainder_s,
                                    1, 1, 8, 8)
            with self.instance.if_scope(remainder > 0):
                self.instance.vadds(mask,
                                    dst_ub[dst_start + repeat * one_cnt],
                                    src_ub[src_start + repeat * one_cnt],
                                    val,
                                    1,
                                    1, 1, 8, 8)

    # 'pylint: disable=too-many-locals
    def vector_binary_op(self, size, ub_list, op_name="add", start_list=None, blk_stride_list=None,
                         rep_stride_list=None):
        """
        vadd, vsub, vmul, vmin, vmax func
        """
        func_map = {
            "add": self.instance.vadd,
            "sub": self.instance.vsub,
            "mul": self.instance.vmul,
            "min": self.instance.vmin,
            "max": self.instance.vmax
        }

        dst_ub, src0_ub, src1_ub = ub_list
        if start_list is None:
            start_list = (0, 0, 0)
        if blk_stride_list is None:
            blk_stride_list = (1, 1, 1)
        if rep_stride_list is None:
            rep_stride_list = (8, 8, 8)
        dst_start, src0_start, src1_start = start_list
        dst_blk_stride, src0_blk_stride, src1_blk_stride = blk_stride_list
        dst_rep_stride, src0_rep_stride, src1_rep_stride = rep_stride_list

        one_cnt = Constant.VECTOR_FP32_SIZE
        repeat = size // one_cnt
        remainder = size % one_cnt
        loop_repeat = repeat // Constant.MAX_REPEAT
        loop_remainder = repeat % Constant.MAX_REPEAT

        if _is_immediate(size):
            if loop_repeat > 0:
                with self.instance.for_range(0, loop_repeat) as l_i:
                    func_map.get(op_name)(one_cnt,
                                      dst_ub[dst_start + l_i * dst_rep_stride * \
                                      Constant.BLOCK_FP32_SIZE * Constant.MAX_REPEAT],
                                      src0_ub[src0_start + l_i * src0_rep_stride * \
                                      Constant.BLOCK_FP32_SIZE * Constant.MAX_REPEAT],
                                      src1_ub[src1_start + l_i * src1_rep_stride * \
                                      Constant.BLOCK_FP32_SIZE * Constant.MAX_REPEAT],
                                      Constant.MAX_REPEAT,
                                      dst_blk_stride, src0_blk_stride, src1_blk_stride,
                                      dst_rep_stride, src0_rep_stride, src1_rep_stride)
            if loop_remainder > 0:
                func_map.get(op_name)(one_cnt,
                                  dst_ub[dst_start + loop_repeat * dst_rep_stride * \
                                  Constant.BLOCK_FP32_SIZE * Constant.MAX_REPEAT],
                                  src0_ub[src0_start + loop_repeat * src0_rep_stride * \
                                  Constant.BLOCK_FP32_SIZE * Constant.MAX_REPEAT],
                                  src1_ub[src1_start + loop_repeat * src1_rep_stride * \
                                  Constant.BLOCK_FP32_SIZE * Constant.MAX_REPEAT],
                                  loop_remainder,
                                  dst_blk_stride, src0_blk_stride, src1_blk_stride,
                                  dst_rep_stride, src0_rep_stride, src1_rep_stride)
            if remainder > 0:
                func_map.get(op_name)(remainder,
                                  dst_ub[dst_start + repeat * dst_rep_stride * Constant.BLOCK_FP32_SIZE],
                                  src0_ub[src0_start + repeat * src0_rep_stride * Constant.BLOCK_FP32_SIZE],
                                  src1_ub[src1_start + repeat * src1_rep_stride * Constant.BLOCK_FP32_SIZE],
                                  1,
                                  dst_blk_stride, src0_blk_stride, src1_blk_stride,
                                  dst_rep_stride, src0_rep_stride, src1_rep_stride)
        else:
            loop_remainder_s = self.instance.Scalar("uint32", name="binary_loop_remainder")
            mask = self.instance.Scalar("uint32", name="binary_mask")
            loop_remainder_s.set_as(loop_remainder)
            mask.set_as(remainder)
            with self.instance.for_range(0, loop_repeat) as l_i:
                func_map.get(op_name)(one_cnt,
                                  dst_ub[dst_start + l_i * dst_rep_stride * \
                                  Constant.BLOCK_FP32_SIZE * Constant.MAX_REPEAT],
                                  src0_ub[src0_start + l_i * src0_rep_stride * \
                                  Constant.BLOCK_FP32_SIZE * Constant.MAX_REPEAT],
                                  src1_ub[src1_start + l_i * src1_rep_stride * \
                                  Constant.BLOCK_FP32_SIZE * Constant.MAX_REPEAT],
                                  self.repeat_max,
                                  dst_blk_stride, src0_blk_stride, src1_blk_stride,
                                  dst_rep_stride, src0_rep_stride, src1_rep_stride)
            with self.instance.if_scope(loop_remainder > 0):
                func_map.get(op_name)(one_cnt,
                                  dst_ub[dst_start + loop_repeat * dst_rep_stride * \
                                  Constant.BLOCK_FP32_SIZE * Constant.MAX_REPEAT],
                                  src0_ub[src0_start + loop_repeat * src0_rep_stride * \
                                  Constant.BLOCK_FP32_SIZE * Constant.MAX_REPEAT],
                                  src1_ub[src1_start + loop_repeat * src1_rep_stride * \
                                  Constant.BLOCK_FP32_SIZE * Constant.MAX_REPEAT],
                                  loop_remainder_s,
                                  dst_blk_stride, src0_blk_stride, src1_blk_stride,
                                  dst_rep_stride, src0_rep_stride, src1_rep_stride)
            with self.instance.if_scope(remainder > 0):
                func_map.get(op_name)(mask,
                                  dst_ub[dst_start + repeat * dst_rep_stride * Constant.BLOCK_FP32_SIZE],
                                  src0_ub[src0_start + repeat * src0_rep_stride * Constant.BLOCK_FP32_SIZE],
                                  src1_ub[src1_start + repeat * src1_rep_stride * Constant.BLOCK_FP32_SIZE],
                                  1,
                                  dst_blk_stride, src0_blk_stride, src1_blk_stride,
                                  dst_rep_stride, src0_rep_stride, src1_rep_stride)

    # 'pylint: disable=too-many-locals
    def vector_sel(self, mask, size, buf_list, start_list):
        """
        vsel func
        """
        grad_scale_weight_ub = buf_list[4]
        fp32_ceil_index = buf_list[5]
        fp32_floor_index = buf_list[7]
        zero_ub = buf_list[9]
        neg_one_ub = buf_list[10]
        h_max_ub = buf_list[11]
        w_max_ub = buf_list[12]
        h_max_ub2 = buf_list[13]
        w_max_ub2 = buf_list[14]
        mask_ub1 = buf_list[15]
        mask_ub2 = buf_list[16]
        mask_ub3 = buf_list[17]
        mask_ub4 = buf_list[18]
        mask_ub5 = buf_list[19]
        mask_ub6 = buf_list[20]
        mask_ub7 = buf_list[21]
        mask_ub8 = buf_list[22]
        mask_ub9 = buf_list[23]
        x_ub_lt = buf_list[24]
        x_ub_lb = buf_list[25]
        x_ub_rt = buf_list[26]
        x_ub_rb = buf_list[27]
        check_offset_ub = buf_list[28]

        index_start, group_c_start = start_list
        mask_bit = 16
        and_mask = (mask + mask_bit - 1) // mask_bit

        # '-1 < h < H_IN
        self.instance.vec_cmpv_gt(mask_ub6, check_offset_ub[size + index_start], neg_one_ub, 1, 8, 0)
        self.instance.vec_cmpv_lt(mask_ub7, check_offset_ub[size + index_start], h_max_ub2, 1, 8, 0)
        self.instance.vand(and_mask, mask_ub6, mask_ub6, mask_ub7, 1, 1, 1, 1, 8, 8, 8)

        # '-1 < w < W_IN
        self.instance.vec_cmpv_gt(mask_ub8, check_offset_ub[index_start], neg_one_ub, 1, 8, 0)
        self.instance.vec_cmpv_lt(mask_ub9, check_offset_ub[index_start], w_max_ub2, 1, 8, 0)
        self.instance.vand(and_mask, mask_ub8, mask_ub8, mask_ub9, 1, 1, 1, 1, 8, 8, 8)

        # '-1 < h < H_IN and '-1 < w < W_IN
        self.instance.vand(and_mask, mask_ub7, mask_ub6, mask_ub8, 1, 1, 1, 1, 8, 8, 8)

        # '0 <= h <= H_IN - 1
        self.instance.vec_cmpv_ge(mask_ub1, fp32_floor_index[size + index_start], zero_ub, 1, 8, 0)
        self.instance.vec_cmpv_le(mask_ub2, fp32_floor_index[size + index_start], h_max_ub, 1, 8, 0)
        self.instance.vand(and_mask, mask_ub3, mask_ub1, mask_ub2, 1, 1, 1, 1, 8, 8, 8)
        self.instance.vand(and_mask, mask_ub3, mask_ub3, mask_ub7, 1, 1, 1, 1, 8, 8, 8)

        self.instance.vec_cmpv_ge(mask_ub1, fp32_ceil_index[size + index_start], zero_ub, 1, 8, 0)
        self.instance.vec_cmpv_le(mask_ub2, fp32_ceil_index[size + index_start], h_max_ub, 1, 8, 0)
        self.instance.vand(and_mask, mask_ub5, mask_ub1, mask_ub2, 1, 1, 1, 1, 8, 8, 8)
        self.instance.vand(and_mask, mask_ub5, mask_ub5, mask_ub7, 1, 1, 1, 1, 8, 8, 8)

        # '0 <= w <= W_IN - 1
        self.instance.vec_cmpv_ge(mask_ub1, fp32_floor_index[index_start], zero_ub, 1, 8, 0)
        self.instance.vec_cmpv_le(mask_ub2, fp32_floor_index[index_start], w_max_ub, 1, 8, 0)
        self.instance.vand(and_mask, mask_ub4, mask_ub1, mask_ub2, 1, 1, 1, 1, 8, 8, 8)
        self.instance.vand(and_mask, mask_ub4, mask_ub4, mask_ub7, 1, 1, 1, 1, 8, 8, 8)

        self.instance.vec_cmpv_ge(mask_ub1, fp32_ceil_index[index_start], zero_ub, 1, 8, 0)
        self.instance.vec_cmpv_le(mask_ub2, fp32_ceil_index[index_start], w_max_ub, 1, 8, 0)
        self.instance.vand(and_mask, mask_ub2, mask_ub1, mask_ub2, 1, 1, 1, 1, 8, 8, 8)
        self.instance.vand(and_mask, mask_ub2, mask_ub2, mask_ub7, 1, 1, 1, 1, 8, 8, 8)

        # 'dx_lt -> fp32_floor_index[1] and fp32_floor_index[0]
        self.instance.vand(and_mask, mask_ub1, mask_ub3, mask_ub4, 1, 1, 1, 1, 8, 8, 8)
        cmp_mask1 = self.instance.mov_tensor_to_cmpmask(mask_ub1)
        self.instance.vsel(mask, 0, fp32_floor_index[index_start], cmp_mask1, x_ub_lt[group_c_start], zero_ub,
                           1, 1, 1, 1, 8, 8, 8)
        self.instance.vsel(mask, 0, x_ub_lt[group_c_start], cmp_mask1, grad_scale_weight_ub[index_start],
                           zero_ub, 1, 1, 1, 1, 8, 8, 8)

        # 'dx_l_b -> fp32_ceil_index[1] and fp32_floor_index[0]
        self.instance.vand(and_mask, mask_ub4, mask_ub5, mask_ub4, 1, 1, 1, 1, 8, 8, 8)
        cmp_mask2 = self.instance.mov_tensor_to_cmpmask(mask_ub4)
        self.instance.vsel(mask, 0, fp32_ceil_index[index_start], cmp_mask2, x_ub_lb[group_c_start], zero_ub,
                           1, 1, 1, 1, 8, 8, 8)
        self.instance.vsel(mask, 0, x_ub_lb[group_c_start], cmp_mask2, grad_scale_weight_ub[size + index_start],
                           zero_ub, 1, 1, 1, 1, 8, 8, 8)

        # 'dx_r_t -> fp32_floor_index[1] and fp32_ceil_index[0]
        self.instance.vand(and_mask, mask_ub3, mask_ub3, mask_ub2, 1, 1, 1, 1, 8, 8, 8)
        cmp_mask3 = self.instance.mov_tensor_to_cmpmask(mask_ub3)
        self.instance.vsel(mask, 0, fp32_floor_index[size + index_start], cmp_mask3, x_ub_rt[group_c_start], zero_ub,
                           1, 1, 1, 1, 8, 8, 8)
        self.instance.vsel(mask, 0, x_ub_rt[group_c_start], cmp_mask3, grad_scale_weight_ub[size * 2 + index_start],
                           zero_ub, 1, 1, 1, 1, 8, 8, 8)

        # 'dx_r_b -> fp32_ceil_index[1] and fp32_ceil_index[0]
        self.instance.vand(and_mask, mask_ub2, mask_ub5, mask_ub2, 1, 1, 1, 1, 8, 8, 8)
        cmp_mask4 = self.instance.mov_tensor_to_cmpmask(mask_ub2)
        self.instance.vsel(mask, 0, fp32_ceil_index[size + index_start], cmp_mask4, x_ub_rb[group_c_start], zero_ub,
                           1, 1, 1, 1, 8, 8, 8)
        self.instance.vsel(mask, 0, x_ub_rb[group_c_start], cmp_mask4, grad_scale_weight_ub[size * 3 + index_start],
                           zero_ub, 1, 1, 1, 1, 8, 8, 8)

    def reduce_sum(self, out_index, size, out_ub):
        """
        summation of group_c axis, [...,kh,kw,group_c] -> [...,kh,kw]
        """
        if self.group_c <= Constant.VECTOR_FP32_SIZE:
            self.reduce_sum_group_c_le_64(out_index, self.group_c, size, out_ub)
        else:
            if self.group_c // Constant.BLOCK_FP32_SIZE > Constant.MAX_REPEAT:
                self.group_c_stride_gt_255(out_index, size, out_ub)
            else:
                self.group_c_stride_le_255(out_index, size, out_ub)

    def reduce_sum_group_c_le_64(self, out_index, one_cnt, size, out_ub):
        """
        processing group_c less than or equal to 64 scenes
        """
        repeat = size // one_cnt
        max_repeat = Constant.MAX_REPEAT // Constant.BLOCK_FP32_SIZE * Constant.BLOCK_FP32_SIZE
        loop_repeat = repeat // max_repeat
        loop_remainder = repeat % max_repeat

        if _is_immediate(size):
            if loop_repeat > 0:
                with self.instance.for_range(0, loop_repeat) as l_i:
                    self.instance.vcadd(one_cnt,
                                        out_ub[l_i * max_repeat],
                                        out_ub[l_i * max_repeat * one_cnt],
                                        max_repeat,
                                        1, 1, one_cnt // Constant.BLOCK_FP32_SIZE)
            if loop_remainder > 0:
                self.instance.vcadd(one_cnt,
                                    out_ub[loop_repeat * max_repeat],
                                    out_ub[loop_repeat * max_repeat * one_cnt],
                                    loop_remainder,
                                    1, 1, one_cnt // Constant.BLOCK_FP32_SIZE)
        else:
            max_repeat_s = self.instance.Scalar("uint32", name="max_repeat_s")
            max_repeat_s.set_as(max_repeat)
            loop_remainder_s = self.instance.Scalar("uint32", name="loop_remainder_s")
            loop_remainder_s.set_as(loop_remainder)
            with self.instance.for_range(0, loop_repeat) as l_i:
                self.instance.vcadd(one_cnt,
                                    out_ub[l_i * max_repeat],
                                    out_ub[l_i * max_repeat * one_cnt],
                                    max_repeat_s,
                                    1, 1, one_cnt // Constant.BLOCK_FP32_SIZE)
            with self.instance.if_scope(loop_remainder > 0):
                self.instance.vcadd(one_cnt,
                                    out_ub[loop_repeat * max_repeat],
                                    out_ub[loop_repeat * max_repeat * one_cnt],
                                    loop_remainder_s,
                                    1, 1, one_cnt // Constant.BLOCK_FP32_SIZE)

        input_dict = {
            "instance": self.instance,
            "out_ub": out_ub,
            "out_gm": self.grad_offsets_gm,
            "gm_offset": out_index,
            "element_num": repeat,
            "dsize": self.dsize,
        }
        common_util.move_out_non32_alignment(input_dict)

    def group_c_stride_gt_255(self, out_index, size, out_ub):
        """
        sum each group_c and summarize later, when the stride of group_c > 255
        """
        group_c_num = size // self.group_c
        with self.instance.for_range(0, group_c_num) as l_i:
            self.vcadd_one_group_c(out_ub, [l_i * Constant.VECTOR_FP32_SIZE, l_i * self.group_c])
        self.reduce_sum_group_c_le_64(out_index, Constant.VECTOR_FP32_SIZE,
                                      group_c_num * Constant.VECTOR_FP32_SIZE, out_ub)

    def group_c_stride_le_255(self, out_index, size, out_ub):
        """
        using repeat to sum group_c, when the stride of group_c <= 255
        """
        group_c_num = size // self.group_c
        loop_repeat = group_c_num // Constant.MAX_REPEAT
        loop_remainder = group_c_num % Constant.MAX_REPEAT
        one_cnt = Constant.VECTOR_FP32_SIZE

        if _is_immediate(size):
            if loop_repeat > 0:
                with self.instance.for_range(0, loop_repeat) as l_i:
                    self.vcadd_stride_group_c(Constant.MAX_REPEAT, out_ub,
                                              [l_i * Constant.MAX_REPEAT * one_cnt,
                                              l_i * Constant.MAX_REPEAT * self.group_c])
            if loop_remainder > 0:
                self.vcadd_stride_group_c(loop_remainder, out_ub,
                                          [loop_repeat * Constant.MAX_REPEAT * one_cnt,
                                          loop_repeat * Constant.MAX_REPEAT * self.group_c])
        else:
            loop_remainder_s = self.instance.Scalar("uint32", name="loop_remainder_s")
            loop_remainder_s.set_as(loop_remainder)
            with self.instance.for_range(0, loop_repeat) as l_i:
                self.vcadd_stride_group_c(self.repeat_max, out_ub,
                                          [l_i * Constant.MAX_REPEAT * one_cnt,
                                          l_i * Constant.MAX_REPEAT * self.group_c])
            with self.instance.if_scope(loop_remainder > 0):
                self.vcadd_stride_group_c(loop_remainder_s, out_ub,
                                          [loop_repeat * Constant.MAX_REPEAT * one_cnt,
                                          loop_repeat * Constant.MAX_REPEAT * self.group_c])

        self.reduce_sum_group_c_le_64(out_index, Constant.VECTOR_FP32_SIZE,
                                      group_c_num * Constant.VECTOR_FP32_SIZE, out_ub)

    def vcadd_stride_group_c(self, repeat, out_ub, start_list):
        """
        using repeat to sum group_c
        """
        dst_start, src_start = start_list
        one_cnt = Constant.VECTOR_FP32_SIZE
        num = self.group_c // Constant.VECTOR_FP32_SIZE
        tail = self.group_c % Constant.VECTOR_FP32_SIZE
        if tail > 0:
            self.instance.vadd(tail,
                               out_ub[src_start],
                               out_ub[src_start],
                               out_ub[src_start + num * one_cnt],
                               repeat,
                               1, 1, 1, self.group_c // 8, self.group_c // 8, self.group_c // 8)
        if num > 1:
            with self.instance.for_range(1, num) as l_i:
                self.instance.vadd(one_cnt,
                                   out_ub[src_start],
                                   out_ub[src_start],
                                   out_ub[src_start + l_i * one_cnt],
                                   repeat,
                                   1, 1, 1, self.group_c // 8, self.group_c // 8, self.group_c // 8)

        self.instance.vadds(one_cnt,
                            out_ub[dst_start],
                            out_ub[src_start],
                            0,
                            repeat,
                            1, 1, 8, self.group_c // 8)

    def vcadd_one_group_c(self, out_ub, start_list):
        """
        sum each group_c
        """
        dst_start, src_start = start_list
        one_cnt = Constant.VECTOR_FP32_SIZE
        num = self.group_c // Constant.VECTOR_FP32_SIZE
        tail = self.group_c % Constant.VECTOR_FP32_SIZE
        loop_repeat = num // Constant.MAX_REPEAT
        loop_remainder = num % Constant.MAX_REPEAT
        if tail > 0:
            self.instance.vadd(tail,
                               out_ub[src_start],
                               out_ub[src_start],
                               out_ub[src_start + num * one_cnt],
                               1, 1, 1, 1, 8, 8, 8)
        if loop_repeat > 1:
            with self.instance.for_range(1, loop_repeat) as l_i:
                self.instance.vadd(one_cnt,
                                   out_ub[src_start],
                                   out_ub[src_start],
                                   out_ub[src_start + l_i * Constant.MAX_REPEAT * one_cnt],
                                   Constant.MAX_REPEAT,
                                   1, 1, 1, 8, 8, 8)
        if loop_remainder > 0 and loop_repeat > 0:
            self.instance.vadd(one_cnt,
                               out_ub[src_start],
                               out_ub[src_start],
                               out_ub[src_start + loop_repeat * Constant.MAX_REPEAT * one_cnt],
                               loop_remainder,
                               1, 1, 1, 8, 8, 8)
        size = Constant.MAX_REPEAT if loop_repeat > 0 else loop_remainder
        cur_size = size // 2
        cur_tail = size % 2
        while cur_size > 0:
            self.instance.vadd(one_cnt,
                               out_ub[src_start],
                               out_ub[src_start],
                               out_ub[src_start + (cur_size + cur_tail) * one_cnt],
                               cur_size,
                               1, 1, 1, 8, 8, 8)
            cur_tail = cur_size % 2
            cur_size = cur_size // 2

        self.instance.vadds(one_cnt,
                            out_ub[dst_start],
                            out_ub[src_start],
                            0, 1, 1, 1, 8, 8)
