#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

is_nan
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_tik_comm_func import ceil_div as _ceil_div
from impl import common_util


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # ub size count
    UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    # aicore count
    CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    # byte count one block
    BLOCK_BYTE_COUNT = 32
    # repeat up limit for mte
    REPEAT_LIMIT = 255
    # max int64 value
    MAX_INT64_VALUE = 2**64 - 1
    # parameters for moving tiling data
    TILING_CTRL_PARAM = ("int64", 9, 4)
    # the data size of bf16
    BF16_DATA_SIZE = 2

    # the constants related to input dtype
    CONSTANT_MAP = {
        "float16": {
            "inf_num": 0x7c00,  # 0x7c00=31744, means positive infinity in float16 case.
            "sign_mask": 0x7fff,  # 0x7fff=32767=2**15-1, do AND operation to ignore sign bit.
            "flag_tensor_multiple": 1
        },
        "float32": {
            "inf_num": 0x7f800000,  # 0x7f800000=2139095040 means positive infinity in foat32 case.
            "sign_mask": 0x7fffffff,  # 0x7fffffff=2147483647=2**31-1, do AND operation to ignore sign bit.
            "flag_tensor_multiple": 2
        },
        "bfloat16": {
            "inf_num": 0x7f80,  # 0x7f80=32640 means positive infinity in bfloat16 case.
            "sign_mask": 0x7fff,  # 0x7fff=32767=2**15-1, do AND operation to ignore sign bit.
            "flag_tensor_multiple": 1
        },
    }


def _get_data_size(dtype):
    return Constant.BF16_DATA_SIZE if dtype == "bfloat16" else common_util.get_data_size(dtype)


def _get_element_cnt_one_block(dtype):
    """
    get element count in a block
    """
    byte_len = common_util.get_data_size(dtype)
    element_cnt = Constant.BLOCK_BYTE_COUNT // byte_len

    return element_cnt


def _get_max_element_in_ub(dtype, ub_part):
    """
    get the up limit elements in UB
    """
    byte_len = common_util.get_data_size(dtype)

    ub_upper_limit = ((Constant.UB_SIZE - 2 * 1024) // 2) // ub_part
    element_size = ub_upper_limit // byte_len

    return element_size


def _check_input_params(input_dtype, output_dtype):
    """
    check whether the input parameters is valid or not
    """
    para_check.check_dtype(input_dtype, ("float16", "float32", "bfloat16"), param_name="x")
    para_check.check_dtype(output_dtype, ("int8",), param_name="y")


def _get_ub_max_size(input_dtype, output_dtype):
    """
    output 32 bytes align
    """
    ub_x_size = _get_max_element_in_ub(input_dtype, 1) // 2
    return ub_x_size - ub_x_size % (Constant.BLOCK_BYTE_COUNT // common_util.get_data_size(output_dtype))


# 'pylint:disable=too-many-arguments
def _scalar_vector_func(tik_inst, vec_func, dst, src, scalar, data_len, data_type):
    """
    do scalar vector operator
    """
    data_one_block = _get_element_cnt_one_block(data_type)
    repeat_data_num = 8 * data_one_block
    repeat = data_len // repeat_data_num
    repeat_tail = data_len % repeat_data_num
    loop_repeat_cnt = repeat // Constant.REPEAT_LIMIT
    with tik_inst.if_scope(repeat >= Constant.REPEAT_LIMIT):
        with tik_inst.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
            offset = repeat_lp_cnt * Constant.REPEAT_LIMIT * repeat_data_num
            vec_func(repeat_data_num, dst[offset], src[offset], scalar, Constant.REPEAT_LIMIT, 1, 1, 8, 8)
    left_repeat = repeat - loop_repeat_cnt * Constant.REPEAT_LIMIT
    with tik_inst.if_scope(left_repeat > 0):
        offset = loop_repeat_cnt * Constant.REPEAT_LIMIT * repeat_data_num
        vec_func(repeat_data_num, dst[offset], src[offset], scalar, left_repeat, 1, 1, 8, 8)
    with tik_inst.if_scope(repeat_tail > 0):
        offset = repeat * repeat_data_num
        vec_func(repeat_tail, dst[offset], src[offset], scalar, 1, 1, 1, 8, 8)


# 'pylint:disable=too-many-arguments
def _vector_single_src_func(tik_inst, vec_func, dst, src, data_len, data_type):
    """
    do vector operator
    """
    data_one_block = _get_element_cnt_one_block(data_type)
    repeat_data_num = 8 * data_one_block
    repeat = data_len // repeat_data_num
    repeat_tail = data_len % repeat_data_num
    loop_repeat_cnt = repeat // Constant.REPEAT_LIMIT


    with tik_inst.if_scope(repeat >= Constant.REPEAT_LIMIT):
        with tik_inst.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
            offset = repeat_lp_cnt * Constant.REPEAT_LIMIT * repeat_data_num
            vec_func(repeat_data_num, dst[offset], src[offset], Constant.REPEAT_LIMIT, 1, 1, 8, 8)
    left_repeat = repeat - loop_repeat_cnt * Constant.REPEAT_LIMIT

    with tik_inst.if_scope(left_repeat > 0):
        offset = loop_repeat_cnt * Constant.REPEAT_LIMIT * repeat_data_num
        vec_func(repeat_data_num, dst[offset], src[offset], left_repeat, 1, 1, 8, 8)
    with tik_inst.if_scope(repeat_tail > 0):
        offset = repeat * repeat_data_num
        vec_func(repeat_tail, dst[offset], src[offset], 1, 1, 1, 8, 8)


# 'pylint:disable=too-many-arguments
def _vector_double_src_func(tik_inst, vec_func, dst, src1, src2, data_len, data_type):
    """
    do vector operator
    """
    data_one_block = _get_element_cnt_one_block(data_type)
    repeat_data_num = 8 * data_one_block
    repeat = data_len // repeat_data_num
    repeat_tail = data_len % repeat_data_num
    loop_repeat_cnt = repeat // Constant.REPEAT_LIMIT

    with tik_inst.if_scope(repeat >= Constant.REPEAT_LIMIT):
        with tik_inst.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
            offset = repeat_lp_cnt * Constant.REPEAT_LIMIT * repeat_data_num
            vec_func(repeat_data_num, dst[offset], src1[offset], src2[offset], Constant.REPEAT_LIMIT, 1, 1, 1, 8, 8, 8)
    left_repeat = repeat - loop_repeat_cnt * Constant.REPEAT_LIMIT
    with tik_inst.if_scope(left_repeat > 0):
        offset = loop_repeat_cnt * Constant.REPEAT_LIMIT * repeat_data_num
        vec_func(repeat_data_num, dst[offset], src1[offset], src2[offset], left_repeat, 1, 1, 1, 8, 8, 8)
    with tik_inst.if_scope(repeat_tail > 0):
        offset = repeat * repeat_data_num
        vec_func(repeat_tail, dst[offset], src1[offset], src2[offset], 1, 1, 1, 1, 8, 8, 8)


def _vector_dup_func(tik_inst, dst, scalar, data_len):
    """
    do vec_dup
    """
    data_one_block = _get_element_cnt_one_block(dst.dtype)
    repeat_data_num = 8 * data_one_block
    repeat = data_len // repeat_data_num
    repeat_tail = data_len % repeat_data_num
    loop_repeat_cnt = repeat // Constant.REPEAT_LIMIT

    with tik_inst.if_scope(repeat >= Constant.REPEAT_LIMIT):
        with tik_inst.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
            offset = repeat_lp_cnt * Constant.REPEAT_LIMIT * repeat_data_num
            tik_inst.vec_dup(repeat_data_num, dst[offset], scalar, Constant.REPEAT_LIMIT, 8)
    left_repeat = repeat - loop_repeat_cnt * Constant.REPEAT_LIMIT
    with tik_inst.if_scope(left_repeat > 0):
        offset = loop_repeat_cnt * Constant.REPEAT_LIMIT * repeat_data_num
        tik_inst.vec_dup(repeat_data_num, dst[offset], scalar, left_repeat, 8)
    with tik_inst.if_scope(repeat_tail > 0):
        offset = repeat * repeat_data_num
        tik_inst.vec_dup(repeat_tail, dst[offset], scalar, 1, 8)


# 'pylint:disable=too-many-locals
def _vconv_func(tik_inst, dst, src, round_mode, data_len):
    dst_data_one_block = _get_element_cnt_one_block(dst.dtype)
    src_data_one_block = _get_element_cnt_one_block(src.dtype)
    data_one_block = dst_data_one_block if dst_data_one_block <= src_data_one_block else src_data_one_block
    repeat_data_num = 8 * data_one_block
    dst_rep_stride = repeat_data_num // dst_data_one_block
    src_rep_stride = repeat_data_num // src_data_one_block
    repeat = data_len // repeat_data_num
    repeat_tail = data_len % repeat_data_num
    loop_repeat_cnt = repeat // Constant.REPEAT_LIMIT
    deq_scalar = 1.0 if src.dtype == "int32" and dst.dtype == "float16" else None

    with tik_inst.if_scope(repeat >= Constant.REPEAT_LIMIT):
        with tik_inst.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
            offset = repeat_lp_cnt * Constant.REPEAT_LIMIT * repeat_data_num
            tik_inst.vconv(repeat_data_num, round_mode, dst[offset], src[offset], Constant.REPEAT_LIMIT, 1, 1,
                           dst_rep_stride, src_rep_stride, deq_scalar)
    left_repeat = repeat - loop_repeat_cnt * Constant.REPEAT_LIMIT
    with tik_inst.if_scope(left_repeat > 0):
        offset = loop_repeat_cnt * Constant.REPEAT_LIMIT * repeat_data_num
        tik_inst.vconv(repeat_data_num, round_mode, dst[offset], src[offset], left_repeat, 1, 1, dst_rep_stride,
                       src_rep_stride, deq_scalar)
    with tik_inst.if_scope(repeat_tail > 0):
        offset = repeat * repeat_data_num
        tik_inst.vconv(repeat_tail, round_mode, dst[offset], src[offset], 1, 1, 1, dst_rep_stride, src_rep_stride,
                       deq_scalar)


def _data_move(tik_inst, dst, src, data_len, support_move_align):
    if support_move_align:
        data_size = data_len * _get_data_size(src.dtype)
        tik_inst.data_move_pad(dst, src, 1, data_size, 0, 0)
    else:
        element_one_block = _get_element_cnt_one_block(src.dtype)
        tik_inst.data_move(dst, src, 0, 1, _ceil_div(data_len, element_one_block), 0, 0)


class IsNan:
    """
    is_nan: Test element-wise for NaN and return result as a boolean array

    calculation process:
    0. init stage
        change input dtype to "int" by force in init stage.  "int" is good for doing AND, ADD operation.
        set inf_num = 0x7c00(FP16) 0x7f800000(FP32)
        mov(data_in_ub, data_in_gm)  # NEW MEMORY_1

    1. ignore sign bit to get uint value
        sign_mask = 2**15-1(FP16)  2**31-1(FP32)
        vec_dup(cache_ub, sign_mask)  # NEW MEMORY_2
        # Because vand only supports 16-bit, 32-bit data should be read as "int16" in FP32 case in cache_ub memory.
        vand(data_in_ub, data_in_ub, cache_ub) 

    2. compare the uint value with inf_num(0x7c00 / 0x7f800000)
        vadds(cache_ub, data_in_ub, -inf_num) to get 3 kinds of results: >0(nan), =0(inf), <0(finite)
        # pay attention to nan case, and binarize the result to 0_1: 0 means result<=0 and 1 means result>0
        vmins(cache_ub, cache_ub, 1)
        vmaxs(cache_ub, cache_ub, 0)

    3. convert 0_1 results to int8
        vconv(data_in_ub, cache_ub)   int16->float16 in FP16 case;  int32->float16 in FP32 case
        vconv(data_out_ub, data_in_ub) float16->int8; data_out_ub, alias cache_ub, share the same memory start address.
        mov(data_out_gm, data_out_ub)
    
    UB allocation: 
        as shown above, there are 3 ub related vars: data_in_ub, cache_ub, data_out_ub, and only 2 vars are used at the
        same time, no var dependencies. So it's reasonable to divide ub into two parts.
    """

    def __init__(self, input_x, output_y, kernel_name="is_nan"):
        self.tik_inst = tik.Tik()
        self.input_ub = None
        self.cache_ub = None
        self._init_inner_params(input_x, output_y, kernel_name)
        self._init_gm()
        self._init_tiling_params()
        self.support_move_align = tbe_platform.api_check_support("tik.data_move_pad")

    def build(self):
        """
        build cce
        """
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.data_input],
                               outputs=[self.data_out],
                               flowtable=[self.data_tiling],
                               config=opt_config)
        return self.tik_inst

    def compute(self):
        """
        do is_nan
        """
        _data_move(self.tik_inst, self.tiling_ub, self.data_tiling, Constant.TILING_CTRL_PARAM[1], False)
        self.need_core_num.set_as(self.tiling_ub[1])
        with self.tik_inst.for_range(0, self.need_core_num, block_num=self.need_core_num) as block_idx:
            self._init_ub()
            self._get_tiling_params(block_idx)
            core_offset = block_idx * self.per_core_size
            self._schedule(core_offset, self.core_loop_cnt, self.core_left_size)

    def _init_inner_params(self, input_x, output_y, kernel_name):
        self.kernel_name = kernel_name
        self.input_dtype = input_x.get("dtype").lower()
        self.output_dtype = output_y.get("dtype").lower()

        if self.output_dtype == "bool":
            self.output_dtype = "int8"

        _check_input_params(self.input_dtype, self.output_dtype)
        constant_map = Constant.CONSTANT_MAP.get(self.input_dtype)
        self.inf_num = constant_map.get("inf_num")
        self.sign_mask = constant_map.get("sign_mask")
        self.flag_tensor_multiple = constant_map.get("flag_tensor_multiple")
        self.input_dtype = "int" + input_x.get("dtype")[-2:]
        self.per_loop_size = _get_ub_max_size(self.input_dtype, self.output_dtype)

    def _init_gm(self):
        self.data_input = self.tik_inst.Tensor(self.input_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm,
                                               "data_input")
        self.data_out = self.tik_inst.Tensor(self.output_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_out")
        self.data_tiling = self.tik_inst.Tensor(Constant.TILING_CTRL_PARAM[0], (Constant.TILING_CTRL_PARAM[1],),
                                                tik.scope_gm, "data_tiling")

    def _init_ub(self):
        ub_max_size = _get_max_element_in_ub(self.input_dtype, 2)
        self.input_ub = self.tik_inst.Tensor(self.input_dtype, (ub_max_size,), tik.scope_ubuf, "input_ub")
        self.cache_ub = self.tik_inst.Tensor(self.input_dtype, (ub_max_size,), tik.scope_ubuf, "cache_ub")

    def _init_tiling_params(self):
        self.tiling_ub = self.tik_inst.Tensor(Constant.TILING_CTRL_PARAM[0], (Constant.TILING_CTRL_PARAM[1],),
                                              tik.scope_ubuf, "tiling_ub")
        self.real_core_num = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "real_core_num")
        self.need_core_num = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "need_core_num")
        self.total_element_size = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "total_element_size")
        self.per_core_size = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "per_core_size")
        self.core_size = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "core_size")
        self.core_loop_cnt = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "core_loop_cnt")
        self.core_left_size = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "core_left_size")
        self.real_per_loop_size = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "per_loop_size")

    def _get_tiling_params(self, block_idx):
        self.total_element_size.set_as(self.tiling_ub[2])
        self.per_core_size.set_as(self.tiling_ub[3])

        with self.tik_inst.if_scope(tik.all(block_idx == self.need_core_num - 1, self.tiling_ub[6] != 0)):
            self.core_size.set_as(self.tiling_ub[6])
            self.core_loop_cnt.set_as(self.tiling_ub[7])
            self.core_left_size.set_as(self.tiling_ub[8])
        with self.tik_inst.else_scope():
            self.core_size.set_as(self.tiling_ub[3])
            self.core_loop_cnt.set_as(self.tiling_ub[4])
            self.core_left_size.set_as(self.tiling_ub[5])

        with self.tik_inst.if_scope(self.core_loop_cnt == 0):
            self.real_per_loop_size.set_as(0)
        with self.tik_inst.else_scope():
            self.real_per_loop_size.set_as((self.core_size - self.core_left_size) // self.core_loop_cnt)

    def _inner_compute(self, offset, element_size):
        self._data_move_in(offset, element_size)
        self._ignore_sign_bit(element_size)
        self._cmp_with_inf(element_size)
        self._data_move_out(offset, element_size)

    def _schedule(self, core_offset, core_loop_cnt, core_left_size):
        with self.tik_inst.if_scope(core_loop_cnt > 0):
            with self.tik_inst.for_range(0, core_loop_cnt, thread_num=2) as lp_cnt:
                lp_offset = core_offset + lp_cnt * self.per_loop_size
                self._inner_compute(lp_offset, self.per_loop_size)
        with self.tik_inst.if_scope(core_left_size > 0):
            offset = core_offset + core_loop_cnt * self.real_per_loop_size
            self._inner_compute(offset, core_left_size)

    def _ignore_sign_bit(self, element_size):
        flag_tensor_size = element_size * self.flag_tensor_multiple
        _vector_dup_func(self.tik_inst, self.cache_ub, self.sign_mask, element_size)
        in_ub = self.input_ub.reinterpret_cast_to("int16")
        mask_ub = self.cache_ub.reinterpret_cast_to("int16")
        _vector_double_src_func(self.tik_inst, self.tik_inst.vand, in_ub, in_ub, mask_ub, flag_tensor_size, "int16")

    def _cmp_with_inf(self, element_size):
        _scalar_vector_func(self.tik_inst, self.tik_inst.vadds, self.cache_ub, self.input_ub, -self.inf_num,
                            element_size, self.input_ub.dtype)
        _scalar_vector_func(self.tik_inst, self.tik_inst.vmins, self.cache_ub, self.cache_ub, 1,
                            element_size, self.cache_ub.dtype)
        _scalar_vector_func(self.tik_inst, self.tik_inst.vmaxs, self.cache_ub, self.cache_ub, 0,
                            element_size, self.cache_ub.dtype)

    def _data_move_out(self, offset, element_size):
        result = self.input_ub.reinterpret_cast_to("float16")
        _vconv_func(self.tik_inst, result, self.cache_ub, "none", element_size)
        out_ub = self.cache_ub.reinterpret_cast_to("int8")
        _vconv_func(self.tik_inst, out_ub, result, "to-zero", element_size)
        _data_move(self.tik_inst, self.data_out[offset], out_ub, element_size, self.support_move_align)

    def _data_move_in(self, offset, element_size):
        # move input data
        _data_move(self.tik_inst, self.input_ub, self.data_input[offset], element_size, self.support_move_align)


@register_operator("IsNan")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def is_nan(input_x, output_y, kernel_name="is_nan"):
    """
    Determine the tensor float member is Nan 

    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape as input,
        and the dtype should be bool
    kernel_name : str
        cce kernel name, default value is is_nan

    Returns
    ----------
    None
    """
    is_nan_instance = IsNan(input_x, output_y, kernel_name)
    is_nan_instance.compute()
    ub_size = _get_max_element_in_ub(is_nan_instance.input_dtype, 1)
    tbe_context.get_context().add_compile_info(
        "vars", {
            "ub_size": ub_size,
            "core_num": Constant.CORE_NUM,
            "input_data_byte": common_util.get_data_size(is_nan_instance.input_dtype)
        })
    return is_nan_instance.build()
