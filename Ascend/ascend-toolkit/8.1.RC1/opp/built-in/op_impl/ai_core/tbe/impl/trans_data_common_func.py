#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

trans_data_common_func
"""


from math import gcd as func_gcd
from functools import reduce as func_reduce
from impl.util.platform_adapter import tbe_platform


CORE_UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
# AICORE count
CORE_DIM_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)


def get_core_num():
    return tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)


def get_ub_size():
    return tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

# C0 length except int8 and uint8
C0_16 = 16
# C0 length int8 and uint8
C0_32 = 32
# Ni length
NI_16 = 16
# bytes in one block
BLOCK_BYTE_SIZE = 32
# repeat up limit for vector command
REPEAT_LIMIT_VECT = 255
# repeat up limit for mte
REPEAT_LIMIT_MTE = 4095
# strides up limit for mte
STRIDE_LIMIT_MTE = 65535
# mask value for float32
MASK_64 = 64
# mask value for float16
MASK_128 = 128
# max int64 value
MAX_INT64_VALUE = 2 ** 62 - 1
# used for vnchwconv
ADDR_IDX_LIST = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
# used for scalar
REG_IDX_LIST = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
# vnchwconv line count
VNC_LINES = 16
TILING_CTRL_PARAM = ("int64", 64)
SAVE_UB = 256


def ceil_div(value_x, value_y):
    """
    do ceil division

    Parameters
    ----------
    value_x : int
        dividend
    value_y : int
        divider

    Returns
    -------
    the ceiling of value_x as an Integral
    """

    result = (value_x + value_y - 1) // value_y

    return result


def ceil_fill(value_x, value_y):
    """
    do ceiling product

    Parameters
    ----------
    value_x : int
        dividend
    value_y : int
        divider

    Returns
    -------
    the ceiling product of value_x and value_y as an Integral
    """

    result = (value_x + value_y - 1) // value_y * value_y

    return result


def floor_div(value_x, value_y):
    """
    do floor division

    Parameters
    ----------
    value_x : int
        dividend
    value_y : int
        divider

    Returns
    -------
    the floor of value_x as an Integral
    """

    result = value_x // value_y

    return result


def lcm(value_x, value_y):
    """
    get lcm of value_x and value_y

    Parameters
    ----------
    value_x : int
        one integral input
    value_y : int
        another integral input

    Returns
    -------
    the lcm of value_x and value_y as an Integral
    """

    result = value_x * value_y // func_gcd(value_x, value_y)

    return result


def get_c0_len(dtype):
    """
    get c0 length according to dtype

    Parameters
    ----------
    dtype : str
        data type name

    Returns
    -------
    the c0 length of dtype as an Integral
    """

    c0_len = C0_32 if dtype.lower() in ("int8", "uint8") else C0_16

    return c0_len


def clean_ubuf(tik_inst, src, src_offset, dup_len):
    """
    set ubuf to zero

    Parameters
    ----------
    tik_inst : tik instance
        used to generate cce code
    src : tensor
        ub tensor
    src_offset: int
        ub offset
    dup_len: int
        ub length to be cleaned

    Returns
    -------
    None
    """

    dtype = src.dtype.lower()
    dtype_factor = 1
    if dtype in ("float16", "int16", "uint16"):
        dtype_factor = 2
    elif dtype in ("float32", "int32", "uint32"):
        dtype_factor = 1
    batch_size = MASK_64

    with tik_inst.new_stmt_scope():
        dup_len_reg = tik_inst.Scalar()
        dup_len_reg.set_as(dup_len)

        with tik_inst.if_scope(dup_len_reg > 0):
            repeat = dup_len_reg // (batch_size * dtype_factor)
            left_elem = dup_len_reg % (batch_size * dtype_factor)
            repeat_loop = repeat // REPEAT_LIMIT_VECT
            repeat_left = repeat % REPEAT_LIMIT_VECT
            dup_value = tik_inst.Scalar(dtype=dtype)
            dup_value.set_as(0)

            with tik_inst.if_scope(repeat_loop > 0):
                with tik_inst.for_range(0, repeat_loop) as rpt_idx:
                    tik_inst.vector_dup(MASK_64 * dtype_factor,
                                        src[src_offset + rpt_idx * REPEAT_LIMIT_VECT * batch_size * dtype_factor],
                                        dup_value, REPEAT_LIMIT_VECT, 1, 8)

            with tik_inst.if_scope(repeat_left > 0):
                tik_inst.vector_dup(MASK_64 * dtype_factor,
                                    src[src_offset + repeat_loop * REPEAT_LIMIT_VECT * batch_size * dtype_factor],
                                    dup_value, repeat_left, 1, 8)

            with tik_inst.if_scope(left_elem > 0):
                tik_inst.vector_dup(left_elem, src[src_offset + repeat * batch_size * dtype_factor],
                                    dup_value, 1, 1, 8)


def get_shape_size(sub_shape):
    """
    return shape size

    Parameters
    ----------
    sub_shape : tuple/list
        input tuple or list

    Returns
    -------
    the product of all values in sub_shape as an Integral
    """

    shape_size = func_reduce(lambda x, y: x * y, sub_shape)

    return shape_size


def get_dtype_len(in_dtype):
    """
    get the byte count of certain dtype

    Parameters
    ----------
    in_dtype : str
        data type name

    Returns
    -------
    the block count of in_dtype as an Integral
    """

    temp_dtype = in_dtype.lower()

    if temp_dtype in ("int8", "uint8"):
        byte_len = 1
    elif temp_dtype in ("float16", "int16", "uint16"):
        byte_len = 2
    elif temp_dtype in ("float32", "int32", "uint32"):
        byte_len = 4
    elif temp_dtype in ("int64", "uint64"):
        byte_len = 8

    return byte_len


def get_max_element_in_ub(in_dtype, ub_part, deduct_size=1024):
    """
    get the up limit elements in one part of UB

    Parameters
    ----------
    in_dtype : str
        data type name
    ub_part : int
        count of UB will be split
    deduct_size: int
        count of UB will be deducted

    Returns
    -------
    the up limit elements in one part of UB as an Integral
    """

    vector_ub_size = get_ub_size()
    if ub_part == 0:
        return vector_ub_size

    byte_len = get_dtype_len(in_dtype)
    max_ub_size = 256 * 1024  # the unit is Byte
    if vector_ub_size >= max_ub_size:
        ub_upper_limit = (max_ub_size - deduct_size) // ub_part
    else:
        ub_upper_limit = (vector_ub_size - deduct_size) // ub_part
    element_size = ub_upper_limit // byte_len

    return element_size


def get_dtype_factor(dtype):
    """
    return 2 for float32, 1 for float16

    Parameters
    ----------
    dtype : str

    Returns
    -------
    the data type factor as an Integral
    """

    size_factor = 2 if dtype.lower() in ("float32", "int32", "uint32") else 1

    return size_factor


def get_tiling_params(tik_inst, tiling_ub, tiling_gm, tiling_params, tiling_dtype_bytes):
    """
    get tiling parameters

    Parameters
    ----------
    tik_inst : tik instance
    tiling_ub: ub tensor
    tiling_gm: gm tensor
    tiling_params: list of tiling parameters
    tiling_dtype_bytes: bytes of dtype

    Returns
    -------
    None
    """

    ele_per_block = BLOCK_BYTE_SIZE // tiling_dtype_bytes
    tik_inst.data_move(tiling_ub, tiling_gm, 0, 1, TILING_CTRL_PARAM[1] // ele_per_block, 0, 0)
    for idx, reg in enumerate(tiling_params):
        reg.set_as(tiling_ub[idx])
