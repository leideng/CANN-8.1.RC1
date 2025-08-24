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
common_util
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tik
from impl import constant_util as constant
from impl.util.platform_adapter import tvm


def get_vector_repeat_times(tik_instance, total_size):
    """
    get vector instruct repeat times

    Parameters
    ----------
    tik_instance: tik_instance
    total_size: the byte of the data

    Returns
    -------
    repeats: repeat times of vector instructs
    """
    repeats = tik_instance.Scalar(constant.DATA_TYPE_INT32)
    repeats.set_as(total_size % constant.VECTOR_BYTE_SIZE)
    with tik_instance.if_scope(repeats == 0):
        repeats.set_as(total_size // constant.VECTOR_BYTE_SIZE)
    with tik_instance.else_scope():
        repeats.set_as(total_size // constant.VECTOR_BYTE_SIZE + 1)

    return repeats


def get_datamove_nburst(tik_instance, total_size):
    """
    get datamove nburst

    Parameters
    ----------
    tik_instance: tik_instance
    total_size: the byte of the data

    Returns
    -------
    nburst: one burst indicating the continueous transfer length
                     in terms of the block size.
    """
    nburst = tik_instance.Scalar(constant.DATA_TYPE_INT32)
    nburst.set_as(total_size % constant.BLOCK_SIZE)
    with tik_instance.if_scope(nburst == 0):
        nburst.set_as(total_size / constant.BLOCK_SIZE)
    with tik_instance.else_scope():
        nburst.set_as(total_size / constant.BLOCK_SIZE + 1)

    return nburst


def conv_s4_to_s8(tik_instance, dst_ub, src_ub, number, mode=''):
    """
    convert bfloat16 to float32

    Parameters
    ----------
    tik_instance: tik_instance
    dst_ub_fp32: the float32 destination ub, The user needs to ensure that the ub is sufficient
                 ub size must be 8*block_size bytes aligned
    src_ub_bfp16: the bfloat16 source ub, The user needs to ensure that the ub is sufficient
                  ub size must be 8*block_size bytes aligned
    number: the number of elements

    Returns
    -------
    
    """
    src_dtype = src_ub.dtype
    dst_dtype = dst_ub.dtype
    if src_dtype in (
            "float16",
            "bfloat16",
    ) and dst_dtype in (
            "float32",
            "int32",
    ):
        loop = number // (constant.MASK64 * constant.MAX_REPEAT_TIMES)
        with tik_instance.if_scope(loop > 0):
            with tik_instance.for_range(0, loop) as index:
                compute_offset = constant.MASK64 * constant.MAX_REPEAT_TIMES * index
                tik_instance.vec_conv(constant.MASK64, mode, dst_ub[compute_offset], src_ub[compute_offset],
                                      constant.MAX_REPEAT_TIMES, constant.REPEAT_STRIDE_EIGHT,
                                      constant.REPEAT_STRIDE_FOUR)

        compute_offset = constant.MASK64 * constant.MAX_REPEAT_TIMES * loop
        repeat_time = number % (constant.MASK64 * constant.MAX_REPEAT_TIMES) // constant.MASK64
        with tik_instance.if_scope(repeat_time > 0):
            tik_instance.vec_conv(constant.MASK64, mode, dst_ub[compute_offset], src_ub[compute_offset], repeat_time,
                                  constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)
        tail = number % constant.MASK64
        tail_offsets = compute_offset + repeat_time * constant.MASK64
        with tik_instance.if_scope(tail > 0):
            tik_instance.vec_conv(tail, mode, dst_ub[tail_offsets], src_ub[tail_offsets], 1,
                                  constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)


# 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments, too-many-lines
def conv_s8_to_s4(tik_instance, dst_ub, src_ub, number, mode='', deqscale=None):
    """
    convert float32/int32 to bfloat16/float16/int16

    Parameters
    ----------
    tik_instance: tik_instance
    dst_ub_bfp16: the bfloat16 destination ub, The user needs to ensure that the ub is sufficient
                  ub size must be 8*block_size bytes aligned
    src_ub_fp32: the float32 source ub, The user needs to ensure that the ub is sufficient
                 ub size must be 8*block_size bytes aligned
    number: the number of elements

    Returns
    -------
    nburst: one burst indicating the continueous transfer length
                     in terms of the block size.    
    """
    src_dtype = src_ub.dtype
    dst_dtype = dst_ub.dtype
    if src_dtype in (
            "float32",
            "int32",
    ) and dst_dtype in ("float16", "bfloat16", "int16"):
        loop = number // (constant.MASK64 * constant.MAX_REPEAT_TIMES)
        with tik_instance.if_scope(loop > 0):
            with tik_instance.for_range(0, loop) as index:
                compute_offset = constant.MASK64 * constant.MAX_REPEAT_TIMES * index
                tik_instance.vec_conv(constant.MASK64, mode, dst_ub[compute_offset], src_ub[compute_offset],
                                      constant.MAX_REPEAT_TIMES, constant.REPEAT_STRIDE_FOUR,
                                      constant.REPEAT_STRIDE_EIGHT, deqscale)

        compute_offset = constant.MASK64 * constant.MAX_REPEAT_TIMES * loop
        repeat_time = number % (constant.MASK64 * constant.MAX_REPEAT_TIMES) // constant.MASK64
        with tik_instance.if_scope(repeat_time > 0):
            tik_instance.vec_conv(constant.MASK64, mode, dst_ub[compute_offset], src_ub[compute_offset], repeat_time,
                                  constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT, deqscale)
        tail = number % constant.MASK64
        tail_offsets = compute_offset + repeat_time * constant.MASK64
        with tik_instance.if_scope(tail > 0):
            tik_instance.vec_conv(tail, mode, dst_ub[tail_offsets], src_ub[tail_offsets], 1,
                                  constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT, deqscale)


def conv_s4_to_i4(tik_instance, dst_ub, src_ub, number, mode=''):
    """
    convert int16 to float16

    Parameters
    ----------
    tik_instance: tik_instance
    dst_ub_int16: the int16 destination ub, The user needs to ensure that the ub is sufficient
                  ub size must be 8*block_size bytes aligned
    src_ub_fp16: the float16 source ub, The user needs to ensure that the ub is sufficient
                 ub size must be 8*block_size bytes aligned
    number: the number of elements

    Returns
    -------
    nburst: one burst indicating the continueous transfer length
                     in terms of the block size.    
    """
    src_dtype = src_ub.dtype
    dst_dtype = dst_ub.dtype
    if src_dtype in ("int16") and dst_dtype in ("float16"):
        loop = number // (constant.MASK128 * constant.MAX_REPEAT_TIMES)
        with tik_instance.if_scope(loop > 0):
            with tik_instance.for_range(0, loop) as index:
                compute_offset = constant.MASK128 * constant.MAX_REPEAT_TIMES * index
                tik_instance.vec_conv(constant.MASK128, mode, dst_ub[compute_offset], src_ub[compute_offset],
                                      constant.MAX_REPEAT_TIMES, constant.REPEAT_STRIDE_EIGHT,
                                      constant.REPEAT_STRIDE_EIGHT)

        compute_offset = constant.MASK128 * constant.MAX_REPEAT_TIMES * loop
        repeat_time = number % (constant.MASK128 * constant.MAX_REPEAT_TIMES) // constant.MASK128
        with tik_instance.if_scope(repeat_time > 0):
            tik_instance.vec_conv(constant.MASK128, mode, dst_ub[compute_offset], src_ub[compute_offset], repeat_time,
                                  constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
        tail = number % constant.MASK128
        tail_offsets = compute_offset + repeat_time * constant.MASK128
        with tik_instance.if_scope(tail > 0):
            tik_instance.vec_conv(tail, mode, dst_ub[tail_offsets], src_ub[tail_offsets], 1,
                                  constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)


def conv_f32_to_s8(tik_instance, dst_ub, src_ub, number, mode=''):
    """
    convert float32 to int32

    Parameters
    ----------
    tik_instance: tik_instance
    dst_ub_int32: the int32 destination ub, The user needs to ensure that the ub is sufficient
                  ub size must be 8*block_size bytes aligned
    src_ub_float32: the float32 source ub, The user needs to ensure that the ub is sufficient
                 ub size must be 8*block_size bytes aligned
    number: the number of elements

    Returns
    -------
    nburst: one burst indicating the continueous transfer length
                     in terms of the block size.    
    """
    src_dtype = src_ub.dtype
    dst_dtype = dst_ub.dtype
    if src_dtype in ("float32",) and dst_dtype in ("int32",):
        loop = number // (constant.MASK64 * constant.MAX_REPEAT_TIMES)
        with tik_instance.if_scope(loop > 0):
            with tik_instance.for_range(0, loop) as index:
                compute_offset = constant.MASK64 * constant.MAX_REPEAT_TIMES * index
                tik_instance.vec_conv(constant.MASK64, mode, dst_ub[compute_offset], src_ub[compute_offset],
                                      constant.MAX_REPEAT_TIMES, constant.REPEAT_STRIDE_EIGHT,
                                      constant.REPEAT_STRIDE_EIGHT)

        compute_offset = constant.MASK64 * constant.MAX_REPEAT_TIMES * loop
        repeat_time = number % (constant.MASK64 * constant.MAX_REPEAT_TIMES) // constant.MASK64
        with tik_instance.if_scope(repeat_time > 0):
            tik_instance.vec_conv(constant.MASK64, mode, dst_ub[compute_offset], src_ub[compute_offset], repeat_time,
                                  constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
        tail = number % constant.MASK64
        tail_offsets = compute_offset + repeat_time * constant.MASK64
        with tik_instance.if_scope(tail > 0):
            tik_instance.vec_conv(tail, mode, dst_ub[tail_offsets], src_ub[tail_offsets], 1,
                                  constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)


def conv_i8_to_s8(tik_instance, dst_ub, src_ub, number, mode=''):
    """
    convert int32 to float

    Parameters
    ----------
    tik_instance: tik_instance
    dst_ub_fp32: the float destination ub, The user needs to ensure that the ub is sufficient
                  ub size must be 8*block_size bytes aligned
    src_ub_int32: the int32 source ub, The user needs to ensure that the ub is sufficient
                 ub size must be 8*block_size bytes aligned
    number: the number of elements

    Returns
    -------
    nburst: one burst indicating the continueous transfer length
                     in terms of the block size.    
    """
    src_dtype = src_ub.dtype
    dst_dtype = dst_ub.dtype
    loop = number // (constant.MASK64 * constant.MAX_REPEAT_TIMES)
    with tik_instance.if_scope(loop > 0):
        with tik_instance.for_range(0, loop) as index:
            compute_offset = constant.MASK64 * constant.MAX_REPEAT_TIMES * index
            tik_instance.vec_conv(constant.MASK64, mode, dst_ub[compute_offset], src_ub[compute_offset],
                                    constant.MAX_REPEAT_TIMES, constant.REPEAT_STRIDE_EIGHT,
                                    constant.REPEAT_STRIDE_EIGHT)

    compute_offset = constant.MASK64 * constant.MAX_REPEAT_TIMES * loop
    repeat_time = number % (constant.MASK64 * constant.MAX_REPEAT_TIMES) // constant.MASK64
    with tik_instance.if_scope(repeat_time > 0):
        tik_instance.vec_conv(constant.MASK64, mode, dst_ub[compute_offset], src_ub[compute_offset], repeat_time,
                                constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
    tail = number % constant.MASK64
    tail_offsets = compute_offset + repeat_time * constant.MASK64
    with tik_instance.if_scope(tail > 0):
        tik_instance.vec_conv(tail, mode, dst_ub[tail_offsets], src_ub[tail_offsets], 1,
                                constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)


def conv_i1_to_s2(tik_instance, dst_ub, src_ub, number, mode=''):
    """
    convert int8 to float16

    Parameters
    ----------
    tik_instance: tik_instance
    dst_ub: the float16 destination ub, The user needs to ensure that the ub is sufficient
                 ub size must be 8*block_size bytes aligned
    src_ub: the int8 source ub, The user needs to ensure that the ub is sufficient
                  ub size must be 8*block_size bytes aligned
    number: the number of elements

    Returns
    -------
    
    """
    src_dtype = src_ub.dtype
    dst_dtype = dst_ub.dtype
    if src_dtype in (
            "int8",
            "uint8",
    ) and dst_dtype in ("float16",):
        loop = number // (constant.MASK128 * constant.MAX_REPEAT_TIMES)
        with tik_instance.if_scope(loop > 0):
            with tik_instance.for_range(0, loop) as index:
                compute_offset = constant.MASK128 * constant.MAX_REPEAT_TIMES * index
                tik_instance.vec_conv(constant.MASK128, mode, dst_ub[compute_offset:], src_ub[compute_offset:],
                                      constant.MAX_REPEAT_TIMES, constant.REPEAT_STRIDE_EIGHT,
                                      constant.REPEAT_STRIDE_FOUR)

        compute_offset = constant.MASK128 * constant.MAX_REPEAT_TIMES * loop
        repeat_time = number % (constant.MASK128 * constant.MAX_REPEAT_TIMES) // constant.MASK128
        with tik_instance.if_scope(repeat_time > 0):
            tik_instance.vec_conv(constant.MASK128, mode, dst_ub[compute_offset:], src_ub[compute_offset:], repeat_time,
                                  constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)
        tail = number % constant.MASK128
        tail_offsets = compute_offset + repeat_time * constant.MASK128
        with tik_instance.if_scope(tail > 0):
            tik_instance.vec_conv(tail, mode, dst_ub[tail_offsets], src_ub[tail_offsets], 1,
                                  constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)


def conv_s2_to_i1(tik_instance, dst_ub, src_ub, number, mode=''):
    """
    convert float16 to int8

    Parameters
    ----------
    tik_instance: tik_instance
    dst_ub: the int8 destination ub, The user needs to ensure that the ub is sufficient
                  ub size must be 8*block_size bytes aligned
    src_ub: the float16 source ub, The user needs to ensure that the ub is sufficient
                 ub size must be 8*block_size bytes aligned
    number: the number of elements

    Returns
    -------
    nburst: one burst indicating the continueous transfer length
                     in terms of the block size.    
    """
    src_dtype = src_ub.dtype
    dst_dtype = dst_ub.dtype
    if src_dtype in ("float16",) and dst_dtype in (
            "int8",
            "uint8",
    ):
        loop = number // (constant.MASK128 * constant.MAX_REPEAT_TIMES)
        with tik_instance.if_scope(loop > 0):
            with tik_instance.for_range(0, loop) as index:
                compute_offset = constant.MASK128 * constant.MAX_REPEAT_TIMES * index
                tik_instance.vec_conv(constant.MASK128, mode, dst_ub[compute_offset:], src_ub[compute_offset:],
                                      constant.MAX_REPEAT_TIMES, constant.REPEAT_STRIDE_FOUR,
                                      constant.REPEAT_STRIDE_EIGHT)

        compute_offset = constant.MASK128 * constant.MAX_REPEAT_TIMES * loop
        repeat_time = number % (constant.MASK128 * constant.MAX_REPEAT_TIMES) // constant.MASK128
        with tik_instance.if_scope(repeat_time > 0):
            tik_instance.vec_conv(constant.MASK128, mode, dst_ub[compute_offset:], src_ub[compute_offset:], repeat_time,
                                  constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)
        tail = number % constant.MASK128
        tail_offsets = compute_offset + repeat_time * constant.MASK128
        with tik_instance.if_scope(tail > 0):
            tik_instance.vec_conv(tail, mode, dst_ub[tail_offsets], src_ub[tail_offsets], 1,
                                  constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)


def get_data_size(datatype):
    """
    get data size unit, means one element of this input datatype takes up nbyte space

    Parameters
    ----------
    datatype: datatype supports float32,float16,int32,int16,int8,uint32,uint16,uint8

    Returns
    -------
    data_size: one element of this input datatype takes up nbyte space
    """
    datatype_map = {
        constant.DATA_TYPE_FP32: constant.DATA_SIZE_FOUR,
        constant.DATA_TYPE_FP16: constant.DATA_SIZE_TWO,
        constant.DATA_TYPE_INT32: constant.DATA_SIZE_FOUR,
        constant.DATA_TYPE_INT16: constant.DATA_SIZE_TWO,
        constant.DATA_TYPE_INT8: constant.DATA_SIZE_ONE,
        constant.DATA_TYPE_UINT32: constant.DATA_SIZE_FOUR,
        constant.DATA_TYPE_UINT16: constant.DATA_SIZE_TWO,
        constant.DATA_TYPE_UINT8: constant.DATA_SIZE_ONE,
        constant.DATA_TYPE_UINT64: constant.DATA_SIZE_EIGHT,
        constant.DATA_TYPE_INT64: constant.DATA_SIZE_EIGHT
    }
    data_size = datatype_map.get(datatype)
    if data_size is None:
        raise RuntimeError("datatype %s is not support!" % (datatype))

    return data_size


def move_out_non32_alignment(input_dict):
    """
  move data from ub to gm when non32 alignment
  usage scenarios: multi core moves out of the scene for the last time,
  in order to prevent covering the data of other core

  Parameters
  ----------
    input_dict: input_dict is a dict, the keys as follow:
            instance: tik instance
            out_ub: a ub tensor
            out_gm: a gm tensor
            gm_offset: a scalar,gm offset
            element_num: element number
            dsize: data size of each type,fp32,data size is 4,
                   fp16 data size is 2 and so on
  Returns
  -------
  None
  """
    instance = input_dict.get("instance")
    out_ub = input_dict.get("out_ub")
    out_gm = input_dict.get("out_gm")
    gm_offset = input_dict.get("gm_offset")
    element_num = input_dict.get("element_num")
    dsize = input_dict.get("dsize")
    each_burst_num = constant.BLOCK_SIZE // dsize
    out_ub_tmp = instance.Tensor(out_ub.dtype, (each_burst_num,), name="out_ub_tmp", scope=tik.scope_ubuf)
    nbursts = instance.Scalar("int32")
    nbursts.set_as((element_num * dsize) // constant.BLOCK_SIZE)
    scalar = instance.Scalar(out_ub.dtype)
    mod = instance.Scalar("int32")
    mod.set_as((element_num * dsize) % constant.BLOCK_SIZE)

    # 32b alignment
    with instance.if_scope(mod == 0):
        instance.data_move(out_gm[gm_offset], out_ub, constant.SID, constant.DEFAULT_NBURST, nbursts,
                           constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    # less than 32b
    with instance.if_scope(nbursts == 0):
        offset = each_burst_num - element_num
        instance.data_move(out_ub_tmp, out_gm[gm_offset - offset], constant.SID, constant.DEFAULT_NBURST,
                           constant.DEFAULT_BURST_LEN, constant.STRIDE_ZERO, constant.STRIDE_ZERO)

        with instance.for_range(0, element_num) as out_cycle:
            scalar.set_as(out_ub[out_cycle])
            out_ub_tmp[offset + out_cycle].set_as(scalar)
        instance.data_move(out_gm[gm_offset - offset], out_ub_tmp, constant.SID, constant.DEFAULT_NBURST,
                           constant.DEFAULT_BURST_LEN, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
    # bigger than 32b
    with instance.else_scope():
        instance.data_move(out_gm[gm_offset], out_ub, constant.SID, constant.DEFAULT_NBURST, nbursts,
                           constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        offset = element_num - each_burst_num
        scalar = instance.Scalar(out_ub.dtype)
        with instance.for_range(0, each_burst_num) as time:
            scalar.set_as(out_ub[offset + time])
            out_ub[time].set_as(scalar)
        instance.data_move(out_gm[gm_offset + offset], out_ub, constant.SID, constant.DEFAULT_NBURST,
                           constant.DEFAULT_BURST_LEN, constant.STRIDE_ZERO, constant.STRIDE_ZERO)


def get_block_element(datatype):
    """
    get the count of element that one block has.
    :param datatype: data type
    :return: count
    """
    data_type_size = get_data_size(datatype)
    return constant.BLOCK_SIZE // data_type_size


def get_attr(attr_value, attr_name, dtype_compute, ir_dtype):
    """
    get the attr

    Parameters
    ----------
    attr_value: value of attr
    attr_name: name of attr
    dtype_compute: is the dtype used for calculation
    ir_dtype: the type of attr is ir

    Returns
    -------
    attr_var
    """
    if attr_value is None:
        attr_dtype = {"src_dtype": ir_dtype}
        attr_var = tbe.var_attr(attr_name, dtype=dtype_compute, addition=attr_dtype)
    else:
        attr_var = tvm.const(attr_value, dtype_compute)
    return attr_var


def get_vlrelu(x, attr_value, attr_name, attr_dtype):
    """
    get vlrelu

    Parameters
    ----------
    x: x tensor
    attr: value of attr
    attr_name: name of attr
    attr_dtype: dtype of attr

    Returns
    -------
    res_vlrelu, attr_value
    """
    if attr_value is None:
        dtype = x.dtype
        scalar = tvm.const(0, dtype)
        tmp_max_x = tbe.vmaxs(x, scalar)
        tmp_min_x = tbe.vmins(x, scalar)
        attr_value = get_attr(attr_value, attr_name, dtype, attr_dtype)
        tmp_mul_x = tbe.vmuls(tmp_min_x, attr_value)
        res_vlrelu = tbe.vadd(tmp_max_x, tmp_mul_x)
    else:
        res_vlrelu = tbe.vlrelu(x, attr_value)
    return res_vlrelu, attr_value


def get_dtype(tensor_dict):
    """
    get tensor_dict dtype

    Parameters
    ----------
    tensor_dict: input or output

    Return
    -------
    dtype of tensor
    """
    get_dict_dtype = tensor_dict.get("dtype").lower()
    tensor_dtype = "int8" if get_dict_dtype == "bool" else get_dict_dtype

    return tensor_dtype


def get_esp_min(dtype):
    """
    get esp_min

    Parameters
    ----------
    dtype: dtype

    Return
    -------
    esp_min
    """
    if dtype == "float16":
        esp_min = tvm.const(1.18e-7, dtype)
    else:
        esp_min = tvm.const(1.18e-38, dtype)

    return esp_min
