#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

strided slice
"""

from __future__ import absolute_import
from __future__ import with_statement
import math

from impl import common_util
from impl import constant_util as constant
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.util_tik_comm_func import ceil_align
from impl.util.util_tik_comm_func import ceil_div
from impl.util.util_tik_comm_func import floor_align


def data_copy(tik_instance: tik.Tik, dst: tik.Tensor, src: tik.Tensor, count, burst=None):
    repeat = 1
    src_stride = 0
    dst_stride = 0
    dtype_size = common_util.get_data_size(src.dtype)
    if tbe_platform.api_check_support("tik.data_move_pad"):
        dest_b8 = dst.reinterpret_cast_to("int8")
        src_b8 = src.reinterpret_cast_to("int8")
        tik_instance.data_move_pad(dest_b8, src_b8, repeat, dtype_size * count, dst_stride, src_stride)
    else:
        block_element = constant.BLOCK_SIZE // dtype_size
        if burst is None:
            burst = ceil_div(count, block_element)
        tik_instance.data_move(dst, src, 0, repeat, burst, src_stride, dst_stride)


def gm2ub(tik_instance: tik.Tik, dst: tik.Tensor, src: tik.Tensor, count, burst=None):
    """
    move data from gm to ub
    :param tik_instance: tik instance
    :param dst: dst ub
    :param src: src gm
    :param count: count to move
    :param burst: burst to move, if is None, burst=ceil(count / block_element), by default None
    :param repeat: repeat to move
    :param src_stride: src_stride to move
    :param dst_stride: dst_stride to move
    :return: None
    """
    if dst.scope != tik.scope_ubuf:
        raise RuntimeError("dst must be UB, but dst is {}.".format(dst.scope))

    if src.scope != tik.scope_gm:
        raise RuntimeError("src must be global, but src is {}.".format(src.scope))

    if dst.dtype != src.dtype:
        raise RuntimeError("dst.dtype[{}] != src.dtype[{}].".format(dst.dtype, src.dtype))

    data_copy(tik_instance, dst, src, count, burst)


def ub2gm(tik_instance: tik.Tik, dst: tik.Tensor, src: tik.Tensor, count, burst=None):
    """
    move data from ub to gm
    :param tik_instance: tik instance
    :param dst: dst gm
    :param src: src ub
    :param count: count to move
    :param burst: burst to move, if is None, burst=ceil(count / block_element), by default None
    :return: None
    """
    if dst.scope != tik.scope_gm:
        raise RuntimeError("dst must be global, but dst is {}.".format(dst.scope))

    if src.scope != tik.scope_ubuf:
        raise RuntimeError("src must be UB, but src is {}.".format(src.scope))

    if dst.dtype != src.dtype:
        raise RuntimeError("dst.dtype[{}] != src.dtype[{}].".format(dst.dtype, src.dtype))

    data_copy(tik_instance, dst, src, count, burst)


def ub2ub(tik_instance: tik.Tik, dst: tik.Tensor, src: tik.Tensor, copy_params):
    """
    move data from ub to ub
    :param tik_instance: tik instance
    :param dst: dst ub
    :param src: src ub
    :param sid: sid
    :param nburst: nburst
    :param burst: burst
    :param src_stride: src gap
    :param dst_stride: dst gap
    :return:
    """
    if dst.scope != tik.scope_ubuf:
        raise RuntimeError("dst must be UB, but dst is {}.".format(dst.scope))

    if src.scope != tik.scope_ubuf:
        raise RuntimeError("src must be UB, but src is {}.".format(src.scope))

    sid, nburst, burst, src_stride, dst_stride = copy_params
    tik_instance.data_move(dst, src, sid, nburst, burst, src_stride, dst_stride)


# the max value of repeat_times in vnchwconv
VNCHW_REPEAT_TIMES = 255
# max value of nburst in data_move
N_BURST = 4095
# max value of src_stride/dst_stride in data_move
MAX_STRIDE = 65535


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    MAX_SIZE = 2**31 - 1
    MAX_INPUT_SIZE = MAX_SIZE
    MAX_OUTPUT_SIZE = MAX_SIZE
    MAX_NBURST = 4095
    MAX_REPEAT = 255
    Tiling_UB_SIZE = 382


class TilingMode:
    """
    The class for tiling mode
    """
    ONLY_LAST_STRIDE_LARGER1_OUT_CONTINUAL_VNCHWCONV = 9
    LAST_STRIDE_LARGER1_SMALL_INOUT_INNER_VNCHWCONV = 10
    LAST_STRIDE_LARGER1_LE_BLOCK_ELE_LARGE_OUT_INNER_VNCHWCONV = 11
    LAST_STRIDE_LARGER1_FUNCTIONAL = 12
    LAST_STRIDE_LARGER_BLOCK_ELE_LARGE_OUT_INNER_VNCHWCONV = 13
    LAST_STRIDE_32B_ALIGN_LARGE_OUT_INNER_VNCHWCONV = 14
    LAST_STRIDE_32B_ALIGN_SMALL_OUT_INNER_VNCHWCONV = 15
    LAST_STRIDE_LARGER1_NOT_32B_ALIGN_SMALL_OUT_INNER_VNCHWCONV = 16
    NOT_LAST_STRIDE_LARGER1_FUNCTIONAL = 17
    NOT_LAST_STRIDE_LARGER1_SMALL_OUT_INNER_VNCHWCONV = 18


def change_dtype(dtype):
    """
    change dtype for others
    """
    if dtype == "bool":
        return "int8", "int8"
    if dtype == "bfloat16":
        return "float16", "float16"
    if dtype == "complex32":
        return "int32", "int32"
    if dtype == "complex64":
        return "int64", "int64"
    return dtype, dtype


# 'pylint: disable=too-many-arguments,unused-argument
def check_supported(input_x,
                    begin,
                    end,
                    strides=None,
                    output_x=None,
                    begin_mask=0,
                    end_mask=0,
                    ellipsis_mask=0,
                    new_axis_mask=0,
                    shrink_axis_mask=0,
                    kernel_name="strided_slice"):
    """
    check_supported
    """
    strides_value = strides.get("const_value")
    strides_const_value_range = strides.get("const_value_range")
    if strides_const_value_range:
        for i in strides_const_value_range:
            if i[0] <= 1 <= i[1]:
                continue
            return False, "strides value range:{} has not 1 value.".format(str(i))
        return "Unknown"

    if not strides_value:
        return "Unknown"

    def is_all_ge_one():
        for stride in strides_value:
            if stride < 1:
                return False
        return True

    if is_all_ge_one():
        return True, ""
    return False, "strides:{} has not 1 value.".format(str(strides_value))


def ceil_32bytes_align_count(count, dtype):
    """
    ceil_32bytes_align_count
    """
    type_size = common_util.get_data_size(dtype)
    block_count = math.ceil(count * type_size / constant.BLOCK_SIZE)
    return block_count * constant.BLOCK_SIZE // type_size


# 'pylint: disable=too-many-locals, too-many-statements, too-many-instance-attributes
# 'pylint: disable=too-few-public-methods
class StridedSlice:
    """
    StridedSlice
    """

    # 'pylint: disable=too-many-locals, too-many-statements, too-many-instance-attributes
    class TilingParam:
        """
        TilingParam
        """

        def __init__(self, input_x_shape, inst: tik.Tik):
            """
            tiling param
            :param input_x_shape: input shape
            :param inst: tik instance
            """
            self.tik_instance = inst
            dtype = "int64"
            self.dtype = dtype
            # mode_type, shape_length, input_shape, output_shape, begin, end, stride
            tiling_gm_size = 3 + len(input_x_shape) * 5
            tiling_gm_size = ceil_32bytes_align_count(tiling_gm_size, self.dtype)
            self.tiling_gm = inst.Tensor(dtype, (tiling_gm_size,), name="tiling_gm", scope=tik.scope_gm)

            self.input_shape = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="input_dims")
            self.begin = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="begin_dims")
            self.end = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="end_dims")
            self.stride = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="stride_dims")
            self.output_shape = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="output_shape_dims")
            self.input_steps = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="input_steps")
            self.output_steps = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="output_steps")

            self.shape_length = inst.Scalar(dtype, name="shape_length", init_value=len(input_x_shape))
            self.tiling_mode = inst.Scalar(dtype, name="tiling_mode")
            self.out_dim = inst.Scalar(dtype, name="out_dim")
            self.out_dim_with_vnchwconv = inst.Scalar(dtype, name="out_dim_with_vnchwconv")
            self.core_num = inst.Scalar(dtype, name="core_num")

        # 'pylint: disable=invalid-name
        def init(self):
            """
            init process data
            """
            with self.tik_instance.new_stmt_scope():
                need_ub_size = self.tiling_gm.shape[0]
                tiling_ub = self.tik_instance.Tensor(self.dtype, (need_ub_size,),
                                                     name="tiling_ub",
                                                     scope=tik.scope_ubuf)
                gm2ub(self.tik_instance, tiling_ub, self.tiling_gm, need_ub_size)

                self.tiling_mode.set_as(tiling_ub[0])
                self.shape_length.set_as(tiling_ub[1])
                index = self.tik_instance.Scalar(init_value=2)
                items = (self.input_shape, self.output_shape, self.begin, self.end, self.stride)
                for item in items:
                    with self.tik_instance.for_range(0, self.shape_length) as dim_idx:
                        item[dim_idx].set_as(tiling_ub[index])
                        index.set_as(index + 1)
                self.core_num.set_as(tiling_ub[index])
            self.out_dim.set_as(1)
            self.out_dim_with_vnchwconv.set_as(1)
            with self.tik_instance.for_range(0, self.shape_length) as index:
                dim = self.output_shape[index]
                with self.tik_instance.if_scope(index < self.shape_length - 1):
                    self.out_dim.set_as(self.out_dim * dim)
                with self.tik_instance.if_scope(index < self.shape_length - 2):
                    self.out_dim_with_vnchwconv.set_as(self.out_dim_with_vnchwconv * dim)

            with self.tik_instance.for_range(0, self.shape_length) as index:
                dim_idx = self.shape_length - 1 - index
                self.input_steps[index].set_as(self.input_shape[dim_idx])
                self.output_steps[index].set_as(self.output_shape[dim_idx])
                with self.tik_instance.if_scope(index > 0):
                    self.input_steps[index].set_as(self.input_steps[index] * self.input_steps[index - 1])
                    self.output_steps[index].set_as(self.output_steps[index] * self.output_steps[index - 1])

    # 'pylint: disable=locally-disabled,too-many-arguments,
    # 'pylint: disable=unused-argument,too-many-locals
    def __init__(self,
                 input_x,
                 strides,
                 begin_mask,
                 end_mask,
                 ellipsis_mask,
                 new_axis_mask,
                 shrink_axis_mask,
                 kernel_name="strided_slice"):
        self.strides = strides
        self.begin_mask = begin_mask
        self.end_mask = end_mask
        self.ellipsis_mask = ellipsis_mask
        self.new_axis_mask = new_axis_mask
        self.shrink_axis_mask = shrink_axis_mask
        self.kernel_name = kernel_name

        inst = tik.Tik()
        self.tik_instance = inst
        self.tik_profiling = tik.Dprofile()
        max_dim_supported = 8
        self.tiling_param = self.TilingParam([1] * max_dim_supported, inst)
        self.shape_length = self.tiling_param.shape_length
        self.input_shape = self.tiling_param.input_shape
        self.output_shape = self.tiling_param.output_shape
        self.begin = self.tiling_param.begin
        self.end = self.tiling_param.end
        self.strides = self.tiling_param.stride
        self.out_dims = self.tiling_param.out_dim
        self.input_inner_dim = self.input_shape[self.shape_length - 1]
        self.output_inner_dim = self.output_shape[self.shape_length - 1]
        self.dtype = input_x.get("dtype").lower()
        self.dtype, input_x["dtype"] = change_dtype(self.dtype)
        self.dtype_size = common_util.get_data_size(self.dtype)
        self.input_gm = inst.Tensor(self.dtype, (Constant.MAX_INPUT_SIZE,), name="input_gm", scope=tik.scope_gm)
        self.begin_gm = inst.Tensor(self.dtype, (Constant.MAX_SIZE,), name="begin_gm", scope=tik.scope_gm)
        self.end_gm = inst.Tensor(self.dtype, (Constant.MAX_SIZE,), name="end_gm", scope=tik.scope_gm)
        self.strides_gm = inst.Tensor(self.dtype, (Constant.MAX_SIZE,), name="strides_gm", scope=tik.scope_gm)
        self.output_gm = inst.Tensor(self.dtype, (Constant.MAX_OUTPUT_SIZE,), name="output_gm", scope=tik.scope_gm)
        self.aicore_num = self.tik_profiling.get_aicore_num()
        self.element_each_block = constant.BLOCK_SIZE // self.dtype_size
        self.reserve_ub_size = Constant.Tiling_UB_SIZE
        self.ub_size = (self.tik_profiling.get_unified_buffer_size() - self.reserve_ub_size) // self.dtype_size // \
                       self.element_each_block * self.element_each_block
        self.ub_size_with_vnchwconv = ((self.tik_profiling.get_unified_buffer_size() - self.reserve_ub_size) // \
                                       self.dtype_size - self.element_each_block) // 2 // \
                                      self.element_each_block * self.element_each_block
        self.max_gap = 65535 * self.element_each_block
        self.max_last_dim = (self.max_gap + self.ub_size) // self.element_each_block
        float16_dtype_size = 2
        if self.dtype_size % float16_dtype_size == 0:
            self.vnchwconv_column = 16
            self.tensor_dtype = "float16"
            self.multi_times = self.dtype_size // float16_dtype_size
            self.tensor_dtype_size = float16_dtype_size
            self.input_gm4vnchwconv = self.input_gm.reinterpret_cast_to("float16")
            self.output_gm4vnchwconv = self.output_gm.reinterpret_cast_to("float16")
        else:
            self.vnchwconv_column = 32
            self.tensor_dtype = self.dtype
            self.multi_times = 1
            self.tensor_dtype_size = self.dtype_size
            self.input_gm4vnchwconv = self.input_gm
            self.output_gm4vnchwconv = self.output_gm
        self.tensor_block_element = constant.BLOCK_SIZE // self.tensor_dtype_size
        self.vnchwconv_rows = 16
        self._opt_config = {"out_of_bound_sync_check": False, "enable_const_fold": True}
        self.core_num_scalar = inst.Scalar("int64", name="core_num_scalar", init_value=self.aicore_num)

    @staticmethod
    def _ceil_div(int1: tik.Scalar, int2):
        """
        get ceil for (int1 / int2)
        """
        return ceil_div(int1, int2)

    def _ceil_32bytes_count(self, count: tik.Scalar, block_element):
        """
        _ceil_32bytes_count
        """
        ceil_num = self._ceil_div(count, block_element)
        return ceil_num * block_element

    def _get_input_gm_addr(self, cur_index: tik.Scalar):
        """
        _get_input_gm_addr
        """
        inst = self.tik_instance
        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        dim_count = self.tiling_param.shape_length
        begin = self.tiling_param.begin
        strides = self.tiling_param.stride
        addr = inst.Scalar(self.tiling_param.dtype, name="input_addr")
        addr.set_as(begin[dim_count - 1])
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")
        with inst.for_range(2, dim_count + 1) as dim_idx:
            dim = self.tiling_param.output_shape[dim_count - dim_idx]
            step = self.tiling_param.input_steps[dim_idx - 2]
            tmp.set_as(tmp_cur_index % dim)
            tmp_begin_idx = begin[dim_count - dim_idx] + tmp * strides[dim_count - dim_idx]
            addr.set_as(addr + step * tmp_begin_idx)
            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _get_output_gm_addr(self, cur_index: tik.Scalar):
        """
        _get_output_gm_addr
        """
        inst = self.tik_instance
        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        dim_count = self.tiling_param.shape_length
        addr = inst.Scalar(self.tiling_param.dtype, name="output_addr")
        addr.set_as(0)
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")

        with inst.for_range(2, dim_count + 1) as dim_idx:
            dim = self.tiling_param.output_shape[dim_count - dim_idx]
            step = self.tiling_param.output_steps[dim_idx - 2]
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * tmp)
            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _data_move(self, dest: tik.Tensor, src: tik.Tensor, count: tik.Scalar):
        """
        _data_move
        """
        dtype_size = common_util.get_data_size(src.dtype)
        burst = self._ceil_div(count * dtype_size, constant.BLOCK_SIZE)
        if tbe_platform.api_check_support("tik.data_move_pad"):
            dest_b8 = dest.reinterpret_cast_to("int8")
            src_b8 = src.reinterpret_cast_to("int8")
            self.tik_instance.data_move_pad(dest_b8, src_b8, 1, count * dtype_size, 0, 0)
        else:
            self.tik_instance.data_move(dest, src, 0, 1, burst, 0, 0)

    def get_opt_config(self):
        return self._opt_config

    def get_vars_info(self):
        return {"block_dim": self.aicore_num,
                "ub_size": tik.Dprofile().get_unified_buffer_size() -
                           Constant.Tiling_UB_SIZE}

    def slice(self):
        """
        for slice, the stride is one
        """
        inst = self.tik_instance
        self.tiling_param.init()
        self.core_num_scalar.set_as(self.tiling_param.core_num)
        if self.dtype == "float16":
            with inst.for_range(0, self.core_num_scalar, block_num=self.core_num_scalar, name="core_idx") as i:
                mode = self.tiling_param.tiling_mode
                self.shape_length = self.tiling_param.shape_length
                with inst.if_scope(mode == 1):
                    self._do_small_last_dim(i)
                with inst.elif_scope(mode == 2):
                    self._do_large_last_dim(i)
                with inst.elif_scope(mode == 3):
                    self._do_small_last_dim_with_vnchwconv(i)
                with inst.elif_scope(mode == 4):
                    self._do_large_last_dim_multi_rows(i)
                with inst.elif_scope(mode == 5):
                    self._do_only_slice_last_dim_with_vnchwconv(i)
                with inst.elif_scope(mode == 6):
                    self._do_only_slice_last_dim_with_datamove(i)
                with inst.elif_scope(mode == 7):
                    self._do_with_one_dim(i)
                with inst.elif_scope(mode == 8):
                    self._do_with_last_dim_equal_one(i)
        elif self.dtype_size % 2 == 0:
            with inst.for_range(0, self.core_num_scalar, block_num=self.core_num_scalar, name="core_idx") as i:
                mode = self.tiling_param.tiling_mode
                self.shape_length = self.tiling_param.shape_length
                with inst.if_scope(tik.any(mode == 1, mode == 3)):
                    self._do_small_last_dim(i)
                with inst.elif_scope(mode == 2):
                    self._do_large_last_dim(i)
                with inst.elif_scope(mode == 4):
                    self._do_large_last_dim_multi_rows(i)
                with inst.elif_scope(mode == 5):
                    self._do_only_slice_last_dim_with_vnchwconv(i)
                with inst.elif_scope(mode == 6):
                    self._do_only_slice_last_dim_with_datamove(i)
                with inst.elif_scope(mode == 7):
                    self._do_with_one_dim(i)
                with inst.elif_scope(mode == 8):
                    self._do_with_last_dim_equal_one(i)
        else:
            with inst.for_range(0, self.core_num_scalar, block_num=self.core_num_scalar, name="core_idx") as i:
                mode = self.tiling_param.tiling_mode
                self.shape_length = self.tiling_param.shape_length
                with inst.if_scope(tik.any(mode == 1, mode == 3)):
                    self._do_small_last_dim(i)
                with inst.elif_scope(tik.any(mode == 2, mode == 5)):
                    self._do_large_last_dim(i)
                with inst.elif_scope(mode == 4):
                    self._do_large_last_dim_multi_rows(i)
                with inst.elif_scope(mode == 6):
                    self._do_only_slice_last_dim_with_datamove(i)
                with inst.elif_scope(mode == 7):
                    self._do_with_one_dim(i)
                with inst.elif_scope(mode == 8):
                    self._do_with_last_dim_equal_one(i)

    def strided_slice(self):
        """
        strided_slice
        """
        inst = self.tik_instance
        self.tiling_param.init()
        self.core_num_scalar.set_as(self.tiling_param.core_num)
        if self.dtype == "float16":
            with inst.for_range(0, self.core_num_scalar, block_num=self.core_num_scalar, name="core_idx") as i:
                mode = self.tiling_param.tiling_mode
                self.shape_length = self.tiling_param.shape_length
                with inst.if_scope(mode == 1):
                    self._do_small_last_dim(i)
                with inst.elif_scope(mode == 2):
                    self._do_large_last_dim(i)
                with inst.elif_scope(mode == 3):
                    self._do_small_last_dim_with_vnchwconv(i)
                with inst.elif_scope(mode == 4):
                    self._do_large_last_dim_multi_rows(i)
                with inst.elif_scope(mode == 5):
                    self._do_only_slice_last_dim_with_vnchwconv(i)
                with inst.elif_scope(mode == 6):
                    self._do_only_slice_last_dim_with_datamove(i)
                with inst.elif_scope(mode == 7):
                    self._do_with_one_dim(i)
                with inst.elif_scope(mode == 8):
                    self._do_with_last_dim_equal_one(i)
                with inst.elif_scope(mode == TilingMode.ONLY_LAST_STRIDE_LARGER1_OUT_CONTINUAL_VNCHWCONV):
                    self._do_only_last_stride_larger1_out_continual_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER1_SMALL_INOUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger1_small_inout_inner_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER1_LE_BLOCK_ELE_LARGE_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger1_large_out_inner_with_vnchwconv_copy_multi_strides(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER1_FUNCTIONAL):
                    self._do_last_stride_larger1_functional(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER_BLOCK_ELE_LARGE_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger_block_element_large_out_inner_with_vnchwconv_copy_block_by_block(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_32B_ALIGN_LARGE_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_block_element_align_large_out_inner_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_32B_ALIGN_SMALL_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_block_element_align_small_out_inner_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER1_NOT_32B_ALIGN_SMALL_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger_block_element_not_align_small_out_inner_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.NOT_LAST_STRIDE_LARGER1_FUNCTIONAL):
                    self._do_large_last_dim(i)
                with inst.elif_scope(mode == TilingMode.NOT_LAST_STRIDE_LARGER1_SMALL_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger1_small_inout_inner_with_vnchwconv(i)
        elif self.dtype_size % 2 == 0:
            with inst.for_range(0, self.core_num_scalar, block_num=self.core_num_scalar, name="core_idx") as i:
                mode = self.tiling_param.tiling_mode
                self.shape_length = self.tiling_param.shape_length
                with inst.if_scope(tik.any(mode == 1, mode == 3)):
                    self._do_small_last_dim(i)
                with inst.elif_scope(mode == 2):
                    self._do_large_last_dim(i)
                with inst.elif_scope(mode == 4):
                    self._do_large_last_dim_multi_rows(i)
                with inst.elif_scope(mode == 5):
                    self._do_only_slice_last_dim_with_vnchwconv(i)
                with inst.elif_scope(mode == 6):
                    self._do_only_slice_last_dim_with_datamove(i)
                with inst.elif_scope(mode == 7):
                    self._do_with_one_dim(i)
                with inst.elif_scope(mode == 8):
                    self._do_with_last_dim_equal_one(i)
                with inst.elif_scope(mode == TilingMode.ONLY_LAST_STRIDE_LARGER1_OUT_CONTINUAL_VNCHWCONV):
                    self._do_only_last_stride_larger1_out_continual_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER1_SMALL_INOUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger1_small_inout_inner_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER1_LE_BLOCK_ELE_LARGE_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger1_large_out_inner_with_vnchwconv_copy_multi_strides(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER1_FUNCTIONAL):
                    self._do_last_stride_larger1_functional(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER_BLOCK_ELE_LARGE_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger_block_element_large_out_inner_with_vnchwconv_copy_block_by_block(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_32B_ALIGN_LARGE_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_block_element_align_large_out_inner_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_32B_ALIGN_SMALL_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_block_element_align_small_out_inner_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER1_NOT_32B_ALIGN_SMALL_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger_block_element_not_align_small_out_inner_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.NOT_LAST_STRIDE_LARGER1_FUNCTIONAL):
                    self._do_large_last_dim(i)
                with inst.elif_scope(mode == TilingMode.NOT_LAST_STRIDE_LARGER1_SMALL_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger1_small_inout_inner_with_vnchwconv(i)
        else:
            with inst.for_range(0, self.core_num_scalar, block_num=self.core_num_scalar, name="core_idx") as i:
                mode = self.tiling_param.tiling_mode
                self.shape_length = self.tiling_param.shape_length
                with inst.if_scope(tik.any(mode == 1, mode == 3)):
                    self._do_small_last_dim(i)
                with inst.elif_scope(tik.any(mode == 2, mode == 5)):
                    self._do_large_last_dim(i)
                with inst.elif_scope(mode == 4):
                    self._do_large_last_dim_multi_rows(i)
                with inst.elif_scope(mode == 6):
                    self._do_only_slice_last_dim_with_datamove(i)
                with inst.elif_scope(mode == 7):
                    self._do_with_one_dim(i)
                with inst.elif_scope(mode == 8):
                    self._do_with_last_dim_equal_one(i)
                with inst.elif_scope(mode == TilingMode.ONLY_LAST_STRIDE_LARGER1_OUT_CONTINUAL_VNCHWCONV):
                    self._do_only_last_stride_larger1_out_continual_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER1_SMALL_INOUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger1_small_inout_inner_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER1_LE_BLOCK_ELE_LARGE_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger1_large_out_inner_with_vnchwconv_copy_multi_strides(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER1_FUNCTIONAL):
                    self._do_last_stride_larger1_functional(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER_BLOCK_ELE_LARGE_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger_block_element_large_out_inner_with_vnchwconv_copy_block_by_block(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_32B_ALIGN_LARGE_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_block_element_align_large_out_inner_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_32B_ALIGN_SMALL_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_block_element_align_small_out_inner_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.LAST_STRIDE_LARGER1_NOT_32B_ALIGN_SMALL_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger_block_element_not_align_small_out_inner_with_vnchwconv(i)
                with inst.elif_scope(mode == TilingMode.NOT_LAST_STRIDE_LARGER1_FUNCTIONAL):
                    self._do_large_last_dim(i)
                with inst.elif_scope(mode == TilingMode.NOT_LAST_STRIDE_LARGER1_SMALL_OUT_INNER_VNCHWCONV):
                    self._do_last_stride_larger1_small_inout_inner_with_vnchwconv(i)

    def _min(self, left, right):
        inst = self.tik_instance
        min_value = inst.Scalar(dtype=self.input_inner_dim.dtype, init_value=right)
        with inst.if_scope(left < right):
            min_value.set_as(left)
        return min_value

    def _get_input_inner_dim_no_begin(self):
        last_begin = self.begin[self.shape_length - 1]
        last_inner = self.input_inner_dim
        dtype_size = self.dtype_size
        float16_dtype_size = 2
        multi_times = max(dtype_size // float16_dtype_size, 1)
        return (last_inner - last_begin) * multi_times

    def _do_with_last_dim_equal_one(self, core_idx):
        """
        slice the last dim and the last dim size of outshape is equal one.
        Parameters
        ----------
        core_idx: number of ai_core

        Returns
        -------
        None
        """
        inst = self.tik_instance
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        max_rows_in_ub = inst.Scalar("int64", name="max_rows_in_ub")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        last_repeat_rows = inst.Scalar("int64", name="last_repeat_rows")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        tail_rows_repeat_times = inst.Scalar("int64", name="tail_rows_repeat_times")
        tail_last_repeat_rows = inst.Scalar("int64", name="tail_last_repeat_rows")
        input_inner_dim = self.tiling_param.input_shape[self.shape_length - 1]
        input_outer_dim = self.tiling_param.input_shape[0]
        max_rows_in_ub.set_as(floor_align(self.ub_size // (input_inner_dim + 1), self.element_each_block))
        rows_each_core.set_as(
            self._ceil_32bytes_count(self._ceil_div(self.tiling_param.input_shape[0], self.core_num_scalar),
                                     self.element_each_block))
        aicore_num_used.set_as(self._ceil_div(input_outer_dim, rows_each_core))
        repeat_times.set_as(rows_each_core // max_rows_in_ub)
        last_repeat_rows.set_as(rows_each_core % max_rows_in_ub)
        tail_rows.set_as(input_outer_dim % rows_each_core)
        tail_rows_repeat_times.set_as(tail_rows // max_rows_in_ub)
        tail_last_repeat_rows.set_as(tail_rows % max_rows_in_ub)

        src_addr = inst.Scalar("int64", name="src_addr")
        dst_addr = inst.Scalar("int64", name="dst_addr")

        with inst.new_stmt_scope():
            input_ub = inst.Tensor(self.dtype, (max_rows_in_ub * input_inner_dim,),
                                   scope=tik.scope_ubuf,
                                   name="input_ub")
            output_ub = inst.Tensor(self.dtype, (max_rows_in_ub,), scope=tik.scope_ubuf, name="output_ub")
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tail_rows != 0):
                    with inst.if_scope(core_idx < aicore_num_used - 1):
                        with inst.for_range(0, repeat_times) as repeat_idx:
                            src_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * \
                                            input_inner_dim)
                            dst_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * 1)
                            self._do_with_last_dim_equal_one_per_loop(src_addr, dst_addr, max_rows_in_ub, input_ub,
                                                                      output_ub)
                        with inst.if_scope(last_repeat_rows > 0):
                            src_addr.set_as((core_idx * rows_each_core + repeat_times * max_rows_in_ub) * \
                                            input_inner_dim)
                            dst_addr.set_as((core_idx * rows_each_core + repeat_times * max_rows_in_ub) * 1)
                            self._do_with_last_dim_equal_one_per_loop(src_addr, dst_addr, last_repeat_rows, input_ub,
                                                                      output_ub)
                    with inst.else_scope():
                        with inst.for_range(0, tail_rows_repeat_times) as tail_repeat_idx:
                            src_addr.set_as((core_idx * rows_each_core + tail_repeat_idx * max_rows_in_ub) * \
                                            input_inner_dim)
                            dst_addr.set_as((core_idx * rows_each_core + tail_repeat_idx * max_rows_in_ub) * 1)
                            self._do_with_last_dim_equal_one_per_loop(src_addr, dst_addr, max_rows_in_ub, input_ub,
                                                                      output_ub)
                        with inst.if_scope(tail_last_repeat_rows > 0):
                            src_addr.set_as((core_idx * rows_each_core + tail_rows_repeat_times * max_rows_in_ub) * \
                                            input_inner_dim)
                            dst_addr.set_as((core_idx * rows_each_core + tail_rows_repeat_times * max_rows_in_ub) * 1)
                            self._do_with_last_dim_equal_one_per_loop(src_addr, dst_addr, tail_last_repeat_rows,
                                                                      input_ub, output_ub)
                with inst.else_scope():
                    with inst.for_range(0, repeat_times) as repeat_idx:
                        src_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * input_inner_dim)
                        dst_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * 1)
                        self._do_with_last_dim_equal_one_per_loop(src_addr, dst_addr, max_rows_in_ub, input_ub,
                                                                  output_ub)
                    with inst.if_scope(last_repeat_rows > 0):
                        src_addr.set_as((core_idx * rows_each_core + repeat_times * max_rows_in_ub) * input_inner_dim)
                        dst_addr.set_as((core_idx * rows_each_core + repeat_times * max_rows_in_ub) * 1)
                        self._do_with_last_dim_equal_one_per_loop(src_addr, dst_addr, last_repeat_rows, input_ub,
                                                                  output_ub)

    def _do_with_last_dim_equal_one_per_loop(self, src_addr, dst_addr, cur_repeat_rows, input_ub, output_ub):
        """
        slice the last dim and the last dim size of outshape is equal one.
        Parameters
        ----------
        src_addr: the ub addr when move data from gm to ub
        dst_addr: the gm addr when move data from ub to gm
        cur_repeat_rows: number of rows processed at one loop
        input_ub: ub for storing input data
        output_ub: ub for storing output data

        Returns
        -------
        None
        """
        inst = self.tik_instance
        input_inner_dim = self.tiling_param.input_shape[self.shape_length - 1]
        start_num = inst.Scalar("int64", name="start_num")
        loop_data = inst.Scalar("int64", name="loop_data")
        start_num.set_as(self.tiling_param.begin[self.shape_length - 1])
        loop_data.set_as(cur_repeat_rows * input_inner_dim)
        self._data_move(input_ub[0], self.input_gm[src_addr], loop_data)
        with inst.for_range(0, cur_repeat_rows) as idx:
            output_ub[idx].set_as(input_ub[idx * input_inner_dim + start_num])
        self._data_move(self.output_gm[dst_addr], output_ub, cur_repeat_rows)

    def _do_with_one_dim(self, core_idx):
        """
        slice the data and the length of outshape is equal one.
        Parameters
        ----------
        core_idx: number of ai_core

        Returns
        -------
        None
        """
        inst = self.tik_instance
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        core_data = inst.Scalar("int64", name="core_data")
        tail_data = inst.Scalar("int64", name="tail_data")
        input_addr = inst.Scalar("int64", name="input_addr")
        output_addr = inst.Scalar("int64", name="output_addr")
        max_data_in_ub = inst.Scalar("int64", name="max_data_in_ub")
        output_inner_dim = self.tiling_param.output_shape[self.shape_length - 1]
        start_num = self.tiling_param.begin[self.shape_length - 1]
        core_data.set_as(ceil_align(ceil_div(output_inner_dim, self.core_num_scalar), self.element_each_block))
        aicore_num_used.set_as(ceil_div(output_inner_dim, core_data))
        max_data_in_ub.set_as(floor_align(self.ub_size, self.element_each_block))
        with inst.if_scope(aicore_num_used == 1):
            core_data.set_as(output_inner_dim)
        tail_data.set_as(output_inner_dim % core_data)
        with inst.if_scope(tail_data == 0):
            tail_data.set_as(core_data)
        with inst.new_stmt_scope():
            one_dim_ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="one_dim_ub")
            with inst.if_scope(core_idx < aicore_num_used - 1):
                input_addr.set_as(core_idx * core_data + start_num)
                output_addr.set_as(core_idx * core_data)
                self._do_with_one_dim_per_core(input_addr, output_addr, core_data, one_dim_ub, max_data_in_ub)
            with inst.elif_scope(core_idx == aicore_num_used - 1):
                input_addr.set_as((aicore_num_used - 1) * core_data + start_num)
                output_addr.set_as((aicore_num_used - 1) * core_data)
                self._do_with_one_dim_per_core(input_addr, output_addr, tail_data, one_dim_ub, max_data_in_ub)

    def _do_with_one_dim_per_core(self, input_addr, output_addr, core_data, input_ub, max_data_in_ub):
        """
        slice the data and the length of outshape is equal one.
        Parameters
        ----------
        input_addr: the ub addr when move data from gm to ub
        output_addr: the gm addr when move data from ub to gm
        core_data: number of data processed at one loop
        input_ub: ub for storing input data
        max_data_in_ub: maximum data stored in UB

        Returns
        -------
        None
        """
        inst = self.tik_instance
        src_addr = inst.Scalar("int64", name="src_addr")
        dst_addr = inst.Scalar("int64", name="dst_addr")
        loop_times = core_data // max_data_in_ub
        last_loop_data = core_data % max_data_in_ub
        with inst.for_range(0, loop_times) as loop_idx:
            src_addr.set_as(input_addr + loop_idx * max_data_in_ub)
            dst_addr.set_as(output_addr + loop_idx * max_data_in_ub)
            self._data_move(input_ub[0], self.input_gm[src_addr], max_data_in_ub)
            self._data_move(self.output_gm[dst_addr], input_ub[0], max_data_in_ub)
        src_addr.set_as(input_addr + loop_times * max_data_in_ub)
        dst_addr.set_as(output_addr + loop_times * max_data_in_ub)
        with inst.if_scope(last_loop_data > 0):
            self._data_move(input_ub[0], self.input_gm[src_addr], last_loop_data)
            self._data_move(self.output_gm[dst_addr], input_ub[0], last_loop_data)

    def _do_only_slice_last_dim_with_datamove(self, core_idx):
        """
        slice the last dim with data_move instruction.
        Parameters
        ----------
        core_idx: number of ai_core

        Returns
        -------
        None
        """
        inst = self.tik_instance
        max_rows_in_ub = inst.Scalar("int64", name="max_rows_in_ub")
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        last_repeat_rows = inst.Scalar("int64", name="last_repeat_rows")
        tail_repeat_rows = inst.Scalar("int64", name="tail_repeat_rows")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        tail_rows_repeat_times = inst.Scalar("int64", name="tail_rows_repeat_times")
        tail_rows_last_repeat = inst.Scalar("int64", name="tail_rows_last_repeat")
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        input_addr = inst.Scalar("int64", name="input_addr")
        output_addr = inst.Scalar("int64", name="output_addr")
        max_rows_in_ub.set_as(
            floor_align(self.ub_size // self.tiling_param.input_shape[self.shape_length - 1], self.element_each_block))
        with inst.if_scope(max_rows_in_ub // self.element_each_block > Constant.MAX_REPEAT):
            max_rows_in_ub.set_as(Constant.MAX_REPEAT * self.element_each_block)
        rows_each_core.set_as(
            self._ceil_32bytes_count(self._ceil_div(self.tiling_param.input_shape[0], self.core_num_scalar),
                                     self.element_each_block))
        repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
        last_repeat_rows.set_as(rows_each_core % max_rows_in_ub)
        with inst.if_scope(last_repeat_rows == 0):
            last_repeat_rows.set_as(max_rows_in_ub)
        tail_rows.set_as(self.tiling_param.input_shape[0] % rows_each_core)
        tail_rows_repeat_times.set_as(self._ceil_div(tail_rows, max_rows_in_ub))
        tail_repeat_rows.set_as(floor_align(tail_rows % max_rows_in_ub, self.element_each_block))
        tail_rows_last_repeat.set_as(tail_rows % max_rows_in_ub - tail_repeat_rows)
        with inst.if_scope(tik.all(tail_rows_last_repeat == 0, tail_repeat_rows == 0)):
            tail_repeat_rows.set_as(max_rows_in_ub)
        aicore_num_used.set_as(self._ceil_div(self.tiling_param.input_shape[0], rows_each_core))
        input_inner_dim = self.tiling_param.input_shape[self.shape_length - 1]
        output_inner_dim = self.tiling_param.output_shape[self.shape_length - 1]
        with inst.new_stmt_scope():
            input_ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="input_ub")
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tail_rows != 0):
                    with inst.if_scope(core_idx < aicore_num_used - 1):
                        with inst.for_range(0, repeat_times - 1) as repeat_idx:
                            input_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * \
                                              input_inner_dim)
                            output_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * \
                                               output_inner_dim)
                            self._do_with_data_move_per_loop(input_addr, output_addr, input_ub, max_rows_in_ub)
                        input_addr.set_as((core_idx * rows_each_core + (repeat_times - 1) * max_rows_in_ub) * \
                                              input_inner_dim)
                        output_addr.set_as((core_idx * rows_each_core + (repeat_times - 1) * max_rows_in_ub) * \
                                            output_inner_dim)
                        self._do_with_data_move_per_loop(input_addr, output_addr, input_ub, last_repeat_rows)
                    with inst.else_scope():
                        with inst.for_range(0, tail_rows_repeat_times - 1) as tail_repeat_idx:
                            input_addr.set_as(((aicore_num_used - 1) * rows_each_core + \
                                               tail_repeat_idx * max_rows_in_ub) * input_inner_dim)
                            output_addr.set_as(((aicore_num_used - 1) * rows_each_core + \
                                                tail_repeat_idx * max_rows_in_ub) * output_inner_dim)
                            self._do_with_data_move_per_loop(input_addr, output_addr, input_ub, max_rows_in_ub)
                        with inst.if_scope(tail_repeat_rows > 0):
                            input_addr.set_as(((aicore_num_used - 1) * rows_each_core + \
                                               (tail_rows_repeat_times - 1) * max_rows_in_ub) * input_inner_dim)
                            output_addr.set_as(((aicore_num_used - 1) * rows_each_core + \
                                                (tail_rows_repeat_times - 1) * max_rows_in_ub) * output_inner_dim)
                            self._do_with_data_move_per_loop(input_addr, output_addr, input_ub, tail_repeat_rows)
                        with inst.if_scope(tail_rows_last_repeat > 0):
                            self._do_with_data_move_tail_rows((aicore_num_used - 1) * rows_each_core + \
                                                            (tail_rows_repeat_times - 1) * max_rows_in_ub + \
                                                            tail_repeat_rows, tail_rows_last_repeat, input_ub)
                with inst.else_scope():
                    with inst.if_scope(core_idx < aicore_num_used):
                        with inst.for_range(0, repeat_times - 1) as repeat_idx:
                            input_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * \
                                              input_inner_dim)
                            output_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * \
                                               output_inner_dim)
                            self._do_with_data_move_per_loop(input_addr, output_addr, input_ub, max_rows_in_ub)
                        input_addr.set_as((core_idx * rows_each_core + (repeat_times - 1) * max_rows_in_ub) * \
                                              input_inner_dim)
                        output_addr.set_as((core_idx * rows_each_core + (repeat_times - 1) * max_rows_in_ub) * \
                                            output_inner_dim)
                        self._do_with_data_move_per_loop(input_addr, output_addr, input_ub, last_repeat_rows)

    def _do_with_data_move_per_loop(self, input_addr, output_addr, input_ub, cur_repeat_rows):
        """
        slice the last dim with data_move instruction, this function copy data with rollback, will not copy/write
        redundant data.
        Parameters
        ----------
        input_addr: the ub addr when move data from gm to ub
        output_addr: the gm addr when move data from ub to gm
        input_ub: ub for storing input data
        cur_repeat_rows: number of rows processed at one loop

        Returns
        -------
        None
        """
        inst = self.tik_instance
        input_inner_dim = self.tiling_param.input_shape[self.shape_length - 1]
        output_inner_dim = self.tiling_param.output_shape[self.shape_length - 1]
        output_last_dim_part1 = output_inner_dim // self.element_each_block * self.element_each_block
        output_last_dim_part1_block = output_last_dim_part1 // self.element_each_block
        output_last_dim_part2 = output_inner_dim - output_last_dim_part1
        start_num = self.tiling_param.begin[self.shape_length - 1]
        src_addr = inst.Scalar("int64", name="src_addr")
        dst_addr = inst.Scalar("int64", name="dst_addr")
        with inst.for_range(0, self.element_each_block) as loop_idx:
            src_addr.set_as(input_addr + loop_idx * input_inner_dim + start_num)
            dst_addr.set_as(loop_idx * output_last_dim_part1)
            inst.data_move(input_ub[dst_addr], self.input_gm[src_addr], 0, cur_repeat_rows // self.element_each_block,
                           output_last_dim_part1_block, input_inner_dim - output_last_dim_part1_block,
                           output_last_dim_part1 - output_last_dim_part1_block)
            src_addr.set_as(loop_idx * output_last_dim_part1)
            dst_addr.set_as(output_addr + loop_idx * output_inner_dim)
            inst.data_move(self.output_gm[dst_addr], input_ub[src_addr], 0, cur_repeat_rows // self.element_each_block,
                           output_last_dim_part1_block, output_last_dim_part1 - output_last_dim_part1_block,
                           output_inner_dim - output_last_dim_part1_block)
            with inst.if_scope(output_last_dim_part2 > 0):
                inst.data_move(
                    input_ub[loop_idx * self.element_each_block],
                    self.input_gm[input_addr + loop_idx * input_inner_dim + start_num +
                                  (output_inner_dim - self.element_each_block)], 0,
                    cur_repeat_rows // self.element_each_block, 1,
                    input_inner_dim - 1, self.element_each_block - 1)
                inst.data_move(
                    self.output_gm[
                        output_addr + loop_idx * output_inner_dim + (output_inner_dim - self.element_each_block)],
                    input_ub[loop_idx * self.element_each_block], 0, cur_repeat_rows // self.element_each_block, 1,
                                                                     self.element_each_block - 1, output_inner_dim - 1)

    def _do_with_data_move_tail_rows(self, rows, tail_rows_last_repeat, input_ub):
        """
        slice the tail data with data_move instruction.
        Parameters
        ----------
        rows: start data row
        tail_rows_last_repeat: rows to be processed
        input_ub: ub for storing input data

        Returns
        -------
        None
        """
        inst = self.tik_instance
        input_inner_dim = self.tiling_param.input_shape[self.shape_length - 1]
        output_inner_dim = self.tiling_param.output_shape[self.shape_length - 1]
        start_num = self.tiling_param.begin[self.shape_length - 1]
        input_addr = rows * input_inner_dim
        output_addr = rows * output_inner_dim
        with inst.for_range(0, tail_rows_last_repeat) as loop_idx:
            self._data_move(input_ub, self.input_gm[input_addr + loop_idx * input_inner_dim + start_num],
                            output_inner_dim)
            self._data_move(self.output_gm[output_addr + loop_idx * output_inner_dim], input_ub, output_inner_dim)

    def _do_only_slice_last_dim_with_vnchwconv(self, core_idx):
        """
        slice the last dim and a column ub can fit multiple rows of input data.
        """
        inst = self.tik_instance
        dtype_size = self.dtype_size
        output_shape = self.tiling_param.output_shape
        input_shape = self.tiling_param.input_shape
        shape_length = self.tiling_param.shape_length
        float16_dtype_size = common_util.get_data_size("float16")
        multi_times = dtype_size // float16_dtype_size
        tensor_dtype = "float16"
        vnchwconv_column = 16
        input_inner_dim = input_shape[shape_length - 1] * multi_times
        output_inner_dim = output_shape[shape_length - 1] * multi_times
        out_dim = self.tiling_param.out_dim
        ub_size = self.ub_size * multi_times // 2
        begin_value = self.tiling_param.begin[shape_length - 1] * multi_times
        element_each_block = self.element_each_block * multi_times
        output_32bytes_align_rows = inst.Scalar("int64",
                                                name="output_32bytes_align_rows",
                                                init_value=element_each_block)
        with inst.if_scope(output_32bytes_align_rows % output_inner_dim == 0):
            output_32bytes_align_rows.set_as(output_32bytes_align_rows // output_inner_dim)
        with inst.elif_scope(output_inner_dim % output_32bytes_align_rows == 0):
            output_32bytes_align_rows.set_as(1)
        reserve_input_32bytes_align_ub = vnchwconv_column * constant.BLOCK_SIZE
        max_rows_in_ub = floor_align((ub_size - reserve_input_32bytes_align_ub) // (input_inner_dim * vnchwconv_column),
                                     output_32bytes_align_rows)
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        rows_each_repeat = inst.Scalar("int64", name="rows_each_repeat")
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        tail_rows_repeat_times = inst.Scalar("int64", name="tail_rows_repeat_times")
        tail_rows_repeat_tail_count = inst.Scalar("int64", name="tail_rows_repeat_tail_count")
        tail_rows_repeat_roll_back_rows = inst.Scalar("int64", name="tail_rows_repeat_roll_back_rows", init_value=0)
        rows_each_core.set_as(
            self._ceil_32bytes_count(self._ceil_div(out_dim, self.core_num_scalar), output_32bytes_align_rows))
        repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
        rows_each_repeat.set_as(
            self._ceil_32bytes_count(self._ceil_div(rows_each_core, repeat_times), output_32bytes_align_rows))
        rows_each_core.set_as(rows_each_repeat * repeat_times)
        aicore_num_used.set_as(self._ceil_div(out_dim, rows_each_core))
        with inst.if_scope(rows_each_core > out_dim):
            rows_each_core.set_as(out_dim)
            repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
            rows_each_repeat.set_as(
                self._ceil_32bytes_count(self._ceil_div(rows_each_core, repeat_times), output_32bytes_align_rows))
            with inst.if_scope(rows_each_repeat > out_dim):
                rows_each_repeat.set_as(out_dim)
        loop_times = self._ceil_div(repeat_times, 16)
        last_loop_rows = inst.Scalar("int64", name="last_loop_rows", init_value=repeat_times % 16)
        with inst.if_scope(last_loop_rows == 0):
            last_loop_rows.set_as(16)

        tail_rows.set_as(out_dim % rows_each_core)
        tail_rows_repeat_times.set_as(self._ceil_div(tail_rows, rows_each_repeat))
        tail_rows_repeat_tail_count.set_as(tail_rows % rows_each_repeat)
        with inst.if_scope(tail_rows_repeat_tail_count == 0):
            tail_rows_repeat_tail_count.set_as(rows_each_repeat)
        with inst.if_scope(tail_rows_repeat_tail_count % output_32bytes_align_rows != 0):
            tail_rows_repeat_roll_back_rows.set_as(self._ceil_32bytes_count(tail_rows_repeat_tail_count,
                                                                            output_32bytes_align_rows) - \
                                                   tail_rows_repeat_tail_count)
        tail_loop_times = self._ceil_div(tail_rows_repeat_times - 1, 16)
        tail_last_loop_rows = inst.Scalar("int64",
                                          name="tail_last_loop_rows",
                                          init_value=(tail_rows_repeat_times - 1) % 16)
        with inst.if_scope(tail_last_loop_rows == 0):
            tail_last_loop_rows.set_as(16)

        if dtype_size % float16_dtype_size == 0:
            input_gm = self.input_gm.reinterpret_cast_to("float16")
            output_gm = self.output_gm.reinterpret_cast_to("float16")
        else:
            input_gm = self.input_gm
            output_gm = self.output_gm

        input_addr = inst.Scalar("int64", name="input_addr")
        output_addr = inst.Scalar("int64", name="output_addr")
        param_dict = {
            "input_gm": input_gm,
            "output_gm": output_gm,
            "rows_each_repeat": rows_each_repeat,
            "input_inner_dim": input_inner_dim,
            "output_inner_dim": output_inner_dim,
            "begin_value": begin_value,
            "element_each_block": element_each_block
        }

        with inst.new_stmt_scope():
            input_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="vnchw_conv_ub")
            param_dict["input_ub"] = input_ub
            param_dict["vnchw_conv_ub"] = vnchw_conv_ub
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tail_rows != 0):
                    with inst.if_scope(core_idx < aicore_num_used - 1):
                        with inst.for_range(0, loop_times - 1) as loop_idx:
                            input_addr.set_as(rows_each_core * input_inner_dim * core_idx + \
                                              loop_idx * rows_each_repeat * input_inner_dim * 16)
                            output_addr.set_as(rows_each_core * output_inner_dim * core_idx + \
                                               loop_idx * rows_each_repeat * output_inner_dim * 16)
                            param_dict["input_addr"] = input_addr
                            param_dict["output_addr"] = output_addr
                            param_dict["loop_rows"] = 16
                            self._do_with_vnchwconv_per_loop(param_dict)
                        input_addr.set_as(rows_each_core * input_inner_dim * core_idx + \
                                          (loop_times - 1) * rows_each_repeat * input_inner_dim * 16)
                        output_addr.set_as(rows_each_core * output_inner_dim * core_idx + \
                                           (loop_times - 1) * 16 * rows_each_repeat * output_inner_dim)
                        param_dict["input_addr"] = input_addr
                        param_dict["output_addr"] = output_addr
                        param_dict["loop_rows"] = last_loop_rows
                        self._do_with_vnchwconv_per_loop(param_dict)
                    with inst.else_scope():
                        with inst.if_scope(tail_rows_repeat_times > 1):
                            with inst.for_range(0, tail_loop_times - 1) as loop_idx:
                                input_addr.set_as(rows_each_core * input_inner_dim * (aicore_num_used - 1) + \
                                                  loop_idx * rows_each_repeat * input_inner_dim * 16)
                                output_addr.set_as(rows_each_core * output_inner_dim * (aicore_num_used - 1) + \
                                                   loop_idx * rows_each_repeat * output_inner_dim * 16)
                                param_dict["input_addr"] = input_addr
                                param_dict["output_addr"] = output_addr
                                param_dict["loop_rows"] = 16
                                self._do_with_vnchwconv_per_loop(param_dict)
                            input_addr.set_as(rows_each_core * input_inner_dim * (aicore_num_used - 1) + \
                                              (tail_loop_times - 1) * rows_each_repeat * input_inner_dim * 16)
                            output_addr.set_as(rows_each_core * output_inner_dim * (aicore_num_used - 1) + \
                                               (tail_loop_times - 1) * 16 * rows_each_repeat * output_inner_dim)
                            param_dict["input_addr"] = input_addr
                            param_dict["output_addr"] = output_addr
                            param_dict["loop_rows"] = tail_last_loop_rows
                            self._do_with_vnchwconv_per_loop(param_dict)
                        input_addr.set_as(rows_each_core * input_inner_dim * (aicore_num_used - 1) + \
                                          (tail_rows_repeat_times - 1) * rows_each_repeat * input_inner_dim - \
                                          tail_rows_repeat_roll_back_rows * input_inner_dim)
                        output_addr.set_as(rows_each_core * output_inner_dim * (aicore_num_used - 1) + \
                                           (tail_rows_repeat_times - 1) * rows_each_repeat * output_inner_dim - \
                                           tail_rows_repeat_roll_back_rows * output_inner_dim)
                        param_dict["input_addr"] = input_addr
                        param_dict["output_addr"] = output_addr
                        param_dict["loop_rows"] = 1
                        param_dict["rows_each_repeat"].set_as(tail_rows_repeat_tail_count +
                                                              tail_rows_repeat_roll_back_rows)
                        self._do_with_vnchwconv_per_loop(param_dict)
                with inst.else_scope():
                    with inst.for_range(0, loop_times - 1) as loop_idx:
                        input_addr.set_as(rows_each_core * input_inner_dim * core_idx + \
                                          loop_idx * rows_each_repeat * input_inner_dim * 16)
                        output_addr.set_as(rows_each_core * output_inner_dim * core_idx + \
                                           loop_idx * 16 * rows_each_repeat * output_inner_dim)
                        param_dict["input_addr"] = input_addr
                        param_dict["output_addr"] = output_addr
                        param_dict["loop_rows"] = 16
                        self._do_with_vnchwconv_per_loop(param_dict)
                    input_addr.set_as(rows_each_core * input_inner_dim * core_idx + \
                                      (loop_times - 1) * rows_each_repeat * input_inner_dim * 16)
                    output_addr.set_as(rows_each_core * output_inner_dim * core_idx + \
                                       (loop_times - 1) * 16 * rows_each_repeat * output_inner_dim)
                    param_dict["input_addr"] = input_addr
                    param_dict["output_addr"] = output_addr
                    with inst.if_scope(rows_each_core % rows_each_repeat == 0):
                        param_dict["loop_rows"] = last_loop_rows
                        self._do_with_vnchwconv_per_loop(param_dict)
                    with inst.elif_scope(last_loop_rows == 1):
                        param_dict["loop_rows"] = 1
                        param_dict["rows_each_repeat"].set_as(rows_each_core % rows_each_repeat)
                        self._do_with_vnchwconv_per_loop(param_dict)
                    with inst.else_scope():
                        param_dict["loop_rows"] = last_loop_rows - 1
                        param_dict["rows_each_repeat"].set_as(rows_each_repeat)
                        self._do_with_vnchwconv_per_loop(param_dict)
                        input_addr.set_as(rows_each_core * input_inner_dim * core_idx +
                                          (loop_times - 1) * rows_each_repeat * input_inner_dim * 16 +
                                          rows_each_repeat * input_inner_dim * (last_loop_rows - 1))
                        output_addr.set_as(rows_each_core * output_inner_dim * core_idx +
                                           (loop_times - 1) * 16 * rows_each_repeat * output_inner_dim +
                                           rows_each_repeat * output_inner_dim * (last_loop_rows - 1))
                        param_dict["input_addr"] = input_addr
                        param_dict["output_addr"] = output_addr
                        param_dict["loop_rows"] = 1
                        param_dict["rows_each_repeat"].set_as(rows_each_core % rows_each_repeat)
                        self._do_with_vnchwconv_per_loop(param_dict)


    def _do_with_vnchwconv_per_loop(self, param_dict):
        """
        slice multiple rows of input data at one loop
        """
        inst = self.tik_instance
        input_gm = param_dict["input_gm"]
        output_gm = param_dict["output_gm"]
        rows_each_repeat = param_dict["rows_each_repeat"]
        input_inner_dim = param_dict["input_inner_dim"]
        output_inner_dim = param_dict["output_inner_dim"]
        begin_value = param_dict["begin_value"]
        element_each_block = param_dict["element_each_block"]
        input_addr = param_dict["input_addr"]
        output_addr = param_dict["output_addr"]
        loop_rows = param_dict["loop_rows"]
        input_ub = param_dict["input_ub"]
        vnchw_conv_ub = param_dict["vnchw_conv_ub"]
        vnchw_conv_repeat_times = inst.Scalar("int64", name="vnchw_conv_repeat_times")
        vnchw_conv_repeat_times.set_as(self._ceil_div(rows_each_repeat * input_inner_dim, element_each_block))
        loop_count = inst.Scalar("int64", name="loop_count")
        loop_count.set_as((self._ceil_div(rows_each_repeat * input_inner_dim, element_each_block)) * \
                          element_each_block)
        with inst.for_range(0, loop_rows) as loop_rows_idx:
            src_addr_in = input_addr + loop_rows_idx * rows_each_repeat * input_inner_dim
            dst_addr_in = loop_rows_idx * ceil_align(rows_each_repeat * input_inner_dim, element_each_block)
            self._data_move(input_ub[dst_addr_in], input_gm[src_addr_in], rows_each_repeat * input_inner_dim)
        self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times, element_each_block)
        ub2ub(inst, input_ub, vnchw_conv_ub[begin_value * 16], [0, rows_each_repeat, output_inner_dim,
              input_inner_dim - output_inner_dim, 0])
        vnchw_conv_repeat_times.set_as(self._ceil_div(rows_each_repeat * output_inner_dim, element_each_block))
        self._do_with_vnchwconv2output(vnchw_conv_ub, input_ub, vnchw_conv_repeat_times, element_each_block)
        self._data_move(output_gm[output_addr], vnchw_conv_ub, loop_rows * rows_each_repeat * output_inner_dim)

    def _do_with_vnchwconv_copy_multi_rows_per_loop_last_stride_larger_than_one(self, param_dict):
        """
        slice multiple rows of input data at one loop
        """
        inst = self.tik_instance
        input_gm = param_dict["input_gm"]
        output_gm = param_dict["output_gm"]
        rows_each_repeat = param_dict["rows_each_repeat"]
        input_inner_dim = param_dict["input_inner_dim"]
        output_inner_dim = param_dict["output_inner_dim"]
        begin_value = param_dict["begin_value"]
        element_each_block = param_dict["element_each_block"]
        input_addr = param_dict["input_addr"]
        output_addr = param_dict["output_addr"]
        loop_rows = param_dict["loop_rows"]
        input_ub = param_dict["input_ub"]
        vnchw_conv_ub = param_dict["vnchw_conv_ub"]
        last_stride = self.tiling_param.stride[self.tiling_param.shape_length - 1]
        float16_dtype_size = common_util.get_data_size("float16")
        multi_times = max(self.dtype_size // float16_dtype_size, 1)
        vnchw_conv_repeat_times = inst.Scalar("int64", name="vnchw_conv_repeat_times")
        vnchw_conv_repeat_times.set_as(self._ceil_div(rows_each_repeat * input_inner_dim, element_each_block))
        loop_count = inst.Scalar("int64", name="loop_count")
        loop_count.set_as((self._ceil_div(rows_each_repeat * input_inner_dim, element_each_block)) * \
                          element_each_block)
        with inst.for_range(0, loop_rows) as loop_rows_idx:
            src_addr_in = input_addr + loop_rows_idx * rows_each_repeat * input_inner_dim
            dst_addr_in = loop_rows_idx * ceil_align(rows_each_repeat * input_inner_dim, element_each_block)
            self._data_move(input_ub[dst_addr_in], input_gm[src_addr_in], rows_each_repeat * input_inner_dim)
        self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times, element_each_block)
        valid_len = (output_inner_dim - multi_times) * last_stride + multi_times
        ub2ub(inst, input_ub, vnchw_conv_ub[begin_value * element_each_block], [0, rows_each_repeat,
              valid_len, input_inner_dim - valid_len, (last_stride - 1) * multi_times])
        ub2ub(inst, vnchw_conv_ub, input_ub, [0, rows_each_repeat * output_inner_dim // multi_times, multi_times,
              (last_stride - 1) * multi_times, 0])

        vnchw_conv_repeat_times.set_as(self._ceil_div(rows_each_repeat * output_inner_dim, element_each_block))
        self._do_with_vnchwconv2output(input_ub, vnchw_conv_ub, vnchw_conv_repeat_times, element_each_block)
        self._data_move(output_gm[output_addr], input_ub, loop_rows * rows_each_repeat * output_inner_dim)

    def _do_with_vnchwconv_copy_each_rows_per_loop_last_stride_larger_than_one(self, param_dict):
        """
        slice multiple rows of input data at one loop
        """
        inst = self.tik_instance
        input_gm = param_dict["input_gm"]
        output_gm = param_dict["output_gm"]
        rows_each_repeat = param_dict["rows_each_repeat"]
        input_inner_dim = param_dict["input_inner_dim"]
        output_inner_dim = param_dict["output_inner_dim"]
        element_each_block = param_dict["element_each_block"]
        output_row_idx = param_dict["output_row_idx"]
        output_addr = output_row_idx * output_inner_dim
        loop_rows = param_dict["loop_rows"]
        input_ub = param_dict["input_ub"]
        vnchw_conv_ub = param_dict["vnchw_conv_ub"]
        input_inner_dim_copy_in = self._min(input_inner_dim, self._get_input_inner_dim_no_begin())
        last_stride = self.tiling_param.stride[self.tiling_param.shape_length - 1]
        float16_dtype_size = common_util.get_data_size("float16")
        multi_times = max(self.dtype_size // float16_dtype_size, 1)
        vnchw_conv_repeat_times = inst.Scalar("int64", name="vnchw_conv_repeat_times")
        vnchw_conv_repeat_times.set_as(self._ceil_div(rows_each_repeat * input_inner_dim, element_each_block))
        loop_count = inst.Scalar("int64", name="loop_count")
        loop_count.set_as((self._ceil_div(rows_each_repeat * input_inner_dim, element_each_block)) * element_each_block)
        with inst.for_range(0, loop_rows) as loop_rows_idx:
            with inst.for_range(0, rows_each_repeat) as rows_each_repeat_idx:
                row_idx_in_ub = loop_rows_idx * rows_each_repeat + rows_each_repeat_idx
                dst_addr_in = row_idx_in_ub * input_inner_dim
                src_addr_in = self._get_input_gm_addr(output_row_idx + row_idx_in_ub) * multi_times
                self._data_move(input_ub[dst_addr_in], input_gm[src_addr_in], input_inner_dim_copy_in)
        self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times, element_each_block)
        ub2ub(inst, input_ub, vnchw_conv_ub[0:], [0, rows_each_repeat,
              output_inner_dim * last_stride,
              input_inner_dim - output_inner_dim * last_stride, 0])
        ub2ub(inst, vnchw_conv_ub, input_ub, [0, rows_each_repeat * output_inner_dim // multi_times, multi_times,
              (last_stride - 1) * multi_times, 0])

        vnchw_conv_repeat_times.set_as(self._ceil_div(rows_each_repeat * output_inner_dim, element_each_block))
        self._do_with_vnchwconv2output(input_ub, vnchw_conv_ub, vnchw_conv_repeat_times, element_each_block)
        self._data_move(output_gm[output_addr], input_ub, loop_rows * rows_each_repeat * output_inner_dim)

    def _do_with_vnchwconv_copy_each_rows_per_loop_last_stride_align(self, param_dict):
        """
        slice multiple rows of input data at one loop
        """
        inst = self.tik_instance
        input_gm = param_dict["input_gm"]
        output_gm = param_dict["output_gm"]
        rows_each_repeat = param_dict["rows_each_repeat"]
        input_inner_dim = param_dict["input_inner_dim"]
        output_inner_dim = param_dict["output_inner_dim"]
        element_each_block = param_dict["element_each_block"]
        output_row_idx = param_dict["output_row_idx"]
        output_addr = output_row_idx * output_inner_dim
        loop_rows = param_dict["loop_rows"]
        input_ub = param_dict["input_ub"]
        vnchw_conv_ub = param_dict["vnchw_conv_ub"]
        last_stride = self.tiling_param.stride[self.tiling_param.shape_length - 1]
        ori_output_inner_dim = self.tiling_param.output_shape[self.tiling_param.shape_length - 1]
        multi_times = self.multi_times
        vnchw_conv_repeat_times = inst.Scalar("int64", name="vnchw_conv_repeat_times")
        vnchw_conv_repeat_times.set_as(self._ceil_div(rows_each_repeat * input_inner_dim, element_each_block))
        loop_count = inst.Scalar("int64", name="loop_count")
        loop_count.set_as((self._ceil_div(rows_each_repeat * input_inner_dim, element_each_block)) * element_each_block)
        with inst.for_range(0, loop_rows) as loop_rows_idx:
            with inst.for_range(0, rows_each_repeat) as rows_each_repeat_idx:
                row_idx_in_ub = loop_rows_idx * rows_each_repeat + rows_each_repeat_idx
                dst_addr_in = row_idx_in_ub * input_inner_dim
                src_addr_in = self._get_input_gm_addr(output_row_idx + row_idx_in_ub) * multi_times
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    inst.data_move_pad(input_ub[dst_addr_in:], input_gm[src_addr_in:], ori_output_inner_dim,
                                       1 * self.dtype_size, 0, (last_stride - 1) * self.dtype_size)
                else:
                    inst.data_move(input_ub[dst_addr_in:], input_gm[src_addr_in:], 0, ori_output_inner_dim, 1,
                                   last_stride * multi_times // element_each_block - 1, 0)
        self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times, element_each_block)
        ub2ub(inst, input_ub, vnchw_conv_ub, [0, rows_each_repeat * output_inner_dim // multi_times, multi_times,
              element_each_block - multi_times, 0])
        vnchw_conv_repeat_times.set_as(self._ceil_div(rows_each_repeat * output_inner_dim, element_each_block))
        self._do_with_vnchwconv2output(vnchw_conv_ub, input_ub, vnchw_conv_repeat_times, element_each_block)
        self._data_move(output_gm[output_addr], vnchw_conv_ub, loop_rows * rows_each_repeat * output_inner_dim)

    def _do_with_vnchwconv_last_stride_not_align_larger_block_element(self, param_dict):
        """
        slice multiple rows of input data at one loop
        """
        inst = self.tik_instance
        input_gm = param_dict["input_gm"]
        output_gm = param_dict["output_gm"]
        rows_each_repeat = param_dict["rows_each_repeat"]
        input_inner_dim = param_dict["input_inner_dim"]
        output_inner_dim = param_dict["output_inner_dim"]
        element_each_block = param_dict["element_each_block"]
        output_row_idx = param_dict["output_row_idx"]
        output_addr = output_row_idx * output_inner_dim
        loop_rows = param_dict["loop_rows"]
        input_ub = param_dict["input_ub"]
        vnchw_conv_ub = param_dict["vnchw_conv_ub"]
        last_stride = self.tiling_param.stride[self.tiling_param.shape_length - 1]
        ori_output_inner_dim = self.tiling_param.output_shape[self.tiling_param.shape_length - 1]
        multi_times = self.multi_times
        vnchw_conv_repeat_times = inst.Scalar("int64", name="vnchw_conv_repeat_times")
        vnchw_conv_repeat_times.set_as(self._ceil_div(rows_each_repeat * input_inner_dim, element_each_block))
        loop_count = inst.Scalar("int64", name="loop_count")
        loop_count.set_as((self._ceil_div(rows_each_repeat * input_inner_dim, element_each_block)) * element_each_block)
        with inst.for_range(0, loop_rows) as loop_rows_idx:
            with inst.for_range(0, rows_each_repeat) as rows_each_repeat_idx:
                row_idx_in_ub = loop_rows_idx * rows_each_repeat + rows_each_repeat_idx
                dst_addr_in = row_idx_in_ub * input_inner_dim
                src_addr_in = self._get_input_gm_addr(output_row_idx + row_idx_in_ub) * multi_times
                with inst.for_range(0, ori_output_inner_dim) as i:
                    dst_addr_cur_stride = dst_addr_in + i * element_each_block
                    src_addr_cur_stride = src_addr_in + i * last_stride * multi_times
                    gm2ub(inst, input_ub[dst_addr_cur_stride:], input_gm[src_addr_cur_stride:], multi_times)
        self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times, element_each_block)
        ub2ub(inst, input_ub, vnchw_conv_ub, [0, rows_each_repeat * output_inner_dim // multi_times, multi_times,
              element_each_block - multi_times, 0])
        vnchw_conv_repeat_times.set_as(self._ceil_div(rows_each_repeat * output_inner_dim, element_each_block))
        self._do_with_vnchwconv2output(vnchw_conv_ub, input_ub, vnchw_conv_repeat_times, element_each_block)
        self._data_move(output_gm[output_addr], vnchw_conv_ub, loop_rows * rows_each_repeat * output_inner_dim)

    def _do_with_vnchwconv2output(self, vnchw_conv_ub, input_ub, vnchw_conv_repeat_times, element_each_block):
        """
        rearrange data in ub with vnchwconv instruction
        """
        inst = self.tik_instance
        float16_type_size = 2
        if self.dtype_size % float16_type_size == 0:
            dst_list = [vnchw_conv_ub[i * vnchw_conv_repeat_times * element_each_block] for i in range(16)]
            src_list = [input_ub[i * element_each_block] for i in range(16)]
            with inst.if_scope(vnchw_conv_repeat_times == 1):
                inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
            with inst.else_scope():
                inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 1, 16)
        else:
            dst_list = [vnchw_conv_ub[i * vnchw_conv_repeat_times * element_each_block] for i in range(16)]
            src_list = [input_ub[i * element_each_block] for i in range(16)]
            with inst.if_scope(vnchw_conv_repeat_times == 1):
                inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
            with inst.else_scope():
                inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 1, 32)

            src_list = [input_ub[32*16 + i * element_each_block] for i in range(16)]
            with inst.if_scope(vnchw_conv_repeat_times == 1):
                inst.vnchwconv(True, False, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
            with inst.else_scope():
                inst.vnchwconv(True, False, dst_list, src_list, vnchw_conv_repeat_times, 1, 32)


    def _do_with_input2vnchwconv(self, vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times,
                                 element_each_block):
        """
        rearrange data in ub with vnchwconv instruction
        """
        inst = self.tik_instance
        float16_type_size = 2
        if self.dtype_size % float16_type_size == 0:
            dst_list = [vnchw_conv_ub[i * element_each_block] for i in range(16)]
            src_list = [input_ub[i * loop_count] for i in range(16)]
            with inst.if_scope(vnchw_conv_repeat_times == 1):
                inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
            with inst.else_scope():
                inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 16, 1)
        else:
            dst_list = [vnchw_conv_ub[i * element_each_block] for i in range(16)]
            src_list = [input_ub[i * loop_count] for i in range(16)]
            with inst.if_scope(vnchw_conv_repeat_times == 1):
                inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
            with inst.else_scope():
                inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 32, 1)

            dst_list = [vnchw_conv_ub[32 * 16 + i * element_each_block] for i in range(16)]
            with inst.if_scope(vnchw_conv_repeat_times == 1):
                inst.vnchwconv(False, True, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
            with inst.else_scope():
                inst.vnchwconv(False, True, dst_list, src_list, vnchw_conv_repeat_times, 32, 1)

    def _do_large_last_dim_multi_rows(self, core_idx):
        """
        _do_large_last_dim_multi_rows
        """
        inst = self.tik_instance
        # `out_dim = 67, num_rows_per_aicore: 3, 3, 3, 2, 2, 2 ...`
        num_rows_per_aicore = inst.Scalar("int64", name="num_rows_per_aicore")
        num_tail_rows = inst.Scalar("int64", name="num_tail_rows")
        num_rows_per_aicore.set_as(self.tiling_param.out_dim // self.core_num_scalar)
        num_tail_rows.set_as(self.tiling_param.out_dim - self.core_num_scalar * num_rows_per_aicore)

        row_idx = inst.Scalar("int64", name="row_idx")
        with inst.if_scope(core_idx < num_tail_rows):
            row_idx.set_as(core_idx + core_idx * num_rows_per_aicore)
        with inst.else_scope():
            row_idx.set_as(num_tail_rows + core_idx * num_rows_per_aicore)

        with inst.if_scope(core_idx < num_tail_rows):
            num_rows_per_aicore.set_as(num_rows_per_aicore + 1)

        with inst.if_scope(row_idx < self.tiling_param.out_dim):
            self._do_large_last_dim_multi_rows_per_aicore(row_idx, num_rows_per_aicore)

    def _do_large_last_dim_multi_rows_per_aicore(self, row_idx, num_rows):
        """
        _do_large_last_dim_multi_rows_per_aicore, input and output's last dim is align
        """
        inst = self.tik_instance
        output_shape = self.tiling_param.output_shape
        input_shape = self.tiling_param.input_shape
        dim_count = self.tiling_param.shape_length
        max_rows_per_data_move = inst.Scalar("int64", name="loops")
        max_rows_per_data_move.set_as(self.ub_size // output_shape[dim_count - 1])
        with inst.if_scope(max_rows_per_data_move > Constant.MAX_NBURST):
            max_rows_per_data_move.set_as(Constant.MAX_NBURST)
        loops = inst.Scalar("int64", name="loops")
        loops.set_as(num_rows // max_rows_per_data_move)

        row_loop_idx = inst.Scalar("int64", name="row_loop_idx")

        with inst.new_stmt_scope():
            ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="ub")
            with inst.for_range(0, loops, name="loops_for_range") as loop_idx:
                row_loop_idx.set_as(row_idx + loop_idx * max_rows_per_data_move)
                input_gm_addr = self._get_input_gm_addr(row_loop_idx)
                output_gm_addr = self._get_output_gm_addr(row_loop_idx)
                inst.data_move(ub, self.input_gm[input_gm_addr], 0, max_rows_per_data_move,
                               output_shape[dim_count - 1] // self.element_each_block,
                               (input_shape[dim_count - 1] - output_shape[dim_count - 1]) // self.element_each_block,
                               0)  # input gm -> ub

                inst.data_move(self.output_gm[output_gm_addr], ub, 0, 1,
                               max_rows_per_data_move * output_shape[dim_count - 1] // self.element_each_block, 0,
                               0)  # ub -> output gm

            num_remain_rows = num_rows - loops * max_rows_per_data_move
            with inst.if_scope(num_remain_rows > 0):
                row_loop_idx.set_as(row_idx + loops * max_rows_per_data_move)
                input_gm_addr = self._get_input_gm_addr(row_loop_idx)
                output_gm_addr = self._get_output_gm_addr(row_loop_idx)
                inst.data_move(ub, self.input_gm[input_gm_addr], 0, num_remain_rows,
                               output_shape[dim_count - 1] // self.element_each_block,
                               (input_shape[dim_count - 1] - output_shape[dim_count - 1]) // self.element_each_block,
                               0)  # input gm -> ub
                inst.data_move(self.output_gm[output_gm_addr], ub, 0, 1,
                               num_remain_rows * output_shape[dim_count - 1] // self.element_each_block, 0,
                               0)  # ub -> output gm

    def _do_large_last_dim(self, core_idx):
        self._do_large_last_dim_normal(core_idx)

    def _do_large_last_dim_normal(self, core_idx):
        """
        _do_large_last_dim_normal
        """
        inst = self.tik_instance
        output_shape = self.tiling_param.output_shape
        inner_loops = self._ceil_div(output_shape[self.shape_length - 1], self.ub_size)
        out_loops = self._ceil_div(self.tiling_param.out_dim, self.core_num_scalar)
        with inst.for_range(0, out_loops, name="out_loop") as loop_idx:
            idx = core_idx * out_loops + loop_idx
            with inst.if_scope(idx < self.tiling_param.out_dim):
                input_gm_addr = self._get_input_gm_addr(idx)
                output_gm_addr = self._get_output_gm_addr(idx)
                with inst.for_range(0, inner_loops, name="inner_loop") as inner_loop_idx:
                    with inst.if_scope(output_shape[self.shape_length - 1] % self.element_each_block == 0):
                        self._do_large_last_dim_align(input_gm_addr, output_gm_addr, inner_loop_idx)
                    with inst.else_scope():
                        self._do_large_last_dim_not_align(input_gm_addr, output_gm_addr, inner_loop_idx)

    # 'pylint: disable=invalid-name
    def _do_small_last_dim(self, core_idx):
        """
        _do_small_last_dim
        """
        inst = self.tik_instance
        output_shape = self.tiling_param.output_shape
        inner_dim = output_shape[self.shape_length - 1]
        out_dim = self.tiling_param.out_dim
        out_loops = self._ceil_div(out_dim, self.core_num_scalar)
        tmp_ub_size = self.element_each_block
        ub_size = self.ub_size - self.element_each_block
        ub_data_count = inst.Scalar("int32", name="out_ub_data_count")
        ub_data_count.set_as(0)
        input_gm = self.input_gm
        output_gm = self.output_gm
        need_update_out_addr = inst.Scalar("int32", name="need_update_out_addr")
        need_update_out_addr.set_as(1)
        output_gm_addr = inst.Scalar(self.tiling_param.dtype, name="output_addr")
        with inst.new_stmt_scope():
            tmp_ub = inst.Tensor(self.dtype, (tmp_ub_size,), scope=tik.scope_ubuf, name="tmp_ub")
            ub = inst.Tensor(self.dtype, (ub_size,), scope=tik.scope_ubuf, name="out_ub")
            with inst.for_range(0, out_loops, name="out_loop") as loop_idx:
                idx = core_idx * out_loops + loop_idx
                with inst.if_scope(idx < self.tiling_param.out_dim):
                    input_gm_addr = self._get_input_gm_addr(idx)
                    with inst.if_scope(need_update_out_addr == 1):
                        need_update_out_addr.set_as(0)
                        output_gm_addr.set_as(self._get_output_gm_addr(idx))

                    with inst.if_scope(ub_data_count + inner_dim > ub_size):
                        self._data_move(output_gm[output_gm_addr], ub, ub_data_count)
                        ub_data_count.set_as(0)
                        output_gm_addr.set_as(self._get_output_gm_addr(idx))
                    self._data_move(tmp_ub, input_gm[input_gm_addr], inner_dim)

                    with inst.for_range(0, inner_dim) as index:
                        ub[ub_data_count + index] = tmp_ub[index]
                    ub_data_count.set_as(ub_data_count + inner_dim)

                    with inst.if_scope(loop_idx == out_loops - 1):
                        self._add_tail(ub, tmp_ub, idx, ub_data_count)
            with inst.if_scope(ub_data_count != 0):
                self._data_move(output_gm[output_gm_addr], ub, ub_data_count)

    def _do_large_last_dim_align(self, input_gm_addr, output_gm_addr, inner_loop_idx):
        """
        _do_large_last_dim_align
        """
        inst = self.tik_instance
        total = self.tiling_param.output_shape[self.shape_length - 1]
        input_gm = self.input_gm
        output_gm = self.output_gm
        count = inst.Scalar("int64", name="remain")
        with inst.new_stmt_scope():
            ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="out_ub")
            count.set_as(total - self.ub_size * inner_loop_idx)
            with inst.if_scope(count > self.ub_size):
                count.set_as(self.ub_size)

            self._data_move(ub, input_gm[input_gm_addr + inner_loop_idx * self.ub_size], count)
            self._data_move(output_gm[output_gm_addr + inner_loop_idx * self.ub_size], ub, count)

    # 'pylint: disable=too-many-locals,invalid-name
    def _do_large_last_dim_not_align(self, input_gm_addr, output_gm_addr, inner_loop_idx):
        """
        _do_large_last_dim_not_align
        """
        inst = self.tik_instance
        total = self.tiling_param.output_shape[self.shape_length - 1]
        input_gm = self.input_gm
        output_gm = self.output_gm
        count = inst.Scalar("int64", name="remain")
        with inst.new_stmt_scope():
            ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="out_ub")
            count.set_as(total - self.ub_size * inner_loop_idx)
            with inst.if_scope(count >= self.ub_size):
                self._data_move(ub, input_gm[input_gm_addr + inner_loop_idx * self.ub_size], self.ub_size)
                self._data_move(output_gm[output_gm_addr + inner_loop_idx * self.ub_size], ub, self.ub_size)
            with inst.else_scope():
                with inst.if_scope(inner_loop_idx > 0):
                    align_count = self._ceil_32bytes_count(count, self.element_each_block)
                    redundant_count = align_count - count
                    new_in_start_index = (input_gm_addr + inner_loop_idx * self.ub_size - redundant_count)
                    new_out_start_index = (output_gm_addr + inner_loop_idx * self.ub_size - redundant_count)
                    self._data_move(ub, input_gm[new_in_start_index:], align_count)
                    self._data_move(output_gm[new_out_start_index:], ub, align_count)
                with inst.else_scope():
                    in_start_index = (input_gm_addr + inner_loop_idx * self.ub_size)
                    out_start_index = (output_gm_addr + inner_loop_idx * self.ub_size)
                    self._data_move(ub, input_gm[in_start_index:], self.element_each_block)
                    self._data_move(output_gm[out_start_index:], ub, self.element_each_block)

                    in_start_index += self.element_each_block
                    out_start_index += self.element_each_block
                    align_count = self._ceil_32bytes_count(count - self.element_each_block, self.element_each_block)
                    redundant_count = align_count - count + self.element_each_block
                    new_in_start_index = in_start_index - redundant_count
                    new_out_start_index = out_start_index - redundant_count
                    self._data_move(ub, input_gm[new_in_start_index:], align_count)
                    self._data_move(output_gm[new_out_start_index:], ub, align_count)

    def _add_tail(self, ub, tmp_ub, idx, ub_data_count):
        """
        _add_tail
        """
        inst = self.tik_instance
        inner_dim = self.tiling_param.output_shape[self.shape_length - 1]
        out_dim = self.tiling_param.out_dim
        align_count = inst.Scalar(init_value=self._ceil_32bytes_count(ub_data_count, self.element_each_block))
        overlap_count = inst.Scalar(init_value=align_count - ub_data_count)
        ext_rows = inst.Scalar(init_value=self._ceil_div(overlap_count, inner_dim))
        input_gm = self.input_gm
        with inst.for_range(1, ext_rows + 1, name="ext_row") as row_idx:
            with inst.if_scope(idx + row_idx < out_dim):
                input_addr = self._get_input_gm_addr(idx + row_idx)
                self._data_move(tmp_ub, input_gm[input_addr], inner_dim)
                with inst.for_range(0, inner_dim) as index:
                    with inst.if_scope(ub_data_count < align_count):
                        ub[ub_data_count] = tmp_ub[index]
                        ub_data_count.set_as(ub_data_count + 1)

    def _add_tail_last_stride_larger1(self, ub, tmp_ub, idx, ub_data_count):
        """
        _add_tail
        """
        inst = self.tik_instance
        inner_dim = self.tiling_param.output_shape[self.shape_length - 1]
        out_dim = self.tiling_param.out_dim
        last_stride = self.tiling_param.stride[self.shape_length - 1]
        align_count = inst.Scalar(init_value=self._ceil_32bytes_count(ub_data_count, self.element_each_block))
        overlap_count = inst.Scalar(init_value=align_count - ub_data_count)
        ext_rows = inst.Scalar(init_value=self._ceil_div(overlap_count, inner_dim))
        input_gm = self.input_gm
        with inst.for_range(1, ext_rows + 1, name="ext_row") as row_idx:
            with inst.if_scope(idx + row_idx < out_dim):
                input_addr = self._get_input_gm_addr(idx + row_idx)
                with inst.for_range(0, inner_dim) as inner_idx:
                    input_addr = input_addr + inner_idx * last_stride
                    self._data_move(tmp_ub, input_gm[input_addr], 1)
                    with inst.if_scope(ub_data_count < align_count):
                        ub[ub_data_count] = tmp_ub[0]
                        ub_data_count.set_as(ub_data_count + 1)

    def _do_small_last_dim_with_vnchwconv(self, core_idx):
        """
        _do_small_last_dim_with_vnchwconv
        """
        inst = self.tik_instance
        input_shape = self.tiling_param.input_shape
        output_shape = self.tiling_param.output_shape
        out_loops = self._ceil_div(self.tiling_param.out_dim_with_vnchwconv, self.core_num_scalar)
        compute_rows_each_inner_loops = self.ub_size_with_vnchwconv // (16 *
                                                                        input_shape[self.shape_length - 1]) // 16 * 16

        inner_loops = self._ceil_div(output_shape[self.shape_length - 2], compute_rows_each_inner_loops) - 1
        compute_rows_tail = output_shape[self.shape_length - 2] - inner_loops * compute_rows_each_inner_loops
        with inst.new_stmt_scope():
            ub1 = inst.Tensor(self.dtype, (self.ub_size_with_vnchwconv,), scope=tik.scope_ubuf, name="ub1")
            ub2 = inst.Tensor(self.dtype, (self.ub_size_with_vnchwconv,), scope=tik.scope_ubuf, name="ub2")
            with inst.for_range(0, out_loops, name="out_loops") as out_loops_idx:
                idx = core_idx * out_loops + out_loops_idx
                with inst.if_scope(idx < self.tiling_param.out_dim_with_vnchwconv):
                    input_gm_base_addr = self._get_input_base_gm_addr_with_vnchwconv(idx)
                    output_gm_base_addr = self._get_output_base_gm_addr_with_vnchwconv(idx)
                    with inst.for_range(0, inner_loops, name="inner_loops") as inner_loops_idx:
                        input_gm_addr = input_gm_base_addr + inner_loops_idx * \
                            compute_rows_each_inner_loops * input_shape[self.shape_length - 1]
                        output_gm_addr = output_gm_base_addr + inner_loops_idx * \
                            compute_rows_each_inner_loops * output_shape[self.shape_length - 1]

                        self._do_each_matrix_align(input_gm_addr, output_gm_addr, compute_rows_each_inner_loops, \
                                                   ub1, ub2)

                    input_gm_addr = input_gm_base_addr + inner_loops * compute_rows_each_inner_loops * input_shape[
                        self.shape_length - 1]
                    output_gm_addr = output_gm_base_addr + inner_loops * compute_rows_each_inner_loops * output_shape[
                        self.shape_length - 1]
                    param_list = [input_gm_addr, output_gm_addr, compute_rows_tail, out_loops_idx, out_loops, ub1, ub2]
                    self._do_each_matrix_tail(param_list)

    def _get_input_base_gm_addr_with_vnchwconv(self, cur_index: tik.Scalar):
        """
        _get_input_base_gm_addr_vnchw
        """
        inst = self.tik_instance
        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        dim_count = self.tiling_param.shape_length
        addr = inst.Scalar(self.tiling_param.dtype, name="input_addr")
        addr.set_as(self.tiling_param.begin[dim_count - 2] * self.tiling_param.input_shape[dim_count - 1])
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")

        with inst.for_range(3, dim_count + 1) as dim_idx:
            dim = self.tiling_param.output_shape[dim_count - dim_idx]
            step = self.tiling_param.input_steps[dim_idx - 2]
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * (tmp + self.tiling_param.begin[dim_count - dim_idx]))
            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _get_output_base_gm_addr_with_vnchwconv(self, cur_index: tik.Scalar):
        """
        _get_output_base_gm_addr_vnchw
        """
        inst = self.tik_instance
        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        dim_count = self.tiling_param.shape_length
        addr = inst.Scalar(self.tiling_param.dtype, name="output_addr")
        addr.set_as(0)
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")

        with inst.for_range(3, dim_count + 1) as dim_idx:
            dim = self.tiling_param.output_shape[dim_count - dim_idx]
            step = self.tiling_param.output_steps[dim_idx - 2]
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * tmp)
            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _do_each_matrix_align(self, input_gm_addr, output_gm_addr, rows, ub1, ub2):
        """
        _do_each_matrix_align
        """
        output_matrix_count = self.tiling_param.output_shape[self.shape_length - 1] * rows
        self._do_each_matrix_except_move_output_gm(input_gm_addr, output_gm_addr, rows, [ub1, ub2])
        self._data_move(self.output_gm[output_gm_addr], ub2, output_matrix_count)

    def _do_each_matrix_except_move_output_gm(self, input_gm_addr, output_gm_addr, rows, ub_list):
        """
        _do_each_matrix_except_move_output_gm
        """
        inst = self.tik_instance
        input_shape = self.tiling_param.input_shape
        output_shape = self.tiling_param.output_shape
        begin = self.tiling_param.begin
        ub1, ub2 = ub_list
        input_matrix_count = input_shape[self.shape_length - 1] * rows
        self._data_move(ub1, self.input_gm[input_gm_addr], input_matrix_count)

        # first vnchwconv_loop: ub1(32, 31) -> ub2(32 * 31, 16)
        vnchwconv_loop = self._ceil_div(input_matrix_count, 16)
        with inst.for_range(0, vnchwconv_loop) as i:
            src_addr = [ub1[16 * i + 16 * j] for j in range(16)]
            dst_addr = [ub2[16 * 16 * i + 16 * j] for j in range(16)]
            inst.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

        nburst_loop = rows // Constant.MAX_NBURST
        with inst.for_range(0, nburst_loop) as i:
            ub2ub(inst,
                  ub1[i * Constant.MAX_NBURST * output_shape[self.shape_length - 1] * 16],
                  ub2[(i * Constant.MAX_NBURST * input_shape[self.shape_length - 1] + begin[
                      self.shape_length - 1]) * 16],
                  [0, Constant.MAX_NBURST, output_shape[self.shape_length - 1],
                  input_shape[self.shape_length - 1] - output_shape[self.shape_length - 1], 0])

        with inst.if_scope(rows % Constant.MAX_NBURST != 0):
            ub2ub(inst,
                  ub1[nburst_loop * Constant.MAX_NBURST * output_shape[self.shape_length - 1] * 16],
                  ub2[(nburst_loop * Constant.MAX_NBURST * input_shape[self.shape_length - 1] +
                       begin[self.shape_length - 1]) * 16], [0, rows % Constant.MAX_NBURST,
                  output_shape[self.shape_length - 1],
                  input_shape[self.shape_length - 1] - output_shape[self.shape_length - 1], 0])

        output_matrix_count = output_shape[self.shape_length - 1] * rows
        vnchwconv_loop = self._ceil_div(output_matrix_count, 16)
        with inst.for_range(0, vnchwconv_loop) as i:
            src_addr = [ub1[16 * 16 * i + 16 * j] for j in range(16)]
            dst_addr = [ub2[16 * i + 16 * j] for j in range(16)]
            inst.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

    def _do_each_matrix_tail(self, param_list):
        """
        _do_each_matrix_tail
        """
        inst = self.tik_instance
        input_gm_addr, output_gm_addr, rows, out_loops_idx, out_loops, ub1, ub2 = param_list
        output_matrix_count = self.tiling_param.output_shape[self.shape_length - 1] * rows
        ub_block = inst.Tensor(self.dtype, (self.element_each_block,), scope=tik.scope_ubuf, name="ub_block")

        self._do_each_matrix_except_move_output_gm(input_gm_addr, output_gm_addr, rows, [ub1, ub2])

        with inst.if_scope(tik.all(out_loops_idx == out_loops - 1,
                                   output_matrix_count > self.element_each_block,
                                   output_matrix_count % self.element_each_block != 0)):
            floor_align_count = output_matrix_count // self.element_each_block * self.element_each_block
            self._data_move(self.output_gm[output_gm_addr], ub2, floor_align_count)
            with inst.for_range(0, self.element_each_block, name="block_element_loop") as element_id:
                ub_block[element_id] = ub2[output_matrix_count - self.element_each_block + element_id]
            self._data_move(self.output_gm[output_gm_addr + output_matrix_count - self.element_each_block], ub_block,
                            self.element_each_block)

        with inst.else_scope():
            self._data_move(self.output_gm[output_gm_addr], ub2, output_matrix_count)

    def _do_only_last_stride_larger1_out_continual_with_vnchwconv(self, core_idx):
        inst = self.tik_instance
        dtype_size = self.dtype_size
        output_shape = self.tiling_param.output_shape
        input_shape = self.tiling_param.input_shape
        shape_length = self.tiling_param.shape_length
        float16_dtype_size = 2
        multi_times = max(dtype_size // float16_dtype_size, 1)
        if dtype_size % float16_dtype_size == 0:
            vnchwconv_column = 16
            tensor_dtype = "float16"
        else:
            vnchwconv_column = 32
            tensor_dtype = "int8"

        input_inner_dim = input_shape[shape_length - 1] * multi_times
        output_inner_dim = output_shape[shape_length - 1] * multi_times
        out_dim = self.tiling_param.out_dim
        ub_size = self.ub_size * multi_times // 2
        begin_value = self.tiling_param.begin[shape_length - 1] * multi_times
        input_addr_start = input_inner_dim * self.tiling_param.begin[0]
        element_each_block = self.element_each_block * multi_times
        output_32bytes_align_rows = self._get_32bytes_align_rows()
        reserve_input_32bytes_align_ub = 0
        max_rows_in_ub = (ub_size - reserve_input_32bytes_align_ub) // vnchwconv_column // \
                         ceil_align(input_inner_dim * output_32bytes_align_rows,
                                    element_each_block) * output_32bytes_align_rows
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        rows_each_repeat = inst.Scalar("int64", name="rows_each_repeat")
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        tail_rows_repeat_times = inst.Scalar("int64", name="tail_rows_repeat_times")
        tail_rows_repeat_tail_count = inst.Scalar("int64", name="tail_rows_repeat_tail_count")
        tail_rows_repeat_roll_back_rows = inst.Scalar("int64", name="tail_rows_repeat_roll_back_rows", init_value=0)
        rows_each_core.set_as(
            self._ceil_32bytes_count(self._ceil_div(out_dim, self.core_num_scalar), output_32bytes_align_rows))
        repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
        rows_each_repeat.set_as(
            self._ceil_32bytes_count(self._ceil_div(rows_each_core, repeat_times), output_32bytes_align_rows))
        rows_each_core.set_as(rows_each_repeat * repeat_times)
        aicore_num_used.set_as(self._ceil_div(out_dim, rows_each_core))
        with inst.if_scope(rows_each_core > out_dim):
            rows_each_core.set_as(out_dim)
            repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
            rows_each_repeat.set_as(
                self._ceil_32bytes_count(self._ceil_div(rows_each_core, repeat_times), output_32bytes_align_rows))
            with inst.if_scope(rows_each_repeat):
                rows_each_repeat.set_as(out_dim)

        loop_times = self._ceil_div(repeat_times, 16)
        last_loop_rows = inst.Scalar("int64", name="last_loop_rows", init_value=repeat_times % 16)
        with inst.if_scope(last_loop_rows == 0):
            last_loop_rows.set_as(16)

        tail_rows.set_as(out_dim % rows_each_core)
        tail_rows_repeat_times.set_as(self._ceil_div(tail_rows, rows_each_repeat))
        tail_rows_repeat_tail_count.set_as(tail_rows % rows_each_repeat)
        with inst.if_scope(tail_rows_repeat_tail_count == 0):
            tail_rows_repeat_tail_count.set_as(rows_each_repeat)
        with inst.if_scope(tail_rows_repeat_tail_count % output_32bytes_align_rows != 0):
            tail_rows_repeat_roll_back_rows.set_as(self._ceil_32bytes_count(tail_rows_repeat_tail_count,
                                                                            output_32bytes_align_rows) - \
                                                   tail_rows_repeat_tail_count)
        tail_loop_times = self._ceil_div(tail_rows_repeat_times - 1, 16)
        tail_last_loop_rows = inst.Scalar("int64",
                                          name="tail_last_loop_rows",
                                          init_value=(tail_rows_repeat_times - 1) % 16)
        with inst.if_scope(tail_last_loop_rows == 0):
            tail_last_loop_rows.set_as(16)

        if dtype_size % float16_dtype_size == 0:
            input_gm = self.input_gm.reinterpret_cast_to("float16")
            output_gm = self.output_gm.reinterpret_cast_to("float16")
        else:
            input_gm = self.input_gm
            output_gm = self.output_gm

        input_addr = inst.Scalar("int64", name="input_addr")
        output_addr = inst.Scalar("int64", name="output_addr")
        param_dict = {
            "input_gm": input_gm,
            "output_gm": output_gm,
            "rows_each_repeat": rows_each_repeat,
            "input_inner_dim": input_inner_dim,
            "output_inner_dim": output_inner_dim,
            "begin_value": begin_value,
            "element_each_block": element_each_block
        }

        with inst.new_stmt_scope():
            input_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="vnchw_conv_ub")
            param_dict["input_ub"] = input_ub
            param_dict["vnchw_conv_ub"] = vnchw_conv_ub
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tail_rows != 0):
                    with inst.if_scope(core_idx < aicore_num_used - 1):
                        with inst.for_range(0, loop_times - 1) as loop_idx:
                            input_addr.set_as(input_addr_start + rows_each_core * input_inner_dim * core_idx + \
                                              loop_idx * rows_each_repeat * input_inner_dim * 16)
                            output_addr.set_as(rows_each_core * output_inner_dim * core_idx + \
                                               loop_idx * rows_each_repeat * output_inner_dim * 16)
                            param_dict["input_addr"] = input_addr
                            param_dict["output_addr"] = output_addr
                            param_dict["loop_rows"] = 16
                            self._do_with_vnchwconv_copy_multi_rows_per_loop_last_stride_larger_than_one(param_dict)
                        input_addr.set_as(input_addr_start + rows_each_core * input_inner_dim * core_idx + \
                                          (loop_times - 1) * rows_each_repeat * input_inner_dim * 16)
                        output_addr.set_as(rows_each_core * output_inner_dim * core_idx + \
                                           (loop_times - 1) * 16 * rows_each_repeat * output_inner_dim)
                        param_dict["input_addr"] = input_addr
                        param_dict["output_addr"] = output_addr
                        param_dict["loop_rows"] = last_loop_rows
                        self._do_with_vnchwconv_copy_multi_rows_per_loop_last_stride_larger_than_one(param_dict)
                    with inst.else_scope():
                        with inst.if_scope(tail_rows_repeat_times > 1):
                            with inst.for_range(0, tail_loop_times - 1) as loop_idx:
                                input_addr.set_as(input_addr_start + rows_each_core * input_inner_dim * (
                                            aicore_num_used - 1) + loop_idx * rows_each_repeat * input_inner_dim * 16)
                                output_addr.set_as(rows_each_core * output_inner_dim * (aicore_num_used - 1) +
                                                   loop_idx * rows_each_repeat * output_inner_dim * 16)
                                param_dict["input_addr"] = input_addr
                                param_dict["output_addr"] = output_addr
                                param_dict["loop_rows"] = 16
                                self._do_with_vnchwconv_copy_multi_rows_per_loop_last_stride_larger_than_one(param_dict)
                            input_addr.set_as(
                                input_addr_start + rows_each_core * input_inner_dim * (aicore_num_used - 1) + (
                                            tail_loop_times - 1) * rows_each_repeat * input_inner_dim * 16)
                            output_addr.set_as(rows_each_core * output_inner_dim * (aicore_num_used - 1) +
                                               (tail_loop_times - 1) * 16 * rows_each_repeat * output_inner_dim)
                            param_dict["input_addr"] = input_addr
                            param_dict["output_addr"] = output_addr
                            param_dict["loop_rows"] = tail_last_loop_rows
                            self._do_with_vnchwconv_copy_multi_rows_per_loop_last_stride_larger_than_one(param_dict)
                        input_addr.set_as(input_addr_start + rows_each_core * input_inner_dim * (aicore_num_used - 1) +
                                          (tail_rows_repeat_times - 1) * rows_each_repeat * input_inner_dim -
                                          tail_rows_repeat_roll_back_rows * input_inner_dim)
                        output_addr.set_as(rows_each_core * output_inner_dim * (aicore_num_used - 1) +
                                           (tail_rows_repeat_times - 1) * rows_each_repeat * output_inner_dim -
                                           tail_rows_repeat_roll_back_rows * output_inner_dim)
                        param_dict["input_addr"] = input_addr
                        param_dict["output_addr"] = output_addr
                        param_dict["loop_rows"] = 1
                        param_dict["rows_each_repeat"].set_as(tail_rows_repeat_tail_count +
                                                              tail_rows_repeat_roll_back_rows)
                        self._do_with_vnchwconv_copy_multi_rows_per_loop_last_stride_larger_than_one(param_dict)
                with inst.else_scope():
                    with inst.for_range(0, loop_times - 1) as loop_idx:
                        input_addr.set_as(input_addr_start + rows_each_core * input_inner_dim * core_idx +
                                          loop_idx * rows_each_repeat * input_inner_dim * 16)
                        output_addr.set_as(rows_each_core * output_inner_dim * core_idx +
                                           loop_idx * 16 * rows_each_repeat * output_inner_dim)
                        param_dict["input_addr"] = input_addr
                        param_dict["output_addr"] = output_addr
                        param_dict["loop_rows"] = 16
                        self._do_with_vnchwconv_copy_multi_rows_per_loop_last_stride_larger_than_one(param_dict)
                    input_addr.set_as(input_addr_start + rows_each_core * input_inner_dim * core_idx +
                                      (loop_times - 1) * rows_each_repeat * input_inner_dim * 16)
                    output_addr.set_as(rows_each_core * output_inner_dim * core_idx +
                                       (loop_times - 1) * 16 * rows_each_repeat * output_inner_dim)
                    param_dict["input_addr"] = input_addr
                    param_dict["output_addr"] = output_addr
                    param_dict["loop_rows"] = last_loop_rows
                    self._do_with_vnchwconv_copy_multi_rows_per_loop_last_stride_larger_than_one(param_dict)

    def _do_last_stride_larger1_small_inout_inner_with_vnchwconv(self, core_idx):
        inst = self.tik_instance
        dtype_size = self.dtype_size
        output_shape = self.tiling_param.output_shape
        shape_length = self.tiling_param.shape_length
        float16_dtype_size = 2
        multi_times = max(dtype_size // float16_dtype_size, 1)
        if dtype_size % float16_dtype_size == 0:
            vnchwconv_column = 16
            tensor_dtype = "float16"
        else:
            vnchwconv_column = 32
            tensor_dtype = "int8"

        last_stride = self.tiling_param.stride[shape_length - 1]
        output_inner_dim = output_shape[shape_length - 1] * multi_times
        input_inner_dim = ceil_align(output_shape[shape_length - 1] * last_stride,
                                     self.element_each_block) * multi_times
        out_dim = self.tiling_param.out_dim
        ub_size = self.ub_size * multi_times // 2
        begin_value = self.tiling_param.begin[shape_length - 1] * multi_times
        element_each_block = self.element_each_block * multi_times
        output_32bytes_align_rows = self._get_32bytes_align_rows()
        reserve_input_32bytes_align_ub = 0
        max_rows_in_ub = floor_align((ub_size - reserve_input_32bytes_align_ub) // (input_inner_dim * vnchwconv_column),
                                     output_32bytes_align_rows)
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        rows_each_repeat = inst.Scalar("int64", name="rows_each_repeat")
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        tail_rows_repeat_times = inst.Scalar("int64", name="tail_rows_repeat_times")
        tail_rows_repeat_tail_count = inst.Scalar("int64", name="tail_rows_repeat_tail_count")
        tail_rows_repeat_roll_back_rows = inst.Scalar("int64", name="tail_rows_repeat_roll_back_rows", init_value=0)
        rows_each_core.set_as(
            self._ceil_32bytes_count(self._ceil_div(out_dim, self.core_num_scalar), output_32bytes_align_rows))
        repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
        rows_each_repeat.set_as(
            self._ceil_32bytes_count(self._ceil_div(rows_each_core, repeat_times), output_32bytes_align_rows))
        rows_each_core.set_as(rows_each_repeat * repeat_times)
        aicore_num_used.set_as(self._ceil_div(out_dim, rows_each_core))
        with inst.if_scope(rows_each_core > out_dim):
            rows_each_core.set_as(out_dim)
            repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
            rows_each_repeat.set_as(
                self._ceil_32bytes_count(self._ceil_div(rows_each_core, repeat_times), output_32bytes_align_rows))
            with inst.if_scope(rows_each_repeat > out_dim):
                rows_each_repeat.set_as(out_dim)
        loop_times = self._ceil_div(repeat_times, 16)
        last_loop_rows = inst.Scalar("int64", name="last_loop_rows", init_value=repeat_times % 16)
        with inst.if_scope(last_loop_rows == 0):
            last_loop_rows.set_as(16)

        tail_rows.set_as(out_dim % rows_each_core)
        tail_rows_repeat_times.set_as(self._ceil_div(tail_rows, rows_each_repeat))
        tail_rows_repeat_tail_count.set_as(tail_rows % rows_each_repeat)
        with inst.if_scope(tail_rows_repeat_tail_count == 0):
            tail_rows_repeat_tail_count.set_as(rows_each_repeat)
        with inst.if_scope(tail_rows_repeat_tail_count % output_32bytes_align_rows != 0):
            tail_rows_repeat_roll_back_rows.set_as(self._ceil_32bytes_count(tail_rows_repeat_tail_count,
                                                                            output_32bytes_align_rows) -
                                                   tail_rows_repeat_tail_count)
        tail_loop_times = self._ceil_div(tail_rows_repeat_times - 1, 16)
        tail_last_loop_rows = inst.Scalar("int64",
                                          name="tail_last_loop_rows",
                                          init_value=(tail_rows_repeat_times - 1) % 16)
        with inst.if_scope(tail_last_loop_rows == 0):
            tail_last_loop_rows.set_as(16)

        if dtype_size % float16_dtype_size == 0:
            input_gm = self.input_gm.reinterpret_cast_to("float16")
            output_gm = self.output_gm.reinterpret_cast_to("float16")
        else:
            input_gm = self.input_gm
            output_gm = self.output_gm

        output_row_idx = inst.Scalar("int64", name="output_row_idx")
        param_dict = {
            "input_gm": input_gm,
            "output_gm": output_gm,
            "rows_each_repeat": rows_each_repeat,
            "input_inner_dim": input_inner_dim,
            "output_inner_dim": output_inner_dim,
            "begin_value": begin_value,
            "element_each_block": element_each_block
        }

        with inst.new_stmt_scope():
            input_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="vnchw_conv_ub")
            param_dict["input_ub"] = input_ub
            param_dict["vnchw_conv_ub"] = vnchw_conv_ub
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tail_rows != 0):
                    with inst.if_scope(core_idx < aicore_num_used - 1):
                        with inst.for_range(0, loop_times - 1) as loop_idx:
                            output_row_idx.set_as(rows_each_core * core_idx + \
                                                  loop_idx * rows_each_repeat * 16)
                            param_dict["output_row_idx"] = output_row_idx
                            param_dict["loop_rows"] = 16
                            self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_larger_than_one(param_dict)
                        output_row_idx.set_as(rows_each_core * core_idx + (loop_times - 1) * 16 * rows_each_repeat)
                        param_dict["output_row_idx"] = output_row_idx
                        param_dict["loop_rows"] = last_loop_rows
                        self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_larger_than_one(param_dict)
                    with inst.else_scope():
                        with inst.if_scope(tail_rows_repeat_times > 1):
                            with inst.for_range(0, tail_loop_times - 1) as loop_idx:
                                output_row_idx.set_as(rows_each_core * (aicore_num_used - 1) + \
                                                      loop_idx * rows_each_repeat * 16)
                                param_dict["output_row_idx"] = output_row_idx
                                param_dict["loop_rows"] = 16
                                self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_larger_than_one(param_dict)
                            output_row_idx.set_as(rows_each_core * (aicore_num_used - 1) + \
                                                  (tail_loop_times - 1) * 16 * rows_each_repeat)
                            param_dict["output_row_idx"] = output_row_idx
                            param_dict["loop_rows"] = tail_last_loop_rows
                            self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_larger_than_one(param_dict)
                        output_row_idx.set_as(
                            rows_each_core * (aicore_num_used - 1) + (tail_rows_repeat_times - 1) * rows_each_repeat -
                            tail_rows_repeat_roll_back_rows)
                        param_dict["output_row_idx"] = output_row_idx
                        param_dict["loop_rows"] = 1
                        param_dict["rows_each_repeat"].set_as(tail_rows_repeat_tail_count +
                                                              tail_rows_repeat_roll_back_rows)
                        self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_larger_than_one(param_dict)
                with inst.else_scope():
                    with inst.for_range(0, loop_times - 1) as loop_idx:
                        output_row_idx.set_as(rows_each_core * core_idx + loop_idx * 16 * rows_each_repeat)
                        param_dict["output_row_idx"] = output_row_idx
                        param_dict["loop_rows"] = 16
                        self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_larger_than_one(param_dict)
                    output_row_idx.set_as(rows_each_core * core_idx + (loop_times - 1) * 16 * rows_each_repeat)
                    param_dict["output_row_idx"] = output_row_idx
                    param_dict["loop_rows"] = last_loop_rows
                    self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_larger_than_one(param_dict)

    def _do_last_stride_larger1_large_out_inner_with_vnchwconv_copy_multi_strides(self, core_idx):
        inst = self.tik_instance
        output_shape = self.tiling_param.output_shape
        shape_length = self.tiling_param.shape_length
        multi_times = self.multi_times
        tensor_dtype = self.tensor_dtype
        last_stride = self.tiling_param.stride[shape_length - 1]
        output_inner_dim = output_shape[shape_length - 1] * multi_times
        ub_size = floor_align(self.ub_size * multi_times // 2, self.tensor_block_element)
        output_inner_dim_factor = inst.Scalar(
            init_value=floor_align(ub_size // self.vnchwconv_column // (last_stride * multi_times),
                                   self.tensor_block_element))
        with inst.if_scope(output_inner_dim_factor > output_inner_dim):
            output_inner_dim_factor.set_as(floor_align(output_inner_dim, self.tensor_block_element))
        out_dim = self.tiling_param.out_dim
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        rows_each_repeat = inst.Scalar("int64", name="rows_each_repeat", init_value=self.vnchwconv_rows)
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        rows_each_core.set_as(ceil_align(self._ceil_div(out_dim, self.core_num_scalar), self.vnchwconv_rows))
        repeat_times.set_as(self._ceil_div(rows_each_core, self.vnchwconv_rows))
        aicore_num_used.set_as(self._ceil_div(out_dim, rows_each_core))
        with inst.if_scope(rows_each_core > out_dim):
            rows_each_core.set_as(out_dim)
            repeat_times.set_as(self._ceil_div(rows_each_core, self.vnchwconv_rows))
            with inst.if_scope(rows_each_repeat > out_dim):
                rows_each_repeat.set_as(out_dim)
        tail_rows.set_as(out_dim % rows_each_core)

        with inst.new_stmt_scope():
            input_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="vnchw_conv_ub")
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tik.all(tail_rows != 0, core_idx == aicore_num_used - 1)):
                    repeat_times.set_as(ceil_div(tail_rows, self.vnchwconv_rows))
                    with inst.for_range(0, repeat_times - 1) as repeat_idx:
                        start_output_row_idx = rows_each_core * core_idx + self.vnchwconv_rows * repeat_idx
                        self._copy_each_row_with_vnchwconv(start_output_row_idx, output_inner_dim_factor,
                                                           self.vnchwconv_rows, input_ub, vnchw_conv_ub,
                                                           self._copy_part_each_row_with_vnchwconv)

                    start_output_row_idx = rows_each_core * core_idx + self.vnchwconv_rows * (repeat_times - 1)
                    rows = inst.Scalar(init_value=tail_rows % self.vnchwconv_rows)
                    with inst.if_scope(rows == 0):
                        rows.set_as(self.vnchwconv_rows)
                    self._copy_each_row_with_vnchwconv(start_output_row_idx, output_inner_dim_factor, rows, input_ub,
                                                       vnchw_conv_ub, self._copy_part_each_row_with_vnchwconv)
                with inst.else_scope():
                    with inst.for_range(0, repeat_times) as repeat_idx:
                        start_output_row_idx = rows_each_core * core_idx + self.vnchwconv_rows * repeat_idx
                        self._copy_each_row_with_vnchwconv(start_output_row_idx, output_inner_dim_factor,
                                                           self.vnchwconv_rows, input_ub, vnchw_conv_ub,
                                                           self._copy_part_each_row_with_vnchwconv)

    def _do_last_stride_larger_block_element_large_out_inner_with_vnchwconv_copy_block_by_block(self, core_idx):
        inst = self.tik_instance
        output_shape = self.tiling_param.output_shape
        shape_length = self.tiling_param.shape_length
        multi_times = self.multi_times
        tensor_dtype = self.tensor_dtype
        output_inner_dim = output_shape[shape_length - 1] * multi_times
        ub_size = floor_align(self.ub_size * multi_times // 2, self.tensor_block_element)
        output_inner_dim_factor = inst.Scalar(
            init_value=floor_align(ub_size // self.vnchwconv_column // (self.tensor_block_element * multi_times),
                                   self.tensor_block_element))
        with inst.if_scope(output_inner_dim_factor > output_inner_dim):
            output_inner_dim_factor.set_as(floor_align(output_inner_dim, self.tensor_block_element))
        out_dim = self.tiling_param.out_dim
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        rows_each_repeat = inst.Scalar("int64", name="rows_each_repeat", init_value=self.vnchwconv_rows)
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        rows_each_core.set_as(ceil_align(self._ceil_div(out_dim, self.core_num_scalar), self.vnchwconv_rows))
        repeat_times.set_as(self._ceil_div(rows_each_core, self.vnchwconv_rows))
        aicore_num_used.set_as(self._ceil_div(out_dim, rows_each_core))
        with inst.if_scope(rows_each_core > out_dim):
            rows_each_core.set_as(out_dim)
            repeat_times.set_as(self._ceil_div(rows_each_core, self.vnchwconv_rows))
            with inst.if_scope(rows_each_repeat > out_dim):
                rows_each_repeat.set_as(out_dim)
        tail_rows.set_as(out_dim % rows_each_core)

        with inst.new_stmt_scope():
            input_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="vnchw_conv_ub")
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tik.all(tail_rows != 0, core_idx == aicore_num_used - 1)):
                    repeat_times.set_as(ceil_div(tail_rows, self.vnchwconv_rows))
                    with inst.for_range(0, repeat_times - 1) as repeat_idx:
                        start_output_row_idx = rows_each_core * core_idx + self.vnchwconv_rows * repeat_idx
                        self._copy_each_row_with_vnchwconv(start_output_row_idx, output_inner_dim_factor,
                                                           self.vnchwconv_rows, input_ub, vnchw_conv_ub,
                                                           self._do_part_each_row_with_vnchwconv_block_by_block)

                    start_output_row_idx = rows_each_core * core_idx + self.vnchwconv_rows * (repeat_times - 1)
                    rows = inst.Scalar(init_value=tail_rows % self.vnchwconv_rows)
                    with inst.if_scope(rows == 0):
                        rows.set_as(self.vnchwconv_rows)
                    self._copy_each_row_with_vnchwconv(start_output_row_idx, output_inner_dim_factor, rows, input_ub,
                                                       vnchw_conv_ub,
                                                       self._do_part_each_row_with_vnchwconv_block_by_block)
                with inst.else_scope():
                    with inst.for_range(0, repeat_times) as repeat_idx:
                        start_output_row_idx = rows_each_core * core_idx + self.vnchwconv_rows * repeat_idx
                        self._copy_each_row_with_vnchwconv(start_output_row_idx, output_inner_dim_factor,
                                                           self.vnchwconv_rows, input_ub, vnchw_conv_ub,
                                                           self._do_part_each_row_with_vnchwconv_block_by_block)

    def _do_last_stride_block_element_align_large_out_inner_with_vnchwconv(self, core_idx):
        inst = self.tik_instance
        output_shape = self.tiling_param.output_shape
        shape_length = self.tiling_param.shape_length
        multi_times = self.multi_times
        tensor_dtype = self.tensor_dtype
        output_inner_dim = output_shape[shape_length - 1] * multi_times
        ub_size = floor_align(self.ub_size * multi_times // 2, self.tensor_block_element)
        output_inner_dim_factor = inst.Scalar(
            init_value=floor_align(ub_size // self.vnchwconv_column // (self.tensor_block_element * multi_times),
                                   self.tensor_block_element))
        with inst.if_scope(output_inner_dim_factor > output_inner_dim):
            output_inner_dim_factor.set_as(floor_align(output_inner_dim, self.tensor_block_element))
        out_dim = self.tiling_param.out_dim
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        rows_each_repeat = inst.Scalar("int64", name="rows_each_repeat", init_value=self.vnchwconv_rows)
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        rows_each_core.set_as(ceil_align(self._ceil_div(out_dim, self.core_num_scalar), self.vnchwconv_rows))
        repeat_times.set_as(self._ceil_div(rows_each_core, self.vnchwconv_rows))
        aicore_num_used.set_as(self._ceil_div(out_dim, rows_each_core))
        with inst.if_scope(rows_each_core > out_dim):
            rows_each_core.set_as(out_dim)
            repeat_times.set_as(self._ceil_div(rows_each_core, self.vnchwconv_rows))
            with inst.if_scope(rows_each_repeat > out_dim):
                rows_each_repeat.set_as(out_dim)
        tail_rows.set_as(out_dim % rows_each_core)

        with inst.new_stmt_scope():
            input_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="vnchw_conv_ub")
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tik.all(tail_rows != 0, core_idx == aicore_num_used - 1)):
                    repeat_times.set_as(ceil_div(tail_rows, self.vnchwconv_rows))
                    with inst.for_range(0, repeat_times - 1) as repeat_idx:
                        start_output_row_idx = rows_each_core * core_idx + self.vnchwconv_rows * repeat_idx
                        self._copy_each_row_with_vnchwconv(start_output_row_idx, output_inner_dim_factor,
                                                           self.vnchwconv_rows, input_ub, vnchw_conv_ub,
                                                           self._do_part_each_row_with_vnchwconv_repeat_by_block)

                    start_output_row_idx = rows_each_core * core_idx + self.vnchwconv_rows * (repeat_times - 1)
                    rows = inst.Scalar(init_value=tail_rows % self.vnchwconv_rows)
                    with inst.if_scope(rows == 0):
                        rows.set_as(self.vnchwconv_rows)
                    self._copy_each_row_with_vnchwconv(start_output_row_idx, output_inner_dim_factor, rows, input_ub,
                                                       vnchw_conv_ub,
                                                       self._do_part_each_row_with_vnchwconv_repeat_by_block)
                with inst.else_scope():
                    with inst.for_range(0, repeat_times) as repeat_idx:
                        start_output_row_idx = rows_each_core * core_idx + self.vnchwconv_rows * repeat_idx
                        self._copy_each_row_with_vnchwconv(start_output_row_idx, output_inner_dim_factor,
                                                           self.vnchwconv_rows, input_ub, vnchw_conv_ub,
                                                           self._do_part_each_row_with_vnchwconv_repeat_by_block)

    def _copy_each_row_with_vnchwconv(self, start_output_row_idx, output_inner_dim_factor, rows, input_ub,
                                      vnchw_conv_ub,
                                      one_vnchwconv_func):
        inst = self.tik_instance
        output_inner_dim = self.output_shape[self.shape_length - 1] * self.multi_times
        output_inner_dim_split = ceil_div(output_inner_dim, output_inner_dim_factor)
        with inst.for_range(0, output_inner_dim_split - 1) as i:
            output_inner_offset = i * output_inner_dim_factor
            one_vnchwconv_func(start_output_row_idx, output_inner_offset,
                               output_inner_dim_factor, rows,
                               input_ub, vnchw_conv_ub)
        row_tail_align_count = inst.Scalar(init_value=ceil_align(output_inner_dim % output_inner_dim_factor,
                                                                 self.tensor_block_element))
        rollback4tail = inst.Scalar(init_value=row_tail_align_count - (output_inner_dim % output_inner_dim_factor))
        with inst.if_scope(row_tail_align_count == 0):
            row_tail_align_count.set_as(output_inner_dim_factor)
            rollback4tail.set_as(0)
        output_inner_offset = (output_inner_dim_split - 1) * output_inner_dim_factor - rollback4tail
        one_vnchwconv_func(start_output_row_idx, output_inner_offset,
                           row_tail_align_count, rows, input_ub,
                           vnchw_conv_ub)

    def _copy_part_each_row_with_vnchwconv(self, start_output_row_idx, offset, count_each_copy, rows, input_ub,
                                           vnchw_conv_ub):
        inst = self.tik_instance
        multi_times = self.multi_times
        last_stride = self.tiling_param.stride[self.tiling_param.shape_length - 1]
        input_gm = self.input_gm4vnchwconv
        output_gm = self.output_gm4vnchwconv
        input_inner_dim = count_each_copy * last_stride
        input_inner_dim_copy_in = self._min(input_inner_dim,
                                            self._get_input_inner_dim_no_begin() - offset * last_stride)
        with inst.for_range(0, rows) as j:
            output_row_idx = start_output_row_idx + j
            input_gm_addr = self._get_input_gm_addr(output_row_idx) * multi_times + offset * last_stride
            in_ub_addr = j * input_inner_dim
            gm2ub(inst, input_ub[in_ub_addr:], input_gm[input_gm_addr:], input_inner_dim_copy_in)
        vnchw_repeat_times = input_inner_dim // self.tensor_block_element
        self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, input_inner_dim, vnchw_repeat_times,
                                      self.tensor_block_element)
        ub2ub(inst, input_ub, vnchw_conv_ub, [0, count_each_copy, multi_times, (last_stride - 1) * multi_times, 0])
        self._do_with_vnchwconv2output(vnchw_conv_ub, input_ub, vnchw_repeat_times, self.tensor_block_element)
        with inst.for_range(0, rows) as j:
            output_row_idx = start_output_row_idx + j
            output_gm_addr = self._get_output_gm_addr(output_row_idx) * multi_times + offset
            out_ub_addr = input_inner_dim * j
            ub2gm(inst, output_gm[output_gm_addr:], vnchw_conv_ub[out_ub_addr:], count_each_copy)

    def _do_part_each_row_with_vnchwconv_block_by_block(self, start_output_row_idx, offset, count_each_copy, rows,
                                                        input_ub,
                                                        vnchw_conv_ub):
        inst = self.tik_instance
        multi_times = self.multi_times
        last_stride = self.tiling_param.stride[self.tiling_param.shape_length - 1]
        input_gm = self.input_gm4vnchwconv
        output_gm = self.output_gm4vnchwconv
        input_inner_dim = count_each_copy * self.tensor_block_element
        with inst.for_range(0, rows) as j:
            output_row_idx = start_output_row_idx + j
            input_gm_addr = self._get_input_gm_addr(output_row_idx) * multi_times + offset * last_stride
            with inst.for_range(0, count_each_copy // multi_times) as i:
                in_ub_addr = j * input_inner_dim + i * self.tensor_block_element
                input_gm_cur_stride_addr = input_gm_addr + i * last_stride * multi_times
                gm2ub(inst, input_ub[in_ub_addr:], input_gm[input_gm_cur_stride_addr:], multi_times)

        vnchw_repeat_times = count_each_copy * self.tensor_block_element // self.tensor_block_element
        self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, input_inner_dim, vnchw_repeat_times,
                                      self.tensor_block_element)
        ub2ub(inst, input_ub, vnchw_conv_ub, [0, count_each_copy, multi_times,
              self.tensor_block_element - multi_times, 0])
        self._do_with_vnchwconv2output(vnchw_conv_ub, input_ub, vnchw_repeat_times, self.tensor_block_element)
        with inst.for_range(0, rows) as j:
            output_row_idx = start_output_row_idx + j
            output_gm_addr = self._get_output_gm_addr(output_row_idx) * multi_times + offset
            out_ub_addr = input_inner_dim * j
            ub2gm(inst, output_gm[output_gm_addr:], vnchw_conv_ub[out_ub_addr:], count_each_copy)

    def _do_part_each_row_with_vnchwconv_repeat_by_block(self, start_output_row_idx, offset, count_each_copy, rows,
                                                         input_ub,
                                                         vnchw_conv_ub):
        inst = self.tik_instance
        multi_times = self.multi_times
        last_stride = self.tiling_param.stride[self.tiling_param.shape_length - 1]
        input_gm = self.input_gm4vnchwconv
        output_gm = self.output_gm4vnchwconv
        input_inner_dim = count_each_copy * self.tensor_block_element
        with inst.for_range(0, rows) as j:
            output_row_idx = start_output_row_idx + j
            input_gm_addr = self._get_input_gm_addr(output_row_idx) * multi_times + offset * last_stride
            in_ub_addr = j * input_inner_dim
            if tbe_platform.api_check_support("tik.data_move_pad"):
                inst.data_move_pad(input_ub[in_ub_addr:], input_gm[input_gm_addr:], count_each_copy, self.dtype_size,
                                   0, (last_stride - 1) * self.dtype_size)
            else:
                inst.data_move(input_ub[in_ub_addr:], input_gm[input_gm_addr:], 0, count_each_copy, 1,
                               last_stride * multi_times // self.tensor_block_element - 1, 0)
        vnchw_repeat_times = count_each_copy * self.tensor_block_element // self.tensor_block_element
        self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, input_inner_dim, vnchw_repeat_times,
                                      self.tensor_block_element)
        ub2ub(inst, input_ub, vnchw_conv_ub, [0, count_each_copy, multi_times,
              self.tensor_block_element - multi_times, 0])
        self._do_with_vnchwconv2output(vnchw_conv_ub, input_ub, vnchw_repeat_times, self.tensor_block_element)
        with inst.for_range(0, rows) as j:
            output_row_idx = start_output_row_idx + j
            output_gm_addr = self._get_output_gm_addr(output_row_idx) * multi_times + offset
            out_ub_addr = input_inner_dim * j
            ub2gm(inst, output_gm[output_gm_addr:], vnchw_conv_ub[out_ub_addr:], count_each_copy)


    def _do_with_vnchwconv_per_loop_large_out_inner_large_inner_stride(self, row_id, loop_rows):
        """
        slice a row of input data multiple times
        Parameters
        ----------
        row_id: id of output row
        loop_rows: number of output rows processed at one loop

        Returns
        -------
        None
        """
        inst = self.tik_instance
        with inst.for_range(0, self.inner_loops - 1) as inner_loops_idx:
            self._do_with_vnchwconv_per_inner_loop_large_out_inner_large_inner_stride(row_id, loop_rows,
                                                                                      inner_loops_idx,
                                                                                      self.inner_loop_count, 0)
        self._do_with_vnchwconv_per_inner_loop_large_out_inner_large_inner_stride(row_id, loop_rows,
                                                                                  self.inner_loops - 1,
                                                                                  self.last_inner_loop_count,
                                                                                  self.roll_back_num)


    def _do_last_stride_larger1_functional(self, core_idx):
        inst = self.tik_instance
        output_shape = self.tiling_param.output_shape
        inner_dim = output_shape[self.shape_length - 1]
        inner_stride = self.tiling_param.stride[self.shape_length - 1]
        out_dim = self.tiling_param.out_dim
        larger_block_rows = ceil_div(self.element_each_block, inner_dim)
        out_loops = ceil_align(self._ceil_div(out_dim, self.core_num_scalar), larger_block_rows)
        tmp_ub_size = self.element_each_block
        ub_size = self.ub_size - self.element_each_block
        ub_data_count = inst.Scalar("int32", name="out_ub_data_count")
        ub_data_count.set_as(0)
        input_gm = self.input_gm
        output_gm = self.output_gm
        with inst.new_stmt_scope():
            tmp_ub = inst.Tensor(self.dtype, (tmp_ub_size,), scope=tik.scope_ubuf, name="tmp_ub")
            ub = inst.Tensor(self.dtype, (ub_size,), scope=tik.scope_ubuf, name="out_ub")
            output_gm_addr = inst.Scalar(self.tiling_param.dtype, name="output_addr",
                                         init_value=self._get_output_gm_addr(core_idx * out_loops))
            with inst.for_range(0, out_loops, name="out_loop") as loop_idx:
                idx = core_idx * out_loops + loop_idx
                with inst.if_scope(idx < self.tiling_param.out_dim):
                    with inst.for_range(0, inner_dim) as inner_idx:
                        input_gm_addr = self._get_input_gm_addr(idx) + inner_idx * inner_stride
                        self._data_move(tmp_ub, input_gm[input_gm_addr], 1)
                        ub[ub_data_count] = tmp_ub[0]
                        ub_data_count.set_as(ub_data_count + 1)
                        with inst.if_scope(ub_data_count == ub_size):
                            self._data_move(output_gm[output_gm_addr], ub, ub_size)
                            ub_data_count.set_as(0)
                            output_gm_addr.set_as(output_gm_addr + ub_size)
                    with inst.if_scope(loop_idx == out_loops - 1):
                        self._add_tail_last_stride_larger1(ub, tmp_ub, idx, ub_data_count)
            with inst.if_scope(ub_data_count != 0):
                self._data_move(output_gm[output_gm_addr], ub, ub_data_count)

    def _do_with_vnchwconv_per_inner_loop_large_out_inner_large_inner_stride(self, row_id, loop_rows, inner_loops_idx,
                                                                             loop_count, roll_back_num):
        """
        slice data with the instruction of vnchwconv
        Parameters
        ----------
        row_id: id of output row
        loop_rows: number of output rows processed at one loop
        inner_loops_idx: id of input data loops
        loop_count: number of data processed at one loop
        roll_back_num: roll back num, in order to output data align of 32B

        Returns
        -------
        None
        """
        inst = self.tik_instance
        output_addr = self.output_inner_dim * row_id
        last_stride = self.strides[self.shape_length-1]
        loop_num = loop_count // last_stride
        with inst.new_stmt_scope():
            ub_size = floor_align(self.ub_size // 2, self.element_each_block)
            input_ub = inst.Tensor(self.tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(self.tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="vnchwconv_ub")
            vnchw_conv_repeat_times = loop_count // self.vnchwconv_column
            with inst.for_range(0, loop_rows) as loop_rows_idx:
                output_idx = row_id + loop_rows_idx
                src_addr_in = self._get_input_gm_addr(output_idx) + inner_loops_idx * self.inner_loop_count - \
                              roll_back_num * last_stride
                dst_addr_in = loop_rows_idx * loop_count
                with inst.for_range(0, loop_num) as i:
                    gm2ub(inst, input_ub[dst_addr_in + i * self.element_each_block],
                          self.input_gm[src_addr_in + last_stride * i], self.multi_times)
            self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times,
                                          self.element_each_block)
            ub2ub(inst, input_ub, vnchw_conv_ub, [0, loop_count // (last_stride * self.multi_times),
                  self.multi_times, (self.element_each_block - 1) * self.multi_times, 0])
            vnchw_conv_repeat_times = ceil_div(loop_num, self.vnchwconv_column)
            self._do_with_vnchwconv2output(vnchw_conv_ub, input_ub, vnchw_conv_repeat_times,
                                           constant.BLOCK_SIZE // self.tensor_dtype_size)
            tmp_stride = (self.inner_loops - 1) * self.inner_loop_num // self.element_each_block
            with inst.if_scope(tik.all(self.roll_back_num == 0, tmp_stride <= MAX_STRIDE)):
                dst_addr_out = output_addr + inner_loops_idx * self.inner_loop_num
                dst_stride = inst.Scalar("int32", name="dst_stride", init_value=0)
                with inst.if_scope(tik.all(self.inner_loops > 1, inner_loops_idx == self.inner_loops - 1)):
                    dst_stride.set_as((self.inner_loops - 1) * self.inner_loop_num // self.element_each_block)
                with inst.elif_scope(self.inner_loops > 1):
                    dst_stride.set_as(((self.inner_loops - 2) * self.inner_loop_num + self.last_inner_loop_num) //
                                      self.element_each_block)
                inst.data_move(self.output_gm[dst_addr_out], vnchw_conv_ub, 0, loop_rows,
                               vnchw_conv_repeat_times, 0, dst_stride)
            with inst.else_scope():
                with inst.for_range(0, loop_rows) as loop_rows_idx:
                    dst_addr_out = output_addr + loop_rows_idx * self.output_inner_dim + \
                                   inner_loops_idx * self.inner_loop_num - roll_back_num
                    ub2gm(inst, self.output_gm[dst_addr_out], vnchw_conv_ub[loop_num * loop_rows_idx], loop_num)

    def _do_last_stride_block_element_align_small_out_inner_with_vnchwconv(self, core_idx):
        inst = self.tik_instance
        dtype_size = self.dtype_size
        output_shape = self.tiling_param.output_shape
        shape_length = self.tiling_param.shape_length
        float16_dtype_size = 2
        multi_times = self.multi_times
        vnchwconv_column = self.vnchwconv_column
        tensor_dtype = self.tensor_dtype
        output_inner_dim = output_shape[shape_length - 1] * multi_times
        input_inner_dim = output_shape[shape_length - 1] * self.element_each_block * multi_times
        out_dim = self.tiling_param.out_dim
        ub_size = self.ub_size * multi_times // 2
        begin_value = self.tiling_param.begin[shape_length - 1] * multi_times
        element_each_block = self.element_each_block * multi_times
        output_32bytes_align_rows = self._get_32bytes_align_rows()
        reserve_input_32bytes_align_ub = 0
        max_rows_in_ub = floor_align((ub_size - reserve_input_32bytes_align_ub) // (input_inner_dim * vnchwconv_column),
                                     output_32bytes_align_rows)
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        rows_each_repeat = inst.Scalar("int64", name="rows_each_repeat")
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        tail_rows_repeat_times = inst.Scalar("int64", name="tail_rows_repeat_times")
        tail_rows_repeat_tail_count = inst.Scalar("int64", name="tail_rows_repeat_tail_count")
        tail_rows_repeat_roll_back_rows = inst.Scalar("int64", name="tail_rows_repeat_roll_back_rows", init_value=0)
        rows_each_core.set_as(
            self._ceil_32bytes_count(self._ceil_div(out_dim, self.core_num_scalar), output_32bytes_align_rows))
        repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
        rows_each_repeat.set_as(
            self._ceil_32bytes_count(self._ceil_div(rows_each_core, repeat_times), output_32bytes_align_rows))
        rows_each_core.set_as(rows_each_repeat * repeat_times)
        aicore_num_used.set_as(self._ceil_div(out_dim, rows_each_core))
        with inst.if_scope(rows_each_core > out_dim):
            rows_each_core.set_as(out_dim)
            repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
            rows_each_repeat.set_as(
                self._ceil_32bytes_count(self._ceil_div(rows_each_core, repeat_times), output_32bytes_align_rows))
            with inst.if_scope(rows_each_repeat > out_dim):
                rows_each_repeat.set_as(out_dim)
        loop_times = self._ceil_div(repeat_times, 16)
        last_loop_rows = inst.Scalar("int64", name="last_loop_rows", init_value=repeat_times % 16)
        with inst.if_scope(last_loop_rows == 0):
            last_loop_rows.set_as(16)

        tail_rows.set_as(out_dim % rows_each_core)
        tail_rows_repeat_times.set_as(self._ceil_div(tail_rows, rows_each_repeat))
        tail_rows_repeat_tail_count.set_as(tail_rows % rows_each_repeat)
        with inst.if_scope(tail_rows_repeat_tail_count == 0):
            tail_rows_repeat_tail_count.set_as(rows_each_repeat)
        with inst.if_scope(tail_rows_repeat_tail_count % output_32bytes_align_rows != 0):
            tail_rows_repeat_roll_back_rows.set_as(self._ceil_32bytes_count(tail_rows_repeat_tail_count,
                                                                            output_32bytes_align_rows) - \
                                                   tail_rows_repeat_tail_count)
        tail_loop_times = self._ceil_div(tail_rows_repeat_times - 1, 16)
        tail_last_loop_rows = inst.Scalar("int64",
                                          name="tail_last_loop_rows",
                                          init_value=(tail_rows_repeat_times - 1) % 16)
        with inst.if_scope(tail_last_loop_rows == 0):
            tail_last_loop_rows.set_as(16)

        if dtype_size % float16_dtype_size == 0:
            input_gm = self.input_gm.reinterpret_cast_to("float16")
            output_gm = self.output_gm.reinterpret_cast_to("float16")
        else:
            input_gm = self.input_gm
            output_gm = self.output_gm

        output_row_idx = inst.Scalar("int64", name="output_row_idx")
        param_dict = {
            "input_gm": input_gm,
            "output_gm": output_gm,
            "rows_each_repeat": rows_each_repeat,
            "input_inner_dim": input_inner_dim,
            "output_inner_dim": output_inner_dim,
            "begin_value": begin_value,
            "element_each_block": element_each_block
        }

        with inst.new_stmt_scope():
            input_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="vnchw_conv_ub")
            param_dict["input_ub"] = input_ub
            param_dict["vnchw_conv_ub"] = vnchw_conv_ub
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tail_rows != 0):
                    with inst.if_scope(core_idx < aicore_num_used - 1):
                        with inst.for_range(0, loop_times - 1) as loop_idx:
                            output_row_idx.set_as(rows_each_core * core_idx + \
                                                  loop_idx * rows_each_repeat * 16)
                            param_dict["output_row_idx"] = output_row_idx
                            param_dict["loop_rows"] = 16
                            self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_align(param_dict)
                        output_row_idx.set_as(rows_each_core * core_idx + (loop_times - 1) * 16 * rows_each_repeat)
                        param_dict["output_row_idx"] = output_row_idx
                        param_dict["loop_rows"] = last_loop_rows
                        self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_align(param_dict)
                    with inst.else_scope():
                        with inst.if_scope(tail_rows_repeat_times > 1):
                            with inst.for_range(0, tail_loop_times - 1) as loop_idx:
                                output_row_idx.set_as(rows_each_core * (aicore_num_used - 1) + \
                                                      loop_idx * rows_each_repeat * 16)
                                param_dict["output_row_idx"] = output_row_idx
                                param_dict["loop_rows"] = 16
                                self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_align(param_dict)
                            output_row_idx.set_as(rows_each_core * (aicore_num_used - 1) + \
                                                  (tail_loop_times - 1) * 16 * rows_each_repeat)
                            param_dict["output_row_idx"] = output_row_idx
                            param_dict["loop_rows"] = tail_last_loop_rows
                            self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_align(param_dict)
                        output_row_idx.set_as(
                            rows_each_core * (aicore_num_used - 1) + (tail_rows_repeat_times - 1) * rows_each_repeat -
                            tail_rows_repeat_roll_back_rows)
                        param_dict["output_row_idx"] = output_row_idx
                        param_dict["loop_rows"] = 1
                        param_dict["rows_each_repeat"].set_as(tail_rows_repeat_tail_count +
                                                              tail_rows_repeat_roll_back_rows)
                        self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_align(param_dict)
                with inst.else_scope():
                    with inst.for_range(0, loop_times - 1) as loop_idx:
                        output_row_idx.set_as(rows_each_core * core_idx + loop_idx * 16 * rows_each_repeat)
                        param_dict["output_row_idx"] = output_row_idx
                        param_dict["loop_rows"] = 16
                        self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_align(param_dict)
                    output_row_idx.set_as(rows_each_core * core_idx + (loop_times - 1) * 16 * rows_each_repeat)
                    param_dict["output_row_idx"] = output_row_idx
                    param_dict["loop_rows"] = last_loop_rows
                    self._do_with_vnchwconv_copy_each_rows_per_loop_last_stride_align(param_dict)

    def _get_32bytes_align_rows(self):
        inst = self.tik_instance
        output_32bytes_align_rows = inst.Scalar("int64",
                                                name="output_32bytes_align_rows",
                                                init_value=self.element_each_block)
        output_inner_dim = self.tiling_param.output_shape[self.tiling_param.shape_length - 1]
        with inst.if_scope(output_32bytes_align_rows % output_inner_dim == 0):
            output_32bytes_align_rows.set_as(output_32bytes_align_rows // output_inner_dim)
        with inst.elif_scope(output_inner_dim % output_32bytes_align_rows == 0):
            output_32bytes_align_rows.set_as(1)
        return output_32bytes_align_rows

    def _do_last_stride_larger_block_element_not_align_small_out_inner_with_vnchwconv(self, core_idx):
        inst = self.tik_instance
        dtype_size = self.dtype_size
        output_shape = self.tiling_param.output_shape
        shape_length = self.tiling_param.shape_length
        float16_dtype_size = 2
        multi_times = self.multi_times
        vnchwconv_column = self.vnchwconv_column
        tensor_dtype = self.tensor_dtype
        output_inner_dim = output_shape[shape_length - 1] * multi_times
        input_inner_dim = output_shape[shape_length - 1] * self.element_each_block * multi_times
        out_dim = self.tiling_param.out_dim
        ub_size = self.ub_size * multi_times // 2
        begin_value = self.tiling_param.begin[shape_length - 1] * multi_times
        element_each_block = self.element_each_block * multi_times
        output_32bytes_align_rows = self._get_32bytes_align_rows()
        reserve_input_32bytes_align_ub = 0
        max_rows_in_ub = floor_align((ub_size - reserve_input_32bytes_align_ub) // (input_inner_dim * vnchwconv_column),
                                     output_32bytes_align_rows)
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        rows_each_repeat = inst.Scalar("int64", name="rows_each_repeat")
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        tail_rows_repeat_times = inst.Scalar("int64", name="tail_rows_repeat_times")
        tail_rows_repeat_tail_count = inst.Scalar("int64", name="tail_rows_repeat_tail_count")
        tail_rows_repeat_roll_back_rows = inst.Scalar("int64", name="tail_rows_repeat_roll_back_rows", init_value=0)
        rows_each_core.set_as(
            self._ceil_32bytes_count(self._ceil_div(out_dim, self.core_num_scalar), output_32bytes_align_rows))
        repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
        rows_each_repeat.set_as(
            self._ceil_32bytes_count(self._ceil_div(rows_each_core, repeat_times), output_32bytes_align_rows))
        rows_each_core.set_as(rows_each_repeat * repeat_times)
        aicore_num_used.set_as(self._ceil_div(out_dim, rows_each_core))
        with inst.if_scope(rows_each_core > out_dim):
            rows_each_core.set_as(out_dim)
            repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
            rows_each_repeat.set_as(
                self._ceil_32bytes_count(self._ceil_div(rows_each_core, repeat_times), output_32bytes_align_rows))
            with inst.if_scope(rows_each_repeat>out_dim):
                rows_each_repeat.set_as(out_dim)
        loop_times = self._ceil_div(repeat_times, 16)
        last_loop_rows = inst.Scalar("int64", name="last_loop_rows", init_value=repeat_times % 16)
        with inst.if_scope(last_loop_rows == 0):
            last_loop_rows.set_as(16)

        tail_rows.set_as(out_dim % rows_each_core)
        tail_rows_repeat_times.set_as(self._ceil_div(tail_rows, rows_each_repeat))
        tail_rows_repeat_tail_count.set_as(tail_rows % rows_each_repeat)
        with inst.if_scope(tail_rows_repeat_tail_count == 0):
            tail_rows_repeat_tail_count.set_as(rows_each_repeat)
        with inst.if_scope(tail_rows_repeat_tail_count % output_32bytes_align_rows != 0):
            tail_rows_repeat_roll_back_rows.set_as(self._ceil_32bytes_count(tail_rows_repeat_tail_count,
                                                                            output_32bytes_align_rows) - \
                                                   tail_rows_repeat_tail_count)
        tail_loop_times = self._ceil_div(tail_rows_repeat_times - 1, 16)
        tail_last_loop_rows = inst.Scalar("int64",
                                          name="tail_last_loop_rows",
                                          init_value=(tail_rows_repeat_times - 1) % 16)
        with inst.if_scope(tail_last_loop_rows == 0):
            tail_last_loop_rows.set_as(16)

        if dtype_size % float16_dtype_size == 0:
            input_gm = self.input_gm.reinterpret_cast_to("float16")
            output_gm = self.output_gm.reinterpret_cast_to("float16")
        else:
            input_gm = self.input_gm
            output_gm = self.output_gm

        output_row_idx = inst.Scalar("int64", name="output_row_idx")
        param_dict = {
            "input_gm": input_gm,
            "output_gm": output_gm,
            "rows_each_repeat": rows_each_repeat,
            "input_inner_dim": input_inner_dim,
            "output_inner_dim": output_inner_dim,
            "begin_value": begin_value,
            "element_each_block": element_each_block
        }

        with inst.new_stmt_scope():
            input_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="vnchw_conv_ub")
            param_dict["input_ub"] = input_ub
            param_dict["vnchw_conv_ub"] = vnchw_conv_ub
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tail_rows != 0):
                    with inst.if_scope(core_idx < aicore_num_used - 1):
                        with inst.for_range(0, loop_times - 1) as loop_idx:
                            output_row_idx.set_as(rows_each_core * core_idx + loop_idx * rows_each_repeat * 16)
                            param_dict["output_row_idx"] = output_row_idx
                            param_dict["loop_rows"] = 16
                            self._do_with_vnchwconv_last_stride_not_align_larger_block_element(param_dict)
                        output_row_idx.set_as(rows_each_core * core_idx + (loop_times - 1) * 16 * rows_each_repeat)
                        param_dict["output_row_idx"] = output_row_idx
                        param_dict["loop_rows"] = last_loop_rows
                        self._do_with_vnchwconv_last_stride_not_align_larger_block_element(param_dict)
                    with inst.else_scope():
                        with inst.if_scope(tail_rows_repeat_times > 1):
                            with inst.for_range(0, tail_loop_times - 1) as loop_idx:
                                output_row_idx.set_as(rows_each_core * (aicore_num_used - 1) + \
                                                      loop_idx * rows_each_repeat * 16)
                                param_dict["output_row_idx"] = output_row_idx
                                param_dict["loop_rows"] = 16
                                self._do_with_vnchwconv_last_stride_not_align_larger_block_element(param_dict)
                            output_row_idx.set_as(rows_each_core * (aicore_num_used - 1) + \
                                                  (tail_loop_times - 1) * 16 * rows_each_repeat)
                            param_dict["output_row_idx"] = output_row_idx
                            param_dict["loop_rows"] = tail_last_loop_rows
                            self._do_with_vnchwconv_last_stride_not_align_larger_block_element(param_dict)
                        output_row_idx.set_as(
                            rows_each_core * (aicore_num_used - 1) + (tail_rows_repeat_times - 1) * rows_each_repeat -
                            tail_rows_repeat_roll_back_rows)
                        param_dict["output_row_idx"] = output_row_idx
                        param_dict["loop_rows"] = 1
                        param_dict["rows_each_repeat"].set_as(tail_rows_repeat_tail_count +
                                                              tail_rows_repeat_roll_back_rows)
                        self._do_with_vnchwconv_last_stride_not_align_larger_block_element(param_dict)
                with inst.else_scope():
                    with inst.for_range(0, loop_times - 1) as loop_idx:
                        output_row_idx.set_as(rows_each_core * core_idx + loop_idx * 16 * rows_each_repeat)
                        param_dict["output_row_idx"] = output_row_idx
                        param_dict["loop_rows"] = 16
                        self._do_with_vnchwconv_last_stride_not_align_larger_block_element(param_dict)
                    output_row_idx.set_as(rows_each_core * core_idx + (loop_times - 1) * 16 * rows_each_repeat)
                    param_dict["output_row_idx"] = output_row_idx
                    param_dict["loop_rows"] = last_loop_rows
                    self._do_with_vnchwconv_last_stride_not_align_larger_block_element(param_dict)


# 'pylint: disable=locally-disabled,too-many-arguments,
# 'pylint: disable=unused-argument,too-many-locals
@register_operator("StridedSlice")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def strided_slice(input_x,
                  begin,
                  end,
                  strides=None,
                  output_x=None,
                  begin_mask=0,
                  end_mask=0,
                  ellipsis_mask=0,
                  new_axis_mask=0,
                  shrink_axis_mask=0,
                  kernel_name="strided_slice"):
    """
    Extracts a strided slice of a tensor (generalized python array indexing).
    Roughly speaking, this op extracts a slice of size (end-begin)/stride
    from the given input_ tensor.
    Starting at the location specified by begin the slice continues
     by adding stride to the index
    until all dimensions are not less than end. Note that a stride
    can be negative, which causes a reverse slice.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of out
    begin: dict.
        shape and dtype of begin, represents the index of the first value to select.
    end: dict.
        shape and dtype of end, represents the index of the last value to select.
    strides: dict.
        shape and dtype of strides, step length to select.
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin
        value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position
        is actually an ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification
        should shrink the dimensionality.
    kernel_name : str
        cce kernel name, default value is "strided_slice"

    Returns
    -------
    tik_instance
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32", "complex32", "int32", "uint8", "bool", "int8", "int64", "complex64",
                  "int16", "uint16", "uint32", "uint64")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    strided_slice_instance = StridedSlice(input_x, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                          shrink_axis_mask, kernel_name)
    strided_slice_instance.strided_slice()
    inst = strided_slice_instance.tik_instance
    opt_config = strided_slice_instance.get_opt_config()
    inst.BuildCCE(kernel_name=strided_slice_instance.kernel_name,
                  inputs=(strided_slice_instance.input_gm, strided_slice_instance.begin_gm,
                          strided_slice_instance.end_gm, strided_slice_instance.strides_gm),
                  outputs=(strided_slice_instance.output_gm,),
                  flowtable=[strided_slice_instance.tiling_param.tiling_gm],
                  config=opt_config,
                  enable_l2=False)

    tbe_context.get_context().add_compile_info("vars", strided_slice_instance.get_vars_info())
    return inst


def generalize_all_shape(in_tensor):
    """
    generalize_all_shape
    """
    dynamic_rank_shape = [-2]
    in_tensor["shape"] = dynamic_rank_shape
    in_tensor["ori_shape"] = dynamic_rank_shape
    in_tensor["format"] = "ND"
    in_tensor["ori_format"] = "ND"

    return in_tensor


def get_bit_len(data_type_string):
    """
    get_bit_len
    """
    dtype_dict = {
        'float16': 16,
        'int16': 16,
        'uint16': 16,
        'bfloat16':16,
        'float': 32,
        'float32': 32,
        'complex32': 32,
        'int32': 32,
        'uint32': 32,
        'int64': 64,
        'uint64': 64,
        'complex64': 64,
        'int8': 8,
        'uint8': 8,
        'bool': 8,
    }
    return dtype_dict.get(data_type_string, 0)


def generalize_input_output_all_shape(in_tensor):
    """
    generalize_input_output_all_shape
    """
    dynamic_rank_shape = [-2]
    in_tensor["shape"] = dynamic_rank_shape
    in_tensor["ori_shape"] = dynamic_rank_shape
    in_tensor["format"] = "ND"
    in_tensor["ori_format"] = "ND"
    input_dtype = in_tensor.get("dtype")
    if get_bit_len(input_dtype) == 32:
        in_tensor["dtype"] = "float32"
    if get_bit_len(input_dtype) == 16:
        in_tensor["dtype"] = "float16"
    if get_bit_len(input_dtype) == 64:
        in_tensor["dtype"] = "int64"
    if get_bit_len(input_dtype) == 8:
        in_tensor["dtype"] = "int8"

    return in_tensor


@tbe_register.register_param_generalization("StridedSlice")
def strided_slice_generalization(input_x,
                                 begin,
                                 end,
                                 strides=None,
                                 output_x=None,
                                 begin_mask=0,
                                 end_mask=0,
                                 ellipsis_mask=0,
                                 new_axis_mask=0,
                                 shrink_axis_mask=0,
                                 generalize_config=None):
    if generalize_config is None:
        generalize_config = {"mode": "keep_rank"}
    if generalize_config.get("mode") == "keep_rank":
        const_value_key = "const_value"
        strides_value = strides.get(const_value_key)

        def is_all_ge_one():
            for stride in strides_value:
                if stride < 1:
                    return False
            return True

        if strides_value and is_all_ge_one():
            # to aicore, aicore only support stride is 1.
            input_x["ori_shape"] = [-1] * len(input_x["ori_shape"])
            input_x["shape"] = input_x["ori_shape"]
            input_x["ori_range"] = [[1, -1]] * len(input_x["ori_shape"])
            input_x["range"] = input_x["ori_range"]
            output_x["ori_shape"] = [-1] * len(output_x["ori_shape"])
            output_x["shape"] = output_x["ori_shape"]
            output_x["ori_range"] = [[1, -1]] * len(output_x["ori_shape"])
            output_x["range"] = output_x["ori_range"]
            to_del_const_value = [begin, end]
            for item in to_del_const_value:
                if const_value_key in item:
                    item[const_value_key] = None
            return [[
                input_x, begin, end, strides, output_x, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                shrink_axis_mask
            ]]
        else:
            # to not aicore
            input_x["ori_shape"] = [-1] * len(input_x["ori_shape"])
            input_x["shape"] = input_x["ori_shape"]
            input_x["ori_range"] = [[1, -1]] * len(input_x["ori_shape"])
            input_x["range"] = input_x["ori_range"]
            output_x["ori_shape"] = [-1] * len(output_x["ori_shape"])
            output_x["shape"] = output_x["ori_shape"]
            output_x["ori_range"] = [[1, -1]] * len(output_x["ori_shape"])
            output_x["range"] = output_x["ori_range"]
            to_del_const_value = [begin, end, strides]
            for item in to_del_const_value:
                if const_value_key in item:
                    item[const_value_key] = None
            min_int32 = -2**31
            max_int32 = 2**31 - 1
            value_range = ([[min_int32, max_int32]] if len(strides["ori_shape"]) == 0 else 
                           [[min_int32, max_int32]] * strides["ori_shape"][0])
            strides["const_value_range"] = value_range
            return [[
                input_x, begin, end, strides, output_x, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                shrink_axis_mask
            ]]

    elif generalize_config.get("mode") == "all_shape":
        input_x = generalize_input_output_all_shape(input_x)
        begin = generalize_all_shape(begin)
        end = generalize_all_shape(end)
        strides = generalize_all_shape(strides)
        output_x = generalize_input_output_all_shape(output_x)
        begin_mask = None
        end_mask = None
        ellipsis_mask = None
        new_axis_mask = None
        shrink_axis_mask = None

        return [[
                input_x, begin, end, strides, output_x, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                shrink_axis_mask
            ]]

    else:
        # to default process
        return None
