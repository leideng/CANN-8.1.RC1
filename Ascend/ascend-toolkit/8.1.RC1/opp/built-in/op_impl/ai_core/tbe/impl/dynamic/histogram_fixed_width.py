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
histogram_fixed_width
"""

from collections import namedtuple
from impl.util.platform_adapter import tbe_platform as cce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods
class Constant(object):
    """
    The class for constant
    """
    # 8k UB buffer is a reserved space
    RESERVE_SIZE = 8 * 1024
    MAX_INT32 = 2 ** 31 - 1
    MAX_UINT8 = 2 ** 8 - 1
    HIGH_PREC_MAX_TENSOR = 2 ** 18
    ELEMENT_UINT8 = 2 ** 8
    MAX_REPEAT_TIME = 255
    SCALAR_TENSOR_SIZE = 32
    TILING_ARG_NUM = 32
    INDICES_TENSOR_NUM = 8
    DST_REP_STRIDE = 8
    DTYPE_FP32 = "float32"
    DTYPE_FP16 = "float16"
    DTYPE_INT32 = "int32"
    DTYPE_INT64 = "int64"
    BYTE_FP16 = 2
    BYTE_FP32 = 4
    BYTE_INT32 = 4
    BYTE_INT64 = 8
    TILING_PARA_DTYPE = DTYPE_INT64
    BLOCK_BYTE_SIZE = 32
    MASK_INDICES = 8
    MASK_16BIT = 128
    MASK_32BIT = 64
    BLOCK_16BIT = 16


class HistogramFixedWidth(object):
    """
    The class of HistogramFixedWidth op
    """

    def __init__(self, x, input_range, nbins, y, dtype=3, kernel_name="HistogramFixedWidth"):
        """
        init func

        Parameters
        ----------
        x: dict
            dict info of input value, must include the keys(shape and dtype).
        input_range: dict
            dict info of input value_range, must include the keys(shape and dtype).
                            the shape must be (2,) or [2]
        nbins: dict
            dict info of nbins value, must include the keys(shape and dtype).
        y: dict
            dict info of output
        dtype: str
            data type for returned histogram.
        kernel_name: str
            cce kernel name, default value is "histogram_fixed_width"


        returns
        -------
        None
        """
        self.x_dtype = x.get("dtype").lower()
        self.range_dtype = input_range.get("dtype").lower()
        self.nbins_dtype = nbins.get("dtype").lower()
        self.y_dtype = y.get("dtype").lower()
        self.dtype = dtype
        self.kernel_name = kernel_name
        self.tiling_dtype = Constant.TILING_PARA_DTYPE

        self.tik_instance = tik.Tik()
        self.ub_size_bytes = cce.get_soc_spec(cce.UB_SIZE) - Constant.RESERVE_SIZE
        self.total_core_number = cce.get_soc_spec(cce.CORE_NUM)
        if cce.get_soc_spec(cce.SHORT_SOC_VERSION) == "AS31XM1":
            self.is_vector_histv2_support = True
        else:
            self.is_vector_histv2_support = False
        self.output_num_each_block = Constant.BLOCK_BYTE_SIZE // Constant.BYTE_FP32
        if self.x_dtype == Constant.DTYPE_FP16:
            self.value_num_each_block = Constant.BLOCK_BYTE_SIZE // Constant.BYTE_FP16
        else:
            self.value_num_each_block = Constant.BLOCK_BYTE_SIZE // Constant.BYTE_FP32
        self.mask = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="mask")
        self.repeat = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="repeat")
        self._init_gm()
        self._init_tiling_data()
        self.x_ub = self.tik_instance.Tensor(self.x_dtype, (self.input_ub_max_tensor_size,), scope=tik.scope_ubuf,
                                             name="x_ub")
        if not cce.api_check_support("tbe.dsl.vexp", "float32") or not cce.api_check_support("tik.scalar_conv") \
                or self.x_dtype == Constant.DTYPE_FP16:
            self.one = self.tik_instance.Tensor(Constant.DTYPE_FP16, (Constant.MASK_16BIT,),
                                                scope=tik.scope_ubuf, name="one")
            self.zero = self.tik_instance.Tensor(Constant.DTYPE_FP16, (Constant.MASK_16BIT,),
                                                 scope=tik.scope_ubuf, name="zero")
            self.x_ub_fp16 = self.tik_instance.Tensor(Constant.DTYPE_FP16, (self.input_ub_max_tensor_size,),
                                                      scope=tik.scope_ubuf, name="x_ub_fp16")
        else:
            self.one = self.tik_instance.Tensor(Constant.DTYPE_FP32, (Constant.MASK_32BIT,),
                                                scope=tik.scope_ubuf, name="one")
            self.zero = self.tik_instance.Tensor(Constant.DTYPE_FP32, (Constant.MASK_32BIT,),
                                                 scope=tik.scope_ubuf, name="zero")
            self.x_ub_fp32 = self.tik_instance.Tensor(Constant.DTYPE_FP32, (self.input_ub_max_tensor_size,),
                                                      scope=tik.scope_ubuf, name="x_ub_fp32")
        self.range_ub = self.tik_instance.Tensor(self.range_dtype, (self.value_num_each_block,), scope=tik.scope_ubuf,
                                                 name="range_ub")
        self.y_ub = self.tik_instance.Tensor(self.y_dtype, (self.output_ub_max_tensor_size,),
                                             scope=tik.scope_ubuf, name="y_ub")
        self._init_range_data()
        self._init_preprocess_factor()

    def scalar_int32_to_fp16(self, dst_scalar, src_scalar):
        """
        scalar int32 to fp16 tool

        Parameters
        ----------
        dst_scalar : dst_scalar
        src_scalar : src_scalar

        Returns
        -------
        None
        """
        int32_tensor = self.tik_instance.Tensor(Constant.DTYPE_INT32, (self.output_num_each_block,),
                                                scope=tik.scope_ubuf,
                                                name="int32_tensor")
        fp16_tensor = self.tik_instance.Tensor(Constant.DTYPE_FP16, (self.output_num_each_block,), scope=tik.scope_ubuf,
                                               name="fp16_tensor")
        int32_tensor[0].set_as(src_scalar)
        self.tik_instance.vec_conv(1, "", fp16_tensor, int32_tensor, 1, 1, 1, 1.0)
        dst_scalar.set_as(fp16_tensor[0])

    def scalar_fp16_to_fp32(self, dst_scalar, src_scalar):
        """
        scalar fp16 to fp32 tool

        Parameters
        ----------
        dst_scalar : dst_scalar
        src_scalar : src_scalar

        Returns
        -------
        None
        """
        fp32_tensor = self.tik_instance.Tensor(Constant.DTYPE_FP32, (self.output_num_each_block,), scope=tik.scope_ubuf,
                                               name="fp32_tensor")
        fp16_tensor = self.tik_instance.Tensor(Constant.DTYPE_FP16, (self.output_num_each_block,), scope=tik.scope_ubuf,
                                               name="fp16_tensor")
        fp16_tensor[0].set_as(src_scalar)
        self.tik_instance.vec_conv(1, "", fp32_tensor, fp16_tensor, 1, 1, 1)
        dst_scalar.set_as(fp32_tensor[0])

    def scalar_fp16_to_int32(self, dst_scalar, src_scalar, round_mode):
        """
        scalar fp16 to int32 tool

        Parameters
        ----------
        dst_scalar : dst_scalar
        src_scalar : src_scalar
        round_mode : round_mode

        Returns
        -------
        None
        """
        int32_tensor = self.tik_instance.Tensor(Constant.DTYPE_INT32, (self.output_num_each_block,),
                                                scope=tik.scope_ubuf,
                                                name="fp32_tensor")
        fp16_tensor = self.tik_instance.Tensor(Constant.DTYPE_FP16, (self.output_num_each_block,), scope=tik.scope_ubuf,
                                               name="fp16_tensor")
        fp16_tensor[0].set_as(src_scalar)
        self.tik_instance.vec_conv(1, round_mode, int32_tensor, fp16_tensor, 1, 1, 1)
        dst_scalar.set_as(int32_tensor[0])

    def scalar_fp32_to_fp16(self, dst_scalar, src_scalar):
        """
        scalar fp32 to fp16 tool

        Parameters
        ----------
        dst_scalar : dst_scalar
        src_scalar : src_scalar

        Returns
        -------
        None
        """
        fp16_tensor = self.tik_instance.Tensor(Constant.DTYPE_FP16, (self.output_num_each_block,), scope=tik.scope_ubuf,
                                               name="fp16_tensor")
        fp32_tensor = self.tik_instance.Tensor(Constant.DTYPE_FP32, (self.output_num_each_block,), scope=tik.scope_ubuf,
                                               name="fp32_tensor")
        fp32_tensor[0].set_as(src_scalar)
        self.tik_instance.vec_conv(1, "", fp16_tensor, fp32_tensor, 1, 1, 1)
        dst_scalar.set_as(fp16_tensor[0])

    def int32_to_fp16(self, dst_ub, src_ub, mask, repeat):
        """
        tensor int32 to fp16

        Parameters
        ----------
        dst_ub : dst_ub
        src_ub : src_ub
        mask : mask
        repeat : repeat

        Returns
        -------
        None
        """
        if not cce.api_check_support("tik.vec_conv", "s322f16"):
            floor_x_ub_int16 = self.tik_instance.Tensor("int16", (self.input_ub_max_tensor_size,),
                                                        name="floor_x_ub_int16", scope=tik.scope_ubuf)
            self.tik_instance.vcbd(mask, floor_x_ub_int16, src_ub, repeat, 1, 1, 4, 8)
            self.tik_instance.vec_conv(mask, '', dst_ub, floor_x_ub_int16, repeat, 8, 8)
        else:
            self.tik_instance.vec_conv(mask, '', dst_ub, src_ub, repeat, 4, 8, 1.0)

    def calc_per_core(self, loop_output_num, output_offset, output_burst_len, back_num):
        """
        compute per core

        Parameters
        ----------
        loop_output_num : output loop num
        output_offset : input offset
        output_burst_len : output_burst_len
        back_num : back_num
        Returns
        -------
        None
        """
        self.init_output_ub()
        input_loop_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="input_loop_num")
        input_tail_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="input_tail_num")
        input_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="input_offset")
        tail_burst_len = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="tail_burst_len")
        tail_mask = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="tail_mask")

        input_loop_num.set_as(self.total_values_num // self.input_ub_max_tensor_size)
        input_tail_num.set_as(self.total_values_num % self.input_ub_max_tensor_size)
        tail_burst_len.set_as((input_tail_num + self.value_num_each_block - 1) // self.value_num_each_block)
        if self.x_dtype == Constant.DTYPE_FP16:
            self.mask.set_as(Constant.MASK_16BIT)
            self.repeat.set_as(self.input_ub_max_tensor_size // Constant.MASK_16BIT)
            tail_mask.set_as(self.input_ub_max_tensor_size % Constant.MASK_16BIT)
        else:
            self.mask.set_as(Constant.MASK_32BIT)
            self.repeat.set_as(self.input_ub_max_tensor_size // Constant.MASK_32BIT)
            tail_mask.set_as(self.input_ub_max_tensor_size % Constant.MASK_32BIT)
        with self.tik_instance.for_range(0, input_loop_num) as loop_id:
            input_offset.set_as(loop_id * self.input_ub_max_tensor_size)
            self.tik_instance.data_move(self.x_ub, self.x_gm[input_offset], 0, 1, self.input_ub_max_block_len, 0, 0)
            self.input_value_preprocess(tail_mask)
            if self.x_dtype == Constant.DTYPE_FP16 and self.is_vector_histv2_support == True:
                with self.tik_instance.if_scope(self.is_datasize_support_vdhist == 1):
                    self.per_core_output_result_vdhist(loop_output_num, output_offset, self.input_ub_max_tensor_size)
                with self.tik_instance.else_scope():
                    self.per_core_output_result(loop_output_num, output_offset, self.input_ub_max_tensor_size)
            else:
                self.per_core_output_result(loop_output_num, output_offset, self.input_ub_max_tensor_size)
        with self.tik_instance.if_scope(input_tail_num != 0):
            if self.x_dtype == Constant.DTYPE_FP16:
                self.repeat.set_as(input_tail_num // Constant.MASK_16BIT)
                tail_mask.set_as(input_tail_num % Constant.MASK_16BIT)
            else:
                self.repeat.set_as(input_tail_num // Constant.MASK_32BIT)
                tail_mask.set_as(input_tail_num % Constant.MASK_32BIT)
            input_offset.set_as(input_loop_num * self.input_ub_max_tensor_size)
            self.tik_instance.data_move(self.x_ub, self.x_gm[input_offset], 0, 1, tail_burst_len, 0, 0)
            self.input_value_preprocess(tail_mask)
            if self.x_dtype == Constant.DTYPE_FP16 and self.is_vector_histv2_support == True:
                with self.tik_instance.if_scope(self.is_datasize_support_vdhist == 1):
                    self.per_core_output_result_vdhist(loop_output_num, output_offset, input_tail_num)
                with self.tik_instance.else_scope():
                    self.per_core_output_result(loop_output_num, output_offset, input_tail_num)
            else:
                self.per_core_output_result(loop_output_num, output_offset, input_tail_num)
        self.y_ub_to_gm(output_burst_len, output_offset, back_num)

    def init_output_ub(self):
        """
        initilize output ub to zero

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.mask.set_as(Constant.MASK_32BIT)
        self.repeat.set_as(self.output_ub_max_tensor_size // Constant.MASK_32BIT)

        self.tik_instance.vec_dup(self.mask, self.y_ub, 0, self.repeat, 8)

    def input_value_preprocess(self, tail_mask):
        """
        input_value_preprocess

        Parameters
        ----------
        tail_mask : tail_mask

        Returns
        -------
        None
        """
        repeat_loop = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="repeat_loop")
        repeat_tail = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="repeat_tail")
        repeat_loop.set_as(self.repeat // 255)
        repeat_tail.set_as(self.repeat % 255)
        if not cce.api_check_support("tbe.dsl.vexp", "float32") or not cce.api_check_support("tik.scalar_conv") \
                or self.x_dtype == Constant.DTYPE_FP16:
            with self.tik_instance.for_range(0, repeat_loop) as index:
                input_offset = index * 255 * self.mask
                self.calc_floor_fp16(self.x_ub_fp16[input_offset], self.x_ub[input_offset], self.mask, 255)
            with self.tik_instance.if_scope(repeat_tail != 0):
                input_offset = repeat_loop * 255 * self.mask
                self.calc_floor_fp16(self.x_ub_fp16[input_offset], self.x_ub[input_offset], self.mask, repeat_tail)
            with self.tik_instance.if_scope(tail_mask != 0):
                self.calc_floor_fp16(self.x_ub_fp16[self.mask * self.repeat], self.x_ub[self.mask * self.repeat],
                                     tail_mask, 1)
        else:
            with self.tik_instance.for_range(0, repeat_loop) as index:
                input_offset = index * 255 * self.mask
                self.calc_floor_fp32(self.x_ub_fp32[input_offset], self.x_ub[input_offset], self.mask, 255)
            with self.tik_instance.if_scope(repeat_tail != 0):
                input_offset = repeat_loop * 255 * self.mask
                self.calc_floor_fp32(self.x_ub_fp32[input_offset], self.x_ub[input_offset], self.mask, repeat_tail)
            with self.tik_instance.if_scope(tail_mask != 0):
                self.calc_floor_fp32(self.x_ub_fp32[self.mask * self.repeat], self.x_ub[self.mask * self.repeat],
                                     tail_mask, 1)

    def reduce_to_bins_fp32(self, bins_value, input_data_size, pre_count):
        """
        reduce to bins for fp32 input

        Parameters
        ----------
        bins_value: bins_value
        input_data_size: input_data_size
        pre_count: pre_count

        Returns
        -------
        None
        """
        repeat = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='repeat')
        tail_mask = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='tail_mask')
        repeat_tail = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='repeat_tail')
        repeat_loop = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='repeat_loop')
        repeat.set_as(input_data_size // Constant.MASK_32BIT)
        tail_mask.set_as(input_data_size % Constant.MASK_32BIT)
        repeat_loop.set_as(repeat // 255)
        repeat_tail.set_as(repeat % 255)
        bins_value_count = self.tik_instance.Scalar(Constant.DTYPE_FP32, name="bins_value_count", init_value=0.0)
        output_reduce_sum_int32 = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="output_reduce_sum_int32")
        bins_value_fp32 = self.tik_instance.Scalar(Constant.DTYPE_FP32, name="bins_value_fp32")
        self.tik_instance.scalar_conv("", bins_value_fp32, bins_value)
        bins_value_fp32.set_as(bins_value_fp32 - 1)
        bins_value_tensor = self.tik_instance.Tensor(Constant.DTYPE_FP32, (input_data_size,), scope=tik.scope_ubuf,
                                                     name="bins_value_tensor")
        output_num_per_repeat = self.tik_instance.Tensor(Constant.DTYPE_FP32, (self.value_num_each_block,),
                                                         scope=tik.scope_ubuf, name="output_num_per_repeat")
        output_num_per_repeat_scalar = self.tik_instance.Scalar(Constant.DTYPE_FP32,
                                                                name="output_num_per_repeat_scalar", init_value=0.0)
        ElementOpTuple = namedtuple("ElementOpTuple", ["mask", "repeat"])
        with self.tik_instance.for_range(0, repeat_loop) as index:
            input_offset = index * 255 * Constant.MASK_32BIT
            self.tik_instance.vec_dup(Constant.MASK_32BIT, bins_value_tensor, bins_value_fp32, 255, 8)
            element_op_tuple = ElementOpTuple(Constant.MASK_32BIT, 255)
            self.reduce_add(output_num_per_repeat, self.x_ub_fp32[input_offset], bins_value_tensor, element_op_tuple)
            output_num_per_repeat_scalar.set_as(output_num_per_repeat[0])
            bins_value_count.set_as(bins_value_count + output_num_per_repeat_scalar)
        with self.tik_instance.if_scope(repeat_tail != 0):
            input_offset = repeat_loop * 255 * Constant.MASK_32BIT
            self.tik_instance.vec_dup(Constant.MASK_32BIT, bins_value_tensor, bins_value_fp32, repeat_tail, 8)
            element_op_tuple = ElementOpTuple(Constant.MASK_32BIT, repeat_tail)
            self.reduce_add(output_num_per_repeat, self.x_ub_fp32[input_offset], bins_value_tensor, element_op_tuple)
            output_num_per_repeat_scalar.set_as(output_num_per_repeat[0])
            bins_value_count.set_as(bins_value_count + output_num_per_repeat_scalar)
        with self.tik_instance.if_scope(tail_mask != 0):
            input_offset = repeat * Constant.MASK_32BIT
            self.tik_instance.vec_dup(tail_mask, bins_value_tensor, bins_value_fp32, 1, 8)
            element_op_tuple = ElementOpTuple(tail_mask, 1)
            self.reduce_add(output_num_per_repeat, self.x_ub_fp32[input_offset], bins_value_tensor, element_op_tuple)
            output_num_per_repeat_scalar.set_as(output_num_per_repeat[0])
            bins_value_count.set_as(bins_value_count + output_num_per_repeat_scalar)
        self.tik_instance.scalar_conv("floor", output_reduce_sum_int32, bins_value_count)
        output_reduce_sum_int32.set_as(output_reduce_sum_int32 - pre_count)

        return output_reduce_sum_int32

    def calc_floor_fp32(self, dst_x_ub, origin_x_ub, mask, repeat):
        """
        calc floor

        Parameters
        ----------
        dst_scalar : dst_scalar
        src_scalar : src_scalar
        mask : mask
        repeat : repeat

        Returns
        -------
        None
        """
        input_data_size = mask * repeat
        floor_x_ub_int32 = self.tik_instance.Tensor(Constant.DTYPE_INT32, (input_data_size,), scope=tik.scope_ubuf,
                                                    name="floor_x_ub_int32")
        if self.x_dtype == Constant.DTYPE_INT32:
            self.tik_instance.vec_conv(mask, "", dst_x_ub, origin_x_ub, repeat, 8, 8)
        else:
            self.tik_instance.data_move(dst_x_ub, origin_x_ub, 0, 1, (input_data_size * 4 + 31) // 32, 0, 0)
        self.tik_instance.vec_adds(mask, dst_x_ub, dst_x_ub, self.range_left_fp32, repeat, 8, 8)
        self.tik_instance.vec_muls(mask, dst_x_ub, dst_x_ub, self.scale_factor_fp32, repeat, 8, 8)
        self.tik_instance.vec_conv(mask, "floor", floor_x_ub_int32, dst_x_ub, repeat, 8, 8)
        self.tik_instance.vec_conv(mask, "", dst_x_ub, floor_x_ub_int32, repeat, 8, 8)
        clip_tmp_ub = self.tik_instance.Tensor(Constant.DTYPE_FP32, (input_data_size,), scope=tik.scope_ubuf,
                                               name="clip_tmp_ub")
        clip_value_int32 = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="clip_value_int32")
        clip_value_int32.set_as(self.nbins_int32 - 1)
        clip_value_fp32 = self.tik_instance.Scalar(Constant.DTYPE_FP32, name="clip_value_fp32")
        self.tik_instance.scalar_conv("", clip_value_fp32, clip_value_int32)
        self.tik_instance.vec_dup(mask, clip_tmp_ub, clip_value_fp32, repeat, 8)
        self.tik_instance.vec_min(mask, dst_x_ub, dst_x_ub, clip_tmp_ub, repeat, 8, 8, 8)
        self.tik_instance.vec_dup(mask, clip_tmp_ub, 0, repeat, 8)
        self.tik_instance.vec_max(mask, dst_x_ub, dst_x_ub, clip_tmp_ub, repeat, 8, 8, 8)

    def vector_floor_fp16(self, dst_x_ub, input_data_size, mask, repeat):
        floor_x_ub_int32 = self.tik_instance.Tensor(Constant.DTYPE_INT32, (input_data_size,), scope=tik.scope_ubuf,
                                                    name="floor_x_ub_int32")
        repeat_tmp = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="repeat_tmp")
        offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="offset")
        with self.tik_instance.if_scope(mask == 128):
            repeat_tmp.set_as(repeat * 2)
            with self.tik_instance.if_scope(repeat_tmp > 255):
                offset.set_as(255 * mask // 2)
                self.tik_instance.vec_conv(Constant.MASK_32BIT, "floor", floor_x_ub_int32, dst_x_ub, 255, 8, 4)
                self.tik_instance.vec_conv(Constant.MASK_32BIT, "floor", floor_x_ub_int32[offset], dst_x_ub[offset],
                                           repeat_tmp - 255, 8, 4)
                self.int32_to_fp16(dst_x_ub, floor_x_ub_int32, Constant.MASK_32BIT, 255)
                self.int32_to_fp16(dst_x_ub[offset], floor_x_ub_int32[offset], Constant.MASK_32BIT, repeat_tmp - 255)
            with self.tik_instance.else_scope():
                self.tik_instance.vec_conv(Constant.MASK_32BIT, "floor", floor_x_ub_int32, dst_x_ub, repeat_tmp, 8, 4)
                self.int32_to_fp16(dst_x_ub, floor_x_ub_int32, Constant.MASK_32BIT, repeat_tmp)
        with self.tik_instance.elif_scope(mask <= 64):
            self.tik_instance.vec_conv(mask, "floor", floor_x_ub_int32, dst_x_ub, repeat, 8, 4)
            self.int32_to_fp16(dst_x_ub, floor_x_ub_int32, mask, repeat)
        with self.tik_instance.else_scope():
            offset.set_as(Constant.MASK_32BIT)
            self.tik_instance.vec_conv(Constant.MASK_32BIT, "floor", floor_x_ub_int32, dst_x_ub, 1, 8, 4)
            self.tik_instance.vec_conv(mask - Constant.MASK_32BIT, "floor", floor_x_ub_int32[offset], dst_x_ub[offset],
                                       1, 8, 4)
            self.int32_to_fp16(dst_x_ub, floor_x_ub_int32, Constant.MASK_32BIT, 1)
            self.int32_to_fp16(dst_x_ub[offset], floor_x_ub_int32[offset], mask - Constant.MASK_32BIT, 1)

    def calc_floor_fp16(self, dst_x_ub, origin_x_ub, mask_pre, repeat_pre):
        """
        calc_floor_fp16

        Parameters
        ----------
        dst_x_ub : dst_x_ub
        origin_x_ub : origin_x_ub
        mask_pre : mask
        repeat_pre : repeat

        Returns
        -------
        None
        """
        input_data_size = mask_pre * repeat_pre
        repeat = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="repeat")
        mask = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="mask")
        if self.x_dtype == Constant.DTYPE_FP16:
            mask.set_as(mask_pre)
            repeat.set_as(repeat_pre)
            self.tik_instance.vec_adds(mask, dst_x_ub, origin_x_ub, self.range_left,
                                       repeat, 8, 8)
        elif self.x_dtype == Constant.DTYPE_INT32:
            self.int32_to_fp16(dst_x_ub, origin_x_ub, mask_pre, repeat_pre)
            mask.set_as(mask_pre * 2)
            repeat.set_as((repeat_pre + 1) / 2)
            self.tik_instance.vec_adds(mask, dst_x_ub, dst_x_ub, self.range_left,
                                       repeat, 8, 8)
        else:
            self.tik_instance.vec_conv(mask_pre, "", dst_x_ub, origin_x_ub, repeat_pre, 4, 8)
            mask.set_as(mask_pre * 2)
            repeat.set_as((repeat_pre + 1) / 2)
            self.tik_instance.vec_adds(mask, dst_x_ub, dst_x_ub, self.range_left, repeat, 8, 8)
        factor_tensor_src_tmp = self.tik_instance.Tensor(Constant.DTYPE_FP16, (Constant.MASK_16BIT,),
                                                         scope=tik.scope_ubuf, name="factor_tensor_src_tmp")
        factor_tensor_dst_tmp = self.tik_instance.Tensor(Constant.DTYPE_FP16, (Constant.MASK_16BIT,),
                                                         scope=tik.scope_ubuf, name="factor_tensor_dst_tmp")
        factor_tmp = self.tik_instance.Scalar(Constant.DTYPE_FP16, name="factor_tmp")
        self.tik_instance.vec_dup(mask, factor_tensor_src_tmp, self.factor_for_fp16, 1, 8)
        if not cce.api_check_support("tik.vec_rec_high_preci"):
            self.tik_instance.vec_rec(mask, factor_tensor_dst_tmp, factor_tensor_src_tmp, 1, 8, 8)
        else:
            wk_size = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="wk_size")
            wk_size.set_as((mask + Constant.BLOCK_16BIT - 1) // Constant.BLOCK_16BIT * Constant.BLOCK_16BIT * 4)
            work_tensor_ub = self.tik_instance.Tensor(Constant.DTYPE_FP32, (wk_size,), name="work_tensor_ub",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.vec_rec_high_preci(mask, factor_tensor_dst_tmp, factor_tensor_src_tmp, work_tensor_ub,
                                                 1, 8, 8)
        factor_tmp.set_as(factor_tensor_dst_tmp[0])
        self.tik_instance.vec_muls(mask, dst_x_ub, dst_x_ub, factor_tmp, repeat, 8, 8)
        self.tik_instance.vec_muls(mask, dst_x_ub, dst_x_ub, self.nbins, repeat, 8, 8)
        self.vector_floor_fp16(dst_x_ub, input_data_size, mask, repeat)

        clip_tmp_ub = self.tik_instance.Tensor(Constant.DTYPE_FP16, (input_data_size,),
                                               scope=tik.scope_ubuf, name="clip_tmp_ub")
        clip_value_int32 = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="clip_value_int32",
                                                    init_value=self.nbins_int32 - 1)
        clip_value_fp32 = self.tik_instance.Scalar(Constant.DTYPE_FP32, name="clip_value_fp32")
        clip_value_fp16 = self.tik_instance.Scalar(Constant.DTYPE_FP16, name="clip_value_fp16")
        if not cce.api_check_support("tik.scalar_conv"):
            self.scalar_int32_to_fp16(clip_value_fp16, clip_value_int32)
        else:
            self.tik_instance.scalar_conv("", clip_value_fp32, clip_value_int32)
            self.tik_instance.scalar_conv("", clip_value_fp16, clip_value_fp32)
        self.tik_instance.vec_dup(mask, clip_tmp_ub, clip_value_fp16, repeat, 8)
        self.tik_instance.vec_min(mask, dst_x_ub, dst_x_ub, clip_tmp_ub, repeat, 8, 8, 8)
        self.tik_instance.vec_dup(mask, clip_tmp_ub, 0, repeat, 8)
        self.tik_instance.vec_max(mask, dst_x_ub, dst_x_ub, clip_tmp_ub, repeat, 8, 8, 8)

    def reduce_to_bins_fp16(self, bins_value, input_data_size, pre_count):
        """
        reduce to bins for fp16 input

        Parameters
        ----------
        bins_value: bins_value
        input_data_size: input_data_size
        pre_count: pre_count

        Returns
        -------
        None
        """
        repeat = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='repeat')
        tail_mask = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='tail_mask')
        repeat_tail = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='repeat_tail')
        repeat_loop = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='repeat_loop')
        repeat.set_as(input_data_size // Constant.MASK_16BIT)
        tail_mask.set_as(input_data_size % Constant.MASK_16BIT)
        repeat_loop.set_as(repeat // 255)
        repeat_tail.set_as(repeat % 255)
        output_reduce_sum_int32 = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="output_reduce_sum_int32",
                                                           init_value=0)
        bins_value_fp32 = self.tik_instance.Scalar(Constant.DTYPE_FP32, name="bins_value_fp32")
        bins_value_fp16 = self.tik_instance.Scalar(Constant.DTYPE_FP16, name="bins_value_fp16")
        once_count_fp32 = self.tik_instance.Scalar(Constant.DTYPE_FP32, name="once_count_fp32")
        once_count_fp16 = self.tik_instance.Scalar(Constant.DTYPE_FP16, name="once_count_fp16")
        once_count_int32 = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="once_count_int32")
        ElementOpTuple = namedtuple("ElementOpTuple", ["mask", "repeat"])

        if not cce.api_check_support("tik.scalar_conv"):
            bins_value.set_as(bins_value - 1)
            self.scalar_int32_to_fp16(bins_value_fp16, bins_value)
        else:
            self.tik_instance.scalar_conv("", bins_value_fp32, bins_value)
            bins_value_fp32.set_as(bins_value_fp32 - 1)
            self.tik_instance.scalar_conv("", bins_value_fp16, bins_value_fp32)
        bins_value_tensor = self.tik_instance.Tensor(Constant.DTYPE_FP16, (input_data_size,), scope=tik.scope_ubuf,
                                                     name="bins_value_tensor")
        output_num_per_repeat = self.tik_instance.Tensor(Constant.DTYPE_FP16, (self.value_num_each_block,),
                                                         scope=tik.scope_ubuf, name="output_num_per_repeat")
        with self.tik_instance.for_range(0, repeat_loop) as index:
            input_offset = index * 255 * Constant.MASK_16BIT
            self.tik_instance.vec_dup(Constant.MASK_16BIT, bins_value_tensor, bins_value_fp16, 255, 8)
            element_op_tuple = ElementOpTuple(Constant.MASK_16BIT, 255)
            self.reduce_add(output_num_per_repeat, self.x_ub_fp16[input_offset], bins_value_tensor, element_op_tuple)
            once_count_fp16.set_as(output_num_per_repeat[0])
            if not cce.api_check_support("tik.scalar_conv"):
                self.scalar_fp16_to_int32(once_count_int32, once_count_fp16, "floor")
            else:
                self.tik_instance.scalar_conv("", once_count_fp32, once_count_fp16)
                self.tik_instance.scalar_conv("floor", once_count_int32, once_count_fp32)
            output_reduce_sum_int32.set_as(output_reduce_sum_int32 + once_count_int32)
        with self.tik_instance.if_scope(repeat_tail != 0):
            input_offset = repeat_loop * 255 * Constant.MASK_16BIT
            self.tik_instance.vec_dup(Constant.MASK_16BIT, bins_value_tensor, bins_value_fp16, repeat_tail, 8)
            element_op_tuple = ElementOpTuple(Constant.MASK_16BIT, repeat_tail)
            self.reduce_add(output_num_per_repeat, self.x_ub_fp16[input_offset], bins_value_tensor, element_op_tuple)
            once_count_fp16.set_as(output_num_per_repeat[0])
            if not cce.api_check_support("tik.scalar_conv"):
                self.scalar_fp16_to_int32(once_count_int32, once_count_fp16, "floor")
            else:
                self.tik_instance.scalar_conv("", once_count_fp32, once_count_fp16)
                self.tik_instance.scalar_conv("floor", once_count_int32, once_count_fp32)
            output_reduce_sum_int32.set_as(output_reduce_sum_int32 + once_count_int32)
        with self.tik_instance.if_scope(tail_mask != 0):
            input_offset = repeat * Constant.MASK_16BIT
            self.tik_instance.vec_dup(tail_mask, bins_value_tensor, bins_value_fp16, 1, 8)
            element_op_tuple = ElementOpTuple(tail_mask, 1)
            self.reduce_add(output_num_per_repeat, self.x_ub_fp16[input_offset], bins_value_tensor, element_op_tuple)
            once_count_fp16.set_as(output_num_per_repeat[0])
            if not cce.api_check_support("tik.scalar_conv"):
                self.scalar_fp16_to_int32(once_count_int32, once_count_fp16, "floor")
            else:
                self.tik_instance.scalar_conv("", once_count_fp32, once_count_fp16)
                self.tik_instance.scalar_conv("floor", once_count_int32, once_count_fp32)
            output_reduce_sum_int32.set_as(output_reduce_sum_int32 + once_count_int32)
        output_reduce_sum_int32.set_as(output_reduce_sum_int32 - pre_count)

        return output_reduce_sum_int32

    def per_core_cmpv_clip(self, src_ub, mask, input_data_size, loop_max_num):
        """
        compare and clip input vector
        Parameters
        ----------
        src_ub : operator ub
        mask : mask
        input_data_size : input_data_size
        loop_max_num : loop_max_num
        Returns
        -------
        None
        """
        repeat = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='repeat')
        tail_mask = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='tail_mask')
        repeat_tail = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='repeat_tail')
        repeat_loop = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='repeat_loop')

        value_dtype = src_ub.dtype
        loop_output_max_num = self.tik_instance.Tensor(value_dtype, (mask,), scope=tik.scope_ubuf,
                                                       name="loop_output_max_num")
        src_tmp = self.tik_instance.Tensor(value_dtype, (input_data_size,), scope=tik.scope_ubuf, name="src_tmp")
        if cce.get_soc_spec(cce.SHORT_SOC_VERSION) == "Ascend310":
            max_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="max_num")
            max_num.set_as(loop_max_num)
            loop_max_num_tmp = self.tik_instance.Tensor(Constant.DTYPE_INT32, (mask,), scope=tik.scope_ubuf,
                                                        name="loop_max_num_tmp")
            if mask > 64:
                self.tik_instance.vec_dup(mask // 2, loop_max_num_tmp, loop_max_num, 2, 8)
                self.tik_instance.vec_conv(mask // 2, "", loop_output_max_num, loop_max_num_tmp, 2, 4, 8, 1.0)
            else:
                self.tik_instance.vec_dup(mask, loop_max_num_tmp, loop_max_num, 1, 8)
                self.tik_instance.vec_conv(mask, "", loop_output_max_num, loop_max_num_tmp, 1, 1, 1, 1.0)
        else:
            self.tik_instance.vec_dup(mask, loop_output_max_num, loop_max_num, 1, 8)
        if value_dtype == Constant.DTYPE_FP16:
            repeat.set_as(input_data_size // Constant.MASK_16BIT)
            tail_mask.set_as(input_data_size % Constant.MASK_16BIT)
        else:
            repeat.set_as(input_data_size // Constant.MASK_32BIT)
            tail_mask.set_as(input_data_size % Constant.MASK_32BIT)
        repeat_loop.set_as(repeat // 255)
        repeat_tail.set_as(repeat % 255)
        self.tik_instance.vec_dup(mask, self.one, 1.0, 1, 8)
        self.tik_instance.vec_dup(mask, self.zero, 0.0, 1, 8)
        with self.tik_instance.for_range(0, repeat_loop) as index:
            input_offset = index * 255 * mask
            self.tik_instance.vec_sub(mask, src_tmp[input_offset], src_ub[input_offset], loop_output_max_num,
                                      255, 8, 8, 0)
            self.tik_instance.vec_min(mask, src_tmp[input_offset], src_tmp[input_offset], self.zero, 255, 8, 8, 0)
            self.tik_instance.vec_abs(mask, src_tmp[input_offset], src_tmp[input_offset], 255, 8, 8)
            self.tik_instance.vec_min(mask, src_tmp[input_offset], src_tmp[input_offset], self.one, 255, 8, 8, 0)
            self.tik_instance.vec_adds(mask, src_ub[input_offset], src_ub[input_offset], 1.0, 255, 8, 8)
            self.tik_instance.vec_mul(mask, src_ub[input_offset], src_ub[input_offset], src_tmp[input_offset],
                                      255, 8, 8, 8)
        with self.tik_instance.if_scope(repeat_tail != 0):
            input_offset = repeat_loop * 255 * mask
            self.tik_instance.vec_sub(mask, src_tmp[input_offset], src_ub[input_offset], loop_output_max_num,
                                      repeat_tail, 8, 8, 0)
            self.tik_instance.vec_min(mask, src_tmp[input_offset], src_tmp[input_offset], self.zero, repeat_tail,
                                      8, 8, 0)
            self.tik_instance.vec_abs(mask, src_tmp[input_offset], src_tmp[input_offset], repeat_tail, 8, 8)
            self.tik_instance.vec_min(mask, src_tmp[input_offset], src_tmp[input_offset], self.one, repeat_tail,
                                      8, 8, 0)
            self.tik_instance.vec_adds(mask, src_ub[input_offset], src_ub[input_offset], 1.0, repeat_tail, 8, 8)
            self.tik_instance.vec_mul(mask, src_ub[input_offset], src_ub[input_offset], src_tmp[input_offset],
                                      repeat_tail, 8, 8, 8)
        with self.tik_instance.if_scope(tail_mask != 0):
            input_offset = repeat * mask
            self.tik_instance.vec_sub(tail_mask, src_tmp[input_offset], src_ub[input_offset], loop_output_max_num,
                                      1, 8, 8, 0)
            self.tik_instance.vec_min(tail_mask, src_tmp[input_offset], src_tmp[input_offset], self.zero, 1,
                                      8, 8, 0)
            self.tik_instance.vec_abs(tail_mask, src_tmp[input_offset], src_tmp[input_offset], 1, 8, 8)
            self.tik_instance.vec_min(tail_mask, src_tmp[input_offset], src_tmp[input_offset], self.one, 1,
                                      8, 8, 0)
            self.tik_instance.vec_adds(tail_mask, src_ub[input_offset], src_ub[input_offset], 1.0, 1, 8, 8)
            self.tik_instance.vec_mul(tail_mask, src_ub[input_offset], src_ub[input_offset], src_tmp[input_offset],
                                      1, 8, 8, 8)

    def per_core_output_result(self, loop_output_num, output_offset, input_data_size):
        """
        per_core_output_result

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pre_count = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='pre_count', init_value=0)
        bins_value = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="bins_value")
        y_value = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="y_value")
        if self.x_dtype == Constant.DTYPE_FP16 and cce.api_check_support("tik.vreduce")\
                and cce.get_soc_spec(cce.SHORT_SOC_VERSION) != "Ascend310B":
            min_num = self.tik_instance.Scalar(self.x_dtype, name="min_num")
            max_num = self.tik_instance.Scalar(self.x_dtype, name="max_num")
            number = self.tik_instance.Scalar("uint32", name="number")
            counter = self.tik_instance.Scalar("uint32", name="counter")
            repeat = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="repeat")
            src_tmp = self.tik_instance.Tensor(self.x_dtype, (Constant.MASK_16BIT,), scope=tik.scope_ubuf,
                                               name='src_tmp')
            mask_ub = self.tik_instance.Tensor("uint16", (Constant.BYTE_FP16 * repeat,), scope=tik.scope_ubuf,
                                               name='mask_ub')
            repeat.set_as((input_data_size+Constant.MASK_16BIT) // Constant.MASK_16BIT)
            min_num.set_as(output_offset)
            max_num.set_as(output_offset+loop_output_num)
            self.tik_instance.vec_dup(Constant.MASK_16BIT, src_tmp, max_num, 1, 8)
            self.tik_instance.vec_cmpv_le(mask_ub, self.x_ub_fp16, src_tmp, repeat, 8, 0)
            self.tik_instance.vreduce(input_data_size, self.x_ub_fp16, self.x_ub_fp16, mask_ub, 1, 1, 8, 1, 0, number,
                                      "counter")
            self.tik_instance.vec_dup(Constant.MASK_16BIT, src_tmp, min_num, 1, 8)
            repeat.set_as((number + Constant.MASK_16BIT) // Constant.MASK_16BIT)
            self.tik_instance.vec_cmpv_ge(mask_ub, self.x_ub_fp16, src_tmp, repeat, 8, 0)
            self.tik_instance.vreduce(number, self.x_ub_fp16, self.x_ub_fp16, mask_ub, 1, 1, 8, 1, 0, counter,
                                      "counter")
            input_data_size = counter

        with self.tik_instance.new_stmt_scope():
            if not cce.api_check_support("tbe.dsl.vexp", "float32") or not cce.api_check_support("tik.scalar_conv") \
                    or self.x_dtype == Constant.DTYPE_FP16:
                self.per_core_cmpv_clip(self.x_ub_fp16, Constant.MASK_16BIT, input_data_size,
                                        (loop_output_num + output_offset))
            else:
                self.per_core_cmpv_clip(self.x_ub_fp32, Constant.MASK_32BIT, input_data_size,
                                        (loop_output_num + output_offset))

        with self.tik_instance.for_range(0, loop_output_num) as y_ub_index:
            bins_value.set_as(output_offset + loop_output_num - y_ub_index)
            if not cce.api_check_support("tbe.dsl.vexp", "float32") \
                    or not cce.api_check_support("tik.scalar_conv") \
                    or self.x_dtype == Constant.DTYPE_FP16:
                output_reduce_sum = self.reduce_to_bins_fp16(bins_value, input_data_size, pre_count)
                pre_count.set_as(pre_count + output_reduce_sum)
            else:
                output_reduce_sum = self.reduce_to_bins_fp32(bins_value, input_data_size, pre_count)
                pre_count.set_as(pre_count + output_reduce_sum)
            y_value.set_as(self.y_ub[loop_output_num - y_ub_index - 1])
            self.y_ub[loop_output_num - y_ub_index - 1].set_as(y_value + output_reduce_sum)

    def y_ub_to_gm(self, output_burst_len, output_offset, back_num):
        """
        output ub to gm

        Parameters
        ----------
        output_burst_len: output_burst_len
        output_offset: output_offset
        back_num: back_num

        Returns
        -------
        None
        """
        ub_move_len = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="ub_move_len")
        gm_move_len = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="gm_move_len")
        with self.tik_instance.if_scope(back_num == self.output_num_each_block):
            back_num.set_as(0)
        with self.tik_instance.if_scope(output_burst_len > 1):
            self.tik_instance.data_move(self.y_gm[output_offset], self.y_ub, 0, 1, output_burst_len - 1, 0, 0)
            ub_move_len.set_as((output_burst_len - 1) * self.output_num_each_block - back_num)
            gm_move_len.set_as(output_offset + (output_burst_len - 1) * self.output_num_each_block - back_num)
            tmp = self.tik_instance.Tensor(Constant.DTYPE_INT32, (self.output_num_each_block,), scope=tik.scope_ubuf,
                                           name="tmp")
            with self.tik_instance.for_range(0, self.output_num_each_block) as index:
                tmp[index].set_as(self.y_ub[ub_move_len + index])
            self.tik_instance.data_move(self.y_gm[gm_move_len], tmp, 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.y_gm[output_offset], self.y_ub, 0, 1, output_burst_len, 0, 0)

    def per_core_output_result_vdhist(self, loop_output_num, output_offset, input_data_size):
        loop_num = (loop_output_num + Constant.ELEMENT_UINT8 - 1) // Constant.ELEMENT_UINT8
        repeat_num = self.tik_instance.Scalar("int32", name="repeat_num")
        repeat_num.set_as((input_data_size + Constant.MASK_16BIT - 1) // Constant.MASK_16BIT)
        self.tik_instance.vec_adds(Constant.MASK_16BIT, self.x_ub_fp16, self.x_ub_fp16, -output_offset,
                                    repeat_num, 8, 8)  
        counter = self.tik_instance.Scalar("uint32", name="counter")
        repeat = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="repeat")
        tail_x_num = self.tik_instance.Scalar("uint32", name="tail_x_num")
        x_ub_tmp = self.tik_instance.Tensor(self.x_dtype, (input_data_size,), 
                                            scope=tik.scope_ubuf, name="x_ub_tmp")
        input_tmp = self.tik_instance.Tensor(self.x_dtype, (input_data_size,), 
                                             scope=tik.scope_ubuf, name="input_tmp")

        src_tmp = self.tik_instance.Tensor(self.x_dtype, (Constant.MASK_16BIT,), scope=tik.scope_ubuf,
                                            name="src_tmp")
        mask_ub = self.tik_instance.Tensor("uint16", (input_data_size,), 
                                           scope=tik.scope_ubuf, name="mask_ub")
        
        dst_tmp = self.tik_instance.Tensor("uint8", (input_data_size,), 
                                           scope=tik.scope_ubuf, name="dst_tmp")
        with self.tik_instance.for_range(0, loop_num) as tile_index:
            self.tik_instance.vec_dup(Constant.MASK_16BIT, src_tmp, 0, 1, 8)
            repeat.set_as((input_data_size + Constant.MASK_16BIT - 1) // Constant.MASK_16BIT)
            self.tik_instance.vec_adds(Constant.MASK_16BIT, x_ub_tmp, self.x_ub_fp16, 
                                       -(Constant.ELEMENT_UINT8 * tile_index), repeat, 8, 8)
            self.tik_instance.vec_cmpv_ge(mask_ub, x_ub_tmp, src_tmp, repeat, 8, 0)
            self.tik_instance.vec_sel(Constant.MASK_16BIT, 1, input_tmp, mask_ub, x_ub_tmp,
                                      Constant.ELEMENT_UINT8, repeat, 8, 8, 8)
            self.tik_instance.vec_dup(Constant.MASK_16BIT, src_tmp, Constant.MAX_UINT8, 1, 8)
            self.tik_instance.vec_cmpv_le(mask_ub, input_tmp, src_tmp, repeat, 8, 0)
            self.tik_instance.vreduce(input_data_size, x_ub_tmp, input_tmp, mask_ub, 1, 1, 8, 1, 0,
                                      counter, "counter")
            with self.tik_instance.if_scope(counter == 0):
                self.tik_instance.tik_continue()
            self.tik_instance.vec_conv(Constant.MASK_16BIT, "floor", dst_tmp, x_ub_tmp, repeat, 4, 8)
            loop_x_num = (counter + Constant.ELEMENT_UINT8 - 1) // Constant.ELEMENT_UINT8
            tail_x_num.set_as(counter % Constant.ELEMENT_UINT8)
            with self.tik_instance.if_scope(tail_x_num == 0):
                tail_x_num.set_as(Constant.ELEMENT_UINT8)
            self._hist_block(loop_x_num, tail_x_num, dst_tmp, tile_index)

    def reduce_add(self, output_ub, x_ub, bins_value_tensor, element_op_tuple):
        """
        reduce add func

        Parameters
        ----------
        output_ub: output_ub
        x_ub: x_ub
        bins_value_tensor: bins_value_tensor
        element_op_tuple: element_op_tuple

        Returns
        -------
        None
        """
        value_dtype = x_ub.dtype
        (mask, repeat) = element_op_tuple
        sub_rslt = self.tik_instance.Tensor(value_dtype, (mask * repeat,), scope=tik.scope_ubuf, name="sub_rslt")
        work_tensor = self.tik_instance.Tensor(value_dtype, (repeat,), scope=tik.scope_ubuf, name="work_tensor")
        self.tik_instance.vec_sub(mask, sub_rslt, x_ub, bins_value_tensor, repeat, 8, 8, 8)
        self.tik_instance.vec_max(mask, sub_rslt, sub_rslt, self.zero, repeat, 8, 8, 0)
        self.tik_instance.vec_min(mask, sub_rslt, sub_rslt, self.one, repeat, 8, 8, 0)
        self.tik_instance.vec_reduce_add(mask, output_ub, sub_rslt, work_tensor, repeat, 8)

    def histogram_fixed_width_compute(self):
        """
        Main process of HistogramFixedWidth
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        per_core_loop = self.tik_instance.Scalar(Constant.DTYPE_INT64, name='per_core_loop', init_value=0)
        tail_num = self.tik_instance.Scalar(Constant.DTYPE_INT64, name='tail_num', init_value=0)
        output_offset = self.tik_instance.Scalar(Constant.DTYPE_INT64, name='output_offset', init_value=0)
        tail_burst_len = self.tik_instance.Scalar(Constant.DTYPE_INT64, name='tail_burst_len', init_value=0)
        per_loop_back_num = self.tik_instance.Scalar(Constant.DTYPE_INT64, name='per_loop_back_num', init_value=0)
        tail_num_back_num = self.tik_instance.Scalar(Constant.DTYPE_INT64, name='tail_num_back_num', init_value=0)
        with self.tik_instance.for_range(0, self.total_core_number, block_num=self.total_core_number) as core_id:
            with self.tik_instance.if_scope(core_id < self.used_core_num):
                with self.tik_instance.if_scope(core_id < self.used_core_num - 1):
                    per_core_loop.set_as(self.per_core_calc_num // self.output_ub_max_tensor_size)
                    tail_num.set_as(self.per_core_calc_num % self.output_ub_max_tensor_size)
                    tail_burst_len.set_as((tail_num + self.output_num_each_block - 1) // self.output_num_each_block)
                    per_loop_back_num.set_as(
                        self.output_num_each_block - self.per_core_calc_num % self.output_num_each_block)
                    tail_num_back_num.set_as(self.output_num_each_block - tail_num % self.output_num_each_block)
                with self.tik_instance.else_scope():
                    per_core_loop.set_as(self.last_core_calc_num // self.output_ub_max_tensor_size)
                    tail_num.set_as(self.last_core_calc_num % self.output_ub_max_tensor_size)
                    tail_burst_len.set_as((tail_num + self.output_num_each_block - 1) // self.output_num_each_block)
                    per_loop_back_num.set_as(
                        self.output_num_each_block - self.last_core_calc_num % self.output_num_each_block)
                    tail_num_back_num.set_as(self.output_num_each_block - tail_num % self.output_num_each_block)
                with self.tik_instance.for_range(0, per_core_loop) as index:
                    output_offset.set_as(core_id * self.per_core_calc_num + index * self.output_ub_max_tensor_size)
                    self.calc_per_core(self.output_ub_max_tensor_size, output_offset, self.output_ub_max_block_len,
                                       per_loop_back_num)
                with self.tik_instance.if_scope(tail_num != 0):
                    output_offset.set_as(
                        core_id * self.per_core_calc_num + per_core_loop * self.output_ub_max_tensor_size)
                    self.calc_per_core(tail_num, output_offset, tail_burst_len, tail_num_back_num)
        tbe_context.get_context().add_compile_info('vars',
                                                   {'core_num': self.total_core_number,
                                                    'ub_size': self.ub_size_bytes,
                                                    'is_vector_histv2_support': self.is_vector_histv2_support
                                                    })
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.x_gm, self.range_gm, self.nbins_gm],
            outputs=[self.y_gm],
            flowtable=[self.tiling_gm])

        return self.tik_instance

    def _hist_block(self, loop_x_num, tail_x_num, dst_tmp, tile_index):
        y_tmp = self.tik_instance.Tensor("uint16", (Constant.ELEMENT_UINT8,), scope=tik.scope_ubuf, name="y_tmp")
        with self.tik_instance.for_range(0, loop_x_num) as x_idx:
            self.tik_instance.vec_dup(Constant.MASK_16BIT, y_tmp, 0, 2, 8)
            mask = self.tik_instance.Scalar("int16", name="mask")
            mask.set_as(Constant.ELEMENT_UINT8)
            with self.tik_instance.if_scope(x_idx == loop_x_num - 1):
                mask.set_as(tail_x_num)
            x_offset = x_idx * Constant.ELEMENT_UINT8
            dst_v0 = self.tik_instance.Vector("uint16", name="vector_dst0")
            dst_v1 = self.tik_instance.Vector("uint16", name="vector_dst1")
            src_v0 = self.tik_instance.Vector("uint8", name="vector_src0")
            self.tik_instance.vector_load(src_v0, dst_tmp[x_offset])
            self.tik_instance.vector_vdup(None, dst_v0, 0, 0, False)
            self.tik_instance.vector_vdup(None, dst_v1, 0, 0, False)
            self.tik_instance.vector_vdhistv2(mask, dst_v0, dst_v1, src_v0)
            self.tik_instance.vector_store(y_tmp[0:128], dst_v0)
            self.tik_instance.vector_store(y_tmp[128:], dst_v1)
            
            y_idx_start = tile_index * Constant.ELEMENT_UINT8
            y_tmp_uint32 = self.tik_instance.Tensor("uint32", (Constant.ELEMENT_UINT8,), 
                                                    scope=tik.scope_ubuf, name="y_tmp_uint32")
            y_tmp_int16 = self.tik_instance.Tensor("int16", (Constant.ELEMENT_UINT8,), 
                                                   scope=tik.scope_ubuf, name="y_tmp_int16")
            y_tmp_int32 = self.tik_instance.Tensor("int32", (Constant.ELEMENT_UINT8,), 
                                                   scope=tik.scope_ubuf, name="y_tmp_int32")
            self.tik_instance.vcbd(Constant.MASK_32BIT, y_tmp_uint32, y_tmp, 4, 1, 1, 8, 4)
            self.tik_instance.vcbd(Constant.MASK_32BIT, y_tmp_int16, y_tmp_uint32, 4, 1, 1, 4, 8)
            self.tik_instance.vcbd(Constant.MASK_32BIT, y_tmp_int32, y_tmp_int16, 4, 1, 1, 8, 4)
            self.tik_instance.vec_add(Constant.MASK_32BIT, self.y_ub[y_idx_start:], y_tmp_int32, 
                                      self.y_ub[y_idx_start:], Constant.ELEMENT_UINT8 // Constant.MASK_32BIT,
                                       8, 8, 8)

    def _init_gm(self):
        """
        init gm

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.nbins_gm = self.tik_instance.Tensor(self.nbins_dtype, (Constant.MAX_INT32,), name="nbins_gm",
                                                 scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,), scope=tik.scope_gm,
                                                  name="tiling_gm")
        self.x_gm = self.tik_instance.Tensor(self.x_dtype, (Constant.MAX_INT32,), scope=tik.scope_gm, name="x_gm")
        self.y_gm = self.tik_instance.Tensor(self.y_dtype, (Constant.MAX_INT32,), scope=tik.scope_gm, name="y_gm")
        self.range_gm = self.tik_instance.Tensor(self.range_dtype, (self.value_num_each_block,), scope=tik.scope_gm,
                                                 name="range_gm")

    def _init_range_data(self):
        """
        init range data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.range_ub, self.range_gm, 0, 1, self.value_num_each_block, 0, 0)
        self.range_left = self.tik_instance.Scalar(Constant.DTYPE_FP16, name="range_left",
                                                   init_value=self.range_ub[0])
        self.range_right = self.tik_instance.Scalar(Constant.DTYPE_FP16, name="range_right",
                                                    init_value=self.range_ub[1])
        range_tmp_int32 = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="range_tmp_int32")
        self.range_left_fp32 = self.tik_instance.Scalar(Constant.DTYPE_FP32, name="range_left_fp32")
        self.range_right_fp32 = self.tik_instance.Scalar(Constant.DTYPE_FP32, name="range_right_fp32")
        if self.range_dtype == Constant.DTYPE_INT32:
            range_tmp_int32.set_as(self.range_ub[0])
            if not cce.api_check_support("tik.scalar_conv"):
                self.scalar_int32_to_fp16(self.range_left, range_tmp_int32)
                range_tmp_int32.set_as(self.range_ub[1])
                self.scalar_int32_to_fp16(self.range_right, range_tmp_int32)
            else:
                self.tik_instance.scalar_conv("", self.range_left_fp32, range_tmp_int32)
                self.tik_instance.scalar_conv("", self.range_left, self.range_left_fp32)
                range_tmp_int32.set_as(self.range_ub[1])
                self.tik_instance.scalar_conv("", self.range_right_fp32, range_tmp_int32)
                self.tik_instance.scalar_conv("", self.range_right, self.range_right_fp32)
        elif self.range_dtype == Constant.DTYPE_FP32:
            self.range_left_fp32.set_as(self.range_ub[0])
            self.range_right_fp32.set_as(self.range_ub[1])
            if not cce.api_check_support("tik.scalar_conv"):
                self.scalar_fp32_to_fp16(self.range_right, self.range_right_fp32)
                self.scalar_fp32_to_fp16(self.range_left, self.range_left_fp32)
            else:
                self.tik_instance.scalar_conv("", self.range_left, self.range_left_fp32)
                self.tik_instance.scalar_conv("", self.range_right, self.range_right_fp32)
        else:
            if not cce.api_check_support("tik.scalar_conv"):
                self.scalar_fp16_to_fp32(self.range_left_fp32, self.range_left)
                self.scalar_fp16_to_fp32(self.range_right_fp32, self.range_right)
            else:
                self.tik_instance.scalar_conv("", self.range_left_fp32, self.range_left)
                self.tik_instance.scalar_conv("", self.range_right_fp32, self.range_right)

    def _init_tiling_data(self):
        """
        init tiling data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,), scope=tik.scope_ubuf,
                                                  name="tiling_ub")
        self.is_datasize_support_vdhist = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                                   name="is_datasize_support_vdhist", init_value=0)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
        self.used_core_num = self.tik_instance.Scalar(self.tiling_dtype, name="used_core_num")
        self.total_output_num = self.tik_instance.Scalar(self.tiling_dtype, name='total_output_num')
        self.per_core_calc_num = self.tik_instance.Scalar(self.tiling_dtype, name='per_core_calc_num')
        self.last_core_calc_num = self.tik_instance.Scalar(self.tiling_dtype, name='last_core_calc_num')
        self.input_ub_max_tensor_size = self.tik_instance.Scalar(self.tiling_dtype, name='ub_max_tensor_size')
        self.output_ub_max_tensor_size = self.tik_instance.Scalar(self.tiling_dtype, name='ub_max_tensor_size')
        self.total_values_num = self.tik_instance.Scalar(self.tiling_dtype, name="total_values_num")
        self.input_ub_max_block_len = self.tik_instance.Scalar(self.tiling_dtype, name="input_ub_max_block_len")
        self.output_ub_max_block_len = self.tik_instance.Scalar(self.tiling_dtype, name="output_ub_max_block_len")
        self.nbins_int32 = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='nbins_int32')
        nbins_int64 = self.tik_instance.Scalar(Constant.DTYPE_INT64, name='nbins_int64')

        self.used_core_num.set_as(self.tiling_ub[0])
        self.total_output_num.set_as(self.tiling_ub[1])
        self.per_core_calc_num.set_as(self.tiling_ub[2])
        self.last_core_calc_num.set_as(self.tiling_ub[3])
        self.input_ub_max_tensor_size.set_as(self.tiling_ub[4])
        self.output_ub_max_tensor_size.set_as(self.tiling_ub[5])
        nbins_int64.set_as(self.tiling_ub[6])
        self.nbins_int32.set_as(nbins_int64)
        self.total_values_num.set_as(self.tiling_ub[7])
        self.input_ub_max_block_len.set_as(self.tiling_ub[8])
        self.output_ub_max_block_len.set_as(self.tiling_ub[9])
        if self.is_vector_histv2_support == True:
            with self.tik_instance.if_scope(self.total_values_num < Constant.HIGH_PREC_MAX_TENSOR):
                self.is_datasize_support_vdhist.set_as(0)
            with self.tik_instance.else_scope():
                self.is_datasize_support_vdhist.set_as(1)
        else:
            self.is_datasize_support_vdhist.set_as(0)

    def _init_preprocess_factor(self):
        """
        init scale factor

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.scale_factor = self.tik_instance.Scalar(Constant.DTYPE_FP16, name="range_dev")
        # nbins
        self.nbins_fp32 = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='nbins_fp32')
        self.nbins = self.tik_instance.Scalar(Constant.DTYPE_FP16, name='nbins')
        self.scale_factor_fp32 = self.tik_instance.Scalar(Constant.DTYPE_FP32, name="scale_factor_fp32")
        self.factor_for_fp16 = self.tik_instance.Scalar(Constant.DTYPE_FP16, name="factor_for_fp16")
        if not cce.api_check_support("tik.scalar_conv"):
            tmp_tensor_int32 = self.tik_instance.Tensor(Constant.DTYPE_INT32, (self.output_num_each_block,),
                                                        scope=tik.scope_ubuf, name="tmp_tensor_int32")
            nbins_fp16 = self.tik_instance.Tensor(Constant.DTYPE_FP16, (self.output_num_each_block,),
                                                  scope=tik.scope_ubuf, name="nbins_fp16")
            tmp_range_fp16 = self.tik_instance.Tensor(Constant.DTYPE_FP16, (self.output_num_each_block,),
                                                      scope=tik.scope_ubuf, name="tmp_range_fp16")
            tmp_tensor_int32[0].set_as(self.nbins_int32)
            tmp_range_fp16[0].set_as(self.range_left)
            self.tik_instance.vec_conv(1, "", nbins_fp16, tmp_tensor_int32, 1, 1, 1, 1.0)
            self.tik_instance.vec_muls(1, tmp_range_fp16, tmp_range_fp16, -1.0, 1, 1, 1)
            self.range_left.set_as(tmp_range_fp16[0])
            self.tik_instance.vec_adds(1, tmp_range_fp16, tmp_range_fp16, self.range_right, 1, 1, 1)
            self.factor_for_fp16.set_as(tmp_range_fp16[0])
            self.scalar_int32_to_fp16(self.nbins, self.nbins_int32)
        else:
            self.tik_instance.scalar_conv("", self.nbins_fp32, self.nbins_int32)
            self.tik_instance.scalar_conv("", self.nbins, self.nbins_fp32)
            # range_left
            self.range_left_fp32.set_as(-self.range_left_fp32)
            self.tik_instance.scalar_conv("", self.range_left, self.range_left_fp32)
            # scale_factor
            self.range_right_fp32.set_as(self.range_right_fp32 + self.range_left_fp32)
            with self.tik_instance.if_scope(self.range_right_fp32 != 0):
                self.scale_factor_fp32.set_as(self.nbins_fp32 / self.range_right_fp32)
            with self.tik_instance.else_scope():
                self.scale_factor_fp32.set_as(0)
            self.tik_instance.scalar_conv("", self.factor_for_fp16, self.range_right_fp32)
            self.tik_instance.scalar_conv("", self.scale_factor, self.scale_factor_fp32)


def _check_param(x, input_range, y, kernel_name):
    """
    check parameters

    Parameters
    ----------
    x: dict
        dict info of input value, must include the keys(shape and dtype).
    input_range: dict
        dict info of input value_range, must include the keys(shape and dtype).
                        the shape must be (2,) or [2]
    nbins: dict
        number of histogram bins.
    y: dict
        dict info of output
    dtype: str
        data type for returned histogram.
    kernel_name: str
        cce kernel name, default value is "histogram_fixed_width"


    returns
    -------
    None
    """
    x_dtype = x.get("dtype").lower()
    para_check.check_dtype(x_dtype, ("float16", "float32", "int32"), param_name="x")
    range_dtype = input_range.get("dtype").lower()
    if range_dtype != x_dtype:
        rule = "range type not match x type"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)

    y_dtype = y.get("dtype").lower()
    para_check.check_dtype(y_dtype, ("int32"), param_name="y")


@register_operator("HistogramFixedWidth")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def histogram_fixed_width(x,
                          range,
                          nbins,
                          y,
                          dtype=3,
                          kernel_name='histogram_fixed_width'):
    """
    this operation returns a rank 1 histogram counting
     the number of entries in `values` that fell into every bin.
      The bins are equal width and determined by the arguments
    `value_range` and `nbins`.

    Parameters
    ----------
    x: dict
        dict info of input value, must include the keys(shape and dtype).
    range: dict
        dict info of input value_range, must include the keys(shape and dtype).
                        the shape must be (2,) or [2]
    nbins: dict
        dict info of input nbins, must include the keys(shape and dtype).
    y: dict
        dict info of output
    dtype: int
        data type for returned histogram.
    kernel_name: str
        cce kernel name, default value is "histogram_fixed_width"


    returns
    -------
    None
    """
    _check_param(x, range, y, kernel_name)
    histogram_fixed_width_instance = HistogramFixedWidth(x, range, nbins, y, dtype, kernel_name)
    tik_instance = histogram_fixed_width_instance.histogram_fixed_width_compute()

    return tik_instance
