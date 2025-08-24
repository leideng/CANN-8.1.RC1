#!/usr/bin/env python
# coding: utf-8
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
tik_op_base
"""
# 'pylint: disable=too-many-arguments,no-self-use
import functools
import impl.constant_util as constant
from impl import common_util

MAX_REPEAT_NUM = 255


class TikOpBase:
    """
        Function: use to store concat base parameters
    """
    def __init__(self, tik_instance):
        self.tik_instance = tik_instance

    def get_data_size(self, data_type):
        """
        get_data_size
        """
        return common_util.get_data_size(data_type)

    def get_shape_size(self, data_shape):
        """
        get_shape_size
        """
        data_size = int(functools.reduce(lambda i, j: i * j, data_shape))
        return data_size

    def _get_mask_and_repeat(self, data_shape, data_type):
        data_size = int(functools.reduce(lambda i, j: i * j, data_shape))
        data_byte_num = common_util.get_data_size(data_type)
        one_block_num = constant.BLOCK_SIZE // data_byte_num

        front_mask = constant.REPEAT_STRIDE_EIGHT*one_block_num
        if data_size <= front_mask:
            front_mask = data_size
            last_mask = data_size
            repeat_times = constant.REPEAT_TIME_ONCE

            return front_mask, last_mask, repeat_times

        # in this case, repeat is greater than 1
        repeat_times = data_size // front_mask
        last_mask = front_mask
        if data_size % front_mask != 0:
            last_mask = data_size - repeat_times*front_mask
            repeat_times = repeat_times + 1

        return front_mask, last_mask, repeat_times

    def _double_vector_func(self, func_name, dest, src0, src1, compute_shape,
                            dest_offset=0, src0_offset=0, src1_offset=0):
        front_mask, last_mask, repeat_times = \
            self._get_mask_and_repeat(compute_shape, dest.dtype)
        if repeat_times == 1:
            func_name(front_mask, dest[dest_offset],
                      src0[src0_offset], src1[src1_offset],
                      constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
        elif repeat_times <= MAX_REPEAT_NUM:
            if front_mask == last_mask:
                func_name(front_mask, dest[dest_offset],
                          src0[src0_offset], src1[src1_offset],
                          repeat_times,
                          constant.STRIDE_ONE, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
            else:
                func_name(front_mask, dest[dest_offset],
                          src0[src0_offset], src1[src1_offset],
                          repeat_times - 1,
                          constant.STRIDE_ONE, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
                vector_offset = (repeat_times - 1)*front_mask
                func_name(last_mask, dest[dest_offset + vector_offset],
                          src0[src0_offset + vector_offset],
                          src1[src1_offset + vector_offset],
                          constant.REPEAT_TIME_ONCE,
                          constant.STRIDE_ONE, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)

        else:
            rest_repeat_num = repeat_times
            count = 0
            while rest_repeat_num > MAX_REPEAT_NUM:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], src0[vector_offset],
                          src1[vector_offset], MAX_REPEAT_NUM, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.STRIDE_ONE,
                          constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
                count = count + 1
                rest_repeat_num = rest_repeat_num - MAX_REPEAT_NUM
            if rest_repeat_num != 1:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], src0[vector_offset],
                          src1[vector_offset], rest_repeat_num - 1,
                          constant.STRIDE_ONE, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], src0[vector_offset],
                      src1[vector_offset], constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)

    def _vector_scalar_func(self, func_name, dest, src0, scalar_val,
                            compute_shape):
        front_mask, last_mask, repeat_times = \
            self._get_mask_and_repeat(compute_shape, dest.dtype)
        if repeat_times == 1:
            func_name(front_mask, dest, src0, scalar_val,
                      constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
        elif repeat_times <= MAX_REPEAT_NUM:
            func_name(front_mask, dest, src0, scalar_val, repeat_times - 1,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], src0[vector_offset],
                      scalar_val, constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
        else:
            rest_repeat_num = repeat_times
            count = 0
            while rest_repeat_num > MAX_REPEAT_NUM:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], src0[vector_offset],
                          scalar_val, MAX_REPEAT_NUM, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
                count = count + 1
                rest_repeat_num = rest_repeat_num - MAX_REPEAT_NUM
            if rest_repeat_num != 1:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], src0[vector_offset],
                          scalar_val, rest_repeat_num - 1, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], src0[vector_offset],
                      scalar_val, constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)

    def vector_dup_func(self, dest, scalar_val, compute_shape):
        """
        vector_dup_func
        """
        func_name = self.tik_instance.vector_dup
        front_mask, last_mask, repeat_times = \
            self._get_mask_and_repeat(compute_shape, dest.dtype)
        if repeat_times == 1:
            func_name(front_mask, dest, scalar_val, constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
        elif repeat_times <= MAX_REPEAT_NUM:
            func_name(front_mask, dest, scalar_val, repeat_times - 1,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], scalar_val,
                      constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
        else:
            rest_repeat_num = repeat_times
            count = 0
            while rest_repeat_num > MAX_REPEAT_NUM:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], scalar_val,
                          MAX_REPEAT_NUM, constant.STRIDE_ONE,
                          constant.REPEAT_STRIDE_EIGHT)
                count = count + 1
                rest_repeat_num = rest_repeat_num - MAX_REPEAT_NUM
            if rest_repeat_num != 1:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], scalar_val,
                          rest_repeat_num - 1,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], scalar_val,
                      constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT)

    def _single_vector_func(self, func_name, dest, src, compute_shape):
        front_mask, last_mask, repeat_times = \
            self._get_mask_and_repeat(compute_shape, dest.dtype)
        if repeat_times == 1:
            func_name(front_mask, dest, src, constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
        elif repeat_times <= MAX_REPEAT_NUM:
            func_name(front_mask, dest, src, repeat_times - 1,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], src[vector_offset],
                      constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
        else:
            rest_repeat_num = repeat_times
            count = 0
            while rest_repeat_num > MAX_REPEAT_NUM:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], src[vector_offset],
                          MAX_REPEAT_NUM, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
                count = count + 1
                rest_repeat_num = rest_repeat_num - MAX_REPEAT_NUM
            if rest_repeat_num != 1:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], src[vector_offset],
                          rest_repeat_num - 1, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], src[vector_offset],
                      constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)

    def vmul_func(self, dest, src0, src1, compute_shape, dest_offset=0,
                  src0_offset=0, src1_offset=0):
        """
        vmul_func
        """
        self._double_vector_func(self.tik_instance.vmul, dest, src0, src1,
                                 compute_shape, dest_offset, src0_offset,
                                 src1_offset)

    def vadd_func(self, dest, src0, src1, compute_shape, dest_offset=0,
                  src0_offset=0, src1_offset=0):
        """
        vadd_func
        """
        self._double_vector_func(self.tik_instance.vadd, dest, src0, src1,
                                 compute_shape, dest_offset, src0_offset,
                                 src1_offset)

    def vsub_func(self, dest, src0, src1, compute_shape):
        """
        vsub_func
        """
        self._double_vector_func(self.tik_instance.vsub, dest, src0, src1,
                                 compute_shape)

    def move_data(self, dest, src, data_type, copy_shape):
        """
        move_data
        """
        byte_num_one = common_util.get_data_size(data_type)
        copy_size = self.get_shape_size(copy_shape)
        one_block_ele_num = constant.BLOCK_SIZE // byte_num_one
        block_num = copy_size // one_block_ele_num
        self.tik_instance.data_move(dest, src, constant.SID,
                                    constant.DEFAULT_NBURST, block_num,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
