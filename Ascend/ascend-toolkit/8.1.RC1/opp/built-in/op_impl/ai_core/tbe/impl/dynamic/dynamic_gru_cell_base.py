"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tik op base
"""
import functools
import impl.constant_util as constant
from impl import common_util


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    MAX_REPEAT_NUM = 255


class TikOpBase:
    """
        Function: use to store concat base parameters
    """

    def __init__(self, tik_instance):
        self.tik_instance = tik_instance

    @staticmethod
    def get_data_size(data_type):
        """
        get_data_size
        """
        return common_util.get_data_size(data_type)

    @staticmethod
    def get_shape_size(data_shape):
        """
        get_shape_size
        """
        data_size = functools.reduce(lambda i, j: i * j, data_shape)
        return data_size

    # 'pylint: disable=too-many-arguments
    def vmul_func(self, dest, src0, src1, compute_shape, dest_offset=0, src0_offset=0, src1_offset=0):
        """
        vmul_func
        """
        self._double_vector_func(self.tik_instance.vmul, dest, src0, src1, compute_shape, dest_offset, src0_offset,
                                 src1_offset)

    # 'pylint: disable=too-many-arguments
    def vadd_func(self, dest, src0, src1, compute_shape, dest_offset=0, src0_offset=0, src1_offset=0):
        """
        vadd_func
        """
        self._double_vector_func(self.tik_instance.vadd, dest, src0, src1, compute_shape, dest_offset, src0_offset,
                                 src1_offset)

    def vsub_func(self, dest, src0, src1, compute_shape):
        """
        vsub_func
        """
        self._double_vector_func(self.tik_instance.vsub, dest, src0, src1, compute_shape)

    def move_data(self, dest, src, data_type, copy_shape):
        """
        move_data
        """
        byte_num_one = common_util.get_data_size(data_type)
        copy_size = self.get_shape_size(copy_shape)
        one_block_ele_num = constant.BLOCK_SIZE // byte_num_one
        block_num = (copy_size + one_block_ele_num - 1) // one_block_ele_num
        self.tik_instance.data_move(dest, src, constant.SID, constant.DEFAULT_NBURST, block_num, constant.STRIDE_ZERO,
                                    constant.STRIDE_ZERO)

    def vector_dup_func(self, dest, scalar_val, compute_shape):
        """
        vector_dup_func
        """
        func_name = self.tik_instance.vector_dup
        front_mask, last_mask, repeat_times = \
            self._get_mask_and_repeat(compute_shape, dest.dtype)
        with self.tik_instance.if_scope(repeat_times == 1):
            func_name(front_mask, dest, scalar_val, constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(repeat_times > Constant.MAX_REPEAT_NUM):
                rest_repeat_num = repeat_times
                count = repeat_times // Constant.MAX_REPEAT_NUM
                with self.tik_instance.for_range(0, count) as loop_i:
                    vector_offset = loop_i * Constant.MAX_REPEAT_NUM * front_mask

                    func_name(front_mask, dest[vector_offset], scalar_val, Constant.MAX_REPEAT_NUM, constant.STRIDE_ONE,
                              constant.REPEAT_STRIDE_EIGHT)
                    rest_repeat_num = rest_repeat_num - Constant.MAX_REPEAT_NUM
                with self.tik_instance.if_scope(rest_repeat_num != 1):
                    vector_offset = count * Constant.MAX_REPEAT_NUM * front_mask
                    func_name(front_mask, dest[vector_offset], scalar_val, rest_repeat_num - 1, constant.STRIDE_ONE,
                              constant.REPEAT_STRIDE_EIGHT)
                vector_offset = (repeat_times - 1) * front_mask
                func_name(last_mask, dest[vector_offset], scalar_val, constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                          constant.REPEAT_STRIDE_EIGHT)
            with self.tik_instance.else_scope():
                func_name(front_mask, dest, scalar_val, repeat_times - 1, constant.STRIDE_ONE,
                          constant.REPEAT_STRIDE_EIGHT)
                vector_offset = (repeat_times - 1) * front_mask
                func_name(last_mask, dest[vector_offset], scalar_val, constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                          constant.REPEAT_STRIDE_EIGHT)

    def _get_mask_and_repeat(self, data_shape, data_type):
        data_size = functools.reduce(lambda i, j: i * j, data_shape)
        data_byte_num = common_util.get_data_size(data_type)
        one_block_num = constant.BLOCK_SIZE // data_byte_num

        front_mask = self.tik_instance.Scalar('int64',
                                              name='front_mask',
                                              init_value=constant.REPEAT_STRIDE_EIGHT * one_block_num)
        last_mask = self.tik_instance.Scalar('int64', name='last_mask')
        repeat_times = self.tik_instance.Scalar('int64', name='repeat_times')

        with self.tik_instance.if_scope(data_size > front_mask):
            repeat_times.set_as(data_size // front_mask)
            last_mask.set_as(front_mask)
            with self.tik_instance.if_scope(data_size % front_mask != 0):
                last_mask.set_as(data_size - repeat_times * front_mask)
                repeat_times.set_as(repeat_times + 1)
        with self.tik_instance.else_scope():
            front_mask.set_as(data_size)
            last_mask.set_as(data_size)
            repeat_times.set_as(constant.REPEAT_TIME_ONCE)

        return front_mask, last_mask, repeat_times

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _double_vector_func(self,
                            func_name,
                            dest,
                            src0,
                            src1,
                            compute_shape,
                            dest_offset=0,
                            src0_offset=0,
                            src1_offset=0):
        front_mask, last_mask, repeat_times = \
            self._get_mask_and_repeat(compute_shape, dest.dtype)
        with self.tik_instance.if_scope(repeat_times == 1):
            func_name(front_mask, dest[dest_offset], src0[src0_offset], src1[src1_offset], constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(repeat_times > Constant.MAX_REPEAT_NUM):
                rest_repeat_num = repeat_times
                loop_count = repeat_times // Constant.MAX_REPEAT_NUM
                with self.tik_instance.for_range(0, loop_count) as i:
                    vector_offset = i * Constant.MAX_REPEAT_NUM * front_mask
                    func_name(front_mask, dest[vector_offset], src0[vector_offset], src1[vector_offset],
                    Constant.MAX_REPEAT_NUM, constant.STRIDE_ONE, constant.STRIDE_ONE, constant.STRIDE_ONE,
                    constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
                    rest_repeat_num = rest_repeat_num - Constant.MAX_REPEAT_NUM
                with self.tik_instance.if_scope(rest_repeat_num != 1):
                    vector_offset = loop_count * Constant.MAX_REPEAT_NUM * front_mask
                    func_name(front_mask, dest[vector_offset], src0[vector_offset], src1[vector_offset],
                              rest_repeat_num - 1, constant.STRIDE_ONE, constant.STRIDE_ONE, constant.STRIDE_ONE,
                              constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
                vector_offset = (repeat_times - 1) * front_mask
                func_name(last_mask, dest[vector_offset], src0[vector_offset], src1[vector_offset],
                          constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE, constant.STRIDE_ONE, constant.STRIDE_ONE,
                          constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)

            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(front_mask == last_mask):
                    func_name(front_mask, dest[dest_offset], src0[src0_offset], src1[src1_offset], repeat_times,
                              constant.STRIDE_ONE, constant.STRIDE_ONE, constant.STRIDE_ONE,
                              constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
                with self.tik_instance.else_scope():
                    func_name(front_mask, dest[dest_offset], src0[src0_offset], src1[src1_offset], repeat_times - 1,
                              constant.STRIDE_ONE, constant.STRIDE_ONE, constant.STRIDE_ONE,
                              constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
                    vector_offset = (repeat_times - 1) * front_mask
                    func_name(last_mask, dest[dest_offset + vector_offset], src0[src0_offset + vector_offset],
                              src1[src1_offset + vector_offset], constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                              constant.STRIDE_ONE, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)

    # 'pylint: disable=too-many-arguments
    def _vector_scalar_func(self, func_name, dest, src0, scalar_val, compute_shape):
        front_mask, last_mask, repeat_times = \
            self._get_mask_and_repeat(compute_shape, dest.dtype)
        with self.tik_instance.if_scope(repeat_times == 1):
            func_name(front_mask, dest, src0, scalar_val, constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(repeat_times > Constant.MAX_REPEAT_NUM):
                rest_repeat_num = repeat_times
                loop_count = rest_repeat_num // Constant.MAX_REPEAT_NUM
                with self.tik_instance.for_range(0, loop_count) as i:
                    vector_offset = i * Constant.MAX_REPEAT_NUM * front_mask
                    func_name(front_mask, dest[vector_offset], src0[vector_offset], scalar_val, Constant.MAX_REPEAT_NUM,
                              constant.STRIDE_ONE, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT)
                    rest_repeat_num = rest_repeat_num - Constant.MAX_REPEAT_NUM

                with self.tik_instance.if_scope(rest_repeat_num != 1):
                    vector_offset = loop_count * Constant.MAX_REPEAT_NUM * front_mask
                    func_name(front_mask, dest[vector_offset], src0[vector_offset], scalar_val, rest_repeat_num - 1,
                              constant.STRIDE_ONE, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                              constant.REPEAT_STRIDE_EIGHT)
                vector_offset = (repeat_times - 1) * front_mask
                func_name(last_mask, dest[vector_offset], src0[vector_offset], scalar_val, constant.REPEAT_TIME_ONCE,
                          constant.STRIDE_ONE, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
            with self.tik_instance.else_scope():
                func_name(front_mask, dest, src0, scalar_val, repeat_times - 1, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
                vector_offset = (repeat_times - 1) * front_mask
                func_name(last_mask, dest[vector_offset], src0[vector_offset], scalar_val, constant.REPEAT_TIME_ONCE,
                          constant.STRIDE_ONE, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)

    def _single_vector_func(self, func_name, dest, src, compute_shape):
        front_mask, last_mask, repeat_times = \
            self._get_mask_and_repeat(compute_shape, dest.dtype)
        if repeat_times == 1:
            func_name(front_mask, dest, src, constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
        elif repeat_times <= Constant.MAX_REPEAT_NUM:
            func_name(front_mask, dest, src, repeat_times - 1, constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1) * front_mask
            func_name(last_mask, dest[vector_offset], src[vector_offset], constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
        else:
            rest_repeat_num = repeat_times
            count = 0
            while rest_repeat_num > Constant.MAX_REPEAT_NUM:
                vector_offset = count * Constant.MAX_REPEAT_NUM * front_mask
                func_name(front_mask, dest[vector_offset], src[vector_offset], Constant.MAX_REPEAT_NUM,
                constant.STRIDE_ONE, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
                count = count + 1
                rest_repeat_num = rest_repeat_num - Constant.MAX_REPEAT_NUM
            if rest_repeat_num != 1:
                vector_offset = count * Constant.MAX_REPEAT_NUM * front_mask
                func_name(front_mask, dest[vector_offset], src[vector_offset], rest_repeat_num - 1, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1) * front_mask
            func_name(last_mask, dest[vector_offset], src[vector_offset], constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
