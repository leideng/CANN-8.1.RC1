#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
one_hot
"""
# 'pylint: disable=too-many-lines
from impl.util.platform_adapter import tbe_platform as cce
from impl.util.platform_adapter import tik
from impl.util import util_common
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from tbe.common.platform import platform_info as tbe_platform_info


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # 16k UB buffer is a reserved space
    RESERVE_SIZE = 16 * 1024
    MAX_INT32 = 2 ** 31 - 1
    SCALAR_TENSOR_SIZE = 32
    TILING_ARG_NUM = 64
    cal_num = 64
    TILING_MODE_1 = 1
    TILING_MODE_2 = 2
    TILING_MODE_3 = 3
    TILING_MODE_4 = 4
    TILING_MODE_5 = 5
    TILING_MODE_6 = 6
    TILING_MODE_7 = 7
    TILING_MODE_8 = 8
    TILING_MODE_9 = 9
    TILING_MODE_10 = 10
    TILING_MODE_11 = 11
    OFF_VALUE_TENSOR_PART = 2
    TOTAL_PART = 3
    ALIGN_16_BYTES = 16
    MOVE_TILING_BLOCK_NUM = 3
    INT64_MASK = 32


# 'pylint: disable=too-many-public-methods,too-many-instance-attributes,too-many-arguments
# 'pylint: disable=unused-argument,too-many-statements,too-many-locals,invalid-name
class OneHot:
    """
    The class of OneHot op
    """

    # 'pylint: disable =too-many-arguments,too-many-statements
    def __init__(
            self,
            x,
            depth,
            on_value,
            off_value,
            axis,
            y,
            kernel_name='one_hot'):
        """
        constructor of OneHot

        Parameters
        ----------
        x: dict
            shape and dtype of input indices tensor
        depth: dict
            the int32 scalar which judge the depth of add dim
        on_value: dict
            the value which set_as by the input tensor x
        off_value: dict
            the value which used to fill the off_value_tensor at first
        axis:int
            the attr judged which dim will be add
        y:dict
            dict with keys(range and dtype) of output
        kernel_name: str
            kernel name, default value is "one_hot"

        Returns
        -------
        None
        """
        self.dtype_x = x.get('dtype')
        self.dtype_depth = depth.get('dtype')
        self.dtype_on_value = on_value.get('dtype')
        self.dtype_off_value = off_value.get('dtype')
        self.kernel_name = kernel_name
        self.tiling_dtype = 'int64'
        block_bite_size = 32
        self.max_repeat_time = 255
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.ub_size_bytes = cce.get_soc_spec(cce.UB_SIZE)
        self.dtype_bytes_size_x = cce.get_bit_len(self.dtype_x) // 8
        self.x_each_block = block_bite_size // self.dtype_bytes_size_x
        self.dtype_bytes_size_depth = cce.get_bit_len(
            self.dtype_depth) // 8
        self.depth_each_block = block_bite_size // self.dtype_bytes_size_depth
        self.dtype_bytes_size_on_value = cce.get_bit_len(
            self.dtype_on_value) // 8
        self.on_value_each_block = block_bite_size // self.dtype_bytes_size_on_value
        self.dtype_bytes_size_off_value = cce.get_bit_len(
            self.dtype_off_value) // 8
        self.off_value_each_block = block_bite_size // self.dtype_bytes_size_off_value
        self.vector_mask_max_x = 8 * self.x_each_block
        self.dump_mask_max_off_value = 8 * self.off_value_each_block
        self.dtype_bytes_size_tiling = cce.get_bit_len(
            self.tiling_dtype) // 8
        self.tiling_each_block = block_bite_size // self.dtype_bytes_size_tiling
        self.index_scalar = self.tik_instance.Scalar(
            self.dtype_x, name='index_scalar')
        self.tiling_gm = self.tik_instance.Tensor(
            self.tiling_dtype, (Constant.TILING_ARG_NUM,), name='tiling_gm', scope=tik.scope_gm)
        self.total_core_number = cce.get_soc_spec(cce.CORE_NUM)
        self.total_part = Constant.TOTAL_PART
        self.off_value_tensor_part = Constant.OFF_VALUE_TENSOR_PART
        self.used_ub = (self.ub_size_bytes -
                        Constant.RESERVE_SIZE) // self.total_part
        self.x_len = self.used_ub // self.dtype_bytes_size_x // self.x_each_block * \
            self.x_each_block // Constant.ALIGN_16_BYTES * Constant.ALIGN_16_BYTES
        self.y_len = (self.used_ub * self.off_value_tensor_part) // self.dtype_bytes_size_on_value // \
            self.on_value_each_block * self.on_value_each_block \
            // Constant.ALIGN_16_BYTES * Constant.ALIGN_16_BYTES
        self.begin = self.tik_instance.Scalar(dtype='int64', name='begin')
        self.end = self.tik_instance.Scalar(dtype='int64', name='end')
        self.numel_shape_x = None
        self.x_gm = None
        self.dump_zero_gm = None
        self.dump_one_gm = None
        self.off_value_ub = None
        self.last_core_index = None
        self.first_dim_x = None
        self.depth_ub = None
        self.on_value_ub = None
        self.on_value_gm = None
        self.x_ub = None
        self.off_value_gm = None
        self.numel_shape_off_value_tensor = None
        self.total_part = None
        self.core_number = None
        self.y_gm = None
        self.off_value = None
        self.max_numel_vec_dup_one_loop = None
        self.last_core_numel = None
        self.on_value = None
        self.last_dim_x = None
        self.depth_gm = None
        self.is_zero_off_value = None
        self.off_value_tensor_ub = None
        self.remain_mask = None
        self.offset_off_value_tensor = None
        self.depth = None
        self.not_last_core_index = None
        self.not_last_core_numel = None
        self.remain_repeat_time = None
        self.core_num_var = None
        self.first_index = 0

    def get_tiling_args(self, tiling_ub):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from one_hot tiling

        Returns
        -------
        None
        """
        self.is_zero_off_value = self.tik_instance.Scalar(
            self.tiling_dtype, name='is_zero_off_value')
        self.core_number = self.tik_instance.Scalar(
            self.tiling_dtype, name='core_number')
        self.numel_shape_x = self.tik_instance.Scalar(
            self.tiling_dtype, name='shape_x')
        self.first_dim_x = self.tik_instance.Scalar(
            self.tiling_dtype, name='first_dim_x')
        self.last_dim_x = self.tik_instance.Scalar(
            self.tiling_dtype, name='last_dim_x')
        self.numel_shape_off_value_tensor = self.tik_instance.Scalar(
            self.tiling_dtype, name='numel_shape_off_value_tensor')
        self.not_last_core_index = self.tik_instance.Scalar(
            self.tiling_dtype, name='not_last_core_index')
        self.last_core_index = self.tik_instance.Scalar(
            self.tiling_dtype, name='last_core_index')
        self.not_last_core_numel = self.tik_instance.Scalar(
            self.tiling_dtype, name='not_last_core_numel')
        self.last_core_numel = self.tik_instance.Scalar(
            self.tiling_dtype, name='last_core_numel')
        self.core_num_var = self.tik_instance.Scalar(
            self.tiling_dtype, name='core_num_var')

        self.is_zero_off_value.set_as(tiling_ub[0])
        self.not_last_core_numel.set_as(tiling_ub[1])
        self.core_number.set_as(tiling_ub[3])
        self.numel_shape_x.set_as(tiling_ub[4])
        self.first_dim_x.set_as(tiling_ub[5])
        self.last_dim_x.set_as(tiling_ub[6])
        self.numel_shape_off_value_tensor.set_as(tiling_ub[7])
        self.last_core_numel.set_as(tiling_ub[8])
        self.not_last_core_index.set_as(tiling_ub[9])
        self.last_core_index.set_as(tiling_ub[10])
        self.core_num_var.set_as(tiling_ub[11])

    def gm_to_data(self):
        """
        GM size to the data of OneHot OP

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.x_gm = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_INT32,),
                                             name='x_gm', scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.dtype_on_value, (Constant.MAX_INT32,),
                                             name='y_gm', scope=tik.scope_gm)
        self.on_value_gm = self.tik_instance.Tensor(
            self.dtype_on_value,
            (Constant.SCALAR_TENSOR_SIZE,
             ),
            name='on_value',
            scope=tik.scope_gm)
        self.off_value_gm = self.tik_instance.Tensor(
            self.dtype_off_value,
            (Constant.SCALAR_TENSOR_SIZE,
             ),
            name='off_value',
            scope=tik.scope_gm)
        self.depth_gm = self.tik_instance.Tensor(
            self.dtype_depth, (Constant.SCALAR_TENSOR_SIZE,), name='depth', scope=tik.scope_gm)

        size = self.y_len
        dump_zero_data = [0 for _ in range(size)]
        dump_one_data = [1 for _ in range(size)]
        self.dump_zero_gm = self.tik_instance.Tensor(
            self.dtype_off_value, (size,), name='dump_zero_gm', scope=tik.scope_gm, init_value=dump_zero_data)
        self.dump_one_gm = self.tik_instance.Tensor(
            self.dtype_off_value, (size,), name='dump_one_gm', scope=tik.scope_gm, init_value=dump_one_data)

    def one_hot_compute_tiling(self):
        """
        Main process of one_hot

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.gm_to_data()
        tiling_ub = self.tik_instance.Tensor(
            self.tiling_dtype, (Constant.TILING_ARG_NUM,), name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(
            tiling_ub,
            self.tiling_gm,
            0,
            1,
            Constant.MOVE_TILING_BLOCK_NUM,
            0,
            0)
        self.get_tiling_args(tiling_ub)
        mode_of_cal_with_axis = self.tik_instance.Scalar(
            self.tiling_dtype, name='mode_of_cal_with_axis')
        mode_of_cal_with_axis.set_as(tiling_ub[2])

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as block_id:
            self.ub_to_data()
            self.data_move()
            with self.tik_instance.if_scope(tik.any((mode_of_cal_with_axis <= Constant.TILING_MODE_5),
                                                    (mode_of_cal_with_axis >= Constant.TILING_MODE_9))):
                self.begin.set_as(self.not_last_core_numel * block_id)
                with self.tik_instance.if_scope(block_id == self.core_number - 1):
                    self.end.set_as(self.last_core_numel + self.begin)
                with self.tik_instance.else_scope():
                    self.end.set_as(self.not_last_core_numel + self.begin)
            with self.tik_instance.else_scope():
                self.begin.set_as(self.not_last_core_index * block_id)
                with self.tik_instance.if_scope(block_id == self.core_number - 1):
                    self.end.set_as(self.last_core_index + self.begin)
                with self.tik_instance.else_scope():
                    self.end.set_as(self.not_last_core_index + self.begin)
            with self.tik_instance.if_scope(block_id < self.core_number):
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_1):
                    self.one_hot_last_axis_first_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_2):
                    self.one_hot_last_axis_second_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_3):
                    self.one_hot_last_axis_third_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_4):
                    self.one_hot_last_axis_fourth_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_5):
                    self.one_hot_last_axis_fifth_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_6):
                    self.one_hot_first_axis_first_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_7):
                    self.one_hot_first_axis_second_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_8):
                    self.one_hot_first_axis_third_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_9):
                    self.one_hot_middle_axis_first_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_10):
                    self.one_hot_middle_axis_second_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_11):
                    self.one_hot_middle_axis_third_mode(block_id)
        opt_config = {"out_of_bound_sync_check": True,
                      "enable_const_fold": True}
        tbe_context.get_context().add_compile_info('vars', {'core_num': self.total_core_number,
                                                            "ub_size": self.ub_size_bytes - Constant.RESERVE_SIZE})
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name, inputs=[
                self.x_gm, self.depth_gm, self.on_value_gm, self.off_value_gm], outputs=[
                self.y_gm], flowtable=[
                self.tiling_gm],
            config=opt_config)
        return self.tik_instance

    def ub_to_data(self):
        """
        UB size to the data of OneHot OP

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.x_ub = self.tik_instance.Tensor(
            self.dtype_x, (self.x_len,), name='x_ub', scope=tik.scope_ubuf)
        self.off_value_tensor_ub = self.tik_instance.Tensor(
            self.dtype_off_value,
            (self.y_len,
             ),
            name='off_value_tensor_ub',
            scope=tik.scope_ubuf)
        self.on_value_ub = self.tik_instance.Tensor(
            self.dtype_on_value,
            (Constant.SCALAR_TENSOR_SIZE,
             ),
            name='on_value_ub',
            scope=tik.scope_ubuf)
        self.off_value_ub = self.tik_instance.Tensor(
            self.dtype_off_value,
            (Constant.SCALAR_TENSOR_SIZE,
             ),
            name='off_value_ub',
            scope=tik.scope_ubuf)
        self.depth_ub = self.tik_instance.Tensor(
            self.dtype_depth, (Constant.SCALAR_TENSOR_SIZE,), name='depth_ub', scope=tik.scope_ubuf)
        self.on_value = self.tik_instance.Scalar(
            self.dtype_on_value, name='on_value_scalar')
        self.off_value = self.tik_instance.Scalar(
            self.dtype_off_value, name='off_value_scalar')
        self.depth = self.tik_instance.Scalar(
            self.dtype_depth, name='depth_scalar')

    def data_move(self):
        """
        move data of OneHot op from gm to ub

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        reinterpret_cast_dtype = "int8"
        if cce.api_check_support("tik.data_move_pad"):
            reinterpret_on_value_gm = self.on_value_gm.reinterpret_cast_to(
                reinterpret_cast_dtype)
            reinterpret_off_value_gm = self.off_value_gm.reinterpret_cast_to(
                reinterpret_cast_dtype)
            reinterpret_depth_gm = self.depth_gm.reinterpret_cast_to(
                reinterpret_cast_dtype)
            on_value_bytes = 1 * self.dtype_bytes_size_on_value
            off_value_bytes = 1 * self.dtype_bytes_size_on_value
            depth_bytes = 1 * self.dtype_bytes_size_depth
            reinterpret_on_value_ub = self.on_value_ub.reinterpret_cast_to(
                reinterpret_cast_dtype)
            reinterpret_off_value_ub = self.off_value_ub.reinterpret_cast_to(
                reinterpret_cast_dtype)
            reinterpret_depth_ub = self.depth_ub.reinterpret_cast_to(
                reinterpret_cast_dtype)
            self.tik_instance.data_move_pad(
                reinterpret_on_value_ub, reinterpret_on_value_gm, 1, on_value_bytes, 0, 0, 0, 0, None)
            self.tik_instance.data_move_pad(
                reinterpret_off_value_ub, reinterpret_off_value_gm, 1, off_value_bytes, 0, 0, 0, 0, None)
            self.tik_instance.data_move_pad(
                reinterpret_depth_ub, reinterpret_depth_gm, 1, depth_bytes, 0, 0, 0, 0, None)
        else:
            self.tik_instance.data_move(
                self.on_value_ub,
                self.on_value_gm,
                0,
                1,
                1,
                0,
                0)

            self.tik_instance.data_move(
                self.off_value_ub,
                self.off_value_gm,
                0,
                1,
                1,
                0,
                0)

            self.tik_instance.data_move(
                self.depth_ub,
                self.depth_gm,
                0,
                1,
                1,
                0,
                0)
        self.on_value.set_as(self.on_value_ub[0])
        self.off_value.set_as(self.off_value_ub[0])
        self.depth.set_as(self.depth_ub[0])

    def vec_dump_off_value_tensor_ub(self, off_value_ub_size):
        """
        the function which vec dump the space of off_value_tensor_ub

        Parameters
        ----------
        off_value_ub_size:
        the size of ub space which should be filled with off_value

        Returns
        -------
        None
        """
        self.max_numel_vec_dup_one_loop = self.max_repeat_time * self.dump_mask_max_off_value
        with self.tik_instance.for_range(0,
                                         off_value_ub_size // self.max_numel_vec_dup_one_loop) as loop:
            self.tik_instance.vec_dup(self.dump_mask_max_off_value,
                                      self.off_value_tensor_ub[loop *
                                                               self.max_numel_vec_dup_one_loop],
                                      self.off_value, self.max_repeat_time, 8)
        self.remain_repeat_time = off_value_ub_size % self.max_numel_vec_dup_one_loop // self.dump_mask_max_off_value
        self.remain_mask = off_value_ub_size % self.max_numel_vec_dup_one_loop % self.dump_mask_max_off_value
        with self.tik_instance.if_scope(self.remain_repeat_time > 0):
            self.tik_instance.vec_dup(self.dump_mask_max_off_value,
                                      self.off_value_tensor_ub[off_value_ub_size //
                                                               self.max_numel_vec_dup_one_loop *
                                                               self.max_numel_vec_dup_one_loop],
                                      self.off_value, self.remain_repeat_time, 8)
        self.offset_off_value_tensor = off_value_ub_size // self.max_numel_vec_dup_one_loop * \
            self.max_numel_vec_dup_one_loop + off_value_ub_size % \
            self.max_numel_vec_dup_one_loop \
            // self.dump_mask_max_off_value * \
            self.dump_mask_max_off_value
        with self.tik_instance.if_scope(self.remain_mask > 0):
            self.tik_instance.vec_dup(self.remain_mask,
                                      self.off_value_tensor_ub[self.offset_off_value_tensor],
                                      self.off_value,
                                      1,
                                      8)

    def move_off_value_to_ub(self, off_value_ub_size):
        with self.tik_instance.if_scope(off_value_ub_size <= 0):
            sum_ub = self.y_len
            self.tik_instance.data_move(self.off_value_tensor_ub, self.dump_zero_gm,
                                        0, 1,
                                        sum_ub // self.off_value_each_block,
                                        0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.off_value_tensor_ub, self.dump_zero_gm,
                                        0, 1, (off_value_ub_size - 1) // self.off_value_each_block + 1, 0, 0)

        with self.tik_instance.if_scope(self.off_value == 1):
            self.tik_instance.data_move(self.off_value_tensor_ub, self.dump_one_gm,
                                        0, 1, (off_value_ub_size - 1) // self.off_value_each_block + 1, 0, 0)

        return

    def align_to_32_last_block(self, begin_offset, first_index, second_index):
        """
        align the last block of data move when the axis is -1 or middle

        Parameters
        ----------
        first_index:int
        the index of the origin begin of last block
        second_index:int
        the index of the offset of the y_gm
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(first_index % self.off_value_each_block > 0):
            block_ub = self.tik_instance.Tensor(
                self.dtype_off_value,
                (self.off_value_each_block,
                 ),
                name='block_ub',
                scope=tik.scope_ubuf)
            offset_begin = first_index - self.off_value_each_block
            with self.tik_instance.for_range(offset_begin, first_index) as index:
                block_ub[index -
                         offset_begin].set_as(self.off_value_tensor_ub[index])
            self.tik_instance.data_move(
                self.y_gm[begin_offset + second_index + offset_begin],
                block_ub, 0, 1, 1, 0, 0)

    # last axis with ub enough for all
    def one_hot_last_axis_set_value(self, i, j, set_value):
        self.index_scalar.set_as(self.x_ub[i])
        with self.tik_instance.if_scope(tik.all(self.index_scalar >= self.first_index,
                                        self.index_scalar < self.depth)):
            self.index_scalar.set_as(j * self.depth + self.index_scalar)
            self.off_value_tensor_ub[self.index_scalar].set_as(set_value)

    def one_hot_move_off_value(self, begin_offset, output_part, off_value_ub_size):
        move_times = output_part // off_value_ub_size
        output_part_32 = move_times * off_value_ub_size
        output_part_tail = output_part % off_value_ub_size
        ub_repeat_times = off_value_ub_size // self.off_value_each_block
        tail_repeat_times = output_part_tail // self.off_value_each_block
        with self.tik_instance.for_range(0, move_times) as i:
            i_offset = i * off_value_ub_size
            self.tik_instance.data_move(self.y_gm[begin_offset + i_offset],
                                        self.off_value_tensor_ub, 0, 1, ub_repeat_times, 0, 0)
        with self.tik_instance.if_scope(tail_repeat_times > 0):
            self.tik_instance.data_move(self.y_gm[begin_offset + output_part_32],
                                        self.off_value_tensor_ub, 0, 1, tail_repeat_times, 0, 0)
            self.align_to_32_last_block(
                begin_offset, output_part_tail, output_part_32)

    def last_axis_move_on_value(self, part_offset, line_repeat_times, line_num, i):
        begin_offset = self.begin * self.depth
        with self.tik_instance.for_range(0, line_repeat_times) as k:
            k_offset = part_offset + k * line_num
            with self.tik_instance.for_range(0, line_num) as ele:
                self.one_hot_last_axis_set_value(
                    k_offset + ele, ele, self.on_value)
            offset = (i * self.x_len + k_offset) * self.depth
            self.tik_instance.data_move(self.y_gm[begin_offset + offset],
                                        self.off_value_tensor_ub, 0, 1, self.depth *
                                        line_num // self.off_value_each_block, 0, 0)
            self.align_to_32_last_block(
                begin_offset, self.depth * line_num, offset)
            with self.tik_instance.for_range(0, line_num) as ele:
                self.one_hot_last_axis_set_value(
                    k_offset + ele, ele, self.off_value)

    def one_hot_last_axis_move_one_depth(self, index, index_offset):
        begin_offset = self.begin * self.depth
        self.index_scalar.set_as(self.x_ub[index])
        with self.tik_instance.if_scope(tik.all(self.index_scalar >= self.first_index,
                                                self.index_scalar < self.depth)):
            i_offset = (index + index_offset * self.x_len) * self.depth
            from_end_index = self.depth - self.index_scalar
            with self.tik_instance.if_scope(from_end_index < self.off_value_each_block):
                block_index = self.off_value_each_block - from_end_index
                self.off_value_tensor_ub[block_index].set_as(self.on_value)
                self.tik_instance.data_move(self.y_gm[begin_offset + i_offset +
                                                      self.depth - self.off_value_each_block],
                                            self.off_value_tensor_ub, 0, 1, 1, 0, 0)
                self.off_value_tensor_ub[block_index].set_as(self.off_value)
            with self.tik_instance.else_scope():
                block_index = self.index_scalar % self.off_value_each_block
                self.off_value_tensor_ub[block_index].set_as(self.on_value)
                self.tik_instance.data_move(self.y_gm[begin_offset + i_offset +
                                            self.index_scalar // self.off_value_each_block *
                                            self.off_value_each_block],
                                            self.off_value_tensor_ub, 0, 1, 1, 0, 0)
                self.off_value_tensor_ub[block_index].set_as(self.off_value)

    def one_hot_first_axis_set_value(self, i, set_value):
        self.index_scalar.set_as(self.x_ub[i])
        with self.tik_instance.if_scope(tik.all(self.index_scalar < self.end,
                                                self.index_scalar >= self.begin)):
            self.index_scalar.set_as(i +
                                     self.numel_shape_x *
                                     (self.index_scalar - self.begin))
            self.off_value_tensor_ub[self.index_scalar].set_as(set_value)

    def one_hot_first_axis_move_on(self, i_offset, task_len):
        begin_offset = self.begin * self.numel_shape_x
        end_offset = self.end * self.numel_shape_x
        with self.tik_instance.for_range(0, task_len) as j:
            self.index_scalar.set_as(self.x_ub[j])
            with self.tik_instance.if_scope(tik.all(self.index_scalar < self.end,
                                            self.index_scalar >= self.begin)):
                block_index = j % self.off_value_each_block
                y_offset = (j // self.off_value_each_block) * \
                    self.off_value_each_block
                block_offset = self.index_scalar * self.numel_shape_x + i_offset + y_offset
                with self.tik_instance.if_scope(tik.all((end_offset - block_offset < self.off_value_each_block),
                                                        (block_offset - begin_offset >= self.off_value_each_block))):
                    last_block_offset = end_offset - self.off_value_each_block
                    last_block_ub_index = block_offset - last_block_offset + block_index
                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                self.y_gm[last_block_offset], 0, 1, 1, 0, 0)
                    self.off_value_tensor_ub[last_block_ub_index].set_as(
                        self.on_value)
                    self.tik_instance.data_move(self.y_gm[last_block_offset],
                                                self.off_value_tensor_ub, 0, 1, 1, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                self.y_gm[block_offset], 0, 1, 1, 0, 0)
                    self.off_value_tensor_ub[block_index].set_as(self.on_value)
                    self.tik_instance.data_move(self.y_gm[block_offset],
                                                self.off_value_tensor_ub, 0, 1, 1, 0, 0)

    def one_hot_middle_axis_move_on(self, i_base_offset, task_len):
        begin_offset = self.begin * self.depth * self.last_dim_x
        end_offset = self.end * self.depth * self.last_dim_x
        with self.tik_instance.for_range(0, task_len) as i:
            x_offset = i_base_offset + i
            core_taks_id = x_offset // self.last_dim_x
            last_dim_id = x_offset % self.last_dim_x
            self.index_scalar.set_as(self.x_ub[i])
            index_offset = self.index_scalar * self.last_dim_x
            with self.tik_instance.if_scope(tik.all(self.index_scalar >= self.first_index,
                                                    self.index_scalar < self.depth)):
                i_offset = core_taks_id * self.depth * self.last_dim_x
                block_index = last_dim_id % self.off_value_each_block
                y_offset = (last_dim_id // self.off_value_each_block) * \
                    self.off_value_each_block
                block_offset = begin_offset + index_offset + i_offset + y_offset
                with self.tik_instance.if_scope(tik.all((end_offset - block_offset < self.off_value_each_block),
                                                        (block_offset - begin_offset >= self.off_value_each_block))):
                    last_block_offset = end_offset - self.off_value_each_block
                    last_block_ub_index = block_offset - last_block_offset + block_index
                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                self.y_gm[last_block_offset], 0, 1, 1, 0, 0)
                    self.off_value_tensor_ub[last_block_ub_index].set_as(
                        self.on_value)
                    self.tik_instance.data_move(self.y_gm[last_block_offset],
                                                self.off_value_tensor_ub, 0, 1, 1, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                self.y_gm[block_offset], 0, 1, 1, 0, 0)
                    self.off_value_tensor_ub[block_index].set_as(self.on_value)
                    self.tik_instance.data_move(self.y_gm[block_offset],
                                                self.off_value_tensor_ub, 0, 1, 1, 0, 0)

    # last axis with ub enough for all
    def one_hot_last_axis_first_mode(self, id_number):
        """
        the first calculate mode when the axis is 0

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        task_len = self.end - self.begin
        x_repeat_times = (task_len - 1) // self.x_each_block + 1
        if cce.api_check_support("tik.data_move_pad"):
            reinterpret_x_gm = self.x_gm[self.begin].reinterpret_cast_to(
                "int32")
            reinterpret_x_ub = self.x_ub.reinterpret_cast_to("int32")
            data_move_bytes = task_len * self.dtype_bytes_size_x
            self.tik_instance.data_move_pad(
                reinterpret_x_ub, reinterpret_x_gm, 1, data_move_bytes, 0, 0, 0, 0, None)
        else:
            self.tik_instance.data_move(self.x_ub, self.x_gm[self.begin],
                                        0, 1, x_repeat_times, 0, 0)
        output_part = task_len * self.depth
        off_value_ub_size = output_part
        begin_offset = self.begin * self.depth
        move_repeat_times = output_part // self.off_value_each_block
        if self.dtype_off_value == "int64":
            self.move_off_value_to_ub(off_value_ub_size)
        else:
            self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        # set as the on_value
        with self.tik_instance.for_range(0, task_len) as i:
            self.one_hot_last_axis_set_value(i, i, self.on_value)
        # data move to gm
        with self.tik_instance.if_scope(move_repeat_times > 0):
            self.tik_instance.data_move(self.y_gm[begin_offset], self.off_value_tensor_ub, 0, 1,
                                        move_repeat_times, 0, 0)
            self.align_to_32_last_block(begin_offset, output_part, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.y_gm[begin_offset], self.off_value_tensor_ub, 0, 1, 1, 0, 0)

    # last axis with ub size is more than x and enough to off_value_tensor
    # some lines
    def one_hot_last_axis_second_mode(self, id_number):
        """
        the second calculate mode when the axis is 0

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        task_len = self.end - self.begin
        x_repeat_times = (task_len - 1) // self.x_each_block + 1
        if cce.api_check_support("tik.data_move_pad"):
            reinterpret_x_gm = self.x_gm[self.begin].reinterpret_cast_to(
                "int32")
            reinterpret_x_ub = self.x_ub.reinterpret_cast_to("int32")
            data_move_bytes = task_len * self.dtype_bytes_size_x
            self.tik_instance.data_move_pad(
                reinterpret_x_ub, reinterpret_x_gm, 1, data_move_bytes, 0, 0, 0, 0, None)
        else:
            self.tik_instance.data_move(self.x_ub, self.x_gm[self.begin],
                                        0, 1, x_repeat_times, 0, 0)
        off_value_ub_size = self.y_len
        line_num = off_value_ub_size // self.depth
        line_repeat_times = task_len // line_num
        task_tail = task_len % line_num
        if self.dtype_off_value == "int64":
            self.move_off_value_to_ub(off_value_ub_size)
        else:
            self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        self.last_axis_move_on_value(0, line_repeat_times, line_num, 0)
        tail_begin = line_repeat_times * line_num
        with self.tik_instance.if_scope(task_tail > 0):
            self.last_axis_move_on_value(tail_begin, 1, task_tail, 0)

    # last axis with ub size is more than x and smaller than off_value_tensor
    # one line
    def one_hot_last_axis_third_mode(self, id_number):
        """
        the third calculate mode when the axis is 0

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        task_len = self.end - self.begin
        x_repeat_times = (task_len - 1) // self.x_each_block + 1
        if cce.api_check_support("tik.data_move_pad"):
            reinterpret_x_gm = self.x_gm[self.begin].reinterpret_cast_to(
                "int32")
            reinterpret_x_ub = self.x_ub.reinterpret_cast_to("int32")
            data_move_bytes = task_len * self.dtype_bytes_size_x
            self.tik_instance.data_move_pad(
                reinterpret_x_ub, reinterpret_x_gm, 1, data_move_bytes, 0, 0, 0, 0, None)
        else:
            self.tik_instance.data_move(self.x_ub, self.x_gm[self.begin],
                                        0, 1, x_repeat_times, 0, 0)
        output_part = task_len * self.depth
        off_value_ub_size = self.y_len
        begin_offset = self.begin * self.depth
        if self.dtype_off_value == "int64":
            self.move_off_value_to_ub(off_value_ub_size)
        else:
            self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        self.one_hot_move_off_value(
            begin_offset, output_part, off_value_ub_size)
        with self.tik_instance.for_range(0, task_len) as i:
            self.one_hot_last_axis_move_one_depth(i, 0)

    # last axis with ub size is less than x and enough to off_value_tensor
    # some lines
    # 'pylint: disable =too-many-statements
    def one_hot_last_axis_fourth_mode(self, id_number):
        """
        the fourth calculate mode when the axis is 0

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        task_len = self.end - self.begin
        line_num = self.y_len // self.depth
        x_move_times = (task_len - 1) // self.x_len + 1
        task_tail = task_len - (x_move_times - 1) * self.x_len
        begin_offset = self.begin * self.depth
        off_value_ub_size = self.y_len
        if self.dtype_off_value == "int64":
            self.move_off_value_to_ub(off_value_ub_size)
        else:
            self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        offset = self.tik_instance.Scalar(
            dtype=self.dtype_depth, name='offset')
        if cce.api_check_support("tik.data_move_pad"):
            with self.tik_instance.for_range(0, x_move_times) as i:
                move_offset = (x_move_times - 1) * self.x_len
                with self.tik_instance.if_scope(i == x_move_times - 1):
                    last_time_handle = task_len - move_offset
                    handle_repeat_times = last_time_handle // line_num
                    handle_tail = last_time_handle % line_num
                    handle_tail_offset = handle_tail * self.depth
                    x_offset = task_tail // line_num * line_num
                    reinterpret_x_gm = self.x_gm[i * self.x_len +
                                                 self.begin].reinterpret_cast_to("int32")
                    reinterpret_x_ub = self.x_ub.reinterpret_cast_to("int32")
                    data_move_bytes = task_tail * self.dtype_bytes_size_x
                    self.tik_instance.data_move_pad(
                        reinterpret_x_ub, reinterpret_x_gm, 1, data_move_bytes, 0, 0, 0, 0, None)
                    with self.tik_instance.if_scope(handle_repeat_times > 0):
                        self.last_axis_move_on_value(
                            0, handle_repeat_times, line_num, i)

                    with self.tik_instance.if_scope(handle_tail > 0):
                        with self.tik_instance.if_scope(handle_tail_offset > self.off_value_each_block):
                            self.last_axis_move_on_value(
                                x_offset, 1, handle_tail, x_move_times - 1)

                        # when tail eles are not enough a block
                        with self.tik_instance.else_scope():
                            x_tail_offset = task_tail - self.off_value_each_block

                            with self.tik_instance.for_range(0, self.off_value_each_block) as ele:
                                self.one_hot_last_axis_set_value(
                                    x_tail_offset + ele, ele, self.on_value)

                            offset.set_as(
                                (move_offset + x_tail_offset) * self.depth)
                            self.tik_instance.data_move(self.y_gm[begin_offset + offset], self.off_value_tensor_ub,
                                                        0, 1, self.depth, 0, 0)

                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(self.x_ub,
                                                self.x_gm[i *
                                                          self.x_len + self.begin],
                                                0, 1, self.x_len // self.x_each_block, 0, 0)
                    line_repeat_times = self.x_len // line_num
                    line_tail = self.x_len % line_num
                    with self.tik_instance.if_scope(line_repeat_times > 0):
                        self.last_axis_move_on_value(
                            0, line_repeat_times, line_num, i)

                    with self.tik_instance.if_scope(line_tail > 0):
                        part_offset = line_repeat_times * line_num
                        self.last_axis_move_on_value(
                            part_offset, 1, line_tail, i)
        else:
            with self.tik_instance.for_range(0, x_move_times) as i:
                move_offset = (x_move_times - 1) * self.x_len
                with self.tik_instance.if_scope(i == x_move_times - 1):
                    last_time_handle = task_len - move_offset
                    handle_repeat_times = last_time_handle // line_num
                    handle_tail = last_time_handle % line_num
                    handle_tail_offset = handle_tail * self.depth
                    x_offset = task_tail // line_num * line_num
                    self.tik_instance.data_move(self.x_ub,
                                                self.x_gm[i *
                                                          self.x_len + self.begin],
                                                0, 1, (task_tail - 1) // self.x_each_block + 1, 0, 0)
                    with self.tik_instance.if_scope(handle_repeat_times > 0):
                        self.last_axis_move_on_value(
                            0, handle_repeat_times, line_num, i)

                    with self.tik_instance.if_scope(handle_tail > 0):
                        with self.tik_instance.if_scope(handle_tail_offset > self.off_value_each_block):
                            self.last_axis_move_on_value(
                                x_offset, 1, handle_tail, x_move_times - 1)

                        # when tail eles are not enough a block
                        with self.tik_instance.else_scope():
                            x_tail_offset = task_tail - self.off_value_each_block

                            with self.tik_instance.for_range(0, self.off_value_each_block) as ele:
                                self.one_hot_last_axis_set_value(
                                    x_tail_offset + ele, ele, self.on_value)

                            offset.set_as(
                                (move_offset + x_tail_offset) * self.depth)
                            self.tik_instance.data_move(self.y_gm[begin_offset + offset], self.off_value_tensor_ub,
                                                        0, 1, self.depth, 0, 0)

                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(self.x_ub,
                                                self.x_gm[i *
                                                          self.x_len + self.begin],
                                                0, 1, self.x_len // self.x_each_block, 0, 0)
                    line_repeat_times = self.x_len // line_num
                    line_tail = self.x_len % line_num
                    with self.tik_instance.if_scope(line_repeat_times > 0):
                        self.last_axis_move_on_value(
                            0, line_repeat_times, line_num, i)

                    with self.tik_instance.if_scope(line_tail > 0):
                        part_offset = line_repeat_times * line_num
                        self.last_axis_move_on_value(
                            part_offset, 1, line_tail, i)

    # last axis with ub size is less than x smaller than off_value_tensor one
    # line
    def one_hot_last_axis_fifth_mode(self, id_number):
        """
        the fifth calculate mode when the axis is 0

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        task_len = self.end - self.begin
        x_move_times = (task_len - 1) // self.x_len + 1
        output_part = task_len * self.depth
        off_value_ub_size = self.y_len
        task_tail = task_len % self.x_len
        begin_offset = self.begin * self.depth
        if self.dtype_off_value == "int64":
            self.move_off_value_to_ub(off_value_ub_size)
        else:
            self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        # move off value to gm
        self.one_hot_move_off_value(
            begin_offset, output_part, off_value_ub_size)

        with self.tik_instance.for_range(0, x_move_times) as i:
            with self.tik_instance.if_scope(i == x_move_times - 1):
                if cce.api_check_support("tik.data_move_pad"):
                    reinterpret_x_gm = self.x_gm[i * self.x_len +
                                                 self.begin].reinterpret_cast_to("int32")
                    reinterpret_x_ub = self.x_ub.reinterpret_cast_to("int32")
                    data_move_bytes = task_tail * self.dtype_bytes_size_x
                    with self.tik_instance.for_range(0, task_tail) as index:
                        self.tik_instance.data_move_pad(
                            reinterpret_x_ub, reinterpret_x_gm, 1, data_move_bytes, 0, 0, 0, 0, None)
                        self.one_hot_last_axis_move_one_depth(
                            index, i * self.x_len)
                else:
                    with self.tik_instance.for_range(0, task_tail) as index:
                        self.tik_instance.data_move(self.x_ub,
                                                    self.x_gm[i *
                                                              self.x_len + self.begin],
                                                    0, 1, (task_tail -
                                                           1) // self.x_each_block + 1,
                                                    0, 0)
                        self.one_hot_last_axis_move_one_depth(
                            index, i * self.x_len)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.x_ub,
                                            self.x_gm[i *
                                                      self.x_len + self.begin],
                                            0, 1, self.x_len // self.x_each_block,
                                            0, 0)
                with self.tik_instance.for_range(0, self.x_len) as index:
                    self.one_hot_last_axis_move_one_depth(
                        index, i * self.x_len)

    # first axis with ub enough for all
    def one_hot_first_axis_first_mode(self, id_number):
        """
        the first calculate mode when the axis is -1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        task_len = self.end - self.begin
        move_x_repeat_times = (self.numel_shape_x - 1) // self.x_each_block + 1
        move_output_times = task_len * self.numel_shape_x // self.off_value_each_block
        begin_offset = self.begin * self.numel_shape_x
        output_part = self.numel_shape_x * task_len
        off_value_ub_size = output_part
        self.tik_instance.data_move(
            self.x_ub, self.x_gm, 0, 1, move_x_repeat_times, 0, 0)
        if self.dtype_off_value == "int64":
            self.move_off_value_to_ub(off_value_ub_size)
        else:
            self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        with self.tik_instance.for_range(0, self.numel_shape_x) as i:
            self.one_hot_first_axis_set_value(i, self.on_value)
        with self.tik_instance.if_scope(move_output_times > 0):
            self.tik_instance.data_move(self.y_gm[begin_offset],
                                        self.off_value_tensor_ub, 0, 1, move_output_times,
                                        0, 0)
            self.align_to_32_last_block(begin_offset, output_part, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.y_gm[begin_offset],
                                        self.off_value_tensor_ub, 0, 1, 1, 0, 0)

    # first axis with ub size is more than x and enough to off_value_tensor
    # some lines
    def one_hot_first_axis_second_mode(self, id_number):
        """
        the second calculate mode when the axis is -1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        task_len = self.end - self.begin
        output_part = self.numel_shape_x * task_len
        off_value_ub_size = self.y_len
        move_x_times = (self.numel_shape_x - 1) // self.x_each_block + 1
        begin_offset = self.begin * self.numel_shape_x
        self.tik_instance.data_move(
            self.x_ub, self.x_gm, 0, 1, move_x_times, 0, 0)

        if self.dtype_off_value == "int64":
            self.move_off_value_to_ub(off_value_ub_size)
        else:
            self.vec_dump_off_value_tensor_ub(off_value_ub_size)

        self.one_hot_move_off_value(
            begin_offset, output_part, off_value_ub_size)
        self.one_hot_first_axis_move_on(0, self.numel_shape_x)

    # first axis with ub size is less than x and enough to off_value_tensor
    # some lines
    def one_hot_first_axis_third_mode(self, id_number):
        """
        the third calculate mode when the axis is -1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        task_len = self.end - self.begin
        output_part = task_len * self.numel_shape_x
        off_value_ub_size = self.y_len
        begin_offset = self.begin * self.numel_shape_x
        x_move_times = (self.numel_shape_x - 1) // self.x_len + 1
        x_tail = self.numel_shape_x % self.x_len
        if self.dtype_off_value == "int64":
            self.move_off_value_to_ub(off_value_ub_size)
        else:
            self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        self.one_hot_move_off_value(
            begin_offset, output_part, off_value_ub_size)
        with self.tik_instance.for_range(0, x_move_times) as i:
            i_offset = i * self.x_len
            with self.tik_instance.if_scope(tik.any((i < x_move_times - 1), (x_tail == 0))):
                self.tik_instance.data_move(self.x_ub,
                                            self.x_gm[i_offset],
                                            0, 1,
                                            self.x_len // self.x_each_block,
                                            0, 0)
                self.one_hot_first_axis_move_on(i_offset, self.x_len)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.x_ub,
                                            self.x_gm[i_offset],
                                            0, 1,
                                            (x_tail - 1) // self.x_each_block + 1,
                                            0, 0)
                self.one_hot_first_axis_move_on(i_offset, x_tail)

    # middle axis with ub enough for all
    def one_hot_middle_axis_first_mode(self, id_number):
        """
        the first calculate mode when the axis is 0 < axis < len(x_shape) - 1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        core_task_len = self.end - self.begin
        task_len = core_task_len * self.last_dim_x
        x_repeat_times = (task_len - 1) // self.x_each_block + 1

        self.tik_instance.data_move(self.x_ub,
                                    self.x_gm[self.begin * self.last_dim_x],
                                    0, 1, x_repeat_times, 0, 0)
        output_part = task_len * self.depth
        off_value_ub_size = output_part
        begin_offset = self.begin * self.depth * self.last_dim_x
        move_repeat_times = output_part // self.off_value_each_block
        if self.dtype_off_value == "int64":
            self.move_off_value_to_ub(off_value_ub_size)
        else:
            self.vec_dump_off_value_tensor_ub(off_value_ub_size)

        with self.tik_instance.for_range(0, core_task_len) as i:
            with self.tik_instance.for_range(0, self.last_dim_x) as j:
                x_id = i * self.last_dim_x + j
                base_offset = i * self.last_dim_x * self.depth + j
                self.index_scalar.set_as(self.x_ub[x_id])
                with self.tik_instance.if_scope(tik.all(self.index_scalar >= self.first_index,
                                                        self.index_scalar < self.depth)):
                    index_offset = self.last_dim_x * self.index_scalar
                    self.index_scalar.set_as(base_offset + index_offset)
                    self.off_value_tensor_ub[self.index_scalar].set_as(
                        self.on_value)

        with self.tik_instance.if_scope(move_repeat_times > 0):
            self.tik_instance.data_move(self.y_gm[begin_offset], self.off_value_tensor_ub, 0, 1,
                                        move_repeat_times, 0, 0)
            self.align_to_32_last_block(begin_offset, output_part, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.y_gm[begin_offset],
                                        self.off_value_tensor_ub, 0, 1, 1, 0, 0)

    # middle axis with ub size is more than x and enough to off_value_tensor
    # some lines
    def one_hot_middle_axis_second_mode(self, id_number):
        """
        the second calculate mode when the axis is 0 < axis < len(x_shape) - 1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        core_task_len = self.end - self.begin
        task_len = core_task_len * self.last_dim_x

        x_repeat_times = (task_len - 1) // self.x_each_block + 1
        x_begin_offset = self.begin * self.last_dim_x
        self.tik_instance.data_move(self.x_ub, self.x_gm[x_begin_offset],
                                    0, 1, x_repeat_times,
                                    0, 0)
        output_part = task_len * self.depth
        off_value_ub_size = self.y_len
        begin_offset = self.begin * self.depth * self.last_dim_x
        if self.dtype_off_value == "int64":
            self.move_off_value_to_ub(off_value_ub_size)
        else:
            self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        self.one_hot_move_off_value(
            begin_offset, output_part, off_value_ub_size)
        self.one_hot_middle_axis_move_on(0, task_len)

    # middle axis with ub size is less than x and enough to off_value_tensor
    def one_hot_middle_axis_third_mode(self, id_number):
        """
        the third calculate mode when the axis is 0 < axis < len(x_shape) - 1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        core_task_len = self.end - self.begin
        task_len = core_task_len * self.last_dim_x
        output_part = task_len * self.depth
        off_value_ub_size = self.y_len
        x_begin_offset = self.begin * self.last_dim_x
        begin_offset = x_begin_offset * self.depth
        if self.dtype_off_value == "int64":
            self.move_off_value_to_ub(off_value_ub_size)
        else:
            self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        self.one_hot_move_off_value(
            begin_offset, output_part, off_value_ub_size)
        x_move_times = (task_len - 1) // self.x_len + 1
        x_tail = task_len % self.x_len
        with self.tik_instance.for_range(0, x_move_times) as i:
            i_offset = i * self.x_len
            with self.tik_instance.if_scope(tik.any((i < x_move_times - 1), (x_tail == 0))):
                self.tik_instance.data_move(self.x_ub,
                                            self.x_gm[x_begin_offset +
                                                      i_offset],
                                            0, 1,
                                            self.x_len // self.x_each_block,
                                            0, 0)
                self.one_hot_middle_axis_move_on(i_offset, self.x_len)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.x_ub,
                                            self.x_gm[x_begin_offset +
                                                      i_offset],
                                            0, 1,
                                            (x_tail - 1) // self.x_each_block + 1,
                                            0, 0)
                self.one_hot_middle_axis_move_on(i_offset, x_tail)
        return


# 'pylint: disable=too-many-return-statements
def check_supported(x, depth, on_value, off_value, y, axis,
                    kernel_name="one_hot"):
    """
    dynamic is support, static and shape[0] is 2048, and axis is 0,
    onehot is support, else static not support, onehotd is support.
    x : dict
        dict with keys(range and dtype) of indices tensor
    depth: dict
        dict whith the scalar depth
    on_value : dict
        dict with the scalar on_value
    off_value: dict
        dict with the scalar off_value
    axis: int
        the axis which judge the mode of calculate
    y : dict
        dict with keys(range and dtype) of output
    kernel_name : str
        kernel name, default value is "one_hot"

    Returns
    -------
    True or False
    """
    reason = "Running dynamic op for all shapes"
    return True, reason


def _check_param(x, depth, on_value, off_value):
    """
    check parameters, if one is invalid, then raise error

    Parameters
    ----------
    x : dict
        dict with keys(range and dtype) of indices tensor
    depth: dict
        dict whith the scalar depth
    on_value : dict
        dict with the scalar on_value
    off_value: dict
        dict with the scalar off_value
    axis: int
        the axis which judge the mode of calculate

    Returns
    -------
    None
    """
    x_dtype = x.get("dtype").lower()
    depth_dtype = depth.get("dtype").lower()
    on_value_dtype = on_value.get("dtype").lower()
    off_value_dtype = off_value.get("dtype").lower()
    para_check.check_dtype(x_dtype, ["int32", "int64"])
    para_check.check_dtype(depth_dtype, ["int32", "int64"])
    para_check.check_dtype(
        on_value_dtype, ["int64", "int32", "float32", "float16"])
    para_check.check_dtype(
        off_value_dtype, ["int64", "int32", "float32", "float16"])


# the register of OneHot op
# 'pylint: disable=unused-argument,too-many-arguments
@register_operator('OneHot')
def one_hot(x,
            depth,
            on_value,
            off_value,
            y,
            axis,
            kernel_name='one_hot'):
    """
    algorithm:one_hot
    Operation for one_hot

    Parameters
    ----------
    x : dict
        dict with keys(range and dtype) of indices tensor
    depth: dict
        dict whith the scalar depth
    on_value : dict
        dict with the scalar on_value
    off_value: dict
        dict with the scalar off_value
    axis: int
        the axis which judge the mode of calculate
    y : dict
        dict with keys(range and dtype) of output
    kernel_name : str
        kernel name, default value is "resize_bilinear_v2_grad"

    Returns
    -------
    None
    """
    _check_param(x, depth, on_value, off_value)
    one_hot_instance = OneHot(
        x, depth, on_value, off_value, axis, y, kernel_name)
    tik_instance = one_hot_instance.one_hot_compute_tiling()
    return tik_instance
