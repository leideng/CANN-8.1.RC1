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
reverse_v2_d
"""
import math
import functools

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util import util_select_op_base
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods,redefined-outer-name,too-many-statements,too-many-locals,too-many-lines
def check_supported(input_x, output_y, axis, kernel_name="reverse_v2_d"):
    """
    check whether reverse_v2_d is supported
    """
    if tbe_platform.api_check_support("tik.vcopy"):
        return False, ""

    return True, ""


class Constant:
    """
    The class for constant
    """
    MAX_BLOCK_NUM = 65536


# 'pylint: disable = unused-argument
def get_op_support_info(input_x, output_y, axis, kernel_name="reverse_v2_d"):
    """
    get_op_support_info
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    format_x = input_x.get("format").upper()
    shape_x_len = len(input_x.get("shape"))
    axis = _param_check(shape_x, dtype_x, axis, kernel_name)
    axis = omit_axis_point_to_dim_1(shape_x, axis)
    if format_x in ("ND", "NHWC", "NCHW", "NC1HWC0"):
        axis_split_matrix = []
        for i in range(0, shape_x_len):
            if i not in axis:
                split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]])]
                axis_split_matrix.append(split_0)
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def op_select_format(input_x, output_y, axis, kernel_name="reverse_v2_d"):
    """
    1. when input x's ori_format equal to "NCHW" and the 1 or 3 are not in axis.
    the Op ReverseV2D can support ND and NCHW.
    > for example:
    > input_x : Tensor of (shape=(16, 16), "ND")
    > the Op Select can process with NC1HWC0:
    > input_x : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    """
    input_ori_shape = input_x.get("ori_shape")
    input_ori_format = input_x.get("ori_format")

    axis = list(set(axis))
    axis = shape_util.axis_check(len(input_ori_shape), axis)

    is_support_5hd = True

    if input_ori_format != "NCHW":
        is_support_5hd = False

    if (input_ori_format == "NCHW" and (1 in axis)) \
            or (input_ori_format == "NHWC" and (3 in axis)):
        is_support_5hd = False

    if (input_ori_format == "NCHW") and len(input_ori_shape) > 1 \
            and (input_ori_shape[1] % 16 != 0):
        is_support_5hd = False

    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype_base = [
            "float16", "int8", "int16", "int32", "int64", "uint8",
            "uint16", "uint32", "uint64"
        ]
        dtype_5hd = [
            "float16", "int8", "int16", "int32", "int64", "uint8",
            "uint16", "uint32", "uint64"
        ]
    else:
        dtype_base = [
            "float16", "float", "int8", "int16", "int32", "int64", "uint8",
            "uint16", "uint32", "uint64"
        ]
        dtype_5hd = [
            "float16", "float", "int8", "int16", "int32", "int64", "uint8",
            "uint16", "uint32", "uint64"
        ]

    format_base = ["ND"] * len(dtype_base)

    if is_support_5hd:
        dtype_base = dtype_base + dtype_5hd
        format_base = format_base + ["NC1HWC0"] * len(dtype_5hd)

    dtype_str = ','.join(dtype_base)
    format_str = ','.join(format_base)

    input0 = util_select_op_base.gen_param(
        classify="input0", name="x", datatype=dtype_str, format=format_str)
    output0 = util_select_op_base.gen_param(
        classify="output0", name="y", datatype=dtype_str, format=format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
# 'pylint: disable=unused-argument
def reverse_v2_d(input_x, output_y, axis, kernel_name="reverse_v2_d"):
    """
    Generate reverse_v2_d operator

    Parameters
    ----------
    input_x: dict
    data of input.
    source data type, support "int8", "int16", "int32", "int64", "uint8",
    "uint16", "uint32", "uint64", "float16", "float32"
    output_y: dict
    data of output.
    axis: list
    the axis list for reverse
    kernel_name: str
    kernel name, default value is "reverse_ext2"

    Returns:
    ----------
    None
    """
    input_format = input_x.get("format")
    ori_format = input_x.get("ori_format")
    adjusted_axis = []
    if input_format == "NC1HWC0":
        for i in axis:
            adj_axis = shape_util.axis_transform_5d(i, ori_format)
            adjusted_axis.append(adj_axis)
        axis = adjusted_axis

    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")

    # 'pylint: disable=variable_type_changed
    axis = _param_check(shape_x, dtype_x, axis, kernel_name)

    axis = omit_axis_point_to_dim_1(shape_x, axis)
    if len(axis) == 0:
        move = MoveFromGm2Gm(shape_x, dtype_x, kernel_name)
        move.move()
        return move.tik_instance
    reverse = ReverseExt2(shape_x, dtype_x, axis, kernel_name)
    if reverse.reverse_bytesize < 32 or axis[-1] == len(shape_x) - 1:
        reverse.reverse_ext2_scalar_compute()
    else:
        reverse.reverse_ext2_data_move_compute()
    return reverse.tik_instance


def _param_check(shape_x, dtype_x, axis, kernel_name):
    """
    Check the input parameter

    Parameters
    ----------
    shape_x: tuple or list
        the shape of input tensor
    dtype_x: string
        the dtype of input tensor
    axis: list
        the axis list for reverse
    kernel_name: str
        kernel name, default value is "reverse_ext2"

    Returns:
    ----------
    axis: list
    """
    para_check.check_shape(shape_x, param_name="input_x")
    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16",
                  "uint32", "uint64", "float16", "float32")
    para_check.check_dtype(dtype_x.lower(), check_list, param_name="input_x")
    axis = list(set(axis))
    axis = shape_util.axis_check(len(shape_x), axis)

    return axis


# 'pylint: disable=invalid-name
def get_max_factor(n):
    """
    Get max factor of n, the max factor is less than MAX_BLOCK_NUM

    Parameters
    ----------
    n: an int number

    Returns
    -------
    the max factor: -1 if not found
    """
    if n < Constant.MAX_BLOCK_NUM:
        return n

    factors = []
    for i in range(2, Constant.MAX_BLOCK_NUM):
        if n % i == 0:
            factors = factors + [i, ]

    if len(factors) == 0:
        return -1

    return factors[-1]


# 'pylint: disable=invalid-name
def get_min_factor(n):
    """
    Get min factor of n

    Parameters
    ----------
    n: an int number

    Returns
    -------
    the min factor: -1 if not found
    """
    min_factor = -1
    for i in range(2, n + 1):
        if n % i == 0:
            min_factor = i
            break
    return min_factor


def omit_axis_point_to_dim_1(shape, axis):
    """
    Omit the axis points to shape of dim 1

    Parameters
    ----------
    shape: tuple or list
    the shape of input tensor

    axis: list
    the original axis list for reverse

    Returns
    -------
    axis: the axis list for reverse
    """
    dst_axis = [i for i in axis if shape[i] != 1]
    return dst_axis


def get_new_shape_axis(shape, axis):
    """
    Omit the ones in front of shape and update the shape and axis

    Parameters
    ----------
    shape: tuple or list
    the original shape of input tensor
    axis: list
    the axis list for reverse

    Returns
    -------
    dst_out_shape: the updated shape of input tensor
    dst_axis: the updated axis list for reverse
    """
    axis = list(set(axis))

    out_shape = shape[:axis[0]]
    out_shape_len = len(out_shape)

    if out_shape_len > 0:
        out_ele_num = functools.reduce(lambda x, y: x * y, out_shape)
        if out_ele_num == 1:
            dst_out_shape = shape[axis[0]:]
            dst_axis = [i - out_shape_len for i in axis]
        else:
            if out_ele_num > Constant.MAX_BLOCK_NUM - 1:
                max_factor = get_max_factor(out_ele_num)
                if max_factor == -1:
                    dst_out_shape = [out_ele_num, ] + list(shape[axis[0]:])
                    dst_axis = [i - out_shape_len + 1 for i in axis]
                else:
                    dst_out_shape = [max_factor, out_ele_num//max_factor] + \
                                    list(shape[axis[0]:])
                    dst_axis = [i - out_shape_len + 2 for i in axis]
            else:
                dst_out_shape = [out_ele_num, ] + list(shape[axis[0]:])
                dst_axis = [i - out_shape_len + 1 for i in axis]
    else:
        dst_out_shape = shape
        dst_axis = axis

    if dst_out_shape[0] > Constant.MAX_BLOCK_NUM - 1:
        max_factor = get_max_factor(dst_out_shape[0])
        if max_factor != -1:
            dst_out_shape = [max_factor, dst_out_shape[0]//max_factor] + \
                            list(dst_out_shape[1:])
            if 0 in dst_axis:
                tmp_axis = [i+1 for i in dst_axis[1:]]
                dst_axis = [0, 1] + tmp_axis

    return dst_out_shape, dst_axis


# 'pylint: disable=too-many-instance-attributes
class ReverseExt2:
    """
    Function: use to store reverse_ext2 schedule parameters
    Modify : 2019-11-05
    """

    def __init__(self, shape_x, dtype_x, axis, kernel_name):
        """
        init scatter_nd base parameters

        Parameters
        ----------
        shape_x: tuple or list
            the shape of input tensor
        dtype_x: string
            the dtype of input tensor
        axis: list
            the axis list for reverse
        kernel_name: str
            kernel name, default value is "reverse_ext2"

        Returns
        -------
        None
        """
        self.old_shape_x = shape_x
        shape_x, axis = get_new_shape_axis(shape_x, axis)

        self.tik_instance = tik.Tik(tik.Dprofile())
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.shape_x = list(shape_x)
        self.dtype_x = dtype_x
        self.axis = axis
        self.kernel_name = kernel_name

        block_bite_size = 32
        ub_size_bytes = (
                tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) -
                block_bite_size)
        self.dsize = get_bit_len(dtype_x) // 8
        self.data_each_block = block_bite_size // self.dsize
        self.ub_element_number = (
                ub_size_bytes // self.dsize // self.data_each_block *
                self.data_each_block)

        self.input_total_num = functools.reduce(lambda x, y: x * y, self.shape_x)

        if axis[-1] == len(shape_x) - 1:
            self.reverse_bytesize = 1
        else:
            self.reverse_bytesize = functools.reduce(lambda x, y: x * y,
                                                     shape_x[axis[-1] + 1:])

        self.replace_set_as_last_axis = axis[-1] == len(shape_x) - 1 and shape_x[-1] > 8
        self.replace_set_as_b32 = axis[-1] != len(shape_x) - 1 and \
                                  8 < self.reverse_bytesize < 32

        self.use_vgather = False
        self.available_ub = self.ub_element_number // 2

        if tbe_platform.api_check_support("tik.vgather"):
            if (self.replace_set_as_last_axis or self.replace_set_as_b32) and \
                (self.shape_x[-1] * self.dsize % 32 == 0):
                self.use_vgather = True

        self.offset_size = 0
        self.offset_last_size = 0
        self.split_w = False

        self.thread_num = 1
        # To initialize the split_axis and the split_factor.
        self.split_axis = 0
        self.split_factor = 0
        if self.use_vgather is not True:
            self.init_split_axis()
        else:
            self.init_split_axis_v200()

        if self.split_factor > 0:
            self.get_outer_inner_shape()

        if self.split_factor == 0 and self.split_axis == 0:
            self.input_ub_num = (
                    math.ceil(self.input_total_num / self.data_each_block) *
                    self.data_each_block)
            self.get_outer_inner_shape_for_small_shape()
        else:
            self.input_ub_num = (
                    self.available_ub // self.data_each_block *
                    self.data_each_block)

        self.data_x_gm = self.tik_instance.Tensor(
            self.dtype_x, self.old_shape_x, name="data_x_gm", scope=tik.scope_gm)
        self.data_y_gm = self.tik_instance.Tensor(
            self.dtype_x, self.shape_x, name="data_y_gm", scope=tik.scope_gm)

        self.data_x_ub = None
        self.sorted_ub = None
        self.sorted_align = None

        self.align_flag = False

        self.move_in_offset = 0
        self.move_out_offset = 0

        self.offset_ub = None
        self.offset_last_ub = None
        self.base_addr = None

    def reverse_ext2_scalar_compute(self):
        """
        Use the scalar method to reverse the input tensor

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.split_axis == 0 and self.split_factor == 0:
            self.data_x_ub = self.tik_instance.Tensor(
                self.dtype_x, (self.input_ub_num,),
                name="data_x_ub",
                scope=tik.scope_ubuf)
            self.sorted_ub = self.tik_instance.Tensor(
                self.dtype_x, (self.input_ub_num,),
                name="sorted_ub",
                scope=tik.scope_ubuf)

            if self.use_vgather:
                self.base_addr = self.tik_instance.Scalar("uint32", name="base_addr")
            self.offset_ub = self.tik_instance.Tensor(
                "int32", (self.offset_size, ),
                name="offset_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.offset_size) as elem_i:
                self.offset_ub[elem_i].set_as(elem_i * self.dsize)

            burst_len = math.ceil(self.input_total_num / self.data_each_block)

            self.tik_instance.data_move(self.data_x_ub, self.data_x_gm, 0, 1,
                                        burst_len, 0, 0)
            self.reverse_small_shape(self.shape_x, 0, 0, 0)
            self.tik_instance.data_move(self.data_y_gm, self.sorted_ub, 0, 1,
                                        burst_len, 0, 0)
        else:
            inner_data_num = functools.reduce(lambda x, y: x * y, self.inner_shape)
            is_multicore = inner_data_num > 32 and self.shape_x[0] < Constant.MAX_BLOCK_NUM
            if self.use_vgather is True and is_multicore is not True:
                self.base_addr = self.tik_instance.Scalar("uint32", name="base_addr")
                self.offset_ub = self.tik_instance.Tensor(
                    "int32", (self.offset_size, ),
                    name="offset_ub", scope=tik.scope_ubuf)
                if self.replace_set_as_b32:
                    with self.tik_instance.for_range(0, self.offset_size) as elem_i:
                        self.offset_ub[elem_i].set_as(elem_i * self.dsize)
                else:
                    with self.tik_instance.for_range(0, self.offset_size) as elem_i:
                        self.offset_ub[elem_i].set_as((self.offset_size - 1 -
                                                       elem_i) * self.dsize)

            self.reverse_big_shape(self.outer_shape, 0, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.data_x_gm),
            outputs=(self.data_y_gm),
            enable_l2=False)

    def move_inner(self, move_in_index, move_out_index, inner_shape):
        """
        Move the inner loop data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        move_num = functools.reduce(lambda x, y: x * y, inner_shape)

        move_loop = move_num // self.input_ub_num
        last_num = move_num - move_loop * self.input_ub_num
        if move_loop != 0:
            multi_thread = 1
            if move_loop > 1:
                multi_thread = 2
            with self.tik_instance.for_range(0, move_loop, thread_num=multi_thread) as loop_index:
                self.data_x_ub = self.tik_instance.Tensor(
                    self.dtype_x, (self.input_ub_num,),
                    name="data_x_ub",
                    scope=tik.scope_ubuf)
                burst_len = self.input_ub_num // self.data_each_block

                self.tik_instance.data_move(
                    self.data_x_ub,
                    self.data_x_gm[self.move_in_offset +
                                   move_in_index * move_num +
                                   loop_index * self.input_ub_num], 0, 1,
                    burst_len, 0, 0)

                self.tik_instance.data_move(
                    self.data_y_gm[self.move_out_offset +
                                   move_out_index * move_num +
                                   loop_index * self.input_ub_num],
                    self.data_x_ub, 0, 1, burst_len, 0, 0)

        self.data_x_ub = self.tik_instance.Tensor(
            self.dtype_x, (self.input_ub_num,),
            name="data_x_ub",
            scope=tik.scope_ubuf)

        burst_len = last_num // self.data_each_block
        if burst_len > 0:
            self.tik_instance.data_move(
                self.data_x_ub,
                self.data_x_gm[self.move_in_offset + move_in_index * move_num +
                               move_loop * self.input_ub_num], 0, 1, burst_len,
                0, 0)

            self.tik_instance.data_move(
                self.data_y_gm[self.move_out_offset +
                               move_out_index * move_num +
                               move_loop * self.input_ub_num], self.data_x_ub,
                0, 1, burst_len, 0, 0)

        tile_num = last_num % self.data_each_block
        if tile_num > 0:
            self.tik_instance.data_move(
                self.data_x_ub,
                self.data_x_gm[self.move_in_offset + move_in_index * move_num +
                               move_num - self.data_each_block], 0, 1, 1, 0, 0)

            self.tik_instance.data_move(
                self.data_y_gm[self.move_out_offset +
                               move_out_index * move_num + move_num -
                               self.data_each_block], self.data_x_ub, 0, 1, 1,
                0, 0)

    def data_move_iteration(self, out_shape, move_in_index, move_out_index,
                            loop_axis):
        """
        Traverse the outer loop of tensor

        Parameters
        ----------
        out_shape:
        the shape of outer loop
        move_in_index:
        index for moving input data from gm to ub
        move_out_index:
        index for moving output data from ub to gm
        loop_axis:
        loop index currently traversed

        Returns
        -------
        None
        """
        inner_shape = self.shape_x[self.axis[-1] + 1:]

        with self.tik_instance.for_range(0, out_shape[0]) as index:
            move_in_index, move_out_index = self.get_move_index(
                loop_axis, move_in_index, move_out_index, out_shape, index)
            if len(out_shape) > 1:
                self.data_move_iteration(out_shape[1:], move_in_index,
                                         move_out_index, loop_axis + 1)
            else:
                self.move_inner(move_in_index, move_out_index, inner_shape)

    # 'pylint: disable=too-many-locals
    def reverse_ext2_data_move_compute(self):
        """
        Reverse the input tensor by the data_move method

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        out_shape = self.shape_x[:self.axis[-1] + 1]

        first_dim = self.shape_x[0]
        inner_num = functools.reduce(lambda x, y: x * y, self.shape_x[1:])

        if first_dim <= self.aicore_num:
            with self.tik_instance.for_range(
                    0, first_dim, block_num=first_dim) as index:

                move_in_index, move_out_index = self.get_move_index(
                    0, 0, 0, out_shape, index)
                if len(out_shape) > 1:
                    self.data_move_iteration(out_shape[1:], move_in_index,
                                             move_out_index, 1)
                else:
                    inner_shape = self.shape_x[self.axis[-1] + 1:]
                    self.move_inner(move_in_index, move_out_index, inner_shape)
        else:
            loop_num1 = first_dim % self.aicore_num
            loop_step1 = math.ceil(first_dim / self.aicore_num)
            loop_step2 = first_dim // self.aicore_num
            if len(out_shape) > 1:
                out_shape1 = [loop_step1] + out_shape[1:]
                out_shape2 = [loop_step2] + out_shape[1:]
            else:
                out_shape1 = [loop_step1]
                out_shape2 = [loop_step2]

            with self.tik_instance.for_range(
                    0, self.aicore_num, block_num=self.aicore_num) as index:

                loop_index = self.tik_instance.Scalar("int32")
                loop_index.set_as(index)
                with self.tik_instance.if_scope(loop_index < loop_num1):
                    move_in_index1 = index * loop_step1
                    if 0 in self.axis:
                        move_out_index1 = (
                                self.shape_x[0] - loop_step1 - index * loop_step1)
                    else:
                        move_out_index1 = index * loop_step1
                    self.move_in_offset = move_in_index1 * inner_num
                    self.move_out_offset = move_out_index1 * inner_num
                    self.data_move_iteration(out_shape1, 0, 0, 0)
                with self.tik_instance.else_scope():
                    move_in_index2 = (
                            loop_num1 * loop_step1 +
                            (index - loop_num1) * loop_step2)
                    if 0 in self.axis:
                        move_out_index2 = (
                                self.shape_x[0] - loop_step1 * loop_num1 -
                                loop_step2 - (index - loop_num1) * loop_step2)
                    else:
                        move_out_index2 = (
                                loop_num1 * loop_step1 +
                                (index - loop_num1) * loop_step2)
                    self.move_in_offset = move_in_index2 * inner_num
                    self.move_out_offset = move_out_index2 * inner_num
                    self.data_move_iteration(out_shape2, 0, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.data_x_gm),
            outputs=(self.data_y_gm),
            enable_l2=False)

    def reverse_small_shape(self, loop_shape, sorted_ub_index, data_x_ub_index,
                            loop_axis):
        """
        Reverse a tensor slice on ub

        Parameters
        ----------
        loop_shape:
        loop shape that needs to be traversed
        sorted_ub_index:
        the offset to write data to sorted_ub
        data_x_ub_index:
        the offset to read data from data_x_ub
        loop_axis:
        the number of the axis currently traversed

        Returns
        -------
        None
        """
        ori_src_idx = data_x_ub_index
        ori_dst_idx = sorted_ub_index
        loop_time = loop_shape[0]
        reverse_num = functools.reduce(lambda x, y: x * y, loop_shape)
        is_set_loop = (self.replace_set_as_last_axis and len(loop_shape) == 1) or \
                      (self.replace_set_as_b32 and self.reverse_bytesize == reverse_num)
        if self.use_vgather and is_set_loop:
            loop_time = 1

        with self.tik_instance.for_range(0, loop_time) as index:
            if loop_axis in self.axis:
                sorted_ub_index = sorted_ub_index * loop_shape[0] + \
                                  loop_shape[0] - 1 - index
            else:
                sorted_ub_index = sorted_ub_index * loop_shape[0] + index
            data_x_ub_index = data_x_ub_index * loop_shape[0] + index
            if self.use_vgather is not True:
                if len(loop_shape) > 1:
                    self.reverse_small_shape(loop_shape[1:], sorted_ub_index,
                                             data_x_ub_index, loop_axis + 1)
                else:
                    self.sorted_ub[data_x_ub_index].set_as(
                        self.data_x_ub[sorted_ub_index])
            else:
                if self.replace_set_as_last_axis and len(loop_shape) > 1:
                    self.reverse_small_shape(loop_shape[1:], sorted_ub_index,
                                             data_x_ub_index, loop_axis + 1)
                elif self.replace_set_as_b32 and reverse_num > self.reverse_bytesize:
                    self.reverse_small_shape(loop_shape[1:], sorted_ub_index,
                                             data_x_ub_index, loop_axis + 1)
                else:
                    src_idx = ori_src_idx * loop_shape[0]
                    dst_idx = ori_dst_idx * loop_shape[0]
                    if self.replace_set_as_b32:
                        src_idx = ori_src_idx * self.reverse_bytesize
                        dst_idx = ori_dst_idx * self.reverse_bytesize
                    self.base_addr.set_as(src_idx * self.dsize)
                    self.tik_instance.vgather(
                        self.offset_size, self.sorted_ub[dst_idx],
                        self.data_x_ub, self.offset_ub, 1, 8,
                        self.base_addr, 0, "counter")

    # 'pylint: disable=too-many-arguments
    def get_move_index(self, loop_axis, move_in_index, move_out_index,
                       outer_loop_shape, index):
        """
        Get the offset of reading and writing UB

        Parameters
        ----------
        loop_axis:
        the number of the axis currently traversed
        move_in_index:
        the offset to read data from data_x_gm
        move_out_index:
        the offset to write data to data_x_ub
        outer_loop_shape:
        the outer loop shape of the current traversal
        index:
        current traversed index

        Returns
        -------
        move_in_index:
        the offset to read data from data_x_gm
        move_out_index:
        the offset to write data to data_x_ub
        """
        if loop_axis in self.axis:
            move_in_index = move_in_index * outer_loop_shape[
                0] + outer_loop_shape[0] - 1 - index
        else:
            move_in_index = move_in_index * outer_loop_shape[0] + index
        move_out_index = move_out_index * outer_loop_shape[0] + index

        return move_in_index, move_out_index

    def reverse_last_axis_small(self, inner_data_num, indices_loop_index,
                                move_out_index):
        """
        Reverse a sequence of the last axis on the UB

        Parameters
        ----------
        inner_data_num:
        number of elements that need to be transported
        indices_loop_index:
        loop index currently traversed
        move_out_index:
        the offset to write data to data_y_gm

        Returns
        -------
        None
        """
        burst_len = math.ceil(inner_data_num / self.data_each_block)
        self.tik_instance.data_move(
            self.data_x_ub, self.data_x_gm[indices_loop_index * inner_data_num],
            0, 1, burst_len, 0, 0)

        self.reverse_small_shape(self.inner_shape, 0, 0, self.split_axis)

        if self.align_flag:
            self.tik_instance.data_move(
                self.data_y_gm[move_out_index * inner_data_num], self.sorted_ub,
                0, 1, burst_len, 0, 0)
        else:
            burst_len = inner_data_num // self.data_each_block
            if burst_len != 0:
                self.tik_instance.data_move(
                    self.data_y_gm[move_out_index * inner_data_num],
                    self.sorted_ub, 0, 1, burst_len, 0, 0)

                with self.tik_instance.for_range(0, self.data_each_block) as index:
                    self.sorted_align[index].set_as(
                        self.sorted_ub[index + inner_data_num -
                                       self.data_each_block])

                self.tik_instance.data_move(
                    self.data_y_gm[move_out_index * inner_data_num +
                                   inner_data_num - self.data_each_block],
                    self.sorted_align, 0, 1, 1, 0, 0)
            else:
                self.tik_instance.data_move(
                    self.data_y_gm[move_out_index * inner_data_num],
                    self.sorted_ub, 0, 1, 1, 0, 0)

    def reverse_last_axis(self, move_in_index, move_out_index):
        """
        Reverse the last axis of input tensor

        Parameters
        ----------
        move_in_index:
        the offset to read data from data_x_gm
        move_out_index:
        the offset to write data to data_y_gm

        Returns
        -------
        None
        """
        self.data_x_ub = self.tik_instance.Tensor(
            self.dtype_x, (self.input_ub_num,),
            name="data_x_ub",
            scope=tik.scope_ubuf)
        self.sorted_ub = self.tik_instance.Tensor(
            self.dtype_x, (self.input_ub_num,),
            name="sorted_ub",
            scope=tik.scope_ubuf)
        self.sorted_align = self.tik_instance.Tensor(
            self.dtype_x, (self.data_each_block,),
            name="sorted_align",
            scope=tik.scope_ubuf)

        inner_data_num = functools.reduce(lambda x, y: x * y, self.inner_shape)

        if inner_data_num % self.data_each_block == 0:
            self.align_flag = True
        else:
            self.align_flag = False

        indices_loop_index = self.tik_instance.Scalar("int32")
        indices_loop_index.set_as(move_in_index)

        if inner_data_num > self.available_ub:
            self.reverse_last_axis_big(indices_loop_index, move_out_index)
        else:
            self.reverse_last_axis_small(inner_data_num, indices_loop_index,
                                         move_out_index)

    def reverse_big_shape(self, outer_loop_shape, move_in_index, move_out_index,
                          loop_axis):
        """
        Traverse the outer loop of tensor

        Parameters
        ----------
        outer_loop_shape:
        the shape of outer loop
        move_in_index:
        index for moving input data from gm to ub
        move_out_index:
        index for moving output data from ub to gm
        loop_axis:
        loop index currently traversed

        Returns
        -------
        None
        """
        inner_data_num = functools.reduce(lambda x, y: x * y, self.inner_shape)
        if loop_axis == 0 and inner_data_num > 32 and self.shape_x[0] < Constant.MAX_BLOCK_NUM:
            with self.tik_instance.for_range(
                    0, outer_loop_shape[0],
                    block_num=self.outer_shape[0]) as index:
                if self.use_vgather and self.replace_set_as_last_axis:
                    self.base_addr = self.tik_instance.Scalar("uint32", name="base_addr")
                    self.offset_ub = self.tik_instance.Tensor(
                        "int32", (self.offset_size, ),
                        name="offset_ub", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, self.offset_size) as elem_i:
                        self.offset_ub[elem_i].set_as((self.offset_size - 1 -
                                                       elem_i) * self.dsize)
                    if self.split_w is True:
                        self.offset_last_ub = self.tik_instance.Tensor(
                            "int32", (self.offset_last_size, ),
                            name="offset_last_ub", scope=tik.scope_ubuf)
                        with self.tik_instance.for_range(0, self.offset_last_size) as elem_i:
                            self.offset_last_ub[elem_i].set_as((self.offset_last_size -
                                                                1 - elem_i) * self.dsize)
                elif self.use_vgather and self.replace_set_as_b32:
                    self.base_addr = self.tik_instance.Scalar("uint32", name="base_addr")
                    self.offset_ub = self.tik_instance.Tensor(
                        "int32", (self.offset_size, ),
                        name="offset_ub", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, self.offset_size) as elem_i:
                        self.offset_ub[elem_i].set_as(elem_i * self.dsize)

                move_in_index, move_out_index = self.get_move_index(
                    loop_axis, move_in_index, move_out_index, outer_loop_shape,
                    index)

                if len(outer_loop_shape) > 1:
                    self.reverse_big_shape(outer_loop_shape[1:], move_in_index,
                                           move_out_index, loop_axis + 1)
                else:
                    self.reverse_last_axis(move_in_index, move_out_index)
        else:
            with self.tik_instance.for_range(0, outer_loop_shape[0],
                                             thread_num=self.thread_num) as index:
                move_in_index, move_out_index = self.get_move_index(
                    loop_axis, move_in_index, move_out_index, outer_loop_shape,
                    index)

                if len(outer_loop_shape) > 1:
                    self.reverse_big_shape(outer_loop_shape[1:], move_in_index,
                                           move_out_index, loop_axis + 1)
                else:
                    self.reverse_last_axis(move_in_index, move_out_index)

    def reverse_last_axis_big(self, indices_loop_index, move_out_index):
        """
        Reverse the last axis of UB capacity

        Parameters
        ----------
        indices_loop_index:
        index for moving input data from gm to ub
        move_out_index:
        index for moving output data from ub to gm

        Returns
        -------
        None
        """
        inner_loop = self.shape_x[-1] // self.split_factor
        gm_read_index = (
                indices_loop_index * self.shape_x[-1] +
                inner_loop * self.split_factor)
        last_num = self.shape_x[-1] % self.split_factor
        burst_len = math.ceil(last_num / self.data_each_block)
        self.tik_instance.data_move(self.data_x_ub,
                                    self.data_x_gm[gm_read_index], 0, 1,
                                    burst_len, 0, 0)
        if self.use_vgather is not True:
            with self.tik_instance.for_range(0, last_num) as index:
                self.sorted_ub[index].set_as(self.data_x_ub[last_num - 1 - index])
        else:
            self.tik_instance.vgather(self.offset_last_size, self.sorted_ub,
                                      self.data_x_ub, self.offset_last_ub,
                                      1, 8, 0, 0, "counter")
        gm_write_index = move_out_index * self.shape_x[-1]
        self.tik_instance.data_move(self.data_y_gm[gm_write_index],
                                    self.sorted_ub, 0, 1, burst_len, 0, 0)
        with self.tik_instance.for_range(0, inner_loop) as inner_index:
            gm_read_index = (
                    indices_loop_index * self.shape_x[-1] +
                    (inner_loop - 1 - inner_index) * self.split_factor)
            gm_write_index = (
                    move_out_index * self.shape_x[-1] + last_num +
                    inner_index * self.split_factor)
            burst_len = math.ceil(self.split_factor / self.data_each_block)
            self.tik_instance.data_move(self.data_x_ub,
                                        self.data_x_gm[gm_read_index], 0, 1,
                                        burst_len, 0, 0)
            if self.use_vgather is not True:
                with self.tik_instance.for_range(0, self.split_factor) as index:
                    self.sorted_ub[index].set_as(self.data_x_ub[self.split_factor
                                                                - 1 - index])
            else:
                self.tik_instance.vgather(self.offset_size, self.sorted_ub,
                                          self.data_x_ub, self.offset_ub,
                                          1, 8, 0, 0, "counter")

            self.tik_instance.data_move(self.data_y_gm[gm_write_index],
                                        self.sorted_ub, 0, 1, burst_len, 0, 0)

    def get_outer_inner_shape(self):
        """
        Get the shape of the outer loop and the inner loop

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.split_axis != len(self.shape_x) - 1:
            outer_last_axis = (
                    self.shape_x[self.split_axis] // self.split_factor)
            self.outer_shape = (
                    self.shape_x[0:self.split_axis] + [outer_last_axis])
            self.inner_shape = ([self.split_factor] +
                                self.shape_x[self.split_axis + 1:])
        else:
            if len(self.shape_x) > 1:
                self.outer_shape = self.shape_x[0:self.split_axis]
            else:
                self.outer_shape = [1]
            self.inner_shape = [self.shape_x[self.split_axis]]

    def init_split_axis(self):
        """
        Get the split axis and split factor

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        half_ub_size = self.ub_element_number // 2

        for index, _ in enumerate(self.shape_x):
            ele_cnt = functools.reduce(lambda x, y: x * y, self.shape_x[index:])
            if ele_cnt <= half_ub_size:
                self.split_axis = index - 1
                self.split_factor = half_ub_size // ele_cnt
                while self.shape_x[self.split_axis] % self.split_factor != 0:
                    self.split_factor -= 1
                break

        if self.shape_x[-1] > half_ub_size:
            self.split_axis = len(self.shape_x) - 1
            self.split_factor = (
                    half_ub_size // self.data_each_block * self.data_each_block)

        if self.split_axis < 0:
            self.split_axis = 0
            self.split_factor = 0

    def init_split_axis_v200(self):
        """
        Get the split axis and split factor of v200

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.thread_num = 2
        if self.ub_element_number > (2 + 4 // self.dsize) * self.shape_x[-1] * 2:
            if self.replace_set_as_b32:
                self.available_ub = (self.ub_element_number - 4 // self.dsize * 32) // 2 // 2
            else:
                self.available_ub = (self.ub_element_number - 4 // self.dsize * self.shape_x[-1]) // 2 // 2

            for index, _ in enumerate(self.shape_x):
                ele_cnt = functools.reduce(lambda x, y: x * y, self.shape_x[index:])
                if ele_cnt <= self.available_ub:
                    self.split_axis = index - 1
                    self.split_factor = self.available_ub // ele_cnt
                    while self.shape_x[self.split_axis] % self.split_factor != 0:
                        self.split_factor -= 1
                    break

            if self.split_axis < 0:
                self.split_axis = 0
                self.split_factor = 0

            if self.axis[-1] == len(self.shape_x) - 1:
                self.offset_size = self.shape_x[-1]
                self.offset_last_size = 0
            else:
                self.offset_size = self.reverse_bytesize
                self.offset_last_size = 0
        else:
            self.split_w = True
            self.split_axis = len(self.shape_x) - 1
            avail_size = self.ub_element_number // 2 // (2 + 2 * 4 // self.dsize)
            self.available_ub = avail_size
            self.split_factor = avail_size // self.data_each_block * self.data_each_block
            self.offset_size = self.split_factor
            self.offset_last_size = self.shape_x[-1] % self.offset_size

    def get_outer_inner_shape_for_small_shape(self):
        """
        Update split axis, split factor, outer_shape and inner_shape
        when split_axis = 0 and split_factor = 0

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        first_dim = self.shape_x[0]
        min_factor = get_min_factor(first_dim)

        self.outer_shape = [min_factor, ]
        self.inner_shape = [first_dim // min_factor, ] + self.shape_x[1:]
        self.split_factor = first_dim // min_factor
        self.split_axis = 0

        if self.use_vgather and self.inner_shape[-1] != self.shape_x[-1]:
            self.offset_size = self.inner_shape[-1]


class MoveFromGm2Gm:
    """
    Function: move the data from gm to gm via ub
    """

    def __init__(self, shape, dtype, kernel_name):
        """
        init the parameters

        Parameters
        ----------
        shape: tuple or list
            the shape of input tensor
        dtype: string
            the dtype of input tensor
        kernel_name: str
            kernel name, default value is "reverse_ext2"

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(tik.Dprofile())

        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.shape = list(shape)
        self.dtype = dtype
        self.kernel_name = kernel_name

        block_byte_size = 32
        dtype_byte_size = get_bit_len(dtype) // 8

        self.data_each_block = block_byte_size // dtype_byte_size

        ub_byte_size = (
                tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - block_byte_size)

        self.ub_element_number = (ub_byte_size // dtype_byte_size //
                                  self.data_each_block * self.data_each_block)
        self.input_total_num = functools.reduce(lambda x, y: x*y, shape)

        self.data_num_each_core = self.input_total_num // self.aicore_num
        self.last_data_num = self.input_total_num % self.aicore_num

        self.input_gm = self.tik_instance.Tensor(self.dtype,
                                                 self.shape,
                                                 name="input_gm",
                                                 scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype,
                                                  self.shape,
                                                  name="output_gm",
                                                  scope=tik.scope_gm)
        self.input_ub = None

    def move(self):
        """
        move the elements

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.data_num_each_core >= self.data_each_block:
            with self.tik_instance.for_range(
                    0, self.aicore_num, block_num=self.aicore_num) as index:
                self.input_ub = self.tik_instance.Tensor(self.dtype,
                                                         (self.ub_element_number, ),
                                                         name="input_ub",
                                                         scope=tik.scope_ubuf)

                move_index = index * self.data_num_each_core
                self.move_each_core(move_index, self.data_num_each_core)

        if self.data_num_each_core >= self.data_each_block:
            if self.last_data_num > 0:
                self.input_ub = self.tik_instance.Tensor(self.dtype,
                                                         (self.ub_element_number, ),
                                                         name="input_ub",
                                                         scope=tik.scope_ubuf)
                move_index = self.data_num_each_core * self.aicore_num
                self.move_each_core(move_index, self.last_data_num)
        else:
            self.input_ub = self.tik_instance.Tensor(self.dtype,
                                                     (self.ub_element_number, ),
                                                     name="input_ub",
                                                     scope=tik.scope_ubuf)
            move_index = 0
            self.move_little_shape(move_index, self.input_total_num)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_gm],
            outputs=[self.output_gm])

        return self.tik_instance

    def move_each_core(self, move_offset, move_num):
        """
        move the elements for each core

        Parameters
        ----------
        move_offset: start index
        move_num: number of elements to move

        Returns
        -------
        None
        """
        move_times = move_num // self.ub_element_number
        last_num = move_num % self.ub_element_number

        new_move_offset = move_offset
        with self.tik_instance.for_range(0, move_times) as index:
            new_move_offset += index * self.ub_element_number
            self.move_each_ub_loop(new_move_offset, self.ub_element_number)

        if last_num > 0:
            last_move_offset = move_offset + move_times * self.ub_element_number
            self.move_each_ub_loop(last_move_offset, last_num)

    def move_each_ub_loop(self, move_offset, move_num):
        """
        move the elements for each ub loop

        Parameters
        ----------
        move_offset: start index
        move_num: number of elements to move

        Returns
        -------
        None
        """
        burst_len = move_num // self.data_each_block
        if burst_len > 0:
            self.tik_instance.data_move(self.input_ub,
                                        self.input_gm[move_offset],
                                        0,
                                        1, burst_len,
                                        0, 0)
            self.tik_instance.data_move(self.output_gm[move_offset],
                                        self.input_ub,
                                        0,
                                        1, burst_len,
                                        0, 0)

        last_num = move_num % self.data_each_block
        if last_num > 0:
            last_offset = move_offset + move_num - self.data_each_block
            self.tik_instance.data_move(self.input_ub,
                                        self.input_gm[last_offset],
                                        0,
                                        1, 1,
                                        0, 0)
            self.tik_instance.data_move(self.output_gm[last_offset],
                                        self.input_ub,
                                        0,
                                        1, 1,
                                        0, 0)

    def move_little_shape(self, move_offset, move_num):
        """
        move small shape

        Parameters
        ----------
        move_offset: start index
        move_num: number of elements to move

        Returns
        -------
        None
        """
        burst_len = math.ceil(move_num / self.data_each_block)
        self.tik_instance.data_move(self.input_ub,
                                    self.input_gm[move_offset],
                                    0,
                                    1, burst_len,
                                    0, 0)
        self.tik_instance.data_move(self.output_gm[move_offset],
                                    self.input_ub,
                                    0,
                                    1, burst_len,
                                    0, 0)
