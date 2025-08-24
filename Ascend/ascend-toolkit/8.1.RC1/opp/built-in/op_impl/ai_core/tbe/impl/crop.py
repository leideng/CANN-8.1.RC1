#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
crop
"""
from collections import namedtuple
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl import constant_util as constant
from impl import common_util
from impl.util import util_select_op_base


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    Constant
    """
    # reserve size for ub
    RESERVE_SIZE = 16 * 1024


# 'pylint: disable=invalid-name,too-many-arguments,unused-argument,too-many-instance-attributes
# 'pylint: disable=too-many-locals,too-many-statements,too-many-lines
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.REQUIRED_ATTR_LIST_INT, para_check.KERNEL_NAME)
def crop(x, size, y, axis=2, offsets=(0), kernel_name="crop"):
    """
    the main function of crop

    Parameters
    ----------
    x1: dict,shape and datatype,datatype supports int8,uint8,int16,uint16,
        int32,uint32,int64,uint64,float16,float32
    x2: dict,shape and datatype,datatype supportsint8,uint8,int16,uint16,
        int32,uint32,int64,uint64,float16,float32
    y: dict,shape and datatype,datatype supports int8,uint8,int16,uint16,
        int32,uint32,int64,uint64,float16,float32
    axis: crop start with axis
    offsets: crop start offset of each axis
    kernel_name: cce kernel name, default value is "crop"

    Returns
    -------
    tik_instance: tik_instance
    """
    input_dict = {
        "x1": x,
        "x2": y,
        "y": y,
        "axis": axis,
        "offset": offsets,
        "kernel_name": kernel_name
    }
    check_and_adjust_offset(input_dict)
    crop_process = Crop(input_dict)
    crop_process.compute_crop()
    crop_process.instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(crop_process.x1_gm, crop_process.x2_gm),
                                   outputs=(crop_process.y_gm,), enable_l2=False)

    return crop_process.instance


# 'pylint: disable=too-many-return-values
class Crop:
    """
    Function: store Crop parameters  and compute crop
    Modify : 2019-11-06
    """

    def __init__(self, input_dict):
        """
      init the Crop parameters

      Parameters
      ----------
        input_dict: input_dict is a dict, the keys as follow:
                x1: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                x2: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                y: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                axis: crop start with axis
                offsets: crop start offset of each axis
                kernel_name: cce kernel name, default value is "crop"
      Returns
      -------
      None
      """
        self.instance = tik.Tik(tik.Dprofile())
        self.dtype = input_dict.get("x1").get("dtype").lower()
        self.dsize = common_util.get_data_size(self.dtype)
        total_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        ub_size = (total_size - Constant.RESERVE_SIZE) // (2 * self.dsize)
        burst_len = constant.BLOCK_SIZE // self.dsize
        ub_size = ((ub_size + burst_len - 1) // burst_len) * burst_len
        self.one_max_size = ub_size
        x1_len = get_shape_total_number(input_dict.get("x1").get("shape"))
        x1_len = ((x1_len + burst_len - 1) // burst_len) * burst_len
        mod = input_dict.get("y").get("shape")[-1] % burst_len
        if mod != 0:
            x1_len = x1_len + burst_len
        self.x1_gm = self.instance.Tensor(self.dtype, (x1_len,), name="x1_gm",
                                          scope=tik.scope_gm)
        self.x2_gm = self.instance.Tensor(self.dtype, (32,), name="x2_gm",
                                          scope=tik.scope_gm)
        y_len = get_shape_total_number(input_dict.get("y").get("shape"))
        y_len = ((y_len + burst_len - 1) // burst_len) * burst_len
        if mod != 0:
            y_len = y_len + burst_len
        self.y_gm = self.instance.Tensor(self.dtype, (y_len,), name="y_gm",
                                         scope=tik.scope_gm)
        self.input_dict = input_dict

    def get_element_num(self):
        """
        get the block size
        """
        shape_y = self.input_dict.get("y").get("shape")
        shape_len = len(shape_y)
        element_num = shape_y[-1]
        if "format" in self.input_dict.get("x1"):
            x1_format = self.input_dict.get("x1").get("format")
            if x1_format == "NC1HWC0":
                element_num = shape_y[-1] * shape_y[-2]
                shape_len = len(shape_y) - 1
        return element_num, shape_len

    def get_blockdim_and_loop_cycle(self):
        """
        get block dim and loop cycle
        """
        limit_size_of_each_block = self.get_limit_size_of_each_block()
        block_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        shape_y = self.input_dict.get("y").get("shape")
        loop = get_shape_total_number(shape_y) // limit_size_of_each_block
        element_num, _ = self.get_element_num()
        all_num = get_shape_total_number(shape_y) // element_num
        if loop <= block_num:
            block_num = loop
        inner_loop = all_num // block_num
        inner_loop_mod = all_num % block_num
        return block_num, limit_size_of_each_block, inner_loop, inner_loop_mod

    def get_limit_size_of_each_block(self):
        """
        get limit size of each block
        """
        shape_y = self.input_dict.get("y").get("shape")
        if "format" in self.input_dict.get("x1"):
            x1_format = self.input_dict.get("x1").get("format")
            if x1_format == "NC1HWC0":
                return shape_y[-1] * shape_y[-2]
        each_size = 1
        each_block_num = constant.BLOCK_SIZE // self.dsize
        each_block_align = (each_block_num + shape_y[-1] - 1) // shape_y[-1] \
                           * shape_y[-1]
        for j in range(len(shape_y) - 1, -1, -1):
            each_size = each_size * shape_y[j]
            if each_size * self.dsize >= constant.BLOCK_SIZE:
                each_size = each_block_align
                break
        return each_size

    def get_thread_num(self, block_num, each_loop, element_num):
        """
        get thread num
        """
        if element_num * self.dsize < constant.BLOCK_SIZE:
            if block_num == 1 and each_loop > 1:
                thread_num = 2
            else:
                thread_num = 1
        else:
            thread_num = 1
            if each_loop > 1:
                thread_num = 2
        return thread_num

    def prepare_src_pattern(self, index, num=1):
        """
        prepare src1_pattern tensor for vreduce instruction
        """
        shape_x = self.input_dict.get("x1").get("shape")[index:]
        shape_y = self.input_dict.get("y").get("shape")[index:]
        offset_in = self.input_dict.get("offset")[index:]
        x_size = get_shape_total_number(shape_x)
        if self.dtype in ["float16", "uint16", "int16"]:
            dtype = "uint16"
            align_size = 16
        else:
            dtype = "uint32"
            align_size = 32
        size_align = (x_size * num + align_size - 1) // align_size * align_size
        offset_list_tuple = namedtuple('OffsetList', "size, shape_x, shape_y, offset_in, element_size, num")
        pattern_list = _get_input_offset_list(
            offset_list_tuple(size_align, shape_x, shape_y, offset_in, shape_y[-1], num)
            )
        src1_pattern = self.instance.Tensor(dtype, (size_align // align_size,),
                                            name="src1_pattern", scope=tbe_platform.scope_ubuf)
        scalar = self.instance.Scalar(dtype, name="pattern_scalar")
        for i in range(len(pattern_list)):
            if (i + 1) % align_size == 0:
                s = ''.join([str(x) for x in pattern_list[i + 1 - align_size:i + 1]])
                scalar.set_as(int(s[::-1], 2))
                src1_pattern[i // align_size].set_as(scalar)
        return src1_pattern

    def get_pattern_index_num(self):
        """
        determine the start axis of the src1_pattern tensor based on the output shape size
        """
        shape_x = self.input_dict.get("x1").get("shape")
        shape_y = self.input_dict.get("y").get("shape")
        total_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        block_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        ub_size = total_size // 2 // (self.dsize * 3)
        align = constant.BLOCK_SIZE // self.dsize
        index_32b = _get_index_num(shape_y, len(shape_y) - 1, -1, -1, align)
        block_index = _get_index_num(shape_y, 0, index_32b + 1, 1, block_num)
        pattern_index = len(shape_x) - 1
        for i in range(block_index + 1, len(shape_x)):
            buf_size_needed = get_shape_total_number(shape_x[i:])
            is_buffer_large_enough = ub_size // buf_size_needed
            if is_buffer_large_enough > 0:
                pattern_index = i
                break
        if block_index >= len(shape_x) - 1:
            return -1
        if block_index >= pattern_index:
            return block_index + 1

        return pattern_index

    def reduce_compute(self, input_dict):
        """
        use the vreduce instruction to move data circularly
        """
        compute_loop = input_dict.get("compute_loop")
        rsvd_index = input_dict.get("rsvd_index")
        in_element = input_dict.get("in_element")
        dst_ub = input_dict.get("dst_ub")
        src_ub = input_dict.get("src_ub")
        src1_pattern_ub = input_dict.get("src1_pattern_ub")
        rsvd_scalar = self.instance.Scalar("uint32", name="reduce_scalar")
        with self.instance.for_range(0, compute_loop) as n_i:
            self.instance.vreduce(in_element, dst_ub[rsvd_index], src_ub[n_i * in_element], src1_pattern_ub,
                                  constant.REPEAT_TIME_ONCE, constant.BLOCK_STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                                  constant.STRIDE_ONE, 0, rsvd_scalar, "counter")
            rsvd_index.set_as(rsvd_index + rsvd_scalar)

    def reduce_loop_compute(self, loop_num, input_dict):
        """
        determine the number of loops processed by each UB based on the input shape size
        """
        shape_x = self.input_dict.get("x1").get("shape")
        shape_y = self.input_dict.get("y").get("shape")
        offset_in = self.input_dict.get("offset")
        rsvd_index = input_dict.get("rsvd_index")
        dst_ub = input_dict.get("dst_ub")
        src_ub = input_dict.get("src_ub")
        src1_pattern_ub = input_dict.get("src1_pattern_ub")
        ub_size = input_dict.get("ub_size")
        in_element = input_dict.get("in_element")
        pattern_index = input_dict.get("pattern_index")
        loop_start = input_dict.get("loop_start")

        ub_in_num = ub_size // in_element
        loop = loop_num // ub_in_num
        tail = loop_num % ub_in_num

        reduce_dict = {
            "rsvd_index": rsvd_index,
            "in_element": in_element,
            "dst_ub": dst_ub,
            "src_ub": src_ub,
            "src1_pattern_ub": src1_pattern_ub
        }
        x1_offset = self.instance.Scalar("uint32", name="x1_offset")
        loop_offset_tuple = namedtuple('InputOffset', "shape_x shape_y offset_in loop_i pattern_index x_offset")
        with self.instance.if_scope(loop > 0):
            with self.instance.for_range(0, loop) as l_i:
                _get_input_offset(loop_offset_tuple(shape_x, shape_y, offset_in, loop_start + l_i * ub_in_num,
                                  pattern_index, x1_offset))
                n_burst = (in_element * ub_in_num * self.dsize + constant.BLOCK_SIZE - 1) // constant.BLOCK_SIZE
                self.instance.data_move(src_ub, self.x1_gm[x1_offset * in_element],
                                        constant.SID, constant.DEFAULT_NBURST, n_burst,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                reduce_dict["compute_loop"] = ub_in_num
                self.reduce_compute(reduce_dict)
        with self.instance.if_scope(tail > 0):
            _get_input_offset(loop_offset_tuple(shape_x, shape_y, offset_in, loop_start + loop * ub_in_num,
                              pattern_index, x1_offset))
            n_burst = common_util.get_datamove_nburst(self.instance, in_element * tail * self.dsize)
            self.instance.data_move(src_ub, self.x1_gm[x1_offset * in_element], constant.SID, constant.DEFAULT_NBURST,
                                    n_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            reduce_dict["compute_loop"] = tail
            self.reduce_compute(reduce_dict)

    def reduce_move(self, input_dict):
        """
        determine the number of loops processed by each UB based on the output shape size
        """
        shape_y = self.input_dict.get("y").get("shape")
        ub_size = input_dict.get("ub_size")
        in_element = input_dict.get("in_element")
        out_element = input_dict.get("out_element")
        pattern_index = input_dict.get("pattern_index")
        reduce_loop = input_dict.get("reduce_loop")
        loop_time = input_dict.get("loop_time")
        src1_pattern_ub = input_dict.get("src1_pattern_ub")
        move_flag = input_dict.get("move_flag")
        out_offset = loop_time * out_element
        src_ub = self.instance.Tensor(self.dtype, (ub_size,), name="src_ub", scope=tbe_platform.scope_ubuf)
        dst_ub = self.instance.Tensor(self.dtype, (ub_size,), name="dst_ub", scope=tbe_platform.scope_ubuf)
        n_burst_out = (out_element * reduce_loop * self.dsize + constant.BLOCK_SIZE - 1) // constant.BLOCK_SIZE
        rsvd_index = self.instance.Scalar("uint32", "reduce_index")
        rsvd_index.set_as(0)
        num = shape_y[pattern_index - 1]

        loop_dict = {
            "dst_ub": dst_ub,
            "src_ub": src_ub,
            "src1_pattern_ub": src1_pattern_ub,
            "ub_size": ub_size,
            "in_element": in_element,
            "pattern_index": pattern_index,
            "rsvd_index": rsvd_index
        }
        with self.instance.if_scope((loop_time % num + reduce_loop) <= num):
            loop_dict["loop_start"] = loop_time
            self.reduce_loop_compute(reduce_loop, loop_dict)
        with self.instance.else_scope():
            first = num - loop_time % num
            loop = (reduce_loop - first) // num
            tail = (reduce_loop - first) % num
            loop_dict["loop_start"] = loop_time
            self.reduce_loop_compute(first, loop_dict)
            with self.instance.if_scope(loop > 0):
                with self.instance.for_range(0, loop) as l_i:
                    loop_dict["loop_start"] = loop_time + first + l_i * num
                    self.reduce_loop_compute(num, loop_dict)
            with self.instance.if_scope(tail > 0):
                loop_dict["loop_start"] = loop_time + first + loop * num
                self.reduce_loop_compute(tail, loop_dict)

        with self.instance.if_scope(move_flag):
            each_burst_num = constant.BLOCK_SIZE // self.dsize
            n_burst = ((out_element * reduce_loop * self.dsize) // constant.BLOCK_SIZE)
            self.instance.data_move(self.y_gm[out_offset], dst_ub, constant.SID,
                                    constant.DEFAULT_NBURST,
                                    n_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            offset = out_element * reduce_loop - each_burst_num
            scalar = self.instance.Scalar(dst_ub.dtype)
            with self.instance.for_range(0, each_burst_num) as time:
                scalar.set_as(dst_ub[offset + time])
                dst_ub[time].set_as(scalar)
            self.instance.data_move(self.y_gm[out_offset + offset], dst_ub,
                                    constant.SID, constant.DEFAULT_NBURST,
                                    constant.DEFAULT_BURST_LEN,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        with self.instance.else_scope():
            self.instance.data_move(self.y_gm[out_offset], dst_ub, constant.SID, constant.DEFAULT_NBURST, n_burst_out,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def reduce_move_consequent(self, input_dict):
        """
        use reduce to continuously process a piece of data
        """
        shape_x = self.input_dict.get("x1").get("shape")
        shape_y = self.input_dict.get("y").get("shape")
        offset_in = self.input_dict.get("offset")
        x1_offset = self.instance.Scalar("uint32", name="x1_offset")
        src1_pattern_ub = input_dict.get("src1_pattern_ub")
        ub_size = input_dict.get("ub_size")
        offset_index = input_dict.get("offset_index")
        in_element = input_dict.get("in_element")
        out_element = input_dict.get("out_element")
        pattern_index = input_dict.get("pattern_index")
        move_flag = input_dict.get("move_flag")
        num = input_dict.get("num")

        move_offset_tuple = namedtuple('MoveOffset', "shape_x shape_y offset_in loop_i pattern_index x_offset")
        _get_input_offset(move_offset_tuple(shape_x, shape_y, offset_in, offset_index, pattern_index, x1_offset))
        src_ub = self.instance.Tensor(self.dtype, (ub_size,), name="src_ub", scope=tik.scope_ubuf)
        dst_ub = self.instance.Tensor(self.dtype, (ub_size,), name="dst_ub", scope=tik.scope_ubuf)
        n_burst = (in_element * num * self.dsize + constant.BLOCK_SIZE - 1) // constant.BLOCK_SIZE
        n_burst_out = (out_element * num * self.dsize + constant.BLOCK_SIZE - 1) // constant.BLOCK_SIZE

        self.instance.data_move(src_ub, self.x1_gm[x1_offset * in_element],
                                constant.SID, constant.DEFAULT_NBURST, n_burst,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)

        self.instance.vreduce(in_element * num, dst_ub, src_ub, src1_pattern_ub,
                              constant.REPEAT_TIME_ONCE, constant.BLOCK_STRIDE_ONE,
                              constant.REPEAT_STRIDE_EIGHT,
                              constant.STRIDE_ONE, 0, None, "counter")

        with self.instance.if_scope(move_flag):
            each_burst_num = constant.BLOCK_SIZE // self.dsize
            n_burst = ((out_element * num * self.dsize) // constant.BLOCK_SIZE)
            self.instance.data_move(self.y_gm[offset_index * out_element], dst_ub, constant.SID,
                                    constant.DEFAULT_NBURST,
                                    n_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            offset = out_element * num - each_burst_num
            scalar = self.instance.Scalar(dst_ub.dtype)
            with self.instance.for_range(0, each_burst_num) as time:
                scalar.set_as(dst_ub[offset + time])
                dst_ub[time].set_as(scalar)
            self.instance.data_move(self.y_gm[offset_index * out_element + offset], dst_ub,
                                    constant.SID, constant.DEFAULT_NBURST,
                                    constant.DEFAULT_BURST_LEN,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        with self.instance.else_scope():
            self.instance.data_move(self.y_gm[offset_index * out_element], dst_ub, constant.SID,
                                    constant.DEFAULT_NBURST, n_burst_out,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def compute_with_reduce(self, pattern_index):
        """
        compute with vreduce instruction
        """
        total_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        block_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        ub_size = total_size // 2 // (self.dsize * 3)
        align = constant.BLOCK_SIZE // self.dsize
        shape_x = self.input_dict.get("x1").get("shape")
        shape_y = self.input_dict.get("y").get("shape")
        y_size = get_shape_total_number(shape_y)
        in_element = get_shape_total_number(shape_x[pattern_index:])
        out_element = get_shape_total_number(shape_y[pattern_index:])
        out_element_align_num = (align + out_element - 1) // out_element
        loop_all = get_shape_total_number(shape_y[:pattern_index])
        if y_size * self.dsize < constant.BLOCK_SIZE:
            block_num = 1
            inner_loop = 1
            tail = 0
        else:
            num = loop_all // out_element_align_num
            if num < block_num:
                block_num = num if num > 0 else 1
            inner_loop = loop_all // block_num
            tail = loop_all % block_num
        with self.instance.for_range(0, block_num, block_num=block_num) as block_id:
            each_loop = self.instance.Scalar("uint32", name="each_loop")
            each_loop.set_as(inner_loop)
            offset = self.instance.Scalar("uint32", name="offset")
            if tail > 0:
                with self.instance.if_scope(block_id < tail):
                    each_loop.set_as(each_loop + 1)
            offset.set_as(block_id * each_loop)
            if tail > 0:
                with self.instance.if_scope(block_id >= tail):
                    offset.set_as(block_id * (each_loop + 1) - (block_id - tail))

            ub_out_num = ub_size // out_element
            ub_in_num = ub_size // in_element
            block_index = _get_index_num(shape_y, 0, len(shape_y), 1, block_num)
            pattern_before = shape_y[pattern_index - 1]
            axis = self.input_dict.get("axis")
            if block_index + 1 == pattern_index:
                pattern_num = inner_loop + 1
            else:
                pattern_num = pattern_before
            if ub_in_num < pattern_num:
                pattern_num = ub_in_num
            # 'pylint: disable=variable_type_changed
            pattern_num = _get_tail_num(inner_loop, pattern_num, out_element_align_num, -1,
                                        out_element_align_num)

            move_consequent_flag = pattern_num >= out_element_align_num and (
                    pattern_index <= axis or
                    (block_index == 0 and pattern_index == 1) or
                    (tail == 0 and pattern_num == pattern_before and inner_loop % pattern_num == 0))
            if move_consequent_flag:
                loop = each_loop // pattern_num
                loop_tail = each_loop % pattern_num
                thread_num = 1
                if inner_loop // pattern_num >= 2:
                    thread_num = 2
                src1_pattern_ub = self.prepare_src_pattern(pattern_index, pattern_num)
                with self.instance.for_range(0, loop, thread_num=thread_num) as l_i:
                    input_dict = {
                        "src1_pattern_ub": src1_pattern_ub,
                        "ub_size": ub_size,
                        "offset_index": offset + l_i * pattern_num,
                        "in_element": in_element,
                        "out_element": out_element,
                        "pattern_index": pattern_index,
                        "move_flag": tik.all(l_i == loop - 1, loop_tail == 0,
                                             (out_element * pattern_num * self.dsize) % constant.BLOCK_SIZE != 0),
                        "num": pattern_num
                    }
                    self.reduce_move_consequent(input_dict)
                with self.instance.if_scope(loop_tail > 0):
                    input_dict = {
                        "src1_pattern_ub": src1_pattern_ub,
                        "ub_size": ub_size,
                        "offset_index": offset + loop * pattern_num,
                        "in_element": in_element,
                        "out_element": out_element,
                        "pattern_index": pattern_index,
                        "move_flag": (out_element * loop_tail * self.dsize) % constant.BLOCK_SIZE != 0,
                        "num": loop_tail
                    }
                    self.reduce_move_consequent(input_dict)
            else:
                pattern_num = _get_tail_num(inner_loop, ub_out_num, out_element_align_num, -1,
                                            out_element_align_num)
                loop_mv = each_loop // pattern_num
                tail_mv = each_loop % pattern_num
                thread_num = 1
                if inner_loop // pattern_num >= 2:
                    thread_num = 2
                reduce_move_dict = {
                    "ub_size": ub_size,
                    "in_element": in_element,
                    "out_element": out_element,
                    "pattern_index": pattern_index
                }
                src1_pattern_ub = self.prepare_src_pattern(pattern_index)
                with self.instance.for_range(0, loop_mv, thread_num=thread_num) as l_i:
                    reduce_move_dict["src1_pattern_ub"] = src1_pattern_ub
                    reduce_move_dict["reduce_loop"] = pattern_num
                    reduce_move_dict["loop_time"] = offset + l_i * pattern_num
                    reduce_move_dict["move_flag"] = tik.all(l_i == loop_mv - 1, tail_mv == 0, block_num > 1, (
                            out_element * pattern_num * self.dsize) % constant.BLOCK_SIZE != 0)
                    self.reduce_move(reduce_move_dict)
                with self.instance.if_scope(tail_mv > 0):
                    reduce_move_dict["src1_pattern_ub"] = src1_pattern_ub
                    reduce_move_dict["reduce_loop"] = tail_mv
                    reduce_move_dict["loop_time"] = offset + loop_mv * pattern_num
                    reduce_move_dict["move_flag"] = tik.all(block_num > 1, (
                            out_element * tail_mv * self.dsize) % constant.BLOCK_SIZE != 0)
                    self.reduce_move(reduce_move_dict)

    def compute_crop(self):
        """
        compute crop

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        shape_x = self.input_dict.get("x1").get("shape")
        dtype_flag = self.dtype in ["int16", "uint16", "float16", "int32", "uint32", "float32"]
        v200_flag = tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend310P", "Ascend610", "BS9SX1A")
        pattern_index = self.get_pattern_index_num()
        flag = dtype_flag and v200_flag
        if flag and 0 < pattern_index < len(shape_x):
            self.compute_with_reduce(pattern_index)
            return

        block_num, each_block_size, loop, tail = \
            self.get_blockdim_and_loop_cycle()
        shape_out = self.input_dict.get("y").get("shape")
        shape_out_len = get_shape_total_number(shape_out)
        offset_in = self.input_dict.get("offset")
        shape = self.input_dict.get("x1").get("shape")
        element_num, shape_len = self.get_element_num()
        x1_shape_list = get_elem_of_each_dim(shape, len(shape))
        shape = self.input_dict.get("x2").get("shape")
        x2_shape_list = get_elem_of_each_dim(shape, shape_len - 1)
        thread_n = self.get_thread_num(block_num, loop, element_num)

        with self.instance.for_range(0, block_num, block_num=block_num) \
                as block_id:
            ub_tmp = self.instance.Tensor(self.dtype, (256,),
                                          name="ub_tmp", scope=tbe_platform.scope_ubuf)
            self.instance.data_move(ub_tmp,
                                    self.x2_gm[0],
                                    constant.SID,
                                    constant.DEFAULT_NBURST, 1,
                                    constant.STRIDE_ZERO,
                                    constant.STRIDE_ZERO)
            count = self.instance.Scalar("int32")
            count.set_as(0)
            each_loop = self.instance.Scalar("int32")
            each_loop.set_as(loop)
            offset = self.instance.Scalar("int32")
            if tail > 0:
                with self.instance.if_scope(block_id < tail):
                    each_loop.set_as(each_loop + 1)
            offset.set_as(block_id * each_loop)
            with self.instance.if_scope(tik.all(block_id >= tail, tail > 0)):
                offset.set_as(block_id * (each_loop + 1) - (block_id - tail))
            out_offset = self.instance.Scalar("int32")
            out_offset.set_as(offset * element_num)
            cycles = shape_out_len // element_num
            tmp_offset = self.instance.Scalar("int32")
            tmp_offset.set_as(0)
            with self.instance.for_range(offset, cycles,
                                         thread_num=thread_n) as times:
                with self.instance.if_scope(count < each_loop):
                    x1_ub = self.instance.Tensor(self.dtype,
                                                 (self.one_max_size,),
                                                 name="x1_ub",
                                                 scope=tbe_platform.scope_ubuf)
                    x1_offset = self.instance.Scalar("int32")
                    x1_offset.set_as(0)
                    for q in range(shape_len):
                        mod = times
                        for s in range(q):
                            mod %= x2_shape_list[s]
                        mod = mod // x2_shape_list[q] + offset_in[q]
                        x1_offset.set_as(
                            x1_offset + mod * x1_shape_list[q])
                    if element_num * self.dsize < constant.BLOCK_SIZE \
                            and block_num > 1:
                        input_dict = {
                            "x1_ub": x1_ub,
                            "ub_tmp": ub_tmp,
                            "x1_offset": x1_offset,
                            "out_offset": out_offset,
                            "tmp_offset": tmp_offset,
                            "element_num": element_num,
                            "each_block_size": each_block_size,
                            "count": count,
                            "each_loop": each_loop, }
                        self.move_out_less_than32b(input_dict)
                        out_offset.set_as(out_offset + element_num)
                    else:
                        input_dict = {
                            "x1_ub": x1_ub,
                            "x1_offset": x1_offset,
                            "out_offset": out_offset,
                            "element_num": element_num,
                            "block_num": block_num,
                        }
                        self.data_move(input_dict)
                        out_offset.set_as(out_offset + element_num)
                    count.set_as(count + 1)

    def move_out_less_than32b(self, input_dict):
        """
      move data from ub to gm

      Parameters
      ----------
        input_dict: input_dict is a dict, the keys as follow:
                x1_ub: x1_ub is a tensor,store data from gm
                ub_tmp: ub_tmp is a tensor,store last loop 32b data from gm
                x1_offset: x1 gm data offset
                out_offset: output data offset
                tmp_offset: ub_tmp's offset
                element_num: each continuous segment
                each_block_size: each block process the number of element
                count: loop count
                each_loop: the total loop of each block
      Returns
      -------
      None
      """
        x1_ub = input_dict.get("x1_ub")
        ub_tmp = input_dict.get("ub_tmp")
        x1_offset = input_dict.get("x1_offset")
        out_offset = input_dict.get("out_offset")
        tmp_offset = input_dict.get("tmp_offset")
        element_num = input_dict.get("element_num")
        count = input_dict.get("count")
        each_loop = input_dict.get("each_loop")
        nburst = common_util.get_datamove_nburst(self.instance,
                                                 element_num * self.dsize)
        self.instance.data_move(x1_ub, self.x1_gm[x1_offset], constant.SID,
                                constant.DEFAULT_NBURST, nburst,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        loop_32b = (constant.BLOCK_SIZE // self.dsize) // element_num
        if (constant.BLOCK_SIZE // self.dsize) % element_num != 0:
            loop_32b = loop_32b + 1
        scalar = self.instance.Scalar(x1_ub.dtype)
        with self.instance.if_scope(count >= each_loop - loop_32b):
            with self.instance.for_range(0, element_num) as time:
                scalar.set_as(x1_ub[time])
                ub_tmp[tmp_offset + time].set_as(scalar)
            tmp_offset.set_as(tmp_offset + element_num)
            with self.instance.if_scope(count == each_loop - 1):
                out_offset.set_as(out_offset - (loop_32b - 1) * element_num)
                input_dict = {
                    "instance": self.instance,
                    "out_ub": ub_tmp,
                    "out_gm": self.y_gm,
                    "gm_offset": out_offset,
                    "element_num": element_num * loop_32b,
                    "dsize": self.dsize,
                }
                common_util.move_out_non32_alignment(input_dict)

        with self.instance.else_scope():
            nburst = common_util.get_datamove_nburst(self.instance,
                                                     element_num * self.dsize)
            self.instance.data_move(self.y_gm[out_offset],
                                    x1_ub,
                                    constant.SID,
                                    constant.DEFAULT_NBURST, nburst,
                                    constant.STRIDE_ZERO,
                                    constant.STRIDE_ZERO)

    def data_move(self, input_dict):
        """
      move data from ub to gm

      Parameters
      ----------
        input_dict: input_dict is a dict, the keys as follow:
                x1_ub: x1_ub is a tensor,store data from gm
                x1_offset: x1 gm data offset
                out_offset: output data offset
                element_num: each continuous segment
                block_num: blcok number
      Returns
      -------
      None
      """
        x1_ub = input_dict.get("x1_ub")
        out_offset = input_dict.get("out_offset")
        element_num = input_dict.get("element_num")
        block_num = input_dict.get("block_num")
        loop_cycle, last_ub_num = get_loop_param(element_num,
                                                 self.one_max_size)
        total_size = self.instance.Scalar("int32")
        total_size.set_as(self.one_max_size * self.dsize)
        ub_size = self.instance.Scalar("int32")
        ub_size.set_as(self.one_max_size)
        offset_x1 = self.instance.Scalar("int32")
        offset_x1.set_as(input_dict.get("x1_offset"))
        offset_out = self.instance.Scalar("int32")
        offset_out.set_as(out_offset)
        each_burst_num = constant.BLOCK_SIZE // self.dsize
        with self.instance.for_range(0, loop_cycle) as cycle:
            with self.instance.if_scope(cycle == loop_cycle - 1):
                total_size.set_as(last_ub_num * self.dsize)
                ub_size.set_as(last_ub_num)
            nburst = common_util.get_datamove_nburst(self.instance,
                                                     total_size)
            with self.instance.if_scope(
                    tik.all(cycle == loop_cycle - 1,
                            total_size % constant.BLOCK_SIZE != 0,
                            block_num > 1)):
                x1_ub_tmp = self.instance.Tensor(self.dtype, (32,),
                                                 name="x1_ub_tmp",
                                                 scope=tbe_platform.scope_ubuf)
                self.instance.data_move(x1_ub_tmp,
                                        self.x1_gm[offset_x1 +
                                                   ub_size - each_burst_num],
                                        constant.SID,
                                        constant.DEFAULT_NBURST, 1,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                self.instance.data_move(self.y_gm[offset_out +
                                                  ub_size - each_burst_num],
                                        x1_ub_tmp,
                                        constant.SID,
                                        constant.DEFAULT_NBURST, 1,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                with self.instance.if_scope(total_size > constant.BLOCK_SIZE):
                    self.instance.data_move(x1_ub,
                                            self.x1_gm[offset_x1],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            nburst - 1,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
                    self.instance.data_move(self.y_gm[offset_out],
                                            x1_ub,
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            nburst - 1,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
            with self.instance.else_scope():
                self.instance.data_move(x1_ub,
                                        self.x1_gm[offset_x1],
                                        constant.SID,
                                        constant.DEFAULT_NBURST, nburst,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                self.instance.data_move(self.y_gm[offset_out],
                                        x1_ub,
                                        constant.SID,
                                        constant.DEFAULT_NBURST, nburst,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
            offset_x1.set_as(offset_x1 + ub_size)
            offset_out.set_as(offset_out + ub_size)


def get_loop_param(length, max_ub_num):
    """
    get loop parameters

    Parameters
    ----------
    length: total number
    max_ub_num: max of ub num

    Returns
    -------
    loop_cycle: loop cycle
    last_ub_num: the last data needs ub num
    """
    loop_cycle = length // max_ub_num
    last_ub_num = length % max_ub_num
    if last_ub_num != 0:
        loop_cycle = loop_cycle + 1
    else:
        last_ub_num = max_ub_num

    return loop_cycle, last_ub_num


def get_op_support_info(x, size, y, axis=2, offsets=(0), kernel_name="crop"):
    """
    get split info
    """
    ori_shape = x.get("ori_shape")
    if axis < 0:
        axis = axis + len(ori_shape)
    dim_x = len(x.get("shape"))
    format_x = x.get("format").upper()
    not_cut_dim = []
    x_shape = x.get("shape")
    size_shape = size.get("shape")
    if format_x == "NC1HWC0":
        not_cut_dim = [1, 4]

    if format_x in ("ND", "NC1HWC0"):
        axis_split_list = []
        for i in range(dim_x):
            if i < axis and i not in not_cut_dim and x_shape[i] == size_shape[i]:
                split = [util_select_op_base.SplitInput([0, [i], [-1], [-1]],
                                                        [1, [i], [-1], [-1]]),
                         util_select_op_base.SplitOutput([0, [i]])]
                axis_split_list.append(split)
    else:
        axis_split_list = None
    axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_list, axis_reduce_list)
    return op_cal_info_in_json


def op_select_format(x, size, y, axis=2, offsets=(0), kernel_name="crop"):
    """
    1. when ori_format of x is equal to "NCHW", the ori_shape of x is
    equal to 4 and the axis is greater than or equal to 2. the Op
    Crop can support HC1HWC0 and ND.
    > for example :
    > x : Tensor of (shape=(16, 16, 16, 16), "NCHW")
    > size : Tensor of (shape=(16, 16, 16, 16), "NCHW")
    > the Op Crop can process with NC1HWC0:
    > x : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    > size : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    """
    dtype_base = [
        "float16", "float", "int32", "int8", "int16", "int64", "uint8",
        "uint16", "uint32", "uint64"
    ]
    dtype_lhisi = [
        "float16", "int32", "int8", "int16", "int64", "uint8",
        "uint16", "uint32", "uint64"
    ]

    ori_format = x.get("ori_format").upper()
    ori_shape = x.get("ori_shape")

    dtype_out = dtype_base
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype_out = dtype_lhisi

    if axis < 0:
        axis = axis + len(ori_shape)

    format_out = ["ND"] * len(dtype_out)
    if ori_format == "NCHW" and len(ori_shape) == 4 and axis >= 2:
        format_out = format_out + ["NC1HWC0"] * len(dtype_out)
        dtype_out = dtype_out + dtype_out

    dtype_str = ','.join(dtype_out)
    format_str = ','.join(format_out)

    input0 = util_select_op_base.gen_param(
        classify="input0", name="x", datatype=dtype_str, format=format_str)
    input1 = util_select_op_base.gen_param(
        classify="input1", name="size", datatype=dtype_str, format=format_str)
    output0 = util_select_op_base.gen_param(
        classify="output0", name="y", datatype=dtype_str, format=format_str)
    param_list = [input0, input1, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def check_and_adjust_offset(input_dict):
    """
    check the parameters is valid, if one is invalid,then raise error
    adjust offset's length as the same as len(x1_shape)

    Parameters
    ----------
    input_dict: input_dict is a dict, the keys as follow:
                x1: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                x2: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                y: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                axis: crop start with axis
                offsets: crop start offset of each axis
                kernel_name: cce kernel name, default value is "crop"
    Returns
    -------
    None
    """

    x1_dtype = input_dict.get("x1").get("dtype").lower()
    x1_shape = input_dict.get("x1").get("shape")
    x2_dtype = input_dict.get("x2").get("dtype").lower()
    x2_shape = input_dict.get("x2").get("shape")
    y_dtype = input_dict.get("y").get("dtype").lower()
    y_shape = input_dict.get("y").get("shape")
    para_check.check_shape(x1_shape, param_name="x1")
    para_check.check_dtype(x1_dtype, ("int8", "uint8", "int16", "uint16", "int32",
                                      "uint32", "int64", "uint64", "float16",
                                      "float32"), param_name="x1")
    para_check.check_shape(x2_shape, param_name="x2")
    para_check.check_dtype(x2_dtype, ("int8", "uint8", "int16", "uint16", "int32",
                                      "uint32", "int64", "uint64", "float16",
                                      "float32"), param_name="x2")

    para_check.check_shape(y_shape, param_name="y")
    para_check.check_dtype(y_dtype, ("int8", "uint8", "int16", "uint16", "int32",
                                     "uint32", "int64", "uint64", "float16", "float32"), param_name="y")
    if x2_dtype != y_dtype or y_dtype != x1_dtype:
        rule_desc = "the dtype of size, x and y should be same"
        error_manager_vector.raise_err_check_params_rules("crop", rule_desc,
                                                        "y_dtype", y_dtype)

    if not check_same_shape(y_shape, x2_shape):
        error_manager_vector.raise_err_inputs_dtype_not_equal("crop", "y", "x2", y_shape, x2_shape)

    if len(x2_shape) != len(x1_shape):
        error_detail = "the parameter[%s][%s] are not equal in shape with shapes[%s][%s]." \
                        % ("x1", "x2", ','.join(str(i) for i in x1_shape),  ','.join(str(i) for i in x2_shape))
        error_manager_vector.raise_err_specific_reson("crop", error_detail)

    # check his-es check offset
    x1_ori_shape = input_dict.get("x1").get("shape")
    if 'ori_shape' in input_dict.get("x1"):
        x1_ori_shape = input_dict.get("x1").get("ori_shape")
    axis = input_dict.get("axis")
    if axis >= len(x1_ori_shape) or axis < -len(x1_ori_shape):
        error_manager_vector.raise_err_input_param_not_in_range("crop", "axis", str(-len(x1_ori_shape)),
                                                                str(len(x1_ori_shape)), axis)

    if axis < 0:
        input_dict["axis"] = axis + len(x1_ori_shape)
        axis = axis + len(x1_ori_shape)
    # the same verify as caffe
    offset = input_dict.get("offset")
    offset_final = [0] * len(x1_shape)
    if len(offset) == 1:
        for i in range(axis, len(x1_ori_shape)):
            offset_final[i] = offset[0]
    elif len(offset) != 0:
        if len(offset) != len(x1_ori_shape) - axis:
            rule_desc = "axis(%d)+len(offset)(%d) must equal to input dim(x)" \
                        % (axis, len(offset))
            error_manager_vector.raise_err_check_params_rules("crop", rule_desc,
                                                    "input dim", len(x1_ori_shape))

        offset_final[axis:len(x1_ori_shape)] = offset
    len_offset_final = len(offset_final)
    for i in range(len_offset_final):
        if x1_shape[i] - offset_final[i] < x2_shape[i]:
            rule_desc = "the ith[%d]'s size's dimension can't be bigger than " \
                        "x's size minus offset" % i
            error_manager_vector.raise_err_check_params_rules("crop", rule_desc,
                                                "size's dimension", x2_shape[i])
    input_dict["offset"] = offset_final


def get_shape_total_number(shape):
    """
    get the number of element from the shape

    Parameters
    ----------
    shape: out put shape

    Returns
    -------
    total_number: the number of element of the shape
    """
    total_number = 1
    for i in shape:
        total_number = total_number * i

    return total_number


def check_same_shape(shape_x, shape_y):
    """
    check shape_x is the same shape as shape_y

    Parameters
    ----------
    shape_x: a tuple or list
    shape_y: a tuple or list

    Returns
    -------
    boolean: True has the same shape, False does't has the same shape
    """
    shape_x_len = len(shape_x)
    shape_y_len = len(shape_y)
    if shape_x_len != shape_y_len:
        return False
    for k in range(shape_x_len):
        if shape_x[k] != shape_y[k]:
            return False

    return True


def get_elem_of_each_dim(shape, element_num):
    """
    get element of each dim

    Parameters
    ----------
    shape: out put shape
    element_num: element num

    Returns
    -------
    elem_of_each_dim
    """
    elem_of_each_dim = [1] * len(shape)
    for i in range(element_num):
        j = i + 1
        while j < element_num:
            elem_of_each_dim[i] = elem_of_each_dim[i] * shape[j]
            j = j + 1

    return elem_of_each_dim


def _get_input_offset_list(offset_list_tuple):
    """
    get the input offset list
    """
    size, shape_x, shape_y, offset_in, element_size, num = offset_list_tuple
    x_size = get_shape_total_number(shape_x)
    y_size = get_shape_total_number(shape_y)
    offset_list = [0] * size
    x1_shape_list = get_elem_of_each_dim(shape_x, len(shape_x))
    x2_shape_list = get_elem_of_each_dim(shape_y, len(shape_y) - 1)
    for m in range(num):
        start = m * x_size
        for i in range(y_size // element_size):
            offset = 0
            for q in range(len(shape_y)):
                mod = i
                for s in range(q):
                    mod %= x2_shape_list[s]
                mod = mod // x2_shape_list[q] + offset_in[q]
                offset = offset + mod * x1_shape_list[q]
            offset_list[start + offset:start + offset + element_size] = [1] * element_size
    return offset_list


def _get_index_num(shape, start, end, step, threshold):
    """
    get the index num
    """
    val_cnt = 1
    index = end - step
    for i in range(start, end, step):
        val_cnt = val_cnt * shape[i]
        if val_cnt >= threshold:
            index = i
            break
    return index


def _get_input_offset(move_offset_tuple):
    """
    calculate the offset position of each input
    """
    shape_x, shape_y, offset_in, loop_i, pattern_index, x_offset = move_offset_tuple
    x1_shape_list = get_elem_of_each_dim(shape_x[:pattern_index], len(shape_x[:pattern_index]))
    x2_shape_list = get_elem_of_each_dim(shape_y[:pattern_index], len(shape_y[:pattern_index]))
    offset = 0
    for j in range(0, pattern_index):
        mod = loop_i
        for k in range(j):
            mod %= x2_shape_list[k]
        mod = mod // x2_shape_list[j] + offset_in[j]
        offset = offset + mod * x1_shape_list[j]
    x_offset.set_as(offset)


def _get_tail_num(size, start, end, step, threshold):
    """
    get the tail num
    """
    if threshold == 1:
        return start
    val_cnt = 1
    for i in range(start, end, step):
        if size % i >= threshold:
            val_cnt = i
            break
    return val_cnt
