#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
shuffle_channel
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl import constant_util as constant
from impl import common_util
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_vector
from impl.transpose_d import transpose_d

# reserve size for ub
RESERVE_SIZE = 16 * 1024
BLOCK_NORMAL = 8
BLOCK_SOC = 10


# 'pylint: disable=invalid-name,too-many-locals,too-many-statements,too-many-arguments,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def shuffle_channel(x, y, group=1, kernel_name="shuffle_channel"):
    """the main function of shuffle_channel

    Parameters
    ----------
    x: dict,shape and datatype,datatype supports int8,uint8,int16,uint16,int32,
         uint32,int64,uint64,float16,float32
    y: dict,shape and datatype,datatype supports int8,uint8,int16,uint16,int32,
         uint32,int64,uint64,float16,float32
    group: 1 channel group
    kernel_name: cce kernel name, default value is "shuffle_channel"

    Returns
    -------
    tik_instance: tik_instance
    """
    input_dict = {
        "x": x,
        "y": y,
        "group": group,
        "kernel_name": kernel_name
    }
    _check_param(input_dict)

    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("BS9SX1A",):
        shuffle_channel_pg2(x, y, group, kernel_name)
        return

    shuffle_channel_common(input_dict, kernel_name)


def shuffle_channel_pg2(x, y, group=1, kernel_name="shuffle_channel"):
    shape = x.get("shape")
    new_shape_input = [shape[0], group, shape[1] // group, shape[2] * shape[3]]
    new_shape_output = [shape[0], shape[1] // group, group, shape[2] * shape[3]]
    x.update({"shape": new_shape_input})
    y.update({"shape": new_shape_output})
    transpose_d(x, y, [0, 2, 1, 3], kernel_name)


def shuffle_channel_common(input_dict, kernel_name="shuffle_channel"):
    shuffle_process = ShuffleChannel(input_dict)
    shuffle_process.compute_shuffle_channel()
    shuffle_process.instance.BuildCCE(kernel_name=kernel_name,
                                      inputs=(shuffle_process.x_gm,),
                                      outputs=(shuffle_process.y_gm,),
                                      enable_l2=False)

    return shuffle_process.instance


def get_op_support_info(x, y, group=1, kernel_name="shuffle_channel"):
    """
    get split info
    """
    dim_x = len(x.get("shape"))
    format_x = x.get("format").upper()
    not_cut_dim = [1]
    if format_x == "NCHW":
        axis_split_list = []
        for i in range(dim_x):
            if i not in not_cut_dim:
                split = [util_select_op_base.SplitInput([0, [i], [-1], [-1]]),
                         util_select_op_base.SplitOutput([0, [i]])]
                axis_split_list.append(split)
    else:
        axis_split_list = None

    axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_list, axis_reduce_list)
    return op_cal_info_in_json


class ShuffleChannel:
    """
    Function: store ShuffleChannel parameters  and compute ShuffleChannel
    Modify : 202--02-12
    """

    def __init__(self, input_dict):
        """init the ShuffleChannel parameters

        Parameters
        ----------
        input_dict: input_dict is a dict, the keys as follow:
            x: dict,shape and datatype,datatype supports int8,uint8,int16,
              uint16,int32,uint32,int64,uint64,float16,float32
            y: dict,shape and datatype,datatype supports int8,uint8,int16,
              uint16,int32,uint32,int64,uint64,float16,float32
            group: 1 channel group
            kernel_name: cce kernel name, default value is "shuffle_channel"
        Returns
        -------
        None
        """
        self.instance = tik.Tik()
        self.dtype = input_dict.get("x").get("dtype").lower()
        self.dsize = common_util.get_data_size(self.dtype)
        total_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        ub_size = (total_size - RESERVE_SIZE) // (2 * self.dsize)
        burnest_len = constant.BLOCK_SIZE // self.dsize
        ub_size = ((ub_size + burnest_len - 1) // burnest_len) * burnest_len
        self.one_max_size = ub_size
        x_len = _get_shape_total_number(input_dict.get("x").get("shape"))
        x_len = ((x_len + burnest_len - 1) // burnest_len) * burnest_len
        hw = input_dict.get("y").get("shape")[2] * \
             input_dict.get("y").get("shape")[3]
        mod = hw % burnest_len
        if mod != 0:
            x_len = x_len + burnest_len
        self.x_gm = self.instance.Tensor(self.dtype, (x_len,), name="x_gm",
                                         scope=tbe_platform.scope_gm)
        self.y_gm = self.instance.Tensor(self.dtype, (x_len,), name="y_gm",
                                         scope=tbe_platform.scope_gm)
        self.input_dict = input_dict

    def get_blockdim_and_loop_cycle(self):
        """get block dim and loop cycle

        Parameters
        ----------
        None
        Returns
        -------
        block_num, inner_loop, inner_loop_mod
        """
        block_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        shape_y = self.input_dict.get("y").get("shape")
        limit_size_of_each_block = shape_y[2] * shape_y[3]
        total_channel = shape_y[0] * shape_y[1]
        each_block_num = constant.BLOCK_SIZE // self.dsize
        each_block_align = \
            ((each_block_num + limit_size_of_each_block - 1) //
             limit_size_of_each_block) * limit_size_of_each_block
        if limit_size_of_each_block * self.dsize < constant.BLOCK_SIZE:
            all_size = total_channel * limit_size_of_each_block * self.dsize
            if all_size < constant.BLOCK_SIZE:
                block_num = 1
                return block_num, total_channel, 0

            limit_size_of_each_block = each_block_align
        limit_channel_of_each_block = limit_size_of_each_block // \
                                      (shape_y[2] * shape_y[3])
        loop = (total_channel * shape_y[2] * shape_y[3]) // limit_size_of_each_block
        mod_channel = ((total_channel * shape_y[2] * shape_y[3]) % \
                       limit_size_of_each_block) // (shape_y[2] * shape_y[3])
        if loop <= block_num:
            block_num = loop
            inner_loop = limit_channel_of_each_block
            inner_loop_mod = mod_channel
        else:
            inner_loop = (loop // block_num) * limit_channel_of_each_block
            inner_loop_mod = (loop % block_num) * limit_channel_of_each_block + mod_channel
            if inner_loop_mod > block_num:
                inner_loop = inner_loop + inner_loop_mod // block_num
                inner_loop_mod = inner_loop_mod % block_num

        return block_num, inner_loop, inner_loop_mod

    def check_can_use_v200_instruction(self):
        """
        check can use v200 instruction or feature
        """
        dtype_flag = self.dtype in ["int16", "float16", "int32", "float32"]
        dtype_cs_flag = self.dtype in ["int16", "float16", "int32"]
        v200_flag = tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend310P", "Ascend610", "BS9SX1A")
        cs_flag = tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Hi3796CV300CS", "SD3403")
        block_num, ub_size = self.get_ub_block_size(self.dsize * 2)
        shape = self.input_dict.get("x").get("shape")
        num = shape[1] * shape[2] * shape[3]

        block_index = 0
        val_cnt = 1
        for index, val in enumerate(shape):
            val_cnt = val_cnt * val
            if val_cnt >= block_num:
                block_index = index
                break

        size_flag = num <= ub_size and block_index < 1
        flag = (dtype_flag and v200_flag) or (dtype_cs_flag and cs_flag)
        if flag and size_flag:
            return True

        return False

    def vadds_compute(self, n_i, dst_ub, src_ub):
        """describe the process of calculating the vadds instruction"""
        shape_out = self.input_dict.get("y").get("shape")
        each_block_num = constant.VECTOR_BYTE_SIZE // self.dsize
        channel = shape_out[1]
        group = self.input_dict.get("group")
        hw_num = shape_out[2] * shape_out[3]
        hw_align = (hw_num + each_block_num - 1) // each_block_num * each_block_num
        repeat = hw_align // each_block_num
        zero_val = 0
        for c_i in range(channel):
            j = c_i // group
            i = c_i % group
            base_index = (n_i * channel + i * (channel // group) + j) * hw_num
            dst_index = (n_i * channel + c_i) * hw_num
            self.instance.vec_adds(each_block_num, dst_ub[dst_index], src_ub[base_index], zero_val, repeat,
                                   constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)

    def vadds_move(self, input_dict):
        """describe the process of moving in and out and calculating the vadds instruction"""
        ub_size = input_dict.get("ub_size")
        data_size = input_dict.get("data_size")
        gm_offset = input_dict.get("gm_offset")
        adds_loop = input_dict.get("adds_loop")
        move_flag = input_dict.get("move_flag")

        src_ub = self.instance.Tensor(self.dtype, (ub_size,), name="src_ub", scope=tbe_platform.scope_ubuf)
        dst_ub = self.instance.Tensor(self.dtype, (ub_size,), name="dst_ub", scope=tbe_platform.scope_ubuf)
        n_burst = common_util.get_datamove_nburst(self.instance, data_size * self.dsize)

        self.instance.data_move(src_ub, self.x_gm[gm_offset], constant.SID, constant.DEFAULT_NBURST,
                                n_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)

        with self.instance.for_range(0, adds_loop) as n_i:
            self.vadds_compute(n_i, dst_ub, src_ub)

        with self.instance.if_scope(move_flag):
            input_dict = {
                "instance": self.instance,
                "out_ub": dst_ub,
                "out_gm": self.y_gm,
                "gm_offset": gm_offset,
                "element_num": data_size,
                "dsize": self.dsize,
            }
            common_util.move_out_non32_alignment(input_dict)
        with self.instance.else_scope():
            self.instance.data_move(self.y_gm[gm_offset], dst_ub, constant.SID, constant.DEFAULT_NBURST,
                                    n_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def get_ub_block_size(self, tensor_size):
        """get ub size and core number"""
        block_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        total_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        soc_flag = tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend310P",) and block_num == BLOCK_SOC
        block_num = BLOCK_NORMAL if soc_flag else block_num
        reserve_size = 4 * 1024
        block_size = constant.BLOCK_SIZE // self.dsize
        ub_size = (total_size - reserve_size) // 2 // tensor_size // block_size * block_size
        return block_num, ub_size

    def move_with_vadds(self):
        """move with vadds instruction"""
        shape_out = self.input_dict.get("y").get("shape")
        shape_size = _get_shape_total_number(shape_out)
        element_num = shape_out[1] * shape_out[2] * shape_out[3]
        num_32b = constant.BLOCK_SIZE // self.dsize
        element_num_32b_align = (element_num + num_32b - 1) // num_32b * num_32b
        block_num, ub_size = self.get_ub_block_size(self.dsize * 2)
        ub_num = ub_size // element_num
        loop = shape_size // element_num_32b_align

        if loop < block_num:
            block_num = loop if loop > 0 else 1
        inner_loop = (shape_size // element_num) // block_num
        tail = (shape_size // element_num) % block_num

        thread_num = 1
        if inner_loop // ub_num > 1:
            thread_num = 2

        with self.instance.for_range(0, block_num, block_num=block_num) as block_id:
            each_loop = self.instance.Scalar("uint32")
            offset = self.instance.Scalar("uint32")
            each_loop.set_as(inner_loop)
            if tail > 0:
                with self.instance.if_scope(block_id < tail):
                    each_loop.set_as(each_loop + 1)
            offset.set_as(block_id * each_loop)
            with self.instance.if_scope(tik.all(block_id >= tail, tail > 0)):
                offset.set_as(block_id * (each_loop + 1) - (block_id - tail))

            loop_mv = self.instance.Scalar("uint32")
            tail_mv = self.instance.Scalar("uint32")
            loop_32b = self.instance.Scalar("uint32")
            block_size = constant.BLOCK_SIZE // self.dsize
            align_32b = (block_size + element_num - 1) // element_num
            with self.instance.if_scope(tik.any(each_loop % ub_num >= align_32b, each_loop <= align_32b)):
                loop_mv.set_as(each_loop // ub_num)
                tail_mv.set_as(each_loop % ub_num)
                loop_32b.set_as(0)
            with self.instance.else_scope():
                loop_mv.set_as((each_loop - align_32b) // ub_num)
                tail_mv.set_as((each_loop - align_32b) % ub_num)
                loop_32b.set_as(align_32b)

            with self.instance.if_scope(loop_mv > 0):
                with self.instance.for_range(0, loop_mv, thread_num=thread_num) as l_i:
                    input_dict = {
                        "ub_size": ub_size,
                        "data_size": ub_num * element_num,
                        "gm_offset": offset * element_num + l_i * ub_num * element_num,
                        "adds_loop": ub_num,
                        "move_flag": tik.all(l_i == loop_mv - 1, tail_mv == 0, loop_32b == 0, block_num > 1)
                    }
                    self.vadds_move(input_dict)
            with self.instance.if_scope(tail_mv > 0):
                input_dict = {
                    "ub_size": ub_size,
                    "data_size": tail_mv * element_num,
                    "gm_offset": offset * element_num + loop_mv * ub_num * element_num,
                    "adds_loop": tail_mv,
                    "move_flag": tik.all(loop_32b == 0, block_num > 1)
                }
                self.vadds_move(input_dict)
            with self.instance.if_scope(loop_32b > 0):
                input_dict = {
                    "ub_size": ub_size,
                    "data_size": loop_32b * element_num,
                    "gm_offset": offset * element_num + (loop_mv * ub_num + tail_mv) * element_num,
                    "adds_loop": loop_32b,
                    "move_flag": tik.all(loop_32b > 0, block_num > 1)
                }
                self.vadds_move(input_dict)

    def move_without_transform(self):
        """when group = 1 or group = channel, directly move data in and out"""
        size = _get_shape_total_number(self.input_dict.get("x").get("shape"))
        ai_core_num, ub_size = self.get_ub_block_size(self.dsize)
        block_size = constant.BLOCK_SIZE // self.dsize
        if size <= block_size:
            block_num = 1
        else:
            all_block_num = size // block_size
            block_num = ai_core_num
            if all_block_num < ai_core_num:
                block_num = all_block_num
        each_len = size // block_num
        each_mod = size % block_num

        thread_num = 1
        if each_len // ub_size > 1:
            thread_num = 2

        with self.instance.for_range(0, block_num, block_num=block_num) as block_id:
            each_size = self.instance.Scalar("uint32")
            offset = self.instance.Scalar("uint32")
            each_size.set_as(each_len)
            if each_mod > 0:
                with self.instance.if_scope(block_id < each_mod):
                    each_size.set_as(each_len + 1)
            offset.set_as(block_id * each_size)
            if each_mod > 0:
                with self.instance.if_scope(block_id >= each_mod):
                    offset.set_as(block_id * (each_size + 1) - (block_id - each_mod))

            ub_loop = each_size // ub_size
            ub_mod = each_size % ub_size
            with self.instance.for_range(0, ub_loop, thread_num=thread_num) as loop_id:
                src_ub = self.instance.Tensor(self.dtype, (ub_size,), name="src_ub", scope=tbe_platform.scope_ubuf)
                burst_len = ub_size // block_size
                self.instance.data_move(src_ub, self.x_gm[offset + loop_id * ub_size],
                                        constant.SID, constant.DEFAULT_NBURST, burst_len,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                self.instance.data_move(self.y_gm[offset + loop_id * ub_size], src_ub,
                                        constant.SID, constant.DEFAULT_NBURST, burst_len,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            with self.instance.if_scope(ub_mod > 0):
                src_ub = self.instance.Tensor(self.dtype, (ub_size,), name="src_ub", scope=tbe_platform.scope_ubuf)
                with self.instance.if_scope(tik.all(block_num > 1, ub_mod % block_size != 0)):
                    src_ub_1 = self.instance.Tensor(self.dtype, (16,), name="src_ub_1", scope=tbe_platform.scope_ubuf)
                    index = offset + ub_loop * ub_size
                    with self.instance.if_scope(ub_mod >= block_size):
                        burst_len = ub_mod // block_size
                        self.instance.data_move(src_ub, self.x_gm[index],
                                                constant.SID, constant.DEFAULT_NBURST, burst_len,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                        self.instance.data_move(self.y_gm[index], src_ub,
                                                constant.SID, constant.DEFAULT_NBURST, burst_len,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                        gm_offset = index + burst_len * block_size - block_size + ub_mod % block_size
                        self.instance.data_move(src_ub_1, self.x_gm[gm_offset],
                                                constant.SID, constant.DEFAULT_NBURST, constant.DEFAULT_BURST_LEN,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                        self.instance.data_move(self.y_gm[gm_offset], src_ub_1,
                                                constant.SID, constant.DEFAULT_NBURST, constant.DEFAULT_BURST_LEN,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                    with self.instance.else_scope():
                        gm_offset = index - block_size + ub_mod % block_size
                        self.instance.data_move(src_ub_1, self.x_gm[gm_offset],
                                                constant.SID, constant.DEFAULT_NBURST, constant.DEFAULT_BURST_LEN,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                        self.instance.data_move(self.y_gm[gm_offset], src_ub_1,
                                                constant.SID, constant.DEFAULT_NBURST, constant.DEFAULT_BURST_LEN,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                with self.instance.else_scope():
                    burst_len = (ub_mod + block_size - 1) // block_size
                    self.instance.data_move(src_ub, self.x_gm[offset + ub_loop * ub_size],
                                            constant.SID, constant.DEFAULT_NBURST, burst_len,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                    self.instance.data_move(self.y_gm[offset + ub_loop * ub_size], src_ub,
                                            constant.SID, constant.DEFAULT_NBURST, burst_len,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def compute_shuffle_channel(self):
        """compute shuffle_channel

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        group = self.input_dict.get("group")
        channel = self.input_dict.get("x").get("shape")[1]
        if group in (1, channel):
            self.move_without_transform()
            return

        if self.check_can_use_v200_instruction():
            self.move_with_vadds()
            return

        block_num, inner_loop, tail = self.get_blockdim_and_loop_cycle()
        shape_out = self.input_dict.get("y").get("shape")
        hw = shape_out[2] * shape_out[3]
        if hw * self.dsize < constant.BLOCK_SIZE:
            if block_num == 1 and inner_loop > 1:
                thread_num = 2
            else:
                thread_num = 1
        else:
            thread_num = 1
            if inner_loop > 1:
                thread_num = 2
        with self.instance.for_range(0, block_num, block_num=block_num) as block_id:
            ub_tmp = self.instance.Tensor(self.dtype, (256,),
                                          name="ub_tmp", scope=tbe_platform.scope_ubuf)
            loop = self.instance.Scalar("int32")
            tmp_offset = self.instance.Scalar("int32")
            tmp_offset.set_as(0)
            with self.instance.for_range(0,
                                         inner_loop,
                                         thread_num=thread_num) as inner_cycle:
                x_ub = self.instance.Tensor(self.dtype, (self.one_max_size,),
                                            name="x_ub", scope=tbe_platform.scope_ubuf)
                loop.set_as(block_id * inner_loop + inner_cycle)
                with self.instance.if_scope(tail > 0):
                    with self.instance.if_scope(block_id < tail):
                        loop.set_as(block_id * inner_loop + inner_cycle + block_id)
                    with self.instance.else_scope():
                        loop.set_as(block_id * inner_loop + inner_cycle + tail)

                src_start, dest_start = self.get_start_address(loop)
                if hw * self.dsize < constant.BLOCK_SIZE and block_num > 1:
                    input_dict = {
                        "x_ub": x_ub,
                        "ub_tmp": ub_tmp,
                        "src_start": src_start,
                        "dest_start": dest_start,
                        "element_num": hw,
                        "each_loop": inner_cycle,
                        "total_loop": inner_loop,
                        "tmp_offset": tmp_offset,
                    }

                    self.move_out_less_than32b(input_dict)

                else:
                    input_dict = {
                        "x_ub": x_ub,
                        "src_start": src_start,
                        "dest_start": dest_start,
                        "element_num": hw,
                        "block_num": block_num,
                    }
                    self.data_move(input_dict)
            if tail > 0:
                with self.instance.if_scope(block_id < tail):
                    x_ub = self.instance.Tensor(self.dtype,
                                                (self.one_max_size,),
                                                name="x_ub",
                                                scope=tbe_platform.scope_ubuf)
                    loop.set_as(loop + 1)
                    src_start, dest_start = self.get_start_address(loop)

                    with self.instance.if_scope((hw * self.dsize) >= constant.BLOCK_SIZE):
                        input_dict = {
                            "x_ub": x_ub,
                            "src_start": src_start,
                            "dest_start": dest_start,
                            "element_num": hw,
                            "block_num": block_num,
                        }
                        self.data_move(input_dict)

                    with self.instance.else_scope():
                        self.instance.data_move(x_ub,
                                                self.x_gm[src_start],
                                                constant.SID,
                                                constant.DEFAULT_NBURST, 1,
                                                constant.STRIDE_ZERO,
                                                constant.STRIDE_ZERO)
                        input_dict = {
                            "instance": self.instance,
                            "out_ub": x_ub,
                            "out_gm": self.y_gm,
                            "gm_offset": dest_start,
                            "element_num": hw,
                            "dsize": self.dsize,
                        }
                        common_util.move_out_non32_alignment(input_dict)

    def get_start_address(self, loop):
        """get the start address of the source and dest tensor

        Parameters
        ----------
        loop: loop times
        Returns
        -------
        src_start, dest_start
        """
        shape_out = self.input_dict.get("y").get("shape")
        channel = shape_out[1]
        group = self.input_dict.get("group")
        src_start = self.instance.Scalar("int32")
        group_row = (loop % channel) // group
        group_col = (loop % channel) % group
        index = (loop // channel) * channel + \
                group_col * (channel // group) + group_row
        hw = shape_out[2] * shape_out[3]
        src_start.set_as(index * hw)
        dest_start = self.instance.Scalar("int32")
        dest_start.set_as(loop * hw)
        return src_start, dest_start

    def move_out_less_than32b(self, input_dict):
        """move data from ub to gm

        Parameters
        ----------
        input_dict: input_dict is a dict, the keys as follow:
                x_ub: x_ub is a tensor,store data from gm
                ub_tmp: ub_tmp is a tensor,store last loop 32b data from gm
                src_start: src address
                dest_start: dest address
                element_num: each continuous segment
                each_loop: loop times
                total_loop: total loop of each block
                tmp_offset: the offset of ub_tmp
        Returns
        -------
        None
        """
        x_ub = input_dict.get("x_ub")
        ub_tmp = input_dict.get("ub_tmp")
        src_start = input_dict.get("src_start")
        dest_start = input_dict.get("dest_start")
        each_loop = input_dict.get("each_loop")
        element_num = input_dict.get("element_num")
        total_loop = input_dict.get("total_loop")
        tmp_offset = input_dict.get("tmp_offset")
        loop_32b = (constant.BLOCK_SIZE // self.dsize) // element_num
        if (constant.BLOCK_SIZE // self.dsize) % element_num != 0:
            loop_32b = loop_32b + 1

        nburst = common_util.get_datamove_nburst(self.instance,
                                                 element_num * self.dsize)
        self.instance.data_move(x_ub, self.x_gm[src_start], constant.SID,
                                constant.DEFAULT_NBURST, nburst,
                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        scalar = self.instance.Scalar(x_ub.dtype)

        with self.instance.if_scope(each_loop >= total_loop - loop_32b):
            with self.instance.for_range(0, element_num) as time:
                scalar.set_as(x_ub[time])
                ub_tmp[tmp_offset + time].set_as(scalar)
            tmp_offset.set_as(tmp_offset + element_num)
            with self.instance.if_scope(each_loop == total_loop - 1):
                dest_start.set_as(dest_start - (loop_32b - 1) * element_num)
                input_dict = {
                    "instance": self.instance,
                    "out_ub": ub_tmp,
                    "out_gm": self.y_gm,
                    "gm_offset": dest_start,
                    "element_num": element_num * loop_32b,
                    "dsize": self.dsize,
                }
                common_util.move_out_non32_alignment(input_dict)

        with self.instance.else_scope():
            nburst = common_util.get_datamove_nburst(self.instance,
                                                     element_num * self.dsize)
            self.instance.data_move(self.y_gm[dest_start],
                                    x_ub,
                                    constant.SID,
                                    constant.DEFAULT_NBURST, nburst,
                                    constant.STRIDE_ZERO,
                                    constant.STRIDE_ZERO)

    def data_move(self, input_dict):
        """move data from ub to gm

        Parameters
        ----------
        input_dict: input_dict is a dict, the keys as follow:
                x_ub: x_ub is a tensor,store data from gm
                src_start: the start address of src tensor
                dest_start: the start address of dest tensor
                element_num: each continuous segment
                block_num: blcok number
        Returns
        -------
        None
        """
        x_ub = input_dict.get("x_ub")
        element_num = input_dict.get("element_num")
        block_num = input_dict.get("block_num")
        loop_num, last_ub_num = _get_loop_param(element_num,
                                                self.one_max_size)
        cur_size = self.instance.Scalar("int32")
        cur_size.set_as(self.one_max_size * self.dsize)
        ub_num = self.instance.Scalar("int32")
        ub_num.set_as(self.one_max_size)
        offset_in = self.instance.Scalar("int32")
        offset_in.set_as(input_dict.get("src_start"))
        offset_out = self.instance.Scalar("int32")
        offset_out.set_as(input_dict.get("dest_start"))
        each_burst_num = constant.BLOCK_SIZE // self.dsize
        with self.instance.for_range(0, loop_num) as cycle:
            with self.instance.if_scope(cycle == loop_num - 1):
                cur_size.set_as(last_ub_num * self.dsize)
                ub_num.set_as(last_ub_num)
            n_burst = common_util.get_datamove_nburst(self.instance,
                                                      cur_size)
            mod = cur_size % constant.BLOCK_SIZE
            with self.instance.if_scope(
                    tik.all(cycle == loop_num - 1, mod != 0, block_num > 1)):
                x_ub_tail = self.instance.Tensor(self.dtype, (32,),
                                                 name="x_ub_tail",
                                                 scope=tbe_platform.scope_ubuf)
                self.instance.data_move(x_ub_tail,
                                        self.x_gm[offset_in +
                                                  ub_num - each_burst_num],
                                        constant.SID,
                                        constant.DEFAULT_NBURST, 1,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                self.instance.data_move(self.y_gm[offset_out +
                                                  ub_num - each_burst_num],
                                        x_ub_tail,
                                        constant.SID,
                                        constant.DEFAULT_NBURST, 1,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                with self.instance.if_scope(cur_size > constant.BLOCK_SIZE):
                    self.instance.data_move(x_ub,
                                            self.x_gm[offset_in],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            n_burst - 1,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
                    self.instance.data_move(self.y_gm[offset_out],
                                            x_ub,
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            n_burst - 1,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
            with self.instance.else_scope():
                self.instance.data_move(x_ub,
                                        self.x_gm[offset_in],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        n_burst, constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
                self.instance.data_move(self.y_gm[offset_out],
                                        x_ub,
                                        constant.SID,
                                        constant.DEFAULT_NBURST, n_burst,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
            offset_in.set_as(offset_in + ub_num)
            offset_out.set_as(offset_out + ub_num)


def _get_loop_param(length, max_ub_num):
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


def _check_param(input_dict):
    """
    check the parameters is valid

    Parameters
    ----------
    input_dict: input_dict is a dict, the keys as follow:
                x: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                y: dict,shape and datatype,datatype supports int8,uint8,
                    int16,uint16,int32,uint32,int64,uint64,float16,float32
                group: channel group default 1
                kernel_name: cce kernel name, default value is "shuffle_channel"
    Returns
    -------
    None
    """
    x_dtype = input_dict.get("x").get("dtype").lower()
    x_shape = input_dict.get("x").get("shape")
    y_dtype = input_dict.get("y").get("dtype").lower()
    y_shape = input_dict.get("y").get("shape")

    para_check.check_shape(x_shape, param_name="input_x")
    para_check.check_dtype(x_dtype,
                           ["int8", "uint8", "int16", "uint16", "int32",
                            "uint32", "int64", "uint64", "float16", "float32"],
                           param_name="input_x")

    para_check.check_shape(y_shape, param_name="output_y")
    para_check.check_dtype(y_dtype,
                           ["int8", "uint8", "int16", "uint16", "int32",
                            "uint32", "int64", "uint64", "float16", "float32"],
                           param_name="output_y")

    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("shuffle_channel", "x", "y", x_dtype, y_dtype)

    if len(x_shape) > 4 or len(x_shape) < 2:
        error_manager_vector.raise_err_input_param_range_invalid("shuffle_channel", "x", "3", "4", str(len(x_shape)))

    if len(x_shape) == 3:
        x_shape = (x_shape[0], x_shape[1], x_shape[2], 1)
    if len(x_shape) == 2:
        x_shape = (x_shape[0], x_shape[1], 1, 1)
    input_dict["x"]["shape"] = x_shape

    if len(y_shape) > 4 or len(y_shape) < 2:
        error_manager_vector.raise_err_input_param_range_invalid("shuffle_channel", "y", "3", "4", str(len(y_shape)))

    if len(y_shape) == 3:
        y_shape = (y_shape[0], y_shape[1], y_shape[2], 1)
    if len(y_shape) == 2:
        y_shape = (y_shape[0], y_shape[1], 1, 1)
    input_dict["y"]["shape"] = y_shape

    if not _check_same_dim(y_shape, x_shape):
        error_manager_vector.raise_err_inputs_shape_not_equal("shuffle_channel", "x", "y",
                                                              str(x_shape), str(y_shape), str(y_shape))

    group = input_dict.get("group")
    if group <= 0:
        error_manager_vector.raise_err_input_param_range_invalid("shuffle_channel", "group",
                                                                 "1", "inf", str(group))

    channel = x_shape[1]
    if channel % group != 0:
        error_manager_vector.raise_err_specific_reson("shuffle_channel", "the channel of input[x] \
                                                      should be divisible by the parameter[group], \
                                                      but actually they are [{}] and [{}].".format(
                                                      str(channel), str(group)))


def _get_shape_total_number(shape):
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


def _check_same_dim(shape_x, shape_y):
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
    for k in range(shape_x_len):
        if shape_x[k] != shape_y[k]:
            return False

    return True
