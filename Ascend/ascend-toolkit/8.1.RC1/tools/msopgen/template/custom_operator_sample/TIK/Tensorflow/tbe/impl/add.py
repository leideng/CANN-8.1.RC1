#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

add
"""
import math
from functools import reduce as functools_reduce
import numpy as np

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check


def get_bit_len(dtype):
    if dtype == 'bool':
        return 16
    if dtype == 'double':
        return 64
    if dtype not in ('bool', 'double'):
        index = 0
        for i in dtype:
            if i.isdigit():
                break
            index += 1
        return int(dtype[index:])
    return None


class Vadd():
    def __init__(self, input_x, input_y, kernel_name="vadd_sample"):
        self.shape_x = input_x.get("shape")
        self.dtype_x = input_x.get("dtype")
        self.shape_y = input_y.get("shape")
        self.dtype_y = input_y.get("dtype")
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        # 获取芯片核数
        tik_dprofile = tik.Dprofile()
        self.aicore_num = tik_dprofile.get_aicore_num()

        # Unified Buffer上数据读取和写入必须32B对齐，此参数用来计算tensor划分和数据搬运指令参数
        block_byte_size = 32
        # 获取Unified Buffer空间大小，单位为bytes
        ub_size_bytes = tbe_platform.get_soc_spec("UB_SIZE")

        # 根据输入的数据类型计算一个block可以存放多少个对应的元素
        dtype_bytes_size = get_bit_len(self.dtype_x) // 8
        self.data_each_block = block_byte_size // dtype_bytes_size

        # 计算在Unified Buffer上给两个输入和计算结果分别分配多少空间（地址重叠），并进行32B对齐
        self.ub_tensor_size = (
                ub_size_bytes // dtype_bytes_size // 2 // self.data_each_block *
                self.data_each_block)

        # 计算输入的元素个数
        self.input_num = functools_reduce(lambda x, y: x * y, self.shape_x)

        # 计算每个aicore需要处理的数据量，当前只考虑均分场景，且均分后32 bytes对齐
        self.data_num_each_core = self.input_num // self.aicore_num

        # vector指令每个repeat最多计算8个block，该参数为mask的最大值
        self.vector_mask_max = 8 * self.data_each_block

        self.input_x_gm = self.tik_instance.Tensor(
            self.dtype_x, self.shape_x, name="input_x_gm", scope=tik.scope_gm)
        self.input_y_gm = self.tik_instance.Tensor(
            self.dtype_x, self.shape_x, name="input_y_gm", scope=tik.scope_gm)
        self.output_z_gm = self.tik_instance.Tensor(
            self.dtype_x, self.shape_x, name="output_z_gm", scope=tik.scope_gm)

    def vadd_compute(self):
        with self.tik_instance.for_range(
                0, self.aicore_num, block_num=self.aicore_num) as index:
            # 创建两个输入在Unified Buffer上的tensor
            self.input_x_ub = self.tik_instance.Tensor(
                self.dtype_x, (self.ub_tensor_size,),
                name="input_x_ub",
                scope=tik.scope_ubuf)
            self.input_y_ub = self.tik_instance.Tensor(
                self.dtype_y, (self.ub_tensor_size,),
                name="input_y_ub",
                scope=tik.scope_ubuf)

            # 将对应的GM上的数据搬运到Unified Buffer，每次搬运的偏移量为已经处理过的数据个数
            move_offset = index * self.data_num_each_core

            # 每个aicore计算自己负责的数据分片
            self.vadd_compute_each_core(move_offset, self.data_num_each_core)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_x_gm, self.input_y_gm],
            outputs=[self.output_z_gm])

        return self.tik_instance

    def vadd_compute_each_core(self, move_offset, move_num):
        loop_time = move_num // self.ub_tensor_size
        move_offset_init = move_offset
        if loop_time > 0:
            with self.tik_instance.for_range(0, loop_time) as loop_index:
                move_offset += loop_index * self.ub_tensor_size
                self.vadd_compute_each_loop(move_offset, self.ub_tensor_size)
            move_offset = move_offset_init + loop_time * self.ub_tensor_size

        last_num = move_num % self.ub_tensor_size
        if last_num > 0:
            self.vadd_compute_each_loop(move_offset, last_num)

    def vadd_compute_each_loop(self, move_offset, move_num):
        # 计算每次搬运的burst_len
        burst_len = math.ceil(move_num / self.data_each_block)

        self.tik_instance.data_move(self.input_x_ub,
                                    self.input_x_gm[move_offset], 0, 1,
                                    burst_len, 0, 0)
        self.tik_instance.data_move(self.input_y_ub,
                                    self.input_y_gm[move_offset], 0, 1,
                                    burst_len, 0, 0)
        vadd_loop = move_num // (self.vector_mask_max * 255)
        add_offset = 0
        if vadd_loop > 0:
            with self.tik_instance.for_range(0, vadd_loop) as add_index:
                add_offset = add_index * self.vector_mask_max * 255
                self.tik_instance.vec_add(self.vector_mask_max,
                                          self.input_x_ub[add_offset],
                                          self.input_x_ub[add_offset],
                                          self.input_y_ub[add_offset],
                                          255, 8, 8, 8)
            add_offset = vadd_loop * self.vector_mask_max * 255
        repeat_time = (
                move_num % (self.vector_mask_max * 255) // self.vector_mask_max)
        if repeat_time > 0:
            self.tik_instance.vec_add(self.vector_mask_max,
                                      self.input_x_ub[add_offset],
                                      self.input_x_ub[add_offset],
                                      self.input_y_ub[add_offset],
                                      repeat_time, 8, 8, 8)
            add_offset += repeat_time * self.vector_mask_max
        last_num = move_num % self.vector_mask_max
        if last_num > 0:
            self.tik_instance.vec_add(last_num,
                                      self.input_x_ub[add_offset],
                                      self.input_x_ub[add_offset],
                                      self.input_y_ub[add_offset],
                                      1, 8, 8, 8)

        self.tik_instance.data_move(self.output_z_gm[move_offset],
                                    self.input_x_ub, 0, 1, burst_len, 0, 0)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def add(input_x, input_y, output_z, kernel_name="add"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    input_y : dict
        shape and dtype of input
    output_z : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "add"

    Returns
    -------
    None
    """
    vadd_instance = Vadd(input_x, input_y, kernel_name)
    tik_instance = vadd_instance.vadd_compute()

    return tik_instance


"""
***************************test for debug**************************************
注意：以下main方法主要用于调试tik_debug使用，调试时请注释掉@para_check.check_op_params装饰器
"""
if __name__=="__main__":
    from tbe.common.platform import set_current_compile_soc_info

    set_current_compile_soc_info("Ascend310")

    add_shape = [64, 64]
    add_type = "float32"
    input_x = {"shape": add_shape, "dtype": add_type}
    input_y = {"shape": add_shape, "dtype": add_type}
    output_z = {"shape": add_shape, "dtype": add_type}

    tik_instance = add(input_x, input_y, output_z, kernel_name="add")

    data_x = np.ones(add_shape).astype(add_type)
    data_y = np.ones(add_shape).astype(add_type)

    feed_dict = {"input_x_gm": data_x,
                 "input_y_gm": data_y}

    out_put = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=True)
    deformable_actual = np.array(out_put, dtype=np.float32)
