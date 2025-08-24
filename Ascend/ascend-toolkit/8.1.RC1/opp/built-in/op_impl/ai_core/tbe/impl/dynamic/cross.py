#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

cross
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import para_check


# 'pylint: disable=invalid-name
def check_supported(x1, x2, y, dim=-65530, kernel_name="cross"):
    dtype = x1.get("dtype")
    if dtype == "float32":
        return False, "not support"
    return True, ""


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    Constant
    """
    BLOCK = 8
    MAX_INT32 = 2 ** 31 - 1


class Cross():
    """
    Function: Compute the cross product of two tensor
    Create: 2020-07-10
    Modify: 2022-11-10
    """

    def __init__(self, x1, x2, dim, kernel_name="cross"):

        self.tik_instance = tik.Tik()
        # Reads the data type of the incoming parameter
        self.dtype_x = x1.get("dtype")
        self.dtype_y = x2.get("dtype")
        self.kernel_name = kernel_name
        self.interval_num = self.tik_instance.Scalar(
            "int32", name="interval_num")
        self.loop_times = self.tik_instance.Scalar("int32", name="loop_times")
        self.tiling_gm = self.tik_instance.Tensor("int32", [Constant.BLOCK], name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.get_tiling_args()

        # Constant for data storage
        self.core_num = 1
        block_bite_size = 32
        ub_size_bytes = 256
        self.dtype_bytes_size = tbe_platform.get_bit_len(self.dtype_x) // 8
        self.data_each_block = block_bite_size // self.dtype_bytes_size
        self.ub_tensor_mask = (
            ub_size_bytes // self.dtype_bytes_size // self.data_each_block * self.data_each_block)
        self.vector_mask_max = 8 * self.data_each_block

        # The input and output tensor in Global Memory
        self.x1_gm = self.tik_instance.Tensor(self.dtype_x,
                                              [Constant.MAX_INT32],
                                              name="x1_gm",
                                              scope=tik.scope_gm)
        self.x2_gm = self.tik_instance.Tensor(self.dtype_y,
                                              [Constant.MAX_INT32],
                                              name="x2_gm",
                                              scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.dtype_x,
                                             [Constant.MAX_INT32],
                                             name="y_gm",
                                             scope=tik.scope_gm)

        # as data_saver, do page_down
        self.x1_ub = self.tik_instance.Tensor(self.dtype_x,
                                              (self.vector_mask_max, ), name="x1_ub", scope=tik.scope_ubuf)
        self.x2_ub = self.tik_instance.Tensor(self.dtype_x,
                                              (self.vector_mask_max, ), name="x2_ub", scope=tik.scope_ubuf)
        self.temp_ub = self.tik_instance.Tensor(self.dtype_x,
                                                (self.vector_mask_max, ), name="temp_ub", scope=tik.scope_ubuf)
        # as data_saver's index
        self.ub_lower_index_1 = self.tik_instance.Scalar(
            dtype="int32", init_value=- self.vector_mask_max - 1)
        self.ub_lower_index_2 = self.tik_instance.Scalar(
            dtype="int32", init_value=- self.vector_mask_max - 1)

        # as data_record, do page_update
        self.y_ub = self.tik_instance.Tensor(self.dtype_x,
                                             (self.vector_mask_max, ), name="y_ub", scope=tik.scope_ubuf)
        # init data_record
        self.gm_lower_index = self.tik_instance.Scalar(
            dtype="int32", init_value=0)

        # compute save tensor
        self.left_i_scalar = self.tik_instance.Scalar(
            self.dtype_x, init_value=0)

        self.left_j_scalar = self.tik_instance.Scalar(
            self.dtype_x, init_value=0)

        self.left_k_scalar = self.tik_instance.Scalar(
            self.dtype_x, init_value=0)

        self.right_i_scalar = self.tik_instance.Scalar(
            self.dtype_x, init_value=0)

        self.right_j_scalar = self.tik_instance.Scalar(
            self.dtype_x, init_value=0)

        self.right_k_scalar = self.tik_instance.Scalar(
            self.dtype_x, init_value=0)

        self.output_i_scalar = self.tik_instance.Scalar(
            self.dtype_x, init_value=0)

        self.output_j_scalar = self.tik_instance.Scalar(
            self.dtype_x, init_value=0)

        self.output_k_scalar = self.tik_instance.Scalar(
            self.dtype_x, init_value=0)

    def get_tiling_args(self):
        """get_tiling_args"""
        tiling_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK],
                                             name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
        self.interval_num.set_as(tiling_ub[0])
        self.loop_times.set_as(tiling_ub[1])

    def cross_compute(self):
        """
        cross_compute
        """
        with self.tik_instance.for_range(0, self.loop_times) as i:
            with self.tik_instance.for_range(0, self.interval_num) as j:
                i_index = self.tik_instance.Scalar(dtype="int32",
                                                   init_value=i * 3 * self.interval_num + j)
                j_index = self.tik_instance.Scalar(dtype="int32",
                                                   init_value=i * 3 * self.interval_num + j + self.interval_num)
                k_index = self.tik_instance.Scalar(dtype="int32",
                                                   init_value=i * 3 * self.interval_num + j + 2 * self.interval_num)
                # data move in
                self.set_data1ub(self.left_i_scalar, i_index)
                self.set_data2ub(self.right_i_scalar, i_index)
                self.set_data1ub(self.left_j_scalar, j_index)
                self.set_data2ub(self.right_j_scalar, j_index)
                self.set_data1ub(self.left_k_scalar, k_index)
                self.set_data2ub(self.right_k_scalar, k_index)

                # compute
                if (self.dtype_x == "float16"):
                    self.compute_each_three_num_fp16()
                else:
                    self.compute_each_three_num_other()

                # data move out
                self.set_data2gm(self.output_i_scalar, i_index)
                self.set_data2gm(self.output_j_scalar, j_index)
                self.set_data2gm(self.output_k_scalar, k_index)

        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.core_num
            })
        self.tik_instance.BuildCCE(
            inputs=[self.x1_gm, self.x2_gm],
            outputs=[self.y_gm],
            kernel_name=self.kernel_name,
            flowtable=[self.tiling_gm])
        return self.tik_instance

    def compute_each_three_num_fp16(self):
        """
        fp16: use saver tensor compute only one tuple answer
        """
        left_1 = self.tik_instance.Tensor(
            "float32", (1, ), name="left_1", scope=tik.scope_ubuf)
        left_2 = self.tik_instance.Tensor(
            "float32", (1, ), name="left_2", scope=tik.scope_ubuf)
        left_3 = self.tik_instance.Tensor(
            "float32", (1, ), name="left_3", scope=tik.scope_ubuf)
        right_1 = self.tik_instance.Tensor(
            "float32", (1, ), name="right_1", scope=tik.scope_ubuf)
        right_2 = self.tik_instance.Tensor(
            "float32", (1, ), name="right_2", scope=tik.scope_ubuf)
        right_3 = self.tik_instance.Tensor(
            "float32", (1, ), name="right_3", scope=tik.scope_ubuf)

        left_11 = self.tik_instance.Scalar("float32", init_value=0)
        left_12 = self.tik_instance.Scalar("float32", init_value=0)
        left_13 = self.tik_instance.Scalar("float32", init_value=0)
        right_21 = self.tik_instance.Scalar("float32", init_value=0)
        right_22 = self.tik_instance.Scalar("float32", init_value=0)
        right_23 = self.tik_instance.Scalar("float32", init_value=0)

        self.tik_instance.scalar_conv("", left_11, self.left_i_scalar)
        self.tik_instance.scalar_conv("", left_12, self.left_j_scalar)
        self.tik_instance.scalar_conv("", left_13, self.left_k_scalar)
        self.tik_instance.scalar_conv("", right_21, self.right_i_scalar)
        self.tik_instance.scalar_conv("", right_22, self.right_j_scalar)
        self.tik_instance.scalar_conv("", right_23, self.right_k_scalar)

        left_1[0] = left_11
        left_2[0] = left_12
        left_3[0] = left_13
        right_1[0] = right_21
        right_2[0] = right_22
        right_3[0] = right_23

        self.compute_tensor_one(
            left_2, right_3, right_2, left_3, self.output_i_scalar)
        self.compute_tensor_one(
            right_1, left_3, left_1, right_3, self.output_j_scalar)
        self.compute_tensor_one(
            left_1, right_2, right_1, left_2, self.output_k_scalar)

    # 'pylint: disable=too-many-arguments
    def compute_tensor_one(self, left_font, left_back, right_font, right_back, dst):
        compute_output = self.tik_instance.Tensor(
            "float32", (1, ), name="compute_output", scope=tik.scope_ubuf)
        output = self.tik_instance.Tensor(
            self.dtype_x, (1, ), name="output", scope=tik.scope_ubuf)
        output_left = self.tik_instance.Tensor(
            "float32", (1, ), name="output_left", scope=tik.scope_ubuf)
        output_right = self.tik_instance.Tensor(
            "float32", (1, ), name="output_right", scope=tik.scope_ubuf)

        self.tik_instance.vec_mul(
            1, output_left, left_font, left_back, 1, 8, 8, 8)
        self.tik_instance.vec_mul(
            1, output_right, right_font, right_back, 1, 8, 8, 8)
        self.tik_instance.vec_sub(
            1, compute_output, output_left, output_right, 1, 8, 8, 8)

        self.tik_instance.h_cast(output, compute_output, "")

        dst.set_as(output[0])

    def compute_each_three_num_other(self):
        """
        use saver scalar compute only one tuple answer
        """
        self.output_i_scalar.set_as(
            self.left_j_scalar * self.right_k_scalar - self.right_j_scalar * self.left_k_scalar)
        self.output_j_scalar.set_as(
            self.right_i_scalar * self.left_k_scalar - self.left_i_scalar * self.right_k_scalar)
        self.output_k_scalar.set_as(
            self.left_i_scalar * self.right_j_scalar - self.right_i_scalar * self.left_j_scalar)

    def set_data1ub(self, dst, index):
        """
        get data from GM to saver x1 scalar
        """
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_instance.data_move_pad(self.x1_ub[0], self.x1_gm[index], 1, self.dtype_bytes_size, 0, 0)
        else:
            self.tik_instance.data_move(self.x1_ub[0], self.x1_gm[index], 0, 1, 1, 0, 0)
        dst.set_as(self.x1_ub[0])

    def set_data2ub(self, dst, index):
        """
        get data from GM to saver x2 scalar
        """
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_instance.data_move_pad(self.x2_ub[0], self.x2_gm[index], 1, self.dtype_bytes_size, 0, 0)
        else:
            self.tik_instance.data_move(self.x2_ub[0], self.x2_gm[index], 0, 1, 1, 0, 0)
        dst.set_as(self.x2_ub[0])

    def set_data2gm(self, src, index):
        """
        set data to GM from saver tensor
        """
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.temp_ub[0].set_as(src)
            self.tik_instance.data_move_pad(self.y_gm[index], self.temp_ub[0], 1, self.dtype_bytes_size, 0, 0)
        else:
            self.tik_instance.data_move(self.temp_ub[0], self.y_gm[index], 0, 1, 1, 0, 0)
            self.temp_ub[0].set_as(src)
            self.tik_instance.data_move(self.y_gm[index], self.temp_ub[0], 0, 1, 1, 0, 0)


# 'pylint: disable=too-many-locals,invalid-name,unused-argument,too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def cross(x1, x2, y, dim=-65530, kernel_name="cross"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        first input shape and dtype of input
    x2 : dict
        second input shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    dim : int
        the dimension to take the cross-product in
    kernel_name : str
        kernel name, default value is "cross"

    Returns
    -------
    None
    """
    cross_instance = Cross(x1, x2, dim, kernel_name=kernel_name)
    tik_instance = cross_instance.cross_compute()
    return tik_instance
