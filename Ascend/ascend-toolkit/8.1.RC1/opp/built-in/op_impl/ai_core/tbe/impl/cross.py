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
cross
"""
import functools
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform as cce
from impl.util.platform_adapter import para_check


class Cross():
    """
    Function: Compute the cross product of two tensor
    Create: 2020-07-10
    Modify: 2021-03-30
    """
    def __init__(self, x1, x2, dim, kernel_name="cross"):

        self.tik_instance = tik.Tik()
        # Reads the shape and data type of the incoming parameter
        self.shape_x = x1.get("shape")
        self.dtype_x = x1.get("dtype")
        self.shape_y = x2.get("shape")
        self.dtype_y = x2.get("dtype")
        self.kernel_name = kernel_name

        # Constant for data storage
        block_bite_size = 32
        ub_size_bytes = 256

        # Processing dim yields the interval value
        self.dim = dim
        self.intervel_num = 1
        self.input_num = functools.reduce(lambda x, y: x * y, self.shape_x)
        if self.dim == -65530:
            for i in range(0, len(self.shape_x)):
                self.intervel_num = self.intervel_num * self.shape_x[i]
                if self.shape_x[i] == 3:
                    self.dim = i
                    break
        elif self.dim < -len(self.shape_x) or self.dim > len(self.shape_x) - 1:
            raise Exception("dimension out of range")
        elif self.shape_x[self.dim] != 3:
            raise Exception("dimension {} does not have size 3".format(self.dim))
        else:
            for i in range(0, self.dim + 1):
                self.intervel_num = self.intervel_num * self.shape_x[i]
        if self.intervel_num == self.input_num and self.shape_x[-1] != 3:
            raise Exception("cannot find dimension")
        self.intervel_num = self.input_num // self.intervel_num

        dtype_bytes_size = cce.get_bit_len(self.dtype_x) // 8
        self.data_each_block = block_bite_size // dtype_bytes_size
        self.ub_tensor_mask = (ub_size_bytes // dtype_bytes_size // self.data_each_block * self.data_each_block)
        self.vector_mask_max = 8 * self.data_each_block

        # The input and output tensor in Global Memory
        self.x1_gm = self.tik_instance.Tensor(self.dtype_x,
                                              self.shape_x,
                                              name="x1_gm",
                                              scope=tik.scope_gm)
        self.x2_gm = self.tik_instance.Tensor(self.dtype_y,
                                              self.shape_y,
                                              name="x2_gm",
                                              scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.dtype_x,
                                             self.shape_x,
                                             name="y_gm",
                                             scope=tik.scope_gm)

        # as data_saver, do page_down
        self.x1_ub = self.tik_instance.Tensor(self.dtype_x,
                                              (self.vector_mask_max, ), name="x1_ub", scope=tik.scope_ubuf)
        self.x2_ub = self.tik_instance.Tensor(self.dtype_x,
                                              (self.vector_mask_max, ), name="x2_ub", scope=tik.scope_ubuf)
        # as data_saver's index
        self.ub_lower_index = self.tik_instance.Scalar(dtype="int32", init_value=- self.vector_mask_max - 1)

        # as data_record, do page_update
        self.y_ub = self.tik_instance.Tensor(self.dtype_x,
                                             (self.vector_mask_max, ), name="y_ub", scope=tik.scope_ubuf)
        # init data_record
        self.gm_lower_index = self.tik_instance.Scalar(dtype="int32", init_value=0)
        self.tik_instance.data_move(self.y_ub[self.gm_lower_index], self.y_gm[self.gm_lower_index], 0, 1, 8, 0, 0)

        # out loop
        self.loop_times = self.input_num // self.intervel_num // 3

        # compute save tensor
        self.left_i_ub = self.tik_instance.Tensor(self.dtype_x,
                                                  (1, ), name="left_i_ub", scope=tik.scope_ubuf)

        self.left_j_ub = self.tik_instance.Tensor(self.dtype_x,
                                                  (1, ), name="left_j_ub", scope=tik.scope_ubuf)

        self.left_k_ub = self.tik_instance.Tensor(self.dtype_x,
                                                  (1, ), name="left_k_ub", scope=tik.scope_ubuf)

        self.right_i_ub = self.tik_instance.Tensor(self.dtype_x,
                                                   (1, ), name="right_i_ub", scope=tik.scope_ubuf)

        self.right_j_ub = self.tik_instance.Tensor(self.dtype_x,
                                                   (1, ), name="right_j_ub", scope=tik.scope_ubuf)

        self.right_k_ub = self.tik_instance.Tensor(self.dtype_x,
                                                   (1, ), name="right_k_ub", scope=tik.scope_ubuf)

        self.output_i_ub = self.tik_instance.Tensor(self.dtype_x,
                                                    (1, ), name="output_i_ub", scope=tik.scope_ubuf)

        self.output_j_ub = self.tik_instance.Tensor(self.dtype_x,
                                                    (1, ), name="output_j_ub", scope=tik.scope_ubuf)

        self.output_k_ub = self.tik_instance.Tensor(self.dtype_x,
                                                    (1, ), name="output_j_ub", scope=tik.scope_ubuf)

    def cross_compute(self):
        """
        cross_compute
        """
        with self.tik_instance.for_range(0, self.loop_times) as i:
            with self.tik_instance.for_range(0, self.intervel_num) as j:
                i_index = self.tik_instance.Scalar(dtype="int32",
                                                   init_value=i * 3 * self.intervel_num + j)
                j_index = self.tik_instance.Scalar(dtype="int32",
                                                   init_value=i * 3 * self.intervel_num + j + self.intervel_num)
                k_index = self.tik_instance.Scalar(dtype="int32",
                                                   init_value=i * 3 * self.intervel_num + j + 2 * self.intervel_num)
                # data move in
                self.set_data2ub(self.left_i_ub, self.x1_ub, i_index)
                self.set_data2ub(self.right_i_ub, self.x2_ub, i_index)
                self.set_data2ub(self.left_j_ub, self.x1_ub, j_index)
                self.set_data2ub(self.right_j_ub, self.x2_ub, j_index)
                self.set_data2ub(self.left_k_ub, self.x1_ub, k_index)
                self.set_data2ub(self.right_k_ub, self.x2_ub, k_index)
                # compute
                self.cross_compute_each_three_num()
                # data move out
                self.set_data2gm(self.output_i_ub, i_index)
                self.set_data2gm(self.output_j_ub, j_index)
                self.set_data2gm(self.output_k_ub, k_index)
        # last data move out
        self.tik_instance.data_move(self.y_gm[self.gm_lower_index], self.y_ub[0], 0, 1, 8, 0, 0)
        self.tik_instance.BuildCCE(
            inputs=[self.x1_gm, self.x2_gm],
            outputs=[self.y_gm],
            kernel_name=self.kernel_name)
        return self.tik_instance

    def cross_compute_each_three_num(self):
        """
        use saver tensor compute only one truple answer
        """
        output_i_left = self.tik_instance.Tensor(self.dtype_x, (1, ), name="output_i_left", scope=tik.scope_ubuf)
        output_i_right = self.tik_instance.Tensor(self.dtype_x, (1, ), name="output_i_right", scope=tik.scope_ubuf)

        self.tik_instance.vec_mul(1, output_i_left, self.left_j_ub, self.right_k_ub, 1, 8, 8, 8)
        self.tik_instance.vec_mul(1, output_i_right, self.right_j_ub, self.left_k_ub, 1, 8, 8, 8)
        self.tik_instance.vec_sub(1, self.output_i_ub, output_i_left, output_i_right, 1, 8, 8, 8)

        output_j_left = self.tik_instance.Tensor(self.dtype_x, (1, ), name="output_j_left", scope=tik.scope_ubuf)
        output_j_right = self.tik_instance.Tensor(self.dtype_x, (1, ), name="output_j_right", scope=tik.scope_ubuf)

        self.tik_instance.vec_mul(1, output_j_left, self.right_i_ub, self.left_k_ub, 1, 8, 8, 8)
        self.tik_instance.vec_mul(1, output_j_right, self.left_i_ub, self.right_k_ub, 1, 8, 8, 8)
        self.tik_instance.vec_sub(1, self.output_j_ub, output_j_left, output_j_right, 1, 8, 8, 8)

        output_k_left = self.tik_instance.Tensor(self.dtype_x, (1, ), name="output_k_left", scope=tik.scope_ubuf)
        output_k_right = self.tik_instance.Tensor(self.dtype_x, (1, ), name="output_k_right", scope=tik.scope_ubuf)

        self.tik_instance.vec_mul(1, output_k_left, self.left_i_ub, self.right_j_ub, 1, 8, 8, 8)
        self.tik_instance.vec_mul(1, output_k_right, self.right_i_ub, self.left_j_ub, 1, 8, 8, 8)
        self.tik_instance.vec_sub(1, self.output_k_ub, output_k_left, output_k_right, 1, 8, 8, 8)

    def set_data2ub(self, dst, src, index):
        """
        get data from GM to saver tensor
        """
        with self.tik_instance.if_scope(index < self.ub_lower_index):
            self.ub_lower_index.set_as(index / self.vector_mask_max * self.vector_mask_max)
            self.tik_instance.data_move(self.x1_ub[0], self.x1_gm[self.ub_lower_index],
                                        0, 1, 8, 0, 0)
            self.tik_instance.data_move(self.x2_ub[0], self.x2_gm[self.ub_lower_index],
                                        0, 1, 8, 0, 0)
            dst[0].set_as(src[index - self.ub_lower_index])
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(index < self.ub_lower_index + self.vector_mask_max):
                dst[0].set_as(src[index - self.ub_lower_index])
            with self.tik_instance.else_scope():
                self.ub_lower_index.set_as(index / self.vector_mask_max * self.vector_mask_max)
                self.tik_instance.data_move(self.x1_ub[0], self.x1_gm[self.ub_lower_index],
                                            0, 1, 8, 0, 0)
                self.tik_instance.data_move(self.x2_ub[0], self.x2_gm[self.ub_lower_index],
                                            0, 1, 8, 0, 0)
                dst[0].set_as(src[index - self.ub_lower_index])

    def set_data2gm(self, src, index):
        """
        set data to GM from saver tensor
        """
        with self.tik_instance.if_scope(index < self.gm_lower_index):
            self.tik_instance.data_move(self.y_gm[self.gm_lower_index], self.y_ub[0], 0, 1, 8, 0, 0)
            self.gm_lower_index.set_as(index / self.vector_mask_max * self.vector_mask_max)
            self.tik_instance.data_move(self.y_ub[0], self.y_gm[self.gm_lower_index], 0, 1, 8, 0, 0)
            self.y_ub[index - self.gm_lower_index].set_as(src)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(index < self.gm_lower_index + self.vector_mask_max):
                self.y_ub[index - self.gm_lower_index].set_as(src)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.y_gm[self.gm_lower_index], self.y_ub[0], 0, 1, 8, 0, 0)
                self.gm_lower_index.set_as(index / self.vector_mask_max * self.vector_mask_max)
                self.tik_instance.data_move(self.y_ub[0], self.y_gm[self.gm_lower_index], 0, 1, 8, 0, 0)
                self.y_ub[index - self.gm_lower_index].set_as(src)


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
