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
resize_d
"""

import math
from collections import namedtuple
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import platform as cce

from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len

# get available ub size
UB_SIZE = cce.get_soc_spec(cce.UB_SIZE)


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # input shape indices NCHW
    N_IDX = 0
    C_IDX = 1
    H_IDX = 2
    W_IDX = 3

    # size shape indices HW
    SIZES_H_IDX = 0
    SIZES_W_IDX = 1

    # constant parameters in calculation
    VECTOR_MASK_MAX = 64
    BLOCK_NUM_FP32 = 8
    STRIDE_FP16 = 4


class ResizeBicubic():
    """
    Function: use to store ResizeTrilinear base parameters
    """

    def __init__(self, x, sizes, scales, coordinate_transformation_mode, cubic_coeff_a, kernel_name="resize_d"):
        """init ResizeBicubic base parameters
        """
        self.tik_instance = tik.Tik(disable_debug=False)
        self.kernel_name = kernel_name
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        
        self.block_byte_size = 32

        # fomular related parameters
        self.coordinate_transformation_mode = coordinate_transformation_mode
        self.scalar_half = 0.5
        self.scalar_negative_half = -0.5
        self.scalar_one = 1
        self.scalar_negative_one = -1
        self.coeff = cubic_coeff_a
        self.coeff_plus_2 = cubic_coeff_a + 2
        self.coeff_plus_3 = cubic_coeff_a + 3
        self.double_coeff_plus_3 = cubic_coeff_a * 2 + 3

        # acquire input info
        self.x_shape = x.get("shape")
        self.x_dtype = x.get("dtype").lower()
        self.batch_size = self.x_shape[Constant.N_IDX]
        self.channel_size = self.x_shape[Constant.C_IDX]
        self.h_in_size = self.x_shape[Constant.H_IDX]
        self.w_in_size = self.x_shape[Constant.W_IDX]

        self.h_out_size = sizes[Constant.SIZES_H_IDX]
        self.w_out_size = sizes[Constant.SIZES_W_IDX]
        self.y_shape = [self.batch_size, self.channel_size, self.h_out_size, self.w_out_size]
        if self.coordinate_transformation_mode == "align_corners":
            self.h_scale = (self.h_in_size - 1) / (self.h_out_size - 1)
            self.w_scale = (self.w_in_size - 1) / (self.w_out_size - 1)
        else:
            self.h_scale = self.h_in_size / self.h_out_size
            self.w_scale = self.w_in_size / self.w_out_size

        # core parameters
        self.inner_dtype = "float32"
        self.nc_num = self.batch_size * self.channel_size
        self.need_core_num = self.batch_size * self.channel_size * self.h_out_size

        self.init_ub_tensor()

    @staticmethod
    def get_dtype_size(dtype):
        """
        :param dtype: data type
        :return:
        """
        dtype_byte_size = get_bit_len(dtype) // 8
        return dtype_byte_size

    def init_ub_tensor(self):
        """
        init tensor in ub
        """

        block_bite_size = 32
        dtype_bytes_size = cce.get_bit_len(self.x_dtype) // 8
        self.data_each_block = block_bite_size // dtype_bytes_size

        self.h_ub_out_size = max(self.data_each_block, self.h_out_size)
        self.w_ub_out_size = max(self.data_each_block, self.w_out_size)

        max_dim_value = list(range(512))
        zero_value = list(0 for i in range(512))
        self.images_gm = self.tik_instance.Tensor(self.x_dtype, self.x_shape,
                                                   name="images_gm", scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.x_dtype, self.y_shape,
                                                   name="out_gm", scope=tik.scope_gm)

        self.dst_idx_gm = self.tik_instance.Tensor(self.inner_dtype, [512],
                                                   name="dst_idx_gm", scope=tik.scope_gm, init_value=max_dim_value)
        self.zero_value_gm = self.tik_instance.Tensor(self.inner_dtype, [512],
                                                   name="zero_value_gm", scope=tik.scope_gm, init_value=zero_value)

        self.index_h_mapping_ub = self.tik_instance.Tensor("int32", [self.h_ub_out_size],
                                                   name="index_h_mapping_ub", scope=tik.scope_ubuf)
        self.index_w_mapping_ub = self.tik_instance.Tensor("int32", [self.w_ub_out_size],
                                                   name="index_w_mapping_ub", scope=tik.scope_ubuf)
        
        self.src_line_1 = self.tik_instance.Tensor(self.inner_dtype, [self.w_in_size],
                                                   name="src_line_1", scope=tik.scope_ubuf)
        self.src_line_2 = self.tik_instance.Tensor(self.inner_dtype, [self.w_in_size],
                                                   name="src_line_2", scope=tik.scope_ubuf)
        self.src_line_3 = self.tik_instance.Tensor(self.inner_dtype, [self.w_in_size],
                                                   name="src_line_3", scope=tik.scope_ubuf)
        self.src_line_4 = self.tik_instance.Tensor(self.inner_dtype, [self.w_in_size],
                                                   name="src_line_4", scope=tik.scope_ubuf)
                
        self.h_diff = self.tik_instance.Tensor(self.inner_dtype, [self.h_ub_out_size],
                                                   name="h_diff", scope=tik.scope_ubuf)
        self.w_diff = self.tik_instance.Tensor(self.inner_dtype, [self.w_ub_out_size],
                                                   name="w_diff", scope=tik.scope_ubuf)
        self.h_weight_1 = self.tik_instance.Tensor(self.inner_dtype, [self.h_ub_out_size],
                                                   name="h_weight_1", scope=tik.scope_ubuf)
        self.h_weight_2 = self.tik_instance.Tensor(self.inner_dtype, [self.h_ub_out_size],
                                                   name="h_weight_2", scope=tik.scope_ubuf)
        self.h_weight_3 = self.tik_instance.Tensor(self.inner_dtype, [self.h_ub_out_size],
                                                   name="h_weight_3", scope=tik.scope_ubuf)
        self.h_weight_4 = self.tik_instance.Tensor(self.inner_dtype, [self.h_ub_out_size],
                                                   name="h_weight_4", scope=tik.scope_ubuf)
        self.w_weight_1 = self.tik_instance.Tensor(self.inner_dtype, [self.w_ub_out_size],
                                                   name="w_weight_1", scope=tik.scope_ubuf)
        self.w_weight_2 = self.tik_instance.Tensor(self.inner_dtype, [self.w_ub_out_size],
                                                   name="w_weight_2", scope=tik.scope_ubuf)
        self.w_weight_3 = self.tik_instance.Tensor(self.inner_dtype, [self.w_ub_out_size],
                                                   name="w_weight_3", scope=tik.scope_ubuf)
        self.w_weight_4 = self.tik_instance.Tensor(self.inner_dtype, [self.w_ub_out_size],
                                                   name="w_weight_4", scope=tik.scope_ubuf)
        self.h_ub_1 = self.tik_instance.Tensor(self.inner_dtype, [self.h_ub_out_size],
                                                   name="h_ub_1", scope=tik.scope_ubuf)
        self.h_ub_2 = self.tik_instance.Tensor(self.inner_dtype, [self.h_ub_out_size],
                                                   name="h_ub_2", scope=tik.scope_ubuf)
        self.h_ub_3 = self.tik_instance.Tensor(self.inner_dtype, [self.h_ub_out_size],
                                                   name="h_ub_3", scope=tik.scope_ubuf)
        self.w_ub_1 = self.tik_instance.Tensor(self.inner_dtype, [self.w_ub_out_size],
                                                   name="w_ub_1", scope=tik.scope_ubuf)
        self.w_ub_2 = self.tik_instance.Tensor(self.inner_dtype, [self.w_ub_out_size],
                                                   name="w_ub_2", scope=tik.scope_ubuf)
        self.w_ub_3 = self.tik_instance.Tensor(self.inner_dtype, [self.w_ub_out_size],
                                                   name="w_ub_3", scope=tik.scope_ubuf)
        self.w_ub_4 = self.tik_instance.Tensor(self.inner_dtype, [self.w_ub_out_size],
                                                   name="w_ub_4", scope=tik.scope_ubuf)

        self.data_move(self.index_h_mapping_ub, self.dst_idx_gm, [0, 0], self.h_out_size)
        self.data_move(self.index_w_mapping_ub, self.dst_idx_gm, [0, 0], self.w_out_size)
        self.data_move(self.h_weight_1, self.zero_value_gm, [0, 0], self.h_out_size)
        self.data_move(self.h_weight_2, self.zero_value_gm, [0, 0], self.h_out_size)
        self.data_move(self.h_weight_3, self.zero_value_gm, [0, 0], self.h_out_size)
        self.data_move(self.h_weight_4, self.zero_value_gm, [0, 0], self.h_out_size)
        self.data_move(self.w_weight_1, self.zero_value_gm, [0, 0], self.w_out_size)
        self.data_move(self.w_weight_2, self.zero_value_gm, [0, 0], self.w_out_size)
        self.data_move(self.w_weight_3, self.zero_value_gm, [0, 0], self.w_out_size)
        self.data_move(self.w_weight_4, self.zero_value_gm, [0, 0], self.w_out_size)
        self.data_move(self.h_ub_1, self.dst_idx_gm, [0, 0], self.h_out_size)
        self.data_move(self.w_ub_1, self.dst_idx_gm, [0, 0], self.w_out_size)
        

        # calc diff
        if self.coordinate_transformation_mode == "align_corners":
            self.scalar_operator_template(self.tik_instance.vec_muls,
                                          self.h_ub_2, self.h_ub_1, self.h_scale, [0, 0], self.h_out_size)
            self.scalar_operator_template(self.tik_instance.vec_muls,
                                          self.w_ub_2, self.w_ub_1, self.w_scale, [0, 0], self.w_out_size)
        else:
            self.scalar_operator_template(self.tik_instance.vec_adds,
                                          self.h_ub_1, self.h_ub_1, self.scalar_half, [0, 0], self.h_out_size)
            self.scalar_operator_template(self.tik_instance.vec_muls,
                                          self.h_ub_2, self.h_ub_1, self.h_scale, [0, 0], self.h_out_size)
            self.scalar_operator_template(self.tik_instance.vec_adds,
                                          self.h_ub_2, self.h_ub_2, self.scalar_negative_half, [0, 0], self.h_out_size)
            self.scalar_operator_template(self.tik_instance.vec_adds,
                                          self.w_ub_1, self.w_ub_1, self.scalar_half, [0, 0], self.w_out_size)
            self.scalar_operator_template(self.tik_instance.vec_muls,
                                          self.w_ub_2, self.w_ub_1, self.w_scale, [0, 0], self.w_out_size)
            self.scalar_operator_template(self.tik_instance.vec_adds,
                                          self.w_ub_2, self.w_ub_2, self.scalar_negative_half, [0, 0], self.w_out_size)

        self.conv_operator_template(self.tik_instance.vec_conv,
                                    "floor", self.index_h_mapping_ub, self.h_ub_2, [0, 0], self.h_out_size)
        self.conv_operator_template(self.tik_instance.vec_conv,
                                    "none", self.h_ub_1, self.index_h_mapping_ub, [0, 0], self.h_out_size)
        self.double_operator_template(self.tik_instance.vec_sub,
                                      self.h_diff, self.h_ub_2, self.h_ub_1, [0, 0, 0], self.h_out_size)
        self.calc_weights(self.h_diff, self.h_weight_1, self.h_weight_2, self.h_weight_3, self.h_weight_4,
                          self.h_ub_1, self.h_ub_2, self.h_ub_3, self.h_out_size)

        self.conv_operator_template(self.tik_instance.vec_conv,
                                    "floor", self.index_w_mapping_ub, self.w_ub_2, [0, 0], self.w_out_size)
        self.conv_operator_template(self.tik_instance.vec_conv,
                                    "none", self.w_ub_1, self.index_w_mapping_ub, [0, 0], self.w_out_size)
        self.double_operator_template(self.tik_instance.vec_sub,
                                      self.w_diff, self.w_ub_2, self.w_ub_1, [0, 0, 0], self.w_out_size)
        self.calc_weights(self.w_diff, self.w_weight_1, self.w_weight_2, self.w_weight_3, self.w_weight_4,
                          self.w_ub_1, self.w_ub_2, self.w_ub_3, self.w_out_size)

    def calc_weights(self, diff, weight_1, weight_2, weight_3, weight_4, ub_1, ub_2, ub_3, weight_size):
        """
        calculate weights of four point by line
        W1 = Ar(r-1)^2
        W2 = (A+2)r^3 - (A+3)r^2 + 1
        W3 = -(A+2)r^3 + (2A+3)r^2 -Ar
        W4 = -Ar(r-1)^2
        :para diff : fractional part of (dst_index * scale)
        :para ub_1 : help buff to save middle value
        :para ub_2 : help buff to save middle value
        :para ub_3 : help buff to save middle value
        :para weight_size : lenght of weight vector
        return:
        :para weight_1 : weight of ll points
        :para weight_2 : weight of left points
        :para weight_3 : weight of right points
        :para weight_4 : weight of rr points
        """

        self.scalar_operator_template(self.tik_instance.vec_adds,
                                      ub_1, diff, self.scalar_negative_one, [0, 0], weight_size)

        # W1
        self.double_operator_template(self.tik_instance.vec_add, weight_1, weight_1, diff, [0, 0, 0], weight_size)
        self.double_operator_template(self.tik_instance.vec_mul, weight_1, weight_1, ub_1, [0, 0, 0], weight_size)
        self.double_operator_template(self.tik_instance.vec_mul, weight_1, weight_1, ub_1, [0, 0, 0], weight_size)
        self.scalar_operator_template(self.tik_instance.vec_muls, weight_1, weight_1, self.coeff, [0, 0], weight_size)

        # W4
        self.double_operator_template(self.tik_instance.vec_sub, weight_4, weight_4, diff, [0, 0, 0], weight_size)
        self.double_operator_template(self.tik_instance.vec_mul, weight_4, weight_4, diff, [0, 0, 0], weight_size)
        self.double_operator_template(self.tik_instance.vec_mul, weight_4, weight_4, ub_1, [0, 0, 0], weight_size)
        self.scalar_operator_template(self.tik_instance.vec_muls, weight_4, weight_4, self.coeff, [0, 0], weight_size)

        # W2
        self.double_operator_template(self.tik_instance.vec_mul, ub_1, diff, diff, [0, 0, 0], weight_size)
        self.scalar_operator_template(self.tik_instance.vec_muls, ub_2, ub_1, self.coeff_plus_3, [0, 0], weight_size)
        self.double_operator_template(self.tik_instance.vec_mul, ub_1, ub_1, diff, [0, 0, 0], weight_size)
        self.scalar_operator_template(self.tik_instance.vec_muls, ub_3, ub_1, self.coeff_plus_2, [0, 0], weight_size)

        self.double_operator_template(self.tik_instance.vec_add, weight_2, weight_2, ub_3, [0, 0, 0], weight_size)
        self.double_operator_template(self.tik_instance.vec_sub, weight_2, weight_2, ub_2, [0, 0, 0], weight_size)
        self.scalar_operator_template(self.tik_instance.vec_adds,
                                      weight_2, weight_2, self.scalar_one, [0, 0], weight_size)

        # W3
        self.double_operator_template(self.tik_instance.vec_mul, ub_1, diff, diff, [0, 0, 0], weight_size)
        self.scalar_operator_template(self.tik_instance.vec_muls,
                                      ub_2, ub_1, self.double_coeff_plus_3, [0, 0], weight_size)
        self.double_operator_template(self.tik_instance.vec_mul, ub_1, ub_1, diff, [0, 0, 0], weight_size)
        self.scalar_operator_template(self.tik_instance.vec_muls, ub_3, ub_1, self.coeff_plus_2, [0, 0], weight_size)

        self.double_operator_template(self.tik_instance.vec_sub, weight_3, weight_3, diff, [0, 0, 0], weight_size)
        self.scalar_operator_template(self.tik_instance.vec_muls, weight_3, weight_3, self.coeff, [0, 0], weight_size)
        self.double_operator_template(self.tik_instance.vec_add, weight_3, weight_3, ub_2, [0, 0, 0], weight_size)
        self.double_operator_template(self.tik_instance.vec_sub, weight_3, weight_3, ub_3, [0, 0, 0], weight_size)

    def resize_bicubic_compute(self):
        """
        op compute
        """
        with self.tik_instance.for_range(0, self.need_core_num, block_num=self.need_core_num) as core_idx:
            self.compute_core(core_idx)

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.images_gm],
                                   outputs=[self.out_gm])
        return self.tik_instance

    def compute_core(self, core_idx):
        h_in_index = self.tik_instance.Scalar("int32")
        h_out_index = self.tik_instance.Scalar("int32")
        temp_h_weight = self.tik_instance.Scalar("float32")
        h_out_index.set_as(core_idx % self.h_out_size)
        h_in_index.set_as(self.index_h_mapping_ub[h_out_index])

        # init ub tensor src_line_1~4
        self.get_src_lines(core_idx, h_in_index)

        # sum up
        temp_h_weight.set_as(self.h_weight_1[h_out_index])
        self.scalar_operator_template(self.tik_instance.vec_muls,
                                      self.src_line_1, self.src_line_1, temp_h_weight, [0, 0], self.w_in_size)
        temp_h_weight.set_as(self.h_weight_2[h_out_index])
        self.scalar_operator_template(self.tik_instance.vec_muls,
                                      self.src_line_2, self.src_line_2, temp_h_weight, [0, 0], self.w_in_size)
        temp_h_weight.set_as(self.h_weight_3[h_out_index])
        self.scalar_operator_template(self.tik_instance.vec_muls,
                                      self.src_line_3, self.src_line_3, temp_h_weight, [0, 0], self.w_in_size)
        temp_h_weight.set_as(self.h_weight_4[h_out_index])
        self.scalar_operator_template(self.tik_instance.vec_muls,
                                      self.src_line_4, self.src_line_4, temp_h_weight, [0, 0], self.w_in_size)
        self.double_operator_template(self.tik_instance.vec_add,
                                      self.src_line_1, self.src_line_2, self.src_line_1, [0, 0, 0], self.w_in_size)
        self.double_operator_template(self.tik_instance.vec_add,
                                      self.src_line_2, self.src_line_4, self.src_line_3, [0, 0, 0], self.w_in_size)
        self.double_operator_template(self.tik_instance.vec_add,
                                      self.src_line_1, self.src_line_2, self.src_line_1, [0, 0, 0], self.w_in_size)

        # init ub tensor w_ub_1~4
        self.calc_oper_lines()
        self.double_operator_template(self.tik_instance.vec_mul,
                                      self.w_ub_1, self.w_ub_1, self.w_weight_1, [0, 0, 0], self.w_out_size)
        self.double_operator_template(self.tik_instance.vec_mul,
                                      self.w_ub_2, self.w_ub_2, self.w_weight_2, [0, 0, 0], self.w_out_size)
        self.double_operator_template(self.tik_instance.vec_mul,
                                      self.w_ub_3, self.w_ub_3, self.w_weight_3, [0, 0, 0], self.w_out_size)
        self.double_operator_template(self.tik_instance.vec_mul,
                                      self.w_ub_4, self.w_ub_4, self.w_weight_4, [0, 0, 0], self.w_out_size)
        self.double_operator_template(self.tik_instance.vec_add,
                                      self.w_ub_1, self.w_ub_2, self.w_ub_1, [0, 0, 0], self.w_out_size)
        self.double_operator_template(self.tik_instance.vec_add,
                                      self.w_ub_2, self.w_ub_4, self.w_ub_3, [0, 0, 0], self.w_out_size)
        self.double_operator_template(self.tik_instance.vec_add,
                                      self.w_ub_1, self.w_ub_2, self.w_ub_1, [0, 0, 0], self.w_out_size)

        if self.x_dtype == "float16":
            ub_cast_line = self.tik_instance.Tensor(self.x_dtype, [self.w_ub_out_size],
                                                    name="ub_cast_line", scope=tik.scope_ubuf)
            self.conv_operator_template(self.tik_instance.vec_conv,
                                        "none", ub_cast_line, self.w_ub_1, [0, 0], self.w_out_size, 8, 4)
            self.move_to_gm(self.out_gm, ub_cast_line, [core_idx * self.w_out_size, 0], self.w_out_size)
        else:
            self.move_to_gm(self.out_gm, self.w_ub_1, [core_idx * self.w_out_size, 0], self.w_out_size)

    def get_src_lines(self, core_idx, h_in_index):
        nc_idx = core_idx // self.h_out_size
        index_1 = self.tik_instance.Scalar("int64")
        index_2 = self.tik_instance.Scalar("int64")
        index_3 = self.tik_instance.Scalar("int64")
        index_4 = self.tik_instance.Scalar("int64")
        h_max_index = self.h_in_size - 1

        index_2.set_as(h_in_index)
        index_1.set_as(index_2 - 1)
        self.tik_instance.scalar_max(index_1, 0, index_1)
        index_3.set_as(index_2 + 1)
        self.tik_instance.scalar_min(index_3, h_max_index, index_3)
        index_4.set_as(index_2 + 2)
        self.tik_instance.scalar_min(index_4, h_max_index, index_4)
        self.tik_instance.scalar_max(index_2, 0, index_2)

        if self.x_dtype == "float16":
            ub_cast_line = self.tik_instance.Tensor(self.x_dtype, [self.w_in_size],
                                                    name="ub_cast_line", scope=tik.scope_ubuf)
            self.data_move(ub_cast_line, self.images_gm,
                           [0, (nc_idx * self.h_in_size + index_1) * self.w_in_size], self.w_in_size)
            self.conv_operator_template(self.tik_instance.vec_conv,
                                        "none", self.src_line_1, ub_cast_line, [0, 0], self.w_in_size, 8, 4)
            self.data_move(ub_cast_line, self.images_gm,
                           [0, (nc_idx * self.h_in_size + index_2) * self.w_in_size], self.w_in_size)
            self.conv_operator_template(self.tik_instance.vec_conv,
                                        "none", self.src_line_2, ub_cast_line, [0, 0], self.w_in_size, 8, 4)
            self.data_move(ub_cast_line, self.images_gm,
                           [0, (nc_idx * self.h_in_size + index_3) * self.w_in_size], self.w_in_size)
            self.conv_operator_template(self.tik_instance.vec_conv,
                                        "none", self.src_line_3, ub_cast_line, [0, 0], self.w_in_size, 8, 4)
            self.data_move(ub_cast_line, self.images_gm,
                           [0, (nc_idx * self.h_in_size + index_4) * self.w_in_size], self.w_in_size)
            self.conv_operator_template(self.tik_instance.vec_conv,
                                        "none", self.src_line_4, ub_cast_line, [0, 0], self.w_in_size, 8, 4)
        else:
            self.data_move(self.src_line_1, self.images_gm,
                           [0, (nc_idx * self.h_in_size + index_1) * self.w_in_size], self.w_in_size)
            self.data_move(self.src_line_2, self.images_gm,
                           [0, (nc_idx * self.h_in_size + index_2) * self.w_in_size], self.w_in_size)
            self.data_move(self.src_line_3, self.images_gm,
                           [0, (nc_idx * self.h_in_size + index_3) * self.w_in_size], self.w_in_size)
            self.data_move(self.src_line_4, self.images_gm,
                           [0, (nc_idx * self.h_in_size + index_4) * self.w_in_size], self.w_in_size)
        
    def calc_oper_lines(self):
        w_in_index_ll = self.tik_instance.Scalar("int64")
        w_in_index_l = self.tik_instance.Scalar("int64")
        w_in_index_r = self.tik_instance.Scalar("int64")
        w_in_index_rr = self.tik_instance.Scalar("int64")
        temp_index = self.tik_instance.Scalar("int64")
        load_index = self.tik_instance.Scalar("int32")
        w_max_index = self.w_in_size - 1
        with self.tik_instance.for_range(0, self.w_out_size) as w_out_index:
            load_index.set_as(self.index_w_mapping_ub[w_out_index])
            w_in_index_l.set_as(load_index)
            temp_index.set_as(w_in_index_l - 1)
            self.tik_instance.scalar_max(w_in_index_ll, 0, temp_index)
            temp_index.set_as(w_in_index_l + 1)
            self.tik_instance.scalar_min(w_in_index_r, w_max_index, temp_index)
            temp_index.set_as(w_in_index_l + 2)
            self.tik_instance.scalar_min(w_in_index_rr, w_max_index, temp_index)
            self.tik_instance.scalar_max(w_in_index_l, 0, w_in_index_l)

            self.w_ub_1[w_out_index] = self.src_line_1[w_in_index_ll]
            self.w_ub_2[w_out_index] = self.src_line_1[w_in_index_l]
            self.w_ub_3[w_out_index] = self.src_line_1[w_in_index_r]
            self.w_ub_4[w_out_index] = self.src_line_1[w_in_index_rr]

    def conv_operator_template(self, op_obj, mode, dst, src, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        vector_mask_max = Constant.VECTOR_MASK_MAX
        dst_offset, src_offset = offsets

        tensor_size = num if num else src.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, mode, dst[tmp_dst_offset], src[tmp_src_offset], 255,
                       dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            op_obj(vector_mask_max, mode, dst[dst_offset], src[src_offset], repeat_time, dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            op_obj(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)

    def scalar_operator_template(self, op_obj, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        vector_mask_max = Constant.VECTOR_MASK_MAX
        dst_offset, src_offset = offsets

        tensor_size = num if num else src.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src[tmp_src_offset], scalar, 255,
                       dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            op_obj(vector_mask_max, dst[dst_offset], src[src_offset], scalar, repeat_time, dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            op_obj(last_num, dst[dst_offset], src[src_offset], scalar, 1, dst_stride, src_stride)

    def double_operator_template(self, op_obj, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8,
                                 src1_stride=8):
        """
        tik api template
        """
        vector_mask_max = Constant.VECTOR_MASK_MAX
        dst_offset, src0_offset, src1_offset = offsets

        tensor_size = num if num else src1.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src0_offset = src0_offset + index * vector_mask_max * 255
                tmp_src1_offset = src1_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src0[tmp_src0_offset], src1[tmp_src1_offset], 255,
                       dst_stride, src0_stride, src1_stride)

            dst_offset += loop * vector_mask_max * 255
            src0_offset += loop * vector_mask_max * 255
            src1_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            op_obj(vector_mask_max, dst[dst_offset], src0[src0_offset], src1[src1_offset], repeat_time, dst_stride,
                   src0_stride, src1_stride)
            dst_offset += repeat_time * vector_mask_max
            src0_offset += repeat_time * vector_mask_max
            src1_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            op_obj(last_num, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1, dst_stride, src0_stride,
                   src1_stride)

    def data_move(self, dst, src, offsets, num, nburst=1, src_stride=0, dst_stride=0):
        """
        move data from ub or gm to ub or gm
        """
        dst_offset, src_offset = offsets
        sid = 0
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        burst_len = (num + data_each_block - 1) // data_each_block
        self.tik_instance.data_move(dst[dst_offset],
                                    src[src_offset],
                                    sid,
                                    nburst,
                                    burst_len,
                                    src_stride=src_stride,
                                    dst_stride=dst_stride)

    def move_to_gm(self, dst, src, offsets, num, nburst=1, src_stride=0, dst_stride=0):
        """
        move data from ub to gm
        """
        dst_offset, src_offset = offsets
        sid = 0
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        burst_len = num // data_each_block
        if burst_len > 0:
            self.tik_instance.data_move(dst[dst_offset],
                                        src[src_offset],
                                        sid,
                                        nburst,
                                        burst_len,
                                        src_stride=src_stride,
                                        dst_stride=dst_stride)
        
        last_num = num % data_each_block
        if last_num > 0:
            tail_block = self.tik_instance.Tensor(self.x_dtype, [data_each_block],
                                                  name="tail_block", scope=tik.scope_ubuf)
            if self.x_dtype == "float32":
                self.tik_instance.set_atomic_add(1)
                self.data_move(tail_block, self.zero_value_gm, [0, 0], 1)
                last_block_start = src_offset + burst_len * data_each_block
                for i in range(0, last_num):
                    tail_block[i].set_as(src[last_block_start + i])
                    
                self.tik_instance.data_move(dst[dst_offset + burst_len * data_each_block],
                                            tail_block[0],
                                            sid,
                                            nburst,
                                            1,
                                            src_stride=src_stride,
                                            dst_stride=dst_stride)
                self.tik_instance.set_atomic_add(0)
            else:
                last_block_start = src_offset + num - data_each_block
                for i in range(0, data_each_block):
                    tail_block[i].set_as(src[last_block_start + i])
                self.tik_instance.data_move(dst[dst_offset + num - data_each_block],
                                            tail_block[0],
                                            sid,
                                            nburst,
                                            1,
                                            src_stride=src_stride,
                                            dst_stride=dst_stride)


class ResizeLinear:
    """
    ResizeLinear main functions
    """

    # 'pylint: disable=too-many-arguments
    # 'pylint: disable=too-many-locals
    def __init__(self, x, sizes, scales, coordinate_transformation_mode="align_corners", kernel_name="resize_d"):

        self.tik_instance = tik.Tik()

        self.x_dtype = x.get("dtype")
        self.x_shape = x.get("shape")
        self.size = sizes[0]
        self.scale = scales[0]
        self.dim0 = self.x_shape[0]
        self.dim1 = self.x_shape[1]
        self.dim_redundancy = self.x_shape[2]
        self.dim2 = self.x_shape[3]
        self.input_num = self.dim0 * self.dim1 * self.dim2
        self.coordinate_transformation_mode = coordinate_transformation_mode
        self.kernel_name = kernel_name

        self.check_param1(self.dim_redundancy, self.dim2, self.size)
        self.check_param2(sizes, scales)

        if self.coordinate_transformation_mode == "align_corners":
            self.scale_w = 0. if self.size == 1 else (self.dim2 - 1) / (self.size - 1)
        else:
            self.scale_w = 1.0 / self.scale if self.scale > 0. else (self.dim2 / self.size)

        self.output_num = self.dim0 * self.dim1 * self.size

        block_bite_size = 32
        dtype_bytes_size = cce.get_bit_len(self.x_dtype) // 8
        self.data_each_block = block_bite_size // dtype_bytes_size

        self.x_gm = self.tik_instance.Tensor(self.x_dtype, (self.dim0, self.dim1, 1, self.dim2),
                                             name="x_gm",
                                             scope=tik.scope_gm)
        self.x_gm.reshape(self.x_shape)

        self.output_gm = self.tik_instance.Tensor(self.x_dtype, (self.dim0, self.dim1, 1, self.size),
                                                  name="output_gm",
                                                  scope=tik.scope_gm)

    # 'pylint: disable=too-many-locals, too-many-branches
    def resize_linear_compute(self):
        """
        ResizeLinear main logic
        """
        self.x_gm.reshape([
            self.input_num,
        ])

        if self.output_num <= self.data_each_block:
            res_lastdim_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block],
                                                      name="res_lastdim_ub",
                                                      scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.dim0) as i:
                with self.tik_instance.for_range(0, self.dim1) as j:
                    current_index_output = self.tik_instance.Scalar("int32",
                                                                    init_value=i * (self.dim1 * self.size) +
                                                                               j * self.size)
                    with self.tik_instance.for_range(0, self.size) as k:
                        with self.tik_instance.if_scope(self.size == 1):
                            res_lastdim_ub[current_index_output + k].set_as(
                                self.get_number_in_global_memory(i * (self.dim1 * self.dim2) + j * self.dim2))
                        with self.tik_instance.else_scope():
                            res_lastdim_ub[current_index_output + k].set_as(
                                self.compute_helper(self.scale_w, k, i * (self.dim1 * self.dim2) + j * self.dim2))

            self.tik_instance.data_move(self.output_gm, res_lastdim_ub, 0, 1, 1, 0, 0)

        elif self.size < self.data_each_block:
            loop_time = self.output_num // self.data_each_block
            with self.tik_instance.for_range(0, loop_time) as i:
                res_lastdim_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block],
                                                          name="res_lastdim_ub",
                                                          scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, self.data_each_block) as j:
                    current_index = i * self.data_each_block + j
                    current_dim1 = current_index // self.size
                    with self.tik_instance.if_scope(self.size == 1):
                        res_lastdim_ub[j].set_as(
                            self.get_number_in_global_memory(current_dim1 * self.dim2))
                    with self.tik_instance.else_scope():
                        res_lastdim_ub[j].set_as(
                            self.compute_helper(self.scale_w, current_index % self.size, current_dim1 * self.dim2))
                self.tik_instance.data_move(self.output_gm[i * self.data_each_block], res_lastdim_ub, 0, 1, 1, 0, 0)

            remainder = self.output_num % self.data_each_block
            with self.tik_instance.if_scope(remainder != 0):
                remainder_begin_index = self.output_num - self.data_each_block
                res_lastdim_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block],
                                                          name="res_lastdim_ub",
                                                          scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, self.data_each_block) as k:
                    current_index = remainder_begin_index + k
                    current_dim1 = current_index // self.size
                    with self.tik_instance.if_scope(self.size == 1):
                        res_lastdim_ub[k].set_as(
                            self.get_number_in_global_memory(current_dim1 * self.dim2))
                    with self.tik_instance.else_scope():
                        res_lastdim_ub[k].set_as(
                            self.compute_helper(self.scale_w, current_index % self.size, current_dim1 * self.dim2))
                self.tik_instance.data_move(self.output_gm[remainder_begin_index], res_lastdim_ub, 0, 1, 1, 0, 0)

        else:
            with self.tik_instance.for_range(0, self.dim0) as i:
                with self.tik_instance.for_range(0, self.dim1) as j:
                    loop_time = self.tik_instance.Scalar("int32", init_value=self.size // self.data_each_block)
                    current_index_output = self.tik_instance.Scalar("int32",
                                                                    init_value=i * (self.dim1 * self.size) +
                                                                               j * self.size)
                    with self.tik_instance.for_range(0, loop_time) as m:
                        res_lastdim_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block],
                                                                  name="res_lastdim_ub",
                                                                  scope=tik.scope_ubuf)

                        with self.tik_instance.for_range(0, self.data_each_block) as n:
                            with self.tik_instance.if_scope(self.size == 1):
                                res_lastdim_ub[n].set_as(
                                    self.get_number_in_global_memory(i * (self.dim1 * self.dim2) + j * self.dim2))
                            with self.tik_instance.else_scope():
                                res_lastdim_ub[n].set_as(
                                    self.compute_helper(self.scale_w, m * self.data_each_block + n,
                                                        i * (self.dim1 * self.dim2) + j * self.dim2))
                        self.tik_instance.data_move(self.output_gm[current_index_output], res_lastdim_ub, 0, 1, 1, 0, 0)
                        current_index_output.set_as(current_index_output + self.data_each_block)
                    res_lastdim_remainder_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block],
                                                                        name="res_lastdim_remainder_ub",
                                                                        scope=tik.scope_ubuf)

                    remainder = self.size % self.data_each_block
                    with self.tik_instance.if_scope(remainder != 0):
                        remainder_begin_index = self.size - self.data_each_block
                        with self.tik_instance.for_range(0, self.data_each_block) as k:
                            with self.tik_instance.if_scope(self.size == 1):
                                res_lastdim_remainder_ub[k].set_as(
                                    self.get_number_in_global_memory(i * self.dim1 * self.dim2 + j * self.dim2))
                            with self.tik_instance.else_scope():
                                res_lastdim_remainder_ub[k].set_as(
                                    self.compute_helper(self.scale_w, remainder_begin_index + k,
                                                        i * (self.dim1 * self.dim2) + j * self.dim2))
                        self.tik_instance.data_move(
                            self.output_gm[i * (self.dim1 * self.size) + (j + 1) * self.size - self.data_each_block],
                            res_lastdim_remainder_ub, 0, 1, 1, 0, 0)

        self.output_gm.reshape([self.dim0, self.dim1, 1, self.size])

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.x_gm], outputs=[self.output_gm])
        return self.tik_instance

    def get_number_in_global_memory(self, index):
        """
        get the value with given index from input tensor (in global memory)

        Parameters
        ----------
        index : int
            the index of required value in the input tensor

        Returns
        -------
        res : input.dtype
            the value under the given index
        """
        max_offset = max(0, self.input_num - self.data_each_block)

        x_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block], name="x_ub", scope=tik.scope_ubuf)

        res = self.tik_instance.Scalar(self.x_dtype, name="res")

        index = self.tik_instance.Scalar("int32", init_value=index)

        with self.tik_instance.if_scope(index < max_offset):
            self.tik_instance.data_move(x_ub, self.x_gm[index], 0, 1, 1, 0, 0)
            res.set_as(x_ub[0])

        with self.tik_instance.else_scope():
            self.tik_instance.data_move(x_ub, self.x_gm[max_offset], 0, 1, 1, 0, 0)
            res.set_as(x_ub[index - max_offset])

        return res

    def compute_helper(self, scale_w, output_block_offset, input_dim_offset):
        """
        ResizeLinear main calculation logic

        Parameters
        ----------
        scale_w : float

        output_block_offset : int

        input_dim_offset : int

        Returns
        -------
        res : input.dtype
            the output value with the given parameters

        """
        # Cal real
        real_w = self.tik_instance.Scalar("float32", name="real_w")
        k = self.tik_instance.Scalar("float32", init_value=output_block_offset)
        temp_w = self.tik_instance.Scalar("float32")
        with self.tik_instance.if_scope(self.coordinate_transformation_mode == "align_corners"):
            temp_w.set_as(scale_w * k)
        with self.tik_instance.else_scope():
            temp = self.tik_instance.Scalar(dtype="float32", init_value=scale_w * (k + 0.5) - 0.5)
            with self.tik_instance.if_scope(temp < 0):
                temp_w.set_as(0.)
            with self.tik_instance.else_scope():
                temp_w.set_as(temp)
        real_w.set_as(temp_w)

        # Cal Integer of real_w
        coefficient_w = self.tik_instance.Scalar("int32", name="coefficient_w")
        self.tik_instance.scalar_conv('floor', coefficient_w, real_w)

        # Cal Decimal of real_w
        coefficient_lambda = self.tik_instance.Scalar("float32", name="coefficient_lambda")
        coefficient_lambda.set_as(real_w - coefficient_w)

        # Cal 1.0 - Decimal of real_w
        coefficient_lambda0 = self.tik_instance.Scalar("float32", name="coefficient_lambda0")
        coefficient_lambda0.set_as(1.0 - coefficient_lambda)

        index = self.tik_instance.Scalar("int32", init_value=input_dim_offset + coefficient_w)
        temp2 = self.tik_instance.Scalar(self.x_dtype, init_value=self.get_number_in_global_memory(index))

        offset = self.tik_instance.Scalar(dtype="int32", init_value=1)
        with self.tik_instance.if_scope(coefficient_w == (self.dim2 - 1)):
            offset.set_as(0)

        temp4 = self.tik_instance.Scalar(self.x_dtype, init_value=self.get_number_in_global_memory(offset + index))

        res = self.tik_instance.Scalar(dtype=self.x_dtype,
                                       init_value=(coefficient_lambda0 * temp2 + coefficient_lambda * temp4))

        return res

    @staticmethod
    def check_param1(dim_redundancy, in_size_w, out_size_w):
        """
        check  in_size_w, out_size_w:
        in_size_w and out_size_w should be greater than 0

        Parameters
        ----------
        in_size_w : int
            the last dim of input
        out_size_w : int
            the output size

        Returns
        -------
        None
        """
        # Since only NCHW format input is currently supported, the input of npu
        # is converted from 3dim to 4dim, so the relevant judgment has also been changed(if dim_redundancy != 1)
        if dim_redundancy != 1:
            raise RuntimeError("The 3rd Dim of Input Tensor should always be 1.")

        if in_size_w <= 0 or out_size_w <= 0:
            raise RuntimeError("Input and output sizes should be greater than 0.")

    def check_param2(self, sizes, scales):
        """
        check sizes, scales:
        the length of sizes and scales should both be 1,
        the value of the scales should equal to x.shape[2] / sizes[0].

        Parameters
        ----------
        sizes : list
            list with values of sizes
        scales : list
            list with values of scales

        Returns
        -------
        None
        """
        # check sizes
        if len(sizes) != 1:
            raise RuntimeError("It is expected len(sizes) equals to 1.")

        # check scales
        if len(scales) != 1 and scales is not None:
            raise RuntimeError("It is expected len(scales) equals to 1.")

        # check scales value
        if scales is not None and (sizes[0] / self.dim2 - scales[0]) > 0.0001:
            raise RuntimeError("It is expected scales[0] equals to x.shape[2] / sizes[0].")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_FLOAT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
# 'pylint: disable=too-many-arguments,too-many-locals
# 'pylint: disable=W0613
def resize_d(x,
             y,
             sizes,
             scales=None,
             roi=None,
             coordinate_transformation_mode="half_pixel",
             cubic_coeff_a=-0.75,
             exclude_outside=0,
             extrapolation_value=0.0,
             mode="nearest",
             nearest_mode="round_prefer_floor",
             kernel_name="resize_d"):
    """
    algorithm: resize_d
    Operation for resize_d

    Parameters
    ----------
    x : dict
        dict with keys(shape and dtype) of x
    y : dict
        dict with keys(shape and dtype) of y
    sizes : list
        the shape of output about 'new_height, new_width'
    scales : list
        the value about 'scale_h, scale_w'
    roi: list
        The RoIs' coordinates are normalized in the coordinate system of the input image.
        It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"
    coordinate_transformation_mode : str
        This attribute describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
    cubic_coeff_a : float
        The coefficient 'a' used in cubic interpolation.
    exclude_outside : int
        If set to 1, the weight of sampling locations outside the tensor will be set to 0
        and the weight will be renormalized so that their sum is 1.0.
    extrapolation_value : float
        When coordinate_transformation_mode is "tf_crop_and_resize" and
        x_original is outside the range [0, length_original - 1],
        this value is used as the corresponding output value. Default is 0.0f.
    mode : str
        Three interpolation modes: nearest (default), linear and cubic.
    nearest_mode : str
        Four modes: round_prefer_floor (default, as known as round half down),
        round_prefer_ceil (as known as round half up), floor, ceil.
        Only used by nearest interpolation.
    kernel_name : str
        kernel name, default value is "resize_d"

    Returns
    -------
    None
    """
    x_dim = len(x.get("shape"))
    if mode == "cubic" and x_dim == 4:
        resize_bicubic_instance = ResizeBicubic(x, sizes, scales, coordinate_transformation_mode, cubic_coeff_a,
                                                kernel_name)
        res = resize_bicubic_instance.resize_bicubic_compute()
    elif mode == "linear" and x_dim == 4:
        resize_linear = ResizeLinear(x, sizes, scales, coordinate_transformation_mode, kernel_name)
        res = resize_linear.resize_linear_compute()
    else:
        raise RuntimeError("Not supported at the moment.")
    return res
