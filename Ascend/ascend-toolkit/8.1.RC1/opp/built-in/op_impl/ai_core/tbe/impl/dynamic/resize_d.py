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

    # tensor shape params
    MAX_INT32 = 2 ** 31 - 1
    MAX_INT32_VALUE = 2 ** 31 - 1
    MAX_INT64 = 2 ** 63 - 1
    MAX_LINE_NUM = 1024
    TILING_ARG_NUM = 24

    TILING_SCALAR_DTYPE = "int32"
    NUM_EACH_BURST = 8


def get_ceil_int(int1, int2):
    ceil_int = (int1 + int2 - 1) // int2
    return ceil_int


class ResizeBicubicHWNC():
    """
    ResizeBicubic main functions
    """
    def __init__(self, input_x, sizes, scales, coordinate_transformation_mode, cubic_coeff_a, kernel_name="resize_d"):
        """
        init ResizeBicubic base parameters
        """
        self.tik_instance = tik.Tik(disable_debug=False)
        self.kernel_name = kernel_name
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.input_x_dtype = input_x.get("dtype").lower()
        if self.input_x_dtype == "float16":
            self.ori_dtype_size = 2
        else:
            self.ori_dtype_size = 4
        self.dtype_size = 4
        self.block_byte_size = 32
        self.dtype_block_ele_num = int(self.block_byte_size / self.dtype_size)
        self.ori_dtype_block_ele_num = int(self.block_byte_size / self.ori_dtype_size)
        self.nc_ele_max = 2048

        self.cubic_coeff_a = cubic_coeff_a
        if coordinate_transformation_mode == "align_corners":
            self.align_corners = True
        else:
            self.align_corners = False

        if scales is not None:
            if len(scales) == 1:
                self.scales_h, self.scales_w = scales[0], scales[0]
            else:
                self.scales_h, self.scales_w = scales[0], scales[1]
        else:
            self.scales_h, self.scales_w = 0, 0

        self.tiling_gm = self.tik_instance.Tensor("float32", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
        self.input_gm = self.tik_instance.Tensor(self.input_x_dtype, (Constant.MAX_INT32_VALUE,),
                                                       name="input_gm", scope=tik.scope_gm)
        self.tiling_args()
        self.init_ub()
        self.init_scalar()
        self.output_gm = self.tik_instance.Tensor(self.input_x_dtype, (Constant.MAX_INT32_VALUE,),
                                                  name="output_gm", scope=tik.scope_gm, is_atomic_add=True)

    @staticmethod
    def area_pixel_compute_source_index(scale, dst_index, align_corners):
        if align_corners:
            return scale * dst_index
        else:
            return scale * (dst_index + 0.5) - 0.5

    @staticmethod
    def cubic_convolution1(x, cubic_coeff_a):
        return ((cubic_coeff_a + 2) * x - (cubic_coeff_a + 3)) * x * x + 1

    @staticmethod
    def cubic_convolution2(x, cubic_coeff_a):
        return ((cubic_coeff_a * x - 5 * cubic_coeff_a) * x + 8 * cubic_coeff_a) * x - 4 * cubic_coeff_a

    def tiling_args(self):
        """
        Get runtime params from tiling
        """
        self.nbatch = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="nbatch")
        self.channels = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="channels")
        self.input_height = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="input_height")
        self.input_width = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="input_width")
        self.output_height = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="output_height")
        self.output_width = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="output_width")
        self.height_scale = self.tik_instance.Scalar("float32", name="height_scale")
        self.width_scale = self.tik_instance.Scalar("float32", name="width_scale")
        self.scales_h = self.tik_instance.Scalar("float32", name="scales_h")
        self.scales_w = self.tik_instance.Scalar("float32", name="scales_w")
        self.tiling_core_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "tiling_core_num")           

        burst_val = get_ceil_int(Constant.TILING_ARG_NUM, Constant.NUM_EACH_BURST)

        self.tiling_ub = self.tik_instance.Tensor("float32", [Constant.TILING_ARG_NUM, ],
                                             name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, burst_val, 0, 0)
        
        self.nbatch.set_as(self.tiling_ub[2])
        self.channels.set_as(self.tiling_ub[3])
        self.input_height.set_as(self.tiling_ub[0])
        self.input_width.set_as(self.tiling_ub[1])
        self.output_height.set_as(self.tiling_ub[4])
        self.output_width.set_as(self.tiling_ub[5])
        if self.align_corners == True:
            self.height_scale.set_as(self.tiling_ub[8])
            self.width_scale.set_as(self.tiling_ub[9])
        else:
            self.height_scale.set_as(self.tiling_ub[10])
            self.width_scale.set_as(self.tiling_ub[11])
        self.tiling_core_num.set_as(self.tiling_ub[16])
        self.scales_h.set_as(self.tiling_ub[19])
        self.scales_w.set_as(self.tiling_ub[20])

        self.area_pixel_compute_scale(self.height_scale, self.input_height, self.output_height,
                                      self.align_corners, scale=self.scales_h)
        self.area_pixel_compute_scale(self.width_scale, self.input_width, self.output_width,
                                      self.align_corners, scale=self.scales_w)
        self.nc_ele_num = self.nbatch * self.channels
        with self.tik_instance.if_scope(self.nc_ele_num < self.ori_dtype_block_ele_num):
            if not tbe_platform.api_check_support("tik.vcopy"):
                self.tiling_core_num.set_as(1)
    
    def resize_bicubic_compute(self):
        """
        main entrance of the calculation process
        """
        hw_compute_num = self.tik_instance.Scalar("int64", name="hw_compute_num")
        hw_compute_tail = self.tik_instance.Scalar("int64", name="hw_compute_tail")
        hw_compute_num.set_as((self.output_height * self.output_width) // self.tiling_core_num)
        hw_compute_tail.set_as((self.output_height * self.output_width) % self.tiling_core_num)

        with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as core_index:

            # Calculate the core, each core starts indexing and counting the number
            core_start_index = self.tik_instance.Scalar("int64", name="core_start_index")
            core_compute_num = self.tik_instance.Scalar("int64", name="core_compute_num")
            with self.tik_instance.if_scope(core_index < hw_compute_tail):
                core_start_index.set_as(hw_compute_num * core_index + core_index)
                core_compute_num.set_as(hw_compute_num + 1)
            with self.tik_instance.else_scope():
                core_start_index.set_as(hw_compute_num * core_index + hw_compute_tail)
                core_compute_num.set_as(hw_compute_num)
            
            with self.tik_instance.for_range(core_start_index, core_start_index + core_compute_num) as hw_index:
                # Calculated output coordinates
                self.output_y.set_as(hw_index // self.output_width)
                self.output_x.set_as(hw_index % self.output_width)

                # Calculate NC length and number of cycles
                self.nc_loop_num.set_as(self.nc_ele_num // self.nc_ele_max + 1)
                self.last_loop_ele_num.set_as((self.nc_ele_num % self.nc_ele_max))

                # Calculate each cycle
                self.clac_each_cycle(hw_index)
         
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance

    def clac_each_cycle(self, hw_index):
        with self.tik_instance.for_range(0, self.nc_loop_num) as loop:
            with self.tik_instance.if_scope(loop == self.nc_loop_num - 1):
                self.move_ele_num.set_as(self.last_loop_ele_num)
                self.compute_ele_num.set_as(get_ceil_int(self.last_loop_ele_num,
                                            self.ori_dtype_block_ele_num) * self.ori_dtype_block_ele_num)
            with self.tik_instance.else_scope():
                self.move_ele_num.set_as(self.nc_ele_max)
                self.compute_ele_num.set_as(self.nc_ele_max)
            
            with self.tik_instance.if_scope(tik.all((self.input_height == self.output_height),
                                                    (self.input_width == self.output_width))):
                # input and output are the same size,just copy
                self.data_move_in_same_size(loop)
            with self.tik_instance.else_scope():
                # Calculate the weight coefficient and input coordinates
                with self.tik_instance.if_scope(loop == 0):
                    self.calc_input_and_coeffs(hw_index)
                # data move in 
                self.data_move_in_general(loop)
                # data calculation
                self.resize_bicubic_data_compute()
            # data move out
            self.data_move_out(loop)

    def resize_bicubic_data_compute(self):
        for index_y in range(4):
            for index_x in range(4):
                if not index_x:
                    dst_offset = index_y * self.need_ub_len
                    src_offset = index_y * 4 * self.need_ub_len + index_x * self.need_ub_len
                    self.single_operator_template(self.tik_instance.vec_muls, 
                                                    self.tmp_tensor,
                                                    self.input_tensor, 
                                                    self.coeffs_x[index_x],
                                                    [dst_offset, src_offset], self.compute_ele_num)
                else:
                    dst_offset = index_y * 4 * self.need_ub_len + index_x * self.need_ub_len
                    src_offset = index_y * 4 * self.need_ub_len + index_x * self.need_ub_len
                    self.single_operator_template(self.tik_instance.vec_muls, 
                                                    self.input_tensor, 
                                                    self.input_tensor, 
                                                    self.coeffs_x[index_x],
                                                    [dst_offset, src_offset], self.compute_ele_num)
                    dst_offset = index_y * self.need_ub_len
                    src_offset0 = index_y * self.need_ub_len
                    src_offset1 = index_y * 4 * self.need_ub_len + index_x * self.need_ub_len
                    self.double_operator_template(self.tik_instance.vec_add,
                                                    self.tmp_tensor,
                                                    self.tmp_tensor, 
                                                    self.input_tensor, 
                                                    [dst_offset, src_offset0, src_offset1], self.compute_ele_num)
        for index_y in range(0, 4):
            if index_y:
                dst_offset = index_y * self.need_ub_len
                src_offset = index_y * self.need_ub_len
                self.single_operator_template(self.tik_instance.vec_muls, 
                                                self.tmp_tensor, 
                                                self.tmp_tensor, 
                                                self.coeffs_y[index_y],
                                                [dst_offset, src_offset], self.compute_ele_num)
                src_offset1 = index_y * self.need_ub_len
                self.double_operator_template(self.tik_instance.vec_add,
                                                self.output_tensor, 
                                                self.output_tensor,  
                                                self.tmp_tensor, 
                                                [0, 0, src_offset1], self.compute_ele_num)
            else:
                self.single_operator_template(self.tik_instance.vec_muls, 
                                                self.output_tensor, 
                                                self.tmp_tensor, 
                                                self.coeffs_y[index_y],
                                                [0, 0], self.compute_ele_num)
        if self.input_x_dtype == "float16":
            # data fp32-->fp16
            self.data_conv(self.tmp_tensor_fp16, self.output_tensor, [0, 0], self.compute_ele_num, 4, 8)
        
    def data_move_in_same_size(self, loop):
        dst_offset = 0
        src_offset = self.output_y * self.input_width * self.nc_ele_num + \
                            self.output_x * self.nc_ele_num + loop * self.nc_ele_max
        if self.input_x_dtype == "float32" :
            self.data_move_general(self.output_tensor, self.input_gm, src_offset, self.move_ele_num, "gm2ub")
            # If the number of NC is less than one block, the value needs to be zero
            self.set_tensor_zero(self.output_tensor, dst_offset)
        else:
            self.data_move_general(self.tmp_tensor_fp16, self.input_gm, src_offset, self.move_ele_num, "gm2ub")
            # If the number of NC is less than one block, the value needs to be zero
            self.set_tensor_zero(self.tmp_tensor_fp16, dst_offset)

    def data_move_in_general(self, loop):
        for index_y, y in enumerate(self.access_y):
            for index_x, x in enumerate(self.access_x):
                dst_offset = index_y * 4 * self.need_ub_len + index_x * self.need_ub_len
                src_offset = y * self.input_width * self.nc_ele_num + x * self.nc_ele_num + loop * self.nc_ele_max
                if self.input_x_dtype == "float32" :
                    self.data_move_general(self.input_tensor[dst_offset:], 
                                         self.input_gm, src_offset, 
                                         self.move_ele_num, "gm2ub")
                else:
                    self.data_move_general(self.tmp_tensor_fp16, 
                                         self.input_gm, 
                                         src_offset, 
                                         self.move_ele_num, "gm2ub")
                    # data fp16-->fp32
                    self.data_conv(self.input_tensor, self.tmp_tensor_fp16, [dst_offset, 0], self.compute_ele_num, 8, 4)
                # If the number of NC is less than one block, the value needs to be zero
                self.set_tensor_zero(self.input_tensor, dst_offset)

    def set_tensor_zero(self, set_tensor, dst_offset):
        with self.tik_instance.if_scope(self.nc_ele_num < self.ori_dtype_block_ele_num):
            with self.tik_instance.for_range(self.nc_ele_num, self.ori_dtype_block_ele_num) as index:
                set_tensor[dst_offset + index].set_as(0.0)

    def data_move_out(self, loop):
        dst_offset = self.output_y * self.output_width * self.nc_ele_num + \
                                    self.output_x * self.nc_ele_num + loop * self.nc_ele_max
        with self.tik_instance.if_scope(self.nc_ele_num < self.ori_dtype_block_ele_num):
            if tbe_platform.api_check_support("tik.vcopy"):
                self.tik_instance.set_atomic_add(self.input_x_dtype)
        if self.input_x_dtype == "float32":
            self.data_move_general(self.output_gm, self.output_tensor, dst_offset, self.move_ele_num, "ub2gm")
        else:
            self.data_move_general(self.output_gm, self.tmp_tensor_fp16, dst_offset, self.move_ele_num, "ub2gm")
        self.tik_instance.set_atomic_add(0)

    def calc_input_and_coeffs(self, hw_index):
        
        # Calculate the x coordinate and the t_x decimal part of the input tensor
        self.real_x.set_as(self.area_pixel_compute_source_index(self.width_scale, self.output_x, self.align_corners))
        self.guard_index_and_lambda(self.real_x, self.input_width, self.input_x, self.t_x)
        # Calculate the y coordinate and the t_y decimal part of the input tensor 
        self.real_y.set_as(self.area_pixel_compute_source_index(self.height_scale, self.output_y, self.align_corners))
        self.guard_index_and_lambda(self.real_y, self.input_height, self.input_y, self.t_y)
        # Calculated weight
        self.get_cubic_coefficients(self.coeffs_x, self.t_x)
        self.get_cubic_coefficients(self.coeffs_y, self.t_y)
        # Calculate the legal coordinates of 16 points
        for i in range(0, 4):
            self.access_x[i].set_as(self.input_x - 1 + i)
            with self.tik_instance.if_scope(self.access_x[i] > self.input_width - 1):
                self.access_x[i].set_as(self.input_width - 1)
            with self.tik_instance.if_scope(self.access_x[i] < 0):
                self.access_x[i].set_as(0)
            
            self.access_y[i].set_as(self.input_y - 1 + i)
            with self.tik_instance.if_scope(self.access_y[i] > self.input_height - 1):
                self.access_y[i].set_as(self.input_height - 1)
            with self.tik_instance.if_scope(self.access_y[i] < 0):
                self.access_y[i].set_as(0)

    def init_ub(self):
        self.need_ub_len = self.tik_instance.Scalar("int64", name="need_ub_len")
        with self.tik_instance.if_scope(self.nc_ele_num > self.nc_ele_max):
            self.need_ub_len.set_as(self.nc_ele_max)
        with self.tik_instance.else_scope():
            self.need_ub_len.set_as(get_ceil_int(self.nc_ele_num,
                                                 self.ori_dtype_block_ele_num) * self.ori_dtype_block_ele_num)

        self.input_tensor = self.tik_instance.Tensor("float32", (4 * 4 * self.need_ub_len, ), 
                                                                name="input_tensor", scope=tik.scope_ubuf)
        self.tmp_tensor = self.tik_instance.Tensor("float32", (4 * self.need_ub_len, ), 
                                                                name="tmp_tensor", scope=tik.scope_ubuf)
        self.output_tensor = self.tik_instance.Tensor("float32", (self.need_ub_len, ), 
                                                                name="output_tensor", scope=tik.scope_ubuf)
        self.tmp_tensor_fp16 = self.tik_instance.Tensor("float16", (self.need_ub_len, ), 
                                                                name="tmp_tensor_fp16", scope=tik.scope_ubuf)

    def init_scalar(self):
        self.output_y = self.tik_instance.Scalar("int64", name="output_y")
        self.output_x = self.tik_instance.Scalar("int64", name="output_x")

        self.real_x = self.tik_instance.Scalar("float32", name="real_x")
        self.input_x = self.tik_instance.Scalar("int32", name="input_x")
        self.t_x = self.tik_instance.Scalar("float32", name="t_x")

        self.real_y = self.tik_instance.Scalar("float32", name="real_y")
        self.input_y = self.tik_instance.Scalar("int32", name="input_y")
        self.t_y = self.tik_instance.Scalar("float32", name="t_y")

        self.coeffs_x0 = self.tik_instance.Scalar("float32", name="coeffs_x0")
        self.coeffs_x1 = self.tik_instance.Scalar("float32", name="coeffs_x1")
        self.coeffs_x2 = self.tik_instance.Scalar("float32", name="coeffs_x2")
        self.coeffs_x3 = self.tik_instance.Scalar("float32", name="coeffs_x3")
        self.coeffs_x = [self.coeffs_x0, self.coeffs_x1, self.coeffs_x2, self.coeffs_x3]

        self.coeffs_y0 = self.tik_instance.Scalar("float32", name="coeffs_y0")
        self.coeffs_y1 = self.tik_instance.Scalar("float32", name="coeffs_y1")
        self.coeffs_y2 = self.tik_instance.Scalar("float32", name="coeffs_y2")
        self.coeffs_y3 = self.tik_instance.Scalar("float32", name="coeffs_y3")
        self.coeffs_y = [self.coeffs_y0, self.coeffs_y1, self.coeffs_y2, self.coeffs_y3]

        self.access_x0 = self.tik_instance.Scalar("int64", name="access_x0")
        self.access_x1 = self.tik_instance.Scalar("int64", name="access_x1")
        self.access_x2 = self.tik_instance.Scalar("int64", name="access_x2")
        self.access_x3 = self.tik_instance.Scalar("int64", name="access_x3")
        self.access_x = [self.access_x0, self.access_x1, self.access_x2, self.access_x3]

        self.access_y0 = self.tik_instance.Scalar("int64", name="access_y0")
        self.access_y1 = self.tik_instance.Scalar("int64", name="access_y1")
        self.access_y2 = self.tik_instance.Scalar("int64", name="access_y2")
        self.access_y3 = self.tik_instance.Scalar("int64", name="access_y3")
        self.access_y = [self.access_y0, self.access_y1, self.access_y2, self.access_y3]

        self.nc_loop_num = self.tik_instance.Scalar("int64", name="nc_loop_num")
        self.last_loop_ele_num = self.tik_instance.Scalar("int64", name="last_loop_ele_num")
        self.last_block_ele_num = self.tik_instance.Scalar("int64", name="last_block_ele_num")

        self.move_ele_num = self.tik_instance.Scalar("int64", name="move_ele_num")

        self.compute_ele_num = self.tik_instance.Scalar("int64", name="compute_ele_num")
        self.move_start_index = self.tik_instance.Scalar("int64", name="move_start_index")
        self.move_end_index = self.tik_instance.Scalar("int64", name="move_end_index")

    def data_move_general(self, dst, src, offset, num, trans):
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.data_copy_pad(dst, src, offset, num, trans)
        else:
            self.data_copy_align(dst, src, offset, num, trans)

    def data_copy_pad(self, dst, src, offset, num, trans, nburst=1):
        dst_gap, src_gap = 0, 0
        burst_len = num * self.ori_dtype_size
        if trans == "gm2ub":
            self.tik_instance.data_move_pad(dst, src[offset], nburst, burst_len, dst_gap, src_gap)
        else:
            self.tik_instance.data_move_pad(dst[offset], src, nburst, burst_len, dst_gap, src_gap)
        
    def data_copy_align(self, dst, src, offset, num, trans):
        """
        move data to align
        """
        num_align = num // self.ori_dtype_block_ele_num * self.ori_dtype_block_ele_num
        num_extra = num % self.ori_dtype_block_ele_num

        if trans == "gm2ub":
            with self.tik_instance.if_scope(num_align > 0):
                self.data_move(dst, src[offset], num=num_align)
            with self.tik_instance.if_scope(tik.all(num_extra > 0, self.nc_ele_num >= self.ori_dtype_block_ele_num)):
                self.data_move(dst[num_align],
                               src[offset + num - self.ori_dtype_block_ele_num],
                               num=self.ori_dtype_block_ele_num)
            with self.tik_instance.if_scope(tik.all(num_extra > 0, self.nc_ele_num < self.ori_dtype_block_ele_num)):
                self.data_move(dst[num_align],
                               src[offset + num_align],
                               num=self.ori_dtype_block_ele_num)

        if trans == "ub2gm":
            with self.tik_instance.if_scope(num_align > 0):
                self.data_move(dst[offset], src, num=num_align)
            with self.tik_instance.if_scope(tik.all(num_extra > 0, self.nc_ele_num >= self.ori_dtype_block_ele_num)):
                self.data_move(dst[offset + num - self.ori_dtype_block_ele_num],
                               src[num_align], num=self.ori_dtype_block_ele_num)
            with self.tik_instance.if_scope(tik.all(num_extra > 0, self.nc_ele_num < self.ori_dtype_block_ele_num)):
                self.data_move(dst[offset + num_align], src[num_align], num=self.ori_dtype_block_ele_num)

    def data_move(self, dst, src, num, nburst=1):
        """
        move data
        """
        sid, src_stride, dst_stride = 0, 0, 0
        burst_len = (num + self.ori_dtype_block_ele_num - 1) // self.ori_dtype_block_ele_num
        self.tik_instance.data_move(dst, src, sid, nburst, burst_len, src_stride=src_stride, dst_stride=dst_stride)

    def guard_index_and_lambda(self, real_input_index, input_size, input_int, input_float):
        self.tik_instance.scalar_conv("floor", input_int, real_input_index)
        with self.tik_instance.if_scope(input_int > input_size - 1):
            input_int.set_as(input_size - 1)

        input_float.set_as(real_input_index - input_int)
        with self.tik_instance.if_scope(input_float < 0):
            input_float.set_as(0)
        with self.tik_instance.if_scope(input_float > 1):
            input_float.set_as(1)

    def area_pixel_compute_scale(self, hw_scale, input_size, output_size, align_corners, scale=None):
        tmp1 = self.tik_instance.Scalar("float32", name="tmp1")
        tmp2 = self.tik_instance.Scalar("float32", name="tmp2")
        self.tik_instance.scalar_conv('', tmp1, input_size)
        self.tik_instance.scalar_conv('', tmp2, output_size)

        if align_corners:
            with self.tik_instance.if_scope(output_size > 1):
                hw_scale.set_as((tmp1 - 1.0) / (tmp2 - 1.0))
            with self.tik_instance.else_scope():
                hw_scale.set_as(0.0)
        else:
            with self.tik_instance.if_scope(scale > 0.0):
                hw_scale.set_as(scale)
            with self.tik_instance.else_scope():
                hw_scale.set_as(tmp1 / tmp2)

    def get_cubic_coefficients(self, coeffs, t):

        x1 = self.tik_instance.Scalar("float32", name="x1")
        x2 = self.tik_instance.Scalar("float32", name="x2")

        x1.set_as(t)
        coeffs[0].set_as(self.cubic_convolution2(x1 + 1.0, self.cubic_coeff_a))
        coeffs[1].set_as(self.cubic_convolution1(x1, self.cubic_coeff_a))

        x2.set_as(1.0 - t)
        coeffs[2].set_as(self.cubic_convolution1(x2, self.cubic_coeff_a))
        coeffs[3].set_as(self.cubic_convolution2(x2 + 1.0, self.cubic_coeff_a))

    # 'pylint: disable=huawei-too-many-arguments
    def single_operator_template(self, op_obj, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        dst_offset, src_offset = offsets[0], offsets[1]
        vector_mask_max = 256 // self.dtype_size

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                op_obj(vector_mask_max, dst[dst_offset + index * vector_mask_max * 255], 
                       src[src_offset + index * vector_mask_max * 255], scalar, 255, dst_stride, src_stride)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset + loop * vector_mask_max * 255], 
                   src[src_offset + loop * vector_mask_max * 255], scalar, repeat_time, dst_stride, src_stride)

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                   src[src_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], scalar, 
                   1, dst_stride, src_stride)

    # 'pylint: disable=huawei-too-many-arguments
    def double_operator_template(self, op_obj, dst, src0, src1, offsets, num=0, dst_stride=8,
                                 src0_stride=8, src1_stride=8):
        """
        tik api template
        """
        dst_offset, src0_offset, src1_offset = offsets[0], offsets[1], offsets[2]
        vector_mask_max = 256 // self.dtype_size

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                op_obj(vector_mask_max, dst[dst_offset + index * vector_mask_max * 255], 
                       src0[src0_offset + index * vector_mask_max * 255], 
                       src1[src1_offset + index * vector_mask_max * 255],
                       255, dst_stride, src0_stride, src1_stride)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset + loop * vector_mask_max * 255], 
                   src0[src0_offset + loop * vector_mask_max * 255], 
                   src1[src1_offset + loop * vector_mask_max * 255], 
                   repeat_time, dst_stride, src0_stride, src1_stride)

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                   src0[src0_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                   src1[src1_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                   1, dst_stride, src0_stride, src1_stride)

    # 'pylint: disable=huawei-too-many-arguments
    def data_conv(self, dst, src, offsets, num=0, dst_rep_stride=8, src_rep_stride=8):
        """
        conv fp16 <--> fp32
        """
        round_mode = 'none'
        dst_offset, src_offset, vector_mask_max = offsets[0], offsets[1], 64

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                self.tik_instance.vec_conv(vector_mask_max, round_mode, dst[dst_offset + index * vector_mask_max * 255],
                                       src[src_offset + index * vector_mask_max * 255], 
                                       255, dst_rep_stride, src_rep_stride)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_conv(vector_mask_max, round_mode, dst[dst_offset + loop * vector_mask_max * 255],
                                   src[src_offset + loop * vector_mask_max * 255], 
                                   repeat_time, dst_rep_stride, src_rep_stride)

        last_num = tensor_size % vector_mask_max

        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_conv(last_num, round_mode, 
                                   dst[dst_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max],
                                   src[src_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                                   1, dst_rep_stride, src_rep_stride)


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
        self.inner_dtype = "float32"

        # formula related
        self.scalar_half = 0.5
        self.scalar_negative_half = -0.5
        self.scalar_one = 1
        self.scalar_negative_one = -1

        # obtain input info and attrs
        self.x_dtype = x.get("dtype").lower()
        self.coordinate_transformation_mode = coordinate_transformation_mode

        self.init_tiling_gm_and_scalar()
        self.get_tiling_data()
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

        max_dim_value = list(range(Constant.MAX_LINE_NUM))
        zero_value = list(0 for i in range(Constant.MAX_LINE_NUM))
        self.images_gm = self.tik_instance.Tensor(self.x_dtype, [Constant.MAX_INT32],
                                                   name="images_gm", scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.x_dtype, [Constant.MAX_INT32],
                                                   name="out_gm", scope=tik.scope_gm, is_atomic_add=True)

        self.dst_idx_gm = self.tik_instance.Tensor(self.inner_dtype, [Constant.MAX_LINE_NUM],
                                                   name="dst_idx_gm", scope=tik.scope_gm, init_value=max_dim_value)
        self.zero_value_gm = self.tik_instance.Tensor(self.inner_dtype, [Constant.MAX_LINE_NUM],
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

        # calc vertical offset and diff
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

    def init_tiling_gm_and_scalar(self):
        """
        init tiling gm and scalar
        """
        self.tiling_gm = self.tik_instance.Tensor("float32", [Constant.TILING_ARG_NUM],
                                                   name="tiling_gm", scope=tik.scope_gm)

        self.batch_size = self.tik_instance.Scalar("int32", "batch_size")
        self.channel_size = self.tik_instance.Scalar("int32", "channel_size")
        self.h_in_size = self.tik_instance.Scalar("int32", "h_in_size")
        self.w_in_size = self.tik_instance.Scalar("int32", "w_in_size")
        self.h_out_size = self.tik_instance.Scalar("int32", "h_out_size")
        self.w_out_size = self.tik_instance.Scalar("int32", "w_out_size")
        self.h_scale = self.tik_instance.Scalar("float32", "h_scale")
        self.w_scale = self.tik_instance.Scalar("float32", "w_scale")            

        self.need_core_num = self.tik_instance.Scalar("int32", "need_core_num")
        self.x_data_each_block = self.tik_instance.Scalar("int32", "x_data_each_block")

        self.h_ub_out_size = self.tik_instance.Scalar("int32", "h_ub_out_size")
        self.w_ub_out_size = self.tik_instance.Scalar("int32", "w_ub_out_size")
        self.cubic_coeff_a = self.tik_instance.Scalar("float32", "cubic_coeff_a")
        self.tiling_core_num = self.tik_instance.Scalar("int32", "tiling_core_num")

    def get_tiling_data(self):
        """
        obtaion tiling info from gm
        """
        self.tiling_ub = self.tik_instance.Tensor("float32", [Constant.TILING_ARG_NUM],
                                                   name="tiling_ub", scope=tik.scope_ubuf)
        self.data_move(self.tiling_ub, self.tiling_gm, [0, 0], Constant.TILING_ARG_NUM)
        self.batch_size.set_as(self.tiling_ub[0])
        self.channel_size.set_as(self.tiling_ub[1])
        self.h_in_size.set_as(self.tiling_ub[2])
        self.w_in_size.set_as(self.tiling_ub[3])
        self.h_out_size.set_as(self.tiling_ub[4])
        self.w_out_size.set_as(self.tiling_ub[5])
        self.cubic_coeff_a.set_as(self.tiling_ub[6])

        if self.coordinate_transformation_mode == "align_corners":
            self.h_scale.set_as(self.tiling_ub[8])
            self.w_scale.set_as(self.tiling_ub[9])
        else:
            self.h_scale.set_as(self.tiling_ub[10])
            self.w_scale.set_as(self.tiling_ub[11])

        self.need_core_num.set_as(self.tiling_ub[12])
        self.x_data_each_block.set_as(self.tiling_ub[13])

        self.h_ub_out_size.set_as(self.tiling_ub[14])
        self.w_ub_out_size.set_as(self.tiling_ub[15])
        self.tiling_core_num.set_as(self.tiling_ub[16])

        self.coeff = self.cubic_coeff_a
        self.coeff_plus_2 = self.cubic_coeff_a + 2
        self.coeff_plus_3 = self.cubic_coeff_a + 3
        self.double_coeff_plus_3 = self.cubic_coeff_a * 2 + 3

    # 'pylint: disable=too-many-arguments
    def calc_weights(self, diff, weight_1, weight_2, weight_3, weight_4, ub_1, ub_2, ub_3, weight_size):
        """
        calculate weights of four point by line
        W1 = Ar(r-1)^2
        W2 = (A+2)r^3 - (A+3)r^2 + 1
        W3 = -(A+2)r^3 + (2A+3)r^2 -Ar
        W4 = -A(r-1)r^2
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
        main entrance of the calculation process
        """
        with self.tik_instance.if_scope(tik.all(self.w_out_size < self.x_data_each_block, self.x_dtype == "float16")):
            with self.tik_instance.for_range(0, self.batch_size * self.channel_size * self.h_out_size) as line_idx:
                self.compute_core(line_idx)

        with self.tik_instance.else_scope():
            batch_core_num = self.need_core_num // self.tiling_core_num
            batch_core_tail = self.need_core_num % self.tiling_core_num
            with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as core_idx:
                with self.tik_instance.for_range(0, batch_core_num) as batch:
                    self.compute_core(core_idx + batch * self.tiling_core_num)
                with self.tik_instance.if_scope(core_idx < batch_core_tail):
                    self.compute_core(batch_core_num * self.tiling_core_num + core_idx)

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.images_gm],
                                   outputs=[self.out_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance

    def compute_core(self, core_idx):
        h_in_index = self.tik_instance.Scalar("int32")
        h_out_index = self.tik_instance.Scalar("int32")
        temp_h_weight = self.tik_instance.Scalar("float32")
        h_out_index.set_as(core_idx % self.h_out_size)
        h_in_index.set_as(self.index_h_mapping_ub[h_out_index])

        # init ub tensor src_line_1~4
        self.get_src_lines(core_idx, h_in_index)

        # reduceSum horizontal dirction reduceSum
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
                                        "none", ub_cast_line, self.w_ub_1, [0, 0], self.w_out_size, 4, 8)
            self.move_to_gm(self.out_gm, ub_cast_line, [core_idx * self.w_out_size, 0], self.w_out_size)
        else:
            self.move_to_gm(self.out_gm, self.w_ub_1, [core_idx * self.w_out_size, 0], self.w_out_size)

    def get_src_lines(self, core_idx, h_in_index):
        nc_idx = core_idx // self.h_out_size
        index_1 = self.tik_instance.Scalar("int64")
        index_2 = self.tik_instance.Scalar("int64")
        index_3 = self.tik_instance.Scalar("int64")
        index_4 = self.tik_instance.Scalar("int64")
        h_max_index = self.tik_instance.Scalar("int64", init_value=self.h_in_size - 1)

        index_2.set_as(h_in_index)
        index_1.set_as(index_2 - 1)
        self.tik_instance.scalar_max(index_1, 0, index_1)
        self.tik_instance.scalar_min(index_1, h_max_index, index_1)
        index_3.set_as(index_2 + 1)
        self.tik_instance.scalar_min(index_3, h_max_index, index_3)
        index_4.set_as(index_2 + 2)
        self.tik_instance.scalar_min(index_4, h_max_index, index_4)
        self.tik_instance.scalar_max(index_2, 0, index_2)
        self.tik_instance.scalar_min(index_2, h_max_index, index_2)

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
        # obtain ll, l, r, rr item of whole line seperately
        w_in_index_ll = self.tik_instance.Scalar("int64")
        w_in_index_l = self.tik_instance.Scalar("int64")
        w_in_index_r = self.tik_instance.Scalar("int64")
        w_in_index_rr = self.tik_instance.Scalar("int64")
        temp_index = self.tik_instance.Scalar("int64")
        load_index = self.tik_instance.Scalar("int32")
        w_max_index = self.tik_instance.Scalar("int64", init_value=self.w_in_size - 1)
        with self.tik_instance.for_range(0, self.w_out_size) as w_out_index:
            load_index.set_as(self.index_w_mapping_ub[w_out_index])
            w_in_index_l.set_as(load_index)
            temp_index.set_as(w_in_index_l - 1)
            self.tik_instance.scalar_max(w_in_index_ll, 0, temp_index)
            self.tik_instance.scalar_min(w_in_index_ll, w_max_index, w_in_index_ll)
            temp_index.set_as(w_in_index_l + 1)
            self.tik_instance.scalar_min(w_in_index_r, w_max_index, temp_index)
            temp_index.set_as(w_in_index_l + 2)
            self.tik_instance.scalar_min(w_in_index_rr, w_max_index, temp_index)
            self.tik_instance.scalar_max(w_in_index_l, 0, w_in_index_l)
            self.tik_instance.scalar_min(w_in_index_l, w_max_index, w_in_index_l)

            self.w_ub_1[w_out_index] = self.src_line_1[w_in_index_ll]
            self.w_ub_2[w_out_index] = self.src_line_1[w_in_index_l]
            self.w_ub_3[w_out_index] = self.src_line_1[w_in_index_r]
            self.w_ub_4[w_out_index] = self.src_line_1[w_in_index_rr]

    # 'pylint: disable=too-many-arguments
    def conv_operator_template(self, op_obj, mode, dst, src, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        vector_mask_max = Constant.VECTOR_MASK_MAX
        dst_offset, src_offset = offsets

        tensor_size = src.size
        with self.tik_instance.if_scope(num > 0):
            tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, mode, dst[tmp_dst_offset], src[tmp_src_offset], 255,
                       dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, mode, dst[dst_offset], src[src_offset], repeat_time, dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)

    # 'pylint: disable=too-many-arguments
    def scalar_operator_template(self, op_obj, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        vector_mask_max = Constant.VECTOR_MASK_MAX
        dst_offset, src_offset = offsets


        tensor_size = src.size
        with self.tik_instance.if_scope(num > 0):
            tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src[tmp_src_offset], scalar, 255,
                       dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset], src[src_offset], scalar, repeat_time, dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset], src[src_offset], scalar, 1, dst_stride, src_stride)

    # 'pylint: disable=too-many-arguments
    def double_operator_template(self, op_obj, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8,
                                 src1_stride=8):
        """
        tik api template
        """
        vector_mask_max = Constant.VECTOR_MASK_MAX
        dst_offset, src0_offset, src1_offset = offsets

        tensor_size = src1.size
        with self.tik_instance.if_scope(num > 0):
            tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_instance.if_scope(loop > 0):
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

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset], src0[src0_offset], src1[src1_offset], repeat_time, dst_stride,
                   src0_stride, src1_stride)
            dst_offset += repeat_time * vector_mask_max
            src0_offset += repeat_time * vector_mask_max
            src1_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1, dst_stride, src0_stride,
                   src1_stride)

    # 'pylint: disable=too-many-arguments
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

    # 'pylint: disable=too-many-arguments
    def move_to_gm(self, dst, src, offsets, num, nburst=1, src_stride=0, dst_stride=0):
        """
        move data from ub to gm
        """
        dst_offset, src_offset = offsets
        sid = 0
        burst_len = num // self.x_data_each_block
        with self.tik_instance.if_scope(burst_len > 0):
            self.tik_instance.data_move(dst[dst_offset],
                                        src[src_offset],
                                        sid,
                                        nburst,
                                        burst_len,
                                        src_stride=src_stride,
                                        dst_stride=dst_stride)

        last_num = num % self.x_data_each_block
        with self.tik_instance.if_scope(last_num > 0):
            tail_block = self.tik_instance.Tensor(self.x_dtype, [self.x_data_each_block],
                                                  name="tail_block", scope=tik.scope_ubuf)
            if self.x_dtype == "float32":
                self.tik_instance.set_atomic_add(1)
                self.data_move(tail_block, self.zero_value_gm, [0, 0], 1)
                last_block_start = src_offset + burst_len * self.x_data_each_block
                with self.tik_instance.for_range(0, last_num) as i:
                    tail_block[i].set_as(src[last_block_start + i])

                self.tik_instance.data_move(dst[dst_offset + burst_len * self.x_data_each_block],
                                            tail_block[0],
                                            sid,
                                            nburst,
                                            1,
                                            src_stride=src_stride,
                                            dst_stride=dst_stride)
                self.tik_instance.set_atomic_add(0)
            else:
                with self.tik_instance.if_scope(burst_len > 0):
                    last_block_start = src_offset + num - self.x_data_each_block
                    with self.tik_instance.for_range(0, self.x_data_each_block) as i:
                        tail_block[i].set_as(src[last_block_start + i])
                    self.tik_instance.data_move(dst[dst_offset + num - self.x_data_each_block],
                                                tail_block[0],
                                                sid,
                                                nburst,
                                                1,
                                                src_stride=src_stride,
                                                dst_stride=dst_stride)
                with self.tik_instance.else_scope():
                    dst_offset = dst_offset + num - self.x_data_each_block
                    self.tik_instance.data_move(tail_block[0],
                                                dst[dst_offset],
                                                sid,
                                                nburst,
                                                1,
                                                src_stride=src_stride,
                                                dst_stride=dst_stride)
                    with self.tik_instance.for_range(0, last_num) as i:
                        tail_block[self.x_data_each_block - last_num + i].set_as(src[i])
                    self.tik_instance.data_move(dst[dst_offset],
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
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.x_dtype = x.get("dtype")
        self.block_byte_size = 32
        self.coordinate_transformation_mode = coordinate_transformation_mode
        self.kernel_name = kernel_name

        self.init_tiling_gm_and_scalar()
        self.get_tiling_data()

        self.x_gm = self.tik_instance.Tensor(self.x_dtype, [Constant.MAX_INT64],
                                             name="x_gm",
                                             scope=tik.scope_gm)

        self.output_gm = self.tik_instance.Tensor(self.x_dtype, [Constant.MAX_INT64],
                                                  name="output_gm",
                                                  scope=tik.scope_gm)


    @staticmethod
    def get_dtype_size(dtype):
        """
        :param dtype: data type
        :return:
        """
        dtype_byte_size = get_bit_len(dtype) // 8
        return dtype_byte_size

    def init_tiling_gm_and_scalar(self):
        """
        init tiling gm and scalar
        """
        self.tiling_gm = self.tik_instance.Tensor("float32", [Constant.TILING_ARG_NUM],
                                                   name="tiling_gm", scope=tik.scope_gm)

        self.dim0 = self.tik_instance.Scalar("int32", "dim0")
        self.dim1 = self.tik_instance.Scalar("int32", "dim1")
        self.dim_redundancy = self.tik_instance.Scalar("int32", "dim_redundancy")
        self.dim2 = self.tik_instance.Scalar("int32", "dim2")
        self.size = self.tik_instance.Scalar("int32", "size")
        self.scale_w = self.tik_instance.Scalar("float32", "scale_w")
        self.data_each_block = self.tik_instance.Scalar("int32", "data_each_block")

    def get_tiling_data(self):
        """
        obtaion tiling info from gm
        """
        self.tiling_ub = self.tik_instance.Tensor("float32", [Constant.TILING_ARG_NUM],
                                                   name="tiling_ub", scope=tik.scope_ubuf)
        self.data_move(self.tiling_ub, self.tiling_gm, [0, 0], Constant.TILING_ARG_NUM)
        self.dim0.set_as(self.tiling_ub[0])
        self.dim1.set_as(self.tiling_ub[1])
        self.dim_redundancy.set_as(self.tiling_ub[2])
        self.dim2.set_as(self.tiling_ub[3])
        self.size.set_as(self.tiling_ub[5])
        if self.coordinate_transformation_mode == "align_corners":
            self.scale_w.set_as(self.tiling_ub[9])
        else:
            self.scale_w.set_as(self.tiling_ub[11])
        self.data_each_block.set_as(self.tiling_ub[13])

        dim0_int64 = self.tik_instance.Scalar("int64", name = "dim0_int64", init_value = self.dim0)
        dim1_int64 = self.tik_instance.Scalar("int64", name = "dim1_int64", init_value = self.dim1)
        dim2_int64 = self.tik_instance.Scalar("int64", name = "dim2_int64", init_value = self.dim2)
        size_int64 = self.tik_instance.Scalar("int64", name = "size_int64", init_value = self.size)
        self.input_num = dim0_int64 * dim1_int64 * dim2_int64
        self.output_num = dim0_int64 * dim1_int64 * size_int64

    # 'pylint: disable=too-many-locals, too-many-branches
    def resize_linear_compute(self):
        """
        ResizeLinear main logic
        """
        self.x_gm.reshape([
            self.input_num,
        ])

        with self.tik_instance.if_scope(self.output_num <= self.data_each_block):
            res_lastdim_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block],
                                                      name="res_lastdim_ub",
                                                      scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.dim0) as i:
                with self.tik_instance.for_range(0, self.dim1) as j:
                    current_index_output = self.tik_instance.Scalar("int32",
                                                                    init_value=i * (self.dim1 * self.size) +
                                                                               j * self.size)
                    with self.tik_instance.for_range(0, self.size) as k:
                        res_lastdim_ub[current_index_output + k].set_as(
                            self.compute_helper(self.scale_w, k, i * (self.dim1 * self.dim2) + j * self.dim2))

            self.tik_instance.data_move(self.output_gm, res_lastdim_ub, 0, 1, 1, 0, 0)

        with self.tik_instance.elif_scope(self.size < self.data_each_block):
            loop_time = self.output_num // self.data_each_block
            with self.tik_instance.for_range(0, loop_time) as i:
                res_lastdim_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block],
                                                          name="res_lastdim_ub",
                                                          scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, self.data_each_block) as j:
                    current_index = i * self.data_each_block + j
                    current_dim1 = current_index // self.size
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
                    res_lastdim_ub[k].set_as(
                        self.compute_helper(self.scale_w, current_index % self.size, current_dim1 * self.dim2))
                self.tik_instance.data_move(self.output_gm[remainder_begin_index], res_lastdim_ub, 0, 1, 1, 0, 0)

        with self.tik_instance.else_scope():
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
                            res_lastdim_remainder_ub[k].set_as(
                                self.compute_helper(self.scale_w, remainder_begin_index + k,
                                                    i * (self.dim1 * self.dim2) + j * self.dim2))
                        self.tik_instance.data_move(
                            self.output_gm[i * (self.dim1 * self.size) + (j + 1) * self.size - self.data_each_block],
                            res_lastdim_remainder_ub, 0, 1, 1, 0, 0)

        self.output_gm.reshape([self.dim0, self.dim1, 1, self.size])

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.x_gm],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm])
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
        max_offset = self.tik_instance.Scalar("int64", name="max_offset")
        temp_value = self.tik_instance.Scalar("int64", name="temp_value",
                                              init_value=self.input_num - self.data_each_block)
        self.tik_instance.scalar_max(max_offset, temp_value, 0)

        x_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block], name="x_ub", scope=tik.scope_ubuf)

        res = self.tik_instance.Scalar(self.x_dtype, name="res")

        index = self.tik_instance.Scalar("int64", init_value=index)

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

        index = self.tik_instance.Scalar("int64", init_value=input_dim_offset + coefficient_w)
        temp2 = self.tik_instance.Scalar(self.x_dtype, init_value=self.get_number_in_global_memory(index))

        offset = self.tik_instance.Scalar(dtype="int64", init_value=1)
        with self.tik_instance.if_scope(coefficient_w == (self.dim2 - 1)):
            offset.set_as(0)

        temp4 = self.tik_instance.Scalar(self.x_dtype, init_value=self.get_number_in_global_memory(offset + index))

        res = self.tik_instance.Scalar(dtype=self.x_dtype,
                                       init_value=(coefficient_lambda0 * temp2 + coefficient_lambda * temp4))

        return res

    # 'pylint: disable=too-many-arguments
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


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_FLOAT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
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
             data_format="NCHW",
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
    data_format : str
        NCHW or HWNC, default = "NCHW".
    kernel_name : str
        kernel name, default value is "resize_d"

    Returns
    -------
    None
    """
    check_tuple = ("float32", "float16", "bfloat16")
    input_data_type = x.get("dtype").lower()
    para_check.check_dtype_rule(input_data_type, check_tuple, "DataType")
    x_dim = len(x.get("shape"))
    if mode == "cubic":
        if data_format == "HWNC":
            resize_bicubic_instance = ResizeBicubicHWNC(x,
                                                        sizes,
                                                        scales,
                                                        coordinate_transformation_mode,
                                                        cubic_coeff_a,
                                                        kernel_name)
        else:
            resize_bicubic_instance = ResizeBicubic(x,
                                                    sizes,
                                                    scales,
                                                    coordinate_transformation_mode,
                                                    cubic_coeff_a,
                                                    kernel_name)
        res = resize_bicubic_instance.resize_bicubic_compute()
    elif mode == "linear":
        resize_linear = ResizeLinear(x, sizes, scales, coordinate_transformation_mode, kernel_name)
        res = resize_linear.resize_linear_compute()
    else:
        raise RuntimeError("Not supported at the moment.")
    return res
