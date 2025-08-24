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
resize_grad_d
"""

import functools as fctool
import math
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
import numpy as np

SHAPE_SIZE_LIMIT = 2147483648
ERROR_TOLERANCE = 0.0001
UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)


# 'pylint: disable=unused-argument
# 'pylint: disable=too-many-arguments
# 'pylint: disable=too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_FLOAT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def resize_grad_d(grads, y, original_size, roi, scales, coordinate_transformation_mode="half_pixel",
                  cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0.0,
                  mode="nearest", nearest_mode="round_prefer_floor", kernel_name="resize_grad_d"):
    """
    Interface of resize_grad_d(TBE opType),it`s corresponding interface is upsample_bicubic2d_backward(PyTorch)
    Parameters
    ----------
    grads : dict
    shape and dtype of input
    y : dict
    shape and dtype of output
    roi : list_float
    1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X.
    The RoIs' coordinates are normalized in the coordinate system of the input image.
    It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"

    original_size : list_int
    shape of original_image

    scales : list_float
    The scale array along each dimension.
    It takes value greater than 0.
    If it's less than 1, it's sampling down, otherwise, it's upsampling.
    The number of elements of 'scales' should be the same as the rank of input 'X'.
    Only one of 'scales' and 'sizes' can be specified. If 'size' is specified,
    then set scales to empty data (zero shape) in this operator's input list.

    coordinate_transformation_mode : str
    This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the
    original tensor.

    cubic_coeff_a : float
    The coefficient 'a' used in cubic interpolation.
    Two common choice are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch).
    This attribute is valid only if "mode" is "cubic".

    exclude_outside : int
    default is 0.
    If set to 1, the weight of sampling locations outside the tensor will be set to 0
    and the weight will be renormalized so that their sum is 1.0. The default value is 0.

    exclude_outside : int
    default is 0.
    If set to 1, the weight of sampling locations outside the tensor will be set to 0
    and the weight will be renormalized so that their sum is 1.0. The default value is 0.

    extrapolation_value : float (default is 0.0)
    When coordinate_transformation_mode is "tf_crop_and_resize" and x_original is outside
    the range [0, length_original - 1], this value is used as the corresponding output value.
    Default is 0.0f.

    mode : string
    default is nearest
    Three interpolation modes: nearest (default), linear and cubic.
    The "linear" mode includes linear interpolation for 1D tensor and N-linear interpolation
    for N-D tensor (for example, bilinear interpolation for 2D tensor).

    The "cubic" mode includes cubic interpolation for 1D tensor and N-cubic interpolation
    for N-D tensor (for example, bicubic interpolation for 2D tensor).
    nearest_mode : string (default is round_prefer_floor)
    Four modes: round_prefer_floor (default, as known as round half down),
    round_prefer_ceil (as known as round half up), floor, ceil.
    Only used by nearest interpolation.

    It indicates how to get "nearest" pixel in input tensor from x_original,
    so this attribute is valid only if "mode" is "nearest".

    kernel_name : str
    kernel name, default value is "resize_grad_d"

    Returns
    -------
    tik_instance
    """
    x_dim = len(grads.get("shape"))
    shape_size = len(original_size)
    if mode == "cubic" and shape_size == 4:
        shape_grads = shape_util.scalar2tensor_one(grads.get("shape"))
        para_check.check_shape_size(shape_grads, SHAPE_SIZE_LIMIT)
        check_tuple = ("float32", "float16")
        input_data_type1 = grads.get("dtype").lower()
        para_check.check_dtype_rule(input_data_type1, check_tuple)
        para_check.check_kernel_name(kernel_name)
        upsamplebicubic2d_backward_instance = UpSampleBicubic2dBackward(
            grads, original_size, scales,
            coordinate_transformation_mode,
            cubic_coeff_a,
            kernel_name=kernel_name)
        res = upsamplebicubic2d_backward_instance.upsamplebicubic2d_backward_compute()
    elif mode == "linear" and x_dim == 4:
        resize_linear = ResizeLinearBackward(grads,
                                             original_size,
                                             scales,
                                             coordinate_transformation_mode,
                                             kernel_name)
        res = resize_linear.resize_linear_compute()
    else:
        raise RuntimeError("Upsample Not supported.")
    return res


class UpSampleBicubic2dBackward:
    """
    Class UpSampleBicubic2dBackward
    Main part of op ResizeGradD(op_type).Upsample_bicubic2d_backward(pytorch interface)
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self, grads, original_size, scales, coordinate_transformation_mode,
                 cubic_coeff_a, kernel_name="resize_grad_d"):
        """
        Init parameters.
        -------
        tik_instance
        """
        self.kernel_name = kernel_name
        self.align_corners = (coordinate_transformation_mode == "align_corners")
        self.shape_grads = grads.get("shape")
        self.dtype_grads = grads.get("dtype")
        self.tik_instance = tik.Tik(tik.Dprofile())

        self.batch_size = original_size[0]
        self.c_size = original_size[1]
        self.in_size_h = original_size[2]
        self.in_size_w = original_size[3]
        self.nc = self.batch_size * self.c_size  # n*c of input in upsample_bicubic2d

        self.out_size_h = self.shape_grads[2]
        self.out_size_w = self.shape_grads[3]

        self.scale_h = \
            self.area_pixel_compute_scale(self.in_size_h, self.out_size_h,
                                          self.align_corners, scales[0])
        self.scale_w = \
            self.area_pixel_compute_scale(self.in_size_w, self.out_size_w,
                                          self.align_corners, scales[1])
        self.cubic_coeff_a = cubic_coeff_a
        self.scales = scales
        block_bite_size = 32

        ub_size_bytes = tbe_platform.get_soc_spec("UB_SIZE")

        dtype_bytes_size = tbe_platform.get_bit_len(
            self.dtype_grads) // 8
        self.data_each_block = block_bite_size // dtype_bytes_size

        self.ub_tensor_size = \
            ub_size_bytes // dtype_bytes_size // self.data_each_block * self.data_each_block

        self.input_num = fctool.reduce(lambda x, y: x * y, self.shape_grads)
        self.output_num = fctool.reduce(lambda x, y: x * y, original_size)

        self.pos_max_in = self.input_num - self.data_each_block
        self.pos_max_out = self.output_num - self.data_each_block

        if self.input_num < self.data_each_block:
            self.pos_max_in = 0
        if self.output_num < self.data_each_block:
            self.pos_max_out = 0

        self.vector_mask_max = 8 * self.data_each_block

        self.input_grads_gm = self.tik_instance.Tensor(
            self.dtype_grads, self.shape_grads, name="input_grads_gm", scope=tik.scope_gm)

        self.output_gm = self.tik_instance.Tensor(
            self.dtype_grads, original_size, name="output_gm", scope=tik.scope_gm, is_atomic_add=True)

    def upsamplebicubic2d_backward_compute(self):
        """
        There are two cases for upsample,one is input image(H axis & W axis) has the same
        size as output, just copy input to output; another is input image do not has
        the same size as output,do cubic interpolation sampling.
        Returns
        -------
        tik_instance
        """
        if self.in_size_h == self.out_size_h and self.in_size_w == self.out_size_w:
            self.upsamplebicubic2d_backward_compute_same_size()
        else:
            self.upsamplebicubic2d_backward_compute_general()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_grads_gm, ],
            outputs=[self.output_gm, ])

        return self.tik_instance

    # special case ,input has the same size as output just copy
    def upsamplebicubic2d_backward_compute_same_size(self):
        """
        Special case, input and output images have the same size,just copy.
        Returns
        -------
        None
        """
        input_grads_ub = self.tik_instance.Tensor(self.dtype_grads, (self.ub_tensor_size,),
                                                  name="input_grads_ub", scope=tik.scope_ubuf)
        loop_time = self.input_num // self.ub_tensor_size
        burst_len = math.ceil(self.ub_tensor_size / self.data_each_block)
        with self.tik_instance.if_scope(loop_time > 0):
            with self.tik_instance.for_range(0, loop_time) as loop:
                move_offset = loop * self.ub_tensor_size
                self.tik_instance.data_move(input_grads_ub, self.input_grads_gm[move_offset],
                                            0, 1, burst_len, 0, 0)
                self.tik_instance.data_move(self.output_gm[move_offset], input_grads_ub,
                                            0, 1, burst_len, 0, 0)
            move_offset = loop_time * self.ub_tensor_size

        last_num = self.input_num % self.ub_tensor_size
        if last_num > 0:
            self.tik_instance.data_move(input_grads_ub, self.input_grads_gm[move_offset],
                                        0, 1, math.ceil(last_num / self.data_each_block), 0, 0)
            self.tik_instance.data_move(self.output_gm[move_offset], input_grads_ub,
                                        0, 1, math.ceil(last_num / self.data_each_block), 0, 0)

    # 'pylint: disable=too-many-locals
    def upsamplebicubic2d_backward_compute_general(self):
        """
        Main compute logic of upsample_bicubic2d_backward(PyTorch Interface, TBE op_name is ResizeGradD)
        for genernal case.
        Returns
        -------
        None
        """
        core_num = self.nc if self.dtype_grads == "float32" else 1
        with self.tik_instance.for_range(0, self.nc, block_num=core_num) as nc:

            input_grads_ub = self.tik_instance.Tensor(self.dtype_grads, (self.data_each_block,),
                                                      name="input_grads_ub", scope=tik.scope_ubuf)

            output_ub = self.tik_instance.Tensor(self.dtype_grads, (self.data_each_block,),
                                                 name="output_ub", scope=tik.scope_ubuf)

            value_ub = self.tik_instance.Tensor(self.dtype_grads, (1,),
                                                name="value_ub", scope=tik.scope_ubuf)

            value_temp = self.tik_instance.Tensor("float32", (self.data_each_block,),
                                                  name="value_temp", scope=tik.scope_ubuf)

            assist1_ub = self.tik_instance.Tensor("float32", (1,),
                                                  name="assist1_ub", scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, self.out_size_h) as output_y:
                with self.tik_instance.for_range(0, self.out_size_w) as output_x:

                    src_scalar = self.tik_instance.Scalar(dtype="int32", init_value=output_x)
                    dst_scalar_output_x = self.tik_instance.Scalar(dtype="float32")
                    self.tik_instance.scalar_conv('none', dst_scalar_output_x, src_scalar)

                    src_scalar = self.tik_instance.Scalar(dtype="int32", init_value=output_y)
                    dst_scalar_output_y = self.tik_instance.Scalar(dtype="float32")
                    self.tik_instance.scalar_conv('none', dst_scalar_output_y, src_scalar)

                    real_x = self.area_pixel_compute_source_index(self.scale_w,
                                                                  dst_scalar_output_x,
                                                                  self.align_corners)

                    src_scalar = self.tik_instance.Scalar(dtype="float32", init_value=real_x)
                    dst_scalar_input_x = self.tik_instance.Scalar(dtype="int32")
                    self.tik_instance.scalar_conv('floor', dst_scalar_input_x, src_scalar)

                    real_y = self.area_pixel_compute_source_index(self.scale_h,
                                                                  dst_scalar_output_y,
                                                                  self.align_corners)

                    src_scalar = self.tik_instance.Scalar(dtype="float32", init_value=real_y)
                    dst_scalar_input_y = self.tik_instance.Scalar(dtype="int32")
                    self.tik_instance.scalar_conv('floor', dst_scalar_input_y, src_scalar)

                    # Corresponding to the interpolation process in upsample_bicubic2d
                    x_coeffs_ub = self.get_cubic_upsample_coefficients(dst_scalar_output_x,
                                                                       dst_scalar_input_x,
                                                                       self.in_size_w,
                                                                       self.out_size_w, 0)

                    y_coeffs_ub = self.get_cubic_upsample_coefficients(dst_scalar_output_y,
                                                                       dst_scalar_input_y,
                                                                       self.in_size_h,
                                                                       self.out_size_h, 1)

                    x_coeffs_scalar = self.tik_instance.Scalar(dtype="float32")
                    y_coeffs_scalar = self.tik_instance.Scalar(dtype="float32")

                    # Move to next channel
                    input_ptr = nc * self.out_size_w * self.out_size_h
                    output_ptr = nc * self.in_size_w * self.in_size_h

                    offset = output_y * self.out_size_w + output_x
                    input_pos = input_ptr + offset

                    with self.tik_instance.if_scope(input_pos > self.pos_max_in):
                        self.tik_instance.data_move(input_grads_ub,
                                                    self.input_grads_gm[self.pos_max_in],
                                                    0, 1, 1, 0, 0)
                        value_ub[0].set_as(
                            input_grads_ub[input_pos - self.pos_max_in])
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(input_grads_ub,
                                                    self.input_grads_gm[input_pos],
                                                    0, 1, 1, 0, 0)
                        value_ub[0].set_as(input_grads_ub[0])

                    with self.tik_instance.for_range(0, 4) as i:
                        with self.tik_instance.for_range(0, 4) as j:

                            x_coeffs_scalar.set_as(x_coeffs_ub[i])
                            y_coeffs_scalar.set_as(y_coeffs_ub[j])

                            self.tik_instance.vec_dup(self.data_each_block, value_temp, 0, 1, 1)
                            # Convert fp16 to fp32 to improve accuracy
                            if self.dtype_grads == "float16":
                                self.tik_instance.vec_conv(
                                    1, 'none', assist1_ub, value_ub, 1, 1, 1)
                                self.tik_instance.vec_muls(1, value_temp,
                                                           assist1_ub, x_coeffs_scalar, 1, 1, 1)
                                self.tik_instance.vec_muls(1, value_temp,
                                                           value_temp, y_coeffs_scalar, 1, 1, 1)
                            else:

                                self.tik_instance.vec_muls(1, value_temp,
                                                           value_ub, x_coeffs_scalar, 1, 1, 1)
                                self.tik_instance.vec_muls(1, value_temp,
                                                           value_temp, y_coeffs_scalar, 1, 1, 1)

                            self.upsample_increment_value_bounded(
                                output_ub,
                                output_ptr,
                                self.in_size_w,
                                self.in_size_h,
                                dst_scalar_input_x - 1 + i,
                                dst_scalar_input_y - 1 + j,
                                value_temp
                            )

    def area_pixel_compute_scale(self, input_size, output_size, align_corners, scale):
        """
        Compute scalses
        Returns
        -------
        float
        """
        res = 0
        if output_size > 1:
            if align_corners:
                res = (input_size - 1) / (output_size - 1)
            else:
                res = self.compute_scales_value(scale, input_size, output_size)
        return res

    @staticmethod
    def compute_scales_value(scale, input_size, output_size):
        """
        Compute scalses
        Returns
        -------
        float
        """
        if scale > 0.0:
            return 1.0 / scale
        return input_size / output_size

    # 'pylint: disable=too-many-arguments
    # 'pylint: disable=too-many-locals
    @staticmethod
    def area_pixel_compute_source_index(scale, dst_index, align_corners):
        """
        Compute index in input image according to scale and index of output image
        :param scale : float
        :param dst_index : int. Index of output image
        :param align_corners : bool
        Returns
        -------
        float
        """
        if align_corners:
            return scale * dst_index * 1.0
        return scale * (dst_index + 0.5) - 0.5

    def get_cubic_upsample_coefficients(self, output_scalar, input_x, in_length, out_length, flag):
        """
        Corresponding to the interpolation process of upsample_bicubic2d

        :param output_scalar: int (output index)
        :param input_x: int (Integer part of real_x or real_y)
        :param in_length: int (in_size_w or in_size_h)
        :param out_length: int (out_size_w or out_size_h)
        :param flag: int
        Returns
        ----------
        :return: tensor shape[4,].
        """
        coeffs_ub = self.tik_instance.Tensor("float32", (4,),
                                             name="coeffs_ub", scope=tik.scope_ubuf)
        temp_scalar = self.tik_instance.Scalar(dtype="int32", init_value=input_x)
        cast_input_index = self.tik_instance.Scalar(dtype="float32")
        self.tik_instance.scalar_conv('none', cast_input_index, temp_scalar)
        input_index_scalar = self.tik_instance.Scalar(dtype="float32", init_value=cast_input_index)
        one_length = self.tik_instance.Scalar(dtype="float32")
        two_length = self.tik_instance.Scalar(dtype="float32")
        three_length = self.tik_instance.Scalar(dtype="float32")
        x1 = self.tik_instance.Scalar(dtype="float32")
        a = self.cubic_coeff_a  # -0.75
        if flag == 0:
            scale = self.scales[1]  # x direction
        else:
            scale = self.scales[0]  # y direction
        if out_length > 1:
            if self.align_corners:
                one_length.set_as(out_length - 1)
                two_length.set_as((out_length - 1) * (out_length - 1))
                three_length.set_as((out_length - 1) *
                                    (out_length - 1) * (out_length - 1))
                x1.set_as(output_scalar * (in_length - 1.0) -
                          input_index_scalar * one_length)
            else:
                if scale > 0.0:
                    one_length.set_as(scale)
                    two_length.set_as(scale * scale)
                    three_length.set_as(scale * scale * scale)
                    x1.set_as((output_scalar + 0.5) -
                              (0.5 + input_index_scalar) * scale)
                else:
                    one_length.set_as(out_length)
                    two_length.set_as(out_length * out_length)
                    three_length.set_as(out_length * out_length * out_length)
                    x1.set_as((output_scalar + 0.5) * in_length -
                              (0.5 + input_index_scalar) * out_length)
            coeffs_ub[0].set_as((((a * (x1 + one_length) - 5.0 * a * one_length) * (x1 + one_length) +
                                  8.0 * a * two_length) * (x1 + one_length) - 4.0 * a * three_length) / three_length)
            coeffs_ub[1].set_as((((a + 2.0) * x1 - (a + 3.0) * one_length)
                                 * x1 * x1 + 1.0 * three_length) / three_length)
            x2 = one_length - x1
            coeffs_ub[2].set_as((((a + 2.0) * x2 - (a + 3.0) * one_length)
                                 * x2 * x2 + 1.0 * three_length) / three_length)
            coeffs_ub[3].set_as((((a * (x2 + one_length) - 5.0 * a * one_length) * (x2 + one_length) +
                                  8.0 * a * two_length) * (x2 + one_length) - 4.0 * a * three_length) / three_length)

        else:
            coeffs_ub[0].set_as(0.0)
            coeffs_ub[1].set_as(1.0)
            coeffs_ub[2].set_as(0.0)
            coeffs_ub[3].set_as(0.0)
        return coeffs_ub

    # 'pylint: disable=too-many-arguments
    # 'pylint: disable=too-many-locals
    def upsample_increment_value_bounded(self, output_ub, output_pos, width, height, x, y, value):
        """
        Compute upsample increment value bounded
        Returns
        -------
        None
        """
        x_scalar = self.tik_instance.Scalar(dtype="int64", init_value=x)
        y_scalar = self.tik_instance.Scalar(dtype="int64", init_value=y)
        w_scalar = self.tik_instance.Scalar(
            dtype="int64", init_value=width - 1)
        h_scalar = self.tik_instance.Scalar(
            dtype="int64", init_value=height - 1)
        temp_ub = self.tik_instance.Tensor(self.dtype_grads, (1,),
                                           name="temp_ub", scope=tik.scope_ubuf)
        assist2_ub = self.tik_instance.Tensor("float32", (1,),
                                              name="assist2_ub", scope=tik.scope_ubuf)

        a_x = self.tik_instance.Scalar(dtype="int64")
        a_y = self.tik_instance.Scalar(dtype="int64")
        temp_x = self.tik_instance.Scalar(dtype="int64")
        temp_y = self.tik_instance.Scalar(dtype="int64")
        self.tik_instance.scalar_min(temp_x, x_scalar, w_scalar)
        self.tik_instance.scalar_min(temp_y, y_scalar, h_scalar)
        self.tik_instance.scalar_max(a_x, temp_x, 0)
        self.tik_instance.scalar_max(a_y, temp_y, 0)

        offset = output_pos + a_y * width + a_x

        with self.tik_instance.if_scope(offset > self.pos_max_out):
            self.tik_instance.data_move(output_ub,
                                        self.output_gm[self.pos_max_out],
                                        0, 1, 1, 0, 0)
            pos_in_output_ub = offset - self.pos_max_out

            # Convert fp16 to fp32 to improve accuracy
            if self.dtype_grads == "float16":
                temp_ub[0].set_as(output_ub[pos_in_output_ub])

                self.tik_instance.vec_conv(
                    1, 'none', assist2_ub, temp_ub, 1, 1, 1)

                self.tik_instance.vec_add(1, assist2_ub,
                                          assist2_ub, value, 1, 1, 1, 1)

                self.tik_instance.vec_conv(
                    1, 'none', temp_ub, assist2_ub, 1, 1, 1)

                output_ub[pos_in_output_ub].set_as(temp_ub[0])
                self.tik_instance.data_move(self.output_gm[self.pos_max_out],
                                            output_ub,
                                            0, 1, 1, 0, 0)
            else:
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.output_gm[self.pos_max_out + pos_in_output_ub],
                                            value, 0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(0)

        with self.tik_instance.else_scope():

            # Convert fp16 to fp32 to improve accuracy
            if self.dtype_grads == "float16":
                self.tik_instance.data_move(output_ub,
                                            self.output_gm[offset],
                                            0, 1, 1, 0, 0)
                self.tik_instance.vec_conv(
                    1, 'none', assist2_ub, output_ub, 1, 1, 1)

                self.tik_instance.vec_add(1, assist2_ub,
                                          assist2_ub, value, 1, 1, 1, 1)

                self.tik_instance.vec_conv(
                    1, 'none', output_ub, assist2_ub, 1, 1, 1)

                self.tik_instance.data_move(self.output_gm[offset],
                                            output_ub,
                                            0, 1, 1, 0, 0)
            else:
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.output_gm[offset],
                                            value, 0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(0)


class ResizeLinearBackward:
    """
    ResizeLinearBackward main functions
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self,
                 x,
                 original_size,
                 scales,
                 coordinate_transformation_mode="half_pixel",
                 kernel_name="resize_grad_d"):

        self.tik_instance = tik.Tik(tik.Dprofile())

        if len(original_size) != 3:
            raise RuntimeError("It is expected input_size equals to 3.")

        self.x_dtype = x.get("dtype")
        self.x_shape = x.get("shape")
        self.size = original_size[-1]
        self.scale = scales[0]
        self.dim0 = self.x_shape[0]
        self.dim1 = self.x_shape[1]
        self.dim_redundancy = self.x_shape[2]
        self.dim2 = self.x_shape[3]
        self.input_num = self.dim0 * self.dim1 * self.dim2
        self.coordinate_transformation_mode = coordinate_transformation_mode
        self.kernel_name = kernel_name

        self.check_param1(self.dim_redundancy, self.dim2, self.size)
        self.check_param2(original_size, scales)

        if self.dim2 > 1:
            if self.coordinate_transformation_mode == "align_corners":
                self.scale_w = (self.size - 1) / (self.dim2 - 1)
            else:
                self.scale_w = 1.0 / self.scale if self.scale > 0. else (self.size / self.dim2)
        else:
            self.scale_w = 0

        self.output_num = self.dim0 * self.dim1 * self.size

        block_bite_size = 32
        dtype_bytes_size = tbe_platform.get_bit_len(self.x_dtype) // 8
        self.data_each_block = block_bite_size // dtype_bytes_size

        self.x_gm = self.tik_instance.Tensor(self.x_dtype,
                                             (self.dim0, self.dim1, self.dim_redundancy, self.dim2),
                                             name="x_gm",
                                             scope=tik.scope_gm)
        self.x_gm.reshape(self.x_shape)

        self.output_gm = self.tik_instance.Tensor(self.x_dtype,
                                                  (self.dim0, self.dim1, self.dim_redundancy, self.size),
                                                  name="output_gm",
                                                  scope=tik.scope_gm)

    def resize_linear_compute(self):
        """
        ResizeLinearBackward main logic
        """
        self.x_gm.reshape([self.input_num, ])

        self.init_output_gm_as_zero()

        with self.tik_instance.for_range(0, self.dim0) as i:
            with self.tik_instance.for_range(0, self.dim1) as j:
                with self.tik_instance.for_range(0, self.dim2) as k:
                    self.compute_helper_backward(self.scale_w,
                                                 k,
                                                 (i * (self.dim1 * self.dim2)) + (j * self.dim2) + k,
                                                 i * (self.dim1 * self.size) + j * self.size)

        self.output_gm.reshape([self.dim0, self.dim1, 1, self.size])

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.x_gm],
                                   outputs=[self.output_gm])
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

    def compute_helper_backward(self, scale_w, output_block_offset, index_in_gm, input_dim_offset):
        """
        compute helper backward
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

        # Cal Offset
        offset = self.tik_instance.Scalar(dtype="int32", init_value=1)
        with self.tik_instance.if_scope(coefficient_w == (self.size - 1)):
            offset.set_as(0)

        # Cal Decimal of real_w
        coefficient_lambda = self.tik_instance.Scalar("float32", name="coefficient_lambda")
        coefficient_lambda.set_as(real_w - coefficient_w)

        # Cal 1.0 - Decimal of real_w
        coefficient_lambda0 = self.tik_instance.Scalar("float32", name="coefficient_lambda0")
        coefficient_lambda0.set_as(1.0 - coefficient_lambda)

        _x = self.get_number_in_global_memory(index_in_gm)

        self.set_output_as(input_dim_offset + coefficient_w, coefficient_lambda0 * _x)
        self.set_output_as(input_dim_offset + coefficient_w + offset, coefficient_lambda * _x)

    def set_output_as(self, index, num):
        """
        set output
        """

        block_num = index // self.data_each_block

        block_offset = index % self.data_each_block

        temp_ub = self.tik_instance.Tensor(self.x_dtype,
                                           [self.data_each_block, ],
                                           name="temp_ub",
                                           scope=tik.scope_ubuf)

        self.tik_instance.data_move(temp_ub, self.output_gm[block_num * self.data_each_block], 0, 1, 1, 0, 0)

        temp_scalar = self.tik_instance.Scalar(self.x_dtype, init_value=temp_ub[block_offset])

        temp_ub[block_offset].set_as(temp_scalar + num)
        self.tik_instance.data_move(self.output_gm[block_num * self.data_each_block], temp_ub, 0, 1, 1, 0, 0)

    def init_output_gm_as_zero(self):
        """
        Init the output_gm as zero

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        temp_ub = self.tik_instance.Tensor(self.x_dtype,
                                           [self.data_each_block, ],
                                           name="temp_ub",
                                           scope=tik.scope_ubuf)

        self.init_ub_as_zero(temp_ub)

        loop_time = np.ceil(self.output_num / self.data_each_block).astype(np.int32)
        with self.tik_instance.for_range(0, loop_time) as i:
            self.tik_instance.data_move(self.output_gm[i * self.data_each_block], temp_ub, 0, 1, 1, 0, 0)

    def init_ub_as_zero(self, ub):
        """
        construct a zero tensor

        Parameters
        ----------
        ub: Tensor

        Returns
        -------
        None
        """
        temp = self.tik_instance.Scalar(self.x_dtype, init_value=0.0)
        with self.tik_instance.for_range(0, self.data_each_block) as i:
            ub[i].set_as(temp)

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

        # check scales
        if len(scales) != 1 and scales is not None:
            raise RuntimeError("It is expected len(scales) equals to 1.")

        # check scales value
        if scales is not None and (self.dim2 / sizes[-1] - scales[0]) > ERROR_TOLERANCE:
            raise RuntimeError("It is expected scales[0] equals to sizes[0] / x.shape[2].")
