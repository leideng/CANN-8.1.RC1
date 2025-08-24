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
max_pool_with_argmax_resnet50
"""
import math

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl import common_util
from impl import constant_util as constant
from impl.util.platform_adapter import tbe_context

# min value of fp16
MIN_VALUE_FP16 = -65504.0
# define dilation size
DILATION = 1
# parameters for vector instruct
MASK = 128
REPEAT_2 = 2
DSTSTRIDEM0 = 1
SRC0STRIDEM0 = 1
SRC1STRIDEM0 = 1
DSTSTRIDEM1 = 8
SRC0STRIDEM1 = 8
SRC1STRIDEM1 = 8
# get available ub size
UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
# get available l1 size
L1_SIZE = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)


def _ceil_div(value, factor):
    """
    caculate ceil value of div

    Parameters
    ----------
    value: dtype of int or float
        original value
    factor: dtype of int or float
        dividend value

    Returns
    -------
    value: dtype of int or float
    """
    return math.ceil(value / factor)


# 'pylint: disable=locally-disabled, too-many-instance-attributes
# 'pylint: disable=too-few-public-methods
class MaxPoolWithargmaxResnet50():
    """
       Function: use to finish MaxPoolWithargmax main functions
    """

    def __init__(self, input_x, ksize, strides, padding):
        """
        init MaxPoolWithargmax parameters

        Parameters
        ----------
        input_x: dict
            shape and datatype
        ksize: list or tuple
            The size of the window for each dimension of the input tensor.
        strides: list or tuple
            The stride of the sliding window of the input tensor.
        padding: str
            The type of padding algorithm to use.

        Returns
        -------
        None
        """
        self.input_shape = input_x.get("shape")
        self.input_dtype = input_x.get("dtype").lower()
        self.input_type_size = common_util.get_data_size(self.input_dtype)
        self.tik_instance = tik.Tik()

        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.batch_size = self.input_shape[0]
        self.c1_size = self.input_shape[1]
        self.in_size_h = self.input_shape[2]
        self.in_size_w = self.input_shape[3]
        self.c_block_size = self.input_shape[4]

        self.window_h = self.ksize[1]
        self.window_w = self.ksize[2]
        self.stride_h = self.strides[1]
        self.stride_w = self.strides[2]
        self.nc1 = self.batch_size * self.c1_size

        # caculate pad and output size
        self.pad, self.out_size_h, self.out_size_w = \
        self._calc_out_size_and_pad()

    # 'pylint: disable=too-many-locals,too-many-statements
    def tik_instance_function_v1(self, kernel_name):
        """
        implementation of max_pool_with_argmax and return the tik instance
        :param kernel_name: the kernel's name
        :return: tik instance
        """
        dtype = self.input_dtype
        input_shape = self.input_shape
        batch_size = input_shape[0]
        c1_dim = input_shape[1]
        c0_dim = constant.C0_SIZE
        window_h = self.window_h
        window_w = self.window_w
        filter_size = window_h * window_w
        stride_h = self.stride_h
        stride_w = self.stride_w
        output_h = self.out_size_h
        output_w = self.out_size_w
        input_h, input_w = input_shape[2:4]

        instance = self.tik_instance

        mask_one_window = ((output_h * output_w + 15) // 16 + 1) * 16

        block_num = batch_size * c1_dim

        gm_input = instance.Tensor(dtype, [block_num, input_h, input_w, c0_dim],
                                   name="input_fmap_gm", scope=tik.scope_gm)
        gm_output_max = instance.Tensor(dtype, [block_num, output_h, output_w, c0_dim],
                                        name="output_max_gm", scope=tik.scope_gm)
        gm_output_mask = instance.Tensor("uint16", [block_num, filter_size, mask_one_window],
                                         name="output_mask_gm", scope=tik.scope_gm)
        output_h_once = 8  # tiling strategy
        input_h_once = (output_h_once - 1) * stride_h + window_h  # 17
        input_w_once = (output_w - 1) * stride_w + window_w  # 113

        with instance.for_range(0, block_num, block_num=block_num) as block_idx:
            ub_fmatrix_temp = instance.Tensor(dtype, [output_h_once, output_w, c0_dim],
                                              name="ub_fmatrix_temp", scope=tik.scope_ubuf)
            ub_mask_temp = instance.Tensor("uint16", [filter_size, output_h_once, output_w],
                                           name="ub_mask", scope=tik.scope_ubuf)
            ub_mask_or_ping = instance.Tensor("uint16", [output_h_once, output_w],
                                              name="ub_mask_or_ping", scope=tik.scope_ubuf)
            ub_mask_or_pong = instance.Tensor("uint16", [output_h_once, output_w],
                                              name="ub_mask_or_pong", scope=tik.scope_ubuf)
            ub_mask_not_ping = instance.Tensor("uint16", [output_h_once, output_w],
                                               name="ub_mask_not_ping", scope=tik.scope_ubuf)
            ub_mask_not_pong = instance.Tensor("uint16", [output_h_once, output_w],
                                               name="ub_mask_not_pong", scope=tik.scope_ubuf)
            ub_input_ping = instance.Tensor(dtype, [input_h_once, input_w_once, c0_dim],
                                            name="ub_input_ping", scope=tik.scope_ubuf)
            ub_input_pong = instance.Tensor(dtype, [input_h_once, input_w_once, c0_dim],
                                            name="ub_input_pong", scope=tik.scope_ubuf)
            ub_max_ping = instance.Tensor(dtype, [output_h_once, output_w, c0_dim],
                                          name="ub_max_ping", scope=tik.scope_ubuf)
            ub_max_pong = instance.Tensor(dtype, [output_h_once, output_w, c0_dim],
                                          name="ub_max_pong", scope=tik.scope_ubuf)
            ub_mask_ping = instance.Tensor("uint16", [filter_size, output_h_once, output_w],
                                           name="ub_mask_ping", scope=tik.scope_ubuf)
            ub_mask_pong = instance.Tensor("uint16", [filter_size, output_h_once, output_w],
                                           name="ub_mask_pong", scope=tik.scope_ubuf)
            ub_input = ub_input_ping
            ub_max = ub_max_ping
            ub_mask = ub_mask_ping
            ub_input_other = ub_input_pong
            ub_max_other = ub_max_pong
            ub_mask_other = ub_mask_pong

            instance.data_move(dst=ub_input,
                               src=gm_input[block_idx, 0, 0, 0],
                               sid=0,
                               nburst=input_h_once,
                               burst=input_w,
                               src_stride=0,
                               dst_stride=input_w_once - input_w)
            for loop_idx in range(0, output_h // output_h_once):
                # dup min value
                instance.vector_dup(mask=constant.MASK128,
                                    dst=ub_input[0, input_w, 0],
                                    scalar=MIN_VALUE_FP16,
                                    repeat_times=(input_h_once - 1) * c0_dim // 128,
                                    dst_blk_stride=input_w_once * 2,
                                    dst_rep_stride=input_w_once)
                instance.vector_dup(mask=16,
                                    dst=ub_input[16, input_w, 0],
                                    scalar=MIN_VALUE_FP16,
                                    repeat_times=1,
                                    dst_blk_stride=1,
                                    dst_rep_stride=8)

                # compute max value
                # filter 0 and filter 1
                with instance.for_range(0, output_h_once) as looph_idx:
                    instance.vmax(mask=constant.MASK128,
                                  dst=ub_max[looph_idx, 0, 0],
                                  src0=ub_input[looph_idx * stride_h, 0, 0],
                                  src1=ub_input[looph_idx * stride_h, 1, 0],
                                  repeat_times=output_w // 8,
                                  dst_blk_stride=1,
                                  src0_blk_stride=stride_w,
                                  src1_blk_stride=stride_w,
                                  dst_rep_stride=8,
                                  src0_rep_stride=8 * stride_w,
                                  src1_rep_stride=8 * stride_w)
                # filter 4 and filter 5
                with instance.for_range(0, output_h_once) as looph_idx:
                    instance.vmax(mask=constant.MASK128,
                                  dst=ub_fmatrix_temp[looph_idx, 0, 0],
                                  src0=ub_input[looph_idx * stride_h + 4 // window_w, 4 % window_w, 0],
                                  src1=ub_input[looph_idx * stride_h + 5 // window_w, 5 % window_w, 0],
                                  repeat_times=output_w // 8,
                                  dst_blk_stride=1,
                                  src0_blk_stride=stride_w,
                                  src1_blk_stride=stride_w,
                                  dst_rep_stride=8,
                                  src0_rep_stride=8 * stride_w,
                                  src1_rep_stride=8 * stride_w)
                # filter 2
                with instance.for_range(0, output_h_once) as looph_idx:
                    instance.vmax(mask=constant.MASK128,
                                  dst=ub_max[looph_idx, 0, 0],
                                  src0=ub_max[looph_idx, 0, 0],
                                  src1=ub_input[looph_idx * stride_h + 2 // window_w, 2 % window_w, 0],
                                  repeat_times=output_w // 8,
                                  dst_blk_stride=1,
                                  src0_blk_stride=1,
                                  src1_blk_stride=stride_w,
                                  dst_rep_stride=8,
                                  src0_rep_stride=8,
                                  src1_rep_stride=8 * stride_w)
                # filter 6
                with instance.for_range(0, output_h_once) as looph_idx:
                    instance.vmax(mask=constant.MASK128,
                                  dst=ub_fmatrix_temp[looph_idx, 0, 0],
                                  src0=ub_fmatrix_temp[looph_idx, 0, 0],
                                  src1=ub_input[looph_idx * stride_h + 6 // window_w, 6 % window_w, 0],
                                  repeat_times=output_w // 8,
                                  dst_blk_stride=1,
                                  src0_blk_stride=1,
                                  src1_blk_stride=stride_w,
                                  dst_rep_stride=8,
                                  src0_rep_stride=8,
                                  src1_rep_stride=8 * stride_w)
                # filter 3
                with instance.for_range(0, output_h_once) as looph_idx:
                    instance.vmax(mask=constant.MASK128,
                                  dst=ub_max[looph_idx, 0, 0],
                                  src0=ub_max[looph_idx, 0, 0],
                                  src1=ub_input[looph_idx * stride_h + 3 // window_w, 3 % window_w, 0],
                                  repeat_times=output_w // 8,
                                  dst_blk_stride=1,
                                  src0_blk_stride=1,
                                  src1_blk_stride=stride_w,
                                  dst_rep_stride=8,
                                  src0_rep_stride=8,
                                  src1_rep_stride=8 * stride_w)
                # filter 7
                with instance.for_range(0, output_h_once) as looph_idx:
                    instance.vmax(mask=constant.MASK128,
                                  dst=ub_fmatrix_temp[looph_idx, 0, 0],
                                  src0=ub_fmatrix_temp[looph_idx, 0, 0],
                                  src1=ub_input[looph_idx * stride_h + 7 // window_w, 7 % window_w, 0],
                                  repeat_times=output_w // 8,
                                  dst_blk_stride=1,
                                  src0_blk_stride=1,
                                  src1_blk_stride=stride_w,
                                  dst_rep_stride=8,
                                  src0_rep_stride=8,
                                  src1_rep_stride=8 * stride_w)
                with instance.for_range(0, output_h_once) as looph_idx:
                    instance.vmax(mask=constant.MASK128,
                                  dst=ub_max[looph_idx, 0, 0],
                                  src0=ub_max[looph_idx, 0, 0],
                                  src1=ub_fmatrix_temp[looph_idx, 0, 0],
                                  repeat_times=output_w // 8,
                                  dst_blk_stride=1,
                                  src0_blk_stride=1,
                                  src1_blk_stride=1,
                                  dst_rep_stride=8,
                                  src0_rep_stride=8,
                                  src1_rep_stride=8)
                # filter 8
                with instance.for_range(0, output_h_once) as looph_idx:
                    instance.vmax(mask=constant.MASK128,
                                  dst=ub_max[looph_idx, 0, 0],
                                  src0=ub_max[looph_idx, 0, 0],
                                  src1=ub_input[looph_idx * stride_h + 8 // window_w, 8 % window_w, 0],
                                  repeat_times=output_w // 8,
                                  dst_blk_stride=1,
                                  src0_blk_stride=1,
                                  src1_blk_stride=stride_w,
                                  dst_rep_stride=8,
                                  src0_rep_stride=8,
                                  src1_rep_stride=8 * stride_w)
                # move max to out
                instance.data_move(dst=gm_output_max[block_idx, loop_idx * output_h_once, 0, 0],
                                   src=ub_max,
                                   sid=0,
                                   nburst=1,
                                   burst=output_h_once * output_w,
                                   src_stride=0,
                                   dst_stride=0)

                # preload move next input to ub
                if loop_idx < 5:  # 5 means output_h // output_h_once - 1 - 1
                    instance.data_move(dst=ub_input_other,
                                       src=gm_input[block_idx, (loop_idx + 1) * 16, 0, 0],
                                       sid=0,
                                       nburst=input_h_once,
                                       burst=input_w,
                                       src_stride=0,
                                       dst_stride=input_w_once - input_w)
                if loop_idx == 5:
                    instance.data_move(dst=ub_input_other,
                                       src=gm_input[block_idx, (loop_idx + 1) * 16, 0, 0],
                                       sid=0,
                                       nburst=16,
                                       burst=input_w,
                                       src_stride=0,
                                       dst_stride=input_w_once - input_w)
                    instance.vector_dup(mask=constant.MASK128,
                                        dst=ub_input_other[16, 0, 0],
                                        scalar=MIN_VALUE_FP16,
                                        repeat_times=input_w // 8,
                                        dst_blk_stride=1,
                                        dst_rep_stride=8)
                # compute mask
                with instance.for_range(0, filter_size) as flt_idx:
                    with instance.for_range(0, output_h_once) as looph_idx:
                        instance.vadds(mask=constant.MASK128,
                                       dst=ub_fmatrix_temp[looph_idx, 0, 0],
                                       src=ub_input[looph_idx * 2 + flt_idx // window_w, flt_idx % window_w, 0],
                                       scalar=0,
                                       repeat_times=output_w // 8,
                                       dst_blk_stride=1,
                                       src_blk_stride=2,
                                       dst_rep_stride=8,
                                       src_rep_stride=16)
                    instance.vcmpv_eq(dst=ub_mask_temp[flt_idx, 0, 0],
                                      src0=ub_fmatrix_temp,
                                      src1=ub_max,
                                      repeat_times=output_h_once * output_w // 8,
                                      src0_blk_stride=1,
                                      src1_blk_stride=1,
                                      src0_rep_stride=8,
                                      src1_rep_stride=8)
                # deduplicate mask
                instance.data_move(ub_mask,
                                   ub_mask_temp,
                                   0,
                                   1,
                                   28,
                                   0, 0)
                ub_mask_not = ub_mask_not_ping
                ub_mask_not_other = ub_mask_not_pong
                ub_mask_or = ub_mask_or_ping
                ub_mask_or_other = ub_mask_or_pong
                for filter_idx in range(1, filter_size):
                    if filter_idx == 1:
                        instance.vnot(constant.MASK128,
                                      ub_mask_not,
                                      ub_mask_temp,
                                      output_h_once * output_w // 128,
                                      1, 1, 8, 8)
                        instance.vor(constant.MASK128,
                                     ub_mask_or,
                                     ub_mask_temp,
                                     ub_mask_temp[filter_idx, 0, 0],
                                     output_h_once * output_w // 128,
                                     1, 1, 1, 8, 8, 8)
                    else:
                        instance.vnot(constant.MASK128,
                                      ub_mask_not,
                                      ub_mask_or_other,
                                      output_h_once * output_w // 128,
                                      1, 1, 8, 8)
                        instance.vor(constant.MASK128,
                                     ub_mask_or,
                                     ub_mask_or_other,
                                     ub_mask_temp[filter_idx, 0, 0],
                                     output_h_once * output_w // 128,
                                     1, 1, 1, 8, 8, 8)
                    instance.vand(constant.MASK128,
                                  ub_mask[filter_idx, 0, 0],
                                  ub_mask_temp[filter_idx, 0, 0],
                                  ub_mask_not,
                                  output_h_once * output_w // 128,
                                  1, 1, 1, 8, 8, 8)
                    ub_mask_not, ub_mask_not_other = ub_mask_not_other, ub_mask_not
                    ub_mask_or, ub_mask_or_other = ub_mask_or_other, ub_mask_or

                # tail
                for filter_idx in range(1, filter_size):
                    if filter_idx == 1:
                        instance.vnot(constant.MASK64,
                                      ub_mask_not[6, 48],
                                      ub_mask_temp[0, 6, 48],
                                      1,
                                      1, 1, 8, 8)
                        instance.vor(constant.MASK64,
                                     ub_mask_or[6, 48],
                                     ub_mask_temp[0, 6, 48],
                                     ub_mask_temp[filter_idx, 6, 48],
                                     1,
                                     1, 1, 1, 8, 8, 8)
                    else:
                        instance.vnot(constant.MASK64,
                                      ub_mask_not[6, 48],
                                      ub_mask_or_other[6, 48],
                                      1,
                                      1, 1, 8, 8)
                        instance.vor(constant.MASK64,
                                     ub_mask_or[6, 48],
                                     ub_mask_or_other[6, 48],
                                     ub_mask_temp[filter_idx, 6, 48],
                                     1,
                                     1, 1, 1, 8, 8, 8)
                    instance.vand(constant.MASK64,
                                  ub_mask[filter_idx, 6, 48],
                                  ub_mask_temp[filter_idx, 6, 48],
                                  ub_mask_not[6, 48],
                                  1,
                                  1, 1, 1, 8, 8, 8)
                    ub_mask_not, ub_mask_not_other = ub_mask_not_other, ub_mask_not
                    ub_mask_or, ub_mask_or_other = ub_mask_or_other, ub_mask_or

                # move mask to out
                instance.data_move(dst=gm_output_mask[block_idx, 0, loop_idx * output_h_once * output_w],
                                   src=ub_mask,
                                   sid=0,
                                   nburst=filter_size,
                                   burst=output_h_once * output_w // 16,
                                   src_stride=0,
                                   dst_stride=(mask_one_window - output_h_once * output_w) // 16)
                ub_input, ub_input_other = ub_input_other, ub_input
                ub_max, ub_max_other = ub_max_other, ub_max
                ub_mask, ub_mask_other = ub_mask_other, ub_mask
        instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[gm_input],
                          outputs=[gm_output_max, gm_output_mask])
        return instance

    def _calc_out_size_and_pad(self):
        """
        caculate output size and padding size
        -------
        pad: include pad_t, pad_b, pad_l, pad_r
        out_size_h: out_size in h direction
        out_size_w: out_size in w direction
        """
        # pad_l, pad_r, pad_t, pad_b is for pad on the left, right, top, bottom
        pad_l, pad_r, pad_t, pad_b = 0, 0, 0, 0

        if self.padding == "SAME":
            # `Hout = ceil(Hi, Sh), Wout = ceil(Wi, Sw)`
            out_size_h = (self.in_size_h + self.stride_h - 1) // self.stride_h
            out_size_w = (self.in_size_w + self.stride_w - 1) // self.stride_w

            # get total pad rows or pad columns
            pad_rows = (out_size_h - 1) * self.stride_h + \
                       ((self.window_h - 1) * DILATION + 1) - self.in_size_h
            pad_cols = (out_size_w - 1) * self.stride_w + \
                       ((self.window_w - 1) * DILATION + 1) - self.in_size_w

            # pad_rows and pad_columns is odd or even number
            if pad_rows % 2 == 0:
                pad_t = pad_rows // 2
                pad_b = pad_rows // 2
            else:
                pad_t = pad_rows // 2
                pad_b = pad_rows - pad_t

            if pad_cols % 2 == 0:
                pad_l = pad_cols // 2
                pad_r = pad_cols // 2
            else:
                pad_l = pad_cols // 2
                pad_r = pad_cols - pad_l

            if pad_t < 0:
                pad_t = 0

            if pad_b < 0:
                pad_b = 0

            if pad_l < 0:
                pad_l = 0

            if pad_r < 0:
                pad_r = 0

        # caculate output size in VALID mode
        if self.padding == "VALID":
            # `Hout = ceil(Hi - Fh + 1, Sh), Wout = ceil(Wi - Fw + 1, Sw)`
            out_size_h = (self.in_size_h - self.window_h + 1 +
                          (self.stride_h - 1)) // self.stride_h
            out_size_w = (self.in_size_w - self.window_w + 1 +
                          (self.stride_w - 1)) // self.stride_w
        pad = (pad_l, pad_r, pad_t, pad_b)

        return pad, out_size_h, out_size_w


# 'pylint: disable=invalid-name
def is_max_pool_with_argmax_param(x, ksize, strides, padding):
    """
    test if the param suitable for this module to treat
    :param x: dict of shape and dtype of the input x
    :param ksize: value of ksize
    :param strides: value of strides
    :param padding: value of padding
    :return: Bool, if the param suitable for this module to treat return True,
             if not return False
    """
    resnet50_x = {"shape": (32, 4, 112, 112, 16), "dtype": "float16"}
    resnet50_ksize = [1, 3, 3, 1]
    resnet50_strides = [1, 2, 2, 1]
    resnet50_padding = "SAME"

    def is_valid_shape(resnet50shape, shape):
        """
        check whether the shape is valid

        Parameters
        ----------
        resnet50shape: original shape
        shape: destination shape

        Returns
        -------
        None
        """
        if shape.get("dtype") != resnet50shape.get("dtype"):
            return False

        if len(shape.get("shape")) != len(resnet50shape.get("shape")):
            return False

        resnet50_last3dims = resnet50shape.get("shape")[2:]
        last3dims = shape.get("shape")[2:]

        return list(resnet50_last3dims) == list(last3dims)

    ksize = list(ksize)
    strides = list(strides)

    if (resnet50_ksize == ksize and resnet50_strides == strides and
            resnet50_padding == padding and
            is_valid_shape(resnet50_x, x)):
        return True

    return False


# 'pylint: disable=invalid-name
def max_pool_with_argmax(x, ksize, strides, padding, kernel_name):
    """
    implementation of max_pool_with_argmax and return the tik instance
    :param x: dict of shape and dtype of the input x
    :param ksize: value of strides
    :param strides: value of strides
    :param padding: value of padding
    :param kernel_name: the kernel's name
    :return: tik instance
    """
    max_pool_grad = MaxPoolWithargmaxResnet50(x, ksize, strides, padding)
    return max_pool_grad.tik_instance_function_v1(kernel_name)
