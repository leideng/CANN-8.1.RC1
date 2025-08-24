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
max_pool_grad_with_argmaxv2
"""
import math
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl import max_pool_grad_with_argmax_cut_h_v2 as argmax_cut_h_v2
from impl import max_pool_grad_with_argmax_v2_resnet50 as resnet50

# size of 5HD format
DIM_5HD = 5
# size of c0 for fp16
C0 = 16
# min shape of attr
ATTR_SHAPE_MIN = 4
# size of useful UB buffer
USEFUL_UB_SIZE = 1024 * 240
# size of vector calc one repeat
ONE_REPEAT = 256
# size of one block
BLOCK_SIZE = 32
# max repeat of vector calc
V_MAX_REPEAT = 255
# max num of fp16 in one repeat
FP16_MAX = 128
# max num of fp32 in one repeat
FP32_MAX = 64
# max num of fp16 mask handle one time
MASK_MAX = 8
DT_INT32 = 3
DT_INT64 = 9


# 'pylint: disable=locally-disabled,too-many-arguments,invalid-name
@para_check.check_input_type(dict, dict, dict, dict, (list, tuple), (list, tuple),
                             (list, tuple), int,
                             (list, tuple), bool, str)
def max_pool_grad_with_argmax(x, grad, argmax, y, ksize, strides, pads,
                              dtype=DT_INT32,
                              dilation=(1, 1, 1, 1), ceil_mode=False,
                              kernel_name="max_pool_grad_with_argmaxv2"):
    """
    the main function of the maxpoolGradWithArgmax
    Parameters
    ----------
    x: input of maxpool, useless for maxpool gard
    grad: input of maxpoolgard or output of maxpool
    argmax:output of maxpool mask or index
    y: output of maxpoolgard
    ksize: kernel or windows size,minimum length is 4,
           just like [1, poolingWindowH, poolingWindowW, 1]
    strides: stride , minimum length is 4, just like
    [1, poolingStrideH, poolingStrideW, 1]
    pads: pad list_int
    kernel_name: kernel_name
    Returns
    -------
    tik_instance: tik_instance
    :param pads:
    :param strides:
    :param ksize:
    :param argmax:
    :param grad:
    :param dilation:
    :param dtype:
    :param kernel_name: max_pool_grad_with_argmaxv2
    :param ceil_mode:
    """
    check_param(x, grad, argmax, y, ksize, strides, pads, dtype, dilation,
                ceil_mode,
                kernel_name)
    if resnet50.is_max_pool_grad_with_argmax_param(grad, argmax, x, ksize,
                                                   strides, pads):
        return resnet50.max_pool_grad_with_argmax(grad, argmax, x, ksize,
                                                  strides, pads,
                                                  dilation, ceil_mode,
                                                  kernel_name)
    maxpoolgard = MaxpoolGard(grad, argmax, x, ksize, strides, pads, dilation,
                              ceil_mode)
    return maxpoolgard.tik_instance_function(kernel_name)


# 'pylint: disable=too-few-public-methods,useless-super-delegation
class MaxpoolGard(argmax_cut_h_v2.MaxpoolGradBase):
    """
    parameter for max_pool_grad_with_pool
    """

    # 'pylint: disable=locally-disabled,too-many-arguments,
    # 'pylint: disable=too-many-locals
    def __init__(self, grad, argmax, input_x, ksize, strides, padding,
                 dilation, ceil_mode):
        """
        init compare and bit pack parameters
        Parameters
        ----------
        grad: input of maxpoolgard or output of maxpool
        argmax: output of maxpool mask or index
        input_x: input of maxpool, useless for maxpool gard
        ksize: the kernel size
        strides: stride , minimum length is 4, just like
        [1, poolingStrideH, poolingStrideW, 1]
        padding: pad num ,should be listint
        dilation: dilation, should be listint
        ceil_mode: ceil_mode, True or False
        Returns
        -------
        None
        """
        # 'pylint: disable=super-with-arguments
        super(MaxpoolGard, self).__init__(grad, argmax, input_x, ksize,
                                          strides, padding,
                                          dilation, ceil_mode)

    # 'pylint: disable=locally-disabled,too-many-locals,too-many-statements,
    # 'pylint: disable=unused-variable,too-many-branches,too-many-return-statements
    def tik_instance_function(self, kernel_name):
        """
        get vector instruct repeat times
        Parameters
        ----------
        kernel_name: cce kernel name, default value is "maxpoolGradWithArgmax"
        Returns
        -------
        None
        """
        batch, c1, dyh, dyw, channel = self.input_gard_shape
        strideh, stridew = self.strides[1:3]
        windowh, windoww = self.ksize[1:3]
        # the minimum part can be dealed
        ho_min = 1 if self.hoverlap == 0 else 2
        hoverlap = self.hoverlap
        woverlap = self.woverlap
        dtype_size = self.dtype_size

        ho_min = 1 if hoverlap == 0 else 2
        ho_max = ho_min
        wo_max = math.ceil(dyw / 16) * 16
        col2img_w = wo_max * stridew if woverlap == 0 else (wo_max - 1) * stridew + windoww
        col2img_h = ho_max * strideh if hoverlap == 0 else (ho_max - 1) * strideh + windowh

        if windowh > 2 * strideh or windoww > 2 * stridew:
            error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmaxv2",
                                                          "windowh > 2*strideh or \
                                                          windoww > 2*stridew not support yet!!")
        if batch * c1 >= self.blocknum or dyh <= self.blocknum:
            if col2img_w * col2img_h * channel * dtype_size > self.ub_limit:
                length = col2img_w * col2img_h * channel * dtype_size
                error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmaxv2",
                                                              "length is bigger than ub_limit!")
            return self.tik_instance_cut_nc1_cut_h(kernel_name)  # support
        if batch * c1 * dyh < self.blocknum:
            self.change_blocknum(batch * c1 * dyh)
        if col2img_w * col2img_h * channel * dtype_size > self.ub_limit:
            length = col2img_w * col2img_h * channel * dtype_size
            error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmaxv2",
                                                          "length is bigger than ub_limit!")
        return self.tik_instance_cut_nc1h_cut_h(kernel_name)  # support


def check_shape_5hd(shape):
    """
    The common check rule for tensor shape, just for 5hd
    """
    para_check.check_shape_rule(shape)
    if len(shape) != DIM_5HD:
        error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmaxv2",
                                                      "The dim of tensor must be " + str(DIM_5HD) +
                                                      ", actual dim is " + str(len(shape)))

    if shape[DIM_5HD - 1] != C0:
        error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmaxv2",
                                                      "The value of C0 must be " + str(C0) +
                                                      ", actual input is " + str(shape[DIM_5HD - 1]))


def check_padding(padding, check_list):
    """
    The common check rule for padding
    """
    if padding not in check_list:
        error_manager_vector.raise_err_pad_mode_invalid("max_pool_grad_with_argmaxv2",
                                                        "SAME or VALID", str(padding))


def _pooling_output_shape_pad_lr(input_size, kernel_size, pad_l,
                                 pad_r, stride, dilation, ceil_mode):
    temp = input_size + pad_l + pad_r - dilation * (kernel_size - 1) - 1

    if ceil_mode:
        output_size = ((temp + (stride - 1)) // stride) + 1
    else:
        output_size = (temp // stride) + 1

    if pad_l > 0:
        # ensure that the last pooling starts inside the image
        # needed to avoid problems in ceil mode
        if (output_size - 1) * stride >= (input_size + pad_l):
            output_size = output_size - 1

    return output_size


def _pooling_output_shape(input_size, kernel_size, pad, stride,
                          dilation, ceil_mode):
    return _pooling_output_shape_pad_lr(input_size, kernel_size,
                                        pad, pad, stride, dilation, ceil_mode)


def _pool2d_shape_check(kernel_h, kernel_w, stride_h, stride_w,
                        pad_h, pad_w, dilation_h, dilation_w, output_h,
                        output_w):
    if kernel_w <= 0 or kernel_h <= 0:
        error_manager_vector.raise_err_input_value_invalid("max_pool_grad_with_argmaxv2",
                                                           "kernel_w or kernel_h",
                                                           "greater than zero",
                                                           str(kernel_w) + str("and") + str(kernel_h))

    if stride_h <= 0 or stride_w <= 0:
        error_manager_vector.raise_err_input_value_invalid("max_pool_grad_with_argmaxv2",
                                                           "stride_h or stride_w",
                                                           "greater than zero",
                                                           str(stride_h) + str("and") + str(stride_w))

    if dilation_h <= 0 or dilation_w <= 0:
        error_manager_vector.raise_err_input_value_invalid("max_pool_grad_with_argmaxv2",
                                                           "dilation_h or dilation_w",
                                                           "greater than zero",
                                                           str(dilation_h) + str("and") + str(dilation_w))

    if (kernel_w // 2) < pad_w or (kernel_h // 2) < pad_h:
        error_manager_vector.raise_err_input_value_invalid("max_pool_grad_with_argmaxv2",
                                                           "pad_w or pad_h",
                                                           "smaller than half of kernel",
                                                           str(pad_w) + str("and") + str(pad_h))

    if output_h < 1 or output_w < 1:
        error_manager_vector.raise_err_input_value_invalid("max_pool_grad_with_argmaxv2",
                                                           "output_h or output_w",
                                                           "greater than or equal to 1",
                                                           str(output_h) + str("and") + str(output_w))


# 'pylint: disable=locally-disabled,too-many-locals
def check_output_dim_with_ksize_stride(padding, input_gard_shape, y_shape,
                                       ksize, strides, dilation, ceil_mode):
    """
    The common check rule for output dim and ksize and strides
    """
    para_check.check_tensor_shape_size(ksize)
    para_check.check_tensor_shape_size(strides)
    if len(ksize) < ATTR_SHAPE_MIN or len(strides) < ATTR_SHAPE_MIN:
        error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmaxv2",
                                                      "The shape length of ksize or strides must be more than 4")
    if ksize[0] != 1 or ksize[3] != 1:
        error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmaxv2",
                                                      "MaxPoolGradWithArgmax only supports pooling across width/height,"
                                                      "and other ksize dimension should be one")
    if strides[0] != 1 or strides[3] != 1:
        error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmaxv2",
                                                      "MaxPoolGradWithArgmax only supports pooling across width/height,"
                                                      "and other strides dimension should be one")
    if ksize[1] * ksize[2] > 255:
        error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmaxv2",
                                                      "invalid window params, window_h*window_w should be <=255")

    input_height = y_shape[2]
    input_width = y_shape[3]
    input_batch = y_shape[0]
    xc1 = y_shape[1]
    xc0 = y_shape[4]
    output_height = input_gard_shape[2]
    output_width = input_gard_shape[3]
    windowh = ksize[1]
    windoww = ksize[2]
    dyn = input_gard_shape[0]
    dyc1 = input_gard_shape[1]
    dyc0 = input_gard_shape[4]
    pad_h = padding[1]
    pad_w = padding[2]
    stride_h = strides[1]
    stride_w = strides[2]
    dilation_h = dilation[1]
    dilation_w = dilation[2]

    dyh = _pooling_output_shape(input_height, windowh, pad_h, stride_h,
                                dilation_h, ceil_mode)
    dyw = _pooling_output_shape(input_width, windoww, pad_w, stride_w,
                                dilation_w, ceil_mode)

    if dyh != output_height or dyw != output_width or \
            input_batch != dyn or xc1 != dyc1 or xc0 != dyc0:
        error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmaxv2",
                                                      "dimentions of dx dy \
                                                       padMode window stride is wrong,please check!")


def check_param(x, grad, argmax, y, ksize, strides, padding, dtype, dilation,
                ceil_mode,
                kernel_name):
    """
    check the parameters is valid, if one is invalid,then raise error
    Parameters
    ----------
    x: dict,shape and datatype
    grad: dict,shape and datatype
    argmax: dict,shape and datatype
    y: dict,shape and datatype
    ksize: kernel or windows size,minimum length is 4,
          just like [1, poolingWindowH, poolingWindowW, 1]
    strides: stride , minimum length is 4, just like
    [1, poolingStrideH, poolingStrideW, 1]
    padding: pad mode
    Returns
    -------
    None
    """
    y_shape = x.get("shape")
    y_dtype = x.get("dtype").lower()
    y_dtype_arg = y.get("dtype").lower()
    input_gard_shape = grad.get("shape")
    grad_dtype = grad.get("dtype").lower()
    argmax_shape = argmax.get("shape")
    argmax_dtype = argmax.get("dtype").lower()
    para_check.check_shape_rule(y_shape)
    para_check.check_shape_rule(input_gard_shape)
    para_check.check_shape_rule(argmax_shape)
    para_check.check_kernel_name(kernel_name)
    check_shape_5hd(y_shape)
    check_shape_5hd(input_gard_shape)
    para_check.check_tensor_shape_size(input_gard_shape)
    para_check.check_tensor_shape_size(argmax_shape)
    para_check.check_tensor_shape_size(y_shape)
    para_check.check_dtype_rule(grad_dtype, ("float16", "float32", "int32"))
    para_check.check_dtype_rule(argmax_dtype, ("uint16"))
    para_check.check_dtype_rule(y_dtype, ("float16", "float32", "int32"))

    if y_dtype != grad_dtype or y_dtype_arg != y_dtype:
        error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmaxv2",
                                                      "The dtype of tensor must be same")

    if dtype not in (DT_INT32, DT_INT64):
        error_manager_vector.raise_err_input_dtype_not_supported("max_pool_grad_with_argmaxv2",
                                                                 "dtype", "int32 or int64", str(dtype))

    check_output_dim_with_ksize_stride(padding, input_gard_shape, y_shape,
                                       ksize, strides,
                                       dilation, ceil_mode)


def _ceil_div(value, factor):
    """
    _ceil_div
    """
    if value % factor == 0:
        quotient = value // factor
    else:
        quotient = value // factor + 1
    return quotient
