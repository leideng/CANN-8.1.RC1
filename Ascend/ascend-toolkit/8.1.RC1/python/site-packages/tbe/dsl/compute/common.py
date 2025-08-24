#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
common
"""
from __future__ import absolute_import
from tbe import tvm
from tbe.common.platform import BLOCK_REDUCE
from tbe.common.platform import BLOCK_REDUCE_INT8
from tbe.common.utils.errormgr.error_manager_util import get_error_message


_BLOCK_SIZE = BLOCK_REDUCE
_BLOCK_INT8_SIZE = BLOCK_REDUCE_INT8


def img2col(input_img,
            col_shape,
            filter_h,
            filter_w,
            pad,
            stride,
            tag=None,
            padding_value=0.0):
    """
    img2col
    """

    # pylint: disable=too-many-locals
    def _img2col_compute(input_img, indices, filter_w, pad, stride,
                         padding_value):
        # fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0
        _, _, fmap_h, fmap_w, _ = input_img.shape
        col_n, col_howo, col_c1, col_hw, col_ww, col_c0 = indices
        stride_h, stride_w = stride
        pad_top, _, pad_left, pad_right = pad

        output_w = (fmap_w.value + pad_left + pad_right - filter_w) \
            // stride_w + 1

        img_n_index = col_n
        img_c1_index = col_c1
        img_h_index = (col_howo // output_w) * stride_h + col_hw
        img_w_index = (col_howo % output_w) * stride_w + col_ww
        img_c0_index = col_c0

        return tvm.select(
            tvm.any(img_h_index < pad_top,
                    img_h_index > fmap_h.value + pad_top - 1,
                    img_w_index < pad_left,
                    img_w_index > fmap_w.value + pad_left - 1),
            tvm.const(padding_value, 'float16'),
            input_img(img_n_index, img_c1_index, img_h_index - pad_top,
                      img_w_index - pad_left, img_c0_index))

    if tag is None:
        tag = 'im2col_row_major'
    return tvm.compute(
        col_shape,
        lambda *indices: _img2col_compute(input_img, indices, filter_w, pad,
                                          stride, padding_value),
        name='im2col_row_major',
        tag=tag,
        attrs={
            'kernel_h': filter_h,
            'kernel_w': filter_w,
            'padding': pad,
            'stride': stride
        })



def im2col_fractal(a_im2col_shape, in_a, dst='ca', tag=None):
    """
    im2col_fractal
    """
    last_dim = in_a.shape[-1]

    # pylint: disable=too-many-locals
    def __im2col_fractal_indices(indices, in_a):
        _, h_w, _, kernel_h, kernel_w, _ = in_a.shape
        if dst == 'ca':
            batch_size, i_1, j_1, i_0, j_0 = indices
        else:
            batch_size, i_1, j_1, j_0, i_0 = indices

        n_index = batch_size
        hw_index = i_1 * _BLOCK_SIZE + i_0
        c1_index = (((j_1 * last_dim + j_0) // last_dim) //
                    kernel_w.value) // kernel_h.value
        kh_index = (((j_1 * last_dim + j_0) // last_dim) //
                    kernel_w.value) % kernel_h.value
        kw_index = ((j_1 * last_dim + j_0) // last_dim) % kernel_w.value
        c0_index = (j_1 * last_dim + j_0) % last_dim

        return tvm.select(
            tvm.any(hw_index < 0, hw_index > h_w.value - 1),
            tvm.const(0.0, 'float16'),
            in_a(n_index, hw_index, c1_index, kh_index, kw_index, c0_index))

    if tag is None:
        tag = 'im2col_fractal'
    return tvm.compute(
        a_im2col_shape,
        lambda *indices: __im2col_fractal_indices(indices, in_a),
        name='im2col_fractal',
        tag=tag)


# pylint: disable=too-many-arguments
def im2col_6d(input_img,
              col_shape,
              filter_h,
              filter_w,
              pad,
              stride,
              padding_value=0.0,
              dilation=[1, 1]):
    """
    im2col_6d
    """

    # pylint: disable=too-many-locals
    def _im2col_compute(input_img, indices, filter_w, pad, stride,
                        padding_value, dilation):
        # fmap_n, fmap_cg, fmap_c1, fmap_h, fmap_w, fmap_c0
        sixd_flag = 0
        if (len(input_img.shape)) == 6:
            _, _, _, fmap_h, fmap_w, _ = input_img.shape
            sixd_flag = 1
        else:
            _, _, fmap_h, fmap_w, _ = input_img.shape
        col_n, col_cg, col_howo, col_c1, col_hw, col_ww, col_c0 = indices
        stride_h, stride_w = stride
        pad_top, _, pad_left, pad_right = pad
        dilation_h, dilation_w = dilation
        effective_filter_w = (filter_w - 1) * dilation_w + 1
        output_w = (fmap_w.value + pad_left + pad_right - effective_filter_w) // stride_w + 1

        img_n_index = col_n
        img_c1_index = col_c1
        img_h_index = (col_howo // output_w) * stride_h + col_hw*dilation_h
        img_w_index = (col_howo % output_w) * stride_w + col_ww*dilation_w
        img_c0_index = col_c0
        from .depthwise_conv2d_compute import DepthwiseConv2dParam
        slice_offset = DepthwiseConv2dParam.fusion_para.get("slice_offset")
        input_memory_type = DepthwiseConv2dParam.fusion_para.get(
            "input_memory_type")
        offset_w = slice_offset[2] if (
            slice_offset and input_memory_type == 1) else 0
        if sixd_flag == 1:
            return tvm.select(
                tvm.any(img_h_index < pad_top,
                        img_h_index > fmap_h.value + pad_top - 1,
                        img_w_index < pad_left,
                        img_w_index > fmap_w.value + pad_left - 1),
                tvm.const(padding_value, input_img.dtype),
                input_img(img_n_index, col_cg, img_c1_index,
                          img_h_index - pad_top + offset_w,
                          img_w_index - pad_left,
                          img_c0_index))
        return tvm.select(
            tvm.any(img_h_index < pad_top,
                    img_h_index > fmap_h.value + pad_top - 1,
                    img_w_index < pad_left,
                    img_w_index > fmap_w.value + pad_left - 1),
            tvm.const(padding_value, input_img.dtype),
            input_img(img_n_index,
                      col_cg,
                      img_h_index - pad_top + offset_w,
                      img_w_index - pad_left,
                      img_c0_index))

    return tvm.compute(
        col_shape,
        lambda *indices: _im2col_compute(input_img, indices, filter_w, pad,
                                         stride, padding_value, dilation),
        name='im2col_row_major',
        tag='im2col_row_major',
        attrs={
            'kernel_h': filter_h,
            'kernel_w': filter_w,
            'padding': pad,
            'stride': stride
        })


def im2col_fractal_6d(a_im2col_shape, in_a):
    """
    im2col_fractal_6d
    """
    last_dim = in_a.shape[-1]

    # pylint: disable=too-many-locals
    def __im2col_fractal_indices(indices, in_a):
        _, c_g, h_w, _, kernel_h, kernel_w, _ = in_a.shape
        batch_size, c_g, i_1, j_1, i_0, j_0 = indices

        n_index = batch_size
        hw_index = i_1 * _BLOCK_SIZE + i_0
        c1_index = (((j_1 * last_dim + j_0) // last_dim) //
                    kernel_w.value) // kernel_h.value
        kh_index = (((j_1 * last_dim + j_0) // last_dim) //
                    kernel_w.value) % kernel_h.value
        kw_index = ((j_1 * last_dim + j_0) // last_dim) % kernel_w.value
        c0_index = (j_1 * last_dim + j_0) % last_dim

        return tvm.select(
            tvm.any(hw_index < 0, hw_index > h_w.value - 1),
            tvm.const(0.0, in_a.dtype),
            in_a(n_index, c_g, hw_index, c1_index, kh_index, kw_index,
                 c0_index))

    return tvm.compute(
        a_im2col_shape,
        lambda *indices: __im2col_fractal_indices(indices, in_a),
        name='im2col_fractal',
        tag='im2col_fractal')


def mad(mad_shape, in_a, in_b, res_type, offset_x=0, v200_flag=False):
    """
    mad
    """
    if res_type in ('int32', 'uint32'):
        r_k0 = tvm.reduce_axis((0, _BLOCK_INT8_SIZE), name='k0')
    else:
        r_k0 = tvm.reduce_axis((0, _BLOCK_SIZE), name='k0')
    r_k1 = tvm.reduce_axis((0, in_b.shape[1]), name='k1')
    # If tag set to 'gemv', computeOp return tensor of specific layout.
    # e.g. gemv of 1x32, tensor C is 1x32 but occupy 16x32 fractal matrix size.
    # gemv of 2x32 also occupy 16x32.
    if res_type == "float16":
        crmode = 'f162f16'
    else:
        crmode = 'f162f32'
    offset_x = offset_x if v200_flag else 0
    return tvm.compute(
        mad_shape,
        lambda n, cg, j1, i, j0: tvm.sum((in_a[
            n, cg, i // _BLOCK_SIZE, r_k1, i % _BLOCK_SIZE, r_k0] - offset_x).astype(
                res_type) * in_b[cg, r_k1, j1, j0, r_k0].astype(res_type),
                                         axis=[r_k1, r_k0]),
        name='mad',
        tag='gemm',
        attrs={'mode': crmode})


# pylint: disable=invalid-name
def tf_get_windowed_output_size(input_size, filter_size, stride, padding_type):
    """
    get output and padding size using tensorflow padding rule

    Parameters
    ----------
    input_size : int, feature map size

    filter_size : int, filter size

    stride: int, stride size

    padding_type: string, support "SAME", "VALID" or "EXPLICIT"

    Returns
    -------
    output_size: int, output feature map size

    padding_size: int, feature map padding size
    """
    if padding_type == 'EXPLICIT':
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "tf_get_windowed_output_size does not" \
                                      " handle EXPLITCIT padding; " \
                                      "call tf_get_windowed_output_size_verbose instead."
        raise RuntimeError(dict_args, get_error_message(dict_args))

    # pylint: disable=invalid-name
    output_size, padding_size, _ = tf_get_windowed_output_size_verbose(
        input_size, filter_size, stride, padding_type)

    return output_size, padding_size


# pylint: disable=invalid-name
def tf_get_windowed_output_size_verbose(input_size, filter_size, stride,
                                        padding_type):
    """
    get output and padding size using tensorflow padding rule

    Parameters
    ----------
    input_size : int, feature map size

    filter_size : int, filter size

    stride: int, stride size

    padding_type: string, support "SAME", "VALID" or "EXPLICIT"

    Returns
    -------
    output_size: int, output feature map size

    padding_before: int, feature map padding before size

    padding_after: int, feature map padding after size
    """
    dilation_rate = 1

    (output_size, padding_before,
     padding_after) = tf_get_windowed_output_size_verbose_v2(
         input_size, filter_size, dilation_rate, stride, padding_type)

    return output_size, padding_before, padding_after


def tf_get_windowed_output_size_verbose_v2(input_size, filter_size,
                                           dilation_rate, stride,
                                           padding_type):
    """
    get output and padding size using tensorflow padding rule

    Parameters
    ----------
    input_size : int, feature map size

    filter_size : int, filter size

    dilation_rate: int, dilation rate

    stride: int, stride size

    padding_type: string, support "SAME", "VALID" or "EXPLICIT"

    Returns
    -------
    output_size: int, output feature map size

    padding_before: int, feature map padding before size

    padding_after: int, feature map padding after size
    """
    if stride <= 0:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "Stride must be > 0, but stride is [%s]" % stride
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if dilation_rate < 1:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "dilation_rate must be >= 1, " \
                                      "but dilation_rate is [%s]" % dilation_rate
        raise RuntimeError(dict_args, get_error_message(dict_args))

    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    if padding_type == "VALID":
        output_size = (input_size - effective_filter_size + stride) // stride
        padding_before = 0
        padding_after = 0
    elif padding_type == "SAME":
        output_size = (input_size + stride - 1) // stride
        padding_needed = max(0, (output_size - 1) * stride +
                             effective_filter_size - input_size)
        padding_before = padding_needed // 2
        padding_after = padding_needed - padding_before
    else:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "Unsupported padding type [%s], " \
                                      "padding_type must be VALID or SAME" % padding_type
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return output_size, padding_before, padding_after
