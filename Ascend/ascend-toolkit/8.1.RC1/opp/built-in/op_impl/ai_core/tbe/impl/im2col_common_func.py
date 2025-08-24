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
extract_image_patches
"""
# 'pylint: disable=unused-import,too-many-lines
import math
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from tbe.dsl.base.operation import get_context
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    BLOCK_SIZE = 16
    BLOCK_SIZE_INT8 = 32

    DOUBLE_BUFFER = 2
    FP16_SIZE = 2
    INT8_SIZE = 1
    NEED_UB_SPACE_NUM = 2
    SIZE_L1 = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
    SIZE_UB = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    LOAD3D_REPEAT_TIME_LIMIT = 255
    DELTA = 0.000001  # aviod div zero, fp32 precision


# 'pylint: disable = unused-argument,redefined-builtin,too-many-arguments,invalid-name
def _ceil_div(value, block):
    """
    integrate the input value by block
    """
    return (value + block - 1) // block


def _prod(val_list):
    """
    caculate product of val_list
    """
    res = 1
    for val in val_list:
        res = res * val
    return res


def _ub_split_c1(ub_split_c1_shape, tensor, ksize):
    # 'pylint: disable=invalid-name
    def _ub_split_c1_indices(indices, tensor):
        n, howo, co1, khw, howo0, co0 = indices
        n_index = n
        hw_index = howo
        hw0_index = howo0
        c1_index = co1 * ksize + khw
        c0_index = co0
        return tensor(n_index, hw_index, c1_index, hw0_index, c0_index)

    return tvm.compute(ub_split_c1_shape, lambda *indices: _ub_split_c1_indices(indices, tensor), name='_ub_split_c1')


def _ub_transpose(ub_transpose_shape, tensor):
    # 'pylint: disable=invalid-name
    def _ub_transpose_indices(indices, tensor):
        n, howo, howo0, khw, co1, co0 = indices
        n_index = n
        hw_index = howo
        c1_index = co1
        khw_index = khw
        hw0_index = howo0
        c0_index = co0

        return tensor(n_index, hw_index, c1_index, khw_index, hw0_index, c0_index)

    return tvm.compute(ub_transpose_shape,
                       lambda *indices: _ub_transpose_indices(indices, tensor),
                       name='_ub_transpose')


def _ub_merge_hw(ub_merge_shape, tensor):
    # 'pylint: disable=invalid-name
    def _ub_merge_hw_indices(indices, tensor):
        _, _, in_hw0, _, _, _ = tensor.shape
        n, howo, khw, co1, co0 = indices
        n_index = n
        hw_index = howo // in_hw0
        hw0_index = howo % in_hw0
        c1_index = co1
        khw_index = khw
        c0_index = co0
        return tensor(n_index, hw_index, hw0_index, khw_index, c1_index, c0_index)

    return tvm.compute(ub_merge_shape, lambda *indices: _ub_merge_hw_indices(indices, tensor), name='_ub_merge_hw')


def _ub_merge_co(ub_merge_co_shape, tensor):
    # 'pylint: disable=invalid-name
    def _ub_merge_co_indices(indices, tensor):
        _, _, _, _, in_c0 = tensor.shape
        n, howo, khw, co = indices
        n_index = n
        hw_index = howo
        khw_index = khw
        c1_index = co // in_c0
        c0_index = co % in_c0
        return tensor(n_index, hw_index, khw_index, c1_index, c0_index)

    return tvm.compute(ub_merge_co_shape, lambda *indices: _ub_merge_co_indices(indices, tensor), name='_ub_merge_co')


# 'pylint: disable=too-many-arguments
def _im2col_row_major_v2(feature_map, im2col_vm_shape, kernel_h, kernel_w, padding, stride, dilate, compute_dtype):
    """
    calculate im2col_row_major tensor
    Parameters
    ----------
    feature_map : feature map

    im2col_vm_shape : shape of A_im2col_row_major

    kernel_h: the kernel value in  h

    kernel_w: the kernel value in  w

    padding: the padding shape

    stride: the stride value

    dilate: the dilation value

    compute_dtype: dtype of compute result
    -------
    Returns : A_im2col_row_major tensor
    """

    # 'pylint: disable=unused-argument,invalid-name,too-many-locals,too-many-arguments
    def _im2col_row_major_indices(indices, feature_map, kernel_h, kernel_w, padding, stride, dilate):
        """
        calculate im2col_row_major tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        feature_map : feature map

        kernel_h: the kernel value in  h

        kernel_w: the kernel value in  w

        padding: the padding shape

        stride: the stride value

        -------
        Returns  im2col_row_major tvm lambda function
        """
        _, _, in_h, in_w, _ = feature_map.shape

        n, hw, c1, kh, kw, c0 = indices
        stride_h, stride_w = stride
        dilate_h, dilate_w = dilate
        padding_top, _, padding_left, padding_right = padding

        kernel_dilate_w = (kernel_w - 1) * dilate[1] + 1

        width_out = (in_w.value + padding_left + padding_right - kernel_dilate_w) // (stride_w) + 1

        n_index = n
        c1_index = c1
        h_index = (hw // width_out) * stride_h + (kh * dilate_h)
        w_index = (hw % width_out) * stride_w + (kw * dilate_w)
        c0_index = c0
        return tvm.select(
            tvm.any(h_index < padding_top, h_index > in_h.value + padding_top - 1, w_index < padding_left,
                    w_index > in_w.value + padding_left - 1), tvm.const(0.0, compute_dtype),
            feature_map(n_index, c1_index, h_index - padding_top, w_index - padding_left, c0_index))

    return tvm.compute(
        im2col_vm_shape,
        lambda *indices: _im2col_row_major_indices(indices, feature_map, kernel_h, kernel_w, padding, stride, dilate),
        name='im2col_row_major',
        tag='im2col_row_major')


def _im2col_fractal_v2(im2col_shape, feature_map, config, compute_dtype):
    """
    calculate im2col_fractal tensor
    Parameters
    ----------
    im2col_shape : shape of A_im2col

    feature_map : feature map

    config: the config of cube

    compute_dtype: dtype of compute result
    -------
    Returns : A_im2col_fractal tensor
    """

    # 'pylint: disable=invalid-name,too-many-locals
    def _im2col_fractal_indices(indices, feature_map):
        """
        calculate im2col_fractal tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        feature_map : feature map

        -------
        Returns : im2col_fractal tvm lambda function
        """
        block_size = config['mac'][1]
        block_size_m = config['mac'][0]
        _, hw, _, kernel_h, kernel_w, c0 = feature_map.shape
        batch_size, i1, j1, i0, j0 = indices
        n_index = batch_size

        hw_index = i1 * block_size_m + i0

        c1_index = (((j1 * block_size + j0) // c0.value) // kernel_w.value) // kernel_h.value

        kh_index = (((j1 * block_size + j0) // c0.value) // kernel_w.value) % kernel_h.value

        kw_index = ((j1 * block_size + j0) // c0.value) % kernel_w.value

        c0_index = (j1 * block_size + j0) % c0.value

        dtype = compute_dtype
        return tvm.select(tvm.any(hw_index < 0, hw_index > hw.value - 1), tvm.const(0.0, dtype),
                          feature_map(n_index, hw_index, c1_index, kh_index, kw_index, c0_index))

    return tvm.compute(im2col_shape,
                       lambda *indices: _im2col_fractal_indices(indices, feature_map),
                       name='im2col_fractal',
                       tag='im2col_fractal')


# 'pylint: disable=too-many-arguments
def _im2col_fractal_v3(shape, fmap, kernel_h, kernel_w, padding, stride, dilate, wo, align_block_size):
    # 'pylint: disable=too-many-arguments
    def __im2col_idx(idx, fmap, kernel_h, kernel_w, padding, stride, dilate):
        _, _, fmap_h, fmap_w, _ = fmap.shape
        n, col_h, col_w, block_size_h, block_size_w = idx
        padding_top, _, padding_left, _ = padding
        stride_h, stride_w = stride
        dilate_h, dilate_w = dilate
        virtual_h = col_h * Constant.BLOCK_SIZE + block_size_h
        virtual_w = col_w * align_block_size + block_size_w

        back_n = n
        back_c1 = virtual_w // align_block_size // kernel_w // kernel_h
        back_h = (virtual_h // wo) * stride_h + (col_w // kernel_w % kernel_h) * dilate_h
        back_w = (virtual_h % wo) * stride_w + (col_w % kernel_w) * dilate_w
        back_c0 = block_size_w

        return tvm.select(
            tvm.any(back_h < padding_top, back_h > fmap_h + padding_top - 1, back_w < padding_left,
                    back_w > fmap_w + padding_left - 1), tvm.const(0, fmap.dtype),
            fmap(back_n, back_c1, back_h - padding_top, back_w - padding_left, back_c0))

    return tvm.compute(shape,
                       lambda *idx: __im2col_idx(idx, fmap, kernel_h, kernel_w, padding, stride, dilate),
                       name="im2col_fractal",
                       tag="im2col_fractal")


# 'pylint: disable=too-many-arguments
def im2col_compute_dynamic(fmap,
                           origin_c_in,
                           ksizes,
                           strides,
                           dilates,
                           pad,
                           out_h,
                           out_w,
                           is_origin_cin_align=True,
                           cin_range=None):
    """
    ops compute

    Parameters
    ----------
    fmap : TVM tensor
        the placeholder of input_x
    origin_c_in : real c size of input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    pad: input attr
    out_h: input attr
    out_w: input attr
    Returns
    -------
    compute results
    """
    # fmap's format is NC1HWC0
    fmap_shape = fmap.shape
    dtype_input = fmap.dtype
    if dtype_input in ('int8', 'uint8'):
        align_block_size = Constant.BLOCK_SIZE_INT8
    else:
        align_block_size = Constant.BLOCK_SIZE

    def _get_dynamic_or_static_value(fmap):
        if isinstance(fmap, tvm.tir.IntImm):
            fmap_value = fmap.value
        else:
            fmap_value = fmap
        return fmap_value

    fmap_n = _get_dynamic_or_static_value(fmap_shape[0])
    fmap_c1 = _get_dynamic_or_static_value(fmap_shape[1])
    fmap_h = _get_dynamic_or_static_value(fmap_shape[2])
    fmap_w = _get_dynamic_or_static_value(fmap_shape[3])
    fmap_c0 = _get_dynamic_or_static_value(fmap_shape[4])

    # out to L1
    fmap_in_l1 = tvm.compute(fmap_shape, lambda *i: fmap[i], name="fmap_in_l1")

    kernel_h, kernel_w = ksizes
    stride_h = strides[0]
    stride_w = strides[0]
    if len(strides) != 1:
        stride_h = strides[0]
        stride_w = strides[1]
    dilate_h = dilates[0]
    dilate_w = dilates[0]
    if len(dilates) != 1:
        dilate_h = dilates[0]
        dilate_w = dilates[1]

    padding_h_before, padding_h_after, padding_w_before, padding_w_after = pad

    stride = (stride_h, stride_w)
    dilate = (dilate_h, dilate_w)

    is_var = get_context().get("ISVAR")

    get_context().add("padding_h_before", padding_h_before)
    get_context().add("padding_h_after", padding_h_after)

    if is_var:
        outh_outw = tbe.var("outh_outw")
        fmapn_outh_outw = tbe.var("fmapn_outh_outw")
        howo = tbe.var("howo")
        howo_div_block_size = tbe.var("howo_div_block_size")
    if isinstance(out_h * out_w, int) or not is_var:
        outh_outw = out_h * out_w
        howo = ((out_h * out_w + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE) * Constant.BLOCK_SIZE
        howo_div_block_size = howo // Constant.BLOCK_SIZE

    if isinstance(fmap_n, int) or isinstance(outh_outw, int) or not is_var:
        fmapn_outh_outw = fmap_n * outh_outw

    fractal_shape = (fmap_n, howo_div_block_size, fmap_c1 * kernel_h * kernel_w, Constant.BLOCK_SIZE, align_block_size)
    fmap_fractal = _im2col_fractal_v3(fractal_shape, fmap_in_l1, kernel_h, kernel_w, pad, stride, dilate, out_w,
                                      align_block_size)

    extract_params = {
        "out_h": out_h,
        "out_w": out_w,
        "fmap_shape": fmap_shape,
        "origin_c_in": origin_c_in,
    }
    setfmatrix_dict = {
        "conv_kernel_h": kernel_h,
        "conv_kernel_w": kernel_w,
        "conv_padding_top": padding_h_before,
        "conv_padding_bottom": padding_h_after,
        "conv_padding_left": padding_w_before,
        "conv_padding_right": padding_w_after,
        "conv_stride_h": stride_h,
        "conv_stride_w": stride_w,
        "conv_dilation_h": dilate_h,
        "conv_dilation_w": dilate_w,
        "conv_fm_c": fmap_c1 * fmap_c0,
        "conv_fm_h": fmap_h,
        "conv_fm_w": fmap_w,
    }

    ub_split_c1_shape = (fmap_n, howo_div_block_size, fmap_c1, kernel_h * kernel_w, Constant.BLOCK_SIZE,
                         align_block_size)
    ub_split_c1_res = _ub_split_c1(ub_split_c1_shape, fmap_fractal, kernel_h * kernel_w)
    ub_transpose_shape = (fmap_n, howo_div_block_size, Constant.BLOCK_SIZE, kernel_h * kernel_w, fmap_c1,
                          align_block_size)
    ub_transpose_res = _ub_transpose(ub_transpose_shape, ub_split_c1_res)

    ub_merge_hw_shape = (fmap_n, howo, kernel_h * kernel_w, fmap_c1, align_block_size)
    ub_merge_hw_res = _ub_merge_hw(ub_merge_hw_shape, ub_transpose_res)
    ub_merge_co_shape = (fmap_n, howo, kernel_h * kernel_w, fmap_c1 * align_block_size)
    ub_merge_co_res = _ub_merge_co(ub_merge_co_shape, ub_merge_hw_res)
    workspace_shape = (fmap_n, outh_outw, kernel_h * kernel_w, fmap_c1 * align_block_size)
    workspace_res = tvm.compute(workspace_shape, lambda *i: ub_merge_co_res[i], name="workspace_res")

    if is_origin_cin_align or (cin_range is not None and cin_range[0] <= align_block_size):
        ub_res_shape = (fmap_n, outh_outw, kernel_h * kernel_w, fmap_c1 * align_block_size)
        ub_res = tvm.compute(ub_res_shape, lambda *i: workspace_res[i], name="ub_res")
    else:
        ub_res_shape = (fmapn_outh_outw * kernel_h * kernel_w, fmap_c1 * align_block_size)
        ub_res = tvm.compute(ub_res_shape,
                             lambda i, c: workspace_res[i // (outh_outw * kernel_h * kernel_w), i //
                                                        (kernel_h * kernel_w) % outh_outw, i %
                                                        (kernel_h * kernel_w), c],
                             name="ub_res")

    if (isinstance(origin_c_in, int) or isinstance(origin_c_in, tvm.tir.IntImm)):
        origin_c_in = int(origin_c_in)

    if origin_c_in == 1 and dtype_input not in ('int8', 'uint8'):
        out_shape = (fmap_n, outh_outw, kernel_h * kernel_w, 1)
        c = tvm.reduce_axis((0, workspace_shape[-1]), "c")
        output_res = tvm.compute(out_shape,
                                 lambda i, j, k, f: tvm.sum(ub_res[i, j, k, c], axis=c),
                                 name="res",
                                 attrs={
                                     'extract_params': extract_params,
                                     'setfmatrix_dict': setfmatrix_dict
                                 })
    else:
        if is_origin_cin_align or (cin_range is not None and cin_range[0] <= align_block_size):
            out_shape = (fmap_n, outh_outw, kernel_h * kernel_w, origin_c_in)
        else:
            out_shape = (fmapn_outh_outw * kernel_h * kernel_w, origin_c_in)
        output_res = tvm.compute(out_shape,
                                 lambda *i: ub_res[i],
                                 name="res",
                                 attrs={
                                     'extract_params': extract_params,
                                     'setfmatrix_dict': setfmatrix_dict
                                 })

    return output_res, workspace_res, workspace_shape


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments,too-many-statements,too-many-lines
@register_operator_compute("Im2col", op_mode="static", support_fusion=True)
def im2col_compute(fmap,
                   origin_c_in,
                   ksizes,
                   strides,
                   dilates,
                   pad,
                   out_h,
                   out_w,
                   is_dynamic=False,
                   is_origin_cin_align=True,
                   cin_range=None):
    """
    ops compute

    Parameters
    ----------
    fmap : TVM tensor
        the placeholder of input_x
    origin_c_in : real c size of input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    pad: input attr
    out_h: input attr
    out_w: input attr
    Returns
    -------
    compute results
    """
    # fmap's format is NC1HWC0
    if is_dynamic:
        output_res, workspace_res, workspace_shape = im2col_compute_dynamic(fmap, origin_c_in, ksizes, strides, dilates,
                                                                            pad, out_h, out_w, is_origin_cin_align,
                                                                            cin_range)
        return output_res, workspace_res, workspace_shape
    fmap_shape = fmap.shape
    dtype_input = fmap.dtype
    if dtype_input in ('int8', 'uint8'):
        align_block_size = Constant.BLOCK_SIZE_INT8
    else:
        align_block_size = Constant.BLOCK_SIZE

    fmap_n = fmap_shape[0].value
    fmap_c1 = fmap_shape[1].value
    fmap_h = fmap_shape[2].value
    fmap_w = fmap_shape[3].value
    fmap_c0 = fmap_shape[4].value

    # out to L1
    fmap_in_l1 = tvm.compute(fmap_shape, lambda *i: fmap[i], name="fmap_in_l1")

    kernel_h, kernel_w = ksizes
    stride_h = strides[0]
    stride_w = strides[0]
    if len(strides) != 1:
        stride_w = strides[1]
    dilate_h = dilates[0]
    dilate_w = dilates[0]
    if len(dilates) != 1:
        dilate_w = dilates[1]

    padding_h_before, padding_h_after, padding_w_before, padding_w_after = pad

    stride = (stride_h, stride_w)
    dilate = (dilate_h, dilate_w)

    fmap_vm_shape = (fmap_n, out_h * out_w, fmap_c1, kernel_h, kernel_w, fmap_c0)
    fmap_im2col = _im2col_row_major_v2(fmap_in_l1, fmap_vm_shape, kernel_h, kernel_w, pad, stride, dilate, dtype_input)

    howo = ((out_h * out_w + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE) * Constant.BLOCK_SIZE
    fractal_shape = (fmap_n, howo // Constant.BLOCK_SIZE, fmap_c1 * kernel_h * kernel_w, Constant.BLOCK_SIZE,
                     align_block_size)
    config = {"mac": [16, align_block_size, 16]}
    fmap_fractal = _im2col_fractal_v2(fractal_shape, fmap_im2col, config, dtype_input)

    extract_params = {
        "out_h": out_h,
        "out_w": out_w,
        "fmap_shape": fmap_shape,
        "origin_c_in": origin_c_in,
    }
    setfmatrix_dict = {
        "conv_kernel_h": kernel_h,
        "conv_kernel_w": kernel_w,
        "conv_padding_top": padding_h_before,
        "conv_padding_bottom": padding_h_after,
        "conv_padding_left": padding_w_before,
        "conv_padding_right": padding_w_after,
        "conv_stride_h": stride_h,
        "conv_stride_w": stride_w,
        "conv_dilation_h": dilate_h,
        "conv_dilation_w": dilate_w,
        "conv_fm_c": fmap_c1 * fmap_c0,
        "conv_fm_h": fmap_h,
        "conv_fm_w": fmap_w,
    }

    ub_split_c1_shape = (fmap_n, howo // Constant.BLOCK_SIZE, fmap_c1, kernel_h * kernel_w, Constant.BLOCK_SIZE,
                         align_block_size)
    ub_split_c1_res = _ub_split_c1(ub_split_c1_shape, fmap_fractal, kernel_h * kernel_w)
    ub_transpose_shape = (fmap_n, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE, kernel_h * kernel_w, fmap_c1,
                          align_block_size)
    ub_transpose_res = _ub_transpose(ub_transpose_shape, ub_split_c1_res)

    ub_merge_hw_shape = (fmap_n, howo, kernel_h * kernel_w, fmap_c1, align_block_size)
    ub_merge_hw_res = _ub_merge_hw(ub_merge_hw_shape, ub_transpose_res)
    ub_merge_co_shape = (fmap_n, howo, kernel_h * kernel_w, fmap_c1 * align_block_size)
    ub_merge_co_res = _ub_merge_co(ub_merge_co_shape, ub_merge_hw_res)
    workspace_shape = (fmap_n, out_h * out_w, kernel_h * kernel_w, fmap_c1 * align_block_size)
    workspace_res = tvm.compute(workspace_shape, lambda *i: ub_merge_co_res[i], name="workspace_res")

    ub_res_shape = (fmap_n, out_h * out_w, kernel_h * kernel_w, fmap_c1 * align_block_size)
    ub_res = tvm.compute(ub_res_shape, lambda *i: workspace_res[i], name="ub_res")

    if origin_c_in == 1 and dtype_input not in ('int8', 'uint8'):
        out_shape = (fmap_n, out_h * out_w, kernel_h * kernel_w)
        c = tvm.reduce_axis((0, workspace_shape[-1]), "c")
        output_res = tvm.compute(out_shape,
                                 lambda i, j, k: tvm.sum(ub_res[i, j, k, c], axis=c),
                                 name="res",
                                 attrs={
                                     'extract_params': extract_params,
                                     'setfmatrix_dict': setfmatrix_dict
                                 })
    else:
        out_shape = (fmap_n, out_h * out_w, kernel_h * kernel_w, origin_c_in)
        output_res = tvm.compute(out_shape,
                                 lambda *i: ub_res[i],
                                 name="res",
                                 attrs={
                                     'extract_params': extract_params,
                                     'setfmatrix_dict': setfmatrix_dict
                                 })

    return output_res, workspace_res, workspace_shape


# 'pylint: disable=too-many-arguments
def _get_tiling_param_cut_howo_col(used_ub_size, lcm_out_w, khkw, cut_h_col, fmap_w, fmap_c0, type_size, origin_c_in,
                                   align_block_size):
    """
    get params for tiling
    """
    # cut howo col
    max_v_ub = (used_ub_size // align_block_size // lcm_out_w + khkw - 1) // (khkw + 1)
    if max_v_ub > Constant.LOAD3D_REPEAT_TIME_LIMIT:
        max_v_ub = Constant.LOAD3D_REPEAT_TIME_LIMIT
    max_v_l1 = Constant.SIZE_L1 // (cut_h_col * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER)
    if max_v_ub > max_v_l1:
        max_v_ub = max_v_l1
    if max_v_ub > 1:
        while origin_c_in % max_v_ub != 0:
            max_v_ub = max_v_ub - 1
    # cut howo col, move_rate
    # move_rate limit according to mte2 bound
    move_rate = 1 / khkw
    return max_v_ub, move_rate


# 'pylint: disable=too-many-locals,too-many-arguments
def _get_tiling_param_cut_howo_row(khkw, fmap_w, fmap_c0, dilated_kernel_h, dilated_kernel_w, stride_h, type_size,
                                   avg_split_ub_size, cut_w_row, cut_h_row, origin_c_in, align_block_size):
    # cut howo row
    max_v_ub = avg_split_ub_size // align_block_size // Constant.BLOCK_SIZE // khkw
    max_v_load3d_limit = Constant.LOAD3D_REPEAT_TIME_LIMIT // khkw
    if max_v_ub > max_v_load3d_limit:
        max_v_ub = max_v_load3d_limit
    max_v_l1 = Constant.SIZE_L1 // (cut_h_row * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER)
    if max_v_ub > max_v_l1:
        max_v_ub = max_v_l1
    if max_v_ub > 1:
        while origin_c_in % max_v_ub != 0:
            max_v_ub = max_v_ub - 1

    # cut howo row, move_rate
    # move_rate useful move rate while mte2 data move
    double_loaded = dilated_kernel_h // 2 - stride_h
    if double_loaded < 0:
        double_loaded = 0
    slide_dis_h = cut_h_row - dilated_kernel_h + 1
    slide_times_h = slide_dis_h // stride_h + 1
    slide_dis_w = cut_w_row - dilated_kernel_w + 1
    move_rate = slide_dis_w / (slide_times_h * fmap_w) * (1 - double_loaded / cut_h_row)
    return max_v_ub, move_rate


# 'pylint: disable=too-many-arguments
def _get_tiling_param_cut_howo_partial_col(out_w, khkw, fmap_w, stride_h, type_size, avg_split_ub_size, cut_h_row,
                                           origin_c_in, align_block_size, dilated_kernel_h):
    """"
    The function is get tiling param cut howo partial col.
    """
    # cut howo col partially
    c_in_align = _ceil_div(origin_c_in, align_block_size) * align_block_size
    max_v_ub = avg_split_ub_size // (khkw * c_in_align * align_block_size)
    max_v_load3d_limit = Constant.LOAD3D_REPEAT_TIME_LIMIT // khkw
    if max_v_ub > max_v_load3d_limit:
        max_v_ub = 0

    w_size = fmap_w * c_in_align * type_size * Constant.DOUBLE_BUFFER
    max_v_l1 = Constant.SIZE_L1 // (dilated_kernel_h * w_size)
    if Constant.SIZE_L1 < (_ceil_div(max_v_l1 * Constant.BLOCK_SIZE, out_w) + 1) * stride_h * w_size \
            or cut_h_row > stride_h + dilated_kernel_h - 1:
        max_v_l1 = Constant.SIZE_L1 // (cut_h_row * w_size)

    if max_v_ub > max_v_l1:
        max_v_ub = max_v_l1
    cut_hw_up_w = (max_v_ub * align_block_size + out_w - 1) // out_w * out_w

    # cut howo col partially, move_rate
    # move_rate useful move rate while mte2 data move
    move_rate = max_v_ub * align_block_size / (cut_hw_up_w + Constant.DELTA)
    return max_v_ub, move_rate


def _get_tiling_param_cut_howo_min(fmap_w, fmap_c0, type_size, avg_split_ub_size, cut_h_row, align_block_size):
    # cut howo khkw c, minimum cut
    max_v_ub = avg_split_ub_size // (1 * align_block_size * Constant.BLOCK_SIZE)
    if max_v_ub > Constant.LOAD3D_REPEAT_TIME_LIMIT:
        max_v_ub = Constant.LOAD3D_REPEAT_TIME_LIMIT
    max_v_l1 = Constant.SIZE_L1 // (cut_h_row * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER)
    if max_v_ub > max_v_l1:
        max_v_ub = max_v_l1

    return max_v_ub


# 'pylint: disable=too-many-locals
def _get_tiling_param(setfmatrix_dict, extract_params, used_ub_size, type_size, avg_split_ub_size, align_block_size):
    out_w = extract_params['out_w']
    fmap_shape = extract_params['fmap_shape']
    origin_c_in = extract_params["origin_c_in"]
    lcm_out_w = extract_params['lcm_out_w']
    cut_h_col = extract_params['cut_h_col']
    cut_w_row = extract_params['cut_w_row']
    cut_h_row = extract_params['cut_h_row']
    dilated_kernel_h = extract_params['dilated_kernel_h']
    dilated_kernel_w = extract_params['dilated_kernel_w']
    fmap_w = fmap_shape[3].value
    fmap_c0 = fmap_shape[4].value
    kernel_h = setfmatrix_dict['conv_kernel_h']
    kernel_w = setfmatrix_dict['conv_kernel_w']
    stride_h = setfmatrix_dict['conv_stride_h']
    khkw = kernel_h * kernel_w

    max_v_cut_col, move_rate_cut_col = _get_tiling_param_cut_howo_col(used_ub_size, lcm_out_w, khkw, cut_h_col, fmap_w,
                                                                      fmap_c0, type_size, origin_c_in, align_block_size)

    max_v_cut_row, move_rate_cut_row = \
        _get_tiling_param_cut_howo_row(khkw, fmap_w, fmap_c0, dilated_kernel_h, dilated_kernel_w, stride_h, type_size,
                                       avg_split_ub_size, cut_w_row, cut_h_row, origin_c_in, align_block_size)

    max_v_cut_col_p, move_rate_cut_col_p = \
        _get_tiling_param_cut_howo_partial_col(out_w, khkw, fmap_w, stride_h, type_size, avg_split_ub_size, cut_h_row,
                                               origin_c_in, align_block_size, dilated_kernel_h)

    max_v_cut_min = _get_tiling_param_cut_howo_min(fmap_w, fmap_c0, type_size, avg_split_ub_size, cut_h_row,
                                                   align_block_size)
    return [
        max_v_cut_col, max_v_cut_row, max_v_cut_col_p, max_v_cut_min, move_rate_cut_col, move_rate_cut_row,
        move_rate_cut_col_p
    ]


# 'pylint: disable=too-many-statements,too-many-branches,too-many-locals
def im2col_schedule(res, sch_list):
    """
    :param res: the multi-results in the operator
    :param sch: schedule list
    """
    sch = sch_list[0]
    setfmatrix_map = res.op.attrs['setfmatrix_dict']
    setfmatrix_dict = {}
    for key, value in setfmatrix_map.items():
        if hasattr(value, "value"):
            setfmatrix_dict[key] = value.value
        else:
            setfmatrix_dict[key] = value

    extract_map = res.op.attrs['extract_params']
    extract_params = {}
    for key, value in extract_map.items():
        if hasattr(value, "value"):
            extract_params[key] = value.value
        else:
            extract_params[key] = value

    out_h = extract_params.get('out_h')
    out_w = extract_params.get('out_w')
    fmap_shape = extract_params.get('fmap_shape')
    origin_c_in = extract_params.get("origin_c_in")
    fmap_n = fmap_shape[0].value
    fmap_c1 = fmap_shape[1].value
    fmap_h = fmap_shape[2].value
    fmap_w = fmap_shape[3].value
    fmap_c0 = fmap_shape[4].value
    kernel_h = setfmatrix_dict.get('conv_kernel_h')
    kernel_w = setfmatrix_dict.get('conv_kernel_w')
    dilate_h = setfmatrix_dict.get('conv_dilation_h')
    dilate_w = setfmatrix_dict.get('conv_dilation_w')
    stride_h = setfmatrix_dict.get('conv_stride_h')
    stride_w = setfmatrix_dict.get('conv_stride_w')

    ub_res = res.op.input_tensors[0]
    workspace_res = ub_res.op.input_tensors[0]
    merge_co_ub = workspace_res.op.input_tensors[0]
    merge_hw_ub = merge_co_ub.op.input_tensors[0]
    transpose_ub = merge_hw_ub.op.input_tensors[0]
    split_c1_ub = transpose_ub.op.input_tensors[0]
    fmap_fractal = split_c1_ub.op.input_tensors[0]
    fmap_im2col = fmap_fractal.op.input_tensors[0]
    fmap_in_l1 = fmap_im2col.op.input_tensors[0]

    sch[fmap_in_l1].set_scope(tbe_platform.scope_cbuf)
    sch[fmap_im2col].set_scope(tbe_platform.scope_cbuf)
    sch[fmap_fractal].set_scope(tbe_platform.scope_ubuf)
    sch[split_c1_ub].set_scope(tbe_platform.scope_ubuf)
    sch[transpose_ub].set_scope(tbe_platform.scope_ubuf)
    sch[merge_hw_ub].set_scope(tbe_platform.scope_ubuf)
    sch[merge_co_ub].set_scope(tbe_platform.scope_ubuf)
    sch[workspace_res].set_scope(tbe_platform.scope_gm)
    sch[ub_res].set_scope(tbe_platform.scope_ubuf)

    dtype_input = ub_res.dtype
    if dtype_input in ('int8', 'uint8'):
        align_block_size = Constant.BLOCK_SIZE_INT8
        type_size = Constant.INT8_SIZE
    else:
        align_block_size = Constant.BLOCK_SIZE
        type_size = Constant.FP16_SIZE

    out_hw_up16 = ((out_h * out_w - 1) // Constant.BLOCK_SIZE + 1) * Constant.BLOCK_SIZE
    dilated_kernel_h = (kernel_h - 1) * dilate_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilate_w + 1
    lcm_out_w = Constant.BLOCK_SIZE // math.gcd(out_w, Constant.BLOCK_SIZE) * out_w
    cut_h_col = (Constant.BLOCK_SIZE // math.gcd(out_w, Constant.BLOCK_SIZE) - 1) * stride_h + 1 + dilated_kernel_h // 2
    if cut_h_col > fmap_h:
        cut_h_col = fmap_h
    # `cut_h_col while cut_hw = Constant.BLOCK_SIZE`
    cut_w_row_s = (Constant.BLOCK_SIZE - 1) * stride_w + 1
    cut_h_row_s = max(stride_h, (((cut_w_row_s - 1) // fmap_w + 1) - 1) * stride_h + 1)
    cut_w_row = cut_w_row_s + dilated_kernel_w - 1
    cut_h_row = cut_h_row_s + dilated_kernel_h - 1
    if lcm_out_w > out_hw_up16:
        lcm_out_w = out_hw_up16

    extract_params['lcm_out_w'] = lcm_out_w
    extract_params['cut_h_col'] = cut_h_col
    extract_params['cut_w_row'] = cut_w_row
    extract_params['cut_h_row'] = cut_h_row
    extract_params['dilated_kernel_h'] = dilated_kernel_h
    extract_params['dilated_kernel_w'] = dilated_kernel_w

    sch[ub_res].buffer_align((1, 1), (1, 1), (1, 1), (1, align_block_size))
    sch[fmap_im2col].buffer_align((1, 1), (out_w, out_w), (1, 1), (1, 1), (1, 1), (1, align_block_size))
    sch[fmap_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE), (1, align_block_size))

    used_ub_size = Constant.SIZE_UB // type_size // Constant.DOUBLE_BUFFER
    avg_split_ub_size = used_ub_size // Constant.NEED_UB_SPACE_NUM
    howo = out_h * out_w
    khkw = kernel_h * kernel_w
    c_out = khkw * fmap_c1 * fmap_c0

    out_shape = [fmap_n, howo, khkw, origin_c_in]
    device_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

    def _split_multi_core_32b_not_aligned(multi_core_factor, dma_split_axis_id, dma_split_factor):
        """
        split multi core, when 32B is not aligned
        """
        res_axis_list = list(res.op.axis).copy()
        workspace_axis_list = list(workspace_res.op.axis).copy()

        res_bind_axis_list = [0 for _ in range(dma_split_axis_id)]
        workspace_bind_axis_list = [0 for _ in range(dma_split_axis_id)]
        for i in range(dma_split_axis_id):
            workspace_bind_axis_list[i], workspace_axis_list[i] = sch[workspace_res].split(workspace_axis_list[i],
                                                                                           factor=multi_core_factor[i])
            res_bind_axis_list[i], res_axis_list[i] = sch[res].split(res_axis_list[i], factor=multi_core_factor[i])
        # 32B not align data copy
        res_axis_list[dma_split_axis_id], dma_copy_axis = sch[res].split(res_axis_list[dma_split_axis_id],
                                                                         factor=dma_split_factor)

        sch[res].reorder(*(res_bind_axis_list + res_axis_list[:dma_split_axis_id] + [dma_copy_axis] +
                           res_axis_list[dma_split_axis_id + 1:]))
        sch[workspace_res].reorder(*(workspace_bind_axis_list + workspace_axis_list))

        res_bind_axis = sch[res].fuse(*(res_bind_axis_list))
        workspace_bind_axis = sch[workspace_res].fuse(*(workspace_bind_axis_list))

        return [[res_bind_axis], res_axis_list, [workspace_bind_axis], workspace_axis_list, dma_copy_axis]

    def _split_multi_core_32b_align(tiling_factor):
        """
        split multi core, when 32B is aligned
        """
        if Constant.SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
            howo_align = Constant.BLOCK_SIZE
        else:
            howo_align = lcm_out_w

        def _get_core_factor(multi_core_factor, core_n, core_howo, core_c):
            multi_core_factor[0] = max(_ceil_div(out_shape[0], core_n), tiling_factor[0])
            multi_core_factor[1] = _ceil_div(max(_ceil_div(out_shape[1], core_howo), tiling_factor[1]),
                                             howo_align) * howo_align
            multi_core_factor[3] = _ceil_div(max(_ceil_div(out_shape[3], core_c), tiling_factor[3]),
                                             align_block_size) * align_block_size
            return multi_core_factor

        pre_core_n, pre_core_c, pre_core_howo = [1], [1], [1]
        for i in range(1, device_core_num + 1):
            multi_core_factor = _get_core_factor(out_shape.copy(), i, i, i)
            pre_core_n.append(_ceil_div(out_shape[0], multi_core_factor[0]))
            pre_core_howo.append(_ceil_div(out_shape[1], multi_core_factor[1]))
            pre_core_c.append(_ceil_div(out_shape[3], multi_core_factor[3]))

        core_n, core_c, core_howo = _cal_multi_core_factor_3d(_ceil_div(out_shape[0], tiling_factor[0]),
                                                              _ceil_div(out_shape[3], tiling_factor[3]),
                                                              _ceil_div(out_shape[1], tiling_factor[1]), pre_core_n,
                                                              pre_core_c, pre_core_howo)
        multi_core_factor = _get_core_factor(out_shape.copy(), core_n, core_howo, core_c)

        res_axis_list = list(res.op.axis).copy()
        res_bind_axis_list = [0 for _ in res_axis_list]
        for i, _ in enumerate(res_bind_axis_list):
            res_bind_axis_list[i], res_axis_list[i] = sch[res].split(res_axis_list[i], factor=multi_core_factor[i])
        sch[res].reorder(*(res_bind_axis_list + res_axis_list))
        res_bind_axis = sch[res].fuse(*(res_bind_axis_list))

        return [res_bind_axis], res_axis_list

    def _get_dma_split_factor(dma_split_axis_id, out_shape, max_value):
        """
        get split factor
        """
        split_eles = _prod(out_shape[dma_split_axis_id:])
        ele_len = _prod(out_shape[dma_split_axis_id + 1:])

        def _could_split_multi_core(val):
            if val * ele_len > max_value:
                return False
            tail_len = split_eles % (val * ele_len)
            return (tail_len > align_block_size) or (val * ele_len > align_block_size and tail_len == 0)

        if _could_split_multi_core(out_shape[dma_split_axis_id]):
            return out_shape[dma_split_axis_id], True, True

        if dma_split_axis_id == 1 and _could_split_multi_core(out_w):  # howo
            return out_w, True, True

        if dma_split_axis_id == 2 and _could_split_multi_core(kernel_w):  # khkw
            return kernel_w, True, True

        for val in range(align_block_size, out_shape[dma_split_axis_id], align_block_size):
            if _could_split_multi_core(val):
                return val, (out_shape[dma_split_axis_id] % val == 0), True

        return 1, False, False

    def _cal_multi_core_factor(m, n, m_list, n_list):
        """
        Return the cut factors for multicore axis.
        """

        m_list = list(set(m_list))
        n_list = list(set(n_list))

        m_list.sort(reverse=True)
        n_list.sort(reverse=True)

        min_cycle_num = m * n
        core_m, core_n = m_list[-1], n_list[-1]

        for i in m_list:
            for j in n_list:
                if i * j > device_core_num:
                    continue
                tmp_cycle_num = _ceil_div(m, i) * _ceil_div(n, j)
                if tmp_cycle_num < min_cycle_num:
                    min_cycle_num = tmp_cycle_num
                    core_m, core_n = i, j
                break
        return core_m, core_n

    def _cal_multi_core_factor_3d(n, c, howo, n_list, c_list, howo_list):
        """
        Return the cut factors for multicore axis.
        """
        n_list = list(set(n_list))
        c_list = list(set(c_list))
        howo_list = list(set(howo_list))

        n_list.sort(reverse=True)
        c_list.sort(reverse=True)
        howo_list.sort(reverse=True)

        min_cycle_num = n * c * howo
        core_n, core_c, core_howo = n_list[-1], c_list[-1], howo_list[-1]

        for i in n_list:
            for j in c_list:
                if i * j > device_core_num:
                    continue
                for k in howo_list:
                    if i * j * k > device_core_num:
                        continue
                    tmp_cycle_num = _ceil_div(n, i) * _ceil_div(c, j) * _ceil_div(howo, k)
                    if tmp_cycle_num < min_cycle_num:
                        min_cycle_num = tmp_cycle_num
                        core_n, core_c, core_howo = i, j, k
                    break
        return core_n, core_c, core_howo

    def _get_multi_core_factor(dma_split_axis_id, tiling_factor):
        """
        get multi core split factor
        """
        multi_core_factor = out_shape.copy()
        if dma_split_axis_id == 0:  # n
            return multi_core_factor
        if dma_split_axis_id == 1:  # howo
            used_core_num = min(device_core_num, _ceil_div(out_shape[0], tiling_factor[0]))
            multi_core_factor[0] = max(_ceil_div(out_shape[0], used_core_num), tiling_factor[0])
            return multi_core_factor

        if Constant.SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
            howo_align = Constant.BLOCK_SIZE
        elif Constant.SIZE_L1 >= cut_h_col * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
            howo_align = lcm_out_w
        else:
            howo_align = howo

        def _get_core_factor(multi_core_factor, core_n, core_howo):
            multi_core_factor[0] = max(_ceil_div(out_shape[0], core_n), tiling_factor[0])
            multi_core_factor[1] = _ceil_div(max(_ceil_div(out_shape[1], core_howo), tiling_factor[1]),
                                             howo_align) * howo_align
            return multi_core_factor

        pre_core_n, pre_core_howo = [1], [1]
        for i in range(1, device_core_num + 1):
            multi_core_factor = _get_core_factor(out_shape.copy(), i, i)
            pre_core_n.append(_ceil_div(out_shape[0], multi_core_factor[0]))
            pre_core_howo.append(_ceil_div(out_shape[1], multi_core_factor[1]))

        core_n, core_howo = _cal_multi_core_factor(_ceil_div(out_shape[0], tiling_factor[0]),
                                                   _ceil_div(out_shape[1], tiling_factor[1]), pre_core_n, pre_core_howo)
        multi_core_factor = _get_core_factor(out_shape.copy(), core_n, core_howo)

        return multi_core_factor

    def _schedule_32b_not_aligned(dma_split_axis_id, dma_split_factor, allow_multi_core, reg_mov=True):
        """
        schedule, when 32B is not aligned
        """
        n_factor = 1
        howo_factor = howo
        khkw_factor = khkw
        c_factor = origin_c_in
        tiling_param = _get_tiling_param(setfmatrix_dict, extract_params, used_ub_size, type_size, avg_split_ub_size,
                                         align_block_size)

        max_v_cut_col, max_v_cut_row, max_v_cut_col_p, _, move_rate_cut_col, move_rate_cut_row, \
            move_rate_cut_col_p = tiling_param
        move_rate = 0
        if max_v_cut_col > 0:
            move_rate = move_rate_cut_col
        if move_rate < move_rate_cut_row and max_v_cut_row > 0:
            move_rate = move_rate_cut_row
        if move_rate < move_rate_cut_col_p and max_v_cut_col_p > 0:
            move_rate = move_rate_cut_col_p

        if lcm_out_w * c_out <= avg_split_ub_size and khkw * fmap_c1 <= Constant.LOAD3D_REPEAT_TIME_LIMIT \
                and max_v_cut_col > 0 and max_v_cut_row > 0 \
                and Constant.SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
            max_v = avg_split_ub_size // lcm_out_w // c_out
            if lcm_out_w * max_v < howo:
                # if True cut n howo else only cut n
                howo_factor = lcm_out_w * max_v
        elif move_rate == move_rate_cut_col and max_v_cut_col > 0:
            # cut howo col
            howo_factor = lcm_out_w
            khkw_factor = 1
            c_factor = align_block_size
        elif move_rate == move_rate_cut_row and max_v_cut_row > 0:
            # cut howo row
            howo_factor = Constant.BLOCK_SIZE
            khkw_factor = khkw
            c_factor = align_block_size
        elif move_rate == move_rate_cut_col_p and max_v_cut_col_p > 0:
            # cut howo col partially
            howo_factor = Constant.BLOCK_SIZE * max_v_cut_col_p
            c_factor = origin_c_in
            khkw_factor = khkw
        else:
            # cut howo khkw c
            howo_factor = Constant.BLOCK_SIZE
            khkw_factor = 1
            c_factor = align_block_size

        tiling_factor = [n_factor, howo_factor, khkw_factor, c_factor]

        if reg_mov:
            reg_mov_ub = sch.cache_write(res, tbe_platform.scope_ubuf)
        if allow_multi_core:
            multi_core_factor = _get_multi_core_factor(dma_split_axis_id, tiling_factor)
        else:
            multi_core_factor = out_shape.copy()

        split_multi_core_axis_list = _split_multi_core_32b_not_aligned(multi_core_factor, dma_split_axis_id,
                                                                       dma_split_factor)
        res_bind_list, res_axis_list, workspace_bind_list, workspace_axis_list, dma_copy_axis = \
            split_multi_core_axis_list

        workspace_res_n_outer, workspace_res_n_inner = sch[workspace_res].split(workspace_axis_list[0],
                                                                                factor=tiling_factor[0])
        workspace_res_howo_outer, workspace_res_howo_inner = sch[workspace_res].split(workspace_axis_list[1],
                                                                                      factor=tiling_factor[1])
        workspace_res_khkw_outer, workspace_res_khkw_inner = sch[workspace_res].split(workspace_axis_list[2],
                                                                                      factor=tiling_factor[2])
        workspace_res_c1_inner_outer, workspace_res_c1_inner = sch[workspace_res].split(workspace_axis_list[3],
                                                                                        factor=align_block_size)
        workspace_res_c1_outer, workspace_res_c1_inner_outer = \
            sch[workspace_res].split(workspace_res_c1_inner_outer, factor=max(tiling_factor[3] // align_block_size, 1))

        workspace_axis_outer_list = [
            workspace_res_n_outer, workspace_res_howo_outer, workspace_res_khkw_outer, workspace_res_c1_outer
        ]
        workspace_axis_inner_list = [
            workspace_res_n_inner, workspace_res_c1_inner_outer, workspace_res_howo_inner, workspace_res_khkw_inner,
            workspace_res_c1_inner
        ]

        if Constant.SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
            compute_at_id = 0
        elif Constant.SIZE_L1 >= cut_h_row * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER \
                and move_rate != move_rate_cut_col:
            compute_at_id = 1
        elif Constant.SIZE_L1 >= cut_h_col * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER \
                and move_rate == move_rate_cut_col:
            compute_at_id = 1
        elif Constant.SIZE_L1 >= cut_h_col * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER \
                and move_rate == move_rate_cut_col:
            compute_at_id = 1
            workspace_c_out_outer, workspace_axis_outer_list[3] = sch[workspace_res].split(workspace_axis_outer_list[3],
                                                                                           factor=1)
            workspace_bind_list.append(workspace_c_out_outer)
        else:
            compute_at_id = 2
            workspace_c_out_outer, workspace_axis_outer_list[3] = sch[workspace_res].split(workspace_axis_outer_list[3],
                                                                                           factor=1)
            workspace_bind_list.append(workspace_c_out_outer)
            workspace_axis_outer_list[2], _ = sch[workspace_res].split(workspace_axis_outer_list[2],
                                                                       factor=max(kernel_w // tiling_factor[2], 1))

        sch[workspace_res].reorder(*(workspace_bind_list + workspace_axis_outer_list + workspace_axis_inner_list))

        sch[fmap_im2col].compute_at(sch[workspace_res], workspace_axis_outer_list[compute_at_id])
        sch[fmap_in_l1].compute_at(sch[workspace_res], workspace_axis_outer_list[compute_at_id])

        sch[merge_co_ub].compute_at(sch[workspace_res], workspace_axis_outer_list[3])
        sch[merge_hw_ub].compute_at(sch[workspace_res], workspace_axis_outer_list[3])
        sch[transpose_ub].compute_at(sch[workspace_res], workspace_axis_outer_list[3])
        sch[split_c1_ub].compute_at(sch[workspace_res], workspace_axis_outer_list[3])
        sch[fmap_fractal].compute_at(sch[workspace_res], workspace_axis_outer_list[3])

        sch[ub_res].compute_at(sch[res], res_axis_list[dma_split_axis_id])
        if reg_mov:
            sch[reg_mov_ub].compute_at(sch[res], res_axis_list[dma_split_axis_id])

        block = tvm.thread_axis("blockIdx.x")
        sch[res].bind(res_bind_list[0], block)
        sch[workspace_res].bind(workspace_bind_list[0], block)

        sch[split_c1_ub].compute_inline()
        sch[merge_co_ub].compute_inline()

        sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], tbe_platform.DMA_COPY)
        sch[fmap_im2col].emit_insn(fmap_im2col.op.axis[0], tbe_platform.SET_FMATRIX, setfmatrix_dict)
        sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], tbe_platform.IM2COL)
        sch[split_c1_ub].emit_insn(split_c1_ub.op.axis[0], tbe_platform.DMA_COPY)

        if dtype_input in ("int8", "uint8"):
            sch[transpose_ub].emit_insn(transpose_ub.op.axis[0], tbe_platform.DMA_COPY)
            sch[merge_hw_ub].emit_insn(merge_hw_ub.op.axis[0], tbe_platform.DMA_COPY)
        else:
            sch[transpose_ub].emit_insn(transpose_ub.op.axis[0], tbe_platform.ADDVS)
            sch[merge_hw_ub].emit_insn(merge_hw_ub.op.axis[0], tbe_platform.ADDVS)

        sch[merge_co_ub].emit_insn(merge_co_ub.op.axis[0], tbe_platform.DMA_COPY)
        sch[workspace_res].emit_insn(workspace_axis_inner_list[0], tbe_platform.DMA_COPY)
        sch[ub_res].emit_insn(ub_res.op.axis[0], tbe_platform.DMA_COPY)
        if reg_mov:
            if origin_c_in == 1 and dtype_input not in ('int8', 'uint8'):
                sch[reg_mov_ub].emit_insn(reg_mov_ub.op.axis[0], tbe_platform.REDUCE_SUM)
            else:
                sch[reg_mov_ub].emit_insn(reg_mov_ub.op.axis[0], tbe_platform.DATA_MOV)
        sch[res].emit_insn(dma_copy_axis, tbe_platform.DMA_PADDING)

    if origin_c_in % align_block_size == 0:
        n_factor = 1
        howo_factor = howo
        khkw_factor = khkw
        c_factor = origin_c_in
        max_v = fmap_c1
        tiling_param = _get_tiling_param(setfmatrix_dict, extract_params, used_ub_size, type_size, avg_split_ub_size,
                                         align_block_size)

        max_v_cut_col, max_v_cut_row, max_v_cut_col_p, max_v_cut_min, move_rate_cut_col, move_rate_cut_row, \
            move_rate_cut_col_p = tiling_param

        move_rate = 0
        if max_v_cut_col > 0:
            move_rate = move_rate_cut_col
        if move_rate < move_rate_cut_row and max_v_cut_row > 0:
            move_rate = move_rate_cut_row
        if move_rate < move_rate_cut_col_p and max_v_cut_col_p > 0:
            move_rate = move_rate_cut_col_p
        split_khkw_mode = False
        if lcm_out_w * c_out <= avg_split_ub_size and khkw * fmap_c1 <= Constant.LOAD3D_REPEAT_TIME_LIMIT \
                and max_v_cut_col > 0 and max_v_cut_row > 0 \
                and Constant.SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
            max_v = avg_split_ub_size // lcm_out_w // c_out
            if lcm_out_w * max_v < howo:
                # if True cut n howo else only cut n
                howo_factor = lcm_out_w * max_v
        elif move_rate == move_rate_cut_col and max_v_cut_col > 0:
            # cut howo col
            howo_factor = lcm_out_w
            max_v = max_v_cut_col
            khkw_factor = 1
            c_factor = align_block_size * max_v
        elif move_rate == move_rate_cut_row and max_v_cut_row > 0:
            # cut howo row
            howo_factor = Constant.BLOCK_SIZE
            khkw_factor = khkw
            max_v = max_v_cut_row
            c_factor = align_block_size * max_v
        elif move_rate == move_rate_cut_col_p and max_v_cut_col_p > 0:
            # cut howo col partially
            howo_factor = Constant.BLOCK_SIZE * max_v_cut_col_p
            c_factor = origin_c_in
            khkw_factor = khkw
            max_v = fmap_c1
        else:
            # cut howo khkw c
            howo_factor = Constant.BLOCK_SIZE
            max_v = max_v_cut_min
            if max_v == 0:
                max_v = 1
                split_khkw_mode = True
            # The instruction parameter is uint8 type.
            if khkw * max_v >= 256:
                max_v = max(255 // khkw, 1)
            khkw_factor = 1
            c_factor = align_block_size * max_v

        tiling_factor = [n_factor, howo_factor, khkw_factor, c_factor]
        res_bind_list, res_axis_list = _split_multi_core_32b_align(tiling_factor)

        res_n_outer, res_n_inner = sch[res].split(res_axis_list[0], factor=tiling_factor[0])
        res_howo_outer, res_howo_inner = sch[res].split(res_axis_list[1], factor=tiling_factor[1])
        res_khkw_outer, res_khkw_inner = sch[res].split(res_axis_list[2], factor=tiling_factor[2])
        res_c_inner_outer, res_c_inner = sch[res].split(res_axis_list[3], factor=align_block_size)
        res_c_outer, res_c_outer_inner = sch[res].split(res_c_inner_outer, factor=tiling_factor[3] // align_block_size)

        res_axis_outer_list = [res_n_outer, res_howo_outer, res_khkw_outer, res_c_outer]

        if Constant.SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
            compute_at_id = 0
        elif Constant.SIZE_L1 >= cut_h_row * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER \
                and move_rate != move_rate_cut_col:
            compute_at_id = 1
        elif Constant.SIZE_L1 >= cut_h_col * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER \
                and move_rate == move_rate_cut_col:
            compute_at_id = 1
        elif Constant.SIZE_L1 >= cut_h_col * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER \
                and move_rate == move_rate_cut_col:
            compute_at_id = 1
            res_c_out_outer, res_axis_outer_list[3] = sch[res].split(res_axis_outer_list[3], factor=1)
            res_bind_list.append(res_c_out_outer)
        elif Constant.SIZE_L1 >= cut_h_row_s * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER \
                and split_khkw_mode:
            compute_at_id = 2
            res_axis_outer_list[2], _ = sch[res].split(res_axis_outer_list[2], factor=max(kernel_w // khkw_factor, 1))
            res_c_out_outer, res_axis_outer_list[3] = sch[res].split(res_axis_outer_list[3], factor=1)
            res_bind_list.append(res_c_out_outer)
        else:
            compute_at_id = 3

        sch[res].reorder(*(res_bind_list + res_axis_outer_list +
                           [res_n_inner, res_c_outer_inner, res_howo_inner, res_khkw_inner, res_c_inner]))

        sch[fmap_im2col].compute_at(sch[res], res_axis_outer_list[compute_at_id])
        sch[fmap_in_l1].compute_at(sch[res], res_axis_outer_list[compute_at_id])
        sch[transpose_ub].compute_at(sch[res], res_axis_outer_list[3])
        sch[fmap_fractal].compute_at(sch[res], res_axis_outer_list[3])

        sch[workspace_res].compute_inline()
        sch[ub_res].compute_inline()
        sch[merge_co_ub].compute_inline()
        sch[merge_hw_ub].compute_inline()
        sch[split_c1_ub].compute_inline()

        block = tvm.thread_axis("blockIdx.x")
        sch[res].bind(res_bind_list[0], block)

        sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], tbe_platform.DMA_COPY)
        sch[fmap_im2col].emit_insn(fmap_im2col.op.axis[0], tbe_platform.SET_FMATRIX, setfmatrix_dict)
        sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], tbe_platform.IM2COL)
        sch[transpose_ub].emit_insn(transpose_ub.op.axis[0], tbe_platform.DMA_COPY)
        sch[res].emit_insn(res_n_inner, tbe_platform.DMA_COPY)
    else:
        out_shape_len = len(out_shape)
        multi_core_i = out_shape_len - 1
        prod = 1
        for i in range(out_shape_len):
            prod = prod * out_shape[i]
            if prod > device_core_num:
                multi_core_i = i
                break
        dma_split_i = 0
        prod = 1
        for i in range(out_shape_len - 1, -1, -1):
            prod = prod * out_shape[i]
            if prod > align_block_size:
                dma_split_i = i
                break

        res_ub_num = _ceil_div(_ceil_div(origin_c_in, align_block_size) * align_block_size, origin_c_in) + 1
        max_ub_limit_32b = min(_ceil_div(_prod(out_shape), device_core_num), avg_split_ub_size // res_ub_num)
        max_ub_limit_32b = max(align_block_size, max_ub_limit_32b)

        for i in range(min(multi_core_i + 1, dma_split_i), dma_split_i + 1):
            dma_split_factor, align_split, allow_multi_core = _get_dma_split_factor(i, out_shape, max_ub_limit_32b)
            if align_split or i == dma_split_i:
                _schedule_32b_not_aligned(i, dma_split_factor, allow_multi_core, reg_mov=(i != out_shape_len - 1))
                break

    sch[fmap_in_l1].double_buffer()
    sch[fmap_im2col].double_buffer()
    sch[fmap_fractal].double_buffer()
    sch[transpose_ub].double_buffer()
    sch[ub_res].double_buffer()
