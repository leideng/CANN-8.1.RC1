#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
extract_volume_patches
"""
# 'pylint: disable=too-many-lines

import functools
import math
import os
import re
import stat

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import tbe_platform
from tbe.dsl.compute.common import tf_get_windowed_output_size_verbose
from impl.util.platform_adapter import tvm
from impl.util.util_common import write_code
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.platform_adapter import build_config


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    BLOCK_SIZE = 16
    BLOCK_SIZE_FP16 = 16
    BLOCK_SIZE_INT8 = 32

    DOUBLE_BUFFER = 2

    MAX_UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    MAX_L1_SIZE = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
    DEVICE_CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)


# 'pylint: disable = unused-argument,redefined-builtin,too-many-locals,too-many-arguments
def get_op_support_info(input_x, output_y, ksizes, strides, padding, kernel_name="extract_volume_patches"):
    """
    get extract_volume_patches slice info
    """
    format_x = input_x.get("format")
    shape_x = input_x.get("shape")
    data_format = input_x.get("ori_format")
    _, filter_d, filter_h, _, _ = ksizes
    if data_format == "NCDHW":
        _, _, filter_d, filter_h, _ = ksizes
    if format_x == "NDC1HWC0":
        input_d = shape_x[1]
        input_h = shape_x[3]
        if (input_h == filter_h and input_d == filter_d) or padding == "SAME":
            axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])]]
        elif padding == "VALID":
            if input_h != filter_h and input_d != filter_d:
                axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]),
                                      SplitOutput([0, [0]])], [SplitInput([0, [1], [0], [0]]),
                                                               SplitOutput([0, [1]])],
                                     [SplitInput([0, [3], [0], [0]]),
                                      SplitOutput([0, [3]])]]
            elif input_h == filter_h and input_d != filter_d:
                axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]),
                                      SplitOutput([0, [0]])], [SplitInput([0, [1], [0], [0]]),
                                                               SplitOutput([0, [1]])]]
            else:
                axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]),
                                      SplitOutput([0, [0]])], [SplitInput([0, [3], [0], [0]]),
                                                               SplitOutput([0, [3]])]]
        else:
            axis_split_matrix = None

        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)

    return op_cal_info_in_json


# 'pylint: disable=too-many-arguments,invalid-name,too-many-statements,too-many-branches,too-many-locals
def _check_shape_and_format_vailded(input_x, output_y, ksizes, strides, padding, kernel_name):
    """
    check whether the input param valid or not
    """
    shape_x = input_x.get("ori_shape")
    shape_y = output_y.get("shape")
    ori_format = input_x.get("ori_format")

    if ori_format not in ("NCDHW", "NDHWC"):
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", 'NDHWC or NCDHW', ori_format)

    if len(ksizes) == 5 and len(strides) == 5:
        if ori_format == "NDHWC":
            _, kernel_d, kernel_h, kernel_w, _ = ksizes
            _, stride_d, stride_h, stride_w, _ = strides
            fmap_n, fmap_d, fmap_h, fmap_w, fmap_c = shape_x
        else:
            _, _, kernel_d, kernel_h, kernel_w = ksizes
            _, _, stride_d, stride_h, stride_w = strides
            fmap_n, fmap_c, fmap_d, fmap_h, fmap_w = shape_x
            shape_x = [fmap_n, fmap_d, fmap_h, fmap_w, fmap_c]
            ksizes = [1, kernel_d, kernel_h, kernel_w, 1]
            strides = [1, stride_d, stride_h, stride_w, 1]
    elif len(ksizes) != 5:
        error_manager_vector.raise_err_input_param_range_invalid(kernel_name, 'ksizes', 5, 5, len(ksizes))
    else:
        error_manager_vector.raise_err_input_param_range_invalid(kernel_name, 'strides', 5, 5, len(strides))

    if kernel_h > 255 or kernel_h < 1:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, 'kernel', 1, 255, kernel_h)
    if kernel_w > 255 or kernel_w < 1:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, 'kernel', 1, 255, kernel_w)

    if stride_h > 63 or stride_h < 1:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, 'strides', 1, 63, stride_h)
    if stride_w > 63 or stride_w < 1:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, 'strides', 1, 63, stride_w)

    if padding not in ("SAME", "VALID"):
        error_manager_vector.raise_err_pad_mode_invalid(kernel_name, 'VALID or SAME', padding)

    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="output_y")

    dtype_x = input_x.get("dtype").lower()
    dtype_y = output_y.get("dtype").lower()

    para_check.check_dtype(dtype_x, ("float16", "uint8", "int8"), param_name="input_x")
    para_check.check_dtype(dtype_y, ("float16", "uint8", "int8"), param_name="output_y")
    if dtype_x != dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "input_x", "output_y", dtype_x, dtype_y)

    out_d, _, _ = tf_get_windowed_output_size_verbose(fmap_d, kernel_d, stride_d, padding)
    out_h, _, _ = tf_get_windowed_output_size_verbose(fmap_h, kernel_h, stride_h, padding)
    out_w, pad_left, pad_right = tf_get_windowed_output_size_verbose(
        fmap_w, kernel_w, stride_w, padding)

    # check whether fmap_d, fmap_h and fmap_w are valid
    dilation_rate = 1
    for input_size, kernel_size, stride in ((fmap_d, kernel_d, stride_d), (fmap_h, kernel_h, stride_h),
                                            (fmap_w, kernel_w, stride_w)):
        effective_kernel_size = (kernel_size - 1) * dilation_rate + 1
        if padding == "SAME":
            output_size = (input_size + stride - 1) // stride
            padding_needed = (output_size - 1) * stride + effective_kernel_size - input_size
            if padding_needed < 0:
                error_manager_vector.raise_err_specific_reson(
                    kernel_name, "The padding in the same mode must be greater than or equal to 0!")

    block_size_align = Constant.BLOCK_SIZE_FP16 if dtype_x == "float16" else Constant.BLOCK_SIZE_INT8
    dtype_size = 2 if dtype_x == "float16" else 1

    if (kernel_h * fmap_w * block_size_align * dtype_size) > Constant.MAX_UB_SIZE:
        error_manager_vector.raise_err_specific_reson(kernel_name,
                                                      "UB's memory space is not enough to support the tiling shape!")

    # cloud out_size_h = 1 or out_size_w = 1, img2col does not act normally
    if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) == "Ascend910" and (out_h != 1 or out_w != 1):
        if fmap_w + pad_left + pad_right - kernel_w < stride_w:
            error_manager_vector.raise_err_specific_reson(
                kernel_name,
                "Invalid params in the platform of cloud, it must be fmap_w + pad_l + pad_r - kernel_w >= stride_w!")

    expect_shape_y = (fmap_n, out_d, (kernel_d * kernel_h * kernel_w * fmap_c + block_size_align - 1) \
                      // block_size_align, out_h, out_w, block_size_align)

    if list(expect_shape_y) != list(shape_y):
        error_manager_vector.raise_err_inputs_shape_not_equal(kernel_name, "expect_shape_y", "output_y",
                                                              list(expect_shape_y), list(shape_y),
                                                              list(expect_shape_y))

    if kernel_d <= 0 or stride_d <= 0:
        error_manager_vector.raise_err_specific_reson(kernel_name,
                                                      "Kernel and strides must be greater than 0 in the D dimension!")

    if len(shape_x) != 5:
        error_manager_vector.raise_err_input_param_range_invalid(kernel_name, 'input_x', 5, 5, len(shape_x))
    if len(shape_y) != 6:
        error_manager_vector.raise_err_input_param_range_invalid(kernel_name, 'output_y', 6, 6, len(shape_y))

    fmap_c1 = (fmap_c + block_size_align - 1) // block_size_align
    fmap_c0 = block_size_align
    type_size = 2 if dtype_x == "float16" else 1
    cut_h = Constant.BLOCK_SIZE // math.gcd(out_w, Constant.BLOCK_SIZE) * stride_h + kernel_h
    if cut_h > fmap_h:
        cut_h = fmap_h
    if cut_h * fmap_w * fmap_c1 * fmap_c0 * type_size * Constant.DOUBLE_BUFFER > Constant.MAX_L1_SIZE:
        error_manager_vector.raise_err_specific_reson(
            kernel_name, "Input size is too large load to L1, while cut h, need size: %d" %
            (cut_h * fmap_w * fmap_c1 * fmap_c0 * type_size * Constant.DOUBLE_BUFFER))

    return shape_x, ksizes, strides


def _ceil_to(value, ceil_value):
    """
    Return the least multiple of ceil_value integer number(output > 0)
    which is greater than or equal to x.
    """
    if ceil_value <= 0:
        return value
    return ((value + ceil_value - 1) // ceil_value) * ceil_value


# 'pylint: disable=too-many-return-statements
def _cal_multi_core_factor_3_axis(m, n, t):
    """
    Return the proper cut factors for 3 multicore axis.
    """
    def _ceil(x, y):
        return (x + y - 1) // y

    # for pruning
    if m * n * t > 65535:
        max_gcd_m = math.gcd(m, Constant.DEVICE_CORE_NUM)
        max_gcd_n = math.gcd(n, Constant.DEVICE_CORE_NUM)
        max_gcd_t = math.gcd(t, Constant.DEVICE_CORE_NUM)
        if max_gcd_m >= Constant.DEVICE_CORE_NUM:
            return Constant.DEVICE_CORE_NUM, 1, 1
        if max_gcd_m * max_gcd_n >= Constant.DEVICE_CORE_NUM:
            return max_gcd_m, Constant.DEVICE_CORE_NUM // max_gcd_m, 1
        if max_gcd_m * max_gcd_n * max_gcd_t >= Constant.DEVICE_CORE_NUM:
            return max_gcd_m, max_gcd_m, Constant.DEVICE_CORE_NUM // (max_gcd_m * max_gcd_n)

    split_loop_partition_bug_list = [[53, 7, 5]]
    if [m, n, t] in split_loop_partition_bug_list:
        if Constant.DEVICE_CORE_NUM == 32:
            return 6, 1, 5
        if Constant.DEVICE_CORE_NUM == 8:
            return 8, 1, 1
        if Constant.DEVICE_CORE_NUM == 2:
            return 2, 1, 1

    def _init_core(m, n, t):
        for i in range(m, 0, -1):
            for j in range(n, 0, -1):
                for k in range(t, 0, -1):
                    if i * j * k < 65536:
                        return i, j, k

    core_m, core_n, core_t = _init_core(m, n, t)
    min_cycle_num = _ceil(m, core_m) * _ceil(n, core_n) * _ceil(t, core_t) * \
                    _ceil(core_m * core_n * core_t, Constant.DEVICE_CORE_NUM)
    min_core = core_m * core_n * core_t

    for i in range(m, 0, -1):
        for j in range(n, 0, -1):
            for k in range(t, 0, -1):
                if _ceil(m, i) * _ceil(n, j) * _ceil(t, k) * \
                    _ceil(i * j * k, Constant.DEVICE_CORE_NUM) > min_cycle_num:
                    continue
                if i * j * k < min_core:
                    min_core = i * j * k
                    core_m, core_n, core_t = i, j, k

    return core_m, core_n, core_t


# 'pylint: disable=invalid-name
def _din_img2col(image_patches_res_shape, image_patches_res, padding, stride_d, float16_align_flag):
    """
    calculate din_img2col tensor
    Parameters
    ----------
    image_patches_res : image_patches_res

    image_patches_res_shape : shape of A_din_img2col

    padding: the padding shape

    stride_d: the stride value in d
    -------
    Returns : A_din_img2col tensor
    """

    # 'pylint: disable=too-many-locals
    def _din_img2col_compute(indices, image_patches_res, padding, stride_d, float16_align_flag):
        """
        calculate din_img2col tensor
        Parameters
        ----------
        indices : indices in lambda function

        image_patches_res : feature map

        padding: the padding shape

        stride_d: the stride value in d
        -------
        Returns : A_din_img2col tvm lambda function
        """
        _, in_d, _, _, _, _, _ = image_patches_res.shape
        if image_patches_res.dtype != "float16" or float16_align_flag:
            n, do, howo1, howo0, kd, c1, khkw, c0 = indices
        else:
            n, do, howo1, c1, kd, khkw, howo0, c0 = indices
        padding_top, _, _, _ = padding

        n_index = n
        d_index = do * stride_d + kd
        howo1_index = howo1
        howo0_index = howo0
        c1_index = c1
        khkw_index = khkw
        c0_index = c0
        if image_patches_res.dtype in ("uint8", "int8"):
            return tvm.select(
                tvm.any(d_index < padding_top, d_index > in_d.value + padding_top - 1),
                tvm.const(0, image_patches_res.dtype),
                image_patches_res(n_index, d_index - padding_top, howo1_index, howo0_index, c1_index, khkw_index,
                                  c0_index))
        if float16_align_flag:
            return tvm.select(
                tvm.any(d_index < padding_top, d_index > in_d.value + padding_top - 1),
                tvm.const(0, image_patches_res.dtype),
                image_patches_res(n_index, d_index - padding_top, howo1_index, howo0_index, c1_index, khkw_index,
                                  c0_index) + tvm.const(0, image_patches_res.dtype))
        return tvm.select(
            tvm.any(d_index < padding_top,
                    d_index > in_d.value + padding_top - 1), tvm.const(0, image_patches_res.dtype),
            image_patches_res(n_index, d_index - padding_top, howo1_index, c1_index, khkw_index, howo0_index, c0_index)
            + tvm.const(0, image_patches_res.dtype))

    return tvm.compute(
        image_patches_res_shape,
        lambda *indices: _din_img2col_compute(indices, image_patches_res, padding, stride_d, float16_align_flag),
        name='image_patches_res',
        tag='image_patches_res')


# 'pylint: disable=too-many-arguments
def _img2col(input_img, col_shape, filter_h, filter_w, pad, stride):
    """
    calculate im2col tensor
    Parameters
    ----------
    input_img : feature map

    col_shape : shape of im2col tensor

    filter_h: the filter value in h

    filter_w: the filter value in w

    pad: the pad shape, list

    stride: the stride value, list

    -------
    Returns : im2col tensor
    """

    # 'pylint: disable=too-many-locals
    def _img2col_compute(input_img, indices, filter_w, pad, stride):
        """
        calculate im2col tensor
        Parameters
        ----------
        input_img: feature map

        indices: indices in lambda function

        filter_w: the filter value in w

        pad: the pad shape, list

        stride: the stride value, list

        -------
        Returns:  im2col tvm lambda function
        """
        _, _, _, fmap_h, fmap_w, _ = input_img.shape
        col_n, col_d, col_howo, col_c1, col_hw, col_ww, col_c0 = indices
        stride_h, stride_w = stride
        pad_top, _, pad_left, pad_right = pad

        output_w = (fmap_w.value + pad_left + pad_right - filter_w) // stride_w + 1

        img_n_index = col_n
        img_d_index = col_d
        img_c1_index = col_c1
        img_h_index = (col_howo // output_w) * stride_h + col_hw
        img_w_index = (col_howo % output_w) * stride_w + col_ww
        img_c0_index = col_c0

        return tvm.select(
            tvm.any(img_h_index < pad_top, img_h_index > fmap_h.value + pad_top - 1, img_w_index < pad_left,
                    img_w_index > fmap_w.value + pad_left - 1), tvm.const(0, input_img.dtype),
            input_img(img_n_index, img_d_index, img_c1_index, img_h_index - pad_top, img_w_index - pad_left,
                      img_c0_index))

    return tvm.compute(col_shape,
                       lambda *indices: _img2col_compute(input_img, indices, filter_w, pad, stride),
                       name='im2col_row_major',
                       tag='im2col_fractal',
                       attrs={
                           'kernel_h': filter_h,
                           'kernel_w': filter_w,
                           'padding': pad,
                           'stride': stride
                       })


# 'pylint: disable=too-many-arguments,invalid-name
def _im2col_fractal(a_im2col_shape, fmap):
    """
    calculate im2col_fractal tensor

    Parameters
    ----------
    a_im2col_shape: shape of a_im2col

    fmap: feature map
    -------
    Returns : a_im2col_fractal tensor
    """

    # 'pylint: disable=too-many-locals
    def _im2col_fractal_indices(indices, fmap):
        """
        calculate im2col_fractal tvm lambda function
        Parameters
        ----------
        indices: indices in lambda function

        fmap: feature map
        -------
        Returns : im2col_fractal tvm lambda function
        """
        _, _, hw, _, kernel_h, kernel_w, _ = fmap.shape
        batch_size, d, i1, j1, j0, i0 = indices

        if fmap.dtype in ("int8", "uint8"):
            block_size_align = Constant.BLOCK_SIZE_INT8
        else:
            block_size_align = Constant.BLOCK_SIZE_FP16  # 16

        n_index = batch_size
        d_index = d
        hw_index = i1 * Constant.BLOCK_SIZE + i0
        c1_index = (((j1 * block_size_align + j0) // block_size_align) // kernel_w.value) // kernel_h.value
        kh_index = (((j1 * block_size_align + j0) // block_size_align) // kernel_w.value) % kernel_h.value
        kw_index = ((j1 * block_size_align + j0) // block_size_align) % kernel_w.value
        c0_index = (j1 * block_size_align + j0) % block_size_align

        return tvm.select(tvm.any(hw_index < 0, hw_index > hw.value - 1), tvm.const(0, fmap.dtype),
                          fmap(n_index, d_index, hw_index, c1_index, kh_index, kw_index, c0_index))

    return tvm.compute(a_im2col_shape,
                       lambda *indices: _im2col_fractal_indices(indices, fmap),
                       name='im2col_fractal',
                       tag='im2col_fractal')


# 'pylint: disable=too-many-arguments,invalid-name,too-many-locals,too-many-statements,too-many-return-statements
def _get_load3d_tiling(fmap_shape, ksize, strides, padding,
                       max_l1_valid_size, max_next_valid_size, dtype, aligned_flag):
    """
    get load3d tiling in davinci.
    ----------
        fmap_shape:
             The shape before load3d, should be (n, di, c1, hi, wi, c0).
        ksize:
             kernel sizes of load3d, should be (kernel_d, kernel_h, kernel_w).
        strides:
             strides of load3d, should be (stride_d, stride_h, stride_w).
        padding:
             "SAME" or "VALID"
        max_l1_valid_size:
            The max buffer size which can used before load3d.
        max_next_valid_size:
            The max buffer size which can used after load3d.
        dtype:
            "float16" or others.
    Returns
    -------
        is_tiling_valid:
            True or False.
        shape_in_l1:
            (n, di, c1, hi, wi, c0).
        is_l1_double_buffer:
            True or False or None.
        shape_after_load3d:
            (n, do, howo, c1, kd, khkw, c0), howo is a multiple of c0.
        is_l0_ub_double_buffer:
            True or False or None
    """
    data_size = tbe_platform.get_bit_len(dtype.lower()) // 8  # 8bits = 1bytes
    block_size_align = 16 if dtype == "float16" else 32
    max_l1_valid_num = max_l1_valid_size // data_size
    max_next_valid_num = max_next_valid_size // data_size

    fmap_n, fmap_d, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap_shape
    fmap_n, fmap_d, fmap_c1, fmap_h, fmap_w, fmap_c0 = \
        fmap_n.value, fmap_d.value, fmap_c1.value, fmap_h.value, fmap_w.value, fmap_c0.value
    kernel_d, kernel_h, kernel_w = ksize
    stride_d, stride_h, stride_w = strides
    output_d, _, _ = tf_get_windowed_output_size_verbose(fmap_d, kernel_d, stride_d,
                                                                               padding.upper())
    output_h, _, _ = tf_get_windowed_output_size_verbose(fmap_h, kernel_h, stride_h,
                                                                               padding.upper())
    output_w, _, _ = tf_get_windowed_output_size_verbose(fmap_w, kernel_w, stride_w,
                                                                               padding.upper())

    l1_n = 1  # init param
    l1_di = fmap_d  # init param
    l1_c1 = 1  # init param
    l1_hi = fmap_h
    l1_wi = fmap_w
    l1_c0 = fmap_c0
    max_dihiwi_l1 = max_l1_valid_num // fmap_c0
    max_dihi_l1 = max_dihiwi_l1 // fmap_w
    max_ho_l1 = (max_dihi_l1 // 1 - (kernel_h - stride_h)) // stride_h
    max_do_l1 = (max_dihi_l1 // 1 - (kernel_d - stride_d)) // stride_d

    # The memory space of l1 is not enough.
    if max_dihiwi_l1 < 1:
        return False, None, None, None, None, None
    if max_ho_l1 < 1 or max_do_l1 < 1:
        # not supported tiling wi in l1 now! must repeat in vertical.
        return False, None, None, None, None, None

    # see if we can do double buffer in l1
    l1_double_buffer = False
    min_com_multi = output_w * block_size_align // math.gcd(output_w, block_size_align)
    if max_ho_l1 >= min_com_multi // output_w * Constant.DOUBLE_BUFFER and \
            max_do_l1 >= min_com_multi // output_w * Constant.DOUBLE_BUFFER:
        max_ho_l1 = max_ho_l1 // Constant.DOUBLE_BUFFER
        max_do_l1 = max_do_l1 // Constant.DOUBLE_BUFFER
        max_dihiwi_l1 = max_dihiwi_l1 // Constant.DOUBLE_BUFFER
        max_dihi_l1 = max_dihiwi_l1 // Constant.DOUBLE_BUFFER
        l1_double_buffer = True

    # l1 memory is enough to put the whole feature map.
    if max_ho_l1 >= output_h:
        max_ho_l1 = output_h
        l1_hi = fmap_h

    else:  # not enough to put the whole feature map
        wo_gcd_c0 = math.gcd(output_w, block_size_align)
        ho_gcd_c0 = block_size_align // wo_gcd_c0
        if max_ho_l1 < ho_gcd_c0:
            return False, None, None, None, None, None
        max_ho_l1 = max_ho_l1 // ho_gcd_c0 * ho_gcd_c0
        l1_hi = max_ho_l1 * stride_h + kernel_h - stride_h
    l1_di = min(fmap_d, max_dihi_l1 // l1_hi) if max_dihi_l1 // l1_hi >= 1 else 1
    max_do_l1 = ((max_dihi_l1 // l1_hi) - kernel_d + 1) // stride_d
    if max_do_l1 < 1:
        max_do_l1 = 1

    howo_pad = _ceil_to(output_h * output_w, block_size_align)
    howo_block = howo_pad // block_size_align
    l0ub_n = 1
    l0ub_do = output_d
    l0ub_c1 = 1
    # The value of l0ub_howo must be multiplied by c0 later.
    l0ub_howo = howo_block
    l0ub_kd = kernel_d
    l0ub_khkw = kernel_h * kernel_w
    l0ub_c0 = fmap_c0
    l0_double_buffer = False

    max_dohowokdkhkw_l0ub = max_next_valid_num // fmap_c0 // fmap_c0
    # The memory space of l0/ub is not enough.
    if max_dohowokdkhkw_l0ub < 1:
        return False, None, None, None, None, None
    # see if we can do double buffer in l0/ub.
    if max_dohowokdkhkw_l0ub >= Constant.DOUBLE_BUFFER:
        max_dohowokdkhkw_l0ub = max_dohowokdkhkw_l0ub // Constant.DOUBLE_BUFFER
        l0_double_buffer = True

    # l0/ub memory is enough to put the whole col.
    if max_dohowokdkhkw_l0ub >= output_d * howo_block * kernel_d * kernel_h * kernel_w:
        pass
    # not enough to put whole kernel
    elif max_dohowokdkhkw_l0ub < kernel_h * kernel_w:
        l0ub_do = 1
        l0ub_howo = 1
        l0ub_khkw = max_dohowokdkhkw_l0ub
        l0ub_kd = 1
    # enough to put a whole khkw, but not enough for kd * khkw
    elif max_dohowokdkhkw_l0ub < kernel_d * kernel_h * kernel_w:
        l0ub_do = 1
        l0ub_howo = 1
        l0ub_khkw = kernel_h * kernel_w
        l0ub_kd = max_dohowokdkhkw_l0ub // (kernel_h * kernel_w)
        if l0ub_kd == 0:
            l0ub_kd = 1
    # enough to put a whole kernel, but not enough for howo
    elif max_dohowokdkhkw_l0ub < howo_block * kernel_d * kernel_h * kernel_w:
        l0ub_do = 1
        l0ub_howo = max_dohowokdkhkw_l0ub // (kernel_d * kernel_h * kernel_w)
        if l0ub_howo == 0:
            l0ub_howo = 1
    # enough to put a whole kernel and howo, but not enough for dohowo
    else:
        l0ub_do = max_dohowokdkhkw_l0ub // (howo_block * kernel_d * kernel_h * kernel_w)
        if l0ub_do == 0:
            l0ub_do = 1
    l0ub_howo *= fmap_c0  # multiplied by c0
    # get min howo in l1 and l0/ub
    l0ub_howo = min(l0ub_howo, max(max_ho_l1 * output_w, block_size_align))
    l0ub_do = min(l0ub_do, max_do_l1)

    howo_split = 1
    l0ub_howo_tmp = l0ub_howo
    tile_l1_wi = fmap_w

    def get_tile_l1_hi(dtype, l0ub_howo, output_w, stride_h, kernel_h, fmap_h, howo_split, aligned_flag):
        """
        Return tile_l1_hi
        """
        l0ub_howo = 2 * l0ub_howo if dtype in ("uint8", "int8") else l0ub_howo
        tile_l1_hi_1 = (l0ub_howo + output_w - 1) // output_w * stride_h + kernel_h
        if not howo_split or aligned_flag or dtype == "float16":
            return min(tile_l1_hi_1, fmap_h)
        #  infer bound calculation info
        min_align = output_w
        extent_align = output_w
        tensor5_num = fmap_c0 * fmap_c0
        tensor42_num = tensor5_num
        tensor7_num = tensor42_num
        if (Constant.MAX_UB_SIZE // 4) // (tensor42_num + tensor5_num + tensor7_num):
            res_double_buffer_flag = 1
        else:
            res_double_buffer_flag = 0

        if res_double_buffer_flag:
            ub_num = Constant.MAX_UB_SIZE // (Constant.DOUBLE_BUFFER * data_size)
        else:
            ub_num = Constant.MAX_UB_SIZE // data_size

        max_mhowo_factor = ub_num // (tensor42_num + tensor5_num + tensor7_num)
        mhowo = (output_h * output_w + fmap_c0 - 1) // fmap_c0
        factor = min(max_mhowo_factor, mhowo)

        howo = output_w * output_h
        howo_outer = (howo // fmap_c0 + factor - 1) // factor * factor
        min_range = (howo_outer * 2) * 16
        # 48 is `Constant.BLOCK_SIZE + BLOCK_SIZE_ALIGN`
        extent = 48
        max_range = min_range + extent - 1
        aligned_min = (min_range // min_align * min_align)
        new_extent = max_range - aligned_min + 1
        aligned_extent = ((new_extent + extent_align - 1) // extent_align + 1) * extent_align
        tile_l1_hi_2 = (((((aligned_extent + (((howo_outer * fmap_c0) // output_w) * output_w)) - 1) // output_w) *
                         stride_h) + kernel_h - 1) - \
                        (((((howo_outer * fmap_c0) // output_w) * output_w) // output_w) * stride_h) + 1
        return min(max(tile_l1_hi_1, tile_l1_hi_2), fmap_h)

    tile_l1_hi = get_tile_l1_hi(dtype, l0ub_howo, output_w, stride_h, kernel_h, fmap_h, howo_split, aligned_flag)

    tile_l1_di = fmap_d
    tile_l1_c0 = l0ub_c0
    if tile_l1_wi * tile_l1_hi * tile_l1_di * tile_l1_c0 * Constant.DOUBLE_BUFFER > max_l1_valid_num:
        l0ub_howo = block_size_align
        tile_l1_hi = get_tile_l1_hi(dtype, l0ub_howo, output_w, stride_h, kernel_h, fmap_h, howo_split, aligned_flag)

    if tile_l1_wi * tile_l1_hi * tile_l1_di * tile_l1_c0 * Constant.DOUBLE_BUFFER > max_l1_valid_num:
        howo_split = 0
    else:
        return True, (l1_n, l1_di, l1_c1, l1_hi, l1_wi,
                      l1_c0), l1_double_buffer, (l0ub_n, l0ub_do, l0ub_howo, l0ub_c1, l0ub_kd, l0ub_khkw,
                                                 l0ub_c0), l0_double_buffer, howo_split

    l0ub_howo = l0ub_howo_tmp
    tile_l1_wi = fmap_w
    tile_l1_hi = get_tile_l1_hi(dtype, l0ub_howo, output_w, stride_h, kernel_h, fmap_h, howo_split, aligned_flag)

    tile_l1_di = max(l0ub_kd, l0ub_do)
    tile_l1_c0 = l0ub_c0
    if tile_l1_wi * tile_l1_hi * tile_l1_di * tile_l1_c0 * Constant.DOUBLE_BUFFER > max_l1_valid_num:
        l0ub_kd = min(max_l1_valid_num // (Constant.DOUBLE_BUFFER * tile_l1_wi * tile_l1_hi * tile_l1_c0), l0ub_kd)
        if l0ub_kd == 0:
            l0ub_kd = 1
        l0ub_do = 1
        tile_l1_di = min(l0ub_kd, fmap_d)
        if tile_l1_wi * tile_l1_hi * tile_l1_di * tile_l1_c0 * Constant.DOUBLE_BUFFER > max_l1_valid_num:
            l0ub_howo = block_size_align
            tile_l1_hi = get_tile_l1_hi(dtype, l0ub_howo, output_w, stride_h, kernel_h,
                                        fmap_h, howo_split, aligned_flag)

    if tile_l1_wi * tile_l1_hi * tile_l1_di * tile_l1_c0 * Constant.DOUBLE_BUFFER > max_l1_valid_num:
        error_manager_vector.raise_err_specific_reson("extract_volume_patches", "L1 Size is not enough")

    return True, (l1_n, l1_di, l1_c1, l1_hi, l1_wi,
                  l1_c0), l1_double_buffer, (l0ub_n, l0ub_do, l0ub_howo, l0ub_c1, l0ub_kd, l0ub_khkw,
                                             l0ub_c0), l0_double_buffer, howo_split


# 'pylint: disable=unnecessary-lambda,too-many-locals,too-many-return-statements
def _extract_volume_patches_compute_6hd(data_input, fmap_c, ksizes, strides, padding):
    """
    calculating data

    Parameters
    ----------
    data_input : TVM tensor
        the placeholder of input_x
    ksizes: input attr
    strides: input attr
    padding: input attr

    Returns
    -------
    out_res: output tensor
    workspace_res: workspace result
    """
    # fmap's format is NDC1HWC0
    fmap_shape = data_input.shape
    original_cin = fmap_c
    fmap_in_l1 = tvm.compute(fmap_shape, lambda *i: data_input(*i), name="fmap_in_l1")

    _, filter_d, filter_h, filter_w, _ = ksizes
    _, stride_d, stride_h, stride_w, _ = strides
    fmap_batch, fmap_d, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap_shape
    out_h, padding_h_before, padding_h_after = tf_get_windowed_output_size_verbose(
        fmap_h.value, filter_h, stride_h, padding)
    out_w, padding_w_before, padding_w_after = tf_get_windowed_output_size_verbose(
        fmap_w.value, filter_w, stride_w, padding)
    pad = (padding_h_before, padding_h_after, padding_w_before, padding_w_after)
    stride = (stride_h, stride_w)
    # set_fmatrix, VM shape
    fmap_vm_shape = (fmap_batch, fmap_d, out_h * out_w, fmap_c1, filter_h, filter_w, fmap_c0)
    # fmap_in_l1: [N, D, C1 ,H, W, C0]
    fmap_im2col = _img2col(fmap_in_l1, fmap_vm_shape, filter_h, filter_w, pad, stride)
    howo = ((out_h * out_w + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE) * Constant.BLOCK_SIZE

    # load 3D, L1 to UB
    if fmap_in_l1.dtype in ("int8", "uint8"):
        blcok_size_align = Constant.BLOCK_SIZE_INT8  # 32
    else:
        blcok_size_align = Constant.BLOCK_SIZE_FP16  # 16

    fractal_shape = (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, fmap_c1 * filter_h * filter_w,
                     Constant.BLOCK_SIZE, blcok_size_align)
    fmap_fractal = _im2col_fractal(fractal_shape, fmap_im2col)

    # 2nd filter and stride
    filter_h_2nd = filter_d
    stride_h_2nd = stride_d
    filter_w_2nd = 1
    stride_w_2nd = 1
    fmap_h_2nd, fmap_w_2nd = fmap_d, howo
    out_h_2nd, padding_h_before_2nd, padding_h_after_2nd = tf_get_windowed_output_size_verbose(
        fmap_h_2nd.value, filter_h_2nd, stride_h_2nd, padding)
    dout = out_h_2nd
    _, padding_w_before_2nd, padding_w_after_2nd = tf_get_windowed_output_size_verbose(
        fmap_w_2nd, filter_w_2nd, stride_w_2nd, padding)
    pad_2nd = (padding_h_before_2nd, padding_h_after_2nd, padding_w_before_2nd, padding_w_after_2nd)

    if original_cin % blcok_size_align != 0 and data_input.dtype == "float16":
        # fmap_fractal_transpose
        # transpose Constant.BLOCK_SIZE
        # [UB]fmap_fractal:
        #  `(fmap_batch, fmap_d, howo // BLOCK_SIZE, fmap_c1 * filter_h * filter_w, BLOCK_SIZE, blcok_size_align)` ->
        # [UB]image_patches:
        #  `(fmap_batch, fmap_d, howo // BLOCK_SIZE, fmap_c1 * filter_h * filter_w, BLOCK_SIZE, blcok_size_align)`
        image_patches_shape = (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, fmap_c1 * filter_h * filter_w,
                               Constant.BLOCK_SIZE, blcok_size_align)
        image_patches = tvm.compute(image_patches_shape,
                                    lambda n, d, howo1, c1khkw, howo0, c0: fmap_fractal[n, d, howo1, c1khkw, howo0, c0],
                                    name="image_patches")

        # [UB]image_patches:
        #  `(fmap_batch, fmap_d, howo // BLOCK_SIZE, fmap_c1 * filter_h * filter_w, BLOCK_SIZE, blcok_size_align)` ->
        # [UB]image_patches_split_c1:
        #  `(fmap_batch, fmap_d, howo // BLOCK_SIZE, fmap_c1, filter_h * filter_w, BLOCK_SIZE, blcok_size_align)`
        image_patches_split_c1_shape = (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, fmap_c1, filter_h * filter_w,
                                        Constant.BLOCK_SIZE, blcok_size_align)
        image_patches_split_c1 = tvm.compute(image_patches_split_c1_shape,
                                             lambda n, d, howo1, c1, khkw, howo0, c0: image_patches[
                                                 n, d, howo1, c1 * filter_h * filter_w + khkw, howo0, c0],
                                             name="image_patches_split_c1")

        # expand d axis
        # [UB]image_patches_split_c1:
        #  `(fmap_batch, fmap_d, howo // BLOCK_SIZE, fmap_c1 * filter_h * filter_w, BLOCK_SIZE, blcok_size_align)` ->
        # [UB]image_patches_res:
        #  `(fmap_batch, dout, howo // BLOCK_SIZE, fmap_c1, filter_d,`
        #  `filter_h * filter_w, BLOCK_SIZE, blcok_size_align)`
        image_patches_res_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, fmap_c1, filter_d,
                                   filter_h * filter_w, Constant.BLOCK_SIZE, blcok_size_align)
        image_patches_res = _din_img2col(image_patches_res_shape,
                                         image_patches_split_c1,
                                         pad_2nd,
                                         stride_d,
                                         float16_align_flag=0)

        # transpose c1 axis
        # [UB]image_patches_res:
        #  (fmap_batch, dout, howo // Constant.BLOCK_SIZE, fmap_c1, filter_d,
        #  filter_h * filter_w, Constant.BLOCK_SIZE, blcok_size_align) ->
        # [UB]image_patches_res_transformat:
        #  (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d, filter_h * filter_w, fmap_c1,
        #   Constant.BLOCK_SIZE, blcok_size_align)
        image_patches_res_transformat_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d,
                                               filter_h * filter_w, fmap_c1, Constant.BLOCK_SIZE, blcok_size_align)
        image_patches_res_transformat = tvm.compute(
            image_patches_res_transformat_shape,
            lambda n, dO, howo_co, kd, khkw, c1, co, c0: image_patches_res[n, dO, howo_co, c1, kd, khkw, co, c0
                                                                           ] + tvm.const(0, data_input.dtype),
            name="image_patches_res_transformat")

        # workspace_res
        # dma from ub to workspace
        # [UB]image_patches_res_transformat:
        #  (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d,
        #  filter_h * filter_w, fmap_c1, Constant.BLOCK_SIZE, blcok_size_align) ->
        # [WorkSpace]workspace_res:
        #  (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d, filter_h * filter_w, fmap_c1,
        #   Constant.BLOCK_SIZE, blcok_size_align)
        workspace_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d, filter_h * filter_w,
                           fmap_c1, Constant.BLOCK_SIZE, blcok_size_align)
        workspace_res = tvm.compute(workspace_shape,
                                    lambda n, dO, howo_co, kd, khkw, c1, co, c0: image_patches_res_transformat[
                                        n, dO, howo_co, kd, khkw, c1, co, c0],
                                    name="workspace_res")

        tensor0_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d, filter_h * filter_w, fmap_c1,
                         Constant.BLOCK_SIZE, blcok_size_align)
        tensor0 = tvm.compute(tensor0_shape, lambda *i: workspace_res(*i), name="tensor0")

        tensor1_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d, filter_h * filter_w, fmap_c1,
                         Constant.BLOCK_SIZE, blcok_size_align)
        tensor1 = tvm.compute(tensor1_shape, lambda *i: tensor0(*i), name="tensor1")

        tensor2_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d, filter_h * filter_w, fmap_c1,
                         blcok_size_align, Constant.BLOCK_SIZE)
        tensor2 = tvm.compute(tensor2_shape,
                              lambda i1, i2, i3, i4, i5, i6, i7, i8: tensor1[i1, i2, i3, i4, i5, i6, i8, i7] + tvm.
                              const(0, dtype=data_input.dtype),
                              name="tensor2")

        tensor3_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d, filter_h * filter_w, fmap_c,
                         Constant.BLOCK_SIZE)
        tensor3 = tvm.compute(tensor3_shape,
                              lambda n, dO, howo_floor, kd, khkw, c, floor: tensor2[
                                  n, dO, howo_floor, kd, khkw, c // blcok_size_align, c % blcok_size_align, floor] + \
                                      tvm.const(0, dtype=data_input.dtype), name="tensor3")

        tensor4_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d, filter_h * filter_w * fmap_c,
                         Constant.BLOCK_SIZE)
        tensor4 = tvm.compute(tensor4_shape,
                              lambda n, dO, howo_floor, kd, khkwc, floor: tensor3[n, dO, howo_floor, kd, khkwc //
                                                                                  fmap_c, khkwc % fmap_c, floor],
                              name="tensor4")

        tensor41_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d * filter_h * filter_w * fmap_c,
                          Constant.BLOCK_SIZE)
        tensor41 = tvm.compute(
            tensor41_shape,
            lambda n, dO, howo_floor, kdkhkwc, floor: tensor4[n, dO, howo_floor, kdkhkwc //
                                                              (filter_h * filter_w * fmap_c), kdkhkwc %
                                                              (filter_h * filter_w * fmap_c), floor],
            name="tensor41")

        workspace2_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d * filter_h * filter_w * fmap_c,
                            Constant.BLOCK_SIZE)
        workspace2 = tvm.compute(workspace2_shape,
                                 lambda n, dO, howo_floor, kdkhkwc, floor: tensor41[n, dO, howo_floor, kdkhkwc, floor],
                                 name="workspace2")

        tensor42_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, filter_d * filter_h * filter_w * fmap_c,
                          Constant.BLOCK_SIZE)
        tensor42 = tvm.compute(tensor42_shape,
                               lambda n, dO, howo_floor, kdkhkwc, floor: workspace2[n, dO, howo_floor, kdkhkwc, floor],
                               name="tensor42")

        tensor5_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE,
                         (filter_d * filter_h * filter_w * fmap_c + blcok_size_align - 1) // blcok_size_align,
                         blcok_size_align, Constant.BLOCK_SIZE)
        tensor5 = tvm.compute(tensor5_shape,
                              lambda n, dO, howo_floor, kdkhkwc_floor2, floor2, floor: tvm.select(
                                  tvm.any(
                                      kdkhkwc_floor2 * blcok_size_align + floor2 > \
                                      filter_d * filter_h * filter_w * fmap_c - 1
                                  ),
                                  tvm.const(0, "float16"),
                                  tensor42[n, dO, howo_floor, kdkhkwc_floor2 * blcok_size_align + floor2, floor] + \
                                  tvm.const(0, dtype="float16")
                              ),
                              name="tensor5")

        tensor6_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE,
                         (filter_d * filter_h * filter_w * fmap_c + blcok_size_align - 1) // blcok_size_align,
                         Constant.BLOCK_SIZE, blcok_size_align)
        tensor6 = tvm.compute(tensor6_shape,
                              lambda n, dO, howo_floor, kdkhkwc_floor2, floor, floor2: tensor5[
                                  n, dO, howo_floor, kdkhkwc_floor2, floor2, floor],
                              name="tensor6")

        tensor7_shape = (fmap_batch, dout,
                         (filter_d * filter_h * filter_w * fmap_c + blcok_size_align - 1) // blcok_size_align,
                         howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE, blcok_size_align)
        tensor7 = tvm.compute(tensor7_shape,
                              lambda n, dO, kdkhkwc_floor2, howo_floor, floor, floor2: tensor6[
                                  n, dO, howo_floor, kdkhkwc_floor2, floor, floor2],
                              name="tensor7")

        extract_params = {}
        extract_params["padding_mode"] = padding
        extract_params["original_cin"] = original_cin
        extract_params["out_d"] = out_h_2nd
        extract_params["out_h"] = out_h
        extract_params["out_w"] = out_w
        extract_params["fmap_shape"] = fmap_shape
        extract_params["ksizes"] = (filter_d, filter_h, filter_w)
        extract_params["strides"] = (stride_d, stride_h, stride_w)
        extract_params["pad"] = pad
        extract_params["fmap_vm_shape"] = fmap_vm_shape
        extract_params["fractal_shape"] = fractal_shape
        extract_params["howo"] = howo

        setfmatrix_dict = {
            "conv_kernel_h": filter_h,
            "conv_kernel_w": filter_w,
            "conv_padding_top": padding_h_before,
            "conv_padding_left": padding_w_before,
            "conv_padding_right": padding_w_after,
            "conv_stride_h": stride_h,
            "conv_stride_w": stride_w,
            "conv_fm_c": fmap_c1 * fmap_c0,
            "conv_fm_h": fmap_h,
            "conv_fm_w": fmap_w
        }

        tensor8_shape = (fmap_batch, dout,
                         (filter_d * filter_h * filter_w * fmap_c + blcok_size_align - 1) // blcok_size_align,
                         out_h * out_w, blcok_size_align)
        tensor8 = tvm.compute(tensor8_shape,
                              lambda n, dO, kdkhkwc_floor2, j, floor2: tensor7[
                                  n, dO, kdkhkwc_floor2, j // blcok_size_align, j % blcok_size_align, floor2],
                              name="out_res",
                              attrs={
                                  'extract_params': extract_params,
                                  'setfmatrix_dict': setfmatrix_dict
                              })
        out_res = tensor8
        return out_res, workspace_res, workspace2

    if original_cin % blcok_size_align != 0 and data_input.dtype in ("uint8", "int8"):
        # fmap_fractal_transpose
        # transpose Constant.BLOCK_SIZE
        # [UB]fmap_fractal:
        #  (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, fmap_c1 * filter_h * filter_w,
        #  Constant.BLOCK_SIZE, blcok_size_align) ->
        # [UB]image_patches:
        #  (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE,
        #  fmap_c1 * filter_h * filter_w, blcok_size_align)
        image_patches_shape = (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE,
                               fmap_c1 * filter_h * filter_w, blcok_size_align)
        image_patches = tvm.compute(image_patches_shape,
                                    lambda n, d, howo1, howo0, c1khkw, c0: fmap_fractal[n, d, howo1, c1khkw, howo0, c0],
                                    name="image_patches")
        # image_patches_split_c1
        # split c1 & khkw
        # [UB]image_patches:
        #  (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE,
        #  fmap_c1 * filter_h * filter_w, blcok_size_align) ->
        # [UB]image_patches_split_c1:
        #  (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE,
        #  fmap_c1, filter_h * filter_w, blcok_size_align)
        image_patches_split_c1_shape = (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE, fmap_c1,
                                        filter_h * filter_w, blcok_size_align)
        image_patches_split_c1 = tvm.compute(image_patches_split_c1_shape,
                                             lambda n, d, howo1, howo0, c1, khkw, c0: image_patches[
                                                 n, d, howo1, howo0, c1 * filter_h * filter_w + khkw, c0],
                                             name="image_patches_split_c1")
        # image_patches_res
        # expand d axis
        # [UB]image_patches_split_c1:
        #  (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE, fmap_c1, filter_h * filter_w,
        #  Constant.BLOCK_SIZE) ->
        # [UB]image_patches_res:
        #  (fmap_batch, dout, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE, filter_d, fmap_c1,
        #  filter_h * filter_w, blcok_size_align)
        image_patches_res_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE, filter_d,
                                   fmap_c1, filter_h * filter_w, blcok_size_align)
        image_patches_res = _din_img2col(image_patches_res_shape,
                                         image_patches_split_c1,
                                         pad_2nd,
                                         stride_d,
                                         float16_align_flag=0)

        # workspace_res
        # dma from ub to workspace and transpose
        # [UB]image_patches_res:
        #  (fmap_batch, dout, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE, filter_d,
        #  fmap_c1, filter_h * filter_w, blcok_size_align) ->
        # [WorkSpace]workspace_res:
        #  `(fmap_batch, dout, out_h * out_w, filter_d, filter_h * filter_w, fmap_c1, blcok_size_align)`
        workspace_shape = (fmap_batch, dout, out_h * out_w, filter_d, filter_h * filter_w, fmap_c1, blcok_size_align)
        workspace_res = tvm.compute(workspace_shape,
                                    lambda n, dO, howo, kd, khkw, c1, c0: image_patches_res[
                                        n, dO, howo // Constant.BLOCK_SIZE,
                                        howo % Constant.BLOCK_SIZE, kd, c1, khkw, c0],
                                    name="workspace_res")

        tensor0_shape = (fmap_batch, dout, out_h * out_w, filter_d, filter_h * filter_w, fmap_c1, blcok_size_align)
        tensor0 = tvm.compute(tensor0_shape, lambda *i: workspace_res(*i), name="tensor0")

        howo_8bit_align = (out_h * out_w + blcok_size_align - 1) // blcok_size_align * blcok_size_align
        tensor00_shape = (fmap_batch, dout, howo_8bit_align // blcok_size_align, blcok_size_align, filter_d,
                          filter_h * filter_w, fmap_c1, blcok_size_align)
        tensor00 = tvm.compute(
            tensor00_shape,
            lambda n, dO, howo_floor, floor, kd, khkw, c1, c0: tvm.select(
                tvm.any(howo_floor * blcok_size_align + floor > out_h * out_w - 1), tvm.const(0, data_input.dtype),
                tensor0[n, dO, howo_floor * blcok_size_align + floor, kd, khkw, c1, c0]),
            name="tensor00")

        workspace_8bit_shape = (fmap_batch, dout, howo_8bit_align // blcok_size_align, blcok_size_align, filter_d,
                                filter_h * filter_w, fmap_c1, blcok_size_align)
        workspace_8bit = tvm.compute(workspace_8bit_shape, lambda *i: tensor00(*i), name="workspace_8bit")

        tensor10_shape = (fmap_batch, dout, howo_8bit_align // blcok_size_align, blcok_size_align, filter_d,
                          filter_h * filter_w, fmap_c1, blcok_size_align)
        tensor10 = tvm.compute(tensor10_shape, lambda *i: workspace_8bit(*i), name="tensor10")

        tensor1_shape = (fmap_batch, dout, howo_8bit_align // blcok_size_align, filter_d, filter_h * filter_w, fmap_c1,
                         blcok_size_align, blcok_size_align)
        tensor1 = tvm.compute(
            tensor1_shape,
            lambda n, dO, howo_floor, kd, khkw, c1, floor, c0: tensor10[n, dO, howo_floor, floor, kd, khkw, c1, c0],
            name="tensor1")

        tensor2_shape = (fmap_batch, dout, howo_8bit_align // blcok_size_align, filter_d, filter_h * filter_w, fmap_c1,
                         blcok_size_align, blcok_size_align)
        tensor2 = tvm.compute(tensor2_shape,
                              lambda i1, i2, i3, i4, i5, i6, i7, i8: tensor1[i1, i2, i3, i4, i5, i6, i8, i7],
                              name="tensor2")

        tensor3_shape = (fmap_batch, dout, howo_8bit_align // blcok_size_align, filter_d, filter_h * filter_w, fmap_c,
                         blcok_size_align)
        tensor3 = tvm.compute(tensor3_shape,
                              lambda n, dO, howo_floor, kd, khkw, c, floor: tensor2[
                                  n, dO, howo_floor, kd, khkw, c // blcok_size_align, c % blcok_size_align, floor],
                              name="tensor3")

        tensor4_shape = (fmap_batch, dout, howo_8bit_align // blcok_size_align, filter_d, filter_h * filter_w * fmap_c,
                         blcok_size_align)
        tensor4 = tvm.compute(tensor4_shape,
                              lambda n, dO, howo_floor, kd, khkwc, floor: tensor3[n, dO, howo_floor, kd, khkwc //
                                                                                  fmap_c, khkwc % fmap_c, floor],
                              name="tensor4")

        tensor41_shape = (fmap_batch, dout, howo_8bit_align // blcok_size_align,
                          filter_d * filter_h * filter_w * fmap_c, blcok_size_align)
        tensor41 = tvm.compute(
            tensor41_shape,
            lambda n, dO, howo_floor, kdkhkwc, floor: tensor4[n, dO, howo_floor, kdkhkwc //
                                                              (filter_h * filter_w * fmap_c), kdkhkwc %
                                                              (filter_h * filter_w * fmap_c), floor],
            name="tensor41")

        workspace2_shape = (fmap_batch, dout, howo_8bit_align // blcok_size_align,
                            filter_d * filter_h * filter_w * fmap_c, blcok_size_align)
        workspace2 = tvm.compute(workspace2_shape,
                                 lambda n, dO, howo_floor, kdkhkwc, floor: tensor41[n, dO, howo_floor, kdkhkwc, floor],
                                 name="workspace2")

        tensor42_shape = (fmap_batch, dout, howo_8bit_align // blcok_size_align,
                          filter_d * filter_h * filter_w * fmap_c, blcok_size_align)
        tensor42 = tvm.compute(tensor42_shape,
                               lambda n, dO, howo_floor, kdkhkwc, floor: workspace2[n, dO, howo_floor, kdkhkwc, floor],
                               name="tensor42")

        tensor5_shape = (fmap_batch, dout, howo_8bit_align // blcok_size_align,
                         (filter_d * filter_h * filter_w * fmap_c + blcok_size_align - 1) // blcok_size_align,
                         blcok_size_align, blcok_size_align)
        tensor5 = tvm.compute(tensor5_shape,
                              lambda n, dO, howo_floor, kdkhkwc_floor2, floor2, floor: tvm.select(
                                  tvm.any(kdkhkwc_floor2 * blcok_size_align + floor2 > \
                                          filter_d * filter_h * filter_w * fmap_c - 1),
                                  tvm.const(0, data_input.dtype),
                                  tensor42[n, dO, howo_floor, kdkhkwc_floor2 * blcok_size_align + floor2, floor] + \
                                      tvm.const(0, data_input.dtype)
                              ),
                              name="tensor5")

        tensor6_shape = (fmap_batch, dout, howo_8bit_align // blcok_size_align,
                         (filter_d * filter_h * filter_w * fmap_c + blcok_size_align - 1) // blcok_size_align,
                         blcok_size_align, blcok_size_align)
        tensor6 = tvm.compute(tensor6_shape,
                              lambda n, dO, howo_floor, kdkhkwc_floor2, floor, floor2: tensor5[
                                  n, dO, howo_floor, kdkhkwc_floor2, floor2, floor],
                              name="tensor6")

        tensor7_shape = (fmap_batch, dout,
                         (filter_d * filter_h * filter_w * fmap_c + blcok_size_align - 1) // blcok_size_align,
                         howo_8bit_align // blcok_size_align, blcok_size_align, blcok_size_align)
        tensor7 = tvm.compute(tensor7_shape,
                              lambda n, dO, kdkhkwc_floor2, howo_floor, floor, floor2: tensor6[
                                  n, dO, howo_floor, kdkhkwc_floor2, floor, floor2],
                              name="tensor7")

        extract_params = {}
        extract_params["padding_mode"] = padding
        extract_params["original_cin"] = original_cin
        extract_params["out_d"] = out_h_2nd
        extract_params["out_h"] = out_h
        extract_params["out_w"] = out_w
        extract_params["fmap_shape"] = fmap_shape
        extract_params["ksizes"] = (filter_d, filter_h, filter_w)
        extract_params["strides"] = (stride_d, stride_h, stride_w)
        extract_params["pad"] = pad
        extract_params["fmap_vm_shape"] = fmap_vm_shape
        extract_params["fractal_shape"] = fractal_shape
        extract_params["howo"] = howo
        extract_params["howo_8bit_align"] = howo_8bit_align

        setfmatrix_dict = {
            "conv_kernel_h": filter_h,
            "conv_kernel_w": filter_w,
            "conv_padding_top": padding_h_before,
            "conv_padding_left": padding_w_before,
            "conv_padding_right": padding_w_after,
            "conv_stride_h": stride_h,
            "conv_stride_w": stride_w,
            "conv_fm_c": fmap_c1 * fmap_c0,
            "conv_fm_h": fmap_h,
            "conv_fm_w": fmap_w
        }

        tensor8_shape = (fmap_batch, dout,
                         (filter_d * filter_h * filter_w * fmap_c + blcok_size_align - 1) // blcok_size_align,
                         out_h * out_w, blcok_size_align)
        tensor8 = tvm.compute(tensor8_shape,
                              lambda n, dO, kdkhkwc_floor2, j, floor2: tensor7[
                                  n, dO, kdkhkwc_floor2, j // blcok_size_align, j % blcok_size_align, floor2],
                              name="out_res",
                              attrs={
                                  'extract_params': extract_params,
                                  'setfmatrix_dict': setfmatrix_dict
                              })
        out_res = tensor8
        return out_res, workspace_res, workspace_8bit, workspace2

    if original_cin % blcok_size_align == 0:
        # fmap_fractal_transpose
        # transpose Constant.BLOCK_SIZE
        # [UB]fmap_fractal:
        #  (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, fmap_c1 * filter_h * filter_w,
        #  Constant.BLOCK_SIZE, blcok_size_align) ->
        # [UB]image_patches:
        #  (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE,
        #  fmap_c1 * filter_h * filter_w, blcok_size_align)
        image_patches_shape = (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE,
                               fmap_c1 * filter_h * filter_w, blcok_size_align)
        image_patches = tvm.compute(image_patches_shape,
                                    lambda n, d, howo1, howo0, c1khkw,
                                           c0: fmap_fractal[n, d, howo1, c1khkw, howo0, c0],
                                    name="image_patches")
        # image_patches_split_c1
        # split c1 & khkw
        # [UB]image_patches:
        #  (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE,
        #  fmap_c1 * filter_h * filter_w, blcok_size_align) ->
        # [UB]image_patches_split_c1:
        #  (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE,
        #  fmap_c1, filter_h * filter_w, blcok_size_align)
        image_patches_split_c1_shape = (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE, fmap_c1,
                                        filter_h * filter_w, blcok_size_align)
        image_patches_split_c1 = tvm.compute(image_patches_split_c1_shape,
                                             lambda n, d, howo1, howo0, c1, khkw, c0: image_patches[
                                                 n, d, howo1, howo0, c1 * filter_h * filter_w + khkw, c0],
                                             name="image_patches_split_c1")
        # image_patches_res
        # expand d axis
        # [UB]image_patches_split_c1:
        #  (fmap_batch, fmap_d, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE,
        #  fmap_c1, filter_h * filter_w, Constant.BLOCK_SIZE) ->
        # [UB]image_patches_res:
        #  (fmap_batch, dout, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE, filter_d, fmap_c1,
        #  filter_h * filter_w, blcok_size_align)
        image_patches_res_shape = (fmap_batch, dout, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE,
                                   filter_d, fmap_c1, filter_h * filter_w, blcok_size_align)
        image_patches_res = _din_img2col(image_patches_res_shape,
                                         image_patches_split_c1,
                                         pad_2nd,
                                         stride_d,
                                         float16_align_flag=1)

        # workspace_res
        # dma from ub to workspace and transpose
        # [UB]image_patches_res:
        #  (fmap_batch, dout, howo // Constant.BLOCK_SIZE, Constant.BLOCK_SIZE, filter_d,
        #  fmap_c1, filter_h * filter_w, blcok_size_align) ->
        # [WorkSpace]workspace_res:
        #  `(fmap_batch, dout, out_h*out_w, filter_d, filter_h * filter_w, fmap_c1, blcok_size_align)`
        workspace_shape = (fmap_batch, dout, out_h * out_w, filter_d, filter_h * filter_w, fmap_c1, blcok_size_align)
        workspace_res = tvm.compute(workspace_shape,
                                    lambda n, dO, howo, kd, khkw, c1, c0: image_patches_res[
                                        n, dO, howo // Constant.BLOCK_SIZE,
                                        howo % Constant.BLOCK_SIZE, kd, c1, khkw, c0],
                                    name="workspace_res")

        # ub_res
        # dma from workspace to ub and merge c1c0
        # [WorkSpace]workspace_res:
        #  (fmap_batch, dout, out_h*out_w, filter_d, filter_h * filter_w, fmap_c1, blcok_size_align) ->
        # [UB]ub_res:
        #  `(fmap_batch, dout, out_h*out_w, filter_d, filter_h * filter_w, fmap_c1, blcok_size_align)`
        ub_res_shape = (fmap_batch, dout, out_h * out_w, filter_d, filter_h * filter_w, fmap_c1, blcok_size_align)
        ub_res = tvm.compute(ub_res_shape,
                             lambda n, dO, howo, kd, khkw, c1, c0: workspace_res[n, dO, howo, kd, khkw, c1, c0],
                             name="ub_res")

        extract_params = {}
        extract_params["padding_mode"] = padding
        extract_params["original_cin"] = original_cin
        extract_params["out_d"] = out_h_2nd
        extract_params["out_h"] = out_h
        extract_params["out_w"] = out_w
        extract_params["fmap_shape"] = fmap_shape
        extract_params["ksizes"] = (filter_d, filter_h, filter_w)
        extract_params["strides"] = (stride_d, stride_h, stride_w)
        extract_params["pad"] = pad
        extract_params["fmap_vm_shape"] = fmap_vm_shape
        extract_params["fractal_shape"] = fractal_shape
        extract_params["howo"] = howo
        extract_params["ub_res_shape"] = ub_res_shape

        setfmatrix_dict = {
            "conv_kernel_h": filter_h,
            "conv_kernel_w": filter_w,
            "conv_padding_top": padding_h_before,
            "conv_padding_left": padding_w_before,
            "conv_padding_right": padding_w_after,
            "conv_stride_h": stride_h,
            "conv_stride_w": stride_w,
            "conv_fm_c": fmap_c1 * fmap_c0,
            "conv_fm_h": fmap_h,
            "conv_fm_w": fmap_w
        }

        # trans_format0
        # trans format from ub to ub.
        # [UB]ub_res: (fmap_batch, dout, out_h * out_w, filter_d,
        #  filter_h * filter_w, fmap_c1, Constant.BLOCK_SIZE) ->
        # [UB]trans_format0: (fmap_batch, dout, fmap_c1, out_h * out_w, filter_d,
        #  filter_h * filter_w, fmap_c0)
        trans_format0_shape = (fmap_batch, dout, filter_d, filter_h * filter_w, fmap_c1, out_h * out_w,
                               blcok_size_align)
        trans_format0 = tvm.compute(trans_format0_shape,
                                    lambda n, dO, kd, khkw, c1, howo, c0: ub_res[n, dO, howo, kd, khkw, c1, c0],
                                    name="trans_format0")

        # out_res
        # from ub to gm
        # [UB]trans_format0: (fmap_batch, dout, filter_d,
        #  filter_h * filter_w, fmap_c1, out_h * out_w, blcok_size_align) ->
        # [GM]out_res: (fmap_batch, dout, out_c1, out_h * out_w, blcok_size_align)
        out_res_shape = (fmap_batch, dout, filter_d, filter_h * filter_w, fmap_c1, out_h * out_w, blcok_size_align)
        out_res = tvm.compute(out_res_shape,
                              lambda *i: trans_format0[i],
                              name="out_res",
                              attrs={
                                  'extract_params': extract_params,
                                  'setfmatrix_dict': setfmatrix_dict
                              })

        return out_res


# 'pylint: disable=unnecessary-lambda,too-many-locals
@register_operator_compute("extract_volume_patches", op_mode="static", support_fusion=True)
def extract_volume_patches_compute(data_input, fmap_c, ksizes, strides, padding):
    """
    ops compute

    Parameters
    ----------
    data_input:  TVM tensor
        the placeholder of input_x
    ksizes: input attr
    strides: input attr
    padding: input attr
    Returns
    -------
    compute results
    """
    output_image_patches = _extract_volume_patches_compute_6hd(data_input, fmap_c, ksizes, strides, padding)

    blcok_size_align = 16 if data_input.dtype == "float16" else 32
    if fmap_c % blcok_size_align == 0:
        return [output_image_patches]
    return list(output_image_patches)


# 'pylint: disable=locally-disabled, too-many-statements, too-many-locals
def _extract_volume_patches_schedule(res, sch_list, original_cin):
    """
    extract_image_patches schedule

    Parameters
    ----------
    res: the multi-results in the operator
    sch_list: schedule list

    Returns
    -------
    None
    """
    sch = sch_list[0]

    dtype_input = res.dtype
    if dtype_input in ("int8", "uint8"):
        blcok_size_align = Constant.BLOCK_SIZE_INT8  # 32
    else:
        blcok_size_align = Constant.BLOCK_SIZE_FP16  # 16

    dtype_size = 2 if dtype_input == "float16" else 1

    if original_cin % blcok_size_align != 0 and dtype_input == "float16":
        tensor7 = res.op.input_tensors[0]
        tensor6 = tensor7.op.input_tensors[0]
        tensor5 = tensor6.op.input_tensors[0]
        tensor42 = tensor5.op.input_tensors[0]
        workspace2 = tensor42.op.input_tensors[0]
        tensor41 = workspace2.op.input_tensors[0]
        tensor4 = tensor41.op.input_tensors[0]
        tensor3 = tensor4.op.input_tensors[0]
        tensor2 = tensor3.op.input_tensors[0]
        tensor1 = tensor2.op.input_tensors[0]
        tensor0 = tensor1.op.input_tensors[0]
        workspace_res = tensor0.op.input_tensors[0]
        image_patches_res_transformat = workspace_res.op.input_tensors[0]
        image_patches_res = image_patches_res_transformat.op.input_tensors[0]
    elif original_cin % blcok_size_align != 0 and dtype_input in ("int8", "uint8"):
        tensor7 = res.op.input_tensors[0]
        tensor6 = tensor7.op.input_tensors[0]
        tensor5 = tensor6.op.input_tensors[0]
        tensor42 = tensor5.op.input_tensors[0]
        workspace2 = tensor42.op.input_tensors[0]
        tensor41 = workspace2.op.input_tensors[0]
        tensor4 = tensor41.op.input_tensors[0]
        tensor3 = tensor4.op.input_tensors[0]
        tensor2 = tensor3.op.input_tensors[0]
        tensor1 = tensor2.op.input_tensors[0]
        tensor10 = tensor1.op.input_tensors[0]
        workspace_8bit = tensor10.op.input_tensors[0]
        tensor00 = workspace_8bit.op.input_tensors[0]
        tensor0 = tensor00.op.input_tensors[0]
        workspace_res = tensor0.op.input_tensors[0]
        image_patches_res = workspace_res.op.input_tensors[0]
    else:
        trans_format0 = res.op.input_tensors[0]
        ub_res = trans_format0.op.input_tensors[0]
        workspace_res = ub_res.op.input_tensors[0]
        image_patches_res = workspace_res.op.input_tensors[0]

    image_patches_split_c1 = image_patches_res.op.input_tensors[0]
    image_patches = image_patches_split_c1.op.input_tensors[0]
    fmap_fractal = image_patches.op.input_tensors[0]
    fmap_im2col = fmap_fractal.op.input_tensors[0]
    fmap_in_l1 = fmap_im2col.op.input_tensors[0]

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

    fmap_shape = extract_params.get("fmap_shape")
    (filter_d, filter_h, filter_w) = extract_params.get("ksizes")
    (filter_d, filter_h, filter_w) = (filter_d.value, filter_h.value, filter_w.value)
    (stride_d, stride_h, stride_w) = extract_params.get("strides")
    (stride_d, stride_h, stride_w) = (stride_d.value, stride_h.value, stride_w.value)
    (fmap_batch, fmap_d, fmap_c1, fmap_h, fmap_w, _) = fmap_shape
    fmap_batch, fmap_d, fmap_c1, fmap_h, fmap_w = \
        fmap_batch.value, fmap_d.value, fmap_c1.value, fmap_h.value, fmap_w.value
    padding = extract_params.get("padding_mode")
    original_cin = extract_params.get("original_cin")
    out_d = extract_params.get("out_d")
    out_w = extract_params.get("out_w")
    out_w = extract_params.get("out_w")
    howo = extract_params.get("howo")

    compute_inline_tensor42 = 1 if original_cin % blcok_size_align != 0 else 0
    compute_inline_tensor3_6 = 1 if original_cin % blcok_size_align != 0 and dtype_input == "float16" else 0
    compute_inline_tensor4_41 = 1

    sch[fmap_in_l1].set_scope(tbe_platform.scope_cbuf)
    sch[fmap_im2col].set_scope(tbe_platform.scope_cbuf)
    sch[fmap_fractal].set_scope(tbe_platform.scope_ubuf)
    sch[image_patches_res].set_scope(tbe_platform.scope_ubuf)
    sch[workspace_res].set_scope(tbe_platform.scope_gm)
    if original_cin % blcok_size_align != 0 and dtype_input == "float16":
        sch[image_patches_res_transformat].set_scope(tbe_platform.scope_ubuf)
        sch[tensor0].set_scope(tbe_platform.scope_ubuf)
        sch[tensor1].set_scope(tbe_platform.scope_ubuf)
        sch[tensor2].set_scope(tbe_platform.scope_ubuf)
        if not compute_inline_tensor4_41:
            sch[tensor4].set_scope(tbe_platform.scope_ubuf)
            sch[tensor41].set_scope(tbe_platform.scope_ubuf)
        if not compute_inline_tensor3_6:
            sch[tensor3].set_scope(tbe_platform.scope_ubuf)
            sch[tensor6].set_scope(tbe_platform.scope_ubuf)
        sch[workspace2].set_scope(tbe_platform.scope_gm)
        sch[tensor42].set_scope(tbe_platform.scope_ubuf)
        sch[tensor5].set_scope(tbe_platform.scope_ubuf)
        sch[tensor7].set_scope(tbe_platform.scope_ubuf)
    elif original_cin % blcok_size_align != 0 and dtype_input in ("uint8", "int8"):
        sch[tensor0].set_scope(tbe_platform.scope_ubuf)
        sch[tensor00].set_scope(tbe_platform.scope_ubuf)
        sch[workspace_8bit].set_scope(tbe_platform.scope_gm)
        sch[tensor10].set_scope(tbe_platform.scope_ubuf)
        sch[tensor1].set_scope(tbe_platform.scope_ubuf)
        sch[tensor2].set_scope(tbe_platform.scope_ubuf)
        if not compute_inline_tensor4_41:
            sch[tensor4].set_scope(tbe_platform.scope_ubuf)
            sch[tensor41].set_scope(tbe_platform.scope_ubuf)
        if not compute_inline_tensor3_6:
            sch[tensor3].set_scope(tbe_platform.scope_ubuf)
            sch[tensor6].set_scope(tbe_platform.scope_ubuf)
        sch[workspace2].set_scope(tbe_platform.scope_gm)
        sch[tensor42].set_scope(tbe_platform.scope_ubuf)
        sch[tensor5].set_scope(tbe_platform.scope_ubuf)
        sch[tensor7].set_scope(tbe_platform.scope_ubuf)
    else:
        sch[ub_res].set_scope(tbe_platform.scope_ubuf)
        sch[trans_format0].set_scope(tbe_platform.scope_ubuf)

    # compute inline
    sch[image_patches].compute_inline()
    sch[image_patches_split_c1].compute_inline()

    # align to 32B
    # c1 * BlockSize must be integer multiple of 16B
    sch[fmap_im2col].buffer_align((1, 1), (1, 1), (out_w, out_w), (1, 1), (1, 1), (1, 1), (1, blcok_size_align))
    sch[fmap_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE), (1, blcok_size_align))
    if original_cin % blcok_size_align != 0 and dtype_input == "float16":
        sch[image_patches_res].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE),
                                            (1, blcok_size_align))
        sch[tensor1].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE),
                                  (1, blcok_size_align))
        sch[tensor2].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, blcok_size_align),
                                  (1, Constant.BLOCK_SIZE))
        sch[tensor3].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE))
        sch[tensor4].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE))
        sch[tensor41].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE))

        sch[tensor42].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE))
        sch[tensor5].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, blcok_size_align), (1, Constant.BLOCK_SIZE))
        sch[tensor6].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE), (1, blcok_size_align))
        sch[tensor7].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE), (1, blcok_size_align))
    elif original_cin % blcok_size_align != 0 and dtype_input in ("uint8", "int8"):
        sch[image_patches_res].buffer_align((1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE), (1, 1), (1, 1), (1, 1),
                                            (1, blcok_size_align))
        sch[tensor0].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, blcok_size_align))
        sch[tensor00].buffer_align((1, 1), (1, 1), (1, 1), (1, blcok_size_align), (1, 1), (1, 1), (1, 1),
                                   (1, blcok_size_align))
        sch[tensor10].buffer_align((1, 1), (1, 1), (1, 1), (1, blcok_size_align), (1, 1), (1, 1), (1, 1),
                                   (1, blcok_size_align))
        sch[tensor1].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, blcok_size_align),
                                  (1, blcok_size_align))
        sch[tensor2].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, blcok_size_align),
                                  (1, blcok_size_align))
        sch[tensor3].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, blcok_size_align))
        sch[tensor4].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, blcok_size_align))
        sch[tensor41].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, blcok_size_align))
        sch[tensor42].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, blcok_size_align))
        sch[tensor5].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, blcok_size_align), (1, blcok_size_align))
        sch[tensor6].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, blcok_size_align), (1, blcok_size_align))
        sch[tensor7].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, blcok_size_align), (1, blcok_size_align))
    else:
        sch[image_patches_res].buffer_align((1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE), (1, 1), (1, 1), (1, 1),
                                            (1, blcok_size_align))
        sch[ub_res].storage_align(ub_res.op.axis[4], blcok_size_align, 0)

    # compute tiling params
    fmap_fractal_size = fmap_batch * fmap_d * howo * fmap_c1 * filter_h * filter_w * blcok_size_align
    image_patches_res_size = fmap_batch * out_d * howo * filter_d * fmap_c1 * filter_h * filter_w * blcok_size_align

    total_size = fmap_fractal_size + image_patches_res_size

    max_next_valid_size = Constant.MAX_UB_SIZE * image_patches_res_size // total_size

    max_l1_valid_size = Constant.MAX_L1_SIZE

    aligned_flag = original_cin % blcok_size_align == 0

    is_tiling_valid, shape_in_l1, is_l1_double_buffer, shape_after_load3d, is_l0_ub_double_buffer, \
    howo_split = _get_load3d_tiling(fmap_shape, (filter_d, filter_h, filter_w), (stride_d, stride_h, stride_w), \
                                    padding, max_l1_valid_size, max_next_valid_size, dtype_input, aligned_flag)

    if (is_tiling_valid, shape_in_l1, is_l1_double_buffer, shape_after_load3d,
            is_l0_ub_double_buffer) == (False, None, None, None, None):
        error_manager_vector.raise_err_specific_reson(
            "extract_volume_patches",
            "Not supported fmap shape = (%s), kernel = (1, %u, %u, %u, 1), stride = (1, %u, %u, %u, 1)" %
            (fmap_shape, filter_d, filter_h, filter_w, stride_d, stride_h, stride_w))

    (_, ub_do, ub_howo, _, ub_kd, ub_khkw, _) = shape_after_load3d

    # for load3d emit_insn
    _, fmap_im2col_d_inner = sch[fmap_im2col].split(fmap_im2col.op.axis[1], factor=1)
    _, fmap_fractal_d_inner = sch[fmap_fractal].split(fmap_fractal.op.axis[1], factor=1)
    if original_cin % blcok_size_align != 0 and dtype_input == "float16":
        # cut workspace_res
        workspace_res_n_outer, workspace_res_n_inner = sch[workspace_res].split(workspace_res.op.axis[0], factor=1)
        workspace_res_do_outer, workspace_res_do_inner = sch[workspace_res].split(workspace_res.op.axis[1], factor=1)
        workspace_res_howo_outer, workspace_res_howo_inner = sch[workspace_res].split(
            workspace_res.op.axis[2], factor=(ub_howo + blcok_size_align - 1) // blcok_size_align)
        workspace_res_kd_outer, workspace_res_kd_inner = sch[workspace_res].split(workspace_res.op.axis[3],
                                                                                  factor=ub_kd)
        workspace_res_khkw_outer, workspace_res_khkw_inner = sch[workspace_res].split(workspace_res.op.axis[4],
                                                                                      factor=ub_khkw)
        workspace_res_c1_outer, workspace_res_c1_inner = sch[workspace_res].split(workspace_res.op.axis[5], factor=1)

        sch[workspace_res].reorder(workspace_res_n_outer, workspace_res_do_outer, workspace_res_c1_outer,
                                   workspace_res_howo_outer, workspace_res_kd_outer, workspace_res_khkw_outer,
                                   workspace_res_n_inner, workspace_res_do_inner, workspace_res_c1_inner,
                                   workspace_res_howo_inner, workspace_res_kd_inner, workspace_res_khkw_inner,
                                   workspace_res.op.axis[6], workspace_res.op.axis[7])

        # for compute_at
        if howo_split:
            sch[fmap_in_l1].compute_at(sch[workspace_res], workspace_res_howo_outer)
            sch[fmap_im2col].compute_at(sch[workspace_res], workspace_res_howo_outer)
        else:
            sch[fmap_in_l1].compute_at(sch[workspace_res], workspace_res_kd_outer)
            sch[fmap_im2col].compute_at(sch[workspace_res], workspace_res_kd_outer)

        sch[fmap_fractal].compute_at(sch[workspace_res], workspace_res_khkw_outer)
        sch[image_patches_res].compute_at(sch[workspace_res], workspace_res_khkw_outer)
        sch[image_patches_res_transformat].compute_at(sch[workspace_res], workspace_res_khkw_outer)

        tensor1_shape = tensor1.shape
        tensor3_shape = tensor3.shape
        fmap_batch, dout, m_ho_wo, kd, khkw, fmap_c = tensor3_shape[:6]

        def _cal_max_devisor(num, factor):
            max_devisor = 1
            for i in range(factor, 0, -1):
                if num % i == 0:
                    max_devisor = i
                    break
            return max_devisor

        # 'pylint: disable=too-many-return-statements
        def _cal_workspace2_factor():
            tensor1_num = functools.reduce(lambda x, y: x * y, tensor1_shape[-3:])
            if hasattr(tensor1_num, "value"):
                tensor1_num = tensor1_num.value
            tensor2_num = tensor1_num
            workspace2_khkw_factor, workspace2_kd_factor, workspace2_mhowo_factor,\
              split_workspace2_kd_flag, split_workspace2_mhowo_flag = 1, 1, 1, 0, 0
            workspace2_double_buffer_flag = 1 if (Constant.MAX_UB_SIZE // 4) // (tensor1_num + tensor2_num) else 0
            if workspace2_double_buffer_flag:
                ub_num = Constant.MAX_UB_SIZE // (Constant.DOUBLE_BUFFER * dtype_size)
            else:
                ub_num = Constant.MAX_UB_SIZE // dtype_size
            max_khkw_factor = ub_num // (tensor1_num + tensor2_num)
            if max_khkw_factor == 0:
                error_manager_vector.raise_err_specific_reson(
                    "extract_volume_patches",
                    "Not Support shape, needs UB memory size: %d" % ((tensor1_num + tensor2_num) * 2))
            workspace2_khkw_factor = min(max_khkw_factor, khkw.value)
            if max_khkw_factor >= khkw.value * 2:
                split_workspace2_kd_flag = 1
                max_kd_factor = max_khkw_factor // khkw.value
                workspace2_kd_factor = min(max_kd_factor, kd.value)
                if max_kd_factor >= kd.value * 2:
                    split_workspace2_mhowo_flag = 1
                    max_mhowo_factor = max_kd_factor // kd.value
                    workspace2_mhowo_factor = min(max_mhowo_factor, m_ho_wo.value)
            return workspace2_khkw_factor, workspace2_kd_factor, workspace2_mhowo_factor, split_workspace2_kd_flag,\
                   split_workspace2_mhowo_flag, workspace2_double_buffer_flag
        workspace2_khkw_factor, workspace2_kd_factor, workspace2_mhowo_factor, split_workspace2_kd_flag,\
          split_workspace2_mhowo_flag, workspace2_double_buffer_flag = _cal_workspace2_factor()
        if hasattr(khkw, "value"):
            khkw = khkw.value
        if hasattr(workspace2_khkw_factor, "value"):
            workspace2_khkw_factor = workspace2_khkw_factor.value
        max_khkw_devisor = _cal_max_devisor(khkw, workspace2_khkw_factor)

        workspace2_n_outer, workspace2_n_inner = sch[workspace2].split(workspace2.op.axis[0], factor=1)
        workspace2_do_outer, workspace2_do_inner = sch[workspace2].split(workspace2.op.axis[1], factor=1)
        workspace2_mhowo_outer, workspace2_mhowo_inner = sch[workspace2].split(workspace2.op.axis[2],
                                                                               factor=workspace2_mhowo_factor)

        workspace2_kdkhkwc_outer, workspace2_kdkhkwc_inner = sch[workspace2].split(workspace2.op.axis[3],
                                                                                   factor=khkw * fmap_c)  # kd, khkwc
        workspace2_kd_outer, workspace2_kd_inner = \
            sch[workspace2].split(workspace2_kdkhkwc_outer,
                                  factor=workspace2_kd_factor)  # kd_outer, kd_inner
        workspace2_khkwc_outer, workspace2_khkwc_inner = sch[workspace2].split(workspace2_kdkhkwc_inner,
                                                                               factor=fmap_c)  # khkw, c
        workspace2_khkw_outer, workspace2_khkw_inner = sch[workspace2].split(workspace2_khkwc_outer,
                                                                             factor=max_khkw_devisor)  # kh, kw
        sch[workspace2].reorder(workspace2_n_outer, workspace2_do_outer, workspace2_mhowo_outer, workspace2_kd_outer,
                                workspace2_khkw_outer, workspace2_khkw_inner, workspace2_n_inner, workspace2_do_inner,
                                workspace2_mhowo_inner, workspace2_kd_inner, workspace2_khkwc_inner,
                                workspace2.op.axis[4])

        if split_workspace2_mhowo_flag:
            sch[tensor0].compute_at(sch[workspace2], workspace2_mhowo_outer)
            sch[tensor1].compute_at(sch[workspace2], workspace2_mhowo_outer)
            sch[tensor2].compute_at(sch[workspace2], workspace2_mhowo_outer)
            if not compute_inline_tensor4_41:
                sch[tensor4].compute_at(sch[workspace2], workspace2_mhowo_outer)
                sch[tensor41].compute_at(sch[workspace2], workspace2_mhowo_outer)
            if not compute_inline_tensor3_6:
                sch[tensor3].compute_at(sch[workspace2], workspace2_mhowo_outer)
        elif split_workspace2_kd_flag:
            sch[tensor0].compute_at(sch[workspace2], workspace2_kd_outer)
            sch[tensor1].compute_at(sch[workspace2], workspace2_kd_outer)
            sch[tensor2].compute_at(sch[workspace2], workspace2_kd_outer)
            if not compute_inline_tensor4_41:
                sch[tensor4].compute_at(sch[workspace2], workspace2_kd_outer)
                sch[tensor41].compute_at(sch[workspace2], workspace2_kd_outer)
            if not compute_inline_tensor3_6:
                sch[tensor3].compute_at(sch[workspace2], workspace2_kd_outer)
        else:
            sch[tensor0].compute_at(sch[workspace2], workspace2_khkw_outer)
            sch[tensor1].compute_at(sch[workspace2], workspace2_khkw_outer)
            sch[tensor2].compute_at(sch[workspace2], workspace2_khkw_outer)
            if not compute_inline_tensor4_41:
                sch[tensor4].compute_at(sch[workspace2], workspace2_khkw_outer)
                sch[tensor41].compute_at(sch[workspace2], workspace2_khkw_outer)
            if not compute_inline_tensor3_6:
                sch[tensor3].compute_at(sch[workspace2], workspace2_khkw_outer)

        mhowo = tensor7.shape[-3].value
        fmap_c = fmap_c.value

        # 'pylint: disable=too-many-return-statements
        def _cal_res_factor():
            tensor5_num = functools.reduce(lambda x, y: x * y, tensor5.shape[-2:])
            tensor5_num = tensor5_num.value
            tensor42_num = tensor5_num
            tensor7_num = tensor42_num
            res_kdkhkwc_floor_factor, res_mhowo_factor, split_res_kdkhkwc_floor_flag, split_res_mhowo_flag = 1, 1, 0, 0
            if (Constant.MAX_UB_SIZE // 4) // (tensor42_num + tensor5_num + tensor7_num):
                res_double_buffer_flag = 1
            else:
                res_double_buffer_flag = 0
            if res_double_buffer_flag:
                ub_num = Constant.MAX_UB_SIZE // (Constant.DOUBLE_BUFFER * dtype_size)
            else:
                ub_num = Constant.MAX_UB_SIZE // dtype_size
            max_mhowo_factor = ub_num // (tensor42_num + tensor5_num + tensor7_num)
            if max_mhowo_factor == 0:
                error_manager_vector.raise_err_specific_reson(
                    "extract_volume_patches",
                    "Not Support shape, needs UB memory size: %d" % (tensor42_num + tensor5_num + tensor7_num) * 2)
            res_mhowo_factor = min(max_mhowo_factor, mhowo)
            split_res_mhowo_flag = 1
            if max_mhowo_factor >= mhowo * 2:
                split_res_kdkhkwc_floor_flag = 1
                max_kdkhkwc_floor_factor = max_mhowo_factor // mhowo
                res_kdkhkwc_floor_factor = \
                    min(max_kdkhkwc_floor_factor,
                        (filter_d * filter_h * filter_w * fmap_c + blcok_size_align - 1) // blcok_size_align)
            return res_kdkhkwc_floor_factor, res_mhowo_factor, split_res_kdkhkwc_floor_flag, split_res_mhowo_flag,\
                   res_double_buffer_flag

        res_kdkhkwc_floor_factor, res_mhowo_factor, split_res_kdkhkwc_floor_flag, split_res_mhowo_flag,\
          res_double_buffer_flag = _cal_res_factor()
        res_n_outer, res_n_inner = sch[res].split(res.op.axis[0], factor=1)
        res_do_outer, res_do_inner = sch[res].split(res.op.axis[1], factor=1)
        res_kdkhkwc_floor_outer, res_kdkhkwc_floor_inner = sch[res].split(res.op.axis[2],
                                                                          factor=res_kdkhkwc_floor_factor)
        # out_h*out_w // blcok_size_align, blcok_size_align
        res_howo_outer, res_howo_inner = sch[res].split(res.op.axis[3], factor=blcok_size_align)
        # down to blcok_size_align multiple
        res_mhowo_outer, res_mhowo_inner = sch[res].split(res_howo_outer, factor=res_mhowo_factor)

        sch[res].reorder(res_n_outer, res_do_outer, res_kdkhkwc_floor_outer, res_mhowo_outer, res_n_inner,
                         res_do_inner, res_kdkhkwc_floor_inner, res_mhowo_inner, res_howo_inner, res.op.axis[4])

        if split_res_mhowo_flag:
            sch[tensor42].compute_at(sch[res], res_mhowo_outer)
            sch[tensor5].compute_at(sch[res], res_mhowo_outer)
            if not compute_inline_tensor3_6:
                sch[tensor6].compute_at(sch[res], res_mhowo_outer)
            sch[tensor7].compute_at(sch[res], res_mhowo_outer)
        elif split_res_kdkhkwc_floor_flag:
            sch[tensor42].compute_at(sch[res], res_kdkhkwc_floor_outer)
            sch[tensor5].compute_at(sch[res], res_kdkhkwc_floor_outer)
            if not compute_inline_tensor3_6:
                sch[tensor6].compute_at(sch[res], res_kdkhkwc_floor_outer)
            sch[tensor7].compute_at(sch[res], res_kdkhkwc_floor_outer)
        else:
            error_manager_vector.raise_err_specific_reson("extract_volume_patches",
                                                          "Not Support shape, UB memory size overflow")

        # for emit_insn
        sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], tbe_platform.DMA_COPY)
        sch[fmap_im2col].emit_insn(fmap_im2col_d_inner, 'set_fmatrix', setfmatrix_dict)
        sch[fmap_fractal].emit_insn(fmap_fractal_d_inner, tbe_platform.IM2COL)

        sch[image_patches_res].emit_insn(image_patches_res.op.axis[0], tbe_platform.ADDVS)
        sch[image_patches_res_transformat].emit_insn(image_patches_res_transformat.op.axis[0],
                                                     tbe_platform.ADDVS)
        sch[workspace_res].emit_insn(workspace_res_n_inner, tbe_platform.DMA_COPY)

        sch[tensor0].emit_insn(tensor0.op.axis[0], tbe_platform.DMA_COPY)
        sch[tensor1].emit_insn(tensor1.op.axis[0], tbe_platform.ADDVS)
        sch[tensor2].emit_insn(tensor2.op.axis[-2], "vnchwconv")
        if not compute_inline_tensor4_41:
            sch[tensor4].emit_insn(tensor4.op.axis[0], tbe_platform.DMA_COPY)
            sch[tensor41].emit_insn(tensor41.op.axis[0], tbe_platform.DMA_COPY)
        if not compute_inline_tensor3_6:
            sch[tensor3].emit_insn(tensor3.op.axis[0], tbe_platform.ADDVS)
            sch[tensor6].emit_insn(tensor6.op.axis[-2], "vnchwconv")
        sch[workspace2].emit_insn(workspace2_n_inner, tbe_platform.DMA_COPY)
        sch[tensor42].emit_insn(tensor42.op.axis[0], tbe_platform.DMA_COPY)
        sch[tensor5].emit_insn(tensor5.op.axis[0], tbe_platform.DMA_PADDING)
        sch[tensor7].emit_insn(tensor7.op.axis[0], tbe_platform.DMA_COPY)
        sch[res].emit_insn(res_n_inner, tbe_platform.DMA_COPY)

        # for multi cores
        res_shape = [int(i.value) for i in res.shape]
        res_n, res_do, _, _, _ = res_shape
        fmap_batch = fmap_batch.value
        dout = dout.value
        fmap_batch_factor, dout_factor, _ = _cal_multi_core_factor_3_axis(fmap_batch, dout, 1)
        workspace_cut_n_outer, workspace_cut_n_inner = sch[workspace_res].split(workspace_res_n_outer,
                                                                                nparts=fmap_batch_factor)
        workspace2_cut_n_outer, workspace2_cut_n_inner = sch[workspace2].split(workspace2_n_outer,
                                                                               nparts=fmap_batch_factor)
        res_cut_n_outer, res_cut_n_inner = sch[res].split(res_n_outer, nparts=fmap_batch_factor)
        workspace_cut_do_outer, workspace_cut_do_inner = sch[workspace_res].split(workspace_res_do_outer,
                                                                                  nparts=dout_factor)
        workspace2_cut_do_outer, workspace2_cut_do_inner = sch[workspace2].split(workspace2_do_outer,
                                                                                 nparts=dout_factor)
        res_cut_do_outer, res_cut_do_inner = sch[res].split(res_do_outer, nparts=dout_factor)
        sch[workspace_res].reorder(workspace_cut_n_outer, workspace_cut_do_outer, workspace_cut_n_inner,
                                   workspace_cut_do_inner)
        sch[workspace2].reorder(workspace2_cut_n_outer, workspace2_cut_do_outer, workspace2_cut_n_inner,
                                workspace2_cut_do_inner)
        sch[res].reorder(res_cut_n_outer, res_cut_do_outer, res_cut_n_inner, res_cut_do_inner)
        workspace_fused_axis = sch[workspace_res].fuse(workspace_cut_n_outer, workspace_cut_do_outer)
        workspace2_fused_axis = sch[workspace2].fuse(workspace2_cut_n_outer, workspace2_cut_do_outer)
        res_fused_axis = sch[res].fuse(res_cut_n_outer, res_cut_do_outer)

        block_idx = tvm.thread_axis('blockIdx.x')
        sch[workspace_res].bind(workspace_fused_axis, block_idx)
        sch[workspace2].bind(workspace2_fused_axis, block_idx)
        sch[res].bind(res_fused_axis, block_idx)

        # for double buffer
        if is_l0_ub_double_buffer:
            sch[fmap_fractal].double_buffer()
            sch[image_patches_res].double_buffer()
            sch[fmap_im2col].double_buffer()
        if is_l1_double_buffer:
            sch[fmap_in_l1].double_buffer()

        if workspace2_double_buffer_flag:
            sch[tensor0].double_buffer()
            sch[tensor1].double_buffer()
            sch[tensor2].double_buffer()
            if not compute_inline_tensor3_6:
                sch[tensor3].double_buffer()
            if not compute_inline_tensor4_41:
                sch[tensor4].double_buffer()
                sch[tensor41].double_buffer()

        if res_double_buffer_flag:
            sch[tensor42].double_buffer()
            sch[tensor5].double_buffer()
            if not compute_inline_tensor3_6:
                sch[tensor6].double_buffer()
            sch[tensor7].double_buffer()

        # for compute_inline
        sch[image_patches_res_transformat].compute_inline()
        sch[tensor1].compute_inline()
        if compute_inline_tensor4_41:
            sch[tensor41].compute_inline()
            sch[tensor4].compute_inline()
        if compute_inline_tensor3_6:
            sch[tensor3].compute_inline()
            sch[tensor6].compute_inline()
        if compute_inline_tensor42:
            sch[tensor42].compute_inline()

    elif original_cin % blcok_size_align != 0 and dtype_input in ("uint8", "int8"):  # cut workspace_res
        workspace_res_n_outer, workspace_res_n_inner = sch[workspace_res].split(workspace_res.op.axis[0], factor=1)
        workspace_res_do_outer, workspace_res_do_inner = sch[workspace_res].split(workspace_res.op.axis[1], factor=1)
        workspace_res_howo_outer, workspace_res_howo_inner = sch[workspace_res].split(workspace_res.op.axis[2],
                                                                                      factor=ub_howo)
        workspace_res_kd_outer, workspace_res_kd_inner = sch[workspace_res].split(workspace_res.op.axis[3],
                                                                                  factor=ub_kd)
        workspace_res_khkw_outer, workspace_res_khkw_inner = sch[workspace_res].split(workspace_res.op.axis[4],
                                                                                      factor=ub_khkw)
        workspace_res_c1_outer, workspace_res_c1_inner = sch[workspace_res].split(workspace_res.op.axis[5], factor=1)

        sch[workspace_res].reorder(workspace_res_n_outer, workspace_res_do_outer, workspace_res_c1_outer,
                                   workspace_res_howo_outer, workspace_res_kd_outer, workspace_res_khkw_outer,
                                   workspace_res_n_inner, workspace_res_do_inner, workspace_res_c1_inner,
                                   workspace_res_howo_inner, workspace_res_kd_inner, workspace_res_khkw_inner,
                                   workspace_res.op.axis[6])

        # for compute_at
        if howo_split:
            sch[fmap_in_l1].compute_at(sch[workspace_res], workspace_res_howo_outer)
            sch[fmap_im2col].compute_at(sch[workspace_res], workspace_res_howo_outer)
        else:
            sch[fmap_in_l1].compute_at(sch[workspace_res], workspace_res_kd_outer)
            sch[fmap_im2col].compute_at(sch[workspace_res], workspace_res_kd_outer)

        sch[fmap_fractal].compute_at(sch[workspace_res], workspace_res_khkw_outer)
        sch[image_patches_res].compute_at(sch[workspace_res], workspace_res_khkw_outer)

        tensor00_shape = tensor00.shape
        fmap_batch, dout, howo_8bit_align_floor, _, kd, khkw, fmap_c1 = tensor00_shape[:7]

        # 'pylint: disable=too-many-return-statements
        def _cal_workspace_8bit_factor():
            tensor0_num = blcok_size_align * blcok_size_align * fmap_c1
            tensor0_num = tensor0_num.value
            tensor00_num = tensor0_num
            workspace_8bit_khkw_factor, workspace_8bit_kd_factor, workspace_8bit_mhowo_factor,\
              split_workspace_8bit_kd_flag, split_workspace_8bit_mhowo_flag = 1, 1, 1, 0, 0
            if (Constant.MAX_UB_SIZE // 2) // (tensor0_num + tensor00_num):
                workspace_8bit_double_buffer_flag = 1
            else:
                workspace_8bit_double_buffer_flag = 0
            if workspace_8bit_double_buffer_flag:
                ub_num = Constant.MAX_UB_SIZE // (Constant.DOUBLE_BUFFER * dtype_size)
            else:
                ub_num = Constant.MAX_UB_SIZE // dtype_size
            max_khkw_factor = ub_num // (tensor0_num + tensor00_num)
            if max_khkw_factor == 0:
                error_manager_vector.raise_err_specific_reson("extract_volume_patches",
                                                              "Not Support shape, needs UB memory size: %d" %
                                                              (tensor00_num + tensor0_num) * 2)
            workspace_8bit_khkw_factor = min(max_khkw_factor, khkw.value)
            if max_khkw_factor >= khkw.value * 2:
                split_workspace_8bit_kd_flag = 1
                max_kd_factor = max_khkw_factor // khkw.value
                workspace_8bit_kd_factor = min(max_kd_factor, kd.value)
                if max_kd_factor >= kd.value * 2:
                    split_workspace_8bit_mhowo_flag = 1
                    max_mhowo_factor = max_kd_factor // kd.value
                    workspace_8bit_mhowo_factor = min(max_mhowo_factor, howo_8bit_align_floor.value)
            return workspace_8bit_khkw_factor, workspace_8bit_kd_factor, workspace_8bit_mhowo_factor,\
                   split_workspace_8bit_kd_flag, split_workspace_8bit_mhowo_flag, workspace_8bit_double_buffer_flag
        workspace_8bit_khkw_factor, workspace_8bit_kd_factor, workspace_8bit_mhowo_factor,\
          split_workspace_8bit_kd_flag, split_workspace_8bit_mhowo_flag, \
          workspace_8bit_double_buffer_flag = _cal_workspace_8bit_factor()
        if hasattr(khkw, "value"):
            khkw = khkw.value
        if hasattr(workspace_8bit_khkw_factor, "value"):
            workspace_8bit_khkw_factor = workspace_8bit_khkw_factor.value

        workspace_8bit_n_outer, workspace_8bit_n_inner = sch[workspace_8bit].split(workspace_8bit.op.axis[0], factor=1)
        workspace_8bit_do_outer, workspace_8bit_do_inner = sch[workspace_8bit].split(workspace_8bit.op.axis[1],
                                                                                     factor=1)
        workspace_8bit_mhowo_outer, workspace_8bit_mhowo_inner = \
            sch[workspace_8bit].split(workspace_8bit.op.axis[2],
                                      factor=workspace_8bit_mhowo_factor)

        workspace_8bit_kd_outer, workspace_8bit_kd_inner = \
            sch[workspace_8bit].split(workspace_8bit.op.axis[4],
                                      factor=workspace_8bit_kd_factor)  # kd, khkwc
        workspace_8bit_khkw_outer, workspace_8bit_khkw_inner = \
            sch[workspace_8bit].split(workspace_8bit.op.axis[5],
                                      factor=workspace_8bit_khkw_factor)  # kh, kw
        sch[workspace_8bit].reorder(workspace_8bit_n_outer, workspace_8bit_do_outer, workspace_8bit_mhowo_outer,
                                    workspace_8bit_kd_outer, workspace_8bit_khkw_outer, workspace_8bit_n_inner,
                                    workspace_8bit_do_inner, workspace_8bit_mhowo_inner, workspace_8bit_kd_inner,
                                    workspace_8bit_khkw_inner, workspace_8bit.op.axis[3], workspace_8bit.op.axis[6],
                                    workspace_8bit.op.axis[7])

        if split_workspace_8bit_mhowo_flag:
            sch[tensor0].compute_at(sch[workspace_8bit], workspace_8bit_mhowo_outer)
            sch[tensor00].compute_at(sch[workspace_8bit], workspace_8bit_mhowo_outer)
        elif split_workspace_8bit_kd_flag:
            sch[tensor0].compute_at(sch[workspace_8bit], workspace_8bit_kd_outer)
            sch[tensor00].compute_at(sch[workspace_8bit], workspace_8bit_kd_outer)
        else:
            sch[tensor0].compute_at(sch[workspace_8bit], workspace_8bit_khkw_outer)
            sch[tensor00].compute_at(sch[workspace_8bit], workspace_8bit_khkw_outer)

        tensor10_shape = tensor10.shape
        tensor1_shape = tensor1.shape
        tensor3_shape = tensor3.shape
        fmap_batch, dout, m_ho_wo, kd, khkw, fmap_c = tensor3_shape[:6]

        def _cal_max_devisor(num, factor):
            max_devisor = 1
            for i in range(factor, 0, -1):
                if num % i == 0:
                    max_devisor = i
                    break
            return max_devisor

        # 'pylint: disable=too-many-return-statements
        def _cal_workspace2_factor():
            tensor10_num = functools.reduce(lambda x, y: x * y, tensor10_shape[-2:])
            if hasattr(tensor10_num, "value"):
                tensor10_num = tensor10_num.value
            tensor1_num = functools.reduce(lambda x, y: x * y, tensor1_shape[-3:])
            if hasattr(tensor1_num, "value"):
                tensor1_num = tensor1_num.value
            tensor2_num = tensor1_num
            workspace2_khkw_factor, workspace2_kd_factor, workspace2_mhowo_factor, split_workspace2_kd_flag,\
              split_workspace2_mhowo_flag = 1, 1, 1, 0, 0
            if (Constant.MAX_UB_SIZE // Constant.DOUBLE_BUFFER) // (tensor10_num + tensor1_num + tensor2_num):
                workspace2_double_buffer_flag = 1
            else:
                workspace2_double_buffer_flag = 0
            if workspace2_double_buffer_flag:
                ub_num = Constant.MAX_UB_SIZE // (Constant.DOUBLE_BUFFER * dtype_size)
            else:
                ub_num = Constant.MAX_UB_SIZE // dtype_size
            max_khkw_factor = ub_num // (tensor10_num + tensor1_num + tensor2_num)
            if max_khkw_factor == 0:
                error_manager_vector.raise_err_specific_reson("extract_volume_patches",
                                                              "Not Support shape, needs UB memory size: %d" %
                                                              (tensor10_num + tensor1_num + tensor2_num) * 2)
            workspace2_khkw_factor = min(max_khkw_factor, khkw.value)
            if max_khkw_factor >= khkw.value * 2:
                split_workspace2_kd_flag = 1
                max_kd_factor = max_khkw_factor // khkw.value
                workspace2_kd_factor = min(max_kd_factor, kd.value)
                if max_kd_factor >= kd.value * 2:
                    split_workspace2_mhowo_flag = 1
                    max_mhowo_factor = max_kd_factor // kd.value
                    workspace2_mhowo_factor = min(max_mhowo_factor, m_ho_wo.value)
            return workspace2_khkw_factor, workspace2_kd_factor, workspace2_mhowo_factor, split_workspace2_kd_flag,\
                   split_workspace2_mhowo_flag, workspace2_double_buffer_flag
        workspace2_khkw_factor, workspace2_kd_factor, workspace2_mhowo_factor, split_workspace2_kd_flag,\
          split_workspace2_mhowo_flag, workspace2_double_buffer_flag = _cal_workspace2_factor()
        if hasattr(khkw, "value"):
            khkw = khkw.value
        if hasattr(workspace2_khkw_factor, "value"):
            workspace2_khkw_factor = workspace2_khkw_factor.value
        max_khkw_devisor = _cal_max_devisor(khkw, workspace2_khkw_factor)

        workspace2_n_outer, workspace2_n_inner = sch[workspace2].split(workspace2.op.axis[0], factor=1)
        workspace2_do_outer, workspace2_do_inner = sch[workspace2].split(workspace2.op.axis[1], factor=1)
        workspace2_mhowo_outer, workspace2_mhowo_inner = sch[workspace2].split(workspace2.op.axis[2],
                                                                               factor=workspace2_mhowo_factor)

        workspace2_kdkhkwc_outer, workspace2_kdkhkwc_inner = sch[workspace2].split(workspace2.op.axis[3],
                                                                                   factor=khkw * fmap_c)  # kd, khkwc
        workspace2_kd_outer, workspace2_kd_inner = \
            sch[workspace2].split(workspace2_kdkhkwc_outer,
                                  factor=workspace2_kd_factor)  # kd_outer, kd_inner
        workspace2_khkwc_outer, workspace2_khkwc_inner = sch[workspace2].split(workspace2_kdkhkwc_inner,
                                                                               factor=fmap_c)  # khkw, c
        workspace2_khkw_outer, workspace2_khkw_inner = sch[workspace2].split(workspace2_khkwc_outer,
                                                                             factor=max_khkw_devisor)  # kh, kw
        sch[workspace2].reorder(workspace2_n_outer, workspace2_do_outer, workspace2_mhowo_outer, workspace2_kd_outer,
                                workspace2_khkw_outer, workspace2_khkw_inner, workspace2_n_inner, workspace2_do_inner,
                                workspace2_mhowo_inner, workspace2_kd_inner, workspace2_khkwc_inner,
                                workspace2.op.axis[4])

        if split_workspace2_mhowo_flag:
            sch[tensor10].compute_at(sch[workspace2], workspace2_mhowo_outer)
            sch[tensor1].compute_at(sch[workspace2], workspace2_mhowo_outer)
            sch[tensor2].compute_at(sch[workspace2], workspace2_mhowo_outer)
            if not compute_inline_tensor4_41:
                sch[tensor4].compute_at(sch[workspace2], workspace2_mhowo_outer)
                sch[tensor41].compute_at(sch[workspace2], workspace2_mhowo_outer)
            if not compute_inline_tensor3_6:
                sch[tensor3].compute_at(sch[workspace2], workspace2_mhowo_outer)
        elif split_workspace2_kd_flag:
            sch[tensor10].compute_at(sch[workspace2], workspace2_kd_outer)
            sch[tensor1].compute_at(sch[workspace2], workspace2_kd_outer)
            sch[tensor2].compute_at(sch[workspace2], workspace2_kd_outer)
            if not compute_inline_tensor4_41:
                sch[tensor4].compute_at(sch[workspace2], workspace2_kd_outer)
                sch[tensor41].compute_at(sch[workspace2], workspace2_kd_outer)
            if not compute_inline_tensor3_6:
                sch[tensor3].compute_at(sch[workspace2], workspace2_kd_outer)
        else:
            sch[tensor10].compute_at(sch[workspace2], workspace2_khkw_outer)
            sch[tensor1].compute_at(sch[workspace2], workspace2_khkw_outer)
            sch[tensor2].compute_at(sch[workspace2], workspace2_khkw_outer)
            if not compute_inline_tensor4_41:
                sch[tensor4].compute_at(sch[workspace2], workspace2_khkw_outer)
                sch[tensor41].compute_at(sch[workspace2], workspace2_khkw_outer)
            if not compute_inline_tensor3_6:
                sch[tensor3].compute_at(sch[workspace2], workspace2_khkw_outer)

        mhowo = tensor7.shape[-3].value
        fmap_c = fmap_c.value

        # 'pylint: disable=too-many-return-statements
        def _cal_res_factor():
            tensor5_num = functools.reduce(lambda x, y: x * y, tensor5.shape[-2:])
            tensor5_num = tensor5_num.value
            tensor42_num = tensor5_num
            tensor7_num = tensor42_num
            res_kdkhkwc_floor_factor, res_mhowo_factor, split_res_kdkhkwc_floor_flag, split_res_mhowo_flag = 1, 1, 0, 0
            if (Constant.MAX_UB_SIZE // 4) // (tensor42_num + tensor5_num + tensor7_num):
                res_double_buffer_flag = 1
            else:
                res_double_buffer_flag = 0
            if res_double_buffer_flag:
                ub_num = Constant.MAX_UB_SIZE // (Constant.DOUBLE_BUFFER * dtype_size)
            else:
                ub_num = Constant.MAX_UB_SIZE // dtype_size
            max_mhowo_factor = ub_num // (tensor42_num + tensor5_num + tensor7_num)
            if max_mhowo_factor == 0:
                error_manager_vector.raise_err_specific_reson("extract_volume_patches",
                                                              "Not Support shape, needs UB memory size: %d" %
                                                              (tensor42_num + tensor5_num + tensor7_num) * 2)
            res_mhowo_factor = min(max_mhowo_factor, mhowo)
            split_res_mhowo_flag = 1
            if max_mhowo_factor >= mhowo * 2:
                split_res_kdkhkwc_floor_flag = 1
                max_kdkhkwc_floor_factor = max_mhowo_factor // mhowo
                res_kdkhkwc_floor_factor = \
                    min(max_kdkhkwc_floor_factor,
                        (filter_d * filter_h * filter_w * fmap_c + blcok_size_align - 1) // blcok_size_align)
            return res_kdkhkwc_floor_factor, res_mhowo_factor, split_res_kdkhkwc_floor_flag, split_res_mhowo_flag,\
                   res_double_buffer_flag

        res_kdkhkwc_floor_factor, res_mhowo_factor, split_res_kdkhkwc_floor_flag, split_res_mhowo_flag,\
          res_double_buffer_flag = _cal_res_factor()
        res_n_outer, res_n_inner = sch[res].split(res.op.axis[0], factor=1)
        res_do_outer, res_do_inner = sch[res].split(res.op.axis[1], factor=1)
        res_kdkhkwc_floor_outer, res_kdkhkwc_floor_inner = sch[res].split(res.op.axis[2],
                                                                          factor=res_kdkhkwc_floor_factor)
        # out_h*out_w // blcok_size_align, blcok_size_align
        res_howo_outer, res_howo_inner = sch[res].split(res.op.axis[3],
                                                        factor=blcok_size_align)  # down to blcok_size_align multiple
        res_mhowo_outer, res_mhowo_inner = sch[res].split(res_howo_outer, factor=res_mhowo_factor)

        sch[res].reorder(res_n_outer, res_do_outer, res_kdkhkwc_floor_outer, res_mhowo_outer, res_n_inner,
                         res_do_inner, res_kdkhkwc_floor_inner, res_mhowo_inner, res_howo_inner, res.op.axis[4])

        if split_res_mhowo_flag:
            sch[tensor42].compute_at(sch[res], res_mhowo_outer)
            sch[tensor5].compute_at(sch[res], res_mhowo_outer)
            if not compute_inline_tensor3_6:
                sch[tensor6].compute_at(sch[res], res_mhowo_outer)
            sch[tensor7].compute_at(sch[res], res_mhowo_outer)
        elif split_res_kdkhkwc_floor_flag:
            sch[tensor42].compute_at(sch[res], res_kdkhkwc_floor_outer)
            sch[tensor5].compute_at(sch[res], res_kdkhkwc_floor_outer)
            if not compute_inline_tensor3_6:
                sch[tensor6].compute_at(sch[res], res_kdkhkwc_floor_outer)
            sch[tensor7].compute_at(sch[res], res_kdkhkwc_floor_outer)
        else:
            error_manager_vector.raise_err_specific_reson("extract_volume_patches",
                                                          "Not Support shape, UB memory size overflow")

        # for emit_insn
        sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], tbe_platform.DMA_COPY)
        sch[fmap_im2col].emit_insn(fmap_im2col_d_inner, 'set_fmatrix', setfmatrix_dict)
        sch[fmap_fractal].emit_insn(fmap_fractal_d_inner, tbe_platform.IM2COL)

        sch[image_patches_res].emit_insn(image_patches_res.op.axis[0], tbe_platform.DMA_COPY)
        sch[workspace_res].emit_insn(workspace_res_n_inner, tbe_platform.DMA_COPY)

        sch[tensor0].emit_insn(tensor0.op.axis[0], tbe_platform.DMA_COPY)
        sch[tensor00].emit_insn(tensor00.op.axis[0], tbe_platform.DMA_COPY)
        sch[workspace_8bit].emit_insn(workspace_8bit_n_inner, tbe_platform.DMA_COPY)
        sch[tensor10].emit_insn(tensor10.op.axis[0], tbe_platform.DMA_COPY)
        sch[tensor1].emit_insn(tensor1.op.axis[0], tbe_platform.DMA_COPY)
        sch[tensor2].emit_insn(tensor2.op.axis[-2], "vnchwconv")
        if not compute_inline_tensor4_41:
            sch[tensor4].emit_insn(tensor4.op.axis[0], tbe_platform.DMA_COPY)
            sch[tensor41].emit_insn(tensor41.op.axis[0], tbe_platform.DMA_COPY)
        if not compute_inline_tensor3_6:
            sch[tensor3].emit_insn(tensor3.op.axis[0], tbe_platform.DMA_COPY)
            sch[tensor6].emit_insn(tensor6.op.axis[-2], "vnchwconv")
        sch[workspace2].emit_insn(workspace2_n_inner, tbe_platform.DMA_COPY)
        sch[tensor42].emit_insn(tensor42.op.axis[0], tbe_platform.DMA_COPY)
        sch[tensor5].emit_insn(tensor5.op.axis[0], tbe_platform.DMA_COPY)
        sch[tensor7].emit_insn(tensor7.op.axis[0], tbe_platform.DMA_COPY)
        sch[res].emit_insn(res_n_inner, tbe_platform.DMA_COPY)

        # for multi cores
        res_shape = [int(i.value) for i in res.shape]
        fmap_batch = fmap_batch.value
        dout = dout.value
        fmap_batch_factor, dout_factor, _ = _cal_multi_core_factor_3_axis(fmap_batch, dout, 1)
        workspace_cut_n_outer, workspace_cut_n_inner = sch[workspace_res].split(workspace_res_n_outer,
                                                                                nparts=fmap_batch_factor)
        workspace_8bit_cut_n_outer, workspace_8bit_cut_n_inner = \
            sch[workspace_8bit].split(workspace_8bit_n_outer, nparts=fmap_batch_factor)
        workspace2_cut_n_outer, workspace2_cut_n_inner = sch[workspace2].split(workspace2_n_outer,
                                                                               nparts=fmap_batch_factor)
        res_cut_n_outer, res_cut_n_inner = sch[res].split(res_n_outer, nparts=fmap_batch_factor)
        workspace_cut_do_outer, workspace_cut_do_inner = sch[workspace_res].split(workspace_res_do_outer,
                                                                                  nparts=dout_factor)
        workspace_8bit_cut_do_outer, workspace_8bit_cut_do_inner = \
            sch[workspace_8bit].split(workspace_8bit_do_outer, nparts=dout_factor)
        workspace2_cut_do_outer, workspace2_cut_do_inner = sch[workspace2].split(workspace2_do_outer,
                                                                                 nparts=dout_factor)
        res_cut_do_outer, res_cut_do_inner = sch[res].split(res_do_outer, nparts=dout_factor)
        sch[workspace_res].reorder(workspace_cut_n_outer, workspace_cut_do_outer, workspace_cut_n_inner,
                                   workspace_cut_do_inner)
        sch[workspace_8bit].reorder(workspace_8bit_cut_n_outer, workspace_8bit_cut_do_outer,
                                    workspace_8bit_cut_n_inner, workspace_8bit_cut_do_inner)
        sch[workspace2].reorder(workspace2_cut_n_outer, workspace2_cut_do_outer, workspace2_cut_n_inner,
                                workspace2_cut_do_inner)
        sch[res].reorder(res_cut_n_outer, res_cut_do_outer, res_cut_n_inner, res_cut_do_inner)
        workspace_fused_axis = sch[workspace_res].fuse(workspace_cut_n_outer, workspace_cut_do_outer)
        workspace_8bit_fused_axis = sch[workspace_8bit].fuse(workspace_8bit_cut_n_outer, workspace_8bit_cut_do_outer)
        workspace2_fused_axis = sch[workspace2].fuse(workspace2_cut_n_outer, workspace2_cut_do_outer)
        res_fused_axis = sch[res].fuse(res_cut_n_outer, res_cut_do_outer)

        block_idx = tvm.thread_axis('blockIdx.x')
        sch[workspace_res].bind(workspace_fused_axis, block_idx)
        sch[workspace_8bit].bind(workspace_8bit_fused_axis, block_idx)
        sch[workspace2].bind(workspace2_fused_axis, block_idx)
        sch[res].bind(res_fused_axis, block_idx)

        # for double buffer
        if is_l0_ub_double_buffer:
            sch[fmap_fractal].double_buffer()
            sch[image_patches_res].double_buffer()
            sch[fmap_im2col].double_buffer()
        if is_l1_double_buffer:
            sch[fmap_in_l1].double_buffer()

        if workspace_8bit_double_buffer_flag:
            sch[tensor0].double_buffer()
            sch[tensor00].double_buffer()

        if workspace2_double_buffer_flag:
            sch[tensor10].double_buffer()
            sch[tensor1].double_buffer()
            sch[tensor2].double_buffer()
            if not compute_inline_tensor3_6:
                sch[tensor3].double_buffer()
            if not compute_inline_tensor4_41:
                sch[tensor4].double_buffer()
                sch[tensor41].double_buffer()

        if res_double_buffer_flag:
            sch[tensor42].double_buffer()
            sch[tensor5].double_buffer()
            if not compute_inline_tensor3_6:
                sch[tensor6].double_buffer()
            sch[tensor7].double_buffer()

        # for compute_inline
        if compute_inline_tensor4_41:
            sch[tensor41].compute_inline()
            sch[tensor4].compute_inline()
        if compute_inline_tensor3_6:
            sch[tensor3].compute_inline()
            sch[tensor6].compute_inline()
        if compute_inline_tensor42:
            sch[tensor42].compute_inline()

    else:
        if fmap_batch < Constant.DEVICE_CORE_NUM:
            ub_do = 1
        # three tensors has the same size on UBuf.
        if (ub_do * ub_kd * ub_khkw * ub_howo * blcok_size_align * dtype_size) * 3 > Constant.MAX_UB_SIZE:
            ub_do = max(min(ub_do, (Constant.MAX_UB_SIZE // 3) // \
                                   (ub_kd * ub_khkw * ub_howo * blcok_size_align * dtype_size)), 1)

        if (ub_do * ub_kd * ub_khkw * ub_howo * blcok_size_align * dtype_size) * 3 > Constant.MAX_UB_SIZE:
            ub_kd = max(min(ub_kd, (Constant.MAX_UB_SIZE // 3) // \
                                   (ub_khkw * ub_howo * blcok_size_align * dtype_size)), 1)

        if (ub_do * ub_kd * ub_khkw * ub_howo * blcok_size_align * dtype_size) * 3 > Constant.MAX_UB_SIZE:
            ub_khkw = max(min(ub_khkw, (Constant.MAX_UB_SIZE // 3) // (ub_howo * blcok_size_align * dtype_size)), 1)

        if padding == "SAME":
            l1_di = ub_do * stride_d - stride_d + 1
            if not howo_split and \
                (l1_di * fmap_h * fmap_w * blcok_size_align * dtype_size) > Constant.MAX_L1_SIZE:
                l1_di = max(Constant.MAX_L1_SIZE // (fmap_h * fmap_w * blcok_size_align * dtype_size), 1)
                ub_do = (l1_di - 1 + stride_d) // stride_d
        else:
            l1_di = ub_do * stride_d - stride_d + filter_d
            if not howo_split and \
                    (l1_di * fmap_h * fmap_w * blcok_size_align * dtype_size) > Constant.MAX_L1_SIZE:
                l1_di = max(Constant.MAX_L1_SIZE // (fmap_h * fmap_w * blcok_size_align * dtype_size), 1)
                ub_do = max((l1_di - filter_d + stride_d) // stride_d, 1)

        # cut res
        res_n_outer, res_n_inner = sch[res].split(res.op.axis[0], factor=1)
        res_do_outer, res_do_inner = sch[res].split(res.op.axis[1], factor=ub_do)
        res_kd_outer, res_kd_inner = sch[res].split(res.op.axis[2], factor=ub_kd)
        res_khkw_outer, res_khkw_inner = sch[res].split(res.op.axis[3], factor=ub_khkw)
        res_howo_outer, res_howo_inner = sch[res].split(res.op.axis[5], factor=ub_howo)

        sch[res].reorder(res_n_outer, res_do_outer, res.op.axis[4], res_howo_outer, res_kd_outer, res_khkw_outer,
                         res_n_inner, res_do_inner, res.op.axis[6], res_howo_inner, res_kd_inner, res_khkw_inner)

        # for compute_at
        if howo_split:
            sch[fmap_in_l1].compute_at(sch[res], res_howo_outer)
            sch[fmap_im2col].compute_at(sch[res], res_howo_outer)
        else:
            sch[fmap_in_l1].compute_at(sch[res], res_kd_outer)
            sch[fmap_im2col].compute_at(sch[res], res_kd_outer)

        sch[fmap_fractal].compute_at(sch[res], res_khkw_outer)
        sch[image_patches_res].compute_at(sch[res], res_khkw_outer)
        sch[ub_res].compute_at(sch[res], res_khkw_outer)
        sch[trans_format0].compute_at(sch[res], res_khkw_outer)

        # for emit_insn
        sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], tbe_platform.DMA_COPY)
        sch[fmap_im2col].emit_insn(fmap_im2col_d_inner, 'set_fmatrix', setfmatrix_dict)
        sch[fmap_fractal].emit_insn(fmap_fractal_d_inner, tbe_platform.IM2COL)
        if dtype_input == "float16":
            sch[image_patches_res].emit_insn(image_patches_res.op.axis[0], tbe_platform.ADDVS)
            sch[ub_res].emit_insn(ub_res.op.axis[0], tbe_platform.ADDVS)
            sch[trans_format0].emit_insn(trans_format0.op.axis[0], tbe_platform.ADDVS)
        else:
            sch[image_patches_res].emit_insn(image_patches_res.op.axis[0], tbe_platform.DMA_COPY)
            sch[ub_res].emit_insn(ub_res.op.axis[0], tbe_platform.DMA_COPY)
            sch[trans_format0].emit_insn(trans_format0.op.axis[0], tbe_platform.DMA_COPY)
        sch[res].emit_insn(res_n_inner, tbe_platform.DMA_COPY)

        # for double buffer
        if is_l0_ub_double_buffer:
            sch[fmap_fractal].double_buffer()
            sch[image_patches_res].double_buffer()
            sch[fmap_im2col].double_buffer()
            sch[trans_format0].double_buffer()
        if is_l1_double_buffer:
            sch[fmap_in_l1].double_buffer()

        # for compute_inline
        sch[workspace_res].compute_inline()
        sch[ub_res].compute_inline()

        # for multi cores
        res_shape = [int(i.value) for i in res.shape]
        res_n, res_do, _, _, _, _, _ = res_shape
        res_c1 = fmap_c1
        res_n_factor, res_do_factor, res_c1_factor = _cal_multi_core_factor_3_axis(res_n, res_do, res_c1)
        res_cut_n_outer, res_cut_n_inner = sch[res].split(res_n_outer, nparts=res_n_factor)
        res_cut_do_outer, res_cut_do_inner = sch[res].split(res_do_outer, nparts=res_do_factor)
        res_cut_c1_outer, res_cut_c1_inner = sch[res].split(res.op.axis[4], nparts=res_c1_factor)
        sch[res].reorder(res_cut_n_outer, res_cut_do_outer, res_cut_c1_outer, res_cut_n_inner, res_cut_do_inner,
                         res_cut_c1_inner)
        res_fused_axis = sch[res].fuse(res_cut_n_outer, res_cut_do_outer, res_cut_c1_outer)
        block_idx = tvm.thread_axis('blockIdx.x')
        sch[res].bind(res_fused_axis, block_idx)


# 'pylint: disable=too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_STR, para_check.KERNEL_NAME)
def extract_volume_patches(input_x, output_y, ksizes, strides, padding, kernel_name="extract_volume_patches"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same type as input
    ksizes: input attr
    strides: input attr
    padding: input attr
    kernel_name : str
        kernel name, default value is "extract_volume_patches"

    Returns
    -------
    None
    """
    ori_shape_5d, ksizes, strides = \
        _check_shape_and_format_vailded(input_x, output_y, ksizes, strides, padding, kernel_name)
    fmap_n, fmap_d, fmap_h, fmap_w, fmap_c = ori_shape_5d
    dtype = input_x.get("dtype")
    input_dtype = dtype.lower()
    blcok_size_align = Constant.BLOCK_SIZE_FP16 if input_dtype == "float16" else Constant.BLOCK_SIZE_INT8
    shape = (fmap_n, fmap_d, (fmap_c + blcok_size_align - 1) // blcok_size_align, fmap_h, fmap_w, blcok_size_align)

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    res = list(extract_volume_patches_compute(data_input, fmap_c, ksizes, strides, padding))

    sch = tvm.create_schedule(res[0].op)

    _extract_volume_patches_schedule(res[0], [sch], fmap_c)

    def _write_workspace_info(workspace_list, kernel_name):
        """
        write workspace information
        """
        def shape_to_list(shape):
            """
            translate tvm.shape to list type in python
            """
            tmp = []
            for i in shape:
                tmp.append(i.value)
            return tmp

        def get_data_width(dtype):
            """
            get data_width
            """
            m = re.search(r'\d+', dtype)
            if m:
                return int(m.group(0)) // 8
            return 0

        num = len(workspace_list)
        if num:
            shape_list = [shape_to_list(i.shape) for i in workspace_list]
            total_size = [functools.reduce(lambda x, y: x * y, list_i) for list_i in shape_list]

            total_size = [i * get_data_width(j.dtype) for i, j in zip(total_size, workspace_list)]
            if not os.path.exists("kernel_meta"):
                os.mkdir("kernel_meta")
                os.chmod("kernel_meta", stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP)
            wkspace_dict = {"workspace": {"num": num, "size": total_size}}
            write_code(wkspace_dict, kernel_name)

    with tbe_build.build_config():
        build_list = [data_input] + res
        tvm.build(sch, build_list, "cce", name=kernel_name)
        if len(res) > 1:
            _write_workspace_info(res[1:], kernel_name)
