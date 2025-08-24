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
cube util.
"""
import warnings
from tbe.dsl.compute.cube_util import shape_to_list as shape_to_list_new
from tbe.dsl.compute.cube_util import check_pad_zero as check_pad_zero_new
from tbe.dsl.compute.cube_util import ceil_div as ceil_div_new
from tbe.dsl.compute.cube_util import raise_cube_util_err as raise_cube_util_err_new
from tbe.dsl.compute.cube_util import im2col_row_major as im2col_row_major_new
from tbe.dsl.compute.cube_util import im2col_fractal as im2col_fractal_new
from tbe.dsl.compute.cube_util import im2col_fractal_3d as im2col_fractal_3d_new
from tbe.dsl.compute.cube_util import im2col_fractal_v2 as im2col_fractal_v2_new
from tbe.dsl.compute.cube_util import is_support_v200 as is_support_v200_new
from tbe.dsl.compute.cube_util import calc_info_of_iter_vars as calc_info_of_iter_vars_new
from tbe.dsl.compute.cube_util import print_iter_vars as print_iter_vars_new


class GroupDictKeys:
    """
    The keys of group_dict
    """
    groups = "groups"
    g_extend = "g_extend"
    multiple_extend = "multiple_extend"
    dx_c1_extend = "dx_c1_extend"
    dy_c1_extend = "dy_c1_extend"
    dx_c_ori = "dx_c_ori"
    dy_c_ori = "dy_c_ori"
    filter_batch_ori = "filter_batch_ori"
    filter_c_ori = "filter_c_ori"
    filter_ori_format = "filter_ori_format"


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    warnings.warn("te.lang.cce.te_compute.cube_util is expired, " \
        "please replace it with the func tbe.dsl.compute.cube_util",
        DeprecationWarning)
    return shape_to_list_new(shape)


def check_pad_zero(pads):
    """
    check if pad is [0, x, 0, x]
    """
    warnings.warn("te.lang.cce.te_compute.cube_util is expired, " \
        "please replace it with the func tbe.dsl.compute.cube_util",
        DeprecationWarning)
    return check_pad_zero_new(pads)


def ceil_div(num1, num2):
    """
    ceil div
    """
    warnings.warn("te.lang.cce.te_compute.cube_util is expired, " \
        "please replace it with the func tbe.dsl.compute.cube_util",
        DeprecationWarning)
    return ceil_div_new(num1, num2)


def raise_cube_util_err(msg):
    """
    In common component: cube_util, [%s] % (msg)
    msg for discribe the error info
    the error info only for cube_util's developers
    """
    warnings.warn("te.lang.cce.te_compute.cube_util is expired, " \
        "please replace it with the func tbe.dsl.compute.cube_util",
        DeprecationWarning)
    return raise_cube_util_err_new(msg)


def im2col_row_major(
        a_im2col_vm_shape,
        tensor_a,
        kernel_w,
        padding,
        stride,
        compute_dtype,
        opti_h_flag=False,
        tag="",
        dilation=(1, 1),
        offset_x=0,
        slice_offset=0):
    """
    calculate im2col_row_major tensor
    Parameters
    ----------
    a_im2col_vm_shape : shape of a_im2col_row_major

    tensor_a: feature map

    kernel_w: width of filter

    padding: the padding shape

    stride: the stride value

    dilation: the dilation value

    compute_dtype: dtype of compute result

    offset_x: offset of x
    -------
    Returns : a_im2col_row_major tensor
    """
    warnings.warn("te.lang.cce.te_compute.cube_util is expired, " \
        "please replace it with the func tbe.dsl.compute.cube_util",
        DeprecationWarning)
    return im2col_row_major_new(a_im2col_vm_shape, tensor_a, kernel_w, padding, stride, compute_dtype,
                                opti_h_flag, tag, dilation, offset_x, slice_offset)


def im2col_fractal(a_im2col_shape, tensor_a_row_major):
    """
    calculate im2col_fractal tensor
    Parameters
    ----------
    a_im2col_shape: shape of a_im2col

    tensor_a_row_major: feature map after row major

    config: the config of cube

    compute_dtype: dtype of compute result
    -------
    Returns : a_im2col_fractal tensor
    """
    warnings.warn("te.lang.cce.te_compute.cube_util is expired, " \
        "please replace it with the func tbe.dsl.compute.cube_util",
        DeprecationWarning)
    return im2col_fractal_new(a_im2col_shape, tensor_a_row_major)


def im2col_fractal_3d(
        a_im2col_shape,
        tensor_a_row_major,
        fmap_c1,
        d_out,
        filter_d,
        stride_d,
        cin1_g,
        cyclebuffer_flag,
        tag=""):
    """
    calculate 3d im2col_fractal tensor
    Parameters
    ----------
    a_im2col_shape : shape of a_im2col

    tensor_a_row_major : feature map after row major

    fmap_c1 : channel c1

    d_out : output d

    filter_d : kernel_d

    strided : stride d

    cyclebuffer_flag : whether to do cyclebuffer
    -------
    Returns : a_im2col_fractal tensor
    """
    warnings.warn("te.lang.cce.te_compute.cube_util is expired, " \
        "please replace it with the func tbe.dsl.compute.cube_util",
        DeprecationWarning)
    return im2col_fractal_3d_new(a_im2col_shape, tensor_a_row_major, fmap_c1, d_out, filter_d, stride_d, cin1_g,
                                 cyclebuffer_flag, tag)


def im2col_fractal_v2(shape, img2col_para):
    """
    calculate im2col_fractal tensor without tensor row_major
    Parameters
    ----------
    shape : shape of a_im2col

    img2col_para : tensor of fmap, kernel_h, kernel_w, padding, stride, fmap_wo, dilation
    -------
    Returns : a_im2col_fractal tensor
    """
    warnings.warn("te.lang.cce.te_compute.cube_util is expired, " \
        "please replace it with the func tbe.dsl.compute.cube_util",
        DeprecationWarning)
    return im2col_fractal_v2_new(shape, img2col_para)


def is_support_v200():
    """
    check if Ascend610/BS9SX1A/Ascend310P/Hi3796CV300CS version
    ----------

    Returns
    -------
    True:  Ascend610/BS9SX1A/Ascend310P/Hi3796CV300CS version
    False: Other version
    """
    warnings.warn("te.lang.cce.te_compute.cube_util is expired, " \
        "please replace it with the func tbe.dsl.compute.cube_util",
        DeprecationWarning)
    return is_support_v200_new()


def calc_info_of_iter_vars(stage):
    """
    Calcuate information of IterVar.

    Args: stage: Stage of schedule.

    Returns:
    A list of elements that are combinations of IterVar.var and information.
    For example:[[i0.inner, IterVar(min=0, extent=3,parent=Parent(var=i0, min=0, extent=6, factor=2, nparts=-1))],
    [i0.outer, IterVar(min=0,extent=2,parent=Parent(var=i0, min=0, extent=6, factor=2, nparts=-1))],[i1, (0, 16)]]
    """
    warnings.warn("te.lang.cce.te_compute.cube_util is expired, " \
        "please replace it with the func tbe.dsl.compute.cube_util",
        DeprecationWarning)
    return calc_info_of_iter_vars_new(stage)


def print_iter_vars(iter_vars):
    """
    Pretty print iter_vars.

    Args: iter_vars: List of iter_var.

    Returns:None.
    """
    warnings.warn("te.lang.cce.te_compute.cube_util is expired, " \
        "please replace it with the func tbe.dsl.compute.cube_util",
        DeprecationWarning)
    return print_iter_vars_new(iter_vars)
