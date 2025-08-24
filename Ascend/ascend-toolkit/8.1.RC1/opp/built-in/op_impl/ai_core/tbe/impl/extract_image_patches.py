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
import functools
import math
import os
import re
import stat

import te.platform as tbe_platform
from tbe import tvm
from te.lang import cce as tbe
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl.im2col_common_func import im2col_compute
from impl.im2col_common_func import im2col_schedule
from impl.util.util_common import write_code
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_common import check_load3d_w_out_1_support
from impl.util.platform_adapter import build_config


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    BLOCK_SIZE = 16
    BLOCK_SIZE_INT8 = 32
    DOUBLE_BUFFER = 2
    FP16_SIZE = 2
    INT8_SIZE = 1
    SIZE_L1 = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)


# 'pylint: disable=unused-argument, too-many-arguments
def get_op_support_info(images, y, ksizes, strides, dilates, padding, kernel_name="extract_image_patches"):
    """
    get extract_image_patches slice info
    """
    format_x = images.get("format")
    images_shape = images.get("shape")
    data_format = images.get("ori_format")
    kernel_h = ksizes[1]
    if data_format == "NCHW":
        kernel_h = ksizes[2]
    if format_x == "NC1HWC0":
        images_h = images_shape[2]
        if images_h == kernel_h or padding == "SAME":
            axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])]]
        elif padding == "VALID":
            axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]),
                                  SplitOutput([0, [0]])], [SplitInput([0, [2], [0], [0]]),
                                                           SplitOutput([0, [1]])]]
        else:
            axis_split_matrix = None
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)

    return op_cal_info_in_json


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments
@tbe_platform.fusion_manager.fusion_manager.register("extract_image_patches")
def extract_image_patches_compute(fmap,
                                  c_in_real,
                                  ksizes,
                                  strides,
                                  dilates,
                                  padding,
                                  kernel_name="extract_image_patches"):
    """
    ops compute

    Parameters
    ----------
    fmap : TVM tensor
        the placeholder of input_x
    c_in_real : real c size of input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    padding: input attr
    kernel_name : str
    kernel name, default value is "extract_image_patches"

    Returns
    -------
    compute results
    """
    # fmap's format is NC1HWC0
    fmap_shape = fmap.shape

    fmap_h = fmap_shape[2].value
    fmap_w = fmap_shape[3].value

    _, kernel_h, kernel_w, _ = ksizes
    _, stride_h, stride_w, _ = strides
    _, dilate_h, dilate_w, _ = dilates

    out_h, padding_h_before, padding_h_after = tbe.te_compute.common.tf_get_windowed_output_size_verbose_v2(
        fmap_h, kernel_h, dilate_h, stride_h, padding)
    out_w, padding_w_before, padding_w_after = tbe.te_compute.common.tf_get_windowed_output_size_verbose_v2(
        fmap_w, kernel_w, dilate_w, stride_w, padding)

    pads = (padding_h_before, padding_h_after, padding_w_before, padding_w_after)
    ksize = (kernel_h, kernel_w)
    stride = (stride_h, stride_w)
    dilate = (dilate_h, dilate_w)

    output_res, workspace_res, workspace_shape = im2col_compute(fmap, c_in_real, ksize, stride, dilate, pads, out_h,
                                                                out_w)

    return output_res, workspace_res, workspace_shape


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,too-many-statements,too-many-locals
# 'pylint: disable=too-many-branches
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_STR, para_check.KERNEL_NAME)
def extract_image_patches(images, y, ksizes, strides, dilates, padding, kernel_name="extract_image_patches"):
    """
    calculating data

    Parameters
    ----------
    images : dict
        shape and dtype of input, only support float16
    y : dict
        shape and dtype of output, should be same shape and type as input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    padding: input attr
    kernel_name : str
        kernel name, default value is "extract_image_patches"

    Returns
    -------
    None
    """
    shape_input_4d = images.get("ori_shape")
    dtype_input = images.get("dtype")
    dtype_input = dtype_input.lower()
    if dtype_input in ('int8', 'uint8'):
        align_block_size = Constant.BLOCK_SIZE_INT8
        type_size = Constant.INT8_SIZE
    else:
        align_block_size = Constant.BLOCK_SIZE
        type_size = Constant.FP16_SIZE

    data_format = images.get('ori_format')
    format_list = ('NHWC', 'NCHW')
    if data_format not in format_list:
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", format_list, data_format)
    if len(ksizes) != 4 or len(strides) != 4 or len(dilates) != 4:
        error_manager_vector.raise_err_check_params_rules(kernel_name, 'input params invalide',
                                                          ['ksizes', 'strides', 'dilates'], [ksizes, strides, dilates])
    # NCHW -> NHWC
    if data_format == 'NCHW':
        shape_input_4d = (shape_input_4d[0], shape_input_4d[2], shape_input_4d[3], shape_input_4d[1])
        ksizes = (ksizes[0], ksizes[2], ksizes[3], ksizes[1])
        strides = (strides[0], strides[2], strides[3], strides[1])
        dilates = (dilates[0], dilates[2], dilates[3], dilates[1])
    fmap_n, fmap_h, fmap_w, fmap_c = shape_input_4d
    fmap_c1 = (fmap_c + align_block_size - 1) // align_block_size
    fmap_c0 = align_block_size
    shape_input = (fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0)

    _, kernel_h, kernel_w, _ = ksizes
    _, stride_h, stride_w, _ = strides
    _, dilate_h, dilate_w, _ = dilates

    if kernel_h >= 256 or kernel_w >= 256:
        error_manager_vector.raise_err_specific_reson(kernel_name, "kernel_h and kernel_w can not >= 256!")
    if stride_h >= 64 or stride_w >= 64:
        error_manager_vector.raise_err_specific_reson(kernel_name, "stride_h and stride_w can not >= 64!")
    if dilate_h >= 256 or dilate_w >= 256:
        error_manager_vector.raise_err_specific_reson(kernel_name, "dilate_h and dilate_w can not >= 256!")

    out_h, padding_h_top, padding_h_bottom = \
        tbe.te_compute.common.tf_get_windowed_output_size_verbose_v2(fmap_h, kernel_h, dilate_h, stride_h, padding)
    out_w, padding_w_before, padding_w_after = tbe.te_compute.common.tf_get_windowed_output_size_verbose_v2(
        fmap_w, kernel_w, dilate_w, stride_w, padding)
    if out_h <= 0 or out_w <= 0:
        error_manager_vector.raise_err_specific_reson(kernel_name, "out_h and out_w can not <= 0!")
    if padding_h_top >= 256 or padding_h_bottom >= 256:
        error_manager_vector.raise_err_specific_reson(kernel_name,
                                                      "padding_h_top and padding_h_bottom can not >= 256!")
    if padding_w_before >= 256 or padding_w_after >= 256:
        error_manager_vector.raise_err_specific_reson(kernel_name,
                                                      "padding_w_before and padding_w_after can not >= 256!")

    if not check_load3d_w_out_1_support() and (out_h != 1 and out_w == 1):
        if fmap_w + padding_w_before + padding_w_after - ((kernel_w - 1) * dilate_w + 1) < stride_w:
            error_manager_vector.raise_err_specific_reson(
                kernel_name, "Platform cloud and DC DO NOT support these invalid params,"
                " it must be fmap_w + pad_l + pad_r - ((kernel_w - 1) * dilate_w + 1) >= stride_w")

    # min cut_h
    dilated_kernel_h = (kernel_h - 1) * dilate_h + 1
    cut_h_col = (Constant.BLOCK_SIZE // math.gcd(out_w, Constant.BLOCK_SIZE) - 1) * \
                stride_h + 1 + dilated_kernel_h // 2
    if cut_h_col > fmap_h:
        cut_h_col = fmap_h

    cut_w_row_s = (Constant.BLOCK_SIZE - 1) * stride_w + 1
    cut_h_row_s = ((cut_w_row_s - 1) // fmap_w + 1) * stride_h + 1
    min_cut_h = min(cut_h_col, cut_h_row_s)

    if min_cut_h * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER > Constant.SIZE_L1:
        error_manager_vector.raise_err_specific_reson(
            kernel_name, "Input size is too large load to L1, while cut h, need size: %d" %
            (min_cut_h * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER))

    data_input = tvm.placeholder(shape_input, name="data", dtype=dtype_input)
    output_res, workspace_res, _ = extract_image_patches_compute(data_input, fmap_c, ksizes, strides, dilates, padding,
                                                                 kernel_name)
    sch = tvm.create_schedule(output_res.op)
    im2col_schedule(output_res, [sch])

    def _write_workspace_info(workspace_list, kernel_name):
        def _shape_to_list(shape):
            """
            translate tvm.shape to list type in python
            """
            tmp = []
            for i in shape:
                tmp.append(i.value)
            return tmp

        def _get_data_width(dtype):
            m = re.search(r'\d+', dtype)
            if m:
                return int(m.group(0)) // 8
            return 0

        num = len(workspace_list)
        if num:
            shape_list = [_shape_to_list(i.shape) for i in workspace_list]
            total_size = [functools.reduce(lambda x, y: x * y, list_i) for list_i in shape_list]

            total_size = [i * _get_data_width(j.dtype) for i, j in zip(total_size, workspace_list)]
            if not os.path.exists("kernel_meta"):
                os.mkdir("kernel_meta")
                os.chmod("kernel_meta", stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP)
            wkspace_dict = {"workspace": {"num": num, "size": total_size}}
            write_code(wkspace_dict, kernel_name)


    if fmap_c % align_block_size == 0:
        tensor_list = [data_input, output_res]
        with build_config(dummy_placeholder=True):
            tvm.build(sch, tensor_list, "cce", name=kernel_name)
    else:
        tensor_list = [data_input, output_res, workspace_res]
        with build_config(dummy_placeholder=True):
            tvm.build(sch, tensor_list, "cce", name=kernel_name)
            _write_workspace_info([workspace_res], kernel_name)
