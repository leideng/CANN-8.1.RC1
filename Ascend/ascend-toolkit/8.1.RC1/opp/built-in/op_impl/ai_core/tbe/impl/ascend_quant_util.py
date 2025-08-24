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
ascend_quant_util
"""
import functools
from impl.util.platform_adapter import tvm
from impl.util import util_select_op_base
from impl.util import util_common
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
import tbe.common.platform.platform_info as platform_info
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import check_support_block_size_16


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant.
    """
    FP16_BLOCK_VALUE = 16
    DTYPE_2_STR_MAP = {
        2: "int8",
        29: "int4",
    }
    C1_TRANS_MAP = {
        "int8": 2,
        "int4": 4
    }
    func_map = {
        "vdeq_cast": tvm.vdeq_cast,
        "deq_cast": tvm.deq_cast
    }


def is_matmul_fuse(tensor):
    """
    check is matmul fuse
    """
    if not isinstance(tensor, tvm.Tensor):
        return False
    is_matmul_fuse_flag = False
    stack = [tensor]
    visited_list = []
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
                if hasattr(in_tensor.op, "tag") and "matmul" in in_tensor.op.tag:
                    is_matmul_fuse_flag = True
                    break
    return is_matmul_fuse_flag


def is_nz_format(tensor, is_quant=False):
    """
    check is nz format
    """
    if "matmul" in tensor.op.tag or "gemm" in tensor.op.tag:
        return True

    tensor_format = "NC1HWC0"
    if tensor.op.attrs:
        if "format" in tensor.op.attrs:
            tensor_format = tensor.op.attrs["format"]
        if "format_" in tensor.op.attrs:
            tensor_format = tensor.op.attrs["format_"]

    if tensor_format == "FRACTAL_NZ":
        return True

    if is_quant:
        if is_matmul_fuse(tensor):
            return True

    return False


# 'pylint: disable = unused-argument
def get_quant_support_info(x, x1=None, dual_output=False, l1_fusion_enable=0):
    """
    obtains the split information of the quantization operator
    """
    dim_x = len(x.get("shape"))
    format_x = x.get("format")

    # C1 C0  can not split
    not_cut_dim = [1, 4]
    if format_x == "FRACTAL_NZ":
        not_cut_dim = [dim_x - 4, dim_x - 1]

    if format_x in ["NC1HWC0", "FRACTAL_NZ"]:
        axis_split_list = []
        for i in range(dim_x):
            if i not in not_cut_dim:
                if x1 is not None:
                    split_in = util_select_op_base.SplitInput([0, [i], [-1], [-1]],
                                                              [2, [i], [-1], [-1]])
                else:
                    split_in = util_select_op_base.SplitInput([0, [i], [-1], [-1]])
                if dual_output:
                    split_out = util_select_op_base.SplitOutput([0, [i]],
                                                                [1, [i]])
                else:
                    split_out = util_select_op_base.SplitOutput([0, [i]])
                axis_split_list.append([split_in, split_out])
    else:
        axis_split_list = None

    axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_list, axis_reduce_list, l1_fusion_enable)
    return op_cal_info_in_json


def get_scale_indices(scale_tensor, tensor_flag, c0, c1):
    """
    scale format only supported nc1hwc0 and ndc1hwc0,
    generate indices from the dimension information of the shape
    """
    scale_shape = scale_tensor.shape
    is_5hd = True if len(scale_shape) == 5 else False
    new_indice = [0] * 5 if is_5hd else [0] * 6
    c0_index_new = 4 if is_5hd else 5
    c1_index_new = 1 if is_5hd else 2
    if tensor_flag:
        new_indice[c0_index_new] = c0
        new_indice[c1_index_new] = c1
    return new_indice


def get_depthwise_conv2d_tensor_info(x, is_dequant=False):
    """
    get depthwise conv2d tensor info
    """
    tensor_dict = dict()
    tensor_dict["mad_ubuf"] = x.op.input_tensors[0]
    if 'bias_flag' in x.op.attrs and x.op.attrs['bias_flag'].value == 1:
        tensor_dict["mad_after_bias"] = tensor_dict.get("mad_ubuf").op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict.get("mad_after_bias").op.input_tensors[1]
        if is_dequant:
            tensor_dict["flag_is_dequant_bias"] = True
            tensor_dict["mad_bias"] = tensor_dict.get("mad_after_bias").op.input_tensors[0]
            tensor_dict["mad_bias_ub_brc"] = tensor_dict.get("mad_bias").op.input_tensors[0]
            tensor_dict["bias_gm"] = tensor_dict.get("mad_bias_ub_brc").op.input_tensors[0]
    else:
        tensor_dict["mad"] = tensor_dict.get("mad_ubuf").op.input_tensors[0]

    tensor_dict["im2col_fractal"] = tensor_dict.get("mad").op.input_tensors[0]
    tensor_dict["im2col_row_major"] = tensor_dict.get("im2col_fractal").op.input_tensors[0]
    if is_dequant:
        tensor_dict["filter_reshape"] = tensor_dict.get("mad").op.input_tensors[1]
        tensor_dict["filter_buf"] = tensor_dict.get("filter_reshape").op.input_tensors[0]

    tensor_dict["temp"] = tensor_dict.get("im2col_row_major").op.input_tensors[0]
    if tensor_dict.get("temp").op.input_tensors:
        tensor_dict["fmap"] = tensor_dict.get("temp").op.input_tensors[0]
    else:
        tensor_dict["fmap"] = tensor_dict.get("im2col_row_major").op.input_tensors[0]
    return tensor_dict


def get_conv_flag(x):
    """
    Check if conv fusion
    """
    conv_flag = 0
    if (len(x.op.input_tensors) > 0) and ('mad1' in x.op.input_tensors[0].name or
                                          'convolution_c_col_bias' in x.op.input_tensors[0].name or
                                          'conv3d_c_col' in x.op.input_tensors[0].name):
        conv_flag = 1
    return conv_flag


def is_conv3d_fuse(x):
    """
    Check if conv3d fusion
    """
    return len(x.op.input_tensors) and ('conv3d_c_col' in x.op.input_tensors[0].name)


def is_support_a100(is_quant=False):
    """
    Check if a100 version.

    Returns
    -------
    True: a100 version.
    False: other version.
    """
    if is_quant:
        if tbe_platform.api_check_support("tik.vcopy") and not platform_info.intrinsic_check_support(
                "Intrinsic_fix_pipe_unit_list", "post_eltwise"):
            return True
        return False

    if tbe_platform.api_check_support("tik.vcopy"):
        return True
    return False


def is_lhisi_version():
    """
    check is Lhisi version
    """
    soc_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if soc_version in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        return True
    return False


def is_support_v200():
    """
    check is v200 version
    """
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in (
            "Ascend310P", "Ascend610", "BS9SX1A", "Hi3796CV300CS",
            "SD3403") or tbe_platform.api_check_support("tik.vcopy"):
        return True
    return False


def is_nano_version():
    """
    check is nano version
    """
    nano_flag = check_support_block_size_16()
    return True if nano_flag else False


def get_antiquant_output_shape(in_shape, nz_format_flag, c1_transform):
    """
    get output shape info
    """
    c0_index = len(in_shape) - 1
    c1_index = 1
    if nz_format_flag:
        c1_index = len(in_shape) - 4
    out_shape = in_shape[:]

    for dim, _ in enumerate(in_shape):
        if dim == c0_index:
            out_shape[dim] = in_shape[dim] // c1_transform
        if dim == c1_index:
            out_shape[dim] = in_shape[dim] * c1_transform
    return out_shape
