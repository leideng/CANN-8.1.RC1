#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
util_tensor_dict.py
"""
import functools
from impl.util import util_common
from impl.util.platform_adapter import shape_util


# 'pylint: disable=too-few-public-methods
class FormatConstant:
    """FormatConstant
    """
    FORMAT_5HD = "NC1HWC0"
    FORMAT_6HD = "NDC1HWC0"
    FORMAT_FZ = "FRACTAL_Z"
    FORMAT_FZ_3D = "FRACTAL_Z_3D"
    FORMAT_NZ = "FRACTAL_NZ"
    SPECIAL_FORMAT = (FORMAT_5HD, FORMAT_6HD, FORMAT_FZ, FORMAT_FZ_3D, FORMAT_NZ)
    FORMAT_ND = "ND"
    FORMAT_NCHW = "NCHW"
    FORMAT_NHWC = "NHWC"


# 'pylint: disable=too-many-instance-attributes,unused-argument
class TensorClass:
    """
    Class: class for Tensor Dict
    """
    # define the unkown dim size
    UNKOWN_DIM = -1
    # define the unkown rank dim size
    UNKOWNRANK_DIM = -2
    # define the scalar shape
    SCALAR_SHAPE = [1]
    # define the default align dim size
    SPECIAL_C0_DIM_SIZE = 16

    def __init__(self, input_dict, is_dividend_input=False):
        self.format = input_dict.get("format")
        self.ori_format = input_dict.get("ori_format")
        self.shape = shape_util.scalar2tensor_one(list(input_dict.get("shape")))
        self.ori_shape = shape_util.scalar2tensor_one(list(input_dict.get("ori_shape")))
        self.dtype = input_dict.get("dtype")
        self.is_dividend_input = is_dividend_input

        # special parmas
        self.is_scalar = self.ori_shape == TensorClass.SCALAR_SHAPE
        support_ori_format = util_common.get_fused_format_str(["N", "H", "W", "C"])
        self.trans_5hd = self.ori_format in support_ori_format and len(self.ori_format) == len(self.ori_shape)
        self.trans_fz = self.trans_5hd
        support_ori_format = util_common.get_fused_format_str(["N", "H", "W", "C", "D"])
        self.trans_6hd = self.ori_format in support_ori_format and len(self.ori_format) == len(self.ori_shape)
        self.trans_fz_3d = self.trans_6hd
        self.trans_nz = len(self.ori_shape) >= 2
        self.ori_n = None
        self.ori_d = None
        self.ori_c = None
        self.ori_h = None
        self.ori_w = None
        self.ori_last_one_dim = None
        self.ori_last_two_dim = None
        self.get_shape_info()
        self.is_dy_rank = TensorClass.UNKOWNRANK_DIM in self.ori_shape
        self.is_dy_shape = TensorClass.UNKOWN_DIM in self.ori_shape or self.is_dy_rank
        self.is_static_shape = not self.is_dy_shape
        self.shape_count = -1 if self.is_dy_shape else functools.reduce(lambda x, y: x * y, self.ori_shape)

    def get_shape_info(self):
        """get_shape_info
        """
        if self.trans_5hd or self.trans_6hd:
            self.ori_n, self.ori_c, self.ori_h, self.ori_w = [
                self.ori_shape[self.ori_format.index("N")], self.ori_shape[self.ori_format.index("C")],
                self.ori_shape[self.ori_format.index("H")], self.ori_shape[self.ori_format.index("W")]
            ]

        if self.trans_6hd:
            self.ori_d = self.ori_shape[self.ori_format.index("D")]

        if self.trans_nz:
            self.ori_last_one_dim = self.ori_shape[-1]
            self.ori_last_two_dim = self.ori_shape[-2]

    def is_support_5hd(self, need_align=False, align_len=SPECIAL_C0_DIM_SIZE):
        """charge whether support 5HD

        condition:
            one: ori_format can transdata to 5HD
            two: ori_c must be aligned when need_align is True
        """
        if self.is_dividend_input:
            need_align = True
        if self.trans_5hd and need_align:
            return self.ori_c % align_len == 0

        return self.trans_5hd

    def is_support_6hd(self, need_align=False, align_len=SPECIAL_C0_DIM_SIZE):
        """charge whether support 6HD

        condition:
            one: ori_format can transdata to 6HD
            two: ori_c must be aligned when need_align is True
        """
        if self.is_dividend_input:
            need_align = True
        if self.trans_6hd and need_align:
            return self.ori_c % align_len == 0

        return self.trans_6hd

    def is_support_nz(self, need_align=False, align_len=SPECIAL_C0_DIM_SIZE):
        """charge whether support NZ

        condition:
            one: ori_format can transdata to NZ
            two: -1, -2 dim size must be aligned when need_align is True
        """
        if self.is_dividend_input:
            need_align = True
        if self.trans_nz and need_align:
            return self.ori_last_one_dim % align_len == 0 and self.ori_last_two_dim % align_len == 0

        return self.trans_nz

    def is_support_fz(self, need_align=False, align_len=SPECIAL_C0_DIM_SIZE):
        """charge whether support FRACTAL_Z

        condition:
            one: ori_format can transdata to FRACTAL_Z
            two: N/C dim size must be aligned when need_align is True
        """
        if self.is_dividend_input:
            need_align = True
        if self.trans_fz and need_align:
            return self.ori_n % align_len == 0 and self.ori_c % align_len == 0

        return self.trans_fz

    def is_support_fz3d(self, need_align=False, align_len=SPECIAL_C0_DIM_SIZE):
        """charge whether support FRACTAL_Z_3D

        condition:
            one: ori_format can transdata to FRACTAL_Z_3D
            two: N/C dim size must be aligned when need_align is True
        """
        if self.is_dividend_input:
            need_align = True
        if self.trans_fz_3d and need_align:
            return self.ori_n % align_len == 0 and self.ori_c % align_len == 0

        return self.trans_fz_3d


def get_format_for_format_ignore(tensor_cls_a, need_align=False):
    """get_format_for_broardcast, do format_ignore op_select
    """
    format_a_list = []
    if need_align or tensor_cls_a.is_dividend_input:
        if tensor_cls_a.is_support_nz(need_align):
            format_a_list += [FormatConstant.FORMAT_NZ]
        if tensor_cls_a.is_support_5hd(need_align):
            format_a_list += [FormatConstant.FORMAT_5HD]
        if tensor_cls_a.is_support_6hd(need_align):
            format_a_list += [FormatConstant.FORMAT_6HD]
        if tensor_cls_a.is_support_fz(need_align):
            format_a_list += [FormatConstant.FORMAT_FZ]
        if tensor_cls_a.is_support_fz3d(need_align):
            format_a_list += [FormatConstant.FORMAT_FZ_3D]
    else:
        format_a_list += FormatConstant.SPECIAL_FORMAT
    return format_a_list


def is_all_supported(func_name, input_tensors, need_align_with_broadcast):
    """is_all_supported, check whether all the tensor is support use func func_name
    """
    if not input_tensors:
        return True
    supported_list = [func_name(tensor, need_align_with_broadcast) for tensor in input_tensors]
    supported = supported_list[0] and supported_list.count(supported_list[0]) == len(supported_list)

    return supported


def is_all_dim_equal(input_dim_list, ignore_dy_dim=False):
    """is_all_dim_equal, check whether all the dim is equal
    """
    if not input_dim_list:
        return True
    is_list_same = input_dim_list.count(input_dim_list[0]) == len(input_dim_list)
    if ignore_dy_dim:
        return is_list_same

    is_dy_dim = input_dim_list[0] == TensorClass.UNKOWN_DIM or input_dim_list[0] == TensorClass.UNKOWNRANK_DIM
    return is_list_same and not is_dy_dim


def is_broadcast_for_5hd(input_tensors, need_align_with_broadcast, need_check_other_shape=True):
    """is_broadcast_for_5hd, check whether all input_tensor support 5hd
    """
    check_func_name = TensorClass.is_support_5hd
    ori_c_dim_list = [tensor_cls.ori_c for tensor_cls in input_tensors]

    is_all_5hd_supported = is_all_supported(check_func_name, input_tensors, need_align_with_broadcast)
    is_all_c_equal = is_all_dim_equal(ori_c_dim_list)

    return is_all_5hd_supported and is_all_c_equal


def is_broadcast_for_6hd(input_tensors, need_align_with_broadcast, need_check_other_shape=True):
    """is_broadcast_for_6hd, check whether all input_tensor support 6hd
    """
    check_func_name = TensorClass.is_support_6hd
    ori_c_dim_list = [tensor_cls.ori_c for tensor_cls in input_tensors]

    is_all_6hd_supported = is_all_supported(check_func_name, input_tensors, need_align_with_broadcast)
    is_all_c_equal = is_all_dim_equal(ori_c_dim_list)

    return is_all_6hd_supported and is_all_c_equal


def is_broadcast_for_nz(input_tensors, need_align_with_broadcast, need_check_other_shape=True):
    """is_broadcast_for_nz, check whether all input_tensor support nz
    """
    check_func_name = TensorClass.is_support_nz
    ori_last_one_dim_list = [tensor_cls.ori_last_one_dim for tensor_cls in input_tensors]
    ori_last_two_dim_list = [tensor_cls.ori_last_two_dim for tensor_cls in input_tensors]

    is_all_nz_supported = is_all_supported(check_func_name, input_tensors, need_align_with_broadcast)
    is_dim_equal = is_all_dim_equal(ori_last_one_dim_list) and is_all_dim_equal(ori_last_two_dim_list)

    return is_all_nz_supported and is_dim_equal


def is_broadcast_for_fz(input_tensors, need_align_with_broadcast, need_check_other_shape=True):
    """is_broadcast_for_fz, check whether all input_tensor support fz

    Parameters
    ----------
    input_tensors: list
        a list of TensorClass
    need_align_with_broadcast: bool
        check whether the n/c dim needs to be aligned
    need_check_other_shape: bool
        check whether the h/w dim needs to be equal
        fz shape is [(N1*H*W), N0, C1, C0]

    Returns
    -------
    res : bool
    """
    check_func_name = TensorClass.is_support_fz
    ori_c_dim_list = [tensor_cls.ori_c for tensor_cls in input_tensors]
    ori_n_dim_list = [tensor_cls.ori_n for tensor_cls in input_tensors]
    ori_h_dim_list = [tensor_cls.ori_h for tensor_cls in input_tensors]
    ori_w_dim_list = [tensor_cls.ori_w for tensor_cls in input_tensors]

    is_all_fz_supported = is_all_supported(check_func_name, input_tensors, need_align_with_broadcast)
    is_all_equal = False
    if is_all_fz_supported:
        is_all_equal = is_all_dim_equal(ori_c_dim_list) and is_all_dim_equal(ori_n_dim_list)
        if need_check_other_shape:
            is_all_equal = is_all_equal and is_all_dim_equal(ori_h_dim_list) and is_all_dim_equal(ori_w_dim_list)

    return is_all_fz_supported and is_all_equal


def is_broadcast_for_fz3d(input_tensors, need_align_with_broadcast, need_check_other_shape=True):
    """is_broadcast_for_fz3d, check whether all input_tensor support fz3d

    Parameters
    ----------
    input_tensors: list
        a list of TensorClass
    need_align_with_broadcast: bool
        check whether the n/c dim needs to be aligned
    need_check_other_shape: bool
        check whether the h/w dim needs to be equal
        fz3d shape is [(N1*D*H*W), N0, C1, C0]

    Returns
    -------
    res : bool
    """
    check_func_name = TensorClass.is_support_fz3d
    ori_c_dim_list = [tensor_cls.ori_c for tensor_cls in input_tensors]
    ori_n_dim_list = [tensor_cls.ori_n for tensor_cls in input_tensors]
    ori_h_dim_list = [tensor_cls.ori_h for tensor_cls in input_tensors]
    ori_w_dim_list = [tensor_cls.ori_w for tensor_cls in input_tensors]
    ori_d_dim_list = [tensor_cls.ori_d for tensor_cls in input_tensors]

    is_all_fz3d_supported = is_all_supported(check_func_name, input_tensors, need_align_with_broadcast)
    is_all_equal = False
    if is_all_fz3d_supported:
        is_all_equal = is_all_dim_equal(ori_c_dim_list) and is_all_dim_equal(ori_n_dim_list)
        if need_check_other_shape:
            is_all_equal = is_all_equal and is_all_dim_equal(ori_h_dim_list) and is_all_dim_equal(ori_w_dim_list)
            is_all_equal = is_all_equal and is_all_dim_equal(ori_d_dim_list)

    return is_all_fz3d_supported and is_all_equal


# 'pylint: disable=too-many-locals
def get_format_for_broardcast(tensor_cls_list,
                              need_align_with_broadcast=False,
                              need_align_without_broadcast=False,
                              need_check_other_dim=True,
                              need_add_default_format=True):
    """get_format_for_broardcast_tensors, do broadcast op_select

    supported list:
        when all shape is static and equal, support all format(additional rule: need_align_without_broadcast)
        when need broadcast or dynamic shape, as follows:
            1. the special dim is not broadcast, support all format(additional rule: need_align_with_broadcast)
                5HD + 5HD + 5HD = 5HD  special dim is C
                FZ + FZ + FZ = FZ      special dim is N/C
                NZ + NZ + NZ = NZ      special dim is the last two dims
            2. Based on the first scenario, if one input is scalar, result is
                5HD + 5HD + ND = 5HD  special dim is C
                FZ + FZ + ND = FZ      special dim is N/C
                NZ + NZ + ND = NZ      special dim is the last two dims

    Parameters
    ----------
    tensor_cls_list: list
        a list of TensorClass.
    need_align_with_broadcast: bool
        A switch that controls whether the special dim needs to be aligned in broadcast scene, default: Flase
    need_align_without_broadcast: bool
        A switch that controls whether the special dim needs to be aligned in without broadcast scene, default: Flase
    need_add_default_format: bool
        whether add ND to result, default: True

    Returns:
    -------
    result_list: list
        list result
            [['ND'], ['FRACTAL_NZ'], ['FRACTAL_NZ']] means ND + FRACTAL_NZ = FRACTAL_NZ
    """
    if not tensor_cls_list:
        return []
    # gen the output list
    result_list = [[] for _ in range(len(tensor_cls_list) + 1)]

    # case: static case, and all shape is equal, supported all format
    static_status_list = [tensor.is_static_shape for tensor in tensor_cls_list]
    set_static_status_list = list(set(static_status_list))
    is_all_status = len(set_static_status_list) == 1 and set_static_status_list[0]

    shape_list = [tensor.ori_shape for tensor in tensor_cls_list]
    is_all_shape_equal = shape_list.count(shape_list[0]) == len(shape_list)

    if is_all_status and is_all_shape_equal:
        # find whether have dividend_input
        is_have_dividend_input = False
        for tensor_cls in tensor_cls_list:
            if tensor_cls.is_dividend_input:
                is_have_dividend_input = True
                break
        align_flag = is_have_dividend_input or need_align_without_broadcast
        special_format = get_format_for_format_ignore(tensor_cls_list[0], align_flag)
        for _format_list in result_list:
            _format_list += special_format

        return result_list

    scalar_idx_list = [i for i, tensor in enumerate(tensor_cls_list) if tensor.shape_count == 1]
    tensor_cls_without_one_data_list = [tensor for tensor in tensor_cls_list if tensor.shape_count != 1]

    special_format_list = []

    def _run_match(match_func_name, match_format):
        """
        _run_match: when match_func_name is true, will add match_format to special_format_list
        """
        if match_func_name(tensor_cls_without_one_data_list, need_align_with_broadcast, need_check_other_dim):
            special_format_list.append(match_format)

    # define the check function dict
    # when key function is true, will add the value to special_format_list
    check_dict = {
        is_broadcast_for_5hd: FormatConstant.FORMAT_5HD,
        is_broadcast_for_6hd: FormatConstant.FORMAT_6HD,
        is_broadcast_for_nz: FormatConstant.FORMAT_NZ,
        is_broadcast_for_fz: FormatConstant.FORMAT_FZ,
        is_broadcast_for_fz3d: FormatConstant.FORMAT_FZ_3D
    }

    for match_key in check_dict.items():
        _run_match(match_key[0], match_key[1])

    # set all format base special_format_list
    # `example: 5HD + 5HD = 5HD,6HD + 6HD = 6HD,NZ + NZ = NZ`
    for _format_list in result_list:
        _format_list += special_format_list

    # set the input scalar to FormatConstant.FORMAT_ND base the scalar_idx
    for scalar_idx in scalar_idx_list:
        result_list[scalar_idx] = [FormatConstant.FORMAT_ND] * len(result_list[scalar_idx])

    # when need default format, add ND to format list
    if need_add_default_format:
        for _format_list in result_list:
            _format_list += [FormatConstant.FORMAT_ND]

    return result_list


def get_format_for_elewise(tensor_cls_list,
                           need_align=False,
                           need_add_default_format=True):
    """get_format_for_elewise, do elewise op_select

    Parameters
    ----------
    tensor_cls_list: list
        a list of TensorClass.
    need_align: bool
        A switch that controls whether the special dim needs to be aligned in elewise scene, default: Flase
    need_add_default_format: bool
        whether add ND to result, default: True

    Returns:
    -------
    result_list: list
        list result
            [['FRACTAL_NZ'], ['FRACTAL_NZ']] means FRACTAL_NZ(in) and FRACTAL_NZ(out)
    """
    # gen the output list
    result_list = [[] for _ in range(len(tensor_cls_list) + 1)]

    # gen the output list
    if tensor_cls_list:
        special_format_list = get_format_for_format_ignore(tensor_cls_list[0], need_align=need_align)

        # set all format base special_format_list
        for _format_list in result_list:
            _format_list += special_format_list

    # when need default format, add ND to format list
    if need_add_default_format:
        for _format_list in result_list:
            _format_list += [FormatConstant.FORMAT_ND]

    return result_list


def is_all_5hd_for_broadcast(tensor_list):
    """judge if all inputs support 5HD format and can do broadcast

    Parameters
    ----------
    tensor_list: list
        a list of input tensor_dict.
    Returns:
    -------
    judge result: bool
        `True` means support '5HD + 5HD' scene, `False` means not.
    """
    tensor_cls_list = [TensorClass(x) for x in tensor_list]
    need_align_with_broadcast = False
    check_func_name = TensorClass.is_support_5hd
    all_5hd_supported = is_all_supported(check_func_name, tensor_cls_list, need_align_with_broadcast)
    if not all_5hd_supported:
        return False  # check all input could convert to 5hd format

    for axis in "NHWC":  # check could broadcast after converted
        expect = 1
        for tensor_cls in tensor_cls_list:
            actual = tensor_cls.ori_shape[tensor_cls.ori_format.index(axis)]
            if actual == -1:
                continue  # dynamic shape scene
            if expect == actual or expect == 1 or actual == 1:
                if expect == 1 and actual != 1:
                    expect = actual
            else:
                return False
    return True


def add_5hd_support_param_desc(param_desc_list, tensor_list, dtype_list):
    """
    add `5HD+5HD->5HD` scene supported param description

    Parameters
    ----------
    param_desc_list: list
        a constrants description of inputs and outputs of the op
    tensor_list : list
        a list of input tensor_dict
    dtype_list: list
        a list of op support dtypes

    Returns:
    -------
    param_desc_list: list
        a constrants description of op params, which add '5HD+5HD->5HD' support scene
    """
    support_5hd = is_all_5hd_for_broadcast(tensor_list)
    if not support_5hd:
        return param_desc_list

    need_add_5hd = False
    for item in param_desc_list:
        has_5hd_dynamic = False
        has_5hd_static = False
        if item.element.unknownshape_format is not None:
            has_5hd_dynamic = "NC1HWC0" in item.element.unknownshape_format
        if item.element.format is not None:
            has_5hd_static = "NC1HWC0" in item.element.format
        if not has_5hd_dynamic and not has_5hd_static:
            need_add_5hd = True
            break

    # 5HD + 5HD -> 5HD
    if need_add_5hd:
        for item in param_desc_list:
            add_5hd_dtype_str = ",{}".format(
                ",".join(dtype_list))
            item.element.datatype += add_5hd_dtype_str

            add_null_str = ",{}".format(
                ",".join(["NULL"] * len(dtype_list)))
            add_5hd_format_str = ",{}".format(
                ",".join(["NC1HWC0"] * len(dtype_list)))
            if item.element.unknownshape_format is not None:
                item.element.unknownshape_format += add_5hd_format_str
            if item.element.format is not None:
                item.element.format += add_null_str
    return param_desc_list
