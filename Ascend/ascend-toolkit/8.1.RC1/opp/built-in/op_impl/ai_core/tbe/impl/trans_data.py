"""
Copyright (C) 2019-2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

trans_data
"""
from te.utils import para_check
from impl import nchw_hwcn_zn
from impl import depthwise_weight_4d_2_6d
from impl import depthwise_weight_6d_2_4d
from impl import trans_data_2d
from impl import transpose_d
from impl import nhwc_2_fractal_z_c04
from impl import nchw_2_fractal_z_c04
from impl import hwcn_2_fractal_z_c04
from impl import four_2_five_c04
from impl import zn_2_hwcn_lstm
from impl import zng_2_nchw_hwcn
from impl import nchw_2_fractal_z_g
from impl import hwcn_2_fractal_z_g
from impl import trans_data_positive_source_ntc
from impl import trans_data_negative_target_ntc
from impl import trans_data_positive_source_tc
from impl import trans_data_negative_target_tc
from tbe import tvm
from impl.util import util_common
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import shape_util
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput


# the NCHW format length
NCDHW_LENTH = 5
FORMAT_LIST = [("NCHW", "NHWC"), ("NCHW", "HWCN"), ("NHWC", "NCHW"), ("NHWC", "HWCN"),
               ("HWCN", "NCHW"), ("HWCN", "NHWC"), ("CHWN", "NCHW"), ("CHWN", "NHWC"), ("CHWN", "HWCN")]


FZ2FZG_SUPPORT = [("FRACTAL_Z", "FRACTAL_Z"), ("FRACTAL_Z_3D", "FRACTAL_Z_3D")]
# 'pylint: disable = unused-argument,too-many-arguments
def get_op_support_info(src, dst, src_format, dst_format,
                        src_subformat=1, dst_subformat=1, groups=1, kernel_name='trans_data'):
    """
    get_op_support_info
    """
    src_shape = src.get("shape")
    src_format = src_format.upper()
    dst_format = dst_format.upper()
    axis_reduce_list = []
    axis_split_matrix = []
    split_0 = []
    nd_format = ("NHWC", "NCHW", "ND")
    if (src_format in nd_format and dst_format == "NC1HWC0") or \
        (src_format == "NC1HWC0" and dst_format in nd_format):
        split_0 = [SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])]
        axis_split_matrix.append(split_0)
    elif src_format in nd_format and dst_format == "FRACTAL_NZ":
        if len(src_shape) == 2:
            split_0 = [SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [1]])]
        else:
            split_0 = [SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])]
        axis_split_matrix.append(split_0)
    elif src_format == "FRACTAL_NZ" and dst_format in nd_format:
        if len(src_shape) == 4:
            split_0 = [SplitInput([0, [1], [-1], [-1]]), SplitOutput([0, [0]])]
        else:
            split_0 = [SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])]
        axis_split_matrix.append(split_0)
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=locally-disabled,redefined-builtin,too-many-statements
# 'pylint: disable=too-many-arguments
def check_whether_2d(input_format, input_dict):
    """Check whether the 4D is 2D extend to 4D

    Parameters
    ----------
    input_format: str
        source data format, can be NHWC, NCHW, FRACTAL_Zn etc.
    input_dict: dict
        shape and dtype of output, should be same shape and type as input

    Returns
    -------
    is_2d : bool
        is_2d
    """
    is_2d = False
    shape = input_dict.get("shape")
    if not (len(list(input_format)) == len(shape) and len(shape) == 4):
        return is_2d

    dict_zip = dict(zip(list(input_format), shape))
    if dict_zip["H"] == 1 and dict_zip["W"] == 1 and \
            dict_zip["C"] % 16 == 0:
        is_2d = True

    return is_2d


def check_group_trans_data_black_list(src, dst):
    black_list = [([7, 7, 1, 24], [49, 2, 16, 16]), ([128, 1, 32, 32], [8192, 1, 16, 16]),
                  ([256, 1, 16, 16], [4096, 1, 16, 16]), ([32, 1, 16, 16], [128, 1, 2, 2]),
                  ([3, 1, 16, 16], [16, 1, 1, 3]), ([9, 1, 16, 16], [48, 1, 1, 3]),
                  ([18, 1, 16, 16], [96, 1, 1, 3]), ([6, 1, 16, 16], [24, 1, 1, 3]),
                  ([6, 1, 16, 16], [32, 1, 1, 3]), ([15, 1, 16, 16], [72, 1, 1, 3]),
                  ([45, 1, 16, 16], [144, 1, 1, 5]), ([9, 1, 16, 16], [40, 1, 1, 3]),
                  ([15, 1, 16, 16], [40, 1, 1, 5]), ([24, 1, 16, 16], [120, 1, 1, 3]),
                  ([45, 1, 16, 16], [240, 1, 1, 3]), ([90, 1, 16, 16], [480, 1, 1, 3]),
                  ([15, 1, 16, 16], [80, 1, 1, 3]), ([39, 1, 16, 16], [200, 1, 1, 3]),
                  ([36, 1, 16, 16], [184, 1, 1, 3]), ([21, 1, 16, 16], [112, 1, 1, 3]),
                  ([30, 1, 16, 16], [160, 1, 1, 13]), ([126, 1, 16, 16], [672, 1, 1, 3]),
                  ([420, 1, 16, 16], [1344, 1, 1, 5]), ([70, 1, 16, 16], [224, 1, 1, 5]),
                  ([180, 1, 16, 16], [960, 1, 1, 3]), ([2, 128, 48, 60], [64, 6, 1, 1]),
                  ([1, 128, 126, 114], [64, 2, 2, 2]), ([1, 256, 126, 114], [128, 8, 2, 2]),
                  ([2, 10, 192], [4, 5, 1]), ([2, 72, 192], [18, 8, 1]),
                  ([32, 1, 16, 16], [64, 32, 2, 2]), ([64, 32, 2, 2], [32, 1, 16, 16]),
                  ([64, 32, 2, 2], [64, 32, 2, 2]), ([128, 32, 3, 3], [36, 4, 16, 16]),
                  ([32, 32, 3, 3], [36, 1, 16, 16]), ([512, 256, 3, 3], [288, 16, 16, 16]),
                  ([512, 128, 3, 3], [144, 16, 16, 16]), ([64, 64, 3, 3], [72, 2, 16, 16]),
                  ([128, 128, 3, 3], [144, 4, 16, 16]), ([128, 64, 3, 3], [72, 4, 16, 16]),
                  ([256, 128, 3, 3], [144, 8, 16, 16]), ([384, 1, 3, 3], [216, 1, 16, 16]),
                  ([96, 1, 3, 3], [54, 1, 16, 16]), ([32, 1, 3, 3], [18, 1, 16, 16]),
                  ([960, 1, 3, 3], [540, 1, 16, 16]), ([64, 1, 3, 3], [36, 1, 16, 16]),
                  ([512, 1, 3, 3], [288, 1, 16, 16]), ([256, 1, 3, 3], [144, 1, 16, 16]),
                  ([576, 1, 3, 3], [324, 1, 16, 16]), ([128, 1, 3, 3], [720, 1, 16, 16]),
                  ([128, 1, 3, 3], [72, 1, 16, 16]), ([1280, 1, 3, 3], [720, 1, 16, 16]),
                  ([192, 1, 3, 3], [108, 1, 16, 16]), ([144, 1, 3, 3], [81, 1, 16, 16])]

    src_shape = list(src.get("shape", []))
    dst_shape = list(dst.get("shape", []))
    return (src_shape, dst_shape) in black_list


def check_group_trans_data_black_list_with_platform(src, dst):
    black_list = [([5, 4, 1, 256], [20, 16, 16, 16]), ([5, 4, 1, 1024], [40, 32, 16, 16])]
    src_shape = list(src.get("shape", []))
    dst_shape = list(dst.get("shape", []))
    return (src_shape, dst_shape) in black_list


def check_supported(src, dst, src_format=None, dst_format=None,
                    src_subformat=1, dst_subformat=1, groups=1, kernel_name="trans_data"):
    src_format = src.get("format", [])
    dst_format = dst.get("format", [])
    if (src_format, dst_format) in FORMAT_LIST and \
            tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ["Ascend310B", "AS31XM1", "Ascend610Lite", 
                                                               "BS9SX1AA", "BS9SX2A", "MC61AM21A"]:
        return False, "not support outerside format transformer in transdata"
    if (src_format, dst_format) in FZ2FZG_SUPPORT and groups <= 1:
        return False, "only support FZ2FZ while groups <= 1"
    reason = "The value of groups is greater than 1, not support aicore"
    if groups > 1 and check_group_trans_data_black_list(src, dst):
        return False, reason
    if groups > 1 and check_group_trans_data_black_list_with_platform(src, dst) \
                  and tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend310B", "AS31XM1"):
        return False, reason

    return True


# 'pylint: disable=locally-disabled,too-many-branches
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_STR,
                            para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def trans_data(src, dst, src_format, dst_format,
               src_subformat=1, dst_subformat=1, groups=1, kernel_name='trans_data'):
    """
    algorithm: format_transfer
    doing format_transfer for various data format
    only support NHWC/NCHW to NC1HWC0 and NC1HWC0 to NHWC/NCHW
    NCHW to FRACTAL_Zn or FRACTAL_Zn to NCHW
    HWCN to FRACTAL_Zn or FRACTAL_Zn to HWCN

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    src_format: str
        source data format, can be NHWC, NCHW, FRACTAL_Zn etc.
    dst_format: str
        target data format, can be NC1HWC0, NCHW, FRACTAL_Zn etc.
    groups: int
        default 1
    kernel_name: str
        kernel name, default value is "format_transfer"

    Returns
    -------
    None
    """
    positive_tc_transfer = [("NHWC", "NC1HWC0"), ("NDHWC", "NDC1HWC0"), ("NHWC", "FRACTAL_NZ"), ("ND", "FRACTAL_NZ"),
                             ("NCHW", "FRACTAL_NZ"), ("NDHWC", "FRACTAL_Z_3D"), ("NC1HWC0", "FRACTAL_Z")]
    positive_ntc_transfer = [("NCHW", "NC1HWC0"), ("NCDHW", "NDC1HWC0"), ("HWCN", "FRACTAL_Z"), ("ND", "FRACTAL_Z"),
                             ("NCHW", "FRACTAL_Z"), ("DHWCN", "FRACTAL_Z_3D"), ("NCDHW", "FRACTAL_Z_3D")]
    negative_tc_transfer = [("NC1HWC0", "NHWC"), ("NDC1HWC0", "NDHWC"), ("FRACTAL_NZ", "NHWC"), ("FRACTAL_NZ", "ND"),
                             ("FRACTAL_NZ", "NCHW"), ("FRACTAL_Z_3D", "NDHWC"), ("FRACTAL_NZ", "NC1HWC0")]
    negative_ntc_transfer = [("NC1HWC0", "NCHW"), ("NDC1HWC0", "NCDHW"), ("FRACTAL_Z", "HWCN"), ("FRACTAL_Z", "ND"),
                             ("FRACTAL_Z", "NCHW"), ("FRACTAL_Z_3D", "DHWCN"), ("FRACTAL_Z_3D", "NCDHW")]

    if (src_format.upper() in ("NHWC", "NCHW") and dst_format.upper() == "NC1HWC0" and
            check_whether_2d(src_format.upper(), src)):
        trans_data_2d(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper() == "NC1HWC0" and dst_format.upper() in ("NHWC", "NCHW") and
          check_whether_2d(dst_format.upper(), dst)):
        trans_data_2d(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper(), dst_format.upper()) in positive_tc_transfer:
        trans_data_positive_source_tc.trans_data_positive_source_tc(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper(), dst_format.upper()) in positive_ntc_transfer and groups == 1:
        trans_data_positive_source_ntc.trans_data_positive_source_ntc(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper(), dst_format.upper()) in negative_tc_transfer:
        trans_data_negative_target_tc.trans_data_negative_target_tc(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper(), dst_format.upper()) in negative_ntc_transfer and groups == 1:
        trans_data_negative_target_ntc.trans_data_negative_target_ntc(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper() == "HWCN" and dst_format.upper() == "FRACTAL_Z" and
          groups > 1 and groups == src.get("shape")[-1]):
        dst_format = "C1HWNCOC0"
        axis_h, axis_w, axis_c, axis_n = src.get("shape")
        axis_c = axis_c * groups
        axis_n = 1
        src["shape"] = (axis_h, axis_w, axis_c, axis_n)
        depthwise_weight_4d_2_6d(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper() == "FRACTAL_Z" and dst_format.upper() == "HWCN" and
          groups > 1 and groups == dst.get("shape")[-1]):
        src_format = "C1HWNCOC0"
        axis_h, axis_w, axis_c, axis_n = dst.get("shape")
        axis_c = axis_c * groups
        axis_n = 1
        axis_c1 = (axis_c + 15) // 16
        src["shape"] = (axis_c1, axis_h, axis_w, axis_n, 16, 16)
        dst["shape"] = (axis_h, axis_w, axis_c, axis_n)
        depthwise_weight_6d_2_4d(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "NCHW" \
            and ((dst_format.upper() == "FRACTAL_ZN" or dst_format.upper() == "FRACTAL_Z") and groups > 1):
        nchw_2_fractal_z_g.nchw_2_fractal_z_g(src, dst, src_format, dst_format, groups, kernel_name)
    elif (src_format.upper() == "FRACTAL_ZN"
          or src_format.upper() == "FRACTAL_Z") \
            and dst_format.upper() == "NCHW" and groups > 1:
        zng_2_nchw_hwcn.zng_2_nchw_hwcn(src, dst, src_format, dst_format, groups, kernel_name)
    elif src_format.upper() == "FRACTAL_Z_3D" and dst_format.upper() == "DHWCN" and groups > 1:
        zng_2_nchw_hwcn.zng_2_nchw_hwcn(src, dst, src_format, dst_format, groups, kernel_name)
    elif src_format.upper() == "HWCN" and dst_format.upper() == "FRACTAL_ZN_LSTM":
        nchw_hwcn_zn.nchw_hwcn_zn(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "HWCN" and dst_format.upper() == "FRACTAL_Z" and groups > 1:
        hwcn_2_fractal_z_g.hwcn_2_fractal_z_g(src, dst, src_format, dst_format, groups, kernel_name)
    elif src_format.upper() == "FRACTAL_ZN_LSTM" and \
            dst_format.upper() == "HWCN":
        zn_2_hwcn_lstm.zn_2_hwcn_lstm(src, dst, src_format,
                                      dst_format, kernel_name)
    elif src_format.upper() == "FRACTAL_Z" and dst_format.upper() == "HWCN" and groups > 1:
        zng_2_nchw_hwcn.zng_2_nchw_hwcn(src, dst, src_format, dst_format, groups, kernel_name)
    elif src_format.upper() == "HWCN" \
            and dst_format.upper() == "C1HWNCOC0":
        depthwise_weight_4d_2_6d(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "C1HWNCOC0" \
            and dst_format.upper() == "HWCN":
        depthwise_weight_6d_2_4d(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "NCHW" and dst_format.upper() == "NHWC":
        transpose_d(src, dst, [0, 2, 3, 1], kernel_name)
    elif src_format.upper() == "NCHW" and dst_format.upper() == "HWCN":
        transpose_d(src, dst, [2, 3, 1, 0], kernel_name)
    elif src_format.upper() == "NHWC" and dst_format.upper() == "NCHW":
        transpose_d(src, dst, [0, 3, 1, 2], kernel_name)
    elif src_format.upper() == "NHWC" and dst_format.upper() == "HWCN":
        transpose_d(src, dst, [1, 2, 3, 0], kernel_name)
    elif src_format.upper() == "HWCN" and dst_format.upper() == "NCHW":
        transpose_d(src, dst, [3, 2, 0, 1], kernel_name)
    elif src_format.upper() == "HWCN" and dst_format.upper() == "NHWC":
        transpose_d(src, dst, [3, 0, 1, 2], kernel_name)
    elif src_format.upper() == "CHWN" and dst_format.upper() == "NCHW":
        transpose_d(src, dst, [3, 0, 1, 2], kernel_name)
    elif src_format.upper() == "CHWN" and dst_format.upper() == "NHWC":
        transpose_d(src, dst, [3, 1, 2, 0], kernel_name)
    elif src_format.upper() == "CHWN" and dst_format.upper() == "HWCN":
        transpose_d(src, dst, [1, 2, 0, 3], kernel_name)
    elif src_format.upper() == "NHWC" and dst_format.upper() == "FRACTAL_Z_C04":
        nhwc_2_fractal_z_c04.nhwc_2_fractal_z_c04(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "NCHW" and dst_format.upper() == "FRACTAL_Z_C04":
        nchw_2_fractal_z_c04.nchw_2_fractal_z_c04(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "HWCN" and dst_format.upper() == "FRACTAL_Z_C04":
        hwcn_2_fractal_z_c04.hwcn_2_fractal_z_c04(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() in ["NHWC", "NCHW", "HWCN"] and dst_format.upper() == "NC1HWC0_C04":
        four_2_five_c04.four_2_five_c04(src, dst, src_format, dst_format, kernel_name)
    else:
        error_manager_vector.raise_err_specific_reson("trans_data", "not support the format transfer!")


# 'pylint: disable=too-many-locals
@tbe_platform.fusion_manager.register("trans_data")
def trans_data_compute(src, dst, src_format, dst_format,
                       src_subformat=1, dst_subformat=1, groups=1, kernel_name='transdata'):
    """
    algorithm: format_transfer
    used for on the fly format transformation , For example NHWC TO NC1HWC0,
    NC1HWC0 TO NHWC, NHWC TO FRACTAL_Z
    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    src_format: str
        source data format, can be NHWC, NCHW, FRACTAL_Zn etc.
    dst_format: str
        target data format, can be NC1HWC0, NCHW, FRACTAL_Zn etc.
    groups: int
        default 1
    kernel_name: str
        kernel name, default value is "format_transfer"

    Returns
    -------
    None
    """
    dst_tensor = None
    c0_dict = {"float32": 8, "float16": 16, "int8": 32, "int4": 64, "bfloat16": 16}
    fractal_n0 = 16 # the third params of fractal_nz(d // d0, n // n0, n0, d0)
    def _ceil_div(dividend, divisor):
        if divisor == 0:
            raise RuntimeError("division by zero")
        return (dividend + divisor - 1) // divisor

    if src_format == "NHWC" and dst_format == "NC1HWC0":
        src_n, src_h, src_w, src_c = tuple(i.value for i in src.shape)
        dst_c0 = c0_dict.get(src.dtype)
        dst_c1 = _ceil_div(src_c, dst_c0)
        dst_shape = (src_n, dst_c1, src_h, src_w, dst_c0)
        dst_tensor = tvm.compute(dst_shape,
            lambda n_idx, c1_idx, h_idx, w_idx, c0_idx: tvm.select(
                tvm.any(c1_idx * dst_c0 + c0_idx < src_c),
                src(n_idx, h_idx, w_idx, c1_idx * dst_c0 + c0_idx)),
                name="res_nc1hwc0",
                attrs={"ori_format": "NHWC", "ori_shape": src.shape,
                       "format": "NC1HWC0"},
                tag="NHWC_trans_5HD")
    elif src_format == "NC1HWC0" and dst_format == "NHWC":
        src_n, src_c1, src_hw, src_c0 = tuple(i.value for i in src.shape)
        dst_n, dst_h, dst_w, dst_c = dst.get("shape")
        dst_shape = (dst_n, dst_h*dst_w, dst_c)

        if dst_n != src_n:
            error_manager_vector.raise_err_specific_reson("trans_data",
                                                          "batch should not be changed when trans NC1HWC0 to NHWC!")

        if dst_h*dst_w != src_hw:
            error_manager_vector.raise_err_specific_reson("trans_data",
                                                          "Ho*Wo should not be changed when trans NC1HWC0 to NHWC!")

        dst_tensor = tvm.compute(
            dst_shape,
            lambda batch_idx, howo_idx, co_idx: src(
                batch_idx, co_idx // src_c0, howo_idx, co_idx % src_c0),
            name="res_nhwc",
            tag="5HD_trans_NHWC")

    elif src_format == "NC1HWC0" and dst_format == "NCHW":
        src_n = src.shape[0].value
        src_c1 = src.shape[1].value
        src_hw = src.shape[2].value
        src_c0 = src.shape[3].value
        dst_ori_shape = dst.get("shape")
        dst_shape = [src_n, dst_ori_shape[1], src_hw]
        m0_fp16 = 16
        src_hw_align = _ceil_div(src_hw, m0_fp16) * m0_fp16
        if src_hw != src_hw_align:
            pad_shape_nc1hwc0 = [src_n, src_c1, src_hw_align, src_c0]
            pad_shape_nchw = [src_n, dst_ori_shape[1], src_hw_align]
            add_pad = tvm.compute(pad_shape_nc1hwc0,
                                  lambda n_idx, c1_idx, hw_idx, c0_idx:
                                      tvm.select(hw_idx < src_hw,
                                                 src(n_idx, c1_idx, hw_idx, c0_idx)),
                                  name="res_add_pad",
                                  tag="5HD_trans_NCHW_add_pad")
            transdata_tensor = tvm.compute(pad_shape_nchw,
                                           lambda n_idx, c_idx, hw_idx:
                                               add_pad(n_idx, c_idx // src_c0, hw_idx, c_idx % src_c0),
                                           name="res_nchw",
                                           tag="5HD_trans_NCHW")
            dst_tensor = tvm.compute(dst_shape,
                                     lambda n_idx, c_idx, hw_idx:
                                         transdata_tensor(n_idx, c_idx, hw_idx),
                                     name="res_rm_pad",
                                     tag="5HD_trans_NCHW_rm_pad")
        else:
            dst_tensor = tvm.compute(dst_shape,
                                     lambda n_idx, c_idx, hw_idx:
                                         src(n_idx, c_idx // src_c0, hw_idx, c_idx % src_c0),
                                     name="res_nchw",
                                     tag="5HD_trans_NCHW")

    elif src_format == "NHWC" and dst_format == "FRACTAL_Z":
        src_n, src_h, src_w, src_c = tuple(i.value for i in src.shape)
        dst_n1 = _ceil_div(src_n, fractal_n0)
        dst_c0 = c0_dict.get(src.dtype)
        dst_c1 = _ceil_div(src_c, dst_c0)
        dst_shape = dst_c1 * src_h * src_w, dst_n1, fractal_n0, dst_c0
        hw = src_h * src_w
        dst_tensor = tvm.compute(
            dst_shape,
            lambda  i, j, k, l: src(j * fractal_n0 + k,
            (i % hw) // src_w, (i % hw) % src_w,
            (i // hw) * dst_c0 + l),
            name="res_fractal_z_weight",
            attrs={"ori_format": "NHWC", "ori_shape": src.shape},
            tag="NHWC_trans_FZ"
        )
    elif src_format in ["ND", "NHWC"] and dst_format == "FRACTAL_NZ":
        src_shape = tuple(i.value for i in src.shape)
        ori_shape = src.op.attrs["ori_shape"] if "ori_shape" in src.op.attrs else src.shape
        block_reduce = c0_dict.get(src.dtype, tbe_platform.BLOCK_REDUCE)
        block_size = tbe_platform.BLOCK_IN
        dst_shape = (
            _ceil_div(src_shape[-1], block_reduce),
            _ceil_div(src_shape[-2], block_size),
            block_size,
            block_reduce
        )
        dst_shape = src_shape[:-2] + dst_shape
        d_axis_origin_length = src_shape[-1]
        dst_tensor = tvm.compute(
            dst_shape,
            lambda *indices: tvm.select(
                tvm.all((indices[-4] * block_reduce + indices[-1]) < d_axis_origin_length),
                src(*indices[:-4],
                    indices[-3] * block_size + indices[-2],
                    indices[-4] * block_reduce + indices[-1])
            ),
            name=src.name + "_fractal",
            attrs={"ori_format": "ND", "ori_shape": ori_shape, "format": dst_format},
            tag="ND_trans_NZ"
        )
    elif src_format == "FRACTAL_NZ" and dst_format == "ND":
        src_shape = tuple(i.value for i in src.shape)
        dst_shape = src.op.attrs["ori_shape"]
        dst_tensor = tvm.compute(
                dst_shape,
                lambda *indices: src(*indices[:-2],
                                     indices[-1] // src_shape[-1],
                                     indices[-2] // src_shape[-2],
                                     indices[-2] % src_shape[-2],
                                     indices[-1] % src_shape[-1]),
                tag="NZ_trans_ND",
                name="res_nd",
                attrs={"ori_format": "FRACTAL_NZ",
                       "ori_shape": src.shape})
    elif src_format == "FRACTAL_Z" and dst_format == "NHWC":
        if src.shape[-1].value == 8:
            # c0 is means float32 situation
            group, src_c1, kk, src_n, src_c0 = tuple(i.value for i in src.shape)
            dst_shape = dst.get("shape")
            _, hw_length, _ = dst_shape

            dst_tensor = tvm.compute(
                dst_shape,
                lambda n_idx, hw_idx, c_idx:
                    # block_dim_reduce, group, c1, khkw, n, c0
                    src(n_idx // src_n,
                        c_idx // src_c0,
                        hw_idx,
                        n_idx,
                        c_idx % src_c0),
                name="res_nhwc",
                tag="FZ_trans_NHWC"
            )
        else:
            group, src_fkk, src_n, src_c0 = tuple(i.value for i in src.shape)
            dst_shape = dst.get("shape")
            _, hw_length, _ = dst_shape

            dst_tensor = tvm.compute(
                dst_shape,
                lambda n_idx, hw_idx, c_idx:
                    # block_dim_reduce, group, fww, n, c0
                    src(n_idx // src_n,
                        c_idx // src_c0 * hw_length + hw_idx,
                        n_idx,
                        c_idx % src_c0),
                name="res_nhwc",
                tag="FZ_trans_NHWC"
            )
    elif src_format == "NDHWC" and dst_format == "NDC1HWC0":
        shape_input = shape_util.shape_to_list(src.shape)
        if len(shape_input) == NCDHW_LENTH:
            src_n, src_d, src_h, src_w, src_c = shape_input
        else:
            src_n, src_d, src_h, src_w, src_c = shape_util.shape_to_list(src.op.attrs["shape"])
        dst_c0 = tbe_platform.CUBE_MKN[src.dtype]["mac"][1]
        dst_c1 = util_common.ceil(src_c, dst_c0)
        dst_shape = (src_n, src_d, dst_c1, src_h, src_w, dst_c0)
        dst_tensor = tvm.compute(
            dst_shape, lambda n_idx, d_idx, c1_idx, h_idx, w_idx, c0_idx: tvm.select(
                tvm.any(c1_idx * dst_c0 + c0_idx < src_c),
                src(n_idx, d_idx, h_idx, w_idx, c1_idx * dst_c0 + c0_idx)),
            name="res_ndc1hwc0",
            attrs={"ori_format": "NDHWC", "ori_shape": src.shape, "format": "NDC1HWC0"},
            tag="NDHWC_trans_6HD"
        )
        return dst_tensor
    else:
        error_manager_vector.raise_err_specific_reson("trans_data", "not support this kind of format transfer !")

    return dst_tensor

