"""
Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.

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
from __future__ import absolute_import
import copy
from tbe.dsl.base.operation import get_te_var
from tbe.common.platform.platform_info import get_soc_spec
from impl.constant_util import BLOCK_SIZE
from impl.util import fusion_util
from impl.util import util_common
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import is_dynamic_input
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.dynamic.transpose import Transpose
from .trans_data_positive_source_tc import trans_data_positive_source_tc
from .trans_data_negative_target_ntc import trans_data_negative_target_ntc
from .trans_data_positive_source_ntc import trans_data_positive_source_ntc
from .trans_data_negative_target_tc import trans_data_negative_target_tc
from . import conv2d_data_rm_compute
from .trans_data_dsl import trans_data_dsl
from .trans_data_format_rule import trans_data_format_rule
from .trans_data_normalization import trans_data_normalization
from .trans_data_groups import trans_data_groups
from .trans_data_c04 import trans_data_c04


# the NCHW format length
NCHW_LENTH = 4
DSL_SUPPORT = [("NHWC", "NC1HWC0"), ("NCHW", "NC1HWC0"), ("ND", "FRACTAL_NZ"),
               ("NC1HWC0", "NHWC"), ("NC1HWC0", "NCHW"), ("FRACTAL_NZ", "ND"),
               ("HWCN", "FRACTAL_Z"), ("FRACTAL_Z", "HWCN"), ("NCHW", "FRACTAL_Z"),
               ("FRACTAL_Z", "NCHW"), ("NCHW", "FRACTAL_Z_C04"), ("FRACTAL_Z_C04", "NCHW"),
               ("NDHWC", "NDC1HWC0"), ("NDC1HWC0", "NDHWC")]
PRIVATE_FORMATS = ["FRACTAL_Z", "NC1HWC0", "NDC1HWC0", "FRACTAL_NZ"]
FZ2FZG_SUPPORT = [("FRACTAL_Z", "FRACTAL_Z"), ("FRACTAL_Z_3D", "FRACTAL_Z_3D")]
C8 = 8
TIK_C8_SUPPORT = [("NCDHW", "NDC1HWC0"), ("NDC1HWC0", "NCDHW"),
                  ("NCDHW", "FRACTAL_Z_3D"), ("FRACTAL_Z_3D", "NCDHW")]


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


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    TILING_MAX_SIZE_GM = 2048  # 16KB
    MAX_INT64_VALUE = 2 ** 64 - 1


def is_do_with_transpose_formats(src_format, dst_format):
    """
    judge src_format and dst_format in the list: ["NCHW", "NHWC", "HWCN", "CHWN"]
    """
    format_list = ["NCHW", "NHWC", "HWCN", "CHWN"]
    if src_format in format_list and dst_format in format_list and src_format != dst_format:
        return True
    return False


def is_do_with_positive_source_ntc_100(src_format, dst_format):
    """
    judge src_format and dst_format in the dict:
    {"NCDHW":"NDC1HWC0", "NCHW":"NC1HWC0", "HWCN":"FRACTAL_Z", "HWCN":"FRACTAL_ZN", "DHWCN":"FRACTAL_Z_3D",
    "ND":"FRACTAL_Z", "ND":"FRACTAL_ZN", "NCHW":"FRACTAL_Z", "NCHW":"FRACTAL_ZN", "NCDHW":"FRACTAL_Z_3D"}
    """
    support_src_dst_formats = {"NCDHW": ["NDC1HWC0", "FRACTAL_Z_3D"], "HWCN": ["FRACTAL_Z", "FRACTAL_ZN"],
                               "DHWCN": ["FRACTAL_Z_3D"], "ND": ["FRACTAL_Z", "FRACTAL_ZN"],
                               "NCHW": ["FRACTAL_Z", "FRACTAL_ZN", "NC1HWC0"]}
    if src_format in support_src_dst_formats and dst_format in support_src_dst_formats.get(src_format):
        return True
    return False


def _is_dynamic(shape):
    if -1 in shape or -2 in shape:
        return True
    return False


def _blocklist(input_, output_, src_format, dst_format, dtype):
    input_ = list(input_)
    output_ = list(output_)

    nz_2_nd = [([32, 1, 16, 16], [1, 512]), ([20, 8, 16, 16], [120, 312]), ([4, 12, 2, 2, 16, 16], [4, 12, 30, 26]),
               ([4, 12, 2, 2, 16, 16], [4, 12, 30, 30]), ([1, 1, 16, 16], [4, 2]), ([20, 1, 16, 16], [4, 312]),
               ([63, 2, 16, 16], [32, 1000]), ([48, 384, 16, 16], [6144, 768]), ([192, 384, 16, 16], [6144, 3072]),
               ([1, 384, 16, 16], [6144, 2]), ([144, 4, 16, 16], [50, 2304]), ([12, 4, 4, 16, 16], [12, 50, 64]),
               ([48, 4, 16, 16], [50, 768]), ([3125, 13, 16, 16], [200, 50000])]

    nd_2_nz = [([1, 1792], [112, 1, 16, 16]), ([120, 312], [20, 8, 16, 16]), ([4, 12, 30, 26], [4, 12, 2, 2, 16, 16]),
               ([4, 12, 30, 30], [4, 12, 2, 2, 16, 16]), ([120, 2], [1, 8, 16, 16]), ([4, 624], [39, 1, 16, 16]),
               ([120, 128], [8, 8, 16, 16]), ([4, 312], [20, 1, 16, 16]), ([16, 12, 384, 384], [16, 12, 24, 24, 16]),
               ([6144, 768], [48, 384, 16, 16]), ([6144, 3072], [192, 384, 16, 16]), ([50, 768], [48, 4, 16, 16]),
               ([120, 50, 64], [12, 4, 4, 16, 16]), ([12, 64, 50], [12, 4, 4, 16, 16]), ([1, 768], [48, 1, 16, 16]),
               ([1, 512], [32, 1, 16, 16])]

    nhwc_2_nc1hwc0 = [([1, 224, 224, 3], [1, 1, 224, 224, 16]), ([1, 324, 576, 3], [1, 1, 324, 576, 16]),
                      ([4, 640, 640, 3], [4, 1, 640, 640, 16]), ([4, 160, 160, 48], (4, 3, 160, 160, 16)),
                      ([1, 640, 640, 3], [1, 1, 640, 640, 16])]

    nc1hwc0_2_nhwc = [([1, 112, 1, 1, 16], [1, 1, 1, 1792]), ([4, 2, 160, 160, 16], [4, 160, 160, 24]),
                      ([4, 6, 20, 20, 16], [4, 20, 20, 84]), ([1, 1, 157, 283, 16], [1, 157, 283, 2]),
                      ([1, 1, 157, 283, 16], [1, 157, 283, 4]), ([1, 1, 157, 283, 16], [1, 157, 283, 10])]

    nc1hwc0_2_nchw = [([32, 6, 55, 55, 16], [32, 96, 55, 55]), ([32, 16, 27, 27, 16], [32, 256, 27, 27]),
                      ([1, 48, 7, 7, 16], [1, 768, 7, 7]), ([2, 2, 512, 512, 16], [2, 19, 512, 512])]

    nchw_2_nc1hwc0 = [([32, 96, 55, 55], [32, 6, 55, 55, 16]), ([32, 256, 27, 27], [32, 16, 27, 27, 16])]

    nc1hwc0_2_nchw_fp32 = [([8, 2, 513, 513, 16], [8, 21, 513, 513])]

    if dtype == "float32":
        if (src_format, dst_format) == ("NC1HWC0", "NCHW") and (input_, output_) in nc1hwc0_2_nchw_fp32:
            return True

    if ((src_format == "NDHWC" and dst_format == "NDC1HWC0" and output_[-1] != 8) or
        (src_format == "NDC1HWC0" and dst_format == "NDHWC" and input_[-1] != 8)):
        return True

    if dtype != "float16":
        return False

    if (src_format, dst_format) == ("FRACTAL_NZ", "ND") and (input_, output_) in nz_2_nd:
        return True
    if (src_format, dst_format) == ("ND", "FRACTAL_NZ") and (input_, output_) in nd_2_nz:
        return True
    if (src_format, dst_format) == ("NHWC", "NC1HWC0") and (input_, output_) in nhwc_2_nc1hwc0:
        return True
    if (src_format, dst_format) == ("NC1HWC0", "NHWC") and (input_, output_) in nc1hwc0_2_nhwc:
        return True
    if (src_format, dst_format) == ("NCHW", "NC1HWC0") and (input_, output_) in nchw_2_nc1hwc0:
        return True
    if (src_format, dst_format) == ("NC1HWC0", "NCHW") and (input_, output_) in nc1hwc0_2_nchw:
        return True
    return False


def _deal_c08(src_format, dst_format, src_shape, dst_shape):
    # DSL only support Fz C0=8
    if "FRACTAL_Z" not in [src_format, dst_format] or int(get_soc_spec("ubblock_size")) < BLOCK_SIZE:
        return True
    c0 = src_shape[-1] if src_format == "FRACTAL_Z" else dst_shape[-1]
    return True if c0 == 8 else False


def _deal_tik_c08(src_format, src_shape, dst_shape):
    c0 = src_shape[-1] if src_format in ("NDC1HWC0", "FRACTAL_Z_3D") else dst_shape[-1]
    return True if c0 == C8 else False


def check_supported(src, dst, src_format=None, dst_format=None,
                    src_subformat=1, dst_subformat=1, groups=1, kernel_name="trans_data"):
    """
    check_supported invoked by framework
    5HD|NZ: do normalization of dyn-static.
    Others: don't do normalization.
    """
    # Not support normal FZ2FZ
    if (src_format, dst_format) in FZ2FZG_SUPPORT:
        if groups <= 1:
            return False, "only support FZ2FZ while groups <= 1"
        if src.get("dtype") in ["int8", "uint8"]:
            return False, "not support yet"
        if src.get("sub_format", 0) > 1 and src.get("ori_format", "NCHW") in ["NCHW", "NCDHW", "DHWCN"]:
            return True, ""
        if dst.get("sub_format", 0) > 1 and dst.get("ori_format", "NCHW") in ["NCHW", "NCDHW", "DHWCN"]:
            return True, ""

    if groups > 1:
        return False, "not support yet"

    src_shape = src.get("shape", [])
    dst_shape = dst.get("shape", [])
    dtype = src.get("dtype", "NA").lower()
    if _is_dynamic(src_shape) or _is_dynamic(dst_shape):
        return True, ""

    # const
    if (src_format, dst_format) in DSL_SUPPORT:
        if not _deal_c08(src_format, dst_format, src_shape, dst_shape):
            return False, "FRACTAL_Z only support C0=8 in DSL"
        if not _blocklist(src_shape, dst_shape, src_format, dst_format, dtype):
            return True, ""
    
    if (src_format, dst_format) in TIK_C8_SUPPORT:
        if _deal_tik_c08(src_format, src_shape, dst_shape):
            return True, "TIK only support NDC1HWC0 and FRACTAL_Z_3D when C0=8"

    return False, "the format only supported 5HD,NZ by DSL or in blocklist now"


# 'pylint: disable=unused-argument, too-many-arguments, too-many-locals, too-many-boolean-expressions
# 'pylint: disable=inconsistent-return-statements
@register_operator("TransData")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def trans_data(src, dst, src_format=None, dst_format=None,
               src_subformat=1, dst_subformat=1, groups=1, kernel_name="trans_data"):
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
    if src_format is None:
        src_format = src.get("format").upper().split(":")[0]
    else:
        src_format = src_format.upper()

    if dst_format is None:
        dst_format = dst.get("format").upper().split(":")[0]
    else:
        dst_format = dst_format.upper()

    tbe_context.get_context().add_build_res("pattern", "TransData")
    src_shape = list(src.get("shape", None))
    dst_shape = list(dst.get("shape", None))
    dtype = src.get("dtype").lower()
    
    if (src_format, dst_format) in DSL_SUPPORT:
        is_unknown_rank = is_unknown_rank_input([src, dst])
        is_unknown_shape = is_dynamic_input([src, dst])
        # const and -2 goto DSL
        if dst_format == "FRACTAL_Z_C04" or src_format == "FRACTAL_Z_C04":
            trans_data_c04([src, dst, None], 0, kernel_name = kernel_name)
            return
        if is_unknown_rank or not is_unknown_shape:
            infos = trans_data_normalization(src, dst, src_format, dst_format)
            axes_map = trans_data_format_rule(infos[0][0], infos[0][1], src_format, dst_format)
            trans_data_dsl(infos, axes_map, kernel_name=kernel_name)
            return
        else:
            # shape is -1 include C8 and C16, DSL-Fz only support C8,
            # others support C8 and C16 in DSL.
            c0 = (src_shape, dst_shape)[0 if src_format in PRIVATE_FORMATS else 1][-1]
            is_fz = "FRACTAL_Z" in (src_format, dst_format)
            is_6hd = "NDC1HWC0" in (src_format, dst_format)
            is_b32 = dtype in ["float32", "int32", "uint32"]
            if not is_fz and not is_6hd:
                infos = trans_data_normalization(src, dst, src_format, dst_format)
                axes_map = trans_data_format_rule(infos[0][0], infos[0][1], src_format, dst_format)
                trans_data_dsl(infos, axes_map, kernel_name=kernel_name)
                return
            elif c0 == C8 and is_b32:
                infos = trans_data_normalization(src, dst, src_format, dst_format)
                axes_map = trans_data_format_rule(infos[0][0], infos[0][1], src_format, dst_format)
                trans_data_dsl(infos, axes_map, kernel_name=kernel_name)
                return

    # TIK for TransData
    tbe_context.get_context().add_compile_info("is_tik", True)
    if dst_format == src_format and dst_format.startswith("FRACTAL_Z"):
        return trans_data_groups(src, dst, kernel_name)

    if ((src_format == "NC1HWC0" and dst_format == "NHWC") or
        (src_format == "FRACTAL_NZ" and dst_format in ("ND", "NHWC", "NCHW", "NC1HWC0")) or
        (src_format == "FRACTAL_Z_3D" and dst_format == "NDHWC") or
        (src_format == "NDC1HWC0" and dst_format == "NDHWC")):
        trans_data_negative_target_tc(src, dst, src_format, dst_format, kernel_name)
    elif (((src_format == "NC1HWC0" and dst_format == "NCHW") or
           (src_format == "FRACTAL_Z_3D" and dst_format == "NCDHW") or
           (src_format == "NDC1HWC0" and dst_format == "NCDHW") or
           ((src_format in ("FRACTAL_Z", "FRACTAL_ZN")) and (dst_format == "HWCN")) or
           ((src_format in ("FRACTAL_Z", "FRACTAL_ZN")) and (dst_format == "NCHW")) or
           ((src_format in ("FRACTAL_Z", "FRACTAL_ZN")) and (dst_format == "ND")) or
           (src_format == "FRACTAL_Z_3D" and dst_format == "DHWCN"))):
        trans_data_negative_target_ntc(src, dst, src_format, dst_format, kernel_name)
    elif is_do_with_transpose_formats(src_format, dst_format):
        x_dtype = src.get("dtype").lower()
        y_dtype = dst.get("dtype").lower()
        tik_inst = tik.Tik()
        data_in = tik_inst.Tensor(x_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_in")
        data_out = tik_inst.Tensor(y_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_out")
        data_workspace = tik_inst.Tensor(y_dtype, (1024, ), tik.scope_gm, "data_workspace", is_workspace=True)
        data_tiling = tik_inst.Tensor("int64", (Constant.TILING_MAX_SIZE_GM,), tik.scope_gm, "data_tiling")
        tensor_list = [data_in, None, data_out, data_workspace, data_tiling]
        input_list = [data_in]
        transpose_instance = Transpose(tik_inst, x_dtype, tensor_list, kernel_name)
        return transpose_instance.compute(input_list)
    elif is_do_with_positive_source_ntc_100(src_format, dst_format):
        trans_data_positive_source_ntc(src, dst, src_format, dst_format, kernel_name)
    else:
        trans_data_positive_source_tc(src, dst, src_format, dst_format, kernel_name)


def _nchw_to_5hd(input_x):
    """
    trans nchw to nc1hwc0
    """
    input_dtype = input_x.dtype
    input_shape = shape_util.shape_to_list(input_x.shape)
    if len(input_shape) == NCHW_LENTH:
        shape_n, shape_c, shape_h, shape_w = input_shape
    else:
        shape_n, shape_c, shape_h, shape_w = shape_util.shape_to_list(input_x.op.attrs["shape"])
    shape_c0 = tbe_platform.CUBE_MKN[input_dtype]["mac"][1]
    shape_c1 = (shape_c + shape_c0 - 1) // shape_c0

    input_align_shape = (shape_n, shape_c1 * shape_c0, shape_h * shape_w)
    reshape_shape = (shape_n, shape_c1, shape_c0, shape_h * shape_w)
    transpose_shape = (shape_n, shape_c1, shape_h * shape_w, shape_c0)
    output_shape = (shape_n, shape_c1, shape_h, shape_w, shape_c0)
    output_attrs = copy.deepcopy(input_x.op.attrs)
    output_attrs["shape"] = output_shape
    output_attrs["ori_format"] = "NCHW"

    if len(input_shape) == NCHW_LENTH:
        input_ub = tvm.compute(input_align_shape,
                               lambda n, c, hw: tvm.select(c < shape_c, input_x(n, c, hw // shape_w, hw % shape_w)),
                               name="input_ub_td"
                              )
    else:
        input_ub = tvm.compute(input_align_shape,
                               lambda n, c, hw: tvm.select(c < shape_c, input_x(n, c, hw)),
                               name="input_ub_td"
                              )
    input_ub_pad = tvm.compute(input_align_shape,
                               lambda n, c, hw: tvm.select(c >= shape_c, tvm.const(0, input_dtype)),
                               name="input_ub_pad"
                              )
    input_ub_vn = tvm.compute(input_align_shape,
                              lambda n, c, hw: input_ub(n, c, hw) + input_ub_pad(n, c, hw),
                              name="input_ub_vn"
                             )
    reshape_c = tvm.compute(reshape_shape,
                            lambda n, c1, c0, hw: input_ub_vn(n, c1*shape_c0 + c0, hw),
                            name="reshape_c"
                           )
    transpose_hw_c0 = tvm.compute(transpose_shape,
                                  lambda n, c1, hw, c0: reshape_c(n, c1, c0, hw),
                                  name="transpose_hw_c0"
                                 )
    res = tvm.compute(output_shape,
                      lambda n, c1, h, w, c0: transpose_hw_c0(n, c1, h*shape_w + w, c0),
                      name="split_hw",
                      tag="NCHW_trans_5HD",
                      attrs=output_attrs
                      )
    return res


def _nhwc_to_5hd(input_x):
    """
    trans nhwc to nc1hwc0
    """
    dtype_input = input_x.dtype
    shape_input = shape_util.shape_to_list(input_x.shape)
    if len(shape_input) == NCHW_LENTH:
        shape_n, shape_h, shape_w, shape_c = shape_input
    else:
        shape_n, shape_h, shape_w, shape_c = shape_util.shape_to_list(input_x.op.attrs["shape"])
    shape_c0 = tbe_platform.CUBE_MKN[dtype_input]["mac"][1]
    shape_c1 = (shape_c + shape_c0 - 1) // shape_c0

    shape_input_aligned = (shape_n, shape_h * shape_w, shape_c1 * shape_c0)
    shape_splited = (shape_n, shape_h * shape_w, shape_c1, shape_c0)
    shape_transposed = (shape_n, shape_c1, shape_h * shape_w, shape_c0)
    shape_output = (shape_n, shape_c1, shape_h, shape_w, shape_c0)
    attrs_output = copy.deepcopy(input_x.op.attrs)
    attrs_output["shape"] = shape_transposed
    attrs_output["ori_format"] = "NHWC"

    if len(shape_input) == NCHW_LENTH:
        input_ub = tvm.compute(shape_input_aligned,
                               lambda n, hw, c: tvm.select(c < shape_c, input_x(n, hw // shape_w, hw % shape_w, c)),
                               name="input_ub_td")
    else:
        input_ub = tvm.compute(shape_input_aligned,
                               lambda n, hw, c: tvm.select(c < shape_c, input_x(n, hw, c)),
                               name="input_ub_td")
    input_ub_pad = tvm.compute(shape_input_aligned,
                               lambda n, hw, c: tvm.select(c >= shape_c, tvm.const(0, dtype_input)),
                               name="input_ub_pad")
    input_ub_vn = tvm.compute(shape_input_aligned,
                              lambda n, hw, c: input_ub(n, hw, c) + input_ub_pad(n, hw, c),
                              name="input_ub_vn")
    reshape_c = tvm.compute(shape_splited,
                            lambda n, hw, c1, c0: input_ub_vn(n, hw, c1*shape_c0 + c0),
                            name="reshape_c")
    transpose_hw_c1 = tvm.compute(shape_transposed,
                                  lambda n, c1, hw, c0: reshape_c(n, hw, c1, c0),
                                  name="transpose_hw_c1")
    output = tvm.compute(shape_output,
                         lambda n, c1, h, w, c0: transpose_hw_c1(n, c1, h * shape_w + w, c0),
                         name="split_hw",
                         tag="NHWC_to_5HD_fusion",
                         attrs=attrs_output)
    return output


def _nc1hwc0_to_nchw(src, dst):
    """
    algorithm: trans nc1hwc0 to nchw

    Parameters
    ----------
    src : Tensor, Tensor of input

    dst: dict, shape and dtype of output, should be same shape and type as input

    Returns
    -------
    Tensor
    """
    src_n, src_c1, src_hw, src_c0 = src.shape
    remove_pad_flag = False
    if src.op.tag == "conv2d_backprop_input":
        real_c = get_te_var("dx_c").get_tvm_var()
    elif src.op.name == "invalid_conv2d_rmpad":
        real_c = get_te_var("c_out").get_tvm_var()
    elif src.op.tag == "convolution_C":
        real_c = get_te_var("c_out").get_tvm_var()
        src = src.op.input_tensors[0]
        remove_pad_flag = True
    else:
        real_c = dst.get("ori_shape")[1]
    transpose_shape = (src_n, src_c1, src_c0, src_hw)
    transpose_tensor = tvm.compute(
        transpose_shape,
        lambda n_idx, c1_idx, c0_idx, hw_idx:
            src(n_idx, c1_idx, hw_idx, c0_idx),
        name="transpose")
    dst_shape = (src_n, real_c, src_hw)
    dst_tensor = tvm.compute(
        dst_shape,
        lambda n_idx, c_idx, hw_idx:
            transpose_tensor(n_idx, c_idx // src_c0, c_idx % src_c0, hw_idx),
        name="res_nchw",
        tag="5HD_TRANS_NCHW")

    if remove_pad_flag:
        dst_tensor = conv2d_data_rm_compute(dst_tensor)

    return dst_tensor


def _nc1hwc0_to_nhwc(src, dst):
    """
    algorithm: trans nc1hwc0 to nhwc

    Parameters
    ----------
    src : Tensor, Tensor of input

    dst: dict, shape and dtype of output, should be same shape and type as input

    Returns
    -------
    Tensor
    """
    src_n, src_c1, src_hw, src_c0 = src.shape
    if src.op.tag == "conv2d_backprop_input":
        real_c = get_te_var("dx_c").get_tvm_var()
    else:
        real_c = dst.get("ori_shape")[-1]
    transpose_shape = (src_n, src_hw, src_c1, src_c0)
    transpose_tensor = tvm.compute(
        transpose_shape,
        lambda n_idx, hw_idx, c1_idx, c0_idx:
            src(n_idx, c1_idx, hw_idx, c0_idx),
        name="transpose")
    dst_shape = (src_n, src_hw, real_c)
    dst_tensor = tvm.compute(
        dst_shape,
        lambda n_idx, hw_idx, c_idx:
            transpose_tensor(n_idx, hw_idx, c_idx // src_c0, c_idx % src_c0),
        name="res_nhwc",
        tag="5HD_to_NHWC_fusion")

    return dst_tensor


@register_operator_compute("TransData", op_mode="dynamic", support_fusion=True)
def trans_data_fusion_compute(src, dst, src_format=None, dst_format=None,
                              src_subformat=1, dst_subformat=1, groups=1, kernel_name="trans_data"):
    """
    algorithm: format_transfer
    used for format transformation , only support transfer between NHWC and NC1HWC0
    Parameters
    ----------
    src : tvm.tensor
    input_tenor
    dst: dict
    shape and dtype of output, should be same shape and type as input
    src_format: str
    source data format, can be NCHW and NC1HWC0, default value is None
    dst_format: str
    target data format, can be NC1HWC0 and NCHW, default value is None
    groups: int
    default 1
    kernel_name: str
    kernel name, default value is "trans_data"

    Returns
    -------
    then tensor after transformation
    """
    if src_format is None:
        src_format = src.op.attrs["format"].upper().split(":")[0]
    else:
        src_format = src_format.upper()

    if dst_format is None:
        dst_format = dst.get("format").upper().split(":")[0]
    else:
        dst_format = dst_format.upper()

    fusion_util.check_fusion_input([src])

    support_out2l1_nd2nz = tbe_platform.intrinsic_check_support("Intrinsic_data_move_out2l1_nd2nz")
    # ND/NHWC to NZ/5HD/FZ is supported from out to L1
    if support_out2l1_nd2nz:
        if src_format == "NHWC" and dst_format == "NC1HWC0":
            shape_input = shape_util.shape_to_list(src.shape)
            if len(shape_input) == NCHW_LENTH:
                src_n, src_h, src_w, src_c = shape_input
            else:
                src_n, src_h, src_w, src_c = shape_util.shape_to_list(src.op.attrs["shape"])
            dst_c0 = tbe_platform.CUBE_MKN[src.dtype]["mac"][1]
            dst_c1 = util_common.ceil(src_c, dst_c0)
            dst_shape = (src_n, dst_c1, src_h, src_w, dst_c0)
            dst_tensor = tvm.compute(
                dst_shape, lambda n_idx, c1_idx, h_idx, w_idx, c0_idx: tvm.select(
                    tvm.any(c1_idx * dst_c0 + c0_idx < src_c),
                    src(n_idx, h_idx, w_idx, c1_idx * dst_c0 + c0_idx)),
                name="res_nc1hwc0",
                attrs={"ori_format": "NHWC", "ori_shape": src.shape, "format": "NC1HWC0"},
                tag="NHWC_trans_5HD"
            )
            return dst_tensor
        elif src_format == "NHWC" and dst_format == "FRACTAL_Z":
            dst_n0 = tbe_platform.CUBE_MKN[src.dtype]["mac"][2]
            src_n, src_h, src_w, src_c = tuple(i.value for i in src.shape)
            dst_n1 = util_common.ceil(src_n, dst_n0)
            dst_c0 = tbe_platform.CUBE_MKN[src.dtype]["mac"][1]
            dst_c1 = util_common.ceil(src_c, dst_c0)
            dst_shape = dst_c1 * src_h * src_w, dst_n1, dst_n0, dst_c0
            hw = src_h * src_w
            dst_tensor = tvm.compute(
                dst_shape, lambda  i, j, k, l: src(
                    j * dst_n0 + k,
                    (i % hw) // src_w, (i % hw) % src_w,
                    (i // hw) * dst_c0 + l),
                name="res_fractal_z_weight",
                attrs={"ori_format": "NHWC", "ori_shape": src.shape},
                tag="NHWC_trans_FZ"
            )
            return dst_tensor
        elif src_format in ["ND", "NHWC"] and dst_format == "FRACTAL_NZ":
            src_shape = tuple(i.value for i in src.shape)
            ori_shape = src.op.attrs["ori_shape"] if "ori_shape" in src.op.attrs else src.shape
            src.op.attrs["src_format"] = "ND"
            block_reduce = tbe_platform.CUBE_MKN[src.dtype]["mac"][1]
            block_size = tbe_platform.BLOCK_IN
            dst_shape = (
                util_common.ceil(src_shape[-1], block_reduce),
                util_common.ceil(src_shape[-2], block_size),
                block_size,
                block_reduce
            )
            dst_shape = src_shape[:-2] + dst_shape
            dst_tensor = tvm.compute(
                dst_shape,
                lambda *indices: tvm.select(
                    tvm.all((indices[-4] * block_reduce + indices[-1]) < src_shape[-1],
                            (indices[-3] * block_reduce + indices[-2]) < src_shape[-2]),
                    src(*indices[:-4],
                        indices[-3] * block_size + indices[-2],
                        indices[-4] * block_reduce + indices[-1])
                ),
                name=src.name + "_fractal",
                attrs={"ori_format": "ND", "ori_shape": ori_shape, "format": dst_format},
                tag="ND_trans_NZ"
            )
            return dst_tensor

    if src_format == "NCHW" and dst_format == "NC1HWC0":
        return _nchw_to_5hd(src)
    elif src_format == "NHWC" and dst_format == "NC1HWC0":
        return _nhwc_to_5hd(src)
    elif src_format == "NC1HWC0" and dst_format == "NCHW":
        return _nc1hwc0_to_nchw(src, dst)
    elif src_format == "NC1HWC0" and dst_format == "NHWC":
        return _nc1hwc0_to_nhwc(src, dst)
    else:
        error_manager_vector.raise_err_specific_reson(
            "trans_data", "only support format transfer between NCHW and NC1HWC0 or between NHWC and NC1HWC0"
        )
