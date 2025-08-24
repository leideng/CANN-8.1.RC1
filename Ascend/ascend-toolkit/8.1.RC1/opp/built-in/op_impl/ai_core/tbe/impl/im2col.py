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
image_to_col
"""
import functools
import math
import os
import re
import stat
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tf_get_windowed_output_size_verbose_v2
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import buildcfg
from tbe.dsl.base import operation
from tbe.dsl.base.operation import get_context
from tbe.common.utils import op_tiling
from impl.util.util_common import write_code
from impl.util.util_common import check_load3d_w_out_1_support
from impl.im2col_common_func import im2col_compute
from impl.im2col_common_func import im2col_schedule
from impl.dynamic.extract_image_patches_without_cbuf import ExtractImagePatchesWithoutCbuf
from impl.dynamic.extract_image_patches import param_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_context
from tbe.common.register import register_operator


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
    SIZE_L1 = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)


def check_supported(images, y, ksizes, strides, dilates, padding_mode, pads, kernel_name="im2col"):
    kernel_h, kernel_w = ksizes
    stride_h = strides[0]
    stride_w = strides[0]
    if len(strides) > 1:
        stride_h = strides[0]
        stride_w = strides[1]
    dilate_h = dilates[0]
    dilate_w = dilates[0]
    if len(dilates) > 1:
        dilate_h = dilates[0]
        dilate_w = dilates[1]
 
    if (kernel_h >= 256) or (kernel_w >= 256):
        reason = "kernel_h and kernel_w can not >= 256!"
        return False, reason
    if (stride_h >= 64) or (stride_w >= 64):
        reason = "stride_h and stride_w can not >= 64!"
        return False, reason
    if (dilate_h >= 256) or (dilate_w >= 256):
        reason = "dilate_h and dilate_w can not >= 256!"
        return False, reason

    return True, ""


def im2col_for_unsupport_directive(images, y, ksizes, strides, dilates, padding_mode, pads, kernel_name="im2col"):
    data_format = images.get("ori_format")
    if len(ksizes) == 2:
        ksizes = (1,) + tuple(ksizes) + (1,) if data_format == "NHWC" else (1, 1) + tuple(ksizes)
    else:
        error_manager_vector.raise_err_check_params_rules(kernel_name, 'input params invalide',
                                                          ['ksizes'], [ksizes])
    if len(strides) != 1:
        strides = (1,) + tuple(strides) + (1,) if data_format == "NHWC" else (1, 1) + tuple(strides)
    else:
        strides = (1, strides[0], strides[0], 1) if data_format == "NHWC" else (1, 1, strides[0], strides[0])
    if len(dilates) != 1:
        dilates = (1,) + tuple(dilates) + (1,) if data_format == "NHWC" else (1, 1) + tuple(dilates)
    else:
        dilates = (1, dilates[0], dilates[0], 1)if data_format == "NHWC" else (1, 1, dilates[0], dilates[0])
    
    images["range"] = [[i, i] for i in images["shape"]]
    shape_input_4d = images.get("ori_shape")
    shape_input_5d = images.get("shape")
    shape_range = images.get("range")
    dtype_input = images.get("dtype").lower()

    if dtype_input not in ("int8", "uint8", "float16", "float", "float32"):
        error_manager_vector.raise_err_specific_reson(kernel_name, 
                                                      "dtype can only be uint8, int8, float16, float or float32!")
    if dtype_input in ('int8', 'uint8'):
        align_block_size = Constant.BLOCK_SIZE_INT8
    else:
        align_block_size = Constant.BLOCK_SIZE

    data_format = images.get('ori_format')
    format_list = ('NHWC', 'NCHW')
    if data_format not in format_list:
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", format_list, data_format)
    if len(ksizes) != 4 or len(strides) != 4 or len(dilates) != 4:
        error_manager_vector.raise_err_check_params_rules(kernel_name, 'input params invalide',
                                                          ['ksizes', 'strides', 'dilates', 'shape_input_4d'],
                                                          [ksizes, strides, dilates, shape_input_4d])

    if len(shape_input_4d) != 4 or len(shape_range) != 5:
        error_manager_vector.raise_err_check_params_rules(kernel_name, 'input shape or range invalide',
                                                          ['shape_range', 'shape_input_4d'],
                                                          [shape_range, shape_input_4d])

    if images.get("ori_range"):
        ori_shape_range = [list(sub_range) for sub_range in images.get("ori_range")]

    # NCHW -> NHWC
    if data_format == 'NCHW':
        shape_input_4d = [shape_input_4d[0], shape_input_4d[2], shape_input_4d[3], shape_input_4d[1]]
        ksizes = [ksizes[0], ksizes[2], ksizes[3], ksizes[1]]
        strides = [strides[0], strides[2], strides[3], strides[1]]
        dilates = [dilates[0], dilates[2], dilates[3], dilates[1]]
        if images.get("ori_range"):
            ori_shape_range = [list(ori_shape_range[0]), list(ori_shape_range[2]),
                               list(ori_shape_range[3]), list(ori_shape_range[1])]
    param_check(ksizes, strides, dilates, kernel_name)

    if not images.get("ori_range"):
        ori_shape_range = [list(shape_range[0]), list(shape_range[2]), list(shape_range[3]), list(shape_range[1])]
        if shape_input_4d[-1] < 0:
            ori_shape_range[-1][0] = (ori_shape_range[-1][0] - 1) * align_block_size
            if ori_shape_range[-1][-1] is not None:
                ori_shape_range[-1][-1] = (ori_shape_range[-1][-1] * align_block_size)

    pads = pads if len(pads) != 1 else pads * 4

    input_list = [{
            "shape": shape_input_5d,
            "ori_shape": shape_input_4d,
            "format": "NC1HWC0",
            "ori_format": "NHWC",
            "dtype": dtype_input
        }]
    output_list = []
    attr_list = [{"name": "ksizes", "dtype": "list_int", "value": ksizes},
                {"name": "strides", "dtype": "list_int", "value": strides},
                {"name": "rates", "dtype": "list_int", "value": dilates},
                {"name": "padding", "dtype": "str", "value": padding_mode}]
    compile_info = {
        "envWithoutCbuf": True,
        "socVersion": tbe_platform.get_soc_spec("SHORT_SOC_VERSION"),
        "coreNum": tbe_platform.get_soc_spec("CORE_NUM"),
        "SIZE_L1": tbe_platform.get_soc_spec("L1_SIZE"),
        "SIZE_UB": tbe_platform.get_soc_spec("UB_SIZE"),
        "dtypeInput": dtype_input,
        "paddingType": padding_mode,
        "pads": pads,
        "isDB": True,
        "isVar": False,
        "isConst": True,
        "isBinary": False
    }
    run_info = op_tiling.do_op_tiling("Im2col",
                                      compile_info, input_list, output_list, None, None, attr_list)
    operation.get_context().add("run_info", run_info)
    operation.get_context().add("is_const_shape", True)
    operation.get_context().add("is_binary", False)

    schedules, tensors = [], []
    pads = None if padding_mode != "CALCULATED" else pads
    context = tbe_context.op_context.get_context()
    if context is not None:
        context.set_op_mode("static")
        eipwc_obj = ExtractImagePatchesWithoutCbuf(shape_input_4d, dtype_input, ksizes, strides, dilates, padding_mode,
                                                   kernel_name, pads)
    else:
        with tbe_context.op_context.OpContext("static"):
            eipwc_obj = ExtractImagePatchesWithoutCbuf(shape_input_4d, dtype_input, ksizes, strides,
                                                       dilates, padding_mode, kernel_name, pads)
    tensor_list, sch = eipwc_obj.do_without_cbuf(True)
    tensors.append(tensor_list)
    schedules.append(sch)
    eipwc_obj.add_compile_info()
    tbe.build(schedules, {"name": kernel_name, "tensor_list": tensors})
    return


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments
def image_to_col_compute(fmap, c_in_real, ksizes, strides, dilates, pads, padding_mode):
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

    Returns
    -------
    compute results
    """
    # fmap's format is NC1HWC0
    fmap_shape = fmap.shape

    fmap_h = fmap_shape[2].value
    fmap_w = fmap_shape[3].value

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

    padding_h_top = pads[0]
    padding_h_bottom = pads[0]
    padding_w_before = pads[0]
    padding_w_after = pads[0]
    if len(pads) != 1:
        padding_h_top, padding_h_bottom, padding_w_before, padding_w_after = pads

    out_h = int((fmap_h + padding_h_top + padding_h_bottom - (dilate_h * (kernel_h - 1) + 1)) / stride_h + 1)
    out_w = int((fmap_w + padding_w_before + padding_w_after - (dilate_w * (kernel_w - 1) + 1)) / stride_w + 1)

    if padding_mode in ("SAME", "VALID"):
        out_h, padding_h_top, padding_h_bottom = tf_get_windowed_output_size_verbose_v2(
        fmap_h, kernel_h, dilate_h, stride_h, padding_mode)
        out_w, padding_w_before, padding_w_after = tf_get_windowed_output_size_verbose_v2(
            fmap_w, kernel_w, dilate_w, stride_w, padding_mode)

    pads = (padding_h_top, padding_h_bottom, padding_w_before, padding_w_after)

    output_res, workspace_res, workspace_shape = im2col_compute(
        fmap, c_in_real, ksizes, strides, dilates, pads, out_h, out_w)

    return output_res, workspace_res, workspace_shape


def op_select_format(images, y, ksizes, strides, dilations, padding_mode, pads, kernel_name="im2col"):
    """
    select format dynamicically
    """
    if util_common.is_unknown([images]):
        x_format_list = "NC1HWC0,NC1HWC0,NC1HWC0"
        x_unkownshape_format_list = "NCHW,NCHW,NCHW"
        y_format_list = "NHWC,NHWC,NHWC"
        y_unkownshape_format_list = "NCHW,NCHW,NCHW"

        x_dtype_list = "float16,float,bfloat16"
        y_dtype_list = "float16,float,bfloat16"
    else:
        x_format_list = "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0"
        x_unkownshape_format_list = "NCHW,NCHW,NCHW,NCHW"
        y_format_list = "NHWC,NHWC,NHWC,NHWC"
        y_unkownshape_format_list = "NCHW,NCHW,NCHW,NCHW"

        x_dtype_list = "float16,float,int8,uint8"
        y_dtype_list = "float16,float,int8,uint8"

    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                           datatype=x_dtype_list,
                                           format=x_format_list,
                                           unknownshape_format=x_unkownshape_format_list)
    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                           datatype=y_dtype_list,
                                           format=y_format_list,
                                           unknownshape_format=y_unkownshape_format_list)
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def check_unsupport_float32_soc(images):
    dtype_is_float32 = (images.get("dtype") == "float32")
    unsupport_l0c2out = not tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    return (dtype_is_float32 and unsupport_l0c2out)


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,too-many-statements,too-many-locals
# 'pylint: disable=too-many-branches
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_LIST_INT, para_check.KERNEL_NAME)
def im2col(images, y, ksizes, strides, dilations, padding_mode, pads, kernel_name="im2col"):
    """.
    calculating data

    Parameters
    ----------
    images : dict
        shape and dtype of input, support float16, float32
    y : dict
        shape and dtype of output, should be same shape and type as input
    ksizes: input attr
    strides: input attr
    dilations: input attr
    pads: input attr
    kernel_name : str
        kernel name, default value is "image_to_col"

    Returns
    -------
    None
    """
    if (not tbe_platform.intrinsic_check_support("Intrinsic_data_move_l12ub")) or check_unsupport_float32_soc(images):
        return im2col_for_unsupport_directive(images, y, ksizes, strides, dilations, padding_mode, pads, kernel_name)

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
    if len(ksizes) != 2:
        error_manager_vector.raise_err_check_params_rules(kernel_name, 'input params invalide',
                                                          ['ksizes'], [ksizes])
    # NCHW -> NHWC
    if data_format == 'NCHW':
        shape_input_4d = (shape_input_4d[0], shape_input_4d[2], shape_input_4d[3], shape_input_4d[1])

    fmap_n, fmap_h, fmap_w, fmap_c = shape_input_4d
    fmap_c1 = (fmap_c + align_block_size - 1) // align_block_size
    fmap_c0 = align_block_size
    shape_input = (fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0)

    kernel_h, kernel_w = ksizes
    stride_h = strides[0]
    stride_w = strides[0]
    if len(strides) != 1:
        stride_h = strides[0]
        stride_w = strides[1]
    dilate_h = dilations[0]
    dilate_w = dilations[0]
    if len(dilations) != 1:
        dilate_h = dilations[0]
        dilate_w = dilations[1]

    padding_h_top = pads[0]
    padding_h_bottom = pads[0]
    padding_w_before = pads[0]
    padding_w_after = pads[0]
    if len(pads) != 1:
        padding_h_top, padding_h_bottom, padding_w_before, padding_w_after = pads

    if (kernel_h >= 256) or (kernel_w >= 256):
        error_manager_vector.raise_err_specific_reson(kernel_name, "kernel_h and kernel_w can not >= 256!")
    if (stride_h >= 64) or (stride_w >= 64):
        error_manager_vector.raise_err_specific_reson(kernel_name, "stride_h and stride_w can not >= 64!")
    if (dilate_h >= 256) or (dilate_w >= 256):
        error_manager_vector.raise_err_specific_reson(kernel_name, "dilate_h and dilate_w can not >= 256!")

    out_h = int((fmap_h + padding_h_top + padding_h_bottom - (dilate_h * (kernel_h - 1) + 1)) / stride_h + 1)
    out_w = int((fmap_w + padding_w_before + padding_w_after - (dilate_w * (kernel_w - 1) + 1)) / stride_w + 1)

    if padding_mode in ("SAME", "VALID"):
        out_h, padding_h_top, padding_h_bottom = tf_get_windowed_output_size_verbose_v2(
            fmap_h, kernel_h, dilate_h, stride_h, padding_mode)
        out_w, padding_w_before, padding_w_after = tf_get_windowed_output_size_verbose_v2(
            fmap_w, kernel_w, dilate_w, stride_w, padding_mode)

    if (out_h <= 0) or (out_w <= 0):
        error_manager_vector.raise_err_specific_reson(kernel_name, "out_h and out_w can not <= 0!")
    if (padding_h_top >= 256) or (padding_h_bottom >= 256):
        error_manager_vector.raise_err_specific_reson(kernel_name, "padding_h_top and padding_h_bottom can not >= 256!")
    if (padding_w_before >= 256) or (padding_w_after >= 256):
        error_manager_vector.raise_err_specific_reson(kernel_name,
                                                      "padding_w_before and padding_w_after can not >= 256!")

    if not check_load3d_w_out_1_support() and (out_h != 1 and out_w == 1):
        if fmap_w + padding_w_before + padding_w_after - ((kernel_w - 1) * dilate_w + 1) < stride_w:
            return im2col_for_unsupport_directive(
                images, y, ksizes, strides, dilations, padding_mode, pads, kernel_name)

    # min cut_h
    dilated_kernel_h = (kernel_h - 1) * dilate_h + 1
    cut_h_col = (Constant.BLOCK_SIZE // math.gcd(out_w, Constant.BLOCK_SIZE) - 1) * stride_h + 1 + dilated_kernel_h // 2
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
    output_res, workspace_res, _ = image_to_col_compute(
        data_input, fmap_c, ksizes, strides, dilations, pads, padding_mode)
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

    with buildcfg.build_config():
        tvm.build(sch, [data_input, output_res, workspace_res], "cce", name=kernel_name)
        if fmap_c % align_block_size != 0:
            _write_workspace_info([workspace_res], kernel_name)
