#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
conv3d_backprop_input_d
"""
import impl.dynamic as dyn_impl
from impl.util import util_common
from impl.util import util_conv3d
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.util_cube_dynamic import generalize_shape_and_range_inplace

from tbe.common.context import op_context
from tbe.common.utils import const
from tbe.common.utils import log

from tbe.dsl.compute.cube_util import Conv3DConsts
from tbe.dsl.compute.cube_util import extract_tensor_info
from tbe.dsl.compute.cube_util import Load3DParam
from tbe.dsl.unify_schedule.conv3d_bp_input_tilingcase import Conv3dBackpropInputBinaryParaProcess

# h,w pad must be in [0, 255], d pad must be in [0, 2147483647]
_PAD_HW_MAX = 255
_PAD_D_MAX = 2147483647

# fmapH, fmapW must be in [1,4096]
_FMAP_HW_MIN = 1
_FMAP_HW_MAX = 4096

# DeDy H,W must be in [1,4096]
_DEDY_HW_MIN = 1
_DEDY_HW_MAX = 4096

# filterH, filterW must be in [1,255]
_FILTER_HW_SIZE = 256

# stride must be in [1,63] and h*w not larger than 256
_STRIDE_SIZE_MAX = 256
_STRIDE_SIZE_HWD_MAX = 343

# the max num of each axis of shape
_DEFAULT_MAX_SHAPE_NUM = 1000000

# the bytes length of several dtype
_BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2, "bfloat16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
_DATA_SIZE_MAX = 9223372036854775807

# align with 16 for chips
_C0_SIZE = tbe_platform.C0_SIZE
_BLOCK_SIZE = 16

_L1FUSION_INPUT_CTR = 2

_OUT_BACKPROP_TARGET_FORMAT = "NDHWC"
_OUT_BACKPROP_FORMAT_WHITE_LIST = ["NDHWC", "NCDHW"]
_FILTER_TARGET_FORMAT = "DHWCN"
_FILTER_FORMAT_WHITE_LIST = ["DHWCN", "NDHWC", "NCDHW"]
_RES_TARGET_FORMAT = "NDHWC"
_RES_FORMAT_WHITE_LIST = ["NDHWC", "NCDHW"]


def get_op_support_info(filters, # pylint: disable=R0913,R0914
                        out_backprop,
                        y_input,
                        input_size,
                        strides,
                        pads,
                        dilations=(1, 1, 1, 1, 1),
                        groups=1,
                        data_format="NDHWC",
                        kernel_name="conv3d_backprop_input",
                        op_slice_info=""):
    """
    algorithm: get_op_support_info

    Parameters
    ----------
    filters: A dict with keys(shape and dtype)
    Input weight tensor

    out_backprop: A dict with keys(shape and dtype)
    The shape of gradients

    y_input: A dict with keys(shape and dtype)
    conv3d_backprop_input output tensor, dtype must be assigned

    input_sizes: The shape of feature map
    5-D with shape [batch, depth, height, weight, channels]

    strides: A tuple/list of 5 integers
    Filter move stride

    pads: A tuple/list of 6 integers
    [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
    Filter expand size of dilated conv3d_backprop_input, default value is (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
    Default value is 1

    data_format: The data format of the input and output data
    Default format is "NDHWC"

    kernel_name: Str
    Kernel name, default value is "conv3d_backprop_input"

    op_slice_info: Str
    Default value is ""

    Returns
    -------
    op_cal_info_in_json: A dict with keys(split_maps, reduce_maps, l1_fusion_enable and min_tbe_l1_space)
    """
    def _cal_min_l1space():
        block_size = 16
        w_value = ori_shape_out_backprop[3] * strides[3]
        filter_d_dilation = (ori_shape_filters[0] - 1) * dilations_formated[1] + 1
        filter_h_dilation = (ori_shape_filters[1] - 1) * dilations_formated[2] + 1
        if ori_shape_res[3] > block_size:
            h_value_max = filter_h_dilation + 1
        elif block_size % ori_shape_res[3] == 0:
            h_value_max = filter_h_dilation + block_size // ori_shape_res[3] - 1
        else:
            h_value_max = filter_h_dilation + block_size // ori_shape_res[3] + 1

        a_l1_size = h_value_max * w_value * ((filter_d_dilation - 2) // strides[1] + 2) * block_size * 2
        b_l1_size = ori_shape_filters[0] * ori_shape_filters[1] * ori_shape_filters[2] * block_size * block_size * 2
        return a_l1_size + b_l1_size

    def _get_slice_info():
        overlap_d = -1 if (filter_d_dilation == 1 and strides_formated[1] == 1) else 0
        overlap_h = -1 if (filter_h_dilation == 1 and strides_formated[2] == 1) else 0
        overlap_w = -1 if (filter_w_dilation == 1 and strides_formated[3] == 1) else 0

        # format
        axis_split_matrix = []
        axis_reduce_list = None
        format_out_backprop = out_backprop.get("format")
        if format_out_backprop == "NDC1HWC0":
            # cut N
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [0], [-1], [-1]]),
                                     util_select_op_base.SplitOutput([0, [0]])])
            # cut D
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [1], [overlap_d], [overlap_d]]),
                                     util_select_op_base.SplitOutput([0, [1]])])
            # cut H
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [3], [overlap_h], [overlap_h]]),
                                     util_select_op_base.SplitOutput([0, [3]])])
            # cut W
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [4], [overlap_w], [overlap_w]]),
                                     util_select_op_base.SplitOutput([0, [4]])])
            # cut Cout
            axis_split_matrix.append(
                [util_select_op_base.SplitInput([1, [0], [-1], [-1]]),
                util_select_op_base.SplitOutput([0, [2]])]
            )
        else:
            axis_split_matrix = None

        return axis_split_matrix, axis_reduce_list

    ori_shape_out_backprop = util_conv3d.transform_shape_with_format(out_backprop.get("ori_format"),
                                                                     _OUT_BACKPROP_TARGET_FORMAT,
                                                                     out_backprop.get("ori_shape"),
                                                                     _OUT_BACKPROP_FORMAT_WHITE_LIST)
    strides_formated = util_conv3d.transform_shape_with_format(out_backprop.get("ori_format"),
                                                               _OUT_BACKPROP_TARGET_FORMAT,
                                                               strides,
                                                               _OUT_BACKPROP_FORMAT_WHITE_LIST)

    dilations_formated = util_conv3d.transform_shape_with_format(out_backprop.get("ori_format"),
                                                                 _OUT_BACKPROP_TARGET_FORMAT,
                                                                 dilations,
                                                                 _OUT_BACKPROP_FORMAT_WHITE_LIST)

    if ori_shape_out_backprop is None or strides_formated is None or dilations_formated is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y_backprop',
            'expected_format_list': ",".join(_OUT_BACKPROP_FORMAT_WHITE_LIST),
            'format': out_backprop.get("ori_format")
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    ori_shape_filters = util_conv3d.transform_shape_with_format(filters.get("ori_format"),
                                                                _FILTER_TARGET_FORMAT,
                                                                filters.get("ori_shape"),
                                                                _FILTER_FORMAT_WHITE_LIST)
    if ori_shape_filters is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'filter',
            'expected_format_list': ",".join(_FILTER_FORMAT_WHITE_LIST),
            'format': filters.get("ori_format")
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    ori_shape_res = util_conv3d.transform_shape_with_format(y_input.get("ori_format"),
                                                            _RES_TARGET_FORMAT,
                                                            y_input.get("ori_shape"),
                                                            _RES_FORMAT_WHITE_LIST)
    if ori_shape_res is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y',
            'expected_format_list': ",".join(_RES_FORMAT_WHITE_LIST),
            'format': y_input.get("ori_format")
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    filter_d_dilation = (ori_shape_filters[0] - 1) * dilations_formated[1] + 1
    filter_h_dilation = (ori_shape_filters[1] - 1) * dilations_formated[2] + 1
    filter_w_dilation = (ori_shape_filters[2] - 1) * dilations_formated[3] + 1

    axis_split_info, axis_reduce_info = _get_slice_info()

    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_info,
                                                              axis_reduce_info,
                                                              _L1FUSION_INPUT_CTR,
                                                              _cal_min_l1space())

    return op_cal_info_in_json


def _get_ndhwc_shape(ori_format_filters, ori_shape_filters,
                     ori_format_out_backprop, ori_shape_out_backprop,
                     ori_shape_strides, ori_shape_dialtions,
                     ori_format_res, ori_shape_res):
    def _ncdhw2ndhwc(shape_ncdhw):
        shape_ndhwc = [shape_ncdhw[0], shape_ncdhw[2], shape_ncdhw[3], shape_ncdhw[4], shape_ncdhw[1]]
        return shape_ndhwc

    if ori_format_filters == "DHWCN":
        shape_filters = ori_shape_filters
    elif ori_format_filters == "NDHWC":
        shape_filters = (ori_shape_filters[1],
                         ori_shape_filters[2],
                         ori_shape_filters[3],
                         ori_shape_filters[4],
                         ori_shape_filters[0],
                        )
    elif ori_format_filters == "NCDHW":
        shape_filters = (ori_shape_filters[2],
                         ori_shape_filters[3],
                         ori_shape_filters[4],
                         ori_shape_filters[1],
                         ori_shape_filters[0],
                        )
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'filter',
            'expected_format_list': '[{}, {}, {}]'
                                    .format('DHWCN', 'NDHWC', 'NCDHW'),
            'format': ori_format_filters
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if ori_format_out_backprop == "NDHWC":
        shape_out_backprop = list(ori_shape_out_backprop)
        shape_strides = ori_shape_strides
        shape_dilations = ori_shape_dialtions
    elif ori_format_out_backprop == "NCDHW":
        shape_out_backprop = _ncdhw2ndhwc(ori_shape_out_backprop)
        shape_strides = _ncdhw2ndhwc(ori_shape_strides)
        shape_dilations = _ncdhw2ndhwc(ori_shape_dialtions)
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y_backprop',
            'expected_format_list': '[{}, {}]'.format('NDHWC', 'NCDHW'),
            'format': ori_format_out_backprop
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if ori_format_res == "NDHWC":
        shape_res = ori_shape_res
    elif ori_format_res == "NCDHW":
        shape_res = _ncdhw2ndhwc(ori_shape_res)
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y',
            'expected_format_list': '[{}, {}]'.format('NDHWC', 'NCDHW'),
            'format': ori_format_res
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    shape_out_backprop[-1] = shape_filters[-1]

    return shape_filters, shape_out_backprop, shape_strides, shape_dilations, shape_res


@tbe_platform.fusion_manager.register("conv3d_backprop_input_d")
def conv3d_backprop_input_fusion_compute(filters,
                                         out_backprop, y_input, input_sizes, strides,
                                         pads, dilations=(1, 1, 1, 1, 1), groups=1,
                                         data_format="NDHWC",
                                         kernel_name="conv3d_backprop_input"):
    """
    algorithm: conv3d_backprop_input_fusion_compute

    Parameters
    ----------
    filters: Input weight tensor

    out_backprop: dedy tensor

    y_input: A dict with keys(shape and dtype)
    Conv3d_backprop_input output tensor, dtype must be assigned

    input_size: The shape of feature map
    5-D with shape [batch, depth, height, weight, channels]

    strides: A tuple/list of 5 integers
    Filter move stride

    pads: A tuple/list of 6 integers
    [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
    filter expand size of dilated conv3d_backprop_input, default value (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
    Default value 1

    data_format: The data format of the input and output data
    Default format "NDHWC"

    kernel_name: Str
    Kernel name, default value is "conv3d_backprop_input"

    Returns
    -------
    dedx
    """
    shape_filter = []
    for i in filters.op.attrs['ori_shape']:
        shape_filter.append(i.value)
    filter_format = filters.op.attrs['ori_format']

    dsl_flag = False
    if isinstance(out_backprop.op, tvm.PlaceholderOp):
        dsl_flag = True

    shape_out_backprop = []
    for i in out_backprop.op.attrs['ori_shape']:
        shape_out_backprop.append(i.value)
    out_backprop_format = out_backprop.op.attrs['ori_format']

    shape_filters, shape_out_backprop, shape_strides, shape_dilations, shape_res = \
        _get_ndhwc_shape(filter_format,
                         shape_filter,
                         out_backprop_format,
                         shape_out_backprop,
                         strides,
                         dilations,
                         data_format,
                         input_sizes)
    filter_dtype = filters.op.dtype
    out_backprop_dtype = out_backprop.dtype
    res_dtype = y_input.get("dtype")
    res = check_conv3dbp_input_params(shape_filters,
                                      shape_out_backprop,
                                      shape_res,
                                      shape_strides,
                                      pads,
                                      groups,
                                      shape_dilations,
                                      filter_dtype,
                                      out_backprop_dtype,
                                      res_dtype,
                                      kernel_name)
    (shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations,
     filter_dtype, out_backprop_dtype, res_dtype, kernel_name, group_dict) = res

    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
    pads = list(pads)

    shape_filter_ncdhw = [filter_batch,
                          filter_channel,
                          filter_depth,
                          filter_h,
                          filter_w]

    para_dict = {
        "strides": strides,
        "pads": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict,
        "dsl_flag": dsl_flag
    }

    dedx = tbe.conv3d_backprop_input(filter=filters,
                                    out_backprop=out_backprop,
                                    filter_size=shape_filter_ncdhw,
                                    input_size=input_sizes,
                                    para_dict=para_dict)
    return dedx


def check_supported(filters,
                    out_backprop,
                    y_input,
                    input_sizes,
                    strides,
                    pads,
                    dilations=(1, 1, 1, 1, 1),
                    groups=1,
                    data_format="NDHWC",
                    kernel_name="conv3d_backprop_input"):
    """
    The H and W dimension of input_sizes should be in range [1, 4096]. \n
    The H and W dimension of dilation should be in range [1, 255]. \n
    The D,H or W dimension of the filter should be in range [1, 255]. \n
    The padding in each dimension should be in range [0, 255]. \n
    The D,H or W dimension of the stride should be in range [1, 63]. \n
    The filter's H * filter 's W should < 256. \n
    The filter's H * W * D should < 343. \n
    The stride's H * W should < 256. \n
    The stride's H * W * D should < 343. \n
    The groups should <= the feature map's and the filter's channel dimension. \n
    The feature map's channel dimension or filter's channel dimension must be divisible by groups. \n
    The channel dimension of feature map should = the filter's channel dimension * groups. \n
    The out_backprop's channel dimension should = the filter's batch dimension. \n
    The feature map's batch dimension should = the out_backprop's batch dimension. \n
    The D,H or W dimension of the feature map after padding should >= the filter's corresponding \
    dimension after dilation. \n
    The out_backprop's H * stride's H should < 4096. \n
    The out_backprop's W * stride's W should < 4096. \n
    If the output H dimension is not 1, the output W dimension should >= 2. \n

    The data in Ubuffer should <= the chip's Ubuffer size. \n
    The data in L1 buffer should <= the chip's L1 buffer size
    """
    ori_shape_filters = filters.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = input_sizes
    ori_shape_strides = strides
    ori_shape_dilations = dilations

    filters_dtype = filters.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y_input.get("dtype")

    ori_format_filters = filters.get("ori_format")
    ori_format_out_backprop = data_format
    ori_format_res = data_format

    try:
        shape_filters, shape_out_backprop, shape_strides, shape_dilations, shape_res = \
        _get_ndhwc_shape(ori_format_filters,
                         ori_shape_filters,
                         ori_format_out_backprop,
                         ori_shape_out_backprop,
                         ori_shape_strides,
                         ori_shape_dilations,
                         ori_format_res,
                         ori_shape_res)
        check_conv3dbp_input_params(shape_filters, shape_out_backprop, shape_res, shape_strides, pads,
                                    groups, shape_dilations, filters_dtype, out_backprop_dtype, res_dtype,
                                    kernel_name)
        return True, ""
    except Exception as e:
        reason = e.args[1]
        return False, reason
    finally:
        pass


def construct_params_for_do_op_tiling(tensors, attrs):
    filters, out_backprop, y = tensors
    input_size, strides, pads, dilations, groups, data_format = attrs
    dyn_inputs = [{
        "name": "input_size",
        "shape": [Conv3DConsts._CONV_BACKPROP_SHAPE_DIM],
        "ori_shape": [Conv3DConsts._CONV_BACKPROP_SHAPE_DIM],
        "dtype": "int32",
        "format": "ND",
        "ori_format": "ND",
        "const_value": input_size
    },
                  extract_tensor_info(filters),
                  extract_tensor_info(out_backprop)]
    dyn_output = [extract_tensor_info(y)]
    dyn_attrs = ({
        "name": "strides",
        "dtype": "list_int",
        "value": strides
    }, {
        "name": "pads",
        "dtype": "list_int",
        "value": pads
    }, {
        "name": "dilations",
        "dtype": "list_int",
        "value": dilations
    }, {
        "name": "groups",
        "dtype": "int",
        "value": groups
    }, {
        "name": "data_format",
        "dtype": "str",
        "value": data_format
    }, {
        "name": "padding",
        "dtype": "str",
        "value": ""
    })

    params = {}
    params['op_type'] = 'Conv3DBackpropInput'
    params['inputs'] = dyn_inputs
    params['outputs'] = dyn_output
    params['attrs'] = dyn_attrs
    op_context.get_context().add_addition('params_do_op_tiling', params)


def enable_binary_constant(context, tensors, attrs, kernel_name):
    context.set_op_mode("static")

    filters, out_backprop, y = tensors
    _, _, _, _, _, data_format = attrs

    input_size = {
        "shape": [Conv3DConsts._CONV_BACKPROP_SHAPE_DIM],
        "ori_shape": [Conv3DConsts._CONV_BACKPROP_SHAPE_DIM],
        "format": "ND",
        "ori_format": "ND"
    }
    generalize_shape_and_range_inplace(filters)
    generalize_shape_and_range_inplace(out_backprop)
    generalize_shape_and_range_inplace(y)
    strides = [-1] * Conv3DConsts._STRIDES_SHAPE_DIM
    pads = [-1] * Conv3DConsts._CONV_BACKPROP_PAD_SHAPE_DIM
    dilations = [-1] * Conv3DConsts._DILATIONS_SHAPE_DIM
    groups = -1

    dyn_impl.conv3d_backprop_input(input_size, filters, out_backprop, y, strides, pads, dilations,
                                   groups, data_format, kernel_name)


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_INT,
    para_check.REQUIRED_ATTR_STR,
    para_check.KERNEL_NAME,
)
def conv3d_backprop_input_d(filters, # pylint: disable=R0913,R0914
                            out_backprop, y_input, input_size, strides,
                            pads, dilations=(1, 1, 1, 1, 1), groups=1,
                            data_format="NDHWC",
                            kernel_name="conv3d_backprop_input"):
    """
    algorithm: Conv3d_backprop_input

    Parameters
    ----------
    filters: A dict with keys(shape and dtype)
    Input weight tensor

    out_backprop: A dict with keys(shape and dtype)
    Gradients tensor

    y_input: A dict with keys(shape and dtype)
    Conv3d_backprop_input output tensor, dtype must be assigned

    input_size: The shape of feature map
    5-D with shape [batch, depth, height, weight, channels]

    strides: A tuple/list of 5 integers
    Filter move stride

    pads: A tuple/list of 6 integers
    [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
    filter expand size of dilated conv3d_backprop_input, default value (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
    Default value 1

    data_format: The data format of the input and output data
    Default format "NDHWC"

    kernel_name: Str
    Kernel name, default value is "conv3d_backprop_input"

    Returns
    -------
    None
    """

    ori_shape_filters = filters.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = input_size
    ori_shape_strides = strides
    ori_shape_dilations = dilations

    filters_dtype = filters.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y_input.get("dtype")

    ori_format_filters = filters.get("ori_format")
    ori_format_out_backprop = data_format
    ori_format_res = data_format

    ori_tensors = [filters, out_backprop, y_input]
    ori_attrs = [input_size, strides, pads, dilations, groups, data_format]

    shape_filters, shape_out_backprop, shape_strides, shape_dilations, shape_res = \
        _get_ndhwc_shape(ori_format_filters,
                         ori_shape_filters,
                         ori_format_out_backprop,
                         ori_shape_out_backprop,
                         ori_shape_strides,
                         ori_shape_dilations,
                         ori_format_res,
                         ori_shape_res)

    _conv3d_backprop_input_cce(shape_filters, shape_out_backprop, shape_res, shape_strides, pads, groups,
                               shape_dilations, filters_dtype, out_backprop_dtype, res_dtype, kernel_name, ori_tensors,
                               ori_attrs)


@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), (list, tuple), int,
                             (list, tuple), str, str, str, str)
def check_conv3dbp_input_params(shape_filter,# pylint:disable=R0913,R0914,R0915
                                shape_out_backprop,
                                input_sizes, strides, pads, groups, dilations,
                                filter_dtype, out_backprop_dtype,
                                res_dtype, kernel_name):
    """
    The params check function of conv3d backprop input

    Parameters
    -------------------------
    shape_filter : The shape of filter
    5-D with shape (depth, height, weight, batch, channels)

    shape_out_backprop : The shape of gradients
    5-D with shape [batch, depth, height, weight,channels]

    input_sizes : The shape of feature map
    5-D with shape [batch, depth, height, weight, channels]

    strides : A list/tuple of ints. The stride of the sliding window

    pads : A list/tuple of ints

    groups : Int of blocked connections from input channels to output channels

    dilations : An optional list/tuple of ints

    filter_dtype : The dtype of filter data

    out_backprop_dtype : The dtype of gradients data

    res_dtype : The dtype of result(De/Dx) data

    kernel_name : Cce kernel name

    Returns
    -----------------------
    All transformed params
    """
    def _check_attr_range(attr_name, attr_value, attr_min, attr_max):
        if attr_value < attr_min or attr_value > attr_max:
            error_manager_cube.raise_err_attr_range_invalid("conv3d_backprop_input",
                "[{},{}]".format(attr_min, attr_max),
                attr_name,
                str(attr_value))

    def _check_64bits_limitation(attr_name, attr_value, dtype):
        bit_ratio = _BIT_RATIO_DICT.get(dtype)
        if attr_value * bit_ratio > _DATA_SIZE_MAX:
            dict_args = {
                'errCode': 'E60020',
                'attr_name': attr_name,
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_ub_limitation():
        w_value = dedy_w * stride_w

        aub_dedy_size_min = dedy_w * _BLOCK_SIZE * 2
        aub_filling_size_min = w_value * _BLOCK_SIZE * 2
        cub_size_min = _BLOCK_SIZE * _BLOCK_SIZE * _BIT_RATIO_DICT.get(res_dtype)
        ub_size = tbe_platform.get_soc_spec("UB_SIZE")

        if (aub_dedy_size_min + aub_filling_size_min + cub_size_min) > ub_size:
            dict_args = {
                'errCode': 'E60119'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_shape_error():
        fmap_h_padding = fmap_h + pad_up + pad_down
        fmap_w_padding = fmap_w + pad_left + pad_right
        fmap_d_padding = fmap_deep + pad_head + pad_tail
        # Check Batch Dimension
        if fmap_channel != filter_channel * groups:
            error_manager_cube.raise_err_specific("conv3d_backprop_input",
                    "Shape error: Fmap's C must be equal to Filter'C * groups.")

        if dedy_channel != filter_batch:
            error_manager_cube.raise_err_specific("conv3d_backprop_input",
                    "Shape error: Dedy's C must be equal to Filter'N.")

        if fmap_batch != dedy_batch:
            error_manager_cube.raise_err_two_paras('E62503', 'conv3d_backprop_input',
                str(dedy_batch), str(fmap_batch))

        # Check HWD dimension
        if filter_h_dilation > fmap_h_padding:
            error_manager_cube.raise_err_three_paras('E62507', 'conv3d_backprop_input', 'H',
                str(filter_h_dilation), str(fmap_h_padding))

        if filter_w_dilation > fmap_w_padding:
            error_manager_cube.raise_err_three_paras('E62507', 'conv3d_backprop_input', 'W',
                str(filter_w_dilation), str(fmap_w_padding))

        if filter_d_dilation > fmap_d_padding:
            error_manager_cube.raise_err_three_paras('E62507', 'conv3d_backprop_input', 'D',
                str(filter_d_dilation), str(fmap_d_padding))

        if ((fmap_h_padding - filter_h_dilation) // stride_h + 1) != dedy_h:
            dict_args = {
                'errCode': 'E60024',
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))
        if ((fmap_w_padding - filter_w_dilation) // stride_w + 1) != dedy_w:
            dict_args = {
                'errCode': 'E60025',
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))
        if ((fmap_d_padding - filter_d_dilation) // stride_d + 1) != dedy_deep:
            dict_args = {
                'errCode': 'E62508',
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    # Base check, Mainly required by interface appearance
    # ===========================================================
    # para_check check
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_filter, Conv3DConsts._CONV_BACKPROP_SHAPE_DIM,
                                Conv3DConsts._CONV_BACKPROP_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(shape_out_backprop, Conv3DConsts._CONV_BACKPROP_SHAPE_DIM,
                                Conv3DConsts._CONV_BACKPROP_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(input_sizes, Conv3DConsts._CONV_BACKPROP_SHAPE_DIM,
                                Conv3DConsts._CONV_BACKPROP_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)
    for i in range(len(list(strides))):
        if list(strides)[i] < 1:
            error_manager_cube.raise_err_specific("conv3ddx",
            "the [{}]th value of strides should be greater than or equal to 1, actual is [{}].".\
            format(i, list(strides)[i]))
    para_check.check_shape_rule(strides, Conv3DConsts._STRIDES_SHAPE_DIM, Conv3DConsts._STRIDES_SHAPE_DIM,
                                _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(dilations, Conv3DConsts._DILATIONS_SHAPE_DIM,
                                Conv3DConsts._DILATIONS_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)

    # pads check
    if isinstance(pads, (tuple, list)) and len(pads) != Conv3DConsts._CONV_BACKPROP_PAD_SHAPE_DIM:
        error_manager_cube.raise_err_one_para('E62501', 'conv3d_backprop_input', 'pads')

    _, dilation_d, dilation_h, dilation_w, _ = dilations
    support_l0c2out_flag = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if not support_l0c2out_flag and dilation_d != 1:
        error_manager_cube.raise_err_specific("conv3ddx",
            "dilation in D dimension only supports 1.")

    # dtype check
    filter_dtype = filter_dtype.lower()
    out_backprop_dtype = out_backprop_dtype.lower()
    res_dtype = res_dtype.lower()
    para_check.check_dtype_rule(filter_dtype, ('float16', 'bfloat16'), "filter")
    para_check.check_dtype_rule(out_backprop_dtype, ('float16', 'bfloat16'), "out_backprop")
    para_check.check_dtype_rule(res_dtype, ('float16', 'float32', 'bfloat16'), "output")

    # the relation limits between shape
    shape_filter = list(shape_filter)
    shape_out_backprop = list(shape_out_backprop)
    input_sizes = list(input_sizes)
    strides = list(strides)
    fmap_batch, fmap_deep, fmap_h, fmap_w, fmap_channel = input_sizes
    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
    _, stride_d, stride_h, stride_w, _ = strides

    if fmap_channel != filter_channel * groups:
        error_manager_cube.raise_err_specific("conv3d_backprop_input",
            f"Shape error: Fmap's C[{fmap_channel}] must be equal to Filter'C[{filter_channel}] * groups[{groups}].")
    group_dict = util_common.calculate_group(fmap_channel,
                                             filter_batch, groups, _C0_SIZE, _C0_SIZE)

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    filter_d_dilation = (filter_depth - 1) * dilation_d + 1

    # pads compute
    pads = list(pads)
    pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right = pads
    # pads value limit
    _check_attr_range("pad head", pad_head, 0, _PAD_D_MAX)
    _check_attr_range("pad tail", pad_tail, 0, _PAD_D_MAX)
    _check_attr_range("pad up", pad_up, 0, _PAD_HW_MAX)
    _check_attr_range("pad down", pad_down, 0, _PAD_HW_MAX)
    _check_attr_range("pad left", pad_left, 0, _PAD_HW_MAX)
    _check_attr_range("pad right", pad_right, 0, _PAD_HW_MAX)
    # filter value limit
    _check_attr_range("filter's H", filter_h, Load3DParam.filter_min(), Load3DParam.filter_max())
    _check_attr_range("filter's W", filter_w, Load3DParam.filter_min(), Load3DParam.filter_max())
    _check_attr_range("filter's D", filter_depth, Load3DParam.filter_min(), Load3DParam.filter_max())

    # Fmap value limit
    _check_attr_range("Fmap's H", fmap_h, _FMAP_HW_MIN, _FMAP_HW_MAX)
    _check_attr_range("Fmap's W", fmap_w, _FMAP_HW_MIN, _FMAP_HW_MAX)

    # stride value limit
    _check_attr_range("stride's H", stride_h, Load3DParam.stride_min(), Load3DParam.stride_max())
    _check_attr_range("stride's W", stride_w, Load3DParam.stride_min(), Load3DParam.stride_max())
    _check_attr_range("stride's H*W",
                      stride_h * stride_w, Load3DParam.stride_min(), _STRIDE_SIZE_MAX)
    _check_attr_range("stride's H*W*D", stride_h * stride_w * stride_d,
                      Load3DParam.stride_min(), _STRIDE_SIZE_HWD_MAX)

    # dilation value limit
    _check_attr_range("dilation's H", dilation_h, Load3DParam.dilation_min(), Load3DParam.dilation_max())
    _check_attr_range("dilation's W", dilation_w, Load3DParam.dilation_min(), Load3DParam.dilation_max())
    _check_attr_range("dilation's D", dilation_d, Load3DParam.dilation_min(), Load3DParam.dilation_max())

    # Dedy value limit
    _check_attr_range("Dedy's H after expands", dedy_h * stride_h,
                      _DEDY_HW_MIN, _DEDY_HW_MAX)
    _check_attr_range("Dedy's W after expands", dedy_w * stride_w,
                      _DEDY_HW_MIN, _DEDY_HW_MAX)

    _check_shape_error()

    if not support_l0c2out_flag and (stride_h > 1 or stride_w > 1):
        _check_ub_limitation()
    block_size_k = tbe_platform.CUBE_MKN.get(out_backprop_dtype).get('mac')[1]
    util_conv3d.check_l1_limitation_dx(dedy_w * stride_w, stride_d, filter_h_dilation, filter_d_dilation, block_size_k)
    # check shape size, 64 bits limitation
    # ===========================================================

    fmap_size = fmap_batch * util_common.align(fmap_channel, _C0_SIZE) * fmap_deep * fmap_h * fmap_w
    dedy_size = dedy_batch * util_common.align(dedy_channel, _C0_SIZE) * dedy_deep * dedy_h * dedy_w
    filter_size = util_common.align(filter_batch, _C0_SIZE) * util_common.align(
        filter_channel, _C0_SIZE) * filter_depth * filter_h * filter_w
    _check_64bits_limitation("input", fmap_size, dtype=res_dtype)
    _check_64bits_limitation("out_backprop", dedy_size,
                             dtype=out_backprop_dtype)
    _check_64bits_limitation("filter", filter_size, dtype=filter_dtype)

    result = (shape_filter, shape_out_backprop, input_sizes, strides,
              pads, dilations, filter_dtype, out_backprop_dtype,
              res_dtype, kernel_name, group_dict)
    return result


@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), (list, tuple), int, (list, tuple),
                             str, str, str, str, (list, tuple), (list, tuple))
def _conv3d_backprop_input_cce(shape_filter, # pylint: disable=R0913,R0914
                              shape_out_backprop, input_sizes,
                              strides, pads, groups, dilations=(1, 1, 1, 1, 1),
                              filter_dtype='float16',
                              out_backprop_dtype='float16',
                              res_dtype='float16',
                              kernel_name="_conv3d_backprop_input_cce",
                              ori_tensors=(),
                              ori_attrs=()):
    """
    Topi interface of conv3d backprop input

    Parameters
    ----------
    shape_filter : The shape of filter
        5-D with shape [ depth, height, weight, batch, channels]

    shape_out_backprop : The shape of gradients
        5-D with shape [batch, depth, height, weight, channels]

    input_sizes : The shape of feature map
        5-D with shape [batch, depth, height, weight, channels]

    strides : A list/tuple of ints. The stride of the sliding window

    pads : A list/tuple of ints

    groups: Int of blocked connections from input channels to output channels

    dilations : An optional list/tuple of ints. Default value (1, 1, 1, 1, 1)

    filter_dtype : The dtype of filter data. Default value is float16

    out_backprop_dtype : The dtype of gradients data. Default value is float16

    res_dtype : The dtype of result(De/Dx) data. Default value is float16

    kernel_name : Cce kernel name. Default value is "_conv3d_backprop_input_cce

    ori_tensors : Original tensor list, in order of [filters, out_backprop, y_input]

    ori_attrs : Original attr list, in order of [input_size, strides, pads, dilations, groups, data_format]

    Returns
    ----------
    None
    """
    def _conv3dbp_input_achieve_with_tvm():
        dedy = tvm.placeholder(shape_dedy,
                               name="dedy",
                               dtype=out_backprop_dtype,
                               attrs={"tag": "conv3d_backprop_input_dy_ddr"})
        filters = tvm.placeholder(shape_filter_frac,
                                  name="filter",
                                  dtype=filter_dtype,
                                  attrs={"tag": "conv3d_backprop_input_filter_ddr"})

        shape_filter_ncdhw = [filter_batch, filter_channel, filter_depth, filter_h, filter_w]
        para_dict = {
            "strides": strides,
            "pads": pads,
            "dilations": dilations,
            "res_dtype": res_dtype,
            "kernel_name": kernel_name,
            "group_dict": group_dict
        }

        dedx = tbe.conv3d_backprop_input(filter=filters,
                                        out_backprop=dedy,
                                        filter_size=shape_filter_ncdhw,
                                        input_size=input_sizes,
                                        para_dict=para_dict)
        tensor_list = [filters, dedy, dedx]

        with tvm.target.cce():
            sch = tbe.auto_schedule(dedx)

        construct_params_for_do_op_tiling(ori_tensors, ori_attrs)

        for var in Conv3dBackpropInputBinaryParaProcess.get_binary_vars():
            Conv3dBackpropInputBinaryParaProcess.define_var(var)

        context = op_context.get_context()
        if context.get_addition("support_binary_constant"):
            log.debug("enable binary constant")
            stride_d = strides[1]
            dedx_d = input_sizes[1]
            dedy_d = shape_dedy[1]
            dilation_d = dilations[1]
            pad_h, pad_t, _, _, _, _ = pads
            filter_d_dilation = (filter_depth - 1) * dilation_d + 1

            if stride_d == filter_d_dilation and (dedx_d + pad_h + pad_t) == dedy_d * stride_d:
                sd_kd_mode = const.SD_EQ_KD_FLAG
            elif stride_d <= filter_d_dilation:
                sd_kd_mode = const.SD_LE_KD_FLAG
            else:
                sd_kd_mode = const.SD_GT_KD_FLAG

            if dilation_d > 1:
                dilation_d_gt_one_flag = True
            else:
                dilation_d_gt_one_flag = False

            context.add_addition("enable_binary_constant", True)
            context.add_addition(const.SD_KD_MODE_KEY, sd_kd_mode)
            context.add_addition("need_expand_stride", any(sd > 1 for sd in strides))
            context.add_addition("dilation_d_gt_one_flag", dilation_d_gt_one_flag)
            enable_binary_constant(context, ori_tensors, ori_attrs, kernel_name)

            return

        config = {"name": kernel_name, "tensor_list": tensor_list, "dummy_placeholder": True}
        tbe.build(sch, config)

    (shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations, filter_dtype, out_backprop_dtype,
     res_dtype, kernel_name, group_dict) = check_conv3dbp_input_params(shape_filter, shape_out_backprop, input_sizes,
                                                                       strides, pads, groups, dilations, filter_dtype,
                                                                       out_backprop_dtype, res_dtype, kernel_name)

    real_g = group_dict.get("real_g")
    cin1_g = group_dict.get("cin1_g")
    cout_g = group_dict.get("cout_g")

    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
    shape_dedy = (dedy_batch, dedy_deep, util_common.ceil(dedy_channel, _C0_SIZE), dedy_h, dedy_w, _C0_SIZE)
    shape_filter_frac = (real_g * filter_depth * cin1_g * filter_h * filter_w, cout_g // _C0_SIZE, _C0_SIZE, _C0_SIZE)

    _conv3dbp_input_achieve_with_tvm()
