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
conv3d_transpose_d
"""
import impl.dynamic as dyn_impl
from impl.conv3d_backprop_input_d import check_conv3dbp_input_params
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
from tbe.dsl.unify_schedule.conv3d_bp_input_tilingcase import Conv3dBackpropInputBinaryParaProcess


_L1FUSION_INPUT_CTR = 2

_OUT_BACKPROP_TARGET_FORMAT = "NDHWC"
_OUT_BACKPROP_FORMAT_WHITE_LIST = ["NDHWC", "NCDHW"]
_FILTER_TARGET_FORMAT = "DHWCN"
_FILTER_FORMAT_WHITE_LIST = ["DHWCN", "NDHWC", "NCDHW"]
_RES_TARGET_FORMAT = "NDHWC"
_RES_FORMAT_WHITE_LIST = ["NDHWC", "NCDHW"]


def get_op_support_info(out_backprop,
                        filters, # pylint: disable=R0913,R0914
                        bias,
                        offset_w,
                        y_input,
                        input_sizes,
                        strides,
                        pads,
                        dilations=(1, 1, 1, 1, 1),
                        groups=1,
                        data_format="NDHWC",
                        output_padding=[0, 0, 0, 0, 0],
                        offset_x=0,
                        kernel_name="conv3d_transpose",
                        op_slice_info=""):
    """
    algorithm: get_op_support_info

    Parameters
    ----------
    out_backprop: A dict with keys(shape and dtype)
    The shape of gradients

    filters: A dict with keys(shape and dtype)
    Input weight tensor

    bias: A dict with keys(shape and dtype) or None
    Input bias tensor

    offset_w: A dict with keys(shape and dtype) or None
    Input offset_w tensor

    y_input: A dict with keys(shape and dtype)
    Conv3d_transpose output tensor, dtype must be assigned

    input_sizes: The shape of feature map
    5-D with shape [batch, depth, height, weight, channels]

    strides: A tuple/list of 5 integers
    Filter move stride

    pads: A tuple/list of 6 integers
    [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
    Filter expand size of dilated conv3d_transpose, default value is (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
    Default value is 1

    data_format: The data format of the input and output data
    Default format is "NDHWC"

    output_padding: The size will be added in the output shape
    Default value is [0, 0, 0, 0, 0]

    offset_x: Int
    Input offset_x value, default value is 0

    kernel_name: Str
    Kernel name, default value is "conv3d_transpose"

    op_slice_info: Str
    Default value is ""

    Returns
    -------
    op_cal_info_in_json: A dict with keys(split_maps, reduce_maps, l1_fusion_enable and min_tbe_l1_space)
    """
    def _cal_min_l1space():
        block_size = 16
        w_value = ori_shape_out_backprop[3] * strides[3]
        filter_d_dilation = (ori_shape_filters[0] - 1) * dilations[1] + 1
        filter_h_dilation = (ori_shape_filters[1] - 1) * dilations[2] + 1
        filter_w_dilation = (ori_shape_filters[2] - 1) * dilations[3] + 1
        if ori_shape_res[3] > block_size:
            h_value_max = filter_h_dilation + 1
        elif block_size % ori_shape_res[3] == 0:
            h_value_max = filter_h_dilation + block_size // ori_shape_res[3] - 1
        else:
            h_value_max = filter_h_dilation + block_size // ori_shape_res[3] + 1

        a_l1_size = h_value_max * w_value * ((filter_d_dilation - 2) // strides[1] + 2) * block_size * 2
        b_l1_size = filter_h_dilation * filter_w_dilation * filter_d_dilation * block_size * block_size * 2
        return a_l1_size + b_l1_size

    def _get_slice_info():
        overlap_d = -1 if (ori_shape_filters[0] == 1 and strides_formated[1] == 1) else 0
        overlap_h = -1 if (ori_shape_filters[1] == 1 and strides_formated[2] == 1) else 0
        overlap_w = -1 if (ori_shape_filters[2] == 1 and strides_formated[3] == 1) else 0
        overlap_c = -1 if ori_shape_filters[3] <= 16 else 0

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
            if bias:
                axis_split_matrix.append(
                    [util_select_op_base.SplitInput([1, [0], [overlap_c], [overlap_c]], [2, [0], [-1], [-1]]),
                    util_select_op_base.SplitOutput([0, [2]])]
                )
            else:
                axis_split_matrix.append(
                    [util_select_op_base.SplitInput([1, [0], [overlap_c], [overlap_c]]),
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

    if ori_shape_out_backprop is None or strides_formated is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y_backprop',
            'expected_format_list': ",".join(_OUT_BACKPROP_FORMAT_WHITE_LIST),
            'format': out_backprop.get("ori_format")
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

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

    axis_split_info, axis_reduce_info = _get_slice_info()

    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_info,
                                                              axis_reduce_info,
                                                              _L1FUSION_INPUT_CTR,
                                                              _cal_min_l1space())

    return op_cal_info_in_json


def _check_output_padding(output_padding_val, dilation_val, stride_val):
    if output_padding_val < 0 or (output_padding_val >= dilation_val and output_padding_val >= stride_val):
        dict_args = {
            'errCode': 'E62305',
            'param_name': 'output_padding',
            'expect_value': '[{}, {})'.format(str(0), str(max(dilation_val, stride_val))),
            'value': str(output_padding_val)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def _process_and_check_input(out_backprop, filters, # pylint: disable=R0913,R0914
                             bias, offset_w, y_input, input_sizes,
                             strides, pads, dilations=(1, 1, 1, 1, 1), groups=1,
                             data_format="NDHWC",
                             output_padding=(0, 0, 0, 0, 0),
                             offset_x=0, kernel_name="conv3d_transpose"):
    """
    """
    ori_shape_filters = filters.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = input_sizes
    ori_shape_strides = strides
    ori_shape_dialtions = dilations
    ori_shape_output_padding = output_padding

    filters_dtype = filters.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y_input.get("dtype")

    ori_format_filters = filters.get("ori_format")
    ori_format_out_backprop = data_format
    ori_format_res = data_format

    if (isinstance(ori_shape_output_padding, (tuple, list)) and
        len(ori_shape_output_padding) != util_common.CONV3D_SHAPE_COMMON_DIM):
        error_manager_cube.raise_err_one_para('E62006', 'conv3d',
            'output_padding should be 5-dim list/tuple')

    # transform filter shape
    shape_filters = util_conv3d.transform_shape_with_format(ori_format_filters,
                                                            _FILTER_TARGET_FORMAT,
                                                            ori_shape_filters,
                                                            _FILTER_FORMAT_WHITE_LIST)
    if shape_filters is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'filter',
            'expected_format_list': ",".join(_FILTER_FORMAT_WHITE_LIST),
            'format': ori_format_filters
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    # transform out_backprop, strides, dilations shape
    shape_out_backprop = util_conv3d.transform_shape_with_format(ori_format_out_backprop,
                                                                 _OUT_BACKPROP_TARGET_FORMAT,
                                                                 ori_shape_out_backprop,
                                                                 _OUT_BACKPROP_FORMAT_WHITE_LIST)

    shape_strides = util_conv3d.transform_shape_with_format(ori_format_out_backprop,
                                                            _OUT_BACKPROP_TARGET_FORMAT,
                                                            ori_shape_strides,
                                                            _OUT_BACKPROP_FORMAT_WHITE_LIST)

    shape_dilations = util_conv3d.transform_shape_with_format(ori_format_out_backprop,
                                                              _OUT_BACKPROP_TARGET_FORMAT,
                                                              ori_shape_dialtions,
                                                              _OUT_BACKPROP_FORMAT_WHITE_LIST)

    shape_output_padding = util_conv3d.transform_shape_with_format(ori_format_out_backprop,
                                                                   _OUT_BACKPROP_TARGET_FORMAT,
                                                                   ori_shape_output_padding,
                                                                   _OUT_BACKPROP_FORMAT_WHITE_LIST)

    if (shape_out_backprop is None or shape_strides is None or shape_dilations is None or
        shape_output_padding is None):
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y_backprop',
            'expected_format_list': ",".join(_OUT_BACKPROP_FORMAT_WHITE_LIST),
            'format': ori_format_out_backprop
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    _, dilation_d, dilation_h, dilation_w, _ = shape_dilations
    _, stride_d, stride_h, stride_w, _ = shape_strides
    _, output_padding_d, output_padding_h, output_padding_w, _ = shape_output_padding

    # transform res shape
    shape_res = util_conv3d.transform_shape_with_format(ori_format_res,
                                                        _RES_TARGET_FORMAT,
                                                        ori_shape_res,
                                                        _RES_FORMAT_WHITE_LIST)
    if shape_res is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y',
            'expected_format_list': ",".join(_RES_FORMAT_WHITE_LIST),
            'format': ori_format_res
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    _check_output_padding(output_padding_d, dilation_d, stride_d)
    _check_output_padding(output_padding_h, dilation_h, stride_h)
    _check_output_padding(output_padding_w, dilation_w, stride_w)

    return (shape_filters, shape_out_backprop, shape_res, shape_strides, pads,
            groups, shape_dilations, filters_dtype, out_backprop_dtype, res_dtype, kernel_name)


def check_supported(out_backprop,
                    filters,
                    bias,
                    offset_w,
                    y_input,
                    input_size,
                    strides,
                    pads,
                    dilations=(1, 1, 1, 1, 1),
                    groups=1,
                    data_format="NDHWC",
                    output_padding=[0, 0, 0, 0, 0],
                    offset_x=0,
                    kernel_name="conv3d_transpose"):
    """
    The H and W dimension of input_size should be in range [1, 4096]. \n
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
    try:
        input_size = y_input.get("ori_shape") if (all(i == 0 for i in input_size)) else input_size
        (shape_filters, shape_out_backprop, shape_res, shape_strides, pads, groups, shape_dilations,
         filters_dtype, out_backprop_dtype, res_dtype,
         kernel_name) = _process_and_check_input(out_backprop, filters, bias, offset_w, y_input, input_size,
                                                 strides, pads, dilations, groups, data_format,
                                                 output_padding, offset_x, kernel_name)

        check_conv3dbp_input_params(shape_filters, shape_out_backprop, shape_res, shape_strides, pads,
                                    groups, shape_dilations, filters_dtype, out_backprop_dtype, res_dtype,
                                    kernel_name)
        if bias:
            util_conv3d.check_bias(bias, res_dtype)
            bias_dtype = bias.get("dtype").lower()
            para_check.check_dtype_rule(bias_dtype, ("float16", "float32"), "bias")

        return True, ""
    except Exception as e:
        reason = e.args[1]
        return False, reason
    finally:
        pass


def construct_params_for_do_op_tiling(tensors, attrs):
    out_backprop, filters, bias, offset_w, y = tensors
    input_size, strides, pads, dilations, groups, data_format, output_padding, offset_x = attrs

    dyn_inputs = [{
        "name": "input_size",
        "shape": [Conv3DConsts._CONV_BACKPROP_SHAPE_DIM],
        "ori_shape": [Conv3DConsts._CONV_BACKPROP_SHAPE_DIM],
        "dtype": "int32",
        "format": "ND",
        "ori_format": "ND",
        "const_value": input_size
    },
                  extract_tensor_info(out_backprop),
                  extract_tensor_info(filters),
                  extract_tensor_info(bias),
                  extract_tensor_info(offset_w)]
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
        "name": "output_padding",
        "dtype": "list_int",
        "value": output_padding
    }, {
        "name": "offset_x",
        "dtype": "int",
        "value": offset_x
    }, {
        "name": "padding",
        "dtype": "str",
        "value": ""
    })

    params = {}
    params['op_type'] = 'Conv3DTranspose'
    params['inputs'] = dyn_inputs
    params['outputs'] = dyn_output
    params['attrs'] = dyn_attrs
    op_context.get_context().add_addition('params_do_op_tiling', params)


def enable_binary_constant(context, tensors, attrs, kernel_name):
    context.set_op_mode("static")

    out_backprop, filters, bias, offset_w, y = tensors
    _, _, _, _, _, data_format, output_padding, offset_x = attrs

    input_size = {
        "shape": [Conv3DConsts._CONV_BACKPROP_SHAPE_DIM],
        "ori_shape": [Conv3DConsts._CONV_BACKPROP_SHAPE_DIM],
        "format": "ND",
        "ori_format": "ND"
    }

    generalize_shape_and_range_inplace(out_backprop)
    generalize_shape_and_range_inplace(filters)
    if bias is not None:
        generalize_shape_and_range_inplace(bias)
    generalize_shape_and_range_inplace(y)
    strides = [-1] * Conv3DConsts._STRIDES_SHAPE_DIM
    pads = [-1] * Conv3DConsts._CONV_BACKPROP_PAD_SHAPE_DIM
    dilations = [-1] * Conv3DConsts._DILATIONS_SHAPE_DIM
    groups = -1

    dyn_impl.conv3d_transpose(input_size, out_backprop, filters, bias, offset_w, y, strides, pads, dilations, groups,
                              data_format, output_padding, offset_x, kernel_name)


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
    para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_STR,
    para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_INT,
    para_check.KERNEL_NAME)
def conv3d_transpose_d(out_backprop, filters, # pylint: disable=R0913,R0914
                       bias, offset_w, y_input, input_size,
                       strides, pads, dilations=(1, 1, 1, 1, 1), groups=1,
                       data_format="NDHWC",
                       output_padding=[0, 0, 0, 0, 0],
                       offset_x=0, kernel_name="conv3d_transpose"):
    """
    algorithm: conv3d_transpose

    Parameters
    ----------
    out_backprop: A dict with keys(shape and dtype)
    The shape of gradients

    filters: A dict with keys(shape and dtype)
    Input weight tensor

    bias: A dict with keys(shape and dtype) or None
    Input bias tensor

    offset_w: A dict with keys(shape and dtype) or None
    Input offset_w tensor

    y_input: A dict with keys(shape and dtype)
    Conv3d_transpose output tensor, dtype must be assigned

    input_size: The shape of feature map
    5-D with shape [batch, depth, height, weight, channels]

    strides: A tuple/list of 5 integers
    Filter move stride

    pads: A tuple/list of 6 integers
    [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
    Filter expand size of dilated conv3d_transpose, default value is (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
    Default value is 1

    data_format: The data format of the input and output data
    Default format is "NDHWC"

    output_padding: The size will be added in the output shape
    Default value is [0, 0, 0, 0, 0]

    offset_x: Int
    Input offset_x value, default value is 0

    kernel_name: Str
    Kernel name, default value is "conv3d_transpose"

    Returns
    -------
    None
    """
    ori_tensors = [out_backprop, filters, bias, offset_w, y_input]
    ori_attrs = [input_size, strides, pads, dilations, groups, data_format, output_padding, offset_x]
    input_size = y_input.get("ori_shape") if (all(i == 0 for i in input_size)) else input_size
    (shape_filters, shape_out_backprop, shape_res, shape_strides,
        pads, groups, shape_dilations, filters_dtype, out_backprop_dtype,
        res_dtype, kernel_name) = _process_and_check_input(
                                      out_backprop, filters,
                                      bias, offset_w, y_input, input_size,
                                      strides, pads, dilations, groups,
                                      data_format, output_padding, offset_x, kernel_name)
    bias_flag = bias is not None
    _conv3d_transpose_cce(shape_filters,
                          shape_out_backprop,
                          shape_res,
                          shape_strides,
                          pads,
                          groups,
                          shape_dilations,
                          bias_flag,
                          filters_dtype,
                          out_backprop_dtype,
                          res_dtype,
                          kernel_name,
                          ori_tensors,
                          ori_attrs)


@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), (str, list, tuple), int,
                             (list, tuple), bool, str, str, str, str, (list, tuple), (list, tuple))
def _conv3d_transpose_cce(shape_filter, # pylint: disable=R0913,R0914
                          shape_out_backprop, input_sizes,
                          strides, pads, groups, dilations=(1, 1, 1, 1, 1),
                          bias_flag=False,
                          filter_dtype='float16',
                          out_backprop_dtype='float16',
                          res_dtype='float16',
                          kernel_name="_conv3d_transpose_cce",
                          ori_tensors=(),
                          ori_attrs=()):
    """
    Topi interface of conv3d transpose

    Parameters:
    ----------
    shape_filter : The shape of filter
        5-D with shape [ depth, height, weight, batch, channels]

    shape_out_backprop : The shape of gradients
        5-D with shape [batch, depth, height, weight, channels]

    input_sizes : The shape of feature map
        5-D with shape [batch, depth, height, weight, channels]

    strides : A list/tuple of ints. The stride of the sliding window

    pads : A list/tuple of ints or str

    groups: Int of blocked connections from input channels to output channels

    dilations : An optional list/tuple of ints. Default value (1, 1, 1, 1, 1)

    filter_dtype : The dtype of filter data. Default value is float16

    out_backprop_dtype : The dtype of gradients data. Default value is float16

    res_dtype : The dtype of result(De/Dx) data. Default value is float16

    kernel_name : Cce kernel name. Default value is "_conv3d_transpose_cce"

    ori_tensors : Original tensor list, in order of [out_backprop, filters, bias, offset_w, y_input]

    ori_attrs : Original attrs list, in order of
                [input_size, strides, pads, dilations, groups, data_format, output_padding, offset_x]

    Returns
    ----------
    None
    """
    def _conv3d_transpose_achieve_with_tvm():
        dedy = tvm.placeholder(shape_dedy,
                               name="dedy",
                               dtype=out_backprop_dtype,
                               attrs={"tag": "conv3d_backprop_input_dy_ddr"})

        filters = tvm.placeholder(shape_filter_frac,
                                  name="filter",
                                  dtype=filter_dtype,
                                  attrs={"tag": "conv3d_backprop_input_filter_ddr"})
        tensor_bias = None
        if bias_flag:
            tensor_bias = tvm.placeholder(
                (util_common.ceil(filter_channel * groups, tbe_platform.C0_SIZE), tbe_platform.C0_SIZE),
                name="bias",
                dtype=res_dtype,
                attrs={"tag": "conv3d_backprop_input_bias_ddr"})

        shape_filter_ncdhw = [filter_batch, filter_channel, filter_depth, filter_h, filter_w]
        para_dict = {
            "strides": strides,
            "pads": pads,
            "dilations": dilations,
            "res_dtype": res_dtype,
            "tensor_bias": tensor_bias,
            "kernel_name": kernel_name,
            "group_dict": group_dict
        }

        dedx = tbe.conv3d_backprop_input(filter=filters,
                                        out_backprop=dedy,
                                        filter_size=shape_filter_ncdhw,
                                        input_size=input_sizes,
                                        para_dict=para_dict)
        if bias_flag:
            tensor_list = [dedy, filters, tensor_bias, dedx]
        else:
            tensor_list = [dedy, filters, dedx]
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

    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter

    # Channel axis should be align with 16
    c0_size = tbe_platform.C0_SIZE
    shape_dedy = (dedy_batch,
                  dedy_deep,
                  util_common.ceil(dedy_channel, c0_size), dedy_h, dedy_w, c0_size)

    real_g = group_dict.get("real_g")
    cin1_g = group_dict.get("cin1_g")
    cout_g = group_dict.get("cout_g")

    shape_filter_frac = (real_g * filter_depth * cin1_g * filter_h * filter_w,
                         cout_g // c0_size, c0_size, c0_size)
    _conv3d_transpose_achieve_with_tvm()
