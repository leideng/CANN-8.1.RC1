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
conv3d_transpose
"""
from impl.dynamic import check_and_config_para
from impl.dynamic.conv3d_backprop_input import check_empty_tensor
from impl.util import util_conv3d
from impl.util import util_select_op_base
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from tbe.common.context import op_context
from tbe.common.utils import const
from tbe.dsl.unify_schedule.conv3d_bp_input_tilingcase import Conv3dBackpropInputBinaryParaProcess

from impl.dynamic.conv3d_backprop_input import check_empty_tensor
from impl.util.util_cube_dynamic import generalize_shape_and_range


Nonetype = type(None)
_X_TARGET_FORMAT = "NDHWC"
_X_FORMAT_WHITE_LIST = ["NDHWC", "NCDHW"]
_FILTER_TARGET_FORMAT = "DHWCN"
_FILTER_FORMAT_WHITE_LIST = ["DHWCN", "NDHWC", "NCDHW"]
_RES_TARGET_FORMAT = "NDHWC"
_RES_FORMAT_WHITE_LIST = ["NDHWC", "NCDHW"]
_L1FUSION_INPUT_CTR = 2


def get_op_support_info(input_size,
                        x,
                        filters,
                        bias,
                        offset_w,
                        y,
                        strides,
                        pads,
                        dilations=(1, 1, 1, 1, 1),
                        groups=1,
                        data_format="NDHWC",
                        output_padding=(0, 0, 0, 0, 0),
                        offset_x=0,
                        kernel_name="conv3d_transpose",
                        op_slice_info=""):
    """
    algorithm: get_op_support_info

    Parameters
    ----------
    input_sizes: The shape of feature map
    5-D with shape [batch, depth, height, weight, channels]

    x: A dict with keys(shape and dtype)
    The shape of gradients

    filters: A dict with keys(shape and dtype)
    Input weight tensor

    bias: A dict with keys(shape and dtype) or None
    Input bias tensor

    offset_w: A dict with keys(shape and dtype) or None
    Input offset_w tensor

    y: A dict with keys(shape and dtype)
    Conv3d_transpose output tensor, dtype must be assigned

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
    def _get_slice_info():
        # headoverlap, tailoverlap, 0 means with overlap, -1 means without it
        overlap_d = -1 if (ori_shape_filters[0] == 1 and strides_formated[1] == 1) else 0
        overlap_h = -1 if (ori_shape_filters[1] == 1 and strides_formated[2] == 1) else 0
        overlap_w = -1 if (ori_shape_filters[2] == 1 and strides_formated[3] == 1) else 0

        # format
        axis_split_matrix = []
        axis_reduce_list = None
        format_out_backprop = x.get("format")
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
                    [util_select_op_base.SplitInput([1, [0], [-1], [-1]], [2, [0], [-1], [-1]]),
                     util_select_op_base.SplitOutput([0, [2]])]
                )
            else:
                axis_split_matrix.append(
                    [util_select_op_base.SplitInput([1, [0], [-1], [-1]]),
                     util_select_op_base.SplitOutput([0, [2]])]
                )
        else:
            axis_split_matrix = None

        return axis_split_matrix, axis_reduce_list

    try:
        input_size_shape = input_size.get("ori_shape")
        if len(input_size_shape) == 1:
            input_size_shape, = input_size_shape
            if input_size_shape < 0:
                error_manager_cube.raise_err_specific(
                    'conv3d_backprop_input', "prebuild failed, not support input size's shape [-1] and [-2]")
    except RuntimeError:
        op_cal_info_in_json = util_select_op_base.get_op_cal_info([], [], _L1FUSION_INPUT_CTR,
                                                                  None)
        return op_cal_info_in_json

    ori_shape_x = util_conv3d.transform_shape_with_format(x.get("ori_format"),
                                                          _X_TARGET_FORMAT,
                                                          x.get("ori_shape"),
                                                          _X_FORMAT_WHITE_LIST)

    strides_formated = util_conv3d.transform_shape_with_format(x.get("ori_format"),
                                                               _X_TARGET_FORMAT,
                                                               strides,
                                                               _X_FORMAT_WHITE_LIST)

    if ori_shape_x is None or strides_formated is None:
        error_manager_cube.raise_err_one_para("E62306", "Conv3d_transpose", "data format should be NDHWC or NCDHW")


    ori_shape_filters = util_conv3d.transform_shape_with_format(filters.get("ori_format"),
                                                                _FILTER_TARGET_FORMAT,
                                                                filters.get("ori_shape"),
                                                                _FILTER_FORMAT_WHITE_LIST)
    if ori_shape_filters is None:
        error_manager_cube.raise_err_one_para("E62306", "Conv3d_transpose",
                                              "filter format should be NDHWC/NCDHW/DHWCN")


    ori_shape_res = util_conv3d.transform_shape_with_format(y.get("ori_format"),
                                                            _RES_TARGET_FORMAT,
                                                            y.get("ori_shape"),
                                                            _RES_FORMAT_WHITE_LIST)
    if ori_shape_res is None:
        error_manager_cube.raise_err_one_para("E62306", "Conv3d_transpose", "y format should be NDHWC or NCDHW")
    axis_split_info, axis_reduce_info = _get_slice_info()

    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_info,
                                                              axis_reduce_info,
                                                              _L1FUSION_INPUT_CTR,
                                                              0)

    return op_cal_info_in_json


@tbe_register.register_param_generalization("Conv3DTranspose")
def conv3d_transpose_generalization(input_size,
                                    x, filter, bias, offset_w, y, strides,
                                    pads, dilations=(1, 1, 1, 1, 1),
                                    groups=1, data_format="NDHWC", output_padding=(0, 0, 0, 0, 0, 0), offset_x=0,
                                    kernel_name="conv3d_transpose",
                                    generalize_config=None):
    """
    conv3d transpose generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    input_size: dict, will not be used
            input tensor size.

    x: dict with keys(ori_shape, ori_format, dtype)
        The shape of gradients.

    filter: dict with keys(ori_shape, ori_format, dtype)
            convolution kernel

    bias: dict with keys(shape and dtype)
        The shape of bias.

    offset_w: dict with keys(shape and dtype) or None
        Input offset_w tensor.

    y: dict with keys(ori_shape, ori_format, dtype and range)
       conv3d_transpose output tensor

    strides: tuple/list of 5 integers
             filter move stride

    pads: tuple/list of 6 integers
          str: "SAME" or "VALID"
          tuple/list of 6 integers: [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 5 integers
               filter expand size of dilated conv3d_transpose

    groups: int
            param for group conv3d_transpose

    data_format: str
            An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC".
            Specify the data format of the input and output data.

    output_padding: tuple/list of 6 integers
        The size will be added in the output shape. Default to (0, 0, 0, 0, 0, 0).

    offset_x: int
        offset of gradients in quant mode. Default to 0.

    kernel_name: str
            kernel name, default value is "conv3d_transpose"

    generalize_config: dict, generaliazation mode.
    support all_shape

    Returns
    -------
    list of params list:
    """
    result = []
    if generalize_config.get("mode") == "keep_rank":
        input_size["const_value"] = None
        filter["ori_range"], filter["ori_shape"] = \
            generalize_shape_and_range(filter.get("ori_format"), filter.get("ori_shape"))
        x["ori_range"], x["ori_shape"] = \
            generalize_shape_and_range(x.get("ori_format"), x.get("ori_shape"))
        result.append([input_size, x, filter, bias, offset_w, y, {"strides": strides}, {"pads": pads},
                       {"dilations": dilations}, {"grouops": groups}, {"data_format": data_format},
                       {"output_padding": output_padding}, {"offset_x": offset_x}])
    elif generalize_config.get("mode") == "all_shape":
        # mod shape and range for binary reuse
        input_size["format"] = "NCDHW"
        input_size["ori_format"] = "NCDHW"
        range_x, shape_x = generalize_shape_and_range(x["format"], x["shape"])
        range_filter, shape_filter = generalize_shape_and_range(filter["format"], filter["shape"])
        range_y, shape_y = generalize_shape_and_range(y["format"], y["shape"])
        x["range"], filter["range"], y["range"] = range_x, range_filter, range_y
        x["shape"], filter["shape"], y["shape"],  = shape_x, shape_filter, shape_y
        if bias is not None:
            range_bias, shape_bias = generalize_shape_and_range(bias["format"], bias["shape"], 1)
            bias["range"] = range_bias
            bias["shape"] = shape_bias
            bias["ori_format"] = "ND"
            bias["format"] = "ND"
        # mod format for binary reuse
        input_size["ori_format"] = "NCDHW"
        input_size["format"] = "NCDHW"
        input_size["dtype"] = "int32"
        x["ori_format"] = "NCDHW"
        filter["ori_format"] = "NCDHW"
        y["ori_format"] = "NCDHW"
        # mod attr for binary reuse
        offset_w = None
        strides = [-1, -1, -1, -1, -1]
        pads = [-1, -1, -1, -1, -1, -1]
        dilations = [-1, -1, -1, -1, -1]
        groups = -1
        output_padding = [-1, -1, -1, -1, -1, -1]
        offset_x = 0
        data_format = "NCDHW"
        result.append([input_size, x, filter, bias, offset_w, y, strides, pads, dilations,
                    groups, data_format, output_padding, offset_x])
    return result


def _check_output_padding(output_padding, stride, dilation, data_format):
    def _ncdhw2ndhwc(shape_ncdhw):
        shape_ndhwc = [shape_ncdhw[0], shape_ncdhw[2], shape_ncdhw[3], shape_ncdhw[4], shape_ncdhw[1]]
        return shape_ndhwc
    if data_format == "NCDHW":
        output_padding = _ncdhw2ndhwc(output_padding)
        stride = _ncdhw2ndhwc(stride)
        dilation = _ncdhw2ndhwc(dilation)
    _, output_padding_d, output_padding_h, output_padding_w, _ = output_padding
    _, stride_d, stride_h, stride_w, _ = stride
    _, dilation_d, dilation_h, dilation_w, _ = dilation
    if output_padding_d < 0 or (output_padding_d >= dilation_d and output_padding_d >= stride_d):
        error_manager_cube.raise_err_three_paras(
            'E62305', 'conv3d_transpose', 'output_padding D',
            '[{}, {})'.format(str(0), 'max(stride D,dilation D)'), str(output_padding_d))

    if output_padding_h < 0 or (output_padding_h >= dilation_h and output_padding_h >= stride_h):
        error_manager_cube.raise_err_three_paras(
            'E62305', 'conv3d_transpose', 'output_padding H',
            '[{}, {})'.format(str(0), 'max(stride H,dilation H)'), str(output_padding_h))

    if output_padding_w < 0 or (output_padding_w >= dilation_w and output_padding_w >= stride_w):
        error_manager_cube.raise_err_three_paras(
            'E62305', 'conv3d_transpose', 'output_padding W',
            '[{}, {})'.format(str(0), 'max(stride W,dilation W)'), str(output_padding_w))


def _conv3d_transpose_compute(filters,
                              out_backprop,
                              y_input,
                              input_size,
                              strides,
                              pads,
                              dilations=(1, 1, 1, 1, 1),
                              groups=1,
                              output_padding=(0, 0, 0, 0, 0),
                              data_format="NDHWC",
                              kernel_name="conv3d_transpose",
                              build_options=None):
    binary_flag = Conv3dBackpropInputBinaryParaProcess.check_binary_flag(filters)
    if binary_flag:
        ori_paras = {
            "input_size": input_size,
            "filters": filters,
            "out_backprop": out_backprop,
            "y": y_input,
            "strides": strides,
            "pads": pads,
            "dilations": dilations,
            "groups": groups,
            "data_format": data_format,
            "kernel_name": kernel_name,
            "build_options": build_options
        }
        conv3dbp_para = Conv3dBackpropInputBinaryParaProcess(ori_paras)
        conv3dbp_para.check_paras()
        input_size_tensor, filter_tensor, dedy_tensor = conv3dbp_para.define_placeholder()
        res_dtype = conv3dbp_para.y.get("dtype")
        group_dict = conv3dbp_para.attrs.get("group_dict")
        shape_filter_ncdhw = conv3dbp_para.shape.get("filter_ncdhw")
        input_sizes = conv3dbp_para.shape.get("dedx_ndc1hwc0")
        strides = conv3dbp_para.attrs.get("strides_ndhwc")
        pads = conv3dbp_para.attrs.get("pads")
        dilations = conv3dbp_para.attrs.get("dilations_ndhwc")
    else:
        check_empty_tensor(out_backprop, filters, y_input, strides, pads, dilations)
        _check_output_padding(output_padding, strides, dilations, data_format)
        (input_size_tensor, dedy_tensor, filter_tensor, shape_filter, input_sizes, strides, pads, dilations, res_dtype,
         kernel_name, group_dict) = check_and_config_para(filters, out_backprop, y_input, input_size, strides, pads,
                                                          dilations, groups, data_format, kernel_name)
        filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
        shape_filter_ncdhw = [filter_batch, filter_channel, filter_depth, filter_h, filter_w]

    para_dict = {
        "strides": strides,
        "pads": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }

    if build_options is not None:
        para_dict.update(build_options)

    dedx = tbe.conv3d_backprop_input(filter=filter_tensor,
                                     out_backprop=dedy_tensor,
                                     filter_size=shape_filter_ncdhw,
                                     input_size=input_sizes,
                                     para_dict=para_dict)

    context = op_context.get_context()
    if context.get_addition("enable_binary_constant"):
        op_placeholder = [dedy_tensor, filter_tensor]
    else:
        op_placeholder = [input_size_tensor, dedy_tensor, filter_tensor]

    return {'op_placeholder': op_placeholder, 'op_res': [dedx]}


@register_operator("Conv3DTranspose")
@para_check.check_input_type(dict, dict, dict, (Nonetype, dict), (Nonetype, dict),
                             dict, (tuple, list), (tuple, list, str),
                             (tuple, list), int, str, (tuple, list), int, str)
def conv3d_transpose(input_size, x, filter,
                     bias, offset_w, y, strides,
                     pads, dilations=(1, 1, 1, 1, 1), groups=1,
                     data_format="NDHWC", output_padding=(0, 0, 0, 0, 0),
                     offset_x=0, kernel_name="conv3d_transpose"):
    """
    algorithm: Conv3d_transpose

    Parameters
    ----------
    input_size: dict, will not be used
    input tensor size.

    x: A dict with keys(shape and dtype)
    Gradients tensor

    filter: A dict with keys(shape and dtype)
    Input weight tensor

    bias: dict with keys(shape and dtype)
    The shape of bias.

    offset_w: dict with keys(shape and dtype) or None
    Input offset_w tensor.

    y: A dict with keys(shape and dtype)
    conv3d_transpose output tensor, dtype must be assigned

    strides: A tuple/list of 5 integers
    Filter move stride

    pads: A tuple/list of 6 integers: [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]
    str: "SAME" or "VALID"

    dilations: A tuple/list of 5 integers
    filter expand size of dilated conv3d_transpose, default value (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
    Default value 1

    data_format: The data format of the input and output data
    Default format "NDHWC"

    output_padding: tuple/list of 5 integers
    The size will be added in the output shape. Default to (0, 0, 0, 0, 0)

    offset_x: int
    offset of gradients in quant mode. Default to 0

    kernel_name: Str
    Kernel name, default value is "conv3d_transpose"

    Returns
    -------
    None
    """
    binary_flag = Conv3dBackpropInputBinaryParaProcess.check_binary_flag(filter)
    if bias:
        if binary_flag:
            error_manager_cube.raise_err_specific('conv3d_transpose', "Bias is not supported on binary condition")
        res_dtype = y.get("dtype").lower()
        util_conv3d.check_bias(bias, res_dtype)
        bias_dtype = bias.get("dtype").lower()
        para_check.check_dtype_rule(bias_dtype, ("float16", "float32"), "bias")

    input_list = [filter, x, y, input_size, strides, pads, dilations, groups, output_padding, data_format, kernel_name]
    context = op_context.get_context()
    extra_params = {
        "single_op": True,
        "split_w": False,
        "binary_flag": binary_flag
    }
    build_args = {
        "constant_realize_extent_in_infer_bound": False
    }

    if context.get_addition("enable_binary_constant"):
        extra_params.update({"need_expand_stride": context.get_addition("need_expand_stride")})
        extra_params.update({const.SD_KD_MODE_KEY: context.get_addition(const.SD_KD_MODE_KEY)})
        extra_params.update({const.DILATION_D_GT_ONE_KEY: context.get_addition(const.DILATION_D_GT_ONE_KEY)})
        input_args = classify(input_list, "conv3d_transpose", extra_params)
    else:
        input_args = classify(input_list, "conv3d_transpose", extra_params)

    schedules, tensors_list = [], []
    for input_param in input_args:
        tensors = []
        with tbe.compute():
            res = _conv3d_transpose_compute(*input_param)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res.get('op_res'))

        schedules.append(sch)
        tensors = res.get('op_placeholder') + res.get('op_res')
        tensors_list.append(tensors)

    config = {'print_ir': False,
              'name': kernel_name,
              'tensor_list': tensors_list,
              'build_args': build_args}

    tbe.build(schedules, config)