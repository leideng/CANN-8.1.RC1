#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
conv2d_backprop_input_d
"""
from collections import OrderedDict
import copy

import impl.dynamic as dyn_impl
import tbe
import tbe.common.utils.log as log
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.util_common import ShapeConverter
from impl.util.util_cube_dynamic import define_operation_var_of_dx
from impl.util.util_cube_dynamic import generalize_shape_and_range_inplace
from impl.util.util_deconv_comm import check_conv2d_transpose
from impl.util.util_deconv_comm import conv2d_transpose_static_compute
from impl.util.util_deconv_comm import conv2d_transpose_static_impl
from impl.util.util_deconv_comm import get_op_support_info_conv2d_transpose
from impl.util.util_deconv_comm import need_exchange_hw_axis
from impl.util.util_deconv_comm import swap_h_w_axes_in_shape
from impl.util.util_deconv_comm import update_ori_shape
from tbe.common.context import op_context
from tbe.dsl.compute.cube_util import extract_tensor_info
from tbe.dsl.static_schedule.util import get_op_impl_mode_enum


def get_op_support_info(  # pylint: disable=invalid-name,R0913,R0914,W0613
        filter,
        out_backprop,
        y,
        input_size,
        strides,
        pads,
        dilations=(1, 1, 1, 1),
        groups=1,
        data_format="NHWC",
        kernel_name="conv2d_backprop_input",
):
    """
    get the conv2d_backprop_input_d split
    """
    tensor_dict = {"weight": filter, "fmap_dedy": out_backprop, "fmap_dedx": y}
    option_input = {}
    attrs_dict = {"input_size": input_size, "strides": strides, "pads": pads, "dilations": dilations,
                  "groups": groups, "data_format": data_format, "kernel_name": kernel_name}
    return get_op_support_info_conv2d_transpose(tensor_dict, option_input, attrs_dict, op_type="conv2d_backprop_input")


def check_supported(
    filter,
    out_backprop,
    y,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",
    kernel_name="conv2d_backprop_input",
):
    """
    check the op support situation:

    batch_input_size == batch_out_backprop

    out_backprop_height == (fmap_height + pad_top + pad_bottom - (dilation_h * (filter_height - 1) + 1)) / stride_h + 1

    out_backprop_width == (fmap_width + pad_left + pad_right - (dilation_w * (filter_width - 1) + 1)) / stride_w + 1
    """

    shape_backprop = out_backprop.get("ori_shape")
    dynamic_flag = any([i < 0 for i in shape_backprop])
    if dynamic_flag:
        return True, ""
    try:
        tensor_dict = {"weight": filter, "fmap_dedy": out_backprop, "fmap_dedx": y}
        option_input = {}
        attrs_dict = {"input_size": input_size, "strides": strides, "pads": pads, "dilations": dilations,
                      "groups": groups, "data_format": data_format, "kernel_name": kernel_name}
        check_conv2d_transpose(tensor_dict, option_input, attrs_dict, op_type="conv2d_backprop_input")
        return True, ""
    except RuntimeError as e:
        reason = e.args[1]
        return False, reason


def construct_params_for_do_op_tiling(inputs, output, attrs):
    # inputs
    #   filter, out_backprop -> input_size, filter, out_backprop
    # output
    #   y -> y
    # attrs
    #   input_size, strides, pads, dilations, groups, data_format ->
    #               strides, pads, dilations, groups, data_format
    dyn_inputs = [{
        "name": "input_size",
        "shape": [4],
        "ori_shape": [4],
        "dtype": "int32",
        "format": "ND",
        "ori_format": "ND",
        "const_value": attrs[0]
    },
                  extract_tensor_info(inputs[0]),
                  extract_tensor_info(inputs[1])]
    dyn_output = [extract_tensor_info(output)]
    dyn_attrs = ({
        "name": "strides",
        "dtype": "list_int",
        "value": attrs[1]
    }, {
        "name": "pads",
        "dtype": "list_int",
        "value": attrs[2]
    }, {
        "name": "dilations",
        "dtype": "list_int",
        "value": attrs[3]
    }, {
        "name": "groups",
        "dtype": "int",
        "value": attrs[4]
    }, {
        "name": "data_format",
        "dtype": "str",
        "value": attrs[5]
    }, {
        "name": "padding",
        "dtype": "str",
        "value": ""
    }, {
        "name": "_op_impl_mode_enum",
        "dtype": "int",
        "value": get_op_impl_mode_enum('Conv2DBackpropInput')
    })

    params = {}
    params['op_type'] = 'Conv2DBackpropInput'
    params['inputs'] = dyn_inputs
    params['outputs'] = dyn_output
    params['attrs'] = dyn_attrs
    op_context.get_context().add_addition('params_do_op_tiling', params)


def enable_binary_constant(context, inputs, output, attrs, kernel_name):
    context.set_op_mode("static")

    dict_input_size = {"shape": [4], "ori_shape": [4], "format": "ND", "ori_format": "ND"}
    dict_filter, dict_out_backprop = inputs
    generalize_shape_and_range_inplace(dict_filter)
    generalize_shape_and_range_inplace(dict_out_backprop)
    generalize_shape_and_range_inplace(output)
    strides = [-1] * 4
    pads = [-1] * 4
    dilations = [-1] * 4
    groups = -1
    data_format = attrs[5]

    dyn_impl.conv2d_backprop_input(dict_input_size, dict_filter, dict_out_backprop, output, strides, pads, dilations,
                                   groups, data_format, kernel_name)


def _need_exchange_hw_optimization(inputs, output, attrs):
    shape_filter_nchw = ShapeConverter.convert(inputs[0]['ori_shape'], inputs[0]['ori_format'], 'NCHW')
    shape_out_backprop_nchw = ShapeConverter.convert(inputs[1]['ori_shape'], inputs[1]['ori_format'], 'NCHW')
    shape_y_nchw = ShapeConverter.convert(output['ori_shape'], output['ori_format'], 'NCHW')
    strides_nchw = ShapeConverter.convert(attrs[1], inputs[0]['ori_format'], 'NCHW')
    pads = attrs[2]
    return need_exchange_hw_axis(shape_filter_nchw, shape_out_backprop_nchw, shape_y_nchw, strides_nchw[2:], pads)


def _exchange_hw(inputs, output, attrs):
    def change_shape(desc_tensor):
        desc_tensor['ori_shape'] = swap_h_w_axes_in_shape(desc_tensor['ori_shape'], desc_tensor['ori_format'])
        desc_tensor['shape'] = swap_h_w_axes_in_shape(desc_tensor['shape'], desc_tensor['format'])
    inputs = copy.deepcopy(list(inputs))
    output = copy.deepcopy(output)
    attrs = copy.deepcopy(list(attrs))

    change_shape(inputs[0])
    change_shape(inputs[1])
    change_shape(output)
    attrs[0] = swap_h_w_axes_in_shape(attrs[0], output['ori_format'])
    attrs[1] = swap_h_w_axes_in_shape(attrs[1], output['ori_format'])
    attrs[2] = [attrs[2][x] for x in (2, 3, 0, 1)]
    attrs[3] = swap_h_w_axes_in_shape(attrs[3], output['ori_format'])
    return inputs, output, attrs


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.OPTION_ATTR_LIST_INT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_STR,
    para_check.KERNEL_NAME
)
def conv2d_backprop_input_d(  # pylint: disable=W0622,C0103,R0913,R0914
    filter,
    out_backprop,
    y,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",
    kernel_name="conv2d_backprop_input",
):
    """
    algorithm: conv2d_backprop_input

    Parameters
    ----------
    filter: dict with keys(ori_shape, ori_format, shape, format, dtype)
        input weight tensor.

    out_backprop: dict with keys(ori_shape, ori_format, shape, format, dtype)
        The shape of gradients.

    y: dict with keys(ori_shape, ori_format, shape, format, dtype)
        conv2d_backprop_input output tensor, dtype must be assigned.

    input_size: tuple/list of 4 integers
        The shape of feature map. 4-D with shape [batch, height, width, channels]
        or [batch, channels, height, filter].

    strides: tuple/list of 4 integers
        filter move stride.

    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_input. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_backprop_input. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    kernel_name: str
        kernel name. Default to "conv2d_backprop_input".

    Returns
    -------
    None
    """
    update_ori_shape(out_backprop)
    tensor_dict = OrderedDict({"weight": filter, "fmap_dedy": out_backprop, "fmap_dedx": y})
    option_input = {}
    attrs_dict = OrderedDict({"input_size": input_size, "strides": strides, "pads": pads, "dilations": dilations,
                              "groups": groups, "data_format": data_format, "kernel_name": kernel_name})
    inputs = tuple(tensor_dict.values())[:2]
    attrs = tuple(attrs_dict.values())

    if _need_exchange_hw_optimization(inputs, y, attrs):
        inputs, y, attrs = _exchange_hw(inputs, y, attrs)
        log.debug(f"after exchange hw {inputs, y, attrs}")
    construct_params_for_do_op_tiling(inputs, y, attrs)

    define_operation_var_of_dx(need_define_input_vars=True)
    tensor_dedx, tensor_list = conv2d_transpose_static_impl(
        tensor_dict, option_input, attrs_dict, "conv2d_backprop_input")
    with tvm.target.cce():
        sch = tbe.dsl.auto_schedule(tensor_dedx)

    context = op_context.get_context()
    if context.get_addition("support_binary_constant"):
        context.add_addition("enable_binary_constant", True)
        log.debug("enable binary constant")
        enable_binary_constant(context, inputs, y, attrs, kernel_name)
        return

    config = {"name": kernel_name, "tensor_list": tensor_list, "limit_ir_lines": [True, ]}
    if context.get_addition("split_w_flag"):
        config.update({"FullySplitLoopPartition": {"partition_only_inner": -1}})
    tbe.dsl.build(sch, config)


@tbe_platform.fusion_manager.register("conv2d_backprop_input_d")
def conv2d_backprop_input_d_compute(  # pylint: disable=C0103,W0622,R0913,R0914
    filter,
    out_backprop,
    y,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",  # pylint: disable=W0613
    kernel_name="conv2d_backprop_input",
):
    """
    used for fusion
    Parameters
    ----------
    filter: Tensor
        input weight tensor.

    out_backprop: Tensor
        conv2d output gradients tenosr.

    y: dict with keys(shape and dtype)
        conv2d_backprop_input output tensor, dtype must be assigned.

    input_size: tuple/list of 4 integers
        The shape of feature map. 4-D with shape [batch, height, width, channels]
        or [batch, channels, height, filter].

    strides: tuple/list of 4 integers
        filter move stride.

    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_input. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_backprop_input. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    kernel_name: str
        kernel name. Default to "conv2d_backprop_input".

    Returns
    -------
    Tensor of conv2d_backprop_input
    """
    tensor_dict = OrderedDict({"weight": filter, "fmap_dedy": out_backprop, "fmap_dedx": y})
    option_input = {}
    attrs_dict = OrderedDict({"input_size": input_size, "strides": strides, "pads": pads, "dilations": dilations,
                              "groups": groups, "data_format": data_format, "kernel_name": kernel_name})
    inputs = tuple(tensor_dict.values())[:2]
    attrs = tuple(attrs_dict.values())

    construct_params_for_do_op_tiling(inputs, y, attrs)
    define_operation_var_of_dx(need_define_input_vars=True)
    return conv2d_transpose_static_compute(tensor_dict, option_input, attrs_dict, "conv2d_backprop_input")
