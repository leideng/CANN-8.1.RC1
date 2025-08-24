#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

conv2d_transpose
"""

from __future__ import absolute_import
import warnings

from impl.dynamic.conv2d_backprop_input import check_empty_tensor
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util.util_cube_dynamic import Conv2dTransposeParaProcess
from impl.util.util_cube_dynamic import check_dynamic_mode
from impl.util.util_cube_dynamic import check_input_output_format_and_shape
from impl.util.util_cube_dynamic import check_l1_size
from impl.util.util_cube_dynamic import generalize_shape_and_range
from impl.util.util_cube_dynamic import set_default_para
from impl.util.util_deconv_comm import get_op_support_info_conv2d_transpose
from tbe.common.context import op_context
from tbe.common.utils.const import SPLIT_AXIS_MODE_STR
from tbe.common.platform import platform_info


H_DIM = 2
W_DIM = 3
OP_TYPE = "conv2d_transpose"
FIX_FLAG = 0
DYNAMIC_FLAG = -1
UNKNOWN_FLAG = -2
ALL_DYNAMIC_VALUE = (-1, -1, -1, -1)
FUZZY_RES_LEN = 14


def get_op_support_info(input_size, x, filter, bias, offset_w, y, strides,
                        pads, dilations=(1, 1, 1, 1), groups=1, data_format="NHWC",
                        output_padding=(0, 0, 0, 0), offset_x=0, kernel_name="conv2d_transpose"):
    """
    get the conv2d_transpose_d split

    """
    tensor_dict = {"weight": filter, "fmap_dedy": x, "fmap_dedx": y, "input_size": input_size}
    option_input = {"bias": bias, "offset_w": offset_w}
    attrs_dict = {"strides": strides, "pads": pads, "dilations": dilations,
                  "groups": groups, "data_format": data_format, "output_padding": output_padding,
                  "offset_x": offset_x, "kernel_name": kernel_name}
    return get_op_support_info_conv2d_transpose(tensor_dict, option_input, attrs_dict,
                                                op_type="conv2d_transpose", mode="dynamic")


def check_supported(input_size, x, weight, bias, offset_w, y, strides, pads,
                    dilations, groups, data_format, output_padding, offset_x):
    """
    check the op support situation
    """
    support_conv_ub_to_ub = platform_info.intrinsic_check_support("Intrinsic_conv_ub_to_ub")
    if not support_conv_ub_to_ub:
        return True, ""
    dtype = "dtype"
    x_data_type = x.get(dtype)
    weight_data_type = weight.get(dtype)
    bias_data_type = bias.get(dtype) if bias is not None else None
    y_data_type = y.get(dtype)

    is_valid_dtype = x_data_type in ["int16", "int8"] and weight_data_type in ["int8"] and \
                     (bias_data_type is None or bias_data_type in ["int32"]) and y_data_type in ["int32"]
    res = True, ""
    if not is_valid_dtype:
        reason = "Not support the given input and output data type. x: %s, weights: %s, bias: %s, y: %s."\
                 % (x_data_type, weight_data_type, bias_data_type, y_data_type)
        res = False, reason
    return res


@tbe_register.register_param_generalization("Conv2DTranspose")
def conv2d_transpose_generalization(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                                    x, filter, bias, offset_w, y, strides,
                                    pads, dilations=(1, 1, 1, 1),
                                    groups=1, data_format="NHWC", output_padding=(0, 0, 0, 0), offset_x=0,
                                    kernel_name=OP_TYPE,
                                    generalize_config=None):
    """
    conv2d transpose generalization

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
       conv2d_transpose output tensor

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
          str: "SAME" or "VALID"
          tuple/list of 4 integers: [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated conv2d_transpose
    groups: int
            param for group conv2d_transpose

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    output_padding: tuple/list of 4 integers
        The size will be added in the output shape. Default to (0, 0, 0, 0).

    offset_x: int
        offset of gradients in quant mode. Default to 0.

    kernel_name: str
            kernel name, default value is "conv2d_transpose"

    generalize_config: dict, generaliazation mode.

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    result = []
    if generalize_config.get("mode") == "keep_rank":
        dynamic_flag = check_dynamic_mode(x)
        if dynamic_flag == UNKNOWN_FLAG:
            warnings.warn("{} not support unknow_rank".format(OP_TYPE))
            return [{"result": "UNSUPPORTED"}]
        result = check_input_output_format_and_shape([x, y], OP_TYPE)
        if result:
            return result
        input_list = [x, filter, bias, y]
        attr_list = [strides, pads, dilations, data_format, output_padding]
        result = check_l1_size(input_list, attr_list, dynamic_flag)
        if result:
            return result
        input_size["const_value"] = None
        result.append([input_size, x, filter, bias, offset_w, y, {"strides": strides},
                    {"pads": pads}, {"dilations": dilations}, {"groups": groups}, {"data_format": data_format},
                    {"output_padding": output_padding}, {"offset_x": offset_x}, {"kernel_name": kernel_name}])
        ori_range_x, ori_shape_x = generalize_shape_and_range(x["ori_format"], x["ori_shape"])
        ori_range_filter, ori_shape_filter = generalize_shape_and_range(filter["ori_format"], filter["ori_shape"])
        if bias is not None:
            ori_range_bias, ori_shape_bias = generalize_shape_and_range(bias["ori_format"], bias["ori_shape"], 1)
            bias["ori_range"] = ori_range_bias
            bias["ori_shape"] = ori_shape_bias
        x["ori_range"], filter["ori_range"] = ori_range_x, ori_range_filter
        x["ori_shape"], filter["ori_shape"] = ori_shape_x, ori_shape_filter
        result[0][1], result[0][2], result[0][3] = x, filter, bias
    elif generalize_config.get("mode") == "all_shape":
        # mod shape and range for binary reuse
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
        input_size["ori_format"] = "NCHW"
        input_size["format"] = "NCHW"
        input_size["dtype"] = "int32"
        x["ori_format"] = "NCHW"
        filter["ori_format"] = "NCHW"
        y["ori_format"] = "NCHW"
        # mod attr for binary reuse
        offset_w = None
        strides = ALL_DYNAMIC_VALUE
        pads = ALL_DYNAMIC_VALUE
        dilations = ALL_DYNAMIC_VALUE
        output_padding = ALL_DYNAMIC_VALUE
        groups = -1
        offset_x = 0
        data_format = "NCHW"
        result.append([input_size, x, filter, bias, offset_w, y, strides, pads, dilations,
                    groups, data_format, output_padding, offset_x])
    return result


def _collect_ori_tensors(ori_paras):
    """
    get valid tensors
    """
    ori_tensors = {}
    for key, value in ori_paras.items():
        valid_tensor = isinstance(value, dict) \
                       and isinstance(value.get("ori_shape"), (list, tuple)) \
                       and len(value.get("ori_shape")) > 0
        if valid_tensor:
            ori_tensors[key] = value
    return ori_tensors


@register_operator_compute("Conv2DTranspose", op_mode="dynamic", support_fusion=True)
@para_check.check_input_type((dict, tvm.Tensor), (dict, tvm.Tensor), (dict, tvm.Tensor),
                             (type(None), dict, tvm.Tensor), (type(None), dict), dict,
                             (tuple, list), (tuple, list), (tuple, list), int, str, (tuple, list), int, str)
def conv2d_transpose_fusion_compute(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                                    x, filter, bias, offset_w, y, strides,
                                    pads, dilations=(1, 1, 1, 1),
                                    groups=1, data_format="NHWC", output_padding=(0, 0, 0, 0), offset_x=0,
                                    kernel_name="conv2d_transpose", build_option=None):
    """
    algorithm: conv2d_transpose

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
       conv2d_transpose output tensor

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
          str: "SAME" or "VALID"
          tuple/list of 4 integers: [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated conv2d_transpose
    groups: int
            param for group conv2d_transpose

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    output_padding: tuple/list of 4 integers
        The size will be added in the output shape. Default to (0, 0, 0, 0).

    offset_x: int
        offset of gradients in quant mode. Default to 0.

    kernel_name: str
            kernel name, default value is "conv2d_transpose"
    build_option: dict
            some extra params, contains [axis_split_mode] currently

    Returns
    -------
    None
    """

    # set fusion build config
    support_conv_ub_to_ub = platform_info.intrinsic_check_support("Intrinsic_conv_ub_to_ub")
    if support_conv_ub_to_ub:
        res = _conv2d_transpose_ub_compute(input_size, x, filter, bias, offset_w, y, strides, pads, dilations, groups,
                                           data_format, output_padding, offset_x, kernel_name)
        return res

    build_cfg = {"constant_realize_extent_in_infer_bound": False}
    tbe_register.set_fusion_buildcfg("Conv2DTranspose", build_cfg)
    res = _conv2d_transpose_compute(input_size, x, filter, bias, offset_w, y, strides, pads, dilations, groups,
                                    data_format, output_padding, offset_x, kernel_name)
    if isinstance(x, tvm.Tensor):
        return res.get('op_res')[0]
    return res


def _conv2d_transpose_ub_compute(input_size, x, kernel, bias, offset_w,
                              y, strides, pads,
                              dilations=(1, 1, 1, 1),
                              groups=1, data_format='NHWC', output_padding=(0, 0, 0, 0), offset_x=0,
                              kernel_name='conv2d_transpose'):
    res_dtype = y.get("dtype").lower()
    kernel_sizes = tuple(map(int, list(kernel.op.attrs.get("ori_shape"))))

    dedx = tbe.conv2d_backprop_input(
        filters=kernel,
        out_backprop=x,
        filter_sizes=kernel_sizes,
        input_sizes=y.get("ori_shape"),
        para_dict={
            "strides":(strides[H_DIM], strides[W_DIM]),
            "padding": pads,
            "dilations": dilations,
            "res_dtype": res_dtype,
            "tensor_bias": bias,
            "offset_x": offset_x,
            "kernel_name": kernel_name,
            "op_type": "Conv2DTranspose"})
    return dedx


def _conv2d_transpose_compute(input_size, x, filter, bias, offset_w,
                              y, strides, pads,
                              dilations=(1, 1, 1, 1),
                              groups=1, data_format='NHWC', output_padding=(0, 0, 0, 0), offset_x=0,
                              kernel_name='conv2d_transpose', build_option=None):
    if isinstance(x, dict):
        check_empty_tensor(filter, x, y, strides, pads)
    ori_paras = {
        "input_size": input_size, "x": x, "filters": filter, "bias": bias, "offset_w": offset_w, "y": y,
        "strides": strides, "pads": pads, "dilations": dilations, "groups": groups, "data_format": data_format,
        "output_padding": output_padding, "offset_x": offset_x, "kernel_name": kernel_name
    }
    if build_option is not None:
        ori_paras[SPLIT_AXIS_MODE_STR] = build_option.get(SPLIT_AXIS_MODE_STR, 0)

    default_para = set_default_para()
    if isinstance(input_size, dict) and not input_size.get("ori_shape"):
        ori_paras["input_size"]["ori_shape"] = default_para["input_size"]["ori_shape"]
    conv2d_transpose_para = Conv2dTransposeParaProcess(ori_paras)
    conv2d_transpose_para.config_paras()
    res_dtype = y.get("dtype").lower()
    dedx = tbe.conv2d_backprop_input(
        filters=conv2d_transpose_para.tensors.get("filter_tensor"),
        out_backprop=conv2d_transpose_para.tensors.get("x_tensor"),
        filter_sizes=conv2d_transpose_para.shape.get("filter_shape_nchw"),
        input_sizes=conv2d_transpose_para.shape.get("dx_shape_nchw"),
        para_dict={
            "strides":(conv2d_transpose_para.strides[H_DIM], conv2d_transpose_para.strides[W_DIM]),
            "padding": conv2d_transpose_para.pads,
            "dilations": conv2d_transpose_para.dilations,
            "res_dtype": res_dtype,
            "tensor_bias": conv2d_transpose_para.tensors.get("bias_tensor"),
            "offset_x": offset_x,
            "kernel_name": kernel_name,
            "group_dict": conv2d_transpose_para.attrs.get("group_para"),
            "correct_range_flag": conv2d_transpose_para.attrs.get("correct_range_flag", False),
            "binary_mode": conv2d_transpose_para.binary_mode,
            "ori_tensors": _collect_ori_tensors(ori_paras),
            "split_axis_mode": conv2d_transpose_para.split_axis_mode,
            "bias_ori_shape": conv2d_transpose_para.shape.get("dx_shape_nchw")[1],
            "op_type": "Conv2DTranspose"})

    op_placeholder = [
        conv2d_transpose_para.tensors.get("input_tensor"),
        conv2d_transpose_para.tensors.get("x_tensor"),
        conv2d_transpose_para.tensors.get("filter_tensor")
    ]
    if bias:
        bias_dtype = bias.get("dtype").lower()
        para_check.check_dtype_rule(bias_dtype, ("float16", "float32", "int32"), "bias")
        op_placeholder.append(conv2d_transpose_para.tensors.get("bias_tensor"))

    context = op_context.get_context()
    if context.get_addition("enable_binary_constant"):
        op_placeholder = op_placeholder[1:]
    return {'op_placeholder': op_placeholder, 'op_res': [dedx]}


@register_operator('Conv2DTranspose')
@para_check.check_input_type(dict, dict, dict, (type(None), dict), (type(None), dict), dict, (tuple, list),
                             (tuple, list), (tuple, list), int, str, (tuple, list), int, str,
                             (type(None), dict))
def conv2d_transpose(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                     x, filter, bias, offset_w, y, strides,
                     pads, dilations=(1, 1, 1, 1),
                     groups=1, data_format="NHWC", output_padding=(0, 0, 0, 0), offset_x=0,
                     kernel_name="conv2d_transpose"):
    """
    algorithm: conv2d_transpose

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
       conv2d_transpose output tensor

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
          str: "SAME" or "VALID"
          tuple/list of 4 integers: [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated conv2d_transpose
    groups: int
            param for group conv2d_transpose

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    output_padding: tuple/list of 4 integers
        The size will be added in the output shape. Default to (0, 0, 0, 0).

    offset_x: int
        offset of gradients in quant mode. Default to 0.

    kernel_name: str
            kernel name, default value is "conv2d_transpose"

    Returns
    -------
    None
    """
    input_list = [input_size, x, filter, bias, offset_w, y]
    attr_list = [strides, pads, dilations, groups, data_format, output_padding, offset_x, kernel_name]
    extra_params = {
        "single_op": True
    }
    context = op_context.get_context()
    build_args = {'constant_realize_extent_in_infer_bound': False,
                  'predicate_total_out_of_bound': False}
    if context.get_addition("enable_binary_constant"):
        extra_params.update({"need_expand_stride": context.get_addition("need_expand_stride"),
                             "split_w": context.get_addition("split_w")})
        ins = classify([input_list, attr_list], "conv2d_transpose", extra_params)
    else:
        ins = classify([input_list, attr_list], "conv2d_transpose", extra_params)
        build_args['enable_db_fold'] = True
    schedules, tensors_list = [], []
    for input_param in ins:
        with tbe.compute():
            res = _conv2d_transpose_compute(*input_param)

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
