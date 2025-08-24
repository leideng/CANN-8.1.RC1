#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

conv2d_backprop_input
"""

from __future__ import absolute_import
import warnings

from impl.util.platform_adapter import classify
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util.util_cube_dynamic import Conv2dBackpropParaProcess
from impl.util.util_cube_dynamic import CubeParaProcess
from impl.util.util_cube_dynamic import check_dynamic_range_lower
from impl.util.util_cube_dynamic import check_tensor_shape
from impl.util.util_cube_dynamic import correct_range
from impl.util.util_cube_dynamic import check_dynamic_mode
from impl.util.util_cube_dynamic import check_input_output_format_and_shape
from impl.util.util_cube_dynamic import check_l1_size
from impl.util.util_cube_dynamic import generalize_shape_and_range
from impl.util.util_cube_dynamic import is_empty_tensor_scene
from impl.util.util_cube_dynamic import set_default_para
from impl.util.util_deconv_comm import get_op_support_info_conv2d_transpose
from tbe.common.context import op_context
from tbe.common.utils.const import SPLIT_AXIS_MODE_STR


H_DIM = 2
W_DIM = 3
OP_TYPE = "conv2d_backprop_input"
ALL_DYNAMIC_VALUE = [-1, -1, -1, -1]
FIX_FLAG = 0
DYNAMIC_FLAG = -1
UNKNOWN_FLAG = -2
# TBE sync data optimiztion
MUTILAYER_SYNC_OPTI = 2


def get_op_support_info(input_size, filter, out_backprop, y, strides,
                        pads, dilations=(1, 1, 1, 1), groups=1,
                        data_format="NHWC", kernel_name="conv2d_backprop_input"):
    """
    get the conv2d_backprop_input split info

    """
    tensor_dict = {"input_size": input_size, "weight": filter, "fmap_dedy": out_backprop, "fmap_dedx": y}
    option_input = {}
    attrs_dict = {"strides": strides, "pads": pads, "dilations": dilations,
                  "groups": groups, "data_format": data_format, "kernel_name": kernel_name}
    return get_op_support_info_conv2d_transpose(tensor_dict, option_input, attrs_dict,
                                                op_type="conv2d_backprop_input", mode="dynamic")


@tbe_register.register_param_generalization("Conv2DBackpropInput")
def conv2d_backprop_input_generalization(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                                         filter, out_backprop, y, strides,
                                         pads, dilations=(1, 1, 1, 1),
                                         groups=1, data_format="NHWC",
                                         kernel_name="conv2d_backprop_input",
                                         generalize_config=None):
    """
    conv2d backprop input generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to conv2d_backprop_input

    generalize_config: generalization mode, string.

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    result = []
    if generalize_config.get("mode") == "keep_rank":
        dynamic_flag = check_dynamic_mode(out_backprop)
        if dynamic_flag == UNKNOWN_FLAG:
            warnings.warn("{} not support unknow_rank".format(OP_TYPE))
            return [{"result": "UNSUPPORTED"}]
        result = check_input_output_format_and_shape([out_backprop, y], OP_TYPE)
        if result:
            return result
        input_list = [out_backprop, filter, None, y]
        attr_list = [strides, pads, dilations, data_format, None]
        result = check_l1_size(input_list, attr_list, dynamic_flag)
        if result:
            return result
        input_size["const_value"] = None
        result.append([input_size, filter, out_backprop, y, {"strides": strides}, {"pads": pads},
                        {"dilations": dilations}, {"groups": groups}, {"data_format": data_format},
                        {"kernel_name": kernel_name}])
        ori_range_x, ori_shape_x = generalize_shape_and_range(
            out_backprop["ori_format"], out_backprop["ori_shape"])
        ori_range_filter, ori_shape_filter = generalize_shape_and_range(filter["ori_format"], filter["ori_shape"])
        out_backprop["ori_range"], filter["ori_range"] = ori_range_x, ori_range_filter
        out_backprop["ori_shape"], filter["ori_shape"] = ori_shape_x, ori_shape_filter
        result[0][1], result[0][2] = filter, out_backprop
    elif generalize_config.get("mode") == "all_shape":
        range_filter, shape_filter = generalize_shape_and_range(filter["format"], filter["shape"])
        range_out_bp, shape_out_bp = generalize_shape_and_range(out_backprop["format"], out_backprop["shape"])
        range_y, shape_y = generalize_shape_and_range(y["format"], y["shape"])
        filter["range"], out_backprop["range"], y["range"] = range_filter, range_out_bp, range_y
        filter["shape"], out_backprop["shape"], y["shape"] = shape_filter, shape_out_bp, shape_y
        input_size["ori_format"] = "NCHW"
        input_size["format"] = "NCHW"
        filter["ori_format"] = "NCHW"
        out_backprop["ori_format"] = "NCHW"
        y["ori_format"] = "NCHW"
        strides = ALL_DYNAMIC_VALUE
        pads = ALL_DYNAMIC_VALUE
        dilations = ALL_DYNAMIC_VALUE
        groups = -1
        data_format = "NCHW"
        result.append([input_size, filter, out_backprop, y, strides, pads, dilations, groups, data_format])
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


@register_operator_compute("Conv2DBackpropInput", op_mode="dynamic", support_fusion=True)
@para_check.check_input_type((dict, tvm.Tensor), (dict, tvm.Tensor), (dict, tvm.Tensor),
                             dict, (tuple, list), (tuple, list), (tuple, list), int, str, str)
def conv2dbp_input_fusion_compute(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                                  filters, out_backprop, y, strides, pads, dilations=(1, 1, 1, 1),
                                  groups=1, data_format='NHWC', kernel_name='conv2d_backprop_input',
                                  build_option=None):
    """
    algorithm: conv2d_backprop_input

    Parameters
    ----------
    input_size: Tensor or dict, will not be used input tensor size.

    filter: Tensor or dict w, convolution kernel.

    out_backprop: Tensor or dict, gradients.

    y: dict with keys(ori_shape, ori_format, dtype and range) conv2d_backprop_input output tensor

    strides: tuple/list of 4 integers, filter move stride

    pads: tuple/list of 4 integers, [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers, filter expand size of dilated conv2d_backprop_input

    groups: int, param for group conv2d_backprop_input

    data_format: str, An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
    Specify the data format of the input and output data.

    kernel_name: str, kernel name, default value is "conv2d_backprop_input"

    build_option: dict, some extra params, contains [axis_split_mode] currently

    Returns
    -------
    None
    """

    # set fusion build config
    build_cfg = {"constant_realize_extent_in_infer_bound": False}
    tbe_register.set_fusion_buildcfg("Conv2DBackpropInput", build_cfg)
    res = _conv2d_backprop_input_compute(input_size, filters, out_backprop, y, strides,
                                          pads, dilations, groups, data_format, kernel_name, build_option)
    if isinstance(out_backprop, tvm.Tensor):
        return res.get('op_res')[0]
    return res


def check_empty_tensor(filters, out_backprop, y, strides, pads, dilations=(1, 1, 1, 1)):
    if check_dynamic_range_lower([filters, out_backprop, y]) or is_empty_tensor_scene([filters, out_backprop, y]):
        data_format = y.get("ori_format")
        paras = {"data_format": data_format, "pads": pads, "strides": strides, "dilations": dilations}
        cube_para = CubeParaProcess(paras)

        cube_para.get_attr_nchw(data_format)
        stride_nchw = cube_para.strides
        dilation_nchw = cube_para.dilations

        w_shape_nchw = cube_para.get_input_nchw(filters.get("ori_shape"), filters.get("ori_format"))
        fmap_shape_nchw, fmap_range_nchw = cube_para.get_input_nchw(y.get("ori_shape"),
                                                                          y.get("ori_format"),
                                                                          y.get("range"))

        if fmap_shape_nchw[1] == 0 or 0 in w_shape_nchw[1:]:
            error_manager_cube.raise_err_specific_user("conv2d_backprop_input", "fmap_c weight_cdhw not support 0")
        check_tensor_shape({"tensor": [filters, out_backprop, y],
                            "value": [1, -1, -1],
                            "range": [(1, 1), (1, 1), (1, 1)]})

        if list(out_backprop.get("ori_shape")) != [-2]:
            correct_range(y, fmap_range_nchw, w_shape_nchw, stride_nchw, dilation_nchw, pads, 'NCHW', True)


def _conv2d_backprop_input_compute(input_size, filters, out_backprop, y, strides, pads, dilations=(1, 1, 1, 1),
                                   groups=1, data_format='NHWC', kernel_name='conv2d_backprop_input',
                                   build_option=None):  # pylint: disable=invalid-name, R0913
    if isinstance(out_backprop, dict):
        check_empty_tensor(filters, out_backprop, y, strides, pads)
    ori_paras = {
        "input_size": input_size, "filters": filters, "out_backprop": out_backprop, "y": y,
        "strides": strides, "pads": pads, "dilations": dilations, "groups": groups, "data_format": data_format,
        "kernel_name": kernel_name
    }
    if build_option is not None:
        ori_paras[SPLIT_AXIS_MODE_STR] = build_option.get(SPLIT_AXIS_MODE_STR, 0)

    default_para = set_default_para()
    if isinstance(input_size, dict) and not input_size.get("ori_shape"):
        ori_paras["input_size"]["ori_shape"] = default_para["input_size"]["ori_shape"]
    conv2dbp_para = Conv2dBackpropParaProcess(ori_paras)
    conv2dbp_para.config_paras()
    res_dtype = y.get("dtype").lower()
    dedx = tbe.conv2d_backprop_input(filters=conv2dbp_para.tensors.get("filter_tensor"),
                                     out_backprop=conv2dbp_para.tensors.get("dy_tensor"),
                                     filter_sizes=conv2dbp_para.shape.get("filter_shape_nchw"),
                                     input_sizes=conv2dbp_para.shape.get("dx_shape_nchw"),
                                     para_dict={
                                         "strides":
                                             (conv2dbp_para.strides[H_DIM], conv2dbp_para.strides[W_DIM]),
                                         "padding": conv2dbp_para.pads,
                                         "dilations": conv2dbp_para.dilations,
                                         "res_dtype": res_dtype,
                                         "kernel_name": kernel_name,
                                         "group_dict": conv2dbp_para.attrs.get("group_para"),
                                         "correct_range_flag": conv2dbp_para.attrs.get("correct_range_flag", False),
                                         "binary_mode": conv2dbp_para.binary_mode,
                                         "ori_tensors": _collect_ori_tensors(ori_paras),
                                         "split_axis_mode": conv2dbp_para.split_axis_mode,
                                         "op_type": "Conv2DBackpropInput"
                                     })

    context = op_context.get_context()
    if context.get_addition("enable_binary_constant"):
        op_placeholder = [conv2dbp_para.tensors.get("filter_tensor"), conv2dbp_para.tensors.get("dy_tensor")]
    else:
        op_placeholder = [
            conv2dbp_para.tensors.get("input_tensor"),
            conv2dbp_para.tensors.get("filter_tensor"),
            conv2dbp_para.tensors.get("dy_tensor")
        ]
    return {'op_placeholder': op_placeholder, 'op_res': [dedx]}


@tbe_register.register_operator('Conv2DBackpropInput')
@para_check.check_input_type(dict, dict, dict, dict, (tuple, list),
                             (tuple, list), (tuple, list), int, str, str,
                             (type(None), dict))
def conv2d_backprop_input(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                          filter, out_backprop, y, strides,
                          pads, dilations=(1, 1, 1, 1),
                          groups=1, data_format="NHWC",
                          kernel_name="conv2d_backprop_input"):
    """
    algorithm: conv2d_backprop_input

    Parameters
    ----------
    input_size: dict, will not be used
            input tensor size.

    filter: dict with keys(ori_shape, ori_format, dtype)
            convolution kernel

    out_backprop: dict with keys(ori_shape, ori_format, dtype)
                  gradients.

    y: dict with keys(ori_shape, ori_format, dtype and range)
       conv2d_backprop_input output tensor

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
          str: "SAME" or "VALID"
          tuple/list of 4 integers: [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated conv2d_backprop_input
    groups: int
            param for group conv2d_backprop_input

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    kernel_name: str
            kernel name, default value is "conv2d_backprop_input"

    Returns
    -------
    None
    """
    input_list = [input_size, filter, out_backprop, y]
    attr_list = [strides, pads, dilations, groups, data_format]

    context = op_context.get_context()
    extra_params = {
        "single_op": True
    }
    build_args = {'constant_realize_extent_in_infer_bound': False,
                  'predicate_total_out_of_bound': False,
                  'sync_opt_for_notail_db': MUTILAYER_SYNC_OPTI,
                  'sync_opt_for_preload_loop_zero': True}
    if context.get_addition("enable_binary_constant"):
        extra_params.update({"need_expand_stride": context.get_addition("need_expand_stride"),
                             "split_w": context.get_addition("split_w")})
        ins = classify([input_list, attr_list], "conv2d_backprop_input", extra_params)
    else:
        ins = classify([input_list, attr_list], "conv2d_backprop_input", extra_params)
        build_args['enable_db_fold'] = True

    schedules, tensors_list = [], []
    for input_param in ins:
        tensors = []
        with tbe.compute():
            res = _conv2d_backprop_input_compute(*input_param)

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
