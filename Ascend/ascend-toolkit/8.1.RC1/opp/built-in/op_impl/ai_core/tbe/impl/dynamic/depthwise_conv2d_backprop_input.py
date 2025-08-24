#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

conv2d_backprop_input
"""

from __future__ import absolute_import
import warnings
from impl.util.util_conv2d import transform_shape_with_format
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_register
from impl.util.util_cube_dynamic import DepthwiseConv2dBackpropParaProcess
from impl.util.util_cube_dynamic import check_graph_mode
from impl.util.util_cube_dynamic import check_graph_range
from impl.util.util_cube_dynamic import Conv2dTransposeParaProcess
from impl.util.util_cube_dynamic import gen_conv_shape_range
from impl.util.util_cube_dynamic import modify_dy_w_range_max_opti
from impl.util.util_cube_dynamic import check_modify_w_range
from impl.util.util_cube_dynamic import set_default_para
from impl.dynamic.conv2d_backprop_input import check_empty_tensor
from tbe.common.utils.const import SPLIT_AXIS_MODE_STR

NONETYPE = type(None)
H_DIM = 2
W_DIM = 3
ORI_SHAPE_LEN = 4
SHAPE_LEN = 5
OP_TYPE = "depthwise_conv2d_backprop_input"
DATA_FORMAT_WHITE_LIST = ["NCHW", "NHWC"]
TAR_FORMAT = "NCHW"
LOWER_STR = [{"result": "UNSUPPORTED", "reason": {"param_index": [2], "type": ["lower_limit"]}}]
UPPER_STR = [{"result": "UNSUPPORTED", "reason": {"param_index": [2], "type": ["upper_limit"]}}]
UNSUPPORED_STR = [{"result": "UNSUPPORTED"}]
DYNAMIC_RANK_FLAG = [-2]
DYNAMIC_FMAP_W_MAX = 4096


def _check_opti_scene(filter_grad):
    """
    Both filter H/W is 1 is opti scene
    """
    filter_h = filter_grad.get("ori_shape")[filter_grad.get("ori_format").find("H")]
    filter_w = filter_grad.get("ori_shape")[filter_grad.get("ori_format").find("W")]
    return filter_h == 1 and filter_w == 1


def check_or_update_w_range(input_list, param_list, op_type, dynamic_flag):
    """
    update w_range
    """
    [_, filter_grad, out_backprop] = input_list

    supports = "no_limit"
    if _check_opti_scene(filter_grad):
        opti_graph_correct, dedy_modify = modify_dy_w_range_max_opti(out_backprop, filter_grad, param_list,
                                                                     op_type, dynamic_flag)
        if not opti_graph_correct:
            supports = dedy_modify[0].get("reason").get("type")[0]
        dedy_range_w = dedy_modify.get("ori_range")[dedy_modify.get("ori_format").find("W")]
        strides, _, _, data_format = param_list
        fmap_range_w = [min(dedy_range_w[0] * strides[data_format.find("W")], DYNAMIC_FMAP_W_MAX),
                        min(dedy_range_w[1] * strides[data_format.find("W")], DYNAMIC_FMAP_W_MAX)]
    else:
        supports, dedy_range_w, fmap_range_w = check_modify_w_range(input_list, param_list, op_type, dynamic_flag)
    return supports, dedy_range_w, fmap_range_w


def check_input_para(input_size, input_list, param_list,
                     kernel_name="depthwise_conv2d_backprop_input"):
    """
    check inputs parameters
    """
    [input_grad, filter_grad, out_backprop] = input_list
    [strides, pads, dilations, data_format] = param_list
    ori_paras = {
        "input_size": input_size, "filters": filter_grad, "out_backprop": out_backprop, "input_grad": input_grad,
        "strides": strides, "pads": pads, "dilations": dilations, "data_format": data_format,
        "kernel_name": kernel_name
    }
    depthwise_conv2dbp_para = DepthwiseConv2dBackpropParaProcess(ori_paras)
    depthwise_conv2dbp_para.check_paras()


def _update_ori_shape(tensor):
    """
    update static ori_shape to dynamic
    """
    tensor["ori_shape"] = (-1, tensor["ori_shape"][1], -1, -1) \
        if tensor.get("ori_format") == TAR_FORMAT else (-1, -1, -1, tensor["ori_shape"][3])
    return tensor


@tbe_register.register_param_generalization("DepthwiseConv2DBackpropInput")
def depthwise_conv2d_backprop_input_generalization(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                                                   filter, out_backprop, input_grad, strides,
                                                   dilations=(1, 1, 1, 1), pads=(0, 0, 0, 0), data_format="NHWC",
                                                   kernel_name="depthwise_conv2d_backprop_input",
                                                   generalize_config=None):
    """
    depthwise conv2d backprop input generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to depthwise_conv2d_backprop_input

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    support_mode = ["keep_rank"]
    is_generalize_config = (generalize_config is not None and generalize_config.get("mode") in support_mode)
    dynamic_flag = check_graph_mode(out_backprop)
    support_format = ["NCHW", "NHWC"]
    unsupported_dict = {"upper_limit": UPPER_STR, "lower_limit": LOWER_STR, "unsupported": UNSUPPORED_STR}
    if not is_generalize_config:
        return
    result = []
    # unknow_rank x ori_shape is [-2], others' shape length is 4
    valid = (isinstance(out_backprop.get("ori_shape"), (list, tuple))
             and len(out_backprop.get("ori_shape")) == ORI_SHAPE_LEN
             and list(out_backprop.get("ori_shape")) != DYNAMIC_RANK_FLAG
             and out_backprop.get("ori_format") in support_format)
    if not valid:
        warnings.warn(f'In {kernel_name}, please check ori_shape and ori_format.')
        return UNSUPPORED_STR
    try:
        out_backprop = gen_conv_shape_range(out_backprop, kernel_name, dynamic_flag)
    except RuntimeError as err:
        warnings.warn(err)
        return UNSUPPORED_STR
    finally:
        pass
    message = check_graph_range(out_backprop, kernel_name, dynamic_flag)
    if message:
        return unsupported_dict.get(message)
    input_list = [input_grad, filter, out_backprop]
    param_list = [strides, pads, dilations, data_format]
    supports, dedy_range_w, fmap_range_w = \
        check_or_update_w_range(input_list, param_list, OP_TYPE, dynamic_flag)
    if unsupported_dict.get(supports):
        return unsupported_dict.get(supports)
    try:
        check_input_para(input_size, input_list, param_list, kernel_name)
    except RuntimeError as err:
        warnings.warn(err)
        return UNSUPPORED_STR
    finally:
        pass

    if not dynamic_flag:
        dy_range = out_backprop.get("ori_range")
        ori_data_format = out_backprop.get("ori_format")
        ori_paras = {
            "input_size": input_size, "x": out_backprop, "filters": filter, "bias": None, "offset_w": None,
            "y": input_grad,
            "strides": strides, "pads": pads, "dilations": dilations, "data_format": data_format,
            "output_padding": (0, 0, 0, 0), "offset_x": 0, "kernel_name": kernel_name
        }
        conv2d_tranpose = Conv2dTransposeParaProcess(ori_paras)
        conv2d_tranpose.correct_pads(input_grad, out_backprop, filter)
        conv2d_tranpose.get_attr_nchw(data_format)
        dy_shape_nchw = conv2d_tranpose.get_input_nchw(out_backprop.get("ori_shape"), out_backprop.get("ori_format"))
        filter_shape_nchw = conv2d_tranpose.get_input_nchw(filter.get("ori_shape"), filter.get("ori_format"))
        _, dy_range_nchw = conv2d_tranpose.get_input_nchw(dy_shape_nchw, ori_data_format, dy_range)
        dy_range_nchw[3] = dedy_range_w
        dx_range_nchw, _, new_dy_range = conv2d_tranpose.get_input_range(filter_shape_nchw, dy_range_nchw)
        dx_c = input_grad.get("ori_shape")[input_grad.get("ori_format").find("C")]
        dx_range_nchw[1] = [dx_c, dx_c]
        if fmap_range_w[0] == fmap_range_w[1]:
            dx_range_nchw[3] = fmap_range_w
        out_backprop["ori_range"] = list(out_backprop["ori_range"])
        out_backprop["ori_range"][out_backprop.get("ori_format").find("H")] = new_dy_range[2]
        out_backprop["ori_range"][out_backprop.get("ori_format").find("W")] = new_dy_range[3]
        input_size["const_value"] = None
        input_size["const_value_range"] = transform_shape_with_format(TAR_FORMAT, data_format,
                                                                      dx_range_nchw, DATA_FORMAT_WHITE_LIST)
        # modify tesnors have range
        out_backprop = _update_ori_shape(out_backprop)
        input_grad = _update_ori_shape(input_grad)
    result.append([input_size, filter, out_backprop, input_grad, {"strides": strides}, {"pads": pads},
                  {"dilations": dilations}, {"data_format": data_format}])
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


def _depthwise_conv2d_backprop_input_compute(
        input_size, filters, out_backprop, input_grad, strides, pads, dilations=(1, 1, 1, 1), data_format='NHWC',
        kernel_name='depthwise_conv2d_backprop_input', build_option=None):  # pylint: disable=invalid-name, R0913
    check_empty_tensor(filters, out_backprop, input_grad, strides, pads)
    ori_paras = {
        "input_size": input_size, "filters": filters, "out_backprop": out_backprop, "input_grad": input_grad,
        "strides": strides, "pads": pads, "dilations": dilations, "data_format": data_format,
        "kernel_name": kernel_name
    }
    if build_option is not None:
        ori_paras[SPLIT_AXIS_MODE_STR] = build_option.get(SPLIT_AXIS_MODE_STR, 0)

    default_para = set_default_para()
    if not input_size.get("ori_shape"):
        ori_paras["input_size"]["ori_shape"] = default_para["input_size"]["ori_shape"]
    depthwise_conv2dbp_para = DepthwiseConv2dBackpropParaProcess(ori_paras)
    depthwise_conv2dbp_para.config_paras()
    res_dtype = input_grad.get("dtype").lower()
    dedx = tbe.conv2d_backprop_input(
        filters=depthwise_conv2dbp_para.tensors.get("filter_tensor"),
        out_backprop=depthwise_conv2dbp_para.tensors.get("dy_tensor"),
        filter_sizes=depthwise_conv2dbp_para.shape.get("filter_shape_nchw"),
        input_sizes=depthwise_conv2dbp_para.shape.get("dx_shape_nchw"),
        para_dict={"strides": (depthwise_conv2dbp_para.strides[H_DIM], depthwise_conv2dbp_para.strides[W_DIM]),
                   "padding": depthwise_conv2dbp_para.pads,
                   "dilations": depthwise_conv2dbp_para.dilations,
                   "res_dtype": res_dtype,
                   "kernel_name": kernel_name,
                   "group_dict": depthwise_conv2dbp_para.attrs.get("group_para"),
                   "correct_range_flag": depthwise_conv2dbp_para.attrs.get("correct_range_flag", False),
                   "ori_tensors": _collect_ori_tensors(ori_paras),
                   "op_type": "depthwise_conv2d_backprop_input",
                   "split_axis_mode": depthwise_conv2dbp_para.split_axis_mode})

    return {'op_placeholder': [depthwise_conv2dbp_para.tensors.get("input_tensor"),
                               depthwise_conv2dbp_para.tensors.get("filter_tensor"),
                               depthwise_conv2dbp_para.tensors.get("dy_tensor")],
            'op_res': [dedx]}


@register_operator('DepthwiseConv2DBackpropInput')
@para_check.check_input_type(dict, dict, dict, dict, (tuple, list),
                             (tuple, list), (tuple, list), str, str)
def depthwise_conv2d_backprop_input(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                                    filter, out_backprop, input_grad, strides,
                                    dilations=(1, 1, 1, 1), pads=(0, 0, 0, 0), data_format="NHWC",
                                    kernel_name="depthwise_conv2d_backprop_input"):
    """
    algorithm: depthwise_conv2d_backprop_input

    Parameters
    ----------
    input_size: dict, shape of input tensor, support [N, C, H, W] or [N, H, W, C], will not be used.

    filter: dict
        4-D origin shape and dtype of filter tensor
        support [H, W, C, K], K is channel_multiplier

    out_backprop: dict
        4-D origin shape and dtype of out_backprop tensor,
        support [N, Co, Ho, Wo] or [N, Ho, Wo, Co],
        gradients w.r.t. the output of the convolution

    input_grad: dict
        4-D origin shape and dtype of input tensor,
        support [N, C, H, W] or [N, H, W, C]

    strides: a list or tuple of four ints
        the stride of the sliding window for height and width of the input of
        the convolution, support [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations: an optional list or tuple of four ints
        the dilation factor for each dimension of input
        if set to k > 1, there will be k-1 skipped cells between each
        filter element on that dimension, support [1, 1, dilation_height,
        dilation_width] or [1, dilation_height, dilation_width, 1]

    pads: a list or tuple of four ints
        padding added to each dimension of the input

    data_format : str
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]

    kernel_name: str
        cce kernel name, default value is "depthwise_conv2d_backprop_input"

    Returns
    -------
    None
    """
    input_list = [input_size, filter, out_backprop, input_grad]
    attr_list = [strides, pads, dilations, data_format, kernel_name]

    extra_params = {
        "single_op": True
    }
    ins = classify([input_list, attr_list], "depthwise_conv2d_backprop_input", extra_params)
    schedules, tensor_list = [], []
    for input_param in ins:
        with tbe.compute():
            res = _depthwise_conv2d_backprop_input_compute(*input_param)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res.get('op_res'))
        schedules.append(sch)
        tensors = res.get('op_placeholder') + res.get('op_res')
        tensor_list.append(tensors)

    config = {'print_ir': False,
              'name': kernel_name,
              'tensor_list': tensor_list,
              'build_args': {'constant_realize_extent_in_infer_bound': False}}
    tbe.build(schedules, config)
