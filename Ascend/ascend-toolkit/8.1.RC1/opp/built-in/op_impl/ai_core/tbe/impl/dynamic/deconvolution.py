#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

deconvolution
"""

from __future__ import absolute_import
import warnings

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_register
from impl.util.util_cube_dynamic import check_dynamic_mode
from impl.util.util_cube_dynamic import check_fuzz_hw_dim
from impl.util.util_cube_dynamic import check_fuzz_input_output
from impl.util.util_cube_dynamic import check_fuzz_n_dim
from impl.util.util_cube_dynamic import check_generalize_config
from impl.util.util_cube_dynamic import DeconvolutionParaProcess
from impl.util.util_cube_dynamic import set_default_para
from impl.util.util_deconv_comm import get_op_support_info_conv2d_transpose
from impl.dynamic.conv2d_backprop_input import check_empty_tensor

H_DIM = 2
W_DIM = 3
OP_TYPE = "deconvolution"
FIX_FLAG = 0
DYNAMIC_FLAG = -1
UNKNOWN_FLAG = -2


def get_op_support_info(x, filter, bias, offset_w, y, strides,
                        pads, dilations=(1, 1, 1, 1),
                        groups=1, data_format="NHWC", offset_x=0,
                        kernel_name="deconvolution"):
    """
    get the deconvolution split info

    """
    tensor_dict = {"weight": filter, "fmap_dedy": x, "fmap_dedx": y}
    option_input = {"bias": bias, "offset_w": offset_w}
    attrs_dict = {"strides": strides, "pads": pads, "dilations": dilations,
                  "groups": groups, "data_format": data_format,
                  "offset_x": offset_x, "kernel_name": kernel_name}
    return get_op_support_info_conv2d_transpose(tensor_dict, option_input, attrs_dict,
                                                op_type="deconvolution", mode="dynamic")


@tbe_register.register_param_generalization("Deconvolution")
def deconvolution_generalization(x, filter, bias, offset_w, y, strides, pads, dilations=(1, 1, 1, 1),
                                 groups=1, data_format='NHWC', offset_x=0, kernel_name=OP_TYPE,
                                 generalize_config=None):
    """
    deconvolution generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to deconvolution

    groups: int
            param for group deconvolution

    data_format: str
            An optional string from: "NCHW". Defaults to "NCHW".
            Specify the data format of the input and output data.

    offset_x: int
        offset of gradients in quant mode. Default to 0.

    kernel_name: str
            kernel name, default value is "deconvolution"

    generalize_config: generalization mode, string.

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    if not check_generalize_config(generalize_config, OP_TYPE):
        return
    result = []
    dynamic_flag = check_dynamic_mode(x)
    if dynamic_flag == UNKNOWN_FLAG:
        warnings.warn("{} not support unknow_rank".format(OP_TYPE))
        return [{"result": "UNSUPPORTED"}]
    # check the format, shape and dilation
    if not check_fuzz_input_output([x, y], dilations, OP_TYPE):
        return [{"result": "UNSUPPORTED"}]
    check_n_dim_flag = check_fuzz_n_dim(x, dynamic_flag, OP_TYPE)
    if check_n_dim_flag:
        return check_n_dim_flag
    check_hw_dim_flag = check_fuzz_hw_dim([x, y, filter], strides, data_format, dynamic_flag, OP_TYPE)
    if check_hw_dim_flag:
        return check_hw_dim_flag
    if dynamic_flag == FIX_FLAG:
        for tensor_mem in [x, y]:
            pos_c = tensor_mem.get("ori_format").find("C")
            c_dim = tensor_mem["ori_shape"][pos_c]
            tensor_mem["ori_shape"] = [-1, -1, -1, -1]
            tensor_mem["ori_shape"][pos_c] = c_dim
    result.append([x, filter, bias, offset_w, y, {"strides": strides},
                   {"pads": pads}, {"dilations": dilations}, {"groups": groups}, {"data_format": data_format},
                   {"offset_x": offset_x}, {"kernel_name": kernel_name}])
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


def _check_empty_tensor(filters, x, y, strides, pads, data_format):
    new_stride = [1, 1, 1, 1]
    dim_h, dim_w = data_format.find('H'), data_format.find('W')
    new_stride[dim_h] = strides[0]
    new_stride[dim_w] = strides[1]
    check_empty_tensor(filters, x, y, new_stride, pads)


def _deconvolution_compute(x, filter, bias, offset_w,
                           y, strides, pads,
                           dilations=(1, 1, 1, 1),
                           groups=1, data_format='NHWC', offset_x=0,
                           kernel_name='deconvolution'):
    _check_empty_tensor(filter, x, y, strides, pads, data_format)
    ori_paras = {
        "x": x, "filters": filter, "bias": bias, "offset_w": offset_w, "y": y,
        "strides": strides, "pads": pads, "dilations": dilations, "groups": groups,
        "data_format": data_format, "offset_x": offset_x, "kernel_name": kernel_name
    }

    deconv_para = DeconvolutionParaProcess(ori_paras)
    deconv_para.config_paras()
    default_para = set_default_para()
    dedx = tbe.conv2d_backprop_input(filters=deconv_para.tensors.get("filter_tensor"),
                                     out_backprop=deconv_para.tensors.get("x_tensor"),
                                     filter_sizes=deconv_para.shape.get("filter_shape_nchw"),
                                     input_sizes=deconv_para.shape.get("dx_shape_nchw"),
                                     para_dict={
                                         "strides":
                                             (deconv_para.strides[H_DIM], deconv_para.strides[W_DIM]),
                                         "padding": deconv_para.pads,
                                         "dilations": deconv_para.dilations,
                                         "res_dtype": default_para.get("res_dtype"),
                                         "tensor_bias": deconv_para.tensors.get("bias_tensor"),
                                         "offset_x": offset_x,
                                         "kernel_name": kernel_name,
                                         "group_dict": deconv_para.attrs.get("group_para"),
                                         "correct_range_flag": deconv_para.attrs.get("correct_range_flag", False),
                                         "ori_tensors": _collect_ori_tensors(ori_paras),
                                         "op_type": "Deconvolution"
                                     })
    if bias:
        return {'op_placeholder': [deconv_para.tensors.get("x_tensor"),
                                   deconv_para.tensors.get("filter_tensor"),
                                   deconv_para.tensors.get("bias_tensor")],
                'op_res': [dedx]}
    return {'op_placeholder': [deconv_para.tensors.get("x_tensor"),
                               deconv_para.tensors.get("filter_tensor")],
            'op_res': [dedx]}


@register_operator('Deconvolution')
@para_check.check_input_type(dict, dict, (type(None), dict), (type(None), dict), dict, (tuple, list),
                             (tuple, list), (tuple, list), int, str, int, str,
                             (type(None), dict))
def deconvolution(x, filter, bias, offset_w, y, strides,
                  pads, dilations=(1, 1, 1, 1),
                  groups=1, data_format="NHWC", offset_x=0,
                  kernel_name="deconvolution"):
    """
    algorithm: deconvolution

    Parameters
    ----------
    x: dict with keys(ori_shape, ori_format, dtype)
        The shape of gradients.

    filter: dict with keys(ori_shape, ori_format, dtype)
            convolution kernel

    bias: dict with keys(shape and dtype)
        The shape of bias.

    offset_w: dict with keys(shape and dtype) or None
        Input offset_w tensor.

    y: dict with keys(ori_shape, ori_format, dtype and range)
       deconvolution output tensor

    strides: tuple/list of 2 integers
             filter move stride

    pads: tuple/list of 4 integers
          str: "SAME" or "VALID"
          tuple/list of 4 integers: [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated deconvolution
    groups: int
            param for group deconvolution

    data_format: str
            An optional string from: "NCHW". Defaults to "NCHW".
            Specify the data format of the input and output data.

    offset_x: int
        offset of gradients in quant mode. Default to 0.

    kernel_name: str
            kernel name, default value is "deconvolution"

    Returns
    -------
    None
    """

    with tbe.compute():
        res = _deconvolution_compute(
            x, filter, bias, offset_w, y,
            strides, pads, dilations, groups, data_format, offset_x, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get('op_res'))

    tensor_list = res.get('op_placeholder') + res.get('op_res')
    config = {'print_ir': False,
              'name': kernel_name,
              'tensor_list': tensor_list,
              'build_args': {'constant_realize_extent_in_infer_bound': False}}
    tbe.build(sch, config)
