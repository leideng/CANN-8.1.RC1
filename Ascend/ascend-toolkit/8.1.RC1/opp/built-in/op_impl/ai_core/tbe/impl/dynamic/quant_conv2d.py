#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2024 Huawei Technologies Co., Ltd
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
static and dynamic QuantConv2d
"""
from __future__ import absolute_import
from typing import List, Tuple, Union

import copy

import tbe.dsl as tbe_base
from tbe import tvm
from tbe.dsl import auto_schedule
from impl.util import fusion_util
from tbe.dsl import build
from tbe.common.register import set_fusion_buildcfg
from tbe.dsl.base.classifier.conv2d_classifier import cube_forward_op_classify
from tbe.common.register import register_op_compute
from tbe.common.register import register_operator
from tbe.common.utils import para_check
from tbe.common.utils.op_util.op_util_conv2d import is_support_fixpipe
from impl.util.util_conv2d_dynamic import enable_db_fold
from impl.util.platform_adapter import error_manager_cube as err_man
from tbe.dsl.compute.quant_conv2d_compute import quant_conv2d_forward_compute
from tbe.common.utils.op_util import op_util_conv2d
from impl.util import util_conv2d
from te.utils import shape_util
from tbe.common.platform import SHORT_SOC_VERSION
from tbe.common.platform.platform_info import get_soc_spec

DYNAMIC_FLAG_1 = -1
DYNAMIC_FLAG_2 = -2
PAD_MAX = 255
STRIDE_MAX = 63
KERNEL_H_MAX = 255
KERNEL_H_MAX_MILAN = 511
DILATION_MAX = 255
H_DIM = 2
W_DIM = 3

NONETYPE = type(None)

DEFAULT_PARA = {
    "optim_dict": {"c0_optim_flg": False,
                   "use_v200_c04_flg": False,
                   "v220_c04_mode": "disabled",
                   "invalid_data_rm": False},
    "fusion_para": {"input_memory_type": 0, "output_memory_type": 0,
                    "valid_shape": (), "slice_offset": (),
                    "l1_fusion_type": -1},
    "ori_shape": [0, 0, 0, 0]
}


def check_parameter_value(offset, round_mode, groups):
    if offset is not None:
        err_man.raise_err_message_cube("QuantConv2d don’t support to set offset parameter, "
                                       "offset is: %s." % offset)
    if round_mode != "rint":
        err_man.raise_err_message_cube("QuantConv2d don’t support to set round_mode parameter, "
                                       "round_mode is: %s." % round_mode)
    if groups != 1:
        err_man.raise_err_message_cube("QuantConv2d only support groups = 1, groups is: %s." % groups)


def check_support_dtype(inputs, weights, scale, bias, outputs):
    data_dtype = "dtype"
    if isinstance(inputs, tvm.Tensor):
        input_data_type = inputs.dtype
        weight_data_type = weights.dtype
        scale_data_type = scale.dtype
        bias_data_type = bias.dtype if bias is not None else "int32"
        output_data_type = outputs.get(data_dtype)
    else:
        input_data_type = inputs.get(data_dtype)
        weight_data_type = weights.get(data_dtype)
        scale_data_type = scale.get(data_dtype)
        bias_data_type = bias.get(data_dtype) if bias is not None else "int32"
        output_data_type = outputs.get(data_dtype)

    if input_data_type != "int8":
        err_man.raise_err_message_cube("Not support the given input dtype."
                                       "input dtype is %s." % input_data_type)
    if weight_data_type != "int8":
        err_man.raise_err_message_cube("Not support the given weight dtype."
                                       "weight dtype is %s." % weight_data_type)
    if scale_data_type != "uint64" and scale_data_type != "int64":
        err_man.raise_err_message_cube("Not support the given scale dtype."
                                       "scale dtype is %s." % scale_data_type)
    if bias_data_type != "int32":
        err_man.raise_err_message_cube("Not support the given bias dtype."
                                       "bias dtype is %s." % bias_data_type)
    if output_data_type != "float16":
        err_man.raise_err_message_cube("Not support the given output dtype."
                                       "output dtype is %s." % output_data_type)


def infer_static_or_dynamic(inputs, weights, scale, bias, strides, pads, dilations):
    if isinstance(inputs, tvm.Tensor):
        shape_lists = [strides, pads, dilations,
                       inputs.shape, weights.shape,
                       scale.shape, None if bias is None else bias.shape]
        for shape in shape_lists:
            if not shape:
                continue
            if DYNAMIC_FLAG_1 in shape or DYNAMIC_FLAG_2 in shape:
                return False
        return True
    else:
        bias = {} if bias is None else bias
        shape_lists = [strides, pads, dilations,
                    inputs.get("shape"), weights.get("shape"),
                    scale.get("shape"), bias.get("shape")]
        for shape in shape_lists:
            if not shape:
                continue
            if DYNAMIC_FLAG_1 in shape or DYNAMIC_FLAG_2 in shape:
                return False

    return True


def check_support_shape(inputs, weights, outputs, strides, dilations, pads, groups, data_format):

    def is_zero_tensor():
        tensor_shapes = [shape_fm, shape_filter, shape_output]
        for shape in tensor_shapes:
            if not shape:
                continue
            if 0 in shape:
                return True

        return False

    def is_nx1():
        if is_support_fixpipe():
            return False

        return wo == 1

    def is_dma():
        for v in strides:
            if v > STRIDE_MAX:
                return True
        for v in dilations:
            if v > DILATION_MAX:
                return True
        for v in pads:
            if v > PAD_MAX:
                return True
        kernel_limit = KERNEL_H_MAX_MILAN if is_support_fixpipe() else KERNEL_H_MAX
        if kh > kernel_limit or kw > kernel_limit:
            return True

        if is_support_fixpipe() and op_util_conv2d.check_load3dv2_postk_params_invalid(kh, kw, weight_dtype):
            return True

        # h_in, w_in, w_out, strideh, stridew, hk_dilation, wk_dilation, w_dtype, c0_optim_flag
        return op_util_conv2d.check_l1_size_invalid([hi, wi, wo, stride_h, None,
                                                     hk_dilation, None, weight_dtype, False])

    default_shape = [-1, -1, -1, -1, -1]
    shape_output = outputs.get("shape", default_shape)
    if isinstance(inputs, tvm.Tensor):
        shape_fm = shape_util.shape_to_list(inputs.shape)
        hi = shape_fm[H_DIM]
        wi = shape_fm[W_DIM]
        weight_dtype = weights.dtype
        para_dict, _ = util_conv2d.calc_para_from_tensor(inputs, weights, None, None,
            strides, pads, dilations, 0, groups, "quantconv2d_fusion", data_format)
        shape_filter = para_dict.get("weight_ori_shape_nchw")
        pad_w = para_dict.get("pad_w")
        stride_h = para_dict.get("stride_h")
        stride_w = para_dict.get("stride_w")
        dilations_h = para_dict.get("dilate_h")
        dilations_w = para_dict.get("dilate_w")
    else:
        hi = inputs.get("shape", default_shape)[H_DIM]
        wi = inputs.get("shape", default_shape)[W_DIM]
        weight_dtype = weights.get("dtype", "int8")
        shape_fm, shape_filter, _, pad_w, stride_h, stride_w, dilations_h, dilations_w, _, _ = \
            util_conv2d.calc_para_from_dict(inputs, weights, strides, pads, dilations, groups, outputs, data_format)
    kh = shape_filter[H_DIM]
    kw = shape_filter[W_DIM]
    hk_dilation = (kh - 1) * dilations_h + 1
    wk_dilation = (kw - 1) * dilations_w + 1
    wo = (wi + pad_w[0] + pad_w[1] - wk_dilation) // stride_w + 1

    if is_zero_tensor():
        err_man.raise_err_message_cube("QuantConv2d don’t support zero tensor, "
                                       "input is %s, weight is %s, output is %s" %
                                       (inputs, weights, outputs))

    if is_nx1():
        err_man.raise_err_message_cube("QuantConv2d don’t support nx1 shape, "
                                       "output is %s" % outputs)
    if is_dma():
        err_man.raise_err_message_cube("QuantConv2d don’t support dma shape, "
                                       "input is %s, weight is %s, strides is %s, "
                                       "dilations is %s, pads is %s, output is %s" %
                                       (inputs, weights, strides, dilations, pads, outputs))


def check_support_soc(support_socs: Union[List, Tuple]):
    soc = get_soc_spec(SHORT_SOC_VERSION)
    if soc not in support_socs:
        err_man.raise_err_message_cube("QuantConv2d only support %s, soc is: %s." % (support_socs, soc))


@register_op_compute("QuantConv2D", op_mode="dynamic", support_fusion=True)
@para_check.check_input_type(tvm.Tensor, tvm.Tensor, tvm.Tensor, (tvm.Tensor, NONETYPE), (tvm.Tensor, NONETYPE),
                             dict, int, (tuple, list), (tuple, list), (tuple, list),
                             (int, NONETYPE), str, (int, NONETYPE),
                             (str, NONETYPE), (str, NONETYPE), (bool, NONETYPE), (dict, NONETYPE))
def quant_conv2d_fusion_compute(inputs, weights, scale, bias, offset,
                                outputs, dtype, strides, pads, dilations,
                                groups=1, data_format='NHWC', offset_x=0,
                                round_mode="rint", kernel_name="quant_conv2d",
                                dsl_flag=True, options=None):
    fusion_util.check_fusion_input([inputs])
    fusion_util.check_fusion_input([weights])

    check_support_soc(["Ascend310P"])
    check_parameter_value(offset, round_mode, groups)
    check_support_dtype(inputs, weights, scale, bias, outputs)
    check_support_shape(inputs, weights, outputs, strides, dilations, pads, groups, data_format)

    binary_static_flag = infer_static_or_dynamic(inputs, weights, scale, bias, strides, pads, dilations)
    if not binary_static_flag:
        err_man.raise_err_message_cube("QuantConv2D fusion don’t support dynamic shape yet.")

    # set fusion build config
    build_cfg = {
        'constant_realize_extent_in_infer_bound': False,
        'enable_branch_eliminator_else_case': False
    }
    set_fusion_buildcfg("QuantConv2D", build_cfg)
    op_type = "QuantConv2D"
    if options is not None:
        options["op_option_dict"] = {}

    dsl_flag = False
    attr_default_para = copy.deepcopy(DEFAULT_PARA)
    attr_default_para["strides"] = strides
    attr_default_para["pads"] = pads
    attr_default_para["dilations"] = dilations
    attr_default_para["offset_x"] = offset_x
    attr_default_para["fusion_op_flag"] = True

    res = quant_conv2d_forward_compute(op_type, inputs, weights, scale, bias,
                                        outputs, dtype, strides, pads, dilations, groups,
                                        data_format, offset_x, kernel_name, dsl_flag,
                                        attr_default_para, binary_static_flag, options)
    return res.get("op_res")[0]


def quant_conv2d_forward(op_type, inputs, weights, scale, bias,
                         outputs, dtype, strides, pads, dilations, groups,
                         data_format, offset_x, kernel_name, op_option_dict,
                         binary_static_flag):
    # format classifier input
    dsl_flag = False
    attr_default_para = copy.deepcopy(DEFAULT_PARA)
    attr_default_para["strides"] = strides
    attr_default_para["pads"] = pads
    attr_default_para["dilations"] = dilations
    attr_default_para["offset_x"] = offset_x
    attr_list = [dtype, strides, pads, dilations, groups, data_format,
                 offset_x, kernel_name, dsl_flag, attr_default_para,
                 binary_static_flag]
    option_dict = copy.deepcopy(DEFAULT_PARA).get("optim_dict")
    option_dict["op_option_dict"] = op_option_dict
    input_list = [inputs, weights, scale, bias]
    ins = (input_list, attr_list, option_dict, outputs)
    extra_parameters = {"binary_static_flag": binary_static_flag}
    # run classifier to generate inputs to for each compute branch
    classified_input_list = cube_forward_op_classify(op_type, ins, extra_parameters)

    # run compute and schedule for each set of input
    sch_list = []
    tensor_list = []
    for compute_input in classified_input_list:
        input_list, attr_list, option_dict, output_dict = compute_input
        with tbe_base.compute():
            res = quant_conv2d_forward_compute(op_type, *input_list, output_dict, *attr_list, option_dict)

        with tvm.target.cce():
            sch_list.append(auto_schedule(res.get("op_res")))

        tensor_list.append(res.get("op_placeholder") + res.get("op_res"))

    # build all kernels
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extent_in_infer_bound": False,
                       "dummy_placeholder": True,
                       "enable_branch_eliminator_else_case": False,
                       "enable_db_fold": enable_db_fold()}
    }
    build(sch_list, config)


@register_operator("QuantConv2d")
@para_check.check_input_type(dict, dict, dict, (dict, type(None)), (dict, type(None)),
                             dict, int, (tuple, list), (tuple, list), (tuple, list),
                             (int, type(None)), str, (int, type(None)),
                             (str, type(None)), str)
def quant_conv2d(inputs, weights, scale, bias, offset,
                 outputs, dtype, strides, pads, dilations,
                 groups=1, data_format='NHWC', offset_x=0,
                 round_mode="rint", kernel_name="quant_conv2d"):
    """
    algorithm: conv2d+dequant
    math:`output = (fmap * weight + bias) * scale`

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: dict with keys(shape and dtype and range)
        input 4d feature map tensor
    weights: dict with keys(shape and dtype)
        input 4d weight tensor
    scale: dict with keys(shape and dtype) or None
        input scale tensor
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    offset: keys(shape and dtype) or None
        input offset tensor
    outputs: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset of fmap
    round_mode: str
        param for requant
    kernel_name: str
        kernel name, default value is "quant_conv2d"

    Returns
    -------
    None
    """

    binary_static_flag = infer_static_or_dynamic(inputs, weights, scale, bias, strides, pads, dilations)
    if not binary_static_flag:
        err_man.raise_err_message_cube("QuantConv2d don’t support dynamic shape.")
    check_support_soc(["Ascend310P", "Ascend910B", "Ascend910_93"])
    check_parameter_value(offset, round_mode, groups)
    check_support_dtype(inputs, weights, scale, bias, outputs)
    check_support_shape(inputs, weights, outputs, strides, dilations, pads, groups, data_format)

    op_type = "QuantConv2D"
    op_option_dict = dict()
    quant_conv2d_forward(op_type, inputs, weights, scale, bias,
                         outputs, dtype, strides, pads, dilations, groups,
                         data_format, offset_x, kernel_name, op_option_dict,
                         binary_static_flag)
