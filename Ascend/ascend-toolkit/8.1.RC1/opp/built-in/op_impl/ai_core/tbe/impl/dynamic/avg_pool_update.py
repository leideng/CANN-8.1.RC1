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
avg_pool_update
"""
import copy
import json
import tbe.dsl as tbe_base
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.register import register_operator
from tbe.common.register import set_fusion_buildcfg
from tbe.common.utils import log
from tbe.dsl import auto_schedule
from tbe.dsl.classifier.elewise_classifier import ElewiseMode
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.util_cube_dynamic import check_dynamic_mode
from impl.util.util_cube_dynamic import FIX_FLAG
from impl.util.util_cube_dynamic import UNKNOWN_FLAG
from impl.util.util_cube_dynamic import DYNAMIC_FLAG
from impl.util.util_cube_dynamic import UNKNOWN_SHAPE
from impl.util.util_conv2d import search_op
from impl.util.util_conv2d import is_support_fixpipe
from impl.util.util_avgpool_update_dynamic import Constant
from impl.util.util_avgpool_update_dynamic import cache_tiling_get_var
from impl.util.util_avgpool_update_dynamic import cache_tiling_paras_process_avgupdate
from impl.util.util_avgpool_update_dynamic import cache_tiling_static
from impl.util.util_avgpool_update_dynamic import get_attr_nchw_format
from impl.util.util_avgpool_update_dynamic import calculate_pads
from impl.util import util_select_op_base
from tbe.common.utils.errormgr import error_manager_cube as err_man

NONETYPE = type(None)


def merge_shape_hw(shape_5hd):
    """
    merge 5hd (n,c1,h,w,c0) to 4d (n,c1,(h*w),c0)
    """
    return [shape_5hd[0], shape_5hd[1], shape_5hd[2]*shape_5hd[3], shape_5hd[4]]



def check_padding(pads, padding, ksize):
    """
    check padding params
    attrs: pads, padding, ksize
    """
    if len(pads) != 4:
        error_manager_vector.raise_err_input_value_invalid("AvgPoolUpdate", "pads", "4", len(pads))
    if padding not in ("CALCULATED", "VALID", "SAME"):
        error_manager_vector.raise_err_specific_reson("AvgPoolUpdate", "Padding mode only support CALCULATED, "
                                                      "VALID or SAME!")


# 'pylint: disable=unused-variable,too-many-arguments,too-many-locals,too-many-arguments,invalid-name
def check_avgpool_update_params(x1, x2, y, attrs):
    """
    check params valid
    attrs: ksize, strides, padding, pads, data_format
    """
    ksize, strides, padding, pads, data_format = attrs

    input1_shape = x1.get("ori_shape")
    input1_type = x1.get("dtype").lower()
    input1_format = x1.get("ori_format")
    input2_shape = x2.get("ori_shape")
    input2_type = x2.get("dtype").lower()
    input2_format = x2.get("ori_format")
    output_shape = y.get("ori_shape")
    output_type = y.get("dtype").lower()
    output_format = y.get("ori_format")

    para_check.check_shape(input1_shape)
    para_check.check_shape(input2_shape)
    para_check.check_shape(output_shape)
    para_check.check_dtype(input1_type, ["float16", "float32"])
    para_check.check_dtype(input2_type, ["int4", "int8", "float16", "float32"])
    para_check.check_dtype(output_type, ["float16", "float32"])
    if input1_shape != output_shape or input1_type != output_type:
        error_manager_vector.raise_err_specific_reson(
            "AvgPoolUpdate", "Data shape and type of the 1st input and the output should be consistent!")

    # check data format
    if input1_format != input2_format or input1_format != output_format or input1_format != data_format:
        error_manager_vector.raise_err_specific_reson("AvgPoolUpdate", "Data formats of the inputs and the output "
                                                      "should be consistent!")

    if len(ksize) != 4:
        error_manager_vector.raise_err_input_value_invalid("AvgPoolUpdate", "ksize", "4", len(ksize))
    if len(strides) != 4:
        error_manager_vector.raise_err_input_value_invalid("AvgPoolUpdate", "strides", "4", len(strides))

    # The ksize/strides of the N and C dimensions are 1
    _, ksize, strides = get_attr_nchw_format(input2_shape, ksize, strides, data_format)
    check_padding(pads, padding, ksize)
    if ksize[Constant.N_DIM] != 1 or ksize[Constant.C_DIM] != 1:
        error_manager_vector.raise_err_specific_reson(
            "AvgPoolUpdate", "The kernel size in the N and C dimensions must be 1")
    if strides[Constant.N_DIM] != 1 or strides[Constant.C_DIM] != 1:
        error_manager_vector.raise_err_specific_reson(
            "AvgPoolUpdate", "The strides in the N and C dimensions must be 1")


def check_dynamic(shape):
    """
    check dynamic or not
    """
    for item in list(shape):
        if isinstance(item, tvm.Var):
            return DYNAMIC_FLAG
    if list(shape) == UNKNOWN_SHAPE:
        return UNKNOWN_FLAG
    if DYNAMIC_FLAG in shape:
        return DYNAMIC_FLAG
    return FIX_FLAG


def is_dynamic(x1, x2):
    if check_dynamic(x1) != FIX_FLAG or check_dynamic(x2) != FIX_FLAG:
        return True
    return False


def _modify_input_for_classify(input_dict: dict):
    input_for_classify = copy.deepcopy(input_dict)

    # Modify shape if 0 dim exists.
    in_shape = input_for_classify.get("shape")
    modified_shape = list()
    for dim in in_shape:
        if dim == 0:
            log.warn("Zero in shape will be modified!")
            modified_shape.append(-1)
        else:
            modified_shape.append(dim)
    input_for_classify["shape"] = tuple(modified_shape)

    # Modify range if 0 in range.
    in_range = input_for_classify.get("range")
    if in_range is None:
        return input_for_classify

    modified_range = list()
    for dim_range in in_range:
        if 0 in dim_range:
            log.warn("Zero in range will be modified!")
            modified_range.append((1, None))
        else:
            modified_range.append(dim_range)
    input_for_classify["range"] = tuple(modified_range)

    return input_for_classify


def _decode_params(params):
    out_w = params.get("out_w")
    out_h = params.get("out_h")
    stride_h = params.get("stride_h")
    stride_w = params.get("stride_w")
    pad_l = params.get("pad_l")
    pad_r = params.get("pad_r")
    pad_t = params.get("pad_t")
    pad_b = params.get("pad_b")
    k_w = params.get("k_w")
    k_h = params.get("k_h")
    min_w = params.get("min_w")
    min_h = params.get("min_h")
    return [out_w, out_h, stride_h, stride_w, pad_l, pad_r, pad_t, pad_b, k_w, k_h, min_w, min_h]


def _avg_pool_update_compute_fp16(x1, params):
    # compute avg_pool_update in fp16 mode
    mean_matrix_shape = x1.shape[2:]
    out_w, out_h, stride_h, stride_w, pad_l, pad_r, pad_t, pad_b, k_w, k_h, min_w, min_h = _decode_params(params)

    mean_matrix_avg_update = tvm.compute(mean_matrix_shape, lambda m, c0:
        (tvm.min(tvm.min((m % out_w) * stride_w - pad_l + k_w,
        (out_w - 1 - (m % out_w)) * stride_w - pad_r + k_w), min_w) *
        tvm.min(tvm.min((m // out_w) * stride_h - pad_t + k_h,
        (out_h - 1 - (m // out_w)) * stride_h - pad_b + k_h), min_h)).astype(x1.dtype),
        name="mean_matrix_" + x1.dtype,
        tag="elewise_set_value_variable")

    c_ub_avg = tvm.compute(x1.shape, lambda n, c1, m, c0:
        tvm.div(x1(n, c1, m, c0), mean_matrix_avg_update(m, c0)),
        name="c_ub_avg",
        tag="elewise_binary_div")

    return c_ub_avg


def _avg_pool_update_compute_fp32(x1, params, data_format):
    # compute avg_pool_update in fp32 mode
    out_w, out_h, stride_h, stride_w, pad_l, pad_r, pad_t, pad_b, k_w, k_h, min_w, min_h = _decode_params(params)
    if data_format == "NCHW":
        mean_matrix_shape = x1.shape[2:]
        mean_matrix_avg_update = tvm.compute(mean_matrix_shape, lambda h, w:
            (tvm.min(tvm.min(w * stride_w - pad_l + k_w,
            (out_w - 1 - w) * stride_w - pad_r + k_w), min_w) *
            tvm.min(tvm.min(h * stride_h - pad_t + k_h,
            (out_h - 1 - h) * stride_h - pad_b + k_h), min_h)).astype(x1.dtype),
            name="mean_matrix_" + x1.dtype,
            tag="elewise_set_value_variable")

        c_ub_avg = tvm.compute(x1.shape, lambda n, c, h, w:
            tvm.div(x1(n, c, h, w), mean_matrix_avg_update(h, w)),
            name="c_ub_avg",
            tag="elewise_binary_div")

    elif data_format == "NHWC":
        mean_matrix_shape = x1.shape[1:]
        mean_matrix_avg_update = tvm.compute(mean_matrix_shape, lambda h, w, c:
            (tvm.min(tvm.min(w * stride_w - pad_l + k_w,
            (out_w - 1 - w) * stride_w - pad_r + k_w), min_w) *
            tvm.min(tvm.min(h * stride_h - pad_t + k_h,
            (out_h - 1 - h) * stride_h - pad_b + k_h), min_h)).astype(x1.dtype),
            name="mean_matrix_" + x1.dtype,
            tag="elewise_set_value_variable")

        c_ub_avg = tvm.compute(x1.shape, lambda n, h, w, c:
            tvm.div(x1(n, h, w, c), mean_matrix_avg_update(h, w, c)),
            name="c_ub_avg",
            tag="elewise_binary_div")
    return c_ub_avg


@register_op_compute("AvgPoolUpdate", op_mode="dynamic", support_fusion=True)
def avg_pool_update_compute(x1, x2, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                            data_format="NCHW", ceil_mode=False,
                            exclusive=True, kernel_name="avg_pool_update"):

    dynamic_flag = is_dynamic(x1.shape, x2.shape)
    if x1.op.name == "input1_data":
        conv_res = x1
    else:
        conv_res = search_op(x1, "convolution_res_conv2d")

    out_h = conv_res.op.attrs["height_out"]
    out_w = conv_res.op.attrs["width_out"]
    if dynamic_flag:
        para_dict = {"dtype": x1.dtype}
        para_dict = cache_tiling_get_var(para_dict)
        input_shape = para_dict.get("input2_shape")
        cor_pads = para_dict.get("pads")
        ksize = para_dict.get("ksize")
        strides = para_dict.get("strides")
    else:
        # nchw format attr
        input_shape, ksize, strides = get_attr_nchw_format(x2.op.attrs["ori_shape"], ksize, strides, data_format)
        out_n, out_c1, _, out_c0 = tuple(i.value for i in x1.shape)
        output_shape = [out_n, out_c1 * out_c0, out_h, out_w] # ori_shape in nchw

        # calculate pads
        cor_pads = calculate_pads(input_shape, output_shape, ksize, strides, padding, pads, ceil_mode)
    input_h = input_shape[Constant.H_DIM]
    input_w = input_shape[Constant.W_DIM]
    pad_t, pad_b, pad_l, pad_r = cor_pads
    k_h, k_w = ksize[Constant.H_DIM], ksize[Constant.W_DIM]
    stride_h, stride_w = strides[Constant.H_DIM], strides[Constant.W_DIM]

    # factor area same when exclusive=False or no padding
    if exclusive is not None and not exclusive or (padding == "VALID" and ceil_mode is not None and not ceil_mode):
        error_manager_vector.raise_err_specific_reson(
            "AvgPoolUpdate", "AvgPoolUpdate op is not required when pooling factor is a constant")
    else:
        min_h = tvm.min(k_h, input_h)
        min_w = tvm.min(k_w, input_w)
        params = {
            "out_w": out_w, "out_h": out_h, "stride_w": stride_w, "stride_h": stride_h, "pad_l": pad_l,
            "pad_r": pad_r, "pad_t": pad_t, "pad_b": pad_b, "k_h": k_h, "k_w": k_w, "min_w": min_w,
            "min_h": min_h
        }
        if x1.dtype == "float16" or dynamic_flag:
            c_ub_avg = _avg_pool_update_compute_fp16(x1, params)
        else:
            # fp32 + ND mode
            c_ub_avg = _avg_pool_update_compute_fp32(x1, params, data_format)
    build_cfg = {"dummy_placeholder": True}
    set_fusion_buildcfg("AvgPoolUpdate", build_cfg)

    return c_ub_avg


@register_operator("AvgPoolUpdate", pattern="AvgPoolUpdate")
@para_check.check_input_type(dict, dict, dict, (list, tuple),
                             (list, tuple), str, (list, tuple), str, (bool, NONETYPE), (bool, NONETYPE), str)
def avg_pool_update(x1, x2, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                    data_format="NCHW", ceil_mode=False,
                    exclusive=True, kernel_name="avg_pool_update"):
    """
        feature map
            |  \
          conv  |
            |   |
          (x1) (x2)
            |   |
        avg_pool_update

    Parameters
    ----------
    x1: dict, shape and dtype of conv's output data, support float16/float32,
    shape is 5HD, ori_shape is 4 dims, format is NCHW/NHWC

    x2: dict, shape and dtype of conv's input data, only support int4/int8/float16/fp32,
    shape is 5HD, ori_shape is 4 dims, format is NCHW/NHWC

    y: dict, shape and dtype of output_data, only support float16

    ksize: list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides: list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding: str, the mode of padding, support VALID, SAME and CALCULATED

    pads: padding value when padding_mode is CALCULATED

    data_format: str, default = "NCHW"

    ceil_mode: use ceil or floor to calculate ho and wo when padding_mode is CALCULATED

    exclusive: ignore padding area or not when calculating the average

    kernel_name: cce kernel name, default value is "avg_pool_update"

    Returns
    -------
    None
    """
    ori_paras = {
        "input_1": x1, "input_2": x2, "output": y, "ksize": ksize,
        "strides": strides, "padding": padding,  "pads": pads,
        "data_format": data_format, "ceil_mode": ceil_mode,
        "exclusive": exclusive, "kernel_name": kernel_name, "dtype": x1.get("dtype").lower(),
    }
    para_dict = {}
    dynamic_flag = is_dynamic(x1.get('shape'), x2.get('shape'))
    if dynamic_flag:
        para_dict = cache_tiling_paras_process_avgupdate(ori_paras)
    else:
        # check params
        check_avgpool_update_params(x1, x2, y, [ksize, strides, padding, pads, data_format])
        para_dict = cache_tiling_static(ori_paras)

    # get tensor shape in UB: N,C1,H*W,C0
    if x1.get("dtype").lower() == "float16" or dynamic_flag:
        # shape in 5HD, get tensor shape in UB: N,C1,H*W,C0
        x1_shape_ub = merge_shape_hw(para_dict.get("input1_realshape"))
        x2_shape_ub = merge_shape_hw(para_dict.get("input2_realshape"))
    else:
        # shape in ND
        x1_shape_ub = para_dict.get("input1_realshape")
        x2_shape_ub = para_dict.get("input2_realshape")
    x1_shape_nchw = para_dict["input1_shape"]

    x1_for_classify = _modify_input_for_classify(x1) if dynamic_flag else x1

    ins = classify([x1_for_classify], OpPatternMode.ELEWISE)

    schedules, tensors = [], []
    for (ins_x,) in ins:
        with tbe_base.compute():
            mode = ins_x.get("mode")
            if mode == ElewiseMode.CONST:
                # variable_shape() method creates variables in different mode for elementwise op.
                # When mode is not const, avg_pool_update's variables are created according to conv2d's variable,
                # this method will be skipped.
                _ = shape_util.variable_shape([ins_x])
            input1_data = tvm.placeholder(
                x1_shape_ub, x1.get('dtype'),
                name="input1_data",
                attrs={"ori_shape": x1.get('ori_shape'),
                       "format": x1.get('format'),
                       "ori_format": x1.get('ori_format'),
                       "height_out": x1_shape_nchw[Constant.H_DIM],
                       "width_out": x1_shape_nchw[Constant.W_DIM]})
            input2_data = tvm.placeholder(
                x2_shape_ub, x2.get('dtype'),
                name="input2_data",
                attrs={"ori_shape": x2.get('ori_shape'),
                       "format": x2.get('format'),
                       "ori_format": x2.get('ori_format')})

            c_ub_avg = avg_pool_update_compute(input1_data, input2_data, y, ksize, strides, padding, pads,
                                               data_format, ceil_mode, exclusive, kernel_name)
            tensors.append([input1_data, input2_data, c_ub_avg])
        with tvm.target.cce():
            sch = auto_schedule(c_ub_avg)
        schedules.append(sch)

    config = {"print_ir": True,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)


def _get_slice_info(x1, data_format):
    if x1.get("dtype").lower() == "float16":
        slice_info = {"_op_slice_info":
            {"splitMaps": [{"inputList": [{"idx": 0, "axis": [0], "headOverLap": [-1], "tailOverLap": [-1]},
                                            {"idx": 1, "axis": [0], "headOverLap": [-1], "tailOverLap": [-1]}],
                            "outputList": [{"idx": 0, "axis": [0]}]},
                            {"inputList": [{"idx": 0, "axis": [2], "headOverLap": [-1], "tailOverLap": [-1]},
                                            {"idx": 1, "axis": [2], "headOverLap": [0], "tailOverLap": [0]}],
                            "outputList": [{"idx": 0, "axis": [2]}]},
                            {"inputList": [{"idx": 0, "axis": [3], "headOverLap": [-1], "tailOverLap": [-1]},
                                            {"idx": 1, "axis": [3], "headOverLap": [0], "tailOverLap": [0]}],
                            "outputList": [{"idx": 0, "axis": [3]}]}],
            "reduceMaps": [],
            "l1FusionEnable": 2,
            "minTbeL1Space": 0}}
    else:
        n_axis = 0
        h_axis = 1 if data_format == "NHWC" else 2
        w_axis = 2 if data_format == "NHWC" else 3

        slice_info = {"_op_slice_info":
            {"splitMaps": [{"inputList": [{"idx": 0, "axis": [n_axis], "headOverLap": [-1], "tailOverLap": [-1]},
                                            {"idx": 1, "axis": [n_axis], "headOverLap": [-1], "tailOverLap": [-1]}],
                            "outputList": [{"idx": 0, "axis": [n_axis]}]},
                            {"inputList": [{"idx": 0, "axis": [h_axis], "headOverLap": [-1], "tailOverLap": [-1]},
                                            {"idx": 1, "axis": [h_axis], "headOverLap": [0], "tailOverLap": [0]}],
                            "outputList": [{"idx": 0, "axis": [h_axis]}]},
                            {"inputList": [{"idx": 0, "axis": [w_axis], "headOverLap": [-1], "tailOverLap": [-1]},
                                            {"idx": 1, "axis": [w_axis], "headOverLap": [0], "tailOverLap": [0]}],
                            "outputList": [{"idx": 0, "axis": [w_axis]}]}],
            "reduceMaps": [],
            "l1FusionEnable": 2,
            "minTbeL1Space": 0}}


    return slice_info


def get_op_support_info(x1, x2, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                        data_format="NCHW", ceil_mode=False,
                        exclusive=True, kernel_name="avg_pool_update"):
    """
    get the avgpool_update split
    """
    slice_info = _get_slice_info(x1, data_format)
    split_maps = slice_info.get("_op_slice_info").get("splitMaps")
    # check input format
    if x1.get("format") != "NC1HWC0" or x2.get("format") != "NC1HWC0":
        split_maps.clear()

    # check input dim
    if check_dynamic_mode(x1) != FIX_FLAG or check_dynamic_mode(x2) != FIX_FLAG:
        split_maps.clear()

    return json.dumps(slice_info)


def op_select_format(x1, x2, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                     data_format="NCHW", ceil_mode=False,
                     exclusive=True, kernel_name="avg_pool_update"):
    r"""
    1.When is dynamic, op select only supports the following specification:

    | Tensor    | x        | filter     | y       |
    | :-------: | :------: | :--------: | :-----: |
    | Data Type | float16  | float16    | float16 |
    | Format    | NC1HWC0  | NC1HWC0    | NC1HWC0 |

    2.When is static, op select supports float, float16, int8, int4:

    """
    def _select_format(params):
        x1 = params[0]
        shape_x1 = x1.get("ori_shape")
        shape_x1 = shape_util.scalar2tensor_one(shape_x1)
        if (shape_x1 == (UNKNOWN_FLAG,)) or (DYNAMIC_FLAG in shape_x1):
            if (is_support_fixpipe()):
                input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                                    datatype="float16,float32",
                                                    format="NC1HWC0,NC1HWC0",
                                                    unknownshape_format="NC1HWC0,NC1HWC0")
                input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                                    datatype="float16,float32",
                                                    format="NC1HWC0,NC1HWC0",
                                                    unknownshape_format="NC1HWC0,NC1HWC0")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16,float32",
                                                        format="NC1HWC0,NC1HWC0",
                                                        unknownshape_format="NC1HWC0,NC1HWC0")
            else:
                input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                                    datatype="float16",
                                                    format="NC1HWC0",
                                                    unknownshape_format="NC1HWC0")
                input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                                    datatype="float16",
                                                    format="NC1HWC0",
                                                    unknownshape_format="NC1HWC0")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16",
                                                        format="NC1HWC0",
                                                        unknownshape_format="NC1HWC0")
        else:
            input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                                   datatype="float16,float16,float32,float16",
                                                   format="NC1HWC0,NC1HWC0,ND,NC1HWC0",
                                                   unknownshape_format="NC1HWC0,NC1HWC0,ND,NC1HWC0")
            input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                                   datatype="float16,int8,float32,int4",
                                                   format="NC1HWC0,NC1HWC0,ND,NC1HWC0",
                                                   unknownshape_format="NC1HWC0,NC1HWC0,ND,NC1HWC0")
            output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                    datatype="float16,float16,float32,float16",
                                                    format="NC1HWC0,NC1HWC0,ND,NC1HWC0",
                                                    unknownshape_format="NC1HWC0,NC1HWC0,ND,NC1HWC0")

        return [input0, input1, output0]

    params = [x1, x2, y, ksize, strides, padding, pads,
             data_format, ceil_mode, exclusive, kernel_name]
    param_list = _select_format(params)
    return util_select_op_base.get_dynamic_param_in_json(param_list)