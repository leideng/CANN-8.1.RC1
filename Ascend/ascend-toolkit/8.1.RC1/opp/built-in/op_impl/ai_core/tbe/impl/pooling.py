#!/usr/bin/env python
# coding: utf-8
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
pooling
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from tbe import tvm
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl.util import util_select_op_base
from impl.conv2d import conv2d
from impl.conv2d import conv2d_compute
from impl.util.util_common import check_op_impl_mode
from impl.util.platform_adapter import OpImplMode
from impl.util.util_conv2d import clear_suffix, is_support_fixpipe
from impl.util.util_cube_dynamic import UNKNOWN_FLAG, DYNAMIC_FLAG
from impl.util import util_select_op_base

# shape limit
# int32's max value
SHAPE_SIZE_LIMIT = 2**31 - 1
# c0 size
C0SIZE = 16

NoneType = type(None)


# 'pylint: disable = unused-argument,redefined-builtin,too-many-arguments,invalid-name,too-many-locals
def get_op_support_info(x,
                        matrix,
                        bias,
                        y,
                        window=(1, 1),
                        stride=(1, 1),
                        offset_x=0,
                        mode=0,
                        pad=(0, 0, 0, 0),
                        global_pooling=False,
                        ceil_mode=0,
                        dilation=(1, 1, 1, 1),
                        kernel_name="pooling_cce",
                        impl_mode="high_performance"):
    """
    get the pooling split
    """
    format_x = x.get("format")
    axis_reduce_list = None
    if format_x == "NC1HWC0":
        axis_split_matrix = [[
            util_select_op_base.SplitInput([0, [0], [-1], [-1]]),
            util_select_op_base.SplitOutput([0, [0]])
        ], [util_select_op_base.SplitInput([0, [2], [0], [0]]),
            util_select_op_base.SplitOutput([0, [2]])]]
    else:
        axis_split_matrix = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_matrix, axis_reduce_list, 2, 0)

    return op_cal_info_in_json


def op_select_format(x,
                     matrix,
                     bias,
                     y,
                     window=(1, 1),
                     stride=(1, 1),
                     offset_x=0,
                     mode=0,
                     pad=(0, 0, 0, 0),
                     global_pooling=False,
                     ceil_mode=0,
                     dilation=(1, 1, 1, 1),
                     kernel_name="pooling_cce",
                     impl_mode="high_performance"):
    r"""
    1. When the scene is dynamic and mode is 2 (sum), op select only supports the following specification:

        | Tensor    | x        | matrix     | bias    | y       |
        | :-------: | :------: | :--------: | :-----: | :-----: |
        | Data Type | float16  | float16    | float16 | float16 |
        | Data Type | float32  | float32    | float32 | float32 |
        | Format    | NC1HWC0  | FRACTAL_Z  | NCHW    | NC1HWC0 |

    2. Other scenes support the following specification:
        | Tensor    | x        | matrix     | bias    | y       |
        | :-------: | :------: | :--------: | :-----: | :-----: |
        | Data Type | float16  | float16    | float16 | float16 |
        | Data Type | int8     | int8       | int32   | int32   |
        | Format    | NC1HWC0  | FRACTAL_Z  | NCHW    | NC1HWC0 |
    """

    mode_sum = 2
    input_shape = x.get("ori_shape", ())
    is_dynamic = (input_shape == (UNKNOWN_FLAG,)) or (DYNAMIC_FLAG in input_shape)
    if mode == mode_sum and is_dynamic and is_support_fixpipe():
        input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                               datatype="float16,float32",
                                               format="NC1HWC0,NC1HWC0",
                                               unknownshape_format="NC1HWC0,NC1HWC0")
        input1 = util_select_op_base.gen_param(classify="input1", name="matrix",
                                               datatype="float16,float32",
                                               format="FRACTAL_Z,FRACTAL_Z",
                                               unknownshape_format="FRACTAL_Z,FRACTAL_Z",
                                               sub_format="0")
        input2 = util_select_op_base.gen_param(classify="input2", name="bias",
                                               datatype="float16,float32",
                                               format="NCHW,NCHW",
                                               unknownshape_format="NCHW,NCHW")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16,float32",
                                                format="NC1HWC0,NC1HWC0",
                                                unknownshape_format="NC1HWC0,NC1HWC0")
    else:
        input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                               datatype="float16,int8",
                                               format="NC1HWC0,NC1HWC0",
                                               unknownshape_format="NC1HWC0,NC1HWC0")
        input1 = util_select_op_base.gen_param(classify="input1", name="matrix",
                                               datatype="float16,int8",
                                               format="FRACTAL_Z,FRACTAL_Z",
                                               unknownshape_format="FRACTAL_Z,FRACTAL_Z",
                                               sub_format="0")
        input2 = util_select_op_base.gen_param(classify="input2", name="bias",
                                               datatype="float16,int32",
                                               format="NCHW,NCHW",
                                               unknownshape_format="NCHW,NCHW")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16,int32",
                                                format="NC1HWC0,NC1HWC0",
                                                unknownshape_format="NC1HWC0,NC1HWC0")

    return util_select_op_base.get_dynamic_param_in_json([input0, input1, input2, output0])


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=too-many-arguments,too-many-locals
def _pooling_check_rule(input_shape, output_dtype, window, stride, kernel_name):
    """
    :param input_shape: shape of input_data
    :param output_dtype: dtype of output_data
    :param window: shape of window
    :param stride: shape of stride
    :param kernel_name: cce kernel name
    :return: None
    """
    # check input and output
    para_check.check_shape(input_shape, min_rank=4, max_size=SHAPE_SIZE_LIMIT, param_name="x")
    para_check.check_dtype(output_dtype, ["float16", "int32"], param_name="y")
    # check window and stride length
    if len(window) != 2:
        error_manager_vector.raise_err_specific_reson("pooling", "the shape of window is not 2 dims")
    if len(stride) != 2:
        error_manager_vector.raise_err_specific_reson("pooling", "the shape of stride is not 2 dims")


def get_fusion_params(input_data, output_data, is_fused_compute=True):
    """
    :param input_data: tensor of input_data
    :param output_data: dict of output_data
    :return: dict fusion_params
    """
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    l1_fusion_type = input_data.op.attrs["L1_fusion_type"].value if "L1_fusion_type" in input_data.op.attrs else -1
    in_l1_flag = input_data.op.attrs["addr_type"].value == 1 if "addr_type" in input_data.op.attrs else False
    l1_addr_flag = input_data.op.attrs["L1_addr_flag"].value if "L1_addr_flag" in input_data.op.attrs else -1
    l1_addr_offset = input_data.op.attrs["L1_addr_offset"] if "L1_addr_offset" in input_data.op.attrs else -1
    l1_valid_size = input_data.op.attrs["L1_valid_size"] if "L1_valid_size" in input_data.op.attrs else -1
    out_l1_flag = output_data.get("addr_type") == 1
    fusion_params = {
        "is_fused_compute": is_fused_compute,
        "l1_fusion_type": l1_fusion_type,
        "in_l1_flag": in_l1_flag,
        "out_l1_flag": out_l1_flag,
        "L1_addr_flag": l1_addr_flag,
        "L1_addr_offset": l1_addr_offset,
        "L1_valid_size": l1_valid_size
    }

    return fusion_params


@tbe_platform.fusion_manager.fusion_manager.register("pooling")
# 'pylint: disable = too-many-branches,too-many-statements,variable_type_changed
def pool_fuse_compute(input_data,
                      matrix,
                      bias,
                      output_data,
                      window,
                      stride,
                      offset_x=0,
                      mode=0,
                      pad=(0, 0, 0, 0),
                      global_pooling=False,
                      ceil_mode=0,
                      dilation=(1, 1, 1, 1),
                      kernel_name="pool_fuse",
                      impl_mode="high_performance"):
    """
    Performs pooling on the input.

    Parameters
    ----------
    input_data: TVM tensor
        A `Tensor`. Must be one of the following types: `float16`
        4-D input to pool over.
    matrix: TVM tensor, shape and dtype of right matrix, only support float16,
        shape is 4 dims, format is NCHW
    bias: TVM tensor, use it to modify bias for fusion op.
    output_data: dict
        dict of output_data, include keys(shape and dtype).
    window: list or tuple
        A list of `ints` that has length 2.
        The size of the window for H, W dimension of the input tensor.
    stride: list or tuple
        A list of `ints` that has length 2.
        The stride of the sliding window for H, W .
    offset_x: avg quantize params.
    mode: str
        A int which stands6 for kinds of  pooling.
    pad : list or tuple, the pad of pooling, only support pooling in H or W
    global_pooling : global pooling params.
    dilation : reserved.
    ceil_mode : caffe round_mode params, 0:CEIL, 1:FLOOR, default value is
    DOMI_POOLING_CEIL
    kernel_name: str
        kernel name, default value is 'pool_fuse'
    impl_mode : high_precision or high_performance for inference
                default value is "high_performance".

    Returns:
    -------
    res: TVM tensor
        output tensor. Has the same type as `input_data`.
    """
    # get input_shape
    input_x = input_data.shape
    input_h = input_x[2].value
    input_w = input_x[3].value

    # convert mode&pad_mode to str for pooling2d
    pad = list(pad)
    if pad[0] >= window[0] or pad[1] >= window[0]:
        error_manager_vector.raise_err_specific_reson("pooling", "pad_h must less than kernel_h")
    if pad[2] >= window[1] or pad[3] >= window[1]:
        error_manager_vector.raise_err_specific_reson("pooling", "pad_w must less than kernel_w")

    if mode == 0:
        conv_pooling_flag = False
        temp_tensor = input_data
        while temp_tensor.op.input_tensors:
            if clear_suffix(temp_tensor.op.tag) in ("convolution_C", "fixpipe_reform"):
                conv_pooling_flag = True
                break
            temp_tensor = temp_tensor.op.input_tensors[0]
        if conv_pooling_flag:
            window_h, window_w = window[0], window[1]
            stride_h, stride_w = stride[0], stride[1]
            res = tbe.max_pool_compute(input_data, (window_h, window_w), (stride_h, stride_w), "SAME", pad, ceil_mode)
        else:
            # call pooling2d for max(pooling)&gmp
            mode_max = "MAX"
            if (input_h == window[0] and input_w == window[1] and
                    pad == [0, 0, 0, 0]) or \
                    global_pooling:
                mode_max = "GMP"
            window = list(window)

            # l1 fusion and l2 fusion
            l1_fusion_type = \
                input_data.op.attrs["L1_fusion_type"].value if "L1_fusion_type" in input_data.op.attrs else -1

            # l1 fusion params assign
            fusion_params = get_fusion_params(input_data, output_data, True)

            if l1_fusion_type == 1:
                input_data.op.attrs["addr_type"].value = 1
                in_l1_flag = True
                fusion_params["in_l1_flag"] = in_l1_flag

                l1_width_fusion_in = \
                    tvm.compute(input_data.shape,
                                lambda n, c1, h, w, c0:
                                input_data(n, c1, h, w, c0),
                                name="l1_width_fusion_tensor_in",
                                attrs=input_data.op.attrs)
                res = tbe.pooling2d(l1_width_fusion_in,
                                    window,
                                    stride,
                                    mode_max,
                                    pad=pad,
                                    data_mode=0,
                                    ceil_mode=ceil_mode,
                                    fusion_params=fusion_params,
                                    impl_mode=impl_mode)
            else:
                res = tbe.pooling2d(input_data,
                                    window,
                                    stride,
                                    mode_max,
                                    pad=pad,
                                    data_mode=0,
                                    ceil_mode=ceil_mode,
                                    fusion_params=fusion_params,
                                    impl_mode=impl_mode)
    elif mode == 1:
        mode_avg = "AVG"
        if (input_h == window[0] and input_w == window[1] and
                pad == [0, 0, 0, 0]) or \
                global_pooling:
            mode_avg = "GAP"

        # call conv2d_compute to fuse for avg_cube
        if mode_avg == "AVG" and matrix is not None:
            # get real pad
            input_c = input_data.op.attrs['ori_shape'][1].value
            _, _, pad_top, pad_bottom, pad_left, pad_right \
                    = tbe.get_caffe_out_size_and_pad(ceil_mode, input_h, input_w,
                                                     window[0], window[1],
                                                     stride[0], stride[1],
                                                     dilation[0], dilation[1],
                                                     pad[0], pad[1], pad[2],
                                                     pad[3])
            conv2d_pad = (pad_top, pad_bottom, pad_left, pad_right)
            strides = (1, 1, stride[0], stride[1])
            res = conv2d_compute(input_data,
                                 matrix,
                                 bias,
                                 None,
                                 output_data,
                                 strides,
                                 conv2d_pad,
                                 dilation,
                                 groups=input_c,
                                 data_format='NCHW',
                                 offset_x=offset_x,
                                 kernel_name=kernel_name)
        else:
            # call pooling2d for gap&avg_old
            window = list(window)

            # l1 fusion and l2 fusion
            l1_fusion_type = \
                input_data.op.attrs["L1_fusion_type"].value if "L1_fusion_type" in input_data.op.attrs else -1

            # l1 fusion params assign
            fusion_params = get_fusion_params(input_data, output_data, True)

            if l1_fusion_type == 1:
                input_data.op.attrs["addr_type"].value = 1
                in_l1_flag = True
                fusion_params["in_l1_flag"] = in_l1_flag

                l1_width_fusion_in = \
                    tvm.compute(input_data.shape,
                                lambda n, c1, h, w, c0:
                                input_data(n, c1, h, w, c0),
                                name="l1_width_fusion_tensor_in",
                                attrs=input_data.op.attrs)
                res = tbe.pooling2d(l1_width_fusion_in,
                                    window,
                                    stride,
                                    mode_avg,
                                    pad=pad,
                                    data_mode=0,
                                    ceil_mode=ceil_mode,
                                    fusion_params=fusion_params,
                                    impl_mode=impl_mode)
            else:
                res = tbe.pooling2d(input_data,
                                    window,
                                    stride,
                                    mode_avg,
                                    pad=pad,
                                    data_mode=0,
                                    ceil_mode=ceil_mode,
                                    fusion_params=fusion_params,
                                    impl_mode=impl_mode)
    else:
        error_manager_vector.raise_err_specific_reson("pooling", "the parameter mode should 0 or 1")

    return res


# 'pylint: disable=unnecessary-lambda, variable_type_changed
def pooling_compute(x,
                    matrix,
                    y,
                    window,
                    stride,
                    mode=0,
                    pad=(0, 0, 0, 0),
                    global_pooling=False,
                    ceil_mode=0,
                    kernel_name="pooling_cce",
                    impl_mode="high_performance"):
    """
    describe compute
    return: tensor
    """
    input_x = x.shape
    input_h = input_x[2].value
    input_w = input_x[3].value

    # convert mode&pad_mode to str for pooling2d
    pad = list(pad)
    if mode == 0:
        mode = "MAX"
        if (input_h == window[0] and input_w == window[1] and
            pad == [0, 0, 0, 0]) or \
                global_pooling:
            mode = "GMP"
    elif mode == 1:
        mode = "AVG"
        if (input_h == window[0] and input_w == window[1] and
            pad == [0, 0, 0, 0]) or \
                global_pooling:
            mode = "GAP"
    else:
        error_manager_vector.raise_err_specific_reson("pooling", "the parameter mode should 0 or 1")

    window = list(window)

    # l1 fusion params assign
    fusion_params = get_fusion_params(x, y, False)
    l1_fusion_type = fusion_params.get("l1_fusion_type")
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if all([impl_mode == "high_precision", mode == "GAP", cce_product in ("Hi3796CV300CS", "SD3403")]):
        # L1 fusion not support
        if l1_fusion_type != -1:
            error_manager_vector.raise_err_specific_reson("pooling", "global avg pooling not support L1 fusion")
        # get window size
        window_h = window[0]
        window_w = window[1]
        size_h_const = tvm.const(1.0 / window_h, dtype=x.dtype)
        size_w_const = tvm.const(1.0 / window_w, dtype=x.dtype)
        input_x_vmuls = tbe.vmuls(x, size_h_const)
        input_x_sum = tbe.sum(input_x_vmuls, axis=[2, 3])
        res = tbe.vmuls(input_x_sum, size_w_const)
        return res

    if l1_fusion_type == 1:
        x.op.attrs["addr_type"].value = 1
        in_l1_flag = True
        fusion_params["in_l1_flag"] = in_l1_flag

        l1_width_fusion_in = tvm.compute(x.shape,
                                         lambda n, c1, h, w, c0: x(n, c1, h, w, c0),
                                         name="l1_width_fusion_tensor_in",
                                         attrs=x.op.attrs)
        res = tbe.pooling2d(l1_width_fusion_in,
                            window,
                            stride,
                            mode,
                            pad=pad,
                            data_mode=0,
                            ceil_mode=ceil_mode,
                            fusion_params=fusion_params,
                            impl_mode=impl_mode)
    else:
        res = tbe.pooling2d(x,
                            window,
                            stride,
                            mode,
                            pad=pad,
                            data_mode=0,
                            ceil_mode=ceil_mode,
                            fusion_params=fusion_params,
                            impl_mode=impl_mode)

    return res


# 'pylint: disable = too-many-branches,too-many-statements,variable_type_changed
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def pooling(x,
            matrix,
            bias,
            y,
            window=(1, 1),
            stride=(1, 1),
            offset_x=0,
            mode=0,
            pad=(0, 0, 0, 0),
            global_pooling=False,
            ceil_mode=0,
            dilation=(1, 1, 1, 1),
            kernel_name="pooling_cce",
            impl_mode="high_performance"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16, shape is 4 dims, format is NCHW
    matrix: dict, shape and dtype of right matrix, only support float16, shape is 4 dims, format is NCHW
    bias: dict, shape and dtype of bias, only support float16, shape is 4 dims, format is NCHW, only use bias in conv2d
    y : output dict, shape and dtype of output_data, only support float16
    window : list or tuple, the window of pooling, only support pooling in H or W
    stride : list or tuple, the stride of pooling window, only support pooling in H or W
    offset_x : avg quantize parmas
    mode : int, the mode of pooling, support 0:max pooling, 1:avg pooling.
    pad : list or tuple, the pad of pooling, only support pooling in H or W
    global_pooling : global pooling params.
    ceil_mode : caffe round_mode params, 0:CEIL, 1:FLOOR, default value is DOMI_POOLING_CEIL
    dilation : reserved.
    kernel_name : cce kernel name, default value is "pooling_cce"
    impl_mode : high_precision or high_performance for inference, default value is "high_performance".
    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    # get shape&dtype
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()
    output_dtype = y.get("dtype")
    output_dtype = output_dtype.lower()

    # check others parameter
    _pooling_check_rule(input_shape, output_dtype, window, stride, kernel_name)

    input_h = input_shape[2]
    input_w = input_shape[3]

    if window[0] == 0 or window[1] == 0:
        error_manager_vector.raise_err_specific_reson("pooling", "window value range is [1, 32768], but it's zero")
    # convert mode&pad_mode to str for pooling2d
    if len(pad) != 4:
        error_manager_vector.raise_err_specific_reson("pooling", "the shape of pad is not 4 dims")
    pad = list(pad)
    if pad[0] >= window[0] or pad[1] >= window[0]:
        error_manager_vector.raise_err_specific_reson("pooling", "pad_h must less than kernel_h")
    if pad[2] >= window[1] or pad[3] >= window[1]:
        error_manager_vector.raise_err_specific_reson("pooling", "pad_w must less than kernel_w")

    if mode == 0:
        modes = "MAX"
        if (input_h == window[0] and input_w == window[1] and pad == [0, 0, 0, 0]) or global_pooling:
            modes = "GMP"
    elif mode == 1:
        modes = "AVG"
        if (input_h == window[0] and input_w == window[1] and pad == [0, 0, 0, 0]) or global_pooling:
            modes = "GAP"
    else:
        error_manager_vector.raise_err_specific_reson("pooling", "the parameter mode should 0 or 1")

    is_use_matrix = False
    is_wpad_zero = False
    if pad[2] == 0 and pad[3] == 0:
        is_wpad_zero = True
    if modes == "AVG" and not (input_h != window[0] and input_w == window[1] and is_wpad_zero):
        is_use_matrix = True
    # avg pooling calls conv2d interface to implement
    if modes == "AVG" and matrix and is_use_matrix:
        input_ori_shape = x.get("ori_shape")
        input_c = input_ori_shape[1]
        # get real pad
        _, _, pad_top, pad_bottom, pad_left, pad_right \
            = tbe.get_caffe_out_size_and_pad(ceil_mode, input_h, input_w, window[0], window[1], stride[0], stride[1],
                                             dilation[0], dilation[1], pad[0], pad[1], pad[2], pad[3])
        pad = (pad_top, pad_bottom, pad_left, pad_right)
        offset_w = None
        dilations = (1, 1, 1, 1)
        strides = (1, 1, stride[0], stride[1])
        conv2d(x,
               matrix,
               bias,
               offset_w,
               y,
               strides,
               pad,
               dilations,
               groups=input_c,
               data_format="NCHW",
               offset_x=offset_x,
               kernel_name=kernel_name)
    else:
        # set tensor attrs
        addr_type = x.get("addr_type", 0)
        l1_fusion_type = x.get("L1_fusion_type", -1)
        l1_addr_flag = x.get("L1_addr_flag", -1)
        l1_addr_offset = x.get("L1_addr_offset", -1)
        l1_valid_size = x.get("L1_valid_size", -1)
        attr = {
            "addr_type": addr_type,
            "L1_fusion_type": l1_fusion_type,
            "L1_addr_flag": l1_addr_flag,
            "L1_addr_offset": l1_addr_offset,
            "L1_valid_size": l1_valid_size
        }
        is_l1fusion = l1_fusion_type in (0, 1)

        tensor_in = tvm.placeholder(input_shape, name="tensor_in", dtype=input_dtype, attrs=attr)
        res = pooling_compute(tensor_in, matrix, y, window, stride, mode, pad, global_pooling, ceil_mode, kernel_name,
                              impl_mode)
        # schedule
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)

        # build
        config = {
            "print_ir": False,
            "need_build": False,
            "name": kernel_name,
            "tensor_list": [tensor_in, res],
            "l1_fusion_option": is_l1fusion
        }

        tbe.cce_build_code(sch, config)
