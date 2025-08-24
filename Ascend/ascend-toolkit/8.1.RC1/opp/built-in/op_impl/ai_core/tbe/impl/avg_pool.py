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
avg_pool
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from tbe import tvm
from te.utils import para_check
from te.utils import shape_util
from te.platform.cce_policy import get_L1_info
from te.utils.error_manager import error_manager_util as err_man
from impl.util import util_select_op_base
from impl.conv2d import conv2d
from impl.conv2d import conv2d_compute
from impl.util.platform_adapter import error_manager_vector
from impl.reduce_mean_d import reduce_mean_d
from te.platform.fusion_manager import fusion_manager
from tbe.common.utils import log


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant.
    """
    AVG_KERNEL_SIZE_H_MUL_W = 255 # kernel_h * kernel_w
    AVG_KERNEL_SIZE = 20 # maximum ksieze
    AVG_STRIDE_SIZE = 64 # maximum strides


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=invalid-name,redefined-builtin,too-many-locals,unused-argument,unused-variable,unnecessary-lambda
def check_supported(x, filter, bias, y, ksize, strides,
                    padding="VALID", data_format="NHWC", offset_x=0,
                    kernel_name="avg_pool",
                    impl_mode=None):
    """
    x : dict, shape and dtype of input_data, only support float16 or int8 \n
    filter : dict, optional input, shape and dtype of input_data, only support float16 or int8 \n
    bias : dict, optional input, shape and dtype of input_data, only support int32 \n
    y : dict, shape and dtype of output_data, only support float16 or int32 \n
    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W \n
    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W \n
    padding : str, the mode of padding, support padding and not padding \n
    data_format : str, default = "NHWC" \n
    offset_x : int, quantization parameter \n
    kernel_name : cce kernel name, default value is "avg_pool_cce" \n
    impl_mode : str, support high_performance and high_precision, default value is None
    """
    ori_shape = y.get("ori_shape")
    input_shape = x.get("ori_shape")
    if data_format == "NHWC":
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        stride_h = strides[1]
        stride_w = strides[2]
        outputh = ori_shape[1]
        outputw = ori_shape[2]
        input_h = input_shape[1]
        input_w = input_shape[2]
    else:
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        stride_h = strides[2]
        stride_w = strides[3]
        outputh = ori_shape[2]
        outputw = ori_shape[3]
        input_h = input_shape[2]
        input_w = input_shape[3]
    is_support_kernel = (ksize_h * ksize_w <= Constant.AVG_KERNEL_SIZE_H_MUL_W) or \
                        (ksize_h <= Constant.AVG_KERNEL_SIZE and ksize_w <= Constant.AVG_KERNEL_SIZE)
    is_global = (outputh == 1) and (outputw == 1)
    is_true_global = (ksize_h == input_h) and (ksize_w == input_w)
    if (ksize_h * ksize_w > Constant.AVG_KERNEL_SIZE_H_MUL_W) and is_global and (not is_true_global):
        reason = "the shape is not supported, ksize:%s ori_shape:%s" % (str(ksize), str(ori_shape))
        return False, reason
    # stride cannot more than 63 because of LOAD3D limitation, modified to 64, use DMA to undertake the usage scenario
    if (not is_global) and (stride_h > Constant.AVG_STRIDE_SIZE or stride_w > Constant.AVG_STRIDE_SIZE):
        reason = "input_shape is not supported by schedule when stride > 64."
        return False, reason
    if not is_support_kernel and outputh != 1 and outputw == 1:
        reason = "the shape is not supported by schedule, ksize:%s ori_shape:%s" % (str(ksize), str(ori_shape))
        return False, reason
    if input_h == 1 and input_w > 100000:
        reason = "input_shape is not supported by schedule when input_h=1 and input_w>100000"
        return False, reason
    return True, ""


def get_op_support_info(x, filter, bias, y, ksize, strides,
                        padding="VALID", data_format="NHWC", offset_x=0,
                        kernel_name="avg_pool",
                        impl_mode=None):
    """
    get the avgpool split
    """
    format_x = x.get("format")
    input_shape = x.get("shape")
    if data_format in ("NHWC",):
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        window = [input_shape[1], input_shape[2]]
    else:
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        window = [input_shape[2], input_shape[3]]

    if format_x == "NC1HWC0":
        if (ksize_h == window[0] and ksize_w == window[1]) or padding == "SAME":
            axis_split_matrix = [[util_select_op_base.SplitInput([0, [0], [-1], [-1]]),
                                 util_select_op_base.SplitOutput([0, [0]])]]
        elif padding == "VALID":
            axis_split_matrix = [
                [util_select_op_base.SplitInput([0, [0], [-1], [-1]]), util_select_op_base.SplitOutput([0, [0]])]]
        else:
            axis_split_matrix = None
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_matrix, axis_reduce_list, 2, 0)

    return op_cal_info_in_json


def _get_fusion_params(input_data, output_data, is_fused_compute=True):
    """
    :param input_data: tensor of input_data
    :param output_data: dict of output_data
    :param is_fused_compute: the default value is true.
    :return: dict fusion_params
    """

    # l1 fusion params assign
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    l1_fusion_type = input_data.op.attrs["L1_fusion_type"].value if "L1_fusion_type" in input_data.op.attrs else -1
    in_l1_flag = input_data.op.attrs["addr_type"].value == 1 if "addr_type" in input_data.op.attrs else False
    l1_addr_flag = input_data.op.attrs["L1_addr_flag"].value if "L1_addr_flag" in input_data.op.attrs else -1
    l1_addr_offset = input_data.op.attrs["L1_addr_offset"] if "L1_addr_offset" in input_data.op.attrs else -1
    l1_valid_size = input_data.op.attrs["L1_valid_size"] if "L1_valid_size" in input_data.op.attrs else -1
    out_l1_flag = output_data.get("addr_type") == 1
    fusion_params = {"is_fused_compute": is_fused_compute,
                     "l1_fusion_type": l1_fusion_type,
                     "in_l1_flag": in_l1_flag,
                     "out_l1_flag": out_l1_flag,
                     "L1_addr_flag": l1_addr_flag,
                     "L1_addr_offset": l1_addr_offset,
                     "L1_valid_size": l1_valid_size}

    return fusion_params


def get_op_specific_info(x, filter, bias, y, ksize, strides,
                         padding="VALID", data_format="NHWC", offset_x=0,
                         kernel_name="avg_pool",
                         impl_mode=None):
    """
    get the avgpool prebuild pattern

    """
    if filter is not None:
        return '{"prebuildPattern": "Convolution"}'
    return '{"prebuildPattern": "undefined"}'


def _avgpool_conv2d_fusion_para(inputs, outputs):
    """
    get L1 fusion para for depthwise_conv2d
    :param inputs: input data
    :param outputs: output data
    :return: l1 convergence parameter
    """

    input_memory_type = inputs.op.attrs["addr_type"] if "addr_type" in inputs.op.attrs else 0
    output_memory_type = outputs["addr_type"] if "addr_type" in outputs else 0
    valid_shape = inputs.op.attrs["valid_shape"] if "valid_shape" in inputs.op.attrs else ()
    slice_offset = inputs.op.attrs["slice_offset"] if "slice_offset" in inputs.op.attrs else ()
    l1_fusion_type = inputs.op.attrs["L1_fusion_type"] if "L1_fusion_type" in inputs.op.attrs else -1

    fmap_l1_addr_flag = inputs.op.attrs["L1_addr_flag"] if "L1_addr_flag" in inputs.op.attrs else -1
    fmap_l1_valid_size = inputs.op.attrs["L1_valid_size"] if "L1_valid_size" in inputs.op.attrs else -1

    l1_fusion_enable_flag = get_L1_info("L1_fusion_enabled")
    if not l1_fusion_enable_flag:
        l1_fusion_type = -1

    valid_shape = shape_util.shape_to_list(valid_shape)
    slice_offset = shape_util.shape_to_list(slice_offset)

    if not l1_fusion_enable_flag:
        input_memory_type = 0
        output_memory_type = 0
        valid_shape = []
        slice_offset = []
        l1_fusion_type = -1

    # 0 is ddr 1 is l1 2 is l2
    if int(input_memory_type) not in (0, 1, 2):
        err_man.raise_err_input_mem_type("depthwise_conv2d",
                                         input_memory_type)
    if int(output_memory_type) not in (0, 1, 2):
        err_man.raise_err_output_mem_type("depthwise_conv2d",
                                          output_memory_type)
    if valid_shape and not slice_offset:
        err_man.raise_err_specific_user(
            "depthwise_conv2d",
            "if valid_shape exists slice_offset can not be []")

    fusion_para = {"input_memory_type": input_memory_type,
                   "output_memory_type": output_memory_type,
                   "valid_shape": valid_shape,
                   "slice_offset": slice_offset,
                   "l1_fusion_type": l1_fusion_type,
                   "fmap_l1_addr_flag": fmap_l1_addr_flag,
                   "fmap_l1_valid_size": fmap_l1_valid_size}

    return fusion_para


# 'pylint: disable=locally-disabled,too-many-arguments
def _pad_compute(padding, input_h, input_w, stride, window, dilations):
    """
    Calculate the pad value.
    :param padding: str, SAME or VALID
    :param input_h: int, input h
    :param output_w: int, output w
    :param stride: list, stride attr
    :param window: list, window attr
    :param dilations: list, dilations attr
    :return: pad
    """

    if padding == "SAME":
        output_h = (input_h + stride[0] - 1) // stride[0]
        output_w = (input_w + stride[1] - 1) // stride[1]
        pad_row = (output_h - 1) * stride[0] + \
                  ((window[0] - 1) * dilations[0] + 1) - input_h
        pad_col = (output_w - 1) * stride[1] + \
                  ((window[1] - 1) * dilations[1] + 1) - input_w
        pad_top = pad_row // 2
        pad_bottom = pad_row - pad_top
        pad_left = pad_col // 2
        pad_right = pad_col - pad_left
        pad_top = _get_pad(int(pad_top))
        pad_bottom = _get_pad(int(pad_bottom))
        pad_left = _get_pad(int(pad_left))
        pad_right = _get_pad(int(pad_right))
        pad = (pad_top, pad_bottom, pad_left, pad_right)
    else:
        pad = (0, 0, 0, 0)
    return pad


# 'pylint: disable=locally-disabled,too-many-arguments,too-many-statements
# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
def _check_window_rule(ksize, strides, data_format):
    """
    check ksize and strides of window in pooling
    :param ksize: list or tuple, the length must be 4
    :param strides: list or tuple, the length must be 4
    :param data_format: input format
    :return: None
    """

    ksize_c = ksize[3] if data_format in ("NHWC",) else ksize[1]
    strides_c = strides[3] if data_format in ("NHWC",) else strides[1]
    if len(ksize) != 4:
        error_manager_vector.raise_err_input_param_range_invalid("avg_pool", "ksize",
                                                                 "4", "4", str(len(ksize)))

    if len(strides) != 4:
        error_manager_vector.raise_err_input_param_range_invalid("avg_pool", "strides",
                                                                 "4", "4", str(len(strides)))

    if ksize[0] != 1 or (ksize_c != 1):
        error_manager_vector.raise_err_input_value_invalid("avg_pool", "ksize[1], ksize[3]",
                                                           "1", str(ksize[1]) + ", " + str(ksize[3]))

    if strides[0] != 1 or strides_c != 1:
        error_manager_vector.raise_err_input_value_invalid("avg_pool", "strides[1], strides[3]",
                                                           "1", str(strides[1]) + ", " + str(strides[3]))

    if data_format not in("NCHW", "NHWC", "NC1HWC0"):
        error_manager_vector.raise_err_input_format_invalid("avg_pool", "x",
                                                            "NC1HWC0, NCHW, NHWC", str(data_format))


def _get_pad(input_pad):
    """
    algorithm:
    obtains the updated pad value.

    Parameters
    ----------
    input_pad: the value of pad
    Returns
    -------
    output_pad: the value of pad
    """

    if input_pad < 0:
        output_pad = 0
    else:
        output_pad = input_pad
    return output_pad


def _avg_pool_check_rule(input_shape, input_dtype,
                         output_dtype, ksize, strides,
                         data_format, kernel_name, impl_mode):
    """
    :param input_shape: shape of input_data
    :param input_dtype: dtype of input_data
    :param output_dtype: dtype of output_data
    :param ksize: the window of avgpooling
    :param strides: the stride of avgpooling window
    :param data_format: NHWC default
    :param kernel_name: cce kernel name
    :param impl_mode: support high_performance and high_precision
    :return: None
    """

    para_check.check_shape(input_shape)
    para_check.check_dtype(input_dtype, ["float16", "int8"])
    para_check.check_dtype(output_dtype, ["float16", "int8", "int32"])
    if (impl_mode is not None) and (impl_mode not in ("high_performance", "high_precision")):
        error_manager_vector.raise_err_input_value_invalid("avg_pool", "impl_mode",
                                                           "high_performance, high_precision", str(impl_mode))
    _check_window_rule(ksize, strides, data_format)


def _avg_pool_global_compute(x, y, ksize, strides,
                             padding="VALID", data_format="NHWC",
                             is_fused_compute=True,
                             kernel_name="avg_pool",
                             impl_mode=None):
    """
    algorithm: avg_pool
    calculating the average pooling

    Parameters
    ----------
    x : placeholders, shape and dtype of input_data, only support float16
    y : dict, shape and dtype of output data, only support float16
    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W
    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W
    padding : str, the mode of padding, support padding and not padding
    data_format : str, default = "NHWC"
    is_fused_compute : bool, default true
    kernel_name : kernel name, default value is "avg_pool"
    impl_mode : str, support high_performance and high_precision, default value is None

    Returns
    -------
    Calculation result
    """

    # create window and stride for pooling2d
    if data_format in ("NHWC",):
        window = [ksize[1], ksize[2]]
        stride = [strides[1], strides[2]]
    else:
        window = [ksize[2], ksize[3]]
        stride = [strides[2], strides[3]]

    # l1 fusion and l2 fusion
    fusion_params = _get_fusion_params(x, y, is_fused_compute)
    l1_fusion_type = fusion_params.get("l1_fusion_type")

    if impl_mode is None:
        # based on pooling2d impl_mode default value
        impl_mode = "high_performance"

    if l1_fusion_type == 1:
        x.op.attrs["addr_type"].value = 1
        in_l1_flag = True
        fusion_params["in_l1_flag"] = in_l1_flag

        l1_width_fusion_in = tvm.compute(x.shape,
                                         lambda n, c1, h, w, c0:
                                         x(n, c1, h, w, c0),
                                         name="l1_width_fusion_tensor_in",
                                         attrs=x.op.attrs)
        res = tbe.pooling2d(l1_width_fusion_in, window, stride,
                            "AVG", padding,
                            fusion_params=fusion_params,
                            impl_mode=impl_mode)
    else:
        res = tbe.pooling2d(x, window, stride, "AVG", padding,
                            fusion_params=fusion_params,
                            impl_mode=impl_mode)

    return res


# 'pylint: disable=unnecessary-lambda,redefined-builtin,too-many-locals
# 'pylint: disable=unnecessary-lambda,too-many-statements
@tbe_platform.fusion_manager.fusion_manager.register("avg_pool")
def avg_pool_compute(x, filter, bias, y, ksize, strides, padding="VALID",
                     data_format="NHWC", offset_x=0, kernel_name="avg_pool",
                     impl_mode=None):
    """
    algorithm: avg_pool
    calculating the average pooling

    Parameters
    ----------
    x : placeholder, shape and dtype of input_data, only support float16 or int8
    filter : optional input, only support float16 or int8
    bias : optional input, only support int32
    y : dict, shape and dtype of output_data, only support float16
    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W
    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W
    padding : str, the mode of padding, support padding and not padding
    data_format : str, default = "NHWC"
    offset_x : quantization parameter
    kernel_name : kernel name, default value is "avg_pool"
    impl_mode : str, support high_performance and high_precision, default value is None

    Returns
    -------
    Calculation result
    """
    output_shape = y.get("ori_shape")
    # create window and stride for pooling2d
    # check  parameter
    _check_window_rule(ksize, strides, data_format)

    if data_format in ("NHWC",):
        window = [ksize[1], ksize[2]]
        stride = [strides[1], strides[2]]
        output_w = output_shape[2]
        inputc = x.op.attrs['ori_shape'][3].value
    else:
        window = [ksize[2], ksize[3]]
        stride = [strides[2], strides[3]]
        output_w = output_shape[3]
        inputc = x.op.attrs['ori_shape'][1].value

    shape_x = x.shape
    input_h = shape_x[2]
    input_w = shape_x[3]
    dilations = (1, 1, 1, 1)

    pad = _pad_compute(padding, input_h, input_w, stride, window, dilations)

    if filter is None:
        res = _avg_pool_global_compute(x, y, ksize, strides, padding, data_format,
                                      is_fused_compute=True, kernel_name=kernel_name, impl_mode=impl_mode)
    else:
        res = conv2d_compute(x, filter, bias, None, y, strides, pad, dilations, groups=inputc,
                             data_format=data_format, offset_x=offset_x, kernel_name=kernel_name)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_STR,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME,
                            para_check.OPTION_ATTR_STR)
def avg_pool(x, filter, bias, y, ksize, strides,
             padding="VALID", data_format="NHWC", offset_x=0,
             kernel_name="avg_pool",
             impl_mode=None):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16 or int8

    filter : dict, optional input, shape and dtype of input_data, only support float16 or int8

    bias : dict, optional input, shape and dtype of input_data, only support int32

    y : dict, shape and dtype of output_data, only support float16 or int32

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support padding and not padding

    data_format : str, default = "NHWC"

    offset_x : int, quantization parameter

    kernel_name : cce kernel name, default value is "avg_pool_cce"

    impl_mode : str, support high_performance and high_precision, default value is None

    Returns
    -------
    None
    """

    # get shape&dtype
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()
    output_dtype = y.get("dtype")
    output_dtype = output_dtype.lower()
    output_shape = y.get("ori_shape")
    output_h = output_shape[1] if data_format in ("NHWC",) else output_shape[2]
    output_w = output_shape[2] if data_format in ("NHWC",) else output_shape[3]

    is_global = (output_h == 1) and (output_w == 1)
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    is_710 = cce_product in ("Ascend310P",)
    axes = [2, 3]

    # check others parameter
    _avg_pool_check_rule(input_shape, input_dtype, output_dtype, ksize, strides,
                         data_format, kernel_name, impl_mode)
    if filter is not None:
        log.info("[%s]: Enter avgpool cube static branch", kernel_name)
        input_h = input_shape[2]
        input_w = input_shape[3]
        input_ori_shape = x.get("ori_shape")
        dilations = (1, 1, 1, 1)
        if data_format in ("NHWC",):
            window = [ksize[1], ksize[2]]
            stride = [strides[1], strides[2]]
            inputc = input_ori_shape[3]
        else:
            window = [ksize[2], ksize[3]]
            stride = [strides[2], strides[3]]
            inputc = input_ori_shape[1]
        pad = _pad_compute(padding, input_h, input_w, stride, window, dilations)
        filter_n = inputc
        offset_w = None
        conv2d(x, filter, bias, offset_w, y, strides, pad, dilations,
               groups=filter_n, data_format=data_format, offset_x=offset_x,
               kernel_name=kernel_name)
    elif is_710 and is_global:
        log.info("[%s]: Enter avgpool vector static branch, using reduce_mean", kernel_name)
        if impl_mode is None:
            # based on 710 global avg pool default value
            impl_mode = "high_precision"
        reduce_mean_d(x, y, axes, keep_dims=True, kernel_name=kernel_name, impl_mode=impl_mode)
        fusion_manager.set_current_op_pattern("Pool2d")
    else:
        log.info("[%s]: Enter avgpool vector static branch", kernel_name)
        # set tensor attrs, during L1 fusion these attrs will assign by te_fusion
        addr_type = x.get("addr_type", 0)
        l1_fusion_type = x.get("L1_fusion_type", -1)
        l1_addr_flag = x.get("L1_addr_flag", -1)
        l1_addr_offset = x.get("L1_addr_offset", -1)
        l1_valid_size = x.get("L1_valid_size", -1)
        attr = {"addr_type": addr_type,
                "L1_fusion_type": l1_fusion_type,
                "L1_addr_flag": l1_addr_flag,
                "L1_addr_offset": l1_addr_offset,
                "L1_valid_size": l1_valid_size}
        is_l1fusion = l1_fusion_type in (0, 1)
        # create tensor_in
        tensor_in = tvm.placeholder(input_shape, name="tensor_in",
                                    dtype=input_dtype, attrs=attr)
        res = _avg_pool_global_compute(tensor_in, y, ksize, strides,
                                       padding, data_format, False, kernel_name, impl_mode=impl_mode)
        tensor_list = [tensor_in, res]
        # schedule
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)

        # build
        config = {"print_ir": False,
                  "need_build": False,
                  "name": kernel_name,
                  "tensor_list": tensor_list,
                  "l1_fusion_option": is_l1fusion}

        tbe.cce_build_code(sch, config)
