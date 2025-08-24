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
avg_pool_v2
"""
import json
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.conv2d import conv2d
from impl.conv2d import conv2d_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import OpImplMode
from impl.util import util_conv2d
from impl.util.util_common import check_op_impl_mode
from tbe.common.utils import log


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant.
    """
    AVGV2_KERNEL_SIZE_H_MUL_W = 255
    AVGV2_KERNEL_SIZE = 20
    AVGV2_DIVISOR_OVERRIDE_MAX = 255


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=invalid-name,redefined-builtin,too-many-locals,unused-argument,no-else-raise,unnecessary-lambda
def check_supported(x, filter, bias, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                    data_format="NCHW", global_pooling=False, ceil_mode=False,
                    exclusive=True, offset_x=0, divisor_override=0, kernel_name="avg_pool_v2",
                    impl_mode="high_performance"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16, shape is 4
        dims, format is NCHW

    filter : assist matrix

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support VALID, SAME and CALCULATED

    pads : padding value when padding_mode is CALCULATED

    data_format : str, default = "NCHW"

    global_pooling : global pooling or not

    ceil_mode : use ceil or floor to calculate ho and wo when padding_mode is CALCULATED

    exclusive : ignore padding area or not when calculating the average

    kernel_name : cce kernel name, default value is "avg_pool_v2"

    impl_mode : assign high_performance or high_precision

    Returns
    -------
    True or False
    """
    ori_shape = y.get("ori_shape")
    if data_format == "NHWC":
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        outputh = ori_shape[1]
        outputw = ori_shape[2]
    else:
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        outputh = ori_shape[2]
        outputw = ori_shape[3]

    # not support schedule for ksize
    ai_core_skize = ksize_h <= 255 and ksize_w <= 255
    is_support_kernel = (ksize_h * ksize_w <= Constant.AVGV2_KERNEL_SIZE_H_MUL_W) or \
                        (ksize_h <= Constant.AVGV2_KERNEL_SIZE and ksize_w <= Constant.AVGV2_KERNEL_SIZE)
    reason = "the shape is not supported by schedule, ksize:%s ori_shape:%s" % (str(ksize), str(ori_shape))
    if not is_support_kernel and outputh != 1 and outputw == 1:
        return False, reason
    if (not global_pooling) and (not ai_core_skize):
        return False, reason

    # not support schedule for divisor_override
    is_divisor_override_valid = divisor_override >= 0 and divisor_override <= Constant.AVGV2_DIVISOR_OVERRIDE_MAX
    reason_divisor_override = "the divisor_overrride is invalid, divisor_override = %s" % (str(divisor_override))
    if not is_divisor_override_valid:
        return False, reason_divisor_override
    return True, ""


def get_op_support_info(x, filter, bias, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                        data_format="NCHW", global_pooling=False, ceil_mode=False,
                        exclusive=True, offset_x=0, divisor_override=0, kernel_name="avg_pool_v2",
                        impl_mode="high_performance"):
    """
    algorithm: get_op_support_info

    Notice
    ------
    get the avgpoolv2 split

    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16, shape is 4
        dims, format is NCHW
    filter : assist matrix
    y : dict, shape and dtype of output_data, only support float16
    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W
    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W
    padding : str, the mode of padding, support VALID, SAME and CALCULATED
    pads : padding value when padding_mode is CALCULATED
    data_format : str, default = "NCHW"
    global_pooling : global pooling or not
    ceil_mode : use ceil or floor to calculate ho and wo when padding_mode
                is CALCULATED
    exclusive : ignore padding area or not when calculating the average
    kernel_name : cce kernel name, default value is "avg_pool_v2"
    impl_mode : assign high_performance or high_precision


    Returns
    -------
    split info of avg_pool_v2
    """


    def remove_cout_split_info(temp_info, slice_info):
        """
        remove_cout_split_info
        """
        temp_info_new = []
        for _, item in enumerate(temp_info):
            if item["inputList"][0]["idx"] != 1:
                temp_info_new.append(item)
        slice_info["_op_slice_info"]["splitMaps"] = temp_info_new


    def check_global_pooling_remove_h_w_split(temp_info, slice_info, global_pooling):
        """
        when glbal_pooling, remove corresponding H/W split info
        """
        temp_global_pooling = []
        for _, item in enumerate(temp_info):
            if item["inputList"][0]["axis"][0] == 0:
                temp_global_pooling.append(item)
        if global_pooling:
            slice_info["_op_slice_info"]["splitMaps"] = temp_global_pooling


    def check_dynamic(x, slice_info):
        """
        process for dynamic shape
        """
        input_shape = []
        if x.get("ori_shape"):
            input_shape = x.get("ori_shape")
        dynamic_flag = False
        if input_shape:
            input_shape = list(input_shape)
            if -1 in input_shape or input_shape == [-2]:
                dynamic_flag = True
        if dynamic_flag:
            slice_info["_op_slice_info"]["splitMaps"] = []


    bias_idx = 2
    bias = None
    format_x = x.get("format")
    slice_info = util_conv2d.get_op_support_info_static_common(bias, bias_idx, format_x)

    if slice_info.get('_op_slice_info'):
        if slice_info.get('_op_slice_info').get("splitMaps"):
            try:
                temp_info = slice_info['_op_slice_info']["splitMaps"]
            except KeyError:
                error_detail = "Key(_op_slice_info or splitMaps) not in the dict"
                error_manager_vector.raise_err_specific_user("avg_pool_v2", error_detail)
            remove_cout_split_info(temp_info, slice_info)
            check_global_pooling_remove_h_w_split(temp_info, slice_info, global_pooling)

            format_x = x.get("format")
            if format_x != "NC1HWC0":
                try:
                    slice_info["_op_slice_info"]["splitMaps"] = []
                except KeyError:
                    error_detail = "Key(_op_slice_info or splitMaps) not in the dict"
                    error_manager_vector.raise_err_specific_user("avg_pool_v2", error_detail)
            check_dynamic(x, slice_info)
    return json.dumps(slice_info)
    

def get_op_specific_info(x, filter, bias, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                         data_format="NCHW", global_pooling=False, ceil_mode=False,
                         exclusive=True, offset_x=0, divisor_override=0, kernel_name="avg_pool_v2",
                         impl_mode="high_performance"):
    """
    get the avgpool prebuild pattern

    """
    if filter is not None:
        return '{"prebuildPattern": "Convolution"}'
    return '{"prebuildPattern": "undefined"}'


def _get_fusion_params(input_data, output_data, is_fused_compute=True):
    """
    function to get fusion params

    Parameters
    ----------
    input_data: tensor of input_data

    output_data: dict of output_data

    is_fused_compute: fused or not

    Returns
    -------
    fusion_params: dict fusion_params
    """
    # l1 fusion params assign
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    l1_fusion_type = input_data.op.attrs["L1_fusion_type"].value if "L1_fusion_type" in input_data.op.attrs else -1
    in_l1_flag = input_data.op.attrs["addr_type"].value == 1 if "addr_type" in input_data.op.attrs else False
    in_valid_shape = input_data.op.attrs["valid_shape"] if "valid_shape" in input_data.op.attrs else []
    in_slice_offset = input_data.op.attrs["slice_offset"] if "slice_offset" in input_data.op.attrs else []
    in_select_read_flag = bool(in_valid_shape)
    in_split_index = input_data.op.attrs["split_index"].value if "split_index" in input_data.op.attrs else 0
    out_l1_flag = output_data.get("addr_type") == 1
    fusion_params = {"is_fused_compute": is_fused_compute,
                     "l1_fusion_type": l1_fusion_type,
                     "in_l1_flag": in_l1_flag,
                     "out_l1_flag": out_l1_flag,
                     "in_select_read_flag": in_select_read_flag,
                     "in_split_index": in_split_index,
                     "in_slice_offset": in_slice_offset}

    return fusion_params


# 'pylint: disable=locally-disabled,too-many-arguments,too-many-statements,redefined-builtin
# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
def _check_window_rule(ksize, strides, pads, data_format):
    """
    check ksize and strides of window in pooling
    """
    if len(pads) != 4:
        error_manager_vector.raise_err_input_value_invalid("avg_pool_v2", "pads", '4', len(pads))

    if data_format in ("NHWC",):
        if len(ksize) != 4:
            error_manager_vector.raise_err_input_param_range_invalid("avg_pool_v2", "ksize", '4', '4', len(ksize))
        elif ksize[0] != 1 or ksize[3] != 1:
            error_manager_vector.raise_err_input_value_invalid("avg_pool_v2", "ksize[0], ksize[3]", '1',
                                                               str(ksize[0]) + "," + str(ksize[3]))

        if len(strides) != 4:
            error_manager_vector.raise_err_input_value_invalid("avg_pool_v2", "strides", '4', len(strides))
        elif strides[0] != 1 or strides[3] != 1:
            error_manager_vector.raise_err_input_value_invalid("avg_pool_v2", "strides[0], strides[3]", '1',
                                                               str(strides[0]) + "," + str(strides[3]))
    elif data_format in ("NC1HWC0", "NCHW"):
        if len(ksize) != 4:
            error_manager_vector.raise_err_input_value_invalid("avg_pool_v2", "ksize", '4', len(ksize))
        elif ksize[0] != 1 or ksize[1] != 1:
            error_manager_vector.raise_err_input_value_invalid("avg_pool_v2", "ksize[0], ksize[1]", '1',
                                                               str(ksize[0]) + "," + str(ksize[1]))

        if len(strides) != 4:
            error_manager_vector.raise_err_input_value_invalid("avg_pool_v2", "strides", '4', len(strides))
        elif strides[0] != 1 or strides[1] != 1:
            error_manager_vector.raise_err_input_value_invalid("avg_pool_v2", "strides[0], strides[1]", '1',
                                                               str(strides[0]) + "," + str(strides[1]))
    else:
        error_manager_vector.raise_err_input_format_invalid("avg_pool_v2", "x",
                                                            ["NC1HWC0", "NCHW", "NHWC"], data_format)


def _get_corrected_pad(input_pad):
    """
    algorithm:
    get corrected pad value

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


def _avg_pool_v2_check_rule(input_shape, input_dtype, output_dtype, input_format, ksize, strides,
                            pads, data_format, kernel_name):
    """
    function to check params

    Parameters
    ----------
    input_shape: shape of input_data

    input_dtype: dtype of input_data

    output_dtype: dtype of output_data

    input_format: format of input

    ksize: the window of avg_pool_v2

    strides: the stride of avg_pool_v2 window

    pads: padding value when padding_mode is CALCULATED

    data_format: NCHW default

    kernel_name: cce kernel name

    Returns
    -------
    None

    """
    # check input and output
    para_check.check_shape(input_shape)
    para_check.check_dtype(input_dtype, ["float16", "int8"])
    para_check.check_dtype(output_dtype, ["float16", "int32"])

    _check_window_rule(ksize, strides, pads, data_format)


def _calculate_pads(padding, input_h, input_w, stride_h, stride_w, ksize_h, ksize_w, dilations, pads, ceil_mode):
    """
    function to calculate pad value
    """
    if padding == "SAME":
        output_h = (input_h + stride_h - 1) // stride_h
        output_w = (input_w + stride_w - 1) // stride_w
        pad_row = (output_h - 1) * stride_h + ((ksize_h - 1) * dilations[0] + 1) - input_h
        pad_col = (output_w - 1) * stride_w + ((ksize_w - 1) * dilations[1] + 1) - input_w

        pad_top = pad_row // 2
        pad_bottom = pad_row - pad_top
        pad_left = pad_col // 2
        pad_right = pad_col - pad_left

        pad_top = _get_corrected_pad(int(pad_top))
        pad_bottom = _get_corrected_pad(int(pad_bottom))
        pad_left = _get_corrected_pad(int(pad_left))
        pad_right = _get_corrected_pad(int(pad_right))

        pad = (pad_top, pad_bottom, pad_left, pad_right)

    elif padding == "CALCULATED":
        pad_top, pad_bottom, pad_left, pad_right = pads

        if ceil_mode:
            ho = (input_h - ksize_h + pad_top + pad_bottom + stride_h - 1) // stride_h + 1
            wo = (input_w - ksize_w + pad_left + pad_right + stride_w - 1) // stride_w + 1
            pad_bottom = _get_corrected_pad(int((ho - 1) * stride_h + ksize_h - input_h - pad_top))
            pad_right = _get_corrected_pad(int((wo - 1) * stride_w + ksize_w - input_w - pad_left))
        else:
            ho = (input_h - ksize_h + pad_top + pad_bottom) // stride_h + 1
            wo = (input_w - ksize_w + pad_left + pad_right) // stride_w + 1
            pad_bottom = _get_corrected_pad(int((ho - 1) * stride_h + ksize_h - input_h - pad_top))
            pad_right = _get_corrected_pad(int((wo - 1) * stride_w + ksize_w - input_w - pad_left))

        pad = (pad_top, pad_bottom, pad_left, pad_right)

    else:
        pad = (0, 0, 0, 0)

    return pad


# 'pylint: disable=unnecessary-lambda,redefined-builtin,too-many-locals
# 'pylint: disable=unnecessary-lambda,too-many-statements
@register_operator_compute("avg_pool_v2", op_mode="static", support_fusion=True)
def avg_pool_v2_compute(x, filter, bias, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                        data_format="NCHW", global_pooling=False, ceil_mode=False,
                        exclusive=True, offset_x=0, divisor_override=0, kernel_name="avg_pool_v2",
                        impl_mode="high_performance"):
    """
    algorithm: avg_pool_V2
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
    data_format : str, default = "NCHW"
    kernel_name : kernel name, default value is "avg_pool_V2"

    Returns
    -------
    Calculation result
    """
    input_shape = list(x.shape)
    input_h = input_shape[2]
    input_w = input_shape[3]
    if data_format in ("NHWC",):
        ksize_h = int(ksize[1])
        ksize_w = int(ksize[2])
        stride_h = int(strides[1])
        stride_w = int(strides[2])
        inputc = x.op.attrs['ori_shape'][3].value
    else:
        ksize_h = int(ksize[2])
        ksize_w = int(ksize[3])
        stride_h = int(strides[2])
        stride_w = int(strides[3])
        inputc = x.op.attrs['ori_shape'][1].value

    if global_pooling:
        ksize = list(ksize)
        if data_format in ("NHWC",):
            ksize[1] = input_h
            ksize[2] = input_w
        else:
            ksize[2] = input_h
            ksize[3] = input_w
        padding = 'VALID'

    if list(pads) == [0, 0, 0, 0] and ksize_h == input_h and ksize_w == input_w:
        if padding == "CALCULATED":
            padding = 'VALID'
        if padding == "SAME" and stride_h == input_h and stride_w == input_w:
            padding = 'VALID'

    if filter is not None:
        dilations = (1, 1, 1, 1)
        pad = _calculate_pads(padding, input_h, input_w, stride_h, stride_w, ksize_h, ksize_w,
                              dilations, pads, ceil_mode)
        res = conv2d_compute(x, filter, bias, None, y, strides, pad, dilations, groups=inputc,
                             data_format=data_format, offset_x=offset_x, kernel_name=kernel_name)
    else:
        res = avg_pool_v2_compute1(x, y, ksize, strides, padding, data_format, False, kernel_name, impl_mode)

    return res


def avg_pool_v2_compute1(x, y, ksize, strides, padding="VALID", data_format="NHWC",
                         is_fused_compute=True, kernel_name="avg_pool_v2",
                         impl_mode="high_performance"):
    """
    function of avg_pool_v2 compute

    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16, shape is 4
        dims, format is NCHW

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support VALID, SAME and CALCULATED

    data_format : str, default = "NCHW"

    is_fused_compute : fuse or not

    kernel_name : cce kernel name, default value is "avg_pool_v2"

    impl_mode : assign high_performance or high_precision

    Returns
    -------
    res : output of avg_pool_v2
    """
    # create window and stride for pooling2d
    if data_format in ("NHWC",):
        window = [int(ksize[1]), int(ksize[2])]
        stride = [int(strides[1]), int(strides[2])]
    else:
        window = [int(ksize[2]), int(ksize[3])]
        stride = [int(strides[2]), int(strides[3])]

    window = list(window)
    stride = list(stride)

    # l1 fusion and l2 fusion
    l1_fusion_type = x.op.attrs["L1_fusion_type"].value if "L1_fusion_type" in x.op.attrs else -1
    fusion_params = _get_fusion_params(x, y, is_fused_compute)
    in_select_read_flag = fusion_params.get("in_select_read_flag")
    in_valid_shape = fusion_params.get("in_valid_shape")
    in_slice_offset = fusion_params.get("in_slice_offset")

    if in_select_read_flag:
        select_tensor_in = tvm.compute(in_valid_shape,
                                       lambda n, c1, h, w, c0:
                                       x(n, c1, h + in_slice_offset[2], w, c0),
                                       name="tensor_read_select",
                                       attrs=x.op.attrs)
        res = tbe.pooling2d(select_tensor_in, window, stride, "AVG", padding,
                            fusion_params=fusion_params, impl_mode=impl_mode)
    elif l1_fusion_type == 1:
        x.op.attrs["addr_type"].value = 1
        in_l1_flag = True
        fusion_params["in_l1_flag"] = in_l1_flag

        l1_width_fusion_in = tvm.compute(x.shape,
                                         lambda n, c1, h, w, c0:
                                         x(n, c1, h, w, c0),
                                         name="l1_width_fusion_tensor_in",
                                         attrs=x.op.attrs)
        res = tbe.pooling2d(l1_width_fusion_in, window, stride, "AVG", padding,
                            fusion_params=fusion_params, impl_mode=impl_mode)
    else:
        res = tbe.pooling2d(x, window, stride, "AVG", padding, fusion_params=fusion_params, impl_mode=impl_mode)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def avg_pool_v2(x, filter, bias, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                data_format="NCHW", global_pooling=False, ceil_mode=False,
                exclusive=True, offset_x=0, divisor_override=0, kernel_name="avg_pool_v2",
                impl_mode="high_performance"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16, shape is 4
        dims, format is NCHW

    filter : assist matrix

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support VALID, SAME and CALCULATED

    pads : padding value when padding_mode is CALCULATED

    data_format : str, default = "NCHW"

    global_pooling : global pooling or not

    ceil_mode : use ceil or floor to calculate ho and wo when padding_mode is CALCULATED

    exclusive : ignore padding area or not when calculating the average

    kernel_name : cce kernel name, default value is "avg_pool_v2"

    impl_mode : assign high_performance or high_precision

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    # get shape&dtype
    input_shape = x.get("shape")
    input_ori_shape = x.get("ori_shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()
    output_dtype = y.get("dtype")
    output_dtype = output_dtype.lower()
    input_format = x.get("format")

    # check others parameter
    _avg_pool_v2_check_rule(input_shape, input_dtype, output_dtype, input_format, ksize, strides,
                            pads, data_format, kernel_name)

    # set tensor attrs, during L1 fusion these attrs will assign by te_fusion
    addr_type = x.get("addr_type", 0)
    valid_shape = x.get("valid_shape", [])
    slice_offset = x.get("slice_offset", [])
    split_index = x.get("split_index", 0)
    l1_fusion_type = x.get("L1_fusion_type", -1)

    is_l1fusion = l1_fusion_type in (0, 1)
    input_h = input_shape[2]
    input_w = input_shape[3]
    
    attr = {"addr_type": addr_type,
        "valid_shape": valid_shape,
        "slice_offset": slice_offset,
        "split_index": split_index,
        "L1_fusion_type": l1_fusion_type}
    
    if data_format in ("NHWC",):
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        stride_h = strides[1]
        stride_w = strides[2]
        inputc = input_ori_shape[3]
    else:
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        stride_h = strides[2]
        stride_w = strides[3]
        inputc = input_ori_shape[1]

    if global_pooling:
        ksize = list(ksize)
        if data_format in ("NHWC",):
            ksize[1] = input_h
            ksize[2] = input_w
        else:
            ksize[2] = input_h
            ksize[3] = input_w
        padding = 'VALID'

    if list(pads) == [0, 0, 0, 0] and ksize_h == input_h and ksize_w == input_w:
        if padding == "CALCULATED":
            padding = 'VALID'
        if padding == "SAME" and stride_h == input_h and stride_w == input_w:
            padding = 'VALID'

    if filter is not None:
        log.info("[%s]: Enter avgpoolv2 cube static branch", kernel_name)
        dilations = (1, 1, 1, 1)
        pad = _calculate_pads(padding, input_h, input_w, stride_h, stride_w, ksize_h, ksize_w,
                              dilations, pads, ceil_mode)

        filter_n = inputc
        conv2d(x, filter, bias, None, y, strides, pad, dilations,
               groups=filter_n, data_format=data_format, offset_x=offset_x,
               kernel_name=kernel_name)
    else:
        log.info("[%s]: Enter avgpoolv2 vector static branch", kernel_name)
        tensor_in = tvm.placeholder(input_shape, name="tensor_in", dtype=input_dtype, attrs=attr)
        res = avg_pool_v2_compute1(tensor_in, y, ksize, strides, padding, data_format, False, kernel_name, impl_mode)

        tensor_list = [tensor_in, res]

        # schedule
        with tvm.target.cce():
            sch = auto_schedule(res)

        # build
        config = {"print_ir": False,
                  "need_build": False,
                  "name": kernel_name,
                  "tensor_list": tensor_list,
                  "l1_fusion_option": is_l1fusion}

        build(sch, config)

