# Copyright 2019 Huawei Technologies Co., Ltd
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
depthwise_conv2d_backprop_filter_d
"""
from impl.conv2d_backprop_filter_d import conv2d_backprop_filter_compute
from impl.conv2d_backprop_filter_d import conv2d_backprop_filter_d
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from tbe.common.utils.conv_util import ConvFeatureMap
from tbe.common.utils.conv_util import ConvKernel


def _get_groups(fmap):
    fm = ConvFeatureMap(fmap)
    return fm.fmap_c


def _convert_kernel(kernel):
    if kernel.get("ori_format").lower() == "hwck":
        kernel["ori_format"] = "HWCN"
    k = ConvKernel(kernel)
    k_format = k.ori_format.lower()
    kernel["ori_shape"] = list(kernel["ori_shape"])
    kernel["ori_shape"][k_format.index("n")] = k.kernel_cout * k.kernel_c
    kernel["ori_shape"][k_format.index("c")] = 1


@tbe_platform.fusion_manager.register("depthwise_conv2d_backprop_filter_d")
def depthwise_conv2d_backprop_filter_d_compute(input_fm,
                                               out_backprop,
                                               filter_grad,
                                               filter_size,
                                               strides,
                                               dilations=(1, 1, 1, 1),
                                               pads=(0, 0, 0, 0),
                                               data_format='NHWC',
                                               kernel_name="depthwise_conv2d_backprop_filter"):
    """
    algorithm: depthwise conv2d

    calculating  depthwise convolution backward filter

    Parameters
    ----------
    input_fm :
    Tvm tensor.
    Placeholder for input feature map.

    out_backprop:
    Tvm tensor.
    Placeholder for derivatives of loss function with respect to output feature map.

    filter_grad :
    a dict.
    4-D origin shape of filter tensor [H, W, C, K],
    K is depthwise_multiplier, support float32.

    filter_size :
    a list/tuple of four ints.
    1-D origin shape of filter tensor with [H, W, C, K],
    K is depthwise_multiplier, support int.

    strides :
    a list/tuple of four ints.
    strides size, [1, 1, stride_height, stride_width] or
    [1, stride_height, stride_width, 1].

    dilations :
    a list/tuple of four ints.
    dilations size, [1, 1, dilation_height, dilation_width] or
    [1, dilation_height, dilation_width, 1].

    pads :
    a list/tuple of four ints.
    padding added to each dimension of the input.

    data_format :
    str.
    shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C].

    kernel_name :
    str.
    cce kernel name.

    Returns
    -------
    None
    """
    x_dict = {
        "ori_format": input_fm.op.attrs["ori_format"],
        "ori_shape": shape_util.shape_to_list(input_fm.op.attrs["ori_shape"]),
        "dtype": input_fm.dtype
    }
    _convert_kernel(filter_grad)
    filter_size = filter_grad.get("ori_shape")
    groups = _get_groups(x_dict)
    return conv2d_backprop_filter_compute(input_fm, out_backprop, filter_grad, filter_size, strides, pads, dilations,
                                          groups, data_format, kernel_name)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def depthwise_conv2d_backprop_filter_d(input_fm,
                                       out_backprop,
                                       filter_grad,
                                       filter_size,
                                       strides,
                                       dilations=(1, 1, 1, 1),
                                       pads=(0, 0, 0, 0),
                                       data_format='NHWC',
                                       kernel_name="depthwise_conv2d_backprop_filter"):
    """
    algorithm: depthwise conv2d

    calculating  depthwise convolution backward filter

    Parameters
    ----------
    input_fm : a dict.
        4-D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
        support float16.

    out_backprop: a dict.
        4-D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
        support float16.

    filter_grad : a dict.
        4-D origin shape of filter tensor [H, W, C, K],
        K is depthwise_multiplier, support float32.

    filter_size : a list/tuple of four ints
        1-D origin shape of filter tensor with [H, W, C, K],
        K is depthwise_multiplier, support int.

    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1].

    dilations : a list/tuple of four ints
        dilations size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1].

    pads : a list/tuple of four ints
        padding added to each dimension of the input.

    data_format : str
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C].

    kernel_name : str
        cce kernel name

    Returns
    -------
    None
    """

    _convert_kernel(filter_grad)
    filter_size = filter_grad.get("ori_shape")
    groups = _get_groups(input_fm)
    conv2d_backprop_filter_d(input_fm, out_backprop, filter_grad, filter_size, strides, pads, dilations, groups,
                             data_format, kernel_name)
