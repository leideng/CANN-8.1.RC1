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
dynamic conv2d_backprop_filter
"""
from impl.dynamic.conv_bp_filter_impl_base import ConvBpFilterImplBase
from impl.util import fusion_util
from impl.util import util_select_op_base
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from tbe.common.utils import log
from tbe.common.utils.conv_util import CubeChecker
from tbe.common.utils.conv_util import CubeConstantConfig

LOWER_STR = [{"result": "UNSUPPORTED", "reason": {"param_index": [0, 2], "type": ["lower_limit", "lower_limit"]}}]
UPPER_STR = [{"result": "UNSUPPORTED", "reason": {"param_index": [0, 2], "type": ["upper_limit", "upper_limit"]}}]


def get_op_support_info(x,
                        filter_size,
                        out_backprop,
                        y,
                        strides,
                        pads,
                        dilations,
                        groups=1,
                        data_format='NHWC',
                        kernel_name=CubeConstantConfig.CONV2D_BACKPROP_FILTER_OP_NAME):
    """
    get the conv2d_backprop_filter split info

    """
    inputs_list = [x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name]
    dw_impl = Conv2dBpFilterDynamicImpl(inputs_list)
    axis_split_matrix, axis_reduce_list = dw_impl.get_op_split_info()
    return util_select_op_base.get_op_cal_info(axis_split_matrix, axis_reduce_list,
                                               CubeConstantConfig.L1FUSION_INPUT_CTR, 0)


@tbe_register.register_param_generalization("Conv2DBackpropFilter")
def conv2d_bp_filter_generalization(x,
                                    filter_size,
                                    out_backprop,
                                    y,
                                    strides,
                                    pads,
                                    dilations,
                                    groups=1,
                                    data_format='NHWC',
                                    kernel_name=CubeConstantConfig.CONV2D_BACKPROP_FILTER_OP_NAME,
                                    generalize_config=None):
    """
    conv2d_backprop_filter generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to conv2d_backprop_filter

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """

    def reset_dtype_bf162fp16(input_list):
        for input_dict in input_list:
            if input_dict["dtype"] == "bfloat16":
                input_dict["dtype"] = "float16"

    result = None
    inputs_list = [x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name]
    if generalize_config.get("mode") == "keep_rank":
        # meet the binary condition and change the range to no_range
        try:
            dw_impl = Conv2dBpFilterDynamicImpl(inputs_list)  # will check inputs
        except RuntimeError as exc:
            return LOWER_STR
        finally:
            pass

        x["ori_shape"], x["ori_range"] = dw_impl.gen_conv_default_shape_range(x.get("ori_format"), x.get("ori_shape"))
        out_backprop["ori_shape"], out_backprop["ori_range"] = dw_impl.gen_conv_default_shape_range(
            out_backprop.get("ori_format"), out_backprop.get("ori_shape"))
        filter_size["const_value"] = None
        result = [[
            x, filter_size, out_backprop, y,
            {"strides": strides}, {"pads": pads}, {"dilations": dilations},
            {"groups": groups}, {"data_format": data_format}
        ]]
    elif generalize_config.get("mode") == "all_shape":
        result = list()
        x["shape"], x["range"] = Conv2dBpFilterDynamicImpl.gen_conv_default_shape_range(x.get("format"), x.get("shape"))
        out_backprop["shape"], out_backprop["range"] = Conv2dBpFilterDynamicImpl.gen_conv_default_shape_range(
            out_backprop.get("format"), out_backprop.get("shape"))
        y["shape"], y["range"] = Conv2dBpFilterDynamicImpl.gen_conv_default_shape_range(y.get("format"), y.get("shape"))
        strides = [-1, -1, -1, -1]
        pads = [-1, -1, -1, -1]
        dilations = [-1, -1, -1, -1]
        groups = -1
        # change formt to ensure reuse
        x["ori_format"] = "NCHW"
        filter_size["ori_format"] = "NCHW"
        out_backprop["ori_format"] = "NCHW"
        y["ori_format"] = "NCHW"
        filter_size["format"] = "NCHW"
        data_format = "NCHW"
        reset_dtype_bf162fp16([x, out_backprop, y])  # for reuse float16 binary
        result.append([x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format])
    return result


@register_operator_compute("Conv2DBackpropFilter", op_mode="dynamic", support_fusion=True)
@para_check.check_input_type(tvm.Tensor, tvm.Tensor, tvm.Tensor, dict, (tuple, list), (tuple, list), (tuple, list), int,
                             str, str, dict)
def conv2d_backprop_filter_fusion_compute(fmap,
                                          filter_tensor,
                                          out_backprop,
                                          y,
                                          strides,
                                          pads,
                                          dilations=(1, 1, 1, 1),
                                          groups=1,
                                          data_format='NHWC',
                                          kernel_name=CubeConstantConfig.CONV2D_BACKPROP_FILTER_OP_NAME,
                                          options=None):
    """
    algorithm: conv2d_backprop_filter

    Parameters
    ----------
    fmap:
    Tvm tensor for input feature map

    filter_tensor:
    Tvm tensor for filter size.

    out_backprop:
    Tvm tensor for input grads.

    y:
    Dict with keys(ori_shape, ori_format, shape, format, dtype, range).

    strides:
    Tuple/list of 4 integers.

    pads:
    Tuple/list of 4 integers
    [pad_top, pad_bottom, pad_left, pad_right]

    dilations:
    Tuple/list of 4 integers
    filter expand size of dilated conv2d_backprop_filter. Default to (1, 1, 1, 1).

    groups:
    int. The number of filter's group. Default value is 1.

    data_format:
    str. An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
    Specify the data format of the input and output data.

    kernel_name:
    str. kernel name, default value is CONV2D_BACKPROP_FILTER_OP_NAME

    Returns
    -------
    Tvm tensor for dedw.
    """
    fusion_util.check_fusion_input([fmap])
    fusion_util.check_fusion_input([filter_tensor])
    fusion_util.check_fusion_input([out_backprop])
    # set fusion build config
    build_cfg = tbe_register.get_fusion_buildcfg()
    if "fusion_op" in build_cfg:
        build_cfg["fusion_op"]["constant_realize_extent_in_infer_bound"] = False
    else:
        build_cfg["fusion_op"] = {"constant_realize_extent_in_infer_bound": False}

    fmap_dict = {
        "ori_format": str(fmap.op.attrs["ori_format"]),
        "ori_shape": shape_util.shape_to_list(fmap.op.attrs["ori_shape"]),
        "dtype": fmap.dtype,
        "shape": shape_util.shape_to_list(fmap.shape),
        "format": str(fmap.op.attrs["format"]),
    }
    filter_dict = {
        "ori_format": str(filter_tensor.op.attrs["ori_format"]),
        "ori_shape": shape_util.shape_to_list(filter_tensor.op.attrs["ori_shape"]),
        "dtype": filter_tensor.dtype,
        "shape": shape_util.shape_to_list(filter_tensor.shape),
        "format": str(fmap.op.attrs["format"]),
    }
    out_backprop_dict = {
        "ori_format": str(out_backprop.op.attrs["ori_format"]),
        "ori_shape": shape_util.shape_to_list(out_backprop.op.attrs["ori_shape"]),
        "dtype": out_backprop.dtype,
        "shape": shape_util.shape_to_list(out_backprop.shape),
        "format": str(fmap.op.attrs["format"]),
    }

    i_list = [
        fmap_dict, filter_dict, out_backprop_dict, y, strides, pads, dilations, groups, data_format, kernel_name
    ]
    dw_impl = Conv2dBpFilterDynamicImpl(i_list, fusion_mode=True, options=options)

    dw_impl.define_vars()
    tensor_list_input = [fmap, filter_tensor, out_backprop]
    return dw_impl.do_compute(tensor_list_input, options)


@register_operator('Conv2DBackpropFilter')
@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.OPTION_ATTR_LIST_INT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_STR,
    para_check.KERNEL_NAME,
)
def conv2d_backprop_filter(x,
                           filter_size,
                           out_backprop,
                           y,
                           strides,
                           pads,
                           dilations,
                           groups=1,
                           data_format='NHWC',
                           kernel_name=CubeConstantConfig.CONV2D_BACKPROP_FILTER_OP_NAME):
    """
    algorithm: conv2d_backprop_filter

    Parameters
    ----------
    x: dict with keys(ori_shape, ori_format, shape, format, dtype, range)
        input feature map tensor.

    filter_size: tuple/list of 4 integers
        The shape of filter. 4-D with shape [filter_height, filter_width, in_channels,
        out_channels] or [out_channels, filter_height, filter_width, in_channels] or
        [out_channels, in_channel, filter_height, filter_width].

    out_backprop: dict with keys(ori_shape, ori_format, shape, format, dtype, range)
        input weight tensor.

    y: dict with keys(ori_shape, ori_format, shape, format, dtype, range)
        output tensor, dtype must be assigned.

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
          [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_filter. Default to (1, 1, 1, 1).

    groups: int
            The number of filter's group. Default value is 1.

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    kernel_name: str
            kernel name, default value is CONV2D_BACKPROP_FILTER_OP_NAME

    Returns
    -------
    None
    """
    inputs_list = [x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name]

    dw_impl = Conv2dBpFilterDynamicImpl(inputs_list)
    dw_impl.format_shape_and_range()

    classified_ins = dw_impl.do_classify()
    sch_list = []
    tensor_list = []
    for cli in classified_ins:
        _, _, option_list = cli
        options = option_list[0].get("options")  # {"compute_template": ct}
        with tbe.compute():
            dw_impl.define_vars()
            tensor_list_input = dw_impl.new_placeholder()
            dedw = dw_impl.do_compute(tensor_list_input, options)
            with tvm.target.cce():
                schs = tbe.auto_schedule(dedw)

            tensor_list.append(tensor_list_input + [dedw])
            sch_list.append(schs)
    log.debug("sch_list num: {}, schs num: {}".format(len(sch_list), len(sch_list[0])))
    dw_impl.do_build(tensor_list, sch_list)


class Conv2dBpFilterDynamicImpl(ConvBpFilterImplBase):

    def __init__(self, inputs_list, fusion_mode=False, options=None) -> None:
        super().__init__(inputs_list, CubeConstantConfig.CONV2D_BACKPROP_FILTER_OP_NAME, fusion_mode, options)
        self.binary_flag = True

    def _check_inputs_format(self):
        x, _, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name = self.inputs_list
        ori_shape_x = x.get("ori_shape")
        ori_shape_out_backprop = out_backprop.get("ori_shape")
        ori_shape_res = y.get("ori_shape")

        dtype_x = x.get("dtype").lower()
        dtype_out_backprop = out_backprop.get("dtype").lower()

        checker = CubeChecker(self.op_name)
        checker.check_kernel_name(kernel_name)
        checker.check_type("x", ori_shape_x, (tuple, list))
        checker.check_type("out_backprop", ori_shape_out_backprop, (tuple, list))
        checker.check_type("y", ori_shape_res, (tuple, list))
        checker.check_type("dilations", dilations, (tuple, list))
        checker.check_type("strides", strides, (tuple, list))
        checker.check_type("pads", pads, (tuple, list))
        checker.check_equal(dtype_x, dtype_out_backprop, "dtype_x", "dtype_out_backprop")

        ori_format_x = x.get("ori_format")
        ori_format_out_backprop = out_backprop.get("ori_format")
        ori_format_res = y.get("ori_format")
        checker.check_format("x", ori_format_x, ("NHWC", "NCHW"))
        checker.check_format("out_backprop", ori_format_out_backprop, ("NHWC", "NCHW"))
        checker.check_format("y", ori_format_res, ("NHWC", "NCHW", "HWCN"))
        checker.check_format("data_format", data_format, ("NHWC", "NCHW"))
        checker.check_equal(ori_format_x, data_format, "fomat_x", "data_format")
        checker.check_equal(ori_format_out_backprop, data_format, "format_dedy", "data_format")

        # check dimension
        checker.check_dims("strides", strides, CubeConstantConfig.CONV_BACKPROP_SHAPE_DIM)
        checker.check_dims("dilations", dilations, CubeConstantConfig.CONV_BACKPROP_SHAPE_DIM)
        checker.check_dims("pads", pads, CubeConstantConfig.CONV_BACKPROP_SHAPE_DIM)
