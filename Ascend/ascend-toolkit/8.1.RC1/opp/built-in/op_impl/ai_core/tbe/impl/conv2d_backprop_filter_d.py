#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
conv2d_backprop_filter_d
"""
import impl.dynamic as dyn_impl
from impl.dynamic.conv_bp_filter_impl_base import ConvBpFilterImplBase
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.util_common import BIT_RATIO_DICT
from impl.util.util_common import ShapeConverter
from impl.util.util_common import align
from tbe.common.utils import log
from tbe.common.utils.conv_util import CubeChecker
from tbe.common.utils.conv_util import CubeConstantConfig
from tbe.common.utils.conv_util import trip_strides
from tbe.dsl.base.operation import get_op_context


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.OPTION_ATTR_LIST_INT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_STR,
    para_check.KERNEL_NAME,
)
def get_op_support_info(
        x,
        out_backprop,
        y,
        filter_size,
        strides,
        pads,
        dilations=(1, 1, 1, 1),
        groups=1,
        data_format="NHWC",
        kernel_name="conv2d_backprop_filter",
):
    """
    get the conv2d_backprop_filter split info

    Parameters
    ----------
    x: dict with keys(ori_shape, ori_format, shape, format, dtype)
        input feature map tensor.

    out_backprop: dict with keys(ori_shape, ori_format, shape, format, dtype)
        input weight tensor.

    y: dict with keys(ori_shape, ori_format, shape, format, dtype)
        output tensor, dtype must be assigned.

    filter_size: tuple/list of 4 integers

    strides: tuple/list of 2 integers

    pads: tuple/list of 4 integers

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_filter. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_backprop_filter. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    kernel_name: str
        kernel name. Default to "conv2d_backprop_filter".

    Returns
    -------
    split info, split axis and min l1 space
    """

    inputs_list = [x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name]
    dw_impl = Conv2dBpFilterStaticImpl(inputs_list)
    dw_impl.check_inputs_logic(check_format=False)

    axis_split_matrix, axis_reduce_list = dw_impl.get_op_split_info()
    min_l1space = dw_impl.cal_min_l1space_for_lxfusion()
    return util_select_op_base.get_op_cal_info(axis_split_matrix, axis_reduce_list,
                                               CubeConstantConfig.L1FUSION_INPUT_CTR, min_l1space)


def check_supported(x,
                    out_backprop,
                    y,
                    filter_size,
                    strides,
                    pads,
                    dilations=(1, 1, 1, 1),
                    groups=1,
                    data_format="NHWC",
                    kernel_name="conv2d_backprop_filter"):
    """
    check the op support situation:

    | Name             | Field    | Scope
    -------------------|----------|--------------
    | x                | H or W   | [1, 1000000]
    -------------------|----------|--------------
    | out_backprop     | H or W   | [1, 1000000]
    -------------------|----------|--------------
    | filter_size      | H or W   | [1, 1000000]
    -------------------|----------|--------------
    | y(filter)        | H or W   | [1, 1000000]
    -------------------|----------|--------------
    | Stride           | H or W   | [1, 1000000]
    -------------------|----------|--------------
    | Dilation         | H or W   | [1, 255]

    In Ascend910, out_backprop's H and W not support 1
    when fmap_h + pad_top + pad_bottom != (filter_height - 1) * dilation_h + 1

    batch_x == batch_out_backprop

    batch_filter == channel_out_backprop

    channel_filter == channel_x * groups

    out_backprop_height == (fmap_height + pad_top + pad_bottom -
                          (dilation_h * (filter_height - 1) + 1))
                           / stride_h + 1

    out_backprop_width == (fmap_width + pad_left + pad_right -
                         (dilation_w * (filter_width - 1) + 1))
                          / stride_w + 1
    """

    if any([i < 0 for i in x.get("ori_shape")]):
        return True, ""
    try:
        inputs_list = [x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name]
        dw_impl = Conv2dBpFilterStaticImpl(inputs_list)
        dw_impl.check_inputs_logic(check_format=False)  # if input ilegal, will raise error
        return True, ""
    except RuntimeError as e:
        reason = e.args[1]
        return False, reason


def op_select_format(x,
                     out_backprop,
                     y,
                     filter_size,
                     strides,
                     pads,
                     dilations=(1, 1, 1, 1),
                     groups=1,
                     data_format="NHWC",
                     kernel_name="conv2d_backprop_filter_d"):
    r"""
    1.When input x type is float16, op select supports the following specification:

    | Tensor    | x        | outbackprop  | y         |
    | :-------: | :------: | :----------: | :-------: |
    | Data Type | float16  | float16      | float32   |
    | Format    | NC1HWC0  | NC1HWC0      | FRACTAL_Z |

    Note:
    a. C0 = 32 / sizeof(data type), C1 = ceil(in_channels / C0), for float16 type, C0 = 16

    2.When input x type is float32, op select supports the following specification:

    | Tensor    | x        | outbackprop  | y         |
    | :-------: | :------: | :----------: | :-------: |
    | Data Type | float32  | float32      | float32   |
    | Format    | NC1HWC0  | NC1HWC0      | FRACTAL_Z |

    3.When input x type is bfloat16, op select supports the following specification:

    | Tensor    | x        | outbackprop  | y         |
    | :-------: | :------: | :----------: | :-------: |
    | Data Type | bfloat16 | bfloat16     | float32   |
    | Format    | NC1HWC0  | NC1HWC0      | FRACTAL_Z |

    4.When in_channels <=4, filter_height > 1, filter_width > 1, op select supports the additional
    FRACTAL_Z_C04 format of input filter:

    | Tensor    | x            | outbackprop    | y              |
    | :-------: | :----------: | :------------: | :-------------:|
    | Format    | NC1HWC0      | NC1HWC0        |  FRACTAL_Z_C04 |

    Note: the data type rules ares the same as above

    Note:
    - convolution input data calculation with in_channels reduce for every out_height line:

            =----------------------------=
            |\^                           \
            | \in_channels                 \
            |  \ v      <- in_width ->      \
            |   =----------------------------=
            |   |  ...                       |
            |   |                      ^     |
            |   | |#| |#| ...          |    -|--
            |   |  |                         |^
            |   | stride           in_height || image data needed for one line out on height
            |   |  |                         |v
            =   | |#| |#| ...          |    -|--
             \  |  ...                 v     |
              \ |                            |
               \|                            |
                =----------------------------=
            "|#|" is the kernel(after dilate) swipe on the input image

    - minimum size calculation:

        `limit_out_height = lcm(out_width, 16) // out_width ("lcm": least common multiple)`
        `dilated_filter_h = dilation_h * (filter_height - 1) + 1`
        `limit_in_height = (limit_out_height - 1) * stride_h + dilated_filter_h - pad_left - pad_right`
        `minimum_size = limit_in_height * in_width * 8 ("8": comes from 2btype*C0=4)`
    """

    inputs_list = [x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name]
    dw_impl = Conv2dBpFilterStaticImpl(inputs_list)
    params_list = dw_impl.select_format()
    return util_select_op_base.get_dynamic_param_in_json(params_list)


@tbe_platform.fusion_manager.register("conv2d_backprop_filter_d")
def conv2d_backprop_filter_compute(x,
                                   out_backprop,
                                   y,
                                   filter_size,
                                   strides,
                                   pads,
                                   dilations=(1, 1, 1, 1),
                                   groups=1,
                                   data_format="NHWC",
                                   kernel_name="conv2d_backprop_filter"):
    """
    used for fusion
    Parameters
    ----------
    x: Tensor
        input tensor.

    out_backprop: Tensor
        conv2d output gradients tenosr.

    y: dict with keys(shape and dtype)
        conv2d_backprop_filter output tensor, dtype must be assigned.

    filter_size: tuple/list of 4 integers
        The shape of feature map. 4-D with shape [batch, height, width, channels]
        or [batch, channels, height, filter].

    strides: tuple/list of 4 integers
        filter move stride.

    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_filter. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_backprop_filter. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    kernel_name: str
        kernel name. Default to "conv2d_backprop_filter".

    Returns
    -------
    Tensor of conv2d_backprop_filter
    """
    x_dict = {
        "ori_format": str(x.op.attrs["ori_format"]),
        "ori_shape": shape_util.shape_to_list(x.op.attrs["ori_shape"]),
        "dtype": x.dtype,
        "shape": shape_util.shape_to_list(x.shape),
        "format": "NC1HWC0",
        "tag": x.op.tag
    }
    out_backprop_dict = {
        "ori_format": str(out_backprop.op.attrs["ori_format"]),
        "ori_shape": shape_util.shape_to_list(out_backprop.op.attrs["ori_shape"]),
        "dtype": out_backprop.dtype,
        "format": "NC1HWC0",
        "shape": shape_util.shape_to_list(out_backprop.shape)
    }

    i_list = [x_dict, filter_size, out_backprop_dict, y, strides, pads, dilations, groups, data_format, kernel_name]
    get_op_context().add_addition("fusion_op", True)

    dw_impl = Conv2dBpFilterStaticImpl(i_list, fusion_mode=True)
    dw_impl.check_inputs_logic()
    dw_impl.save_input_info()

    classified_ins = dw_impl.do_classify()
    _, _, option_list = classified_ins[0]
    options = option_list[0].get("options")  # {"compute_template": ct}
    dw_impl.define_vars()
    return dw_impl.do_compute([x, out_backprop], options)


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.OPTION_ATTR_LIST_INT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_STR,
    para_check.KERNEL_NAME,
)
def conv2d_backprop_filter_d(
        x,
        out_backprop,
        y,
        filter_size,
        strides,
        pads,
        dilations=(1, 1, 1, 1),
        groups=1,
        data_format="NHWC",
        kernel_name="conv2d_backprop_filter",
):
    """
    algorithm: conv2d_backprop_filter

    Parameters
    ----------
    x: dict with keys(ori_shape, ori_format, shape, format, dtype)
        input feature map tensor.

    out_backprop: dict with keys(ori_shape, ori_format, shape, format, dtype)
        input weight tensor.

    y: dict with keys(ori_shape, ori_format, shape, format, dtype)
        output tensor, dtype must be assigned.

    filter_size: tuple/list of 4 integers
        The shape of filter. 4-D with shape [filter_height, filter_width, in_channels,
        out_channels] or [out_channels, filter_height, filter_width, in_channels] or
        [out_channels, in_channel, filter_height, filter_width].

    strides: tuple/list of 2 integers
        filter move stride.

    pads: tuple/list of 4 integers
        [pad_up, pad_down, pad_left, pad_right].

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_filter. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_backprop_filter. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    kernel_name: str
        kernel name. Default to "conv2d_backprop_filter".

    Returns
    -------
    None
    """
    inputs_list = [x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name]
    dw_impl = Conv2dBpFilterStaticImpl(inputs_list)

    # exchange hw for performance
    if dw_impl.need_exchange_hw():
        dw_impl.exchange_hw()

    dw_impl.check_inputs_logic()
    dw_impl.save_input_info()

    classified_ins = dw_impl.do_classify()
    _, _, option_list = classified_ins[0]
    options = option_list[0].get("options")  # {"compute_template": ct}
    dw_impl.define_vars()
    tensor_list_input = dw_impl.new_placeholder()
    dedw = dw_impl.do_compute(tensor_list_input, options)
    with tvm.target.cce():
        sch = tbe.auto_schedule(dedw)

    if dedw.op.attrs.get("is_dynamic_constantization"):
        log.info("[in dynamic constantization scene] static bin no need to build")
        dw_impl.dynamic_constantization()
        return

    real_outs = sch.cce_special["real_out_tensor"]
    tensor_list = tensor_list_input + real_outs
    dw_impl.do_build(tensor_list, sch)


class Conv2dBpFilterStaticImpl(ConvBpFilterImplBase):

    def __init__(self, inputs_list, fusion_mode=False, options=None) -> None:
        super().__init__(inputs_list, CubeConstantConfig.CONV2D_BACKPROP_FILTER_D_OP_NAME, fusion_mode, options)

    def check_inputs_logic(self, check_format=True):
        checker = CubeChecker(self.op_name)
        checker.check_equal(self.dilations.dilation_n, 1, "dilation_n", "1")
        checker.check_equal(self.dilations.dilation_c, 1, "dilation_c", "1")

        checker.check_multiple(self.fm.fmap_c, self.groups, "fmap's channel", "groups")
        checker.check_multiple(self.grads.grads_c, self.groups, "outbackprop's channel", "groups")

        checker.check_equal(self.fm.fmap_batch, self.grads.grads_batch, "x's N", "out_backprop's N")
        checker.check_equal(self.grads.grads_c, self.kernel.kernel_cout, "out_backprop's C", "Filter's N")
        checker.check_equal(self.fm.fmap_c, self.kernel.kernel_c * self.groups, "x's C", "y's C")

        if check_format:  # ori_graph before insert transdata maybe have different format
            checker.check_equal(self.fm.format, self.grads.format, "x's format", "out_backprop's format")
            if self.fm.format in ("NC1HWC0", ):
                checker.check_equal(
                    list(self.fm.shape),
                    ShapeConverter.convert(self.fm.ori_shape, self.fm.ori_format, "NC1HWC0", self.fm.dtype),
                    "fmap's 5HD shape transed from ori_shape", "fmap's shape")
                checker.check_equal(
                    list(self.grads.shape),
                    ShapeConverter.convert(self.grads.ori_shape, self.grads.ori_format, "NC1HWC0", self.grads.dtype),
                    "out_backprop's 5HD shape transed from ori_shape", "out_backprop's shape")

        kh_dilation = (self.kernel.kernel_h - 1) * self.dilations.dilation_h + 1
        kw_dilation = (self.kernel.kernel_w - 1) * self.dilations.dilation_w + 1
        fmap_w_padding = self.fm.fmap_w + self.pads.pad_l + self.pads.pad_r
        fmap_h_padding = self.fm.fmap_h + self.pads.pad_u + self.pads.pad_d

        if kw_dilation > fmap_w_padding:
            error_manager_cube.raise_err_specific(self.op_name, "kw_dilation should less than fmap_w_padding")
        if kh_dilation > fmap_h_padding:
            error_manager_cube.raise_err_specific(self.op_name, "kh_dilation should less than fmap_h_padding")

        checker.check_equal((fmap_w_padding - kw_dilation) // self.strides.stride_w + 1, self.grads.grads_w,
                            "calc_dedy_w", "dedy_w")
        checker.check_equal((fmap_h_padding - kh_dilation) // self.strides.stride_h + 1, self.grads.grads_h,
                            "calc_dedy_h", "dedy_h")

        # check shape size, 64 bits limitation
        c0_size = tbe_platform.CUBE_MKN.get(self.fm.dtype).get("mac")[1]
        fmap_size = self.fm.fmap_batch * align(self.fm.fmap_c, c0_size) * self.fm.fmap_h * self.fm.fmap_w
        dedy_size = (self.grads.grads_batch * align(self.grads.grads_c, c0_size) * self.grads.grads_h *
                     self.grads.grads_w)
        kernel_size = (align(self.kernel.kernel_cout, c0_size) * align(self.kernel.kernel_c, c0_size) *
                       self.kernel.kernel_h * self.kernel.kernel_h)
        checker.check_64bits_limitation("fmap_size", fmap_size, self.fm.dtype)
        checker.check_64bits_limitation("dedy_size", dedy_size, self.grads.dtype)
        checker.check_64bits_limitation("filter_size", kernel_size, self.kernel.dtype)

    def need_exchange_hw(self):
        return all([
            self.fm.fmap_w == 1, self.kernel.kernel_w == 1, self.grads.grads_w == 1, self.pads.pad_l == 0,
            self.pads.pad_r == 0
        ])

    def exchange_hw(self):
        # exchange h and w will not change date in memory
        def _exchange(shape, shape_format):
            shape = list(shape)
            if shape_format.lower() == "nc1hwc0":
                shape[2], shape[3] = shape[3], shape[2]
            elif set(shape_format.lower()) == set("nchw"):
                id_h = shape_format.lower().find("h")
                id_w = shape_format.lower().find("w")
                shape[id_h], shape[id_w] = shape[id_w], shape[id_h]
            elif shape_format.lower() == "fractal_z":
                pass
            else:
                error_manager_cube.raise_err_specific(self.op_name,
                                                      "do not support this format: {}".format(shape_format))
            return shape

        def _exchage_pads_hw(pads):
            return [pads[2], pads[3], pads[0], pads[1]]

        x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name = self.inputs_list
        x["ori_shape"] = _exchange(x["ori_shape"], x["ori_format"])
        out_backprop["ori_shape"] = _exchange(out_backprop["ori_shape"], out_backprop["ori_format"])
        y["ori_shape"] = _exchange(y["ori_shape"], y["ori_format"])

        x["shape"] = _exchange(x["shape"], x["format"])
        out_backprop["shape"] = _exchange(out_backprop["shape"], out_backprop["format"])
        y["shape"] = _exchange(y["shape"], y["format"])
        strides = _exchange(strides, data_format)
        pads = _exchage_pads_hw(pads)
        dilations = _exchange(dilations, data_format)

        self.inputs_list = [x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name]
        self._new_or_update_self_mem()

    def define_vars(self):
        """
        fake vars. just to creat var_map like dynamic
        """
        var_shape_map = {}
        var_shape_map["fmap_nchw"] = self.fm.get_nchw_shape()
        var_shape_map["dedy_nchw"] = self.grads.get_nchw_shape()
        var_shape_map["dedw_nchw"] = self.kernel.get_nchw_shape()
        var_shape_map["fmap_nc1hwc0"] = ShapeConverter.convert(self.fm.ori_shape, self.fm.ori_format, "NC1HWC0",
                                                               self.fm.dtype)
        var_shape_map["dedy_nc1hwc0"] = ShapeConverter.convert(self.grads.ori_shape, self.grads.ori_format, "NC1HWC0",
                                                               self.grads.dtype)
        var_shape_map["strides"] = self.strides.strides
        var_shape_map["pads"] = self.pads.pads
        var_shape_map["dilations"] = self.dilations.get_nchw_shape()
        var_shape_map["groups"] = self.groups

        self.var_map.update(var_shape_map)

    def format_shape_and_range(self):
        pass

    def dynamic_constantization(self):
        """
        static shape use dynamic process
        """
        x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name = self.inputs_list
        # change attr to input tensor for filter_size
        filter_size = {
            "shape": [4],
            "ori_shape": [4],
            'dtype': "int32",
            "format": 'NCHW',
            "ori_format": 'NCHW',
            "const_value": filter_size
        }
        context = get_op_context()
        context.set_op_mode("dynamic")
        context.add_addition("is_dynamic_constantization", True)

        dyn_impl.conv2d_backprop_filter(x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format,
                                        kernel_name)

    def cal_min_l1space_for_lxfusion(self):
        """
        cal the mini l1space using in lxfusion
        """
        # Waiting for FE support fp32, need to be deleted later
        dtype = "float16"

        kh_dilation = (self.kernel.kernel_h - 1) * self.dilations.dilation_h + 1
        kw_dilation = (self.kernel.kernel_w - 1) * self.dilations.dilation_w + 1
        fmap_h_padding = self.fm.fmap_h + self.pads.pad_u + self.pads.pad_d

        if fmap_h_padding == 1 and kh_dilation == 1 and self.strides.stride_h == 1:
            kl1_min = (CubeConstantConfig.C0_SIZE - 1) * self.strides.stride_w + kw_dilation
        else:
            kl1_min = self.fm.fmap_w

        if self.grads.grads_w % CubeConstantConfig.C0_SIZE == 0:
            bl1_min_byte = kh_dilation * kl1_min * CubeConstantConfig.C0_SIZE * BIT_RATIO_DICT.get(dtype)
        else:
            bl1_min_byte = ((kh_dilation + self.strides.stride_h) * kl1_min * CubeConstantConfig.C0_SIZE *
                            BIT_RATIO_DICT.get(dtype))

        return bl1_min_byte

    def select_format(self):

        def _is_c04():
            if (self.kernel.kernel_h == 1) and (self.kernel.kernel_w == 1):
                return False
            if (self.fm.fmap_batch == 48) and (self.fm.fmap_h == 224):
                return False
            return self.fm.fmap_c <= 4

        if self.fm.ori_format not in ["NCHW", "NHWC"]:
            error_manager_cube.raise_err_input_format_invalid("conv2d_backprop_filter_d", "inputs", ["NCHW", "NHWC"],
                                                              self.fm.ori_format)

        if (not isinstance(self.kernel.ori_shape, (tuple, list))) or len(self.kernel.ori_shape) != 4:
            error_manager_cube.raise_err_should_be_4d("conv2d_backprop_filter_d", "filter_size")

        if self.kernel.ori_format not in ["NCHW", "NHWC", "HWCN"]:
            error_manager_cube.raise_err_input_format_invalid("conv2d_backprop_filter_d", "y", ["NCHW", "NHWC", "HWCN"],
                                                              self.kernel.ori_format)

        #select format by c0_optim_flg
        if _is_c04():
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x",
                                                   datatype="float16, float32, bfloat16, float16, bfloat16",
                                                   format="NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0")
            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="out_backprop",
                                                   datatype="float16, float32, bfloat16, float16, bfloat16",
                                                   format="NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0")
            output0 = util_select_op_base.gen_param(
                classify="output0",
                name="y",
                datatype="float32, float32, float32, float32, float32",
                format="FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z_C04, FRACTAL_Z_C04",
                sub_format="0")
        else:
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x",
                                                   datatype="float16, float32, bfloat16, float16, float32, bfloat16",
                                                   format="NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0")
            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="out_backprop",
                                                   datatype="float16, float32, bfloat16, float16, float32, bfloat16",
                                                   format="NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0")
            output0 = util_select_op_base.gen_param(
                classify="output0",
                name="y",
                datatype="float32, float32, float32, float32, float32, float32",
                format="FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z",
                sub_format="0")
        return [input0, input1, output0]

    def _check_inputs_format(self):
        x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name = self.inputs_list
        ori_shape_x = x.get("ori_shape")
        ori_shape_out_backprop = out_backprop.get("ori_shape")
        ori_shape_res = y.get("ori_shape")

        dtype_x = x.get("dtype").lower()
        dtype_out_backprop = out_backprop.get("dtype").lower()

        checker = CubeChecker(self.op_name)
        checker.check_kernel_name(kernel_name)

        checker.check_type("x", ori_shape_x, (tuple, list))
        checker.check_dims("x", ori_shape_x, CubeConstantConfig.CONV_BACKPROP_SHAPE_DIM)
        checker.check_shape_dims_positive("x", ori_shape_x)

        checker.check_type("out_backprop", ori_shape_out_backprop, (tuple, list))
        checker.check_dims("out_backprop", ori_shape_out_backprop, CubeConstantConfig.CONV_BACKPROP_SHAPE_DIM)
        checker.check_shape_dims_positive("out_backprop", ori_shape_out_backprop)


        checker.check_type("y", ori_shape_res, (tuple, list))
        checker.check_dims("y", ori_shape_res, CubeConstantConfig.CONV_BACKPROP_SHAPE_DIM)
        checker.check_shape_dims_positive("y", ori_shape_res)

        checker.check_type("filter_size", filter_size, (tuple, list))
        checker.check_dims("filter_size", filter_size, CubeConstantConfig.CONV_BACKPROP_SHAPE_DIM)

        checker.check_equal(list(filter_size), list(ori_shape_res), "filter_size", "y")
        checker.check_equal(dtype_x, dtype_out_backprop, "dtype_x", "dtype_out_backprop")

        checker.check_type("dilations", dilations, (tuple, list))
        checker.check_dims("dilations", dilations, CubeConstantConfig.CONV_BACKPROP_SHAPE_DIM)
        checker.check_shape_dims_positive("dilations", dilations)

        checker.check_type("strides", strides, (tuple, list))
        strides = trip_strides(strides, data_format)
        checker.check_dims("strides", strides, CubeConstantConfig.STRIDES_SHAPE_DIM)
        checker.check_shape_dims_positive("strides", strides)

        checker.check_type("pads", pads, (tuple, list))
        checker.check_dims("pads", pads, CubeConstantConfig.CONV_BACKPROP_SHAPE_DIM)
        checker.check_shape_dims_positive("pads", pads, allow_zero=True)

        ori_format_x = x.get("ori_format")
        ori_format_out_backprop = out_backprop.get("ori_format")
        ori_format_res = y.get("ori_format")

        checker.check_format("x", ori_format_x, ("NHWC", "NCHW"))
        checker.check_format("out_backprop", ori_format_out_backprop, ("NHWC", "NCHW"))
        checker.check_format("y", ori_format_res, ("NHWC", "NCHW", "HWCN"))
        checker.check_format("data_format", data_format, ("NHWC", "NCHW"))
        checker.check_equal(ori_format_x, data_format, "fmap_ori_format", "data_format")
        checker.check_equal(ori_format_out_backprop, data_format, "out_backprop_ori_format", "data_format")
