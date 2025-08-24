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
dynamic conv3d_backprop_filter
"""
from __future__ import absolute_import

import warnings
from itertools import product
from impl.util import util_common
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_register
from impl.util.util_cube_dynamic import gen_conv_shape_range
from impl.util.util_cube_dynamic import correct_conv2d_backprop_range_start
from impl.util.util_cube_dynamic import check_tensor_shape
from impl.util.util_cube_dynamic import check_dynamic_range_lower
from impl.util.util_cube_dynamic import is_empty_tensor_scene
from impl.util.util_cube_dynamic import correct_range
from impl.util.util_cube_dynamic import check_binary_flag
from impl.dynamic.conv_bp_filter_impl_base import ConvBpFilterImplBase
from impl.util.util_common import align
from impl.util.util_common import ceil
from tbe.common.utils import log
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.conv_util import CubeConstantConfig
from tbe.common.utils.conv_util import CubeChecker
from tbe.dsl.base.operation import get_op_context
from tbe.dsl.classifier.conv3d_bp_filter_classifier import Conv3dBpFilterClassifier
from tbe.dsl.compute.cube_util import Load3DParam
from tbe.dsl.unify_schedule.conv3d_bp_filter_tilingcase import ATTR_VARS
from tbe.dsl.unify_schedule.conv3d_bp_filter_tilingcase import SHAPE_VARS
from tbe.dsl.unify_schedule.conv3d_bp_filter_tilingcase import TILING_VARS


# the dim of shape in conv_backprop must be 5
_CONV_BACKPROP_SHAPE_DIM = 5
# the dim of strides in conv_backprop must be 3
_STRIDES_SHAPE_DIM = 3
# the dim of pads in conv_backprop must be 6
_PADDING_SHAPE_DIM = 6
# fmapH, fmapW must be in [1,4096]
_FMAP_HW_MAX = 4096
_FMAP_HW_MIN = 1

# DeDy H,W must be in [1,4096]
_DEDY_HW_MAX = 4096
_DEDY_HW_MIN = 1

# filterH, filterW must be in [1,255]
_FILTER_HW_MAX = 255
_FILTER_HW_MIN = 1

# stride must be in [1,63]
_STRIDE_HW_MAX = 63
_STRIDE_HW_MIN = 1

# pad must be in [0,255]
_PAD_MAX = 255
_PAD_MIN = 0

# dilation must be in [1,255]
_DILATION_MIN = 1
_DILATION_MAX = 255

# the max num of each axis of shape
_DEFAULT_MAX_SHAPE_NUM = 1000000
# the minimum dim of shape
_DEFAULT_MIN_SHAPE_DIM = 1
# the max dim of shape
_DEFAULT_MAX_SHAPE_DIM = 1

# the max size is 2**63-1
_DATA_SIZE_MAX = 9223372036854775807

# the bytes length of several dtype
_BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                   "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}

# C0_SIZE
_C0_SIZE = 16
# pads valid mode to be [0, 0, 0, 0]
_PADDING_VAILD = [0, 0, 0, 0, 0, 0]
# If pads is string , only support "SAME" or "VALID"
_PADDING_SUPPORT = ('SAME', 'VALID')
_DYNAMIC_RANK_FLAG = [-2]
_DYNAMIC_SHAPE_VAL = -1

# fuzzy compile constant
FUZZ_MODE_N_MAX = 2147483647
FUZZ_MODE_D_MAX = 4096
FUZZ_MODE_HW_MAX = 4096
FMAP_PARAM_IDX = 0
DEDY_PARAM_IDX = 2
LOEWR_LIMIT_FLAG = "lower_limit"
UPPER_LIMIT_FLAG = "upper_limit"
OP_TYPE = "conv3d_backprop_filter"
UNSUPPORTED_FUZZ_RES = [{"result": "UNSUPPORTED"}]


def _get_pos_from_format(format_in):
    return (format_in.find("N"), format_in.find("D"), format_in.find("H"),
            format_in.find("W"), format_in.find("C"))


def _is_fuzzy_input_valid(tensor_dict):
    """
    Validate input for fuzzy compile.
    """
    is_valid = (
        isinstance(tensor_dict.get("ori_shape"), (list, tuple)) and
        len(tensor_dict.get("ori_shape")) == _CONV_BACKPROP_SHAPE_DIM and
        list(tensor_dict.get("ori_shape")) != _DYNAMIC_RANK_FLAG and
        tensor_dict.get("ori_format") in ("NDHWC", "NCDHW")
        )
    if is_valid and _DYNAMIC_SHAPE_VAL in tensor_dict.get("ori_shape") and tensor_dict.get("ori_range"):
        is_valid &= len(tensor_dict.get("ori_range")) == _CONV_BACKPROP_SHAPE_DIM

    return is_valid


def _is_generalized_range_valid(tensor_dict):
    """
    Helper function for tensor range validation.
    """
    for each_range in tensor_dict.get("ori_range"):
        if each_range[0] > each_range[1]:
            return False
    return True


def _check_tensor_range(tensor_dict, upper_limit_dict):
    """
    Helper function for tensor range validation.
    """
    n_dim, d_dim, h_dim, w_dim, _ = _get_pos_from_format(tensor_dict.get("ori_format"))
    n_dim_max, d_dim_max, h_dim_max, w_dim_max = \
        upper_limit_dict.get("N"), upper_limit_dict.get("D"), upper_limit_dict.get("H"), upper_limit_dict.get("W")

    lower_limit_flag = (
        tensor_dict.get("ori_range")[n_dim][0] > n_dim_max or
        tensor_dict.get("ori_range")[d_dim][0] > d_dim_max or
        tensor_dict.get("ori_range")[h_dim][0] > h_dim_max or
        tensor_dict.get("ori_range")[w_dim][0] > w_dim_max
    )

    if lower_limit_flag:
        warnings.warn("Lower range of N/D/H/W dim exceeds upper limit, please check.")
        return LOEWR_LIMIT_FLAG

    upper_limit_flag = (
        None in list(zip(*tensor_dict.get("ori_range")))[1] or
        -1 in list(zip(*tensor_dict.get("ori_range")))[1] or
        tensor_dict.get("ori_range")[n_dim][1] > n_dim_max or
        tensor_dict.get("ori_range")[d_dim][1] > d_dim_max or
        tensor_dict.get("ori_range")[h_dim][1] > h_dim_max or
        tensor_dict.get("ori_range")[w_dim][1] > w_dim_max
    )

    if upper_limit_flag:
        warnings.warn("Upper range of N/D/H/W dim exceeds upper limit, please check.")
        return UPPER_LIMIT_FLAG

    return ""


def _check_tensor_shape(tensor_dict, upper_limit_dict):
    """
    Helper function for tensor shape check.
    """
    n_dim, d_dim, h_dim, w_dim, _ = _get_pos_from_format(tensor_dict.get("ori_format"))
    n_dim_max, d_dim_max, h_dim_max, w_dim_max = \
        upper_limit_dict.get("N"), upper_limit_dict.get("D"), upper_limit_dict.get("H"), upper_limit_dict.get("W")
    shape_invalid = lambda shape_val, upper_limit: shape_val >= upper_limit or shape_val <= 0
    return (
        shape_invalid(tensor_dict.get("ori_shape")[n_dim], n_dim_max) or
        shape_invalid(tensor_dict.get("ori_shape")[d_dim], d_dim_max) or
        shape_invalid(tensor_dict.get("ori_shape")[h_dim], h_dim_max) or
        shape_invalid(tensor_dict.get("ori_shape")[w_dim], w_dim_max)
    )


def _generalize_tensor_shape(tensor_dict):
    """
    Helper function for tensor shape generalization.
    """
    general_shape = [-1] * len(tensor_dict.get("ori_shape"))
    c_dim = tensor_dict.get("ori_format").find("C")
    general_shape[c_dim] = tensor_dict.get("ori_shape")[c_dim]

    return general_shape


def _get_bl1_max_load_width(dedw_dict, strides, dilations, data_format):
    """
    Calculate max load width of fmap in L1, considering L1 size.
    """
    dedw_shape = dedw_dict.get("ori_shape")
    h_dim = data_format.find("H")
    stride_h = strides[h_dim]
    dilation_h = dilations[h_dim]
    filter_h_dilation = (dedw_shape[h_dim] - 1) * dilation_h + 1

    l1_size = tbe_platform.get_soc_spec("L1_SIZE")
    al1_min_byte = _C0_SIZE * _C0_SIZE * 2
    bl1_max_byte = l1_size - al1_min_byte
    bl1_w_max = bl1_max_byte // ((filter_h_dilation + stride_h) * _C0_SIZE * 2)

    return bl1_w_max


def _correct_fmap_w_range(fmap_dict, dedw_dict, attr_param, load3d_constraints):
    """
    Correct w range of fmap. Each dim is supposed to be within [kernel_dilated - pad, upper_limit].
    """
    strides, pads, dilations, data_format = attr_param
    _, _, _, _, pad_left, pad_right = pads
    kernel_w = dedw_dict.get("ori_shape")[dedw_dict.get("ori_format").find("W")]
    w_lower, w_upper = fmap_dict.get("ori_range")[data_format.find("W")]

    # correct w_lower such that dedy_w_lower >= 2
    if -1 in (pad_left, pad_right):
        w_lower_min = strides[data_format.find("W")] + 1
    else:
        w_lower_min = (strides[data_format.find("W")] - pad_left - pad_right +
                       (kernel_w - 1) * dilations[data_format.find("W")] + 1)

    bl1_max_width = _get_bl1_max_load_width(dedw_dict, strides, dilations, data_format)
    if load3d_constraints == "1":
        w_lower = max(w_lower_min, w_lower)
    w_upper = min(bl1_max_width, w_upper)
    fmap_dict["ori_range"][data_format.find("W")] = [w_lower, w_upper]

    return fmap_dict


def _get_output(x_in, k_size, pads, stride, dilation):
    if not x_in:
        return None
    return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1


def _generalization_ori_range(dw_impl, inputs_list):
    result = []
    x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name = inputs_list
    dedy_upper_limit_dict = {"N": FUZZ_MODE_N_MAX, "D": FUZZ_MODE_D_MAX, "H": FUZZ_MODE_HW_MAX, "W": FUZZ_MODE_HW_MAX}
    bl1_max_width = _get_bl1_max_load_width(y, strides, dilations, data_format)
    fmap_upper_limit_dict = {"N": FUZZ_MODE_N_MAX, "D": FUZZ_MODE_D_MAX, "H": FUZZ_MODE_HW_MAX,
                            "W": min(bl1_max_width, FUZZ_MODE_HW_MAX)}
    _check_dynamic = lambda tensor: _DYNAMIC_SHAPE_VAL in tensor.get("ori_shape")
    dynamic_flag = _check_dynamic(x) or _check_dynamic(out_backprop)
    # now obp not support opc: not support load3d special and have perf issue
    if dynamic_flag and not dw_impl.support_l0c2out:
        result = [[x, filter_size, out_backprop, y, {"strides": strides}, {"pads": pads},
                   {"dilations": dilations}, {"groups": groups}, {"data_format": data_format},
                   {"kernel_name": kernel_name}]]
        fmap_range_status = _check_tensor_range(x, fmap_upper_limit_dict)
        dedy_range_status = _check_tensor_range(out_backprop, dedy_upper_limit_dict)

        param_idx = []
        support_info = []
        if fmap_range_status:
            param_idx.append(FMAP_PARAM_IDX)
            support_info.append(fmap_range_status)
        if dedy_range_status:
            param_idx.append(DEDY_PARAM_IDX)
            support_info.append(dedy_range_status)
        if param_idx:
            result = [{"result": "UNSUPPORTED", "reason": {"param_index": param_idx, "type": support_info}}]

    elif not dynamic_flag:
        fmap_invalid_flag = _check_tensor_shape(x, fmap_upper_limit_dict)
        dedy_invalid_flag = _check_tensor_shape(out_backprop, dedy_upper_limit_dict)

        if fmap_invalid_flag or dedy_invalid_flag:
            return UNSUPPORTED_FUZZ_RES

        # generalize input range
        x = gen_conv_shape_range(x, OP_TYPE)
        out_backprop = gen_conv_shape_range(out_backprop, OP_TYPE)
        # correct fmap range of w dim
        x = correct_conv2d_backprop_range_start(x, y, dilations, pads, data_format)
        attr_param = strides, pads, dilations, data_format
        x = _correct_fmap_w_range(x, y, attr_param, dw_impl.load3d_constraints)
        if not _is_generalized_range_valid(x) or not _is_generalized_range_valid(out_backprop):
            return UNSUPPORTED_FUZZ_RES
        # modify input shape
        x["ori_shape"] = _generalize_tensor_shape(x)
        out_backprop["ori_shape"] = _generalize_tensor_shape(out_backprop)

        if dw_impl.binary_flag and dw_impl.support_l0c2out:
            x["ori_shape"], x["ori_range"] = dw_impl.gen_conv_default_shape_range(x.get("ori_format"),
                                                                                  x.get("ori_shape"))
            out_backprop["ori_shape"], out_backprop["ori_range"] = dw_impl.gen_conv_default_shape_range(
                out_backprop.get("ori_format"), out_backprop.get("ori_shape"))

        result.append([x, filter_size, out_backprop, y, {"strides": strides}, {"pads": pads},
                    {"dilations": dilations}, {"groups": groups}, {"data_format": data_format},
                    {"kernel_name": kernel_name}])
    return result


@tbe_register.register_param_generalization("Conv3DBackpropFilter")
def conv3d_backprop_filter_generalization(x, filter_size, out_backprop, y, strides, pads,
                                          dilations=(1, 1, 1, 1, 1), groups=1,
                                          data_format='NDHWC',
                                          kernel_name="conv3d_backprop_filter",
                                          generalize_config=None):
    """
    algorithm: conv3d_backprop_filter generalization api.

    Parameters
    ----------
    x: dict with keys(shape, dtype and range), input feature map tensor

    filter_size: dict, will not be used

    out_backprop: dict with keys(shape and dtype), out_backprop tensor

    y: dict with keys(shape and dtype), output tensor, dtype must be assigned

    strides: tuple/list of 5 integers, filter move stride

    pads: tuple/list of 6 integers, [pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 5 integers, filter expand size of dilated conv3d_backprop_filter

    groups: int, The number of filter's group. Default value is 1.

    data_format: str
    An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC".
    Specify the data format of the input and output data.

    kernel_name: str, kernel name, default value is "conv3d_backprop_filter"

    generalize_config: dict
    An optional dict used for specify the operation mode, support "keep_rank" for now.

    Returns
    -------
    Param list.
    """

    def reset_dtype_bf162fp16(input_list):
        for input_dict in input_list:
            if input_dict["dtype"] == "bfloat16":
                input_dict["dtype"] = "float16"

    inputs_list = [x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name]
    dw_impl = Conv3dBackpropFilterImpl(inputs_list)
    support_mode = ["keep_rank", "all_shape"] if dw_impl.support_l0c2out else ["keep_rank"]
    if generalize_config.get("mode") not in support_mode:
        warnings.warn("Only support keep_rank in generalize_config")
        return

    if not _is_fuzzy_input_valid(x) or not _is_fuzzy_input_valid(out_backprop):
        return UNSUPPORTED_FUZZ_RES

    result = []
    if generalize_config.get("mode") == "keep_rank":
        result = _generalization_ori_range(dw_impl, inputs_list)
    else:
        x["shape"], x["range"] = dw_impl.gen_conv_default_shape_range(x.get("format"), x.get("shape"))
        out_backprop["shape"], out_backprop["range"] = dw_impl.gen_conv_default_shape_range(
            out_backprop.get("format"), out_backprop.get("shape"))
        y["shape"], y["range"] = dw_impl.gen_conv_default_shape_range(y.get("format"), y.get("shape"))
        strides = [-1, -1, -1, -1, -1]
        pads = [-1, -1, -1, -1, -1, -1]
        dilations = [-1, -1, -1, -1, -1]
        groups = -1
        # change formt to ensure reuse
        x["ori_format"] = "NCDHW"
        filter_size["ori_format"] = "NCDHW"
        out_backprop["ori_format"] = "NCDHW"
        y["ori_format"] = "NCDHW"
        filter_size["format"] = "NCDHW"
        data_format = "NCDHW"
        reset_dtype_bf162fp16([x, out_backprop, y])  # for reuse float16 binary
        result.append([x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format])
    return result


@register_operator('Conv3DBackpropFilter')
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
def conv3d_backprop_filter(x, filter_size, out_backprop, y, strides, pads,
                           dilations=(1, 1, 1, 1, 1), groups=1,
                           data_format='NDHWC',
                           kernel_name="conv3d_backprop_filter"):
    """
    algorithm: conv3d_backprop_filter

    Parameters
    ----------
    x: dict with keys(shape, dtype and range), input feature map tensor

    filter_size: dict, will not be used

    out_backprop: dict with keys(shape and dtype), out_backprop tensor

    y: dict with keys(shape and dtype), output tensor, dtype must be assigned

    strides: tuple/list of 5 integers, filter move stride

    pads: tuple/list of 5 integers, [pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 5 integers, filter expand size of dilated conv3d_backprop_filter

    groups: int, The number of filter's group. Default value is 1.

    data_format: str
    An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC".
    Specify the data format of the input and output data.

    kernel_name: str, kernel name, default value is "conv3d_backprop_filter"

    Returns
    -------
    None
    """

    inputs_list = [x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name]
    dw_impl = Conv3dBackpropFilterImpl(inputs_list)
    dw_impl.format_shape_and_range()
    if not dw_impl.binary_flag:
        dw_impl.check_inputs_logic()

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
                sch = tbe.auto_schedule([dedw])

            tensor_list.append(tensor_list_input + [dedw])
            sch_list.append(sch)
    log.debug("sch_list num: {}, sch num: {}".format(len(sch_list), len(sch_list[0])))
    dw_impl.build(tensor_list, sch_list)


class Conv3dBackpropFilterImpl(ConvBpFilterImplBase):

    def __init__(self, inputs_list: list, fusion_mode=False, options=None) -> None:
        """
        inputs are same for static and dynamic
        inputs_list: x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name
        """
        super().__init__(inputs_list, CubeConstantConfig.CONV3D_BACKPROP_FILTER_OP_NAME, fusion_mode, options)
        self.inputs_list = inputs_list
        self.op_name = CubeConstantConfig.CONV3D_BACKPROP_FILTER_OP_NAME
        self.var_map = {}
        self.support_l0c2out = tbe_platform_info.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
        self.load3d_constraints = tbe_platform_info.get_soc_spec("load3d_constraints")
        self.binary_flag = self.get_binary_flag()

    @staticmethod
    def _calc_range(fmap_range, stride, pad_list, weight, dilation):
        out_upper = None
        filter_dilation = (weight - 1) * dilation + 1
        if CubeConstantConfig.DYNAMIC_FLAG in pad_list:
            out_lower = util_common.ceil(fmap_range[0], stride)
            if fmap_range[1]:
                out_upper = util_common.ceil(fmap_range[1], stride)
        else:
            out_lower = _get_output(fmap_range[0], weight, pad_list, stride, dilation)
            if out_lower < 1:
                fmap_range_lower = min(filter_dilation, fmap_range[1]) if fmap_range[1] else filter_dilation
                fmap_range = (fmap_range_lower, fmap_range[1])
                out_lower = _get_output(fmap_range[0], weight, pad_list, stride, dilation)
                warnings.warn("feature map range has been corrected due to invalid output shape")
            if fmap_range[1]:
                out_upper = _get_output(fmap_range[1], weight, pad_list, stride, dilation)
        return [out_lower, out_upper], fmap_range

    def get_binary_flag(self):
        """
        confirm whether it is in binary mode
        """
        if self.support_l0c2out:
            return True
        return check_binary_flag(self.inputs_list[0].get("ori_range"), self.inputs_list[3].get("ori_shape"))

    def correct_input_batch(self):
        x, _, out_backprop, *_ = self.inputs_list
        if self.fm.fmap_batch != self.grads.grads_batch:
            if self.fm.fmap_batch != -1 and self.grads.grads_batch == -1:
                out_backprop["shape"][0] = x["shape"][0]
                out_backprop["ori_shape"][0] = x["ori_shape"][0]
            elif self.fm.fmap_batch == -1 and self.grads.grads_batch != -1:
                x["shape"][0] = out_backprop["shape"][0]
                x["ori_shape"][0] = out_backprop["ori_shape"][0]
            else:
                error_manager_cube.raise_err_specific_user("conv3d_backprop_filter", "Fmap's N not equal to Dedy's N")

        for value in self.fm.get_ncdhw_shape():
            if value < 1 and value != -1:
                error_manager_cube.raise_err_specific_user("conv3d_backprop_filter",
                                                           "Each dimension of fmap has to be -1 or positive integer.")
        if self.fm.get_ncdhw_shape()[1] == -1:
            error_manager_cube.raise_err_specific_user(
                "conv3d_backprop_filter", "dynamic c dimension is not supported yet.")
        self._new_or_update_self_mem()

    def check_value_range(self, lower_bound, upper_bound, lower_bound_dedy, upper_bound_dedy):
        checker = CubeChecker(self.op_name)
        checker.check_value_range("dilations's H", self.dilations.dilation_h, Load3DParam.dilation_min(),
                                  Load3DParam.dilation_max())
        checker.check_value_range("dilations's W", self.dilations.dilation_w, Load3DParam.dilation_min(),
                                  Load3DParam.dilation_max())
        checker.check_value_range("dilations's D", self.dilations.dilation_d, Load3DParam.dilation_min(),
                                  Load3DParam.dilation_max())
        checker.check_equal(self.dilations.dilation_n, 1, "dilation_n", "1")
        checker.check_equal(self.dilations.dilation_c, 1, "dilation_c", "1")

        filter_d_dilation = (self.kernel.kernel_d - 1) * self.dilations.dilation_d + 1
        filter_h_dilation = (self.kernel.kernel_h - 1) * self.dilations.dilation_h + 1
        filter_w_dilation = (self.kernel.kernel_w - 1) * self.dilations.dilation_w + 1

        if self.fm.fmap_d > 0 and -1 not in [self.pads.pad_f, self.pads.pad_b]:
            checker.check_equal(
                (self.fm.fmap_d - filter_d_dilation + self.pads.pad_f + self.pads.pad_b) // self.strides.stride_d + 1,
                self.grads.grads_d, "calc_dedy_d", "dedy_d")
        if self.fm.fmap_w > 0 and -1 not in [self.pads.pad_l, self.pads.pad_r]:
            checker.check_equal(
                (self.fm.fmap_w - filter_w_dilation + self.pads.pad_l + self.pads.pad_r) // self.strides.stride_w + 1,
                self.grads.grads_w, "calc_dedy_w", "dedy_w")
        if self.fm.fmap_h > 0 and -1 not in [self.pads.pad_u, self.pads.pad_d]:
            checker.check_equal(
                (self.fm.fmap_h - filter_h_dilation + self.pads.pad_u + self.pads.pad_d) // self.strides.stride_h + 1,
                self.grads.grads_h, "calc_dedy_h", "dedy_h")
        # special cases
        fmap_hw_max = _FMAP_HW_MAX
        fmap_h_min, fmap_w_min = _FMAP_HW_MIN, _FMAP_HW_MIN
        fmap_d_min = _FMAP_HW_MIN
        dedy_hw_max = _DEDY_HW_MAX
        dedy_hw_min = _DEDY_HW_MIN

        _, _, lower_fmap_d, lower_fmap_h, lower_fmap_w = lower_bound
        upper_fmap_n, upper_fmap_c, upper_fmap_d, upper_fmap_h, upper_fmap_w = upper_bound
        _, _, _, lower_dedy_h, lower_dedy_w = lower_bound_dedy
        _, _, upper_dedy_d, upper_dedy_h, upper_dedy_w = upper_bound_dedy
        # Fmap value limit
        checker.check_value_range("Fmap's minH", lower_fmap_h, fmap_h_min, fmap_hw_max)
        checker.check_value_range("Fmap's minW", lower_fmap_w, fmap_w_min, fmap_hw_max)
        checker.check_value_range("Fmap's minD", lower_fmap_d, fmap_d_min, fmap_hw_max)
        checker.check_value_range("Fmap's maxH", upper_fmap_h, fmap_h_min, fmap_hw_max)
        checker.check_value_range("Fmap's maxW", upper_fmap_w, fmap_w_min, fmap_hw_max)
        checker.check_value_range("Fmap's maxD", upper_fmap_d, fmap_d_min, fmap_hw_max)

        # Dedy value limit
        checker.check_value_range("Dedy's minH inferenced from Fmap's minH", lower_dedy_h, dedy_hw_min, dedy_hw_max)
        checker.check_value_range("Dedy's minW inferenced from Fmap's minW", lower_dedy_w, dedy_hw_min, dedy_hw_max)
        checker.check_value_range("Dedy's maxH inferenced from Fmap's maxH", upper_dedy_h, dedy_hw_min, dedy_hw_max)
        checker.check_value_range("Dedy's maxW inferenced from Fmap's maxW", upper_dedy_w, dedy_hw_min, dedy_hw_max)

        # stride value limit
        checker.check_value_range("stride's H", self.strides.stride_h, _STRIDE_HW_MIN, _STRIDE_HW_MAX)
        checker.check_value_range("stride's W", self.strides.stride_w, _STRIDE_HW_MIN, _STRIDE_HW_MAX)
        checker.check_value_range("stride's D", self.strides.stride_d, _STRIDE_HW_MIN, _STRIDE_HW_MAX)

    def min_l1_byte(self, upper_dedy_w, upper_fmap_w):
        if not upper_dedy_w or not upper_fmap_w:
            return
        # Forth : L1 limitation, Mainly required by chip
        filter_h_dilation = (self.kernel.kernel_h - 1) * self.dilations.dilation_h + 1
        al1_min_byte = _C0_SIZE * _C0_SIZE * 2
        if upper_dedy_w % _C0_SIZE == 0:
            bl1_min_byte = filter_h_dilation * upper_fmap_w * _C0_SIZE * 2
        else:
            bl1_min_byte = (filter_h_dilation + self.strides.stride_h) * upper_fmap_w * _C0_SIZE * 2
        l1_size = tbe_platform.get_soc_spec("L1_SIZE") # L1 size
        if (al1_min_byte + bl1_min_byte) > l1_size:
            error_manager_cube.raise_err_specific_user("conv3d_backprop_filter",
                                                    "for this input shape range, the minimum tiling may exceed \
                                                        L1_Buffer, please lower the upper_bound of fmap_w and retry")

    def check_inputs_logic(self):
        x, _, out_backprop, _, _, pads, *_ = self.inputs_list
        checker = CubeChecker(self.op_name)

        fmap_range = x.get("ori_range")
        dedy_range = out_backprop.get("ori_range")
        lower_bound, upper_bound = zip(*fmap_range)
        lower_bound_dedy, upper_bound_dedy = zip(*dedy_range)

        upper_fmap_n, upper_fmap_c, upper_fmap_d, upper_fmap_h, upper_fmap_w = upper_bound
        _, _, _, lower_dedy_h, lower_dedy_w = lower_bound_dedy
        _, _, upper_dedy_d, upper_dedy_h, upper_dedy_w = upper_bound_dedy

        self.correct_input_batch()
        self.check_value_range(lower_bound, upper_bound, lower_bound_dedy, upper_bound_dedy)
        self.min_l1_byte(upper_dedy_w, upper_fmap_w)
        checker.check_dims("fmap_lower_bound", lower_bound, CubeConstantConfig.CONV3D_BACKPROP_SHAPE_DIM)
        if None not in upper_bound:
            checker.check_dims("fmap_upper_bound", upper_bound, CubeConstantConfig.CONV3D_BACKPROP_SHAPE_DIM)

        # special cases
        upper_dy_flag = upper_dedy_w and upper_dedy_h
        load3d_special_flag = not self.support_l0c2out and \
            upper_dy_flag and lower_dedy_w <= 1 and upper_dedy_w >= 1 and upper_dedy_h > 1
        if (load3d_special_flag):
            # Chip Design demand dedy_w must >=2 when dedy_h != 1
            error_manager_cube.raise_err_specific_user("conv3d_backprop_filter",
                                                       "Chip Design demand dedy_w must >=2 when dedy_h != 1.")


        c0_size = tbe_platform.C0_SIZE
        if upper_fmap_n and upper_fmap_d and upper_fmap_h and upper_fmap_w:
            upper_fmap_size = (upper_fmap_n * align(upper_fmap_c, c0_size) *
                               upper_fmap_d * upper_fmap_h * upper_fmap_w)
            checker.check_64bits_limitation("fmap_size", upper_fmap_size, self.fm.dtype)
            if CubeConstantConfig.DYNAMIC_FLAG not in pads:
                upper_dedy_size = (upper_fmap_n * align(self.kernel.kernel_cout, c0_size) * \
                                   upper_dedy_d * upper_dedy_h * upper_dedy_w)
                checker.check_64bits_limitation("dedy_size", upper_dedy_size, self.grads.dtype)

        kernel_size = (align(self.kernel.kernel_cout, c0_size) * align(self.kernel.kernel_c, c0_size) *
                       self.kernel.kernel_d * self.kernel.kernel_h * self.kernel.kernel_h)
        checker.check_64bits_limitation("filter_size", kernel_size, self.kernel.dtype)

        self._new_or_update_self_mem()

    def get_pads_attr(self):
        def _convert_shape_to_list(shape):
            for i, var in enumerate(shape):
                if isinstance(var, tvm.tir.IntImm):
                    shape[i] = var.value

        # get pads
        *_, pads, _, _, _, _ = self.inputs_list
        filter_d_dilation = (self.kernel.kernel_d - 1) * self.dilations.dilation_d + 1
        filter_h_dilation = (self.kernel.kernel_h - 1) * self.dilations.dilation_h + 1
        filter_w_dilation = (self.kernel.kernel_w - 1) * self.dilations.dilation_w + 1

        pad_front, pad_back, pad_up, pad_down, pad_left, pad_right = pads
        pad_front, pad_back = self._update_pads([pad_front, pad_back], self.var_map.get("fmap_ncdhw")[2],
                                                self.strides.stride_d, filter_d_dilation)
        pad_up, pad_down = self._update_pads([pad_up, pad_down], self.var_map.get("fmap_ncdhw")[3],
                                             self.strides.stride_h, filter_h_dilation)
        pad_left, pad_right = self._update_pads([pad_left, pad_right], self.var_map.get("fmap_ncdhw")[4],
                                                self.strides.stride_w, filter_w_dilation)

        pads = [pad_front, pad_back, pad_up, pad_down, pad_left, pad_right]
        _convert_shape_to_list(pads)
        self.var_map["pads"] = pads
        self._new_or_update_self_mem()

    def get_ncdhw_shape(self, shape, ori_format):
        pos_n, pos_d, pos_h, pos_w, pos_c = _get_pos_from_format(ori_format)
        return [shape[pos_n], shape[pos_c], shape[pos_d], shape[pos_h], shape[pos_w]]

    def get_ncdhw_range(self, shape_range):
        # fmap format is NC1HWC0
        return [shape_range[0], [self.fm.fmap_c, self.fm.fmap_c], shape_range[1], shape_range[3], shape_range[4]]

    def get_fmap_ncdhw_range(self):
        x, *_ = self.inputs_list
        if list(x.get("ori_shape")) == CubeConstantConfig.DYNAMIC_RANK_SHAPE:
            x_shape_ncdhw = [-1, self.kernel.kernel_c, -1, -1, -1]
            x_range_ncdhw = [(1, None), (self.kernel.kernel_c, self.kernel.kernel_c), (1, None), (1, None), (1, None)]
        else:
            x_shape_ncdhw = self.get_ncdhw_shape(x.get("ori_shape"), x.get("ori_format"))
            x_range_ncdhw = self.get_ncdhw_range(x.get("range"))

            for i, r in enumerate(x_range_ncdhw):
                if x_shape_ncdhw[i] > 0:
                    x_range_ncdhw[i] = (x_shape_ncdhw[i], x_shape_ncdhw[i])

                if r[1] and r[0] > r[1]:
                    error_manager_cube.raise_err_specific_user("conv3d_backprop_filter",
                                                               "range lower bound should less equal than upper bound")
        if x_shape_ncdhw[1] == 0:
            error_manager_cube.raise_err_specific_user("conv3d_backprop_filter", "fmap_c not support 0")

        return x_range_ncdhw

    def format_shape_and_range(self):
        if not self.binary_flag:
            self._format_fmap_empty_tensor()
        x, _, out_backprop, y, _, _, _, _, _, _ = self.inputs_list
        tensor_list = [x, out_backprop, y]
        for input_tensor, is_origin in list(product(tensor_list, [True, False])):
            self._format_unknown_rank_shape(input_tensor, is_origin)
            self._format_empty_tensor(input_tensor, is_origin)
            self._format_range(input_tensor, is_origin)
        if not self.binary_flag:
            self._format_c_shape()
            self._range_correction()
        self._new_or_update_self_mem()

    def define_binary_mode_vars(self):
        '''
        variablization
        '''

        def _define_optional_vars(var_name):
            if operation.get_te_var(var_name) is None:
                return operation.var(var_name)
            return operation.get_te_var(var_name).get_tvm_var()

        shape_var_map = {}
        attr_var_map = {}
        tiling_var_map = {}

        for var in SHAPE_VARS:
            shape_var_map[var] = _define_optional_vars(var)

        for var in ATTR_VARS:
            attr_var_map[var] = operation.var(var)

        for var in TILING_VARS:
            tiling_var_map[var] = operation.var(var)

        c0_size = tbe_platform.CUBE_MKN.get(self.fm.dtype).get("mac")[1]
        w_block_m = tbe_platform.CUBE_MKN.get(self.fm.dtype).get("mac")[0]
        var_shape_map = {}
        real_g = attr_var_map.get("real_g")
        cin1_g = attr_var_map.get("cin1_g")
        cout1_g = attr_var_map.get("cout1_g")
        mag_factor = attr_var_map.get("mag_factor")
        var_shape_map["fmap_ncdhw"] = (shape_var_map.get("batch_n"), shape_var_map.get("fmap_c"),
                                       shape_var_map.get("fmap_d"),
                                       shape_var_map.get("fmap_h"), shape_var_map.get("fmap_w"))
        var_shape_map["dedy_ncdhw"] = (shape_var_map.get("batch_n"), shape_var_map.get("dedy_c"),
                                       shape_var_map.get("dedy_d"),
                                       shape_var_map.get("dedy_h"), shape_var_map.get("dedy_w"))
        var_shape_map["dedw_ncdhw"] = (shape_var_map.get("dedy_c"), shape_var_map.get("fmap_c"),
                                       attr_var_map.get("kernel_d"),
                                       attr_var_map.get("kernel_h"), attr_var_map.get("kernel_w"))
        var_shape_map["fmap_ndc1hwc0"] = (shape_var_map.get("batch_n"), shape_var_map.get("fmap_d"),
                                          attr_var_map.get("fmap_c1"),
                                          shape_var_map.get("fmap_h"), shape_var_map.get("fmap_w"), c0_size)
        var_shape_map["dedy_ndc1hwc0"] = (shape_var_map.get("batch_n"), shape_var_map.get("dedy_d"),
                                          attr_var_map.get("dedy_c1"),
                                          shape_var_map.get("dedy_h"), shape_var_map.get("dedy_w"), c0_size)
        var_shape_map["strides"] = (attr_var_map.get("stride_d"),
                                    attr_var_map.get("stride_h"), attr_var_map.get("stride_w"))
        var_shape_map["pads"] = (attr_var_map.get("padf"), attr_var_map.get("padb"), attr_var_map.get("padu"),
                                 attr_var_map.get("padd"), attr_var_map.get("padl"), attr_var_map.get("padr"))
        var_shape_map["dilations"] = (1, 1, attr_var_map.get("dilation_d"), attr_var_map.get("dilation_h"),
                                      attr_var_map.get("dilation_w"))
        var_shape_map["groups"] = -1
        var_shape_map["group_dict"] = {"real_g": real_g, "mag_factor": mag_factor, "cin1_g": cin1_g,
                                       "cout_g": cout1_g * w_block_m, "cin_ori": shape_var_map.get("fmap_c"),
                                       "cout_ori": shape_var_map.get("dedy_c")}

        self.var_map = var_shape_map

    def define_dynamic_mode_vars(self):
        x, _, _, _, _, _, _, groups, *_ = self.inputs_list
        fmap_range = x.get("ori_range")
        var_shape_map = {}
        c0_size = tbe_platform.C0_SIZE
        cin0 = tbe_platform.CUBE_MKN[self.fm.dtype]['mac'][2]
        cout0 = tbe_platform.CUBE_MKN[self.fm.dtype]['mac'][0]

        fmap_n, fmap_d, fmap_h, fmap_w = self.fm.fmap_batch, self.fm.fmap_d, self.fm.fmap_h, self.fm.fmap_w
        dedy_d, dedy_h, dedy_w = self.grads.grads_d, self.grads.grads_h, self.grads.grads_w
        if self.fm.fmap_batch == -1:
            fmap_n = operation.var("batch_n", bound=fmap_range[0])
            operation.add_exclude_bound_var(fmap_n)

        if self.fm.fmap_d == -1:
            fmap_d = operation.var("fmap_d", bound=fmap_range[2])
            dedy_d = operation.var("dedy_d")
            operation.add_exclude_bound_var(fmap_d)
            operation.add_exclude_bound_var(dedy_d)

        if self.fm.fmap_h == -1:
            fmap_h = operation.var("fmap_h", bound=fmap_range[3])
            dedy_h = operation.var("dedy_h")
            operation.add_exclude_bound_var(fmap_h)
            operation.add_exclude_bound_var(dedy_h)

        if self.fm.fmap_w == -1:
            fmap_w = operation.var("fmap_w", bound=fmap_range[4])
            dedy_w = operation.var("dedy_w")
            operation.add_exclude_bound_var(fmap_w)
            operation.add_exclude_bound_var(dedy_w)

        group_dict = util_common.calculate_group(self.fm.fmap_c, self.grads.grads_c, groups, cout0, cin0)
        var_shape_map["fmap_ncdhw"] = (fmap_n, self.fm.fmap_c, fmap_d, fmap_h, fmap_w)
        var_shape_map["dedy_ncdhw"] = (fmap_n, self.grads.grads_c, dedy_d, dedy_h, dedy_w)
        var_shape_map["fmap_ndc1hwc0"] = (fmap_n, fmap_d, ceil(self.fm.fmap_c, c0_size), fmap_h, fmap_w, c0_size)
        var_shape_map["dedy_ndc1hwc0"] = (fmap_n, dedy_d, ceil(self.grads.grads_c, c0_size), dedy_h, dedy_w, c0_size)
        var_shape_map["group_dict"] = group_dict
        var_shape_map["strides"] = self.strides.get_ncdhw_shape()[2:]
        var_shape_map["dilations"] = self.dilations.get_ncdhw_shape()
        var_shape_map["dedw_ncdhw"] = self.kernel.get_ncdhw_shape()
        self.var_map = var_shape_map
        self.get_pads_attr()

    def define_vars(self):
        if self.binary_flag:
            self.define_binary_mode_vars()
        else:
            self.define_dynamic_mode_vars()

    def do_classify(self):
        x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name = self.inputs_list
        input_list = [x, filter_size, out_backprop, y]
        attr_list = [strides, pads, dilations, groups, data_format, kernel_name]
        option_list = []
        extra_parameters = {"single_op_flag": True}
        ins = (input_list, attr_list, option_list)
        classifier = Conv3dBpFilterClassifier(ins, self.op_name, False, extra_parameters)
        return classifier.classify()

    def new_placeholder(self):
        fmap_shape = self.var_map.get("fmap_ndc1hwc0")
        dedy_shape = self.var_map.get("dedy_ndc1hwc0")

        fmap = tvm.placeholder(fmap_shape, name="fmap", dtype=self.fm.dtype)
        filter_size = tvm.placeholder([5], name="filter_size", dtype="int32")
        dedy = tvm.placeholder(dedy_shape, name="dedy", dtype=self.grads.dtype)
        if get_op_context().get_addition("is_dynamic_constantization") is True:
            return [fmap, dedy]
        return [fmap, filter_size, dedy]

    def do_compute(self, tensor_list, options):
        if len(tensor_list) == 2:
            fmap_tensor, dedy_tensor = tensor_list
        else:
            fmap_tensor, _, dedy_tensor = tensor_list

        para_dict = {
            "strides": self.var_map.get("strides"),
            "pads": self.var_map.get("pads"),
            "dilations": self.var_map.get("dilations"),
            "group_dict": self.var_map.get("group_dict"),
            "res_dtype": self.kernel.dtype,
            "kernel_name": self.kernel_name,
            "binary_flag": self.binary_flag,
            "fmap_format_in_gm": self.fm.format,
        }
        if options:
            para_dict.update(options)
            log.debug("[ComputeTemplate] {}".format(options.get("compute_template").get_debug_info()))
        log.debug("[{}] compute param para_dict: {}".format(self.op_name, para_dict))
        return tbe.conv3d_backprop_filter(x=fmap_tensor,
                                          out_backprop=dedy_tensor,
                                          filter_size=self.var_map.get("dedw_ncdhw"),
                                          para_dict=para_dict)

    def build(self, tensor_list, sch_list):
        config = {
            'print_ir': False,
            'name': self.kernel_name,
            'tensor_list': tensor_list,
            'build_args': {
                'constant_realize_extent_in_infer_bound': False,
            }
        }

        if get_op_context().get_addition("is_dynamic_constantization") is True:
            get_op_context().set_op_mode("static")

        log.debug("[{}] build start. kernel_name = {}".format(self.op_name, self.kernel_name))
        tbe.build(sch_list, config)
        log.debug("[{}] build end. kernel_name = {}".format(self.op_name, self.kernel_name))

    def _check_inputs_format(self):
        x, _, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name = self.inputs_list
        ori_shape_x = x.get("ori_shape")
        ori_shape_out_backprop = out_backprop.get("ori_shape")
        ori_shape_res = y.get("ori_shape")

        dtype_x = x.get("dtype").lower()
        dtype_out_backprop = out_backprop.get("dtype").lower()
        dtype_y = y.get("dtype").lower()

        checker = CubeChecker(self.op_name)
        checker.check_kernel_name(kernel_name)
        checker.check_type("x", ori_shape_x, (tuple, list))
        checker.check_type("out_backprop", ori_shape_out_backprop, (tuple, list))
        checker.check_type("y", ori_shape_res, (tuple, list))
        checker.check_type("dilations", dilations, (tuple, list))
        checker.check_type("strides", strides, (tuple, list))
        checker.check_type("pads", pads, (tuple, list))
        checker.check_equal(dtype_x, dtype_out_backprop, "dtype_x", "dtype_out_backprop")
        para_check.check_dtype_rule(dtype_x, ("float16", "bfloat16"), "x")
        para_check.check_dtype_rule(dtype_out_backprop, ("float16", "bfloat16"), "out_backprop")
        para_check.check_dtype_rule(dtype_y, ("float16", "float32"), "output")

        ori_format_x = x.get("ori_format")
        ori_format_out_backprop = out_backprop.get("ori_format")
        ori_format_res = y.get("ori_format")
        checker.check_format("x", ori_format_x, ("NDHWC", "NCDHW"))
        checker.check_format("out_backprop", ori_format_out_backprop, ("NDHWC", "NCDHW"))
        checker.check_format("y", ori_format_res, ("NDHWC", "NCDHW", "DHWCN"))
        checker.check_format("data_format", data_format, ("NDHWC", "NCDHW"))
        checker.check_equal(ori_format_x, data_format, "fomat_x", "data_format")
        checker.check_equal(ori_format_out_backprop, data_format, "format_dedy", "data_format")

        # check dimension
        checker.check_dims("strides", strides, CubeConstantConfig.CONV3D_BACKPROP_SHAPE_DIM)
        checker.check_dims("dilations", dilations, CubeConstantConfig.CONV3D_BACKPROP_SHAPE_DIM)
        checker.check_dims("pads", pads, CubeConstantConfig.CONV3D_BACKPROP_PAD_DIM)

    def _update_pads(self, pad_list, fmap, stride, filter_dilation):
        pad_f, pad_b = pad_list
        if -1 in [pad_f, pad_b]:
            if fmap is not None:
                pad_d = util_common.align(fmap, stride) - stride + filter_dilation - fmap
                pad_d = tvm.max(pad_d, 0)
                pad_f = pad_d // 2
                pad_b = pad_d - pad_f
        else:
            if pad_f >= filter_dilation or pad_b >= filter_dilation:
                error_manager_cube.raise_err_specific("conv3d_backprop_filter",
                                                      "k_dilation should less than fmap_padding")
        return pad_f, pad_b

    def _format_fmap_empty_tensor(self):
        x, _, out_backprop, y, _, pads, _, _, _, _ = self.inputs_list
        if check_dynamic_range_lower([x, out_backprop, y]) or is_empty_tensor_scene([x, out_backprop, y]):
            dx_range = self.get_fmap_ncdhw_range()
            if 0 in self.kernel.get_ncdhw_shape()[1:]:
                error_manager_cube.raise_err_specific_user("conv3d_backprop_filter", "weight_cdhw not support 0")
            check_tensor_shape({"tensor": [x, out_backprop, y],
                                "value": [-1, -1, 1],
                                "range": [(1, 1), (1, 1), (1, 1)]})

            if list(x.get("ori_shape")) != CubeConstantConfig.DYNAMIC_RANK_SHAPE:
                correct_range(x, dx_range, self.kernel.get_ncdhw_shape(), self.strides.get_ncdhw_shape(),
                              self.dilations.get_ncdhw_shape(), pads, "NCDHW")

    def _range_correction(self):
        x, _, out_backprop, *_ = self.inputs_list
        dx_range = self.get_fmap_ncdhw_range()
        dedy_d_range, fmap_d_range = self._calc_range(dx_range[2], self.strides.stride_d,
                                                      (self.pads.pad_f, self.pads.pad_b), self.kernel.kernel_d,
                                                      self.dilations.dilation_d)
        dedy_h_range, fmap_h_range = self._calc_range(dx_range[3], self.strides.stride_h,
                                                      (self.pads.pad_u, self.pads.pad_d), self.kernel.kernel_h,
                                                      self.dilations.dilation_h)
        dedy_w_range, fmap_w_range = self._calc_range(dx_range[4], self.strides.stride_w,
                                                      (self.pads.pad_l, self.pads.pad_r), self.kernel.kernel_w,
                                                      self.dilations.dilation_w)

        dedy_range = [dx_range[0], (self.grads.grads_c, self.grads.grads_c), dedy_d_range, dedy_h_range, dedy_w_range]
        dx_range = [dx_range[0], dx_range[1], fmap_d_range, fmap_h_range, fmap_w_range]
        x["ori_range"] = dx_range
        out_backprop["ori_range"] = dedy_range

    def _format_c_shape(self):
        x, _, out_backprop, _, _, _, _, groups, _, _ = self.inputs_list
        _, _, _, _, pos_fmap_c = _get_pos_from_format(x.get("ori_format"))
        _, _, _, _, pos_out_backprop_c = _get_pos_from_format(out_backprop.get("ori_format"))
        fmap_shape, out_shape = x.get("ori_shape"), out_backprop.get("ori_shape")
        fmap_shape[pos_fmap_c], out_shape[pos_out_backprop_c] = self.kernel.kernel_c * groups, self.kernel.kernel_cout
        self._new_or_update_self_mem()
