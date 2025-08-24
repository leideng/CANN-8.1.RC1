#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
fixpipe common function
"""
from typing import List
from typing import Optional
import functools
import tbe
from tbe import tvm
from tbe.tvm import Tensor
from tbe.common.utils import shape_to_list
from tbe.dsl.base.operation import in_dynamic
from te.platform.cce_params import scope_fb0
from te.platform.cce_params import scope_fb1
from te.platform.cce_params import scope_fb2
from te.platform.cce_params import scope_fb3
from te.platform.cce_params import scope_cbuf
from tbe.common.register import set_fusion_buildcfg
from tbe.common.utils import cast
from tbe.common.utils.op_util.op_util_conv2d import support_conv_instruction

DTYPE_TRANS_MAP = {
    "int4": "S4",
    "int8": "B8",
    "float16": "F16",
    "float32": "F32",
    "int32": "S32",
    "bfloat16": "BF16",
    "int16": "S16"
}

ANTI_QUANT_MAP = {
    "int4": "S4",
    "int8": "S8"
}

LUT_MODE_MAP = {
    "Sigmoid": "SIGMOID",
    "Tanh": "TANH",
    "Elu": "ELU"
}

QUANT_SCALE_0_STR = "quant_scale_0"
QUANT_SCALE_1_STR = "quant_scale_1"
RELU_WEIGHT_0_STR = "relu_weight_0"
RELU_WEIGHT_1_STR = "relu_weight_1"
ELTWISE_SRC_STR = "eltwise_src"
RELU_STR = "Relu"

FIXPIPE_OP_TAG = "fixpipe"
FIXPIPE_REFORM_TAG = "fixpipe_reform"
WINOGRAD_POST_TAG = "cub_wino_post"
WINOGRAD_RES_TAG = "conv_wino_res"

PASS_PRE_CONVERT_MODE = ["F322F32", "S322S32"]

PRE_CONVERT_MODE = ["F322F16", "F322B8", "F322S4", "F322BF16", "S322F16", "S322B8", "S322S4", "S322S16",
                    "VF322F16", "VF322B8", "VF322S4", "VF322BF16", "VS322F16", "VS322B8", "VS322S4", "VS322S16"]

POST_QUANT_MODE = ["F162S4", "F162B8", "VF162S4", "VF162B8", "F162S16"]

FIXPIPE_VECTOR_TENSOR_LIST = [QUANT_SCALE_0_STR, QUANT_SCALE_1_STR, RELU_WEIGHT_0_STR,
                              RELU_WEIGHT_1_STR, ELTWISE_SRC_STR]

NC1HWC0_C1_IDX = 1
NC1HWC0_C0_IDX = 4

DTYPE_FLOAT32 = "float32"
DTYPE_FLOAT16 = "float16"
DTYPE_INT32 = "int32"

VECTOR_RELU_MODE = "VECTOR_RELU"
SCALAR_RELU_MODE = "SCALAR_RELU"
NORMAL_RELU_MODE = "NORMAL_RELU"

PRE_ACT_UNIT_STR = "pre_act"
POST_ACT_UNIT_STR = "post_act"

FIXPIPE_SCOPE_MAP = {
    QUANT_SCALE_0_STR: scope_fb0,
    QUANT_SCALE_1_STR: scope_fb3,
    RELU_WEIGHT_0_STR: scope_fb1,
    RELU_WEIGHT_1_STR: scope_fb2,
    ELTWISE_SRC_STR: scope_cbuf
}

CONV3D_TAG = ["conv3d_C", "conv3d_c_col"]


def set_build_cfg():
    build_cfg = {
            "dummy_placeholder": True
    }

    set_fusion_buildcfg("FixPipe", build_cfg)


def create_placeholder(input_dict, name):
    if "ori_shape" not in input_dict:
        raise RuntimeError("ori_shape not in dict")

    attrs = {}
    if is_scaler_input(input_dict):
        if "const_value" not in input_dict.keys():
            raise RuntimeError("const_value not in dict")
        attrs["const_value"] = input_dict.get("const_value")

    attrs["ori_shape"] = input_dict.get("ori_shape")
    return tvm.placeholder(input_dict.get("shape"), input_dict.get("dtype"), name=name, attrs=attrs)


def get_op_type(x: Tensor):
    if x.op.tag in ["gemm", "matmul_ub_to_ub"]:
        return "matmul"

    if x.op.tag in ["conv2d_backprop_input", "conv2d_backprop_input_ub"]:
        return "conv2d_backprop_input"
    if x.op.name == "res_conv2d_wino":
        return "conv2d_wino"

    if len(x.op.input_tensors) == 1 and \
            x.op.input_tensors[0].op.input_tensors[0].name in ["mad1", "res_conv2d"]:
        return "conv2d"
    if "conv_res" in x.op.tag and support_conv_instruction():
        return "conv2d"
    if x.op.tag.startswith("conv2d_backprop_filter"):
        return "conv2d_backprop_filter"

    if x.op.tag in CONV3D_TAG:
        return "conv3d"

    if x.op.tag == "conv3d_backprop_input_c_ddr_vn":
        return "conv3d_backprop_input"
    return "None"


def get_op_info_from_attrs(key: str, tensor: Tensor):
    if key not in tensor.op.attrs:
        raise RuntimeError("key [{}] not in attrs".format(key))
    return tensor.op.attrs[key]


def calc_shape_total_dim(shape: List):
    if len(shape) == 0:
        raise RuntimeError("shape cannot be []")
    dim = functools.reduce(lambda x, y: x * y, shape[:])
    return dim


def is_scaler_input(input: (Tensor, dict, None)) -> bool:
    if input is None:
        return False

    if isinstance(input, Tensor):
        input_shape = shape_to_list(get_op_info_from_attrs("ori_shape", input))
    else:
        input_shape = input.get("ori_shape")

    # scalar: ori_shape:(), check shape:[1]
    if len(input_shape) == 0:
        if isinstance(input, Tensor):
            input_shape = shape_to_list(input.shape)
        else:
            input_shape = input.get("shape")

        if len(input_shape) != 1 or input_shape[0] != 1:
            raise RuntimeError("shape should be 1 when ori_shape is empty")

    dim = calc_shape_total_dim(input_shape)
    if dim == 1:
        return True

    return False


def is_vector_input(input:(Tensor, dict, None), mode: Optional[str] = None) -> bool:
    if input is None:
        return False

    if isinstance(input, Tensor):
        input_shape = shape_to_list(get_op_info_from_attrs("ori_shape", input))
    else:
        input_shape = input.get("ori_shape")

    if len(input_shape) == 0:
        return False

    dim = calc_shape_total_dim(input_shape)
    if not isinstance(dim, int) or dim > 1:
        return True

    # fixpipe param is wrong when pre_act is vector relu and pre_conv without "V" for v350
    if support_conv_instruction() and dim == 1 and mode == VECTOR_RELU_MODE:
        return True

    return False


def get_input_scalar_value(input: (Tensor, dict, None)):
    if input is None:
        return None
    if isinstance(input, Tensor):
        const_value = get_op_info_from_attrs("const_value", input)
    else:
        const_value = input.get("const_value")

    if len(const_value) == 0:
        raise RuntimeError("scalar's const_value is empty")

    return const_value[0]


def check_fixpipe_support():
    """
    fixpipe support check
    """
    is_support_fixpipe = tbe.common.platform.platform_info.intrinsic_check_support(
        "Intrinsic_fix_pipe_unit_list")
    if not is_support_fixpipe:
        raise RuntimeError("fixpipe is not supported for current soc")


def get_conv_antiq_offset(antiq_offset_tensor, x2):
    antiq_offset_value = get_input_scalar_value(antiq_offset_tensor)
    if antiq_offset_value is not None and x2 is not None and not isinstance(antiq_offset_value, tvm.tir.expr.IntImm):
        antiq_offset_value = cast(antiq_offset_value, x2.dtype)
    return antiq_offset_value
