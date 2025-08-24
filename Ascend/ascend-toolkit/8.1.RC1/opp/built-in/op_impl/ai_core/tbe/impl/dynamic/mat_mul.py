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
dynamic mat_mul
"""
from impl.dynamic.batch_matmul_v2 import _binary_constant_build
from impl.dynamic.batch_matmul_v2 import _dynamic_build
from impl.dynamic.batch_matmul_v2 import gen_op_select_format_params
from impl.dynamic.batch_matmul_v2 import check_support_gemv
from impl.dynamic.batch_matmul_v2 import check_support_nd
from impl.dynamic.batch_matmul_v2 import add_fallback_convert_to_json
from impl.dynamic.batch_matmul_v2 import check_shape
from impl.util import util_gemm
from impl.util import util_select_op_base
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util.util_gemm import get_op_support_info_gemm
from impl.util.util_gemm import gemm_compute
from impl.util.util_gemm import get_prebuild_pattern
from impl.util.util_gemm import reset_dtype
from impl.util.util_gemm import reset_format
from impl.util.util_gemm import op_select_format_fix_pipe_l0c2out
from impl.util.util_gemm import op_select_no_data_move_out2l1_nd2nz

from tbe.common.context import get_context
from tbe.common.buildcfg import get_current_build_config
from tbe.common.utils.const import ComputeFlow


DYNAMIC_FLAG = -1
FUZZY_SUCC_LEN = 8


def get_op_specific_info(input_x: dict, input_y: dict, bias: dict = None, offset_w: dict = None,
                     output_z: dict = None, transpose_x1: bool = False, transpose_x2: bool = False,
                     offset_x: int = 0, kernel_name: str = "matmul"):
    """
    get the matmul prebuild pattern

    """
    return get_prebuild_pattern(input_x, op_pattern="Matmul")


def get_op_support_info(input_x1, input_x2, bias, offset_w=None, output_y=None,
                        transpose_x1=False, transpose_x2=False, offset_x=0, kernel_name="matmul"):
    """
    get the matmul split, which only split the m and n, cannot cut k with bias

    """
    inputs = [input_x1, input_x2, bias]
    return get_op_support_info_gemm(inputs, output_y, transpose_x1, transpose_x2, "mat_mul")


def _op_select_format_fix_pipe() -> tuple:
    """
    dynamic format of new architecture

    return : dynamic format combination, static format combination
    """
    if tbe_platform.intrinsic_check_support("Intrinsic_data_move_out2l1_nd2nz"):
        dyn_scenario, stc_scenario = op_select_format_fix_pipe_l0c2out()
    else:
        dyn_scenario, stc_scenario = op_select_no_data_move_out2l1_nd2nz()

    return list(dyn_scenario.values()), list(stc_scenario.values())


def base_op_select_format(input_x, input_y, transpose_x1, transpose_x2,
                          bias_fp32_flag: bool, impl_mode: str = "") -> tuple:
    """
    provide dynamic format to FE(Base processing)
    This funciton contains all basic format combinations

    return : dynamic format combination, static format combination
    """
    if tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out"):
        return _op_select_format_fix_pipe()

    dyn_case_scenario_list = []
    full_case_scenario_list = []
    # The order from left to right is input1, input2, input3(bias), output
    base_case_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"), ("float16", "ND"),
                           ("int8", "ND"), ("float16", "FRACTAL_NZ"))]

    base_case_bias_fp32_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"), ("float", "ND"),
                                    ("int8", "ND"), ("float16", "FRACTAL_NZ"))]

    base_case_fp32_out_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"), ("float", "ND"),
                                    ("int8", "ND"), ("float", "FRACTAL_NZ"))]

    base_quant_case_scenario = [(("int8", "FRACTAL_NZ"), ("int8", "FRACTAL_Z"), ("int32", "ND"), ("int8", "ND"),
                                 ("int32", "FRACTAL_NZ"))]

    fp32_dtype_scenario = [(("float", "NHWC"), ("float", "NHWC"), ("float", "NHWC"), ("int8", "ND"), ("float", "NHWC")),
                           (("float", "ND"), ("float", "ND"), ("float", "ND"), ("int8", "ND"), ("float", "ND"))]

    int32_dtype_scenario = [(("int32", "NHWC"), ("int32", "NHWC"), ("int32", "NHWC"), ("int8", "ND"),
                            ("int32", "NHWC")),
                            (("int32", "ND"), ("int32", "ND"), ("int32", "ND"), ("int8", "ND"), ("int32", "ND"))]

    base_case_nzz_fp16_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_ZN_RNN"), ("float16", "ND"),
                                   ("int8", "ND"), ("float16", "FRACTAL_NZ"))]
    # ND input and output scenario
    nd_case_scenario = [
            (("float16", "ND"), ("float16", "ND"), ("float16", "ND"), ("int8", "ND"), ("float16", "ND")),
            (("float16", "ND"), ("float16", "FRACTAL_NZ"), ("float16", "ND"), ("int8", "ND"), ("float16", "ND"))
        ]
    nd_case_scenario = []
    nd_fp32out_scenario = [
            (("float16", "ND"), ("float16", "ND"), ("float", "ND"), ("int8", "ND"), ("float", "ND")),
            (("float16", "ND"), ("float16", "FRACTAL_NZ"), ("float", "ND"), ("int8", "ND"), ("float", "ND"))
        ]
    nd_fp32out_scenario = []
    support_s322f32 = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "s322f32")
    if check_support_nd(input_x, input_y, transpose_x1, transpose_x2):
        base_case_scenario += [(("float16", "ND"), ("float16", "ND"), ("float16", "ND"),
                                ("int8", "ND"), ("float16", "ND"))]
    if check_support_gemv(input_x, input_y, transpose_x1, transpose_x2):
        base_case_scenario += [
            (("float16", "ND"), ("float16", "ND"), ("float16", "ND"), ("int8", "ND"), ("float16", "ND")),
            (("float16", "ND"), ("float16", "FRACTAL_NZ"), ("float16", "ND"), ("int8", "ND"), ("float16", "ND"))
        ]
    dyn_case_scenario_list = base_case_scenario + nd_case_scenario + base_case_nzz_fp16_scenario + \
        base_case_fp32_out_scenario
    # Construct scenario list for static
    if bias_fp32_flag and impl_mode == "keep_bias_fp32":
        full_case_scenario_list = base_case_bias_fp32_scenario + base_quant_case_scenario + fp32_dtype_scenario + \
            base_case_fp32_out_scenario + nd_case_scenario + nd_fp32out_scenario
    else:
        full_case_scenario_list = base_case_scenario + base_quant_case_scenario + fp32_dtype_scenario + \
            base_case_fp32_out_scenario + nd_case_scenario + nd_fp32out_scenario
    if support_s322f32:
        full_case_scenario_list += int32_dtype_scenario
    return dyn_case_scenario_list, full_case_scenario_list


def op_select_format(input_x: dict, input_y: dict, bias: dict = None, offset_w: dict = None,
                     output_z: dict = None, transpose_x1: bool = False, transpose_x2: bool = False,
                     offset_x: int = 0, kernel_name: str = "matmul") -> str:
    """
    provide dynamic format to FE
    """
    # BatchMatMulV1 does not support offset_w
    bias_fp32_flag = False
    if bias:
        bias_dtype = bias.get("dtype", "float16")
        bias_fp32_flag = (bias_dtype == "float32")
    scenario_combinations, _ = base_op_select_format(input_x, input_y, transpose_x1,
                                                     transpose_x2, bias_fp32_flag)
    param_list = gen_op_select_format_params(scenario_combinations, support_offset_w=True)
    return add_fallback_convert_to_json(param_list, scenario_combinations, check_shape(input_x, input_y, bias))


@register_operator_compute("MatMulV2", op_mode="dynamic", support_fusion=True)
@register_operator_compute("MatMul", op_mode="dynamic", support_fusion=True)
def mat_mul_fuse_compute(input_x1, input_x2, bias, offset_w, output_y,
                         transpose_x1=False, transpose_x2=False, offset_x=0,
                         kernel_name="matmul"):
    """
    matmul computer for fusion

    Parameters:
    input_x1: tensor
    input_x2: tensor
    bias: tensor or None
    offset_w: None
    output_y: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    transpose_x1: bool
        If true, shape_a == transposed before multiplication
    transpose_x2: bool
        If true, shape_a == transposed before multiplication
    kernel_name: str
        cce kernel_name
    Returns
    -------
    res : dict
        A dict object, dict with input tensor and output tensor
    """
    attrs = {
        "tensor_c": bias,
        "trans_a": transpose_x1,
        "trans_b": transpose_x2,
        "kernel_name": kernel_name,
    }

    return gemm_compute(input_x1, input_x2, output_y, attrs, mode="dynamic")


@tbe_register.register_param_generalization("MatMul")
@tbe_register.register_param_generalization("MatMulV2")
def matmul_generalization(input_x1, input_x2, bias, offset_w=None, output_y=None,
                           transpose_x1=False, transpose_x2=False, offset_x=0, kernel_name="matmul",
                           generalize_config=None):
    result = None
    reset_format(input_x1, input_x2, bias, output_y)
    reset_dtype(input_x1, input_x2, output_y)
    if generalize_config.get("mode") == "all_shape":
        result = []
        input_x1["shape"] = util_gemm.DYNAMIC_UNKNOWN_RANK
        input_x2["shape"] = util_gemm.DYNAMIC_UNKNOWN_RANK
        output_y["shape"] = util_gemm.DYNAMIC_UNKNOWN_RANK
        input_x1["ori_format"] = "ND"
        input_x2["ori_format"] = "ND"
        output_y["ori_format"] = "ND"
        if bias:
            bias["shape"] = util_gemm.DYNAMIC_UNKNOWN_RANK
            bias["ori_format"] = "ND"
        offset_x = None

        result.append([input_x1, input_x2, bias, offset_w, output_y, transpose_x1, transpose_x2, offset_x])
    return result


@register_operator("MatMul")
@register_operator("MatMulV2")
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT, para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_BOOL,
    para_check.REQUIRED_ATTR_BOOL,
    para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def mat_mul(input_x1, input_x2, bias, offset_w=None, output_y=None,
            transpose_x1=False, transpose_x2=False, offset_x=0, kernel_name="matmul"):
    """
    caculating matrix multiplication with bias, C = A * B + bias
    only support input with NZ or ND format and fp16 in dynamic mode

    Parameters:
    input_x1: dict
        A dict object, dict with keys(shape, dtype, and range)
        the dtype must be fp16
        the format can be FRACTAL_NZ or ND
    input_x2: dict
        A dict object, dict with keys(shape, dtype and range)
        the dtype must be fp16
        the format can be FRACTAL_NZ or ND
    bias: dict
        A dict object, dict with keys(shape and dtype) or None
        the dtype must be fp16
        the format must be ND
    offset_w: None
        input offset_w tensor
    output_y: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format can be FRACTAL_NZ or ND
    transpose_x1: bool
        If true, shape_a == transposed before multiplication
    transpose_x2: bool
        If true, shape_a == transposed before multiplication
    kernel_name: str
        cce kernel_name
    Returns
    -------
    res : dict
        None
    """
    reset_format(input_x1, input_x2, bias, output_y)
    if input_x2.get("format") == "FRACTAL_ZN_RNN":
        input_x2["format"] = "FRACTAL_Z"

    is_prebuild = get_current_build_config("enable_op_prebuild")
    if is_prebuild:
        get_context().add_build_res("pattern", "MatMul")

    extra_params = {"op_type": "MatMulV2"}
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    ndnd = input_x1.get("format") == "ND" and input_x2.get("format") == "ND"
    ndnz_nd = input_x1.get("format") == "ND" and input_x2.get("format") == "FRACTAL_NZ" and \
              output_y.get("format") == "ND"
    # not build mix-l2 currently
    if ndnd and support_l0c2out:
        extra_params["nd2nz_type"] = ComputeFlow.on_the_fly.value
    elif ndnz_nd:
        extra_params["nd2nz_type"] = ComputeFlow.weight_nz.value

    mat_mul_inputs = [input_x1, input_x2, bias, offset_w, output_y, transpose_x1, transpose_x2, offset_x]
    context = tbe_context.op_context.get_context()
    if context.get_addition("is_binary_constant") == 1:
        _binary_constant_build(context, mat_mul_inputs, kernel_name, extra_params, support_l0c2out)
    else:
        _dynamic_build(mat_mul_inputs, kernel_name, extra_params, support_l0c2out)
