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
matmul
"""
from impl.util import util_select_op_base
from impl.dynamic.mat_mul import base_op_select_format
from impl.dynamic.batch_matmul_v2 import gen_op_select_format_params
from impl.dynamic.batch_matmul_v2 import add_fallback_convert_to_json
from impl.dynamic.batch_matmul_v2 import check_shape
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.util_gemm import get_op_support_info_gemm
from impl.util.util_gemm import gemm_compute
from impl.util.util_gemm import gemm_impl
from impl.util.util_gemm import get_prebuild_pattern
from tbe.common.context import op_context


def get_op_specific_info(input_x: dict, input_y: dict, bias: dict = None, offset_w: dict = None,
                     output_z: dict = None, transpose_x1: bool = False, transpose_x2: bool = False,
                     offset_x: int = 0, kernel_name: str = "matmul"):
    """
    get the matmul prebuild pattern

    """
    return get_prebuild_pattern(input_x, op_pattern="Matmul")


def op_select_format(input_x: dict, input_y: dict, bias: dict = None, offset_w: dict = None,
                     output_z: dict = None, transpose_x1: bool = False, transpose_x2: bool = False,
                     offset_x: int = 0, kernel_name: str = "matmul", impl_mode: str = "") -> str:
    """
    provide dynamic format to FE
    """
    # BatchMatMulV1 does not support offset_w
    bias_fp32_flag = False
    if bias:
        bias_dtype = bias.get("dtype", "float16")
        bias_fp32_flag = (bias_dtype == "float32")
    context = op_context.get_context()
    if context:
        impl_mode_dict = context.get_addition("op_impl_mode_dict")
        if impl_mode_dict:
            impl_mode = impl_mode_dict.get("MatMul", impl_mode_dict.get("MatMulV2", ""))
    _, full_case_senario_combinations = base_op_select_format(input_x, input_y, transpose_x1,
                                                              transpose_x2, bias_fp32_flag,
                                                              impl_mode=impl_mode)

    param_list = gen_op_select_format_params(full_case_senario_combinations, support_offset_w=True)
    return add_fallback_convert_to_json(param_list, full_case_senario_combinations, check_shape(input_x, input_y, bias))


def get_op_support_info(input_x1: dict,
                        input_x2: dict,
                        bias: dict,
                        offset_w: dict = None,
                        output_y: dict = None,
                        transpose_x1: bool = False,
                        transpose_x2: bool = False,
                        offset_x: int = 0,
                        kernel_name: str = "matmul",
                        impl_mode: str = "") -> str:
    """
    get the matmul split, which only split the m and n, cannot cut k with bias

    """
    inputs = [input_x1, input_x2, bias]
    return get_op_support_info_gemm(inputs, output_y, transpose_x1, transpose_x2, "mat_mul")


def _is_fuzzily_build():
    """
    check fuzzily build flag
    """
    context = op_context.get_context()
    return (context and context.get_build_type() == "fuzzily_build")


def check_supported(input_x1: dict,
                    input_x2: dict,
                    bias: dict,
                    offset_w: dict = None,
                    output_y: dict = None,
                    transpose_x1: bool = False,
                    transpose_x2: bool = False,
                    offset_x: int = 0,
                    kernel_name: str = "matmul",
                    impl_mode: str = "") -> tuple:
    """
    check the op support situation
    """
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    src_dtype = input_x1.get("dtype")
    cube_type = ["float16", "int8", "int4"]
    if (format_a == "FRACTAL_NZ" or format_b == "FRACTAL_NZ") and src_dtype in cube_type:
        return True, ""
    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")

    if any(v == 0 for v in shape_a) or any(v == 0 for v in shape_b):
        return True, ""

    dynamic_flag = any(v < 0 for v in shape_a) or any(v < 0 for v in shape_b)
    if not dynamic_flag:
        para_check.check_shape(shape_a, param_name="input_x1")
        para_check.check_shape(shape_b, param_name="input_x2")
    target_type = ["float32", "int32", "float16", "int8", "int4"]
    res = True, ""
    if src_dtype in target_type and not dynamic_flag:
        if len(shape_a) != 2 and len(shape_b) != 2:
            reason = "the input_shape is not supported, shape_a:%s, shape_b%s"\
                     % (str(shape_a), str(shape_b))
            res = False, reason
        else:
            km_shape = shape_a[0] if transpose_x1 else shape_a[1]
            kn_shape = shape_b[1] if transpose_x2 else shape_b[0]
            if km_shape != kn_shape:
                reason = "the shape not equal, km_shape:%s, kn_shape:%s" % (km_shape, kn_shape)
                res = False, reason
    if _is_fuzzily_build() and src_dtype not in ["float16", "float32"]:
        reason = "in dynamic mode, src dtype only support float16 and float32"
        res = False, reason

    return res


@tbe_platform.fusion_manager.register("mat_mul")
def mat_mul_compute(input_x1,
                    input_x2,
                    bias,
                    offset_w=None,
                    output_y=None,
                    transpose_x1=False,
                    transpose_x2=False,
                    offset_x=0,
                    kernel_name="matmul",
                    impl_mode=""):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    input_x1: dict
        A matrix(2D Tensor), the type can be float16,
        float32, int32, the shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    input_x2: dict
        A matrix(2D Tensor), the type can be float16,
        float32, int32, the shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type can be float16,
        float32, int32, the shape must be 1-dimensional,
        the format can be [ND, NHWC]
    output_y: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be float16,
        float32, int32, the shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "matmul"

    Returns
    -------
    None
    """
    attrs = {
        "offset_a": offset_x,
        "offset_b": offset_w,
        "tensor_c": bias,
        "trans_a": transpose_x1,
        "trans_b": transpose_x2,
        "kernel_name": kernel_name,
        "op_type": "MatMulV2",
        "impl_mode": impl_mode,
        "is_fusion": True
    }
    return gemm_compute(input_x1, input_x2, output_y, attrs)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_BOOL,
                            para_check.REQUIRED_ATTR_BOOL, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def mat_mul(input_x1,
            input_x2,
            bias,
            offset_w=None,
            output_y=None,
            transpose_x1=False,
            transpose_x2=False,
            offset_x=0,
            kernel_name="matmul",
            impl_mode=""):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    input_x1: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be float16,
        float32, int32, the origin shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    input_x2: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be float16,
        float32, int32, the origin shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type can be float16,
        float32, int32, the shape must be 1-dimensional,
        the format can be [ND, NHWC]
    output_y: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be float16,
        float32, int32, the origin shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "matmul"

    Returns
    -------
    None
    """
    attrs = {
        "offset_a": offset_x,
        "offset_b": offset_w,
        "tensor_c": bias,
        "trans_a": transpose_x1,
        "trans_b": transpose_x2,
        "kernel_name": kernel_name,
        "op_type": "MatMulV2",
        "impl_mode": impl_mode,
        "zero_flag": False
    }
    gemm_impl(input_x1, input_x2, output_y, attrs, mode="static")
