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
batch_matmul
"""
from impl.dynamic.batch_matmul import base_op_select_format
from impl.dynamic.batch_matmul_v2 import gen_op_select_format_params
from impl.dynamic.batch_matmul_v2 import add_fallback_convert_to_json
from impl.dynamic.batch_matmul_v2 import check_shape
from impl.util import util_select_op_base
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_gemm import get_op_support_info_gemm
from impl.util.util_gemm import gemm_compute
from impl.util.util_gemm import gemm_impl
from impl.util.util_gemm import get_prebuild_pattern


def get_op_support_info(input_x,
                        input_y,
                        bias=None,
                        output_z=None,
                        adj_x1=False,
                        adj_x2=False,
                        kernel_name="matmul"):
    """
    get the batch_matmul split, which only split batch, m and n, cannot cut k with bias

    """
    inputs = [input_x, input_y, bias]
    return get_op_support_info_gemm(inputs, output_z, adj_x1, adj_x2, "batch_matmul")


def op_select_format(input_x: dict, input_y: dict, bias: dict = None, output_z: dict = None, adj_x1: bool = False,
                     adj_x2: bool = False, kernel_name: str = "matmul"):
    """
    provide dynamic format to FE
    """
    # BatchMatMulV1 does not support offset_w
    src_dtype = input_x.get("dtype")
    _, full_case_senario_combinations = base_op_select_format(input_x, input_y, src_dtype, adj_x1, adj_x2)
    param_list = gen_op_select_format_params(full_case_senario_combinations, support_offset_w=False)
    return add_fallback_convert_to_json(param_list, full_case_senario_combinations, check_shape(input_x, input_y, bias))


def get_op_specific_info(input_x, input_y, bias=None, offset_w=None, output_z=None, adj_x1=False,
                         adj_x2=False, offset_x=0, kernel_name="batch_matmul"):
    """
    get the BatchMatmul prebuild pattern

    """
    return get_prebuild_pattern(input_x, op_pattern="BatchMatmul")


@register_operator_compute("batch_matmul", op_mode="static", support_fusion=True)
def batch_matmul_compute(input_x, input_y, bias=None, output_z=None, adj_x1=False,
                         adj_x2=False, kernel_name="matmul"):
    """
    algorithm: batch_matmul
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters
    ---------
    input_x: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    input_y: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type can be float16,
        float32, int32, the shape must be 1-dimensional,
        the format can be [ND, NHWC]
    output_z: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    adj_x1: bool
        If True, the shape of input_x1 must be transposed before multiplication
    adj_x2: bool
        If True, the shape of input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "matmul"

    Return
    ------
    None
    """
    attrs = {
        "trans_a": adj_x1,
        "trans_b": adj_x2,
        "tensor_c": bias,
        "kernel_name": kernel_name,
        "is_fusion": True,
        "op_type": "BatchMatMul"
    }
    return gemm_compute(input_x, input_y, output_z, attrs)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_BOOL,
                            para_check.REQUIRED_ATTR_BOOL, para_check.KERNEL_NAME)
def batch_matmul(input_x, input_y, bias=None, output_z=None, adj_x1=False,
                 adj_x2=False, kernel_name="matmul"):
    """ algorithm: batch_matmul
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters
    ---------
    input_x: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    input_y: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type can be float16,
        float32, int32, the shape must be 1-dimensional,
        the format can be [ND, NHWC]
    output_z: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    adj_x1: bool
        If True, the shape of input_x1 must be transposed before multiplication
    adj_x2: bool
        If True, the shape of input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "matmul"

    Return
    ------
    None
    """
    attrs = {
        "trans_a": adj_x1,
        "trans_b": adj_x2,
        "tensor_c": bias,
        "kernel_name": kernel_name,
        "op_type": "BatchMatMul",
        "zero_flag": False
    }
    gemm_impl(input_x, input_y, output_z, attrs, mode="static")
