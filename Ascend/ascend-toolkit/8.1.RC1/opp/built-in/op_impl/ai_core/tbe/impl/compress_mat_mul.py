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
matmulcompress
"""
from impl.util.util_gemm import get_op_support_info_gemm
from impl.util.util_gemm import gemm_compute
from impl.util.util_gemm import gemm_impl
from impl.util.util_gemm import get_prebuild_pattern
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import WEIGHT_SPARSE_4_2
from impl.util.platform_adapter import WEIGHT_UNZIP


def check_supported(input_x1,
                    input_x2,
                    compress_index,
                    bias,
                    offset_w=None,
                    output_y=None,
                    trans_a=False,
                    trans_b=False,
                    offset_x=0,
                    kernel_name="compress_matmul"):
    """
    check the op support situation
    """
    cube_type = ["float16", "int8"]
    if (input_x1.get("format") == "FRACTAL_NZ" or input_x2.get("format") == "FRACTAL_NZ") and \
            input_x1.get("dtype") in cube_type:
        return True, ""
    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")
    dynamic_flag = any(v < 0 for v in shape_a) or any(v < 0 for v in shape_b)
    if not dynamic_flag:
        para_check.check_shape(shape_a, param_name="input_x1")
        para_check.check_shape(shape_b, param_name="input_x2")
    target_type = ["float32", "int32", "float16", "int8"]
    res = True, ""
    reason = "shape of input_x1 and input_x2 is not supported by aicore, shape_a is %s,shape_b is %s" \
              % (str(shape_a), str(shape_b))
    if input_x1.get("dtype") in target_type and not dynamic_flag:
        if len(shape_a) != 2 and len(shape_b) != 2:
            res = False, reason
        else:
            km_shape = shape_a[0] if trans_a else shape_a[1]
            kn_shape = shape_b[1] if trans_b else shape_b[0]
            if km_shape != kn_shape:
                res = False, reason
    return res


@register_operator_compute("compress_mat_mul", op_mode="static", support_fusion=True)
def compress_mat_mul_compute(input_x1,
                             input_x2,
                             compress_index,
                             bias,
                             offset_w=None,
                             output_y=None,
                             trans_a=False,
                             trans_b=False,
                             offset_x=0,
                             alg=WEIGHT_UNZIP,
                             kernel_name="compress_matmul"):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    input_x1: dict
        A dict object, contain a matrix(2D Tensor) 's type and
        shape and format, the type can be int8, the shape must
        be 2-dimensional, the format can be [FRACTAL_NZ]
    input_x2: dict
        A dict object, contain a matrix(2D Tensor) 's type and
        shape and format, the type can be int8, the shape must
        be 2-dimensional, the format can be [FRACTAL_Z]
    compress_index: the dict of input compress index
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type must be [int32, float16],
        the shape must be 1-dimensional, the format can be [ND, NHWC]
    output_y: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be [float16, int32], the
        shape must be 2-dimensional, the format can be [FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "compress_matmul"

    Returns
    -------
    None
    """
    attrs = {
        "compress_index": compress_index,
        "offset_a": offset_x,
        "offset_b": offset_w,
        "tensor_c": bias,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "kernel_name": kernel_name,
        "op_type": "CompressMatMul",
        "alg": alg
    }
    return gemm_compute(input_x1, input_x2, output_y, attrs)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_BOOL, para_check.REQUIRED_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def get_op_support_info(input_x1,
                        input_x2,
                        compress_index,
                        bias=None,
                        offset_w=None,
                        output_y=None,
                        trans_a=False,
                        trans_b=False,
                        offset_x=0,
                        kernel_name="compress_matmul"):
    """
    get the matmul split, which only split the m and n, cannot cut k with bias

    """
    inputs = [input_x1, input_x2, bias]
    return get_op_support_info_gemm(inputs, output_y, trans_a, trans_b, "compress_mat_mul")


def get_op_specific_info(input_x1,
                         input_x2,
                         compress_index,
                         bias=None,
                         offset_w=None,
                         output_y=None,
                         trans_a=False,
                         trans_b=False,
                         offset_x=0,
                         kernel_name="compress_matmul"):
    """
    get the matmul prebuild pattern

    """
    return get_prebuild_pattern(input_x1, op_pattern="Matmul")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_BOOL, para_check.REQUIRED_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def compress_mat_mul(input_x1,
                     input_x2,
                     compress_index,
                     bias,
                     offset_w=None,
                     output_y=None,
                     trans_a=False,
                     trans_b=False,
                     offset_x=0,
                     alg=WEIGHT_UNZIP,
                     kernel_name="compress_matmul"):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    input_x1: dict
        A dict object, contain a matrix(2D Tensor) 's type and
        shape and format, the type can be int8, the shape must
        be 2-dimensional, the format can be [FRACTAL_NZ]
    input_x2: dict
        A dict object, contain a matrix(2D Tensor) 's type and
        shape and format, the type can be int8, the shape must
        be 2-dimensional, the format can be [FRACTAL_Z]
    compress_index: the dict of input compress index
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type must be [int32, float16],
        the shape must be 1-dimensional, the format can be [ND, NHWC]
    output_y: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be [float16, int32], the
        shape must be 2-dimensional, the format can be [FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "compress_matmul"

    Returns
    -------
    None
    """
    attrs = {
        "compress_index": compress_index,
        "offset_a": offset_x,
        "offset_b": offset_w,
        "tensor_c": bias,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "kernel_name": kernel_name,
        "alg": alg,
        "op_type": "CompressMatMul"
    }
    gemm_impl(input_x1, input_x2, output_y, attrs, mode="static")
