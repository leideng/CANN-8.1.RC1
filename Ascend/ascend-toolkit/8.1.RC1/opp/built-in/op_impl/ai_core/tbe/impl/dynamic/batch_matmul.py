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
dynamic batch_matmul
"""
import collections
import warnings

from impl.dynamic.batch_matmul_v2 import batch_matmul_v2
from impl.dynamic.batch_matmul_v2 import is_reuse_offline_binary
from impl.dynamic.batch_matmul_v2 import check_batch_range
from impl.dynamic.batch_matmul_v2 import check_fp32_case_scenario
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
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tbe_platform
from impl.util.util_cube_dynamic import check_supported_mm_ub
from impl.util.util_gemm import get_op_support_info_gemm
from impl.util.util_gemm import gemm_compute
from impl.util.util_gemm import get_prebuild_pattern
from impl.util.util_gemm import reset_dtype
from impl.util.util_gemm import op_select_format_fix_pipe_l0c2out
from impl.util.util_gemm import op_select_no_data_move_out2l1_nd2nz


DYNAMIC_FLAG = -1
FUZZY_SUCC_LEN = 8


def get_op_specific_info(input_x: dict, input_y: dict, bias: dict = None, output_z: dict = None, adj_x1: bool = False,
                         adj_x2: bool = False, kernel_name: str = "matmul"):
    """
    get the batch_matmul prebuild pattern

    """
    return get_prebuild_pattern(input_x, op_pattern="BatchMatmul")


def _op_select_format_fix_pipe(not_support_dyn: bool) -> tuple:
    """
    dynamic format of new architecture

    Parameters : support dynamic flag

    return : dynamic format combination, static format combination
    """

    if tbe_platform.intrinsic_check_support("Intrinsic_data_move_out2l1_nd2nz"):
        dyn_scenario, stc_scenario = op_select_format_fix_pipe_l0c2out(no_offset_w=True)
    else:
        dyn_scenario, stc_scenario = op_select_no_data_move_out2l1_nd2nz(no_offset_w=True)

    if not_support_dyn:
        warnings.warn("input_x, input_y out of batch_range")
        dyn_scenario = collections.OrderedDict()

    return list(dyn_scenario.values()), list(stc_scenario.values())


def base_op_select_format(input_x, input_y, src_dtype, adj_x1, adj_x2) -> tuple:
    """
    provide dynamic format to FE(Base processing)
    This funciton contains all basic format combinations

    return : dynamic format combination, static format combination
    """
    shape_a = input_x.get("ori_shape")
    shape_b = input_y.get("ori_shape")
    dynamic_flag = any(v < 0 for v in shape_a) or any(v < 0 for v in shape_b)
    support_fp32 = check_fp32_case_scenario(shape_a, shape_b, adj_x2, src_dtype)
    not_support_dyn = dynamic_flag and not check_batch_range(input_x, input_y)

    if tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out"):
        return _op_select_format_fix_pipe(not_support_dyn)

    dyn_case_scenario_list = []
    full_case_scenario_list = []

    # The order from left to right is input1, input2, input3(bias), output
    base_case_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"),
                           ("float16", "ND"), ("float16", "FRACTAL_NZ"))]
    fp32_out_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"),
                          ("float", "ND"), ("float", "FRACTAL_NZ"))]

    fp32_dtype_scenario = [(("float", "NHWC"), ("float", "NHWC"), ("float", "NHWC"), ("float", "NHWC")),
                           (("float", "ND"), ("float", "ND"), ("float", "ND"), ("float", "ND"))]
    int32_dtype_scenario = [(("int32", "NHWC"), ("int32", "NHWC"), ("int32", "NHWC"), ("int32", "NHWC")),
                            (("int32", "ND"), ("int32", "ND"), ("int32", "ND"), ("int32", "ND"))]

    if not support_fp32:
        fp32_dtype_scenario = []
        int32_dtype_scenario = []

    # ND input and output scenario
    nd_case_scenario = [(("float16", "ND"), ("float16", "ND"), ("float16", "ND"), ("float16", "ND")),
                        (("float16", "ND"), ("float16", "FRACTAL_NZ"), ("float16", "ND"), ("float16", "ND"))]
    nd_case_scenario = []
    nd_fp32out_scenario = [(("float16", "ND"), ("float16", "ND"), ("float", "ND"), ("float", "ND")),
                           (("float16", "ND"), ("float16", "FRACTAL_NZ"), ("float", "ND"), ("float", "ND"))]
    nd_fp32out_scenario = []
    if check_support_nd(input_x, input_y, adj_x1, adj_x2):
        base_case_scenario += [(("float16", "ND"), ("float16", "ND"), ("float16", "ND"),
                                ("float16", "ND"))]
    if check_support_gemv(input_x, input_y, adj_x1, adj_x2):
        base_case_scenario += [
            (("float16", "ND"), ("float16", "ND"), ("float16", "ND"), ("float16", "ND")),
            (("float16", "ND"), ("float16", "FRACTAL_NZ"), ("float16", "ND"), ("float16", "ND"))
        ]
    support_s322f32 = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "s322f32")

    dyn_case_scenario_list = base_case_scenario + nd_case_scenario
    if not_support_dyn:
        warnings.warn("input_x, input_y out of batch_range")
        dyn_case_scenario_list = []
    # Construct scenario list for static
    full_case_scenario_list = base_case_scenario + fp32_dtype_scenario + fp32_out_scenario + nd_case_scenario + \
        nd_fp32out_scenario
    if support_s322f32:
        full_case_scenario_list += int32_dtype_scenario
    return dyn_case_scenario_list, full_case_scenario_list


def op_select_format(input_x: dict, input_y: dict, bias: dict = None, output_z: dict = None, adj_x1: bool = False,
                     adj_x2: bool = False, kernel_name: str = "matmul") -> str:
    """
    provide dynamic format to FE
    """
    src_dtype = input_x.get("dtype")
    scenario_combinations, _ = base_op_select_format(input_x, input_y, src_dtype, adj_x1, adj_x2)

    param_list = gen_op_select_format_params(scenario_combinations, support_offset_w=False)
    return add_fallback_convert_to_json(param_list, scenario_combinations, check_shape(input_x, input_y, bias))


def get_op_support_info(input_x1, input_x2, bias=None, output_z=None,
                        adj_x1=False, adj_x2=False, kernel_name="matmul"):
    """
    get the batch_matmul split, which only split batch, m and n, cannot cut k with bias
    """
    inputs = [input_x1, input_x2, bias]
    return get_op_support_info_gemm(inputs, output_z, adj_x1, adj_x2, "batch_matmul")


def check_supported(input_x1, input_x2, bias, output_z, adj_x1=False, adj_x2=False):
    """
    check the op support situation
    """
    res = check_supported_mm_ub(input_x1, input_x2, bias, output_z)
    return res


@register_operator_compute("BatchMatMul", op_mode="dynamic", support_fusion=False)
def batch_matmul_fuse_compute(input_x1, input_x2, bias, output_z,
                         adj_x1=False, adj_x2=False,
                         kernel_name="matmul"):
    """
    matmul computer for fusion

    Parameters:
    input_x1: tensor
    input_x2: tensor
    bias: tensor or None
    output_z: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ or ND
    adj_x1: bool
        If true, shape_a == transposed before multiplication
    adj_x2: bool
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
        "trans_a": adj_x1,
        "trans_b": adj_x2,
        "kernel_name": kernel_name,
    }
    return gemm_compute(input_x1, input_x2, output_z, attrs, mode="dynamic")


@tbe_register.register_param_generalization("BatchMatMul")
def batch_matmul_generalization(input_x1, input_x2, bias=None, output_z=None,
                                adj_x1=False, adj_x2=False, kernel_name="batchmatmul",
                                generalize_config=None):
    result = None
    reset_dtype(input_x1, input_x2, output_z)
    if generalize_config.get("mode") == "all_shape":
        result = []
        attrs = {"adj_x1": adj_x1, "adj_x2": adj_x2, "op_type": "BatchMatMul"}
        if is_reuse_offline_binary(input_x1, input_x2, bias, output_z, attrs):
            input_x1["shape"] = util_gemm.DYNAMIC_UNKNOWN_RANK
            input_x2["shape"] = util_gemm.DYNAMIC_UNKNOWN_RANK
            output_z["shape"] = util_gemm.DYNAMIC_UNKNOWN_RANK
        result.append([input_x1, input_x2, bias, output_z, adj_x1, adj_x2])
    return result


@register_operator("BatchMatMul")
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_BOOL, para_check.REQUIRED_ATTR_BOOL,
    para_check.KERNEL_NAME)
def batch_matmul(input_x1, input_x2, bias=None, output_z=None,
                 adj_x1=False, adj_x2=False, kernel_name="matmul"):
    """
    caculating matrix multiplication with bias, C = A * B + bias
    only support input with nz format and fp16 in dynamic mode

    Parameters:
    input_x1: dict
        A dict object, dict with keys(shape, dtype, and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    input_x2: dict
        A dict object, dict with keys(shape, dtype and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    bias: dict
        A dict object, dict with keys(shape and dtype) or None
        the dtype must be fp16
        the format must be ND
    output_z: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    adj_x1: bool
        If true, shape_a == transposed before multiplication
    adj_x2: bool
        If true, shape_a == transposed before multiplication
    kernel_name: str
        cce kernel_name
    Returns
    -------
    res : dict
        None
    """
    batch_matmul_v2(input_x1, input_x2, bias=bias, output_z=output_z,
                    adj_x1=adj_x1, adj_x2=adj_x2, kernel_name=kernel_name)
