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
dynamic batch_matmul_v2
"""
import collections
import math
import warnings
import json
from itertools import product
from enum import Enum
from typing import Optional
from tbe.dsl.compute.gemm_compute_util import GEMMComputeParam

from impl.util import util_gemm
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util.util_cube_dynamic import check_tensor_shape
from impl.util.util_cube_dynamic import is_empty_tensor_scene
from impl.util.util_cube_dynamic import check_dynamic_range_lower
from impl.util.util_cube_dynamic import ceil_div
from impl.util.util_cube_dynamic import align
from impl.util.util_cube_dynamic import check_supported_mm_ub
from impl.util.util_gemm import get_op_support_info_gemm
from impl.util.util_gemm import gemm_compute
from impl.util.util_gemm import get_prebuild_pattern
from impl.util.util_gemm import reset_dtype
from impl.util.util_gemm import reset_format
from impl.util.util_gemm import op_select_format_fix_pipe_l0c2out
from impl.util.util_gemm import op_select_no_data_move_out2l1_nd2nz
from tbe.common.context import op_context
from tbe.common.utils.const import ComputeFlow
from tbe.dsl.base.operation import get_te_var

# General limitation of the size for input shape: 2**31 - 1
SHAPE_SIZE_LIMIT = 2147483647
DYNAMIC_FLAG = -1
BLOCK_CUBE = 16
DYNAMIC_FLAG_UNRANK = [-2]
BATCH_NZ_LENGTH = 5
MATMUL_NZ_LENGTH = 4
BATCH_ND_LENGTH = 3
MATMUL_ND_LENGTH = 2
ND_LENGTH = 2
MKN_MIN = 1
LOWER_LIMIT_STR = "LOWER_LIMIT"
FUZZY_SUCC_LEN = 8
ALIGN_NUM_TWO = 2
PAD_NONE = 0
PAD_B = 1
PAD_A = 2
PAD_AB = 3
SHAPE_ALIGN = 256
PAD_ALIGN = 128
SHAPE_ALIGN_FP32 = 64
SHAPE_LOWER = 1000
SHAPE_LOWER_FP32 = 512
SHAPE_LOWER_BIAS_FP32 = 200
LARGE_OUTER_LIMIT = 680
SHAPE_UPPER = 65280
NO_NZ_FUSION = 0
NZ_VEC_B = 1
NZ_VEC_A = 2
NZ_VEC_AB = 3
NZ_SERIAL = 1
NZ_PIPELINE_ATTACH = 2
NZ_PIPELINE_NOT_ATTACH = 3


class Format(str, Enum):
    """
    class of format
    """
    FRACTAL_NZ = 'FRACTAL_NZ'
    ND = 'ND'


def get_op_specific_info(input_x1, input_x2, bias=None, offset_w=None, output_z=None,
                        adj_x1=False, adj_x2=False, offset_x=0, kernel_name="matmul"):
    """
    get the batch_matmulv2 prebuild pattern

    """
    return get_prebuild_pattern(input_x1, op_pattern="BatchMatmul")


def get_op_support_info(input_x1, input_x2, bias=None, offset_w=None, output_z=None,
                        adj_x1=False, adj_x2=False, offset_x=0, kernel_name="matmul"):
    """
    get the batch_matmul_v2 split, which only split batch, m and n, cannot cut k with bias

    """
    inputs = [input_x1, input_x2, bias]
    return get_op_support_info_gemm(inputs, output_z, adj_x1, adj_x2, "batch_matmul")


def _op_select_format_fix_pipe(not_support_dyn: bool) -> tuple:
    """
    dynamic format of new architecture

    Parameters : support dynamic flag

    return : dynamic format combination, static format combination
    """
    if tbe_platform.intrinsic_check_support("Intrinsic_data_move_out2l1_nd2nz"):
        dyn_scenario, stc_scenario = op_select_format_fix_pipe_l0c2out(no_offset_w=False)
    else:
        dyn_scenario, stc_scenario = op_select_no_data_move_out2l1_nd2nz(no_offset_w=False)

    if not_support_dyn:
        warnings.warn("input_x, input_y out of batch_range")
        dyn_scenario = collections.OrderedDict()

    return list(dyn_scenario.values()), list(stc_scenario.values())


def check_support_nd(input_x, input_y, adj_x1, adj_x2):
    block = 16
    shape_a = input_x.get("ori_shape")
    src_dtype = input_x.get("dtype")
    shape_b = input_y.get("ori_shape")
    return (src_dtype == "float16" and shape_a[-1] % block == 0 and
            shape_a[-2] % block == 0 and shape_b[-1] % block == 0 and
            shape_b[-2] % block == 0)


def check_support_gemv(input_x, input_y, adj_x1, adj_x2):
    shape_a = input_x.get("ori_shape")
    src_dtype = input_x.get("dtype")
    shape_b = input_y.get("ori_shape")
    dynamic_flag = any(v < 0 for v in shape_a) or any(v < 0 for v in shape_b)
    if dynamic_flag or src_dtype not in ("float16", ):
        return False
    n_dim = shape_b[-2] if adj_x2 else shape_b[-1]
    m_dim, k_dim = shape_a[-2], shape_a[-1]
    block_reduce = tbe_platform.CUBE_MKN.get(src_dtype).get("mac")[1]
    gemv_valid = m_dim == 1 and k_dim % (block_reduce * tbe_platform.BLOCK_IN) == 0

    if gemv_valid and not adj_x1 and n_dim % tbe_platform.BLOCK_OUT == 0:
        return True

    return False


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

    # The order from left to right is input1, input2, input3(bias), input4(offset_w), output
    base_case_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"), ("float16", "ND"),
                           ("int8", "ND"), ("float16", "FRACTAL_NZ"))]
    base_quant_case_scenario = [
        (("int8", "FRACTAL_NZ"), ("int8", "FRACTAL_Z"), ("int32", "ND"), ("int8", "ND"), ("int32", "FRACTAL_NZ")),
        (("int8", "FRACTAL_NZ"), ("int8", "FRACTAL_Z"), ("float16", "ND"), ("int8", "ND"), ("float16", "FRACTAL_NZ"))
    ]
    # Vector Logic
    fp32_dtype_scenario = [
        (("float", "NHWC"), ("float", "NHWC"), ("float", "NHWC"), ("int8", "ND"), ("float", "NHWC")),
        (("float", "ND"), ("float", "ND"), ("float", "ND"), ("int8", "ND"), ("float", "ND"))
    ]

    int32_dtype_scenario = [
        (("int32", "NHWC"), ("int32", "NHWC"), ("int32", "NHWC"), ("int8", "ND"), ("int32", "NHWC")),
        (("int32", "ND"), ("int32", "ND"), ("int32", "ND"), ("int8", "ND"), ("int32", "ND"))
    ]

    if not support_fp32:
        fp32_dtype_scenario = []
        int32_dtype_scenario = []

    fp32_out_scenatio = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"),
                          ("float", "ND"), ("int8", "ND"), ("float", "FRACTAL_NZ"))]
    rnn_scenatio = [
        (("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_ZN_RNN"), ("float", "ND"), ("int8", "ND"),
         ("float", "FRACTAL_NZ")),
        (("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_ZN_RNN"), ("float16", "ND"), ("int8", "ND"),
         ("float16", "FRACTAL_NZ"))]
    if check_support_nd(input_x, input_y, adj_x1, adj_x2):
        base_case_scenario += [(("float16", "ND"), ("float16", "ND"), ("float16", "ND"),
                                ("int8", "ND"), ("float16", "ND"))]
    if check_support_gemv(input_x, input_y, adj_x1, adj_x2):
        base_case_scenario += [
            (("float16", "ND"), ("float16", "ND"), ("float16", "ND"), ("int8", "ND"), ("float16", "ND")),
            (("float16", "ND"), ("float16", "FRACTAL_NZ"), ("float16", "ND"), ("int8", "ND"), ("float16", "ND"))
        ]
    support_s322f32 = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "s322f32")

    dyn_case_scenario_list = base_case_scenario
    if not_support_dyn:
        warnings.warn("input_x, input_y out of batch_range")
        dyn_case_scenario_list = []

    # Construct scenario list for static
    full_case_scenario_list = base_case_scenario + base_quant_case_scenario + fp32_dtype_scenario + \
        fp32_out_scenatio + rnn_scenatio
    if support_s322f32:
        full_case_scenario_list += int32_dtype_scenario

    return dyn_case_scenario_list, full_case_scenario_list


def check_fp32_case_scenario(shape_a, shape_b, adj_x2, src_dtype):
    """
    check if support float32 or int32 type

    Paramaters

    shape_a: list or tuple ,information of shape_a
    shape_b: list or tuple ,information of shape_b
    adj_x2: bool
    src_type: type of input_x

    Returns

    support format for float32 or int32
    """
    dynamic_flag = any(v < 0 for v in shape_a) or any(v < 0 for v in shape_b)

    if not dynamic_flag:
        shape_a_length = len(shape_a)
        shape_b_length = len(shape_b)
        shape_a_batch = shape_a[0:shape_a_length-2]
        shape_b_batch = shape_b[0:shape_b_length-2]
        if shape_a_batch != shape_b_batch:
            return False
        elif shape_a_length == 2:
            return False
        elif adj_x2:
            if shape_b[shape_a_length - 2] == 1:
                return False
        elif bool(1-adj_x2):
            if shape_b[shape_a_length - 1] == 1:
                return False

    if _is_fuzzily_build() and src_dtype != "float16":
        return False

    return True


def _is_fuzzily_build():
    """
    check fuzzily build flag
    """
    context = op_context.get_context()
    return (context and context.get_build_type() == "fuzzily_build")


def check_batch_range(input_x, input_y):
    """
    Check the batch shape and range legal

    Parameters
    ----------
    input_x: dict with shape and range
    input_y: dict with shape and range

    Returns
    -------
    legit or not
    """
    shape_a = input_x.get("ori_shape")
    shape_b = input_y.get("ori_shape")
    if list(shape_a) == DYNAMIC_FLAG_UNRANK or list(shape_b) == DYNAMIC_FLAG_UNRANK:
        return True

    range_x1 = input_x.get("range")
    range_x2 = input_y.get("range")

    range_x1 = [[v, v] for v in shape_a] if not range_x1 and all(v > 0 for v in shape_a) else range_x1
    range_x2 = [[v, v] for v in shape_b] if not range_x2 and all(v > 0 for v in shape_b) else range_x2
    if len(shape_a) < ND_LENGTH:
        warnings.warn("shape_a length is at least 2-dimensional")
        return False
    if len(shape_b) < ND_LENGTH:
        warnings.warn("shape_b length is at least 2-dimensional")
        return False

    batch_range_x1 = range_x1[:(len(shape_a) - ND_LENGTH)]
    batch_range_x2 = range_x2[:(len(shape_b) - ND_LENGTH)]

    if (not batch_range_x1) or (not batch_range_x2):
        return True

    return True


def gen_op_select_format_params(scenario_combinations: list, support_offset_w: bool = False) -> list:
    """
    generate format
    """
    input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                           datatype=','.join(x[0][0] for x in scenario_combinations),
                                           format=','.join(x[0][1] for x in scenario_combinations))
    input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                           datatype=','.join(x[1][0] for x in scenario_combinations),
                                           format=','.join(x[1][1] for x in scenario_combinations))
    # Bias supports only ND format
    input2 = util_select_op_base.gen_param(classify="input2", name="bias",
                                           datatype=','.join(x[2][0] for x in scenario_combinations),
                                           format=','.join(x[2][1] for x in scenario_combinations))
    if support_offset_w:
        input3 = util_select_op_base.gen_param(classify="input3", name="offset_w",
                                               datatype=','.join(x[3][0] for x in scenario_combinations),
                                               format=','.join(x[3][1] for x in scenario_combinations))
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype=','.join(x[4][0] for x in scenario_combinations),
                                                format=','.join(x[4][1] for x in scenario_combinations))
        param_list = [input0, input1, input2, input3, output0]
    else:
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype=','.join(x[3][0] for x in scenario_combinations),
                                                format=','.join(x[3][1] for x in scenario_combinations))
        param_list = [input0, input1, input2, output0]
    return param_list


def get_aclnn_support_param(scenario_combinations: list, param_dynamic_dict: dict, support_aclnn: bool) -> list:
    """
    provide aclnn support info to FE
    """
    scenario_combinations = [tuple(tuple(item) for item in x)
                             for x in scenario_combinations]
    if not tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out"):
        return param_dynamic_dict

    fall_back_senario = [(("float16", "ND"), ("float16", "ND"),
                          ("float16", "ND"), ("int8", "ND"), ("float16", "ND")),
                         (("float16", "ND"), ("float16", "ND"), ("float",
                           "ND"), ("int8", "ND"), ("float16", "ND")),
                         (("float", "ND"), ("float", "ND"),
                          ("float", "ND"), ("int8", "ND"), ("float", "ND")),
                         (("bfloat16", "ND"), ("bfloat16", "ND"),
                          ("float", "ND"), ("int8", "ND"), ("bfloat16", "ND")),
                         (("float16", "ND"), ("float16", "FRACTAL_NZ"),
                          ("float16", "ND"), ("int8", "ND"), ("float16", "ND")),
                         (("float16", "ND"), ("float16", "FRACTAL_NZ"),
                          ("float", "ND"), ("int8", "ND"), ("float16", "ND")),
                         (("float", "ND"), ("float", "FRACTAL_NZ"),
                          ("float", "ND"), ("int8", "ND"), ("float", "ND")),
                         (("bfloat16", "ND"), ("bfloat16", "FRACTAL_NZ"),
                          ("float", "ND"), ("int8", "ND"), ("bfloat16", "ND")),
                         (("float16", "ND"), ("float16", "ND"), ("float32",
                          "ND"), ("int8", "ND"), ("float16", "ND")),
                         (("float32", "ND"), ("float32", "ND"), ("float32",
                          "ND"), ("int8", "ND"), ("float32", "ND")),
                         (("bfloat16", "ND"), ("bfloat16", "ND"),
                          ("float32", "ND"), ("int8", "ND"), ("bfloat16", "ND")),
                         (("float16", "ND"), ("float16", "FRACTAL_NZ"),
                          ("float32", "ND"), ("int8", "ND"), ("float16", "ND")),
                         (("float32", "ND"), ("float32", "FRACTAL_NZ"),
                          ("float32", "ND"), ("int8", "ND"), ("float32", "ND")),
                         (("bfloat16", "ND"), ("bfloat16", "FRACTAL_NZ"),
                          ("float32", "ND"), ("int8", "ND"), ("bfloat16", "ND"))]
    if support_aclnn:
        enable_fall_back = [("1" if x in fall_back_senario else "0")
                            for x in scenario_combinations]
    else:
        enable_fall_back = ["0" for x in scenario_combinations]
    param_dict = {}
    param_dict["enable"] = ','.join(["0" for x in scenario_combinations])
    param_dict["unknownshape_enable"] = ','.join(enable_fall_back)
    param_dynamic_dict["fallback"] = param_dict
    return param_dynamic_dict


def add_fallback_convert_to_json(param_list: list, scenario_combinations: list, support_aclnn: bool = True):
    """
    add fallback parameters and convert to json
    """

    param_dynamic_dict = util_select_op_base.get_dynamic_param_dict(param_list)
    param_dict = get_aclnn_support_param(
        scenario_combinations, param_dynamic_dict, support_aclnn)
    param_dynamic_in_json = json.dumps(param_dict, indent=4)
    return param_dynamic_in_json


def check_shape(input_x, input_y, bias=None):
    """
    check input shape
    """
    support_aclnn = True
    if bias is not None:
        shape_x = input_x.get("shape")
        shape_y = input_y.get("shape")
        unknownshape_flag = False
        if len(shape_x) == 1 and shape_x[0] == -2:
            unknownshape_flag = True
        if len(shape_y) == 1 and shape_y[0] == -2:
            unknownshape_flag = True
        equal_len = True
        if len(shape_x) != len(shape_y):
            equal_len = False
        batch_len_support = True
        if len(shape_x) < 2 or len(shape_x) > 3:
            batch_len_support = False
        if unknownshape_flag or (not equal_len) or (not batch_len_support):
            support_aclnn = False
    return support_aclnn


def op_select_format(input_x, input_y, bias=None, offset_w=None, output_z=None, adj_x1=False,
                     adj_x2=False, offset_x=0, kernel_name="matmul"):
    """
    provide dynamic format to FE
    """
    src_dtype = input_x.get("dtype")
    scenario_combinations, _ = base_op_select_format(
        input_x, input_y, src_dtype, adj_x1, adj_x2)
    param_list = gen_op_select_format_params(
        scenario_combinations, support_offset_w=True)
    return add_fallback_convert_to_json(param_list, scenario_combinations, check_shape(input_x, input_y, bias))


@register_operator_compute("BatchMatMulV2", op_mode="dynamic", support_fusion=False)
def batch_matmul_v2_fuse_compute(input_x1, input_x2, bias=None, offset_w=None, output_z=None,
                                 adj_x1=False, adj_x2=False, offset_x=0,
                                 kernel_name="matmul"):
    """
    matmul computer for fusion

    Parameters:
    input_x1: tensor
    input_x2: tensor
    bias: tensor or None
    offset_w: tensor or None
    output_z: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ or ND
    adj_x1: bool
        If true, shape_a == transposed before multiplication
    adj_x2: bool
        If true, shape_a == transposed before multiplication
    offset_x: int
        offset of gradients in quant mode
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


def _check_args(args: tuple, expect_args: list, msg: str) -> None:
    """
    check args
    """
    if args not in expect_args:
        error_manager_vector.raise_err_input_format_invalid(
            "mat_mul", msg, expect_args, args)


def _check_dynamic_mode_of_batch_matmul(shape_x1: tuple, shape_x2: tuple) -> None:
    """
    check dynamic mode
    """
    if len(shape_x1) < BATCH_ND_LENGTH - 1:
        error_manager_vector.raise_err_input_shape_invalid(
            "batch_matmul", "x1", "ori_shape dim must more than 1"
        )

    if len(shape_x2) < BATCH_ND_LENGTH - 1:
        error_manager_vector.raise_err_input_shape_invalid(
            "batch_matmul", "x2", "ori_shape dim must more than 1"
        )

    if all(i != DYNAMIC_FLAG for i in shape_x1) and all(i != DYNAMIC_FLAG for i in shape_x2):
        error_manager_vector.raise_err_specific_reson(
            "batch_matmul", "dynamic must at least one of batch, m, k, n"
        )


def _check_dynamic_mode_of_matmul(shape_x1: tuple, shape_x2: tuple) -> None:
    """
    check dynamic mode
    """
    if len(shape_x1) != ND_LENGTH:
        error_manager_vector.raise_err_input_shape_invalid(
            "mat_mul", "x1", "ori_shape dim must be 2"
        )

    if len(shape_x2) != ND_LENGTH:
        error_manager_vector.raise_err_input_shape_invalid(
            "mat_mul", "x2", "ori_shape dim must be 2"
        )

    if all(i != DYNAMIC_FLAG for i in shape_x1) and all(i != DYNAMIC_FLAG for i in shape_x2):
        error_manager_vector.raise_err_specific_reson(
            "mat_mul", "dynamic must at least one in m,k,n"
        )


def _reset_range_by_shape(input_x1: dict, input_x2: dict, range_x1: tuple, range_x2: tuple) -> tuple:
    shape_x1 = input_x1.get("shape")
    shape_x2 = input_x2.get("shape")
    # if all dim known, range may be empty
    range_x1 = tuple([dim, dim] for dim in shape_x1) if not range_x1 and all(dim > 0 for dim in shape_x1) else range_x1
    range_x2 = tuple([dim, dim] for dim in shape_x2) if not range_x2 and all(dim > 0 for dim in shape_x2) else range_x2

    # if shape is not -1, reset range as known
    if len(shape_x1) == len(range_x1):
        range_x1 = tuple(dim_range if shape_x1[idx] == -1 else (shape_x1[idx], shape_x1[idx])
                         for idx, dim_range in enumerate(range_x1))
    if len(shape_x2) == len(range_x2):
        range_x2 = tuple(dim_range if shape_x2[idx] == -1 else (shape_x2[idx], shape_x2[idx])
                         for idx, dim_range in enumerate(range_x2))
    return range_x1, range_x2


def _get_matmul_unrank_shape_and_range(input_x1: dict, input_x2: dict) -> list:
    """Get range in uniform format.

    Four scenarios are handled:
    unrank shape: ((1, None), (1, None)) or ((1, None), (1, None), (BLOCK_CUBE, BLOCK_CUBE), (BLOCK_CUBE, BLOCK_CUBE))
    static shape, empty range: adjust the upper and lower bounds of range to be consistent with dim.
    static dim, upper and lower bounds of range is differ: adjust upper and lower of bounds to be consistent with dim.
    dim is 0 in shape: adjust the range to start at 1.

    Args:
        input_x1: dict of input x1
        input_x2: dict of input x2

    Returns:
        A list of [shape_x1, range_x1, shape_x2, range_x2]
    """
    shape_x1 = input_x1.get("ori_shape")
    shape_x2 = input_x2.get("ori_shape")
    range_x1 = input_x1.get("range")
    range_x2 = input_x2.get("range")
    format_x1 = input_x1.get("format")
    format_x2 = input_x2.get("format")

    range_nd = ((1, None), (1, None))
    range_nz = ((1, None), (1, None), (BLOCK_CUBE, BLOCK_CUBE), (BLOCK_CUBE, BLOCK_CUBE))
    if list(shape_x1) == DYNAMIC_FLAG_UNRANK:
        shape_x1 = (-1, -1)
        range_x1 = range_nd if format_x1 == "ND" else range_nz
    if list(shape_x2) == DYNAMIC_FLAG_UNRANK:
        shape_x2 = (-1, -1)
        range_x2 = range_nd if format_x2 == "ND" else range_nz
    range_x1 = tuple(dim_range if dim_range[0] >= MKN_MIN else (MKN_MIN, dim_range[1]) for dim_range in range_x1)
    range_x2 = tuple(dim_range if dim_range[0] >= MKN_MIN else (MKN_MIN, dim_range[1]) for dim_range in range_x2)
    range_x1, range_x2 = _reset_range_by_shape(input_x1, input_x2, range_x1, range_x2)
    return [list(shape_x1), range_x1, list(shape_x2), range_x2]


def _get_batch_matmul_unrank_shape_and_range(input_x1: dict, input_x2: dict) -> list:
    """
    Get range in uniform format.

    Four scenarios are handled:
    unrank shape: ((1, None), (1, None), (1, None)) or
        ((1, None), (1, None), (1, None), (BLOCK_CUBE, BLOCK_CUBE), (BLOCK_CUBE, BLOCK_CUBE))
    static shape, empty range: adjust the upper and lower bounds of range to be consistent with dim.
    static dim, upper and lower bounds of range is differ: adjust upper and lower of bounds to be consistent with dim.
    dim is 0 in shape: adjust the range to start at 1.

    Args:
        input_x1: dict of input x1
        input_x2: dict of input x2

    Returns:
        A list of [shape_x1, range_x1, shape_x2, range_x2]
    """
    shape_x1 = input_x1.get("ori_shape")
    shape_x2 = input_x2.get("ori_shape")
    range_x1 = input_x1.get("range")
    range_x2 = input_x2.get("range")
    format_x1 = input_x1.get("format")
    format_x2 = input_x2.get("format")

    range_nd = ((1, None), (1, None), (1, None))
    range_nz = ((1, None), (1, None), (1, None), (BLOCK_CUBE, BLOCK_CUBE), (BLOCK_CUBE, BLOCK_CUBE))
    if list(shape_x1) == DYNAMIC_FLAG_UNRANK:
        shape_x1 = (-1, -1, -1)
        range_x1 = range_nd if format_x1 == "ND" else range_nz
    if list(shape_x2) == DYNAMIC_FLAG_UNRANK:
        shape_x2 = (-1, -1, -1)
        range_x2 = range_nd if format_x2 == "ND" else range_nz
    range_x1 = tuple(dim_range if dim_range[0] >= MKN_MIN else (MKN_MIN, dim_range[1]) for dim_range in range_x1)
    range_x2 = tuple(dim_range if dim_range[0] >= MKN_MIN else (MKN_MIN, dim_range[1]) for dim_range in range_x2)
    return [list(shape_x1), range_x1, list(shape_x2), range_x2]


def _get_dynamic_shape_and_range(input_x1: dict, input_x2: dict, bias: dict, op_type: str) -> tuple:
    """
    get the shape and range of matmul
    """
    bias_range = None

    if op_type in ("MatMul", "MatMulV2", "CompressMatMul"):
        shape_x1, range_x1, shape_x2, range_x2 = _get_matmul_unrank_shape_and_range(
            input_x1, input_x2)
    elif op_type in ("BatchMatMul", "BatchMatMulV2"):
        shape_x1, range_x1, shape_x2, range_x2 = _get_batch_matmul_unrank_shape_and_range(
            input_x1, input_x2)
    else:
        reason = f"not support op_type: {op_type}"
        error_manager_vector.raise_err_specific_reson(op_type, reason)

    if bias:
        bias_range = bias.get("range")

    return [shape_x1, shape_x2], [range_x1, range_x2, bias_range]


def _get_range_intersection(range1: list, range2: list, param_name: str, is_graph_mode: bool = False) -> list:
    """
    get range intersection of two range
    """
    if range1[1] is None:
        return range2
    if range2[1] is None:
        return range1

    range_ins = [max(range1[0], range2[0]), min(range1[1], range2[1])]
    # batch_broadcast should be considered in batch_range
    if range_ins[0] > range_ins[1] and not (param_name == "batch_range" and min(range1[0], range2[0]) == 1):
        if not is_graph_mode:
            reason = (f"the range of {param_name} is invalid because it has no intersection, "
                      "and the actual values are {range1}, {range2}")
            error_manager_vector.raise_err_specific_reson("mat_mul", reason)
        else:
            return LOWER_LIMIT_STR
    range_ins = [min(range_ins[0], range_ins[1]), max(range_ins[0], range_ins[1])]
    return range_ins


def _get_batch_range(range_x1: tuple, range_x2: tuple) -> list:
    """Get reduce range of batch.

    Args:
        range_x1: batch range of input x1
        range_x2: batch range of input x2

    Returns:
        A list of reduced range of batch. Returns None when both range_x1 and range_x2 are [] or None.
    """
    if not range_x1 and not range_x2:
        return None
    if len(range_x1) != 0 and len(range_x2) != 0 and len(range_x1) != len(range_x2):
        return [1, None]

    batch_range = [1, 1]
    range_x = []
    if range_x1 and range_x2:
        for range_mem1, range_mem2 in zip(range_x1, range_x2):
            range_ins = _get_range_intersection(range_mem1, range_mem2, "batch_range")
            range_x.append(range_ins)
    elif range_x2:
        range_x = range_x2
    else:
        range_x = range_x1

    for range_mem in range_x:
        if range_mem[1] is None:
            batch_range = [1, None]
            break
        else:
            batch_range[0] = min(batch_range[0] * range_mem[0], SHAPE_SIZE_LIMIT)
            batch_range[1] = min(batch_range[1] * range_mem[1], SHAPE_SIZE_LIMIT)

    return batch_range


def _get_input_x1_range(range_x1: tuple, format_x1: str, adj_x1: bool, op_type: str,
                        input_dtype: str, is_graph_mode: bool) -> list:
    """Get range of m, k, batch.

    Args:
        range_x1: range of x1
        format_x1: format of x1, when format is FRACTAL_NZ, return the range of m1, k1
        adj_x1: whether to swap m, k before matrix multiplication
        op_type: just for log
        is_graph_mode: mode of fuzzy generalization

    Returns:
        A list of range, like [range_m, range_k, range_batch]. when batch does not exist, range_batch is [].
    """
    range_len = BATCH_ND_LENGTH if format_x1 == 'ND' else BATCH_NZ_LENGTH
    if len(range_x1) >= range_len - 1:
        if format_x1 == 'FRACTAL_NZ':
            # adj_x1 True:  m1, k1, k0, m0
            # adj_x1 False: k1, m1, m0, k0
            k_x1_index = -4
            m_index = -3
            batch_range_x1 = range_x1[:-4]
        elif format_x1 == 'ND':
            m_index = -2
            k_x1_index = -1
            batch_range_x1 = range_x1[:-2]
    else:
        error_manager_vector.raise_err_specific_reson(op_type, "Lenth of x1_range illegal")
    m_range = list(range_x1[m_index])
    k_range_x1 = list(range_x1[k_x1_index])
    if not is_graph_mode and operation.get_op_context():
        operation.get_op_context().add_addition("batch_range_x1", batch_range_x1)
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if adj_x1:
        m_range, k_range_x1 = k_range_x1, m_range
        if format_x1 == "FRACTAL_NZ" and support_l0c2out and input_dtype == "int8":
            for index, val in enumerate(k_range_x1):
                k_range_x1[index] = ceil_div(val, ALIGN_NUM_TWO)
    return [m_range, k_range_x1, batch_range_x1]


def _get_input_x2_range(range_x2: tuple, format_x2: str, adj_x2: bool, op_type: str,
                        input_dtype: str, is_graph_mode: bool) -> list:
    """Get range of k, n, batch.

    Args:
        range_x1: range of x1
        format_x1: format of x1, when format is FRACTAL_NZ, return the range of m1, k1
        adj_x1: whether to swap m, k before matrix multiplication
        op_type: just for log
        is_graph_mode: mode of fuzzy generalization

    Returns:
        A list of range, like [range_m, range_k, range_batch]. when batch does not exist, range_batch is [].
    """
    range_len = BATCH_ND_LENGTH if format_x2 == 'ND' else BATCH_NZ_LENGTH
    if len(range_x2) >= range_len - 1:
        if format_x2 == 'FRACTAL_NZ':
            # adj_x2 True:  k1, n1, n0, k0
            # adj_x2 False: n1, k1, k0, n0
            n_index = -4
            k_x2_index = -3
            batch_range_x2 = range_x2[:-4]
        elif format_x2 == 'ND':
            k_x2_index = -2
            n_index = -1
            batch_range_x2 = range_x2[:-2]
        elif format_x2 == 'FRACTAL_Z':
            n_index = -3
            k_x2_index = -4
            batch_range_x2 = range_x2[:-4]
    else:
        error_manager_vector.raise_err_specific_reson(op_type, "Lenth of x1_range illegal")
    k_range_x2 = list(range_x2[k_x2_index])
    n_range = list(range_x2[n_index])
    if not is_graph_mode and operation.get_op_context():
        operation.get_op_context().add_addition("batch_range_x2", batch_range_x2)
    if adj_x2:
        k_range_x2, n_range = n_range, k_range_x2
    elif format_x2 == "FRACTAL_NZ" and input_dtype == "int8":
        for index, val in enumerate(k_range_x2):
            k_range_x2[index] = ceil_div(val, ALIGN_NUM_TWO)
    return [k_range_x2, n_range, batch_range_x2]


def _get_input_range(range_x1: tuple, format_x1: str, range_x2: tuple, format_x2: str, range_bias: tuple,
                     adj_x1: bool, adj_x2: bool, op_type: str, input_dtype: str, is_graph_mode: bool = False) -> list:
    """
    get range in batch, m, k, n and check range
    """
    batch_range_x1, batch_range_x2 = None, None
    if range_x1:
        m_range, k_range_x1, batch_range_x1 = _get_input_x1_range(range_x1, format_x1, adj_x1, op_type,
                                                                  input_dtype, is_graph_mode)
    else:
        # NOTE range_x1 is empty only in check of fuzzy generalization
        m_range = [1, None]
        k_range_x1 = [1, None]

    if range_x2:
        k_range_x2, n_range, batch_range_x2 = _get_input_x2_range(range_x2, format_x2, adj_x2, op_type,
                                                                  input_dtype, is_graph_mode)
    else:
        # NOTE range_x2 is empty only in check of fuzzy generalization
        k_range_x2 = [1, None]
        n_range = [1, None]

    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    k_range = _get_range_intersection(k_range_x1, k_range_x2, "k_range", is_graph_mode)
    if range_bias:
        range_bias_n = list(range_bias[0])
        if format_x2 in ('FRACTAL_NZ', 'FRACTAL_Z'):
            block_size = BLOCK_CUBE
            if input_dtype == "int8" and (not adj_x2) and support_l0c2out:
                block_size = BLOCK_CUBE * 2
            for i, val in enumerate(range_bias[0]):
                range_bias_n[i] = math.ceil(val / block_size) if val is not None else val
        n_range = _get_range_intersection(n_range, range_bias_n, "n_range", is_graph_mode)

    # in fuzzy compile, if n/k's range has no intersection return LOWER_LIMIT
    wrong_range_flag = LOWER_LIMIT_STR in (n_range, k_range)
    if wrong_range_flag:
        return LOWER_LIMIT_STR

    # in generalization func of fuzzy compile, only need check. Not add_addition
    batch_range = None
    if not is_graph_mode:
        batch_range = _get_batch_range(batch_range_x1, batch_range_x2)

    return [batch_range, m_range, k_range, n_range]


def check_and_config_para(input_x1: dict, input_x2: dict, bias: dict, output_z: dict,
                          adj_x1: bool, adj_x2: bool, kernel_name: str, op_type: str) -> tuple:
    """
    check and config dynamic mode
    """
    # get format and dtype
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    format_out = output_z.get("format")
    dtype_a = input_x1.get("dtype").lower()
    dtype_b = input_x2.get("dtype").lower()
    dtype_out = output_z.get("dtype").lower()

    # check kernel_name dtype and format
    para_check.check_kernel_name(kernel_name)
    expect_input_format_a = ['FRACTAL_NZ', 'ND']
    expect_input_format_b = ['FRACTAL_NZ', 'ND', 'FRACTAL_Z']
    expect_out_format = ['FRACTAL_NZ', 'ND']
    expect_args = list(product(expect_input_format_a, ['float16', 'bfloat16', 'int8'],
                               expect_input_format_b, ['float16', 'bfloat16', 'int8'],
                               expect_out_format, ['float16', 'float32', 'bfloat16', 'int32', 'int8']))
    support_f322f32 = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322f32r")
    if support_f322f32:
        expect_args.append(('ND', 'float32', 'ND', 'float32', 'ND', 'float32'))
    _check_args((format_a, dtype_a, format_b, dtype_b, format_out, dtype_out),
                expect_args, "format_a, dtype_a, format_b, dtype_b, format_out, dtype_out")
    # check bias if bias in not None
    if bias:
        dtype_bias = bias.get("dtype")
        para_check.check_dtype_rule(dtype_bias, ("float16", "float32", "int32"), "bias")

    # get range and ori_shape
    shape_input, range_input = _get_dynamic_shape_and_range(input_x1, input_x2, bias, op_type)
    range_x1, range_x2, range_bias = range_input
    shape_x1, shape_x2 = shape_input

    # check dynamic mode
    if op_type in ("MatMul", "MatMulV2", "CompressMatMul"):
        _check_dynamic_mode_of_matmul(shape_x1, shape_x2)
    elif op_type in ("BatchMatMul", "BatchMatMulV2"):
        _check_dynamic_mode_of_batch_matmul(shape_x1, shape_x2)
    else:
        reason = f"not support op_type: {op_type}"
        error_manager_vector.raise_err_specific_reson(op_type, reason)
    # get range in m,k,n
    if dtype_a != dtype_b:
        reason = f"dtype of x1 and x2 must be same, actual is {dtype_a}, {dtype_b}"
        error_manager_vector.raise_err_specific_reson(op_type, reason)
    input_range = _get_input_range(range_x1, format_a,
                                   range_x2, format_b,
                                   range_bias, adj_x1, adj_x2, op_type, dtype_a)

    input_range = [[1, None] if input_range[0] else input_range[0], [1, None], [1, None], [1, None]]
    is_cache_tiling = util_gemm.get_cache_tiling_flag(input_range, bias, dtype_out)
    if is_cache_tiling:
        dtype_a = "float16" if (dtype_a == "bfloat16") else dtype_a
        dtype_out = "float16" if (dtype_out == "bfloat16") else dtype_out
    ori_input_range = input_range
    return [is_cache_tiling, dtype_a, dtype_out, input_range, ori_input_range, shape_input]


def _set_shape(info_x1_x2: dict, is_cache_tiling: bool, unaligned_flag:bool = False) -> tuple:
    """
    Set input shape
    """
    format_a, format_b = info_x1_x2.get("formats")
    adj_x1, adj_x2, input_dtype = info_x1_x2.get("attrs")
    m_var = get_te_var("m").get_tvm_var()
    ka_var = get_te_var("k").get_tvm_var()
    kb_var = get_te_var("k").get_tvm_var()
    n_var = get_te_var("n").get_tvm_var()
    if is_cache_tiling:
        m_ori_var = get_te_var("m_ori").get_tvm_var()
        k_ori_var = get_te_var("k_ori").get_tvm_var()
        n_ori_var = get_te_var("n_ori").get_tvm_var()
    if format_a == "ND" and is_cache_tiling:
        m_var = m_var * BLOCK_CUBE if not unaligned_flag else m_ori_var
        ka_var = k_ori_var
    if format_b == "ND" and is_cache_tiling:
        n_var = n_var * BLOCK_CUBE if not unaligned_flag else n_ori_var
        kb_var = k_ori_var
    # unsupport ND with Nz input now
    if (tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out") and
        format_a == "FRACTAL_NZ" and format_b == "FRACTAL_NZ" and input_dtype == "int8"):
        m_var = ceil_div(m_var, ALIGN_NUM_TWO) if adj_x1 else m_var
        ka_var = ka_var if adj_x1 else ceil_div(ka_var, ALIGN_NUM_TWO)
        n_var = ceil_div(n_var, ALIGN_NUM_TWO) if (not adj_x2) else n_var
        kb_var = kb_var if (not adj_x2) else ceil_div(kb_var, ALIGN_NUM_TWO)
    return [ka_var, m_var, n_var, kb_var]


def _get_m_k_index(format_a: str, adj_x1: bool) -> list:
    """
    get the correct m, k position for shape_x1.
    """
    if adj_x1:
        m_index = -1 if format_a == "ND" else -2
        k_index = -2 if format_a == "ND" else -1
    else:
        m_index = -2 if format_a == "ND" else -1
        k_index = -1 if format_a == "ND" else -2
    return [m_index, k_index]


def _get_k_n_index(format_b: str, adj_x2: bool) -> list:
    """
    get the correct k, n position for shape_x2.
    """
    if adj_x2:
        n_index = -2 if format_b in ("ND", "FRACTAL_Z") else -1
        k_index = -1 if format_b in ("ND", "FRACTAL_Z") else -2
    else:
        n_index = -1 if format_b in ("ND", "FRACTAL_Z") else -2
        k_index = -2 if format_b in ("ND", "FRACTAL_Z") else -1
    return [k_index, n_index]


def _get_bias_tensor(bias: dict, format_b: str, is_cache_tiling: bool):
    """
    Get Bias Tensor
    """
    if bias:
        var_nori = get_te_var("n_ori").get_tvm_var()
        bias_dtype = bias.get("dtype")
        bias_ori_shape = [var_nori]
        tensor_bias = tvm.placeholder(
            bias_ori_shape, name="bias", dtype=bias_dtype, attrs={'ori_shape': bias_ori_shape})
    else:
        tensor_bias = None
    return tensor_bias


def _get_real_trans(format_a: str, format_b: str, adj_x1: bool, adj_x2: bool) -> list:
    """
    Get the correct trans values used in compute
    """
    if format_a == Format.FRACTAL_NZ:
        adj_x1 = not adj_x1
    if format_b == Format.FRACTAL_NZ:
        adj_x2 = not adj_x2
    return [adj_x1, adj_x2]


def _check_nd2nz_need_ub(input_x1: dict, input_x2: dict, nd2nz_type:Optional[int]=None) -> bool:
    """
    check if nd2nz need ub
    """
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    nd_input = _check_nd_in(input_x1, input_x2)
    is_nd_nd_in = nd_input and nd2nz_type != ComputeFlow.on_the_fly.value
    is_nd_nz_in = (nd2nz_type == ComputeFlow.weight_nz.value) and (not support_l0c2out)
    return is_nd_nd_in or is_nd_nz_in


def is_reuse_offline_binary(input_x1, input_x2, bias, output_z, attrs):
    dtype_a = input_x1.get("dtype").lower()
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")

    adj_x1 = attrs.get("adj_x1")
    adj_x2 = attrs.get("adj_x2")
    op_type = attrs.get("op_type")

    dtype_out = output_z.get("dtype").lower()

    shape_input, range_input = _get_dynamic_shape_and_range(input_x1, input_x2, bias, op_type)
    range_x1, range_x2, range_bias = range_input
    input_range = _get_input_range(range_x1, format_a,
                                   range_x2, format_b,
                                   range_bias, adj_x1, adj_x2, op_type, dtype_a)
    input_range = [[1, None] if input_range[0] else input_range[0], [1, None], [1, None], [1, None]]
    return util_gemm.get_cache_tiling_flag(input_range, bias, dtype_out)


def _check_nd_in(input_x1: dict, input_x2: dict) -> bool:
    """
    check format of inputs
    support ND in ND out or ND in NZ out
    """
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    return format_a == "ND" and format_b == "ND"


def define_cache_tiling_var(input_x1: dict, input_x2: dict,
                            nd2nz_type:Optional[int]=None, extra_params:Optional[dict]=None) -> None:
    """
    define variables in cache tiling
    """
    def _create_var(list_var):
        for name_var in list_var:
            operation.var(name_var)

    pad_flag = extra_params.get("pad_flag", 0)
    nz_fusion_flag = extra_params.get("nz_fusion_flag", 0)

    list_var = ("batch_single_core", "m_single_core", "n_single_core", "batch_dim", "n_dim", "m_dim", "k_dim",
                "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor", "kal1_factor",
                "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch_l1_factor", "batch_ub_l0_time", "batch_cub",
                "out_branch_flag", "bias_flag", "hf32_flag", "datatype_bf16", "al1_db", "bl1_db", "l0c_db",
                "l2_cache_flag", "close_k_shift")
    _create_var(list_var)

    if _check_nd2nz_need_ub(input_x1, input_x2, nd2nz_type):
        list_var = ("m_aub", "n_bub", "k_aub", "k_bub", "batch_aub", "batch_bub", "multi_n_ub_l1", "multi_m_ub_l1",
                    "multi_k_aub_l1", "multi_k_bub_l1", "multi_batch_aub_l1", "multi_batch_bub_l1", "a_align_value",
                    "b_align_value", "aub_align_bound", "bub_align_bound", "flag_cub_solving_bank_conflict")
        _create_var(list_var)

        if get_te_var("batch"):
            # in order to reduce the number of variables, the variables perm_x1/perm_x2 are concatenated with the
            # perm_x1/x2 attributes of the list int type
            list_var = ("dim0_x1", "dim1_x1", "dim2_x1", "dim0_x2", "dim1_x2", "dim2_x2", "perm_x1", "perm_x2")
            for name_var in list_var:
                operation.var(name_var, [1, None])
    elif nz_fusion_flag:
        list_var = ("m1_aub", "n1_bub", "k1_aub", "k1_bub", "m_aub_dim", "n_bub_dim", "k_aub_dim", "k_bub_dim")
        _create_var(list_var)
    elif pad_flag:
        list_var = ("m_aub", "n_bub", "k_aub", "k_bub", "aub_dim", "bub_dim")
        _create_var(list_var)


def _check_empty_tensor(input_x1, input_x2, output_z, bias):
    # if tensor'shape has 0 or range's lower is 0
    if (check_dynamic_range_lower([input_x1, input_x2, output_z]) or
        is_empty_tensor_scene([input_x1, input_x2, output_z])):
        if bias:
            bias_range = bias.get("range")
            if bias_range and bias_range[0][0] == 0:
                bias["range"] = ((1, bias_range[0][1]), )
        check_tensor_shape({"tensor": [input_x1, input_x2, output_z],
                            "value": [-1, -1, -1],
                            "range": [(2, 2), (2, 2), (2, 2)]})


def _check_soc_support():
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        error_manager_vector.raise_err_specific_reson(
            "batch_matmul", "Hi3796CV300ES and Hi3796CV300CS and SD3403 don't support dynamic shape")


def _construct_shape_x1_x2(info_x1_x2, is_cache_tiling, nd2nz_type, unaligned_flag):
    shape_x1, shape_x2 = info_x1_x2.get("shapes")
    format_x1, format_x2 = info_x1_x2.get("formats")
    adj_x1, adj_x2, input_dtype = info_x1_x2.get("attrs")

    if is_cache_tiling:
        # NOTE Only the batches of a and b are the same or the batch dimensions of a and b are different, and
        #      the dimension can only be 1.
        #      case1: (b1, b2, x, x) (b1, b2, x, x)
        #      case2: (1, 1, x, x) (x, x)
        # NOTE Use the dim*_x* variable only in the ND format, binary scenario, must have batch in tensor a and b
        if (get_te_var("batch")
            and format_x1 == "ND" and format_x2 == "ND"
            and nd2nz_type != ComputeFlow.on_the_fly.value):
            dim0_x1 = get_te_var("dim0_x1").get_tvm_var()
            dim1_x1 = get_te_var("dim1_x1").get_tvm_var()
            dim2_x1 = get_te_var("dim2_x1").get_tvm_var()
            dim0_x2 = get_te_var("dim0_x2").get_tvm_var()
            dim1_x2 = get_te_var("dim1_x2").get_tvm_var()
            dim2_x2 = get_te_var("dim2_x2").get_tvm_var()

            return [dim0_x1, dim1_x1, dim2_x1], [dim0_x2, dim1_x2, dim2_x2]

    # NOTE Only the batches of a and b are the same or the batch dimensions of a and b are different, and
    #      full broadcast to a or b.
    #      case1: (b1, b2, x, x) (b1, b2, x, x)
    #      case2: (2, 3, x, x) (x, x)
    if len(shape_x1) >= BATCH_ND_LENGTH:
        shape_x1 = [get_te_var("batch").get_tvm_var(), DYNAMIC_FLAG, DYNAMIC_FLAG]
    if len(shape_x2) >= BATCH_ND_LENGTH:
        shape_x2 = [get_te_var("batch").get_tvm_var(), DYNAMIC_FLAG, DYNAMIC_FLAG]

    m_index, ka_index = _get_m_k_index(format_x1, adj_x1)
    kb_index, n_index = _get_k_n_index(format_x2, adj_x2)
    shape_x1[ka_index], shape_x1[m_index], shape_x2[n_index], shape_x2[kb_index] = _set_shape(info_x1_x2,
        is_cache_tiling, unaligned_flag)
    block_size_inner_axis = BLOCK_CUBE
    if input_dtype == "int8" and tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out"):
        block_size_inner_axis = tbe_platform.CUBE_MKN.get(input_dtype).get("mac")[1]
    if format_x1 != "ND":
        shape_x1 = shape_x1 + [BLOCK_CUBE, block_size_inner_axis]
    if format_x2 != "ND":
        shape_x2 = shape_x2 + [BLOCK_CUBE, block_size_inner_axis]
    return shape_x1, shape_x2


def get_ori_batch_shape(input_x1, input_x2):
    '''
    get ori_batch_shape of tensor_a/tensor_b to calculate batch axes indices in mmad_compute
    '''
    ori_shape_x1 = input_x1.get("ori_shape")
    ori_shape_x2 = input_x2.get("ori_shape")
    ori_shape_x1 = (-1, -1, -1, -1, -1, -1) if list(ori_shape_x1) == DYNAMIC_FLAG_UNRANK else ori_shape_x1
    ori_shape_x2 = (-1, -1, -1, -1, -1, -1) if list(ori_shape_x2) == DYNAMIC_FLAG_UNRANK else ori_shape_x2
    if get_te_var("batch") and (len(ori_shape_x1) >= BATCH_ND_LENGTH or len(ori_shape_x2) >= BATCH_ND_LENGTH):
        batch_x1_vars = [get_te_var("batch_a1").get_tvm_var(), get_te_var("batch_a2").get_tvm_var(),
                         get_te_var("batch_a3").get_tvm_var(), get_te_var("batch_a4").get_tvm_var()]
        batch_x2_vars = [get_te_var("batch_b1").get_tvm_var(), get_te_var("batch_b2").get_tvm_var(),
                         get_te_var("batch_b3").get_tvm_var(), get_te_var("batch_b4").get_tvm_var()]
        batch_len_x1 = len(ori_shape_x1) - ND_LENGTH
        batch_len_x2 = len(ori_shape_x2) - ND_LENGTH
        if not (len(ori_shape_x1) >= BATCH_ND_LENGTH and len(ori_shape_x2) >= BATCH_ND_LENGTH):
            # the max batch dim is 4
            return batch_x1_vars[4 - batch_len_x1:], batch_x2_vars[4 - batch_len_x2:]
        batch_len_max = max(batch_len_x1, batch_len_x2)
        return batch_x1_vars[4 - batch_len_max:], batch_x2_vars[4 - batch_len_max:]
    if get_te_var("batch") and input_x1.get("format") == "ND":
        return [get_te_var("batch").get_tvm_var()], [get_te_var("batch").get_tvm_var()]
    return [], []


def define_batch_matmul_var(op_type, extra_params, is_cache_tiling, input_range, ori_input_range):
    batch_range, m_range, k_range, n_range = input_range
    _, ori_m_range, ori_k_range, ori_n_range = ori_input_range
    operation.var("k_ori", ori_k_range)
    operation.var("m_ori", ori_m_range)
    operation.var("n_ori", ori_n_range)
    operation.var("k", k_range)
    operation.var("m", m_range)
    operation.var("n", n_range)

    # NOTE For the 2-dimensional input BatchMatMul scenario, it is believed that there must be a batch variable, and
    #      the batch is 1 at runtime.
    #      scenario 1: onnx
    #      scenario 2: batch, m-axis fusion
    if op_type in ("BatchMatMulV2", "BatchMatMul"):
        if batch_range is None:
            operation.var("batch", [1, 1])
        else:
            operation.var("batch", batch_range)
        if is_cache_tiling:
            list_var = ["batch_a1", "batch_a2", "batch_a3", "batch_a4",
                        "batch_b1", "batch_b2", "batch_b3", "batch_b4",
                        "batch_c1", "batch_c2", "batch_c3", "batch_c4"]
            for name_var in list_var:
                operation.var(name_var, [1, None])
    if extra_params.get("pad_flag", 0) and is_cache_tiling:
        operation.var("k_pad")
        operation.var("m_pad")
        operation.var("n_pad")


def get_input_shape(input_x1: dict, input_x2: dict, bias: dict, output_z: dict, adj_x1: bool,
                    adj_x2: bool, kernel_name: str, extra_params: Optional[dict] = None,
                    unaligned_flag: bool = False):
    _check_empty_tensor(input_x1, input_x2, output_z, bias)
    _check_soc_support()
    if extra_params is None:
        op_type = "BatchMatMulV2"
        nd2nz_type = ComputeFlow.tacit.value
    else:
        op_type = extra_params.get("op_type", "BatchMatMulV2")
        nd2nz_type = extra_params.get("nd2nz_type", 0)
    is_cache_tiling, _, _, input_range, ori_input_range, shape_input = check_and_config_para(
        input_x1, input_x2, bias, output_z, adj_x1, adj_x2, kernel_name, op_type
    )

    format_a = input_x1.get("format")
    format_b = input_x2.get("format")

    # define var
    define_batch_matmul_var(op_type, extra_params, is_cache_tiling, input_range, ori_input_range)
    if is_cache_tiling:
        define_cache_tiling_var(input_x1, input_x2, nd2nz_type, extra_params)
    else:
        nd2nz_type = ComputeFlow.tacit.value

    info_x1_x2 = {"shapes": shape_input, "formats": (format_a, format_b),
                  "attrs": (adj_x1, adj_x2, input_x1.get("dtype"))}
    shape_x1, shape_x2 = _construct_shape_x1_x2(info_x1_x2, is_cache_tiling, nd2nz_type, unaligned_flag)
    return [shape_x1, shape_x2, is_cache_tiling, nd2nz_type, input_range]


@tbe_register.register_param_generalization("BatchMatMulV2")
def batch_matmul_v2_generalization(input_x1, input_x2, bias=None, offset_w=None, output_z=None,
                                   adj_x1=False, adj_x2=False, offset_x=0, kernel_name="batch_matmul",
                                   generalize_config: dict = None) -> list:
    result = None
    reset_format(input_x1, input_x2, bias, output_z)
    reset_dtype(input_x1, input_x2, output_z)
    if generalize_config.get("mode") == "all_shape":
        result = []
        input_x1["shape"] = util_gemm.DYNAMIC_UNKNOWN_RANK
        input_x2["shape"] = util_gemm.DYNAMIC_UNKNOWN_RANK
        output_z["shape"] = util_gemm.DYNAMIC_UNKNOWN_RANK
        input_x1["ori_format"] = "ND"
        input_x2["ori_format"] = "ND"
        output_z["ori_format"] = "ND"
        if bias:
            bias["shape"] = util_gemm.DYNAMIC_UNKNOWN_RANK
            bias["ori_format"] = "ND"
        offset_x = None

        result.append([input_x1, input_x2, bias, offset_w, output_z, adj_x1, adj_x2, offset_x])
    return result


def batch_matmul_compute(input_x1: dict, input_x2: dict, bias: dict, offset_w: dict, output_z: dict, adj_x1: bool,
                         adj_x2: bool, offset_x: int, kernel_name: str, extra_params:Optional[dict]=None,
                         unaligned_flag: bool = False):
    """
    batch_matmul computer

    Parameters:
    input_x1: dict
    A dict object, dict with keys(shape, dtype and range)
    the dtype must be fp16
    the format must be FRACTAL_NZ
    input_x2: dict
    A dict object, dict with keys(shape, dtype and range)
    the dtype must be fp16
    the format must be FRACTAL_NZ
    bias: dict
    A dict object, dict with keys(shape and format) or None
    the dtype must be fp16
    the format must be ND
    output_z: dict
    A dict object, dict with keys(shape and dtype)
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
    A dict object, dict with input tensor and output tensor
    """
    (shape_x1, shape_x2, is_cache_tiling, nd2nz_type,
     input_range) = get_input_shape(input_x1, input_x2, bias, output_z, adj_x1, adj_x2,
                                    kernel_name, extra_params, unaligned_flag)
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    dtype_in = input_x1.get("dtype").lower()
    dtype_out = output_z.get("dtype").lower()
    tensor_x1 = tvm.placeholder(shape_x1, name="tensor_a", dtype=dtype_in)
    tensor_x2 = tvm.placeholder(shape_x2, name="tensor_b", dtype=dtype_in)
    if is_cache_tiling:
        ori_batch_x1, ori_batch_x2 = get_ori_batch_shape(input_x1, input_x2)
        tensor_x1.op.attrs["ori_batch_shape"] = ori_batch_x1
        tensor_x2.op.attrs["ori_batch_shape"] = ori_batch_x2
    tensor_bias = _get_bias_tensor(bias, format_b, is_cache_tiling)
    adj_x1, adj_x2 = _get_real_trans(format_a, format_b, adj_x1, adj_x2)

    para_dict = {
        "trans_a": adj_x1,
        "trans_b": adj_x2,
        "format_a": format_a,
        "format_b": format_b,
        "format_out": output_z.get("format"),
        "dst_dtype": dtype_out,
        "tensor_c": tensor_bias,
        "cache_tiling_flag": is_cache_tiling,
        "unaligned_flag": unaligned_flag,
        "kernel_name": kernel_name,
        "input_range": input_range,
        "nd2nz_type": nd2nz_type,
        "pad_flag": extra_params.get("pad_flag", 0),
        "nz_fusion_flag":extra_params.get("nz_fusion_flag", 0),
        "nz_fusion_mode":extra_params.get("nz_fusion_mode", 0)
    }
    op_res = tbe.gemm(tensor_x1, tensor_x2, para_dict)

    tensor_list = [tensor_x1, tensor_x2]
    if bias:
        tensor_list.append(tensor_bias)

    return {"op_placeholder": tensor_list, "op_res": [op_res], "cache_tiling_flag": is_cache_tiling}


def _update_config(res, kernel_name, tensor_lists, config_flag, extra_params):
    support_l0c2out, bias = config_flag
    cache_tiling_flag = res.get("cache_tiling_flag")
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_lists,
        "build_args": {"constant_realize_extent_in_infer_bound": False,
                       "enable_db_fold": True}
    }
    attr_cache_tiling = dict(res.get("op_res")[0].op.attrs.items()).get("cache_tiling", 0)
    if cache_tiling_flag and int(attr_cache_tiling) == 1:
        if extra_params["op_type"] != "CompressMatMul":
            config.get("build_args")["predicate_realize_bound"] = False
        config.get("build_args")["enable_branch_eliminator_else_case"] = False
        # The following switch may be set to be default_on.
        # It is used to solve sync problem between Pre-Vector pipe and MTE3 pipe.
        if extra_params["op_type"] == "BatchMatMulV2":
            config.get("build_args")["InjectSync"] = {"sync_opt_for_notail_db": 1,
                                                    "sync_opt_for_preload_loop_zero": True}
            config.get("build_args")["enable_cce_licm_safe_hoist"] = True
            if not support_l0c2out:
                config.get("build_args")["InjectSync"] = {"sync_opt_for_notail_db": 2,
                                                        "sync_opt_for_preload_loop_zero": True}
                config.get("build_args")["enable_cce_licm_safe_hoist"] = False
        elif extra_params["op_type"] == "MatMulV2":
            config.get("build_args")["enable_cce_licm_safe_hoist"] = True
            config.get("build_args")["InjectSync"] = {"sync_opt_for_notail_db": 1,
                                                    "sync_opt_for_preload_loop_zero": True}
            if not support_l0c2out:
                config.get("build_args")["enable_cce_licm_safe_hoist"] = False
                config.get("build_args")["InjectSync"] = {"sync_opt_for_notail_db": 2,
                                                        "sync_opt_for_preload_loop_zero": True}
    return config


def _normal_compile(batch_matmul_inputs, kernel_name, extra_params):
    input_x1, input_x2, bias, offset_w, output_z, adj_x1, adj_x2, offset_x = batch_matmul_inputs
    with tbe.compute():
        res = batch_matmul_compute(
            input_x1, input_x2, bias, offset_w, output_z, adj_x1, adj_x2, offset_x, kernel_name, extra_params)
    tensor_list = res.get("op_placeholder") + res.get("op_res")
    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get("op_res"))
    compile_result = [res, tensor_list, sch]
    return compile_result


def _unaligned_compile(batch_matmul_inputs, kernel_name, extra_params):
    input_x1, input_x2, bias, offset_w, output_z, adj_x1, adj_x2, offset_x = batch_matmul_inputs
    unaligned_compile_result = []
    with tbe.compute():
        res_unaligned = batch_matmul_compute(
            input_x1, input_x2, bias=bias, offset_w=offset_w, output_z=output_z,
            adj_x1=adj_x1, adj_x2=adj_x2, offset_x=offset_x, kernel_name=kernel_name,
            extra_params=extra_params, unaligned_flag=True)
    if res_unaligned is not None:
        tensor_list_unaligned = res_unaligned.get("op_placeholder") + res_unaligned.get("op_res")
        with tvm.target.cce():
            sch_unaligned = tbe.auto_schedule(res_unaligned.get("op_res"))
        unaligned_compile_result = [tensor_list_unaligned, sch_unaligned]
    return unaligned_compile_result


def _judge_unaligned_int8(context, support_l0c2out):
    a_dtype = context.get_addition("input_a_dtype")
    if not support_l0c2out or (a_dtype != "int8"):
        return False
    block_size = 16

    block_reduce = tbe_platform.CUBE_MKN.get(a_dtype).get("mac")[1]
    trans_a = context.get_addition("trans_a")
    trans_b = context.get_addition("trans_b")
    format_a = context.get_addition("input_a_format")
    format_b = context.get_addition("input_b_format")
    shape_a = context.get_addition("input_a_shape")
    shape_b = context.get_addition("input_b_shape")
    m_dim = shape_a[-1] if trans_a else shape_a[-2]
    n_dim = shape_b[-2] if trans_b else shape_b[-1]
    # all nz int8 is unalign, for disable shift_inwards
    unalign = (format_a == "FRACTAL_NZ") or (format_b == "FRACTAL_NZ")
    # ND
    unalign = unalign or ((not trans_b) and (format_b == "ND") and (ceil_div(n_dim, block_size) % ALIGN_NUM_TWO != 0))
    unalign = unalign or (trans_a and (format_a == "ND") and (ceil_div(m_dim, block_size) % ALIGN_NUM_TWO != 0))
    return unalign


def _judge_unaligned(context, support_l0c2out):
    block_size = 16
    a_dtype = context.get_addition("input_a_dtype")
    block_reduce = tbe_platform.CUBE_MKN.get(a_dtype).get("mac")[1]
    shape_a = context.get_addition("input_a_shape")
    shape_b = context.get_addition("input_b_shape")
    trans_a = context.get_addition("trans_a")
    trans_b = context.get_addition("trans_b")
    m_dim = shape_a[-1] if trans_a else shape_a[-2]
    n_dim = shape_b[-2] if trans_b else shape_b[-1]
    k_dim = shape_a[-2] if trans_a else shape_a[-1]
    shape_out = context.get_addition("output_y_shape")
    unalign_int8 = _judge_unaligned_int8(context, support_l0c2out)
    if (m_dim % block_size != 0 or n_dim % block_size != 0 or k_dim % block_reduce != 0 or
        shape_out[-1] % block_size != 0 or shape_out[-2] % block_size != 0 or unalign_int8):
        return True
    return False


def _unsupport_mix(batch_matmul_inputs, extra_params, support_l0c2out):
    input_x1, input_x2, bias, _, output_z, _, _, _ = batch_matmul_inputs
    dtype_support_list = [("float16", "float16", "float16"), ("bfloat16", "bfloat16", "bfloat16")]
    if extra_params.get("fusion_type", "") != "nz_fusion":
        dtype_support_list += [("float32", "float32", "float32")]
    dtype_support = (input_x1.get("dtype").lower(), input_x2.get("dtype").lower(),
                     output_z.get("dtype").lower()) in dtype_support_list
    is_nd_in_out = input_x1.get("format") == "ND" and input_x2.get("format") == "ND" and output_z.get("format") == "ND"
    support_fixpipe_l0c2ub = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2ub")
    l2_mix_unsupport_bias = (bias is not None) and (extra_params.get("fusion_type", "") == "nz_fusion")
    unsupport_scenario = (not dtype_support or l2_mix_unsupport_bias
                          or (extra_params["op_type"] == "BatchMatMulV2")
                          or (not is_nd_in_out)
                          or (not support_l0c2out)
                          or support_fixpipe_l0c2ub)
    return unsupport_scenario


def _dynamic_build(batch_matmul_inputs, kernel_name, extra_params, support_l0c2out):
    input_x1, input_x2, bias, offset_w, output_z, adj_x1, adj_x2, offset_x = batch_matmul_inputs
    compile_result = _normal_compile(batch_matmul_inputs, kernel_name, extra_params)
    res, tensor_list, sch = compile_result
    cache_tiling_flag = res.get("cache_tiling_flag")
    tensor_lists, schs = [tensor_list], [sch]
    is_nd_in_out = input_x1.get("format") == "ND" and input_x2.get("format") == "ND" and output_z.get("format") == "ND"
    is_nz_in_nd_out = input_x1.get("format") == "FRACTAL_NZ" and input_x2.get("format") == "FRACTAL_NZ" \
        and output_z.get("format") == "ND"
    is_nz_quant = (input_x1.get("dtype") == "int8" and input_x1.get("format") == "FRACTAL_NZ" and
                   input_x2.get("format") == "FRACTAL_NZ" and support_l0c2out)
    is_nd_quant = (input_x1.get("dtype") == "int8" and input_x1.get("format") == "ND" and
                   input_x2.get("format") == "ND" and support_l0c2out)
    if cache_tiling_flag and ((is_nd_in_out and support_l0c2out) or is_nz_in_nd_out or is_nz_quant or is_nd_quant or
                              ((extra_params.get("nd2nz_type") == ComputeFlow.weight_nz.value) and support_l0c2out)):
        unaligned_compile_result = _unaligned_compile(batch_matmul_inputs, kernel_name, extra_params)
        if unaligned_compile_result:
            tensor_list_unaligned, sch_unaligned = unaligned_compile_result
            tensor_lists.append(tensor_list_unaligned)
            schs.append(sch_unaligned)
    unsupport_scenario = _unsupport_mix(batch_matmul_inputs, extra_params, support_l0c2out)
    if cache_tiling_flag and not unsupport_scenario and not input_x1.get("dtype") == "float32":
        for pad_ab in [PAD_B, PAD_A, PAD_AB]:
            extra_params["pad_flag"] = pad_ab
            compile_result = _normal_compile(batch_matmul_inputs, kernel_name, extra_params)
            _, pad_tensor_list, pad_sch = compile_result
            tensor_lists.append(pad_tensor_list)
            schs.append(pad_sch)
            pad_unaligned_compile_res = _unaligned_compile(batch_matmul_inputs, kernel_name, extra_params)
            if pad_unaligned_compile_res:
                pad_tensor_list_unaligned, pad_sch_unaligned = pad_unaligned_compile_res
                tensor_lists.append(pad_tensor_list_unaligned)
                schs.append(pad_sch_unaligned)
    # Dynamic Mode cannot support Nz-Fusion for now
    config_flag = [support_l0c2out, bias]
    config = _update_config(res, kernel_name, tensor_lists, config_flag, extra_params)
    tbe.build(schs, config)


def _is_invalid_mac_time(m_dim, k_dim, n_dim, dtype, threshold):
    default_cube_freq = 1800
    cube_freq_str = tbe_platform.get_soc_spec("CUBE_FREQ")
    cube_freq = int(cube_freq_str if cube_freq_str.isdigit() else default_cube_freq)
    aic_core_num = int(tbe_platform.get_soc_spec("CUBE_CORE_CNT"))
    block_reduce = tbe_platform.CUBE_MKN.get(dtype).get("mac")[1]
    current_mac_time = ceil_div(m_dim, BLOCK_CUBE) * ceil_div(k_dim, block_reduce) * \
                         ceil_div(n_dim, BLOCK_CUBE) // aic_core_num // cube_freq
    current_mac_time = current_mac_time * 2 if dtype == "float32" else current_mac_time
    return current_mac_time < threshold


def _pad_compare_mac_time(m_dim, k_dim, n_dim, dtype, bias):
    mac_time_threshold = 50
    if (bias is not None) and dtype == "float32":
        mac_time_threshold = 100
    return _is_invalid_mac_time(m_dim, k_dim, n_dim, dtype, mac_time_threshold)


def is_padfusion_large_outer(shape_dim, trans):
    m_dim, k_dim, n_dim = shape_dim
    trans_a, trans_b = trans
    if trans_a:
        large_outer_a = (k_dim // m_dim) > LARGE_OUTER_LIMIT
    else:
        large_outer_a = (m_dim // k_dim) > LARGE_OUTER_LIMIT

    if trans_b:
        large_outer_b = (n_dim // k_dim) > LARGE_OUTER_LIMIT
    else:
        large_outer_b = (k_dim // n_dim) > LARGE_OUTER_LIMIT
    return large_outer_a, large_outer_b


def _get_pad_flag_sub_func(shape_dim, trans, unaligned_flag, dtype, bias):
    m_dim, k_dim, n_dim = shape_dim
    align_size = SHAPE_ALIGN_FP32 if dtype == "float32" else PAD_ALIGN
    lower_shape = SHAPE_LOWER_FP32 if dtype == "float32" else SHAPE_LOWER
    pad_size = SHAPE_ALIGN_FP32 if dtype == "float32" else SHAPE_ALIGN
    if (dtype == "float32") and (bias is not None):
        lower_shape = SHAPE_LOWER_BIAS_FP32
    large_outer_a, large_outer_b = is_padfusion_large_outer(shape_dim, trans)
    m_unalign = (m_dim % align_size != 0 and m_dim > lower_shape and m_dim <= SHAPE_UPPER)
    n_unalign = (n_dim % align_size != 0 and n_dim > lower_shape and n_dim <= SHAPE_UPPER)
    k_unalign = (k_dim % align_size != 0 and k_dim > lower_shape and k_dim <= SHAPE_UPPER)
    trans_a, trans_b = trans
    if trans_a and not trans_b:
        m_unalign = (m_dim % pad_size != 0 and m_dim > lower_shape and m_dim <= SHAPE_UPPER)
        n_unalign = (n_dim % pad_size != 0 and n_dim > lower_shape and n_dim <= SHAPE_UPPER)
    m_k_k_n = (not trans_a and not trans_b and (n_unalign or k_unalign))
    m_k_n_k = (not trans_a and trans_b and k_unalign)
    k_m_k_n = (trans_a and not trans_b and (m_unalign or n_unalign))
    k_m_n_k = (trans_a and trans_b and (m_unalign or k_unalign))
    pad_flag = PAD_NONE
    m_pad, k_pad, n_pad = m_dim, k_dim, n_dim
    if m_k_k_n:
        enable_pad_k = k_unalign and (not large_outer_a)
        enable_pad_n = n_unalign and (not large_outer_b)
        n_pad = align(n_dim, pad_size) if enable_pad_n else n_dim
        k_pad = align(k_dim, pad_size) if enable_pad_k else k_dim
        pad_flag = PAD_B if enable_pad_n else PAD_NONE
        pad_flag += PAD_A if enable_pad_k else PAD_NONE
        unaligned_flag = n_unalign if (not large_outer_b) else unaligned_flag
    elif m_k_n_k:
        k_pad = align(k_dim, pad_size)
        pad_flag = PAD_AB
        unaligned_flag = False
    elif k_m_k_n:
        n_pad = align(n_dim, pad_size) if n_unalign else n_dim
        m_pad = align(m_dim, pad_size) if m_unalign else m_dim
        pad_flag = PAD_B if n_unalign else PAD_NONE
        pad_flag += PAD_A if m_unalign else PAD_NONE
        unaligned_flag = n_unalign or m_unalign
    elif k_m_n_k:
        k_pad = align(k_dim, pad_size) if k_unalign else k_dim
        m_pad = align(m_dim, pad_size) if m_unalign else m_dim
        pad_flag = PAD_B if k_unalign else PAD_NONE
        pad_flag += PAD_A if m_unalign else PAD_NONE
        unaligned_flag = m_unalign
    block_reduce = tbe_platform.CUBE_MKN.get(dtype).get("mac")[1]
    unaligned_flag = (unaligned_flag or
                      (m_pad % BLOCK_CUBE) != 0 or (k_pad % block_reduce) != 0 or (n_pad % BLOCK_CUBE) != 0)
    return pad_flag, unaligned_flag


def _get_m_k_n_dim(context):
    shape_a = context.get_addition("input_a_shape")
    shape_b = context.get_addition("input_b_shape")
    trans_a = context.get_addition("trans_a")
    trans_b = context.get_addition("trans_b")
    m_dim = shape_a[-1] if trans_a else shape_a[-2]
    k_dim = shape_a[-2] if trans_a else shape_a[-1]
    n_dim = shape_b[-2] if trans_b else shape_b[-1]
    return m_dim, k_dim, n_dim


def _get_pad_flag(context, batch_matmul_inputs, extra_params, support_l0c2out, unaligned_flag):
    if _unsupport_mix(batch_matmul_inputs, extra_params, support_l0c2out):
        return PAD_NONE, unaligned_flag
    pad_fusion_black_list = [[3456, 8192, 2560, True, False]]
    m_dim, k_dim, n_dim = _get_m_k_n_dim(context)
    trans_a = context.get_addition("trans_a")
    trans_b = context.get_addition("trans_b")
    dtype_a = context.get_addition("input_a_dtype")
    if [m_dim, k_dim, n_dim, trans_a, trans_b] in pad_fusion_black_list:
        return PAD_NONE, unaligned_flag
    # skip fp32 cases where k_dim is more than 100 times of max(m_dim, n_dim)
    is_fp32_splitk = dtype_a == "float32" and k_dim >= 100 * max(m_dim, n_dim)
    bias = batch_matmul_inputs[2]
    if is_fp32_splitk or _pad_compare_mac_time(m_dim, k_dim, n_dim, dtype_a, bias):
        return PAD_NONE, unaligned_flag
    return _get_pad_flag_sub_func([m_dim, k_dim, n_dim], [trans_a, trans_b], unaligned_flag, dtype_a, bias)


def _nz_fusion_invalid_scenario(context, dtype, support_l0c2out):
    shape_a = context.get_addition("input_a_shape")
    shape_b = context.get_addition("input_b_shape")
    a_use_nz = shape_a[-1] > SHAPE_UPPER
    b_use_nz = shape_b[-1] > SHAPE_UPPER
    if (not a_use_nz and not b_use_nz) or not support_l0c2out:
        return True
    m_dim, k_dim, n_dim = _get_m_k_n_dim(context)
    mac_time_threshold = 1 if dtype == "float32" else 10
    invalid_mac = _is_invalid_mac_time(m_dim, k_dim, n_dim, dtype, mac_time_threshold)
    return invalid_mac


def _binary_constant_build(context, batch_matmul_inputs, kernel_name, extra_params, support_l0c2out):
    input_x1, input_x2, bias, offset_w, output_z, adj_x1, adj_x2, offset_x = batch_matmul_inputs
    unaligned_flag = _judge_unaligned(context, support_l0c2out)
    extra_params["fusion_type"] = "nz_fusion"
    nz_fusion_flag, nz_fusion_mode = NO_NZ_FUSION, NO_NZ_FUSION
    pad_flag = PAD_NONE
    if not nz_fusion_flag:
        extra_params["fusion_type"] = "pad_fusion"
        pad_flag, unaligned_flag = _get_pad_flag(context, batch_matmul_inputs,
                                                 extra_params, support_l0c2out, unaligned_flag)
    extra_params["nz_fusion_flag"] = nz_fusion_flag
    extra_params["nz_fusion_mode"] = nz_fusion_mode
    context.add_addition("enable_nz_fusion", True if (nz_fusion_flag != 0) else False)
    if pad_flag:
        extra_params["pad_flag"] = pad_flag
        context.add_addition("enable_pad", pad_flag)
    if unaligned_flag:
        unaligned_compile_result = _unaligned_compile(batch_matmul_inputs, kernel_name, extra_params)
        tensor_list, sch = unaligned_compile_result
        config = {
            "print_ir": False,
            "name": kernel_name,
            "tensor_list": tensor_list,
            "build_args": {"constant_realize_extent_in_infer_bound": False}
        }
    else:
        compile_result = _normal_compile(batch_matmul_inputs, kernel_name, extra_params)
        res, tensor_list, sch = compile_result
        config_flag = [support_l0c2out, bias]
        config = _update_config(res, kernel_name, tensor_list, config_flag, extra_params)
    context.set_op_mode("static")
    config.get("build_args")["dynamic_shape"] = True
    config.get("build_args")["enable_db_fold"] = True
    tbe.build(sch, config)


def check_supported(input_x1, input_x2, bias, offset_w, output_z, adj_x1=False, adj_x2=False, offset_x=0):
    """
    check the op support situation
    """
    res = check_supported_mm_ub(input_x1, input_x2, bias, output_z)
    return res


@register_operator("BatchMatMulV2")
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
    para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_BOOL,
    para_check.REQUIRED_ATTR_BOOL, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def batch_matmul_v2(input_x1, input_x2, bias=None, offset_w=None, output_z=None,
                    adj_x1=False, adj_x2=False, offset_x=0, kernel_name="matmul"):
    """
    caculating matrix multiplication with bias, C = A * B + bias
    only support input with nz format and fp16 in dynamic mode

    Parameters:
    input_x1: dict
        A dict object, dict with keys(shape, dtype, and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ or ND
    input_x2: dict
        A dict object, dict with keys(shape, dtype and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ or ND
    bias: dict
        A dict object, dict with keys(shape and dtype) or None
        the dtype must be fp16
        the format must be ND
    offset_w: dict
        A dict object, dict with keys(shape and dtype) or None
    output_z: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ or ND
    adj_x1: bool
        If true, shape_a == transposed before multiplication
    adj_x2: bool
        If true, shape_b == transposed before multiplication
    offset_x: int
        offset of gradients in quant mode
    kernel_name: str
        cce kernel_name
    Returns
    -------
    res : dict
        None
    """
    reset_format(input_x1, input_x2, bias, output_z)
    extra_params = {"op_type": "BatchMatMulV2"}
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if support_l0c2out and input_x1.get("format") == "ND" and input_x2.get("format") == "ND":
        extra_params["nd2nz_type"] = ComputeFlow.on_the_fly.value

    batch_matmul_inputs = [input_x1, input_x2, bias, offset_w, output_z, adj_x1, adj_x2, offset_x]
    context = tbe_context.op_context.get_context()
    if context.get_addition("is_binary_constant") == 1:
        _binary_constant_build(context, batch_matmul_inputs, kernel_name, extra_params, support_l0c2out)
    else:
        _dynamic_build(batch_matmul_inputs, kernel_name, extra_params, support_l0c2out)
    tbe_platform.fusion_manager.set_current_op_pattern("BatchMatmul")
