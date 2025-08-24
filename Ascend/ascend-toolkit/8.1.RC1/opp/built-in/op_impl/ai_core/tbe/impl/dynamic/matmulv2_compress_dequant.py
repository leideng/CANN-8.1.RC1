#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
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
import copy
from typing import Optional

from impl.dynamic.batch_matmul_v2 import _check_empty_tensor
from impl.dynamic.batch_matmul_v2 import _check_soc_support
from impl.dynamic.batch_matmul_v2 import _get_bias_tensor
from impl.dynamic.batch_matmul_v2 import _get_k_n_index
from impl.dynamic.batch_matmul_v2 import _get_m_k_index
from impl.dynamic.batch_matmul_v2 import _get_real_trans
from impl.dynamic.batch_matmul_v2 import _set_shape
from impl.dynamic.batch_matmul_v2 import _update_config
from impl.dynamic.batch_matmul_v2 import check_and_config_para
from impl.dynamic.batch_matmul_v2 import get_ori_batch_shape
from impl.dynamic.ascend_dequant import ascend_dequant_compute
from impl.util.util_gemm import reset_format
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import WEIGHT_SPARSE_4_2
from impl.util.platform_adapter import WEIGHT_UNZIP
from impl.util.util_cube_dynamic import ceil_div
from tbe.common.platform import platform_info
from tbe.dsl.base.operation import get_te_var

BLOCK_CUBE = 16
BLOCK_CUBE_INT8 = 32


def define_cache_tiling_var() -> None:
    """
    define variables in cache tiling
    """
    def _create_var(list_var):
        for name_var in list_var:
            operation.var(name_var)

    list_var = ("batch_single_core", "m_single_core", "n_single_core", "batch_dim", "n_dim", "m_dim", "k_dim",
                "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor", "kal1_factor",
                "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch_l1_factor", "batch_ub_l0_time", "batch_cub",
                "out_branch_flag", "bias_flag", "hf32_flag", "datatype_bf16", "al1_db", "bl1_db", "l0c_db",
                "l2_cache_flag", "close_k_shift")
    _create_var(list_var)


def _construct_shape_x1_x2(info_x1_x2):
    shape_x1, shape_x2 = info_x1_x2.get("shapes")
    format_x1, format_x2 = info_x1_x2.get("formats")
    transpose_x1, transpose_x2, input_dtype = info_x1_x2.get("attrs")

    m_index, ka_index = _get_m_k_index(format_x1, transpose_x1)
    kb_index, n_index = _get_k_n_index(format_x2, transpose_x2)
    is_cache_tiling = True
    unaligned_flag = False
    shape_x1[ka_index], shape_x1[m_index], shape_x2[n_index], shape_x2[kb_index] = _set_shape(info_x1_x2,
        is_cache_tiling, unaligned_flag)
    shape_x1 = shape_x1 + [BLOCK_CUBE, BLOCK_CUBE_INT8]
    shape_x2 = shape_x2 + [BLOCK_CUBE, BLOCK_CUBE_INT8]
    return shape_x1, shape_x2


def matmul_compress_compute(input_x1: dict, input_x2: dict, output_z: dict, para_dict: dict,
                            unaligned_flag: bool = False):
    """
    matmul_compress_compute computer

    Parameters:
    input_x1: dict
    A dict object, dict with keys(shape, dtype and range)
    the dtype must be int8
    the format must be FRACTAL_NZ
    input_x2: dict
    A dict object, dict with keys(shape, dtype and range)
    the dtype must be int8
    the format must be FRACTAL_Z
    -------
    res : dict
    A dict object, dict with input tensor and output tensor
    """
    compress_index = para_dict.get("compress_index")
    bias = para_dict.get("bias")
    transpose_x1 = para_dict.get("trans_a")
    transpose_x2 = para_dict.get("trans_b")
    kernel_name = para_dict.get("kernel_name")
    extra_params = para_dict.get("extra_params")
    offset_x = para_dict.get("offset_a")
    offset_w = para_dict.get("offset_b")
    compress_info = para_dict.get("compress_info")
    alg = para_dict.get("alg")
    kernel_name = para_dict.get("kernel_name")
    op_type = para_dict.get("op_type")

    _check_empty_tensor(input_x1, input_x2, output_z, bias)
    _check_soc_support()

    is_cache_tiling, dtype_in, dtype_out, input_range, ori_input_range, shape_input = check_and_config_para(
        input_x1, input_x2, bias, output_z, transpose_x1, transpose_x2, kernel_name, op_type
    )

    # define var
    batch_range, m_range, k_range, n_range = input_range
    _, ori_m_range, ori_k_range, ori_n_range = ori_input_range
    operation.var("k_ori", ori_k_range)
    operation.var("m_ori", ori_m_range)
    operation.var("n_ori", ori_n_range)
    operation.var("k", k_range)
    operation.var("m", m_range)
    operation.var("n", n_range)

    if is_cache_tiling:
        define_cache_tiling_var()

    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    info_x1_x2 = {"shapes": shape_input, "formats": (format_a, format_b),
                  "attrs": (transpose_x1, transpose_x2, input_x1.get("dtype"))}
    shape_x1, shape_x2 = _construct_shape_x1_x2(info_x1_x2)
    tensor_x1 = tvm.placeholder(shape_x1, name="tensor_a", dtype=dtype_in)
    tensor_x2 = tvm.placeholder(shape_x2, name="tensor_b", dtype=dtype_in)
    operation.var("offset_x", [0, 0])
    index_size = operation.var("index_size", dtype="int32")
    dtype_compress_index = compress_index.get("dtype")
    ori_shape_compress_index = compress_index.get("ori_shape")
    attr_compress_index = {"ori_shape": ori_shape_compress_index}
    tensor_compress_index = tvm.placeholder(index_size, name="compress_index",
                                            dtype=dtype_compress_index, attrs=attr_compress_index)
    if is_cache_tiling:
        ori_batch_x1, ori_batch_x2 = get_ori_batch_shape(input_x1, input_x2)
        tensor_x1.op.attrs["ori_batch_shape"] = ori_batch_x1
        tensor_x2.op.attrs["ori_batch_shape"] = ori_batch_x2
    tensor_bias = _get_bias_tensor(bias, format_b, is_cache_tiling)
    transpose_x1, transpose_x2 = _get_real_trans(format_a, format_b, transpose_x1, transpose_x2)

    deq_vec_flag = True
    para_dict = {
        "trans_a": transpose_x1,
        "trans_b": transpose_x2,
        "format_a": format_a,
        "format_b": format_b,
        "format_out": output_z.get("format"),
        "dst_dtype": dtype_out,
        "tensor_c": tensor_bias,
        "cache_tiling_flag": is_cache_tiling,
        "unaligned_flag": unaligned_flag,
        "kernel_name": kernel_name,
        "op_type": "CompressMatMul",
        "input_range": input_range,
        "nd2nz_type": 0,
        "compress_index": tensor_compress_index,
        "offset_a": get_te_var("offset_x").get_tvm_var(),
        "offset_b": offset_w,
        "alg": alg,
        "deq_vec_flag": deq_vec_flag,
        "compress_info": compress_info
    }
    op_res = tbe.gemm(tensor_x1, tensor_x2, para_dict)

    tensor_list = [tensor_x1, tensor_x2]
    if compress_index:
        tensor_list.append(tensor_compress_index)

    return {"op_placeholder": tensor_list, "op_res": [op_res],
            "cache_tiling_flag": is_cache_tiling, "bias": tensor_bias}


def _normal_compile(input_x1, input_x2, output_z, para_dict):
    if not para_dict.get("deq_scale", None):
        reason = f"intput dequant scale must not be None."
        error_manager_vector.raise_err_specific_reson(op_type, reason)
    input_deq = para_dict.get("deq_scale", None)
    kernel_name = para_dict.get("kernel_name", None)
    output_matmul_compress = copy.deepcopy(output_z)
    if output_matmul_compress.get("dtype"):
        output_matmul_compress["dtype"] = "int32"
    with tbe.compute():
        res_matmul_compress = matmul_compress_compute(input_x1, input_x2, output_matmul_compress, para_dict)
        tensor_list = res_matmul_compress.get("op_placeholder")
        n_ori = get_te_var("n_ori").get_tvm_var()
        n = get_te_var("n").get_tvm_var()
        block_n0 = int(platform_info.get_soc_spec("cube_n_size"))
        ori_shape_deq = (1, 1, 1, n_ori)
        shape_deq = (1, n, 1, 1, block_n0)
        dtype_deq = input_deq.get("dtype")
        attr_deq = {"ori_shape": ori_shape_deq}
        tensor_deq = tvm.placeholder(shape_deq, name="deq_scale", dtype=dtype_deq, attrs=attr_deq)
        if input_deq:
            tensor_list.append(tensor_deq)
        bias = res_matmul_compress.get("bias")
        if bias is not None:
            tensor_list.append(bias)
        res_dequant = ascend_dequant_compute(res_matmul_compress.get("op_res")[0],
                                             tensor_deq, output_z, False, False, kernel_name)
    tensor_list.append(res_dequant)

    is_cache_tiling = res_matmul_compress.get("cache_tiling_flag", True)
    res = {"op_placeholder": tensor_list, "op_res": [res_dequant], "cache_tiling_flag": is_cache_tiling}

    with tvm.target.cce():
        sch = tbe.auto_schedule(res_dequant)
    compile_result = [res, tensor_list, sch]
    return compile_result


def _dynamic_build(input_x1, input_x2, output_y, para_dict):
    bias = para_dict.get("bias", None)
    extra_params = para_dict.get("extra_params", None)
    kernel_name = para_dict.get("kernel_name", None)
    compile_result = _normal_compile(input_x1, input_x2, output_y, para_dict)
    res, tensor_list, sch = compile_result
    tensor_lists, schs = [tensor_list], [sch]
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    config_flag = [support_l0c2out, bias]
    config = _update_config(res, kernel_name, tensor_lists, config_flag, extra_params)
    tbe.build(schs, config)


@register_operator("MatMulV2CompressDequant")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_BOOL, para_check.REQUIRED_ATTR_BOOL,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def matmulv2_compress_dequant(input_x1,
                            input_x2,
                            compress_index,
                            deq_scale,
                            bias,
                            offset_w=None,
                            output_y=None,
                            transpose_x1=False,
                            transpose_x2=False,
                            compress_info=None,
                            offset_x=0,
                            alg=WEIGHT_UNZIP,
                            kernel_name="compress_matmul_dequant"):
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
    compress_index: dict
        the dict of input compress index
    deq_scale: dict
        the dict of dequant num
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
    compress_info: listint
        [tiling_k, tiling_n, input_x2_ori_k, input_x2_ori_n, tiling_flag]
    kernel_name: str
        cce kernel name, default value is "compress_matmul_dequant"

    Returns
    -------
    None
    """
    if compress_info is None:
        compress_info = [1, 1, 1, 1, 1]
    reset_format(input_x1, input_x2, bias, output_y)
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if support_l0c2out:
        reason = f"not support support_l0c2out."
        error_manager_vector.raise_err_specific_reson(op_type, reason)

    extra_params = {"op_type": "CompressMatMul"}
    para_dict = {
        "compress_index": compress_index,
        "deq_scale": deq_scale,
        "bias": bias,
        "offset_a": offset_x,
        "offset_b": offset_w,
        "tensor_c": bias,
        "trans_a": transpose_x1,
        "trans_b": transpose_x2,
        "compress_info": compress_info,
        "alg": alg,
        "kernel_name": kernel_name,
        "op_type": "CompressMatMul",
        "extra_params": extra_params
    }
    _dynamic_build(input_x1, input_x2, output_y, para_dict)