#!/usr/bin/python
# -*- coding: utf-8 -*-
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
util_gemm
"""
import collections
from functools import reduce
import warnings
import json

from impl.matmul_vector import matmul_vector_cce as mm_vector_cce
from impl.batch_matmul_vector import matmul_vector_cce as bmm_vector_cce
import impl.dynamic as dyn_impl
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util.util_common import align


RANGE_MAX = 2147483647 # 2**31-1
BATCH_GEAR = [0, 1, 3, 7, 15, 31, RANGE_MAX]
SHAPE_GEAR_MATMUL_ND = [0, 16*3, 16*7, 16*15, 16*31, 16*63, 16*127, 16*191, 16*255,
                        16*511, 16*767, 16*1023, RANGE_MAX] # for fp16
DYNAMIC_DIM_VAL = -1
DYNAMIC_UNKNOWN_RANK = [-2]
NO_RANGE = (1, None)

ND_LENGTH = 2
NZ_LENGTH = 4
# 2 means L1 enable
L1FUSION_INPUT_CTR = 2
OFFSET_W_INDEX = 3


def _get_shape_gear(dim, shape_gear):
    pos = 1
    while(pos < len(shape_gear) and shape_gear[pos] < dim):
        pos += 1
    return (shape_gear[pos - 1] + 1, shape_gear[pos])


def cal_gemm_shape_range(shape, set_range_none=False):
    """
    cal gemm shape range
    """
    shape_range = []
    shape_len = len(shape)
    if not set_range_none:
        # shape like (batch1, ..., batchn, m, k)
        # process batch dim
        for i in range(0, shape_len - 2):
            if shape[i] > RANGE_MAX:
                return "LOWER_LIMIT"
            shape_range.append(_get_shape_gear(shape[i], BATCH_GEAR))

        # process m/k/n dim and bias
        for i in range(-min(shape_len, 2), 0):
            if shape[i] > RANGE_MAX:
                return "LOWER_LIMIT"
            shape_range.append(_get_shape_gear(shape[i], SHAPE_GEAR_MATMUL_ND))
    else:
        for _ in range(shape_len):
            shape_range.append((1, -1))

    return tuple(shape_range)


def cal_gemm_shape_binary(input_desc):
    """
    cal gemm shape when binary mode
    """
    shape_range = input_desc.get("range")
    if NO_RANGE in shape_range:
        return DYNAMIC_UNKNOWN_RANK
    return input_desc.get("shape")


def generalize_input_keep_rank_gemm(input_dict):
    """
    generalize input keep rank gemm
    """
    input_dict["ori_shape"] = [DYNAMIC_DIM_VAL] * len(input_dict["ori_shape"])


def is_graph_mode(tensor):
    """
    check whether is graph mode
    """
    # check graph mode or single mode in fuzzy compile
    if ((DYNAMIC_DIM_VAL in tensor.get("ori_shape") and "ori_range" in tensor.keys()) or
        list(tensor.get("ori_shape")) == DYNAMIC_UNKNOWN_RANK):
        return True
    return False


def matmul_range_check(input_x1, input_x2, bias):
    """
    check matmul range
    """
    x1_ori_range = input_x1.get("range")
    x2_ori_range = input_x2.get("range")
    input_list = [x1_ori_range, x2_ori_range]
    param_index_info = []
    type_info = []

    op_type = "MatMul" if len(input_x1.get("ori_shape")) == 2 else "BatchMatMul"
    if bias is not None:
        bias_ori_range = bias.get("range")
        input_list.append(bias_ori_range)

    if None in input_list:
        warnings.warn("{}, input list has none object".format(op_type))

    unknown_range = (list(input_x1.get("ori_shape")) == DYNAMIC_UNKNOWN_RANK or
                     list(input_x2.get("ori_shape")) == DYNAMIC_UNKNOWN_RANK)
    if unknown_range:
        # if x1 and x2 are -2, fe should excute static compile
        warnings.warn("{}, input x1 and input_x2 should not be -2".format(op_type))
        type_info = ["lower_limit"]

    for idx, item in enumerate(input_list):
        for range_val in item:
            # if upper range exceed int32 or -1, return upper_limit
            if range_val[1] is None or range_val[1] > RANGE_MAX:
                param_index_info.append(idx)
                type_info.append("upper_limit")
                warnings.warn("{}, if range is none or exceed int32, it's upper limit".format(op_type))
            # if lower range exceed int32, return lower_limit
            if range_val[1] is not None and (range_val[0] > RANGE_MAX or range_val[0] > range_val[1]):
                param_index_info.append(idx)
                type_info.append("lower_limit")
                warnings.warn("{}, if lower range exceed int32 or be larger than upper limit, "
                              "it's lower limit".format(op_type))
    if type_info:
        if "lower_limit" in type_info:
            # for lower_limit, fe should excute static compile
            param_index_info = list(range(len(input_list)))
            type_info = len(input_list) * ["lower_limit"]
        json_info = [{"result": "UNSUPPORTED", "reason": {"param_index": param_index_info, "type": type_info}}]
        return False, json_info
    return True, []


# supported info for mm, bmm, fc, gemm
def _cal_min_l1space(weight, op_type="gemm"):
    """
    cal the mini l1_size in BL1
    """
    dtype_b = weight.get("dtype")
    format_b = weight.get("format")
    block_reduce = tbe_platform.CUBE_MKN[dtype_b]["mac"][1]
    block_out = tbe_platform.CUBE_MKN[dtype_b]["mac"][2]
    mini_l1space = block_out * block_reduce * util_common.BIT_RATIO_DICT.get(dtype_b)
    if op_type == "gemm" and format_b == "ND" and dtype_b == "int8":
        mini_l1space *= 2
    return mini_l1space


def _get_split_batch_axis(a_shape, b_shape, batch_dims):
    """
    get the split info of batch dims
    """
    batch_len_a, batch_len_b, batch_len = batch_dims
    axis_split_matrix_batch = []
    for i in range(batch_len):
        batch_a_dim = a_shape[i] if batch_len_a >= batch_len - i else None
        batch_b_dim = b_shape[i] if batch_len_b >= batch_len - i else None
        if not batch_a_dim and not batch_b_dim:
            continue
        elif batch_a_dim and batch_b_dim:
            if batch_a_dim == batch_b_dim:
                batch_split_list = [[0, [i], [-1], [-1]], [1, [i], [-1], [-1]]]
            else:
                batch_split_list = [[0, [i], [-1], [-1]]] if batch_a_dim != 1 else [[1, [i], [-1], [-1]]]
        elif batch_a_dim:
            batch_split_list = [[0, [i], [-1], [-1]]]
        else:
            batch_split_list = [[1, [i], [-1], [-1]]]
        axis_split_matrix_batch.append(
            [util_select_op_base.SplitInput(*batch_split_list), util_select_op_base.SplitOutput([0, [i]])]
        )
    return axis_split_matrix_batch


def _get_split_mn_axis(bias, y, trans_flag, batch_dims, op_type):
    """
    get the split info of m and n
    """
    batch_len_a, batch_len_b, batch_len = batch_dims
    # cut m
    m_in_dim, mk_in_dim = (batch_len_a + 1, batch_len_a) if trans_flag[0] else (batch_len_a, batch_len_a + 1)
    m_split_list = [[0, [m_in_dim], [-1], [-1]]]
    mk_split_list = [0, [mk_in_dim]]
    # cut n
    n_in_dim, nk_in_dim = (batch_len_b, batch_len_b + 1) if trans_flag[1] else (batch_len_b + 1, batch_len_b)
    n_split_list = [[1, [n_in_dim], [-1], [-1]]]
    nk_split_list = [1, [nk_in_dim]]
    if op_type == "gemm":
        # gemm cannot split reduce dim
        axis_reduce_list = None
        c_m_dim, c_n_dim = (1, 0) if bias.get("format") == "FRACTAL_NZ" else (0, 1)
        m_split_list.append([2, [c_m_dim], [-1], [-1]])
        n_split_list.append([2, [c_n_dim], [-1], [-1]])
    elif bias:
        bias_dim = 3 if "compress" in op_type else 2
        axis_reduce_list = None
        n_split_list.append([bias_dim, [0], [-1], [-1]])
    else:
        # cut k_dim which is reduce dim
        axis_reduce_list = [[util_select_op_base.ReduceInput(mk_split_list, nk_split_list),
                             util_select_op_base.ReduceOutput([0, 1, False])]]
    out_m_idx, out_n_idx = (batch_len + 1, batch_len) if y.get("format") == "FRACTAL_NZ" else (batch_len, batch_len + 1)
    axis_split_matrix_a = [
        [util_select_op_base.SplitInput(*m_split_list),
         util_select_op_base.SplitOutput([0, [out_m_idx]])],
    ]
    axis_split_matrix_b = [
        [util_select_op_base.SplitInput(*n_split_list),
         util_select_op_base.SplitOutput([0, [out_n_idx]])],
    ]
    return axis_split_matrix_a, axis_split_matrix_b, axis_reduce_list


def get_op_support_info_gemm(inputs, y, trans_a, trans_b, op_type="mat_mul"):
    """
    get the mm/bmm/gemm split, which only split the m, n and batch, cannot cut k with bias

    """
    format_a, format_b = inputs[0].get("format"), inputs[1].get("format")
    a_shape, b_shape = inputs[0].get("ori_shape"), inputs[1].get("ori_shape")
    if format_a == 'FRACTAL_NZ':
        trans_a = not trans_a
    if format_b == 'FRACTAL_NZ':
        trans_b = not trans_b

    batch_len_a = len(a_shape) - ND_LENGTH
    batch_len_b = len(b_shape) - ND_LENGTH
    if list(a_shape) == DYNAMIC_UNKNOWN_RANK:
        batch_len_a = 1
    if list(b_shape) == DYNAMIC_UNKNOWN_RANK:
        batch_len_b = 1
    batch_len =  max(batch_len_a, batch_len_b)
    # in gemm, without batch dims
    if op_type == "gemm":
        batch_len_a = batch_len_b = batch_len = 0
    # cut m, n and reduce k
    axis_split_matrix_a, axis_split_matrix_b, axis_reduce_list = \
        _get_split_mn_axis(inputs[2], y, [trans_a, trans_b], [batch_len_a, batch_len_b, batch_len], op_type)
    # cut batch
    axis_split_matrix_batch = _get_split_batch_axis(a_shape, b_shape, [batch_len_a, batch_len_b, batch_len])

    axis_split_matrix = axis_split_matrix_a + axis_split_matrix_b + axis_split_matrix_batch
    min_l1space = _cal_min_l1space(inputs[1], op_type)
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, min_l1space)

    return op_cal_info_in_json


def get_op_support_info_fc(inputs, y, axis, op_type="fully_connection"):
    """
    get the fc split, which only split the m, n and batch, cannot cut k with bias

    """
    input_x, input_w, input_b = inputs
    format_x = input_x.get('format')
    format_y = y.get('format')
    n_split_list = [[1, [1], [-1], [-1]]]
    if input_b is not None:
        bias_dim = 3 if "compress" in op_type else 2
        n_split_list.append([bias_dim, [1], [-1], [-1]])
        axis_reduce_list = None
    else:
        km_reduce_list, kn_reduce_list = [0, [1]], [1, [0]]
        if axis != 2 and format_x != 'NC1HWC0':
            km_reduce_list, kn_reduce_list = [0, [0]], [1, [0]]
        axis_reduce_list = [[util_select_op_base.ReduceInput(km_reduce_list, kn_reduce_list),
                             util_select_op_base.ReduceOutput([0, 1, False])]]
    if axis == 2:
        b_split_list = [0, [0], [-1], [-1]]
        m_split_list = [0, [2], [-1], [-1]]
        axis_split_matrix = [
            [util_select_op_base.SplitInput(b_split_list),
             util_select_op_base.SplitOutput([0, [0]])],
            [util_select_op_base.SplitInput(m_split_list),
             util_select_op_base.SplitOutput([0, [2]])],
            [util_select_op_base.SplitInput(*n_split_list),
             util_select_op_base.SplitOutput([0, [1]])]
        ]
    else:
        m_in_axis = 0 if format_x == "NC1HWC0" else 1
        m_split_list = [0, [m_in_axis], [-1], [-1]]
        m_out_axis, n_out_axis = (0, 1) if format_y == "NC1HWC0" else (1, 0)
        axis_split_matrix = [
            [util_select_op_base.SplitInput(m_split_list),
             util_select_op_base.SplitOutput([0, [m_out_axis]])],
            [util_select_op_base.SplitInput(*n_split_list),
             util_select_op_base.SplitOutput([0, [n_out_axis]])]
        ]
    min_l1space = _cal_min_l1space(input_w)
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, min_l1space)

    return op_cal_info_in_json


def reset_format(input_x1, input_x2, bias, output_y):
    for i in (input_x1, input_x2, bias, output_y):
        if isinstance(i, dict) and i.get("format") in ("NHWC", "NCHW", "HWCN", "NCDHW", "NDHWC"):
            i["format"] = "ND"
        if isinstance(i, tvm.Tensor) and i.op.attrs and "format" in i.op.attrs:
            if i.op.attrs["format"] in ("NHWC", "NCHW", "HWCN", "NCDHW", "NDHWC"):
                i.op.attrs["format"] = "ND"


def reset_dtype(input_x1, input_x2, output_y):
    for i_dict in (input_x1, input_x2, output_y):
        if i_dict["dtype"] == "bfloat16":
            i_dict["dtype"] = "float16"


def gemm_compute(input_a, input_b, output_y, para_dict, mode="static"):
    """
    the compute of mm/bmm/gemm in static and dynamic
    """
    if output_y is None:
        output_y = {}
    bias = para_dict.get("tensor_c")
    reset_format(input_a, input_b, bias, output_y)
    format_a = input_a.op.attrs.get("format") if input_a.op.attrs.get("format") else input_a.op.attrs.get("format_")
    format_b = input_b.op.attrs.get("format") if input_b.op.attrs.get("format") else input_b.op.attrs.get("format_")

    if format_a == 'FRACTAL_NZ':
        para_dict["trans_a"] = not para_dict.get("trans_a")
    if format_b == 'FRACTAL_NZ':
        para_dict["trans_b"] = not para_dict.get("trans_b")
    format_out = output_y.get("format")
    if mode == "static" and para_dict.get("op_type") in ("BatchMatMul", "BatchMatMulV2"):
        # get shape and batch_shape from input and output
        shape_a = input_a.op.attrs["current_shape"] if "current_shape" in input_a.op.attrs else input_a.shape
        shape_b = input_b.op.attrs["current_shape"] if "current_shape" in input_b.op.attrs else input_b.shape
        shape_out = output_y.get("shape")
        batch_shape_a = shape_a[:-2] if format_a in ("ND",) else shape_a[:-4]
        batch_shape_b = shape_b[:-2] if format_b in ("ND",) else shape_b[:-4]
        batch_shape_out = shape_out[:-2] if format_out in ("ND",) else shape_out[:-4]
        # ori_batch_shape_out is used as batch_shape in bmm_elemwise fusion
        ori_batch_shape_out = batch_shape_out
        if len(batch_shape_a) == 1 and len(batch_shape_b) == 1 and len(batch_shape_out) > 1:
            # batch_shape_out is reduced to maske sure bmm+confusionTranspose fusion functions well
            batch_shape_out = [reduce(lambda x, y: x * y, list(batch_shape_out))]
        # ori_batch_shape means batch_shape before reduced
        input_a.op.attrs["ori_batch_shape"] = batch_shape_a
        input_b.op.attrs["ori_batch_shape"] = batch_shape_b
        para_dict.update({"batch_shape_a": batch_shape_a,
                          "batch_shape_b": batch_shape_b,
                          "batch_shape_out": batch_shape_out,
                          "ori_batch_shape_out": ori_batch_shape_out})
    if mode == "dynamic":
        # set fusion build config
        build_cfg = {'constant_realize_extent_in_infer_bound': False}
        tbe_register.set_fusion_buildcfg("gemm_op", build_cfg)

    if para_dict.get("offset_b") is not None:
        error_manager_vector.raise_err_specific_reson("mat_mul", "input offset_w must be None!")
    para_dict.update(
        {"format_a": format_a,
         "format_b": format_b,
         "dst_dtype": output_y.get("dtype").lower(),
         "format_out": format_out}
    )
    return tbe.gemm(tensor_a=input_a, tensor_b=input_b, para_dict=para_dict)


def _gemm_para_check(input_a, input_b, para_dict):
    """
    check the input and output of mm/bmm
    """
    ori_shape_a = list(input_a.get("ori_shape"))
    ori_shape_b = list(input_b.get("ori_shape"))
    if ori_shape_a is not None and len(ori_shape_a) < ND_LENGTH:
        ori_shape_a.insert(0, 1)
    if input_b.get("format") == "FRACTAL_ZN_RNN":
        input_b["format"] = "FRACTAL_Z"
        shape_b =  list(input_b.get("shape"))
        ori_shape_b = [shape_b[0]*16, shape_b[1]*16]
        input_b["ori_shape"] = ori_shape_b
    if ori_shape_b is not None and len(ori_shape_b) < ND_LENGTH:
        ori_shape_b.append(1)

    # the bias may be 2 dims, for matmulreshapeadd fusion pass
    bias = para_dict.get("tensor_c")
    ori_shape_bias, shape_bias, shape_bias_align = [], [], []
    if bias is not None and bool(bias):
        ori_shape_bias_len = reduce(lambda x, y: x * y, list(bias.get("ori_shape")))
        shape_bias_len = reduce(lambda x, y: x * y, list(bias.get("shape")))
        ori_shape_bias = [ori_shape_bias_len]
        shape_bias = [shape_bias_len]
        shape_bias_align = [align(shape_bias_len, 16)]
    if para_dict.get("op_type") not in ("CompressMatMul", ):
        # check the input shape
        para_check.check_shape(ori_shape_a, param_name="input_a")
        para_check.check_shape(ori_shape_b, param_name="input_b")
        if para_dict.get("op_type") in ("MatMulV2", "MatMul"):
            if len(ori_shape_a) != len(ori_shape_b) or len(ori_shape_a) != 2:
                error_detail = "length of input shape must be 2"
                error_manager_vector.raise_err_input_shape_invalid("mat_mul", "input", error_detail)
        else:
            if max(len(ori_shape_a), len(ori_shape_b)) < 2:
                error_detail = "shape length for batch matmul greater than or equal to 2"
                error_manager_vector.raise_err_input_shape_invalid('batch_matmul', 'input', error_detail)
        km_shape = ori_shape_a[-2] if para_dict.get("trans_a") else ori_shape_a[-1]
        if para_dict.get("trans_b"):
            n_shape, kn_shape = ori_shape_b[-2], ori_shape_b[-1]
        else:
            n_shape, kn_shape = ori_shape_b[-1], ori_shape_b[-2]
        if km_shape != kn_shape:
            error_detail = "reduce axis of x1 and x2 should be same"
            error_manager_vector.raise_err_two_input_shape_invalid("mat_mul", "x1", "x2", error_detail)
        if ori_shape_bias and ori_shape_bias[0] != n_shape:
            error_detail = "broadcast bias shape must be equal to shape n"
            error_manager_vector.raise_err_input_shape_invalid("mat_mul", "bias", error_detail)

    return [ori_shape_a, ori_shape_b, shape_bias, shape_bias_align]


def _get_input_shape(input_a, input_b):
    """
    get the shape for placeholder
    """
    shape_a = list(input_a.get("shape"))
    shape_b = list(input_b.get("shape"))
    if input_a.get("format") in ("FRACTAL_NZ", "FRACTAL_Z"):
        if len(shape_a) > NZ_LENGTH:
            batch_shape_a = reduce(lambda x, y: x * y, shape_a[:-4])
            shape_a = [batch_shape_a, ] + list(shape_a[-4:])
    else:
        input_a["format"] = "ND"
        if len(shape_a) > ND_LENGTH:
            batch_shape_a = reduce(lambda x, y: x * y, shape_a[:-2])
            shape_a = [batch_shape_a, ] + list(shape_a[-2:])
    if input_b.get("format") in ("FRACTAL_NZ", "FRACTAL_Z"):
        if len(shape_b) > NZ_LENGTH:
            batch_shape_b = reduce(lambda x, y: x * y, shape_b[:-4])
            shape_b = [batch_shape_b, ] + list(shape_b[-4:])
    else:
        input_b["format"] = "ND"
        if len(shape_b) > ND_LENGTH:
            batch_shape_b = reduce(lambda x, y: x * y, shape_b[:-2])
            shape_b = [batch_shape_b, ] + list(shape_b[-2:])
    return shape_a, shape_b


def _matmul_vector_one_compute(tensor_a, tensor_b, tensor_bias, axis):
    """
    algorithm: _matmul_vector_one
    calculating  matrix multiplication with bias, use vector mode ,C = A*B + bias

    Parameters
    ----------
    tensor_a: TVM tensor
        The dtype support "float32", "int32".
    tensor_b: TVM tensor
        The dtype support "float32", "int32".
    tensor_bias: TVM tensor
        The dtype support "float32", "int32".
    axis: int
        the axis for reduce.

    Returns
    -------
    res: TVM tensor
        output tensor. has the same type as tensor_a.
    """
    dtype = tensor_a.dtype
    shape_a = shape_util.shape_to_list(tensor_a.shape)
    shape_b = shape_util.shape_to_list(tensor_b.shape)
    if tensor_bias is not None:
        shape_bias = shape_util.shape_to_list(tensor_bias.shape)

    shape_a, shape_b, shape_max = \
        shape_util.broadcast_shapes(shape_a, shape_b, param_name_input1="tensor_a", param_name_input2="tensor_b")
    tensor_b = tbe.broadcast(tensor_b, shape_max, dtype)
    res_tmp = tbe.vmul(tensor_a, tensor_b)
    res = tbe.reduce_sum(res_tmp, axis=axis)

    shape_res = shape_util.shape_to_list(res.shape)
    if tensor_bias is not None:
        shape_res, shape_bias, shape_max2 = \
            shape_util.broadcast_shapes(shape_res, shape_bias, param_name_input1="res", param_name_input2="bias")
        tensor_bias = tbe.broadcast(tensor_bias, shape_max2, dtype)
        res = tbe.vadd(res, tensor_bias)

    return res


def _matmul_vector_one(shape_a, shape_b, para_dict, src_type):
    """
    algorithm: _matmul_vector_one
    calculating  matrix multiplication with bias, use vector mode ,C = A*B + bias

    Parameters
    ----------
    shape_a : list or tuple
        shape of tensor_a
    shape_b : list or tuple
        shape of tensor_b
    para_dict : dict
        para dict of matmul

    Returns
    -------
    None
    """
    axis = 0 if para_dict.get("trans_a") else 1
    if para_dict.get("trans_a") == para_dict.get("trans_b"):
        shape_b = (shape_b[1], shape_b[0])

    tensor_a = tvm.placeholder(shape_a, name='tensor_a', dtype=src_type)
    tensor_b = tvm.placeholder(shape_b, name='tensor_b', dtype=src_type)
    bias = para_dict.get("tensor_c")
    if bias is not None and bool(bias):
        shape_bias = list(bias.get("shape"))
        tensor_bias = tvm.placeholder(shape_bias, name='tensor_bias', dtype=src_type)
    else:
        tensor_bias = None

    result = _matmul_vector_one_compute(tensor_a, tensor_b, tensor_bias, axis)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)

    tensor_list = [tensor_a, tensor_b, result]
    if tensor_bias is not None:
        tensor_list = [tensor_a, tensor_b, tensor_bias, result]

    config = {"print_ir": False,
              "name": para_dict.get("kernel_name"),
              "tensor_list": tensor_list}
    tbe.build(schedule, config)


def _get_shape_info(tensor):
    range_dim = []
    dynamic_shape = []
    dynamic_ori_shape = []
    for _, _ in enumerate(tensor.get("shape")):
        range_dim.append((1, None))
        dynamic_shape.append(-1)
    for _, _ in enumerate(tensor.get("ori_shape")):
        dynamic_ori_shape.append(-1)
    return range_dim, dynamic_shape, dynamic_ori_shape


def _binary_constant_branch(input_a, input_b, output_y, para_dict, result):
    """
    the mm/bmm use dynamic process
    """
    context = tbe_context.op_context.get_context()
    context.set_op_mode("dynamic")

    context.add_addition("input_a_shape", input_a.get("shape"))
    context.add_addition("input_b_shape", input_b.get("shape"))
    context.add_addition("output_y_shape", output_y.get("shape"))

    context.add_addition("input_a_ori_shape", input_a.get("ori_shape"))
    context.add_addition("input_b_ori_shape", input_b.get("ori_shape"))
    context.add_addition("output_y_ori_shape", output_y.get("ori_shape"))

    context.add_addition("input_a_dtype", input_a.get("dtype"))
    context.add_addition("input_b_dtype", input_b.get("dtype"))
    context.add_addition("output_y_dtype", output_y.get("dtype"))

    context.add_addition("input_a_format", input_a.get("format"))
    context.add_addition("input_b_format", input_b.get("format"))
    context.add_addition("output_y_format", output_y.get("format"))

    context.add_addition("trans_a", para_dict.get("trans_a"))
    context.add_addition("trans_b", para_dict.get("trans_b"))

    bias = para_dict.get("tensor_c")
    if bias is not None and bool(bias):
        context.add_addition("bias_shape", bias.get("shape"))
        context.add_addition("bias_dtype", bias.get("dtype"))
        context.add_addition("bias_format", bias.get("format"))
        context.add_addition("bias_ori_shape", bias.get("ori_shape"))
    context.add_addition("is_binary_constant", 1)

    range_dim_a, dynamic_shape_a, dynamic_ori_shape_a = _get_shape_info(input_a)
    range_dim_b, dynamic_shape_b, dynamic_ori_shape_b = _get_shape_info(input_b)
    range_dim_y, dynamic_shape_y, dynamic_ori_shape_y = _get_shape_info(output_y)

    input_a["range"] = range_dim_a
    input_b["range"] = range_dim_b
    output_y["range"] = range_dim_y

    input_a["ori_shape"] = dynamic_ori_shape_a
    input_b["ori_shape"] = dynamic_ori_shape_b
    output_y["ori_shape"] = dynamic_ori_shape_y

    input_a["shape"] = dynamic_shape_a
    input_b["shape"] = dynamic_shape_b
    output_y["shape"] = dynamic_shape_y

    if bias is not None and bool(bias):
        range_dim_bias, dynamic_shape_bias, dynamic_ori_shape_bias = _get_shape_info(bias)
        bias["range"] = range_dim_bias
        bias["shape"] = dynamic_shape_bias
        bias["ori_shape"] = dynamic_ori_shape_bias
    mm_format_shape = {"FRACTAL_NZ": 4, "ND": 2}
    format_check = output_y.get("format") == "ND" and input_b.get("format") == "FRACTAL_NZ"
    # bmm not support binary constant in cvcoupling scene, mm support nznznd and ndnznd,
    # when bmm's input has no batch, run as mm
    if para_dict.get("op_type") in ("MatMulV2", "MatMul") or (format_check and
        len(output_y.get("shape")) == mm_format_shape.get(output_y.get("format"))):
        context.add_addition("binary_constant_type", "matmul")
        dyn_impl.mat_mul(input_a, input_b, bias, para_dict.get("offset_b"), output_y, para_dict.get("trans_a"),
                         para_dict.get("trans_b"), para_dict.get("offset_a"), para_dict.get("kernel_name"))
    elif para_dict.get("op_type") in ("BatchMatMul", "BatchMatMulV2"):
        dyn_impl.batch_matmul(input_a, input_b, bias, output_y, para_dict.get("trans_a"),
                              para_dict.get("trans_b"), para_dict.get("kernel_name"))


def gemm_impl(input_a, input_b, output_y, para_dict, mode="static"):
    """
    the impl of mm/bmm/gemm in static and dynamic
    """
    bias = para_dict.get("tensor_c")
    reset_format(input_a, input_b, bias, output_y)
    ori_shape_a, ori_shape_b, shape_bias, shape_bias_align = _gemm_para_check(input_a, input_b, para_dict)
    dtype_a = input_a.get("dtype").lower()
    dtype_b = input_b.get("dtype").lower()

    # use vector to cal matmul
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if get_use_vector_core_flag(input_a, support_l0c2out):
        if para_dict.get("op_type") in ("MatMulV2", "MatMul"):
            shape_n_dim = ori_shape_b[0] if para_dict.get("trans_b") else ori_shape_b[1]
            if shape_n_dim == 1:
                _matmul_vector_one(ori_shape_a, ori_shape_b, para_dict, dtype_a)
            else:
                mm_vector_cce(ori_shape_a, ori_shape_b, para_dict, dtype_a, shape_bias_align)
        else:
            bmm_vector_cce(ori_shape_a, ori_shape_b, para_dict, dtype_a, shape_bias)
        return
    #fix shape while k is zero
    ka_dict = {"ND_True": -2, "ND_False": -1, "FRACTAL_NZ_True": -3, "FRACTAL_NZ_False": -4}
    kb_dict = {"ND_True": -1, "ND_False": -2, "FRACTAL_NZ_True": -4, "FRACTAL_NZ_False": -3}
    input_a_attr = str(input_a.get("format")) + "_" + str(para_dict.get("trans_a"))
    input_b_attr = str(input_b.get("format")) + "_" + str(para_dict.get("trans_b"))
    idx_k_of_a = ka_dict.get(input_a_attr)
    idx_k_of_b = kb_dict.get(input_b_attr)
    shape_k_dim = input_a.get("shape")[idx_k_of_a]
    if shape_k_dim == 0:
        new_a_shape = list(input_a.get("shape"))
        new_b_shape = list(input_b.get("shape"))
        new_a_shape[idx_k_of_a] = 16 if input_a.get("format") == "ND" else 1
        new_b_shape[idx_k_of_b] = 16 if input_b.get("format") == "ND" else 1
        input_a["shape"] = tuple(new_a_shape)
        input_b["shape"] = tuple(new_b_shape)
        para_dict["zero_flag"] = True
    shape_a, shape_b = _get_input_shape(input_a, input_b)
    format_out = output_y.get("format")
    if support_l0c2out and format_out == "ND" and para_dict.get("op_type") in ("MatMulV2", "MatMul"):
        if input_a.get("format") == "ND" and shape_a[-1] == 1 and para_dict.get("trans_a"):
            trans_a_shape = (1, shape_a[-2])
            para_dict["trans_a"] = False
            input_a["shape"] = trans_a_shape
            input_a["ori_shape"] = trans_a_shape
            shape_a = list(trans_a_shape)
        if input_b.get("format") == "ND" and shape_b[-1] == 1 and not para_dict.get("trans_b"):
            trans_b_shape = (1, shape_b[-2])
            para_dict["trans_b"] = True
            input_b["shape"] = trans_b_shape
            input_b["ori_shape"] = trans_b_shape
            shape_b = list(trans_b_shape)
    batchmm_2_mm = support_l0c2out and format_out == "ND" and input_a.get("format") == "ND" and \
        input_b.get("format") == "ND" and para_dict.get("op_type") in ("BatchMatMul", "BatchMatMulV2")
    if batchmm_2_mm:
        double_input_b_size = reduce(lambda x, y: x * y, shape_b) * util_common.BIT_RATIO_DICT.get(dtype_b) * 2
        b_l1_fullload = len(shape_a) > 2 and len(shape_b) == 2 and para_dict.get("trans_a") == False and \
            double_input_b_size <= tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
        if b_l1_fullload:
            batch_m = reduce(lambda x, y: x * y, shape_a[:-1])
            shape_a = [batch_m, shape_a[-1]]
            input_a["shape"] = shape_a
            input_a["ori_shape"] = shape_a
            shape_out = output_y.get("shape")
            shape_out = [batch_m, shape_out[-1]]
            output_y["shape"] = shape_out
            output_y["ori_shape"] = shape_out
    tensor_a = tvm.placeholder(shape_a, name='tensor_a',
                               dtype=dtype_a,
                               attrs={'format': input_a.get("format"),
                                      'ori_shape': input_a.get("ori_shape"),
                                      'current_shape': input_a.get("shape"),
                                      'ori_format': input_a.get("ori_format")})
    tensor_b = tvm.placeholder(shape_b, name='tensor_b',
                               dtype=dtype_b,
                               attrs={'format': input_b.get("format"),
                                      'ori_shape': input_b.get("ori_shape"),
                                      'current_shape': input_b.get("shape"),
                                      'ori_format': input_b.get("ori_format")})
    compress_index, tensor_bias = None, None
    if para_dict.get("op_type") in ("CompressMatMul", ):
        if para_dict.get("alg") == "weight_sparse_4_2":
            compress_index = tvm.placeholder(para_dict.get("compress_index").get("shape"),
                                             name='compress_index', dtype="int8")
        else:
            index_size = tvm.var("index_size", dtype="int32")
            compress_index = tvm.placeholder([index_size, ],
                                             name='compress_index', dtype="int8")
        para_dict["compress_index"] = compress_index
    if shape_bias:
        tensor_bias = tvm.placeholder(shape_bias if support_l0c2out else shape_bias_align, name='tensor_bias',
                                      dtype=bias.get("dtype").lower(),
                                      attrs={'ori_shape': bias.get("ori_shape"),
                                             'format': bias.get("format"),
                                             'ori_format': bias.get("ori_format")})
    para_dict["tensor_c"] = tensor_bias

    result = gemm_compute(tensor_a, tensor_b, output_y, para_dict, mode)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)

    # skip static compilation
    attr_is_binary_constant = dict(result.op.attrs.items()).get("is_binary_constant", 0)
    if int(attr_is_binary_constant) == 1:
        para_dict["tensor_c"] = bias
        _binary_constant_branch(input_a, input_b, output_y, para_dict, result)
        return

    tensor_list = [tensor_a, tensor_b]
    if compress_index is not None:
        tensor_list.append(compress_index)
    if tensor_bias is not None:
        tensor_list.append(tensor_bias)
    if para_dict.get("op_type") in ("MatMulV2", "MatMul"):
        real_outs = schedule.cce_special["real_out_tensor"]
    else:
        real_outs = [result]
    tensor_list += real_outs
    config = {"print_ir": False,
              "name": para_dict.get("kernel_name"),
              "tensor_list": tensor_list}
    tbe.build(schedule, config)
    if para_dict.get("op_type") in ("BatchMatMul", "BatchMatMulV2"):
        tbe_platform.fusion_manager.set_current_op_pattern("BatchMatmul")


def get_cache_tiling_flag(input_range: list, bias: dict, dtype_out: str) -> bool:
    """
    config if cache_tiling
    """
    if input_range:
        for dim_range in input_range:
            # NOTE when neither x1 nor x2 has a batch, the range of the batch is None, which does not depend on whether
            #      it is cache_tiling.
            if dim_range is None:
                continue

            input_none_range = not dim_range or None in dim_range
            if input_none_range:
                return True
    else:
        return True
    return False


def get_use_vector_core_flag(input_a: dict, support_l0c2out: bool):
    """
    get use_vector_core_flag with platform and dtype_a
    """
    use_vector_core_flag = False
    dtype_a = input_a.get("dtype").lower()
    # use vector to cal matmul
    # if C0 == 16 and fractal, use vector to cal matmul in support_l0c2out
    # 16 is C0, C0 always fractal matrix_a's latest axis
    # when fully_connection is converted by conv2d, NC1HWC0 is ND
    use_vector_c16 = support_l0c2out and (input_a.get("format") not in ("ND", "NC1HWC0")) and \
        (input_a.get("shape")[-1] == 16)
    if dtype_a == "int32" or (dtype_a == "float32" and ((not support_l0c2out) or use_vector_c16)):
        use_vector_core_flag = True

    return use_vector_core_flag


def get_prebuild_pattern(input_x1: dict, op_pattern: str):
    """
    get the prebuild pattern by the input format and op_pattern
    """
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if get_use_vector_core_flag(input_x1, support_l0c2out):
        return json.dumps({"prebuildPattern": "undefined"})

    return json.dumps({"prebuildPattern": op_pattern})


def op_select_format_fix_pipe_l0c2out(no_offset_w: bool = False) -> tuple:
    """
    dynamic format of new architecture

    return : dynamic format combination, static format combination
    """
    stc_scenario = collections.OrderedDict()
    dyn_scenario = collections.OrderedDict()
    # x1_dtype_format, x2_dtype_format, bias_dtype_format, y_dtype_format
    stc_scenario_keys = [
        "fp16_nz_fp16_nz_fp32_nd_fp16_nz",
        "fp16_nz_fp16_nz_fp16_nd_fp16_nz",
        "fp16_nz_fp16_nz_fp32_nd_fp32_nz",
        "int8_nz_int8_nz_int32_nd_int32_nz",
        "int8_nz_int8_z_int32_nd_int32_nz",
        "bf16_nz_bf16_nz_fp32_nd_bf16_nz",
        "bf16_nd_bf16_nd_fp32_nd_bf16_nd",
        "bf16_nd_bf16_nz_fp32_nd_bf16_nd",
        "fp16_nz_fp16_rnn_fp32_nd_fp16_nz",
        "fp16_nz_fp16_rnn_fp16_nd_fp16_nz",
        "fp16_nz_fp16_rnn_fp32_nd_fp32_nz",
        "fp16_nd_fp16_nd_fp32_nd_fp16_nd",
        "fp16_nd_fp16_nd_fp16_nd_fp16_nd",
        "fp16_nd_fp16_nz_fp32_nd_fp16_nd",
        "fp16_nd_fp16_nz_fp16_nd_fp16_nd",
        "fp16_nd_fp16_nd_fp32_nd_fp32_nd",
        "fp32_nd_fp32_nd_fp32_nd_fp32_nd"
    ]
    nz = "FRACTAL_NZ"
    # list all scenario in order according to FE's dtype & format select rule
    stc_scenario_values = [
        [["float16", nz], ["float16", nz], ["float32", "ND"], ["int8", "ND"], ["float16", nz]],
        [["float16", nz], ["float16", nz], ["float16", "ND"], ["int8", "ND"], ["float16", nz]],
        [["float16", nz], ["float16", nz], ["float32", "ND"], ["int8", "ND"], ["float32", nz]],
        [["int8", nz], ["int8", nz], ["int32", "ND"], ["int8", "ND"], ["int32", nz]],
        [["int8", nz], ["int8", "FRACTAL_Z"], ["int32", "ND"], ["int8", "ND"], ["int32", nz]],
        [["bfloat16", nz], ["bfloat16", nz], ["float32", "ND"], ["int8", "ND"], ["bfloat16", nz]],
        [["bfloat16", "ND"], ["bfloat16", "ND"], ["float32", "ND"], ["int8", "ND"], ["bfloat16", "ND"]],
        [["bfloat16", "ND"], ["bfloat16", nz], ["float32", "ND"], ["int8", "ND"], ["bfloat16", "ND"]],
        [["float16", nz], ["float16", "FRACTAL_ZN_RNN"], ["float32", "ND"], ["int8", "ND"], ["float16", nz]],
        [["float16", nz], ["float16", "FRACTAL_ZN_RNN"], ["float16", "ND"], ["int8", "ND"], ["float16", nz]],
        [["float16", nz], ["float16", "FRACTAL_ZN_RNN"], ["float32", "ND"], ["int8", "ND"], ["float32", nz]],
        [["float16", "ND"], ["float16", "ND"], ["float32", "ND"], ["int8", "ND"], ["float16", "ND"]],
        [["float16", "ND"], ["float16", "ND"], ["float16", "ND"], ["int8", "ND"], ["float16", "ND"]],
        [["float16", "ND"], ["float16", nz], ["float32", "ND"], ["int8", "ND"], ["float16", "ND"]],
        [["float16", "ND"], ["float16", nz], ["float16", "ND"], ["int8", "ND"], ["float16", "ND"]],
        [["float16", "ND"], ["float16", "ND"], ["float32", "ND"], ["int8", "ND"], ["float32", "ND"]],
        [["float32", "ND"], ["float32", "ND"], ["float32", "ND"], ["int8", "ND"], ["float32", "ND"]],
    ]
    int32_scenario = {"int32_nhwc_int32_nhwc_int32_nhwc_int32_nhwc":
                      [["int32", "NHWC"], ["int32", "NHWC"], ["int32", "NHWC"], ["int8", "ND"], ["int32", "NHWC"]],
                      "int32_nd_int32_nd_int32_nd_int32_nd":
                      [["int32", "ND"], ["int32", "ND"], ["int32", "ND"], ["int8", "ND"], ["int32", "ND"]]}
    # 所有list都要检查是否有offst_w
    if no_offset_w:
        for stc_scenario_value in stc_scenario_values:
            stc_scenario_value.pop(OFFSET_W_INDEX)
        for _, int32_scenario_value in int32_scenario.items():
            int32_scenario_value.pop(OFFSET_W_INDEX)
    # make pair of key and value
    for stc_scenario_key, stc_scenario_value in zip(stc_scenario_keys, stc_scenario_values):
        stc_scenario[stc_scenario_key] = stc_scenario_value
    if (not tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2ub") or
        tbe_platform.intrinsic_check_support("Intrinsic_vadd", "int8")):
        stc_scenario.update(int32_scenario)

    dyn_scenario_keys = [
        "int8_nz_int8_nz_int32_nd_int32_nz",
        "fp16_nz_fp16_nz_fp32_nd_fp16_nz",
        "fp16_nz_fp16_nz_fp16_nd_fp16_nz",
        "fp16_nd_fp16_nd_fp32_nd_fp16_nd",
        "fp16_nd_fp16_nd_fp16_nd_fp16_nd",
        "fp16_nd_fp16_nd_fp32_nd_fp32_nd",
        "fp32_nd_fp32_nd_fp32_nd_fp32_nd",
        "bf16_nz_bf16_nz_fp32_nd_bf16_nz",
        "bf16_nd_bf16_nd_fp32_nd_bf16_nd"
    ]
    dyn_scenario_values = [
        [["int8", nz], ["int8", nz], ["int32", "ND"], ["int8", "ND"], ["int32", nz]],
        [["float16", nz], ["float16", nz], ["float32", "ND"], ["int8", "ND"], ["float16", nz]],
        [["float16", nz], ["float16", nz], ["float16", "ND"], ["int8", "ND"], ["float16", nz]],
        [["float16", "ND"], ["float16", "ND"], ["float32", "ND"], ["int8", "ND"], ["float16", "ND"]],
        [["float16", "ND"], ["float16", "ND"], ["float16", "ND"], ["int8", "ND"], ["float16", "ND"]],
        [["float16", "ND"], ["float16", "ND"], ["float32", "ND"], ["int8", "ND"], ["float32", "ND"]],
        [["float32", "ND"], ["float32", "ND"], ["float32", "ND"], ["int8", "ND"], ["float32", "ND"]],
        [["bfloat16", nz], ["bfloat16", nz], ["float32", "ND"], ["int8", "ND"], ["bfloat16", nz]],
        [["bfloat16", "ND"], ["bfloat16", "ND"], ["float32", "ND"], ["int8", "ND"], ["bfloat16", "ND"]],
    ]
    if no_offset_w:
        for dyn_scenario_value in dyn_scenario_values:
            dyn_scenario_value.pop(OFFSET_W_INDEX)
    for dyn_scenario_key, dyn_scenario_value in zip(dyn_scenario_keys, dyn_scenario_values):
        dyn_scenario[dyn_scenario_key] = dyn_scenario_value

    return dyn_scenario, stc_scenario


def op_select_no_data_move_out2l1_nd2nz(no_offset_w: bool = False) -> tuple:
    """
    dynamic format of new architecture

    return : dynamic format combination, static format combination
    """
    stc_scenario = collections.OrderedDict()
    dyn_scenario = collections.OrderedDict()
    # x1_dtype_format, x2_dtype_format, bias_dtype_format, y_dtype_format
    stc_scenario_keys = [
        "fp16_nz_fp16_nz_fp32_nd_fp16_nz",
        "fp16_nz_fp16_nz_fp16_nd_fp16_nz",
        "fp16_nz_fp16_nz_fp32_nd_fp32_nz",
        "int8_nz_int8_nz_int32_nd_int32_nz",
        "int8_nz_int8_z_int32_nd_int32_nz",
        "fp16_nz_fp16_rnn_fp32_nd_fp16_nz",
        "fp16_nz_fp16_rnn_fp16_nd_fp16_nz",
        "fp16_nz_fp16_rnn_fp32_nd_fp32_nz",
        "fp16_nd_fp16_nd_fp32_nd_fp16_nd",
        "fp16_nd_fp16_nd_fp16_nd_fp16_nd",
        "fp16_nd_fp16_nd_fp32_nd_fp32_nd"
    ]
    nz = "FRACTAL_NZ"
    # list all scenario in order according to FE's dtype & format select rule
    stc_scenario_values = [
        [["float16", nz], ["float16", nz], ["float32", "ND"], ["int8", "ND"], ["float16", nz]],
        [["float16", nz], ["float16", nz], ["float16", "ND"], ["int8", "ND"], ["float16", nz]],
        [["float16", nz], ["float16", nz], ["float32", "ND"], ["int8", "ND"], ["float32", nz]],
        [["int8", nz], ["int8", nz], ["int32", "ND"], ["int8", "ND"], ["int32", nz]],
        [["int8", nz], ["int8", "FRACTAL_Z"], ["int32", "ND"], ["int8", "ND"], ["int32", nz]],
        [["float16", nz], ["float16", "FRACTAL_ZN_RNN"], ["float32", "ND"], ["int8", "ND"], ["float16", nz]],
        [["float16", nz], ["float16", "FRACTAL_ZN_RNN"], ["float16", "ND"], ["int8", "ND"], ["float16", nz]],
        [["float16", nz], ["float16", "FRACTAL_ZN_RNN"], ["float32", "ND"], ["int8", "ND"], ["float32", nz]],
        [["float16", "ND"], ["float16", "ND"], ["float32", "ND"], ["int8", "ND"], ["float16", "ND"]],
        [["float16", "ND"], ["float16", "ND"], ["float16", "ND"], ["int8", "ND"], ["float16", "ND"]],
        [["float16", "ND"], ["float16", "ND"], ["float32", "ND"], ["int8", "ND"], ["float32", "ND"]],
    ]
    if no_offset_w:
        for stc_scenario_value in stc_scenario_values:
            stc_scenario_value.pop(OFFSET_W_INDEX)
    # make pair of key and value
    for stc_scenario_key, stc_scenario_value in zip(stc_scenario_keys, stc_scenario_values):
        stc_scenario[stc_scenario_key] = stc_scenario_value

    dyn_scenario_keys = [
        "fp16_nz_fp16_nz_fp32_nd_fp16_nz", "fp16_nz_fp16_nz_fp16_nd_fp16_nz"
    ]
    dyn_scenario_values = [
        [["float16", nz], ["float16", nz], ["float32", "ND"], ["int8", "ND"], ["float16", nz]],
        [["float16", nz], ["float16", nz], ["float16", "ND"], ["int8", "ND"], ["float16", nz]]
    ]
    if no_offset_w:
        for dyn_scenario_value in dyn_scenario_values:
            dyn_scenario_value.pop(OFFSET_W_INDEX)
    for dyn_scenario_key, dyn_scenario_value in zip(dyn_scenario_keys, dyn_scenario_values):
        dyn_scenario[dyn_scenario_key] = dyn_scenario_value

    return dyn_scenario, stc_scenario
