#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
basic_lstm_inpalce_fill_window_cache
"""
# 'pylint: disable=too-many-lines
import operator

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from te.lang.cce import vadd
from te.lang.cce import vmul
from te.lang.cce import vmins
from te.tik import scope_gm
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from tbe import tik
from tbe import tvm
from tbe.dsl.compute.util import DTYPE_BYTE
from tbe.common.platform import platform_info
import numpy as np


class Constant:
    '''
    Constant of lstm
    '''
    K0_SIZE = int(platform_info.get_soc_spec("cube_k_size"))
    LSTM_LAYERS = 8


# 'pylint: disable=unused-argument,too-many-arguments,too-many-return-statements,huawei-too-many-arguments
def check_supported(x, w, r, h, c, b, seq_length, clean_cache, deq_scale,
                    y, out_h, out_c, hidden_size, activation_alpha, activation_beta,
                    activations, clip=-1.0, direction="forward", input_forget=0,
                    quant_scale_x=0.0, quant_offset_x=0.0, quant_sqrt_mode_x=False,
                    quant_scale_h=0.0, quant_offset_h=0.0, quant_sqrt_mode_h=False,
                    quant_dtype=2, kernel_name="basic_lstm_inplace_fill_window_cache"):
    """
    Check supported for BasicLSTMInplaceFillWindowCache.

    Parameters:
    x: dict. shape and dtype of input data x
    w: dict. shape and dtype of input data w
    r: dict. shape and dtype of input data r
    h: dict. shape and dtype of input data h
    c: dict. shape and dtype of input data c
    b: dict. shape and dtype of input data b
    seq_length: dict. shape and dtype of input data seq_length
    clean_cache: dict. shape and dtype of input data clean_cache
    deq_scale: dict. shape and dtype of input data deq_scale
    y: dict. shape and dtype of output data y
    out_h: dict. shape and dtype of output data h
    out_c: dict. shape and dtype of output data c
    hidden_size: value of attr hidden_size
    activation_alpha: value of attr activation_alpha
    activation_beta: value of attr activation_beta
    activations: value of attr activations
    clip: value of attr clip
    direction: value of attr direction
    input_forget: value of attr input_forget
    quant_scale_x: value of attr quant_scale_x
    quant_offset_x: value of attr quant_offset_x
    quant_sqrt_mode_x: value of attr quant_sqrt_mode_x
    quant_scale_h: value of attr quant_scale_h
    quant_offset_h: value of attr quant_offset_h
    quant_sqrt_mode_h: value of attr quant_sqrt_mode_h
    quant_dtype: value of attr quant_dtype

    Returns
    -------
    True or False
    """
    
    (seq_length, batch_size, input_size) = tuple(x.get("ori_shape"))
    (num_directions, _, hidden_size) = tuple(h.get("ori_shape"))
    if any((seq_length != 1, batch_size != 1, num_directions != 1)) :
        return False, "seq_length/batch_size/num_directions only support 1 for BasicLSTMInplaceFillWindowCache"
    return True, ""


def get_emit_insn_map(tensor):
    """
    get tensor's emit_insn key
    """
    insn_map = {"elewise_single_cast": "vector_conv",
                "elewise_single_VS_max": "vector_maxs",
                "elewise_single_VS_min": "vector_mins",
                "elewise_single_log": "vector_ln",
                "elewise_single_exp": "vector_exp",
                "elewise_single_rec": "vector_rec",
                "elewise_single_relu": "vector_relu",
                "elewise_single_abs": "vector_abs",
                "elewise_single_not": "vector_not",
                "elewise_single_sqrt": "vector_sqrt",
                "elewise_single_rsqrt": "vector_rsqrt",
                "elewise_binary_mul": "vector_mul",
                "elewise_single_VS_mul": "vector_muls",
                "elewise_binary_div": "vector_div",
                "elewise_binary_add": "vector_add",
                "elewise_single_VS_add": "vector_adds",
                "elewise_binary_min": "vector_min",
                "elewise_binary_max": "vector_max",
                "elewise_binary_vcmpv_gt": "vector_gt",
                "elewise_binary_vcmpv_ge": "vector_ge",
                "elewise_binary_vcmpv_lt": "vector_lt",
                "elewise_binary_vcmpv_le": "vector_le",
                "elewise_binary_vcmpv_eq": "vector_eq",
                "elewise_binary_vcmpv_ne": "vector_ne",
                "elewise_binary_or": "vector_or",
                "elewise_binary_and": "vector_and",
                "elewise_multiple_mla": "vector_multiple",
                "elewise_multiple_madd": "vector_multiple",
                "elewise_multiple_maddrelu": "vector_multiple",
                "broadcast_for_tensor": "broadcast_for_tensor",
                "elewise_binary_sub": "vector_sub",
                "broadcast": "broadcast"}
    if tensor.op.tag.find("|") != -1:
        str_list = tensor.op.tag.split("|")
        insn = insn_map.get(str_list[0])
    else:
        insn = insn_map.get(tensor.op.tag)
    return insn


def dst_type_conversion(quant_dtype):
    """
    convert dst_type from int to string
    """
    dst_type_str = ""
    if quant_dtype == 2:
        dst_type_str = "int8"
    elif quant_dtype == 6:
        dst_type_str = "int16"
    return dst_type_str


def get_shape_info(input_shape, k1_transform):
    k0_index = len(input_shape) - 1
    k1_index = len(input_shape) - 4

    out_shape = input_shape[:]
    read_shape = input_shape[:]
    read_shape[k1_index] = (read_shape[k1_index] + k1_transform - 1) // k1_transform * k1_transform
    for dim, _ in enumerate(input_shape):
        if dim == k0_index:
            out_shape[dim] = input_shape[dim] * k1_transform
        if dim == k1_index:
            out_shape[dim] = (input_shape[dim] + k1_transform - 1) // k1_transform
    return read_shape, out_shape


def reform_compute(input_tensor, input_shape, out_shape, val_info):
    k0_index = len(input_shape) - 1
    k1_index = len(input_shape) - 4
    new_idx = [0] * len(input_shape)
    def lambda_func(*indices):
        for index in range(len(input_shape)):
            if index == k0_index:
                new_idx[index] = (indices[k1_index] * out_shape[k0_index] + indices[k0_index]) % input_shape[k0_index]
            elif index == k1_index:
                new_idx[index] = (indices[k1_index] * out_shape[k0_index] + indices[k0_index]) // input_shape[k0_index]
            else:
                new_idx[index] = indices[index]
        if val_info is None:
            return input_tensor(*new_idx)
        if val_info[0]:
            return input_tensor(*new_idx) + val_info[1]
        return input_tensor(*new_idx) * val_info[2]
    return lambda_func


def scale_compute(input_ub, input_shape, out_shape, attr_list):
    scale = attr_list[0]
    offset = attr_list[1]
    sqrt_mode = attr_list[2]
    out_dtype = attr_list[3]
    tensor_scale_ub_list = []
    if scale != 1:
        scale_value = tvm.const(scale, "float16")
        scale_ub = tvm.compute(out_shape, reform_compute(input_ub, input_shape, out_shape,
                              (False, -1, scale_value)), name=f"{input_ub.name}_scale_reform")
        tensor_scale_ub_list.append(scale_ub)
        if sqrt_mode:
            scale_sqrt_ub = tvm.compute(out_shape, lambda *indice: scale_ub(*indice) * scale_value,
                                        name=f"{input_ub.name}_scale_sqrt_ub")
            tensor_scale_ub_list.append(scale_sqrt_ub)
            tensor_offset_ub_list, res = offset_compute(scale_sqrt_ub, input_shape, out_shape,
                                                        (offset, False, scale, out_dtype))
            tensor_scale_ub_list.extend(tensor_offset_ub_list)
        else:
            tensor_offset_ub_list, res = offset_compute(scale_ub, input_shape, out_shape,
                                                        (offset, False, scale, out_dtype))
            tensor_scale_ub_list.extend(tensor_offset_ub_list)
    else:
        tensor_offset_ub_list, res = offset_compute(input_ub, input_shape, out_shape,
                                                    (offset, True, scale, out_dtype))
        tensor_scale_ub_list.extend(tensor_offset_ub_list)
    return tensor_scale_ub_list, res


def offset_compute(input_ub, input_shape, out_shape, attr_list):
    offset = attr_list[0]
    reform_flag = attr_list[1]
    scale = attr_list[2]
    out_dtype = attr_list[3]
    tensor_offset_ub_list = []
    res = None

    if offset != 0 or scale == 1:
        offset_value = tvm.const(offset, "float16")
        if reform_flag:
            offset_ub = tvm.compute(out_shape, reform_compute(input_ub, input_shape, out_shape,
                                    (True, offset_value, -1)), name=f"{input_ub.name}_offset_reform")
            tensor_offset_ub_list.append(offset_ub)
        else:
            offset_ub = tvm.compute(out_shape, lambda *indices: input_ub(*indices) + offset_value,
                                    name=f"{input_ub.name}_offset_ub")
            tensor_offset_ub_list.append(offset_ub)
        if out_dtype == "int16":
            res = tvm.compute(out_shape, lambda *indices: shape_util.cast(offset_ub(*indices), "int16"),
                              name=f"cast_i16_ub")
            tensor_offset_ub_list.append(res)
        elif out_dtype == "int8":
            res = tvm.compute(out_shape, lambda *indices: shape_util.cast(offset_ub(*indices), "int8"),
                              name=f"cast_i8_ub")
            tensor_offset_ub_list.append(res)
    else:
        if out_dtype == "int16":
            res = tvm.compute(out_shape, lambda *indices: shape_util.cast(input_ub(*indices), "int16"),
                                name=f"cast_i16_ub")
            tensor_offset_ub_list.append(res)
        elif out_dtype == "int8":
            res = tvm.compute(out_shape, lambda *indices: shape_util.cast(input_ub(*indices), "int8"),
                              name=f"cast_i8_ub")
            tensor_offset_ub_list.append(res)
    return tensor_offset_ub_list, res


# 'pylint: disable=huawei-too-many-arguments
def quant_compute(x_ub, h_ub, quant_scale_x, quant_offset_x, quant_scale_h, quant_offset_h,
                  quant_sqrt_mode_x=False, quant_sqrt_mode_h=False, quant_dtype=6):
    input_x_shape = shape_util.shape_to_list(x_ub.shape)
    input_h_shape = shape_util.shape_to_list(h_ub.shape)
    # dtype after quant
    out_dtype = dst_type_conversion(quant_dtype)
    attr_x_list = [quant_scale_x, quant_offset_x, quant_sqrt_mode_x, out_dtype]
    attr_h_list = [quant_scale_h, quant_offset_h, quant_sqrt_mode_h, out_dtype]

    if out_dtype == "int16":
        k1_transform = 1
    else:
        k1_transform = 2
    _, out_x_shape = get_shape_info(input_x_shape, k1_transform)
    _, out_h_shape = get_shape_info(input_h_shape, k1_transform)

    # quant x and quant h
    tensor_ub_list = []
    quant_list_x, quant_x = scale_compute(x_ub, input_x_shape, out_x_shape, attr_x_list)
    quant_list_h, quant_h = scale_compute(h_ub, input_h_shape, out_h_shape, attr_h_list)
    tensor_ub_list.extend(quant_list_x)
    tensor_ub_list.extend(quant_list_h)
    return tensor_ub_list, quant_x, quant_h


def deqscale_compute(deq_scale_gm, hidden_size, recurrence_flag, is_per_tensor):
    """
    deq_scale split and move to fb
    """
    # input deq_scale shape: (8, ) for per-tensor or (8 * hidden_size, ) for per-channel
    # split deq_scale shape: (4, ) for per-tensor or (4 * hidden_size, ) for per-channel
    shape_deq_ub = (4, 1) if is_per_tensor else (4 * hidden_size, )
    shape_deq_broadcast = (4 * hidden_size, )
    shape_deq_fb = (4 * hidden_size, )

    if recurrence_flag == 'hr':
        scale_ub = tvm.compute(shape_deq_ub, lambda *indices: deq_scale_gm[indices[0] + shape_deq_ub[0], ],
                               name="deqscale_" + recurrence_flag + "_ub",
                               tag="split_deqscale_" + recurrence_flag)
    else:
        scale_ub = tvm.compute(shape_deq_ub, lambda *indices: deq_scale_gm(indices[0]),
                               name="deqscale_" + recurrence_flag + "_ub",
                               tag="split_deqscale_" + recurrence_flag)

    scale_broadcast = None
    if is_per_tensor:
        scale_broadcast = tvm.compute(shape_deq_broadcast,
                                      lambda *indices: tvm.select(indices[0] / hidden_size == 0, scale_ub[0][0],
                                                                  tvm.select(indices[0] / hidden_size == 1, 
                                                                             scale_ub[1][0],
                                                                             tvm.select(indices[0] / hidden_size == 2,
                                                                                        scale_ub[2][0],
                                                                                        scale_ub[3][0]))),
                                      name="deqscale_" + recurrence_flag + "_broadcast",
                                      tag="deqscale_" + recurrence_flag + "_broadcast")
        scale_fb = tvm.compute(shape_deq_fb,
                               lambda *indices: scale_broadcast(*indices),
                               name="deqscale_" + recurrence_flag + "_fb",
                               tag="dma_deqscale_" + recurrence_flag + "_fb")
    else:
        scale_fb = tvm.compute(shape_deq_fb, lambda *indices: scale_ub(*indices),
                               name="deqscale" + recurrence_flag + "_fb",
                               tag="dma_deqscale_" + recurrence_flag + "_fb")
    return scale_ub, scale_broadcast, scale_fb


def matmul_fixpipe_compute(mm_fxp_args):
    """
    matmul and fix-pipe
    """
    def _inner_get_k_value(src_dtype):
        k0 = Constant.K0_SIZE // DTYPE_BYTE.get(src_dtype)
        return k0

    lmatrix, rmatrix, bias, use_bias, deq_scale_fb, recurrence_flag = mm_fxp_args
    matrix_m = lmatrix.shape[-3] * lmatrix.shape[-2]
    matrix_k = lmatrix.shape[-4] * lmatrix.shape[-1]
    matrix_n = rmatrix.shape[-3] * rmatrix.shape[-2]

    # src_type is "s8s8" or "s16s8"
    shape_m1 = matrix_m
    shape_m0 = 1
    shape_lk0 = _inner_get_k_value(lmatrix.dtype)
    # `lK1 = matrix_k // lK0`
    shape_rk0 = 16
    # `rK1 = matrix_k // rK0`
    shape_n0 = 16
    shape_n1 = matrix_n // shape_n0
    # lmatrix [matrix_m * matrix_k] in ub in format K1M1M0K0
    # For s8s8, the (K0, M0) number for feature map is (16, 1).
    # Fors16s8, the (K0, M0) number for feature map is (8, 1).
    # rmatrix [matrix_k * matrix_n] in ub in format K1N1N0K0
    # For s8s8 and s16s8, the (K0, N0) number for weight is (16, 16).
    # `bias [N1, N0]`
    mm_res_shape = [lmatrix.shape[0], shape_n1, shape_m1, shape_m0, shape_n0]
    # the output result is stored to UB with N1M1M0N0 format
    # For s32 output, Nout0 number is N0(16, without channel split),
    # since it must be partial sum for next convolution instruction.
    # Nout0 size is 16*4B.
    k_axis = tvm.reduce_axis((0, matrix_k), name='k_axis')
    mm_op_dict = {"use_bias": use_bias,
                  "matrix_m": matrix_m,
                  "matrix_n": matrix_n,
                  "matrix_k": matrix_k}
    matmul_res = tvm.compute(mm_res_shape,
                             lambda batch, n1, m1, m0, n0: tvm.sum(
                                 tvm.matmul_op(
                                     lmatrix[batch, k_axis // shape_lk0, m1, m0, k_axis % shape_lk0],
                                     rmatrix[k_axis // shape_rk0, n1, n0, k_axis % shape_rk0],
                                     bias[n1, 0, 0, n0] if use_bias else None,
                                     dst_dtype="int32",
                                     op_dict=mm_op_dict),
                                 axis=[k_axis, ]),
                             name="mm_res_" + recurrence_flag,
                             tag="matmul_ub_to_ub")

    # The output data type depends on the quantization mode.
    # Then the data will be moved to UB with normal DMA (automatic channel merge or split).
    fxp_res = tvm.compute(mm_res_shape,
                          lambda batch, n1, m1, m0, n0: tvm.fixpipe_op(
                              matmul_res[batch, n1, m1, m0, n0],
                              "float16",
                              pre_conv_param=deq_scale_fb[n1 * 16 + n0],
                              op_dict={"pre_conv": "VS322F16"}),
                          name="fixpipe_res_" + recurrence_flag)

    reform_shape = [mm_res_shape[0], mm_res_shape[1] * 2, mm_res_shape[2], mm_res_shape[3], mm_res_shape[4] // 2]
    reform_res = tvm.compute(reform_shape,
                             lambda batch, n1, m1, m0, n0: fxp_res[batch, (n1 * 8 + n0) // 16, m1, m0,
                                                                   (n1 * 8 + n0) % 16],
                             name="fixpipe_reform_res_" + recurrence_flag)

    return matmul_res, fxp_res, reform_res


# 'pylint: disable=too-many-return-values
def active_compute(reform_res_xw, reform_res_hr):
    """
    activation
    """
    reform_res = tvm.compute(reform_res_xw.shape,
                                lambda *indices: reform_res_xw(*indices) + reform_res_hr(*indices),
                                name="fixpipe_reform_res_xw_hr", tag="add_fixpipe_reform_res_xw_hr")

    reform_shape = reform_res.shape
    v_shape = (reform_shape[0], reform_shape[1] // 4, reform_shape[2], reform_shape[3], reform_shape[4])
    sigmoid_i = tvm.compute(v_shape, lambda *indices: tvm.sigmoid(reform_res(*indices)),
                            name="sigmoid_i", tag="sigmoid_i")
    sigmoid_o = tvm.compute(v_shape,
                            lambda i0, i1, i2, i3, i4: tvm.sigmoid(reform_res[i0, i1 + v_shape[1], i2, i3, i4]),
                            name="sigmoid_o", tag="sigmoid_o")
    sigmoid_f = tvm.compute(v_shape,
                            lambda i0, i1, i2, i3, i4: tvm.sigmoid(reform_res[i0, i1 + v_shape[1] * 2, i2, i3, i4]),
                            name="sigmoid_f", tag="sigmoid_f")
    tanh_j = tvm.compute(v_shape,
                         lambda i0, i1, i2, i3, i4: tvm.tanh(reform_res[i0, i1 + v_shape[1] * 3, i2, i3, i4]),
                         name="tanh_j", tag="tanh_j")

    return reform_res, sigmoid_f, sigmoid_i, tanh_j, sigmoid_o


# 'pylint: disable=too-many-return-values
def matmul_fixpipe_active_compute(mm_fxp_args, is_per_tensor):
    """
    matmul, fix-pipe, activation
    """
    (x_quant_ub, w_gm, r_gm, h_quant_ub, b_gm, deq_scale_gm) = mm_fxp_args

    w_ub = tvm.compute(w_gm.shape, lambda *i: w_gm(*i), name="w_ub")
    r_ub = tvm.compute(r_gm.shape, lambda *i: r_gm(*i), name="r_ub")

    # input b shape: (8 * hidden_size // 16, 1, 1, 16) , data type: int32
    # matmul bias shape: (4 * hidden_size // 16, 1, 1, 16)
    use_bias = True if b_gm is not None else False
    b_xw_ub, b_hr_ub = None, None
    if use_bias:
        shape_bias = (b_gm.shape[0] // 2, *(b_gm.shape[1:]))
        b_xw_ub = tvm.compute(shape_bias, lambda *indices: b_gm(*indices), name="b_xw_ub", tag="split_input_b_xw")
        b_hr_ub = tvm.compute(shape_bias, lambda *indices: b_gm[indices[0] + shape_bias[0], 0, 0, indices[3]],
                             name="b_hr_ub", tag="split_input_b_hr")

    hidden_size = w_gm.shape[-3] * w_gm.shape[-2] // 4
    (deqscale_xw_ub, deqscale_xw_broadcast, deqscale_xw_fb) = deqscale_compute(deq_scale_gm, hidden_size,
                                                                               'xw', is_per_tensor)
    mm_xw_args = (x_quant_ub, w_ub, b_xw_ub, use_bias, deqscale_xw_fb, "xw")
    (mm_res_xw, fxp_res_xw, reform_res_xw) = matmul_fixpipe_compute(mm_xw_args)

    (deqscale_hr_ub, deqscale_hr_broadcast, deqscale_hr_fb) = deqscale_compute(deq_scale_gm, hidden_size,
                                                                               'hr', is_per_tensor)
    mm_hr_args = (h_quant_ub, r_ub, b_hr_ub, use_bias, deqscale_hr_fb, "hr")
    (mm_res_hr, fxp_res_hr, reform_res_hr) = matmul_fixpipe_compute(mm_hr_args)

    (reform_res_xw_hr, sigmoid_f, sigmoid_i, tanh_j, sigmoid_o) = active_compute(reform_res_xw, reform_res_hr)

    return (w_ub, r_ub, b_xw_ub, b_hr_ub,
            deqscale_xw_ub, deqscale_xw_broadcast, deqscale_hr_ub, deqscale_hr_broadcast,
            mm_res_xw, mm_res_hr, fxp_res_xw, fxp_res_hr,
            reform_res_xw, reform_res_hr, reform_res_xw_hr,
            sigmoid_f, sigmoid_i, tanh_j, sigmoid_o), (deqscale_xw_fb, deqscale_hr_fb)


def get_tiling_func(tiling_case, input_size, hidden_size_k, hidden_size_n):
    if tiling_case == 100:
        tiling_func = 20 * input_size + 20 * hidden_size_k + 16 * input_size * hidden_size_n + 300 * hidden_size_n + 16
    elif tiling_case == 200:
        tiling_func = 20 * input_size + 20 * hidden_size_k + 16 * input_size * hidden_size_n + 300 * hidden_size_n + 144
    elif tiling_case == 300:
        tiling_func = 20 * input_size + 20 * hidden_size_k + 16 * input_size * hidden_size_n + 300 * hidden_size_n
    elif tiling_case == 400:
        tiling_func = 20 * input_size + 20 * hidden_size_k + 16 * input_size * hidden_size_n + 300 * hidden_size_n + 128
    elif tiling_case == 500:
        tiling_func = 20 * input_size + 20 * hidden_size_k + 16 * input_size * hidden_size_n + 308 * hidden_size_n + 16
    elif tiling_case == 600:
        tiling_func = 20 * input_size + 20 * hidden_size_k + 16 * input_size * hidden_size_n + 308 * hidden_size_n + 144
    elif tiling_case == 700:
        tiling_func = 20 * input_size + 20 * hidden_size_k + 16 * input_size * hidden_size_n + 308 * hidden_size_n
    elif tiling_case == 800:
        tiling_func = 20 * input_size + 20 * hidden_size_k + 16 * input_size * hidden_size_n + 308 * hidden_size_n + 128
    else:
        tiling_func = None
    return tiling_func


# 'pylint: disable=huawei-too-many-arguments
def get_tiling_case(is_clean_cache, is_per_tensor, input_size, hidden_size, ub_size):
    if is_clean_cache and not is_per_tensor:
        tiling_case = 100
    elif is_clean_cache and is_per_tensor:
        tiling_case = 200
    elif not is_clean_cache and not is_per_tensor:
        tiling_case = 300
    elif not is_clean_cache and is_per_tensor:
        tiling_case = 400
    else:
        tiling_case = -1

    for case_item in [100, 200, 300, 400]:
        if tiling_case == case_item:
            total_size = get_tiling_func(case_item, input_size, hidden_size, 16)
            if total_size > ub_size:
                tiling_case = case_item + 400
    return tiling_case


# 'pylint: disable=too-many-return-values
def get_lstm_tiling(input_x, s_init_h_gm, is_clean_cache, is_per_tensor):
    input_size = input_x.shape[-4] * input_x.shape[-1]
    hidden_size = s_init_h_gm.shape[-4] * s_init_h_gm.shape[-1]
    ub_size = tbe_platform.get_soc_spec("UB_SIZE")
    db_flag = True if hidden_size >= 16 else False
    m_ub_factor, k_ub_factor, n_ub_factor = 1, None, None
    tiling_case = get_tiling_case(is_clean_cache, is_per_tensor, input_size, hidden_size, ub_size)

    if tiling_case in [100, 200, 300, 400]:
        total_size = get_tiling_func(tiling_case, input_size, hidden_size, hidden_size)
        if total_size < ub_size:  # no cut
            m_ub_factor = 1
            k_ub_factor = input_size + hidden_size
            n_ub_factor = hidden_size
        else:  # only cut N
            m_ub_factor = 1
            k_ub_factor = input_size + hidden_size
            n_ub_factor = 16
    elif tiling_case in [500, 600, 700, 800]:
        total_size = get_tiling_func(tiling_case, input_size, hidden_size, hidden_size)
        if total_size < ub_size:  # only cut K
            m_ub_factor = 1
            k_ub_factor = 16
            n_ub_factor = hidden_size
        else:  # only cut N and K
            m_ub_factor = 1
            k_ub_factor = 16
            n_ub_factor = 16

    # check FB
    if 32 * hidden_size > 2048:
        n_ub_factor = 16
    n1_ub_factor = n_ub_factor // 8
    return db_flag, m_ub_factor, k_ub_factor, n1_ub_factor


def set_tensor_scope(sch, ub_list, fb_list):
    """
    set scope for tensors
    """
    for tensor in ub_list:
        if tensor is not None:
            sch[tensor].set_scope("local.UB")
    for tensor in fb_list:
        if tensor is not None:
            sch[tensor].set_scope("local.FB")


def enable_double_buffer(sch, enable_db, ub_list):
    """
    enable double buffer for ub tensor
    """
    if enable_db:
        for tensor in ub_list:
            if tensor is not None:
                sch[tensor].double_buffer()


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name,invalid-name
def check_prama_dtype(input_x, weight, r, h, c, bias, seq_length, clean_cache, deq_scale, y, output_h, output_c):
    """
    check parameters dtype
    """
    x_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(x_dtype, ["float16"], param_name="x")

    w_dtype = weight.get("dtype").lower()
    para_check.check_dtype(w_dtype, ["int8"], param_name="w")

    r_dtype = r.get("dtype").lower()
    para_check.check_dtype(r_dtype, ["int8"], param_name="w")

    h_dtype = h.get("dtype").lower()
    para_check.check_dtype(h_dtype, ["float16"], param_name="h")

    c_dtype = c.get("dtype").lower()
    para_check.check_dtype(c_dtype, ["float16"], param_name="c")

    if bias is not None:
        bias_dtype = bias.get("dtype").lower()
        para_check.check_dtype(bias_dtype, ["int32"], param_name="b")

    if seq_length is not None:
        seq_length_dtype = seq_length.get("dtype").lower()
        para_check.check_dtype(seq_length_dtype, ["int32"], param_name="seq_length")

    if clean_cache is not None:
        clean_cache_dtype = clean_cache.get("dtype").lower()
        para_check.check_dtype(clean_cache_dtype, ["int32"], param_name="clean_cache")

    deq_scale_dtype = deq_scale.get("dtype").lower()
    para_check.check_dtype(deq_scale_dtype, ["uint64"], param_name="deq_scale")

    y_dtype = y.get("dtype").lower()
    para_check.check_dtype(y_dtype, ["float16"], param_name="y")

    output_h_dtype = output_h.get("dtype").lower()
    para_check.check_dtype(output_h_dtype, ["float16"], param_name="output_h")

    output_c_dtype = output_c.get("dtype").lower()
    para_check.check_dtype(output_c_dtype, ["float16"], param_name="output_c")


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name,invalid-name
def check_prama_inshape(input_x, weight, r, h, c, bias, y, output_h, output_c):
    """
    check parameters
    """
    # check seq_length
    if input_x["shape"][0] != output_h["shape"][0] != 1 :
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "seq_length only support 1 for now!")
    # check batch dim
    if input_x["shape"][2] != y["shape"][2] != output_h["shape"][1] != output_c["shape"][1] != 1:
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "batch_size only support 1, please check!")
    # check num_direction
    if h["shape"][0] != c["shape"][0] != 1:
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "num_direction only support 1, please check!")
    # hidden_size dim check
    if weight["ori_shape"][1] != r["ori_shape"][1] != 4 * output_h["ori_shape"][2]:
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "w, r, h shape is wrong, please check!")
    if bias:
        if bias["ori_shape"][0] != weight["ori_shape"][1] * 2:
            error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                          "bias shape is wrong, please check!")


def check_prama_outshape(h, c, y, output_h, output_c):
    """
    check parameters
    """
    if not operator.eq(h["shape"], output_h["shape"]):
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "h, output_h shape is different, please check!")

    if not operator.eq(c["shape"], output_c["shape"]):
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "c, output_c shape is different, please check!")

    if not operator.eq(h["shape"], c["shape"]):
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "c, h shape is different, please check!")

    if not operator.eq(output_h["shape"], output_c["shape"]):
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "output_c, output_h shape is different, please check!")

    if not operator.eq(output_h["shape"], y["shape"][1:]):
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "y[1:], output_h shape is different, please check!")


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals
def check_attr(activation_alpha, activation_beta, activations, direction, input_forget, quant_dtype):
    """
    check parameters
    """
    if direction not in ["forward"]:
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "attr direction only support 'forward' now, please check!")
    if len(activation_alpha) != 0:
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "attr activation_alpha is not support, please check!")
    if len(activation_beta) != 0:
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "attr activation_beta is not support, please check!")
    if len(activations) != 0:
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "attr activations is not support, please check!")
    if input_forget != 0:
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "attr input_forget is not support, please check!")
    if quant_dtype != 2 and quant_dtype != 6:
        error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
                                                      "attr quant_dtype is not support, please check!")


# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-function-args,too-many-statements
# 'pylint: disable=unused-argument
@register_operator("BasicLSTMInplaceFillWindowCache")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.OPTION_ATTR_LIST_FLOAT,
                            para_check.OPTION_ATTR_LIST_FLOAT, para_check.NONE_TYPE,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def basic_lstm_inplace_fill_window_cache(x, w, r, h, c, b, seq_length, clean_cache, deq_scale,
                                         y, out_h, out_c, hidden_size, activation_alpha, activation_beta,
                                         activations, clip=-1.0, direction="forward", input_forget=0,
                                         quant_scale_x=0.0, quant_offset_x=0.0, quant_sqrt_mode_x=False,
                                         quant_scale_h=0.0, quant_offset_h=0.0, quant_sqrt_mode_h=False,
                                         quant_dtype=2, kernel_name="basic_lstm_inplace_fill_window_cache"):
    """
    basic_lstm_inplace_fill_window_cache op.

    Parameters:

    Inputs:
    x: Tensor, required, dtype=Float16
    w: Tensor, required, dtype=Int8
    r: Tensor, required, dtype=Int8
    h: Tensor, required, dtype=Float16
    c: Tensor, required, dtype=Float16
    b: Tensor, optional, dtype=int32
    seq_length: Tensor, optional, dtype=int32
    clean_cache: Tensor, optional, dtype=bool
    deq_scale: Tensor, required, dtype=uint64

    Outputs:
    y: Tensor, required, dtype=Float16
    out_h: Tensor, required, dtype=Float16
    out_c: Tensor, required, dtype=Float16

    Attributes:
    hidden_size: Tensor, required, dtype=int
    activation_alpha: optional, a list of float
    activation_beta: optional, a list of float
    activations: optional, a list of string
    clip: optional, a float number, default is -0.1
    direction: optional, string , only support "forward"
    input_forget: optional, an int number, default is 0
    quant_scale_x: optional, a float number, default is 0.0
    quant_offset_x: optional, a float number, default is 0.0
    quant_sqrt_mode_x: optional, a bool number, default is false
    quant_scale_h: optional, a float number, default is 0.0
    quant_offset_h: optional, a float number, default is 0.0
    quant_sqrt_mode_h: optional, a float number, default is false
    quant_dtype: optional, a int number, 2 means quant to int8, 6 means quant to int16
    """
    # check param and attr
    check_prama_dtype(x, w, r, h, c, b, seq_length, clean_cache, deq_scale, y, out_h, out_c)
    check_prama_inshape(x, w, r, h, c, b, y, out_h, out_c)
    check_prama_outshape(h, c, y, out_h, out_c)
    check_attr(activation_alpha, activation_beta, activations, direction, input_forget, quant_dtype)

    if quant_scale_x > 65504:
        if quant_sqrt_mode_x:
            error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
        "quant_scale_x should not over 65504")
        else:
            quant_scale_x = np.sqrt(quant_scale_x)
            quant_sqrt_mode_x = True
    if quant_scale_h > 65504:
        if quant_sqrt_mode_h:
            error_manager_vector.raise_err_specific_reson("basic_lstm_inplace_fill_window_cache",
        "quant_scale_h should not over 65504")
        else:
            quant_scale_h = np.sqrt(quant_scale_h)
            quant_sqrt_mode_h = True

    input_dtype = x.get("dtype").lower()
    weight_dtype = w.get("dtype").lower()

    shape_x_input = x.get("shape")
    shape_w = w.get("shape")
    shape_r = r.get("shape")
    shape_hc = h.get("shape")
    shape_deq_scale = deq_scale.get("shape")

    t_size = shape_x_input[0]
    m_size = shape_x_input[2]

    tik_instance = tik.Tik()
    # nano
    is_clean_cache = False
    clean_cache_gm = None
    if clean_cache is not None:
        clean_cache_gm = tik_instance.Tensor(shape=(1,), dtype="int32", scope=scope_gm, name='clean_cache_gm')
        is_clean_cache = True

    x_gm = tik_instance.Tensor(shape=shape_x_input, dtype=input_dtype, scope=scope_gm, name='x_gm')
    w_gm = tik_instance.Tensor(shape=shape_w, dtype=weight_dtype, scope=scope_gm, name='w_gm')
    r_gm = tik_instance.Tensor(shape=shape_r, dtype=weight_dtype, scope=scope_gm, name='r_gm')
    deq_scale_gm = tik_instance.Tensor(shape=shape_deq_scale, dtype="uint64", scope=scope_gm, name='deq_scale_gm')

    b_gm = None
    use_bias = False
    use_seq_length = False
    if b is not None:
        use_bias = True
        b_dtype = b.get("dtype").lower()
        shape_b = (b.get("shape")[0] // 16, 1, 1, 16)
        b_gm = tik_instance.Tensor(shape=shape_b, dtype=b_dtype, scope=scope_gm, name='b_gm')
    if seq_length is not None:    
        seq_length_gm = tik_instance.Tensor(shape=(1,), dtype="int32", scope=scope_gm, name='seq_length_gm')
        use_seq_length = True

    s_init_h_gm = tik_instance.Tensor(shape=shape_hc, dtype=input_dtype, scope=scope_gm, name='s_init_h_gm')
    s_init_c_gm = tik_instance.Tensor(shape=shape_hc, dtype=input_dtype, scope=scope_gm, name='s_init_c_gm')

    update_h_gm = tik_instance.Tensor(shape=shape_hc, dtype=input_dtype, scope=scope_gm, name='update_h_gm')
    update_c_gm = tik_instance.Tensor(shape=shape_hc, dtype=input_dtype, scope=scope_gm, name='update_c_gm')
    update_h_gm_as_y = tik_instance.Tensor(shape=shape_hc, dtype=input_dtype, scope=scope_gm, name='update_h_gm_as_y')

    build_input_list = [x_gm, w_gm, r_gm, s_init_h_gm, s_init_c_gm]
    if use_bias:
        build_input_list.append(b_gm)
    if use_seq_length:
        build_input_list.append(seq_length_gm)
    if is_clean_cache:
        build_input_list.append(clean_cache_gm)
    build_input_list.append(deq_scale_gm)
    build_output_list = [update_h_gm_as_y, update_h_gm, update_c_gm]

    cut_t = 1
    cut_m = m_size
    loop_t = t_size // cut_t
    loop_m = m_size // cut_m
    with tik_instance.for_range(0, loop_t) as loop_i:
        if direction == "REDIRECTIONAL":
            valid_loop_i = loop_t - 1 - loop_i
        else:
            valid_loop_i = loop_i
        with tik_instance.for_range(0, loop_m) as loop_j:
            input_x_var = x_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t, :,
                               loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            s_init_c_gm_var = s_init_c_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            s_init_h_gm_var = s_init_h_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            update_h_gm_var = update_h_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t, :,
                                          loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            update_c_gm_var = update_c_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t,
                                          :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            update_h_gm_as_y_var = update_h_gm_as_y[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t:,
                                                    :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]

            input_list = [input_x_var, w_gm, r_gm, b_gm, s_init_h_gm_var,
                          s_init_c_gm_var, deq_scale_gm, clean_cache_gm]
            output_list = [update_h_gm_var, update_c_gm_var, update_h_gm_as_y_var]

            with tik_instance.if_scope(loop_i == 0):
                is_first_round = True
                tik_instance.call_module(
                    dynamic_rnn_tik,
                    input_list,
                    output_list,
                    [is_first_round, clip, quant_scale_x, quant_offset_x, quant_sqrt_mode_x, quant_scale_h,
                     quant_offset_h, quant_sqrt_mode_h, quant_dtype, is_clean_cache])

            with tik_instance.if_scope(loop_i > 0):
                is_first_round = False
                tik_instance.call_module(
                    dynamic_rnn_tik,
                    input_list,
                    output_list,
                    [is_first_round, clip, quant_scale_x, quant_offset_x, quant_sqrt_mode_x, quant_scale_h,
                     quant_offset_h, quant_sqrt_mode_h, quant_dtype, is_clean_cache])

    config_map = {"dump_cce_code": False, }
    tik_instance.BuildCCE(kernel_name, build_input_list, build_output_list, config=config_map)


def dynamic_rnn_tik(input_list, custom_list):
    """
    inside part of tik loop
    :return:
    """
    (x_gm, w_gm, r_gm, b_gm, s_init_h_gm,
     s_init_c_gm, deq_scale_gm, clean_cache_gm) = input_list

    (is_first_round, cell_clip,
     quant_scale_x, quant_offset_x, quant_sqrt_mode_x,
     quant_scale_h, quant_offset_h, quant_sqrt_mode_h,
     quant_dtype, is_clean_cache) = custom_list

    return dynamic_rnn_core(x_gm, w_gm, r_gm, b_gm, s_init_h_gm, s_init_c_gm, deq_scale_gm,
                            clean_cache_gm, is_first_round, cell_clip,
                            quant_scale_x, quant_offset_x, quant_sqrt_mode_x,
                            quant_scale_h, quant_offset_h, quant_sqrt_mode_h, quant_dtype, is_clean_cache)


# 'pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-statements,unnecessary-lambda
def dynamic_rnn_core(x_gm, w_gm, r_gm, b_gm, s_init_h_gm, s_init_c_gm, deq_scale_gm,
                     clean_cache_gm, is_first_round, cell_clip,
                     quant_scale_x, quant_offset_x, quant_sqrt_mode_x,
                     quant_scale_h, quant_offset_h, quant_sqrt_mode_h, quant_dtype, is_clean_cache):
    """
    implement of dynamic rnn
    """
    is_per_tensor = True if deq_scale_gm.shape[0] == Constant.LSTM_LAYERS else False

    shape_i = shape_h = shape_c = s_init_h_gm.shape

    x_ub = tvm.compute(x_gm.shape, lambda *indices: x_gm(*indices), name="x_ub")

    pad_zero = tvm.const(0, dtype=s_init_h_gm.dtype)

    if clean_cache_gm is not None:
        clean_cache_dtype = clean_cache_gm.dtype
        clean_cache_one = tvm.const(1, dtype=clean_cache_dtype)
        clean_cache_ub = tvm.compute((1,), lambda *indices: clean_cache_gm(*indices),
                                                            name="clean_cache_ub", tag="dma_copy")
        s_state_h_ub = tvm.compute(shape_h, lambda *indices: tvm.select(clean_cache_ub[0] > clean_cache_one,
                                                                        s_init_h_gm(*indices),
                                                                        (tvm.select(clean_cache_ub[0] < clean_cache_one,
                                                                        s_init_h_gm(indices[0] + 1, *indices[1:]),
                                                                        pad_zero))),
                                                                        name="s_init_h", tag="dma_copy")
        s_state_c_ub = tvm.compute(shape_c, lambda *indices: tvm.select(clean_cache_ub[0] > clean_cache_one,
                                                                        s_init_c_gm(*indices),
                                                                        (tvm.select(clean_cache_ub[0] < clean_cache_one,
                                                                        s_init_c_gm(indices[0] + 1, *indices[1:]),
                                                                        pad_zero))),
                                                                        name="s_init_c", tag="dma_copy")
    else:
        s_state_h_ub = tvm.compute(shape_h, lambda *indices: s_init_h_gm(*indices), 
                                                        name="s_init_h", tag="dma_copy")
        s_state_c_ub = tvm.compute(shape_c, lambda *indices: s_init_c_gm(*indices), 
                                                        name="s_init_c", tag="dma_copy")

    # quant x, quant h
    tensor_ub_list, quant_x, quant_h = quant_compute(x_ub, s_state_h_ub, quant_scale_x, quant_offset_x,
                                                     quant_scale_h, quant_offset_h,
                                                     quant_sqrt_mode_x, quant_sqrt_mode_h,
                                                     quant_dtype)

    mm_fxp_active_args = (quant_x, w_gm, r_gm, quant_h, b_gm, deq_scale_gm)
    tensor_list_mm_ub, tensor_list_mm_fb = matmul_fixpipe_active_compute(mm_fxp_active_args, is_per_tensor)
    (w_ub, r_ub, b_xw_ub, b_hr_ub,
     deqscale_xw_ub, deqscale_xw_broadcast, deqscale_hr_ub, deqscale_hr_broadcast,
     matmul_res_xw, matmul_res_hr, fixpipe_res_xw, fixpipe_res_hr,
     reform_res_xw, reform_res_hr, reform_res_xw_hr,
     sigmoid_f, sigmoid_i, tanh_j, sigmoid_o) = tensor_list_mm_ub
    (deqscale_xw_fb, deqscale_hr_fb) = tensor_list_mm_fb

    # vector only support fp16 on this chip
    c_t_tmp1 = vmul(s_state_c_ub, sigmoid_f)
    c_t_tmp2 = vmul(tanh_j, sigmoid_i)
    update_c = vadd(c_t_tmp1, c_t_tmp2)

    new_out_shape = list(shape_h[:])
    new_out_shape[0] = new_out_shape[0] * 2 

    if cell_clip > 0:
        dtype = update_c.dtype
        clip_const = tvm.const(cell_clip, dtype=dtype)
        update_c = vmins(update_c, clip_const)

    update_c_gm = tvm.compute(new_out_shape,
                                lambda batch, n1, m1, m0, n0:
                                    tvm.select(batch > 0, update_c(batch - 1, n1, m1, m0, n0),
                                                update_c(batch, n1, m1, m0, n0)), name="update_c_gm", tag="dma_copy")

    c_t_tanh_ub = tvm.compute(shape_i, lambda *indices: tvm.tanh(update_c(*indices)), name='c_t_tanh_ub')

    update_h = vmul(c_t_tanh_ub, sigmoid_o)

    update_h_gm_as_y = tvm.compute(shape_i, lambda *indices: update_h(*indices),
                                   name="update_h_gm_as_y", tag="ub_to_out")

    update_h_gm = tvm.compute(new_out_shape,
                              lambda batch, n1, m1, m0, n0:
                                  tvm.select(batch > 0, update_h(batch - 1, n1, m1, m0, n0),
                                             update_h(batch, n1, m1, m0, n0)), name="update_h_gm", tag="dma_copy")

    # end compute
    return_list = [update_h_gm, update_c_gm, update_h_gm_as_y]
    # schedule
    s = tvm.create_schedule([update_h_gm.op, update_c_gm.op, update_h_gm_as_y.op])

    def gen_reversed_subgraph_list(out_tensor, tensor_list):
        """
        traverse tensors by Depth-First-Search
        """
        if out_tensor is None:
            return
        stack = [out_tensor]
        visited_list = []
        while stack:
            cur_tensor = stack.pop()
            visited_list.append(cur_tensor)
            for in_tensor in cur_tensor.op.input_tensors:
                if in_tensor in visited_list:
                    continue
                stack.append(in_tensor)
                if any(("elewise" in in_tensor.op.tag, in_tensor.op.tag == "broadcast")):
                    if any((in_tensor.name.endswith("_drnn_cast"),
                            in_tensor.name in ["s_state_h_ub", "s_state_c_ub", "s_state_h_ub_for_element"])):
                        continue
                    if in_tensor not in tensor_list:
                        tensor_list.append(in_tensor)

    elewise_tensors = []
    gen_reversed_subgraph_list(update_h_gm, elewise_tensors)

    # barrier_tensor
    elewise_before_barrier_tensors = []
    if b_xw_ub is not None:
        elewise_before_barrier_tensors = [b_xw_ub, b_hr_ub]
    for tensor in tensor_ub_list:
        elewise_before_barrier_tensors.append(tensor)

    # set scope
    ub_list = [x_ub, quant_x, quant_h, s_state_h_ub, s_state_c_ub, c_t_tanh_ub, update_c, update_h]
    if clean_cache_gm is not None:
        ub_list.append(clean_cache_ub)
    ub_list = ub_list + list(elewise_tensors) + list(elewise_before_barrier_tensors) + list(tensor_list_mm_ub)
    set_tensor_scope(s, ub_list, tensor_list_mm_fb)
    
    # h,c reused by
    s[s_init_c_gm].reused_by(update_c_gm)
    s[s_init_h_gm].reused_by(update_h_gm)

    # matmul compute inline with fix-pipe
    compute_inline_tensors = [matmul_res_xw, matmul_res_hr]
    for tensor in compute_inline_tensors:
        s[tensor].compute_inline()

    fix_pipe_compute_inline_tensors = [fixpipe_res_xw, fixpipe_res_hr]
    for tensor in fix_pipe_compute_inline_tensors:
        s[tensor].compute_inline()

    # get tiling data
    enable_db, factor_m, _, factor_n = get_lstm_tiling(x_gm, s_init_h_gm, is_clean_cache, is_per_tensor)

    # matmul split, compute at
    i_n_outer, i_n_inner = s[reform_res_xw].split(reform_res_xw.op.axis[1], factor=factor_n)
    i_m_outer, i_m_inner = s[reform_res_xw].split(reform_res_xw.op.axis[2], factor=factor_m)
    s[reform_res_xw].reorder(i_n_outer, i_m_outer, i_n_inner, i_m_inner,
                             reform_res_xw.op.axis[3], reform_res_xw.op.axis[4])
    s[w_ub].compute_at(s[reform_res_xw], i_m_outer)
    if b_xw_ub is not None:
        s[b_xw_ub].compute_at(s[reform_res_xw], i_m_outer)
    s[deqscale_xw_fb].compute_at(s[reform_res_xw], i_m_outer)
    s[deqscale_xw_ub].compute_at(s[reform_res_xw], i_m_outer)
    if deqscale_xw_broadcast is not None:
        s[deqscale_xw_broadcast].compute_at(s[reform_res_xw], i_m_outer)
    
    if is_per_tensor:
        s[deqscale_hr_ub].storage_align(deqscale_hr_ub.op.axis[0], 16, 0)
        s[deqscale_xw_ub].storage_align(deqscale_hr_ub.op.axis[0], 16, 0)

    j_n_outer, j_n_inner = s[reform_res_hr].split(reform_res_hr.op.axis[1], factor=factor_n)
    j_m_outer, j_m_inner = s[reform_res_hr].split(reform_res_hr.op.axis[2], factor=factor_m)
    s[reform_res_hr].reorder(j_n_outer, j_m_outer, j_n_inner, j_m_inner,
                             reform_res_hr.op.axis[3], reform_res_hr.op.axis[4])
    s[r_ub].compute_at(s[reform_res_hr], j_m_outer)
    if b_hr_ub is not None:
        s[b_hr_ub].compute_at(s[reform_res_hr], j_m_outer)
    s[deqscale_hr_fb].compute_at(s[reform_res_hr], j_m_outer)
    s[deqscale_hr_ub].compute_at(s[reform_res_hr], j_m_outer)
    if deqscale_hr_broadcast is not None:
        s[deqscale_hr_broadcast].compute_at(s[reform_res_hr], j_m_outer)

    r_n_outer, r_n_inner = s[reform_res_xw_hr].split(reform_res_xw_hr.op.axis[1], factor=factor_n)
    s[reform_res_xw].compute_at(s[reform_res_xw_hr], r_n_outer)
    s[reform_res_hr].compute_at(s[reform_res_xw_hr], r_n_outer)

    # emit_insn
    s[x_ub].emit_insn(x_ub.op.axis[0], 'dma_copy')
    s[w_ub].emit_insn(w_ub.op.axis[0], 'dma_copy')
    s[r_ub].emit_insn(r_ub.op.axis[0], 'dma_copy')
    s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'dma_copy')
    s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'dma_copy')
    if b_xw_ub is not None:
        s[b_xw_ub].emit_insn(b_xw_ub.op.axis[0], 'dma_copy')
        s[b_hr_ub].emit_insn(b_hr_ub.op.axis[0], 'dma_copy')
    if clean_cache_gm is not None:
        s[clean_cache_ub].emit_insn(clean_cache_ub.op.axis[0], 'dma_copy')
    s[deqscale_xw_ub].emit_insn(deqscale_xw_ub.op.axis[0], 'dma_copy')
    s[deqscale_xw_fb].emit_insn(deqscale_xw_fb.op.axis[0], 'dma_copy')
    s[deqscale_hr_ub].emit_insn(deqscale_hr_ub.op.axis[0], 'dma_copy')
    s[deqscale_hr_fb].emit_insn(deqscale_hr_fb.op.axis[0], 'dma_copy')
    s[reform_res_xw].emit_insn(i_n_inner, 'fixpipe_op')
    s[reform_res_hr].emit_insn(j_n_inner, 'fixpipe_op')
    s[reform_res_xw_hr].emit_insn(r_n_inner, 'vector_add')

    for tensor in elewise_tensors:
        insn = get_emit_insn_map(tensor)
        s[tensor].emit_insn(tensor.op.axis[0], insn)

    for tensor in elewise_before_barrier_tensors:
        if "offset_reform" in tensor.op.name:
            s[tensor].emit_insn(tensor.op.axis[0], 'vector_adds')
        if "scale_reform" in tensor.op.name:
            s[tensor].emit_insn(tensor.op.axis[0], 'vector_muls')
        if "scale_sqrt_ub" in tensor.op.name:
            s[tensor].emit_insn(tensor.op.axis[0], 'vector_muls')
        if "offset_ub" in tensor.op.name:
            s[tensor].emit_insn(tensor.op.axis[0], 'vector_adds')
        if "cast_" in tensor.op.name:
            s[tensor].emit_insn(tensor.op.axis[0], 'vector_conv_rint')

    s[sigmoid_i].emit_insn(sigmoid_i.op.axis[1], 'vector_sigmoid')
    s[tanh_j].emit_insn(tanh_j.op.axis[1], 'vector_tanh')
    s[sigmoid_f].emit_insn(sigmoid_f.op.axis[1], 'vector_sigmoid')
    s[sigmoid_o].emit_insn(sigmoid_o.op.axis[1], 'vector_sigmoid')
    s[c_t_tanh_ub].emit_insn(c_t_tanh_ub.op.axis[1], 'vector_tanh')

    s[update_c_gm].emit_insn(s[update_c_gm].op.axis[1], 'dma_copy')
    s[update_h_gm].emit_insn(s[update_h_gm].op.axis[1], 'dma_copy')
    s[update_h_gm_as_y].emit_insn(update_h_gm_as_y.op.axis[0], 'dma_copy')

    return return_list, s
