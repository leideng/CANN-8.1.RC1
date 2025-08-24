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
basic_gru_inplace_fill_window_cache
"""
# 'pylint: disable=too-many-lines
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_build
from tbe.dsl.compute.util import DTYPE_BYTE
from tbe.common.platform import platform_info
import numpy as np


class Constant:
    '''
    Constant of gru
    '''
    BLOCK_SIZE = int(platform_info.get_soc_spec("ubblock_size"))
    M0_SIZE = int(platform_info.get_soc_spec("cube_m_size"))
    N0_SIZE = int(platform_info.get_soc_spec("cube_n_size"))
    K0_SIZE = int(platform_info.get_soc_spec("cube_k_size"))
    GRU_LAYERS = 6
    CONST_ZERO = 0
    CONST_ONE = 1
    CONST_TWO = 2
    CONST_THREE = 3
    CONST_FOUR = 4
    LAST_DIM_INDEX = -1
    F162S8 = 2
    F162S16 = 6


# 'pylint: disable=unused-argument,too-many-arguments,too-many-return-statements,huawei-too-many-arguments
def check_supported(x, w, r, h, b, seq_length, clean_cache, deq_scale,
                    y, out_h, hidden_size, activation_alpha, activation_beta,
                    activations, clip=-1.0, direction="forward", linear_before_reset=1,
                    quant_scale_x=0.0, quant_offset_x=0.0, quant_sqrt_mode_x=False,
                    quant_scale_h=0.0, quant_offset_h=0.0, quant_sqrt_mode_h=False,
                    quant_dtype=2, kernel_name="basic_gru_inplace_fill_window_cache"):
    """
    Check supported for BasicGRUInplaceFillWindowCache.

    Parameters:
    x: dict. shape and dtype of input data x
    w: dict. shape and dtype of input data w
    r: dict. shape and dtype of input data r
    h: dict. shape and dtype of input data h
    b: dict. shape and dtype of input data b
    seq_length: dict. shape and dtype of input data seq_length
    clean_cache: dict. shape and dtype of input data clean_cache
    deq_scale: dict. shape and dtype of input data deq_scale
    y: dict. shape and dtype of output data y
    out_h: dict. shape and dtype of output data h
    hidden_size: value of attr hidden_size
    activation_alpha: value of attr activation_alpha
    activation_beta: value of attr activation_beta
    activations: value of attr activations
    clip: value of attr clip
    direction: value of attr direction
    linear_before_reset: value of attr input_forget
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

    (seq_length, _, input_size) = tuple(x.get("ori_shape"))
    (num_directions, _, hidden_size) = tuple(h.get("ori_shape"))
    if any((seq_length != 1, num_directions != 1)):
        return False, "seq_length/num_directions only support 1 for BasicGRUInplaceFillWindowCache"
    # only support network shape
    support_list = [(114, 114), ]
    if (input_size, hidden_size) not in support_list:
        return False, "no support for input_size/hidden_size value"

    return True, ""


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name,invalid-name
def basic_gru_inplace_fill_window_cache_check_dtype(x, w, r, h, b, seq_length, clean_cache, deq_scale, y, out_h):
    """
    check parameters dtype
    """
    x_dtype = x.get("dtype").lower()
    para_check.check_dtype(x_dtype, ["float16"], param_name="x")

    w_dtype = w.get("dtype").lower()
    para_check.check_dtype(w_dtype, ["int8"], param_name="w")

    r_dtype = r.get("dtype").lower()
    para_check.check_dtype(r_dtype, ["int8"], param_name="w")

    h_dtype = h.get("dtype").lower()
    para_check.check_dtype(h_dtype, ["float16"], param_name="h")

    bias_dtype = b.get("dtype").lower()
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

    output_h_dtype = out_h.get("dtype").lower()
    para_check.check_dtype(output_h_dtype, ["float16"], param_name="out_h")


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name,invalid-name
def basic_gru_inplace_fill_window_cache_check(x, w, r, h, b, deq_scale, out_h, direction, activation_alpha,
                                              activation_beta, activations, linear_before_reset, quant_dtype,
                                              quant_scale_x, quant_scale_h, quant_sqrt_mode_x, quant_sqrt_mode_h):
    shapes = [x, w, r, h, b]
    if all(shape["ori_shape"][0] != 1 for shape in shapes):
        error_manager_vector.raise_err_specific_reson("basic_gru_inplace_fill_window_cache",
        "please check x, w, r, h, b's ori_shape")

    if b is None or deq_scale is None:
        error_manager_vector.raise_err_specific_reson("basic_gru_inplace_fill_window_cache",
                                                      "bias/deq_scale is None, please check!")

    if h.get("shape") != out_h.get("shape"):
        error_manager_vector.raise_err_specific_reson("basic_gru_inplace_fill_window_cache",
        "h, out_h is different, please check!")

    if direction not in ["forward"]:
        error_manager_vector.raise_err_specific_reson("basic_gru_inplace_fill_window_cache",
        "attr direction only support 'forward' now, please check!")

    if any([activation_alpha, activation_beta, activations]):
        error_manager_vector.raise_err_specific_reson("basic_gru_inplace_fill_window_cache",
            "attr activation_alpha/activation_beta/activations is not None, please check!")

    if linear_before_reset != 1:
        error_manager_vector.raise_err_specific_reson("basic_gru_inplace_fill_window_cache",
        "attr linear_before_reset only support 1, please check!")

    if quant_dtype not in [Constant.F162S8, Constant.F162S16]:
        error_manager_vector.raise_err_specific_reson("basic_gru_inplace_fill_window_cache",
        "attr quant_dtype is not support, please check!")


def init_gm_tensor(x, w, r, h, b, seq_length, clean_cache, deq_scale, y):
    shape_x = x.get("shape")
    shape_w = w.get("shape")
    shape_r = r.get("shape")
    shape_h = h.get("shape")
    shape_h_new = list(shape_h[:])
    shape_b = list(b.get("shape"))
    shape_deq_scale = deq_scale.get("shape")
    shape_y = y.get("shape")

    x_dtype = x.get("dtype").lower()
    w_dtype = w.get("dtype").lower()
    b_dtype = b.get("dtype").lower()
    deq_scale_type = deq_scale.get("dtype").lower()

    x_gm = tvm.placeholder(shape=shape_x, dtype=x_dtype, name='x_gm')
    w_gm = tvm.placeholder(shape=shape_w, dtype=w_dtype, name='w_gm')
    r_gm = tvm.placeholder(shape=shape_r, dtype=w_dtype, name='r_gm')
    h_gm = tvm.placeholder(shape=shape_h_new, dtype=x_dtype, name='h_gm')
    b_gm = tvm.placeholder(shape=shape_b, dtype=b_dtype, name='b_gm')
    deq_scale_gm = tvm.placeholder(shape=shape_deq_scale, dtype=deq_scale_type, name='deq_scale_gm')
    in_tensors = [x_gm, w_gm, r_gm, h_gm, b_gm]

    if seq_length is not None:
        seq_length_gm = tvm.placeholder(shape=seq_length.get("shape"), dtype=b_dtype, name='seq_length_gm')
        in_tensors.append(seq_length_gm)

    if clean_cache is not None:
        clean_cache_gm = tvm.placeholder(shape=(1,), dtype=b_dtype, name='clean_cache_gm')
        in_tensors.append(clean_cache_gm)
    in_tensors.append(deq_scale_gm)

    out_h_gm = tvm.placeholder(shape=shape_h_new, dtype=x_dtype, name='out_h_gm')
    y_gm = tvm.placeholder(shape=shape_y, dtype=x_dtype, name='y_gm')
    out_tensors = [y_gm, out_h_gm]
    return in_tensors, out_tensors


def do_loop(gm_in_tensors, gm_out_tensors, direction):
    x_gm, h_gm = gm_in_tensors[0], gm_in_tensors[3]
    out_h_gm, y_gm = gm_out_tensors[0], gm_out_tensors[1]
    shape_x = x_gm.shape
    m1_size = cut_m = shape_x[2]
    cut_t = 1
    loop_t = shape_x[0] // cut_t
    loop_m = m1_size // cut_m
    for loop_i in range(0, loop_t):
        if direction == "REDIRECTIONAL":
            valid_loop_i = loop_t - 1 - loop_i
        else:
            valid_loop_i = loop_i
        for loop_j in range(0, loop_m):
            x_gm_new = x_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t, :,
                       loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            h_gm_new = h_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            out_h_gm_new = out_h_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t, :,
                           loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            y_gm_new = y_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t:,
                       :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            gm_in_tensors[0] = x_gm_new
            gm_in_tensors[3] = h_gm_new
            gm_out_tensors[0] = out_h_gm_new
            gm_out_tensors[1] = y_gm_new
    return gm_in_tensors, gm_out_tensors


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


def offset_compute(input_ub, input_shape, out_shape, attr_list):
    offset, reform_flag, scale, out_dtype = attr_list
    tensor_offset_ub_list = []

    if offset != 0 or scale == 1:
        offset_value = tvm.const(offset, "float16")
        if reform_flag:
            offset_ub = tvm.compute(out_shape, reform_compute(input_ub, input_shape, out_shape,
                                                              (True, offset_value, -1)),
                                    name=f"{input_ub.name}_offset", tag="vector_adds")
            tensor_offset_ub_list.append(offset_ub)
        else:
            offset_ub = tvm.compute(out_shape, lambda *indices: input_ub(*indices) + offset_value,
                                    name=f"{input_ub.name}_offset", tag="vector_adds")
            tensor_offset_ub_list.append(offset_ub)
        if out_dtype == "int16":
            res = tvm.compute(out_shape, lambda *indices: shape_util.cast(offset_ub(*indices), "int16"),
                              name=f"{input_ub.name}_cast_i16_ub", tag="vector_conv_rint")
            tensor_offset_ub_list.append(res)
        elif out_dtype == "int8":
            res = tvm.compute(out_shape, lambda *indices: shape_util.cast(offset_ub(*indices), "int8"),
                              name=f"{input_ub.name}_cast_i8_ub", tag="vector_conv_rint")
            tensor_offset_ub_list.append(res)
    else:
        if out_dtype == "int16":
            res = tvm.compute(out_shape, lambda *indices: shape_util.cast(input_ub(*indices), "int16"),
                              name=f"{input_ub.name}_cast_i16_ub", tag="vector_conv_rint")
            tensor_offset_ub_list.append(res)
        elif out_dtype == "int8":
            res = tvm.compute(out_shape, lambda *indices: shape_util.cast(input_ub(*indices), "int8"),
                              name=f"{input_ub.name}_cast_i8_ub", tag="vector_conv_rint")
            tensor_offset_ub_list.append(res)
    return tensor_offset_ub_list


def scale_compute(input_ub, input_shape, out_shape, attr_list):
    scale, offset, sqrt_mode, out_dtype = attr_list
    quant_ub_list = []
    if scale != 1:
        scale_value = tvm.const(scale, "float16")
        scale_ub = tvm.compute(out_shape, reform_compute(input_ub, input_shape, out_shape,
                                                         (False, -1, scale_value)), 
                                name=f"{input_ub.name}_scale", tag="vector_muls")
        quant_ub_list.append(scale_ub)
        if sqrt_mode:
            scale_sqrt_ub = tvm.compute(out_shape, lambda *indice: scale_ub(*indice) * scale_value,
                                        name=f"{input_ub.name}_scale_sqrt", tag="vector_muls")
            quant_ub_list.append(scale_sqrt_ub)
            tensor_offset_ub_list = offset_compute(scale_sqrt_ub, input_shape, out_shape,
                                                        (offset, False, scale, out_dtype))
            quant_ub_list.extend(tensor_offset_ub_list)
        else:
            tensor_offset_ub_list = offset_compute(scale_ub, input_shape, out_shape,
                                                        (offset, False, scale, out_dtype))
            quant_ub_list.extend(tensor_offset_ub_list)
    else:
        tensor_offset_ub_list = offset_compute(input_ub, input_shape, out_shape,
                                                    (offset, True, scale, out_dtype))
        quant_ub_list.extend(tensor_offset_ub_list)
    return quant_ub_list


def do_quant_x(ori_x_shape, x_ub, attr_x_list):
    x_shape = shape_util.shape_to_list(x_ub.shape)
    out_dtype = attr_x_list[-1]
    out_x_shape = after_quant_shape(x_shape, ori_x_shape, out_dtype)
    quant_x_list = scale_compute(x_ub, x_shape, out_x_shape, attr_x_list)
    return quant_x_list


def dp_quant_h(ori_h_shape, h_ub, attr_h_list):
    h_shape = shape_util.shape_to_list(h_ub.shape)
    out_dtype = attr_h_list[-1]
    out_h_shape = after_quant_shape(h_shape, ori_h_shape, out_dtype)
    quant_h_list = scale_compute(h_ub, h_shape, out_h_shape, attr_h_list)
    return quant_h_list


def do_quant(ori_x_shape, ori_h_shape, x_ub, h_ub, attr_list):
    _, _, _, quant_scale_x, quant_offset_x, quant_sqrt_mode_x, \
        quant_scale_h, quant_offset_h, quant_sqrt_mode_h, quant_dtype = attr_list

    out_dtype = dst_type_conversion(quant_dtype)

    attr_x_list = [quant_scale_x, quant_offset_x, quant_sqrt_mode_x, out_dtype]
    attr_h_list = [quant_scale_h, quant_offset_h, quant_sqrt_mode_h, out_dtype]

    quant_x_list = do_quant_x(ori_x_shape, x_ub, attr_x_list)
    quant_h_list = dp_quant_h(ori_h_shape, h_ub, attr_h_list)
    return quant_x_list, quant_h_list


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


def after_quant_shape(input_shape, ori_shape, out_dtype):
    out_shape = input_shape[:]
    if out_dtype == "int8":
        actual_k_size = ori_shape[-1]
        out_shape[-4] = (actual_k_size + Constant.K0_SIZE - 1) // Constant.K0_SIZE
        out_shape[-1] = Constant.K0_SIZE
    return out_shape


def do_mm_fixpipe_reform(lmatrix, rmatrix, bias, deq_scale_fb, index, ori_m, ori_k, ori_n):
    m0 = Constant.M0_SIZE
    n0 = Constant.N0_SIZE // DTYPE_BYTE.get(rmatrix.dtype)
    l_k0 = Constant.K0_SIZE // DTYPE_BYTE.get(lmatrix.dtype)
    r_k0 = Constant.K0_SIZE // DTYPE_BYTE.get(rmatrix.dtype)

    m1 = (lmatrix.shape[-3] * lmatrix.shape[-2]) // m0 
    n1 = (rmatrix.shape[-3] * rmatrix.shape[-2]) // n0
    k_size = lmatrix.shape[-4] * lmatrix.shape[-1]

    mm_res_shape = [lmatrix.shape[0], n1, m1, m0, n0]

    reform_shape = [mm_res_shape[Constant.CONST_ZERO], mm_res_shape[Constant.CONST_ONE] * 2,
                    mm_res_shape[Constant.CONST_TWO], mm_res_shape[Constant.CONST_THREE], 
                    mm_res_shape[Constant.LAST_DIM_INDEX] // 2]

    k_axis = tvm.reduce_axis((0, k_size), name='k_axis')
    mm_op_dict = {"use_bias": True,
                  "matrix_m": ori_m,
                  "matrix_n": ori_n,
                  "matrix_k": ori_k}
    mm_res = tvm.compute(mm_res_shape,
                         lambda batch, n1, m1, m0, n0: tvm.sum(
                             tvm.matmul_op(
                                 lmatrix[batch, k_axis // l_k0, m1, m0, k_axis % l_k0],
                                 rmatrix[k_axis // r_k0, n1, n0, k_axis % r_k0],
                                 bias[n1 * Constant.N0_SIZE + n0],
                                 dst_dtype="int32",
                                 op_dict=mm_op_dict),
                             axis=[k_axis, ]),
                         name=f'mm_res_{index}',
                         tag="matmul_ub_to_ub")

    fxp_res = tvm.compute(mm_res_shape, lambda batch, n1, m1, m0, n0: tvm.fixpipe_op(
                                                    mm_res[batch, n1, m1, m0, n0],
                                                    "float16",
                                                    pre_conv_param=deq_scale_fb[n1 * Constant.N0_SIZE + n0],
                                                    op_dict={"pre_conv": "VS322F16"}),
                                                    name=f'fixpipe_res_{index}')

    reform_res = tvm.compute(reform_shape, lambda batch, n1, m1, m0, n0:
    fxp_res[batch, (n1 * 8 + n0) // 16, m1, m0, (n1 * 8 + n0) % 16],
                             name=f'reform_res_{index}')
    return mm_res, fxp_res, reform_res


def do_matmul(mm_paras):
    quant_x, quant_h, w_list, r_list, b_list, deq_list, ori_m, ori_k, ori_n = mm_paras
    mm_ub_list, fxp_list, reform_list = [], [], []

    for i in range(3):
        mm_res, fxp_res, reform_res = do_mm_fixpipe_reform(quant_x, w_list[i], b_list[i], deq_list[i],
        i, ori_m, ori_k, ori_n)
        mm_ub_list.append(mm_res)
        fxp_list.append(fxp_res)
        reform_list.append(reform_res)
    for i in range(3, 6):
        mm_res, fxp_res, reform_res = do_mm_fixpipe_reform(quant_h, r_list[i - 3], b_list[i], deq_list[i],
        i, ori_m, ori_k, ori_n)
        mm_ub_list.append(mm_res)
        fxp_list.append(fxp_res)
        reform_list.append(reform_res)
    return mm_ub_list, fxp_list, reform_list


def do_output(add_final, res_shape):
    y_gm = tvm.compute(res_shape, lambda *indices: add_final(*indices), name="y_gm", tag="dma_copy")

    new_out_shape = list(res_shape[:])
    new_out_shape[0] = new_out_shape[0] * 2 
    out_h_gm = tvm.compute(new_out_shape, lambda batch, n1, m1, m0, n0: tvm.select(batch > 0, 
    add_final(batch - 1, n1, m1, m0, n0), add_final(batch, n1, m1, m0, n0)), 
    name="out_h_gm", tag="dma_copy")

    res_list = [y_gm, out_h_gm]
    return res_list


def do_vector(reform_list, h_ub, attr_list, res_shape):
    reform_res_00, reform_res_01, reform_res_02, reform_res_03, reform_res_04, reform_res_05 = reform_list
    input_dtype = h_ub.dtype
    out_shape = h_ub.shape

    add_01 = tvm.compute(out_shape, lambda *indices: reform_res_00(*indices) + reform_res_03(*indices), 
    name="add_01", tag="vector_add")
    sigmoid_01 = tvm.compute(out_shape, lambda *indices: tvm.sigmoid(add_01(*indices)), 
    name="sigmoid_01", tag="vector_sigmoid")
    negtive_const_one = tvm.const(-1.0, dtype=input_dtype)
    vmuls_01 = tvm.compute(out_shape, lambda *indices: sigmoid_01(*indices) * negtive_const_one, 
    name="vmuls_01", tag="vector_muls")
    postive_const_one = tvm.const(1.0, dtype=input_dtype)
    vadds_ub = tvm.compute(out_shape, lambda *indices: vmuls_01(*indices) + postive_const_one, 
    name="vadds_ub", tag="vector_adds")

    add_02 = tvm.compute(out_shape, lambda *indices: reform_res_01(*indices) + reform_res_04(*indices), 
    name="add_02", tag="vector_add")
    sigmoid_02 = tvm.compute(out_shape, lambda *indices: tvm.sigmoid(add_02(*indices)), 
    name="sigmoid_02", tag="vector_sigmoid")
    mul_01 = tvm.compute(out_shape, lambda *indices: sigmoid_02(*indices) * reform_res_05(*indices), 
    name="mul_01", tag="vector_mul")

    add_03 = tvm.compute(out_shape, lambda *indices: reform_res_02(*indices) + mul_01(*indices), 
    name="add_03", tag="vector_add")
    tanh_ub = tvm.compute(out_shape, lambda *indices: tvm.tanh(add_03(*indices)), 
    name="tanh_ub", tag="vector_tanh")

    mul_02 = tvm.compute(out_shape, lambda *indices: vadds_ub(*indices) * tanh_ub(*indices), 
    name="mul_02", tag="vector_mul")
    mul_03 = tvm.compute(out_shape, lambda *indices: sigmoid_01(*indices) * h_ub(*indices), 
    name="mul_03", tag="vector_mul")
    add_final = tvm.compute(out_shape, lambda *indices: mul_02(*indices) + mul_03(*indices), 
    name="add_final", tag="vector_add")

    res_list = do_output(add_final, res_shape)

    elewise_tensors = [add_01, sigmoid_01, vmuls_01, vadds_ub, add_02, sigmoid_02, mul_01, add_03,
    tanh_ub, mul_02, mul_03, add_final]

    return res_list, elewise_tensors


def do_clean_cache(gm_in_tensors, out_dtype):
    def lambda_func(indices):
        return tvm.select(clean_cache_ub[0] > clean_cache_one,
                          h_gm(*indices),
                          (tvm.select(clean_cache_ub[0] < clean_cache_one,
                          h_gm(indices[0] + 1, *indices[1:]),
                          pad_zero)))

    clean_cache_gm, h_ub, clean_cache_ub = None, None, None
    h_gm = gm_in_tensors[3]
    shape_h = h_gm.shape
    pad_zero = tvm.const(0, dtype=h_gm.dtype)

    for tensor in gm_in_tensors:
        if tensor.name == "clean_cache_gm":
            clean_cache_gm = tensor
            clean_cache_dtype = clean_cache_gm.dtype
            clean_cache_one = tvm.const(1, dtype=clean_cache_dtype)

    if clean_cache_gm is not None:
        clean_cache_ub = tvm.compute((1,), lambda *indices: clean_cache_gm(*indices), 
                                                            name="clean_cache_ub", tag="dma_copy")

    if out_dtype == "int8":
        read_h_shape = shape_h[:]
        read_h_shape[-4] = (read_h_shape[-4] + 2 - 1) // 2 * 2
        if clean_cache_gm is None:
            h_ub = tvm.compute(read_h_shape, lambda *indices: tvm.select(indices[1] <= shape_h[1] - 1, 
                                                                            h_gm(*indices),
                                                                            pad_zero), name="h_ub", tag="dma_copy")
        else:
            h_ub = tvm.compute(read_h_shape, lambda *indices: tvm.select(indices[1] <= shape_h[1] - 1, 
                                                                         (lambda_func(indices)), pad_zero),
                                                                          name="h_ub", tag="dma_copy")
    else:
        if clean_cache_gm is None:
            h_ub = tvm.compute(shape_h, lambda *indices: h_gm(*indices), 
                                                         name="h_ub", tag="dma_copy")
        else:
            h_ub = tvm.compute(shape_h, lambda *indices: lambda_func(indices),
                                        name="h_ub", tag="dma_copy")
    return h_ub, clean_cache_ub


def do_w_r_bias(w_gm, r_gm, b_gm):
    new_w_shape = w_gm.shape[:]
    split_len_w = new_w_shape[-3] = new_w_shape[-3] // 3

    new_r_shape = r_gm.shape[:]
    split_len_r = new_r_shape[-3] = new_r_shape[-3] // 3

    hidden_size = r_gm.shape[-4] * r_gm.shape[-1]
    bias_shape = (hidden_size,)

    w_list, r_list = [], []
    split_dim = 1
    for i in range(3):
        w_ub = tvm.compute(new_w_shape, lambda *indices: w_gm(*indices[:split_dim], indices[split_dim] + 
                                                               split_len_w * i, *indices[split_dim + 1:]), 
                                                               name=f"w_ub_{i}", 
                                                               tag="dma_copy")
        w_list.append(w_ub)
        r_ub = tvm.compute(new_r_shape, lambda *indices: r_gm(*indices[:split_dim], indices[split_dim] + 
                                                              split_len_r * i, *indices[split_dim + 1:]), 
                                                              name=f"r_ub_{i}", 
                                                              tag="dma_copy")
        r_list.append(r_ub)

    # get b_ub
    b_list = []
    for i in range(6):
        b_ub = tvm.compute(bias_shape, lambda *indices: b_gm(indices[0] + hidden_size * i), 
                                                        name=f"b_ub_{i}", 
                                                        tag="dma_copy")
        b_list.append(b_ub)
    return w_list, r_list, b_list


def do_deq_scale(deq_scale_gm, r_gm):
    hidden_size = r_gm.shape[-4] * r_gm.shape[-1]

    is_per_tensor = True if deq_scale_gm.shape[0] == Constant.GRU_LAYERS else False

    shape_deq_ub = [1] if is_per_tensor else [hidden_size]
    shape_deq_broadcast = [hidden_size]
    shape_deq_fb = [hidden_size]

    deq_ub_list, fb_list, deq_list = [], [], []

    for i in range(Constant.GRU_LAYERS):
        deq_ub = tvm.compute(shape_deq_ub, lambda *indices: deq_scale_gm(indices[0] + shape_deq_ub[0] * i), 
                                                            name=f"deq_scale_ub_{i}", 
                                                            tag="dma_copy")
        deq_ub_list.append(deq_ub)

    if is_per_tensor:
        for i in range(Constant.GRU_LAYERS):
            deq_broadcast_ub = tvm.compute(shape_deq_broadcast, lambda *indices: deq_ub_list[i](indices[0]),
                                                                name=f"deq_broadcast_ub_{i}", 
                                                                tag="dma_copy")

            deq_ub_list.append(deq_broadcast_ub)

            deq_fb = tvm.compute(shape_deq_fb, lambda *indices: deq_broadcast_ub[indices[0] // hidden_size],
                                                                name=f"deq_scale_fb_{i}", 
                                                                tag="dma_copy")
            fb_list.append(deq_fb)
            deq_list.append(deq_fb)
    else:
        for i in range(Constant.GRU_LAYERS):
            deq_fb = tvm.compute(shape_deq_fb, lambda *indices: deq_ub_list[i](*indices),
                                                                name=f"deq_scale_fb_{i}", 
                                                                tag="dma_copy")
            fb_list.append(deq_fb)
            deq_list.append(deq_fb)
    return deq_ub_list, fb_list, deq_list


def do_input(gm_in_tensors, quant_dtype):
    x_gm, w_gm, r_gm, h_gm, b_gm = gm_in_tensors[:5]
    deq_scale_gm = gm_in_tensors[-1]
    # get x_ub
    out_dtype = dst_type_conversion(quant_dtype)
    x_shape = x_gm.shape
    h_shape = h_gm.shape
    if out_dtype == "int8":
        read_x_shape = x_shape[:]
        read_x_shape[-4] = (read_x_shape[-4] + 2 - 1) // 2 * 2
        zero = tvm.const(0, dtype=x_gm.dtype)
        x_ub = tvm.compute(read_x_shape,
                          lambda *indices: tvm.select(indices[1] <= x_shape[1] - 1, x_gm(*indices), zero),
                          name=f'x_ub', tag="dma_copy")
    else:
        x_ub = tvm.compute(x_gm.shape, lambda *indices: x_gm(*indices), name="x_ub", tag="dma_copy")

    # get h_ub, clean_cache_ub
    h_ub, clean_cache_ub = do_clean_cache(gm_in_tensors, out_dtype)

    w_list, r_list, b_list = do_w_r_bias(w_gm, r_gm, b_gm)

    deq_ub_list, fb_list, deq_list = do_deq_scale(deq_scale_gm, r_gm)

    input_tensor_list = [x_ub, h_ub, clean_cache_ub]
    input_tensor_list.extend(w_list)
    input_tensor_list.extend(r_list)
    input_tensor_list.extend(b_list)
    input_tensor_list.extend(deq_ub_list)
    ub_in_tensors = [x_ub, h_ub, w_list, r_list, clean_cache_ub, b_list, deq_ub_list]
    res = (ub_in_tensors, fb_list, deq_list, input_tensor_list)
    return res


def basic_gru_inplace_fill_window_cache_compute(in_para, gm_in_tensors, attr_list):
    ori_x_shape = in_para[0].get("ori_shape")
    ori_h_shape = in_para[1].get("ori_shape")
    ori_w_shape = in_para[2].get("ori_shape")
    res_shape = in_para[1].get("shape")

    ori_m = ori_x_shape[-2]
    ori_k = ori_x_shape[-1]
    ori_n = ori_w_shape[-2]

    ub_list = []
    quant_type = attr_list[-1]
    res = do_input(gm_in_tensors, quant_type)
    ub_in_tensors, fb_list, deq_list, input_tensor_list = res
    x_ub, h_ub, w_list, r_list, clean_cache_ub, b_list, deq_ub_list = ub_in_tensors
    ub_list.extend(input_tensor_list)
    ub_list = [x for x in ub_list if x is not None]

    quant_x_list, quant_h_list = do_quant(ori_x_shape, ori_h_shape, x_ub, h_ub, attr_list)
    ub_list.extend(quant_x_list) 
    ub_list.extend(quant_h_list) 
    quant_x, quant_h = quant_x_list[-1], quant_h_list[-1]

    mm_paras = (quant_x, quant_h, w_list, r_list, b_list, deq_list, ori_m, ori_k, ori_n)
    mm_ub_list, fxp_list, reform_list = do_matmul(mm_paras)
    ub_list.extend(mm_ub_list) 
    ub_list.extend(fxp_list) 
    ub_list.extend(reform_list) 

    res_list, elewise_list = do_vector(reform_list, h_ub, attr_list, res_shape)
    ub_list.extend(elewise_list) 

    return {
        "res_list": res_list,
        "ub_list": ub_list,
        "fb_list": fb_list,
        "quant_x_list": quant_x_list,
        "quant_h_list": quant_h_list,
        "mm_ub_list": mm_ub_list,
        "fxp_list": fxp_list,
        "reform_list": reform_list,
        "elewise_tensors": elewise_list,
    }


def get_buffer_size(m, n, k1, k2, is_clean_cache, is_per_tensor):
    # when no cleancache and is-per-channel are met, the bytes of all buffers are func_no_cc.
    func_no_cc = 4 * m * k1 + 4 * m * k2 + 3 * n * k1 + 3 * n * k2 + 24 * n + 24 * m * n
    if is_clean_cache and not is_per_tensor:
        # cleancache'shape size is 1 and dtype is int32, 4 means the bytes of clean cache ub.
        return func_no_cc + 4
    elif not is_clean_cache and is_per_tensor:
        # deqscale'shape size is 6 and dtype is int64, 48 means the bytes of deq scale ub.
        return func_no_cc + 48
    elif is_clean_cache and is_per_tensor:
        # there are two more buffers, clean cache ub and deq scale ub.
        return func_no_cc + 48 + 4
    else:
        return func_no_cc


def get_min_repeat(m, n, k1, k2, is_clean_cache, is_per_tensor, ub_size):
    repeat_list = []
    if get_buffer_size(1, 16, k1, k2, is_clean_cache, is_per_tensor) < ub_size:
        repeat_size = (m - 1) * k1 * n + (m - 1) * k2 * n
        repeat_list.append((901, repeat_size))
    if get_buffer_size(1, 16, 16, k2, is_clean_cache, is_per_tensor) < ub_size:
        repeat_size = None
        repeat_list.append((902, repeat_size))
    if get_buffer_size(1, 16, k1, 16, is_clean_cache, is_per_tensor) < ub_size:
        repeat_size = None
        repeat_list.append((903, repeat_size))
    if get_buffer_size(1, 16, 16, 16, is_clean_cache, is_per_tensor) < ub_size:
        repeat_size = None
        repeat_list.append((904, repeat_size))
    tiling_case = min(repeat_list, key=lambda x: x[1])[0]
    factor_list = get_factors(tiling_case)
    return factor_list


def get_factors(tiling_case, m, n, k1, k2):
    factors = {
        100: (m, n, k1, k2, False, False),  # not cut
        200: (1, n, k1, k2, False, False),  # only cut m
        300: (m, 16, k1, k2, False, False),  # only cut n
        400: (m, n, 16, k2, True, False),  # only cut k1
        500: (m, n, k1, 16, False, True),   # only cut k2
        600: (m, 16, 16, n, True, True),   # cut k1 and k2
        700: (1, n, 16, k2, True, False),  # cut m and k1
        800: (m, 16, k1, k2, False, False),  # cut m and k2
        900: (m, 16, k1, 16, False, True),  # cut n and k2
        110: (m, 16, 16, k2, False, True),  # cut n and k1
        210: (1, n, 16, 16, True, True),  # cut m, k1, k2
        310: (m, 16, 16, 16, True, True),  # cut n, k1, k2
        901: (1, 16, k1, k2, False, False),  # repeat 1
        902: (1, 16, k1, 16, False, True),  # repeat 2
        903: (1, 16, 16, k2, True, False),  # repeat 3
        904: (1, 16, 16, 16, True, True),  # repeat 4
    }
    factor_list = factors.get(tiling_case)
    return factor_list


def get_tiling_para(gm_in_tensors):
    batch_size = gm_in_tensors[0].shape[-2] * gm_in_tensors[0].shape[-3] 
    input_size = gm_in_tensors[0].shape[-4] * gm_in_tensors[0].shape[-1]
    hidden_size = gm_in_tensors[3].shape[-4] * gm_in_tensors[3].shape[-1]
    input_size_align = ((input_size + (16 - 1)) // 16) * 16
    hidden_size_align = ((hidden_size + (16 - 1)) // 16) * 16
    k1, k2, m, n = input_size_align // 8, hidden_size_align // 8, batch_size // 1, hidden_size_align // 8
    db_flag = True if hidden_size >= 16 else False
    paras = (k1, k2, m, n, db_flag)
    return paras


def basic_gru_inplace_fill_window_cache_tilling(gm_in_tensors):
    paras = get_tiling_para(gm_in_tensors)
    k1, k2, m, n, db_flag = paras
    ub_size = tbe_platform.get_soc_spec("UB_SIZE")
    deq_scale_tensor = None
    for tensor in gm_in_tensors:
        is_clean_cache = True if tensor.op.name == "clean_cache_gm" else False
        if tensor.op.name == "deq_scale_gm":
            deq_scale_tensor = tensor
            is_per_tensor = True if deq_scale_tensor.shape[0] == Constant.GRU_LAYERS else False

    buffer_size_list = [
        (get_buffer_size(m, n, k1, k2, is_clean_cache, is_per_tensor), 100), # not cut
        (get_buffer_size(1, n, k1, k2, is_clean_cache, is_per_tensor), 200), # only cut m
        (get_buffer_size(m, 16, k1, k2, is_clean_cache, is_per_tensor), 300),  # only cut n
        (get_buffer_size(m, n, 16, k2, is_clean_cache, is_per_tensor), 400), # only cut k1
        (get_buffer_size(m, n, k1, 16, is_clean_cache, is_per_tensor), 500), # only cut k2
        (get_buffer_size(m, n, 16, 16, is_clean_cache, is_per_tensor), 600), # cut k1 and k2
        (get_buffer_size(1, n, 16, k2, is_clean_cache, is_per_tensor), 700), # cut m and k1
        (get_buffer_size(1, n, k1, 16, is_clean_cache, is_per_tensor), 800), # cut m and k2
        (get_buffer_size(m, 16, k1, 16, is_clean_cache, is_per_tensor), 900), # cut n and k2
        (get_buffer_size(m, 16, 16, k2, is_clean_cache, is_per_tensor), 110), # cut n and k1
        (get_buffer_size(1, n, 16, 16, is_clean_cache, is_per_tensor), 210), # cut m, k1, k2
        (get_buffer_size(m, 16, 16, 16, is_clean_cache, is_per_tensor), 310), # cut n, k1, k2
    ]

    need_repeat_trans = False
    for (func, tiling_case) in buffer_size_list:
        if func < ub_size:
            factor_list = get_factors(tiling_case, m, n, k1, k2)
            break
        need_repeat_trans = True

    if need_repeat_trans:
        factor_list = get_min_repeat(m, n, k1, k2, is_clean_cache, is_per_tensor, ub_size)
    m1_ub_factor, k1_ub_factor, k2_ub_factor, n1_ub_factor, cut_k1_flag, cut_k2_flag = factor_list
    tiling_data = {
        "db_flag": False,
        "m1_factor": m1_ub_factor,
        "k1_factor": k1_ub_factor,
        "k2_factor": k2_ub_factor,
        "n1_factor": n1_ub_factor,
        "cut_k1_flag": cut_k1_flag,
        "cut_k2_flag": cut_k2_flag,
    }
    return tiling_data


def do_emit_insn(sch, reform_list, ub_list, fb_list):
    for tensor in reform_list:
        sch[tensor].emit_insn(tensor.op.axis[0], 'fixpipe_op')
    for tensor in ub_list:
        if tensor.op.tag == "vector_add":
            sch[tensor].emit_insn(tensor.op.axis[0], 'vector_add')
        elif tensor.op.tag == "vector_adds":
            sch[tensor].emit_insn(tensor.op.axis[0], 'vector_adds')
        elif tensor.op.tag == "vector_mul":
            sch[tensor].emit_insn(tensor.op.axis[0], 'vector_mul')
        elif tensor.op.tag == "vector_muls":
            sch[tensor].emit_insn(tensor.op.axis[0], 'vector_muls')
        elif tensor.op.tag == "vector_sigmoid":
            sch[tensor].emit_insn(tensor.op.axis[0], 'vector_sigmoid')
        elif tensor.op.tag == "vector_tanh":
            sch[tensor].emit_insn(tensor.op.axis[0], 'vector_tanh')
        elif tensor.op.tag == "dma_copy":
            sch[tensor].emit_insn(tensor.op.axis[0], 'dma_copy')
        elif tensor.op.tag == "vector_conv_rint":
            sch[tensor].emit_insn(tensor.op.axis[0], 'vector_conv_rint')
    for tensor in fb_list:
        sch[tensor].emit_insn(tensor.op.axis[0], 'dma_copy')


def do_no_cut_split(sch, m1_ub_factor, n1_ub_factor, y_gm, out_h_gm, cut_m_list, cut_n_list):
    y_m1_outer, y_m1_inner = sch[y_gm].split(y_gm.op.axis[-3], factor=m1_ub_factor)
    y_n1_outer, y_n1_inner = sch[y_gm].split(y_gm.op.axis[-4], factor=n1_ub_factor)
    sch[y_gm].reorder(y_m1_outer, y_m1_inner, y_n1_outer, y_n1_inner, 
                        y_gm.op.axis[0], y_gm.op.axis[-2], y_gm.op.axis[-1])
    
    h_m1_outer, h_m1_inner = sch[out_h_gm].split(out_h_gm.op.axis[-3], factor=m1_ub_factor)
    h_n1_outer, h_n1_inner = sch[out_h_gm].split(out_h_gm.op.axis[-4], factor=n1_ub_factor)
    sch[out_h_gm].reorder(h_m1_outer, h_m1_inner, h_n1_outer, h_n1_inner, 
                            out_h_gm.op.axis[0], out_h_gm.op.axis[-2], out_h_gm.op.axis[-1], )

    sch.compute_with([y_gm, out_h_gm], 4)

    for tensor in cut_m_list:
        sch[tensor].compute_at(sch[y_gm], y_m1_outer)
    for tensor in cut_n_list:
        sch[tensor].compute_at(sch[y_gm], y_n1_outer)

    sch[y_gm].emit_insn(y_n1_inner, 'dma_copy')
    sch[out_h_gm].emit_insn(h_n1_inner, 'dma_copy')


def do_set_scope(sch, ub_list, fb_list):
    for tensor in ub_list:
        sch[tensor].set_scope("local.UB")
    for tensor in fb_list:
        sch[tensor].set_scope("local.FB")


def do_reused_by(sch, ub_list, gm_in_tensors, out_h_gm):
    sub_tensor, add_tensor_01, mul_tensor_01, add_tensor_02, mul_tensor_02, add_tensor_03 = \
        None, None, None, None, None, None
    for tensor in gm_in_tensors:
        if tensor.op.name == "h_gm":
            sch[tensor].reused_by(out_h_gm)
    for tensor in ub_list:
        if tensor.op.name == "vadds_ub":
            sub_tensor = tensor
        if tensor.op.name == "add_01":
            add_tensor_01 = tensor
        if tensor.op.name == "mul_01":
            mul_tensor_01 = tensor
        if tensor.op.name == "add_02":
            add_tensor_02 = tensor
        if tensor.op.name == "mul_02":
            mul_tensor_02 = tensor
        if tensor.op.name == "add_03":
            add_tensor_03 = tensor
    sch[sub_tensor].reused_by(add_tensor_01)
    sch[mul_tensor_01].reused_by(add_tensor_02)
    sch[mul_tensor_02].reused_by(add_tensor_03)


def do_compute_inline(sch, mm_ub_list, fxp_list):
    for tensor in mm_ub_list:
        sch[tensor].compute_inline()
    for tensor in fxp_list:
        sch[tensor].compute_inline()


def get_mn_list(ub_list, mm_ub_list, fxp_list, fb_list, quant_x_list, quant_h_list):
    compute_at_list = ub_list[:]
    for tensor in compute_at_list:
        if tensor.op.name == "clean_cache_ub":
            compute_at_list.remove(tensor)
    compute_at_list = list(set(compute_at_list) - set(mm_ub_list) - set(fxp_list))
    compute_at_list.extend(fb_list)

    cut_m_list = []
    cut_k1_list = []
    cut_k2_list = []
    cut_m_list.extend(quant_x_list)
    cut_m_list.extend(quant_h_list)

    for tensor in ub_list:
        if tensor.op.name == "x_ub":
            cut_m_list.append(tensor)
        elif tensor.op.name == "h_ub":
            cut_m_list.append(tensor)

    cut_m_list = list(set(cut_m_list))
    cut_n_list = list(set(compute_at_list) - set(cut_m_list))
    return cut_m_list, cut_n_list


def basic_gru_inplace_fill_window_cache_schedule(gm_in_tensors, compute_res_dict, tiling_data):
    # sourcery no-metric
    # init
    res_list = compute_res_dict.get("res_list")
    ub_list = compute_res_dict.get("ub_list")
    fb_list = compute_res_dict.get("fb_list")
    quant_x_list = compute_res_dict.get("quant_x_list")
    quant_h_list = compute_res_dict.get("quant_h_list")
    mm_ub_list = compute_res_dict.get("mm_ub_list")
    fxp_list = compute_res_dict.get("fxp_list")
    reform_list = compute_res_dict.get("reform_list")
    elewise_list = compute_res_dict.get("elewise_tensors")

    y_gm, out_h_gm = res_list
    sch = tvm.create_schedule([y_gm.op, out_h_gm.op])
    # set scope
    do_set_scope(sch, ub_list, fb_list)
    # reused by
    do_reused_by(sch, ub_list, gm_in_tensors, out_h_gm)
    # compute inline
    do_compute_inline(sch, mm_ub_list, fxp_list)
    cut_m_list, cut_n_list = get_mn_list(ub_list, mm_ub_list, fxp_list, fb_list, quant_x_list, quant_h_list)
    # tiling data
    enable_db = tiling_data.get("db_flag")
    m1_ub_factor = tiling_data.get("m1_factor")
    n1_ub_factor = tiling_data.get("n1_factor")
    k1_ub_factor = tiling_data.get("k1_factor")
    k2_ub_factor = tiling_data.get("k2_factor")
    cut_k1_flag = tiling_data.get("cut_k1_flag")
    cut_k2_flag = tiling_data.get("cut_k2_flag")
    # double buffer
    if enable_db:
        for tensor in ub_list:
            sch[tensor].double_buffer()
    # split, reorder and compute_at
    if cut_k1_flag and not cut_k2_flag:
        pass
    elif cut_k2_flag and not cut_k1_flag:
        pass
    elif cut_k1_flag and cut_k2_flag:
        pass
    else:
        do_no_cut_split(sch, m1_ub_factor, n1_ub_factor, y_gm, out_h_gm, cut_m_list, cut_n_list)

    # emit_insn
    do_emit_insn(sch, reform_list, ub_list, fb_list)
    return sch, res_list


# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-function-args,too-many-statements
# 'pylint: disable=unused-argument
@register_operator("BasicGRUInplaceFillWindowCache")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_LIST_FLOAT, para_check.OPTION_ATTR_LIST_FLOAT,
                            para_check.NONE_TYPE, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def basic_gru_inplace_fill_window_cache(x, w, r, h, b, seq_length, clean_cache, deq_scale,
                                        y, out_h, hidden_size, activation_alpha, activation_beta,
                                        activations, clip=-1.0, direction="forward", linear_before_reset=1,
                                        quant_scale_x=0.0, quant_offset_x=0.0, quant_sqrt_mode_x=False,
                                        quant_scale_h=0.0, quant_offset_h=0.0, quant_sqrt_mode_h=False,
                                        quant_dtype=2, kernel_name="basic_gru_inplace_fill_window_cache"):
    """
    basic_gru_inplace_fill_window_cache op.

    Parameters:

    Inputs:
    x: Dict, required, dtype=float16
    w: Dict, required, dtype=int8
    r: Dict, required, dtype=int8
    h: Dict, required, dtype=float16
    b: Dict, optional, dtype=int32
    seq_length: Dict, optional, dtype=int32
    clean_cache: Dict, optional, dtype=int32
    deq_scale: Dict, optional, dtype=uint64

    Outputs:
    y: Dict, required, dtype=float16
    out_h: Dict, required, dtype=float16

    Attributes:
    hidden_size: required, an int number
    activation_alpha: optional, a list of float
    activation_beta: optional, a list of float
    activations: optional, a list of string
    clip: optional, a float number, default is -0.1
    direction: optional, string , only support "forward"
    linear_before_reset: optional, an int number, default is 1
    quant_scale_x: optional, a float number, default is 0.0
    quant_offset_x: optional, a float number, default is 0.0
    quant_sqrt_mode_x: optional, a bool number, default is false
    quant_scale_h: optional, a float number, default is 0.0
    quant_offset_h: optional, a float number, default is 0.0
    quant_sqrt_mode_h: optional, a float number, default is false
    quant_dtype: optional, a int number, 2 means quant to int8, 6 means quant to int16
    """
    # check param and attr
    basic_gru_inplace_fill_window_cache_check(x, w, r, h, b, deq_scale, out_h, direction, activation_alpha,
                                              activation_beta, activations, linear_before_reset, quant_dtype,
                                              quant_scale_x, quant_scale_h, quant_sqrt_mode_x, quant_sqrt_mode_h)

    basic_gru_inplace_fill_window_cache_check_dtype(x, w, r, h, b, seq_length, clean_cache, deq_scale, y, out_h)

    if quant_scale_x >= 65504:
        if quant_sqrt_mode_x:
            error_manager_vector.raise_err_specific_reson("basic_gru_inplace_fill_window_cache",
        "quant_scale_x should not over 65504")
        else:
            quant_scale_x = np.sqrt(quant_scale_x)
            quant_sqrt_mode_x = True

    if quant_scale_h >= 65504:
        if quant_sqrt_mode_h:
            error_manager_vector.raise_err_specific_reson("basic_gru_inplace_fill_window_cache",
        "quant_scale_h should not over 65504")
        else:
            quant_scale_h = np.sqrt(quant_scale_h)
            quant_sqrt_mode_h = True

    gm_in_tensors, gm_out_tensors = init_gm_tensor(x, w, r, h, b, seq_length, clean_cache, deq_scale, y)

    if x.get("shape")[0] != 1:
        gm_in_tensors, gm_out_tensors = do_loop(gm_in_tensors, gm_out_tensors, direction)

    attr_list = [hidden_size, clip, linear_before_reset, quant_scale_x, quant_offset_x, quant_sqrt_mode_x, 
    quant_scale_h, quant_offset_h, quant_sqrt_mode_h, quant_dtype]

    in_para = [x, h, w]

    compute_res_dict = basic_gru_inplace_fill_window_cache_compute(in_para, gm_in_tensors, attr_list)

    tiling_data = basic_gru_inplace_fill_window_cache_tilling(gm_in_tensors)

    sch, res_list = basic_gru_inplace_fill_window_cache_schedule(gm_in_tensors, compute_res_dict, tiling_data)

    with tbe_build.build_config():
        tvm.build(sch, [*gm_in_tensors, *res_list], "cce", name=kernel_name)
