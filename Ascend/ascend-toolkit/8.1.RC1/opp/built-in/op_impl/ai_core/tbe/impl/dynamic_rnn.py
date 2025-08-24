#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
dynamic_rnn
"""
# 'pylint: disable=too-many-lines
import operator

import numpy as np
from impl.util.platform_adapter import tbe_platform
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from te.lang.cce import broadcast
from te.lang.cce import cast_to
from te.lang.cce import vabs
from te.lang.cce import vadd
from te.lang.cce import vadds
from te.lang.cce import vdiv
from te.lang.cce import vexp
from te.lang.cce import vmul
from te.lang.cce import vmuls
from te.lang.cce import vrec
from te.lang.cce import vsub
from te.lang.cce import vmins
from te.lang.cce import vmaxs
from te.domain.rl_bank import rl_bank
from te.platform import scope_ca
from te.platform import scope_cb
from te.platform import scope_cbuf
from te.platform import scope_cc
from te.platform import scope_ubuf
from te.platform.fusion_manager import fusion_manager
from tbe.common.platform import api_check_support
from tbe.common.rl_bank import bank_manager
from te.tik import Dprofile
from te.tik import Tik
from te.tik import scope_gm
from tbe import tvm
from tbe.tvm import create_schedule
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl.dynamic_rnn_block_lstm import block_lstm_core


# 'pylint: disable=huawei-too-many-arguments
def op_select_format(input_x, weight, bias, seq_length, init_h, init_c, wci, wcf,
                     wco, mask, y, output_h, output_c, i, j, f, o, tanhc,
                     cell_type="LSTM", direction="UNIDIRECTIONAL", cell_depth=1,
                     use_peephole=False, keep_prob=1.0, cell_clip=-1.0, num_proj=0,
                     time_major=True, activation="tanh", forget_bias=0.0, gate_order="ijfo", is_training=True,
                     kernel_name="dynamic_rnn"):
    """
    select format dynamically
    """
    x_format_list = "FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ"
    w_format_list = "FRACTAL_ZN_RNN,FRACTAL_ZN_RNN,FRACTAL_ZN_RNN,FRACTAL_ZN_RNN"
    b_format_list = "ND_RNN_BIAS,ND_RNN_BIAS,ND_RNN_BIAS,ND_RNN_BIAS"
    seq_format_list = "FRACTAL_NZ,FRACTAL_NZ,ND,ND"
    mask_format_list = "ND,ND,ND,ND"

    dtype_list = "float16,float16,float16,float16"
    b_dtype_list = "float16,float32,float16,float32"
    seq_dtype_list = "float16,float16,int32,int32"
    mask_dtype_list = "uint8,uint8,uint8,uint8"

    input0 = gen_param(classify="input0", name="x",
                       datatype=dtype_list,
                       format=x_format_list,
                       unknownshape_format=mask_format_list)
    input1 = gen_param(classify="input1", name="w",
                       datatype=dtype_list,
                       format=w_format_list,
                       unknownshape_format=mask_format_list)
    input2 = gen_param(classify="input2", name="b",
                       datatype=b_dtype_list,
                       format=b_format_list,
                       unknownshape_format=mask_format_list)
    input3 = gen_param(classify="input3", name="seq_length",
                       datatype=seq_dtype_list,
                       format=seq_format_list,
                       unknownshape_format=mask_format_list)
    input4 = gen_param(classify="input4", name="init_h",
                       datatype=dtype_list,
                       format=x_format_list,
                       unknownshape_format=mask_format_list)
    input5 = gen_param(classify="input5", name="init_c",
                       datatype=b_dtype_list,
                       format=x_format_list,
                       unknownshape_format=mask_format_list)
    input6 = gen_param(classify="input6", name="wci",
                       datatype=dtype_list,
                       format=x_format_list,
                       unknownshape_format=mask_format_list)
    input7 = gen_param(classify="input7", name="wcf",
                       datatype=dtype_list,
                       format=x_format_list,
                       unknownshape_format=mask_format_list)
    input8 = gen_param(classify="input8", name="wco",
                       datatype=dtype_list,
                       format=x_format_list,
                       unknownshape_format=mask_format_list)
    input9 = gen_param(classify="input9", name="mask",
                       datatype=mask_dtype_list,
                       format=mask_format_list,
                       unknownshape_format=mask_format_list)

    output0 = gen_param(classify="output0", name="y",
                        datatype=b_dtype_list,
                        format=x_format_list,
                        unknownshape_format=mask_format_list)
    output1 = gen_param(classify="output1", name="output_h",
                        datatype=dtype_list,
                        format=x_format_list,
                        unknownshape_format=mask_format_list)
    output2 = gen_param(classify="output2", name="output_c",
                        datatype=b_dtype_list,
                        format=x_format_list,
                        unknownshape_format=mask_format_list)
    output3 = gen_param(classify="output3", name="i",
                        datatype=b_dtype_list,
                        format=x_format_list,
                        unknownshape_format=mask_format_list)
    output4 = gen_param(classify="output4", name="j",
                        datatype=b_dtype_list,
                        format=x_format_list,
                        unknownshape_format=mask_format_list)
    output5 = gen_param(classify="output5", name="f",
                        datatype=b_dtype_list,
                        format=x_format_list,
                        unknownshape_format=mask_format_list)
    output6 = gen_param(classify="output6", name="o",
                        datatype=b_dtype_list,
                        format=x_format_list,
                        unknownshape_format=mask_format_list)
    output7 = gen_param(classify="output7", name="tanhc",
                        datatype=b_dtype_list,
                        format=x_format_list,
                        unknownshape_format=mask_format_list)

    param_list = [input0, input1, input2, input3, input4, input5, input6, input7, input8, input9,
                  output0, output1, output2, output3, output4, output5, output6, output7]

    param_dynamic_in_json =  get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# 'pylint: disable=invalid-name
def sigmoid(x):
    """
    sigmoid
    """

    s = 1 / (1 + np.exp(-x))
    return s


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,too-many-nested-blocks,invalid-name
def matrix_to_zz(matrix, shape, dtype):
    """
    ND(m, k) to zZ
    """

    h = shape[-2]
    w = shape[-1]
    tmp = np.zeros(np.prod(shape), dtype=dtype)
    idx = 0
    if h == 1:
        if len(shape) > 2:
            for batch in range(np.prod(shape[:-2])):
                for j in range(0, w):
                    tmp[idx] = matrix[batch][0][idx]
                    idx = idx + 1
        else:
            for j in range(0, w):
                tmp[idx] = matrix[0][idx]
                idx = idx + 1
    else:
        if len(shape) > 2:
            for batch in range(np.prod(shape[:-2])):
                for i in range(0, h // 16):
                    for j in range(0, w // 16):
                        for ii in range(0, 16):
                            for jj in range(0, 16):
                                tmp[idx] = matrix[batch][i * 16 + ii][j * 16 + jj]
                                idx = idx + 1
        else:
            for i in range(0, h // 16):
                for j in range(0, w // 16):
                    for ii in range(0, 16):
                        for jj in range(0, 16):
                            tmp[idx] = matrix[i * 16 + ii][j * 16 + jj]
                            idx = idx + 1
    return tmp


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name
def matrix_to_nz(matrix, shape, dtype):
    """
    ND(k, n) to nZ
    """

    h = shape[-2]
    w = shape[-1]
    tmp = np.zeros(np.prod(shape), dtype=dtype)
    idx = 0
    if w == 1:
        if len(shape) > 2:
            for batch in range(np.prod(shape[:-2])):
                for i in range(0, h):
                    tmp[idx] = matrix[batch][idx][0]
                    idx = idx + 1
        else:
            for i in range(0, h // 16):
                tmp[idx] = matrix[idx][0]
                idx = idx + 1
    else:
        if len(shape) > 2:
            for batch in range(0, np.prod(shape[:-2])):
                for i in range(0, h // 16):
                    for j in range(0, w // 16):
                        for jj in range(0, 16):
                            for ii in range(0, 16):
                                tmp[idx] = matrix[batch][i * 16 + ii][j * 16 +
                                                                      jj]
                                idx = idx + 1
        else:
            for i in range(0, h // 16):
                for j in range(0, w // 16):
                    for jj in range(0, 16):
                        for ii in range(0, 16):
                            tmp[idx] = matrix[i * 16 + ii][j * 16 + jj]
                            idx = idx + 1
    return tmp


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name
def matrix_to_zn(matrix, shape, dtype):
    """
    ND(m, n) to zN
    """

    h = shape[-2]
    w = shape[-1]
    tmp = np.zeros(np.prod(shape), dtype=dtype)
    idx = 0
    if len(shape) > 2:
        if h == 1:
            for batch in range(np.prod(shape[:-2])):
                for j in range(0, w):
                    tmp[idx] = matrix[batch][0][idx]
                    idx = idx + 1
        elif w == 1:
            for batch in range(np.prod(shape[:-2])):
                for i in range(0, h):
                    tmp[idx] = matrix[batch][idx][0]
                    idx = idx + 1
        else:
            for batch in range(np.prod(shape[:-2])):
                for j in range(0, w // 16):
                    for i in range(0, h // 16):
                        for ii in range(0, 16):
                            for jj in range(0, 16):
                                tmp[idx] = matrix[batch][i * 16 + ii][j * 16 + jj]
                                idx = idx + 1
    else:
        if h == 1:
            for j in range(0, w):
                tmp[idx] = matrix[0][idx]
                idx = idx + 1
        elif w == 1:
            for i in range(0, h):
                tmp[idx] = matrix[idx][0]
                idx = idx + 1
        else:
            for j in range(0, w // 16):
                for i in range(0, h // 16):
                    for ii in range(0, 16):
                        for jj in range(0, 16):
                            tmp[idx] = matrix[i * 16 + ii][j * 16 + jj]
                            idx = idx + 1
    return tmp


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,too-many-nested-blocks,invalid-name
def maxtrix_zn_reverse(matrix, shape, dtype):
    """
    maxtrix zn reverse
    """

    idx = 0
    j_outer, i_outer, i_inner, j_inner = shape[-4], shape[-3], shape[-2], shape[-1]
    h = i_outer * i_inner
    w = j_outer * j_inner

    if len(shape) == 5:
        batch_shape = shape[0]
        tmp = np.zeros((batch_shape, h, w), dtype=dtype)
        for batch in range(batch_shape):
            for j in range(0, j_outer):
                for i in range(0, i_outer):
                    for ii in range(0, i_inner):
                        for jj in range(0, j_inner):
                            tmp[batch][i * 16 + ii][j * 16 + jj] = matrix[idx]
                            idx = idx + 1
    elif len(shape) == 4:
        tmp = np.zeros((h, w), dtype=dtype)
        for j in range(0, j_outer):
            for i in range(0, i_outer):
                for ii in range(0, i_inner):
                    for jj in range(0, j_inner):
                        tmp[i * 16 + ii][j * 16 + jj] = matrix[idx]
                        idx = idx + 1

    return tmp


# 'pylint: disable=too-many-nested-blocks,invalid-name
def maxtrix_nz_reverse(matrix, shape, dtype):
    """
    maxtrix nz reverse
    """

    idx = 0
    i_outer, j_outer, j_inner, i_inner = shape[-4], shape[-3], shape[-2], shape[-1]
    h = i_outer * i_inner
    w = j_outer * j_inner

    if len(shape) == 5:
        batch_shape = shape[0]
        tmp = np.zeros((batch_shape, h, w), dtype=dtype)
        for batch in range(batch_shape):
            for i in range(0, i_outer):
                for j in range(0, j_outer):
                    for jj in range(0, j_inner):
                        for ii in range(0, i_inner):
                            tmp[batch][i * 16 + ii][j * 16 + jj] = matrix[idx]
                            idx = idx + 1
    elif len(shape) == 4:
        tmp = np.zeros((h, w), dtype=dtype)
        for i in range(0, i_outer):
            for j in range(0, j_outer):
                for jj in range(0, j_inner):
                    for ii in range(0, i_inner):
                        tmp[i * 16 + ii][j * 16 + jj] = matrix[idx]
                        idx = idx + 1

    return tmp


# 'pylint: disable=too-many-arguments,too-many-locals,unbalanced-tuple-unpacking
# 'pylint: disable=too-many-function-args,too-many-statements,unused-argument,invalid-name
def dynamic_rnn_np(input_data_list,
                   input_x,
                   weight,
                   bias,
                   seq_length,
                   init_h,
                   init_c,
                   wci,
                   wcf,
                   wco,
                   mask,
                   y,
                   output_h,
                   output_c,
                   i,
                   j,
                   f,
                   o,
                   tanhc,
                   cell_type="LSTM",
                   direction="UNIDIRECTIONAL",
                   cell_depth=1,
                   use_peephole=False,
                   keep_prob=1.0,
                   cell_clip=-1.0,
                   num_proj=0,
                   time_major=True,
                   activation="tanh",
                   forget_bias=0.0,
                   is_training=True,
                   kernel_name="dynamic_rnn"):
    """
    for RL Tune gen golden
    """

    shape_x_input = input_x.get("shape")
    shape_w_input = weight.get("shape")
    shape_output = output_h.get("shape")

    src_type = bias.get("dtype").lower()

    t_size = shape_x_input[0]
    m_size = shape_x_input[2]
    n_size = shape_w_input[1]

    hidden_size = n_size // 4
    batch = m_size * 16
    hidden = hidden_size * 16

    init_from_gm = False
    if init_h is not None:
        init_from_gm = True

    x_data = input_data_list[0]
    x_data = maxtrix_zn_reverse(x_data.flatten(), shape_x_input, np.float16)
    w_data = input_data_list[1]
    w_data = maxtrix_nz_reverse(w_data.flatten(), shape_w_input, np.float16)
    bias_num = input_data_list[2].flatten()

    if init_from_gm:
        c_data = input_data_list[3]
        h_data = input_data_list[4]
    else:
        h_data = np.zeros([batch, hidden]).astype("float16")
        c_data = np.zeros([batch, hidden]).astype(src_type)

    h_new = h_data
    c_new = c_data.astype(src_type)

    t = t_size + 1
    for var in range(t - 1):
        x_new = np.concatenate((x_data[var], h_new), axis=1)

        res = np.matmul(x_new, w_data).astype("float32")

        bias_num = bias_num.astype("float32")
        res = res + bias_num

        res_i, res_j, res_f, res_o = np.split(res, 4, axis=1)

        res_f = res_f + forget_bias
        res_i = sigmoid(res_i)
        res_j = np.tanh(res_j)
        res_f = sigmoid(res_f)
        res_o = sigmoid(res_o)

        c_tmp1 = c_new * res_f
        c_tmp2 = res_j * res_i
        c_new = c_tmp1 + c_tmp2

        c_tmph = np.tanh(c_new)
        h_new = c_tmph * res_o
        h_new = h_new.astype('float32')

        if var == 0:
            output_h = h_new
            output_c = c_new
            output_i = res_i
            output_j = res_j
            output_f = res_f
            output_o = res_o
            output_tanc = c_tmph

        else:
            output_h = np.concatenate((output_h, h_new), axis=0)
            output_c = np.concatenate((output_c, c_new), axis=0)
            output_i = np.concatenate((output_i, res_i), axis=0)
            output_j = np.concatenate((output_j, res_j), axis=0)
            output_f = np.concatenate((output_f, res_f), axis=0)
            output_o = np.concatenate((output_o, res_o), axis=0)
            output_tanc = np.concatenate((output_tanc, c_tmph), axis=0)

    output_h = output_h.reshape(t_size, batch, hidden).astype(src_type)
    output_c = output_c.reshape(t_size, batch, hidden).astype(src_type)
    output_i = output_i.reshape(t_size, batch, hidden).astype(src_type)
    output_j = output_j.reshape(t_size, batch, hidden).astype(src_type)
    output_f = output_f.reshape(t_size, batch, hidden).astype(src_type)
    output_o = output_o.reshape(t_size, batch, hidden).astype(src_type)
    output_tanc = output_tanc.reshape(t_size, batch, hidden).astype(src_type)
    outputy = output_h

    output_c = matrix_to_zn(output_c, output_c.shape, src_type).reshape(shape_output)
    output_h = matrix_to_zn(output_h, output_h.shape, src_type).reshape(shape_output)
    output_i = matrix_to_zn(output_i, output_i.shape, src_type).reshape(shape_output)
    output_j = matrix_to_zn(output_j, output_j.shape, src_type).reshape(shape_output)
    output_f = matrix_to_zn(output_f, output_f.shape, src_type).reshape(shape_output)
    output_o = matrix_to_zn(output_o, output_o.shape, src_type).reshape(shape_output)
    output_tanc = matrix_to_zn(output_tanc, output_tanc.shape, src_type).reshape(shape_output)
    outputy = output_h

    return [outputy, output_h, output_c, output_i, output_j, output_f, output_o, output_tanc]


def sigmoid_compute(input_x):
    """
    calculating sigmoid
    """
    data_input = input_x
    dtype = input_x.dtype
    exp_support = api_check_support(
        "te.lang.cce.vexp", "float32")
    mul_support = api_check_support(
        "te.lang.cce.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_manager_vector.raise_err_specific_reson("DynamicRNN",
                                                      "Input dtype only support float16 while input dtype is float32")

    const_num_neg_one = tvm.const(-1, dtype=dtype)
    const_num_one = tvm.const(1, dtype=dtype)
    tmp_negative = vmuls(data_input, const_num_neg_one)
    if dtype == "float32" and not exp_support:
        tmp_negative = cast_to(tmp_negative, "float16")
    tmp_exp = vexp(tmp_negative)
    if dtype == "float32" and not exp_support:
        tmp_exp = cast_to(tmp_exp, "float32")
    tmp_sum = vadds(tmp_exp, const_num_one)
    if dtype == "float32":
        inp_shape = tmp_sum.shape
        tensor_one = broadcast(tvm.const(1, dtype), inp_shape)
        res = vdiv(tensor_one, tmp_sum)
    else:
        res = vrec(tmp_sum)

    return res


def tanh_compute(input_x):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)
    """
    input_dtype = input_x.dtype
    # positive min float32 value
    min_fp_data = 2 ** (-126)
    const_dtype = input_dtype
    # positive min float16 value
    if input_dtype == "float16":
        min_fp_data = 2 ** (-14)

    has_improve_precision = False

    if input_dtype == "float16" and \
            api_check_support("vexp", "float32"):
        input_x = cast_to(input_x, "float32")
        has_improve_precision = True
        const_dtype = "float32"

    input_abs = vabs(input_x)
    power_val = vmuls(input_abs, tvm.const(-2, const_dtype))
    exp_val = vexp(power_val)

    up_val_tmp = vmul(exp_val, input_x)
    up_val = vsub(input_x, up_val_tmp)

    input_x_tmp = vadds(input_abs, min_fp_data)
    down_val_tmp = vadds(exp_val, tvm.const(1, const_dtype))
    down_val = vmul(down_val_tmp, input_x_tmp)

    res = vdiv(up_val, down_val)

    if has_improve_precision:
        res = cast_to(res, "float16")

    return res


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


# 'pylint: disable=too-many-return-values
def get_lstm_tiling():
    """
    get no RL default lstm element wise tiling
    :return:
    """
    return (1, 1, 12, 1, 1, 12)


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name,invalid-name
def check_prama_dtype(input_x, weight, bias, init_h, init_c, y, output_h,
                      output_c, i, j, f, o, tanhc):
    """
    check parameters dtype
    :return:
    """
    x_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(x_dtype, ["float16"], param_name="x")

    w_dtype = weight.get("dtype").lower()
    para_check.check_dtype(w_dtype, ["float16"], param_name="w")

    output_h_dtype = output_h.get("dtype").lower()
    para_check.check_dtype(output_h_dtype, ["float16"], param_name="output_h")

    if init_h is not None:
        init_h_dtype = init_h.get("dtype").lower()
        para_check.check_dtype(init_h_dtype, ["float16"], param_name="init_h")

    bias_dtype = bias.get("dtype").lower()
    para_check.check_dtype(bias_dtype, ["float16", "float32"], param_name="bias")

    # check optional input
    if init_c is not None:
        if init_c.get("dtype").lower() != bias_dtype:
            error_manager_vector.raise_err_specific_reson("DynamicRNN", "init_c dtype is not the same as bias dtype !")

    # check output
    if y["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "y dtype is not the same as bias dtype !")
    if output_c["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "output_c dtype is not the same as bias dtype !")

    # check additional output
    if i["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "i dtype is not the same as bias dtype !")
    if j["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "j dtype is not the same as bias dtype !")
    if f["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "f dtype is not the same as bias dtype !")
    if o["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "o dtype is not the same as bias dtype !")
    if tanhc["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "tanhc dtype is not the same as bias dtype !")


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name,invalid-name
def check_prama_shape(input_x, weight, bias, seq_length, init_h, init_c,
                      wci, wcf, wco, mask, y, output_h, output_c, i, j, f, o,
                      tanhc):
    """
    check parameters
    """
    # check batch dim
    if input_x["shape"][2] != output_h["shape"][2]:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "x, output_h shape is wrong, please check!")

    if weight["shape"][0] != input_x["shape"][1] + output_h["shape"][1]:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "x, w, output_h shape is wrong, please check!")

    # hidden_size dim check
    if weight["shape"][1] != 4 * output_h["shape"][1]:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "w, output_h shape is wrong, please check!")
    if (bias["shape"][0] + 15) // 16 != weight["shape"][1]:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "w, b shape is wrong, please check!")

    # seq_length
    if seq_length is not None:
        if seq_length.get("dtype").lower() == "int32":
            if (seq_length["shape"][0] + 15) // 16 != output_h["shape"][2]:
                error_manager_vector.raise_err_check_params_rules("DynamicRNN",
                                                          "(seq_length.shape[0] + 15)/16 != output_h.shape[2]",
                                                          "seq_length.shape[0]", output_h["shape"][2])
        else:
            if seq_length["shape"] != output_h["shape"]:
                error_manager_vector.raise_err_check_params_rules("DynamicRNN: seq_length.shape != output_h.shape")

    # check init
    if (init_h is None and init_c is not None) or (
            init_h is not None and init_c is None):
        error_manager_vector.raise_err_specific_reson("DynamicRNN",
                                                      "init_h, init_c should appear together, please check!")

    # check init state should support one time
    if init_h is not None:
        if init_h["shape"][0] != 1:
            error_manager_vector.raise_err_specific_reson("DynamicRNN", "init_h only support 1 time, please check!")

        if init_h["shape"][1] != output_h["shape"][1] or init_h["shape"][2] != \
                output_h["shape"][2]:
            error_manager_vector.raise_err_specific_reson("DynamicRNN",
                                                          "init_h, output_h shape is different, please check!")

        if not operator.eq(init_h["shape"], init_c["shape"]):
            error_manager_vector.raise_err_specific_reson("DynamicRNN",
                                                          "init_c, init_h shape is different, please check!")

    # check output
    if not operator.eq(output_h["shape"], output_c["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNN",
                                                      "output_c, output_h shape is different, please check!")

    if not operator.eq(output_h["shape"], i["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNN",
                                                      "i, output_h shape is different, please check!")

    if not operator.eq(output_h["shape"], j["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNN",
                                                      "i, output_h shape is different, please check!")

    if not operator.eq(output_h["shape"], f["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNN",
                                                      "f, output_h shape is different, please check!")

    if not operator.eq(output_h["shape"], o["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNN",
                                                      "o, output_h shape is different, please check!")

    if not operator.eq(output_h["shape"], tanhc["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNN",
                                                      "tanhc, output_h shape is different, please check!")

    if not operator.eq(output_h["shape"], y["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNN",
                                                      "y, output_h shape is different, please check!")

    # check unsupport pramas
    if wci is not None:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "wci only support None, please check!")

    if wcf is not None:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "wcf only support None, please check!")

    if wco is not None:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "wco only support None, please check!")

    if mask is not None:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "mask only support None, please check!")


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals
def check_attr(cell_type, direction, cell_depth, use_peephole, keep_prob,
               cell_clip, num_proj, time_major, activation, gate_order):
    """
    check parameters
    """
    if cell_type not in ["LSTM", "GRU", "BLOCKLSTM", "RNN_RELU", "RNN_TANH"]:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "attr cell_type is not support, please check!")

    if direction not in ["UNIDIRECTIONAL", "REDIRECTIONAL", "BIDIRECTIONAL"]:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "attr direction is not support, please check!")

    if cell_depth != 1:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "attr cell_depth is not support, please check!")

    if use_peephole is not False:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "attr use_peephole is not support, please check!")

    if keep_prob != 1.0:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "attr keep_prob is not support, please check!")

    if num_proj != 0:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "attr num_proj is not support, please check!")

    if time_major is not True:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "attr time_major only support True, please check!")

    if activation not in ["tanh"]:
        error_manager_vector.raise_err_specific_reson("DynamicRNN", "attr activation only support tanh, please check!")

    if gate_order not in ["ijfo", "ifjo"]:
        error_manager_vector.raise_err_check_params_rules("DynamicRNN",
                                                          "gate_order in ['ijfo', 'ifjo']",
                                                          "gate_order", str(gate_order))


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-function-args,too-many-statements
# 'pylint: disable=unused-argument
def dynamic_rnn(input_x, weight, bias, seq_length, init_h, init_c, wci, wcf,
                wco, mask, y, output_h, output_c, i, j, f, o, tanhc,
                cell_type="LSTM", direction="UNIDIRECTIONAL", cell_depth=1,
                use_peephole=False, keep_prob=1.0, cell_clip=-1.0, num_proj=0,
                time_major=True, activation="tanh", forget_bias=0.0, gate_order="ijfo", is_training=True,
                kernel_name="dynamic_rnn"):
    """
    dynamic_rnn
    """

    check_prama_dtype(input_x, weight, bias, init_h, init_c, y, output_h,
                      output_c, i, j, f, o, tanhc)

    check_prama_shape(input_x, weight, bias, seq_length, init_h, init_c, wci,
                      wcf, wco, mask, y, output_h, output_c, i, j, f, o, tanhc)

    check_attr(cell_type, direction, cell_depth, use_peephole, keep_prob,
               cell_clip, num_proj, time_major, activation, gate_order)

    input_dtype = input_x.get("dtype").lower()
    bias_dtype = bias.get("dtype").lower()

    shape_x_input = input_x.get("shape")
    shape_w_input = weight.get("shape")

    k_size = shape_w_input[0]
    n_size = shape_w_input[1]
    t_size = shape_x_input[0]
    m_size = shape_x_input[2]

    block_size = 4
    hidden_size = n_size // 4
    in_x = k_size - hidden_size

    shape_bias = (1, block_size, hidden_size, 1, 1, 16)
    shape_w = (1, k_size, block_size, hidden_size, 16, 16)
    shape_x = (t_size, in_x, m_size, 16, 16)
    shape_hc_init = (1, hidden_size, m_size, 16, 16)
    shape_hc = (t_size, hidden_size, m_size, 16, 16)

    # due to FE/GE not support, now set default value
    is_gate_output = True

    tik_instance = Tik(Dprofile())

    sync = tik_instance.Tensor(shape=(128,), dtype="int64", scope=scope_gm, name='sync',
                               is_workspace=True, is_atomic_add=True)

    input_x = tik_instance.Tensor(shape=shape_x, dtype=input_dtype,
                                  scope=scope_gm, name='input_x')
    weight = tik_instance.Tensor(shape=shape_w, dtype=input_dtype,
                                 scope=scope_gm, name='weight')
    bias = tik_instance.Tensor(shape=shape_bias, scope=scope_gm,
                               dtype=bias_dtype, name='bias')

    is_using_seq_mask = False
    is_valid_mask = False
    if seq_length is not None:
        is_using_seq_mask = True
        if seq_length.get("dtype").lower() == "int32":
            seq_mask_gm = tik_instance.Tensor(shape=seq_length.get("shape"), scope=scope_gm,
                                              dtype="int32", name='seq_mask_gm')
        else:
            is_valid_mask = True
            seq_mask_gm = tik_instance.Tensor(shape=seq_length.get("shape"), scope=scope_gm,
                                          dtype="float16", name='seq_mask_gm')
    else:
        seq_mask_gm = None

    is_global_init = False
    if init_h is not None:
        is_global_init = True
    if is_global_init:
        s_init_h_gm = tik_instance.Tensor(shape=shape_hc_init,
                                          dtype=input_dtype,
                                          scope=scope_gm,
                                          name='s_init_h_gm')
        s_init_c_gm = tik_instance.Tensor(shape=shape_hc_init,
                                          dtype=bias_dtype, scope=scope_gm,
                                          name='s_init_c_gm')

    update_h_gm = tik_instance.Tensor(shape=shape_hc, dtype=input_dtype,
                                      scope=scope_gm, name='update_h_gm')
    update_c_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                      scope=scope_gm, name='update_c_gm')
    update_h_gm_as_y = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                           scope=scope_gm,
                                           name='update_h_gm_as_y')

    if is_gate_output:
        f_t_sigmoid_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                             scope=scope_gm,
                                             name='f_t_sigmoid_gm')
        i_t_sigmoid_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                             scope=scope_gm,
                                             name='i_t_sigmoid_gm')
        o_t_sigmoid_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                             scope=scope_gm,
                                             name='o_t_sigmoid_gm')
        j_t_tanh_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                          scope=scope_gm,
                                          name='j_t_tanh_gm')
        c_t_tanh_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                          scope=scope_gm,
                                          name='c_t_tanh_gm')

    build_input_list = [input_x, weight, bias]
    if is_using_seq_mask:
        build_input_list.append(seq_mask_gm)
    if is_global_init:
        build_input_list.append(s_init_h_gm)
        build_input_list.append(s_init_c_gm)

    if is_gate_output:
        build_output_list = [update_h_gm_as_y, update_h_gm, update_c_gm,
                             i_t_sigmoid_gm, j_t_tanh_gm, f_t_sigmoid_gm,
                             o_t_sigmoid_gm, c_t_tanh_gm]
    else:
        build_output_list = [update_h_gm_as_y, update_h_gm, update_c_gm]

    # for RL tune getting tik input&output tensor
    fusion_manager.set_tik_tensor(build_input_list, build_output_list)
    bank_manager.init_bank_hit_info(kernel_name)

    cut_t = 1
    last = 1
    # RL default tiling
    cut_m = m_size
    loop_t = t_size // cut_t
    loop_m = m_size // cut_m

    with tik_instance.for_range(0, loop_t) as loop_i:

        if direction == "REDIRECTIONAL":
            valid_loop_i = loop_t - 1 - loop_i
        else:
            valid_loop_i = loop_i
        with tik_instance.for_range(0, loop_m) as loop_j:

            input_x_var = input_x[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t,
                                  :,
                                  loop_j * cut_m: loop_j * cut_m + cut_m,
                                  :, :]

            if is_global_init:
                s_init_c_gm_var = s_init_c_gm[:, :,
                                              loop_j * cut_m: loop_j * cut_m + cut_m,
                                              :, :]
                s_init_h_gm_var = s_init_h_gm[:, :,
                                              loop_j * cut_m: loop_j * cut_m + cut_m,
                                              :, :]
            else:
                s_init_c_gm_var = None
                s_init_h_gm_var = None

            if direction == "REDIRECTIONAL":
                state_h_last = update_h_gm[valid_loop_i * cut_t + last: valid_loop_i * cut_t + cut_t + last:,
                                           :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
                state_c_last = update_c_gm[valid_loop_i * cut_t + last: valid_loop_i * cut_t + cut_t + last:,
                                           :,
                                           loop_j * cut_m: loop_j * cut_m + cut_m,
                                           :, :]
            else:
                state_h_last = update_h_gm[valid_loop_i * cut_t - last: valid_loop_i * cut_t + cut_t - last:,
                                       :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
                state_c_last = update_c_gm[valid_loop_i * cut_t - last: valid_loop_i * cut_t + cut_t - last:,
                                       :,
                                       loop_j * cut_m: loop_j * cut_m + cut_m,
                                       :, :]
            update_h_gm_var = update_h_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t,
                                          :,
                                          loop_j * cut_m: loop_j * cut_m + cut_m,
                                          :, :]
            update_c_gm_var = update_c_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t,
                                          :,
                                          loop_j * cut_m: loop_j * cut_m + cut_m,
                                          :, :]
            update_h_gm_as_y_var = update_h_gm_as_y[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t:,
                                                    :,
                                                    loop_j * cut_m: loop_j * cut_m + cut_m,
                                                    :, :]

            if is_gate_output:
                f_t_sigmoid_gm_var = f_t_sigmoid_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t,
                                                    :,
                                                    loop_j * cut_m: loop_j * cut_m + cut_m,
                                                    :, :]
                i_t_sigmoid_gm_var = i_t_sigmoid_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t,
                                                    :,
                                                    loop_j * cut_m: loop_j * cut_m + cut_m,
                                                    :, :]
                o_t_sigmoid_gm_var = o_t_sigmoid_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t,
                                                    :,
                                                    loop_j * cut_m: loop_j * cut_m + cut_m,
                                                    :, :]
                j_t_tanh_gm_var = j_t_tanh_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t,
                                              :,
                                              loop_j * cut_m: loop_j * cut_m + cut_m,
                                              :, :]
                c_t_tanh_gm_var = c_t_tanh_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t,
                                              :,
                                              loop_j * cut_m: loop_j * cut_m + cut_m,
                                              :, :]
            else:
                f_t_sigmoid_gm_var = None
                i_t_sigmoid_gm_var = None
                o_t_sigmoid_gm_var = None
                j_t_tanh_gm_var = None
                c_t_tanh_gm_var = None

            if is_valid_mask:
                seq_mask_gm_var = seq_mask_gm[valid_loop_i * cut_t: valid_loop_i * cut_t + cut_t,
                                              :,
                                              loop_j * cut_m: loop_j * cut_m + cut_m,
                                              :, :]
            else:
                seq_mask_gm_var = None

            input_list = [input_x_var, weight, bias, s_init_h_gm_var,
                          s_init_c_gm_var, state_h_last,
                          state_c_last, sync, seq_mask_gm_var]

            if is_gate_output:
                output_list = [update_h_gm_var, update_c_gm_var,
                               update_h_gm_as_y_var, i_t_sigmoid_gm_var,
                               j_t_tanh_gm_var, f_t_sigmoid_gm_var,
                               o_t_sigmoid_gm_var, c_t_tanh_gm_var]
            else:
                output_list = [update_h_gm_var, update_c_gm_var,
                               update_h_gm_as_y_var]

            with tik_instance.if_scope(loop_i == 0):
                is_first_round = True
                tik_instance.call_module(
                    dynamic_rnn_tik,
                    input_list,
                    output_list,
                    [is_gate_output, is_first_round, is_global_init,
                     forget_bias, gate_order, cell_clip, cell_type])

            with tik_instance.if_scope(loop_i > 0):
                is_first_round = False
                tik_instance.call_module(
                    dynamic_rnn_tik,
                    input_list,
                    output_list,
                    [is_gate_output, is_first_round, is_global_init,
                     forget_bias, gate_order, cell_clip, cell_type])

    config_map = {
        "dump_cce_code": False,
    }

    tik_instance.BuildCCE(kernel_name,
                          build_input_list,
                          build_output_list,
                          config=config_map)


def dynamic_rnn_tik(input_list, custom_list):
    """
    inside part of tik loop
    :return:
    """
    input_x = input_list[0]
    weight = input_list[1]
    bias = input_list[2]
    s_init_h_gm = input_list[3]
    s_init_c_gm = input_list[4]
    s_state_h_gm_last = input_list[5]
    s_state_c_gm_last = input_list[6]
    sync0 = input_list[7]
    seq_mask_gm = input_list[8]

    is_gate_output = custom_list[0]
    is_first_round = custom_list[1]
    is_global_init = custom_list[2]
    forget_bias = custom_list[3]
    gate_order = custom_list[4]
    cell_clip = custom_list[5]
    cell_type = custom_list[6]

    if cell_type == "BLOCKLSTM":
        return block_lstm_core(input_x, weight, bias, s_init_h_gm, s_init_c_gm,
                                s_state_h_gm_last, s_state_c_gm_last, sync0, seq_mask_gm,
                                is_gate_output, is_first_round, is_global_init,
                                forget_bias, gate_order, cell_clip, cell_type)
    else:
        return dynamic_rnn_core(input_x, weight, bias, s_init_h_gm, s_init_c_gm,
                                s_state_h_gm_last, s_state_c_gm_last, sync0, seq_mask_gm,
                                is_gate_output, is_first_round, is_global_init,
                                forget_bias, gate_order, cell_clip, cell_type)


# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-statements,unnecessary-lambda
def dynamic_rnn_core_sub0(shape_h, shape_i, input_dtype, bias_dtype,
                          s_init_h_gm, s_init_c_gm, s_state_h_gm_last, s_state_c_gm_last, seq_mask_gm,
                          is_first_round, is_global_init):
    s_state_h_ub_for_element = None
    if is_first_round:
        if is_global_init:
            s_state_h_ub = tvm.compute(shape_h,
                                       lambda _, a_i, b_i, c_i, d_i: s_init_h_gm[
                                           0, a_i, b_i, c_i, d_i], name="s_init_h")
            s_state_c_ub = tvm.compute(shape_i,
                                       lambda _, a_i, b_i, c_i, d_i: s_init_c_gm[
                                           0, a_i, b_i, c_i, d_i], name="s_init_c")
            if seq_mask_gm is not None:
                s_state_h_ub_for_element = tvm.compute(shape_h,
                                                       lambda _, a_i, b_i, c_i, d_i: s_init_h_gm[
                                           0, a_i, b_i, c_i, d_i], name="s_state_h_ub_for_element")
        else:
            s_state_h_ub = \
                tvm.compute(shape_h,
                            lambda *indices: tvm.const(0.0, dtype=input_dtype),
                            name='s_state_h_ub',
                            tag="broadcast")

            s_state_c_ub = \
                tvm.compute(shape_i,
                            lambda *indices: tvm.const(0.0, dtype=bias_dtype),
                            name='s_state_c_ub',
                            tag="broadcast")
            if seq_mask_gm is not None:
                s_state_h_ub_for_element = \
                            tvm.compute(shape_h,
                            lambda *indices: tvm.const(0.0, dtype=input_dtype),
                            name='s_state_h_ub_for_element',
                            tag="broadcast")
    else:
        s_state_h_ub = tvm.compute(shape_h,
                                   lambda _, a_i, b_i, c_i, d_i: s_state_h_gm_last[
                                       0, a_i, b_i, c_i, d_i], name="s_state_h_ub")
        s_state_c_ub = tvm.compute(shape_i,
                                   lambda _, a_i, b_i, c_i, d_i: s_state_c_gm_last[
                                       0, a_i, b_i, c_i, d_i], name="s_state_c_ub")
        if seq_mask_gm is not None:
            s_state_h_ub_for_element = tvm.compute(shape_h,
                                       lambda _, a_i, b_i, c_i, d_i: s_state_h_gm_last[
                                       0, a_i, b_i, c_i, d_i], name="s_state_h_ub_for_element")

    return s_state_h_ub, s_state_c_ub, s_state_h_ub_for_element


# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-statements,unnecessary-lambda
def dynamic_rnn_core(input_x, weight, bias, s_init_h_gm, s_init_c_gm,
                     s_state_h_gm_last, s_state_c_gm_last, sync0, seq_mask_gm,
                     is_gate_output, is_first_round, is_global_init,
                     forget_bias, gate_order, cell_clip, cell_type):
    """
    implement of dynamic rnn
    :return:
    """

    shape_x_input = input_x.shape
    shape_w_input = weight.shape

    m_size = shape_x_input[2].value
    k_size = shape_w_input[1].value
    n_size = shape_w_input[2].value * shape_w_input[3].value

    block_size = 4
    t_size = 1
    hidden_size = n_size // block_size
    in_x = k_size - hidden_size

    shape_a_z_bigz = (t_size, m_size, k_size, 16, 16)
    shape_b = (t_size, k_size, block_size, hidden_size, 16, 16)
    shape_c = (t_size, block_size, hidden_size, m_size, 16, 16)
    shape_bias = (t_size, block_size, hidden_size, 1, 1, 16)
    shape_h = (t_size, k_size - in_x, m_size, 16, 16)
    shape_i = (t_size, hidden_size, m_size, 16, 16)

    k0_size = 16
    input_dtype = input_x.dtype
    bias_dtype = bias.dtype

    fp16_input_output = False
    if bias_dtype == 'float16':
        fp16_input_output = True

    s_state_h_ub, s_state_c_ub, s_state_h_ub_for_element = dynamic_rnn_core_sub0(shape_h, shape_i, input_dtype,
                                                                                 bias_dtype, s_init_h_gm, s_init_c_gm,
                                                                                 s_state_h_gm_last, s_state_c_gm_last,
                                                                                 seq_mask_gm, is_first_round,
                                                                                 is_global_init)

    # input and s_start_h is Nz, need trans to zZ
    # so change axis 1 and 2
    a_ub = tvm.compute(shape_a_z_bigz,
                       lambda *indice:
                       tvm.select(indice[2] < in_x,
                                  input_x[indice[0],
                                          indice[2],
                                          indice[1],
                                          indice[3],
                                          indice[4]],
                                  s_state_h_ub[0,
                                               indice[2] - in_x,
                                               indice[1],
                                               indice[3],
                                               indice[4]]
                                  ),
                       name="a_ub", tag="concat")

    a_l1 = tvm.compute(shape_a_z_bigz,
                       lambda *indices: a_ub(*indices),
                       name='a_l1',
                       tag="out_to_l1")
    b_l1 = tvm.compute(shape_b,
                       lambda *indices: weight(*indices),
                       name='b_l1',
                       tag="out_to_l1")

    a_l0a = tvm.compute(shape_a_z_bigz, lambda *indices: a_l1(*indices),
                        name="a_l0a", tag="l1_to_l0")
    b_l0b = tvm.compute(shape_b,
                        lambda *indices: b_l1(*indices),
                        name="b_l0b",
                        tag="l1_to_l0")

    k1 = tvm.reduce_axis((0, k_size), name='k1')
    k0 = tvm.reduce_axis((0, k0_size), name='k0')

    if tbe_platform.api_check_support("tik.vec_conv", "f322f16"):
        data_type = 'float32'
    else:
        data_type = 'float16'
    c_l0c = tvm.compute(shape_c,
                        lambda t, nb_0, nb_1, mb, mp, np:
                        tvm.sum((a_l0a[t, mb, k1, mp, k0] * \
                                 b_l0b[t, k1, nb_0, nb_1, np, k0]) \
                                .astype(data_type),
                                axis=[k1, k0]),
                        name='c_l0c',
                        tag="matmul")

    c_ub = tvm.compute(shape_c, lambda *indices: c_l0c(*indices), name="c_ub")

    bias_ub = tvm.compute(shape_bias,
                          lambda *indices: bias(*indices),
                          name='bias_ub')

    bias_ub_mid = bias_ub
    if fp16_input_output:
        bias_ub_fp32 = \
            tvm.compute(shape_bias,
                        lambda *indices: bias_ub(*indices).astype(data_type),
                        name="bias_ub_fp32_drnn_cast",
                        tag="elewise_single_cast")
        bias_ub_mid = bias_ub_fp32

    bias_bc_ub = broadcast(bias_ub_mid, shape_c)
    c_ub_bias = vadd(c_ub, bias_bc_ub)

    # split matmul res
    if gate_order == "ijfo":
        i_t_index = 0
        j_t_index = 1
        f_t_index = 2
        o_t_index = 3
    else:
        i_t_index = 0
        j_t_index = 2
        f_t_index = 1
        o_t_index = 3

    i_t = \
        tvm.compute(shape_i,
                    lambda t, a_i, b_i, c_i, d_i: c_ub_bias(t, i_t_index, a_i, b_i, c_i, d_i),
                    name="i_t",
                    tag="split_com")
    j_t = \
        tvm.compute(shape_i,
                    lambda t, a_i, b_i, c_i, d_i: c_ub_bias(t, j_t_index, a_i, b_i, c_i, d_i),
                    name="j_t",
                    tag="split_com")
    f_t = \
        tvm.compute(shape_i,
                    lambda t, a_i, b_i, c_i, d_i: c_ub_bias(t, f_t_index, a_i, b_i, c_i, d_i),
                    name="f_t",
                    tag="split_com")
    o_t = \
        tvm.compute(shape_i,
                    lambda t, a_i, b_i, c_i, d_i: c_ub_bias(t, o_t_index, a_i, b_i, c_i, d_i),
                    name="o_t",
                    tag="split_com")

    f_t_bias = vadds(f_t, tvm.const(forget_bias, dtype=bias_dtype))
    f_t_sigmoid = sigmoid_compute(f_t_bias)
    i_t_sigmoid = sigmoid_compute(i_t)
    o_t_sigmoid = sigmoid_compute(o_t)
    j_t_tanh = tanh_compute(j_t)

    f_t_sigmoid_ub = f_t_sigmoid
    i_t_sigmoid_ub = i_t_sigmoid
    o_t_sigmoid_ub = o_t_sigmoid
    j_t_tanh_ub = j_t_tanh

    if is_gate_output:
        f_t_sigmoid_mid = f_t_sigmoid
        i_t_sigmoid_mid = i_t_sigmoid
        o_t_sigmoid_mid = o_t_sigmoid
        j_t_tanh_mid = j_t_tanh

        if fp16_input_output:
            f_t_sigmoid_fp16 = \
                tvm.compute(shape_i,
                            lambda *indices: f_t_sigmoid(*indices).astype('float16'),
                            name="f_t_sigmoid_fp16_drnn_cast",
                            tag="elewise_single_cast")
            i_t_sigmoid_fp16 = \
                tvm.compute(shape_i,
                            lambda *indices: i_t_sigmoid(*indices).astype('float16'),
                            name="i_t_sigmoid_fp16_drnn_cast",
                            tag="elewise_single_cast")
            o_t_sigmoid_fp16 = \
                tvm.compute(shape_i,
                            lambda *indices: o_t_sigmoid(*indices).astype('float16'),
                            name="o_t_sigmoid_fp16_drnn_cast",
                            tag="elewise_single_cast")
            j_t_tanh_fp16 = \
                tvm.compute(shape_i,
                            lambda *indices: j_t_tanh(*indices).astype('float16'),
                            name="j_t_tanh_fp16_drnn_cast",
                            tag="elewise_single_cast")
            f_t_sigmoid_mid = f_t_sigmoid_fp16
            i_t_sigmoid_mid = i_t_sigmoid_fp16
            o_t_sigmoid_mid = o_t_sigmoid_fp16
            j_t_tanh_mid = j_t_tanh_fp16

        f_t_sigmoid_gm = tvm.compute(shape_i,
                                     lambda *indices: f_t_sigmoid_mid(*indices),
                                     name="f_t_sigmoid_gm",
                                     tag="ub_to_out")
        i_t_sigmoid_gm = tvm.compute(shape_i,
                                     lambda *indices: i_t_sigmoid_mid(*indices),
                                     name="i_t_sigmoid_gm",
                                     tag="ub_to_out")
        o_t_sigmoid_gm = tvm.compute(shape_i,
                                     lambda *indices: o_t_sigmoid_mid(*indices),
                                     name="o_t_sigmoid_gm",
                                     tag="ub_to_out")
        j_t_tanh_gm = tvm.compute(shape_i,
                                  lambda *indices: j_t_tanh_mid(*indices),
                                  name="j_t_tanh_gm",
                                  tag="ub_to_out")

        f_t_sigmoid_back = tvm.compute(shape_i,
                                       lambda *indices: f_t_sigmoid_gm(*indices),
                                       name="f_t_sigmoid_back",
                                       tag="out_to_ub")
        i_t_sigmoid_back = tvm.compute(shape_i,
                                       lambda *indices: i_t_sigmoid_gm(*indices),
                                       name="i_t_sigmoid_back",
                                       tag="out_to_ub")
        o_t_sigmoid_back = tvm.compute(shape_i,
                                       lambda *indices: o_t_sigmoid_gm(*indices),
                                       name="o_t_sigmoid_back",
                                       tag="out_to_ub")
        j_t_tanh_back = tvm.compute(shape_i,
                                    lambda *indices: j_t_tanh_gm(*indices),
                                    name="j_t_tanh_back",
                                    tag="out_to_ub")

        if fp16_input_output:
            f_t_sigmoid_back_fp32 = tvm.compute(shape_i,
                                                lambda *indices: f_t_sigmoid_back(*indices).astype(data_type),
                                                name="f_t_sigmoid_back_fp32_drnn_cast",
                                                tag="elewise_single_cast")
            i_t_sigmoid_back_fp32 = tvm.compute(shape_i,
                                                lambda *indices: i_t_sigmoid_back(*indices).astype(data_type),
                                                name="i_t_sigmoid_back_fp32_drnn_cast",
                                                tag="elewise_single_cast")
            o_t_sigmoid_back_fp32 = tvm.compute(
                shape_i,
                lambda *indices: o_t_sigmoid_back(*indices).astype(data_type),
                name="o_t_sigmoid_back_fp32_drnn_cast",
                tag="elewise_single_cast")
            j_t_tanh_back_fp32 = tvm.compute(
                shape_i,
                lambda *indices: j_t_tanh_back(*indices).astype(data_type),
                name="j_t_tanh_back_fp32_drnn_cast",
                tag="elewise_single_cast")

        f_t_sigmoid_ub = f_t_sigmoid_back
        i_t_sigmoid_ub = i_t_sigmoid_back
        o_t_sigmoid_ub = o_t_sigmoid_back
        j_t_tanh_ub = j_t_tanh_back

        if fp16_input_output:
            f_t_sigmoid_ub = f_t_sigmoid_back_fp32
            i_t_sigmoid_ub = i_t_sigmoid_back_fp32
            o_t_sigmoid_ub = o_t_sigmoid_back_fp32
            j_t_tanh_ub = j_t_tanh_back_fp32

    # auto cast support both fp16 fp32
    c_t_tmp1 = vmul(s_state_c_ub, f_t_sigmoid_ub)
    c_t_tmp2 = vmul(j_t_tanh_ub, i_t_sigmoid_ub)
    update_c = vadd(c_t_tmp1, c_t_tmp2)

    if cell_clip > 0:
        dtype = update_c.dtype
        clip_const = tvm.const(cell_clip, dtype=dtype)
        if cell_type == "BLOCKLSTM":
            update_c = vmaxs(update_c, -(clip_const))
        update_c = vmins(update_c, clip_const)

    if seq_mask_gm is not None:
        seq_mask_ub = tvm.compute(shape_h, lambda _, a_i, b_i, c_i, d_i: seq_mask_gm[0, a_i, b_i, c_i, d_i],
                                    name="seq_mask_ub")

        update_c_diff = vsub(update_c, s_state_c_ub)
        update_c_tmp = vmul(update_c_diff, seq_mask_ub)
        update_c = vadd(update_c_tmp, s_state_c_ub)

        if cell_type == "BLOCKLSTM":
            update_c = vmul(update_c, seq_mask_ub)

    # c_gm fp32 case need flag
    if bias_dtype == 'float16':
        update_c_fp16 = tvm.compute(shape_i,
                                    lambda *indices: update_c(*indices).astype('float16'),
                                    name="update_c_fp16_drnn_cast",
                                    tag="elewise_single_cast")
        update_c_gm = tvm.compute(shape_i,
                                  lambda *indices: update_c_fp16(*indices),
                                  name="update_c_gm",
                                  tag="ub_to_out")
    else:
        update_c_gm = tvm.compute(shape_i,
                                  lambda *indices: update_c(*indices),
                                  name="update_c_gm",
                                  tag="ub_to_out")
    if bias_dtype == 'float16':
        update_c_fp16_back = tvm.compute(shape_i,
                                         lambda *indices: update_c_gm(*indices),
                                         name="update_c_fp16_back",
                                         tag="out_to_ub")
        update_c_fp16_back_fp32 = tvm.compute(shape_i,
                                              lambda *indices: update_c_fp16_back(*indices).astype(data_type),
                                              name="update_c_fp16_back_fp32_drnn_cast",
                                              tag="elewise_single_cast")
        c_t_tanh = tanh_compute(update_c_fp16_back_fp32)
    else:
        update_c_fp32_back = tvm.compute(shape_i,
                                         lambda *indices: update_c_gm(*indices),
                                         name="update_c_fp32_back",
                                         tag="out_to_ub")
        c_t_tanh = tanh_compute(update_c_fp32_back)

    c_t_tanh_ub = c_t_tanh

    if is_gate_output:
        c_t_tanh_mid = c_t_tanh

        if fp16_input_output:
            c_t_tanh_fp16 = tvm.compute(shape_i,
                                        lambda *indices: c_t_tanh(*indices).astype('float16'),
                                        name="c_t_tanh_fp16_drnn_cast",
                                        tag="elewise_single_cast")
            c_t_tanh_mid = c_t_tanh_fp16
        c_t_tanh_gm = tvm.compute(shape_i,
                                  lambda *indices: c_t_tanh_mid(*indices),
                                  name="c_t_tanh_gm",
                                  tag="ub_to_out")
        c_t_tanh_back = tvm.compute(shape_i,
                                    lambda *indices: c_t_tanh_gm(*indices),
                                    name="c_t_tanh_back",
                                    tag="out_to_ub")

        if fp16_input_output:
            c_t_tanh_back_fp32 = tvm.compute(shape_i,
                                             lambda *indices: c_t_tanh_back(*indices).astype(data_type),
                                             name="c_t_tanh_back_fp32_drnn_cast",
                                             tag="elewise_single_cast")

        c_t_tanh_ub = c_t_tanh_back

        if fp16_input_output:
            c_t_tanh_ub = c_t_tanh_back_fp32

    update_h = vmul(c_t_tanh_ub, o_t_sigmoid_ub)

    if seq_mask_gm is not None:
        update_h = vmul(update_h, seq_mask_ub)

    if fp16_input_output:
        update_h_fp16 = tvm.compute(shape_i,
                                    lambda *indices: update_h(*indices).astype('float16'),
                                    name="update_h_fp16_drnn_cast",
                                    tag="elewise_single_cast")
        update_h_gm_as_y = tvm.compute(shape_i,
                                       lambda *indices: update_h_fp16(*indices),
                                       name="update_h_gm_as_y",
                                       tag="ub_to_out")
        update_h_gm_as_y_back = tvm.compute(shape_i,
                                            lambda *indices: update_h_gm_as_y(*indices),
                                            name="update_h_gm_as_y_back",
                                            tag="out_to_ub")
    else:
        update_h_gm_as_y = tvm.compute(shape_i,
                                       lambda *indices: update_h(*indices),
                                       name="update_h_gm_as_y",
                                       tag="ub_to_out")
        update_h_gm_as_y_back = tvm.compute(shape_i,
                                            lambda *indices: update_h_gm_as_y(*indices),
                                            name="update_h_gm_as_y_back",
                                            tag="out_to_ub")

    update_h_gm_as_y_back_mid = update_h_gm_as_y_back
    if seq_mask_gm is not None:
        update_h_diff = vsub(update_h_gm_as_y_back, s_state_h_ub_for_element)
        update_h_diff_mask = vmul(update_h_diff, seq_mask_ub)
        update_h_gm_as_y_back_mid = vadd(update_h_diff_mask, s_state_h_ub_for_element)

        if cell_type == "BLOCKLSTM":
            update_h_gm_as_y_back_mid = vmul(update_h_gm_as_y_back_mid, seq_mask_ub)

    if fp16_input_output:
        update_h_gm = tvm.compute(shape_i,
                                  lambda *indices: update_h_gm_as_y_back_mid(*indices),
                                  name="update_h_gm",
                                  tag="ub_to_out")
    else:
        update_h_fp16_cast = tvm.compute(shape_i,
                                         lambda *indices: update_h_gm_as_y_back_mid(*indices).astype('float16'),
                                         name="update_h_fp16_cast_drnn_cast",
                                         tag="elewise_single_cast")
        update_h_gm = tvm.compute(shape_i,
                                  lambda *indices: update_h_fp16_cast(*indices),
                                  name="update_h_gm",
                                  tag="ub_to_out")

    # end compute
    if is_gate_output:
        return_list = [update_h_gm, update_c_gm, update_h_gm_as_y,
                       i_t_sigmoid_gm,
                       j_t_tanh_gm, f_t_sigmoid_gm, o_t_sigmoid_gm,
                       c_t_tanh_gm]
    else:
        return_list = [update_h_gm, update_c_gm, update_h_gm_as_y]
    return_list, s = rl_bank.tik_dsl_bank_proc(return_list, sync_tensor=sync0)
    if s is not None:
        return return_list, s

    bank_manager.update_bank_hit_info(True)
    # schedule
    s = create_schedule([update_h_gm.op])

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
                if in_tensor not in visited_list:
                    stack.append(in_tensor)
                    if "elewise" in in_tensor.op.tag or in_tensor.op.tag == "broadcast":
                        if in_tensor.name.endswith("_drnn_cast"):
                            continue
                        if in_tensor.name in ["s_state_h_ub", "s_state_c_ub", "s_state_h_ub_for_element"]:
                            continue
                        if in_tensor not in tensor_list:
                            tensor_list.append(in_tensor)

    elewise_tensors = []
    gen_reversed_subgraph_list(update_h_gm, elewise_tensors)

    barrier_tensor = c_ub_bias
    elewise_before_barrier_tensors = [bias_bc_ub]

    # set scope
    s[a_l1].set_scope(scope_cbuf)
    s[b_l1].set_scope(scope_cbuf)
    s[a_l0a].set_scope(scope_ca)
    s[b_l0b].set_scope(scope_cb)
    s[c_l0c].set_scope(scope_cc)
    s[c_ub].set_scope(scope_ubuf)
    s[bias_ub].set_scope(scope_ubuf)
    s[bias_bc_ub].set_scope(scope_ubuf)
    s[s_state_h_ub].set_scope(scope_ubuf)
    s[s_state_c_ub].set_scope(scope_ubuf)
    if seq_mask_gm is not None:
        s[s_state_h_ub_for_element].set_scope(scope_ubuf)

    s[a_ub].set_scope(scope_ubuf)

    if fp16_input_output:
        s[bias_ub_fp32].set_scope(scope_ubuf)

    for tensor in elewise_tensors:
        s[tensor].set_scope(scope_ubuf)

    if is_gate_output:
        if fp16_input_output:
            s[f_t_sigmoid_fp16].set_scope(scope_ubuf)
            s[i_t_sigmoid_fp16].set_scope(scope_ubuf)
            s[o_t_sigmoid_fp16].set_scope(scope_ubuf)
            s[j_t_tanh_fp16].set_scope(scope_ubuf)
            s[c_t_tanh_fp16].set_scope(scope_ubuf)
            s[f_t_sigmoid_back_fp32].set_scope(scope_ubuf)
            s[i_t_sigmoid_back_fp32].set_scope(scope_ubuf)
            s[o_t_sigmoid_back_fp32].set_scope(scope_ubuf)
            s[j_t_tanh_back_fp32].set_scope(scope_ubuf)
            s[c_t_tanh_back_fp32].set_scope(scope_ubuf)

        s[f_t_sigmoid_back].set_scope(scope_ubuf)
        s[i_t_sigmoid_back].set_scope(scope_ubuf)
        s[o_t_sigmoid_back].set_scope(scope_ubuf)
        s[j_t_tanh_back].set_scope(scope_ubuf)
        s[c_t_tanh_back].set_scope(scope_ubuf)

    # fp16 in
    if bias_dtype == 'float16':
        s[update_c_fp16].set_scope(scope_ubuf)
        s[update_c_fp16_back].set_scope(scope_ubuf)
        s[update_c_fp16_back_fp32].set_scope(scope_ubuf)
        s[update_h_fp16].set_scope(scope_ubuf)
    else:
        s[update_c_fp32_back].set_scope(scope_ubuf)
        s[update_h_fp16_cast].set_scope(scope_ubuf)

    s[update_h_gm_as_y_back].set_scope(scope_ubuf)

    if seq_mask_gm is not None:
        s[seq_mask_ub].set_scope(scope_ubuf)
    # compute inline
    compute_inline_tensors = [i_t, j_t, f_t, o_t]
    for tensor in compute_inline_tensors:
        s[tensor].compute_inline()

    # matmul tiling
    factor_l1_m, factor_l1_n, factor_l1_k, \
    factor_l0_m, factor_l0_n, factor_l0_k = \
        get_lstm_tiling()

    l1_n_outer, l1_n_inner = \
        s[c_l0c].split(c_l0c.op.axis[2],
                       factor=factor_l1_n)

    l1_m_outer, l1_m_inner = \
        s[c_l0c].split(c_l0c.op.axis[3],
                       factor=factor_l1_m)
    l1_k_outer, l1_k_inner = \
        s[c_l0c].split(c_l0c.op.reduce_axis[0],
                       factor=factor_l1_k)

    l0_n_outer, l0_n_inner = s[c_l0c].split(l1_n_inner,
                                            factor=factor_l0_n)
    l0_m_outer, l0_m_inner = s[c_l0c].split(l1_m_inner,
                                            factor=factor_l0_m)
    l0_k_outer, l0_k_inner = s[c_l0c].split(l1_k_inner,
                                            factor=factor_l0_k)

    s[c_l0c].reorder(l1_n_outer, c_l0c.op.axis[1],
                     l1_m_outer, l1_k_outer,
                     l0_n_outer, l0_m_outer, l0_k_outer,
                     l0_n_inner, l0_m_inner, c_l0c.op.axis[3 + 1],
                     c_l0c.op.axis[4 + 1], l0_k_inner,
                     c_l0c.op.reduce_axis[1])

    s[a_ub].compute_at(s[c_l0c], l1_k_outer)
    s[s_state_h_ub].compute_at(s[c_l0c], l1_k_outer)

    s[a_l0a].compute_at(s[c_l0c], l0_k_outer)
    s[b_l0b].compute_at(s[c_l0c], l0_k_outer)
    s[a_l1].compute_at(s[c_l0c], l1_k_outer)
    s[b_l1].compute_at(s[c_l0c], l1_k_outer)

    ub_n_outer, ub_n_inner = \
        s[c_ub].split(c_ub.op.axis[2],
                      factor=factor_l1_n)

    ub_m_outer, ub_m_inner = s[c_ub].split(c_ub.op.axis[3],
                                           factor=factor_l1_m)
    s[c_ub].reorder(ub_m_outer, ub_n_outer, c_ub.op.axis[1],
                    ub_n_inner, ub_m_inner, c_ub.op.axis[4],
                    c_ub.op.axis[5])

    s[c_l0c].compute_at(s[c_ub], ub_n_outer)

    # elewise compute_at
    barrier_outer, barrier_inner = \
        s[barrier_tensor].split(barrier_tensor.op.axis[2],
                                factor=factor_l1_n)
    barrier_m_outer, barrier_m_inner = \
        s[barrier_tensor].split(barrier_tensor.op.axis[3],
                                factor=factor_l1_m)
    s[barrier_tensor].reorder(
        barrier_tensor.op.axis[0], barrier_m_outer, barrier_outer,
        barrier_tensor.op.axis[1], barrier_inner, barrier_m_inner,
        barrier_tensor.op.axis[4],
        barrier_tensor.op.axis[5])

    s[c_ub].compute_at(s[barrier_tensor], barrier_outer)
    s[bias_ub].compute_at(s[barrier_tensor], barrier_outer)
    if fp16_input_output:
        s[bias_ub_fp32].compute_at(s[barrier_tensor], barrier_outer)

    for tensor in elewise_before_barrier_tensors:
        s[tensor].compute_at(s[barrier_tensor], barrier_outer)

    vn_outer, vn_inner = \
        s[update_h_gm].split(update_h_gm.op.axis[0 + 1],
                             factor=factor_l1_n)
    vn_m_outer, vn_m_inner = \
        s[update_h_gm].split(update_h_gm.op.axis[0 + 2],
                             factor=factor_l1_m)
    second_split_factor = \
        (hidden_size // factor_l1_n) // 1

    vn_o_outer, vn_o_inner = \
        s[update_h_gm].split(vn_outer,
                             factor=second_split_factor)
    s[update_h_gm].reorder(update_h_gm.op.axis[0], vn_m_outer,
                           vn_o_outer, vn_o_inner, vn_inner,
                           vn_m_inner, update_h_gm.op.axis[3],
                           update_h_gm.op.axis[4])

    s[s_state_c_ub].compute_at(s[update_h_gm], vn_o_inner)
    if seq_mask_gm is not None:
        s[s_state_h_ub_for_element].compute_at(s[update_h_gm], vn_o_inner)
    s[barrier_tensor].compute_at(s[update_h_gm], vn_o_inner)

    for tensor in elewise_tensors:
        if tensor not in elewise_before_barrier_tensors:
            s[tensor].compute_at(s[update_h_gm], vn_o_inner)

    s[update_c_gm].compute_at(s[update_h_gm], vn_o_inner)

    # fp16 in
    if bias_dtype == 'float16':
        s[update_c_fp16].compute_at(s[update_h_gm], vn_o_inner)
        s[update_c_fp16_back].compute_at(s[update_h_gm], vn_o_inner)
        s[update_c_fp16_back_fp32].compute_at(s[update_h_gm], vn_o_inner)
        s[update_h_fp16].compute_at(s[update_h_gm], vn_o_inner)
    else:
        s[update_c_fp32_back].compute_at(s[update_h_gm], vn_o_inner)
        s[update_c].compute_at(s[update_h_gm], vn_o_inner)
        s[update_h_fp16_cast].compute_at(s[update_h_gm], vn_o_inner)
    if seq_mask_gm is not None:
        s[seq_mask_ub].compute_at(s[update_h_gm], vn_o_inner)
    s[update_h_gm_as_y].compute_at(s[update_h_gm], vn_o_inner)
    s[update_h_gm_as_y_back].compute_at(s[update_h_gm], vn_o_inner)

    if is_gate_output:
        if fp16_input_output:
            s[f_t_sigmoid_fp16].compute_at(s[update_h_gm], vn_o_inner)
            s[i_t_sigmoid_fp16].compute_at(s[update_h_gm], vn_o_inner)
            s[o_t_sigmoid_fp16].compute_at(s[update_h_gm], vn_o_inner)
            s[j_t_tanh_fp16].compute_at(s[update_h_gm], vn_o_inner)
            s[c_t_tanh_fp16].compute_at(s[update_h_gm], vn_o_inner)
            s[f_t_sigmoid_back_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[i_t_sigmoid_back_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[o_t_sigmoid_back_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[j_t_tanh_back_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[c_t_tanh_back_fp32].compute_at(s[update_h_gm], vn_o_inner)

        s[f_t_sigmoid_gm].compute_at(s[update_h_gm], vn_o_inner)
        s[i_t_sigmoid_gm].compute_at(s[update_h_gm], vn_o_inner)
        s[o_t_sigmoid_gm].compute_at(s[update_h_gm], vn_o_inner)
        s[j_t_tanh_gm].compute_at(s[update_h_gm], vn_o_inner)
        s[c_t_tanh_gm].compute_at(s[update_h_gm], vn_o_inner)

        s[f_t_sigmoid_back].compute_at(s[update_h_gm], vn_o_inner)
        s[i_t_sigmoid_back].compute_at(s[update_h_gm], vn_o_inner)
        s[o_t_sigmoid_back].compute_at(s[update_h_gm], vn_o_inner)
        s[j_t_tanh_back].compute_at(s[update_h_gm], vn_o_inner)
        s[c_t_tanh_back].compute_at(s[update_h_gm], vn_o_inner)

    if is_gate_output:
        s[f_t_sigmoid].reused_by(f_t_sigmoid_ub)
        s[i_t_sigmoid].reused_by(i_t_sigmoid_ub)
        s[o_t_sigmoid].reused_by(o_t_sigmoid_ub)
        s[j_t_tanh].reused_by(j_t_tanh_ub)
        s[c_t_tanh].reused_by(c_t_tanh_ub)

        s[f_t_sigmoid_ub].reused_by(reuse_data=True)
        s[i_t_sigmoid_ub].reused_by(reuse_data=True)
        s[o_t_sigmoid_ub].reused_by(reuse_data=True)
        s[j_t_tanh_ub].reused_by(reuse_data=True)
        s[c_t_tanh_ub].reused_by(reuse_data=True)

    if bias_dtype == 'float16':
        s[update_c].reused_by(update_c_fp16_back_fp32)
        s[update_c_fp16_back_fp32].reused_by(reuse_data=True)
        s[update_h_fp16].reused_by(update_h_gm_as_y_back)
        s[update_h_gm_as_y_back].reused_by(reuse_data=True)
    else:
        s[update_c].reused_by(update_c_fp32_back)
        s[update_c_fp32_back].reused_by(reuse_data=True)
        s[update_h].reused_by(update_h_gm_as_y_back)
        s[update_h_gm_as_y_back].reused_by(reuse_data=True)

    s[a_l1].double_buffer()
    s[b_l1].double_buffer()
    s[a_l0a].double_buffer()
    s[b_l0b].double_buffer()
    s[c_l0c].double_buffer()
    s[c_ub].double_buffer()

    # emit_insn
    s[a_l1].emit_insn(a_l1.op.axis[0], 'dma_copy')
    s[b_l1].emit_insn(b_l1.op.axis[0], 'dma_copy')
    s[a_l0a].emit_insn(a_l0a.op.axis[0], 'dma_copy')
    s[b_l0b].emit_insn(b_l0b.op.axis[0], 'dma_copy')

    s[a_ub].emit_insn(a_ub.op.axis[0], 'dma_copy')

    if fp16_input_output and data_type == 'float32':
        s[bias_ub_fp32].emit_insn(bias_ub_fp32.op.axis[0], 'vector_conv')

    mad_dict = {"mad_pattern": 0, "k_outer": [l1_k_outer, l0_k_outer]}
    s[c_l0c].emit_insn(l0_n_inner, 'mad', mad_dict)
    s[c_ub].emit_insn(ub_n_inner, 'dma_copy')

    s[bias_bc_ub].emit_insn(bias_bc_ub.op.axis[0], 'unified_broadcast')

    if is_first_round:
        if is_global_init:
            s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'dma_copy')
            s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'dma_copy')
            if seq_mask_gm is not None:
                s[s_state_h_ub_for_element].emit_insn(s_state_h_ub_for_element.op.axis[0], 'dma_copy')
        else:
            s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'broadcast')
            s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'broadcast')
            if seq_mask_gm is not None:
                s[s_state_h_ub_for_element].emit_insn(s_state_h_ub_for_element.op.axis[0], 'broadcast')
    else:
        s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'dma_copy')
        s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'dma_copy')
        if seq_mask_gm is not None:
            s[s_state_h_ub_for_element].emit_insn(s_state_h_ub_for_element.op.axis[0], 'dma_copy')

    s[barrier_tensor].emit_insn(barrier_tensor.op.axis[1], 'vector_add')

    for tensor in elewise_tensors:
        if tensor != barrier_tensor:
            insn = get_emit_insn_map(tensor)
            s[tensor].emit_insn(tensor.op.axis[0], insn)

    s[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')

    s[update_c_gm].emit_insn(s[update_c_gm].op.axis[1], 'dma_copy')
    s[update_h_gm].emit_insn(s[update_h_gm].op.axis[3], 'dma_copy')

    block = tvm.thread_axis('blockIdx.x')
    s[update_h_gm].bind(vn_o_outer, block)
    s[update_h_gm].wait_block_sync(axis=vn_m_outer, tensor=sync0[0], bottom=True)
    s[update_h_gm].set_block_sync(axis=vn_m_outer, tensor=sync0[0], bottom=True)

    if is_gate_output:
        if fp16_input_output:
            if data_type == 'float32':
                s[f_t_sigmoid_fp16].emit_insn(s[f_t_sigmoid_fp16].op.axis[1],
                                            'vector_conv')
                s[i_t_sigmoid_fp16].emit_insn(s[i_t_sigmoid_fp16].op.axis[1],
                                            'vector_conv')
                s[o_t_sigmoid_fp16].emit_insn(s[o_t_sigmoid_fp16].op.axis[1],
                                            'vector_conv')
                s[j_t_tanh_fp16].emit_insn(s[j_t_tanh_fp16].op.axis[1],
                                        'vector_conv')
                s[c_t_tanh_fp16].emit_insn(s[c_t_tanh_fp16].op.axis[1],
                                        'vector_conv')
            s[f_t_sigmoid_back_fp32].emit_insn(
                s[f_t_sigmoid_back_fp32].op.axis[1], 'phony_insn')
            s[i_t_sigmoid_back_fp32].emit_insn(
                s[i_t_sigmoid_back_fp32].op.axis[1], 'phony_insn')
            s[o_t_sigmoid_back_fp32].emit_insn(
                s[o_t_sigmoid_back_fp32].op.axis[1], 'phony_insn')
            s[j_t_tanh_back_fp32].emit_insn(s[j_t_tanh_back_fp32].op.axis[1],
                                            'phony_insn')
            s[c_t_tanh_back_fp32].emit_insn(s[c_t_tanh_back_fp32].op.axis[1],
                                            'phony_insn')

        s[f_t_sigmoid_gm].emit_insn(s[f_t_sigmoid_gm].op.axis[1], 'dma_copy')
        s[i_t_sigmoid_gm].emit_insn(s[i_t_sigmoid_gm].op.axis[1], 'dma_copy')
        s[o_t_sigmoid_gm].emit_insn(s[o_t_sigmoid_gm].op.axis[1], 'dma_copy')
        s[j_t_tanh_gm].emit_insn(s[j_t_tanh_gm].op.axis[1], 'dma_copy')
        s[c_t_tanh_gm].emit_insn(s[c_t_tanh_gm].op.axis[1], 'dma_copy')

        s[f_t_sigmoid_back].emit_insn(s[f_t_sigmoid_back].op.axis[1],
                                      'phony_insn')
        s[i_t_sigmoid_back].emit_insn(s[i_t_sigmoid_back].op.axis[1],
                                      'phony_insn')
        s[o_t_sigmoid_back].emit_insn(s[o_t_sigmoid_back].op.axis[1],
                                      'phony_insn')
        s[j_t_tanh_back].emit_insn(s[j_t_tanh_back].op.axis[1], 'phony_insn')
        s[c_t_tanh_back].emit_insn(s[c_t_tanh_back].op.axis[1], 'phony_insn')

    # fp16 in
    if bias_dtype == 'float16':
        if data_type == 'float32':
            s[update_c_fp16].emit_insn(update_c_fp16.op.axis[0], 'vector_conv')
            s[update_h_fp16].emit_insn(update_h_fp16.op.axis[0], 'vector_conv')
        s[update_c_fp16_back].emit_insn(update_c_fp16_back.op.axis[0],
                                        'phony_insn')
        s[update_c_fp16_back_fp32].emit_insn(
            update_c_fp16_back_fp32.op.axis[0], 'phony_insn')
    else:
        s[update_c_fp32_back].emit_insn(update_c_fp32_back.op.axis[0],
                                        'phony_insn')
        s[update_h_fp16_cast].emit_insn(update_h_fp16_cast.op.axis[0],
                                        'vector_conv')
    if seq_mask_gm is not None:
        s[seq_mask_ub].emit_insn(seq_mask_ub.op.axis[0],
                                        'dma_copy')
    s[update_h_gm_as_y].emit_insn(update_h_gm_as_y.op.axis[0], 'dma_copy')
    s[update_h_gm_as_y_back].emit_insn(update_h_gm_as_y_back.op.axis[0],
                                       'phony_insn')

    return return_list, s
