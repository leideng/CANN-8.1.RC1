#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

dynamic_lstm_v2
"""
# 'pylint: disable=too-many-lines
import te.platform as tbe_platform

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
from te.domain.rl_bank import rl_bank
from te.domain.rl_bank import bank_manager
from te.platform import scope_ca
from te.platform import scope_cb
from te.platform import scope_cbuf
from te.platform import scope_cc
from te.platform import scope_ubuf
from te.platform.fusion_manager import fusion_manager
from tbe.common.platform import api_check_support
from te.tik import Dprofile
from te.tik import Tik
from te.tik import scope_gm
from tbe import tvm
from tbe.tvm import create_schedule
from te.utils import para_check
from te.utils.error_manager import error_manager_vector


# 'pylint: disable=invalid-name
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
        error_manager_vector.raise_err_specific_reson("DynamicLSTM",
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


def tanh_compute_high_precision(input_x):
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


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name,too-many-branches
# 'pylint: disable=too-many-function-args,too-many-statements,unused-argument
def dynamic_lstm_v2(input_x, weight, bias, cont, w_xc_x_static, h0, c0, wci, wcf,
                wco, mask, y, output_h, output_c, last_output_h, last_output_c,
                num_output=0, expose_hidden=False, need_output_last=False, forget_bias=0.0,
                kernel_name="dynamic_lstm", impl_mode='high_performance'):
    """
    dynamic_lstm_v2
    """

    shape_x_input = input_x.get("shape")
    shape_w_input = weight.get("shape")

    input_dtype = input_x.get("dtype").lower()
    bias_dtype = bias.get("dtype").lower()

    if bias_dtype == 'float32':
        impl_mode = "high_precision"

    product_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if product_version in ('Ascend310B', "AS31XM1", 'Ascend910B', 'Ascend910_93'):
        impl_mode = "high_precision"
    if product_version in ('Hi3796CV300ES', 'Hi3796CV300CS', 'Hi3796CV300SD3403'):
        impl_mode = "high_performance"

    t_size = shape_x_input[0]
    m_size = shape_x_input[2]
    k_size = shape_w_input[0]
    n_size = shape_w_input[1]

    block_size = 4
    hidden_size = n_size // 4
    in_x = k_size - hidden_size

    shape_x = (t_size, in_x, m_size, 16, 16)
    shape_w = (1, k_size, block_size, hidden_size, 16, 16)
    shape_hc = (t_size, hidden_size, m_size, 16, 16)
    shape_bias = (1, block_size, hidden_size, 1, 1, 16)
    shape_hc_init = (1, hidden_size, m_size, 16, 16)

    is_global_init = False
    if h0 is not None:
        is_global_init = True

    has_static = False
    if w_xc_x_static is not None:
        has_static = True

    tik_instance = Tik(Dprofile())

    input_x = tik_instance.Tensor(shape=shape_x, dtype=input_dtype,
                                  scope=scope_gm, name='input_x')
    weight = tik_instance.Tensor(shape=shape_w, dtype=input_dtype,
                                 scope=scope_gm, name='weight')
    bias = tik_instance.Tensor(shape=shape_bias, scope=scope_gm,
                               dtype=bias_dtype, name='bias')
    sync = tik_instance.Tensor(shape=(128, ), dtype='int64', scope=scope_gm, name='sync',
                               is_workspace=True, is_atomic_add=True)
    cont = tik_instance.Tensor(shape=cont['shape'], dtype=cont['dtype'], scope=scope_gm, name='cont')

    if is_global_init:
        s_init_h_gm = tik_instance.Tensor(shape=shape_hc_init,
                                          dtype=input_dtype,
                                          scope=scope_gm,
                                          name='s_init_h_gm')
        s_init_c_gm = tik_instance.Tensor(shape=shape_hc_init, dtype=bias_dtype, scope=scope_gm, name='s_init_c_gm')
    if need_output_last:
        last_output_h_gm = tik_instance.Tensor(shape=shape_hc_init, dtype=input_dtype,
                                               scope=scope_gm, name='last_output_h_gm')
        last_output_c_gm = tik_instance.Tensor(shape=shape_hc_init, dtype=input_dtype,
                                               scope=scope_gm, name='last_output_c_gm')

    update_h_gm = tik_instance.Tensor(shape=shape_hc, dtype=input_dtype,
                                      scope=scope_gm, name='update_h_gm')
    update_c_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                      scope=scope_gm, name='update_c_gm')
    update_h_gm_as_y = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                           scope=scope_gm,
                                           name='update_h_gm_as_y')
    build_input_list = [input_x, weight, bias, cont]
    if has_static:
        w_xc_x_static_gm = tik_instance.Tensor(shape=w_xc_x_static['shape'], dtype=w_xc_x_static['dtype'],
                                               scope=scope_gm, name='w_xc_x_static_gm')
        build_input_list.append(w_xc_x_static_gm)
    else:
        w_xc_x_static_gm = None
    if is_global_init:
        build_input_list.extend([s_init_h_gm, s_init_c_gm])

    build_output_list = [update_h_gm_as_y, update_h_gm, update_c_gm]
    if need_output_last:
        build_output_list.extend([last_output_h_gm, last_output_c_gm])

    # for RL tune getting tik input&output tensor
    fusion_manager.set_tik_tensor(build_input_list, build_output_list)
    bank_manager.init_bank_hit_info(kernel_name)

    last = 1
    cut_t = 1
    # RL default tiling
    cut_m = m_size
    loop_t = t_size // cut_t
    loop_m = m_size // cut_m

    with tik_instance.for_range(0, loop_t) as loop_i:
        with tik_instance.for_range(0, loop_m) as loop_j:

            input_x_var = input_x[loop_i * cut_t: loop_i * cut_t + cut_t,
                          :,
                          loop_j * cut_m: loop_j * cut_m + cut_m,
                          :, :]
            cont_var = cont[loop_i, :]
            if need_output_last:
                last_output_h_gm_var = last_output_h_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m,
                                  :, :]
                last_output_c_gm_var = last_output_c_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m,
                                  :, :]
            if is_global_init:
                s_init_c_gm_var = s_init_c_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m,
                                  :, :]
                s_init_h_gm_var = s_init_h_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m,
                                  :, :]
            else:
                s_init_c_gm_var = None
                s_init_h_gm_var = None

            state_h_last = update_h_gm[
                           loop_i * cut_t - last: loop_i * cut_t + cut_t - last:,
                           :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            state_c_last = update_c_gm[
                           loop_i * cut_t - last: loop_i * cut_t + cut_t - last:,
                           :,
                           loop_j * cut_m: loop_j * cut_m + cut_m,
                           :, :]

            update_h_gm_var = update_h_gm[
                              loop_i * cut_t: loop_i * cut_t + cut_t,
                              :,
                              loop_j * cut_m: loop_j * cut_m + cut_m,
                              :,
                              :]
            update_c_gm_var = update_c_gm[
                              loop_i * cut_t: loop_i * cut_t + cut_t,
                              :,
                              loop_j * cut_m: loop_j * cut_m + cut_m,
                              :,
                              :]
            update_h_gm_as_y_var = update_h_gm_as_y[
                                   loop_i * cut_t: loop_i * cut_t + cut_t:,
                                   :,
                                   loop_j * cut_m: loop_j * cut_m + cut_m,
                                   :, :]

            input_list = [input_x_var, weight, bias, s_init_h_gm_var,
                          s_init_c_gm_var, state_h_last, state_c_last, cont_var, w_xc_x_static_gm, sync]

            if need_output_last:
                output_list = [update_h_gm_var, update_c_gm_var,
                                update_h_gm_as_y_var, last_output_h_gm_var, last_output_c_gm_var]
            else:
                output_list = [update_h_gm_var, update_c_gm_var,
                                update_h_gm_as_y_var]

            if impl_mode == "high_performance":
                with tik_instance.if_scope(loop_i == 0):
                    is_first_round = True
                    tik_instance.call_module(
                        dynamic_rnn_tik_high_performance,
                        input_list,
                        output_list,
                        [is_first_round, is_global_init, has_static, need_output_last])

                with tik_instance.if_scope(loop_i > 0):
                    is_first_round = False
                    tik_instance.call_module(
                        dynamic_rnn_tik_high_performance,
                        input_list,
                        output_list,
                        [is_first_round, is_global_init, has_static, need_output_last])
            else:
                with tik_instance.if_scope(loop_i == 0):
                    is_first_round = True
                    tik_instance.call_module(
                        dynamic_rnn_tik_high_precision,
                        input_list,
                        output_list,
                        [is_first_round, is_global_init, has_static, need_output_last])

                with tik_instance.if_scope(loop_i > 0):
                    is_first_round = False
                    tik_instance.call_module(
                        dynamic_rnn_tik_high_precision,
                        input_list,
                        output_list,
                        [is_first_round, is_global_init, has_static, need_output_last])


    config_map = {
        "dump_cce_code": False,
    }

    tik_instance.BuildCCE(kernel_name,
                          build_input_list,
                          build_output_list,
                          config=config_map)


# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-function-args,too-many-statements
def dynamic_rnn_tik_high_performance(input_list, custom_list):
    """
    dynamic_rnn_tik_high_performance
    """
    input_x = input_list[0]
    weight = input_list[1]
    bias = input_list[2]
    s_init_h_gm = input_list[3]
    s_init_c_gm = input_list[4]
    s_state_h_gm_last = input_list[5]
    s_state_c_gm_last = input_list[6]
    seq_length_gm = input_list[7]
    static_gm = input_list[8]
    sync0 = input_list[9]

    is_first_round = custom_list[0]
    is_global_init = custom_list[1]
    has_static = custom_list[2]
    need_output_last = custom_list[3]

    return dynamic_rnn_core_high_preformance(input_x, weight, bias, seq_length_gm, static_gm, s_init_h_gm, s_init_c_gm,
                            s_state_h_gm_last, s_state_c_gm_last, sync0,
                            is_first_round, is_global_init, has_static, need_output_last)


# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-function-args,too-many-statements
def dynamic_rnn_tik_high_precision(input_list, custom_list):
    """
    dynamic_rnn_tik_high_precision
    """
    input_x = input_list[0]
    weight = input_list[1]
    bias = input_list[2]
    s_init_h_gm = input_list[3]
    s_init_c_gm = input_list[4]
    s_state_h_gm_last = input_list[5]
    s_state_c_gm_last = input_list[6]
    seq_length_gm = input_list[7]
    static_gm = input_list[8]
    sync0 = input_list[9]

    is_first_round = custom_list[0]
    is_global_init = custom_list[1]
    has_static = custom_list[2]
    need_output_last = custom_list[3]

    return dynamic_rnn_core_high_precision(input_x, weight, bias, seq_length_gm, static_gm, s_init_h_gm, s_init_c_gm,
                            s_state_h_gm_last, s_state_c_gm_last, sync0,
                            is_first_round, is_global_init, has_static, need_output_last)


# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-statements,unnecessary-lambda,too-many-branches
def dynamic_rnn_core_high_preformance(input_x, weight, bias, seq_length, static, s_init_h_gm, s_init_c_gm,
                     s_state_h_gm_last, s_state_c_gm_last, sync0, is_first_round, is_global_init,
                     has_static, need_output_last):
    """
    dynamic rnn core tvm
    """
    shape_x_input = input_x.shape
    shape_w_input = weight.shape

    t_size = 1
    m_size = shape_x_input[2].value
    k_size = shape_w_input[1].value
    n_size = shape_w_input[2].value * shape_w_input[3].value

    block_size = 4
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

    # compute

    if is_first_round:
        if is_global_init:
            s_state_h_ub = tvm.compute(shape_h,
                                       lambda _, i, j, m, n: s_init_h_gm[0, i, j, m, n], name="s_init_h")
            s_state_c_ub = tvm.compute(shape_i,
                                       lambda _, i, j, m, n: s_init_c_gm[0, i, j, m, n], name='s_init_c')
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
    else:
        s_state_h_ub = tvm.compute(shape_h,
                                   lambda _, i, j, m, n: s_state_h_gm_last[0, i, j, m, n],
                                    name="s_state_h_ub")
        s_state_c_ub = tvm.compute(shape_i,
                                   lambda _, i, j, m, n: s_state_c_gm_last[0, i, j, m, n],
                                    name="s_state_c_ub")

    # handle cont mul h  caffe
    tmp_shape = [1, 1, (seq_length.shape[1] + 15) // 16, 16, 1]
    tensor_seq_length_ub = tvm.compute(
        tmp_shape, lambda i, j, k, m, n: seq_length[0, k * 16 + m], name='tensor_seq_length_ub'
        )
    tensor_seq_length_bc_ub = broadcast(tensor_seq_length_ub, shape_h)
    s_state_h_mul_cont_ub = vmul(s_state_h_ub, tensor_seq_length_bc_ub)


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
                                  s_state_h_mul_cont_ub[0,
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

    c_l0c = tvm.compute(shape_c,
                        lambda t, nb_0, nb_1, mb, mp, np:
                        tvm.sum((a_l0a[t, mb, k1, mp, k0] * \
                                 b_l0b[t, k1, nb_0, nb_1, np, k0]),
                                axis=[k1, k0]),
                        name='c_l0c',
                        tag="matmul")

    c_ub = tvm.compute(shape_c, lambda *indices: c_l0c(*indices), name="c_ub")

    bias_ub = tvm.compute(shape_bias,
                          lambda *indices: bias(*indices),
                          name='bias_ub')

    bias_ub_mid = bias_ub
    bias_bc_ub = broadcast(bias_ub_mid, shape_c)
    c_ub_bias = vadd(c_ub, bias_bc_ub)

    # split matmul res
    i_t_index = 0
    j_t_index = 3
    f_t_index = 1
    o_t_index = 2

    j_t = tvm.compute(shape_i,
                    lambda t, i, j, m, n: c_ub_bias(t, j_t_index, i, j, m, n),
                    name="j_t",
                    tag="split_com")
    if has_static:
        i_t = tvm.compute(shape_i,
                    lambda t, i, j, m, n: c_ub_bias(t, i_t_index, i, j, m, n),
                    name="i_t",
                    tag="split_com")
        f_t = tvm.compute(shape_i,
                    lambda t, i, j, m, n: c_ub_bias(t, f_t_index, i, j, m, n),
                    name="f_t",
                    tag="split_com")
        o_t = tvm.compute(shape_i,
                    lambda t, i, j, m, n: c_ub_bias(t, o_t_index, i, j, m, n),
                    name="o_t",
                    tag="split_com")
        shape_static = (1, n_size, m_size, 16, 16)
        output_dim = shape_i[1]
        tensor_static_ub = tvm.compute(shape_static, lambda _, j, k, m, n: static(j, k, m, n), name="tensor_static_ub")
        it_add_static = tvm.compute(
            shape_i, lambda t, i, j, m, n: i_t(t, i, j, m, n) + tensor_static_ub(t, i, j, m, n),
            name='it_add_static', tag='elewise_binary_add'
        )
        ft_add_static = tvm.compute(
            shape_i, lambda t, i, j, m, n: f_t(t, i, j, m, n) + tensor_static_ub(t, i + output_dim, j, m, n),
            name='ft_add_static', tag='elewise_binary_add'
        )
        ot_add_static = tvm.compute(
            shape_i, lambda t, i, j, m, n: o_t(t, i, j, m, n) + tensor_static_ub(t, i + output_dim * 2, j, m, n),
            name='ot_add_static', tag='elewise_binary_add'
        )
        jt_add_static = tvm.compute(
            shape_i, lambda t, i, j, m, n: j_t(t, i, j, m, n) + tensor_static_ub(t, i + output_dim * 3, j, m, n),
            name='jt_add_static', tag='elewise_binary_add'
        )
        f_t_sigmoid = sigmoid_compute(ft_add_static)
        i_t_sigmoid = sigmoid_compute(it_add_static)
        o_t_sigmoid = sigmoid_compute(ot_add_static)
        j_t_tanh = tanh_compute(jt_add_static)
    else:
        shape_fio = (t_size, 3, hidden_size, m_size, 16, 16)
        f_i_o = tvm.compute(shape_fio, lambda t, x, i, j, m, n: c_ub_bias(t, x, i, j, m, n),
         name='f_i_o', tag="split_com")
        f_i_o_sigmoid = sigmoid_compute(f_i_o)
        f_t_sigmoid = tvm.compute(shape_i,
                    lambda t, i, j, m, n: f_i_o_sigmoid(t, f_t_index, i, j, m, n),
                    name="f_t_sigmoid", tag="split_com")
        i_t_sigmoid = tvm.compute(shape_i,
                lambda t, i, j, m, n: f_i_o_sigmoid(t, i_t_index, i, j, m, n),
                name="i_t_sigmoid", tag="split_com")
        o_t_sigmoid = tvm.compute(shape_i,
                lambda t, i, j, m, n: f_i_o_sigmoid(t, o_t_index, i, j, m, n),
                name="o_t_sigmoid", tag="split_com")
        j_t_tanh = tanh_compute(j_t)

    if ''.join([str(i) for i in shape_h]) == ''.join([str(i) for i in shape_i]):
        tensor_cont_ub = tensor_seq_length_bc_ub
    else:
        tensor_cont_ub = tensor_seq_length_ub
    tensor_seq_length_ub_conv = tensor_cont_ub
    if tensor_cont_ub.dtype != f_t_sigmoid.dtype:
        tensor_seq_length_ub_conv = tvm.compute(
            tensor_cont_ub.shape, lambda *i: tensor_cont_ub(*i).astype(f_t_sigmoid.dtype),
            name='tensor_seq_length_ub_conv', tag='elewise_single_cast'
            )
    tensor_seq_length_ub_bc_conv = tensor_seq_length_ub_conv
    if ''.join((str(i) for i in shape_h)) != ''.join((str(i) for i in shape_i)):
        tensor_seq_length_ub_bc_conv = broadcast(tensor_seq_length_ub_conv, shape_i)

    f_t_sigmoid_mul_cont = vmul(f_t_sigmoid, tensor_seq_length_ub_bc_conv)
    f_t_sigmoid_ub = f_t_sigmoid_mul_cont
    i_t_sigmoid_ub = i_t_sigmoid
    o_t_sigmoid_ub = o_t_sigmoid
    j_t_tanh_ub = j_t_tanh

    # auto cast support both fp16 fp32
    c_t_tmp1 = vmul(s_state_c_ub, f_t_sigmoid_ub)
    c_t_tmp2 = vmul(j_t_tanh_ub, i_t_sigmoid_ub)
    update_c = vadd(c_t_tmp1, c_t_tmp2)
    update_c_gm = tvm.compute(shape_i,
                                lambda *indices: update_c(*indices),
                                name="update_c_gm",
                                tag="ub_to_out")
    update_c_fp16_back = tvm.compute(shape_i,
                                         lambda *indices: update_c_gm(*indices),
                                         name="update_c_fp16_back",
                                         tag="out_to_ub")
    if need_output_last:
        last_update_c_gm = tvm.compute(
            shape_i, lambda *indices: update_c_fp16_back(*indices), name='last_update_c_gm',
            tag='ub_to_out'
        )
        last_update_c_back = tvm.compute(
            shape_i, lambda *indices: last_update_c_gm(*indices), name='last_update_c_back',
            tag='out_to_ub'
        )
        c_t_tanh = tanh_compute(last_update_c_back)
    else:
        c_t_tanh = tanh_compute(update_c_fp16_back)
    c_t_tanh_ub = c_t_tanh

    update_h = vmul(c_t_tanh_ub, o_t_sigmoid_ub)
    update_h_gm_as_y = tvm.compute(shape_i,
                                    lambda *indices: update_h(*indices),
                                    name="update_h_gm_as_y",
                                    tag="ub_to_out")
    update_h_gm_as_y_back = tvm.compute(shape_i,
                                        lambda *indices: update_h_gm_as_y(*indices),
                                        name="update_h_gm_as_y_back",
                                        tag="out_to_ub")
    if need_output_last:
        last_update_h_gm = tvm.compute(
            shape_i, lambda *indices: update_h_gm_as_y_back(*indices), name='last_update_h_gm',
            tag='ub_to_out'
        )
        last_update_h_back = tvm.compute(
            shape_i, lambda *indices: last_update_h_gm(*indices), name='last_update_h_back',
            tag='out_to_ub'
        )
        update_h_gm = tvm.compute(shape_i,
                                lambda *indices: last_update_h_back(*indices),
                                name="update_h_gm",
                                tag="ub_to_out")
    else:
        update_h_gm = tvm.compute(shape_i,
                                lambda *indices: update_h_gm_as_y_back(*indices),
                                name="update_h_gm",
                                tag="ub_to_out")

    # end compute
    if not need_output_last:
        return_list = [update_h_gm, update_c_gm, update_h_gm_as_y]
    else:
        return_list = [update_h_gm, update_c_gm, update_h_gm_as_y, last_update_h_gm, last_update_c_gm]
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
                        if in_tensor.name in ["s_state_h_ub", "s_state_c_ub"]:
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
    s[tensor_seq_length_ub_bc_conv].set_scope(scope_ubuf)
    s[tensor_seq_length_bc_ub].set_scope(scope_ubuf)
    s[s_state_h_ub].set_scope(scope_ubuf)
    s[tensor_seq_length_ub].set_scope(scope_ubuf)
    if tensor_seq_length_ub.dtype != f_t_sigmoid.dtype:
        s[tensor_seq_length_ub_conv].set_scope(scope_ubuf)
    s[s_state_c_ub].set_scope(scope_ubuf)
    s[a_ub].set_scope(scope_ubuf)

    if has_static:
        s[tensor_static_ub].set_scope(scope_ubuf)

    if need_output_last:
        s[last_update_c_back].set_scope(scope_ubuf)
        s[last_update_h_back].set_scope(scope_ubuf)

    for tensor in elewise_tensors:
        s[tensor].set_scope(scope_ubuf)

    # fp16 in
    s[update_c_fp16_back].set_scope(scope_ubuf)
    s[update_h_gm_as_y_back].set_scope(scope_ubuf)

    # compute inline
    if has_static:
        compute_inline_tensors = [j_t, i_t, o_t, f_t, it_add_static, ft_add_static, ot_add_static, jt_add_static]
    else:
        compute_inline_tensors = [j_t, f_i_o, i_t_sigmoid, f_t_sigmoid, o_t_sigmoid]

    for tensor in compute_inline_tensors:
        s[tensor].compute_inline()

    # matmul tiling
    factor_l1_m, factor_l1_n, factor_l1_k, \
    factor_l0_m, factor_l0_n, factor_l0_k = 1, 1, 12, 1, 1, 12

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
    s[s_state_h_mul_cont_ub].compute_at(s[c_l0c], l1_k_outer)

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

    for tensor in elewise_before_barrier_tensors:
        s[tensor].compute_at(s[barrier_tensor], barrier_outer)

    vn_outer, vn_inner = \
        s[update_h_gm].split(update_h_gm.op.axis[0 + 1],
                             factor=factor_l1_n)
    vn_m_outer, vn_m_inner = \
        s[update_h_gm].split(update_h_gm.op.axis[0 + 2],
                             factor=factor_l1_m)
    # second_split_factor default is (hidden_size // factor_l1_n) // 1
    second_split_factor = (hidden_size // factor_l1_n) // 1

    vn_o_outer, vn_o_inner = \
        s[update_h_gm].split(vn_outer,
                             factor=second_split_factor)
    s[update_h_gm].reorder(update_h_gm.op.axis[0], vn_m_outer,
                           vn_o_outer, vn_o_inner, vn_inner,
                           vn_m_inner, update_h_gm.op.axis[3],
                           update_h_gm.op.axis[4])

    s[s_state_c_ub].compute_at(s[update_h_gm], vn_o_inner)
    s[s_state_h_ub].compute_at(s[update_h_gm], vn_o_inner)
    s[tensor_seq_length_ub_bc_conv].compute_at(s[update_h_gm], vn_o_inner)
    s[barrier_tensor].compute_at(s[update_h_gm], vn_o_inner)

    for tensor in elewise_tensors:
        if tensor not in elewise_before_barrier_tensors:
            s[tensor].compute_at(s[update_h_gm], vn_o_inner)

    s[update_c_gm].compute_at(s[update_h_gm], vn_o_inner)

    # fp16 in
    s[update_c_fp16_back].compute_at(s[update_h_gm], vn_o_inner)
    s[update_c].compute_at(s[update_h_gm], vn_o_inner)
    s[update_h_gm_as_y].compute_at(s[update_h_gm], vn_o_inner)
    s[update_h_gm_as_y_back].compute_at(s[update_h_gm], vn_o_inner)

    if need_output_last:
        s[last_update_c_gm].compute_at(s[update_h_gm], vn_o_inner)
        s[last_update_h_gm].compute_at(s[update_h_gm], vn_o_inner)
        s[last_update_c_back].compute_at(s[update_h_gm], vn_o_inner)
        s[last_update_h_back].compute_at(s[update_h_gm], vn_o_inner)
        s[update_c].reused_by(last_update_c_back)
        s[last_update_c_back].reused_by(reuse_data=True)
        s[update_h].reused_by(last_update_h_back)
        s[last_update_h_back].reused_by(reuse_data=True)

    s[update_c].reused_by(update_c_fp16_back)
    s[update_c_fp16_back].reused_by(reuse_data=True)
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

    s[tensor_seq_length_ub].emit_insn(tensor_seq_length_ub.op.axis[0], 'dma_copy')
    s[a_ub].emit_insn(a_ub.op.axis[0], 'dma_copy')

    if has_static:
        s[tensor_static_ub].emit_insn(tensor_static_ub.op.axis[0], 'dma_copy')
        s[tensor_static_ub].compute_at(s[update_h_gm], vn_o_inner)

    mad_dict = {"mad_pattern": 0, "k_outer": [l1_k_outer, l0_k_outer]}
    s[c_l0c].emit_insn(l0_n_inner, 'mad', mad_dict)
    s[c_ub].emit_insn(ub_n_inner, 'dma_copy')

    s[bias_bc_ub].emit_insn(bias_bc_ub.op.axis[0], 'unified_broadcast')
    s[tensor_seq_length_bc_ub].emit_insn(tensor_seq_length_bc_ub.op.axis[0], 'unified_broadcast')
    if ''.join((str(i) for i in shape_h)) != ''.join((str(i) for i in shape_i)):
        s[tensor_seq_length_ub_bc_conv].emit_insn(tensor_seq_length_ub_bc_conv.op.axis[0], 'unified_broadcast')

    if is_first_round:
        if is_global_init:
            s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'dma_copy')
            s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'dma_copy')
        else:
            s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'broadcast')
            s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'broadcast')
    else:
        s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'dma_copy')
        s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'dma_copy')

    s[barrier_tensor].emit_insn(barrier_tensor.op.axis[1], 'vector_add')

    for tensor in elewise_tensors:
        if tensor.op.name == 's_state_h_mul_cont_ub' or tensor.op.name == 'f_t_sigmoid_mul_cont':
            s[tensor].reorder(tensor.op.axis[2], tensor.op.axis[3], tensor.op.axis[1],
                        tensor.op.axis[0], tensor.op.axis[4])
        if tensor != barrier_tensor:
            insn = get_emit_insn_map(tensor)
            s[tensor].emit_insn(tensor.op.axis[0], insn)

    s[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')

    s[update_c_gm].emit_insn(s[update_c_gm].op.axis[1], 'dma_copy')
    if need_output_last:
        s[last_update_c_gm].emit_insn(s[last_update_c_gm].op.axis[1], 'dma_copy')
        s[last_update_h_gm].emit_insn(s[last_update_h_gm].op.axis[1], 'dma_copy')
    s[update_h_gm].emit_insn(s[update_h_gm].op.axis[3], 'dma_copy')

    block = tvm.thread_axis('blockIdx.x')
    s[update_h_gm].bind(vn_o_outer, block)
    s[update_h_gm].wait_block_sync(axis=vn_m_outer, tensor=sync0[0], bottom=True)
    s[update_h_gm].set_block_sync(axis=vn_m_outer, tensor=sync0[0], bottom=True)

    # fp16 in
    if need_output_last:
        s[last_update_c_back].emit_insn(last_update_c_back.op.axis[0], 'phony_insn')
        s[last_update_h_back].emit_insn(last_update_h_back.op.axis[0], 'phony_insn')
    s[update_c_fp16_back].emit_insn(update_c_fp16_back.op.axis[0],
                                        'phony_insn')
    s[update_h_gm_as_y].emit_insn(update_h_gm_as_y.op.axis[0], 'dma_copy')
    s[update_h_gm_as_y_back].emit_insn(update_h_gm_as_y_back.op.axis[0],
                                       'phony_insn')

    return return_list, s


# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-statements,unnecessary-lambda,too-many-branches
def dynamic_rnn_core_high_precision(input_x, weight, bias, seq_length, static, s_init_h_gm, s_init_c_gm,
                     s_state_h_gm_last, s_state_c_gm_last, sync0, is_first_round, is_global_init,
                     has_static, need_output_last):
    """
    dynamic rnn core tvm
    """
    shape_x_input = input_x.shape
    shape_w_input = weight.shape

    t_size = 1
    m_size = shape_x_input[2].value
    k_size = shape_w_input[1].value
    n_size = shape_w_input[2].value * shape_w_input[3].value

    block_size = 4
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

    # compute
    if is_first_round:
        if is_global_init:
            s_state_h_ub = tvm.compute(shape_h,
                                       lambda _, i, j, m, n: s_init_h_gm[
                                           0, i, j, m, n], name="s_init_h")
            s_state_c_ub = tvm.compute(shape_i,
                                       lambda _, i, j, m, n: s_init_c_gm[0, i, j, m, n], name='s_init_c')
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
    else:
        s_state_h_ub = tvm.compute(shape_h,
                                   lambda _, i, j, m, n: s_state_h_gm_last[
                                       0, i, j, m, n], name="s_state_h_ub")
        s_state_c_ub = tvm.compute(shape_i,
                                   lambda _, i, j, m, n: s_state_c_gm_last[
                                       0, i, j, m, n], name="s_state_c_ub")

    # handle cont mul h  caffe
    tmp_shape = [1, 1, (seq_length.shape[1] + 15) // 16, 16, 1]
    tensor_seq_length_ub = tvm.compute(
        tmp_shape, lambda i, j, k, m, n: seq_length[0, k * 16 + m], name='tensor_seq_length_ub'
        )
    tensor_seq_length_bc_ub = broadcast(tensor_seq_length_ub, shape_h)
    s_state_h_mul_cont_ub_tmp = vmul(s_state_h_ub, tensor_seq_length_bc_ub)
    s_state_h_mul_cont_ub = s_state_h_mul_cont_ub_tmp
    if s_state_h_mul_cont_ub_tmp.dtype != input_x.dtype:
        s_state_h_mul_cont_ub_tmp_conv = tvm.compute(
            s_state_h_mul_cont_ub_tmp.shape,
            lambda *i: s_state_h_mul_cont_ub_tmp(*i).astype(input_x.dtype),
            name='s_state_h_mul_cont_ub_tmp_conv', tag='elewise_single_cast'
        )
        s_state_h_mul_cont_ub = s_state_h_mul_cont_ub_tmp_conv


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
                                  s_state_h_mul_cont_ub[0,
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

    c_l0c = tvm.compute(shape_c,
                        lambda t, nb_0, nb_1, mb, mp, np:
                        tvm.sum((a_l0a[t, mb, k1, mp, k0] * \
                                 b_l0b[t, k1, nb_0, nb_1, np, k0]).astype('float32'),
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
                        lambda *indices: bias_ub(*indices).astype('float32'),
                        name="bias_ub_fp32_drnn_cast",
                        tag="elewise_single_cast")
        bias_ub_mid = bias_ub_fp32
    bias_bc_ub = broadcast(bias_ub_mid, shape_c)
    c_ub_bias = vadd(c_ub, bias_bc_ub)

    # split matmul res
    i_t_index = 0
    j_t_index = 3
    f_t_index = 1
    o_t_index = 2

    j_t = tvm.compute(shape_i,
                    lambda t, i, j, m, n: c_ub_bias(t, j_t_index, i, j, m, n),
                    name="j_t",
                    tag="split_com")
    if has_static:
        i_t = tvm.compute(shape_i,
                    lambda t, i, j, m, n: c_ub_bias(t, i_t_index, i, j, m, n),
                    name="i_t",
                    tag="split_com")
        f_t = tvm.compute(shape_i,
                    lambda t, i, j, m, n: c_ub_bias(t, f_t_index, i, j, m, n),
                    name="f_t",
                    tag="split_com")
        o_t = tvm.compute(shape_i,
                    lambda t, i, j, m, n: c_ub_bias(t, o_t_index, i, j, m, n),
                    name="o_t",
                    tag="split_com")

        output_dim = shape_i[1]
        tensor_static_it_ub_fp16 = tvm.compute(shape_i, lambda _, j, k, n, m: static(j, k, n, m),
         name="tensor_static_it_ub_fp16")
        tensor_static_ft_ub_fp16 = tvm.compute(shape_i, lambda _, j, k, n, m: static(j, k + output_dim, n, m),
         name="tensor_static_ft_ub_fp16")
        tensor_static_ot_ub_fp16 = tvm.compute(shape_i, lambda _, j, k, n, m: static(j, k + 2 * output_dim, n, m),
         name="tensor_static_ot_ub_fp16")
        tensor_static_jt_ub_fp16 = tvm.compute(shape_i, lambda _, j, k, n, m: static(j, k + 3 * output_dim, n, m),
         name="tensor_static_jt_ub_fp16")

        tensor_static_it_ub = tensor_static_it_ub_fp16
        tensor_static_ft_ub = tensor_static_ft_ub_fp16
        tensor_static_ot_ub = tensor_static_ot_ub_fp16
        tensor_static_jt_ub = tensor_static_jt_ub_fp16
        if fp16_input_output:
            tensor_static_it_ub_fp32 = tvm.compute(
                shape_i, lambda *indices: tensor_static_it_ub_fp16(*indices).astype('float32'),
                name="tensor_static_it_ub_fp32_drnn_cast", tag="elewise_single_cast"
            )
            tensor_static_it_ub = tensor_static_it_ub_fp32
            tensor_static_ft_ub_fp32 = tvm.compute(
                shape_i, lambda *indices: tensor_static_ft_ub_fp16(*indices).astype('float32'),
                name="tensor_static_ft_ub_fp32_drnn_cast", tag="elewise_single_cast"
            )
            tensor_static_ft_ub = tensor_static_ft_ub_fp32
            tensor_static_ot_ub_fp32 = tvm.compute(
                shape_i, lambda *indices: tensor_static_ot_ub_fp16(*indices).astype('float32'),
                name="tensor_static_ot_ub_fp32_drnn_cast", tag="elewise_single_cast"
            )
            tensor_static_ot_ub = tensor_static_ot_ub_fp32
            tensor_static_jt_ub_fp32 = tvm.compute(
                shape_i, lambda *indices: tensor_static_jt_ub_fp16(*indices).astype('float32'),
                name="tensor_static_jt_ub_fp32_drnn_cast", tag="elewise_single_cast"
            )
            tensor_static_jt_ub = tensor_static_jt_ub_fp32

        it_add_static = tvm.compute(
            shape_i, lambda t, i, j, m, n: i_t(t, i, j, m, n) + tensor_static_it_ub(t, i, j, m, n),
            name='it_add_static', tag='elewise_binary_add'
        )
        ft_add_static = tvm.compute(
            shape_i, lambda t, i, j, m, n: f_t(t, i, j, m, n) + tensor_static_ft_ub(t, i, j, m, n),
            name='ft_add_static', tag='elewise_binary_add'
        )
        ot_add_static = tvm.compute(
            shape_i, lambda t, i, j, m, n: o_t(t, i, j, m, n) + tensor_static_ot_ub(t, i, j, m, n),
            name='ot_add_static', tag='elewise_binary_add'
        )
        jt_add_static = tvm.compute(
            shape_i, lambda t, i, j, m, n: j_t(t, i, j, m, n) + tensor_static_jt_ub(t, i, j, m, n),
            name='jt_add_static', tag='elewise_binary_add'
        )
        f_t_sigmoid = sigmoid_compute(ft_add_static)
        i_t_sigmoid = sigmoid_compute(it_add_static)
        o_t_sigmoid = sigmoid_compute(ot_add_static)
        j_t_tanh = tanh_compute(jt_add_static)
    else:
        shape_fio = (t_size, 3, hidden_size, m_size, 16, 16)
        f_i_o = tvm.compute(shape_fio, lambda t, x, i, j, m, n: c_ub_bias(t, x, i, j, m, n),
         name='f_i_o', tag="split_com")
        f_i_o_sigmoid = sigmoid_compute(f_i_o)
        f_t_sigmoid = tvm.compute(shape_i,
                    lambda t, i, j, m, n: f_i_o_sigmoid(t, f_t_index, i, j, m, n),
                    name="f_t_sigmoid", tag="split_com")
        i_t_sigmoid = tvm.compute(shape_i,
                lambda t, i, j, m, n: f_i_o_sigmoid(t, i_t_index, i, j, m, n),
                name="i_t_sigmoid", tag="split_com")
        o_t_sigmoid = tvm.compute(shape_i,
                lambda t, i, j, m, n: f_i_o_sigmoid(t, o_t_index, i, j, m, n),
                name="o_t_sigmoid", tag="split_com")
        j_t_tanh = tanh_compute_high_precision(j_t)

    if ''.join((str(i) for i in shape_h)) == ''.join((str(i) for i in shape_i)):
        tensor_cont_ub = tensor_seq_length_bc_ub
    else:
        tensor_cont_ub = tensor_seq_length_ub
    tensor_seq_length_ub_conv = tensor_cont_ub
    if tensor_cont_ub.dtype != f_t_sigmoid.dtype:
        tensor_seq_length_ub_conv = tvm.compute(
            tensor_cont_ub.shape, lambda *i: tensor_cont_ub(*i).astype(f_t_sigmoid.dtype),
            name='tensor_seq_length_ub_conv', tag='elewise_single_cast'
            )
    tensor_seq_length_ub_bc_conv = tensor_seq_length_ub_conv
    if ''.join((str(i) for i in shape_h)) != ''.join((str(i) for i in shape_i)):
        tensor_seq_length_ub_bc_conv = broadcast(tensor_seq_length_ub_conv, shape_i)

    f_t_sigmoid_mul_cont = vmul(f_t_sigmoid, tensor_seq_length_ub_bc_conv)
    f_t_sigmoid_ub = f_t_sigmoid_mul_cont
    i_t_sigmoid_ub = i_t_sigmoid
    o_t_sigmoid_ub = o_t_sigmoid
    j_t_tanh_ub = j_t_tanh

    # auto cast support both fp16 fp32
    c_t_tmp1 = vmul(s_state_c_ub, f_t_sigmoid_ub)
    c_t_tmp2 = vmul(j_t_tanh_ub, i_t_sigmoid_ub)
    update_c = vadd(c_t_tmp1, c_t_tmp2)

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
        if need_output_last:
            last_update_c_gm = tvm.compute(shape_i,
                                        lambda *indices: update_c_fp16_back(*indices),
                                        name="last_update_c_gm",
                                        tag="ub_to_out")
            last_update_c_back = tvm.compute(shape_i,
                                        lambda *indices: last_update_c_gm(*indices),
                                        name="last_update_c_back",
                                        tag="out_to_ub")
            update_c_fp16_back_fp32 = tvm.compute(shape_i,
                                            lambda *indices: last_update_c_back(*indices).astype('float32'),
                                            name="update_c_fp16_back_fp32_drnn_cast",
                                            tag="elewise_single_cast")
        else:
            update_c_fp16_back_fp32 = tvm.compute(shape_i,
                                            lambda *indices: update_c_fp16_back(*indices).astype('float32'),
                                            name="update_c_fp16_back_fp32_drnn_cast",
                                            tag="elewise_single_cast")
        c_t_tanh = tanh_compute_high_precision(update_c_fp16_back_fp32)
    else:
        update_c_fp32_back = tvm.compute(shape_i,
                                        lambda *indices: update_c_gm(*indices),
                                        name="update_c_fp32_back",
                                        tag="out_to_ub")
        if need_output_last:
            last_update_c_gm = tvm.compute(shape_i,
                                        lambda *indices: update_c_fp32_back(*indices),
                                        name="last_update_c_gm",
                                        tag="ub_to_out")
            last_update_c_back = tvm.compute(shape_i,
                                        lambda *indices: last_update_c_gm(*indices),
                                        name="last_update_c_back",
                                        tag="out_to_ub")
            c_t_tanh = tanh_compute_high_precision(last_update_c_back)
        else:
            c_t_tanh = tanh_compute_high_precision(update_c_fp32_back)

    c_t_tanh_ub = c_t_tanh
    update_h = vmul(c_t_tanh_ub, o_t_sigmoid_ub)

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
        if need_output_last:
            last_update_h_gm = tvm.compute(shape_i,
                                    lambda *indices: update_h_gm_as_y_back(*indices),
                                    name="last_update_h_gm",
                                    tag="ub_to_out")
            last_update_h_back = tvm.compute(shape_i,
                                    lambda *indices: last_update_h_gm(*indices),
                                    name="last_update_h_back",
                                    tag="out_to_ub")
            update_h_gm = tvm.compute(shape_i,
                                    lambda *indices: last_update_h_back(*indices),
                                    name="update_h_gm",
                                    tag="ub_to_out")
        else:
            update_h_gm = tvm.compute(shape_i,
                                    lambda *indices: update_h_gm_as_y_back(*indices),
                                    name="update_h_gm",
                                    tag="ub_to_out")
    else:
        update_h_gm_as_y = tvm.compute(shape_i,
                                    lambda *indices: update_h(*indices),
                                    name="update_h_gm_as_y",
                                    tag="ub_to_out")
        update_h_gm_as_y_back = tvm.compute(shape_i,
                                        lambda *indices: update_h_gm_as_y(*indices),
                                        name="update_h_gm_as_y_back",
                                        tag="out_to_ub")
        update_h_fp16_cast = tvm.compute(shape_i,
                                        lambda *indices: update_h_gm_as_y_back(*indices).astype('float16'),
                                        name="update_h_fp16_cast_drnn_cast",
                                        tag="elewise_single_cast")
        if need_output_last:
            last_update_h_gm = tvm.compute(shape_i,
                                    lambda *indices: update_h_fp16_cast(*indices),
                                    name="last_update_h_gm",
                                    tag="ub_to_out")
            last_update_h_back = tvm.compute(shape_i,
                                    lambda *indices: last_update_h_gm(*indices),
                                    name="last_update_h_back",
                                    tag="out_to_ub")
            update_h_gm = tvm.compute(shape_i,
                                    lambda *indices: last_update_h_back(*indices),
                                    name="update_h_gm",
                                    tag="ub_to_out")

        else:
            update_h_gm = tvm.compute(shape_i,
                                    lambda *indices: update_h_fp16_cast(*indices),
                                    name="update_h_gm",
                                    tag="ub_to_out")

    #end compute
    return_list=[update_h_gm, update_c_gm, update_h_gm_as_y]
    if need_output_last:
        return_list.extend([last_update_h_gm, last_update_c_gm])
    return_list, s = rl_bank.tik_dsl_bank_proc(return_list, sync_tensor = sync0)
    if s is not None:
        return return_list, s

    bank_manager.update_bank_hit_info(True)
    #schedule
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
                        if in_tensor.name in ["s_state_h_ub", "s_state_c_ub"]:
                            continue
                        if in_tensor not in tensor_list:
                            tensor_list.append(in_tensor)

    elewise_tensors = []
    gen_reversed_subgraph_list(update_h_gm, elewise_tensors)

    barrier_tensor = c_ub_bias
    elewise_before_barrier_tensors = [bias_bc_ub]

    #set scope
    s[a_l1].set_scope(scope_cbuf)
    s[b_l1].set_scope(scope_cbuf)
    s[a_l0a].set_scope(scope_ca)
    s[b_l0b].set_scope(scope_cb)
    s[c_l0c].set_scope(scope_cc)
    s[c_ub].set_scope(scope_ubuf)
    s[bias_ub].set_scope(scope_ubuf)
    s[bias_bc_ub].set_scope(scope_ubuf)
    s[tensor_seq_length_ub_bc_conv].set_scope(scope_ubuf)
    s[tensor_seq_length_bc_ub].set_scope(scope_ubuf)
    s[s_state_h_ub].set_scope(scope_ubuf)
    s[tensor_seq_length_ub].set_scope(scope_ubuf)
    if tensor_seq_length_ub.dtype != f_t_sigmoid.dtype:
        s[tensor_seq_length_ub_conv].set_scope(scope_ubuf)
    s[s_state_c_ub].set_scope(scope_ubuf)

    s[a_ub].set_scope(scope_ubuf)
    if has_static:
        s[tensor_static_it_ub_fp16].set_scope(scope_ubuf)
        s[tensor_static_ft_ub_fp16].set_scope(scope_ubuf)
        s[tensor_static_ot_ub_fp16].set_scope(scope_ubuf)
        s[tensor_static_jt_ub_fp16].set_scope(scope_ubuf)
        if fp16_input_output:
            s[tensor_static_it_ub_fp32].set_scope(scope_ubuf)
            s[tensor_static_ft_ub_fp32].set_scope(scope_ubuf)
            s[tensor_static_ot_ub_fp32].set_scope(scope_ubuf)
            s[tensor_static_jt_ub_fp32].set_scope(scope_ubuf)

    if need_output_last:
        s[last_update_c_back].set_scope(scope_ubuf)
        s[last_update_h_back].set_scope(scope_ubuf)

    if fp16_input_output:
        s[bias_ub_fp32].set_scope(scope_ubuf)

    for tensor in elewise_tensors:
        s[tensor].set_scope(scope_ubuf)
    #fp16 in
    if bias_dtype == 'float16':
        s[update_c_fp16].set_scope(scope_ubuf)
        s[update_c_fp16_back].set_scope(scope_ubuf)
        s[update_c_fp16_back_fp32].set_scope(scope_ubuf)
        s[update_h_fp16].set_scope(scope_ubuf)
    else:
        s[update_c_fp32_back].set_scope(scope_ubuf)
        s[update_h_fp16_cast].set_scope(scope_ubuf)

    s[update_h_gm_as_y_back].set_scope(scope_ubuf)

    #compute inline
    if has_static:
        compute_inline_tensors = [j_t, i_t, o_t, f_t, it_add_static, ft_add_static, ot_add_static, jt_add_static]
    else:
        compute_inline_tensors = [j_t, f_i_o, i_t_sigmoid, f_t_sigmoid, o_t_sigmoid]
    for tensor in compute_inline_tensors:
        s[tensor].compute_inline()

    #matmul tiling
    factor_l1_m, factor_l1_n, factor_l1_k, \
    factor_l0_m, factor_l0_n, factor_l0_k = 1, 1, 12, 1, 1, 12

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
    s[a_l0a].compute_at(s[c_l0c], l0_k_outer)
    s[b_l0b].compute_at(s[c_l0c], l0_k_outer)
    s[a_l1].compute_at(s[c_l0c], l1_k_outer)
    s[b_l1].compute_at(s[c_l0c], l1_k_outer)

    ub_n_outer, ub_n_inner = s[c_ub].split(c_ub.op.axis[2], factor=factor_l1_n)

    ub_m_outer, ub_m_inner = s[c_ub].split(c_ub.op.axis[3], factor=factor_l1_m)
    s[c_ub].reorder(ub_m_outer, ub_n_outer, c_ub.op.axis[1],
                    ub_n_inner, ub_m_inner, c_ub.op.axis[4],
                    c_ub.op.axis[5])

    s[c_l0c].compute_at(s[c_ub], ub_n_outer)

    #elewise compute_at
    barrier_outer, barrier_inner = \
        s[barrier_tensor].split(barrier_tensor.op.axis[2], factor=factor_l1_n)
    barrier_m_outer, barrier_m_inner = \
        s[barrier_tensor].split(barrier_tensor.op.axis[3], factor=factor_l1_m)
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
        s[update_h_gm].split(update_h_gm.op.axis[0 + 1], factor=factor_l1_n)
    vn_m_outer, vn_m_inner = \
        s[update_h_gm].split(update_h_gm.op.axis[0 + 2], factor=factor_l1_m)

    second_split_factor = (hidden_size // factor_l1_n) // 1

    vn_o_outer, vn_o_inner = \
        s[update_h_gm].split(vn_outer,
                                factor=second_split_factor)
    s[update_h_gm].reorder(update_h_gm.op.axis[0], vn_m_outer,
                            vn_o_outer, vn_o_inner, vn_inner,
                            vn_m_inner, update_h_gm.op.axis[3],
                            update_h_gm.op.axis[4])

    s[s_state_c_ub].compute_at(s[update_h_gm], vn_o_inner)
    s[s_state_h_ub].compute_at(s[update_h_gm], vn_o_inner)
    s[tensor_seq_length_ub_bc_conv].compute_at(s[update_h_gm], vn_o_inner)
    s[tensor_seq_length_ub].compute_at(s[update_h_gm], vn_o_inner)
    s[tensor_seq_length_bc_ub].compute_at(s[update_h_gm], vn_o_inner)
    s[s_state_h_mul_cont_ub_tmp].compute_at(s[update_h_gm], vn_o_inner)
    s[barrier_tensor].compute_at(s[update_h_gm], vn_o_inner)

    for tensor in elewise_tensors:
        if tensor not in elewise_before_barrier_tensors:
            s[tensor].compute_at(s[update_h_gm], vn_o_inner)

    s[update_c_gm].compute_at(s[update_h_gm], vn_o_inner)

    #fp16 in
    if bias_dtype == 'float16':
        s[update_c_fp16].compute_at(s[update_h_gm], vn_o_inner)
        s[update_c_fp16_back].compute_at(s[update_h_gm], vn_o_inner)
        s[update_c_fp16_back_fp32].compute_at(s[update_h_gm], vn_o_inner)
        s[update_h_fp16].compute_at(s[update_h_gm], vn_o_inner)
    else:
        s[update_c_fp32_back].compute_at(s[update_h_gm], vn_o_inner)
        s[update_c].compute_at(s[update_h_gm], vn_o_inner)
        s[update_h_fp16_cast].compute_at(s[update_h_gm], vn_o_inner)


    s[update_h_gm_as_y].compute_at(s[update_h_gm], vn_o_inner)
    s[update_h_gm_as_y_back].compute_at(s[update_h_gm], vn_o_inner)

    if bias_dtype == 'float16':
        s[update_c].reused_by(update_c_fp16_back_fp32)
        s[update_c_fp16_back_fp32].reused_by(reuse_data=True)
        s[update_h_fp16].reused_by(update_h_gm_as_y_back)
        s[update_h_gm_as_y_back].reused_by(reuse_data=True)
        if need_output_last:
            s[update_c_fp16].reused_by(last_update_c_back)
            s[update_c_fp16].reused_by(update_c_fp16_back)
            s[update_c_fp16_back].reused_by(reuse_data=True)
            s[last_update_c_back].reused_by(reuse_data=True)
            s[update_h_fp16].reused_by(last_update_h_back)
            s[last_update_h_back].reused_by(reuse_data=True)
    else:
        s[update_c].reused_by(update_c_fp32_back)
        s[update_c_fp32_back].reused_by(reuse_data=True)
        s[update_h].reused_by(update_h_gm_as_y_back)
        s[update_h_gm_as_y_back].reused_by(reuse_data=True)
        if need_output_last:
            s[update_c].reused_by(last_update_c_back)
            s[last_update_c_back].reused_by(reuse_data=True)
            s[update_h].reused_by(last_update_h_back)
            s[last_update_h_back].reused_by(reuse_data=True)

    if need_output_last:
        s[last_update_c_gm].compute_at(s[update_h_gm], vn_o_inner)
        s[last_update_h_gm].compute_at(s[update_h_gm], vn_o_inner)
        s[last_update_c_back].compute_at(s[update_h_gm], vn_o_inner)
        s[last_update_h_back].compute_at(s[update_h_gm], vn_o_inner)


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

    s[tensor_seq_length_ub].emit_insn(tensor_seq_length_ub.op.axis[0], 'dma_copy')
    s[a_ub].emit_insn(a_ub.op.axis[0], 'dma_copy')

    if fp16_input_output:
        s[bias_ub_fp32].emit_insn(bias_ub_fp32.op.axis[0], 'vector_conv')

    if has_static:
        s[tensor_static_it_ub_fp16].emit_insn(tensor_static_it_ub_fp16.op.axis[3], 'dma_copy')
        s[tensor_static_ft_ub_fp16].emit_insn(tensor_static_ft_ub_fp16.op.axis[3], 'dma_copy')
        s[tensor_static_ot_ub_fp16].emit_insn(tensor_static_ot_ub_fp16.op.axis[3], 'dma_copy')
        s[tensor_static_jt_ub_fp16].emit_insn(tensor_static_jt_ub_fp16.op.axis[3], 'dma_copy')
        s[tensor_static_it_ub_fp16].compute_at(s[update_h_gm], vn_o_inner)
        s[tensor_static_ft_ub_fp16].compute_at(s[update_h_gm], vn_o_inner)
        s[tensor_static_ot_ub_fp16].compute_at(s[update_h_gm], vn_o_inner)
        s[tensor_static_jt_ub_fp16].compute_at(s[update_h_gm], vn_o_inner)
        if fp16_input_output:
            s[tensor_static_it_ub_fp32].emit_insn(tensor_static_it_ub_fp32.op.axis[3], 'vector_conv')
            s[tensor_static_ft_ub_fp32].emit_insn(tensor_static_ft_ub_fp32.op.axis[3], 'vector_conv')
            s[tensor_static_ot_ub_fp32].emit_insn(tensor_static_ot_ub_fp32.op.axis[3], 'vector_conv')
            s[tensor_static_jt_ub_fp32].emit_insn(tensor_static_jt_ub_fp32.op.axis[3], 'vector_conv')
            s[tensor_static_it_ub_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[tensor_static_ft_ub_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[tensor_static_ot_ub_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[tensor_static_jt_ub_fp32].compute_at(s[update_h_gm], vn_o_inner)

    mad_dict = {"mad_pattern":0, "k_outer":[l1_k_outer, l0_k_outer]}
    s[c_l0c].emit_insn(l0_n_inner, 'mad', mad_dict)
    s[c_ub].emit_insn(ub_n_inner, 'dma_copy')

    s[bias_bc_ub].emit_insn(bias_bc_ub.op.axis[0], 'unified_broadcast')
    s[tensor_seq_length_bc_ub].emit_insn(tensor_seq_length_bc_ub.op.axis[0], 'unified_broadcast')
    if ''.join((str(i) for i in shape_h)) != ''.join((str(i) for i in shape_i)):
        s[tensor_seq_length_ub_bc_conv].emit_insn(tensor_seq_length_ub_bc_conv.op.axis[0], 'unified_broadcast')

    if is_first_round:
        if is_global_init:
            s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'dma_copy')
            s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'dma_copy')
        else:
            s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'broadcast')
            s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'broadcast')
    else:
        s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'dma_copy')
        s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'dma_copy')

    s[barrier_tensor].emit_insn(barrier_tensor.op.axis[1], 'vector_add')

    for tensor in elewise_tensors:
        if tensor.op.name == 's_state_h_mul_cont_ub' or tensor.op.name == 'f_t_sigmoid_mul_cont':
            #continue
            s[tensor].reorder(tensor.op.axis[2], tensor.op.axis[3],
                                tensor.op.axis[1], tensor.op.axis[0], tensor.op.axis[4])
        if tensor != barrier_tensor:
            insn = get_emit_insn_map(tensor)
            s[tensor].emit_insn(tensor.op.axis[0], insn)

    s[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')
    if need_output_last:
        s[last_update_c_gm].emit_insn(s[last_update_c_gm].op.axis[1], 'dma_copy')
        s[last_update_h_gm].emit_insn(s[last_update_h_gm].op.axis[1], 'dma_copy')

    s[update_c_gm].emit_insn(s[update_c_gm].op.axis[1], 'dma_copy')
    s[update_h_gm].emit_insn(s[update_h_gm].op.axis[3], 'dma_copy')

    block = tvm.thread_axis('blockIdx.x')
    s[update_h_gm].bind(vn_o_outer, block)
    s[update_h_gm].wait_block_sync(axis=vn_m_outer, tensor=sync0[0], bottom=True)
    s[update_h_gm].set_block_sync(axis=vn_m_outer, tensor=sync0[0], bottom=True)
    if need_output_last:
        s[last_update_c_back].emit_insn(last_update_c_back.op.axis[0], 'phony_insn')
        s[last_update_h_back].emit_insn(last_update_h_back.op.axis[0], 'phony_insn')

    # fp16 in
    if bias_dtype == 'float16':
        s[update_c_fp16].emit_insn(update_c_fp16.op.axis[0], 'vector_conv')
        s[update_c_fp16_back].emit_insn(update_c_fp16_back.op.axis[0], 'phony_insn')
        s[update_c_fp16_back_fp32].emit_insn(update_c_fp16_back_fp32.op.axis[0], 'phony_insn')
        s[update_h_fp16].emit_insn(update_h_fp16.op.axis[0], 'vector_conv')
    else:
        s[update_c_fp32_back].emit_insn(update_c_fp32_back.op.axis[0], 'phony_insn')
        s[update_h_fp16_cast].emit_insn(update_h_fp16_cast.op.axis[0], 'vector_conv')

    s[update_h_gm_as_y].emit_insn(update_h_gm_as_y.op.axis[0], 'dma_copy')
    s[update_h_gm_as_y_back].emit_insn(update_h_gm_as_y_back.op.axis[0], 'phony_insn')

    return return_list, s
