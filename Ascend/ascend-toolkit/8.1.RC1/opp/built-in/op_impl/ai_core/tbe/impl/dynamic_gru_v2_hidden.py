#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

gru v2 hidden
"""
# 'pylint: disable=too-many-lines
import operator
import math

from impl.dynamic_gru_v2 import check_gru_v2_attr
from impl.dynamic_gru_v2 import ReuseType
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import rl_bank
from impl.util.platform_adapter import bank_manager


def _sigmoid_compute(input_x):
    """
    calculating sigmoid
    """
    data_input = input_x
    dtype = input_x.dtype
    exp_support = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
    mul_support = tbe_platform.api_check_support("tbe.dsl.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden",
                                                          "vmuls should support float32",
                                                          "mul_support", str(mul_support))

    const_num_neg_one = tvm.const(-1, dtype=dtype)
    const_num_one = tvm.const(1, dtype=dtype)
    tmp_negative = tbe.vmuls(data_input, const_num_neg_one)
    if dtype == "float32" and not exp_support:
        tmp_negative = tbe.cast_to(tmp_negative, "float16")
    tmp_exp = tbe.vexp(tmp_negative)
    if dtype == "float32" and not exp_support:
        tmp_exp = tbe.cast_to(tmp_exp, "float32")
    tmp_sum = tbe.vadds(tmp_exp, const_num_one)
    if dtype == "float32":
        inp_shape = tmp_sum.shape
        tensor_one = tbe.broadcast(tvm.const(1, dtype), inp_shape)
        res = tbe.vdiv(tensor_one, tmp_sum)
    else:
        res = tbe.vrec(tmp_sum)

    return res


def _tanh_compute(input_x):
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

    if input_dtype == "float16" and tbe_platform.api_check_support("vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True
        const_dtype = "float32"

    input_abs = tbe.vabs(input_x)
    power_val = tbe.vmuls(input_abs, tvm.const(-2, const_dtype))
    power_val = tbe.cast_to(power_val, "float16")
    exp_val = tbe.vexp(power_val)
    exp_val = tbe.cast_to(exp_val, "float32")

    up_val_tmp = tbe.vmul(exp_val, input_x)
    up_val = tbe.vsub(input_x, up_val_tmp)

    input_x_tmp = tbe.vadds(input_abs, min_fp_data)
    down_val_tmp = tbe.vadds(exp_val, tvm.const(1, const_dtype))
    down_val = tbe.vmul(down_val_tmp, input_x_tmp)

    res = tbe.vdiv(up_val, down_val)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


def _get_emit_insn_map(tensor):
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
def _get_tiling(hidden_size):
    """
    get tiling
    :return:
    """
    if hidden_size * hidden_size <= 128:
        return 1, hidden_size, hidden_size, 1, hidden_size, hidden_size

    n_cut = 256 // hidden_size if hidden_size <= 256 else 1
    while hidden_size % n_cut != 0 and n_cut != 1:
        n_cut -= 1
    k = 128 // n_cut if n_cut <= 128 else 1
    return 1, n_cut, k, 1, n_cut, k


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name
def _check_dtype(x_weight_input, weight_hidden, bias_hidden, init_h,
                 y, output_h, update, reset, new, hidden_new):
    """
    check parameters dtype
    :return:
    """
    para_check.check_dtype(x_weight_input["dtype"], ["float32", ], "x_weight_input")
    para_check.check_dtype(weight_hidden["dtype"], ["float16", ], "weight_hidden")

    bias_dtype = y["dtype"]
    para_check.check_dtype(bias_dtype, ["float16", "float32"], "y")

    def _check_equal_bias_dtype(tensor_dict, name):
        if tensor_dict["dtype"] != bias_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal("DynamicGRUV2Hidden", "y",
                                                                  name, bias_dtype, tensor_dict["dtype"])
    if bias_hidden is not None:
        _check_equal_bias_dtype(bias_hidden, "bias_hidden")
    _check_equal_bias_dtype(output_h, "output_h")
    if init_h is not None:
        _check_equal_bias_dtype(init_h, "init_h")
    if update is not None:
        _check_equal_bias_dtype(update, "update")
    if reset is not None:
        _check_equal_bias_dtype(reset, "reset")
    if new is not None:
        _check_equal_bias_dtype(new, "new")
    if hidden_new is not None:
        _check_equal_bias_dtype(hidden_new, "hidden_new")


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name
def _check_param(x_weight_input, weight_hidden, bias_hidden, seq_length,
                 y, output_h, update, reset, new, hidden_new):
    """
    check parameters
    :return:
    """
    # t size
    if x_weight_input["shape"][0] != output_h["shape"][0]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden",
                                                          "x_weight_input.shape[0] == output_h.shape[0]",
                                                          "output_h.shape[0]", output_h["shape"][0])

    # batch_size
    if x_weight_input["shape"][2] != output_h["shape"][2]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden",
                                                          "x_weight_input.shape[2] == output_h.shape[2]",
                                                          "output_h.shape[2]", output_h["shape"][2])

    if seq_length is not None:
        if seq_length.get("dtype").lower() == "int32":
            if (seq_length["shape"][0] + 15) // 16 != output_h["shape"][2]:
                error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden",
                                                          "(seq_length.shape[0] + 15)/16 == output_h.shape[2]",
                                                          "output_h.shape[2]", output_h["shape"][2])
        else:
            if seq_length["shape"] != output_h["shape"]:
                error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden",
                                                           "seq_length.shape == output_h.shape",
                                                           "output_h.shape", output_h["shape"])

    # k_size
    if weight_hidden["shape"][0] != output_h["shape"][1]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden",
                                                          "weight_hidden.shape[0] == output_h.shape[1]",
                                                          "weight_hidden.shape[0]", weight_hidden["shape"][0])

    # hidden_size
    if weight_hidden["shape"][1] != 3 * output_h["shape"][1]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden",
                                                          "weight_hidden.shape[1] == 3 * output_h.shape[1]",
                                                          "weight_hidden.shape[1]", weight_hidden["shape"][1])

    if bias_hidden is not None and (bias_hidden["shape"][0] + 15) // 16 != weight_hidden["shape"][1]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden",
                                                          "(bias_hidden.shape[0] + 15) // 16 == weight_hidden.shape[1]",
                                                          "bias_hidden.shape[0]", bias_hidden["shape"][0])

    # check output
    if not operator.eq(output_h["shape"], y["shape"]):
        error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden", "y.shape == output_h.shape",
                                                          "y.shape", str(y["shape"]))

    if update is not None and not operator.eq(output_h["shape"], update["shape"]):
        error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden", "update.shape == output_h.shape",
                                                          "update.shape", str(update["shape"]))

    if reset is not None and not operator.eq(output_h["shape"], reset["shape"]):
        error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden", "reset.shape == output_h.shape",
                                                          "reset.shape", str(reset["shape"]))

    if new is not None and not operator.eq(output_h["shape"], new["shape"]):
        error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden", "new.shape == output_h.shape",
                                                          "new.shape", str(new["shape"]))

    if hidden_new is not None and not operator.eq(output_h["shape"], hidden_new["shape"]):
        error_manager_vector.raise_err_check_params_rules("DynamicGRUV2Hidden", "hidden_new.shape == output_h.shape",
                                                          "hidden_new.shape", str(hidden_new["shape"]))


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-function-args,too-many-statements,unused-argument
def dynamic_gru_v2_hidden(x_weight_input, weight_hidden, bias_hidden, seq_length, init_h,
                          y, output_h, update, reset, new, hidden_new,
                          direction="UNIDIRECTIONAL", cell_depth=1, keep_prob=1.0,
                          cell_clip=-1.0, num_proj=0, time_major=True, activation="tanh",
                          gate_order="zrh", reset_after=True, is_training=True, kernel_name="dynamic_gru_v2_hidden"):
    """
    interface of op
    :return:
    """
    _check_dtype(x_weight_input, weight_hidden, bias_hidden, init_h,
                 y, output_h, update, reset, new, hidden_new)
    _check_param(x_weight_input, weight_hidden, bias_hidden, seq_length,
                 y, output_h, update, reset, new, hidden_new)
    check_gru_v2_attr("DynamicGRUV2Hidden", direction, cell_depth, keep_prob,
                      cell_clip, num_proj, time_major, activation, gate_order, reset_after)

    shape_output = y.get("shape")
    m_size = shape_output[2]
    hidden_size = shape_output[1]
    core_num = tbe_platform.get_soc_spec("CORE_NUM")
    l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
    weight_size = hidden_size * 3 * hidden_size * 16 * 16 * 2

    is_m_full_core = m_size >= core_num
    is_weight_all_in_l1 = weight_size < l1_size * 0.75
    if is_m_full_core and is_weight_all_in_l1:
        is_sync = False
        reuse_type = ReuseType.REUSE_ALL
        _solution(bias_hidden, seq_length, init_h, y, update, gate_order, kernel_name, is_sync, reuse_type, direction)
    else:
        is_w_in_l1_cut_core = weight_size / hidden_size * math.ceil(hidden_size / core_num) < l1_size * 0.75
        if is_w_in_l1_cut_core:
            is_sync = True
            reuse_type = ReuseType.REUSE_AFTERCUT
            _solution(bias_hidden, seq_length, init_h, y, update, gate_order, kernel_name, is_sync, reuse_type,
                      direction)
        else:
            is_sync = True
            reuse_type = ReuseType.NO_REUSE
            _solution(bias_hidden, seq_length, init_h, y, update, gate_order, kernel_name, is_sync, reuse_type,
                      direction)


# 'pylint: disable=invalid-name,too-many-statements
def _solution(bias_hidden, seq_length, init_h, y, update, gate_order, kernel_name, is_sync, reuse_type, direction):
    """
    solutions of op
    :return:
    """
    is_gate_output = update is not None
    is_global_init = init_h is not None
    shape_output = y.get("shape")
    t_size = shape_output[0]
    m_size = shape_output[2]
    hidden_size = shape_output[1]
    bias_dtype = y.get("dtype").lower()
    fp16_input_output = bias_dtype == "float16"
    has_bias_hidden = bias_hidden is not None
    core_num = tbe_platform.get_soc_spec("CORE_NUM")

    shape_x_weight_input = (t_size, 3, hidden_size, m_size, 16, 16)
    shape_w_2 = (1, hidden_size, 3, hidden_size, 16, 16)
    shape_h = (t_size, hidden_size, m_size, 16, 16)
    shape_bias = (1, 3, hidden_size, 1, 1, 16)
    shape_h_init = (1, hidden_size, m_size, 16, 16)
    shape_sync = (core_num * 4,)

    tik_instance = tik.Tik(tik.Dprofile())
    x_weight_input = tik_instance.Tensor(shape=shape_x_weight_input, dtype="float32",
                                         scope=tik.scope_gm, name="x_weight_input")
    weight2 = tik_instance.Tensor(shape=shape_w_2, dtype="float16", scope=tik.scope_gm, name="weight2")
    if has_bias_hidden:
        bias2 = tik_instance.Tensor(shape=shape_bias, dtype=bias_dtype, scope=tik.scope_gm, name="bias2")
    else:
        bias2 = None

    is_using_seq_mask = False
    is_valid_mask = False
    if seq_length is not None:
        is_using_seq_mask = True
        if seq_length.get("dtype").lower() == "int32":
            seq_mask_gm = tik_instance.Tensor(shape=seq_length.get("shape"), scope=tik.scope_gm,
                                              dtype="int32", name='seq_mask_gm')
        else:
            is_valid_mask = True
            seq_mask_gm = tik_instance.Tensor(shape=seq_length.get("shape"), scope=tik.scope_gm,
                                          dtype="float16", name='seq_mask_gm')
    else:
        seq_mask_gm = None

    if is_global_init:
        s_init_h_gm = tik_instance.Tensor(shape=shape_h_init, dtype=bias_dtype, scope=tik.scope_gm, name="s_init_h_gm")
    update_h_gm = tik_instance.Tensor(shape=shape_h, dtype=bias_dtype, scope=tik.scope_gm, name="update_h_gm")
    update_y_gm = tik_instance.Tensor(shape=shape_h, dtype=bias_dtype, scope=tik.scope_gm, name="update_y_gm")
    if is_gate_output:
        i_t_gm = tik_instance.Tensor(shape=shape_h, dtype=bias_dtype, scope=tik.scope_gm, name="i_t_gm")
        r_t_gm = tik_instance.Tensor(shape=shape_h, dtype=bias_dtype, scope=tik.scope_gm, name="r_t_gm")
        n_t_gm = tik_instance.Tensor(shape=shape_h, dtype=bias_dtype, scope=tik.scope_gm, name="n_t_gm")
        hn_t_gm = tik_instance.Tensor(shape=shape_h, dtype=bias_dtype, scope=tik.scope_gm, name="hn_t_gm")
    sync = tik_instance.Tensor(shape=shape_sync, dtype="int64", scope=tik.scope_gm, name='sync',
                               is_workspace=True, is_atomic_add=True)

    build_input_list = [x_weight_input, weight2]
    if has_bias_hidden:
        build_input_list.append(bias2)

    if is_using_seq_mask:
        build_input_list.append(seq_mask_gm)

    if is_global_init:
        build_input_list.append(s_init_h_gm)
    build_output_list = [update_y_gm, update_h_gm]
    if is_gate_output:
        build_output_list.extend([i_t_gm, r_t_gm, n_t_gm, hn_t_gm])

    # for RL tune getting tik input&output tensor
    bank_manager.set_tik_tensor(build_input_list, build_output_list)
    bank_manager.init_bank_hit_info(kernel_name)

    last = 1
    sub_t = 1
    loop_t = t_size // sub_t
    with tik_instance.for_range(0, loop_t) as i:
        if direction == "REDIRECTIONAL":
            valid_loop_i = loop_t - 1 - i
        else:
            valid_loop_i = i
        x_weight_input_var = x_weight_input[valid_loop_i * sub_t: valid_loop_i * sub_t + sub_t, :, :, :, :, :]
        if is_global_init:
            s_init_h_gm_var = s_init_h_gm[:, :, :, :, :]
        else:
            s_init_h_gm_var = None
        if direction == "REDIRECTIONAL":
            last_h = update_h_gm[valid_loop_i * sub_t + last: valid_loop_i * sub_t + sub_t + last:, :, :, :, :]
        else:
            last_h = update_h_gm[valid_loop_i * sub_t - last: valid_loop_i * sub_t + sub_t - last:, :, :, :, :]
        update_h_gm_var = update_h_gm[valid_loop_i * sub_t: valid_loop_i * sub_t + sub_t, :, :, :, :]
        update_y_gm_var = update_y_gm[valid_loop_i * sub_t: valid_loop_i * sub_t + sub_t, :, :, :, :]
        if is_gate_output:
            r_t_gm_var = r_t_gm[valid_loop_i * sub_t: valid_loop_i * sub_t + sub_t, :, :, :, :]
            i_t_gm_var = i_t_gm[valid_loop_i * sub_t: valid_loop_i * sub_t + sub_t, :, :, :, :]
            n_t_gm_var = n_t_gm[valid_loop_i * sub_t: valid_loop_i * sub_t + sub_t, :, :, :, :]
            hn_t_gm_var = hn_t_gm[valid_loop_i * sub_t: valid_loop_i * sub_t + sub_t, :, :, :, :]
        else:
            r_t_gm_var = None
            i_t_gm_var = None
            n_t_gm_var = None
            hn_t_gm_var = None

        if is_valid_mask:
            seq_mask_gm_var = seq_mask_gm[valid_loop_i * sub_t: valid_loop_i * sub_t + sub_t, :, :, :, :]
        else:
            seq_mask_gm_var = None

        input_list = [x_weight_input_var, weight2, bias2, s_init_h_gm_var, last_h, sync, seq_mask_gm_var]
        if is_gate_output:
            output_list = [update_y_gm_var, update_h_gm_var, i_t_gm_var, r_t_gm_var, n_t_gm_var, hn_t_gm_var]
        else:
            output_list = [update_y_gm_var, update_h_gm_var]

        with tik_instance.if_scope(i == 0):
            is_first_round = True
            tik_instance.call_module(
                _dynamic_gru_v2_hidden_inner,
                input_list,
                output_list,
                [is_gate_output, is_first_round, is_global_init, gate_order, fp16_input_output, is_sync, reuse_type])

        with tik_instance.if_scope(i > 0):
            is_first_round = False
            tik_instance.call_module(
                _dynamic_gru_v2_hidden_inner,
                input_list,
                output_list,
                [is_gate_output, is_first_round, is_global_init, gate_order, fp16_input_output, is_sync, reuse_type])

    tik_instance.BuildCCE(kernel_name,
                          build_input_list,
                          build_output_list)


# 'pylint: disable=too-many-statements,unused-variable,unnecessary-lambda
def _dynamic_gru_v2_hidden_inner(input_list, custom_list):
    """
    inner part of tik loop
    :return:
    """
    x_weight_input = input_list[0]
    weight2 = input_list[1]
    bias2 = input_list[2]
    s_init_h_gm = input_list[3]
    s_state_h_gm_last = input_list[4]
    sync = input_list[5]
    seq_mask_gm = input_list[6]
    is_gate_output = custom_list[0]
    is_first_round = custom_list[1]
    is_global_init = custom_list[2]
    gate_order = custom_list[3]
    fp16_input_output = custom_list[4]
    is_sync = custom_list[5]
    reuse_type = custom_list[6]

    shape_x_weight_input = x_weight_input.shape
    t_size = shape_x_weight_input[0].value
    m_size = shape_x_weight_input[3].value
    hidden_size = shape_x_weight_input[2].value

    shape_b_2 = (1, hidden_size, 3, hidden_size, 16, 16)
    shape_c_2 = (1, 3, hidden_size, m_size, 16, 16)
    shape_bias = (1, 3, hidden_size, 1, 1, 16)
    shape_i = (1, hidden_size, m_size, 16, 16)
    shape_i_t = (t_size, hidden_size, m_size, 16, 16)
    k0_size = 16

    exceed_ub = False
    if not fp16_input_output and hidden_size > 160 or fp16_input_output and hidden_size > 470:
        exceed_ub = True
    if not is_global_init and hidden_size > 160:
        exceed_ub = True

    # compute
    if is_first_round and not is_global_init:
        s_state_h = tvm.compute(shape_i,
                                lambda *indices: tvm.const(0.0, dtype="float32"),
                                name="s_state_h_ign",
                                tag="broadcast")
        if not exceed_ub:
            s_state_h_fp16 = tvm.compute(shape_i,
                                         lambda *indices: s_state_h(*indices).astype("float16"),
                                         name="s_state_h_fp16_ign",
                                         tag="elewise_single_cast")
        else:
            s_state_h_fp16 = tvm.compute(shape_i,
                                         lambda *indices: tvm.const(0.0, dtype="float16"),
                                         name="s_state_h_fp16_ign",
                                         tag="broadcast")
        if seq_mask_gm is not None:
            s_state_h_ub_for_element = \
                        tvm.compute(shape_i,
                        lambda *indices: tvm.const(0.0, dtype="float32"),
                        name='s_state_h_ub_for_element',
                        tag="broadcast")
    else:
        last_h = s_init_h_gm if is_first_round else s_state_h_gm_last
        if fp16_input_output:
            s_state_h_fp16 = tvm.compute(shape_i,
                                         lambda *indices: last_h(*indices),
                                         name="s_state_h_fp16",
                                         tag="out_to_ub")
            if not exceed_ub:
                s_state_h = tvm.compute(shape_i,
                                        lambda *indices: s_state_h_fp16(*indices).astype("float32"),
                                        name="s_state_h_ign",
                                        tag="elewise_single_cast")
            else:
                s_state_h_tmp = tvm.compute(shape_i,
                                            lambda *indices: last_h(*indices),
                                            name="s_state_h_tmp",
                                            tag="out_to_ub")
                s_state_h = tvm.compute(shape_i,
                                        lambda *indices: s_state_h_tmp(*indices).astype("float32"),
                                        name="s_state_h_ign",
                                        tag="elewise_single_cast")
            if seq_mask_gm is not None:
                s_state_h_ub_for_element_fp16 = tvm.compute(shape_i,
                                                      lambda *indices: last_h(*indices),
                                                      name="s_state_h_ub_for_element_fp16",
                                                      tag="out_to_ub")
                s_state_h_ub_for_element = tvm.compute(shape_i,
                                           lambda *indices: s_state_h_ub_for_element_fp16(*indices).astype("float32"),
                                           name="s_state_h_ub_for_element",
                                           tag="elewise_single_cast")
        else:
            s_state_h = tvm.compute(shape_i,
                                    lambda *indices: last_h(*indices),
                                    name="s_state_h",
                                    tag="out_to_ub")
            if not exceed_ub:
                s_state_h_fp16 = tvm.compute(shape_i,
                                             lambda *indices: s_state_h(*indices).astype("float16"),
                                             name="s_state_h_fp16_ign",
                                             tag="elewise_single_cast")
            else:
                s_state_h_tmp = tvm.compute(shape_i,
                                            lambda *indices: last_h(*indices),
                                            name="s_state_h_tmp",
                                            tag="out_to_ub")
                s_state_h_fp16 = tvm.compute(shape_i,
                                             lambda *indices: s_state_h_tmp(*indices).astype("float16"),
                                             name="s_state_h_fp16_ign",
                                             tag="elewise_single_cast")
            if seq_mask_gm is not None:
                s_state_h_ub_for_element = tvm.compute(shape_i,
                                                      lambda *indices: last_h(*indices),
                                                      name="s_state_h_ub_for_element",
                                                      tag="out_to_ub")

    # second matmul
    # input and s_start_h is Nz, need trans to zZ, so change axis 1 and 2
    shape_a_z_bigz_2 = (1, m_size, hidden_size, 16, 16)
    a_l1_2 = tvm.compute(shape_a_z_bigz_2,
                         lambda *indice: s_state_h_fp16[indice[0], indice[2], indice[1], indice[3], indice[4]],
                         name="a_l1_2",
                         tag="out_to_l1")
    b_l1_2 = tvm.compute(shape_b_2,
                         lambda *indices: weight2(*indices),
                         name="b_l1_2",
                         tag="out_to_l1")
    a_l0a_2 = tvm.compute(shape_a_z_bigz_2, lambda *indices: a_l1_2(*indices), name="a_l0a_2", tag="l1_to_l0")
    b_l0b_2 = tvm.compute(shape_b_2, lambda *indices: b_l1_2(*indices), name="b_l0b_2", tag="l1_to_l0")
    k1_2 = tvm.reduce_axis((0, hidden_size), name="k1_2")
    k0_2 = tvm.reduce_axis((0, k0_size), name="k0_2")
    c_l0c_2 = tvm.compute(shape_c_2,
                          lambda t, nb_0, nb_1, mb, mp, np:
                          tvm.sum((a_l0a_2[t, mb, k1_2, mp, k0_2] * \
                                   b_l0b_2[t, k1_2, nb_0, nb_1, np, k0_2]) \
                                  .astype("float32"),
                                  axis=[k1_2, k0_2]),
                          name="c_l0c_2",
                          tag="matmul")
    c_ub_2 = tvm.compute(shape_c_2, lambda *indices: c_l0c_2(*indices), name="c_ub_2")
    if bias2 is not None:
        bias_ub_2 = tvm.compute(shape_bias,
                                lambda *indices: bias2(*indices),
                                name="bias_ub_2",
                                tag="out_to_ub")
        bias_ub_2_fp32 = bias_ub_2
        if fp16_input_output:
            bias_ub_2_fp32 = tvm.compute(shape_bias,
                                         lambda *indices: bias_ub_2(*indices).astype("float32"),
                                         name="bias_ub_2_fp32_ign",
                                         tag="elewise_single_cast")
        bias_bc_ub_2 = tbe.broadcast(bias_ub_2_fp32, shape_c_2)
        c_ub_bias_2 = tbe.vadd(c_ub_2, bias_bc_ub_2)
    else:
        c_ub_bias_2 = c_ub_2

    # split matmul res
    if gate_order == "zrh":
        i_t_index = 0
        r_t_index = 1
        n_t_index = 2
    else:
        r_t_index = 0
        i_t_index = 1
        n_t_index = 2
    r_t_1 = tvm.compute(shape_i,
                        lambda t, i, j, k, m: x_weight_input(t, r_t_index, i, j, k, m),
                        name="r_t_1",
                        tag="split_com")
    i_t_1 = tvm.compute(shape_i,
                        lambda t, i, j, k, m: x_weight_input(t, i_t_index, i, j, k, m),
                        name="i_t_1",
                        tag="split_com")
    n_t_1 = tvm.compute(shape_i,
                        lambda t, i, j, k, m: x_weight_input(t, n_t_index, i, j, k, m),
                        name="n_t_1",
                        tag="split_com")
    r_t_2 = tvm.compute(shape_i,
                        lambda t, i, j, k, m: c_ub_bias_2(t, r_t_index, i, j, k, m),
                        name="r_t_2",
                        tag="split_com")
    i_t_2 = tvm.compute(shape_i,
                        lambda t, i, j, k, m: c_ub_bias_2(t, i_t_index, i, j, k, m),
                        name="i_t_2",
                        tag="split_com")
    n_t_2 = tvm.compute(shape_i,
                        lambda t, i, j, k, m: c_ub_bias_2(t, n_t_index, i, j, k, m),
                        name="n_t_2",
                        tag="split_com")
    # output n_t_2
    n_t_2_mid = n_t_2
    if is_gate_output:
        if fp16_input_output:
            n_t_2_fp16 = tvm.compute(shape_i,
                                     lambda *indices: n_t_2(*indices).astype("float16"),
                                     name="n_t_2_fp16_ign",
                                     tag="elewise_single_cast")
            hn_t_gm = tvm.compute(shape_i,
                                  lambda *indices: n_t_2_fp16(*indices),
                                  name="hn_t_gm",
                                  tag="ub_to_out")
            hn_t_gm_back = tvm.compute(shape_i,
                                       lambda *indices: hn_t_gm(*indices),
                                       name="hn_t_gm_back",
                                       tag="out_to_ub")
            hn_t_gm_back_fp32 = tvm.compute(shape_i,
                                            lambda *indices: hn_t_gm_back(*indices).astype("float32"),
                                            name="hn_t_gm_back_fp32_ign",
                                            tag="elewise_single_cast")
            n_t_2_mid = hn_t_gm_back_fp32
        else:
            hn_t_gm = tvm.compute(shape_i,
                                  lambda *indices: n_t_2(*indices),
                                  name="hn_t_gm",
                                  tag="ub_to_out")
            hn_t_gm_back = tvm.compute(shape_i,
                                       lambda *indices: hn_t_gm(*indices),
                                       name="hn_t_gm_back",
                                       tag="out_to_ub")
            n_t_2_mid = hn_t_gm_back

    r_t = tbe.vadd(r_t_1, r_t_2)
    i_t = tbe.vadd(i_t_1, i_t_2)
    r_t_sigmoid = _sigmoid_compute(r_t)
    i_t_sigmoid = _sigmoid_compute(i_t)

    # output r_t_sigmoid i_t_sigmoid
    r_t_mid = r_t_sigmoid
    i_t_mid = i_t_sigmoid
    if is_gate_output:
        if fp16_input_output:
            r_t_sigmoid_fp16 = tvm.compute(shape_i,
                                           lambda *indices: r_t_sigmoid(*indices).astype("float16"),
                                           name="r_t_sigmoid_fp16_ign",
                                           tag="elewise_single_cast")
            i_t_sigmoid_fp16 = tvm.compute(shape_i,
                                           lambda *indices: i_t_sigmoid(*indices).astype("float16"),
                                           name="i_t_sigmoid_fp16_ign",
                                           tag="elewise_single_cast")

            r_t_gm = tvm.compute(shape_i,
                                 lambda *indices: r_t_sigmoid_fp16(*indices),
                                 name="r_t_gm",
                                 tag="ub_to_out")
            i_t_gm = tvm.compute(shape_i,
                                 lambda *indices: i_t_sigmoid_fp16(*indices),
                                 name="i_t_gm",
                                 tag="ub_to_out")

            r_t_gm_back = tvm.compute(shape_i,
                                      lambda *indices: r_t_gm(*indices),
                                      name="r_t_gm_back",
                                      tag="out_to_ub")
            i_t_gm_back = tvm.compute(shape_i,
                                      lambda *indices: i_t_gm(*indices),
                                      name="i_t_gm_back",
                                      tag="out_to_ub")

            r_t_gm_back_fp32 = tvm.compute(shape_i,
                                           lambda *indices: r_t_gm_back(*indices).astype("float32"),
                                           name="r_t_gm_back_fp32_ign",
                                           tag="elewise_single_cast")
            i_t_gm_back_fp32 = tvm.compute(shape_i,
                                           lambda *indices: i_t_gm_back(*indices).astype("float32"),
                                           name="i_t_gm_back_fp32_ign",
                                           tag="elewise_single_cast")

            r_t_mid = r_t_gm_back_fp32
            i_t_mid = i_t_gm_back_fp32
        else:
            r_t_gm = tvm.compute(shape_i,
                                 lambda *indices: r_t_sigmoid(*indices),
                                 name="r_t_gm",
                                 tag="ub_to_out")
            i_t_gm = tvm.compute(shape_i,
                                 lambda *indices: i_t_sigmoid(*indices),
                                 name="i_t_gm",
                                 tag="ub_to_out")

            r_t_gm_back = tvm.compute(shape_i,
                                      lambda *indices: r_t_gm(*indices),
                                      name="r_t_gm_back",
                                      tag="out_to_ub")
            i_t_gm_back = tvm.compute(shape_i,
                                      lambda *indices: i_t_gm(*indices),
                                      name="i_t_gm_back",
                                      tag="out_to_ub")

            r_t_mid = r_t_gm_back
            i_t_mid = i_t_gm_back

    r_t_h = tbe.vmul(r_t_mid, n_t_2_mid)
    n_t = tbe.vadd(n_t_1, r_t_h)
    n_t_tanh = _tanh_compute(n_t)

    # output n_t_tanh
    n_t_tanh_mid = n_t_tanh
    if is_gate_output:
        if fp16_input_output:
            n_t_tanh_fp16 = tvm.compute(shape_i,
                                        lambda *indices: n_t_tanh(*indices).astype("float16"),
                                        name="n_t_tanh_fp16_ign",
                                        tag="elewise_single_cast")
            n_t_gm = tvm.compute(shape_i,
                                 lambda *indices: n_t_tanh_fp16(*indices),
                                 name="n_t_gm",
                                 tag="ub_to_out")
            n_t_gm_back = tvm.compute(shape_i,
                                      lambda *indices: n_t_gm(*indices),
                                      name="n_t_gm_back",
                                      tag="out_to_ub")
            n_t_gm_back_fp32 = tvm.compute(shape_i,
                                           lambda *indices: n_t_gm_back(*indices).astype("float32"),
                                           name="n_t_gm_back_fp32_ign",
                                           tag="elewise_single_cast")
            n_t_tanh_mid = n_t_gm_back_fp32
        else:
            n_t_gm = tvm.compute(shape_i,
                                 lambda *indices: n_t_tanh(*indices),
                                 name="n_t_gm",
                                 tag="ub_to_out")
            n_t_gm_back = tvm.compute(shape_i,
                                      lambda *indices: n_t_gm(*indices),
                                      name="n_t_gm_back",
                                      tag="out_to_ub")
            n_t_tanh_mid = n_t_gm_back

    c_t_tmp1 = tbe.vsub(s_state_h, n_t_tanh_mid)
    c_t_tmp2 = tbe.vmul(c_t_tmp1, i_t_mid)
    update_h = tbe.vadd(c_t_tmp2, n_t_tanh_mid)
    if seq_mask_gm is not None:
        seq_mask_ub = tvm.compute(shape_i, lambda *indices: seq_mask_gm(*indices),
                                  name="seq_mask_ub")
        seq_mask_ub_fp32 = tvm.compute(shape_i,
                                    lambda *indices: seq_mask_ub(*indices).astype("float32"),
                                    name="seq_mask_ub_fp32",
                                    tag="elewise_single_cast")
        update_h_diff = tbe.vsub(update_h, s_state_h_ub_for_element)
        update_h_tmp = tbe.vmul(update_h_diff, seq_mask_ub_fp32)
        update_h = tbe.vadd(update_h_tmp, s_state_h_ub_for_element)
    update_h_ub = update_h
    if fp16_input_output:
        update_h_fp16 = tvm.compute(shape_i_t,
                                    lambda *indices: update_h(*indices).astype("float16"),
                                    name="update_h_fp16_ign",
                                    tag="elewise_single_cast")
        update_h_ub = update_h_fp16
    update_y_gm = tvm.compute(shape_i_t,
                              lambda t, i, j, k, m: update_h_ub(0, i, j, k, m),
                              name="update_y_gm",
                              tag="ub_to_out")
    update_y_gm_back = tvm.compute(shape_i_t,
                                   lambda t, i, j, k, m: update_y_gm(0, i, j, k, m),
                                   name="update_y_gm_back",
                                   tag="out_to_ub")
    update_h_gm = tvm.compute(shape_i_t,
                              lambda t, i, j, k, m: update_y_gm_back(0, i, j, k, m),
                              name="update_h_gm",
                              tag="ub_to_out")
    # end compute

    output_list = [update_y_gm, update_h_gm]
    if is_gate_output:
        output_list.append(i_t_gm)
        output_list.append(r_t_gm)
        output_list.append(n_t_gm)
        output_list.append(hn_t_gm)

    # for RL tuning
    output_list, sch_rl = rl_bank.tik_dsl_bank_proc(output_list, sync_tensor=sync)
    if sch_rl is not None:
        return output_list, sch_rl

    bank_manager.update_bank_hit_info(True)
    # schedule
    sch = tvm.create_schedule([update_h_gm.op])

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
                        if in_tensor.name.endswith("_ign"):
                            continue
                        if in_tensor not in tensor_list:
                            tensor_list.append(in_tensor)

    elewise_tensors = []
    gen_reversed_subgraph_list(update_h_gm, elewise_tensors)

    # set scope
    sch[s_state_h].set_scope(tbe_platform.scope_ubuf)
    sch[s_state_h_fp16].set_scope(tbe_platform.scope_ubuf)
    sch[a_l1_2].set_scope(tbe_platform.scope_cbuf)
    sch[b_l1_2].set_scope(tbe_platform.scope_cbuf)
    sch[a_l0a_2].set_scope(tbe_platform.scope_ca)
    sch[b_l0b_2].set_scope(tbe_platform.scope_cb)
    sch[c_l0c_2].set_scope(tbe_platform.scope_cc)
    sch[c_ub_2].set_scope(tbe_platform.scope_ubuf)
    if bias2 is not None:
        sch[bias_ub_2].set_scope(tbe_platform.scope_ubuf)
        sch[bias_bc_ub_2].set_scope(tbe_platform.scope_ubuf)
        if fp16_input_output:
            sch[bias_ub_2_fp32].set_scope(tbe_platform.scope_ubuf)
    if seq_mask_gm is not None:
        sch[s_state_h_ub_for_element].set_scope(tbe_platform.scope_ubuf)
        if (not is_first_round or is_global_init) and fp16_input_output:
            sch[s_state_h_ub_for_element_fp16].set_scope(tbe_platform.scope_ubuf)
    if is_gate_output:
        sch[r_t_gm_back].set_scope(tbe_platform.scope_ubuf)
        sch[i_t_gm_back].set_scope(tbe_platform.scope_ubuf)
        sch[n_t_gm_back].set_scope(tbe_platform.scope_ubuf)
        sch[hn_t_gm_back].set_scope(tbe_platform.scope_ubuf)
        if fp16_input_output:
            sch[r_t_sigmoid_fp16].set_scope(tbe_platform.scope_ubuf)
            sch[i_t_sigmoid_fp16].set_scope(tbe_platform.scope_ubuf)
            sch[n_t_tanh_fp16].set_scope(tbe_platform.scope_ubuf)
            sch[n_t_2_fp16].set_scope(tbe_platform.scope_ubuf)
            sch[r_t_gm_back_fp32].set_scope(tbe_platform.scope_ubuf)
            sch[i_t_gm_back_fp32].set_scope(tbe_platform.scope_ubuf)
            sch[n_t_gm_back_fp32].set_scope(tbe_platform.scope_ubuf)
            sch[hn_t_gm_back_fp32].set_scope(tbe_platform.scope_ubuf)
    sch[update_y_gm_back].set_scope(tbe_platform.scope_ubuf)
    if fp16_input_output:
        sch[update_h_fp16].set_scope(tbe_platform.scope_ubuf)
    sch[n_t_2].set_scope(tbe_platform.scope_ubuf)
    sch[r_t_1].set_scope(tbe_platform.scope_ubuf)
    sch[i_t_1].set_scope(tbe_platform.scope_ubuf)
    sch[n_t_1].set_scope(tbe_platform.scope_ubuf)

    if seq_mask_gm is not None:
        sch[seq_mask_ub].set_scope(tbe_platform.scope_ubuf)
        sch[seq_mask_ub_fp32].set_scope(tbe_platform.scope_ubuf)

    # compute inline
    compute_inline_tensors = [i_t_2, r_t_2]
    for tensor in compute_inline_tensors:
        sch[tensor].compute_inline()

    # matmul tiling
    factor_l1_m, factor_l1_n, factor_l1_k_2, factor_l0_m, factor_l0_n, factor_l0_k_2 = _get_tiling(hidden_size)

    l1_n_outer_2, l1_n_inner_2 = sch[c_l0c_2].split(c_l0c_2.op.axis[2], factor=factor_l1_n)
    l1_m_outer_2, l1_m_inner_2 = sch[c_l0c_2].split(c_l0c_2.op.axis[3], factor=factor_l1_m)
    l1_k_outer_2, l1_k_inner_2 = sch[c_l0c_2].split(c_l0c_2.op.reduce_axis[0], factor=factor_l1_k_2)
    l0_n_outer_2, l0_n_inner_2 = sch[c_l0c_2].split(l1_n_inner_2, factor=factor_l0_n)
    l0_m_outer_2, l0_m_inner_2 = sch[c_l0c_2].split(l1_m_inner_2, factor=factor_l0_m)
    l0_k_outer_2, l0_k_inner_2 = sch[c_l0c_2].split(l1_k_inner_2, factor=factor_l0_k_2)
    sch[c_l0c_2].reorder(c_l0c_2.op.axis[0],
                         l1_m_outer_2,
                         l1_n_outer_2,
                         l1_k_outer_2,
                         c_l0c_2.op.axis[1],
                         l0_n_outer_2,
                         l0_m_outer_2,
                         l0_k_outer_2,
                         l0_n_inner_2,
                         l0_m_inner_2,
                         c_l0c_2.op.axis[4],
                         c_l0c_2.op.axis[5],
                         l0_k_inner_2,
                         c_l0c_2.op.reduce_axis[1])
    sch[a_l1_2].double_buffer()
    sch[b_l1_2].double_buffer()
    sch[a_l0a_2].double_buffer()
    sch[b_l0b_2].double_buffer()
    sch[c_l0c_2].double_buffer()
    sch[a_l1_2].compute_at(sch[c_l0c_2], l1_k_outer_2)
    sch[b_l1_2].compute_at(sch[c_l0c_2], c_l0c_2.op.axis[1])
    sch[a_l0a_2].compute_at(sch[c_l0c_2], l1_k_outer_2)
    sch[b_l0b_2].compute_at(sch[c_l0c_2], l0_k_outer_2)

    update_h_gm_t_outer, update_h_gm_t_inner = sch[update_h_gm].split(update_h_gm.op.axis[0], factor=1)
    update_h_gm_m_outer, update_h_gm_m_inner = sch[update_h_gm].split(update_h_gm.op.axis[2], factor=factor_l1_m)
    update_h_gm_outer, update_h_gm_inner = sch[update_h_gm].split(update_h_gm.op.axis[1], factor=factor_l1_n)
    update_h_gm_o_outer, update_h_gm_outer = sch[update_h_gm].split(update_h_gm_outer, factor=1)
    sch[update_h_gm].reorder(update_h_gm_t_outer,
                             update_h_gm_t_inner,
                             update_h_gm_m_outer,
                             update_h_gm_m_inner,
                             update_h_gm_o_outer,
                             update_h_gm_outer,
                             update_h_gm_inner,
                             update_h_gm.op.axis[3],
                             update_h_gm.op.axis[4])
    sch[c_l0c_2].compute_at(sch[update_h_gm], update_h_gm_outer)
    sch[c_ub_2].compute_at(sch[update_h_gm], update_h_gm_outer)
    if bias2 is not None:
        sch[bias_ub_2].compute_at(sch[update_h_gm], update_h_gm_outer)
        sch[bias_bc_ub_2].compute_at(sch[update_h_gm], update_h_gm_outer)
        if fp16_input_output:
            sch[bias_ub_2_fp32].compute_at(sch[update_h_gm], update_h_gm_outer)
    sch[update_y_gm].compute_at(sch[update_h_gm], update_h_gm_outer)
    sch[update_y_gm_back].compute_at(sch[update_h_gm], update_h_gm_outer)
    if fp16_input_output:
        sch[update_h_fp16].compute_at(sch[update_h_gm], update_h_gm_outer)

    if is_gate_output:
        sch[r_t_gm].compute_at(sch[update_h_gm], update_h_gm_outer)
        sch[r_t_gm_back].compute_at(sch[update_h_gm], update_h_gm_outer)
        sch[i_t_gm].compute_at(sch[update_h_gm], update_h_gm_outer)
        sch[i_t_gm_back].compute_at(sch[update_h_gm], update_h_gm_outer)
        sch[n_t_gm].compute_at(sch[update_h_gm], update_h_gm_outer)
        sch[n_t_gm_back].compute_at(sch[update_h_gm], update_h_gm_outer)
        sch[hn_t_gm].compute_at(sch[update_h_gm], update_h_gm_outer)
        sch[hn_t_gm_back].compute_at(sch[update_h_gm], update_h_gm_outer)
        if fp16_input_output:
            sch[r_t_sigmoid_fp16].compute_at(sch[update_h_gm], update_h_gm_outer)
            sch[r_t_gm_back_fp32].compute_at(sch[update_h_gm], update_h_gm_outer)
            sch[i_t_sigmoid_fp16].compute_at(sch[update_h_gm], update_h_gm_outer)
            sch[i_t_gm_back_fp32].compute_at(sch[update_h_gm], update_h_gm_outer)
            sch[n_t_tanh_fp16].compute_at(sch[update_h_gm], update_h_gm_outer)
            sch[n_t_gm_back_fp32].compute_at(sch[update_h_gm], update_h_gm_outer)
            sch[n_t_2_fp16].compute_at(sch[update_h_gm], update_h_gm_outer)
            sch[hn_t_gm_back_fp32].compute_at(sch[update_h_gm], update_h_gm_outer)

    for tensor in elewise_tensors:
        sch[tensor].set_scope(tbe_platform.scope_ubuf)
        sch[tensor].compute_at(sch[update_h_gm], update_h_gm_outer)
        insn = _get_emit_insn_map(tensor)
        sch[tensor].emit_insn(tensor.op.axis[0], insn)

    sch[n_t_2].compute_at(sch[update_h_gm], update_h_gm_outer)
    sch[r_t_1].compute_at(sch[update_h_gm], update_h_gm_outer)
    sch[i_t_1].compute_at(sch[update_h_gm], update_h_gm_outer)
    sch[n_t_1].compute_at(sch[update_h_gm], update_h_gm_outer)

    if exceed_ub:
        sch[s_state_h].compute_at(sch[update_h_gm], update_h_gm_outer)
        sch[s_state_h_fp16].compute_at(sch[c_l0c_2], l1_k_outer_2)
    else:
        sch[s_state_h].compute_at(sch[update_h_gm], update_h_gm_m_inner)
        sch[s_state_h_fp16].compute_at(sch[update_h_gm], update_h_gm_m_inner)

    if seq_mask_gm is not None:
        sch[s_state_h_ub_for_element].compute_at(sch[update_h_gm], update_h_gm_m_inner)
        sch[seq_mask_ub].compute_at(sch[update_h_gm], update_h_gm_m_inner)
        sch[seq_mask_ub_fp32].compute_at(sch[update_h_gm], update_h_gm_m_inner)
        if (not is_first_round or is_global_init) and fp16_input_output:
            sch[s_state_h_ub_for_element_fp16].compute_at(sch[update_h_gm], update_h_gm_m_inner)

    if reuse_type == ReuseType.REUSE_ALL:
        sch[update_h_gm].bind(update_h_gm_m_outer, tvm.thread_axis("blockIdx.x"))
    else:
        core_num = tbe_platform.get_soc_spec("CORE_NUM")
        update_h_gm_m_outer_size = (m_size + factor_l1_m - 1) // factor_l1_m
        update_h_gm_o_outer_size = (hidden_size + factor_l1_n - 1) // factor_l1_n
        fused_axis_size = update_h_gm_o_outer_size * update_h_gm_m_outer_size
        while fused_axis_size % core_num != 0 and core_num != 1:
            core_num -= 1
        sch[update_h_gm].reorder(update_h_gm_t_outer,
                                 update_h_gm_t_inner,
                                 update_h_gm_m_outer,
                                 update_h_gm_o_outer,
                                 update_h_gm_m_inner,
                                 update_h_gm_outer,
                                 update_h_gm_inner,
                                 update_h_gm.op.axis[3],
                                 update_h_gm.op.axis[4])
        fused_axis = sch[update_h_gm].fuse(update_h_gm_m_outer, update_h_gm_o_outer)
        bind_axis, bind_axis_inner = sch[update_h_gm].split(fused_axis, nparts=core_num)
        sch[update_h_gm].bind(bind_axis, tvm.thread_axis("blockIdx.x"))
    if is_sync:
        sch[update_h_gm].wait_block_sync(axis=update_h_gm_t_inner, tensor=sync[0], bottom=True)
        sch[update_h_gm].set_block_sync(axis=update_h_gm_t_inner, tensor=sync[0], bottom=True)

    # emit insn
    if is_first_round and not is_global_init:
        sch[s_state_h].emit_insn(s_state_h.op.axis[0], "broadcast")
        if not exceed_ub:
            sch[s_state_h_fp16].emit_insn(s_state_h_fp16.op.axis[0], "vector_conv")
        else:
            sch[s_state_h_fp16].emit_insn(s_state_h_fp16.op.axis[0], "broadcast")
        if seq_mask_gm is not None:
            sch[s_state_h_ub_for_element].emit_insn(s_state_h_ub_for_element.op.axis[0], 'broadcast')
    else:
        if fp16_input_output:
            sch[s_state_h_fp16].emit_insn(s_state_h_fp16.op.axis[0], "dma_copy")
            sch[s_state_h].emit_insn(s_state_h.op.axis[0], "vector_conv")
            if exceed_ub:
                sch[s_state_h_tmp].set_scope(tbe_platform.scope_ubuf)
                sch[s_state_h_tmp].compute_at(sch[update_h_gm], update_h_gm_outer)
                sch[s_state_h_tmp].emit_insn(s_state_h_tmp.op.axis[0], "dma_copy")
            if seq_mask_gm is not None:
                sch[s_state_h_ub_for_element_fp16].emit_insn(s_state_h_ub_for_element_fp16.op.axis[0], 'dma_copy')
                sch[s_state_h_ub_for_element].emit_insn(s_state_h_ub_for_element.op.axis[0], 'vector_conv')
        else:
            sch[s_state_h].emit_insn(s_state_h.op.axis[0], "dma_copy")
            sch[s_state_h_fp16].emit_insn(s_state_h_fp16.op.axis[0], "vector_conv")
            if exceed_ub:
                sch[s_state_h_tmp].set_scope(tbe_platform.scope_ubuf)
                sch[s_state_h_tmp].compute_at(sch[c_l0c_2], l1_k_outer_2)
                sch[s_state_h_tmp].emit_insn(s_state_h_tmp.op.axis[0], "dma_copy")
            if seq_mask_gm is not None:
                sch[s_state_h_ub_for_element].emit_insn(s_state_h_ub_for_element.op.axis[0], 'dma_copy')

    sch[a_l1_2].emit_insn(a_l1_2.op.axis[0], "dma_copy")
    sch[b_l1_2].emit_insn(b_l1_2.op.axis[0], "dma_copy")
    sch[a_l0a_2].emit_insn(a_l0a_2.op.axis[0], "dma_copy")
    sch[b_l0b_2].emit_insn(b_l0b_2.op.axis[0], "dma_copy")
    mad_dict = {"mad_pattern": 0, "k_outer": [l1_k_outer_2, l0_k_outer_2]}
    sch[c_l0c_2].emit_insn(l0_n_inner_2, "mad", mad_dict)
    sch[c_ub_2].emit_insn(c_ub_2.op.axis[0], "dma_copy")
    sch[n_t_2].emit_insn(n_t_2.op.axis[0], "dma_copy")
    sch[r_t_1].emit_insn(r_t_1.op.axis[0], "dma_copy")
    sch[i_t_1].emit_insn(i_t_1.op.axis[0], "dma_copy")
    sch[n_t_1].emit_insn(n_t_1.op.axis[0], "dma_copy")

    if bias2 is not None:
        sch[bias_ub_2].emit_insn(bias_ub_2.op.axis[0], "dma_copy")
        sch[bias_bc_ub_2].emit_insn(bias_bc_ub_2.op.axis[0], "unified_broadcast")
        if fp16_input_output:
            sch[bias_ub_2_fp32].emit_insn(bias_ub_2_fp32.op.axis[0], "vector_conv")

    if is_gate_output:
        sch[r_t_gm].emit_insn(r_t_gm.op.axis[0], "dma_copy")
        sch[i_t_gm].emit_insn(i_t_gm.op.axis[0], "dma_copy")
        sch[n_t_gm].emit_insn(n_t_gm.op.axis[0], "dma_copy")
        sch[hn_t_gm].emit_insn(hn_t_gm.op.axis[0], "dma_copy")
        sch[r_t_gm_back].emit_insn(r_t_gm_back.op.axis[0], "phony_insn")
        sch[i_t_gm_back].emit_insn(i_t_gm_back.op.axis[0], "phony_insn")
        sch[n_t_gm_back].emit_insn(n_t_gm_back.op.axis[0], "phony_insn")
        sch[hn_t_gm_back].emit_insn(hn_t_gm_back.op.axis[0], "phony_insn")
        if fp16_input_output:
            sch[r_t_sigmoid_fp16].emit_insn(r_t_sigmoid_fp16.op.axis[0], "vector_conv")
            sch[i_t_sigmoid_fp16].emit_insn(i_t_sigmoid_fp16.op.axis[0], "vector_conv")
            sch[n_t_tanh_fp16].emit_insn(n_t_tanh_fp16.op.axis[0], "vector_conv")
            sch[n_t_2_fp16].emit_insn(n_t_2_fp16.op.axis[0], "vector_conv")
            sch[r_t_gm_back_fp32].emit_insn(r_t_gm_back_fp32.op.axis[0], "phony_insn")
            sch[i_t_gm_back_fp32].emit_insn(i_t_gm_back_fp32.op.axis[0], "phony_insn")
            sch[n_t_gm_back_fp32].emit_insn(n_t_gm_back_fp32.op.axis[0], "phony_insn")
            sch[hn_t_gm_back_fp32].emit_insn(hn_t_gm_back_fp32.op.axis[0], "phony_insn")
            sch[r_t_gm_back_fp32].reused_by(r_t_sigmoid)
            sch[i_t_gm_back_fp32].reused_by(i_t_sigmoid)
            sch[n_t_gm_back_fp32].reused_by(n_t_tanh)
            sch[hn_t_gm_back_fp32].reused_by(n_t_2)
            sch[r_t_gm_back].reused_by(r_t_sigmoid_fp16)
            sch[i_t_gm_back].reused_by(i_t_sigmoid_fp16)
            sch[n_t_gm_back].reused_by(n_t_tanh_fp16)
            sch[hn_t_gm_back].reused_by(n_t_2_fp16)
        else:
            sch[r_t_gm_back].reused_by(r_t_sigmoid)
            sch[i_t_gm_back].reused_by(i_t_sigmoid)
            sch[n_t_gm_back].reused_by(n_t_tanh)
            sch[hn_t_gm_back].reused_by(n_t_2)

    if fp16_input_output:
        sch[update_h_fp16].emit_insn(update_h_fp16.op.axis[0], "vector_conv")
    if seq_mask_gm is not None:
        sch[seq_mask_ub].emit_insn(seq_mask_ub.op.axis[0], 'dma_copy')
        sch[seq_mask_ub_fp32].emit_insn(seq_mask_ub_fp32.op.axis[0], 'vector_conv')
    sch[update_y_gm].emit_insn(update_y_gm.op.axis[0], "dma_copy")
    sch[update_y_gm_back].emit_insn(update_y_gm_back.op.axis[0], "phony_insn")
    sch[update_y_gm_back].reused_by(update_h_ub)
    sch[update_h_gm].emit_insn(update_h_gm_inner, "dma_copy")

    return output_list, sch
