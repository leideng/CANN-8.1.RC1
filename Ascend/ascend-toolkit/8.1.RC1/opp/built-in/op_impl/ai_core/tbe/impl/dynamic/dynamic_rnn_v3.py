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

dynamic_rnn_v3
"""
# 'pylint: disable=too-many-lines
import copy
import operator

import numpy as np
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
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
from te.tvm import expr
from tbe.tvm import create_schedule
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from tbe.common.buildcfg.default_buildcfg import dynamic_build_config_dict
from tbe.common.register import register_param_generalization
from tbe.common.rl_bank import rl_bank


def sigmoid_compute(input_x):
    """
    calculating sigmoid
    """
    data_input = input_x
    dtype = input_x.dtype
    exp_support = api_check_support("te.lang.cce.vexp", "float32")
    mul_support = api_check_support("te.lang.cce.vmuls", "float32")
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
    exp_support = api_check_support("te.lang.cce.vexp", "float32")
    # positive min float32 value
    min_fp_data = 2 ** (-126)
    const_dtype = input_dtype
    # positive min float16 value
    if input_dtype == "float16":
        min_fp_data = 2 ** (-14)

    has_improve_precision = False

    if input_dtype == "float16" and exp_support:
        input_x = cast_to(input_x, "float32")
        has_improve_precision = True
        const_dtype = "float32"

    input_abs = vabs(input_x)
    power_val = vmuls(input_abs, tvm.const(-2, const_dtype))
    if input_dtype == "float32" and not exp_support:
        power_val = cast_to(power_val, "float16")
    exp_val = vexp(power_val)
    if input_dtype == "float32" and not exp_support:
        exp_val = cast_to(exp_val, "float32")

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
                "broadcast": "vector_broadcast"}
    if tensor.op.tag.find("|") != -1:
        str_list = tensor.op.tag.split("|")
        insn = insn_map.get(str_list[0])
    else:
        insn = insn_map.get(tensor.op.tag)
    return insn


def get_lstm_tiling():
    """
    get no RL default lstm element wise tiling
    :return:
    """
    return (1, 1, 12, 1, 1, 12)


def ceil_value(value, factor):
    """
    if not divide exactly then plus 1

    Parameters
    ----------
    value:  input number
    factor: factor

    Returns
    -------
    ceil value
    """
    return (value + factor - 1) // factor


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
            error_manager_vector.raise_err_specific_reson("DynamicRNNV3",
                                                          "init_c dtype is not the same as bias dtype !")

    # check output
    if y["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3", "y dtype is not the same as bias dtype !")
    if output_c["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3", "output_c dtype is not the same as bias dtype !")

    # check additional output
    if i["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3", "i dtype is not the same as bias dtype !")
    if j["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3", "j dtype is not the same as bias dtype !")
    if f["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3", "f dtype is not the same as bias dtype !")
    if o["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3", "o dtype is not the same as bias dtype !")
    if tanhc["dtype"] != bias_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3", "tanhc dtype is not the same as bias dtype !")


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name,invalid-name
def check_prama_shape(weight, bias, seq_length, init_h, init_c, y, output_h, output_c, i, j, f, o, tanhc):
    """
    check parameters
    """
    if ceil_value(bias["shape"][0], 16) != weight["shape"][1]:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3", "w, b shape is wrong, please check!")

    # check init
    if (init_h is None and init_c is not None) or (
            init_h is not None and init_c is None):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3",
                                                      "init_h, init_c should appear together, please check!")

    # check output
    if not operator.eq(output_h["shape"], y["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3",
                                                      "y, output_h shape is different, please check!")

    if not operator.eq(output_c["shape"], i["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3",
                                                      "i, output_c shape is different, please check!")

    if not operator.eq(output_c["shape"], j["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3",
                                                      "j, output_c shape is different, please check!")

    if not operator.eq(output_c["shape"], f["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3",
                                                      "f, output_c shape is different, please check!")

    if not operator.eq(output_c["shape"], o["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3",
                                                      "o, output_c shape is different, please check!")

    if not operator.eq(output_c["shape"], tanhc["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3",
                                                      "tanhc, output_c shape is different, please check!")


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals
def check_attr(cell_type, direction, cell_depth, activation):
    """
    check parameters
    """
    if cell_type not in ["LSTM", "GRU", "RNN_RELU", "RNN_TANH"]:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3", "attr cell_type is not support, please check!")

    if direction not in ["UNIDIRECTIONAL", "BIDIRECTIONAL"]:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3", "attr direction is not support, please check!")

    if cell_depth != 1:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3", "attr cell_depth is not support, please check!")

    if activation not in ["tanh"]:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV3",
                                                      "attr activation only support tanh, please check!")


@register_operator("DynamicRNNV3")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-function-args,too-many-statements
# 'pylint: disable=unused-argument
def dynamic_rnn_v3(input_x, weight, bias, seq_length, init_h, init_c, wci, wcf,
                   wco, mask, real_mask, project, y, output_h, output_c, i, j, f, o, tanhc,
                   cell_type="LSTM", direction="UNIDIRECTIONAL", cell_depth=1,
                   use_peephole=False, keep_prob=1.0, cell_clip=-1.0, num_proj=0,
                   time_major=True, activation="tanh", forget_bias=0.0, is_training=True,
                   kernel_name="dynamic_rnn_v3"):
    """
    dynamic_rnn_v3
    """
    # one block size takes up 32b
    block_size_1 = 32
    # data type of int32
    int32 = "int32"
    tiling_arg_num = 3
    type_len_dict = {"float16": 2, "float32": 4, "int8": 1, "uint8": 1,
                 "int32": 4, "int64": 8, }
    is_dynamic = True

    pipe_hole_fun = False
    project_fun = False
    real_mask_fun = False
    if wci is not None:
        pipe_hole_fun = True
    if real_mask is not None:
        real_mask_fun = True
    if project is not None:
        project_fun = True
    if init_h is not None and len(init_h["shape"]) == 4:
        init_h["shape"] = [1, init_h["shape"][0], init_h["shape"][1], init_h["shape"][2], init_h["shape"][3]]
    if init_c is not None and len(init_c["shape"]) == 4:
        init_c["shape"] = [1, init_c["shape"][0], init_c["shape"][1], init_c["shape"][2], init_c["shape"][3]]
    if pipe_hole_fun and len(wci["shape"]) == 4:
        wci["shape"] = [1, wci["shape"][0], wci["shape"][1], wci["shape"][2], wci["shape"][3]]
    if pipe_hole_fun and len(wcf["shape"]) == 4:
        wcf["shape"] = [1, wcf["shape"][0], wcf["shape"][1], wcf["shape"][2], wcf["shape"][3]]
    if pipe_hole_fun and len(wco["shape"]) == 4:
        wco["shape"] = [1, wco["shape"][0], wco["shape"][1], wco["shape"][2], wco["shape"][3]]
    if real_mask_fun and len(real_mask["shape"]) == 4:
        real_mask["shape"] = [1, real_mask["shape"][0], real_mask["shape"][1], real_mask["shape"][2],
                              real_mask["shape"][3]]
    if project_fun and len(project["shape"]) == 4:
        project["shape"] = [1, project["shape"][0], project["shape"][1], project["shape"][2], project["shape"][3]]

    check_prama_dtype(input_x, weight, bias, init_h, init_c, y, output_h,
                      output_c, i, j, f, o, tanhc)

    check_prama_shape(weight, bias, seq_length, init_h, init_c, y, output_h,
                      output_c, i, j, f, o, tanhc)

    check_attr(cell_type, direction, cell_depth, activation)

    shape_x_input = input_x.get("shape")
    shape_w_input = weight.get("shape")

    input_dtype = input_x.get("dtype").lower()
    bias_dtype = bias.get("dtype").lower()

    tik_instance = Tik(Dprofile())

    if is_dynamic:
        # dynamic shape get seq_length
        tiling_shape = (tiling_arg_num,)
        tiling_dtype = int32

        tiling_gm = tik_instance.Tensor(tiling_dtype, tiling_shape, name="ddr_arg", scope=scope_gm)

        tiling_ub = tik_instance.Tensor(tiling_dtype, tiling_shape, name="tiling_ub", scope=scope_ubuf)

        tik_instance.data_move(tiling_ub, tiling_gm, 0, 1,
                               ceil_value(tiling_arg_num * type_len_dict.get(tiling_dtype), block_size_1),
                               0, 0)

        # get run tiling mode
        tiling_seq = tik_instance.Scalar(dtype=tiling_dtype, name="tiling_seq")
        tiling_seq.set_as(tiling_ub[0])

        tiling_batch = tik_instance.Scalar(dtype=tiling_dtype, name="tiling_batch")
        tiling_batch.set_as(tiling_ub[1])

        tiling_index = tik_instance.Scalar(dtype=tiling_dtype, name="tiling_index")
        tiling_index.set_as(tiling_ub[2])

        t_size = tiling_seq
        m_size = tiling_batch
    else:
        t_size = shape_x_input[0]
        m_size = shape_x_input[2]
    k_size = shape_w_input[0]
    n_size = shape_w_input[1]

    block_size = 4
    hidden_size = n_size // 4
    if project_fun:
        state_size = project.get("shape")[1]
    else:
        state_size = hidden_size
    in_x = k_size - state_size
    shape_x = (t_size, in_x, m_size, 16, 16)
    shape_w = (1, k_size, block_size, hidden_size, 16, 16)
    shape_hc = (t_size, hidden_size, m_size, 16, 16)
    shape_bias = (1, block_size, hidden_size, 1, 1, 16)
    shape_h_init = (1, state_size, m_size, 16, 16)
    shape_hc_y = (t_size, state_size, m_size, 16, 16)
    shape_hc_init = (1, hidden_size, m_size, 16, 16)
    shape_pipe_hole = (1, hidden_size, m_size, 16, 16)
    shape_mask = (t_size, 1, m_size, 16, 16)
    shape_project = (1, state_size, hidden_size, 16, 16)

    is_global_init = False
    if init_h is not None:
        is_global_init = True

    # due to FE/GE not support, now set default value
    is_gate_output = True

    sync = tik_instance.Tensor(shape=(512,), dtype="int64", scope=scope_gm, name='sync',
                               is_workspace=True, is_atomic_add=True)

    # `sync = None`
    input_x = tik_instance.Tensor(shape=shape_x, dtype=input_dtype,
                                  scope=scope_gm, name='input_x')
    weight = tik_instance.Tensor(shape=shape_w, dtype=input_dtype,
                                 scope=scope_gm, name='weight')
    bias = tik_instance.Tensor(shape=shape_bias, scope=scope_gm,
                               dtype=bias_dtype, name='bias')
    wci_gm = None
    wco_gm = None
    wcf_gm = None
    mask_gm = None
    project_gm = None

    if pipe_hole_fun:
        wci_gm = tik_instance.Tensor(shape=shape_pipe_hole, scope=scope_gm,
                                     dtype=bias_dtype, name='wci')
        wco_gm = tik_instance.Tensor(shape=shape_pipe_hole, scope=scope_gm,
                                     dtype=bias_dtype, name='wco')
        wcf_gm = tik_instance.Tensor(shape=shape_pipe_hole, scope=scope_gm,
                                     dtype=bias_dtype, name='wcf')
    if real_mask_fun:
        mask_gm = tik_instance.Tensor(shape=shape_mask, scope=scope_gm,
                                      dtype=bias_dtype, name='mask')

    if project_fun:
        project_gm = tik_instance.Tensor(shape=shape_project, scope=scope_gm,
                                         dtype=input_dtype, name='project')

    is_using_seq_mask = False
    if seq_length is not None:
        shape_seq_length = (t_size, hidden_size, m_size, 16, 16)
        is_using_seq_mask = True
        if seq_length.get("dtype").lower() == "int32":
            seq_mask_gm = tik_instance.Tensor(shape=shape_seq_length, scope=scope_gm,
                                              dtype="int32", name='seq_mask_gm')
        else:
            seq_mask_gm = tik_instance.Tensor(shape=shape_seq_length, scope=scope_gm,
                                              dtype="float16", name='seq_mask_gm')
    else:
        seq_mask_gm = None

    if is_global_init:
        s_init_h_gm = tik_instance.Tensor(shape=shape_h_init,
                                          dtype=input_dtype,
                                          scope=scope_gm,
                                          name='s_init_h_gm')
        s_init_c_gm = tik_instance.Tensor(shape=shape_hc_init,
                                          dtype=bias_dtype, scope=scope_gm,
                                          name='s_init_c_gm')

    update_h_gm = tik_instance.Tensor(shape=shape_hc_y, dtype=input_dtype,
                                      scope=scope_gm, name='update_h_gm')
    update_c_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                      scope=scope_gm, name='update_c_gm')
    update_h_gm_as_y = tik_instance.Tensor(shape=shape_hc_y, dtype=bias_dtype,
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
    if pipe_hole_fun:
        build_input_list.append(wci_gm)
        build_input_list.append(wcf_gm)
        build_input_list.append(wco_gm)
    if real_mask_fun:
        build_input_list.append(mask_gm)
    if project_fun:
        build_input_list.append(project_gm)

    if is_gate_output:
        build_output_list = [update_h_gm_as_y, update_h_gm, update_c_gm,
                             i_t_sigmoid_gm, j_t_tanh_gm, f_t_sigmoid_gm,
                             o_t_sigmoid_gm, c_t_tanh_gm]
    else:
        build_output_list = [update_h_gm_as_y, update_h_gm, update_c_gm]

    # for RL tune getting tik input&output tensor
    fusion_manager.set_tik_tensor(build_input_list, build_output_list)
    bank_manager.init_bank_hit_info(kernel_name)

    last = 1
    cut_t = 1
    # RL default tiling 
    if is_dynamic:
        cut_m = tiling_batch
        loop_t = tiling_seq // cut_t
        loop_m = tiling_batch // cut_m
    else:
        cut_m = m_size
        loop_t = t_size // cut_t
        loop_m = m_size // cut_m

    rl_idx_list_first = []

    with tik_instance.for_range(0, loop_t) as loop_i:
        with tik_instance.for_range(0, loop_m) as loop_j:

            input_x_var = input_x[loop_i * cut_t: loop_i * cut_t + cut_t,
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

            state_h_last = update_h_gm[loop_i * cut_t - last: loop_i * cut_t + cut_t - last:,
                                       :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            state_c_last = update_c_gm[loop_i * cut_t - last: loop_i * cut_t + cut_t - last:,
                                       :,
                                       loop_j * cut_m: loop_j * cut_m + cut_m,
                                       :, :]

            update_h_gm_var = update_h_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                          :,
                                          loop_j * cut_m: loop_j * cut_m + cut_m,
                                          :, :]
            update_c_gm_var = update_c_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                          :,
                                          loop_j * cut_m: loop_j * cut_m + cut_m,
                                          :, :]
            update_h_gm_as_y_var = update_h_gm_as_y[loop_i * cut_t: loop_i * cut_t + cut_t:,
                                                    :,
                                                    loop_j * cut_m: loop_j * cut_m + cut_m,
                                                    :, :]

            if is_gate_output:
                f_t_sigmoid_gm_var = f_t_sigmoid_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                                    :,
                                                    loop_j * cut_m: loop_j * cut_m + cut_m,
                                                    :, :]
                i_t_sigmoid_gm_var = i_t_sigmoid_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                                    :,
                                                    loop_j * cut_m: loop_j * cut_m + cut_m,
                                                    :, :]
                o_t_sigmoid_gm_var = o_t_sigmoid_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                                    :,
                                                    loop_j * cut_m: loop_j * cut_m + cut_m,
                                                    :, :]
                j_t_tanh_gm_var = j_t_tanh_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                              :,
                                              loop_j * cut_m: loop_j * cut_m + cut_m,
                                              :, :]
                c_t_tanh_gm_var = c_t_tanh_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                              :,
                                              loop_j * cut_m: loop_j * cut_m + cut_m,
                                              :, :]
            else:
                f_t_sigmoid_gm_var = None
                i_t_sigmoid_gm_var = None
                o_t_sigmoid_gm_var = None
                j_t_tanh_gm_var = None
                c_t_tanh_gm_var = None
            mask_gm_cur = None
            if real_mask_fun:
                mask_gm_cur = mask_gm[loop_i * cut_t: loop_i * cut_t + cut_t:,
                                      :,
                                      loop_j * cut_m: loop_j * cut_m + cut_m,
                                      :, :]
            wci_gm_cur = None
            wco_gm_cur = None
            wcf_gm_cur = None
            if pipe_hole_fun:
                wci_gm_cur = wci_gm[:, :,
                                    loop_j * cut_m: loop_j * cut_m + cut_m,
                                    :, :]
                wco_gm_cur = wco_gm[:, :,
                                  loop_j * cut_m: loop_j * cut_m + cut_m,
                                  :, :]
                wcf_gm_cur = wcf_gm[:, :,
                                    loop_j * cut_m: loop_j * cut_m + cut_m,
                                    :, :]
            input_list = [input_x_var, weight, bias, s_init_h_gm_var,
                          s_init_c_gm_var, state_h_last,
                          state_c_last, sync, wci_gm_cur, wco_gm_cur, wcf_gm_cur, mask_gm_cur, project_gm]

            if is_gate_output:
                output_list = [update_h_gm_var, update_c_gm_var,
                               update_h_gm_as_y_var, i_t_sigmoid_gm_var,
                               j_t_tanh_gm_var, f_t_sigmoid_gm_var,
                               o_t_sigmoid_gm_var, c_t_tanh_gm_var]
            else:
                output_list = [update_h_gm_var, update_c_gm_var,
                               update_h_gm_as_y_var]

            if is_dynamic:
                with tik_instance.if_scope(loop_i == 0):
                    is_first_round = True
                    rl_idx_list_first = tik_instance.call_module(
                        function=dynamic_rnn_tik,
                        input_tensors=input_list,
                        output_tensors=output_list,
                        config_map={"tiling_key":tiling_index},
                        input_params=[is_gate_output, is_first_round, is_global_init,
                                      forget_bias, pipe_hole_fun, real_mask_fun, project_fun, is_dynamic])

                with tik_instance.if_scope(loop_i > 0):
                    is_first_round = False
                    tik_instance.call_module(
                        function=dynamic_rnn_tik,
                        input_tensors=input_list,
                        output_tensors=output_list,
                        config_map={"tiling_key":tiling_index},
                        input_params=[is_gate_output, is_first_round, is_global_init,
                                      forget_bias, pipe_hole_fun, real_mask_fun, project_fun, is_dynamic])
            else:
                with tik_instance.if_scope(loop_i == 0):
                    is_first_round = True
                    tik_instance.call_module(
                        dynamic_rnn_tik,
                        input_list,
                        output_list,
                        [is_gate_output, is_first_round, is_global_init,
                        forget_bias, pipe_hole_fun, real_mask_fun, project_fun, is_dynamic])

                with tik_instance.if_scope(loop_i > 0):
                    is_first_round = False
                    tik_instance.call_module(
                        dynamic_rnn_tik,
                        input_list,
                        output_list,
                        [is_gate_output, is_first_round, is_global_init,
                        forget_bias, pipe_hole_fun, real_mask_fun, project_fun, is_dynamic])

    tiling_key_value_list = []
    for idx in rl_idx_list_first:
        tiling_key_value_list.append([idx])

    config_map = {
        "dump_cce_code": False,
        "save_temp_cce_file": False,
    }

    if is_dynamic:
        dynamic_config_a = copy.deepcopy(dynamic_build_config_dict)
        dynamic_config_a["dump_cce_code"] = False
        dynamic_config_a["tir.InjectSync"] = {"sync_mode": 2}
        dynamic_config_a["debug_message"] = False
        dynamic_config_a["save_temp_cce_file"] = False

        tik_instance.BuildCCE(kernel_name,
                              build_input_list,
                              build_output_list,
                              config=dynamic_config_a,
                              flowtable=(tiling_gm,),
                              extend_params={"build_multi_kernels":{
                                      "tiling_key":[tiling_index],
                                      "tiling_key_value":tiling_key_value_list
                              }}
                              )
    else:
        tik_instance.BuildCCE(kernel_name,
                              build_input_list,
                              build_output_list,
                              config=config_map)


def dynamic_rnn_tik(input_list, custom_list):
    """
    inside part of tik loop
    :return:
    """
    is_gate_output = custom_list[0]
    is_first_round = custom_list[1]
    is_global_init = custom_list[2]
    forget_bias = custom_list[3]
    is_dynamic = custom_list[7]

    (input_x, weight, bias, s_init_h_gm, s_init_c_gm, s_state_h_gm_last, s_state_c_gm_last, sync0,
     wci_gm, wco_gm, wcf_gm, mask_gm, project_gm) = input_list

    return dynamic_rnn_core(input_x, weight, bias, s_init_h_gm, s_init_c_gm,
                            s_state_h_gm_last, s_state_c_gm_last, sync0,
                            wci_gm, wco_gm, wcf_gm, mask_gm, project_gm,
                            is_gate_output, is_first_round, is_global_init,
                            forget_bias, is_dynamic)


# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-statements,unnecessary-lambda,too-many-lines
def dynamic_rnn_core(input_x, weight, bias, s_init_h_gm, s_init_c_gm,
                     s_state_h_gm_last, s_state_c_gm_last, sync0,
                     wci_gm, wco_gm, wcf_gm, mask_gm, project_gm,
                     is_gate_output, is_first_round, is_global_init,
                     forget_bias, is_dynamic):
    """
    implement of dynamic rnnv3
    :return:
    """
    shape_x_input = input_x.shape
    shape_w_input = weight.shape

    t_size = 1
    if is_dynamic:
        m_size = shape_x_input[2]
    else:
        m_size = shape_x_input[2].value
    k_size = shape_w_input[1].value
    n_size = shape_w_input[2].value * shape_w_input[3].value

    block_size = 4
    hidden_size = n_size // block_size
    if project_gm is not None:
        state_size = project_gm.shape[1].value
    else:
        state_size = hidden_size

    in_x = k_size - state_size

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
                                       lambda _, a_i, b_i, c_i, d_i: s_init_h_gm[
                                       0, a_i, b_i, c_i, d_i], name="s_init_h")
            s_state_h_ub_tmp = tvm.compute(shape_h,
                                           lambda _, a_i, b_i, c_i, d_i: s_init_h_gm[
                                           0, a_i, b_i, c_i, d_i], name="s_init_h_tmp")
            s_state_c_ub = tvm.compute(shape_i,
                                       lambda _, a_i, b_i, c_i, d_i: s_init_c_gm[
                                       0, a_i, b_i, c_i, d_i], name="s_init_c")
        else:
            s_state_h_ub = \
                tvm.compute(shape_h,
                            lambda *indices: tvm.const(0.0, dtype=input_dtype),
                            name='s_state_h_ub',
                            tag="broadcast")
            s_state_h_ub_tmp = \
                tvm.compute(shape_h,
                            lambda *indices: tvm.const(0.0, dtype=input_dtype),
                            name='s_state_h_ub_tmp',
                            tag="broadcast")
            s_state_c_ub = \
                tvm.compute(shape_i,
                            lambda *indices: tvm.const(0.0, dtype=bias_dtype),
                            name='s_state_c_ub',
                            tag="broadcast")
    else:
        s_state_h_ub = tvm.compute(shape_h,
                                   lambda _, a_i, b_i, c_i, d_i: s_state_h_gm_last[
                                   0, a_i, b_i, c_i, d_i], name="s_state_h_ub")
        s_state_h_ub_tmp = tvm.compute(shape_h,
                                       lambda _, a_i, b_i, c_i, d_i: s_state_h_gm_last[
                                       0, a_i, b_i, c_i, d_i], name="s_state_h_ub_tmp")
        s_state_c_ub = tvm.compute(shape_i,
                                   lambda _, a_i, b_i, c_i, d_i: s_state_c_gm_last[
                                   0, a_i, b_i, c_i, d_i], name="s_state_c_ub")

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

    c_l0c = tvm.compute(shape_c,
                        lambda t, nb_0, nb_1, mb, mp, np:
                        tvm.sum((a_l0a[t, mb, k1, mp, k0] * \
                                 b_l0b[t, k1, nb_0, nb_1, np, k0]) \
                                .astype('float32'),
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
    j_t_index = 1
    f_t_index = 2
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

    pipehole_tensors = []
    mask_tensors = []
    project_tensors = []
    if wci_gm is not None:
        wci_ub = tvm.compute(shape_i, lambda _, a_i, b_i, c_i, d_i: wci_gm[0, a_i, b_i, c_i, d_i], name="wci_ub")
        wco_ub = tvm.compute(shape_i, lambda _, a_i, b_i, c_i, d_i: wco_gm[0, a_i, b_i, c_i, d_i], name="wco_ub")
        wcf_ub = tvm.compute(shape_i, lambda _, a_i, b_i, c_i, d_i: wcf_gm[0, a_i, b_i, c_i, d_i], name="wcf_ub")
        wci_ct_mul = vmul(wci_ub, s_state_c_ub)
        wcf_ct_mul = vmul(wcf_ub, s_state_c_ub)
        wci_ct_mul_mid = wci_ct_mul
        wcf_ct_mul_mid = wcf_ct_mul
        # trans wcf_ct_mul fp16 to fp32
        if fp16_input_output:
            wcf_ct_mul_fp32 = \
                tvm.compute(shape_i,
                            lambda *indices: wcf_ct_mul(*indices).astype('float32'),
                            name="wcf_ct_mul_fp32_drnn_cast",
                            tag="elewise_single_cast")
            wci_ct_mul_fp32 = \
                tvm.compute(shape_i,
                            lambda *indices: wci_ct_mul(*indices).astype('float32'),
                            name="wci_ct_mul_fp32_drnn_cast",
                            tag="elewise_single_cast")
            pipehole_tensors.append(wcf_ct_mul_fp32)
            pipehole_tensors.append(wci_ct_mul_fp32)
            wci_ct_mul_mid = wci_ct_mul_fp32
            wcf_ct_mul_mid = wcf_ct_mul_fp32
        f_t_tmp = vadd(f_t, wcf_ct_mul_mid)
        i_t_tmp = vadd(i_t, wci_ct_mul_mid)
        pipehole_tensors.append(wci_ub)
        pipehole_tensors.append(wco_ub)
        pipehole_tensors.append(wcf_ub)
        pipehole_tensors.append(wci_ct_mul)
        pipehole_tensors.append(wcf_ct_mul)
        pipehole_tensors.append(f_t_tmp)
        pipehole_tensors.append(i_t_tmp)
    else:
        f_t_tmp = f_t
        i_t_tmp = i_t

    if mask_gm is not None:
        shape_mask_ub_ct = [1, hidden_size, m_size, 16, 16]
        mask_ub_ct = tvm.compute(shape_mask_ub_ct,
                                 lambda _, a_i, b_i, c_i, d_i: mask_gm[0, 0, b_i, c_i, d_i], name="mask_ub_ct")
        shape_mask_ub_ht = [1, state_size, m_size, 16, 16]
        mask_ub_ht = tvm.compute(shape_mask_ub_ht,
                                 lambda _, a_i, b_i, c_i, d_i: mask_gm[0, 0, b_i, c_i, d_i], name="mask_ub_ht")
        mask_tensors.append(mask_ub_ct)
        mask_tensors.append(mask_ub_ht)
        mask_ub_ct_fp32_mid = mask_ub_ct
        mask_ub_ht_fp32_mid = mask_ub_ht
        if fp16_input_output:
            mask_ub_ct_fp32 = \
                    tvm.compute(shape_mask_ub_ct,
                                lambda *indices: mask_ub_ct(*indices).astype('float32'),
                                name="mask_ub_ct_fp32_drnn_cast",
                                tag="elewise_single_cast")
            mask_ub_ht_fp32 = \
                    tvm.compute(shape_mask_ub_ht,
                                lambda *indices: mask_ub_ht(*indices).astype('float32'),
                                name="mask_ub_ht_fp32_drnn_cast",
                                tag="elewise_single_cast")
            mask_tensors.append(mask_ub_ct_fp32)
            mask_tensors.append(mask_ub_ht_fp32)
            mask_ub_ct_fp32_mid = mask_ub_ct_fp32
            mask_ub_ht_fp32_mid = mask_ub_ht_fp32
    f_t_bias = vadds(f_t_tmp, tvm.const(forget_bias, dtype=bias_dtype))
    f_t_sigmoid = sigmoid_compute(f_t_bias)
    i_t_sigmoid = sigmoid_compute(i_t_tmp)
    j_t_tanh = tanh_compute(j_t)

    f_t_sigmoid_ub = f_t_sigmoid
    i_t_sigmoid_ub = i_t_sigmoid
    j_t_tanh_ub = j_t_tanh

    if is_gate_output:
        f_t_sigmoid_mid = f_t_sigmoid
        i_t_sigmoid_mid = i_t_sigmoid
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
            j_t_tanh_fp16 = \
                tvm.compute(shape_i,
                            lambda *indices: j_t_tanh(*indices).astype('float16'),
                            name="j_t_tanh_fp16_drnn_cast",
                            tag="elewise_single_cast")
            f_t_sigmoid_mid = f_t_sigmoid_fp16
            i_t_sigmoid_mid = i_t_sigmoid_fp16
            j_t_tanh_mid = j_t_tanh_fp16

        f_t_sigmoid_gm = tvm.compute(shape_i,
                                     lambda *indices: f_t_sigmoid_mid(*indices),
                                     name="f_t_sigmoid_gm",
                                     tag="ub_to_out")
        i_t_sigmoid_gm = tvm.compute(shape_i,
                                     lambda *indices: i_t_sigmoid_mid(*indices),
                                     name="i_t_sigmoid_gm",
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
        j_t_tanh_back = tvm.compute(shape_i,
                                    lambda *indices: j_t_tanh_gm(*indices),
                                    name="j_t_tanh_back",
                                    tag="out_to_ub")

        if fp16_input_output:
            f_t_sigmoid_back_fp32 = tvm.compute(shape_i,
                                                lambda *indices: f_t_sigmoid_back(*indices).astype('float32'),
                                                name="f_t_sigmoid_back_fp32_drnn_cast",
                                                tag="elewise_single_cast")
            i_t_sigmoid_back_fp32 = tvm.compute(shape_i,
                                                lambda *indices: i_t_sigmoid_back(*indices).astype('float32'),
                                                name="i_t_sigmoid_back_fp32_drnn_cast",
                                                tag="elewise_single_cast")
            j_t_tanh_back_fp32 = tvm.compute(shape_i,
                                             lambda *indices: j_t_tanh_back(*indices).astype('float32'),
                                             name="j_t_tanh_back_fp32_drnn_cast",
                                             tag="elewise_single_cast")

            s_state_c_back_fp32 = tvm.compute(shape_i,
                                              lambda *indices: s_state_c_ub(*indices).astype('float32'),
                                              name="s_state_c_back_fp32_drnn_cast",
                                              tag="elewise_single_cast")

        f_t_sigmoid_ub = f_t_sigmoid_back
        i_t_sigmoid_ub = i_t_sigmoid_back
        j_t_tanh_ub = j_t_tanh_back
        s_state_c_ub_temp = s_state_c_ub

        if fp16_input_output:
            f_t_sigmoid_ub = f_t_sigmoid_back_fp32
            i_t_sigmoid_ub = i_t_sigmoid_back_fp32
            j_t_tanh_ub = j_t_tanh_back_fp32
            s_state_c_ub_temp = s_state_c_back_fp32

    c_t_tmp1 = vmul(s_state_c_ub_temp, f_t_sigmoid_ub)
    c_t_tmp2 = vmul(j_t_tanh_ub, i_t_sigmoid_ub)
    update_c = vadd(c_t_tmp1, c_t_tmp2)

    c_t_tanh = tanh_compute(update_c)

    if wci_gm is not None:
        wco_ub_mid = wco_ub
        if fp16_input_output:
            wco_ub_fp32 = tvm.compute(shape_i,
                                      lambda *indices: wco_ub(*indices).astype('float32'),
                                      name="wco_ub_fp32_drnn_cast",
                                      tag="elewise_single_cast")
            wco_ub_mid = wco_ub_fp32
            pipehole_tensors.append(wco_ub_fp32)
        wco_ct_add = vmul(wco_ub_mid, update_c)
        o_t_tmp = vadd(wco_ct_add, o_t)
        pipehole_tensors.append(wco_ct_add)
        pipehole_tensors.append(o_t_tmp)
        pipehole_tensors.append(wco_ub_mid)
    else:
        o_t_tmp = o_t
    o_t_sigmoid = sigmoid_compute(o_t_tmp)
    o_t_sigmoid_ub = o_t_sigmoid

    if is_gate_output:
        o_t_sigmoid_mid = o_t_sigmoid
        if fp16_input_output:
            o_t_sigmoid_fp16 = tvm.compute(shape_i,
                                           lambda *indices: o_t_sigmoid(*indices).astype('float16'),
                                           name="o_t_sigmoid_fp16_drnn_cast",
                                           tag="elewise_single_cast")
            o_t_sigmoid_mid = o_t_sigmoid_fp16

        o_t_sigmoid_gm = tvm.compute(shape_i,
                                     lambda *indices: o_t_sigmoid_mid(*indices),
                                     name="o_t_sigmoid_gm",
                                     tag="ub_to_out")
        o_t_sigmoid_back = tvm.compute(shape_i,
                                       lambda *indices: o_t_sigmoid_gm(*indices),
                                       name="o_t_sigmoid_back",
                                       tag="out_to_ub")
        if fp16_input_output:
            o_t_sigmoid_back_fp32 = tvm.compute(shape_i,
                                                lambda *indices: o_t_sigmoid_back(*indices).astype('float32'),
                                                name="o_t_sigmoid_back_fp32_drnn_cast",
                                                tag="elewise_single_cast")
        o_t_sigmoid_ub = o_t_sigmoid_back
        if fp16_input_output and project_gm is None:
            o_t_sigmoid_ub = o_t_sigmoid_back_fp32

    if mask_gm is not None:
        update_c_tmp1 = vmul(mask_ub_ct_fp32_mid, update_c)
        broadcast_one = broadcast(tvm.const(1, bias_dtype), [1, hidden_size, m_size, 16, 16], bias_dtype)
        one_sub_mtc = vsub(broadcast_one, mask_ub_ct)
        update_c_tmp2 = vmul(one_sub_mtc, s_state_c_ub)
        update_c_tmp2_mid = update_c_tmp2
        if fp16_input_output:
            update_c_tmp2_fp32 = tvm.compute(shape_i,
                                             lambda *indices: update_c_tmp2(*indices).astype('float32'),
                                             name="update_c_tmp2_fp32_drnn_cast",
                                             tag="elewise_single_cast")
            update_c_tmp2_mid = update_c_tmp2_fp32
            mask_tensors.append(update_c_tmp2_fp32)
        update_c_tmp = vadd(update_c_tmp1, update_c_tmp2_mid)
        mask_tensors.append(update_c_tmp1)
        mask_tensors.append(broadcast_one)
        mask_tensors.append(one_sub_mtc)
        mask_tensors.append(update_c_tmp2)
        mask_tensors.append(update_c_tmp)
    else:
        update_c_tmp = update_c

    if bias_dtype == 'float16':
        update_c_fp16 = tvm.compute(shape_i,
                                    lambda *indices: update_c_tmp(*indices).astype('float16'),
                                    name="update_c_fp16_drnn_cast",
                                    tag="elewise_single_cast")
        update_c_gm = tvm.compute(shape_i,
                                  lambda *indices: update_c_fp16(*indices),
                                  name="update_c_gm",
                                  tag="ub_to_out")
    else:
        update_c_gm = tvm.compute(shape_i,
                                  lambda *indices: update_c_tmp(*indices),
                                  name="update_c_gm",
                                  tag="ub_to_out")

    if bias_dtype == 'float16':
        update_c_fp16_back_fake = tvm.compute(shape_i,
                                              lambda *indices: update_c_gm(*indices),
                                              name="update_c_fp16_back_fake",
                                              tag="out_to_ub")
        update_c_fp16_back_fp32_fake = tvm.compute(shape_i,
                                                   lambda *indices: update_c_fp16_back_fake(*indices).astype('float32'),
                                                   name="update_c_fp16_back_fp32_drnn_cast_fake",
                                                   tag="elewise_single_cast")
        c_t_tanh_fake = tvm.compute(shape_i,
                                    lambda *indices: update_c_fp16_back_fp32_fake(*indices) + c_t_tanh(*indices),
                                    name="c_t_tanh_fake",
                                    tag="phony_insn")
    else:
        c_t_tanh_fake = tvm.compute(shape_i,
                                    lambda *indices: c_t_tanh(*indices) + update_c_gm(*indices),
                                    name="c_t_tanh_fake",
                                    tag="phony_insn")

    if is_gate_output:
        c_t_tanh_mid = c_t_tanh_fake

        if fp16_input_output:
            c_t_tanh_fp16 = tvm.compute(shape_i,
                                        lambda *indices: c_t_tanh_fake(*indices).astype('float16'),
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
                                             lambda *indices: c_t_tanh_back(*indices).astype('float32'),
                                             name="c_t_tanh_back_fp32_drnn_cast",
                                             tag="elewise_single_cast")
        c_t_tanh_ub = c_t_tanh_back

        if fp16_input_output and project_gm is None:
            c_t_tanh_ub = c_t_tanh_back_fp32

    update_h = vmul(c_t_tanh_ub, o_t_sigmoid_ub)

    pjc_shape_c = [t_size, state_size, m_size, 16, 16]
    if project_gm is not None:
        shape_dh = [t_size, hidden_size, m_size, 16, 16]
        update_h_mid = update_h
        if not fp16_input_output:
            update_h_mid_fp16 = tvm.compute(shape_dh,
                                            lambda *indices: update_h(*indices).astype('float16'),
                                            name="update_h_fp16_drnn_cast",
                                            tag="elewise_single_cast")
            update_h_mid = update_h_mid_fp16
        pjc_a_l1 = tvm.compute(shape_dh,
                               lambda *indices: update_h_mid(*indices),
                               name='pjc_a_l1',
                               tag="conv_l1fuse_reshape")
        shape_project = [t_size, state_size, hidden_size, 16, 16]
        pjc_b_l1 = tvm.compute(shape_project,
                               lambda *indices: project_gm(*indices),
                               name="pjc_b_l1",
                               tag="out_to_l1")
        shape_pjc_l0a = [t_size, m_size, hidden_size, 16, 16]
        pjc_a_l0a = tvm.compute(shape_pjc_l0a, lambda b, a_i, b_i, c_i, d_i:pjc_a_l1[b, b_i, a_i, c_i, d_i],
                                name="pjc_a_l0a", tag="l1_to_l0")
        shape_pjc_l0b = [t_size, hidden_size, state_size, 16, 16]
        pjc_b_l0b = tvm.compute(shape_pjc_l0b, lambda b, a_i, b_i, c_i, d_i: pjc_b_l1[b, b_i, a_i, d_i, c_i],
                                name="pjc_b_l0b", tag="l1_to_l0")
        pjc_k1 = tvm.reduce_axis((0, hidden_size), name='k1')
        pjc_k0 = tvm.reduce_axis((0, k0_size), name='k0')
        pjc_shape_c = [t_size, state_size, m_size, 16, 16]
        pjc_c_l0c = tvm.compute(pjc_shape_c,
                                lambda t, n, m, mp, np:
                                tvm.sum((pjc_a_l0a[t, m, pjc_k1, mp, pjc_k0] *
                                         pjc_b_l0b[t, pjc_k1, n, np, pjc_k0])
                                        .astype('float32'),
                                        axis=[pjc_k1, pjc_k0]),
                                name='pjc_c_l0c',
                                tag="matmul")
        pjc_c_ub = tvm.compute(pjc_shape_c, lambda *indices: pjc_c_l0c(*indices), name="pjc_c_ub")
        update_h_mad = pjc_c_ub
        project_tensors.append(pjc_a_l1)
        project_tensors.append(pjc_b_l1)
        project_tensors.append(pjc_a_l0a)
        project_tensors.append(pjc_b_l0b)
        project_tensors.append(pjc_c_l0c)
        project_tensors.append(pjc_c_ub)
    else:
        update_h_mad = update_h

    pjc_after_tensors = []
    if mask_gm is not None:
        update_h_tmp1 = vmul(mask_ub_ht_fp32_mid, update_h_mad)
        broadcast_one_dh = broadcast(tvm.const(1, "float16"), [1, state_size, m_size, 16, 16], "float16")
        mask_ub_ht_fp16_mid = mask_ub_ht
        if not fp16_input_output:
            mask_ub_ht_fp16 = tvm.compute(shape_mask_ub_ht,
                                          lambda *indices: mask_ub_ht(*indices).astype('float16'),
                                          name="mask_ub_ht_fp16_drnn_cast",
                                          tag="elewise_single_cast")
            mask_ub_ht_fp16_mid = mask_ub_ht_fp16
            mask_tensors.append(mask_ub_ht_fp16)
            pjc_after_tensors += [mask_ub_ht_fp16]
        one_sub_mth = vsub(broadcast_one_dh, mask_ub_ht_fp16_mid)
        update_h_tmp2 = vmul(one_sub_mth, s_state_h_ub_tmp)
        update_h_tmp2_fp32 = tvm.compute(shape_mask_ub_ht,
                                         lambda *indices: update_h_tmp2(*indices).astype('float32'),
                                         name="update_h_tmp2_fp32_drnn_cast",
                                         tag="elewise_single_cast")
        update_h_tmp = vadd(update_h_tmp1, update_h_tmp2_fp32)
        mask_tensors.append(update_h_tmp1)
        mask_tensors.append(broadcast_one_dh)
        mask_tensors.append(one_sub_mth)
        mask_tensors.append(update_h_tmp2)
        mask_tensors.append(update_h_tmp2_fp32)
        mask_tensors.append(update_h_tmp)
        mask_tensors.append(s_state_h_ub_tmp)
        pjc_after_tensors += [mask_ub_ht, update_h_mad, broadcast_one_dh, one_sub_mth, s_state_h_ub_tmp, update_h_tmp2,
                             update_h_tmp1, update_h_tmp, update_h_tmp2_fp32]
    else:
        update_h_tmp = update_h_mad
        pjc_after_tensors += [update_h_mad, s_state_h_ub_tmp]

    if fp16_input_output:
        update_h_fp16 = tvm.compute(pjc_shape_c,
                                    lambda *indices: update_h_tmp(*indices).astype('float16'),
                                    name="update_h_fp16_drnn_cast",
                                    tag="elewise_single_cast")
        update_h_gm_as_y = tvm.compute(pjc_shape_c,
                                       lambda *indices: update_h_fp16(*indices),
                                       name="update_h_gm_as_y",
                                       tag="ub_to_out")
        update_h_gm_as_y_back = tvm.compute(pjc_shape_c,
                                            lambda *indices: update_h_gm_as_y(*indices),
                                            name="update_h_gm_as_y_back",
                                            tag="out_to_ub")
        update_h_gm = tvm.compute(pjc_shape_c,
                                  lambda *indices: update_h_gm_as_y_back(*indices),
                                  name="update_h_gm",
                                  tag="ub_to_out")
    else:
        update_h_gm_as_y = tvm.compute(pjc_shape_c,
                                       lambda *indices: update_h_tmp(*indices),
                                       name="update_h_gm_as_y",
                                       tag="ub_to_out")
        update_h_gm_as_y_back = tvm.compute(pjc_shape_c,
                                            lambda *indices: update_h_gm_as_y(*indices),
                                            name="update_h_gm_as_y_back",
                                            tag="out_to_ub")
        update_h_fp16_cast = tvm.compute(pjc_shape_c,
                                         lambda *indices: update_h_gm_as_y_back(*indices).astype('float16'),
                                         name="update_h_fp16_cast_drnn_cast",
                                         tag="elewise_single_cast")
        update_h_gm = tvm.compute(pjc_shape_c,
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

    return_list, s = rl_bank.tik_dsl_bank_proc(return_list, sync_tensor=sync0, dynamic=True)
    sch_list, tune_shape_list = s
    index_list = []
    for index, _ in enumerate(tune_shape_list):
        index_list.append(tune_shape_list[index][2])
    if sch_list is not None and len(sch_list) > 0:
        if is_dynamic:
            for index, sch_list_value in enumerate(sch_list):
                sch_list_value.set_constraint( \
                    expr.And(input_x.shape[2] <= tune_shape_list[index][1], input_x.shape[2] > 0))
    else:
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
                        if in_tensor.name in ["s_state_h_ub", "s_state_c_ub", "s_state_h_ub_tmp"]:
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

    s[a_ub].set_scope(scope_ubuf)
    s[c_t_tanh_fake].set_scope(scope_ubuf)
    if bias_dtype == "float16":
        s[update_c_fp16_back_fake].set_scope(scope_ubuf)
        s[update_c_fp16_back_fp32_fake].set_scope(scope_ubuf)

    if wci_gm is not None:
        for tensor in pipehole_tensors:
            s[tensor].set_scope(scope_ubuf)
    if mask_gm is not None:
        for tensor in mask_tensors:
            s[tensor].set_scope(scope_ubuf)
    if project_gm is not None:
        s[pjc_b_l1].set_scope(scope_cbuf)
        s[pjc_a_l1].set_scope(scope_cbuf)
        s[pjc_a_l0a].set_scope(scope_ca)
        s[pjc_b_l0b].set_scope(scope_cb)
        s[pjc_c_l0c].set_scope(scope_cc)
        s[pjc_c_ub].set_scope(scope_ubuf)
        if not fp16_input_output:
            s[update_h_mid_fp16].set_scope(scope_ubuf)
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
            s[j_t_tanh_back_fp32].set_scope(scope_ubuf)
            s[s_state_c_back_fp32].set_scope(scope_ubuf)

        s[f_t_sigmoid_back].set_scope(scope_ubuf)
        s[i_t_sigmoid_back].set_scope(scope_ubuf)
        s[o_t_sigmoid_back].set_scope(scope_ubuf)
        s[j_t_tanh_back].set_scope(scope_ubuf)
        s[c_t_tanh_back].set_scope(scope_ubuf)

    # fp16 in
    if bias_dtype == 'float16':
        s[update_c_fp16].set_scope(scope_ubuf)
        s[update_h_fp16].set_scope(scope_ubuf)
    else:
        s[update_h_fp16_cast].set_scope(scope_ubuf)

    s[update_h_gm_as_y_back].set_scope(scope_ubuf)

    # compute inline
    compute_inline_tensors = [i_t, j_t, f_t, o_t]
    for tensor in compute_inline_tensors:
        s[tensor].compute_inline()

    # matmul tiling
    factor_l1_m, factor_l1_n, factor_l1_k, \
    factor_l0_m, factor_l0_n, factor_l0_k = get_lstm_tiling()

    shapeazbigz = 1 * factor_l1_m * factor_l1_k * 16 * 16
    shapeb = 1 * factor_l1_k * 1 * factor_l1_n * 16 * 16
    shapec = 1 * 4 * factor_l1_n * factor_l1_m * 16 * 16
    shapebias = 1 * 4 * factor_l1_n * 1 * 16 * 16
    shapeh = 1 * factor_l1_k * factor_l1_m * 16 * 16
    shapei = 1 * factor_l1_n * factor_l1_m * 16 * 16
    shapepjc = 1 * factor_l1_k * factor_l1_m * 16 * 16

    l1_n_outer, l1_n_inner = s[c_l0c].split(c_l0c.op.axis[2], factor=factor_l1_n)
    l1_m_outer, l1_m_inner = s[c_l0c].split(c_l0c.op.axis[3], factor=factor_l1_m)
    l1_k_outer, l1_k_inner = s[c_l0c].split(c_l0c.op.reduce_axis[0], factor=factor_l1_k)

    l0_n_outer, l0_n_inner = s[c_l0c].split(l1_n_inner, factor=factor_l0_n)
    l0_m_outer, l0_m_inner = s[c_l0c].split(l1_m_inner, factor=factor_l0_m)
    l0_k_outer, l0_k_inner = s[c_l0c].split(l1_k_inner, factor=factor_l0_k)

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

    ub_n_outer, ub_n_inner = s[c_ub].split(c_ub.op.axis[2], factor=factor_l1_n)
    ub_m_outer, ub_m_inner = s[c_ub].split(c_ub.op.axis[3], factor=factor_l1_m)
    s[c_ub].reorder(ub_m_outer, ub_n_outer, c_ub.op.axis[1],
                    ub_n_inner, ub_m_inner, c_ub.op.axis[4],
                    c_ub.op.axis[5])

    s[c_l0c].compute_at(s[c_ub], ub_n_outer)

    # elewise compute_at
    barrier_outer, barrier_inner = s[barrier_tensor].split(barrier_tensor.op.axis[2], factor=factor_l1_n)
    barrier_m_outer, barrier_m_inner = s[barrier_tensor].split(barrier_tensor.op.axis[3], factor=factor_l1_m)
    s[barrier_tensor].reorder(
        barrier_tensor.op.axis[0], barrier_m_outer, barrier_outer,
        barrier_tensor.op.axis[1], barrier_inner, barrier_m_inner,
        barrier_tensor.op.axis[4],
        barrier_tensor.op.axis[5])
    if project_gm is not None:
        pjc_l1_n_outer, pjc_l1_n_inner = s[pjc_c_l0c].split(pjc_c_l0c.op.axis[1], factor=factor_l1_n)
        pjc_l1_m_outer, pjc_l1_m_inner = s[pjc_c_l0c].split(pjc_c_l0c.op.axis[2], factor=factor_l1_m)
        pjc_l1_k_outer, pjc_l1_k_inner = s[pjc_c_l0c].split(pjc_c_l0c.op.reduce_axis[0], factor=factor_l1_k)
        pjc_l0_n_outer, pjc_l0_n_inner = s[pjc_c_l0c].split(pjc_l1_n_inner, factor=factor_l0_n)
        pjc_l0_m_outer, pjc_l0_m_inner = s[pjc_c_l0c].split(pjc_l1_m_inner, factor=factor_l0_m)
        pjc_l0_k_outer, pjc_l0_k_inner = s[pjc_c_l0c].split(pjc_l1_k_inner, factor=factor_l0_k)
        s[pjc_c_l0c].reorder(pjc_l1_n_outer, pjc_l1_m_outer, pjc_l1_k_outer,
                             pjc_l0_n_outer, pjc_l0_m_outer, pjc_l0_k_outer,
                             pjc_l0_n_inner, pjc_l0_m_inner, pjc_c_l0c.op.axis[3],
                             pjc_c_l0c.op.axis[4], pjc_l0_k_inner,
                             pjc_c_l0c.op.reduce_axis[1])
        s[pjc_a_l0a].compute_at(s[pjc_c_l0c], pjc_l0_k_outer)
        s[pjc_b_l0b].compute_at(s[pjc_c_l0c], pjc_l0_k_outer)
        s[pjc_a_l1].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
        s[pjc_b_l1].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)

        pjc_ub_n_outer, pjc_ub_n_inner = s[pjc_c_ub].split(pjc_c_ub.op.axis[1], factor=factor_l1_n)
        pjc_ub_m_outer, pjc_ub_m_inner = s[pjc_c_ub].split(pjc_c_ub.op.axis[2], factor=factor_l1_m)
        s[pjc_c_ub].reorder(pjc_ub_m_outer, pjc_ub_n_outer,
                            pjc_ub_n_inner, pjc_ub_m_inner,
                            pjc_c_ub.op.axis[3], pjc_c_ub.op.axis[4])
        s[pjc_c_l0c].compute_at(s[pjc_c_ub], pjc_ub_n_outer)
        if not fp16_input_output:
            s[update_h_mid_fp16].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)

    s[c_ub].compute_at(s[barrier_tensor], barrier_outer)
    s[bias_ub].compute_at(s[barrier_tensor], barrier_outer)
    if fp16_input_output:
        s[bias_ub_fp32].compute_at(s[barrier_tensor], barrier_outer)

    for tensor in elewise_before_barrier_tensors:
        s[tensor].compute_at(s[barrier_tensor], barrier_outer)

    vn_outer, vn_inner = s[update_h_gm].split(update_h_gm.op.axis[0 + 1], factor=factor_l1_n)
    vn_m_outer, vn_m_inner = s[update_h_gm].split(update_h_gm.op.axis[0 + 2], factor=factor_l1_m)
    second_split_factor = (hidden_size // factor_l1_n) // 1

    vn_o_outer, vn_o_inner = s[update_h_gm].split(vn_outer, factor=second_split_factor)
    s[update_h_gm].reorder(update_h_gm.op.axis[0], vn_m_outer,
                           vn_o_outer, vn_o_inner, vn_inner,
                           vn_m_inner, update_h_gm.op.axis[3],
                           update_h_gm.op.axis[4])

    if mask_gm is not None:
        if project_gm is not None:
            s[s_state_c_ub].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[mask_ub_ct].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            if fp16_input_output:
                s[mask_ub_ct_fp32].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
                s[update_c_tmp2_fp32].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
        else:
            s[s_state_c_ub].compute_at(s[update_h_gm], vn_o_inner)
            s[mask_ub_ct].compute_at(s[update_h_gm], vn_o_inner)
            if fp16_input_output:
                s[mask_ub_ct_fp32].compute_at(s[update_h_gm], vn_o_inner)
                s[update_c_tmp2_fp32].compute_at(s[update_h_gm], vn_o_inner)

        s[s_state_h_ub_tmp].compute_at(s[update_h_gm], vn_o_inner)
        s[mask_ub_ht].compute_at(s[update_h_gm], vn_o_inner)
        s[update_h_tmp2_fp32].compute_at(s[update_h_gm], vn_o_inner)
        if fp16_input_output:
            s[mask_ub_ht_fp32].compute_at(s[update_h_gm], vn_o_inner)
        else:
            s[mask_ub_ht_fp16].compute_at(s[update_h_gm], vn_o_inner)
    else:
        s[s_state_c_ub].compute_at(s[update_h_gm], vn_o_inner)

    if wci_gm is not None and project_gm is not None:
        s[wcf_ub].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
        s[wci_ub].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
        s[wco_ub].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
        if fp16_input_output:
            s[wcf_ct_mul_fp32].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[wci_ct_mul_fp32].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[wco_ub_fp32].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
    elif wci_gm is not None:
        s[wcf_ub].compute_at(s[update_h_gm], vn_o_inner)
        s[wci_ub].compute_at(s[update_h_gm], vn_o_inner)
        s[wco_ub].compute_at(s[update_h_gm], vn_o_inner)
        if fp16_input_output:
            s[wcf_ct_mul_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[wci_ct_mul_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[wco_ub_fp32].compute_at(s[update_h_gm], vn_o_inner)
    if project_gm is not None:
        s[barrier_tensor].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
    else:
        s[barrier_tensor].compute_at(s[update_h_gm], vn_o_inner)
    elewise_after_tensors = []
    if project_gm is not None:
        if mask_gm is not None:
            pjc_after_tensors.append(update_h_tmp2)
            pjc_after_tensors.append(update_h_tmp.op.input_tensors[1])
            pjc_after_tensors.append(update_h_tmp1)
            pjc_after_tensors.append(update_h_tmp1.op.input_tensors[0])
            pjc_after_tensors.append(update_h_tmp1.op.input_tensors[1])
            pjc_after_tensors.append(one_sub_mth.op.input_tensors[1])
        for tensor in elewise_tensors:
            if tensor not in elewise_before_barrier_tensors and tensor not in pjc_after_tensors:
                s[tensor].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            else:
                elewise_after_tensors.append(tensor)
        s[c_t_tanh_fake].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
        if bias_dtype == 'float16':
            s[update_c_fp16_back_fake].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[update_c_fp16_back_fp32_fake].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
        s[update_c_gm].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
        if bias_dtype == 'float16':
            s[update_c_fp16].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[update_h_fp16].compute_at(s[update_h_gm], vn_o_inner)
        else:
            s[update_c].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[update_h_fp16_cast].compute_at(s[update_h_gm], vn_o_inner)

        s[update_h_gm_as_y].compute_at(s[update_h_gm], vn_o_inner)
        s[update_h_gm_as_y_back].compute_at(s[update_h_gm], vn_o_inner)

        if is_gate_output:
            if fp16_input_output:
                s[f_t_sigmoid_fp16].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
                s[i_t_sigmoid_fp16].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
                s[o_t_sigmoid_fp16].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
                s[j_t_tanh_fp16].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
                s[c_t_tanh_fp16].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
                s[f_t_sigmoid_back_fp32].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
                s[i_t_sigmoid_back_fp32].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
                s[j_t_tanh_back_fp32].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
                s[s_state_c_back_fp32].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)

            s[f_t_sigmoid_gm].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[i_t_sigmoid_gm].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[o_t_sigmoid_gm].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[j_t_tanh_gm].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[c_t_tanh_gm].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)

            s[f_t_sigmoid_back].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[i_t_sigmoid_back].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[o_t_sigmoid_back].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[j_t_tanh_back].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)
            s[c_t_tanh_back].compute_at(s[pjc_c_l0c], pjc_l1_k_outer)

        if is_gate_output:
            s[f_t_sigmoid].reused_by(f_t_sigmoid_ub)
            s[i_t_sigmoid].reused_by(i_t_sigmoid_ub)
            s[o_t_sigmoid_mid].reused_by(o_t_sigmoid_ub)
            s[j_t_tanh].reused_by(j_t_tanh_ub)

            s[f_t_sigmoid_ub].reused_by(reuse_data=True)
            s[i_t_sigmoid_ub].reused_by(reuse_data=True)
            s[o_t_sigmoid_ub].reused_by(reuse_data=True)
            s[j_t_tanh_ub].reused_by(reuse_data=True)
        s[pjc_c_ub].compute_at(s[update_h_gm], vn_o_inner)
        for tensor in elewise_after_tensors:
            if tensor not in elewise_before_barrier_tensors:
                s[tensor].compute_at(s[update_h_gm], vn_o_inner)
    else:
        if fp16_input_output:
            s[c_t_tanh_back_fp32].set_scope(scope_ubuf)
            s[o_t_sigmoid_back_fp32].set_scope(scope_ubuf)
        for tensor in elewise_tensors:
            s[tensor].compute_at(s[update_h_gm], vn_o_inner)

        s[c_t_tanh_fake].compute_at(s[update_h_gm], vn_o_inner)
        if bias_dtype == 'float16':
            s[update_c_fp16_back_fake].compute_at(s[update_h_gm], vn_o_inner)
            s[update_c_fp16_back_fp32_fake].compute_at(s[update_h_gm], vn_o_inner)

        s[update_c_gm].compute_at(s[update_h_gm], vn_o_inner)

        # fp16 in
        if bias_dtype == 'float16':
            s[update_c_fp16].compute_at(s[update_h_gm], vn_o_inner)
            s[update_h_fp16].compute_at(s[update_h_gm], vn_o_inner)
        else:
            s[update_c].compute_at(s[update_h_gm], vn_o_inner)
            s[update_h_fp16_cast].compute_at(s[update_h_gm], vn_o_inner)

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
                s[s_state_c_back_fp32].compute_at(s[update_h_gm], vn_o_inner)

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

            s[f_t_sigmoid_ub].reused_by(reuse_data=True)
            s[i_t_sigmoid_ub].reused_by(reuse_data=True)
            s[o_t_sigmoid_ub].reused_by(reuse_data=True)
            s[j_t_tanh_ub].reused_by(reuse_data=True)
    if bias_dtype == 'float16':
        s[update_h_fp16].reused_by(update_h_gm_as_y_back)
        s[update_h_gm_as_y_back].reused_by(reuse_data=True)
    else:
        s[update_h_tmp].reused_by(update_h_gm_as_y_back)
        s[update_h_gm_as_y_back].reused_by(reuse_data=True)

    s[c_t_tanh].reused_by(c_t_tanh_fake)
    s[c_t_tanh_fake].reused_by(reuse_data=True)

    if is_gate_output:
        if project_gm is not None:
            s[c_t_tanh_mid].reused_by(c_t_tanh_ub)
        else:
            s[c_t_tanh_fake].reused_by(c_t_tanh_ub)
        s[c_t_tanh_ub].reused_by(reuse_data=True)
    # emit_insn
    s[a_l1].emit_insn(a_l1.op.axis[0], 'dma_copy')
    s[b_l1].emit_insn(b_l1.op.axis[0], 'dma_copy')
    s[a_l0a].emit_insn(a_l0a.op.axis[0], 'dma_copy')
    s[b_l0b].emit_insn(b_l0b.op.axis[0], 'dma_copy')

    s[a_ub].emit_insn(a_ub.op.axis[0], 'dma_copy')
    if wci_gm is not None:
        s[wci_ub].emit_insn(wci_ub.op.axis[0], 'dma_copy')
        s[wco_ub].emit_insn(wco_ub.op.axis[0], 'dma_copy')
        s[wcf_ub].emit_insn(wcf_ub.op.axis[0], 'dma_copy')
        if fp16_input_output:
            s[wcf_ct_mul_fp32].emit_insn(wcf_ct_mul_fp32.op.axis[0], 'vector_conv')
            s[wci_ct_mul_fp32].emit_insn(wci_ct_mul_fp32.op.axis[0], 'vector_conv')
            s[wco_ub_fp32].emit_insn(wco_ub_fp32.op.axis[0], 'vector_conv')

    if mask_gm is not None:
        s[mask_ub_ct].emit_insn(mask_ub_ct.op.axis[0], 'dma_copy')
        s[mask_ub_ht].emit_insn(mask_ub_ht.op.axis[0], 'dma_copy')
        if is_first_round:
            if is_global_init:
                s[s_state_h_ub_tmp].emit_insn(s_state_h_ub_tmp.op.axis[0], 'dma_copy')
            else:
                s[s_state_h_ub_tmp].emit_insn(s_state_h_ub_tmp.op.axis[0], 'vector_broadcast')
        else:
            s[s_state_h_ub_tmp].emit_insn(s_state_h_ub_tmp.op.axis[0], 'dma_copy')
        s[broadcast_one].emit_insn(broadcast_one.op.axis[0], 'vector_broadcast')
        s[broadcast_one_dh].emit_insn(broadcast_one_dh.op.axis[0], 'vector_broadcast')
        s[update_h_tmp2_fp32].emit_insn(update_h_tmp2_fp32.op.axis[0], 'vector_conv')
        if fp16_input_output:
            s[mask_ub_ct_fp32].emit_insn(mask_ub_ct_fp32.op.axis[0], 'vector_conv')
            s[mask_ub_ht_fp32].emit_insn(mask_ub_ht_fp32.op.axis[0], 'vector_conv')
            s[update_c_tmp2_fp32].emit_insn(update_c_tmp2_fp32.op.axis[0], 'vector_conv')
        else:
            s[mask_ub_ht_fp16].emit_insn(mask_ub_ht_fp16.op.axis[0], 'vector_conv')

    if project_gm is not None:
        s[pjc_a_l1].emit_insn(pjc_a_l1.op.axis[0], 'dma_copy')
        s[pjc_b_l1].emit_insn(pjc_b_l1.op.axis[0], 'dma_copy')
        s[pjc_a_l0a].emit_insn(pjc_a_l0a.op.axis[0], 'dma_copy')
        s[pjc_b_l0b].emit_insn(pjc_b_l0b.op.axis[0], 'dma_copy')
        pjc_mad_dict = {"mad_pattern": 0, "k_outer": [pjc_l1_k_outer, pjc_l0_k_outer]}
        s[pjc_c_l0c].emit_insn(pjc_l0_n_inner, 'mad', pjc_mad_dict)
        s[pjc_c_ub].emit_insn(pjc_ub_n_inner, 'dma_copy')
        if not fp16_input_output:
            s[update_h_mid_fp16].emit_insn(update_h_mid_fp16.op.axis[0], 'vector_conv')

    if fp16_input_output:
        s[bias_ub_fp32].emit_insn(bias_ub_fp32.op.axis[0], 'vector_conv')

    mad_dict = {"mad_pattern": 0, "k_outer": [l1_k_outer, l0_k_outer]}
    s[c_l0c].emit_insn(l0_n_inner, 'mad', mad_dict)
    s[c_ub].emit_insn(ub_n_inner, 'dma_copy')

    s[bias_bc_ub].emit_insn(bias_bc_ub.op.axis[0], 'vector_broadcast', attrs={"storage_bound": 16384})
    if is_first_round:
        if is_global_init:
            s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'dma_copy')
            s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'dma_copy')
        else:
            s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'vector_broadcast')
            s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'vector_broadcast')
    else:
        s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'dma_copy')
        s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'dma_copy')
  
    s[barrier_tensor].emit_insn(barrier_tensor.op.axis[1], 'vector_add')

    for tensor in elewise_tensors:
        if tensor != barrier_tensor:
            insn = get_emit_insn_map(tensor)
            if tensor.op.name != "c_t_tanh_fake":
                s[tensor].emit_insn(tensor.op.axis[0], insn)

    s[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')

    s[update_c_gm].emit_insn(s[update_c_gm].op.axis[1], 'dma_copy')
    s[update_h_gm].emit_insn(s[update_h_gm].op.axis[3], 'dma_copy')

    if is_gate_output:
        if fp16_input_output:
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
            s[j_t_tanh_back_fp32].emit_insn(s[j_t_tanh_back_fp32].op.axis[1],
                                            'phony_insn')
            s[s_state_c_back_fp32].emit_insn(s[s_state_c_back_fp32].op.axis[1],
                                            'vector_conv')
            if project_gm is None and fp16_input_output:
                s[c_t_tanh_back_fp32].emit_insn(s[c_t_tanh_back_fp32].op.axis[1], 'phony_insn')
                s[o_t_sigmoid_back_fp32].emit_insn(s[o_t_sigmoid_back_fp32].op.axis[1], 'phony_insn')

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

    s[c_t_tanh_fake].emit_insn(s[c_t_tanh_fake].op.axis[0], 'phony_insn')
    # fp16 in
    if bias_dtype == 'float16':
        s[update_c_fp16].emit_insn(update_c_fp16.op.axis[0], 'vector_conv')
        s[update_c_fp16_back_fake].emit_insn(s[update_c_fp16_back_fake].op.axis[0], 'phony_insn')
        s[update_c_fp16_back_fp32_fake].emit_insn(s[update_c_fp16_back_fp32_fake].op.axis[0], 'phony_insn')
        s[update_h_fp16].emit_insn(update_h_fp16.op.axis[0], 'vector_conv')
    else:
        s[update_h_fp16_cast].emit_insn(update_h_fp16_cast.op.axis[0],
                                        'vector_conv')

    s[update_h_gm_as_y].emit_insn(update_h_gm_as_y.op.axis[0], 'dma_copy')
    s[update_h_gm_as_y_back].emit_insn(update_h_gm_as_y_back.op.axis[0],
                                       'phony_insn')

    default_index = 0
    if index_list is not None and len(index_list) > 0:
        default_index = index_list[-1] + 1
    tune_shape_list.append([-1, -1, default_index])
    tbe_context.get_context().add_compile_info("vars", {"tune_shape_list": tune_shape_list})
    sch_list.append(s)
    index_list.append(default_index)
    return return_list, sch_list, index_list
