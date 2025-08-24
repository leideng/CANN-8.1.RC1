"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

dynamic_rnn_v2
"""
# 'pylint: disable=too-many-lines
import copy
import operator
from collections import namedtuple
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from tbe.common.buildcfg.default_buildcfg import dynamic_build_config_dict
from tbe.common.rl_bank import rl_bank
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
from tbe.common.rl_bank import bank_manager
from tbe.common.platform import scope_ca
from tbe.common.platform import scope_cb
from tbe.common.platform import scope_cbuf
from tbe.common.platform import scope_cc
from tbe.common.platform import scope_ubuf
from tbe.common.platform.platform_info import api_check_support
from tbe.common.register import register_param_generalization
from tbe.tik import Dprofile
from tbe.tik import Tik
from tbe.tik import scope_gm
from tbe import tvm
from tbe.tvm.tir import expr
from tbe.tvm import create_schedule
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


def sigmoid_compute(input_x):
    """
    calculating sigmoid
    """
    data_input = input_x
    dtype = input_x.dtype
    exp_support = api_check_support("tbe.dsl.vexp", "float32")
    mul_support = api_check_support("tbe.dsl.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
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


def hard_sigmoid_compute(input_x):
    """
    calculating hard sigmoid
    input_x dtype is float32
    """
    dtype = input_x.dtype
    one_const = tvm.const(1, dtype)
    zero_const = tvm.const(0, dtype)
    alpha_x = vmuls(input_x, tvm.const(0.2, input_x.dtype))
    alpha_x_beta = vadds(alpha_x, tvm.const(0.5, input_x.dtype))
    vmin_res = vmins(alpha_x_beta, one_const)
    vmax_res = vmaxs(vmin_res, zero_const)
    return vmax_res


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
    exp_support = api_check_support("tbe.dsl.vexp", "float32")
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


def clip_compute(input_x):
    """
    algorithm: clip
    """
    dtype = input_x.dtype
    one_const = tvm.const(1, dtype)
    negative_one_const = tvm.const(-1, dtype)
    vmin_res = vmins(input_x, one_const)
    vmax_res = vmaxs(vmin_res, negative_one_const)
    return vmax_res


def activation_compute(activation, input_x):
    """
    activation compute
    """
    if activation == "clip":
        res = clip_compute(input_x)
    else:
        res = tanh_compute(input_x)
    return res


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
                "elewise_binary_cmpsel_gt": "vector_select_gt",
                "elewise_binary_cmpsel_ge": "vector_select_ge",
                "elewise_binary_or": "vector_or",
                "elewise_binary_and": "vector_and",
                "elewise_multiple_mla": "vector_multiple",
                "elewise_multiple_madd": "vector_multiple",
                "elewise_multiple_maddrelu": "vector_multiple",
                "broadcast_for_tensor": "broadcast_for_tensor",
                "elewise_binary_sub": "vector_sub",
                "elewise_multiple_sel": "vector_select_bool",
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
    Tiling = namedtuple("Tiling", ["factor_l1_m", "factor_l1_n", "factor_l1_k_1", "factor_l1_k_2",
                                   "factor_l0_m", "factor_l0_n", "factor_l0_k_1", "factor_l0_k_2"])
    tiling = Tiling(1, 1, 12, 12, 1, 1, 12, 12)
    return tiling


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name,invalid-name
def check_param_dtype(input_x, weight_input, weight_hidden, bias, init_h, init_c, y, output_h,
                      output_c, i, j, f, o, tanhc):
    """
    check parameters dtype
    :return:
    """

    x_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(x_dtype, ["float16"], param_name="x")

    w_i_dtype = weight_input.get("dtype").lower()
    para_check.check_dtype(w_i_dtype, ["float16"], param_name="weight_input")

    w_h_dtype = weight_hidden.get("dtype").lower()
    para_check.check_dtype(w_h_dtype, ["float16"], param_name="weight_hidden")

    output_h_dtype = output_h.get("dtype").lower()
    para_check.check_dtype(output_h_dtype, ["float16"], param_name="output_h")

    if init_h is not None:
        init_h_dtype = init_h.get("dtype").lower()
        para_check.check_dtype(init_h_dtype, ["float16"], param_name="init_h")

    # check optional bias input
    if bias is not None:
        bias_dtype = bias.get("dtype").lower()
        para_check.check_dtype(bias_dtype, ["float16", "float32"], param_name="bias")

    # check optional input
    if init_c is not None:
        init_c_dtype = init_c.get("dtype").lower()
        para_check.check_dtype(init_c_dtype, ["float16", "float32"], param_name="init_c")

    # check output
    y_dtype = y.get("dtype").lower()
    para_check.check_dtype(y_dtype, ["float16", "float32"], param_name="y")

    if output_c["dtype"] != y_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "output_c dtype is not the same as y dtype !")

    # check additional output
    if i["dtype"] != y_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "i dtype is not the same as y dtype !")
    if j["dtype"] != y_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "j dtype is not the same as y dtype !")
    if f["dtype"] != y_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "f dtype is not the same as y dtype !")
    if o["dtype"] != y_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "o dtype is not the same as y dtype !")
    if tanhc["dtype"] != y_dtype:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "tanhc dtype is not the same as y dtype !")


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals,invalid-name,invalid-name,line-too-long
def check_param_shape(input_x, weight_input, weight_hidden, bias, seq_length, init_h, init_c,
                      wci, wcf, wco, mask, y, output_h, output_c, i, j, f, o,
                      tanhc):
    """
    check parameters
    """
    if weight_input["shape"][0] != input_x["shape"][1]:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "x, weight_input shape is wrong, please check!")

    if weight_hidden["shape"][0] != output_h["shape"][1]:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "weight_hidden, output_h shape is wrong, please check!")

    # weight_input dim check
    if weight_input["shape"][1] != 4 * output_h["shape"][1]:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "weight_input, output_h shape is wrong, please check!")

    # weight_hidden dim check
    if weight_hidden["shape"][1] != 4 * output_h["shape"][1]:
        error_manager_vector.raise_err_specific_reson(
            "DynamicRNNV2", "weight_hidden, output_h shape is wrong, please check!")

    if bias is not None:
        if (bias["shape"][0] + 15) // 16 != weight_input["shape"][1]:
            error_manager_vector.raise_err_specific_reson("DynamicRNNV2", "w, b shape is wrong, please check!")

    # check init
    if (init_h is None and init_c is not None) or (init_h is not None and init_c is None):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "init_h, init_c should appear together, please check!")

    # check init state should support one time
    if init_h is not None:
        if init_h["shape"][0] != 1:
            error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                          "init_h only support 1 time, please check!")

        if init_h["shape"][1] != output_h["shape"][1] or init_h["shape"][2] != output_h["shape"][2]:
            error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                          "init_h, output_h shape is different, please check!")

        if not operator.eq(init_h["shape"], init_c["shape"]):
            error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                          "init_c, init_h shape is different, please check!")
    # check output
    if y["shape"][1] != output_h["shape"][1] or y["shape"][2] != output_h["shape"][2]:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "y, output_h shape is wrong, please check!")

    if not operator.eq(output_h["shape"], output_c["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "output_c, output_h shape is different, please check!")

    if not operator.eq(y["shape"], i["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "i, output_h shape is different, please check!")

    if not operator.eq(y["shape"], j["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "i, output_h shape is different, please check!")

    if not operator.eq(y["shape"], f["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "f, output_h shape is different, please check!")

    if not operator.eq(y["shape"], o["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "o, output_h shape is different, please check!")

    if not operator.eq(y["shape"], tanhc["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "tanhc, output_h shape is different, please check!")

    if not operator.eq(y["shape"], y["shape"]):
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2",
                                                      "y, output_h shape is different, please check!")

    # check unsupport pramas
    if wci is not None:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2", "wci only support None, please check!")

    if wcf is not None:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2", "wcf only support None, please check!")

    if wco is not None:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2", "wco only support None, please check!")

    if mask is not None:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2", "mask only support None, please check!")


# 'pylint: disable=too-many-arguments,too-many-branches,too-many-locals
def check_attr(cell_type, direction, cell_depth, use_peephole, keep_prob,
               cell_clip, num_proj, time_major, activation, recurrent_activation, gate_order,
               stateful, merge_mode):
    """
    check parameters
    """
    if cell_type not in ["LSTM", "GRU", "RNN_RELU", "RNN_TANH"]:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "cell_type in ['LSTM', 'GRU', 'RNN_RELU', 'RNN_TANH']",
                                                          "cell_type", str(cell_type))
    if direction not in ["UNIDIRECTIONAL", "BIDIRECTIONAL"]:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "direction in ['UNIDIRECTIONAL', 'BIDIRECTIONAL']",
                                                          "direction", str(direction))

    if cell_depth != 1:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "cell_depth == 1",
                                                          "cell_depth", str(cell_depth))

    if use_peephole is not False:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "use_peephole is not False",
                                                          "use_peephole", str(use_peephole))

    if keep_prob != 1.0:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "keep_prob == 1.0",
                                                          "keep_prob", str(keep_prob))

    if cell_clip != -1.0:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "cell_clip == -1.0",
                                                          "cell_clip", str(cell_clip))
    if num_proj != 0:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "num_proj == 0",
                                                          "num_proj", str(num_proj))

    if time_major is not True:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "time_major is not True",
                                                          "time_major", str(time_major))

    if activation not in ["tanh", "clip"]:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "activation in ['tanh', 'clip']",
                                                          "activation", str(activation))

    if recurrent_activation not in ["sigmoid", "hard_sigmoid"]:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "recurrent_activation in ['sigmoid', 'hard_sigmoid']",
                                                          "recurrent_activation", str(recurrent_activation))

    if gate_order not in ["ijfo", "ifco"]:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "gate_order in ['ijfo', 'ifco']",
                                                          "gate_order", str(gate_order))

    if stateful is not False:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "stateful is not False",
                                                          "stateful", str(stateful))

    if merge_mode not in ["concat", "add"]:
        error_manager_vector.raise_err_check_params_rules("DynamicRNNV2",
                                                          "merge_mode in ['concat', 'add']",
                                                          "merge_mode", str(merge_mode))


def register_generalization(rnn_op_params):
    """
    do generalization register
    """
    input_x, weight_input, weight_hidden, bias, seq_length, init_h, init_c, wci, wcf, wco, mask, \
        y, output_h, output_c, i, j, f, o, tanhc, cell_type, direction, cell_depth, use_peephole, \
        keep_prob, cell_clip, num_proj, time_major, activation, recurrent_activation, forget_bias, \
        gate_order, stateful, merge_mode, is_training = rnn_op_params
    shape_input = [-2]
    range_input_x = [(1, -1), (1, -1), (1, -1)]
    range_weight = [(1, -1), (1, -1)]

    shape_output = [-2]
    range_output = [(1, -1), (1, -1), (1, -1)]
    input_x["shape"], input_x["range"] = shape_input, range_input_x
    weight_input["shape"], weight_input["range"] = shape_input, range_weight
    weight_hidden["shape"], weight_hidden["range"] = shape_input, range_weight

    y["shape"], y["range"] = shape_output, range_output
    output_h["shape"], output_h["range"] = shape_output, range_output
    output_c["shape"], output_c["range"] = shape_output, range_output
    i["shape"], i["range"] = shape_output, range_output
    j["shape"], j["range"] = shape_output, range_output
    f["shape"], f["range"] = shape_output, range_output
    o["shape"], o["range"] = shape_output, range_output
    tanhc["shape"], tanhc["range"] = shape_output, range_output

    if bias is not None:
        range_bias = [(1, -1)]
        bias["shape"], bias["range"] = shape_input, range_bias

    if seq_length is not None:
        range_seq_length = [(1, -1), (1, -1), (1, -1)]
        seq_length["shape"], seq_length["range"] = shape_input, range_seq_length

    if (init_h is not None) and (init_c is not None):
        range_init = [(1, -1), (1, -1), (1, -1)]
        init_h["shape"], init_h["range"] = shape_input, range_init
        init_c["shape"], init_c["range"] = shape_input, range_init

    result = []
    result.append([input_x, weight_input, weight_hidden, bias, seq_length, init_h, init_c, wci, wcf, wco, mask,
                   y, output_h, output_c, i, j, f, o, tanhc, cell_type, direction, cell_depth, use_peephole, 
                   keep_prob, cell_clip, num_proj, time_major, activation, recurrent_activation, forget_bias, gate_order,
                   stateful, merge_mode, is_training])
    return result


@register_param_generalization("DynamicRNNV2")
def dynamic_rnn_v2_generalization(input_x, weight_input, weight_hidden, bias, seq_length, init_h, init_c,
                                  wci, wcf, wco, mask, y, output_h, output_c, i, j, f, o, tanhc,
                                  cell_type="LSTM", direction="UNIDIRECTIONAL", cell_depth=1,
                                  use_peephole=False, keep_prob=1.0, cell_clip=-1.0, num_proj=0,
                                  time_major=True, activation="tanh", recurrent_activation="sigmoid",
                                  forget_bias=0.0, gate_order="ijfo", stateful=False, merge_mode="concat",
                                  is_training=True, kernel_name="dynamic_rnn_v2", generalize_config=None):
                                  
    """
    only T, b support -1
    input:
    input_x (T, b, input_size)
    weight_input (input_size, hidden_size * 4)
    weight_hidden (hidden_size, hidden_size * 4)

    bias (hidden_size * 4)  optional input
    seq_length (T, b, hidden_size)  optional input
    init_h (1, b, hidden_size)  optional input
    init_c (1, b, hidden_size)  optional input

    output:
    y (T, b, hidden_size)
    output_h (T, b, hidden_size)
    output_c (T, b, hidden_size)
    i (T, b, hidden_size)
    j (T, b, hidden_size)
    f (T, b, hidden_size)
    o (T, b, hidden_size)
    tanhc (T, b, hidden_size)
    """
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if support_l0c2out and generalize_config["mode"] == "all_shape":
        result = register_generalization([input_x, weight_input, weight_hidden, bias, seq_length, init_h, init_c,
                                          wci, wcf, wco, mask, y, output_h, output_c, i, j, f, o, tanhc, cell_type,
                                          direction, cell_depth, use_peephole, keep_prob, cell_clip, num_proj,
                                          time_major, activation, recurrent_activation, forget_bias, gate_order,
                                          stateful, merge_mode, is_training])
        return result

    if len(input_x["ori_shape"]) != 3:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2", "input_x's shape illegal, please check!")
    if len(weight_input["ori_shape"]) != 2:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2", "weight_input's shape illegal, please check!")
    if len(weight_hidden["ori_shape"]) != 2:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2", "weight_hidden's shape illegal, please check!")

    param_input_size = weight_input["ori_shape"][0]
    param_hidden_size = weight_hidden["ori_shape"][0]

    range_input_x = [(1, -1), (1, -1), (param_input_size, param_input_size)]
    shape_input_x = (-1, -1, param_input_size)

    range_weight_input = [(weight_input["ori_shape"][0], weight_input["ori_shape"][0]),
                          (weight_input["ori_shape"][1], weight_input["ori_shape"][1])]
    shape_weight_input = (weight_input["ori_shape"][0], weight_input["ori_shape"][1])

    range_weight_hidden = [(weight_hidden["ori_shape"][0], weight_hidden["ori_shape"][0]),
                           (weight_hidden["ori_shape"][1], weight_hidden["ori_shape"][1])]
    shape_weight_hidden = (weight_hidden["ori_shape"][0], weight_hidden["ori_shape"][1])

    range_output = [(1, -1), (1, -1), (param_hidden_size, param_hidden_size)]
    shape_output = (-1, -1, param_hidden_size)

    input_x["ori_shape"], input_x["ori_range"] = shape_input_x, range_input_x
    weight_input["ori_shape"], weight_input["ori_range"] = shape_weight_input, range_weight_input
    weight_hidden["ori_shape"], weight_hidden["ori_range"] = shape_weight_hidden, range_weight_hidden

    y["ori_shape"], y["ori_range"] = shape_output, range_output
    output_h["ori_shape"], output_h["ori_range"] = shape_output, range_output
    output_c["ori_shape"], output_c["ori_range"] = shape_output, range_output
    i["ori_shape"], i["ori_range"] = shape_output, range_output
    j["ori_shape"], j["ori_range"] = shape_output, range_output
    f["ori_shape"], f["ori_range"] = shape_output, range_output
    o["ori_shape"], o["ori_range"] = shape_output, range_output
    tanhc["ori_shape"], tanhc["ori_range"] = shape_output, range_output

    if bias is not None:
        range_bias = [(bias["ori_shape"][0], bias["ori_shape"][0])]
        shape_bias = (bias["ori_shape"][0],)
        bias["ori_shape"], bias["ori_range"] = shape_bias, range_bias

    if seq_length is not None:
        range_seq_length = [(1, -1), (1, -1), (param_hidden_size, param_hidden_size)]
        shape_seq_length = (-1, -1, param_hidden_size)
        seq_length["ori_shape"], seq_length["ori_range"] = shape_seq_length, range_seq_length

    if init_h is not None:
        range_init = [(1, 1), (1, -1), (param_hidden_size, param_hidden_size)]
        shape_init = (1, -1, param_hidden_size)
        init_h["ori_shape"], init_h["ori_range"] = shape_init, range_init
        init_c["ori_shape"], init_c["ori_range"] = shape_init, range_init

    result = []
    result.append([input_x, weight_input, weight_hidden, bias, seq_length, init_h, init_c,
                   wci, wcf, wco, mask, y, output_h, output_c, i, j, f, o, tanhc,
                   {"cell_type": cell_type}, {"direction": direction}, {"cell_depth": cell_depth},
                   {"use_peephole": use_peephole}, {"keep_prob": keep_prob}, {"cell_clip": cell_clip},
                   {"num_proj": num_proj}, {"time_major": time_major}, {"activation": activation},
                   {"recurrent_activation": recurrent_activation}, {"forget_bias": forget_bias},
                   {"gate_order": gate_order}, {"stateful": stateful}, {"merge_mode": merge_mode},
                   {"is_training": is_training}])

    return result


@register_operator("DynamicRNNV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-function-args,too-many-statements
# 'pylint: disable=unused-argument
def dynamic_rnn_v2(input_x, weight_input, weight_hidden, bias, seq_length, init_h, init_c, wci, wcf,
                   wco, mask, y, output_h, output_c, i, j, f, o, tanhc,
                   cell_type="LSTM", direction="UNIDIRECTIONAL", cell_depth=1,
                   use_peephole=False, keep_prob=1.0, cell_clip=-1.0, num_proj=0,
                   time_major=True, activation="tanh", recurrent_activation="sigmoid",
                   forget_bias=0.0, gate_order="ijfo", stateful=False, merge_mode="concat",
                   is_training=True, kernel_name="dynamic_rnn_v2"):
    """
    dynamic_rnn_v2

    Parameters:
    ----------
    input_x : dict
        A dict object, contains a Tensor 's type and shape and format
    weight_input : dict
                   A dict object, contains a Tensor 's type and shape and format
    weight_hidden: dict
                   A dict object, contains a Tensor 's type and shape and format
    bias : dict
        A dict object, contains a Tensor 's type and shape and format
    seq_length : dict
                 A dict object, contains a Tensor 's type and shape and format
    init_h: dict
            A dict object, contains a Tensor 's type and shape and format
    init_c: dict
            A dict object, contains a Tensor 's type and shape and format
    wci: dict
            A dict object, contains a Tensor 's type and shape and format
    wcf: dict
            A dict object, contains a Tensor 's type and shape and format
    wco: dict
            A dict object, contains a Tensor 's type and shape and format
    mask: dict
            A dict object, contains a Tensor 's type and shape and format
    y: dict
       A dict object, contains a Tensor 's type and shape and format
    output_h: dict
              A dict object, contains a Tensor 's type and shape and format
    output_c: dict
              A dict object, contains a Tensor 's type and shape and format
    i: dict
       A dict object, contains a Tensor 's type and shape and format
    j: dict
      A dict object, contains a Tensor 's type and shape and format
    f: dict
       A dict object, contains a Tensor 's type and shape and format
    o: dict
       A dict object, contains a Tensor 's type and shape and format
    tanhc: dict
           A dict object, contains a Tensor 's type and shape and format
    cell_type: An string identifying the cell type in the op. Default to "LSTM"
    direction: An string identifying the direction in the op. Default to "UNIDIRECTIONAL"
    cell_depth: An integer identifying the cell depth in the op. Default to 1
    use_peephole: An bool identifying if use peephole in the op. Default to False
    keep_prob: An float identifying the keep prob in the op. Default to 1
    cell_clip: An float identifying the cell clip in the op. Default to -1
    num_proj: An integer identifying the num projection in the op. Default to 0
    time_major: An bool identifying the time major in the op. Default to True
    activation: An string identifying the type of activation function in the op. Default to "tanh"
    recurrent_activation: An string identifying the type of activation function in the op. Default to "sigmoid"
    forget_bias: An float identifying the forget bias in the op. Default to 0
    gate_order: An string identifying the type of gate order in the op. Support "ijfo" and "ifco". Default to "ijfo"
    stateful: An bool identifying the type of stateful in the op. Default to False
    merge_mode: An string identifying the type of merge_modein the op. Default to "concat"
    is_training: An bool identifying is training in the op. Default to True
    kernel_name : cce kernel name, default value is "dynamic_rnn_v2"

    Returns:
    -------
    None
    """
    # one block size takes up 32b
    block_size_1 = 32
    # data type of int32
    int32 = "int32"
    tiling_arg_num = 3
    type_len_dict = {"float16": 2,
                    "float32": 4,
                    "int8": 1,
                    "uint8": 1,
                    "int32": 4,
                    "int64": 8, }
    is_dynamic = True
    compile_mode = tbe_context.get_context().get_addition("hybrid_compile_mode")

    if compile_mode is None:
        compile_mode = "multi_template"

    if compile_mode not in ["single_template", "multi_template"]:
        error_manager_vector.raise_err_specific_reson("DynamicRNNV2", "compile_mode is not support, please check!")

    if compile_mode == "single_template":
        is_dynamic = False

    check_param_dtype(input_x, weight_input, weight_hidden, bias, init_h, init_c, y, output_h,
                      output_c, i, j, f, o, tanhc)

    check_param_shape(input_x, weight_input, weight_hidden, bias, seq_length, init_h, init_c, wci,
                      wcf, wco, mask, y, output_h, output_c, i, j, f, o, tanhc)

    check_attr(cell_type, direction, cell_depth, use_peephole, keep_prob,
               cell_clip, num_proj, time_major, activation, recurrent_activation, gate_order,
               stateful, merge_mode)

    tik_instance = Tik(Dprofile())

    shape_x_input = input_x.get("shape")
    shape_w_i = weight_input.get("shape")
    shape_w_h = weight_hidden.get("shape")

    input_dtype = input_x.get("dtype").lower()
    y_dtype = y.get("dtype").lower()
    has_bias_input = bias is not None

    if is_dynamic:
        # dynamic shape get seq_length
        tiling_shape = (tiling_arg_num,)
        tiling_dtype = int32
        tiling_gm = tik_instance.Tensor(tiling_dtype, tiling_shape,
                                        name="ddr_arg", scope=scope_gm)
        tiling_ub = tik_instance.Tensor(tiling_dtype, tiling_shape,
                                        name="tiling_ub", scope=scope_ubuf)
        tik_instance.data_move(tiling_ub, tiling_gm, 0,
                               1, ceil_value(tiling_arg_num * type_len_dict.get(tiling_dtype), block_size_1),
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

    in_x = shape_x_input[1]
    n_size = shape_w_i[1]
    hidden_size = n_size // 4
    block_size = 4

    shape_x = (t_size, in_x, m_size, 16, 16)
    shape_w_i = (1, in_x, block_size, hidden_size, 16, 16)
    shape_w_h = (1, hidden_size, block_size, hidden_size, 16, 16)

    shape_y = (t_size, hidden_size, m_size, 16, 16)
    shape_hc = (t_size, hidden_size, m_size, 16, 16)
    shape_bias = (1, block_size, hidden_size, 1, 1, 16)
    shape_hc_init = (1, hidden_size, m_size, 16, 16)

    is_global_init = init_h is not None

    # due to FE/GE not support, now set default value
    is_gate_output = True

    sync = tik_instance.Tensor(shape=(512,), dtype="int64", scope=scope_gm, name='sync',
                               is_workspace=True, is_atomic_add=True)

    input_x = tik_instance.Tensor(shape=shape_x, dtype=input_dtype,
                                  scope=scope_gm, name='input_x')
    weight_i = tik_instance.Tensor(shape=shape_w_i, dtype=input_dtype,
                                   scope=scope_gm, name='weight_i')
    weight_h = tik_instance.Tensor(shape=shape_w_h, dtype=input_dtype,
                                   scope=scope_gm, name='weight_h')
    if has_bias_input:
        bias = tik_instance.Tensor(shape=shape_bias, scope=scope_gm,
                                   dtype=y_dtype, name='bias')
    else:
        bias = None
    is_using_seq_mask = False
    is_valid_mask = False
    if seq_length is not None:
        shape_seq_length = (t_size, hidden_size, m_size, 16, 16)
        is_using_seq_mask = True
        if seq_length.get("dtype").lower() == "int32":
            seq_mask_gm = tik_instance.Tensor(shape=shape_seq_length, scope=scope_gm,
                                              dtype="int32", name='seq_mask_gm')
        else:
            is_valid_mask = True
            seq_mask_gm = tik_instance.Tensor(shape=shape_seq_length, scope=scope_gm,
                                              dtype="float16", name='seq_mask_gm')
    else:
        seq_mask_gm = None

    if is_global_init:
        s_init_h_gm = tik_instance.Tensor(shape=shape_hc_init,
                                          dtype=input_dtype,
                                          scope=scope_gm,
                                          name='s_init_h_gm')
        s_init_c_gm = tik_instance.Tensor(shape=shape_hc_init,
                                          dtype=y_dtype, scope=scope_gm,
                                          name='s_init_c_gm')
    last_output_h_gm = tik_instance.Tensor(shape=shape_hc_init, dtype=input_dtype,
                                           scope=scope_gm, name='last_output_h_gm')
    last_output_c_gm = tik_instance.Tensor(shape=shape_hc_init, dtype=y_dtype,
                                           scope=scope_gm, name='last_output_c_gm')
    if is_dynamic:
        update_h_gm = tik_instance.Tensor(shape=shape_hc, dtype=input_dtype,
                                          scope=scope_gm, name='update_h_gm',
                                          is_workspace=True, max_mem_size=1024 * 1024 * 512)
    else:
        update_h_gm = tik_instance.Tensor(shape=shape_hc, dtype=input_dtype,
                                          scope=scope_gm, name='update_h_gm',
                                          is_workspace=True)
    if is_dynamic:
        update_c_gm = tik_instance.Tensor(shape=shape_hc, dtype=y_dtype,
                                          scope=scope_gm, name='update_c_gm',
                                          is_workspace=True, max_mem_size=1024 * 1024 * 512)
    else:
        update_c_gm = tik_instance.Tensor(shape=shape_hc, dtype=y_dtype,
                                          scope=scope_gm, name='update_c_gm',
                                          is_workspace=True)
    update_h_gm_as_y = tik_instance.Tensor(shape=shape_y, dtype=y_dtype,
                                           scope=scope_gm,
                                           name='update_h_gm_as_y')

    if is_gate_output:
        f_t_sigmoid_gm = tik_instance.Tensor(shape=shape_y, dtype=y_dtype,
                                             scope=scope_gm,
                                             name='f_t_sigmoid_gm')
        i_t_sigmoid_gm = tik_instance.Tensor(shape=shape_y, dtype=y_dtype,
                                             scope=scope_gm,
                                             name='i_t_sigmoid_gm')
        o_t_sigmoid_gm = tik_instance.Tensor(shape=shape_y, dtype=y_dtype,
                                             scope=scope_gm,
                                             name='o_t_sigmoid_gm')
        j_t_tanh_gm = tik_instance.Tensor(shape=shape_y, dtype=y_dtype,
                                          scope=scope_gm,
                                          name='j_t_tanh_gm')
        c_t_tanh_gm = tik_instance.Tensor(shape=shape_y, dtype=y_dtype,
                                          scope=scope_gm,
                                          name='c_t_tanh_gm')

    build_input_list = [input_x, weight_i, weight_h]
    if has_bias_input:
        build_input_list.append(bias)
    if is_using_seq_mask:
        build_input_list.append(seq_mask_gm)
    if is_global_init:
        build_input_list.append(s_init_h_gm)
        build_input_list.append(s_init_c_gm)

    if is_gate_output:
        build_output_list = [update_h_gm_as_y, last_output_h_gm, last_output_c_gm,
                             i_t_sigmoid_gm, j_t_tanh_gm, f_t_sigmoid_gm,
                             o_t_sigmoid_gm, c_t_tanh_gm]
    else:
        build_output_list = [update_h_gm_as_y, last_output_h_gm, last_output_c_gm]

    # for RL tune getting tik input&output tensor
    bank_manager.set_tik_tensor(build_input_list, build_output_list)
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

            input_x_var = input_x[loop_i * cut_t: loop_i * cut_t + cut_t, :, loop_j * cut_m: loop_j * cut_m + cut_m,
                          :, :]

            if is_global_init:
                s_init_c_gm_var = s_init_c_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
                s_init_h_gm_var = s_init_h_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            else:
                s_init_c_gm_var = None
                s_init_h_gm_var = None

            last_output_h_gm_var = last_output_h_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            last_output_c_gm_var = last_output_c_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]

            state_h_last = update_h_gm[loop_i * cut_t - last: loop_i * cut_t + cut_t - last:, :,
                           loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            state_c_last = update_c_gm[loop_i * cut_t - last: loop_i * cut_t + cut_t - last:, :,
                           loop_j * cut_m: loop_j * cut_m + cut_m, :, :]

            update_h_gm_var = update_h_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                              :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            update_c_gm_var = update_c_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                              :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            update_h_gm_as_y_var = update_h_gm_as_y[loop_i * cut_t: loop_i * cut_t + cut_t:,
                                   :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]

            if is_gate_output:
                f_t_sigmoid_gm_var = f_t_sigmoid_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                     :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
                i_t_sigmoid_gm_var = i_t_sigmoid_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                     :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
                o_t_sigmoid_gm_var = o_t_sigmoid_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                     :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
                j_t_tanh_gm_var = j_t_tanh_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                  :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
                c_t_tanh_gm_var = c_t_tanh_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                  :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            else:
                f_t_sigmoid_gm_var = None
                i_t_sigmoid_gm_var = None
                o_t_sigmoid_gm_var = None
                j_t_tanh_gm_var = None
                c_t_tanh_gm_var = None
            
            if is_valid_mask:
                seq_mask_gm_var = seq_mask_gm[loop_i * cut_t: loop_i * cut_t + cut_t,
                                              :,
                                              loop_j * cut_m: loop_j * cut_m + cut_m,
                                              :, :]
            else:
                seq_mask_gm_var = None

            input_list = [input_x_var, weight_i, weight_h, bias, s_init_h_gm_var,
                          s_init_c_gm_var, state_h_last,
                          state_c_last, sync, seq_mask_gm_var]

            if is_gate_output:
                output_list = [update_h_gm_var, update_c_gm_var,
                               update_h_gm_as_y_var, i_t_sigmoid_gm_var,
                               j_t_tanh_gm_var, f_t_sigmoid_gm_var,
                               o_t_sigmoid_gm_var, c_t_tanh_gm_var,
                               last_output_h_gm_var, last_output_c_gm_var]
            else:
                output_list = [update_h_gm_var, update_c_gm_var,
                               update_h_gm_as_y_var, last_output_h_gm_var,
                               last_output_c_gm_var]

            if is_dynamic:
                with tik_instance.if_scope(loop_i == 0):
                    is_first_round = True
                    rl_idx_list_first = tik_instance.call_module(
                        function=dynamic_rnn_tik,
                        input_tensors=input_list,
                        output_tensors=output_list,
                        config_map={"tiling_key": tiling_index},
                        input_params=[is_gate_output, is_first_round, is_global_init,
                                      forget_bias, recurrent_activation, gate_order, y_dtype, activation, is_dynamic])

                with tik_instance.if_scope(loop_i > 0):
                    is_first_round = False
                    tik_instance.call_module(
                        function=dynamic_rnn_tik,
                        input_tensors=input_list,
                        output_tensors=output_list,
                        config_map={"tiling_key": tiling_index},
                        input_params=[is_gate_output, is_first_round, is_global_init,
                                      forget_bias, recurrent_activation, gate_order, y_dtype, activation, is_dynamic])
            else:
                with tik_instance.if_scope(loop_i == 0):
                    is_first_round = True
                    tik_instance.call_module(
                        dynamic_rnn_tik,
                        input_list,
                        output_list,
                        [is_gate_output, is_first_round, is_global_init,
                         forget_bias, recurrent_activation, gate_order, y_dtype, activation, is_dynamic])

                with tik_instance.if_scope(loop_i > 0):
                    is_first_round = False
                    tik_instance.call_module(
                        dynamic_rnn_tik,
                        input_list,
                        output_list,
                        [is_gate_output, is_first_round, is_global_init,
                         forget_bias, recurrent_activation, gate_order, y_dtype, activation, is_dynamic])

    tiling_key_value_list = []
    if rl_idx_list_first:
        for idx in rl_idx_list_first:
            tiling_key_value_list.append([idx])

    config_map = {
        "dump_cce_code": False,
    }

    if is_dynamic:
        dynamic_config = copy.deepcopy(dynamic_build_config_dict)
        dynamic_config["dump_cce_code"] = False
        dynamic_config["tir.InjectSync"] = {"sync_mode": 2}
        dynamic_config["debug_message"] = False
        dynamic_config["save_temp_cce_file"] = False

        tik_instance.BuildCCE(kernel_name,
                              build_input_list,
                              build_output_list,
                              config=dynamic_config,
                              flowtable=(tiling_gm,),
                              extend_params={
                                  "build_multi_kernels": {
                                      "tiling_key": [tiling_index],
                                      "tiling_key_value": tiling_key_value_list
                                  }
                              }
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
    input_x = input_list[0]
    weight_i = input_list[1]
    weight_h = input_list[2]
    bias = input_list[3]
    s_init_h_gm = input_list[4]
    s_init_c_gm = input_list[5]
    s_state_h_gm_last = input_list[6]
    s_state_c_gm_last = input_list[7]
    sync0 = input_list[8]
    seq_mask_gm = input_list[9]

    is_gate_output = custom_list[0]
    is_first_round = custom_list[1]
    is_global_init = custom_list[2]
    forget_bias = custom_list[3]
    recurrent_activation = custom_list[4]
    gate_order = custom_list[5]
    y_dtype = custom_list[6]
    activation = custom_list[7]
    is_dynamic = custom_list[8]

    return dynamic_rnn_core(input_x, weight_i, weight_h, bias, s_init_h_gm, s_init_c_gm,
                            s_state_h_gm_last, s_state_c_gm_last, sync0, seq_mask_gm,
                            is_gate_output, is_first_round, is_global_init,
                            forget_bias, recurrent_activation, gate_order, y_dtype, activation, is_dynamic)


# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-statements,unnecessary-lambda
def dynamic_rnn_core(input_x, weight_i, weight_h, bias, s_init_h_gm, s_init_c_gm,
                     s_state_h_gm_last, s_state_c_gm_last, sync0, seq_mask_gm,
                     is_gate_output, is_first_round, is_global_init,
                     forget_bias, recurrent_activation, gate_order, y_dtype, activation, is_dynamic):
    """
    implement of dynamic rnn
    :return:
    """

    shape_x_input = input_x.shape
    shape_w_i = weight_i.shape

    t_size = 1
    if is_dynamic:
        m_size = shape_x_input[2]
    else:
        m_size = shape_x_input[2].value
    hidden_size = shape_w_i[3].value
    block_size = 4
    in_x = shape_x_input[1].value

    shape_a_z_bigz = (t_size, m_size, in_x, 16, 16)
    shape_b_1 = (t_size, in_x, block_size, hidden_size, 16, 16)
    shape_b_2 = (t_size, hidden_size, block_size, hidden_size, 16, 16)
    shape_c_1 = (t_size, block_size, hidden_size, m_size, 16, 16)
    shape_c_2 = (t_size, block_size, hidden_size, m_size, 16, 16)
    shape_bias = (t_size, block_size, hidden_size, 1, 1, 16)
    shape_h = (t_size, hidden_size, m_size, 16, 16)
    shape_h_z_bigz = (t_size, m_size, hidden_size, 16, 16)
    shape_i = (t_size, hidden_size, m_size, 16, 16)

    k0_size = 16
    input_dtype = input_x.dtype
    bias_dtype = y_dtype

    fp16_input_output = False
    if bias_dtype == 'float16':
        fp16_input_output = True

    # compute
    if is_first_round:
        if is_global_init:
            s_state_h_ub = tvm.compute(shape_h,
                                       lambda _, a_i, b_i, c_i, d_i: s_init_h_gm[0, a_i, b_i, c_i, d_i],
                                       name="s_init_h")
            s_state_c_ub = tvm.compute(shape_i,
                                       lambda _, a_i, b_i, c_i, d_i: s_init_c_gm[0, a_i, b_i, c_i, d_i],
                                       name="s_init_c")
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
                s_state_h_ub_for_element = tvm.compute(shape_h,
                                                       lambda *indices: tvm.const(0.0, dtype=input_dtype),
                                                       name='s_state_h_ub_for_element',
                                                       tag="broadcast")
    else:
        s_state_h_ub = tvm.compute(shape_h,
                                   lambda _, a_i, b_i, c_i, d_i: s_state_h_gm_last[0, a_i, b_i, c_i, d_i],
                                   name="s_state_h_ub")
        s_state_c_ub = tvm.compute(shape_i,
                                   lambda _, a_i, b_i, c_i, d_i: s_state_c_gm_last[0, a_i, b_i, c_i, d_i],
                                   name="s_state_c_ub")
        if seq_mask_gm is not None:
            s_state_h_ub_for_element = tvm.compute(shape_h,
                                                   lambda _, a_i, b_i, c_i, d_i: s_state_h_gm_last[
                                                   0, a_i, b_i, c_i, d_i], name="s_state_h_ub_for_element")

    # input and s_start_h is Nz, need trans to zZ
    # so change axis 1 and 2
    a_l1_1 = tvm.compute(shape_a_z_bigz,
                         lambda *indice: input_x[indice[0], indice[2], indice[1], indice[3], indice[4]],
                         name="a_l1_1", tag="out_to_l1")
    b_l1_1 = tvm.compute(shape_b_1,
                         lambda *indices: weight_i(*indices),
                         name='b_l1_1',
                         tag="out_to_l1")

    a_l0a_1 = tvm.compute(shape_a_z_bigz, lambda *indices: a_l1_1(*indices),
                          name="a_l0a_1", tag="l1_to_l0")
    b_l0b_1 = tvm.compute(shape_b_1,
                          lambda *indices: b_l1_1(*indices),
                          name="b_l0b_1",
                          tag="l1_to_l0")

    k1_1 = tvm.reduce_axis((0, in_x), name='k1_1')
    k0_1 = tvm.reduce_axis((0, k0_size), name='k0_1')

    c_l0c_1 = tvm.compute(
        shape_c_1,
        lambda t, nb_0, nb_1, mb, mp, np: tvm.sum((a_l0a_1[t, mb, k1_1, mp, k0_1] * \
                                                   b_l0b_1[t, k1_1, nb_0, nb_1, np, k0_1]).astype('float32'),
                                                  axis=[k1_1, k0_1]),
        name='c_l0c_1',
        tag="matmul")

    c_ub_1 = tvm.compute(shape_c_1, lambda *indices: c_l0c_1(*indices), name="c_ub_1")

    if bias is not None:
        bias_ub = tvm.compute(shape_bias,
                              lambda *indices: bias(*indices),
                              name='bias_ub')

        bias_ub_mid = bias_ub
        if fp16_input_output:
            bias_ub_fp32 = tvm.compute(shape_bias,
                                       lambda *indices: bias_ub(*indices).astype('float32'),
                                       name="bias_ub_fp32_drnn_cast",
                                       tag="elewise_single_cast")
            bias_ub_mid = bias_ub_fp32
        bias_bc_ub = broadcast(bias_ub_mid, shape_c_1)
        c_ub_bias_1 = vadd(c_ub_1, bias_bc_ub)
    else:
        c_ub_bias_1 = c_ub_1

    # second matmul for hidden
    # input and s_start_h is Nz, need trans to zZ, so change axis 1 and 2
    a_l1_2 = tvm.compute(shape_h_z_bigz,
                         lambda *indice: s_state_h_ub[indice[0], indice[2], indice[1], indice[3], indice[4]],
                         name="a_l1_2",
                         tag="out_to_l1")
    b_l1_2 = tvm.compute(shape_b_2,
                         lambda *indices: weight_h(*indices),
                         name="b_l1_2",
                         tag="out_to_l1")
    a_l0a_2 = tvm.compute(shape_h_z_bigz,
                          lambda *indices: a_l1_2(*indices),
                          name="a_l0a_2",
                          tag="l1_to_l0")
    b_l0b_2 = tvm.compute(shape_b_2,
                          lambda *indices: b_l1_2(*indices),
                          name="b_l0b_2",
                          tag="l1_to_l0")
    k1_2 = tvm.reduce_axis((0, hidden_size), name="k1_2")
    k0_2 = tvm.reduce_axis((0, k0_size), name="k0_2")
    c_l0c_2 = tvm.compute(
        shape_c_2,
        lambda t, nb_0, nb_1, mb, mp, np: tvm.sum((a_l0a_2[t, mb, k1_2, mp, k0_2] * \
                                                   b_l0b_2[t, k1_2, nb_0, nb_1, np, k0_2]).astype("float32"),
                                                  axis=[k1_2, k0_2]),
        name="c_l0c_2",
        tag="matmul")
    c_ub_2 = tvm.compute(shape_c_2, lambda *indices: c_l0c_2(*indices), name="c_ub_2")
    c_ub_bias = vadd(c_ub_bias_1, c_ub_2)

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

    if recurrent_activation == "hard_sigmoid":
        f_t_sigmoid = hard_sigmoid_compute(f_t_bias)
        i_t_sigmoid = hard_sigmoid_compute(i_t)
        o_t_sigmoid = hard_sigmoid_compute(o_t)
    else:
        f_t_sigmoid = sigmoid_compute(f_t_bias)
        i_t_sigmoid = sigmoid_compute(i_t)
        o_t_sigmoid = sigmoid_compute(o_t)

    j_t_tanh = activation_compute(activation, j_t)

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
            f_t_sigmoid_back_fp32 = tvm.compute(
                shape_i,
                lambda *indices: f_t_sigmoid_back(*indices).astype('float32'),
                name="f_t_sigmoid_back_fp32_drnn_cast",
                tag="elewise_single_cast")
            i_t_sigmoid_back_fp32 = tvm.compute(
                shape_i,
                lambda *indices: i_t_sigmoid_back(*indices).astype('float32'),
                name="i_t_sigmoid_back_fp32_drnn_cast",
                tag="elewise_single_cast")
            o_t_sigmoid_back_fp32 = tvm.compute(
                shape_i,
                lambda *indices: o_t_sigmoid_back(*indices).astype('float32'),
                name="o_t_sigmoid_back_fp32_drnn_cast",
                tag="elewise_single_cast")
            j_t_tanh_back_fp32 = tvm.compute(
                shape_i,
                lambda *indices: j_t_tanh_back(*indices).astype('float32'),
                name="j_t_tanh_back_fp32_drnn_cast",
                tag="elewise_single_cast")
            s_state_c_back_fp32 = tvm.compute(
                shape_i,
                lambda *indices: s_state_c_ub(*indices).astype('float32'),
                name="s_state_c_back_fp32_drnn_cast",
                tag="elewise_single_cast")

        f_t_sigmoid_ub = f_t_sigmoid_back
        i_t_sigmoid_ub = i_t_sigmoid_back
        o_t_sigmoid_ub = o_t_sigmoid_back
        j_t_tanh_ub = j_t_tanh_back
        s_state_c_ub_temp = s_state_c_ub

        if fp16_input_output:
            f_t_sigmoid_ub = f_t_sigmoid_back_fp32
            i_t_sigmoid_ub = i_t_sigmoid_back_fp32
            o_t_sigmoid_ub = o_t_sigmoid_back_fp32
            j_t_tanh_ub = j_t_tanh_back_fp32
            s_state_c_ub_temp = s_state_c_back_fp32

    # auto cast support both fp16 fp32
    c_t_tmp1 = vmul(s_state_c_ub_temp, f_t_sigmoid_ub)
    c_t_tmp2 = vmul(j_t_tanh_ub, i_t_sigmoid_ub)
    update_c = vadd(c_t_tmp1, c_t_tmp2)

    if seq_mask_gm is not None:
        seq_mask_ub = tvm.compute(shape_h, lambda _, a_i, b_i, c_i, d_i: seq_mask_gm[0, a_i, b_i, c_i, d_i],
                                  name="seq_mask_ub")
        update_c_diff = vsub(update_c, s_state_c_ub_temp)
        seq_mask_ub_fp32 = tvm.compute(shape_h,
                                       lambda *indices: seq_mask_ub(*indices).astype('float32'),
                                       name="seq_mask_ub_fp32_drnn_cast",
                                       tag="elewise_single_cast")
        update_c_tmp = vmul(update_c_diff, seq_mask_ub_fp32)
        update_c = vadd(update_c_tmp, s_state_c_ub_temp)

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
        c_t_tanh = activation_compute(activation, update_c_fp16_back_fp32)
    else:
        update_c_fp32_back = tvm.compute(shape_i,
                                         lambda *indices: update_c_gm(*indices),
                                         name="update_c_fp32_back",
                                         tag="out_to_ub")
        last_update_c_gm = tvm.compute(shape_i,
                                       lambda *indices: update_c_fp32_back(*indices),
                                       name="last_update_c_gm",
                                       tag="ub_to_out")
        last_update_c_back = tvm.compute(shape_i,
                                         lambda *indices: last_update_c_gm(*indices),
                                         name="last_update_c_back",
                                         tag="out_to_ub")
        c_t_tanh = activation_compute(activation, last_update_c_back)

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
                                             lambda *indices: c_t_tanh_back(*indices).astype('float32'),
                                             name="c_t_tanh_back_fp32_drnn_cast",
                                             tag="elewise_single_cast")

        c_t_tanh_ub = c_t_tanh_back

        if fp16_input_output:
            c_t_tanh_ub = c_t_tanh_back_fp32

    update_h = vmul(c_t_tanh_ub, o_t_sigmoid_ub)
    if seq_mask_gm is not None:
        update_h = vmul(update_h, seq_mask_ub_fp32)

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
        if fp16_input_output:
            update_h_diff = vsub(update_h_gm_as_y_back, s_state_h_ub_for_element)
            update_h_diff_mask = vmul(update_h_diff, seq_mask_ub)
            update_h_gm_as_y_back_mid = vadd(update_h_diff_mask, s_state_h_ub_for_element)
        else:
            s_state_h_ub_for_element_fp32 = \
                tvm.compute(shape_h,
                            lambda *indices: s_state_h_ub_for_element(*indices).astype('float32'),
                            name="s_state_h_ub_for_element_cast",
                            tag="elewise_single_cast")
            update_h_diff = vsub(update_h_gm_as_y_back, s_state_h_ub_for_element_fp32)
            update_h_diff_mask = vmul(update_h_diff, seq_mask_ub_fp32)
            update_h_gm_as_y_back_mid = vadd(update_h_diff_mask, s_state_h_ub_for_element_fp32)
    if fp16_input_output:
        last_update_h_gm = tvm.compute(shape_i,
                                       lambda *indices: update_h_gm_as_y_back_mid(*indices),
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
        update_h_fp16_cast = tvm.compute(shape_i,
                                         lambda *indices: update_h_gm_as_y_back_mid(*indices).astype('float16'),
                                         name="update_h_fp16_cast_drnn_cast",
                                         tag="elewise_single_cast")
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
    # end compute

    if is_gate_output:
        return_list = [update_h_gm, update_c_gm, update_h_gm_as_y,
                       i_t_sigmoid_gm,
                       j_t_tanh_gm, f_t_sigmoid_gm, o_t_sigmoid_gm,
                       c_t_tanh_gm, last_update_h_gm, last_update_c_gm]
    else:
        return_list = [update_h_gm, update_c_gm, update_h_gm_as_y,
                       last_update_h_gm, last_update_c_gm]

    return_list, s = rl_bank.tik_dsl_bank_proc(return_list, sync_tensor=sync0, dynamic=is_dynamic)
    if s is not None:
        if is_dynamic:
            sch_list, tune_shape_list = s
            index_list = []
            for index, _ in enumerate(tune_shape_list):
                index_list.append(tune_shape_list[index][2])

            if sch_list is not None and len(sch_list) > 0:
                for index, sch_list_value in enumerate(sch_list):
                    sch_list_value.set_constraint(
                        expr.And(input_x.shape[2] <= tune_shape_list[index][1], input_x.shape[2] > 0))
                tbe_context.get_context().add_compile_info("vars", {"tune_shape_list": tune_shape_list})
            else:
                bank_manager.update_bank_hit_info(True)
        else:
            return return_list, s
    
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
    elewise_before_barrier_tensors = []
    if bias is not None:
        elewise_before_barrier_tensors.extend([bias_bc_ub, c_ub_bias_1])

    # set scope
    s[a_l1_1].set_scope(scope_cbuf)
    s[b_l1_1].set_scope(scope_cbuf)
    s[a_l0a_1].set_scope(scope_ca)
    s[b_l0b_1].set_scope(scope_cb)
    s[c_l0c_1].set_scope(scope_cc)
    s[c_l0c_1].set_scope(scope_cc)
    s[c_ub_1].set_scope(scope_ubuf)
    s[a_l1_2].set_scope(scope_cbuf)
    s[b_l1_2].set_scope(scope_cbuf)
    s[a_l0a_2].set_scope(scope_ca)
    s[b_l0b_2].set_scope(scope_cb)
    s[c_l0c_2].set_scope(scope_cc)
    s[c_ub_2].set_scope(scope_ubuf)
    if bias is not None:
        s[bias_ub].set_scope(scope_ubuf)
        s[bias_bc_ub].set_scope(scope_ubuf)
        if fp16_input_output:
            s[bias_ub_fp32].set_scope(scope_ubuf)
    s[s_state_h_ub].set_scope(scope_ubuf)
    s[s_state_c_ub].set_scope(scope_ubuf)
    if seq_mask_gm is not None:
        s[s_state_h_ub_for_element].set_scope(scope_ubuf)
        if not fp16_input_output:
            s[s_state_h_ub_for_element_fp32].set_scope(scope_ubuf)

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
            s[s_state_c_back_fp32].set_scope(scope_ubuf)

        s[f_t_sigmoid_back].set_scope(scope_ubuf)
        s[i_t_sigmoid_back].set_scope(scope_ubuf)
        s[o_t_sigmoid_back].set_scope(scope_ubuf)
        s[j_t_tanh_back].set_scope(scope_ubuf)
        s[c_t_tanh_back].set_scope(scope_ubuf)
    s[last_update_c_back].set_scope(scope_ubuf)
    s[last_update_h_back].set_scope(scope_ubuf)
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
        s[seq_mask_ub_fp32].set_scope(scope_ubuf)

    # compute inline
    compute_inline_tensors = [i_t, j_t, f_t, o_t]
    for tensor in compute_inline_tensors:
        s[tensor].compute_inline()

    # matmul tiling
    tiling = get_lstm_tiling()
    factor_l1_m, factor_l1_n, factor_l1_k_1, factor_l1_k_2 = tiling.factor_l1_m, tiling.factor_l1_n, \
                                                             tiling.factor_l1_k_1, tiling.factor_l1_k_2
    factor_l0_m, factor_l0_n, factor_l0_k_1, factor_l0_k_2 = tiling.factor_l0_m, tiling.factor_l0_n, \
                                                             tiling.factor_l0_k_1, tiling.factor_l0_k_2

    l1_n_outer_1, l1_n_inner_1 = s[c_l0c_1].split(c_l0c_1.op.axis[2], factor=factor_l1_n)
    l1_m_outer_1, l1_m_inner_1 = s[c_l0c_1].split(c_l0c_1.op.axis[3], factor=factor_l1_m)
    l1_k_outer_1, l1_k_inner_1 = s[c_l0c_1].split(c_l0c_1.op.reduce_axis[0], factor=factor_l1_k_1)
    l0_n_outer_1, l0_n_inner_1 = s[c_l0c_1].split(l1_n_inner_1, factor=factor_l0_n)
    l0_m_outer_1, l0_m_inner_1 = s[c_l0c_1].split(l1_m_inner_1, factor=factor_l0_m)
    l0_k_outer_1, l0_k_inner_1 = s[c_l0c_1].split(l1_k_inner_1, factor=factor_l0_k_1)

    s[c_l0c_1].reorder(l1_n_outer_1,
                       c_l0c_1.op.axis[1],
                       l1_m_outer_1,
                       l1_k_outer_1,
                       l0_n_outer_1,
                       l0_m_outer_1,
                       l0_k_outer_1,
                       l0_n_inner_1,
                       l0_m_inner_1,
                       c_l0c_1.op.axis[3 + 1],
                       c_l0c_1.op.axis[4 + 1],
                       l0_k_inner_1,
                       c_l0c_1.op.reduce_axis[1])
    s[a_l0a_1].compute_at(s[c_l0c_1], l0_k_outer_1)
    s[b_l0b_1].compute_at(s[c_l0c_1], l0_k_outer_1)
    s[a_l1_1].compute_at(s[c_l0c_1], l1_k_outer_1)
    s[b_l1_1].compute_at(s[c_l0c_1], l1_k_outer_1)

    l1_n_outer_2, l1_n_inner_2 = s[c_l0c_2].split(c_l0c_2.op.axis[2], factor=factor_l1_n)
    l1_m_outer_2, l1_m_inner_2 = s[c_l0c_2].split(c_l0c_2.op.axis[3], factor=factor_l1_m)
    l1_k_outer_2, l1_k_inner_2 = s[c_l0c_2].split(c_l0c_2.op.reduce_axis[0], factor=factor_l1_k_2)
    l0_n_outer_2, l0_n_inner_2 = s[c_l0c_2].split(l1_n_inner_2, factor=factor_l0_n)
    l0_m_outer_2, l0_m_inner_2 = s[c_l0c_2].split(l1_m_inner_2, factor=factor_l0_m)
    l0_k_outer_2, l0_k_inner_2 = s[c_l0c_2].split(l1_k_inner_2, factor=factor_l0_k_2)

    s[c_l0c_2].reorder(l1_n_outer_2,
                       c_l0c_2.op.axis[1],
                       l1_m_outer_2,
                       l1_k_outer_2,
                       l0_n_outer_2,
                       l0_m_outer_2,
                       l0_k_outer_2,
                       l0_n_inner_2,
                       l0_m_inner_2,
                       c_l0c_2.op.axis[3 + 1],
                       c_l0c_2.op.axis[4 + 1],
                       l0_k_inner_2,
                       c_l0c_2.op.reduce_axis[1])
    s[a_l0a_2].compute_at(s[c_l0c_2], l0_k_outer_2)
    s[b_l0b_2].compute_at(s[c_l0c_2], l0_k_outer_2)
    s[a_l1_2].compute_at(s[c_l0c_2], l1_k_outer_2)
    s[b_l1_2].compute_at(s[c_l0c_2], l1_k_outer_2)
    s[s_state_h_ub].compute_at(s[c_l0c_2], l1_k_outer_2)

    ub_n_outer_1, ub_n_inner_1 = s[c_ub_1].split(c_ub_1.op.axis[2], factor=factor_l1_n)

    ub_m_outer_1, ub_m_inner_1 = s[c_ub_1].split(c_ub_1.op.axis[3], factor=factor_l1_m)
    s[c_ub_1].reorder(ub_m_outer_1, ub_n_outer_1, c_ub_1.op.axis[1],
                      ub_n_inner_1, ub_m_inner_1, c_ub_1.op.axis[4],
                      c_ub_1.op.axis[5])

    s[c_l0c_1].compute_at(s[c_ub_1], ub_n_outer_1)

    ub_n_outer_2, ub_n_inner_2 = s[c_ub_2].split(c_ub_2.op.axis[2], factor=factor_l1_n)

    ub_m_outer_2, ub_m_inner_2 = s[c_ub_2].split(c_ub_2.op.axis[3], factor=factor_l1_m)
    s[c_ub_2].reorder(ub_m_outer_2, ub_n_outer_2, c_ub_2.op.axis[1],
                      ub_n_inner_2, ub_m_inner_2, c_ub_2.op.axis[4],
                      c_ub_2.op.axis[5])

    s[c_l0c_2].compute_at(s[c_ub_2], ub_n_outer_2)

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

    s[c_ub_1].compute_at(s[barrier_tensor], barrier_outer)
    s[c_ub_2].compute_at(s[barrier_tensor], barrier_outer)
    if bias is not None:
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
        if not fp16_input_output:
            s[s_state_h_ub_for_element_fp32].compute_at(s[update_h_gm], vn_o_inner)
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
        s[seq_mask_ub_fp32].compute_at(s[update_h_gm], vn_o_inner)
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
        s[update_c_fp16].reused_by(last_update_c_back)
        s[update_c_fp16].reused_by(update_c_fp16_back)
        s[update_c_fp16_back].reused_by(reuse_data=True)
        s[last_update_c_back].reused_by(reuse_data=True)
        s[update_h_gm_as_y_back_mid].reused_by(last_update_h_back)
        s[last_update_h_back].reused_by(reuse_data=True)

    else:
        s[update_c].reused_by(update_c_fp32_back)
        s[update_c_fp32_back].reused_by(reuse_data=True)
        s[update_h].reused_by(update_h_gm_as_y_back)
        s[update_h_gm_as_y_back].reused_by(reuse_data=True)
        s[update_c].reused_by(last_update_c_back)
        s[last_update_c_back].reused_by(reuse_data=True)
        s[update_h_fp16_cast].reused_by(last_update_h_back)
        s[last_update_h_back].reused_by(reuse_data=True)

    s[last_update_c_gm].compute_at(s[update_h_gm], vn_o_inner)
    s[last_update_h_gm].compute_at(s[update_h_gm], vn_o_inner)
    s[last_update_c_back].compute_at(s[update_h_gm], vn_o_inner)
    s[last_update_h_back].compute_at(s[update_h_gm], vn_o_inner)

    s[a_l1_1].double_buffer()
    s[b_l1_1].double_buffer()
    s[a_l0a_1].double_buffer()
    s[b_l0b_1].double_buffer()
    s[c_l0c_1].double_buffer()
    s[c_ub_1].double_buffer()
    s[a_l1_2].double_buffer()
    s[b_l1_2].double_buffer()
    s[a_l0a_2].double_buffer()
    s[b_l0b_2].double_buffer()
    s[c_l0c_2].double_buffer()
    s[c_ub_2].double_buffer()

    # set bound
    shape_a_z_bigz = 1 * factor_l1_m * factor_l1_k_1 * 16 * 16
    shape_b_1 = 1 * factor_l1_k_1 * 1 * factor_l1_n * 16 * 16
    shape_b_2 = 1 * factor_l1_k_2 * 1 * factor_l1_n * 16 * 16
    shape_c_1 = 1 * 4 * factor_l1_n * factor_l1_m * 16 * 16
    shape_c_2 = 1 * 4 * factor_l1_n * factor_l1_m * 16 * 16
    shape_bias = 1 * 4 * factor_l1_n * 1 * 16 * 16
    shape_h = 1 * factor_l1_k_2 * factor_l1_m * 16 * 16
    shape_i = 1 * factor_l1_n * factor_l1_m * 16 * 16
    shape_h_z_bigz = 1 * factor_l1_m * factor_l1_k_2 * 16 * 16

    # emit_insn
    s[a_l1_1].emit_insn(a_l1_1.op.axis[0], 'dma_copy')
    s[b_l1_1].emit_insn(b_l1_1.op.axis[0], 'dma_copy')
    s[a_l0a_1].emit_insn(a_l0a_1.op.axis[0], 'dma_copy')
    s[b_l0b_1].emit_insn(b_l0b_1.op.axis[0], 'dma_copy')

    mad_dict = {"mad_pattern": 0, "k_outer": [l1_k_outer_1, l0_k_outer_1]}
    s[c_l0c_1].emit_insn(l0_n_inner_1, 'mad', mad_dict)
    s[c_ub_1].emit_insn(ub_n_inner_1, 'dma_copy')

    # emit_insn
    s[a_l1_2].emit_insn(a_l1_2.op.axis[0], 'dma_copy')
    s[b_l1_2].emit_insn(b_l1_2.op.axis[0], 'dma_copy')
    s[a_l0a_2].emit_insn(a_l0a_2.op.axis[0], 'dma_copy')
    s[b_l0b_2].emit_insn(b_l0b_2.op.axis[0], 'dma_copy')

    mad_dict = {"mad_pattern": 0, "k_outer": [l1_k_outer_2, l0_k_outer_2]}
    s[c_l0c_2].emit_insn(l0_n_inner_2, 'mad', mad_dict)
    s[c_ub_2].emit_insn(ub_n_inner_2, 'dma_copy')

    if bias is not None:
        s[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')
        if fp16_input_output:
            s[bias_ub_fp32].emit_insn(bias_ub_fp32.op.axis[0], 'vector_conv')
        s[bias_bc_ub].emit_insn(bias_bc_ub.op.axis[0], 'vector_broadcast', attrs={"storage_bound": 16384})
        s[c_ub_bias_1].emit_insn(c_ub_bias_1.op.axis[0], 'vector_add')
    if is_first_round:
        if is_global_init:
            s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'dma_copy')
            s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'dma_copy')
            if seq_mask_gm is not None:
                s[s_state_h_ub_for_element].emit_insn(s_state_h_ub_for_element.op.axis[0], 'dma_copy')
        else:
            s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'vector_broadcast')
            s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'vector_broadcast')
            if seq_mask_gm is not None:
                s[s_state_h_ub_for_element].emit_insn(s_state_h_ub_for_element.op.axis[0], 'vector_broadcast')
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

    s[last_update_c_gm].emit_insn(s[last_update_c_gm].op.axis[1], 'dma_copy')
    s[last_update_h_gm].emit_insn(s[last_update_h_gm].op.axis[1], 'dma_copy')

    s[update_c_gm].emit_insn(s[update_c_gm].op.axis[1], 'dma_copy')
    s[update_h_gm].emit_insn(s[update_h_gm].op.axis[3], 'dma_copy')

    block = tvm.thread_axis('blockIdx.x')
    s[update_h_gm].bind(vn_o_outer, block)
    s[update_h_gm].wait_block_sync(axis=vn_m_outer, tensor=sync0[0], bottom=True)
    s[update_h_gm].set_block_sync(axis=vn_m_outer, tensor=sync0[0], bottom=True)

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
            s[o_t_sigmoid_back_fp32].emit_insn(
                s[o_t_sigmoid_back_fp32].op.axis[1], 'phony_insn')
            s[j_t_tanh_back_fp32].emit_insn(s[j_t_tanh_back_fp32].op.axis[1],
                                            'phony_insn')
            s[c_t_tanh_back_fp32].emit_insn(s[c_t_tanh_back_fp32].op.axis[1],
                                            'phony_insn')
            s[s_state_c_back_fp32].emit_insn(s[s_state_c_back_fp32].op.axis[1],
                                             'vector_conv')

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

    s[last_update_c_back].emit_insn(last_update_c_back.op.axis[0], 'phony_insn')
    s[last_update_h_back].emit_insn(last_update_h_back.op.axis[0], 'phony_insn')

    # fp16 in
    if bias_dtype == 'float16':
        s[update_c_fp16].emit_insn(update_c_fp16.op.axis[0], 'vector_conv')
        s[update_c_fp16_back].emit_insn(update_c_fp16_back.op.axis[0],
                                        'phony_insn')
        s[update_c_fp16_back_fp32].emit_insn(
            update_c_fp16_back_fp32.op.axis[0], 'phony_insn')
        s[update_h_fp16].emit_insn(update_h_fp16.op.axis[0], 'vector_conv')
    else:
        s[update_c_fp32_back].emit_insn(update_c_fp32_back.op.axis[0],
                                        'phony_insn')
        s[update_h_fp16_cast].emit_insn(update_h_fp16_cast.op.axis[0],
                                        'vector_conv')
    if seq_mask_gm is not None:
        s[seq_mask_ub].emit_insn(seq_mask_ub.op.axis[0], 'dma_copy')
        s[seq_mask_ub_fp32].emit_insn(seq_mask_ub_fp32.op.axis[0], 'vector_conv')
        if not fp16_input_output:
            s[s_state_h_ub_for_element_fp32].emit_insn(s_state_h_ub_for_element_fp32.op.axis[0], 'vector_conv')
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
